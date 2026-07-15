"""
landslide_risk_feature_engineering.py

Standardizes raw data downloaded by the sub-packages into an aligned
``input_data/`` folder ready for the Steinbaum fuzzy-logic model.

raster standardization pipeline: load (mosaic if tiled) -> clip -> reprojection -> resample -> standardize nodata -> derive features

engineered features:
 - lithology: categorical turns into scale versions based on ()
 - Road vector is rasterized and the distance to a road pixel for every road is computed
 - Slope is derived from the DEM raster

Usage:
    python landslide_risk_feature_engineering.py --config config.yaml
                                                 [--save-intermediates]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
import yaml
from rasterio.enums import Resampling
from xrspatial import slope

from utils import raster_utils

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover
    distance_transform_edt = None


TARGET_RESOLUTION_M = 30.0     # reference grid pixel size, in UTM meters
CLIP_BUFFER_DEG = 0.02         # bbox buffer applied at the EPSG:4326 clip step
                               # (~2 km at mid-latitudes). Prevents NaN halos
                               # at raster edges after reprojection.

# Layer registry. Order doesn't matter; exactly one layer must set is_reference.
#   config_key   : top-level key in config.yaml whose `output_dir` holds the raw data
#   source       : "vector" for plugins whose output is rasterized onto the
#                  reference grid instead of loaded as a GeoTIFF; omitted for
#                  raster sources (the default)
#   resampling   : rioxarray resampling method (raster sources only)
#   feature      : optional post/pre-processing transform to apply
#   is_reference : the layer whose grid every other layer is matched to

LAYERS: dict[str, dict] = {
    "dem": {
        "config_key": "dem_copernicus",
        "is_reference": True,
        "feature": "calculate_slope",
        "resampling": Resampling.bilinear,
    },
    "forest_loss": {
        "config_key": "gfc_hansen",
        "resampling": Resampling.nearest,
        "feature": "binarize_forest_loss",
    },
    "lithology": {
        "config_key": "lithology_glim",
        "resampling": Resampling.nearest,
        "feature": "remap_lithology",
    },
    "roads": {
        "config_key": "pois_osm",
        "source": "vector",
        "feature": "distance_to_features",
    },
    "seismology": {
        "config_key": "seismology_gemf",
        "resampling": Resampling.bilinear,
    },
}


# ---------------------------------------------------------------------------
# helper function
# ---------------------------------------------------------------------------


def save_to_tmp(da: xr.DataArray, tmp_dir: Path, name: str) -> xr.DataArray:
    """Saves a DataArray to disk and returns the computed array to avoid recalculation."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{name}.tif"
    
    # Realize the computation graph
    realized = da.compute() if hasattr(da, "compute") else da
    realized.rio.to_raster(str(out))
    print(f"  [tmp] {out}")
    
    # Return the realized array to the pipeline!
    return realized


# ---------------------------------------------------------------------------
# Feature Engineering functions
# ---------------------------------------------------------------------------


def calculate_slope(da: xr.DataArray) -> xr.DataArray:
    """Calculate slope in degrees from a DEM raster using xrspatial."""
    
    # 1. Bring into memory and squeeze down to 2D (y, x) because xrspatial expects a 2D grid
    dem_2d = da.compute().squeeze()
    
    # 2. Calculate the slope (automatically uses the raster's X/Y meter coordinates)
    slope_deg = slope(dem_2d, name="slope")
    
    # 3. Restore the 3D shape (band, y, x) required by rioxarray for saving GeoTIFFs
    out = slope_deg.expand_dims(dim="band")
    
    # 4. Copy over the vital spatial metadata from the original array
    out.rio.write_crs(da.rio.crs, inplace=True)
    out.rio.write_transform(da.rio.transform(), inplace=True)
    out.rio.write_nodata(np.nan, inplace=True)
    
    return out

def remap_lithology(da: xr.DataArray, mapping: dict[int, float] | None = None) -> xr.DataArray:
    """Remap GLiM lithology classes to a susceptibility membership; unmapped -> NaN.

    The output is fed straight into the gamma overlay with no fuzzy membership
    function, mirroring Stanley & Kirschbaum Fig. 2, where the geologic ranking is
    the only input without a Fuzzy Membership step: the rescaled rating *is* the
    membership.

    `mapping` gives GLiM class -> rock strength on a 1-7 scale (higher = stronger =
    less susceptible), with 0 as a sentinel for "other" (water, ice, no-data).
    """
    if mapping is None:
        mapping = {
            0: 0, # No Data > Other
            1: 1, # Evaporites > Weak Sedimentary
            2: 0, # Ice and Glaciers > Other
            3: 5, # Metamorphics > Metamorphic
            4: 7, # Acid plutonic rocks > Pluton
            5: 7, # Basic plutonic rocks > Pluton
            6: 7, # Intermediate plutonic rocks > Pluton
            7: 2, # Pyroclastics > Volcanic
            8: 6, # Carbonate sedimentary rocks > Carbonate
            9: 4, # Mixed sedimentary rocks > Mixed Sedimentary
            10: 3, # Siliciclastic sedimentary rocks > Siliciclastic Sedimentary
            11: 1, # Unconsolidated sediments > Weak Sedimentary
            12: 2, # Acid volcanic rocks > Volcanic
            13: 2, # Basic volcanic rocks > Volcanic
            14: 2, # Intermediate volcanic rocks > Volcanic
            15: 0, # Water Bodies > Other
        }

    # Strength (1-7, higher = stronger) -> susceptibility, rescaled onto the 0.2-1.0
    # span of S&K Table 3: their strongest rocks rate 0.2 and their weakest 1.0.
    # Class 0 ("other") takes the 0.1 the paper gives its Null/Unknown materials.
    #
    # Nothing may reach exactly 0.0: gamma_fuzzy multiplies the memberships, so a
    # single zero annihilates the pixel no matter what the other layers say. That is
    # why Table 3 floors at 0.1 rather than assigning water bodies a literal zero.
    susceptibility = {
        k: round(0.2 + 0.8 * (7 - v) / 6, 2) if v > 0 else 0.1
        for k, v in mapping.items()
    }

    src = np.asarray(da.compute())
    out = np.full(src.shape, np.nan, dtype=np.float32)
    for old, new in susceptibility.items():
        out[src == old] = new

    # GLiM declares nodata = 0, so load_raster's masked=True has already turned class
    # 0 into NaN by the time it reaches the loop above and it would drop out of the
    # model entirely. Class 0 is a real category in the mapping ("other"), and in this
    # AOI it is overwhelmingly flat inland water, so it takes the same rating as the
    # other "other" classes. This handler runs at the "pre" stage, before reprojection,
    # so NaN here is GLiM's own nodata and not a reprojection edge.
    out[np.isnan(src)] = susceptibility[0]

    remapped = xr.DataArray(out, dims=da.dims, coords=da.coords)
    remapped.rio.write_crs(da.rio.crs, inplace=True)
    remapped.rio.write_transform(da.rio.transform(), inplace=True)
    remapped.rio.write_nodata(np.nan, inplace=True)
    return remapped


def distance_to_features(mask: xr.DataArray, feature_value: float = 1) -> xr.DataArray:
    """Compute per-pixel Euclidean distance to the nearest feature pixel."""
    if distance_transform_edt is None:
        raise ImportError("distance_to_features requires scipy")

    arr = np.asarray(mask.compute())
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    background = (arr != feature_value).astype(np.uint8)
    px = abs(float(mask.rio.resolution()[0]))
    dist = distance_transform_edt(background) * px
    dist = dist.astype(np.float32)

    out = xr.DataArray(
        dist,
        dims=("y", "x"),
        coords={"y": mask["y"].values, "x": mask["x"].values},
    )
    out.rio.write_crs(mask.rio.crs, inplace=True)
    out.rio.write_transform(mask.rio.transform(), inplace=True)
    out.rio.write_nodata(np.nan, inplace=True)
    out = out.expand_dims(dim="band")
    return out

def binarize_forest_loss(da: xr.DataArray) -> xr.DataArray:
    """Reduce Hansen GFC `lossyear` to binary forest-loss presence.

    The band is a year index (0 = no loss; 1..N encode the year of a
    stand-replacement event, 1 -> 2001). Stanley & Kirschbaum p.7 use presence or
    absence of any loss rather than its date or extent, so any non-zero code is
    loss and the year is discarded. That also means no release-specific ceiling is
    needed: the encoding's top end is irrelevant to `> 0`.

    Emits {0, 1}; `binary_fuzzy` in the heuristic model lifts the 0 off the floor
    before the gamma overlay (see Fig. 2's `Fuzzy Membership (4)`).
    """
    arr = np.asarray(da.compute()).astype(np.float32)
    out = np.where(np.isnan(arr), np.nan, (arr > 0).astype(np.float32))
    binarized = xr.DataArray(out.astype(np.float32), dims=da.dims, coords=da.coords)
    binarized.rio.write_crs(da.rio.crs, inplace=True)
    binarized.rio.write_transform(da.rio.transform(), inplace=True)
    binarized.rio.write_nodata(np.nan, inplace=True)
    return binarized


FEATURE_HANDLERS = {
    "calculate_slope": {"func": calculate_slope, "stage": "post"},
    "remap_lithology": {"func": remap_lithology, "stage": "pre"},
    "distance_to_features": {"func": distance_to_features, "stage": "post"},
    "binarize_forest_loss": {"func": binarize_forest_loss, "stage": "post"},
    }


def apply_feature(feature: str | None, stage: str, da: xr.DataArray) -> xr.DataArray:
    """Run the named feature handler if it is registered for this stage."""
    handler = FEATURE_HANDLERS.get(feature) if feature else None
    if not handler or handler["stage"] != stage:
        return da
    return handler["func"](da)

# ---------------------------------------------------------------------------
# Per-layer pipelines
# ---------------------------------------------------------------------------


def process_reference(layer_name: str, spec: dict, src_dir, bbox, save_intermediates: bool = False, tmp_dir: Path | None = None):
    """Create the reference grid from the flagged reference layer."""
    da = raster_utils.load_raster(src_dir, layer_name, tmp_dir)
    da = raster_utils.clip_to_bbox(da, bbox, buffer=CLIP_BUFFER_DEG)

    feature = spec.get("feature")
    da = apply_feature(feature, "pre", da)

    da = raster_utils.reproject(da, auto_utm=True, resampling=spec["resampling"])
    da = raster_utils.resample(da, TARGET_RESOLUTION_M, resampling=spec["resampling"])
    da = raster_utils.standardize_nodata(da, target_nodata=np.nan)

    if save_intermediates:
        save_to_tmp(da, tmp_dir, f"{layer_name}")

    da = apply_feature(feature, "post", da)

    return da.compute()


def process_layer(layer_name: str, spec: dict, src_dir, reference, bbox, save_intermediates: bool = False, tmp_dir: Path | None = None):
    """Load, standardize, and optionally transform a single layer to the reference grid."""
    feature = spec.get("feature")

    if spec.get("source") == "vector":
        gdf = raster_utils.load_vector(src_dir)
        da = raster_utils.rasterize_vector(gdf, reference, all_touched=True)
    else:
        da = raster_utils.load_raster(src_dir, layer_name, tmp_dir)
        da = raster_utils.clip_to_bbox(da, bbox, buffer=CLIP_BUFFER_DEG)

        da = apply_feature(feature, "pre", da)

        da = raster_utils.reproject(
            da,
            match_raster=reference,
            resampling=spec["resampling"],
        )

    da = raster_utils.standardize_nodata(da, target_nodata=np.nan)

    if save_intermediates:
        save_to_tmp(da, tmp_dir, f"{layer_name}")

    da = apply_feature(feature, "post", da)

    return da.compute() if hasattr(da, "compute") else da


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    save_intermediates = config["save_intermediates"]

    base_dir = config_path.parent
    bbox = tuple(float(x) for x in config["bbox"].split(","))
    out_dir = base_dir / "input_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = None
    if save_intermediates:
        tmp_dir = base_dir / "tmp_feature_engineering"
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def src_dir(spec):
        return (base_dir / config[spec["config_key"]]["output_dir"]).resolve()

    # 1. reference layer first; its grid defines everyone else's.
    ref_name = next(n for n, s in LAYERS.items() if s.get("is_reference"))
    ref_spec = LAYERS[ref_name]
    print(f"[reference] {ref_name}")
    reference = process_reference(
        ref_name,
        ref_spec,
        src_dir(ref_spec),
        bbox,
        save_intermediates,
        tmp_dir
    )
    reference.rio.to_raster(str(out_dir / f"{ref_name}.tif"))
    print(f"            shape={tuple(reference.shape)} crs={reference.rio.crs}")

    # 2. every other layer matched to the reference grid.
    for name, spec in LAYERS.items():
        if name == ref_name:
            continue
        print(f"[layer]     {name}")
        da = process_layer(
            name,
            spec,
            src_dir(spec),
            reference,
            bbox,
            save_intermediates,
            tmp_dir
        )
        da.rio.to_raster(str(out_dir / f"{name}.tif"))

    print(f"\nDone. Layers in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to top-level config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
















    


