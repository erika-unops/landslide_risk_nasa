import yaml
from pathlib import Path
import argparse
import numpy as np
import xarray as xr
from utils import raster_utils

# Class written where any input is missing. 0 is free: the paper's categories are 1-5.
NODATA_CLASS = 0

LAYERS: dict[str, dict] = {
    "dem": {
        "config_key": "dem_copernicus",
        "path": "dem.tif",
        "is_slope": True,
        "fuzzy": "large",
        "midpoint": 15.0,
        "spread": 5.0
    },
    "forest_loss":{
        "config_key": "gfc_hansen",
        "path": "forest_loss.tif",
        "fuzzy": "binary",
        "floor": 0.1
    },
    "lithology": {
        "config_key": "lithology_glim",
        "path": "lithology.tif",
        "fuzzy": "none"
    },
    "roads": {
        "config_key": "pois_osm",
        "path": "roads.tif",
        "fuzzy": "small",
        # midpoint 100 m follows Larsen & Parks (1997), cited by S&K p.8: landslide
        # scars were far more common within 85 m of a road, with effects beyond 100 m
        # much less pronounced. The floor is required because this is a distance
        # sigmoid rather than the paper's 1 km binary -- see apply_floor.
        "midpoint": 100.0,
        "spread": 2.0,
        "floor": 0.1
    },
    "seismology": {
        "config_key": "seismology_gemf",
        # GEM v2023.1 PGA, 475-yr return period, rock. S&K used distance-to-faults as
        # their "proxy for tectonic activity" (p.7); PGA substitutes for it here because
        # fault vectors are absent near Madagascar and PGA is absolute rather than
        # AOI-relative, so memberships stay comparable between study areas.
        #
        # midpoint 0.24 g is GSHAP's moderate/high hazard boundary, which shares this
        # raster's return period and rock reference. With spread 2 the curve reproduces
        # the GSHAP bands: 0.08 g (low/moderate) -> 0.10, 0.24 -> 0.50, 0.40 (high/very
        # high) -> 0.74. GEM itself publishes no class breaks, and Keefer's landslide
        # thresholds are event-triggering intensities (MMI, not PGA) rather than the
        # long-term rock damage this variable stands for -- so this is a judgment with a
        # citable basis, not a derived value.
        "path": "seismology.tif",
        "fuzzy": "large",
        "midpoint": 0.24,
        "spread": 2.0
    },
}


def large_fuzzy(x: xr.DataArray, midpoint: float, spread: float) -> xr.DataArray:
    """Sigmoidal 'large' membership — high values get membership near 1.

    This replicates the ArcGIS Fuzzy Membership "Large" function.
    Used for slope: susceptibility grows quickly between ~10° and ~30°[cite: 391].

    Parameters
    ----------
    x : xr.DataArray
        Input values (e.g. slope in degrees).
    midpoint : float
        The x value at which membership = 0.5.
    spread : float
        Controls steepness of the sigmoid. Higher = sharper transition.

    Returns
    -------
    xr.DataArray of floats in [0, 1]
    """
    # Note: 0 values raised to negative spread may trigger a runtime warning but will resolve to 0.
    return 1.0 / (1.0 + (x / midpoint) ** (-spread))


def small_fuzzy(x: xr.DataArray, midpoint: float, spread: float) -> xr.DataArray:
    """Sigmoidal 'small' membership — small values get membership near 1.

    Used for distance-to-faults: closer to fault → higher susceptibility.

    Parameters
    ----------
    x : xr.DataArray
        Input values (e.g. distance to nearest fault in meters).
    midpoint : float
        Distance at which membership = 0.5.
    spread : float
        Steepness. Higher = sharper drop-off.
        
    Returns
    -------
    xr.DataArray of floats in [0, 1]
    """
    return 1.0 / (1.0 + (x / midpoint) ** spread)


def apply_floor(mu: xr.DataArray, floor: float) -> xr.DataArray:
    """Rescale a [0, 1] membership onto [floor, 1].

    Every input to `gamma_fuzzy` must sit well above zero. The overlay multiplies the
    memberships, so an exact 0 annihilates the pixel outright and a near-zero drags it
    down hard through `fuzzy_and ** (1 - gamma)` — regardless of what every other layer
    says. S&K keep all of their overlay inputs inside roughly [0.1, 1]: Table 3's
    ratings floor at 0.1 (water bodies get 0.1, not 0), and the binary layers are lifted
    off zero by the Fuzzy Membership steps in Fig. 2.

    Two of our layers break that band without a floor:
      - forest_loss is binary, and raw {0, 1} would zero every unburned pixel.
      - roads is a distance sigmoid at 30 m, which decays to ~2e-5 far from any road.
        The paper's roads layer is binary at ~1 km pixels, so its non-road pixels get
        the floor instead — about 60x higher than our median of 0.0016. Our decay is a
        claim the paper's design never makes, and it suppresses the whole map.

    Slope is deliberately exempt: it enters as a product rather than through the
    overlay, and flat ground *should* zero the result (S&K p.10).

    The paper publishes neither the function nor the parameters behind its Fuzzy
    Membership steps, so `floor` is a judgment call; 0.1 matches the floor it chose
    everywhere it is visible.
    """
    if not 0.0 < floor < 1.0:
        raise ValueError(f"floor must be in (0, 1), got {floor}")
    return floor + (1.0 - floor) * mu


def gamma_fuzzy(arrays: list[xr.DataArray], gamma: float = 0.9) -> xr.DataArray:
    r"""Fuzzy gamma overlay (Eq. 1 in Stanley & Kirschbaum 2017).

        $$ \mu = (1 - \prod(1 - \mu_i))^\gamma \times (\prod \mu_i)^{1-\gamma} $$

    This interpolates between fuzzy OR (γ=1) and fuzzy AND (γ=0)[cite: 394].
    γ = 0.9 emphasises the OR-like (algebraic sum) component, meaning
    no single factor is strictly necessary for susceptibility[cite: 500, 501].

    Parameters
    ----------
    arrays : list of xr.DataArray
        Each array is a fuzzy membership layer in [0, 1].
    gamma : float
        Gamma parameter in [0, 1]. Paper uses 0.9[cite: 501].

    Returns
    -------
    xr.DataArray of float in [0, 1]
    """
    # Concatenate the arrays along a new 'layer' dimension
    stack = xr.concat(arrays, dim="layer")

    # ∏ μᵢ (fuzzy AND component)
    fuzzy_and = stack.prod(dim="layer")

    # 1 - ∏(1 - μᵢ) (fuzzy OR / algebraic sum component)
    fuzzy_or = 1.0 - (1.0 - stack).prod(dim="layer")

    # Gamma combination
    result = (fuzzy_or ** gamma) * (fuzzy_and ** (1.0 - gamma))
    return result


def susceptibility(slope: xr.DataArray, other_factors: xr.DataArray,
                   thresholds: tuple = (0.11, 0.49, 0.671, 0.75)) -> xr.DataArray:
    """Classify continuous susceptibility into 5 categories.

    Categories (Stanley & Kirschbaum 2017):
        0 = No Data    (any input missing)
        1 = Very Low   (≤ 0.11)
        2 = Low        (0.11 – 0.49)
        3 = Moderate   (0.49 – 0.671)
        4 = High       (0.671 – 0.75)
        5 = Very High  (> 0.75)
    """
    # Capture spatial metadata up front. Plain xarray ops (arithmetic,
    # xr.where, reductions) are not CRS-aware and drop rioxarray's spatial_ref
    # coordinate / grid_mapping attribute, so the result would otherwise be
    # written as a GeoTIFF with no CRS. `slope` is a reliable source: it only
    # went through unary arithmetic, which preserves the metadata.
    crs = slope.rio.crs
    transform = slope.rio.transform()

    # Slope gradient is applied as a critical predictor separate from the gamma overlay
    susc_continuous = slope * other_factors

    t1, t2, t3, t4 = thresholds

    # Initialize array with 1 (Very Low)
    classes = xr.full_like(susc_continuous, 1, dtype=np.uint8)

    # Apply thresholds iteratively using xarray's where
    classes = xr.where(susc_continuous > t1, 2, classes)
    classes = xr.where(susc_continuous > t2, 3, classes)
    classes = xr.where(susc_continuous > t3, 4, classes)
    classes = xr.where(susc_continuous > t4, 5, classes)

    # Re-mask nodata areas using the original valid data mask.
    #
    # This cannot be guarded on `susc_continuous.rio.nodata`: `slope * other_factors`
    # is plain xarray arithmetic, which strips rio metadata, so that attribute is
    # always None here and the mask would never be applied. Every comparison above is
    # False for NaN, so an unmasked no-data pixel silently falls through to class 1 and
    # claims stable terrain where there is no data at all.
    #
    # This does NOT mask the sea. Copernicus encodes ocean as elevation 0.0 rather than
    # nodata, so sea pixels carry a valid ~0 deg slope, are never NaN, and legitimately
    # reach class 1 here. Masking them needs a land/sea mask upstream, not this guard.
    classes = xr.where(susc_continuous.notnull(), classes, NODATA_CLASS).astype(np.uint8)

    # Re-attach the CRS / transform that the xarray ops above stripped off.
    classes = classes.rio.write_crs(crs)
    classes = classes.rio.write_transform(transform)
    classes.rio.write_nodata(NODATA_CLASS, inplace=True)

    return classes


def main(config_path: Path) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    save_intermediates = config.get("save_intermediates", False)

    base_dir = config_path.parent
    input_dir = base_dir / "input_data"
    output_dir = base_dir / "output_data" 
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tmp_dir = None
    if save_intermediates:
        tmp_dir = base_dir / "tmp_heuristic"
        tmp_dir.mkdir(parents=True, exist_ok=True)

    # Process each layer and convert to fuzzy membership
    for name, spec in LAYERS.items():
        print(f"Processing [layer]: {name}")
        img = raster_utils.load_raster(input_dir / spec["path"], layer_name=name, tmp_dir=tmp_dir)
        
        kind = spec["fuzzy"]
        if kind == "large":
            fuzzy_da = large_fuzzy(img, spec["midpoint"], spec["spread"])
        elif kind == "small":
            fuzzy_da = small_fuzzy(img, spec["midpoint"], spec["spread"])
        elif kind in ("binary", "none"):
            # Already a membership value: binary presence/absence, or a rating table.
            # Fig. 2 feeds the geologic ranking straight into the overlay this way.
            fuzzy_da = img
        else:
            raise ValueError(f"Unknown fuzzy type for layer {name}")

        # Lift the membership off zero where the layer declares a floor. See apply_floor.
        floor = spec.get("floor")
        if floor is not None:
            fuzzy_da = apply_floor(fuzzy_da, floor)
            
        if save_intermediates:
            out_path = tmp_dir / f"fuzzy_{name}.tif"
            fuzzy_da.rio.to_raster(out_path)
            LAYERS[name]["fuzzy_data"] = out_path  # store path to manage memory
        else:
            LAYERS[name]["fuzzy_data"] = fuzzy_da  # store DataArray in memory

    # Separate the slope layer from other layers
    slope_key = next(n for n, s in LAYERS.items() if s.get("is_slope"))
    
    if save_intermediates:
        slope_da = raster_utils.load_raster(LAYERS[slope_key]["fuzzy_data"], layer_name="slope_fuzzy")
    else:
        slope_da = LAYERS[slope_key]["fuzzy_data"]

    # Gather the non-slope DataArrays
    other_factors_das = []
    for name, spec in LAYERS.items():
        if name == slope_key:
            continue
            
        if save_intermediates:
            other_factors_das.append(raster_utils.load_raster(spec["fuzzy_data"], layer_name=f"{name}_fuzzy"))
        else:
            other_factors_das.append(spec["fuzzy_data"])
    
    # Calculate Gamma Overlay
    print("Calculating Gamma Overlay...")
    gamma_da = gamma_fuzzy(other_factors_das)
    
    if save_intermediates:
        gamma_da.rio.to_raster(tmp_dir / "gamma_overlay.tif")

    # Final Susceptibility
    print("Calculating final susceptibility categories...")
    susc_classes = susceptibility(slope_da, gamma_da)
    
    output_path = output_dir / "susceptibility.tif"
    susc_classes.rio.to_raster(output_path)
    print(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stanley and Kirschbaum (2017) Landslide Susceptibility logic')
    parser.add_argument("--config", type=str, help='Path to config file')
    args = parser.parse_args()
    main(config_path=Path(args.config))



