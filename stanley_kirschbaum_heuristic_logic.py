import yaml
from pathlib import Path
import argparse
import numpy as np
import xarray as xr
from utils import raster_utils

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
        "fuzzy": "small",
        "midpoint": 500.0,
        "spread": 2.0
    },
    "lithology": {
        "config_key": "lithology_glim",
        "path": "lithology.tif",
        "fuzzy": "large",
        "midpoint": 0.5,
        "spread": 2.0
    },
    "roads": {
        "config_key": "pois_osm",
        "path": "roads.tif",
        "fuzzy": "large",
        "midpoint": 100.0,
        "spread": 2.0
    },
    "seismology": {
        "config_key": "seismology_gemf",
        "path": "seismology.tif",
        "fuzzy": "large",
        "midpoint": 0.5,
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


def gamma_fuzzy(arrays: list[xr.DataArray], gamma: float = 0.9) -> xr.DataArray:
    """Fuzzy gamma overlay (Eq. 1 in Stanley & Kirschbaum 2017).

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
        1 = Very Low   (≤ 0.11)
        2 = Low        (0.11 – 0.49)
        3 = Moderate   (0.49 – 0.671)
        4 = High       (0.671 – 0.75)
        5 = Very High  (> 0.75) 
    """
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
    
    # Re-mask nodata areas using the original valid data mask if necessary
    if susc_continuous.rio.nodata is not None:
        classes = classes.where(susc_continuous.notnull(), susc_continuous.rio.nodata)
        classes.rio.write_nodata(susc_continuous.rio.nodata, inplace=True)
        
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
        
        if spec["fuzzy"] == "large":
            fuzzy_da = large_fuzzy(img, spec["midpoint"], spec["spread"])
        elif spec["fuzzy"] == "small":
            fuzzy_da = small_fuzzy(img, spec["midpoint"], spec["spread"])
        else:
            raise ValueError(f"Unknown fuzzy type for layer {name}")
            
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



