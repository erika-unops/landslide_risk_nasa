"""
raster_utils.py
Thin wrappers around rioxarray / rasterio for common satellite imagery
preprocessing: VRT creation, tiling, resolution harmonization, nodata handling,
bbox clipping, and CRS ↔ UTM conversion.

Dependencies:
    pip install rioxarray rasterio xarray numpy pyproj
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence
import tempfile
import dask
import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.merge import merge as rio_merge

import geopandas as gpd

# GDAL/rasterio file handles are thread-affine; dask's default threaded
# scheduler tears its worker threads down at interpreter exit and can end up
# closing a handle from the wrong thread, which surfaces as noisy (but
# harmless) "Error in sys.excepthook" spam on exit. Run compute graphs on the
# main thread instead — chunks=True stays lazy, only the execution scheduler
# changes.
dask.config.set(scheduler="synchronous")


# ---------------------------------------------------------------------------
# Load rasters (point to a directory or a file path)
# ---------------------------------------------------------------------------

def load_raster(src, layer_name: str, tmp_dir: Path | None = None) -> xr.DataArray:
    """Lazily open the GeoTIFFs in ``src_dir``, mosaicking first if there are many.

    ``src`` may be a directory (scanned for GeoTIFFs, mosaicking if there are
    several) or a path to a single raster file, opened directly.
    """
    if isinstance(src, Path) and src.is_dir():
        images = collect_geotiffs(src)
        if not images:
            raise FileNotFoundError(f"No GeoTIFFs found in {src}")
        if len(images) > 1:
            if tmp_dir is None:
                tmp_dir = Path(tempfile.gettempdir())
            else:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                
            path = make_vrt(images, tmp_dir / f"{layer_name}_mosaic.tif")
        else:
            path = images[0]
    else:
        path = src
    return rioxarray.open_rasterio(str(path), masked=True, chunks=True)

# ---------------------------------------------------------------------------
# Make VRT (mosaic of adjacent/overlapping tiles into a single virtual raster)
# ---------------------------------------------------------------------------

def make_vrt(
    src_paths: Sequence[str | Path],
    dst_path: str | Path,
    *,
    resolution: str = "highest",
    separate: bool = False,
    src_nodata: float | None = None,
    vrt_nodata: float | None = None,
    band_list: list[int] | None = None,
) -> Path:
    """Build a VRT-like mosaic from a list of rasters using rasterio only.

    This replaces ``gdal.BuildVRT`` (which requires the standalone GDAL Python
    bindings that are painful to install on Windows). Under the hood it uses
    ``rasterio.merge`` and writes a real GeoTIFF — not a lazy XML VRT — but the
    output is a drop-in replacement when the goal is "give me one raster I can
    open with rioxarray".

    Parameters
    ----------
    src_paths : list of paths
        Input raster files.
    dst_path : path
        Output file. If suffix is ``.vrt`` it will be silently changed to
        ``.tif`` since this implementation does not emit XML VRTs.
    resolution : {"highest", "lowest", "average", "user"}
        How to resolve differing resolutions. Maps to ``rasterio.merge``'s
        ``res`` argument ("highest" -> finest, "lowest" -> coarsest).
    separate : bool
        If True, stack inputs as separate bands (time-series style).
    src_nodata / vrt_nodata : float, optional
        Override source / output nodata values.
    band_list : list[int], optional
        Subset of bands to read from each input (1-indexed).

    Returns
    -------
    Path to the created mosaic raster.
    """
    dst_path = Path(dst_path)
    if dst_path.suffix.lower() == ".vrt":
        dst_path = dst_path.with_suffix(".tif")

    src_paths = [str(p) for p in src_paths]
    if not src_paths:
        raise ValueError("make_vrt: src_paths is empty")

    # Map resolution mode to rasterio.merge's `res` kwarg.
    if resolution == "highest":
        merge_res = None  # rasterio.merge default = finest input
    elif resolution == "lowest":
        # compute coarsest pixel size across inputs
        pixel_sizes = []
        for p in src_paths:
            with rasterio.open(p) as s:
                pixel_sizes.append(max(abs(s.transform.a), abs(s.transform.e)))
        merge_res = max(pixel_sizes)
    elif resolution == "average":
        pixel_sizes = []
        for p in src_paths:
            with rasterio.open(p) as s:
                pixel_sizes.append((abs(s.transform.a) + abs(s.transform.e)) / 2)
        merge_res = sum(pixel_sizes) / len(pixel_sizes)
    else:
        merge_res = None

    indexes = band_list if band_list else None

    if separate:
        # Stack each input as its own band.
        arrays = []
        transform = None
        crs = None
        for p in src_paths:
            arr, transform = rio_merge(
                [p], res=merge_res, nodata=src_nodata, indexes=indexes
            )
            arrays.append(arr)
            with rasterio.open(p) as s:
                crs = s.crs
        mosaic = np.concatenate(arrays, axis=0)
    else:
        mosaic, transform = rio_merge(
            src_paths, res=merge_res, nodata=src_nodata, indexes=indexes
        )
        with rasterio.open(src_paths[0]) as s:
            crs = s.crs

    out_nodata = vrt_nodata if vrt_nodata is not None else src_nodata
    profile = {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": mosaic.shape[0],
        "dtype": mosaic.dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "compress": "deflate",
    }
    if out_nodata is not None:
        profile["nodata"] = out_nodata

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(mosaic)

    return dst_path


# ---------------------------------------------------------------------------
# Break rasters into tiles (size management)
# ---------------------------------------------------------------------------

def tile_raster(
    src_path: str | Path,
    out_dir: str | Path,
    *,
    tile_size: int = 2048,
    overlap: int = 0,
    prefix: str = "tile",
) -> list[Path]:
    """Split a raster into tiles of fixed pixel dimensions.

    Parameters
    ----------
    src_path : path
        Input raster.
    out_dir : path
        Directory to write tiles into (created if needed).
    tile_size : int
        Width/height of each tile in pixels.
    overlap : int
        Pixel overlap between adjacent tiles (useful for ML inference).
    prefix : str
        Filename prefix for tiles.

    Returns
    -------
    List of Paths to tile files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[Path] = []

    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        step = tile_size - overlap

        for row_off in range(0, src.height, step):
            for col_off in range(0, src.width, step):
                win_h = min(tile_size, src.height - row_off)
                win_w = min(tile_size, src.width - col_off)
                window = rasterio.windows.Window(col_off, row_off, win_w, win_h)
                transform = rasterio.windows.transform(window, src.transform)

                tile_meta = meta.copy()
                tile_meta.update(
                    width=win_w,
                    height=win_h,
                    transform=transform,
                )

                tile_path = out_dir / f"{prefix}_{row_off}_{col_off}.tif"
                with rasterio.open(tile_path, "w", **tile_meta) as dst:
                    dst.write(src.read(window=window))
                tiles.append(tile_path)

    return tiles


# ---------------------------------------------------------------------------
# Scale / resample to a common resolution
# ---------------------------------------------------------------------------

def harmonize_resolution(
    datasets: Sequence[xr.DataArray | xr.Dataset],
    target_res: float | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> list[xr.DataArray | xr.Dataset]:
    """Reproject / resample a list of rioxarray objects to a common resolution.

    Parameters
    ----------
    datasets : sequence of xr.DataArray or xr.Dataset
        Must have CRS set via rioxarray (.rio.crs).
    target_res : float, optional
        Target resolution in the datasets' CRS units.
        If None, uses the finest (smallest) resolution among inputs.
    resampling : rasterio.enums.Resampling
        Resampling method.

    Returns
    -------
    List of reprojected datasets at uniform resolution.
    """
    if target_res is None:
        resolutions = []
        for ds in datasets:
            rx, ry = ds.rio.resolution()
            resolutions.append(min(abs(rx), abs(ry)))
        target_res = min(resolutions)

    out = []
    for ds in datasets:
        rx, ry = ds.rio.resolution()
        if not math.isclose(abs(rx), target_res, rel_tol=1e-6):
            ds = ds.rio.reproject(
                ds.rio.crs,
                resolution=target_res,
                resampling=resampling,
            )
        out.append(ds)
    return out


# ---------------------------------------------------------------------------
# Handle NaN / nodata
# ---------------------------------------------------------------------------

def standardize_nodata(
    da: xr.DataArray,
    target_nodata: float = np.nan,
) -> xr.DataArray:
    """Replace source nodata with a consistent value (default NaN).

    Also converts integer dtypes to float when target_nodata is NaN.
    Sets .rio.nodata / .rio.encoded_nodata accordingly.
    """
    src_nodata = da.rio.nodata
    if src_nodata is None:
        src_nodata = da.rio.encoded_nodata

    if np.isnan(target_nodata) and np.issubdtype(da.dtype, np.integer):
        da = da.astype(np.float32)

    if src_nodata is not None:
        if np.isnan(src_nodata):
            mask = np.isnan(da.values)
        else:
            mask = da.values == src_nodata
        da = da.where(~xr.DataArray(mask, dims=da.dims, coords=da.coords), other=target_nodata)

    da = da.rio.write_nodata(target_nodata)
    return da


def fill_nodata_interpolate(
    da: xr.DataArray,
    method: str = "nearest",
    max_gap: int | None = None,
) -> xr.DataArray:
    """Fill nodata pixels via spatial interpolation.

    Parameters
    ----------
    method : {"nearest", "linear", "cubic"}
    max_gap : int, optional
        Max number of consecutive NaN pixels to fill along each axis.
    """
    filled = da.interpolate_na(dim="x", method=method, limit=max_gap)
    filled = filled.interpolate_na(dim="y", method=method, limit=max_gap)
    return filled


# ---------------------------------------------------------------------------
# Clip to bounding box
# ---------------------------------------------------------------------------

def clip_to_bbox(
    da: xr.DataArray | xr.Dataset,
    bbox: tuple[float, float, float, float],
    bbox_crs: str | CRS = "EPSG:4326",
    buffer: float = 0.0,
) -> xr.DataArray | xr.Dataset:
    """Clip a rioxarray object to a bounding box.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
    bbox : (min_x, min_y, max_x, max_y)
        Bounding box coordinates.
    bbox_crs : str or CRS
        CRS of the bbox. Reprojected to the raster's CRS automatically.
    buffer : float
        Buffer to expand the bbox by, in units of ``bbox_crs``. Useful when a
        downstream reproject/resample step would otherwise produce NaN halos at
        the edges. Default 0 (no buffer).
    """
    if buffer:
        minx, miny, maxx, maxy = bbox
        bbox = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
    return da.rio.clip_box(*bbox, crs=bbox_crs)


def clip_to_geometry(
    da: xr.DataArray | xr.Dataset,
    geometry,
    geometry_crs: str | CRS = "EPSG:4326",
    all_touched: bool = True,
) -> xr.DataArray | xr.Dataset:
    """Clip a rioxarray object to a vector geometry (GeoJSON-like or shapely).

    Parameters
    ----------
    geometry : shapely geometry, GeoJSON dict, or list thereof
    geometry_crs : CRS of the geometry
    all_touched : include pixels touched by geometry edge
    """
    if not isinstance(geometry, (list, tuple)):
        geometry = [geometry]
    return da.rio.clip(geometry, crs=geometry_crs, all_touched=all_touched)


# ---------------------------------------------------------------------------
# CRS → UTM converter
# ---------------------------------------------------------------------------

def get_utm_crs(
    lon: float,
    lat: float,
) -> CRS:
    """Return the UTM CRS for a given lon/lat point.

    Handles both hemispheres. Uses standard 6° zones.
    """
    zone_number = int((lon + 180) / 6) + 1
    is_north = lat >= 0
    epsg = 32600 + zone_number if is_north else 32700 + zone_number
    return CRS.from_epsg(epsg)


def get_utm_crs_from_raster(da: xr.DataArray | xr.Dataset) -> CRS:
    """Infer UTM CRS from a rioxarray object's spatial extent."""
    bounds = da.rio.bounds()  # (left, bottom, right, top) in raster CRS
    # transform centroid to EPSG:4326 if needed
    src_crs = da.rio.crs
    if src_crs and not CRS(src_crs).is_geographic:
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        lon, lat = transformer.transform(cx, cy)
    else:
        lon = (bounds[0] + bounds[2]) / 2
        lat = (bounds[1] + bounds[3]) / 2
    return get_utm_crs(lon, lat)


def reproject(
    da: xr.DataArray | xr.Dataset,
    *,
    target_crs: str | CRS | None = None,
    match_raster: xr.DataArray | xr.Dataset | None = None,
    auto_utm: bool = False,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray | xr.Dataset:
    """Reproject a rioxarray object using one of three modes.

    Exactly one of ``target_crs``, ``match_raster``, or ``auto_utm`` must be
    provided. Resolution is **not** changed by this function — use
    :func:`resample` (single raster) or :func:`harmonize_resolution` (list)
    for that, unless ``match_raster`` is used (in which case the reference
    grid dictates resolution, transform, and shape simultaneously).

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Source raster.
    target_crs : str or CRS, optional
        Reproject to an explicit CRS (e.g. ``"EPSG:32738"``). Resolution is
        preserved from the source.
    match_raster : xr.DataArray or xr.Dataset, optional
        Reproject to exactly match the CRS, transform, and shape of another
        raster. Use this to guarantee pixel-perfect alignment for downstream
        per-pixel math.
    auto_utm : bool
        If True, infer the appropriate UTM zone from the raster's extent and
        reproject to it. Resolution is preserved from the source.
    resampling : rasterio.enums.Resampling
        Resampling method (default bilinear). Use ``Resampling.nearest`` for
        categorical data.

    Returns
    -------
    Reprojected DataArray / Dataset.
    """
    modes = sum(x is not None and x is not False for x in
                (target_crs, match_raster, auto_utm or None))
    if modes != 1:
        raise ValueError(
            "reproject: specify exactly one of target_crs, match_raster, "
            "or auto_utm=True"
        )

    if match_raster is not None:
        return da.rio.reproject_match(match_raster, resampling=resampling)

    if auto_utm:
        dst_crs = get_utm_crs_from_raster(da)
    else:
        dst_crs = CRS.from_user_input(target_crs)

    return da.rio.reproject(dst_crs=dst_crs, resampling=resampling)


# ---------------------------------------------------------------------------
# Collect GeoTIFFs
# ---------------------------------------------------------------------------

def collect_geotiffs(
    directory: str | Path,
    recursive: bool = True,
) -> list[Path]:
    """Collect all GeoTIFF files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to search for GeoTIFFs.
    recursive : bool
        If True, searches subdirectories recursively. Default is True.

    Returns
    -------
    list of Path
        Sorted list of GeoTIFF file paths (*.tif, *.tiff).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    pattern = "**/*.tif*" if recursive else "*.tif*"
    geotiffs = sorted(directory.glob(pattern))

    # Filter to only .tif and .tiff files (exclude .aux.xml, etc.)
    geotiffs = [p for p in geotiffs if p.suffix.lower() in ('.tif', '.tiff')]

    return geotiffs


# ---------------------------------------------------------------------------
# Load a vector layer (point to a directory holding a plugin's output)
# ---------------------------------------------------------------------------

def load_vector(directory: str | Path, pattern: str = "*_combined.geojson") -> gpd.GeoDataFrame:
    """Load the vector file matching ``pattern`` out of a plugin's output directory."""
    directory = Path(directory)
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No vector file matching {pattern!r} found in {directory}")
    return gpd.read_file(matches[0])


# ---------------------------------------------------------------------------
# Resample a single raster to a target resolution (CRS preserved)
# ---------------------------------------------------------------------------

def resample(
    da: xr.DataArray | xr.Dataset,
    target_resolution: float,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray | xr.Dataset:
    """Resample a raster to a target resolution without changing its CRS.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Source raster (already opened via rioxarray).
    target_resolution : float
        Target resolution in map units of the raster's CRS (meters if
        projected, degrees if geographic).
    resampling : rasterio.enums.Resampling
        Resampling method (default bilinear; use nearest for categorical).

    Returns
    -------
    Resampled DataArray / Dataset.
    """
    return da.rio.reproject(
        da.rio.crs,
        resolution=target_resolution,
        resampling=resampling,
    )


# ---------------------------------------------------------------------------
# Rasterize a vector layer onto an existing raster grid
# ---------------------------------------------------------------------------

def rasterize_vector(
    gdf,
    reference: xr.DataArray | xr.Dataset,
    *,
    attribute: str | None = None,
    fill: float = 0,
    burn_value: float = 1,
    all_touched: bool = False,
    dtype: str = "float32",
) -> xr.DataArray:
    """Rasterize a GeoDataFrame onto the grid of a reference raster.

    The result has identical CRS, transform, and shape to ``reference`` so it
    can be stacked with other layers without further resampling.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Vector features. Will be reprojected to ``reference``'s CRS if needed.
    reference : xr.DataArray or xr.Dataset
        Raster whose grid defines the output.
    attribute : str, optional
        Column to burn into the raster. If None, ``burn_value`` is used for
        every feature.
    fill : float
        Value for pixels not covered by any feature. Default 0.
    burn_value : float
        Value burned when ``attribute`` is None. Default 1.
    all_touched : bool
        If True, every pixel touched by a geometry is burned. Useful for
        thin linear features (roads, faults).
    dtype : str
        Output dtype. Default ``"float32"``.

    Returns
    -------
    xr.DataArray with the same spatial dims as ``reference``.
    """
    if gpd is None:
        raise ImportError("rasterize_vector requires geopandas")

    ref_crs = reference.rio.crs
    if gdf.crs is None:
        raise ValueError("rasterize_vector: input GeoDataFrame has no CRS")
    if ref_crs is None:
        raise ValueError("rasterize_vector: reference raster has no CRS")
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)

    transform = reference.rio.transform()
    height = int(reference.rio.height)
    width = int(reference.rio.width)

    if attribute is not None:
        shapes = ((geom, float(val)) for geom, val in
                  zip(gdf.geometry, gdf[attribute]))
    else:
        shapes = ((geom, burn_value) for geom in gdf.geometry)

    arr = rio_rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )

    out = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": reference["y"].values,
            "x": reference["x"].values,
        },
    )
    out.rio.write_crs(ref_crs, inplace=True)
    out.rio.write_transform(transform, inplace=True)
    out.rio.write_nodata(fill, inplace=True)
    return out


