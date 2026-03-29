"""
extract.py — Pull raw band/pixel values from satellite sources at point locations.

How it works:
  1. You provide a CSV/DataFrame with Latitude, Longitude (your sample points)
  2. We download the satellite raster for your bbox + date range
  3. We use .sel(method="nearest") to grab the pixel value at each of YOUR points
  4. Output DataFrame has exactly the same number of rows as your input CSV

Data quality:
  - Nodata pixels (0 in Sentinel, 0 in Landsat) are replaced with NaN
  - Landsat thermal is properly scaled: DN → Kelvin → °C, with nodata masked first
  - Sentinel compositing (median) naturally reduces cloud contamination

Cache rules:
  - DEM + Buildings: keyed by (bbox, points) → immune to date/resolution changes
  - Sentinel + Landsat: keyed by (bbox, points, date, resolution) → re-run when these change
"""

import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rio
import rasterio
import rasterio.transform
from rasterio.merge import merge as rio_merge
from odc.stac import stac_load
import pystac_client
import planetary_computer

from .cache import cache_get, cache_put, DEFAULT_CACHE_DIR


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_points(source):
    """
    Load point locations from a CSV path, URL, or existing DataFrame.

    Only normalizes Latitude/Longitude column casing.
    All other columns (UHI_Class, etc.) are preserved exactly as-is.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    col_map = {}
    for c in df.columns:
        if c.lower() == "latitude":
            col_map[c] = "Latitude"
        elif c.lower() == "longitude":
            col_map[c] = "Longitude"
    df.rename(columns=col_map, inplace=True)
    return df


def points_fingerprint(pts_df):
    """Short string identifying a points DataFrame for cache keys."""
    n = len(pts_df)
    first_lat = pts_df["Latitude"].iloc[0]
    last_lon = pts_df["Longitude"].iloc[-1]
    return f"{n}_{first_lat}_{last_lon}"


def _bbox_to_stac(bbox):
    """(south, west, north, east) → (west, south, east, north) for STAC API."""
    south, west, north, east = bbox
    return (west, south, east, north)


def _stac_client():
    """Connect to Microsoft Planetary Computer STAC catalog (free, no auth)."""
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )


# ---------------------------------------------------------------------------
# Sentinel-2 L2A
# ---------------------------------------------------------------------------

SENTINEL_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
]


def extract_sentinel(
    pts_df,
    bbox,
    time_window,
    resolution=250,
    bands=None,
    max_cloud=30,
    composite="median",
    use_cache=True,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """
    Extract Sentinel-2 L2A band values at each point location.

    Steps:
      1. Search Planetary Computer for scenes in bbox + date range
      2. Load all scenes into xarray (chunked, lazy)
      3. Mask nodata (raw DN=0 → NaN) so compositing ignores bad pixels
      4. Compute median composite across all scenes → one clean raster
      5. Extract the nearest pixel value at each (lat, lon) from your CSV

    Output: DataFrame with same row count as pts_df.
    Columns: Latitude, Longitude, B01, B02, ..., B12

    Parameters
    ----------
    pts_df : DataFrame with Latitude, Longitude columns
    bbox : (south, west, north, east) in degrees
    time_window : "2024-01-15/2024-02-15"
    resolution : pixel size in meters (10–1000)
    max_cloud : max cloud cover % for scene filtering
    composite : "median" or "mean"
    """
    bands = bands or SENTINEL_BANDS

    cache_params = dict(
        bbox=bbox, time_window=time_window,
        resolution=resolution, pid=points_fingerprint(pts_df),
    )
    if use_cache:
        cached = cache_get("sentinel", cache_dir, **cache_params)
        if cached is not None:
            return cached

    stac_bbox = _bbox_to_stac(bbox)
    scale = resolution / 111320.0

    stac = _stac_client()
    search = stac.search(
        bbox=stac_bbox,
        datetime=time_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.get_items())
    print(f"[Sentinel-2] {len(items)} scenes | res={resolution}m | {time_window}")
    if not items:
        return None

    # load all scenes (lazy via dask chunks)
    data = stac_load(
        items, bands=bands, crs="EPSG:4326", resolution=scale,
        chunks={"x": 2048, "y": 2048}, dtype="uint16",
        patch_url=planetary_computer.sign, bbox=stac_bbox,
    )

    # mask nodata: Sentinel-2 L2A uses 0 as nodata/fill value
    # without this, 0-pixels contaminate the median and produce bad indices
    data = data.where(data > 0)

    # composite collapses multiple scenes into one raster
    # median is robust to outliers (clouds that slipped past the filter)
    comp = getattr(data, composite)(dim="time").compute()

    # extract pixel values at each point
    lats = xr.DataArray(pts_df["Latitude"].values, dims="points")
    lons = xr.DataArray(pts_df["Longitude"].values, dims="points")

    df = pd.DataFrame({
        "Latitude": pts_df["Latitude"].values,
        "Longitude": pts_df["Longitude"].values,
    })
    for band in bands:
        df[band] = comp[band].sel(
            longitude=lons, latitude=lats, method="nearest"
        ).values.astype(float)

    del data, comp

    # report how many points got valid data
    valid = df[bands].notna().all(axis=1).sum()
    print(f"[Sentinel-2] {len(df)} points × {len(bands)} bands ({valid} valid, {len(df)-valid} have NaN)")
    if use_cache:
        cache_put(df, "sentinel", cache_dir, **cache_params)
    return df


# ---------------------------------------------------------------------------
# Landsat-8 (visible/NIR + thermal)
# ---------------------------------------------------------------------------

LANDSAT_VIS_BANDS = ["red", "green", "blue", "nir08"]
LANDSAT_THERMAL = ["lwir11"]

# Landsat Collection 2 Level 2 scale factors (from USGS documentation)
LANDSAT_VIS_SCALE = 0.0000275
LANDSAT_VIS_OFFSET = -0.2
LANDSAT_THERM_SCALE = 0.00341802
LANDSAT_THERM_OFFSET = 149.0  # converts to Kelvin


def extract_landsat(
    pts_df,
    bbox,
    time_window,
    resolution=250,
    max_cloud=50,
    scene_index=0,
    use_cache=True,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """
    Extract Landsat-8 visible/NIR + thermal at each point.

    Single scene (not composite) because LST from different dates can't be
    meaningfully averaged — each scene captures a specific thermal snapshot.

    Data quality fix for lwir11:
      - Raw DN=0 is nodata (fill value outside scene footprint or under clouds)
      - We mask these BEFORE applying scale factors
      - Without this: 0 * 0.00341802 + 149.0 - 273.15 = -124°C (garbage)
      - With mask: those pixels become NaN instead

    Output: Latitude, Longitude, red, green, blue, nir08, lwir11
      - vis bands: surface reflectance (0–1 range)
      - lwir11: brightness temperature in °C (typical range 15–50°C for urban)
    """
    cache_params = dict(
        bbox=bbox, time_window=time_window,
        resolution=resolution, pid=points_fingerprint(pts_df),
    )
    if use_cache:
        cached = cache_get("landsat", cache_dir, **cache_params)
        if cached is not None:
            return cached

    stac_bbox = _bbox_to_stac(bbox)
    scale = resolution / 111320.0

    stac = _stac_client()
    search = stac.search(
        bbox=stac_bbox,
        datetime=time_window,
        collections=["landsat-c2-l2"],
        query={"eo:cloud_cover": {"lt": max_cloud}, "platform": {"in": ["landsat-8"]}},
    )
    items = list(search.get_items())
    print(f"[Landsat] {len(items)} scenes | res={resolution}m | {time_window}")
    if not items:
        return None

    # load vis and thermal separately (different scale factors)
    data_vis = stac_load(
        items, bands=LANDSAT_VIS_BANDS, crs="EPSG:4326", resolution=scale,
        chunks={"x": 2048, "y": 2048}, dtype="uint16",
        patch_url=planetary_computer.sign, bbox=stac_bbox,
    )
    data_therm = stac_load(
        items, bands=LANDSAT_THERMAL, crs="EPSG:4326", resolution=scale,
        chunks={"x": 2048, "y": 2048}, dtype="uint16",
        patch_url=planetary_computer.sign, bbox=stac_bbox,
    )

    # MASK NODATA BEFORE scaling
    # Landsat C2L2 uses 0 as fill/nodata for pixels outside scene or under clouds
    # Without masking: 0 * scale + offset = garbage values (-124°C for thermal)
    data_vis = data_vis.where(data_vis > 0)
    data_therm = data_therm.where(data_therm > 0)

    # apply Landsat C2L2 scale factors to valid pixels only
    # vis: DN → surface reflectance (0 to ~0.5 range)
    # thermal: DN → brightness temperature in °C
    #
    # IMPORTANT: lwir11 output is BRIGHTNESS TEMPERATURE, not Land Surface Temperature.
    # BT is what the satellite sensor measures. To get actual LST you need an
    # emissivity correction (done in features.py landsat_features()).
    # Do NOT apply the DN→°C conversion again downstream — it's done here once.
    data_vis = data_vis.astype(float) * LANDSAT_VIS_SCALE + LANDSAT_VIS_OFFSET
    data_therm = data_therm.astype(float) * LANDSAT_THERM_SCALE + LANDSAT_THERM_OFFSET - 273.15

    # pick a single scene
    d_vis = data_vis.isel(time=scene_index).compute()
    d_therm = data_therm.isel(time=scene_index).compute()

    # extract at points
    lats = xr.DataArray(pts_df["Latitude"].values, dims="points")
    lons = xr.DataArray(pts_df["Longitude"].values, dims="points")

    df = pd.DataFrame({
        "Latitude": pts_df["Latitude"].values,
        "Longitude": pts_df["Longitude"].values,
    })
    for band in LANDSAT_VIS_BANDS:
        df[band] = d_vis[band].sel(
            latitude=lats, longitude=lons, method="nearest"
        ).values.astype(float)
    df["lwir11"] = d_therm["lwir11"].sel(
        latitude=lats, longitude=lons, method="nearest"
    ).values.astype(float)

    del data_vis, data_therm, d_vis, d_therm

    # report quality
    valid_thermal = df["lwir11"].notna().sum()
    bad_thermal = ((df["lwir11"] < -50) | (df["lwir11"] > 80)).sum()
    print(f"[Landsat] {len(df)} points × {len(LANDSAT_VIS_BANDS)+1} bands")
    print(f"[Landsat] lwir11: {valid_thermal} valid, {bad_thermal} out-of-range (< -50 or > 80°C)")

    if use_cache:
        cache_put(df, "landsat", cache_dir, **cache_params)
    return df


# ---------------------------------------------------------------------------
# DEM — Copernicus GLO-30 (STATIC: no date, no resolution)
# ---------------------------------------------------------------------------

def extract_dem(
    pts_df,
    bbox,
    use_cache=True,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """
    Extract elevation at each point from Copernicus DEM GLO-30.

    Static data — no date or resolution parameter.
    Only re-runs if bbox or points change.

    Output: Latitude, Longitude, elevation (meters above sea level)
    """
    cache_params = dict(bbox=bbox, pid=points_fingerprint(pts_df))
    if use_cache:
        cached = cache_get("dem", cache_dir, **cache_params)
        if cached is not None:
            return cached

    stac_bbox = _bbox_to_stac(bbox)

    stac = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = stac.search(bbox=stac_bbox, collections=["cop-dem-glo-30"])
    items = list(search.get_items())
    print(f"[DEM] {len(items)} tiles")

    datasets = [
        rasterio.open(planetary_computer.sign(item.assets["data"]).href)
        for item in items
    ]
    merged_array, merged_transform = rio_merge(datasets)
    merged_array = merged_array[0].astype("float32")
    merged_array[merged_array < -1000] = float("nan")

    for ds in datasets:
        ds.close()

    n_rows, n_cols = merged_array.shape
    west, south, east, north = rasterio.transform.array_bounds(
        n_rows, n_cols, merged_transform
    )
    lat_coords = np.linspace(north, south, n_rows)
    lon_coords = np.linspace(west, east, n_cols)

    xr_elev = xr.Dataset({
        "elevation": xr.DataArray(
            merged_array, dims=["y", "x"],
            coords={"y": lat_coords, "x": lon_coords},
        )
    })

    s, w, n, e = bbox
    clipped = xr_elev.sel(x=slice(w, e), y=slice(n, s))

    lats_pts = xr.DataArray(pts_df["Latitude"].values, dims="points")
    lons_pts = xr.DataArray(pts_df["Longitude"].values, dims="points")

    df = pd.DataFrame({
        "Latitude": pts_df["Latitude"].values,
        "Longitude": pts_df["Longitude"].values,
        "elevation": clipped.elevation.sel(
            x=lons_pts, y=lats_pts, method="nearest"
        ).values.astype(float),
    })

    del merged_array, xr_elev, clipped

    print(f"[DEM] {len(df)} points | elevation range: {df['elevation'].min():.0f}–{df['elevation'].max():.0f}m")
    if use_cache:
        cache_put(df, "dem", cache_dir, **cache_params)
    return df


# ---------------------------------------------------------------------------
# Building Density — STATIC, from pre-computed CSV or shapefile
# ---------------------------------------------------------------------------

def extract_buildings(
    pts_df,
    buildings_csv=None,
    buildings_shp=None,
    buffer_m=100,
    use_cache=True,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """
    Get building density at each point.

    Two input modes (pick one):
      buildings_csv — pre-computed CSV with Latitude, Longitude, building_density_100m
      buildings_shp — compute on-the-fly from shapefile using spatial index

    Output: Latitude, Longitude, building_density_100m
    """
    cache_params = dict(pid=points_fingerprint(pts_df), buffer_m=buffer_m)
    if use_cache:
        cached = cache_get("buildings", cache_dir, **cache_params)
        if cached is not None:
            return cached

    if buildings_csv is not None:
        bld = load_points(buildings_csv)
        for c in bld.columns:
            if "building" in c.lower() and "density" in c.lower():
                bld.rename(columns={c: "building_density_100m"}, inplace=True)
                break

        df = pts_df[["Latitude", "Longitude"]].merge(
            bld[["Latitude", "Longitude", "building_density_100m"]],
            on=["Latitude", "Longitude"],
            how="left",
        )
        matched = df["building_density_100m"].notna().sum()
        print(f"[Buildings] {matched}/{len(df)} points matched from CSV")
        if use_cache:
            cache_put(df, "buildings", cache_dir, **cache_params)
        return df

    if buildings_shp is not None:
        import geopandas as gpd
        from shapely.geometry import Point, box as shapely_box

        gdf_points = gpd.GeoDataFrame(
            pts_df,
            geometry=[
                Point(lon, lat)
                for lat, lon in zip(pts_df["Latitude"], pts_df["Longitude"])
            ],
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        gdf_buildings = gpd.read_file(buildings_shp)
        if gdf_buildings.crs != "EPSG:3857":
            gdf_buildings = gdf_buildings.to_crs(epsg=3857)

        sindex = gdf_buildings.sindex
        half = buffer_m / 2
        densities = []

        for _, row in gdf_points.iterrows():
            buf = shapely_box(
                row.geometry.x - half, row.geometry.y - half,
                row.geometry.x + half, row.geometry.y + half,
            )
            candidates = list(sindex.intersection(buf.bounds))
            hits = gdf_buildings.iloc[candidates]
            count = hits[hits.geometry.intersects(buf)].shape[0]
            densities.append(count / buf.area)

        df = pd.DataFrame({
            "Latitude": pts_df["Latitude"].values,
            "Longitude": pts_df["Longitude"].values,
            "building_density_100m": densities,
        })
        print(f"[Buildings] Computed {len(df)} points from shapefile")
        if use_cache:
            cache_put(df, "buildings", cache_dir, **cache_params)
        return df

    print("[Buildings] No source — returning NaN (add buildings_csv or buildings_shp)")
    return pd.DataFrame({
        "Latitude": pts_df["Latitude"].values,
        "Longitude": pts_df["Longitude"].values,
        "building_density_100m": np.nan,
    })
