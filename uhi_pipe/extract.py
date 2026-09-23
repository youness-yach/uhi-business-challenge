"""
extract.py — Pull raw band/pixel values from satellite sources at point locations.

How it works:
  1. You provide a CSV/DataFrame with Latitude, Longitude (your sample points)
  2. We download the satellite raster for your bbox + date range
  3. For each point, we extract pixel values using one of two methods:
     - neighborhood=0 (default): single nearest pixel via .sel(method="nearest")
     - neighborhood=N: NxN pixel bounding box median around each point
       (Chase Kusterer's approach — better represents the area around each station)
  4. Output DataFrame has exactly the same number of rows as your input CSV

Data quality:
  - Sentinel-2 uses SCL band to mask clouds, shadows, snow, water before compositing
  - Nodata pixels (0 in Sentinel, 0 in Landsat) are replaced with NaN
  - Landsat thermal is properly scaled: DN → Kelvin → °C, with nodata masked first
  - Sentinel compositing (median over time + SCL masking) gives clean cloud-free data

Cache rules:
  - DEM + Buildings: keyed by (bbox, points) → immune to date/resolution changes
  - Sentinel + Landsat: keyed by (bbox, points, date, resolution, neighborhood)
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

# SCL values to EXCLUDE (clouds, shadows, snow, nodata, saturated)
SCL_BAD = {0, 1, 3, 8, 9, 10}


def _extract_at_points(comp, bands, pts_df, neighborhood, scale):
    """
    Extract band values at point locations from a composited raster.

    neighborhood=0  → nearest pixel (.sel method="nearest")
    neighborhood=N  → NxN pixel bounding box median around each point
                      (Chase Kusterer's approach: captures spatial context)
    """
    lats = pts_df["Latitude"].values
    lons = pts_df["Longitude"].values

    df = pd.DataFrame({"Latitude": lats, "Longitude": lons})

    if neighborhood <= 0:
        # single nearest pixel
        lat_da = xr.DataArray(lats, dims="points")
        lon_da = xr.DataArray(lons, dims="points")
        for band in bands:
            df[band] = comp[band].sel(
                longitude=lon_da, latitude=lat_da, method="nearest"
            ).values.astype(float)
    else:
        # NxN bounding box median (half-window = N/2 pixels)
        half_pixels = neighborhood / 2
        lon_offset = scale * half_pixels
        lat_offset = scale * half_pixels

        for band in bands:
            vals = []
            for lat, lon in zip(lats, lons):
                subset = comp[band].sel(
                    longitude=slice(lon - lon_offset, lon + lon_offset),
                    latitude=slice(lat + lat_offset, lat - lat_offset),
                )
                vals.append(float(np.nanmedian(subset.values)))
            df[band] = vals

    return df


def extract_sentinel(
    pts_df,
    bbox,
    time_window,
    resolution=250,
    bands=None,
    max_cloud=30,
    composite="median",
    neighborhood=0,
    use_cache=True,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """
    Extract Sentinel-2 L2A band values at each point location.

    Steps:
      1. Search Planetary Computer for scenes in bbox + date range
      2. Load all scenes + SCL cloud mask into xarray
      3. Apply SCL mask: remove clouds, shadows, snow, water, nodata
      4. Mask remaining nodata (DN=0 → NaN)
      5. Compute median composite across time
      6. Extract values at each point (nearest pixel or NxN neighborhood median)

    Parameters
    ----------
    neighborhood : int
        0 = nearest pixel (fast, default)
        100 = 100×100 pixel bounding box median around each point
              (Chase Kusterer's approach — better for UHI station data)
    """
    bands = bands or SENTINEL_BANDS

    cache_params = dict(
        bbox=bbox, time_window=time_window,
        resolution=resolution, pid=points_fingerprint(pts_df),
        neighborhood=neighborhood,
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

    # load spectral bands + SCL cloud classification band
    load_bands = bands + ["SCL"]
    data = stac_load(
        items, bands=load_bands, crs="EPSG:4326", resolution=scale,
        chunks={"x": 2048, "y": 2048}, dtype="uint16",
        patch_url=planetary_computer.sign, bbox=stac_bbox,
    )

    # SCL cloud mask: exclude nodata(0), saturated(1), shadow(3),
    # cloud-medium(8), cloud-high(9), cirrus(10)
    scl = data["SCL"]
    cloud_mask = scl.copy(data=np.ones_like(scl.values, dtype=bool))
    for bad_val in SCL_BAD:
        cloud_mask = cloud_mask & (scl != bad_val)

    # apply SCL mask to spectral bands (keeps only clear pixels)
    data_clean = data[bands].where(cloud_mask)

    # also mask raw nodata (DN=0 can still exist in non-masked areas)
    data_clean = data_clean.where(data_clean > 0)

    # composite: median across time collapses to one clean raster
    comp = getattr(data_clean, composite)(dim="time").compute()

    # extract at points
    df = _extract_at_points(comp, bands, pts_df, neighborhood, scale)

    del data, data_clean, comp

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

# QA_PIXEL bit flags for Landsat Collection 2 Level 2
# Bit 1 = dilated cloud, Bit 3 = cloud, Bit 4 = cloud shadow, Bit 5 = snow
# We mask any pixel where ANY of these bits are set.
QA_CLOUD_BITS = [1, 3, 4, 5]


def _qa_cloud_mask(qa_band):
    """
    Build a boolean mask from Landsat QA_PIXEL band.

    Returns True for CLEAR pixels, False for cloud/shadow/snow.
    Same role as Sentinel's SCL mask — removes contaminated pixels
    before compositing so clouds don't leak into the median.

    Bit reference (USGS Landsat C2 L2 QA_PIXEL):
      Bit 0: fill          Bit 1: dilated cloud
      Bit 2: cirrus        Bit 3: cloud
      Bit 4: cloud shadow  Bit 5: snow
    """
    mask = np.ones_like(qa_band.values, dtype=bool)
    for bit in QA_CLOUD_BITS:
        mask = mask & ((qa_band.values.astype(int) >> bit) & 1 == 0)
    # also mask fill pixels (bit 0)
    mask = mask & ((qa_band.values.astype(int) >> 0) & 1 == 0)
    return qa_band.copy(data=mask)


def extract_landsat(
    pts_df,
    bbox,
    time_window,
    resolution=250,
    max_cloud=50,
    composite="median",
    neighborhood=0,
    use_cache=True,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """
    Extract Landsat-8 visible/NIR + thermal at each point.

    Uses QA_PIXEL cloud masking + median composite across all scenes,
    matching the approach used for Sentinel-2. This is critical for
    cross-location consistency — without masking, cloudy pixels show up
    as artificially cold LST values (clouds are cold in thermal IR),
    which corrupts the UHI signal.

    Processing steps:
      1. Search Planetary Computer for Landsat-8 scenes in bbox + date range
      2. Load vis bands, thermal band, and QA_PIXEL quality band
      3. Apply QA_PIXEL mask: remove clouds, cloud shadow, snow, fill
      4. Mask remaining nodata (DN=0 → NaN)
      5. Apply USGS scale factors (DN → reflectance for vis, DN → °C for thermal)
      6. Median composite across time (collapses to one clean raster)
      7. Extract values at each point

    Data units after extraction:
      - vis bands (red, green, blue, nir08): surface reflectance (0 to ~0.5)
      - lwir11: brightness temperature in °C (emissivity correction → LST
        is applied later in features.py)

    Parameters
    ----------
    pts_df : DataFrame
        Must have Latitude, Longitude columns.
    bbox : tuple
        (south, west, north, east) in degrees.
    time_window : str
        Date range, e.g. "2024-01-15/2024-02-15".
    resolution : int
        Pixel size in meters. Higher = faster + less RAM.
    max_cloud : int
        Max scene-level cloud cover % for STAC search filter.
        Individual pixels are still masked by QA_PIXEL regardless.
    composite : str
        How to combine scenes across time. Default "median".
        Median is robust to outliers from any unmasked cloud edges.
    neighborhood : int
        0 = nearest pixel (fast, default).
        N = NxN pixel bounding box median around each point.
    """
    all_bands = LANDSAT_VIS_BANDS + LANDSAT_THERMAL

    cache_params = dict(
        bbox=bbox, time_window=time_window,
        resolution=resolution, pid=points_fingerprint(pts_df),
        neighborhood=neighborhood, composite=composite,
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

    # log scene dates so users can verify temporal coverage
    for i, item in enumerate(items):
        dt = item.properties.get("datetime", "?")
        cc = item.properties.get("eo:cloud_cover", "?")
        print(f"[Landsat]   scene {i}: {dt}  cloud={cc}%")

    # load vis, thermal, and QA_PIXEL together
    load_vis = LANDSAT_VIS_BANDS + ["qa_pixel"]
    data_vis = stac_load(
        items, bands=load_vis, crs="EPSG:4326", resolution=scale,
        chunks={"x": 2048, "y": 2048}, dtype="uint16",
        patch_url=planetary_computer.sign, bbox=stac_bbox,
    )
    data_therm = stac_load(
        items, bands=LANDSAT_THERMAL, crs="EPSG:4326", resolution=scale,
        chunks={"x": 2048, "y": 2048}, dtype="uint16",
        patch_url=planetary_computer.sign, bbox=stac_bbox,
    )

    # QA_PIXEL cloud mask: True = clear, False = cloud/shadow/snow/fill
    qa = data_vis["qa_pixel"]
    clear_mask = _qa_cloud_mask(qa)

    # apply cloud mask to vis and thermal bands
    data_vis_clean = data_vis[LANDSAT_VIS_BANDS].where(clear_mask)
    data_therm_clean = data_therm.where(clear_mask)

    # mask raw nodata (DN=0 can still exist in non-masked areas)
    data_vis_clean = data_vis_clean.where(data_vis_clean > 0)
    data_therm_clean = data_therm_clean.where(data_therm_clean > 0)

    # apply USGS Landsat C2 L2 scale factors AFTER masking
    # lwir11 output is BRIGHTNESS TEMPERATURE, not LST.
    # Emissivity correction → LST is done later in features.py.
    data_vis_scaled = data_vis_clean.astype(float) * LANDSAT_VIS_SCALE + LANDSAT_VIS_OFFSET
    data_therm_scaled = data_therm_clean.astype(float) * LANDSAT_THERM_SCALE + LANDSAT_THERM_OFFSET - 273.15

    # median composite across time — collapses all scenes into one clean raster.
    # This is safe within a ~30-day window in the same season.
    # Median is robust to residual cloud edges that QA_PIXEL misses.
    comp_vis = getattr(data_vis_scaled, composite)(dim="time").compute()
    comp_therm = getattr(data_therm_scaled, composite)(dim="time").compute()

    # merge vis + thermal composites
    combined = xr.merge([comp_vis, comp_therm])

    # extract at points
    df = _extract_at_points(combined, all_bands, pts_df, neighborhood, scale)

    del data_vis, data_therm, data_vis_clean, data_therm_clean
    del data_vis_scaled, data_therm_scaled, comp_vis, comp_therm, combined

    # report quality
    valid_thermal = df["lwir11"].notna().sum()
    bad_thermal = ((df["lwir11"] < -50) | (df["lwir11"] > 80)).sum()
    nan_thermal = df["lwir11"].isna().sum()
    print(f"[Landsat] {len(df)} points × {len(all_bands)} bands")
    print(f"[Landsat] lwir11: {valid_thermal} valid, {nan_thermal} NaN, "
          f"{bad_thermal} out-of-range (< -50 or > 80°C)")

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
