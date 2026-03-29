"""
merge.py — Join extracted sources and provide one-call pipeline functions.

Three entry points:
  build_dataset()          → one location, one resolution → DataFrame
  build_multi_location()   → N locations, one resolution → combined DataFrame
  build_resolution_sweep() → N locations × M resolutions → dict of DataFrames
                             (smart: static sources run once, not M times)
"""

import pandas as pd

from .extract import (
    load_points, extract_sentinel, extract_landsat,
    extract_dem, extract_buildings,
)
from .features import sentinel_features, landsat_features, dem_features
from .cache import DEFAULT_CACHE_DIR


def merge_sources(dfs):
    """
    Merge a list of DataFrames on (Latitude, Longitude) using left join.

    Each extractor returns a DataFrame with Latitude, Longitude as the first
    two columns (copied from your input CSV). Since they all come from the
    same points, the merge keys match exactly.
    """
    base = dfs[0].copy()
    for df in dfs[1:]:
        # avoid duplicate columns (other than the join keys)
        overlap = [
            c for c in df.columns
            if c in base.columns and c not in ("Latitude", "Longitude")
        ]
        right = df.drop(columns=overlap)
        base = base.merge(right, on=["Latitude", "Longitude"], how="left")
    return base


def build_dataset(
    points,
    bbox,
    time_window,
    resolution=250,
    sources=("sentinel", "landsat", "dem"),
    compute_features=True,
    max_cloud_s2=30,
    max_cloud_landsat=50,
    landsat_scene=0,
    neighborhood=0,
    buildings_shp=None,
    buildings_csv=None,
    label=None,
    save_csv=None,
    use_cache=True,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """
    One call → extract all sources + compute features + merge → ML-ready DataFrame.

    What you provide:
      points      — CSV path, URL, or DataFrame with Latitude, Longitude columns.
                    This controls how many rows the output has.
                    If it has a UHI_Class column, that column is preserved.
      bbox        — (south, west, north, east) in degrees.
                    Defines the satellite image download area.
      time_window — "2024-01-15/2024-02-15" style date range for scene search.
      resolution  — pixel size in meters (10, 50, 100, 250, 500, 750, 1000).
                    Higher = faster + less RAM. Lower = more spatial detail.

    What you get back:
      DataFrame with one row per input point and columns for:
        - Latitude, Longitude (from your CSV)
        - UHI_Class (if present in your CSV)
        - Sentinel-2 bands (B01-B12) + spectral indices (NDVI, NDBI, etc.)
        - Landsat bands (red, green, blue, nir08, lwir11) + LST
        - elevation (meters from DEM)
        - building_density_100m (if buildings source provided)
        - label column (if you set label="Chile" etc.)

    Smart caching:
      DEM and buildings only depend on bbox + points, so changing
      resolution or time_window won't re-download them.
    """
    # load points ONCE — all extractors share this DataFrame
    pts_df = load_points(points)

    dfs = []

    # --- resolution + date sensitive sources ---
    if "sentinel" in sources:
        df_s = extract_sentinel(
            pts_df, bbox, time_window, resolution,
            max_cloud=max_cloud_s2, neighborhood=neighborhood,
            use_cache=use_cache, cache_dir=cache_dir,
        )
        if df_s is not None:
            if compute_features:
                df_s = sentinel_features(df_s)
            dfs.append(df_s)

    if "landsat" in sources:
        df_l = extract_landsat(
            pts_df, bbox, time_window, resolution,
            max_cloud=max_cloud_landsat, scene_index=landsat_scene,
            neighborhood=neighborhood,
            use_cache=use_cache, cache_dir=cache_dir,
        )
        if df_l is not None:
            if compute_features:
                df_l = landsat_features(df_l)
            dfs.append(df_l)

    # --- static sources (no date, no resolution) ---
    if "dem" in sources:
        df_d = extract_dem(
            pts_df, bbox,
            use_cache=use_cache, cache_dir=cache_dir,
        )
        if df_d is not None:
            if compute_features:
                df_d = dem_features(df_d)
            dfs.append(df_d)

    if "buildings" in sources:
        df_b = extract_buildings(
            pts_df,
            buildings_csv=buildings_csv, buildings_shp=buildings_shp,
            use_cache=use_cache, cache_dir=cache_dir,
        )
        if df_b is not None:
            dfs.append(df_b)

    if not dfs:
        print("WARNING: No data extracted from any source.")
        return None

    # merge all sources on (Latitude, Longitude)
    merged = merge_sources(dfs)

    # carry through UHI_Class from the original points CSV if it exists
    if "Uhi_class" not in merged.columns and "UHI_Class" in pts_df.columns:
        merged["UHI_Class"] = pts_df["UHI_Class"].values

    if label:
        merged["label"] = label

    if save_csv:
        merged.to_csv(save_csv, index=False)
        print(f"Saved: {save_csv} ({merged.shape[0]} rows × {merged.shape[1]} cols)")

    return merged


def build_multi_location(locations, **shared_kwargs):
    """
    Run build_dataset for each location and combine into one DataFrame.

    Parameters
    ----------
    locations : list of dict
        Each dict needs at minimum: points, bbox, time_window.
        Optional: label, buildings_csv, buildings_shp.

    **shared_kwargs
        Applied to all locations. Individual location dicts override these.

    Returns
    -------
    Combined DataFrame with all locations stacked. Has a 'label' column
    if you set it in the location dicts.
    """
    all_dfs = []
    for loc in locations:
        # merge shared defaults with per-location overrides
        kwargs = {**shared_kwargs, **loc}
        name = kwargs.get("label", "unknown")
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        df = build_dataset(**kwargs)
        if df is not None:
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined: {combined.shape[0]} rows × {combined.shape[1]} cols")
    return combined


def build_resolution_sweep(
    locations,
    resolutions=(10, 50, 100, 250, 500, 750, 1000),
    sources=("sentinel", "landsat", "dem"),
    compute_features=True,
    **shared_kwargs,
):
    """
    Run all locations × all resolutions. Smart about what to re-run.

    The key optimization: DEM and buildings are extracted ONCE per location
    (they don't depend on resolution). Sentinel and Landsat are extracted
    per resolution, but cached individually so re-runs skip completed work.

    Parameters
    ----------
    locations : list of dict
        Same as build_multi_location.
    resolutions : tuple of int
        Pixel sizes in meters to sweep.
    sources : tuple of str
        Which sources to include.
    compute_features : bool
        Whether to compute spectral indices.

    Returns
    -------
    dict: {resolution_m: DataFrame}
        One ML-ready DataFrame per resolution.

    Example
    -------
    results = build_resolution_sweep(LOCATIONS, resolutions=[10, 250, 1000])
    df_250 = results[250]  # ← 64K rows × 44 cols at 250m
    """
    results = {}

    static_sources = [s for s in sources if s in ("dem", "buildings")]
    dynamic_sources = [s for s in sources if s in ("sentinel", "landsat")]

    use_cache = shared_kwargs.get("use_cache", True)
    cache_dir = shared_kwargs.get("cache_dir", DEFAULT_CACHE_DIR)

    # --- STEP 1: extract static sources once per location ---
    print("=" * 60)
    print("  STATIC SOURCES (extracted once per location)")
    print("=" * 60)

    # keyed by label → list of DataFrames
    static_by_location = {}
    pts_by_location = {}

    for loc in locations:
        kwargs = {**shared_kwargs, **loc}
        name = kwargs.get("label", "unknown")
        pts_df = load_points(kwargs["points"])
        pts_by_location[name] = pts_df

        loc_static_dfs = []

        if "dem" in static_sources:
            df_d = extract_dem(pts_df, kwargs["bbox"],
                               use_cache=use_cache, cache_dir=cache_dir)
            if df_d is not None:
                if compute_features:
                    df_d = dem_features(df_d)
                loc_static_dfs.append(df_d)

        if "buildings" in static_sources:
            df_b = extract_buildings(
                pts_df,
                buildings_csv=kwargs.get("buildings_csv"),
                buildings_shp=kwargs.get("buildings_shp"),
                use_cache=use_cache, cache_dir=cache_dir,
            )
            if df_b is not None:
                loc_static_dfs.append(df_b)

        static_by_location[name] = loc_static_dfs
        print(f"  {name}: {len(loc_static_dfs)} static source(s)")

    # --- STEP 2: sweep resolutions (only dynamic sources re-run) ---
    for res in resolutions:
        print(f"\n{'='*60}")
        print(f"  RESOLUTION: {res}m")
        print(f"{'='*60}")

        all_loc_dfs = []

        for loc in locations:
            kwargs = {**shared_kwargs, **loc}
            name = kwargs.get("label", "unknown")
            pts_df = pts_by_location[name]

            dfs_for_loc = []

            if "sentinel" in dynamic_sources:
                df_s = extract_sentinel(
                    pts_df, kwargs["bbox"], kwargs["time_window"], res,
                    max_cloud=kwargs.get("max_cloud_s2", 30),
                    neighborhood=kwargs.get("neighborhood", 0),
                    use_cache=use_cache, cache_dir=cache_dir,
                )
                if df_s is not None:
                    if compute_features:
                        df_s = sentinel_features(df_s)
                    dfs_for_loc.append(df_s)

            if "landsat" in dynamic_sources:
                df_l = extract_landsat(
                    pts_df, kwargs["bbox"], kwargs["time_window"], res,
                    max_cloud=kwargs.get("max_cloud_landsat", 50),
                    scene_index=kwargs.get("landsat_scene", 0),
                    neighborhood=kwargs.get("neighborhood", 0),
                    use_cache=use_cache, cache_dir=cache_dir,
                )
                if df_l is not None:
                    if compute_features:
                        df_l = landsat_features(df_l)
                    dfs_for_loc.append(df_l)

            # attach the already-extracted static data
            dfs_for_loc.extend(static_by_location.get(name, []))

            if dfs_for_loc:
                merged = merge_sources(dfs_for_loc)

                # preserve UHI_Class from original CSV
                if "UHI_Class" in pts_df.columns and "UHI_Class" not in merged.columns:
                    merged["UHI_Class"] = pts_df["UHI_Class"].values

                merged["label"] = name
                merged["resolution_m"] = res
                all_loc_dfs.append(merged)

        if all_loc_dfs:
            results[res] = pd.concat(all_loc_dfs, ignore_index=True)
            shape = results[res].shape
            print(f"  → {shape[0]} rows × {shape[1]} cols")

    print(f"\nSweep done: {len(results)} resolutions")
    return results
