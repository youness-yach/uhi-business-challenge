"""
uhi_pipe — Satellite data extraction + feature engineering for UHI classification.

What this package does:
  1. Downloads satellite imagery from Microsoft Planetary Computer (free, no auth)
  2. Extracts pixel values at YOUR specific point locations (not the whole raster)
  3. Computes spectral indices (NDVI, NDBI, LST, etc.)
  4. Merges all sources into one ML-ready DataFrame

What YOU provide:
  - A CSV with Latitude, Longitude columns (and optionally UHI_Class)
  - A bounding box: (south, west, north, east) in degrees
  - A date range: "2024-01-15/2024-02-15"
  - A resolution in meters: 10, 50, 100, 250, 500, 750, or 1000

What you get back:
  - A DataFrame with one row per input point
  - Columns: your lat/lon + satellite bands + spectral indices + elevation + LST

Quick start:
  from uhi_pipe import build_dataset
  from uhi_pipe.config import CHILE

  df = build_dataset(**CHILE, resolution=250)
"""

from .extract import (
    load_points,
    extract_sentinel,
    extract_landsat,
    extract_dem,
    extract_buildings,
)
from .features import (
    sentinel_features,
    landsat_features,
    compute_all,
)
from .merge import (
    build_dataset,
    build_multi_location,
    build_resolution_sweep,
    merge_sources,
)
from .cache import cache_clear
