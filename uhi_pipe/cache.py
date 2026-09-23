"""
Simple file-based caching layer.

Each extraction result is saved as a parquet file keyed by its parameters.
Before running an extraction, we check if the result already exists.

Cache key logic:
- DEM:       hash(bbox, points_id)              ← no date, no resolution
- Buildings: hash(bbox, points_id)              ← no date, no resolution
- Sentinel:  hash(bbox, points_id, date, res)   ← depends on everything
- Landsat:   hash(bbox, points_id, date, res)   ← depends on everything
"""

import os
import hashlib
import json
import pandas as pd

DEFAULT_CACHE_DIR = ".uhi_cache"


def _make_key(source, **params):
    """Create a deterministic hash from source name + params."""
    raw = json.dumps({"source": source, **params}, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(source, cache_dir, **params):
    key = _make_key(source, **params)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{source}_{key}.parquet")


def cache_get(source, cache_dir=DEFAULT_CACHE_DIR, **params):
    """Return cached DataFrame if it exists, else None."""
    path = _cache_path(source, cache_dir, **params)
    if os.path.exists(path):
        print(f"[cache] HIT  {source} → {os.path.basename(path)}")
        return pd.read_parquet(path)
    return None


def cache_put(df, source, cache_dir=DEFAULT_CACHE_DIR, **params):
    """Save DataFrame to cache."""
    path = _cache_path(source, cache_dir, **params)
    df.to_parquet(path, index=False)
    print(f"[cache] SAVE {source} → {os.path.basename(path)}")


def cache_clear(cache_dir=DEFAULT_CACHE_DIR):
    """Delete all cached files."""
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))
        print(f"[cache] Cleared {cache_dir}")
