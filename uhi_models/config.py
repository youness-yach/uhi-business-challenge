"""Where the data lives: a Microsoft Fabric Lakehouse or this repository.

The notebooks were developed and run on Microsoft Fabric. There they read
their inputs from, and write their predictions to, the Files area of the
attached Lakehouse. The same notebooks also run locally from the committed
``data/`` folder.

This module works out which environment it is in when it is imported, so no
notebook hard-codes a path. The layout on each side:

    Fabric Lakehouse (/lakehouse/default/Files)   Local repository
    ├── dataset_100m_<city>.csv                   data/raw/
    ├── buildings_<city>.csv                      data/raw/
    └── uhi_pipe_output/
        ├── datasets/ML_Dataset_<res>m.csv        data/sweep/   (not committed, see load_sweep)
        └── predictions/                          data/processed/

Set ``UHI_DATA_ROOT`` to point somewhere else, for example a mounted drive.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

LAKEHOUSE = Path("/lakehouse/default/Files")
REPO_ROOT = Path(__file__).resolve().parent.parent

if os.environ.get("UHI_DATA_ROOT"):
    _root = Path(os.environ["UHI_DATA_ROOT"])
    ENVIRONMENT = "custom"
    RAW_DIR, SWEEP_DIR, OUT_DIR = _root / "raw", _root / "sweep", _root / "processed"
elif LAKEHOUSE.is_dir():
    ENVIRONMENT = "fabric"
    RAW_DIR = LAKEHOUSE
    SWEEP_DIR = LAKEHOUSE / "uhi_pipe_output" / "datasets"
    OUT_DIR = LAKEHOUSE / "uhi_pipe_output" / "predictions"
else:
    ENVIRONMENT = "local"
    RAW_DIR = REPO_ROOT / "data" / "raw"
    SWEEP_DIR = REPO_ROOT / "data" / "sweep"
    OUT_DIR = REPO_ROOT / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# File-name key -> city. The `label` column inside the files uses "Sierra Leone".
CITIES = {"Brazil": "Rio de Janeiro", "Chile": "Santiago", "Sierra_Leone": "Freetown"}
LABELLED = ("Brazil", "Chile")


def load_grid(city: str) -> pd.DataFrame:
    """100 m satellite feature grid for one city (see data/README.md)."""
    return pd.read_csv(RAW_DIR / f"dataset_100m_{city}.csv")


def load_buildings(city: str) -> pd.DataFrame:
    """Grid plus building-morphology columns. Only exists for Brazil and Chile."""
    return pd.read_csv(RAW_DIR / f"buildings_{city}.csv")


def load_sweep(resolution_m: int) -> pd.DataFrame:
    """All three cities at one extraction resolution, as used by the research notebooks.

    On Fabric these tables were written to the Lakehouse by
    ``uhi_pipe.build_resolution_sweep``. The 100 m table can be rebuilt from
    the committed grids. The coarser ones (250 m, 500 m) have to be
    re-extracted from Planetary Computer, which is too large to commit.
    """
    path = SWEEP_DIR / f"ML_Dataset_{resolution_m}m.csv"
    if path.exists():
        return pd.read_csv(path)
    if resolution_m == 100:
        df = pd.concat([load_grid(c) for c in CITIES], ignore_index=True)
        df["resolution_m"] = 100
        return df
    raise FileNotFoundError(
        f"{path} not found. Build it with:\n"
        "    from uhi_pipe import build_resolution_sweep\n"
        "    from uhi_pipe.config import LOCATIONS\n"
        f"    build_resolution_sweep(LOCATIONS, resolutions=[{resolution_m}])[{resolution_m}]"
        f".to_csv('{path}', index=False)"
    )


def describe() -> str:
    return f"environment={ENVIRONMENT}  raw={RAW_DIR}  out={OUT_DIR}"
