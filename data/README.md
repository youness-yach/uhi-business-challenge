# Data

Everything the models need is committed here, so the notebooks run without
access to Microsoft Fabric or the Planetary Computer.

```
data/
├── raw/          model inputs: one row per labelled point, satellite features attached
└── processed/    model outputs: predicted UHI class per point
```

## How the raw tables were produced

1. **Points.** The challenge supplied lat/lon points with a `UHI_Class` label
   (Low / Medium / High) for Rio de Janeiro and Santiago, plus unlabelled
   validation points for Freetown
   ([chase-kusterer/bc-II](https://github.com/chase-kusterer/bc-II/tree/main/Data)).
2. **Satellite features.** Sentinel-2 L2A (SCL cloud-masked median composite),
   Landsat-8 thermal (land-surface temperature), and the Copernicus DEM were
   sampled at each point on a 100 m grid. The team's extraction package is
   [Mickias-Ambaye/uhi-pipe](https://github.com/Mickias-Ambaye/uhi-pipe);
   `notebooks/01_data_extraction.ipynb` is the version run for this project.
3. **Building morphology.** 3D building footprints from 3D-GloBFP
   ([Li et al., 2024](https://doi.org/10.5281/zenodo.11319912)) were aggregated
   in a 100 m buffer around each point (height, area, volume, coverage). The
   footprint tiles (~380 MB) are not committed. Only the aggregated columns are.

## Where this ran: Microsoft Fabric

The team worked in Microsoft Fabric notebooks (Python). Inputs and outputs lived
in the Files area of the attached Lakehouse:

```
/lakehouse/default/Files/
├── dataset_100m_<city>.csv, buildings_<city>.csv   ← same tables as data/raw/
└── uhi_pipe_output/
    ├── datasets/ML_Dataset_<res>m.csv              ← resolution sweep (10 m – 1000 m)
    └── predictions/                                ← model outputs
```

`uhi_models/config.py` detects the Lakehouse and uses these paths. Anywhere
else it uses `data/`, so the notebooks run unchanged in both places.

The extraction step was built to fit notebook memory:
- Scenes load lazily in 2048 × 2048-pixel chunks (`odc-stac`) with `uint16` bands.
- Each raster is released (`del`) as soon as it has been sampled at the points.
- Elevation is cast to `float32`, and downloads stream in 8 KB pieces.
- `uhi_pipe` samples values at the points without writing images to disk, and
  caches each source as Parquet, so re-runs skip completed downloads.

## raw/

| File | Rows | Cols | City | Labelled |
|---|---:|---:|---|---|
| `dataset_100m_Brazil.csv` | 28,488 | 43 | Rio de Janeiro | yes |
| `dataset_100m_Chile.csv` | 21,662 | 43 | Santiago | yes |
| `dataset_100m_Sierra_Leone.csv` | 14,105 | 43 | Freetown | no (prediction target) |
| `buildings_Brazil.csv` | 28,488 | 58 | Rio de Janeiro | yes |
| `buildings_Chile.csv` | 21,662 | 58 | Santiago | yes |

`buildings_*` = `dataset_100m_*` + 15 building-morphology columns. No
Freetown buildings table was built. The transfer notebook loads
`buildings_<city>.csv` when it exists and falls back to `dataset_100m_<city>.csv`,
so the Freetown model uses spectral, thermal and terrain features only.

Class balance:

| City | High | Medium | Low |
|---|---:|---:|---:|
| Rio | 12,748 | 5,149 | 10,591 |
| Santiago | 5,317 | 10,816 | 5,529 |

**Known gap:** 589 of the 14,105 Freetown points (4.2%) have no Sentinel-2 or
Landsat values (cloud cover). The transfer notebook fills the six core
features (`LST`, `NDVI`, `NDBI`, `emissivity`, `NDVI_Landsat`, `elevation`)
with the Freetown median and the remaining spectral columns with 0, so every
point still gets a prediction.

### Columns

| Group | Columns | Source |
|---|---|---|
| Location | `Latitude`, `Longitude` | challenge points |
| Target | `UHI_Class` (Low/Medium/High), `label` (encoded) | challenge labels |
| Sentinel-2 bands | `B01`–`B12`, `B8A` | Sentinel-2 L2A, reflectance DN |
| Vegetation | `NDVI`, `EVI`, `SAVI`, `GNDVI`, `LAI` | derived |
| Water / moisture | `NDWI`, `MNDWI`, `NDMI` | derived |
| Built-up | `NDBI`, `ISA`, `BU`, `NDISI` | derived |
| Bare soil | `NDBaI`, `BSI`, `DBSI` | derived |
| Surface | `LSE`, `Albedo`, `SWIR1_NIR`, `SWIR2_NIR` | derived |
| Landsat-8 | `red`, `green`, `blue`, `nir08`, `lwir11`, `NDVI_Landsat`, `emissivity`, `LST` (°C) | Landsat-8 L2 |
| Terrain | `elevation` (m) | Copernicus DEM |
| Buildings (`buildings_*` only) | `building_density_100m`, `bldg_count_100m`, `bldg_mean_height`, `bldg_max_height`, `bldg_total_area`, `bldg_total_volume`, `bldg_height_std`, `bldg_height_range`, `bldg_height_max_ratio`, `sky_view_proxy`, `volume_density`, `mean_bldg_volume`, `building_coverage_ratio`, `open_space_ratio`, `compactness` | footprints, 100 m buffer |

Index formulas follow the definitions documented in
[uhi-pipe's column reference](https://github.com/Mickias-Ambaye/uhi-pipe#column-reference).

## processed/

| File | Rows | What it is |
|---|---:|---|
| `rio_predictions_in_sample.csv` | 28,488 | Final Rio XGBoost's predictions on its **own training points** (`predicted`, `correct`). The 99.7% hit rate is in-sample. The 5-fold cross-validated F1 of 0.959 is the real performance figure. Written by `notebooks/02_uhi_classification.ipynb`. |
| `santiago_predictions.csv` | 21,662 | Santiago model's predicted class per point, exported during the challenge. |
| `freetown_predictions.csv` | 14,105 | Freetown transfer: class specialists trained on Rio and Santiago, RF + XGBoost ensemble. This is the challenge submission, written by `notebooks/02_uhi_classification.ipynb`. A fresh run reproduces it row for row. |

## sweep/ (not committed)

The research notebooks compare extraction resolutions and read
`ML_Dataset_<res>m.csv`. The 100 m table is rebuilt on the fly from `raw/`.
250 m and 500 m tables have to be re-extracted with `uhi_pipe` (see
`uhi_models.config.load_sweep`).

## Licence and use

The satellite data is public (Copernicus / USGS via Microsoft Planetary
Computer). The point labels come from the Hult Business Challenge II
materials and are shared here for reproducibility of this coursework.
