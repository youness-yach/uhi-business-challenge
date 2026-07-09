# Predicting Urban Heat Islands — A Machine Learning Approach Across Three Cities

Satellite-driven classification of Urban Heat Island (UHI) intensity in Rio de
Janeiro and Santiago, with a combined model transferred to predict UHI risk in
Freetown, Sierra Leone.

**Team project** — Hult International Business School, Business Challenge II.
Team: Carolina Trovisco, Filippo Beni, João Ponte, Mickias Ambaye,
**Youness Yachruti**, Yousra Sajjad. My focus on this team was model
research: testing and tuning multiple tree-based and gradient-boosting models
across all three locations to improve prediction accuracy. See a teammate's
complementary extraction-pipeline repo:
[Mickias-Ambaye/uhi-pipe](https://github.com/Mickias-Ambaye/uhi-pipe).

## Overview

Urban Heat Islands are urban areas that run warmer than their surroundings —
sometimes by 10°C+ locally — driven by dense building layouts, impervious
surface heat absorption, and waste heat from industry and transport. The
challenge: build a model that predicts UHI intensity from satellite data
alone, and test whether a model trained on two cities can transfer to predict
a third city it has never seen.

We extracted Sentinel-2 spectral indices and Landsat-8 thermal/elevation data
for Rio de Janeiro and Santiago (50,150 combined data points), engineered
interaction features (e.g. LST × NDVI, elevation × LST), trained per-city
classifiers, then tested whether a model combining both cities' signal could
predict UHI intensity in Freetown, Sierra Leone — a city with no training
labels of its own.

## Key results

All figures below are held-out classification performance (F1 / precision /
recall), taken directly from the team's final report — not backtested or
live figures, this is a classification task, not a trading strategy.

- **Rio de Janeiro:** F1 = 0.959 (Precision 0.960, Recall 0.959) — best model:
  XGBoost. Rio's heat signal is dominated by raw thermal response and building
  morphology; intense year-round solar exposure combined with high shares of
  concrete/asphalt makes UHI comparatively easy to detect.
- **Santiago:** F1 = 0.690 (Precision 0.694, Recall 0.690) — best model:
  Random Forest. Santiago's heat pattern is structurally harder to model —
  shaped by elevation, slope, and built form (Andes Mountains / Chilean Coast
  Range), which makes hotspot boundaries less clean.
- **Freetown (combined Chile+Brazil → Sierra Leone transfer):** F1 = 0.58
  (Precision 0.59, Recall 0.58) — best model: ensemble of XGBoost and Random
  Forest. Transferability is limited by raw features encoding location
  identity rather than heat patterns; separate models per UHI class (High/
  Medium/Low) were needed since the classes weren't cleanly separable
  end-to-end.

**Cross-city takeaway:** urban heat drivers are city-specific, not universal
— Rio's heat is legible through building materials and solar exposure,
Santiago's through terrain, and Freetown's classification is complicated by
roofing material and color effects not fully captured by the transferred
model.

## Approach

1. **Data extraction** (`UHI_Data_Extraction.ipynb`) — Sentinel-2 L2A and
   Landsat-8 imagery pulled per location via the Microsoft Planetary
   Computer STAC API; ~50% cloud coverage in one scene was handled with SCL
   (Scene Classification Layer) filtering and median compositing across
   multiple scenes.
2. **Feature engineering** — 25 Sentinel-2 spectral indices (vegetation,
   water, built-up, and surface categories), Landsat land-surface temperature
   + elevation, and 6 engineered interaction features (e.g. LST × NDVI).
3. **Per-city modeling** (`notebooks/Brazil.ipynb`, `notebooks/Chile.ipynb`)
   — city-specific classifiers tested across tree-based and gradient-boosting
   model families (Random Forest, XGBoost), tuned separately per city.
4. **Cross-city transfer** (`notebooks/Combined.ipynb`,
   `notebooks/SL_Class_based_Pipeline_V2.ipynb`) — combined Chile+Brazil
   training set, with Quantile Transformation + PCA used to isolate
   UHI-relevant signal from city-specific characteristics before predicting
   Freetown.
5. **Reporting** (`UHI Classification.ipynb`) — consolidated written summary:
   contributing factors, mitigation strategy, and additional-data
   recommendations.

`notebooks/landsat_lst.ipynb` is a standalone utility for generating an
interactive Folium land-surface-temperature map. `src/app.py` is a Streamlit
dashboard built on top of the feature-extraction modules
(`src/get_geo_features.py`, `src/get_satellite_features.py`).

## Stack

Python: pandas, numpy, scikit-learn, xgboost, shap, xarray, rioxarray,
rasterio, rasterstats, geopandas, pystac-client, planetary-computer,
odc-stac, streamlit, folium, matplotlib, seaborn

## How to run

```bash
git clone https://github.com/youness-yach/uhi-business-challenge.git
cd uhi-business-challenge
pip install pandas numpy scipy scikit-learn xgboost shap xarray rioxarray \
  rasterio rasterstats geopandas pystac-client planetary-computer odc-stac \
  streamlit folium matplotlib seaborn
```

Run `UHI_Data_Extraction.ipynb` first to pull satellite features for each
location, then the per-city notebooks (`notebooks/Brazil.ipynb`,
`notebooks/Chile.ipynb`), then `notebooks/Combined.ipynb` /
`notebooks/SL_Class_based_Pipeline_V2.ipynb` for the cross-city transfer.
`UHI Classification.ipynb` consolidates the written findings.

## Notes / limitations

- Freetown had no ground-truth UHI labels for training — the combined-city
  model's 0.58 F1 there reflects transfer performance, not in-city validation.
- Cloud contamination and single-composite satellite snapshots limit temporal
  resolution; the team's report recommends hourly/seasonal time-series data
  and higher-resolution building morphology as natural next steps.
- Full findings, methodology detail, and the team's mitigation recommendations
  are in `Urban Heat Indicator - Team 4 - BC2.pdf`.

---
Youness Yachruti · [LinkedIn](https://www.linkedin.com/in/youness-yachruti/)
