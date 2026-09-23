# Notebooks

Two kinds of notebook live here: the pipeline, which is numbered and runs end
to end from `data/raw/`, and the research log, which records how the final
models were chosen.

Install first, from the repo root:

```bash
pip install -e .              # modelling
pip install -e ".[extract]"   # + satellite re-extraction (notebook 01)
```

## Pipeline

| Notebook | What it does | Runs from committed data | Author |
|---|---|---|---|
| `01_data_extraction.ipynb` | Pulls Sentinel-2, Landsat-8, DEM and 3D-GloBFP buildings, then writes `data/raw/`. Takes roughly 15–45 min (5–15 min per city) and needs network access. | n/a, it *creates* the data | Mickias Ambaye |
| `02_uhi_classification.ipynb` | The final pipeline. Part 1 is the Rio model (XGBoost vs RF, 5-fold CV). Part 2 is the Santiago model (quantile RF). Part 3 is the Freetown transfer (quantile → PCA, per-class specialists, RF + XGBoost ensemble). Writes `data/processed/`. | ✅ ~4 min, reproduces every published metric | Team 4 |

## Research log (`research/`)

These notebooks ran on Microsoft Fabric. Their saved outputs are the original
Fabric runs, kept as the record of the model search behind `02`.

| Notebook | Question it answered |
|---|---|
| `rio_model_search.ipynb` | Which model family and resolution work best for Rio? A tuned sweep of GB, XGBoost, RF, ExtraTrees, HistGB, LR, SVC and KNN, plus Rio → Santiago transfer. |
| `santiago_model_search.ipynb` | Why is Santiago harder? The effect of the Medium class, quantile-RF tuning, a threshold sweep, and SHAP. |
| `combined_transfer.ipynb` | Does pooling Rio and Santiago transfer to Freetown? A quantile-transformed ensemble across resolutions. |
| `freetown_specialists.ipynb` | Leave-one-city-out diagnostics and per-class specialist routing, which fed the final Freetown design in `02`. |

The 100 m sections re-run locally, because `load_sweep(100)` rebuilds that
table from `data/raw/`. Sections at 250 m or 500 m need those resolutions
re-extracted first (see `data/README.md`).
