"""Build reports/figures/ from the pipeline notebook and the committed predictions.

    python reports/make_figures.py

1. Copies the key charts out of notebooks/02_uhi_classification.ipynb, so the
   README always shows what the notebook actually produced.
2. Draws uhi_maps.png: labelled classes for Rio and Santiago next to the
   predicted classes for Freetown.
"""

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from uhi_models.config import OUT_DIR, load_grid

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
NOTEBOOK = ROOT / "notebooks" / "02_uhi_classification.ipynb"

# Notebook chart -> file name. Matched on the code of the cell that draws it.
FROM_NOTEBOOK = {
    "Brazil — top 10 importance": "rio_feature_importance.png",
    "feature ablation": "rio_feature_ablation.png",
    "Chile — top 10 feature importances": "santiago_feature_importance.png",
    "{name} LST by class": "cross_city_lst_shift.png",
    "pca_full = PCA": "transfer_pca_variance.png",
}
# In a cell with several charts, which one to take (0-based).
PICK = {"rio_feature_ablation.png": 0}

# UHI classes are ordered, so they take one hue stepped light -> dark
# (validated ordinal ramp: monotone lightness, light end >= 2:1 on the surface).
CLASS_COLORS = {"Low": "#ef9763", "Medium": "#d9581f", "High": "#8f2f0c"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def export_notebook_charts():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for key, name in FROM_NOTEBOOK.items():
        cell = next(c for c in cells if c["cell_type"] == "code" and key in "".join(c["source"]))
        pngs = [o["data"]["image/png"] for o in cell.get("outputs", [])
                if "data" in o and "image/png" in o["data"]]
        (FIG / name).write_bytes(base64.b64decode(pngs[PICK.get(name, 0)]))
        print("wrote", name)


def draw_maps():
    panels = [
        ("Rio de Janeiro", "labelled", load_grid("Brazil")[["Longitude", "Latitude", "UHI_Class"]]),
        ("Santiago", "labelled", load_grid("Chile")[["Longitude", "Latitude", "UHI_Class"]]),
        ("Freetown", "predicted, no local labels", pd.read_csv(OUT_DIR / "freetown_predictions.csv")),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE)
    for ax, (city, kind, df) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for cls in ["Low", "Medium", "High"]:  # High drawn last so hotspots sit on top
            sub = df[df["UHI_Class"] == cls]
            ax.scatter(sub["Longitude"], sub["Latitude"], s=4, c=CLASS_COLORS[cls],
                       linewidths=0, rasterized=True)
        ax._uhi_labels = (city, f"{kind} · {len(df):,} points")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [Line2D([], [], marker="s", ls="", markersize=9, color=CLASS_COLORS[c], label=c)
               for c in ["Low", "Medium", "High"]]
    fig.legend(handles=handles, title="UHI class", loc="upper right", frameon=False,
               ncol=3, fontsize=10, title_fontsize=10, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Urban heat island class at every sample point", x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    for ax in axes:  # panel titles on one shared baseline, left-aligned to each map
        city, sub = ax._uhi_labels
        x0 = ax.get_position().x0
        fig.text(x0, 0.855, city, fontsize=13, fontweight="bold", color=INK)
        fig.text(x0, 0.82, sub, fontsize=9.5, color=MUTED)
    fig.savefig(FIG / "uhi_maps.png", dpi=150, facecolor=SURFACE)
    print("wrote uhi_maps.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    export_notebook_charts()
    draw_maps()
