"""
Shared plot styling for all LSTM-µHC figures.

Consistent colors, labels, and formatting matching AGENTS.md Section 8.
All plots save at 300 DPI in both PDF and PNG.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

# Model colors and styles (from AGENTS.md Section 8)
COLORS = {
    "model_a": "#1f77b4",   # Blue
    "model_b": "#d62728",   # Red
    "model_c": "#2ca02c",   # Green
    "vanilla": "#1f77b4",
    "hc": "#d62728",
    "mhc": "#2ca02c",
    "classical": "#7f7f7f",  # Gray
    "neural": "#9467bd",     # Purple
    "sota": "#111111",       # Near black
    "persistence": "#aaaaaa",
    "seasonal_naive": "#999999",
    "ridge": "#888888",
    "dlinear": "#7f7f7f",
    "gru": "#9467bd",
    "tcn": "#e377c2",
    "patchtst": "#111111",
}

LABELS = {
    "vanilla": "Model A (Standard LSTM)",
    "hc": "Model B (Naive HC-LSTM)",
    "mhc": "Model C (mHC-LSTM, Ours)",
    "model_a": "Model A (Standard LSTM)",
    "model_b": "Model B (Naive HC-LSTM)",
    "model_c": "Model C (mHC-LSTM, Ours)",
    "persistence": "Persistence",
    "seasonal_naive": "Seasonal Naive",
    "ridge": "Ridge Regression",
    "dlinear": "DLinear",
    "gru": "GRU (matched)",
    "tcn": "TCN (matched)",
    "patchtst": "PatchTST",
}

STYLES = {
    "vanilla": {"color": COLORS["vanilla"], "linestyle": "-", "linewidth": 2},
    "hc": {"color": COLORS["hc"], "linestyle": "--", "linewidth": 2},
    "mhc": {"color": COLORS["mhc"], "linestyle": "-", "linewidth": 2.5},
    "model_a": {"color": COLORS["model_a"], "linestyle": "-", "linewidth": 2},
    "model_b": {"color": COLORS["model_b"], "linestyle": "--", "linewidth": 2},
    "model_c": {"color": COLORS["model_c"], "linestyle": "-", "linewidth": 2.5},
}

# Reference lines
REF_IDENTITY = {"y": 1.0, "color": "gray", "linestyle": ":", "alpha": 0.5, "label": "Identity (y=1)"}
REF_MHC_BOUND = {"y": 1.6, "color": "#2ca02c", "linestyle": ":", "alpha": 0.5, "label": "mHC bound (~1.6)"}


def setup_style():
    """Apply consistent plot style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save_fig(fig, path: str, name: str):
    """Save figure in both PDF and PNG at 300 DPI."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {out / name}.pdf + .png")
    plt.close(fig)


def get_style(model_key: str) -> Dict:
    """Get plot style dict for a model key."""
    return STYLES.get(model_key, {"color": "gray", "linestyle": "-", "linewidth": 1.5})


def get_label(model_key: str) -> str:
    """Get display label for a model key."""
    return LABELS.get(model_key, model_key)


def get_color(model_key: str) -> str:
    """Get color for a model key."""
    return COLORS.get(model_key, "#888888")
