"""
Accuracy plots: Horizon degradation, baseline comparison leaderboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from .plot_style import setup_style, save_fig, get_color, get_label, COLORS


def plot_horizon_degradation(
    log_dir: str,
    run_names: List[str],
    output_dir: str = "outputs/plots",
    depth: int = 4,
    horizons: List[int] = None,
):
    """
    Grouped bar chart: MAE at each forecasting horizon (6h, 12h, 24h, 48h, 72h).
    
    Shows Model A degrades at long range while Model C maintains accuracy.
    """
    setup_style()
    horizons = horizons or [6, 12, 24, 48, 72]

    # Load best epoch metrics
    model_data = {}
    for name in run_names:
        path = Path(log_dir) / f"{name}_epoch.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        model_key = name.split("_")[0]
        # Get best epoch (lowest val_mse)
        best_idx = df["val_mse"].idxmin()
        horizon_cols = [c for c in df.columns if "horizon" in c]
        maes = []
        for col in sorted(horizon_cols):
            maes.append(df.loc[best_idx, col])
        model_data[model_key] = maes

    if not model_data:
        print(f"  No epoch logs found in {log_dir}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(horizons))
    n_models = len(model_data)
    width = 0.8 / n_models

    for i, (model_key, maes) in enumerate(model_data.items()):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, maes, width, label=get_label(model_key),
                      color=get_color(model_key), alpha=0.85, edgecolor="white")
        # Add value labels on bars
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Forecasting Horizon")
    ax.set_ylabel("MAE")
    ax.set_title(f"Forecasting Horizon Degradation (L={depth})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.legend(loc="upper left")
    save_fig(fig, output_dir, f"horizon_degradation_l{depth}")


def plot_baseline_comparison(
    results: Dict[str, Dict],
    output_dir: str = "outputs/plots",
    metric: str = "test_mse",
):
    """
    Grouped bar chart comparing all baselines and models.
    
    Args:
        results: Dict of {model_name: {test_mse, test_mae, params, ...}}
        output_dir: Where to save.
        metric: Which metric to plot (test_mse or test_mae).
    """
    setup_style()

    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [get_color(n) for n in names]
    labels = [get_label(n) for n in names]

    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.85, edgecolor="white")

    for bar, val, label in zip(bars, values, labels):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="left", va="center", fontsize=9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Baseline Accuracy Comparison ({metric.replace('_', ' ').title()})")
    ax.invert_yaxis()
    save_fig(fig, output_dir, "baseline_accuracy_comparison")
