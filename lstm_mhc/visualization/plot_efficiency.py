"""
Efficiency plots: Time vs accuracy trade-off, VRAM comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from .plot_style import setup_style, save_fig, get_color, get_label, get_style


def plot_accuracy_vs_compute_overhead(
    log_dir: str,
    run_names: List[str],
    output_dir: str = "outputs/plots",
    depth: int = 4,
):
    """
    Scatter plot: Training time per epoch vs final validation MSE.
    Shows Model C gets accuracy boost with only ~7% time overhead.
    """
    setup_style()

    points = {}
    for name in run_names:
        path = Path(log_dir) / f"{name}_epoch.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        model_key = name.split("_")[0]
        best_mse = df["val_mse"].min()
        avg_time = df["epoch_time_s"].mean() if "epoch_time_s" in df.columns else 0
        points[model_key] = {"mse": best_mse, "time": avg_time}

    if not points:
        print(f"  No epoch logs found in {log_dir}")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for model_key, data in points.items():
        ax.scatter(data["time"], data["mse"], s=200, zorder=5,
                   color=get_color(model_key), edgecolors="black", linewidth=1.5)
        ax.annotate(get_label(model_key), (data["time"], data["mse"]),
                    textcoords="offset points", xytext=(10, 10), fontsize=9)

    # Shade acceptable overhead region
    if points:
        min_time = min(d["time"] for d in points.values())
        ax.axvspan(min_time, min_time * 1.15, alpha=0.1, color="green", label="<15% overhead")

    ax.set_xlabel("Training Time per Epoch (seconds)")
    ax.set_ylabel("Best Validation MSE (lower is better)")
    ax.set_title(f"Accuracy vs Compute Overhead (L={depth})")
    ax.legend(loc="upper right")
    save_fig(fig, output_dir, f"accuracy_vs_compute_overhead_l{depth}")


def plot_vram_comparison(
    log_dir: str,
    run_names: List[str],
    output_dir: str = "outputs/plots",
    depth: int = 4,
):
    """
    Bar chart: Peak GPU VRAM usage per model variant.
    Shows Model C's memory footprint is only marginally larger than Model A.
    """
    setup_style()

    vram_data = {}
    for name in run_names:
        path = Path(log_dir) / f"{name}_epoch.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        model_key = name.split("_")[0]
        if "peak_vram_mb" in df.columns:
            vram_data[model_key] = df["peak_vram_mb"].max()

    if not vram_data:
        print(f"  No VRAM data found in {log_dir}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    models = list(vram_data.keys())
    values = [vram_data[m] for m in models]
    colors = [get_color(m) for m in models]
    labels = [get_label(m) for m in models]

    bars = ax.bar(range(len(models)), values, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val:.0f} MB", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Peak GPU VRAM (MB)")
    ax.set_title(f"VRAM Usage Comparison (L={depth})")
    save_fig(fig, output_dir, f"vram_comparison_l{depth}")
