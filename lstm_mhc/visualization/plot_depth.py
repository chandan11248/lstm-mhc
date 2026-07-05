"""
Depth scaling plots: Performance vs depth, gradient vs depth, Amax vs depth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from .plot_style import setup_style, save_fig, get_style, get_label, get_color


def load_depth_results(log_dir: str, depths: List[int], seeds: List[int] = None
                       ) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Load epoch logs for all depths and models."""
    seeds = seeds or [42, 43, 44, 45, 46]
    results = {}
    for d in depths:
        results[d] = {}
        for model in ["vanilla", "hc", "mhc"]:
            seed_vals = []
            for s in seeds:
                name = f"{model}_l{d}_s{s}"
                path = Path(log_dir) / f"{name}_epoch.csv"
                if path.exists():
                    seed_vals.append(pd.read_csv(path))
            if seed_vals:
                results[d][model] = seed_vals
    return results


def plot_performance_vs_depth(
    log_dir: str,
    depths: List[int] = None,
    output_dir: str = "outputs/plots",
    seeds: List[int] = None,
):
    """
    Grouped bar chart: Validation MSE at each depth (L=4, 8, 16).
    Shows Model A degrades with depth, Model C maintains or improves.
    """
    setup_style()
    depths = depths or [4, 8, 16]
    results = load_depth_results(log_dir, depths, seeds)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(depths))
    models = ["vanilla", "hc", "mhc"]
    width = 0.25

    for i, model in enumerate(models):
        means, stds = [], []
        for d in depths:
            if d in results and model in results[d]:
                best_mses = [df["val_mse"].min() for df in results[d][model]]
                means.append(np.mean(best_mses))
                stds.append(np.std(best_mses))
            else:
                means.append(0)
                stds.append(0)
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=get_label(model),
               color=get_color(model), alpha=0.85, edgecolor="white", capsize=3)

    ax.set_xlabel("Network Depth (L)")
    ax.set_ylabel("Best Validation MSE")
    ax.set_title("Performance vs Depth")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={d}" for d in depths])
    ax.legend(loc="upper right")
    save_fig(fig, output_dir, "performance_vs_depth")


def plot_gradient_norm_vs_depth(
    log_dir: str,
    depths: List[int] = None,
    output_dir: str = "outputs/plots",
    seeds: List[int] = None,
):
    """
    3-panel line chart: Gradient norm vs steps at each depth.
    Shows instability worsens with depth for A and B, not C.
    """
    setup_style()
    depths = depths or [4, 8, 16]

    fig, axes = plt.subplots(1, len(depths), figsize=(18, 5), sharey=True)
    if len(depths) == 1:
        axes = [axes]

    for ax, d in zip(axes, depths):
        for model in ["vanilla", "hc", "mhc"]:
            style = get_style(model)
            seed_max_norms = []
            for s in (seeds or [42, 43, 44, 45, 46]):
                name = f"{model}_l{d}_s{s}"
                path = Path(log_dir) / f"{name}_step.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    if "grad_norm" in df.columns:
                        norms = df["grad_norm"].dropna().values.astype(float)
                        norms = np.nan_to_num(norms, nan=0.0, posinf=1e6)
                        smoothed = pd.Series(norms).rolling(50, min_periods=1).mean()
                        ax.plot(df["step"], smoothed, color=style["color"],
                                linestyle=style["linestyle"], alpha=0.7, linewidth=1.5)
                        seed_max_norms.append(np.max(norms))
            if seed_max_norms:
                max_norm = max(seed_max_norms)
                ax.annotate(f"max={max_norm:.1f}", xy=(0.95, 0.95),
                            xycoords="axes fraction", ha="right", va="top", fontsize=8)

        ax.set_title(f"L={d}")
        ax.set_xlabel("Training Steps")
        ax.set_yscale("log")

    axes[0].set_ylabel("L₂ Gradient Norm")
    fig.suptitle("Gradient Norm vs Depth", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, output_dir, "gradient_norm_vs_depth")


def plot_amax_gain_vs_depth(
    log_dir: str,
    depths: List[int] = None,
    output_dir: str = "outputs/plots",
    seeds: List[int] = None,
):
    """
    Grouped bar chart: Max composite Amax gain at each depth.
    Shows Model B's gain grows exponentially, Model C stays bounded.
    """
    setup_style()
    depths = depths or [4, 8, 16]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(depths))
    width = 0.35

    for i, model in enumerate(["hc", "mhc"]):
        max_gains = []
        for d in depths:
            gains = []
            for s in (seeds or [42, 43, 44, 45, 46]):
                name = f"{model}_l{d}_s{s}"
                path = Path(log_dir) / f"{name}_step.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    if "fwd_amax" in df.columns:
                        vals = df["fwd_amax"].dropna().values.astype(float)
                        if len(vals) > 0:
                            gains.append(np.max(vals))
            max_gains.append(np.mean(gains) if gains else 0)

        offset = (i - 0.5) * width
        ax.bar(x + offset, max_gains, width, label=get_label(model),
               color=get_color(model), alpha=0.85, edgecolor="white")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=1.6, color="#2ca02c", linestyle=":", alpha=0.5)
    ax.set_xlabel("Network Depth (L)")
    ax.set_ylabel("Max Forward Amax Gain")
    ax.set_title("Composite Amax Gain vs Depth")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={d}" for d in depths])
    ax.legend(loc="upper left")
    save_fig(fig, output_dir, "amax_gain_vs_depth")


def plot_depth_loss_convergence(
    log_dir: str,
    depths: List[int] = None,
    output_dir: str = "outputs/plots",
    seeds: List[int] = None,
):
    """
    Multi-panel line chart: Validation MSE vs epochs at each depth.
    """
    setup_style()
    depths = depths or [4, 8, 16]

    fig, axes = plt.subplots(1, len(depths), figsize=(18, 5), sharey=True)
    if len(depths) == 1:
        axes = [axes]

    for ax, d in zip(axes, depths):
        for model in ["vanilla", "hc", "mhc"]:
            style = get_style(model)
            for s in (seeds or [42, 43, 44, 45, 46]):
                name = f"{model}_l{d}_s{s}"
                path = Path(log_dir) / f"{name}_epoch.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    ax.plot(df["epoch"] + 1, df["val_mse"], color=style["color"],
                            linestyle=style["linestyle"], alpha=0.6, linewidth=1.5)
        ax.set_title(f"L={d}")
        ax.set_xlabel("Epoch")

    axes[0].set_ylabel("Validation MSE")
    fig.suptitle("Loss Convergence vs Depth", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, output_dir, "depth_loss_convergence")
