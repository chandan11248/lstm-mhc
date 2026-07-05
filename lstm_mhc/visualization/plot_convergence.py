"""
Convergence plots: Training loss smoothness, Validation MSE vs epochs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from .plot_style import setup_style, save_fig, get_style, get_label


def load_epoch_logs(log_dir: str, run_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Load epoch-level CSV logs for multiple runs."""
    logs = {}
    for name in run_names:
        path = Path(log_dir) / f"{name}_epoch.csv"
        if path.exists():
            logs[name] = pd.read_csv(path)
    return logs


def load_step_logs(log_dir: str, run_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Load step-level CSV logs for multiple runs."""
    logs = {}
    for name in run_names:
        path = Path(log_dir) / f"{name}_step.csv"
        if path.exists():
            logs[name] = pd.read_csv(path)
    return logs


def plot_training_loss_smoothness(
    log_dir: str,
    run_names: List[str],
    output_dir: str = "outputs/plots",
    depth: int = 4,
    window: int = 100,
):
    """
    Plot training loss convergence with rolling average overlay.
    
    Shows Model C converges smoother and lower than Model A,
    while Model B oscillates wildly.
    """
    setup_style()
    logs = load_step_logs(log_dir, run_names)
    if not logs:
        print(f"  No step logs found in {log_dir}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, df in logs.items():
        if "train_loss" not in df.columns:
            continue
        model_key = name.split("_")[0]
        style = get_style(model_key)
        label = get_label(model_key)

        raw = df["train_loss"].values.astype(float)
        raw = np.nan_to_num(raw, nan=0.0, posinf=np.nanmax(raw[~np.isinf(raw)]) if np.any(~np.isinf(raw)) else 1.0)
        smoothed = pd.Series(raw).rolling(window=window, min_periods=1).mean()

        ax.plot(df["step"], raw, alpha=0.08, color=style["color"], linewidth=0.5)
        ax.plot(df["step"], smoothed, label=label, color=style["color"],
                linestyle=style["linestyle"], linewidth=style["linewidth"])

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title(f"Training Loss Convergence (L={depth})")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    save_fig(fig, output_dir, f"training_loss_smoothness_l{depth}")


def plot_validation_mse_vs_epochs(
    log_dir: str,
    run_names: List[str],
    output_dir: str = "outputs/plots",
    depth: int = 4,
):
    """
    Plot validation MSE vs epochs for all models.
    
    Shows Model C converges to lower error than Model A.
    """
    setup_style()
    logs = load_epoch_logs(log_dir, run_names)
    if not logs:
        print(f"  No epoch logs found in {log_dir}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in logs.items():
        if "val_mse" not in df.columns:
            continue
        model_key = name.split("_")[0]
        style = get_style(model_key)
        label = get_label(model_key)

        ax.plot(df["epoch"] + 1, df["val_mse"], label=label,
                color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_title(f"Validation MSE vs Epochs (L={depth})")
    ax.legend(loc="upper right")
    save_fig(fig, output_dir, f"validation_mse_vs_epochs_l{depth}")
