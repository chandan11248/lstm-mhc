"""
Stability plots: Gradient norm vs steps, Amax gain vs steps.

These are the 'smoking gun' visuals proving the Sinkhorn-Knopp constraint works.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from .plot_style import setup_style, save_fig, get_style, get_label, get_color


def load_step_logs(log_dir: str, run_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Load step-level CSV logs for multiple runs."""
    logs = {}
    for name in run_names:
        path = Path(log_dir) / f"{name}_step.csv"
        if path.exists():
            logs[name] = pd.read_csv(path)
    return logs


def plot_gradient_norm_vs_steps(
    log_dir: str,
    run_names: List[str],
    output_dir: str = "outputs/plots",
    depth: int = 4,
    window: int = 50,
):
    """
    Plot L2 gradient norm vs training steps for all models.
    
    Shows Model B spiking wildly while Model C and Model A stay stable.
    Uses log-scale Y-axis with rolling average for clarity.
    
    Args:
        log_dir: Directory containing step CSV logs.
        run_names: List of run names (e.g., ["vanilla_l4_s42", "hc_l4_s42", "mhc_l4_s42"]).
        output_dir: Where to save the plot.
        depth: Network depth for title.
        window: Rolling average window size.
    """
    setup_style()
    logs = load_step_logs(log_dir, run_names)
    if not logs:
        print(f"  No step logs found in {log_dir}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, df in logs.items():
        if "grad_norm" not in df.columns or df["grad_norm"].dropna().empty:
            continue
        model_key = name.split("_")[0]  # vanilla, hc, or mhc
        style = get_style(model_key)
        label = get_label(model_key)

        raw = df["grad_norm"].values.astype(float)
        # Replace NaN/inf for plotting
        raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=0.0)
        smoothed = pd.Series(raw).rolling(window=window, min_periods=1).mean()

        # Faint raw line + bold smoothed line
        ax.plot(df["step"], raw, alpha=0.1, color=style["color"], linewidth=0.5)
        ax.plot(df["step"], smoothed, label=label, color=style["color"],
                linestyle=style["linestyle"], linewidth=style["linewidth"])

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("L₂ Gradient Norm (raw)")
    ax.set_title(f"Gradient Norm vs Training Steps (L={depth})")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3, label="Clip threshold")
    save_fig(fig, output_dir, f"gradient_norm_vs_steps_l{depth}")


def plot_amax_gain_vs_steps(
    log_dir: str,
    run_names: List[str],
    output_dir: str = "outputs/plots",
    depth: int = 4,
):
    """
    Plot composite Amax gain (forward + backward) vs training steps.
    
    Shows Model B reaching ~3000 (exploding) while Model C stays bounded ~1.6.
    
    Args:
        log_dir: Directory containing step CSV logs.
        run_names: List of run names (HC and mHC models only).
        output_dir: Where to save the plot.
        depth: Network depth for title.
    """
    setup_style()
    logs = load_step_logs(log_dir, run_names)
    if not logs:
        print(f"  No step logs found in {log_dir}")
        return

    fig, (ax_fwd, ax_bwd) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for name, df in logs.items():
        model_key = name.split("_")[0]
        style = get_style(model_key)
        label = get_label(model_key)

        # Forward Amax
        if "fwd_amax" in df.columns and df["fwd_amax"].dropna().any():
            steps = df.loc[df["fwd_amax"].notna(), "step"]
            vals = df.loc[df["fwd_amax"].notna(), "fwd_amax"].astype(float)
            ax_fwd.plot(steps, vals, label=label, color=style["color"],
                        linestyle=style["linestyle"], linewidth=style["linewidth"])

        # Backward Amax
        if "bwd_amax" in df.columns and df["bwd_amax"].dropna().any():
            steps = df.loc[df["bwd_amax"].notna(), "step"]
            vals = df.loc[df["bwd_amax"].notna(), "bwd_amax"].astype(float)
            ax_bwd.plot(steps, vals, label=label, color=style["color"],
                        linestyle=style["linestyle"], linewidth=style["linewidth"])

    # Reference lines
    for ax in [ax_fwd, ax_bwd]:
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=1.6, color="#2ca02c", linestyle=":", alpha=0.5)
        ax.set_yscale("log")
        ax.legend(loc="upper left")

    ax_fwd.set_ylabel("Forward Amax Gain (max row sum)")
    ax_fwd.set_title(f"Composite Amax Gain vs Training Steps (L={depth})")
    ax_bwd.set_ylabel("Backward Amax Gain (max col sum)")
    ax_bwd.set_xlabel("Training Steps")
    save_fig(fig, output_dir, f"amax_gain_vs_steps_l{depth}")
