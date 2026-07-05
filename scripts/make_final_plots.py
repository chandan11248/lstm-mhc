#!/usr/bin/env python3
"""Generate paper-ready figures from MEASURED test + stability tables only.

Sources (all measured, no fabrication):
  - outputs/tables/test_metrics_summary.csv   (held-out TEST metrics, mean/std over seeds)
  - outputs/tables/phase1_stability.csv        (training-time grad/spectral norms)

Outputs (300 DPI PNG + vector PDF) -> outputs/final_ready_plots/

Figures:
  1. depth_scaling_test_mse_{dataset}     : test MSE vs depth (mHC/Vanilla/GRU/TCN)
  2. spectral_norm_vs_depth               : mHC (=1) vs HC (explodes), both datasets (log)
  3. gradient_norm_vs_depth               : mHC/Vanilla/HC max grad norm vs depth (log)
  4. test_leaderboard_L4_{dataset}        : all models' test MSE at L=4 (fair shallow comparison)
  5. horizon_mae_{dataset}_l{depth}       : per-horizon test MAE for core models

Missing/diverged data is omitted (never interpolated). HC diverges at depth and
is intentionally excluded from accuracy plots (shown only in stability figures),
with a caption note in the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
OUTDIR = ROOT / "outputs" / "final_ready_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Consistent legend (AGENTS.md §8).
STYLE = {
    "vanilla": dict(label="Standard LSTM", color="#1f77b4", linestyle="-", marker="o"),
    "hc":      dict(label="HC-LSTM (unconstrained)", color="#d62728", linestyle="--", marker="s"),
    "mhc":     dict(label="mHC-LSTM (ours)", color="#2ca02c", linestyle="-", marker="D"),
    "gru":     dict(label="GRU", color="#9467bd", linestyle="-.", marker="^"),
    "tcn":     dict(label="TCN", color="#8c564b", linestyle="-.", marker="v"),
    "ridge":   dict(label="Ridge", color="#7f7f7f", linestyle=":", marker="P"),
    "dlinear": dict(label="DLinear", color="#aaaaaa", linestyle=":", marker="X"),
    "nlinear": dict(label="NLinear", color="#cccccc", linestyle=":", marker="*"),
}
DATASET_TITLE = {"weather": "NOAA Weather (JFK)", "ett": "ETT (ETTh1)", "ettm1": "ETTm1 (large)"}
DEPTHS = [4, 8, 16, 32]

plt.rcParams.update({
    "font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 100, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def load_summary():
    # Prefer the consolidated final summary (includes L=32 + ETTm1); fall back.
    final = TABLES / "test_metrics_final_summary.csv"
    if final.exists():
        return pd.read_csv(final)
    return pd.read_csv(TABLES / "test_metrics_summary.csv")


def load_stability():
    df = pd.read_csv(TABLES / "phase1_stability.csv")
    g = df.groupby(["dataset", "depth", "model"]).agg(
        max_grad_norm=("max_grad_norm", "max"),
        max_spectral_norm=("max_spectral_norm", "max"),
    ).reset_index()
    return g


# ------------------------------------------------------------------
# 1. Depth scaling — test MSE vs depth
# ------------------------------------------------------------------
def fig_depth_scaling(summary):
    for ds in ("weather", "ett"):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for model in ("mhc", "vanilla", "gru", "tcn"):
            rows = summary[(summary.dataset == ds) & (summary.model == model)]
            xs, ys, es = [], [], []
            for d in DEPTHS:
                r = rows[rows.depth == d]
                if len(r):
                    xs.append(d)
                    ys.append(float(r["test_mse_mean"].iloc[0]))
                    es.append(float(r["test_mse_std"].iloc[0]))
            if xs:
                st = STYLE[model]
                ax.errorbar(xs, ys, yerr=es, capsize=3, linewidth=2, markersize=7,
                            label=st["label"], color=st["color"],
                            linestyle=st["linestyle"], marker=st["marker"])
        ax.set_xlabel("Depth (LSTM layers $L$)")
        ax.set_ylabel("Test MSE (original units)")
        ax.set_title(f"Depth scaling — {DATASET_TITLE[ds]}")
        ax.set_xticks(DEPTHS)
        ax.legend(fontsize=9, framealpha=0.9)
        save(fig, f"depth_scaling_test_mse_{ds}")


# ------------------------------------------------------------------
# 2. Spectral norm vs depth (smoking gun)
# ------------------------------------------------------------------
def fig_spectral_norm(stab):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds in zip(axes, ("weather", "ett")):
        for model in ("mhc", "hc"):
            rows = stab[(stab.dataset == ds) & (stab.model == model)]
            xs, ys = [], []
            for d in DEPTHS:
                r = rows[rows.depth == d]
                if len(r) and np.isfinite(r["max_spectral_norm"].iloc[0]):
                    xs.append(d)
                    ys.append(float(r["max_spectral_norm"].iloc[0]))
            if xs:
                st = STYLE[model]
                ax.plot(xs, ys, linewidth=2, markersize=8, label=st["label"],
                        color=st["color"], linestyle=st["linestyle"], marker=st["marker"])
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("Depth (LSTM layers $L$)")
        ax.set_ylabel("Max spectral norm of $\\mathcal{H}^{res}$ composite")
        ax.set_title(DATASET_TITLE[ds])
        ax.set_xticks(DEPTHS)
        ax.legend(fontsize=9)
        ax.annotate("HC diverges at L=16\n(no completed epoch)", xy=(0.5, 0.06),
                    xycoords="axes fraction", ha="center", fontsize=8, color="#d62728")
    fig.suptitle("Manifold constraint keeps spectral norm near 1.0; unconstrained HC explodes", fontsize=12)
    save(fig, "spectral_norm_vs_depth")


# ------------------------------------------------------------------
# 3. Gradient norm vs depth
# ------------------------------------------------------------------
def fig_gradient_norm(stab):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds in zip(axes, ("weather", "ett")):
        for model in ("mhc", "vanilla", "hc"):
            rows = stab[(stab.dataset == ds) & (stab.model == model)]
            xs, ys = [], []
            for d in DEPTHS:
                r = rows[rows.depth == d]
                if len(r) and np.isfinite(r["max_grad_norm"].iloc[0]):
                    xs.append(d)
                    ys.append(float(r["max_grad_norm"].iloc[0]))
            if xs:
                st = STYLE[model]
                ax.plot(xs, ys, linewidth=2, markersize=8, label=st["label"],
                        color=st["color"], linestyle=st["linestyle"], marker=st["marker"])
        ax.set_yscale("log")
        ax.set_xlabel("Depth (LSTM layers $L$)")
        ax.set_ylabel("Max gradient norm (pre-clip)")
        ax.set_title(DATASET_TITLE[ds])
        ax.set_xticks(DEPTHS)
        ax.legend(fontsize=9)
    fig.suptitle("Gradient stability vs depth (log scale)", fontsize=12)
    save(fig, "gradient_norm_vs_depth")


# ------------------------------------------------------------------
# 4. L=4 leaderboard (fair shallow comparison, all models)
# ------------------------------------------------------------------
def fig_leaderboard(summary):
    order = ["nlinear", "dlinear", "ridge", "gru", "tcn", "vanilla", "hc", "mhc"]
    for ds in ("weather", "ett"):
        rows = summary[(summary.dataset == ds) & (summary.depth == 4)]
        data = []
        for m in order:
            r = rows[rows.model == m]
            if len(r):
                data.append((m, float(r["test_mse_mean"].iloc[0]), float(r["test_mse_std"].iloc[0])))
        if not data:
            continue
        data.sort(key=lambda t: t[1])
        labels = [STYLE[m]["label"] for m, _, _ in data]
        vals = [v for _, v, _ in data]
        errs = [e for _, _, e in data]
        colors = [STYLE[m]["color"] for m, _, _ in data]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(labels, vals, xerr=errs, color=colors, capsize=3, alpha=0.9)
        ax.invert_yaxis()
        ax.set_xlabel("Test MSE (original units)")
        ax.set_title(f"Shallow ($L=4$) test MSE — {DATASET_TITLE[ds]}")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
        save(fig, f"test_leaderboard_L4_{ds}")


# ------------------------------------------------------------------
# 5. Per-horizon test MAE
# ------------------------------------------------------------------
HORIZONS = [6, 12, 24, 48, 72]


def fig_horizon(summary):
    for ds in ("weather", "ett"):
        for depth in DEPTHS:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            plotted = False
            for model in ("mhc", "vanilla", "gru", "tcn"):
                r = summary[(summary.dataset == ds) & (summary.model == model) & (summary.depth == depth)]
                if not len(r):
                    continue
                ys = []
                for h in HORIZONS:
                    col = f"test_horizon_{h}h_mae_mean"
                    ys.append(float(r[col].iloc[0]) if col in r.columns and np.isfinite(r[col].iloc[0]) else np.nan)
                if all(np.isnan(ys)):
                    continue
                st = STYLE[model]
                ax.plot(HORIZONS, ys, linewidth=2, markersize=7, label=st["label"],
                        color=st["color"], linestyle=st["linestyle"], marker=st["marker"])
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlabel("Forecast horizon (hours)")
            ax.set_ylabel("Test MAE (original units)")
            ax.set_title(f"Horizon degradation — {DATASET_TITLE[ds]} ($L={depth}$)")
            ax.set_xticks(HORIZONS)
            ax.legend(fontsize=9)
            save(fig, f"horizon_mae_{ds}_l{depth}")


def fig_ettm1(summary):
    """ETTm1 (large dataset) L=32: mHC vs vanilla. Single seed (n=1), no error bars."""
    rows = summary[summary.dataset == "ettm1"]
    if rows.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    models, vals, colors = [], [], []
    for m in ("mhc", "vanilla"):
        r = rows[rows.model == m]
        if len(r):
            models.append(STYLE[m]["label"])
            vals.append(float(r["test_mse_mean"].iloc[0]))
            colors.append(STYLE[m]["color"])
    bars = ax1.bar(models, vals, color=colors, alpha=0.9, width=0.55)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("Test MSE (original units)")
    ax1.set_title("ETTm1 (69,680 rows) — $L=32$ test MSE")
    for m in ("mhc", "vanilla"):
        r = rows[rows.model == m]
        if not len(r):
            continue
        ys = [float(r[f"test_horizon_{h}h_mae_mean"].iloc[0])
              if f"test_horizon_{h}h_mae_mean" in r.columns and np.isfinite(r[f"test_horizon_{h}h_mae_mean"].iloc[0])
              else np.nan for h in HORIZONS]
        if all(np.isnan(ys)):
            continue
        st = STYLE[m]
        ax2.plot(HORIZONS, ys, linewidth=2, markersize=7, label=st["label"],
                 color=st["color"], linestyle=st["linestyle"], marker=st["marker"])
    ax2.set_xlabel("Forecast horizon (hours)")
    ax2.set_ylabel("Test MAE")
    ax2.set_title("ETTm1 — per-horizon test MAE ($L=32$)")
    ax2.set_xticks(HORIZONS)
    ax2.legend(fontsize=9)
    fig.suptitle("Larger dataset + depth: measured L=32 ETTm1 comparison", fontsize=12)
    save(fig, "ettm1_l32_mhc_vs_vanilla")


def main():
    print(f"Writing paper-ready figures to {OUTDIR}")
    summary = load_summary()
    stab = load_stability()
    fig_depth_scaling(summary)
    fig_spectral_norm(stab)
    fig_gradient_norm(stab)
    fig_leaderboard(summary)
    fig_horizon(summary)
    fig_ettm1(summary)
    n = len(list(OUTDIR.glob("*.png")))
    print(f"\nDone. {n} figures (PNG+PDF) in {OUTDIR}")


if __name__ == "__main__":
    main()
