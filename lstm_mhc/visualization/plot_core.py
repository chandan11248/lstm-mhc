"""
Core publication-quality plots for the LSTM-µHC L=4 base experiment.

Implements the headline figures from required_visual.md §1-§2:
    - Gradient norm vs training steps (§1C)
    - Amax gain vs training steps (§1D)
    - Training loss smoothness (§2)
    - Validation MSE vs epochs (§2)
    - Horizon degradation (MAE vs forecast horizon) (§2)

Color/label scheme follows AGENTS.md §8 (Model Legend).

Each plot is rendered at 300 DPI in BOTH PNG and PDF.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (Kaggle/server safe)
import matplotlib.pyplot as plt
import numpy as np

# --- Model Legend (AGENTS.md §8) ------------------------------------------
COLORS = {
    "model_a": "#1f77b4",   # Blue
    "model_b": "#d62728",   # Red
    "model_c": "#2ca02c",   # Green
    "classical": "#7f7f7f",  # Gray
    "neural": "#9467bd",     # Purple
    "sota": "#111111",       # Near black
}
LABELS = {
    "model_a": "Model A (Standard LSTM)",
    "model_b": "Model B (Naive HC-LSTM)",
    "model_c": "Model C (mHC-LSTM, Ours)",
}
STYLES = {
    "model_a": {"color": COLORS["model_a"], "linestyle": "-", "linewidth": 1.8},
    "model_b": {"color": COLORS["model_b"], "linestyle": "--", "linewidth": 1.8},
    "model_c": {"color": COLORS["model_c"], "linestyle": "-", "linewidth": 2.0},
}

# Map run directory name -> (model_key, friendly label)
# Trainer produces runs named like: mhc_l4_s42, hc_l4_s42, vanilla_l4_s42
RUN_MODEL_MAP = {
    "mhc": "model_c",
    "hc": "model_b",
    "vanilla": "model_a",
}


def _model_key_from_run(run_name: str) -> str:
    """Infer the model key (model_a/b/c) from a run name like 'mhc_l4_s42'."""
    prefix = run_name.split("_")[0].lower()
    return RUN_MODEL_MAP.get(prefix, prefix)


def _depth_from_run(run_name: str) -> Optional[int]:
    """Extract depth (4/8/16) from a run name like 'mhc_l8_s42'. Returns None if not present."""
    import re
    m = re.search(r"_l(\d+)_", run_name)
    return int(m.group(1)) if m else None


def _filter_runs_by_depth(
    runs: Dict[str, Tuple[Path, Path]], depth: int
) -> Dict[str, Tuple[Path, Path]]:
    """Return only runs whose name encodes the requested depth (e.g. l4)."""
    return {
        name: paths
        for name, paths in runs.items()
        if _depth_from_run(name) == depth
    }


def _group_runs_by_model(
    runs: Dict[str, Tuple[Path, Path]],
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Group runs by model key (model_a/b/c), preserving the run order."""
    grouped: Dict[str, List[Tuple[Path, Path]]] = {"model_a": [], "model_b": [], "model_c": []}
    for name, paths in runs.items():
        mk = _model_key_from_run(name)
        grouped.setdefault(mk, []).append(paths)
    return grouped


def _step_arrays_per_seed(
    runs: Dict[str, Tuple[Path, Path]], col: str
) -> Optional[Dict[str, np.ndarray]]:
    """For each model, stack the requested step-CSV column across all seeds.

    Returns a dict: model_key -> (n_seeds, n_steps) array, plus "_steps" for the
    common step axis. Models whose column is all-NaN (e.g. vanilla has no H_res,
    so its fwd_amax/bwd_amax are always NaN) are simply omitted. Returns None
    only if NO model produced any data.
    """
    grouped = _group_runs_by_model(runs)
    steps_ref: Optional[np.ndarray] = None
    out: Dict[str, np.ndarray] = {}
    for mk in ["model_a", "model_b", "model_c"]:
        seed_csvs = grouped.get(mk, [])
        if not seed_csvs:
            continue
        per_seed = []
        steps_this: Optional[np.ndarray] = None
        for step_csv, _ in seed_csvs:
            d = read_step_csv(step_csv)
            arr = d[col]
            if arr.size == 0 or np.all(np.isnan(arr)):
                continue
            per_seed.append(arr)
            steps_this = d["step"] if steps_this is None else steps_this
        if not per_seed:
            continue
        # Align by common length
        n = min(s.size for s in per_seed)
        per_seed = np.stack([s[:n] for s in per_seed], axis=0)
        out[mk] = per_seed
        if steps_ref is None:
            steps_ref = steps_this[:n]
        else:
            steps_ref = steps_ref[: min(steps_ref.size, n)]
    if not out:
        return None
    return {"_steps": steps_ref, **out}


def _epoch_arrays_per_seed(
    runs: Dict[str, Tuple[Path, Path]], col: str
) -> Optional[Dict[str, np.ndarray]]:
    """For each model, stack the requested epoch-CSV column across all seeds."""
    grouped = _group_runs_by_model(runs)
    epochs_ref: Optional[np.ndarray] = None
    out: Dict[str, np.ndarray] = {}
    for mk in ["model_a", "model_b", "model_c"]:
        seed_csvs = grouped.get(mk, [])
        if not seed_csvs:
            continue
        per_seed = []
        epochs_this: Optional[np.ndarray] = None
        for _, epoch_csv in seed_csvs:
            d = read_epoch_csv(epoch_csv)
            arr = d.get(col, np.array([]))
            if arr.size == 0 or np.all(np.isnan(arr)):
                continue
            per_seed.append(arr)
            epochs_this = d.get("epoch", np.array([])) if epochs_this is None else epochs_this
        if not per_seed:
            return None
        n = min(s.size for s in per_seed)
        per_seed = np.stack([s[:n] for s in per_seed], axis=0)
        out[mk] = per_seed
        epochs_ref = epochs_this[:n] if epochs_ref is None else epochs_ref[:n]
    return {"_epochs": epochs_ref, **out}


def _plot_mean_std_band(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.8,
    alpha_band: float = 0.18,
) -> None:
    """Plot a mean line with a translucent ±1 std band (the spec's required format).

    The band is computed multiplicatively in log space: ``lower = mean / (1 + std/mean)``
    and ``upper = mean * (1 + std/mean)``. This avoids the vertical-line artifacts
    that appear on log-scale axes when a single seed spikes (making std > mean) and
    the naive ``mean - std`` would go negative and be clipped to a small constant.
    """
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    ratio = std / np.maximum(mean, 1e-30)
    lower = mean / (1.0 + ratio)
    upper = mean * (1.0 + ratio)
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=alpha_band,
        linewidth=0,
    )


def read_step_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read a *_step.csv produced by MetricsLogger.

    Columns: step, train_loss, grad_norm, fwd_amax, bwd_amax
    Returns a dict of column -> np.ndarray (empty cols become NaN-filled arrays).
    """
    cols = ["step", "train_loss", "grad_norm", "fwd_amax", "bwd_amax"]
    data: Dict[str, list] = {c: [] for c in cols}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in cols:
                v = row.get(c, "")
                data[c].append(float(v) if v not in (None, "") else np.nan)
    return {c: np.array(data[c], dtype=float) for c in cols}


def read_epoch_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read a *_epoch.csv produced by MetricsLogger."""
    data: Dict[str, list] = {}
    header = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for h in header:
            data[h] = []
        for row in reader:
            for h, v in zip(header, row):
                data[h].append(float(v) if v not in (None, "") else np.nan)
    return {h: np.array(data[h], dtype=float) for h in header}


def _save(fig, out_dir: Path, name: str) -> Tuple[Path, Path]:
    """Save a figure in both PNG (300 DPI) and PDF. Returns (png_path, pdf_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _smooth(y: np.ndarray, window: int = 51, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay style smoothing; falls back to moving average if too short."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    if n < window:
        w = max(1, min(n, 5))
        if w % 2 == 0:
            w -= 1
        if w < 1:
            return y
        kernel = np.ones(w) / w
        return np.convolve(y, kernel, mode="same")
    try:
        from scipy.signal import savgol_filter
        win = window if window % 2 == 1 else window + 1
        win = min(win, n if n % 2 == 1 else n - 1)
        return savgol_filter(y, win, poly)
    except Exception:
        w = 11
        kernel = np.ones(w) / w
        return np.convolve(y, kernel, mode="same")


def discover_runs(runs_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Find all training runs in runs_dir.

    Returns: {run_name: (step_csv_path, epoch_csv_path)}
    A run is identified by a pair of files named f"{run}_step.csv" and f"{run}_epoch.csv".
    """
    runs = {}
    if not runs_dir.exists():
        return runs
    step_files = {p.stem.replace("_step", ""): p for p in runs_dir.glob("*_step.csv")}
    epoch_files = {p.stem.replace("_epoch", ""): p for p in runs_dir.glob("*_epoch.csv")}
    for run in step_files.keys() & epoch_files.keys():
        runs[run] = (step_files[run], epoch_files[run])
    return runs


# =====================================================================
#  Plot functions
# =====================================================================

def plot_gradient_norm_vs_steps(
    runs_dir: Path, out_dir: Path, name: str = "gradient_norm_vs_steps_l4"
) -> Tuple[Path, Path]:
    """§1C — L2 gradient norm vs training steps, mean ± std across 3 seeds (L=4)."""
    runs = _filter_runs_by_depth(discover_runs(runs_dir), 4)
    agg = _step_arrays_per_seed(runs, "grad_norm")
    if agg is None:
        raise FileNotFoundError(f"No L=4 step CSVs with grad_norm found in {runs_dir}")
    steps = agg.pop("_steps")
    fig, ax = plt.subplots(figsize=(8, 5))
    for mk in ["model_a", "model_b", "model_c"]:
        if mk not in agg:
            continue
        per_seed = agg[mk]  # (n_seeds, n_steps)
        per_seed_smooth = np.stack([_smooth(row) for row in per_seed], axis=0)
        mean = per_seed_smooth.mean(axis=0)
        std = per_seed_smooth.std(axis=0)
        _plot_mean_std_band(
            ax, steps, mean, std,
            label=LABELS[mk], color=COLORS[mk],
            linestyle=STYLES[mk]["linestyle"], linewidth=STYLES[mk]["linewidth"],
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("L₂ Gradient Norm (smoothed, mean ± std, 3 seeds)")
    ax.set_yscale("log")
    ax.set_title("Gradient Norm vs Training Steps (L=4)")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, out_dir, name)


def plot_amax_gain_vs_steps(
    runs_dir: Path, out_dir: Path, name: str = "amax_gain_vs_steps_l4"
) -> Tuple[Path, Path]:
    """§1D — Composite Amax gain (forward + backward) vs steps, mean ± std, 3 seeds (L=4).

    Amax is only logged every 100 steps in the trainer (expensive metric), so
    ~99% of step rows are NaN. We drop the NaN rows and plot the surviving
    sparse points with markers so the reader sees the actual sampling density.
    """
    runs = _filter_runs_by_depth(discover_runs(runs_dir), 4)
    agg_fwd = _step_arrays_per_seed(runs, "fwd_amax")
    agg_bwd = _step_arrays_per_seed(runs, "bwd_amax")
    if agg_fwd is None and agg_bwd is None:
        raise FileNotFoundError(f"No L=4 step CSVs with fwd_amax/bwd_amax found in {runs_dir}")
    steps = (agg_fwd or agg_bwd).pop("_steps")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for mk in ["model_b", "model_c"]:
        for agg, ax, title in [
            (agg_fwd, axes[0], "Forward Amax Gain (max row sum)"),
            (agg_bwd, axes[1], "Backward Amax Gain (max col sum)"),
        ]:
            if agg is None or mk not in agg:
                continue
            per_seed = agg[mk]
            mean = np.nanmean(per_seed, axis=0)
            std = np.nanstd(per_seed, axis=0)
            # Drop NaN steps (the Amax is only logged every 100 steps) so
            # the marker scatter actually shows up instead of being hidden
            # by sparse line segments.
            mask = np.isfinite(mean) & np.isfinite(std)
            x_valid = steps[mask]
            mean_valid = mean[mask]
            std_valid = std[mask]
            style = STYLES[mk]
            ax.errorbar(
                x_valid, mean_valid, yerr=std_valid,
                label=LABELS[mk], color=style["color"],
                linestyle=style["linestyle"], linewidth=1.0,
                marker="o", markersize=3, capsize=0, alpha=0.85,
            )
    axes[0].set_xlabel("Training Step"); axes[0].set_ylabel("Amax Gain (mean ± std, 3 seeds)")
    axes[0].set_title("Forward Amax Gain (max row sum)")
    axes[0].set_yscale("log"); axes[0].grid(True, which="both", alpha=0.3)
    # Place legend in the lower-left so it doesn't overlap the Model C = 1.0 line
    axes[0].legend(frameon=False, loc="lower left")
    axes[1].set_xlabel("Training Step"); axes[1].set_title("Backward Amax Gain (max col sum)")
    axes[1].set_yscale("log"); axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Composite Amax Gain vs Training Steps (L=4, 3 seeds, logged every 100 steps)")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_training_loss_smoothness(
    runs_dir: Path, out_dir: Path, name: str = "training_loss_smoothness_l4"
) -> Tuple[Path, Path]:
    """§2 — Training loss vs steps, smoothed, mean ± std across 3 seeds (L=4)."""
    runs = _filter_runs_by_depth(discover_runs(runs_dir), 4)
    agg = _step_arrays_per_seed(runs, "train_loss")
    if agg is None:
        raise FileNotFoundError(f"No L=4 step CSVs with train_loss found in {runs_dir}")
    steps = agg.pop("_steps")
    fig, ax = plt.subplots(figsize=(8, 5))
    for mk in ["model_a", "model_b", "model_c"]:
        if mk not in agg:
            continue
        per_seed = agg[mk]
        per_seed_smooth = np.stack([_smooth(row) for row in per_seed], axis=0)
        mean = per_seed_smooth.mean(axis=0)
        std = per_seed_smooth.std(axis=0)
        _plot_mean_std_band(
            ax, steps, mean, std,
            label=LABELS[mk], color=COLORS[mk],
            linestyle=STYLES[mk]["linestyle"], linewidth=STYLES[mk]["linewidth"],
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss (MSE, smoothed, mean ± std)")
    ax.set_yscale("log")
    ax.set_title("Training Loss Smoothness (L=4, 3 seeds)")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, out_dir, name)


def plot_validation_mse_vs_epochs(
    runs_dir: Path, out_dir: Path, name: str = "validation_mse_vs_epochs_l4"
) -> Tuple[Path, Path]:
    """§2 — Validation MSE vs epoch, mean ± std across 3 seeds (L=4)."""
    runs = _filter_runs_by_depth(discover_runs(runs_dir), 4)
    agg = _epoch_arrays_per_seed(runs, "val_mse")
    if agg is None:
        raise FileNotFoundError(f"No L=4 epoch CSVs with val_mse found in {runs_dir}")
    epochs = agg.pop("_epochs")
    fig, ax = plt.subplots(figsize=(8, 5))
    for mk in ["model_a", "model_b", "model_c"]:
        if mk not in agg:
            continue
        per_seed = agg[mk]
        mean = per_seed.mean(axis=0)
        std = per_seed.std(axis=0)
        _plot_mean_std_band(
            ax, epochs, mean, std,
            label=LABELS[mk], color=COLORS[mk],
            linestyle=STYLES[mk]["linestyle"], linewidth=STYLES[mk]["linewidth"],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (mean ± std)")
    ax.set_title("Validation MSE vs Epochs (L=4, 3 seeds)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, name)


def plot_horizon_degradation(
    runs_dir: Path, out_dir: Path, horizons: List[int], name: str = "horizon_degradation_l4"
) -> Tuple[Path, Path]:
    """§2 — MAE vs forecast horizon at best epoch, mean ± std across 3 seeds (L=4)."""
    runs = _filter_runs_by_depth(discover_runs(runs_dir), 4)
    grouped = _group_runs_by_model(runs)
    h_keys = [f"horizon_{h}h_mae" for h in horizons]
    per_model: Dict[str, np.ndarray] = {}
    for mk in ["model_a", "model_b", "model_c"]:
        rows = []
        for _, epoch_csv in grouped.get(mk, []):
            d = read_epoch_csv(epoch_csv)
            if "val_mse" not in d or d["val_mse"].size == 0:
                continue
            best_idx = int(np.nanargmin(d["val_mse"]))
            maes = [d[hk][best_idx] if hk in d and d[hk].size > best_idx else np.nan for hk in h_keys]
            rows.append(maes)
        if not rows:
            continue
        per_model[mk] = np.array(rows, dtype=float)  # (n_seeds, n_horizons)
    if not per_model:
        raise FileNotFoundError("No L=4 horizon data found")
    fig, ax = plt.subplots(figsize=(8, 5))
    for mk in ["model_a", "model_b", "model_c"]:
        if mk not in per_model:
            continue
        arr = per_model[mk]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.errorbar(
            horizons, mean, yerr=std,
            label=LABELS[mk], color=COLORS[mk],
            linestyle=STYLES[mk]["linestyle"], linewidth=STYLES[mk]["linewidth"],
            marker="o", markersize=6, capsize=3,
        )
    ax.set_xlabel("Forecast Horizon (hours)")
    ax.set_ylabel("MAE (z-score units, mean ± std)")
    ax.set_title("Horizon Degradation — MAE at Best Epoch (L=4, 3 seeds)")
    ax.set_xticks(horizons)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, name)


def generate_all_l4(runs_dir: Path, out_dir: Path, horizons: Optional[List[int]] = None) -> List[Path]:
    """Generate every L=4 core plot. Returns list of produced PNG paths.

    Skips (with a printed warning) any plot that can't be built from the
    available logs, so a partial run still yields the plots it can.
    """
    horizons = horizons or [6, 12, 24, 48, 72]
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []
    for fn, args in [
        (plot_gradient_norm_vs_steps, (runs_dir, out_dir)),
        (plot_amax_gain_vs_steps, (runs_dir, out_dir)),
        (plot_training_loss_smoothness, (runs_dir, out_dir)),
        (plot_validation_mse_vs_epochs, (runs_dir, out_dir)),
        (plot_horizon_degradation, (runs_dir, out_dir, horizons)),
    ]:
        try:
            png, _ = fn(*args)
            produced.append(png)
            print(f"  ✓ {png.name}")
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    return produced
