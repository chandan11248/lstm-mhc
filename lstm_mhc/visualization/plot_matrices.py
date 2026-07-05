"""
Matrix visualization: H_res heatmaps, doubly stochastic verification.

Provenance
----------
The H_res matrices are *dynamic* (recomputed each forward pass from the
learned phi_res / b_res / alpha_res), so a heatmap is only meaningful when
extracted from a forward pass on **real data**. The previous implementation
fed ``torch.randn(...)`` (pure noise) into the model, which produced matrices
that reflected the input distribution of random noise rather than the actual
weather/ETT signal the model was trained on — an experimental-validity bug.
This module now runs a forward pass on a real held-out batch from the
checkpoint's dataset, falling back to a clearly-warned synthetic input only
if the real CSV cannot be located (so the function still degrades gracefully
on a machine without the data, instead of crashing).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
from .plot_style import setup_style, save_fig


def _load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
):
    """Rebuild a Model C (mHC-LSTM) from a checkpoint and return (model, config).

    Uses ``ExperimentConfig._from_mapping`` (not the raw constructor) so that
    unknown/legacy keys in the saved config are warned-and-dropped rather than
    raising ``TypeError``. Also redirects ``output_dir`` to a temp dir so the
    config's ``__post_init__`` doesn't create real output dirs as a side effect.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = ckpt.get("config", {})

    from lstm_mhc.models.mhc_lstm import MHCLSTM
    from lstm_mhc.utils.config import ExperimentConfig

    config_dict = dict(config_dict)  # copy so we can mutate
    config_dict.pop("output_path", None)
    config_dict["output_dir"] = "/tmp/_hres_viz"
    exp_config = ExperimentConfig._from_mapping(config_dict)

    model = MHCLSTM(exp_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, exp_config, ckpt


def _real_input_batch(config, ckpt) -> Optional[torch.Tensor]:
    """Load one real held-out batch matching the checkpoint's dataset.

    Returns a ``(B, T, input_dim)`` tensor from the validation split of the
    checkpoint's dataset, or ``None`` if the data CSV cannot be found. The
    caller falls back to a clearly-warned synthetic input in that case so the
    function still produces *a* figure rather than crashing.
    """
    dataset = config.dataset
    try:
        if dataset == "ett":
            from lstm_mhc.data.ett_dataset import build_ett_dataloaders
            _, val_loader, _, _ = build_ett_dataloaders(
                config, csv_name=config.ett_file
            )
        else:
            from lstm_mhc.data.weather_dataset import build_weather_dataloaders
            candidates = [
                Path("data/noaa-weather-data-jfk-airport/jfk_weather_cleaned.csv"),
                Path("data/jfk_weather_cleaned.csv"),
                Path("/kaggle/input/noaa-weather-data-jfk-airport/jfk_weather_cleaned.csv"),
            ]
            csv_path = next((p for p in candidates if p.exists()), None)
            if csv_path is None:
                return None
            _, val_loader, _, _ = build_weather_dataloaders(str(csv_path), config)
        x, _ = next(iter(val_loader))
        return x
    except Exception:
        return None


def _h_res_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[List[np.ndarray], int, str]:
    """Load the model and return per-layer H_res averaged over a real B*T batch.

    Returns ``(h_res_mean_list, num_layers, input_source)`` where
    ``input_source`` is ``"real"`` or ``"synthetic (WARNING: no data CSV)"``
    so the caller can annotate the figure honestly.
    """
    model, exp_config, ckpt = _load_model_from_checkpoint(checkpoint_path, device)
    num_layers = exp_config.num_layers

    x = _real_input_batch(exp_config, ckpt)
    if x is not None:
        input_source = "real"
        with torch.no_grad():
            _, h_res_matrices = model(x.to(device))
    else:
        input_source = "synthetic (WARNING: no data CSV found)"
        print(
            f"  WARNING: could not locate real data CSV for dataset="
            f"{exp_config.dataset}; falling back to synthetic input. "
            f"The resulting H_res reflects random-noise inputs, NOT the "
            f"trained distribution. Re-run on a machine with the data CSV "
            f"for a publication-valid figure."
        )
        with torch.no_grad():
            _, h_res_matrices = model(torch.randn(2, 100, exp_config.input_dim))

    mean_list = [h.mean(dim=(0, 1)).detach().cpu().numpy() for h in h_res_matrices]
    return mean_list, num_layers, input_source


def plot_h_res_heatmaps(
    checkpoint_path: str,
    output_dir: str = "outputs/plots",
    device: str = "cpu",
):
    """
    2D color heatmaps of H_res matrices from trained Model C.
    Shows all layers in a grid with row/column sum annotations.

    The matrices are extracted from a forward pass on a **real held-out batch**
    from the checkpoint's dataset, falling back to a clearly-warned synthetic
    input only if the data CSV is unavailable.
    
    Args:
        checkpoint_path: Path to trained Model C checkpoint.
        output_dir: Where to save plots.
        device: Device to load checkpoint on.
    """
    setup_style()
    mean_mats, num_layers, input_source = _h_res_from_checkpoint(checkpoint_path, device)

    n = mean_mats[0].shape[0] if mean_mats else 4

    n_cols = min(4, num_layers)
    n_rows = (num_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if num_layers == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    stream_labels = [f"Stream {i+1}" for i in range(n)]

    for idx in range(num_layers):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        if idx < len(mean_mats):
            h_res = mean_mats[idx]
        else:
            h_res = np.eye(n)

        sns.heatmap(h_res, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                    xticklabels=stream_labels, yticklabels=stream_labels,
                    vmin=0, vmax=1, linewidths=0.5)

        row_sums = h_res.sum(axis=1)
        col_sums = h_res.sum(axis=0)
        ax.set_title(f"Layer {idx+1} — $\\mathcal{{H}}^{{res}}$\n"
                     f"Row sums: [{', '.join(f'{s:.3f}' for s in row_sums)}]\n"
                     f"Col sums: [{', '.join(f'{s:.3f}' for s in col_sums)}]",
                     fontsize=10)

    # Hide unused subplots
    for idx in range(num_layers, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Feature Routing Heatmaps — Doubly Stochastic $\\mathcal{{H}}^{{res}}$\n"
        f"(input: {input_source})",
        fontsize=14, y=1.03,
    )
    plt.tight_layout()
    save_fig(fig, output_dir, "h_res_heatmaps_all_layers")


def plot_doubly_stochastic_verification(
    checkpoint_path: str,
    output_dir: str = "outputs/plots",
    device: str = "cpu",
):
    """
    Bar chart of row sums and column sums for each layer's H_res.
    Verifies all sums are approximately 1.0.

    Like :func:`plot_h_res_heatmaps`, the H_res matrices are extracted from a
    forward pass on a real held-out batch (with a warned synthetic fallback).
    """
    setup_style()
    mean_mats, num_layers, input_source = _h_res_from_checkpoint(checkpoint_path, device)

    fig, (ax_rows, ax_cols) = plt.subplots(1, 2, figsize=(14, 5))

    all_row_devs, all_col_devs = [], []
    for idx in range(num_layers):
        if idx < len(mean_mats):
            h_res = mean_mats[idx]
            row_sums = h_res.sum(axis=1)
            col_sums = h_res.sum(axis=0)
            all_row_devs.append(np.abs(row_sums - 1.0).max())
            all_col_devs.append(np.abs(col_sums - 1.0).max())
        else:
            all_row_devs.append(0)
            all_col_devs.append(0)

    layers = [f"Layer {i+1}" for i in range(num_layers)]
    x = np.arange(num_layers)

    ax_rows.bar(x, all_row_devs, color="#2ca02c", alpha=0.85)
    ax_rows.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Tolerance (0.05)")
    ax_rows.set_xticks(x)
    ax_rows.set_xticklabels(layers)
    ax_rows.set_ylabel("Max |Row Sum - 1|")
    ax_rows.set_title("Row Sum Verification")
    ax_rows.legend()

    ax_cols.bar(x, all_col_devs, color="#1f77b4", alpha=0.85)
    ax_cols.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Tolerance (0.05)")
    ax_cols.set_xticks(x)
    ax_cols.set_xticklabels(layers)
    ax_cols.set_ylabel("Max |Col Sum - 1|")
    ax_cols.set_title("Column Sum Verification")
    ax_cols.legend()

    fig.suptitle("Doubly Stochastic Verification — All H_res Matrices", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, output_dir, "doubly_stochastic_verification")
