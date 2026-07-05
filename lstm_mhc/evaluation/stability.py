"""Stability analysis: composite signal gain of the µHC residual highway.

This module measures the *signal-propagation stability* of the multi-layer
residual mixing ``X_{l+1} = H_res_l . X_l + ...``. The central claim of the
project is that unconstrained HC (Model B) lets the composite gain explode,
while mHC (Model C) keeps it bounded.

Why the previous implementation was wrong (review §1 / C1)
---------------------------------------------------------
The old code computed ``max(abs-row-sum)`` of the **batch-and-time-averaged**
product ``prod_l mean_{B,T}(H_res_l)``. Two problems:

1. Averaging over (B,T) *before* multiplying destroys the per-sample composite
   the paper actually analyzes. The mean of products != product of means.
2. For doubly-stochastic (Model C) matrices the row sum is *analytically* 1.0,
   so the metric was a tautology (always 1.0), and for unconstrained matrices
   the signed row-sum-of-abs is neither a norm bound nor what the paper plots.

Correct metrics (implemented here)
----------------------------------
For each sample ``s`` we form the true composite ``C_s = prod_l H_res_l^{(s)}``
(a product of ``(B,T,n,n)`` tensors, accumulated without averaging). Then:

- ``spectral_norm`` : ``max_s sigma_max(C_s)``  — the induced L2 operator norm;
  bounded by 1 for doubly-stochastic, and the quantity the paper bounds.
- ``fwd_amax`` (induced inf-norm)  : ``max_s max_i sum_j |C_s[i,j]|``
- ``bwd_amax`` (induced 1-norm)    : ``max_s max_j sum_i |C_s[i,j]|``

The two amax quantities are kept for continuity with existing CSV columns and
the original paper figures; ``spectral_norm`` is the recommended headline
metric because it is a true, tight operator-norm bound.

Analytic guarantees (reported separately, never plotted as "results")
---------------------------------------------------------------------
``ds_row_dev`` / ``ds_col_dev`` measure how close each *single-layer* H_res is
to doubly stochastic — these verify the constraint *was applied* and are not a
learned result.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


@functools.lru_cache(maxsize=8)
def _cached_eye(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Cached identity matrix for ``composite_gain``.

    Avoids re-allocating an ``n×n`` identity on every logged step (was
    previously an O(1) but allocation-heavy op). Keyed on ``(n, dtype,
    device)``; the small cache size is plenty for the typical "few depths,
    one device, one dtype" workload.
    """
    return torch.eye(n, dtype=dtype, device=device).contiguous()


# ----------------------------------------------------------------------
# Composite gain (the headline stability metrics)
# ----------------------------------------------------------------------
@torch.no_grad()
def composite_gain(
    h_res_matrices: List[torch.Tensor],
    reduce: str = "max",
) -> Dict[str, float]:
    """Composite signal gain of the multi-layer H_res product.

    Args:
        h_res_matrices: list of per-layer ``(B, T, n, n)`` residual matrices.
            Tensors may live on any device and need not be detached.
        reduce: ``"max"`` (worst-case over all samples) or ``"mean"``.

    Returns:
        Dict with ``spectral_norm``, ``fwd_amax``, ``bwd_amax`` (the measured
        composite gain) and ``n_layers``.
    """
    if not h_res_matrices:
        return {"spectral_norm": 1.0, "fwd_amax": 1.0, "bwd_amax": 1.0, "n_layers": 0}

    # Flatten (B,T) into a single sample axis S so we compose per-sample.
    mats = [m.reshape(-1, *m.shape[-2:]) for m in h_res_matrices]  # each (S, n, n)
    S, n, _ = mats[0].shape
    eye = _cached_eye(n, mats[0].dtype, mats[0].device)
    composite = eye.expand(S, n, n).contiguous()                  # (S, n, n)
    for m in mats:
        composite = m @ composite                                  # per-sample product

    # Induced inf-norm (max abs row sum) and 1-norm (max abs col sum).
    row_abs = composite.abs().sum(dim=-1)                          # (S, n)
    col_abs = composite.abs().sum(dim=-2)                          # (S, n)
    fwd_per = row_abs.max(dim=-1).values                           # (S,)
    bwd_per = col_abs.max(dim=-1).values                           # (S,)

    # Spectral norm via SVD (largest singular value). Robust on tiny n=4.
    spec = torch.linalg.svdvals(composite)[:, 0]                   # (S,)

    if reduce == "mean":
        agg = lambda t: t.float().mean().item()
    else:
        agg = lambda t: t.float().max().item()

    return {
        "spectral_norm": agg(spec),
        "fwd_amax": agg(fwd_per),
        "bwd_amax": agg(bwd_per),
        "n_layers": len(h_res_matrices),
    }


def compute_composite_amax(h_res_matrices: List[torch.Tensor]) -> Dict[str, float]:
    """Backward-compatible alias returning ``fwd_amax`` / ``backward_amax``.

    Kept for scripts that import the old name; new code should call
    :func:`composite_gain` which also returns the spectral norm.
    """
    g = composite_gain(h_res_matrices, reduce="max")
    return {
        "forward_amax": g["fwd_amax"],
        "backward_amax": g["bwd_amax"],
        "spectral_norm": g["spectral_norm"],
    }


# ----------------------------------------------------------------------
# Doubly-stochastic verification (analytic, not a learned result)
# ----------------------------------------------------------------------
@torch.no_grad()
def verify_doubly_stochastic(h_res: torch.Tensor, tol: float = 1e-3) -> Dict[str, object]:
    """Verify single-layer H_res matrices are approximately doubly stochastic.

    Reports worst-case (over the batch/time/layer axis) deviation of row/column
    sums from 1.0, plus non-negativity. This checks that the *constraint* was
    applied — it is an analytic property, not something the model learns.

    Args:
        h_res: (B, T, n, n) residual matrices.
        tol: row/col-sum deviation tolerance for the ``is_doubly_stochastic``
            flag (1e-3 is comfortably within Sinkhorn-20 convergence).

    Returns:
        Dict with row/col sums, max deviations, non-negativity, and the
        (B,T)-mean matrix for plotting.
    """
    h = h_res.detach()
    # Worst-case over the sample axis, then report mean matrix for heatmaps.
    row_sums = h.sum(dim=-1)                     # (B,T,n)
    col_sums = h.sum(dim=-2)                     # (B,T,n)
    max_row_dev = (row_sums - 1.0).abs().max().item()
    max_col_dev = (col_sums - 1.0).abs().max().item()
    min_entry = h.min().item()
    h_mean = h.mean(dim=(0, 1))                  # (n,n)

    return {
        "mean_matrix_row_sums": h_mean.sum(dim=-1).cpu().numpy().tolist(),
        "mean_matrix_col_sums": h_mean.sum(dim=-2).cpu().numpy().tolist(),
        "max_row_deviation": max_row_dev,
        "max_col_deviation": max_col_dev,
        "min_entry": min_entry,
        "is_doubly_stochastic": (
            max(max_row_dev, max_col_dev) < tol and min_entry >= -tol
        ),
        "matrix": h_mean.cpu().numpy(),
    }


# ----------------------------------------------------------------------
# Per-layer diagnostics
# ----------------------------------------------------------------------
@torch.no_grad()
def layer_diagnostics(h_res_matrices: List[torch.Tensor]) -> List[Dict[str, float]]:
    """Per-layer spectral norm, non-negativity, and row/col deviation.

    Useful for the "is the model collapsing H_res to identity?" check
    (effective-rank collapse) and the learned-vs-prior decomposition.
    """
    out = []
    for h in h_res_matrices:
        m = h.detach().reshape(-1, *h.shape[-2:])           # (S,n,n)
        spec = torch.linalg.svdvals(m)[:, 0].float().mean().item()
        row_dev = (m.sum(dim=-1) - 1.0).abs().mean().item()
        col_dev = (m.sum(dim=-2) - 1.0).abs().mean().item()
        eff_rank = (torch.linalg.svdvals(m) > 1e-4).float().sum(dim=-1).mean().item()
        out.append({
            "spectral_norm_mean": spec,
            "row_dev_mean": row_dev,
            "col_dev_mean": col_dev,
            "effective_rank_mean": float(eff_rank),
        })
    return out


# ----------------------------------------------------------------------
# CSV log loaders
# ----------------------------------------------------------------------
def load_step_logs(log_dir: str, run_name: str) -> pd.DataFrame:
    """Load step-level CSV logs for a run."""
    path = Path(log_dir) / f"{run_name}_step.csv"
    return pd.read_csv(path)


def load_epoch_logs(log_dir: str, run_name: str) -> pd.DataFrame:
    """Load epoch-level CSV logs for a run."""
    path = Path(log_dir) / f"{run_name}_epoch.csv"
    return pd.read_csv(path)
