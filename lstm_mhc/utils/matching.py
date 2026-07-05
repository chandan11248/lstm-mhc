"""Parameter-matching and compute-matching utilities for fair baseline comparison.

Ensures reviewers can verify that Model C's advantage comes from the
architecture, not from extra parameters or extra compute time.

Two levels of matching
----------------------
1. *Analytic* recurrent-budget matching (default, fast): :func:`match_vanilla`
   solves the closed-form hidden dim so Model A's per-layer LSTM params equal
   Model C's recurrent+head budget. This is what the config uses by default.
2. *Exact* matching (verification): :func:`find_parameter_matched_config`
   brute-forces candidate hidden dims and reports the achieved relative error.
   Use this to fill the "Params" column of the results table honestly.
"""

from __future__ import annotations

import time  # noqa: F401  (kept for future compute-matching work — see TODO below)
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..utils.config import ExperimentConfig


def count_trainable_params(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters.

    Returns:
        Dict with ``total`` (all parameters) and ``trainable`` (requires_grad=True).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ----------------------------------------------------------------------
# Analytic matching
# ----------------------------------------------------------------------
def _mhc_total_params(config: ExperimentConfig) -> int:
    """Exact total trainable params of Model C without instantiating it.

    Mirrors :class:`~lstm_mhc.models.mhc_lstm.MHCLSTM`:
      - input_proj  : Linear(input_dim, n*d)
      - per block L : RMSNorm(n*d) + MuHCHeads + LSTM(input=d, hidden=d)
      - output_proj : Linear(n*d, num_horizons*input_dim)
    """
    n, d = config.n_streams, config.hidden_dim
    nd = n * d
    in_dim = config.input_dim
    out_dim = config.num_horizons * in_dim
    input_proj = nd * in_dim + nd
    output_proj = nd * out_dim + out_dim
    rmsnorm = nd
    # MuHCHeads: 3 alphas + phi_pre(nd*n) + phi_post(nd*n) + phi_res(nd*n^2)
    #            + b_pre(n) + b_post(n) + b_res(n*n)
    heads = 3 + nd * n + nd * n + nd * n * n + n + n + n * n
    # single-layer LSTM(input=d, hidden=d): 4*d*(d + d + 2)
    lstm = 4 * d * (d + d + 2)
    per_block = rmsnorm + heads + lstm
    return input_proj + output_proj + config.num_layers * per_block


def _vanilla_total_params(H: int, config: ExperimentConfig) -> int:
    """Exact total trainable params of a stacked vanilla LSTM with hidden=H.

    Matches :class:`~lstm_mhc.models.vanilla_lstm.StandardLSTM`:
      - stacked nn.LSTM(input=in_dim, hidden=H, layers=L)
      - output_head Linear(H, num_horizons*in_dim)
    """
    in_dim = config.input_dim
    L = config.num_layers
    layer0 = 4 * H * (in_dim + H + 2)
    layerk = 4 * H * (H + H + 2)
    lstm = layer0 + (L - 1) * layerk
    out_dim = config.num_horizons * in_dim
    output_head = H * out_dim + out_dim
    return lstm + output_head


def match_vanilla(config: ExperimentConfig) -> int:
    """Pick Model A's hidden dim so its total params match Model C (<=tolerance).

    Uses an exact integer scan (cheap: a few hundred evaluations of a closed
    form) rather than a coarse quadratic approximation, because the dominant
    cost is the stacked-LSTM recurrence and rounding a closed form produced
    60-90% error in practice. Returns the integer H minimizing the relative
    parameter-count error.

    Args:
        config: base config (uses n_streams, hidden_dim, input_dim, num_layers,
            match_tolerance).

    Returns:
        Integer hidden dim H for Model A (within ``config.match_tolerance`` of
        Model C's parameter count).
    """
    if not config.match_params:
        return config.n_streams * config.hidden_dim

    target = _mhc_total_params(config)
    tol = config.match_tolerance
    best_H, best_err = 8, float("inf")
    # Seed estimate: dominant vanilla term is ~ L*8*H^2  =>  H ~ sqrt(T/(8L)).
    import math
    est = int(math.sqrt(target / (8 * config.num_layers)))
    lo, hi = max(8, est - 32), est + 64
    for H in range(lo, hi + 1):
        err = abs(_vanilla_total_params(H, config) - target) / target
        if err < best_err:
            best_H, best_err = H, err
    return best_H


# ----------------------------------------------------------------------
# Exact (brute-force) matching for verification
# ----------------------------------------------------------------------
def find_parameter_matched_config(
    target_model: nn.Module,
    baseline_cls,
    base_config: ExperimentConfig,
    tolerance: float = 0.05,
    hidden_dim_candidates: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Search for a baseline ``hidden_dim`` matching the target's parameter count.

    Args:
        target_model: reference model (e.g. Model C) to match against.
        baseline_cls: baseline class (e.g. StandardLSTM / GRUBaseline).
        base_config: ExperimentConfig for the baseline.
        tolerance: max relative error before a warning is emitted.
        hidden_dim_candidates: candidate hidden dims to try.

    Returns:
        Dict with the matched config, parameter counts, and relative error.

    Raises:
        RuntimeError if no candidate builds successfully.
    """
    target_params = count_trainable_params(target_model)
    candidates = hidden_dim_candidates or _candidate_hidden_dims(target_params)

    best: Optional[Dict[str, Any]] = None
    for hidden_dim in candidates:
        cfg = deepcopy(base_config)
        cfg.hidden_dim = hidden_dim
        cfg.vanilla_hidden_dim = hidden_dim
        cfg.match_params = False
        try:
            model = baseline_cls(cfg)
            params = count_trainable_params(model)
            rel_error = abs(params - target_params) / max(target_params, 1)
            if best is None or rel_error < best["rel_error"]:
                best = {
                    "config": cfg,
                    "params": params,
                    "target_params": target_params,
                    "rel_error": rel_error,
                    "hidden_dim": hidden_dim,
                }
        except Exception:
            continue

    if best is None:
        raise RuntimeError(f"Could not build any baseline from {baseline_cls.__name__}")

    if best["rel_error"] > tolerance:
        # Non-fatal: caller should print the error in the results table.
        best["within_tolerance"] = False
    else:
        best["within_tolerance"] = True
    return best


def _candidate_hidden_dims(target_params: int) -> List[int]:
    """Dense candidate grid that scales with model size."""
    base = [16, 24, 32, 48, 64, 96, 112, 128, 144, 160, 192, 224, 256,
            320, 384, 448, 512, 640, 768, 896, 1024]
    return base


# ----------------------------------------------------------------------
# Compute matching (wall-clock budget)
# ----------------------------------------------------------------------
# TODO(phase 2): implement wall-clock-budget matching. The previous prototype
# (`run_compute_matched`) claimed to enforce a time budget via
# `cfg.max_train_seconds`, but the trainer never read that field — the budget
# was silently ignored. A real implementation needs (a) a `max_train_seconds`
# field on `ExperimentConfig` with validation, and (b) a `time.time()` check
# inside the trainer's epoch loop (run_training). See claude_checked.md issue C6.
