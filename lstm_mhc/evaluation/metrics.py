"""Forecasting metrics computed in original (de-normalized) units.

Every metric reported in a results table or figure MUST be in the original
physical units (°F, %RH, inHg, mph for NOAA weather) so that:
  - Numbers from different datasets are comparable.
  - The figure axes are human-interpretable.
  - Metrics match prior work (Informer, PatchTST, etc.) exactly.

Functions accept raw arrays and the (mean, std) scaler so predictions/targets
can be de-normalized *before* the metric computation (the only correct way).

References
----------
- Hyndman & Koehler "Another look at measures of forecast accuracy" (2006)
- MAPE, sMAPE: Makridakis et al. M4 competition (2018)
- MASE: scale-free; compares to naive seasonal forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# (De)normalization helpers (the scaler lives in memory; callers pass it).
# ------------------------------------------------------------------
def denormalize(tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Denormalize a (B, ..., F) tensor using z-score mean/std arrays."""
    device = tensor.device
    m = torch.tensor(mean, dtype=tensor.dtype, device=device)
    s = torch.tensor(std, dtype=tensor.dtype, device=device)
    return tensor * s + m


def denormalize_np(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Denormalize a numpy array."""
    return arr * std + mean


# ------------------------------------------------------------------
# Metric containers
# ------------------------------------------------------------------
@dataclass
class ForecastMetrics:
    """All standard forecasting metrics in original units (scalar unless noted)."""
    mse: float          # Mean Squared Error
    rmse: float         # Root Mean Squared Error
    mae: float          # Mean Absolute Error
    mape: float         # Mean Absolute Percentage Error (in %)
    smape: float        # Symmetric MAPE (in %, bounded [0,200])
    mase: float         # Mean Absolute Scaled Error
    horizon_maes: List[float]  # per-horizon MAE (original units)

    def as_dict(self) -> Dict[str, float]:
        d = {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "smape": self.smape,
            "mase": self.mase,
        }
        for i, v in enumerate(self.horizon_maes):
            d[f"horizon_{i+1}_mae"] = v
        return d


# ------------------------------------------------------------------
# Core metric computation
# ------------------------------------------------------------------
def _safe_mape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%), ignoring near-zero targets."""
    denom = np.maximum(np.abs(target), eps)
    return float(np.mean(np.abs(pred - target) / denom) * 100)


def _safe_smape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE (%) — bounded in [0, 200]."""
    denom = np.abs(pred) + np.abs(target) + eps
    return float(np.mean(2 * np.abs(pred - target) / denom) * 100)


def _safe_mase(
    pred: np.ndarray,
    target: np.ndarray,
    train_targets: Optional[np.ndarray] = None,
    season: int = 24,
) -> float:
    """Mean Absolute Scaled Error.

    Scaled by the in-sample MAE of the naive seasonal (lag-``season``) forecast
    on the training data. If ``train_targets`` is None returns NaN.

    ``mase = MAE(model) / MAE(naive seasonal)``.
    """
    if train_targets is None or len(train_targets) <= season:
        return float("nan")
    mae_naive = np.mean(np.abs(train_targets[season:] - train_targets[:-season]))
    if mae_naive < 1e-12:
        return float("nan")
    return float(np.mean(np.abs(pred - target)) / mae_naive)


def compute_forecast_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    horizons: List[int],
    train_targets: Optional[np.ndarray] = None,
    season: int = 24,
) -> ForecastMetrics:
    """Compute all standard forecasting metrics on arrays in *original units*.

    Args:
        preds: ``(N, H, F)`` predictions in original units.
        targets: ``(N, H, F)`` ground truth in original units.
        horizons: list of horizon lengths (for per-horizon MAE).
        train_targets: optional ``(N_train,)`` or ``(N_train, F)`` training
            targets in original units (for MASE scaling).
        season: naive seasonal lag for MASE (24 = daily for hourly data).

    Returns:
        :class:`ForecastMetrics` with all values in original units.
    """
    # Overall (all horizons pooled).
    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - targets)))
    mape = _safe_mape(preds, targets)
    smape = _safe_smape(preds, targets)
    mase = _safe_mase(preds.reshape(-1, preds.shape[-1]),
                      targets.reshape(-1, targets.shape[-1]),
                      train_targets, season)

    # Per-horizon MAE (original units; the interpretability metric).
    horizon_maes = []
    for h in range(preds.shape[1]):
        horizon_maes.append(float(np.mean(np.abs(preds[:, h, :] - targets[:, h, :]))))

    return ForecastMetrics(
        mse=mse, rmse=rmse, mae=mae, mape=mape, smape=smape,
        mase=mase, horizon_maes=horizon_maes,
    )


# ------------------------------------------------------------------
# NaN/Inf guard (trainer calls before checkpointing)
# ------------------------------------------------------------------
def is_finite_metric(metrics: ForecastMetrics) -> bool:
    """Return False if any metric is NaN or Inf (catches silent corruption)."""
    vals = [metrics.mse, metrics.rmse, metrics.mae,
            metrics.mape, metrics.smape, metrics.mase]
    return all(np.isfinite(v) for v in vals) and all(np.isfinite(h) for h in metrics.horizon_maes)
