"""Main training loop with full metric hooks and reproducible checkpointing.

Handles (per AGENTS.md §2, §4, §7):
    - Forward/backward with raw gradient-norm logging (pre-clip).
    - Composite signal-gain via :func:`composite_gain` (spectral norm, fwd/bwd).
    - Epoch-level validation with **de-normalized** MSE/MAE/RMSE/MAPE/MASE.
    - Full checkpoint (model, optimizer, scheduler, scaler, config, kaggle_user,
      per-epoch + best + emergency on NaN/crash).
    - NaN/Inf loss guard: on first non-finite loss the loop saves an emergency
      checkpoint and aborts rather than producing corrupt outputs.
    - Resume from the latest per-epoch checkpoint.
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..evaluation.stability import composite_gain
from ..evaluation.metrics import (
    ForecastMetrics,
    compute_forecast_metrics,
    denormalize_np,
    is_finite_metric,
)
from ..utils.config import ExperimentConfig
from .metrics_logger import MetricsLogger


# ------------------------------------------------------------------
# Optimizer parameter-group split
# ------------------------------------------------------------------
# Standard practice: biases, norm scales, and scalar gating parameters
# (alpha_pre/post/res) should NOT be weight-decayed.  Without this split,
# AdamW's decoupled weight decay pulls the mHC dynamic-routing scalars
# toward 0, suppressing the input-dependent heads.
_NO_DECAY_PATTERNS = (
    "alpha_",
    "b_pre",
    "b_post",
    "b_res",
    "norm.weight",
    "rmsnorm.weight",
    ".bias",
)


def split_param_groups(
    model: nn.Module,
    weight_decay: float,
    no_decay_wd: float = 0.0,
) -> List[Dict[str, Any]]:
    """Split model parameters into (decay, no_decay) AdamW parameter groups."""
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(p in name for p in _NO_DECAY_PATTERNS):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    if not decay_params:
        decay_params = no_decay_params
    if not no_decay_params:
        no_decay_params = decay_params
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": no_decay_wd},
    ]


# ------------------------------------------------------------------
# Checkpoint helpers (AGENTS.md §4: the single source of truth for what to save)
# ------------------------------------------------------------------
def _checkpoint_state(
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ExperimentConfig,
    scaler_info: Dict[str, Any],
    best_val_mse: float,
    val_mse: float,
    val_mae: float,
    horizon_maes: List[float],
    grad_norm_history: List[float],
    fwd_amax_history: List[float],
    bwd_amax_history: List[float],
) -> Dict[str, Any]:
    """Assemble the checkpoint dict mandated by AGENTS.md §4."""
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_mse": val_mse,
        "val_mae": val_mae,
        "best_val_mse": best_val_mse,
        "horizon_maes": horizon_maes,
        "grad_norm_history": list(grad_norm_history),  # deque → list for pickle
        "amax_fwd_history": list(fwd_amax_history),
        "amax_bwd_history": list(bwd_amax_history),
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "seed": config.seed,
        "kaggle_user": config.kaggle_user,
        "scaler_mean": scaler_info.get("mean"),
        "scaler_std": scaler_info.get("std"),
        "scaler_columns": scaler_info.get("columns"),
    }


def save_checkpoint(path: Path, state: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ------------------------------------------------------------------
# Evaluation (de-normalized, NaN-safe)
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_std: Optional[np.ndarray] = None,
    horizons: Optional[List[int]] = None,
    train_targets_for_mase: Optional[np.ndarray] = None,
) -> ForecastMetrics:
    """Evaluate on a validation or test set with de-normalized metrics.

    Args:
        model: any model implementing ``forward(x) -> (pred, _)``.
        val_loader: DataLoader yielding ``(x, y)`` normalized batches.
        device: CUDA or CPU.
        scaler_mean/scaler_std: fit-on-train mean/std arrays. If both are
            provided, metrics are computed in original units; otherwise in
            normalized units (backward-compat, not recommended).
        horizons: horizon lengths for per-horizon MAE.
        train_targets_for_mase: raw (de-normalized) training targets for MASE.
    """
    model.eval()
    all_preds, all_targets = [], []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        pred, _ = model(x)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    if not all_preds:
        n_h = len(horizons) if horizons else 5
        return ForecastMetrics(float("inf"), float("inf"), float("inf"),
                               float("inf"), float("inf"), float("inf"),
                               [float("inf")] * n_h)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    if horizons is None:
        horizons = list(range(preds.shape[1]))

    # De-normalize if scaler available (the correct default).
    if scaler_mean is not None and scaler_std is not None:
        preds = denormalize_np(preds, scaler_mean, scaler_std)
        targets = denormalize_np(targets, scaler_mean, scaler_std)
        train_targets_for_mase = denormalize_np(train_targets_for_mase, scaler_mean, scaler_std) \
            if train_targets_for_mase is not None else None

    metrics = compute_forecast_metrics(preds, targets, horizons, train_targets_for_mase)
    model.train()
    return metrics


# ------------------------------------------------------------------
# Single epoch
# ------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    logger: MetricsLogger,
    device: torch.device,
    global_step: int,
    epoch: int,
    config: ExperimentConfig,
    grad_norm_history: List[float],
    fwd_amax_history: List[float],
    bwd_amax_history: List[float],
) -> int:
    """Train one epoch. Returns the updated global step.

    Returns ``-1`` on NaN/Inf loss (caller must handle the emergency stop).
    """
    model.train()
    logger.begin_epoch()  # clear per-epoch accumulators (claude_checked.md C12)
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred, h_res_matrices = model(x)
        loss = F.mse_loss(pred, y)

        # NaN/Inf guard (AGENTS.md §2): abort this epoch on first non-finite.
        if not torch.isfinite(loss):
            print(f"  NaN/Inf loss at step {global_step} — aborting epoch {epoch}.")
            return -1

        loss.backward()

        # Raw (pre-clip) gradient norm — shows Model B explosion vs Model C stability.
        # Compatible with PyTorch >= 2.0 (get_total_norm was added in a later version).
        # error_if_nonfinite=True: a non-finite grad norm indicates a divergence
        # in progress; we catch the raise and convert it to the emergency
        # checkpoint path so the run aborts gracefully rather than crashing
        # with an unhandled exception (claude_checked.md H1).
        try:
            raw_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf"), error_if_nonfinite=True
            )
            grad_norm_history.append(raw_norm.item())
        except RuntimeError as grad_exc:
            # Non-finite gradient at the param level (rare but possible after
            # an exploding activation that the loss check at line 186 missed).
            # Treat exactly like a NaN training loss: abort, save emergency.
            print(f"  Non-finite gradient at step {global_step}: {grad_exc}")
            grad_norm_history.append(float("inf"))
            return -1

        # Log composite gain every N steps.
        fwd_amax = bwd_amax = None
        spec = None
        if h_res_matrices is not None and global_step % config.log_amax_every == 0:
            gain = composite_gain(h_res_matrices, reduce="max")
            fwd_amax, bwd_amax = gain["fwd_amax"], gain["bwd_amax"]
            spec = gain["spectral_norm"]
            fwd_amax_history.append(fwd_amax)
            bwd_amax_history.append(bwd_amax)

        logger.log_step(
            step=global_step,
            train_loss=loss.item(),
            grad_norm=raw_norm.item(),
            fwd_amax=fwd_amax,
            bwd_amax=bwd_amax,
            spectral_norm=spec,
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        global_step += 1

    return global_step


# ------------------------------------------------------------------
# Full training run
# ------------------------------------------------------------------
def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    logger: MetricsLogger,
    device: torch.device,
    scaler_info: Optional[Dict[str, Any]] = None,
    test_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """Complete training run: warmup+cosine LR, early stopping, checkpoints.

    Args:
        model: PyTorch model (A, B, or C).
        train_loader/val_loader: data.
        config: :class:`ExperimentConfig`.
        logger: :class:`MetricsLogger`.
        device: CUDA or CPU.
        scaler_info: dict with ``mean``/``std`` arrays (from the data loader).
            If provided, validation/test metrics are reported in original units.
        test_loader: optional test set for final evaluation.
    """
    model = model.to(device)
    scaler_mean = scaler_info["mean"] if scaler_info else None
    scaler_std = scaler_info["std"] if scaler_info else None

    # Collect training targets for MASE computation (de-normalized).
    train_targets_for_mase: Optional[np.ndarray] = None
    if scaler_mean is not None:
        _targets = []
        for _, _y in train_loader:
            _targets.append(_y.numpy())
        _raw = np.concatenate(_targets, axis=0)           # (N, H, F) normalized
        train_targets_for_mase = denormalize_np(
            _raw.reshape(-1, _raw.shape[-1]), scaler_mean, scaler_std,
        )

    # Optimizer: AdamW matching mHC paper settings (or config overrides).
    # Parameters are split into a decay group (LSTM/Linear weights) and a
    # no-decay group (biases, RMSNorm scales, alpha scalars, b_* biases) so
    # that AdamW's decoupled weight decay does not pull mHC's dynamic-routing
    # scalars (alpha_pre/post/res) toward 0 and suppress the heads.
    optimizer = torch.optim.AdamW(
        split_param_groups(model, config.weight_decay, 0.0),
        lr=config.learning_rate,
        betas=config.adamw_betas,
        eps=config.adamw_eps,
    )

    # LR scheduler: linear warmup + cosine decay to ``min_lr_ratio``.
    warmup_epochs = config.warmup_epochs
    total_epochs = config.num_epochs
    min_ratio = config.min_lr_ratio

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Paths.
    ckpt_dir = config.output_path / "checkpoints"
    best_ckpt_path = ckpt_dir / f"{config.run_name}_best.pt"
    last_ckpt_path = ckpt_dir / f"{config.run_name}_last.pt"

    # State trackers.
    best_val_mse = float("inf")
    best_epoch = -1
    patience_counter = 0
    global_step = 0
    # Bounded in-memory histories (deque auto-evicts the oldest entry on
    # append once the cap is hit). The checkpoint cap (5000) is matched here
    # so memory growth is bounded regardless of epoch count (claude_checked.md H15).
    grad_norm_history: deque[float] = deque(maxlen=5000)
    fwd_amax_history: deque[float] = deque(maxlen=5000)
    bwd_amax_history: deque[float] = deque(maxlen=5000)

    # Dump the config to disk (reproducibility: every result is traceable).
    config.dump()

    # --- Resume (AGENTS.md §4) ---
    start_epoch = 0
    if config.resume and last_ckpt_path.exists():
        ckpt = load_checkpoint(last_ckpt_path, model, optimizer, scheduler)
        # Validate config matches the checkpoint to prevent silent corruption.
        saved_hash = ckpt.get("config_hash", "")
        if saved_hash and saved_hash != config.config_hash:
            raise ValueError(
                f"Config mismatch on resume! Checkpoint hash={saved_hash}, "
                f"current hash={config.config_hash}. Aborting to prevent corruption."
            )
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_mse = ckpt.get("best_val_mse", float("inf"))
        best_epoch = ckpt.get("epoch", -1)
        grad_norm_history = deque(ckpt.get("grad_norm_history", []), maxlen=5000)
        fwd_amax_history = deque(ckpt.get("amax_fwd_history", []), maxlen=5000)
        bwd_amax_history = deque(ckpt.get("amax_bwd_history", []), maxlen=5000)
        print(f"Resumed from epoch {start_epoch} (step {global_step})")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Training: {config.run_name}")
    print(f"Model: {config.model_type} | L={config.num_layers} | n={config.n_streams} | "
          f"d={config.hidden_dim}")
    print(f"Device: {device} | Params: {params:,} | Seed: {config.seed}")
    print(f"LR: {config.learning_rate} | WD: {config.weight_decay} | "
          f"Warmup: {warmup_epochs}ep | Dropout: {config.dropout}")
    print(f"Scaler: {'original units' if scaler_mean is not None else 'normalized (WARNING)'}")
    print(f"Config hash: {config.config_hash}")
    print(f"{'='*60}\n")

    total_start = time.time()
    nan_abort = False
    metrics = None          # guard: referenced after loop even if loop body never ran
    epoch = start_epoch - 1 # guard: referenced after loop even if range is empty

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        # --- Train one epoch ---
        global_step = train_one_epoch(
            model, train_loader, optimizer, logger, device,
            global_step, epoch, config,
            grad_norm_history, fwd_amax_history, bwd_amax_history,
        )

        # NaN/Inf emergency stop.
        if global_step == -1:
            nan_abort = True
            emergency_path = ckpt_dir / f"{config.run_name}_emergency.pt"
            state = _checkpoint_state(
                epoch, 0, model, optimizer, scheduler, config, scaler_info or {},
                best_val_mse, float("nan"), float("nan"), [],
                grad_norm_history, fwd_amax_history, bwd_amax_history,
            )
            save_checkpoint(emergency_path, state)
            print(f"  EMERGENCY checkpoint saved: {emergency_path}")
            break

        # --- Validate ---
        metrics = evaluate(model, val_loader, device, scaler_mean, scaler_std,
                           config.horizons, train_targets_for_mase)
        val_mse = metrics.mse
        val_mae = metrics.mae

        # NaN in metrics = silent corruption, abort.
        if not is_finite_metric(metrics):
            nan_abort = True
            emergency_path = ckpt_dir / f"{config.run_name}_emergency.pt"
            state = _checkpoint_state(
                epoch, global_step, model, optimizer, scheduler, config, scaler_info or {},
                best_val_mse, val_mse, val_mae, metrics.horizon_maes,
                grad_norm_history, fwd_amax_history, bwd_amax_history,
            )
            save_checkpoint(emergency_path, state)
            print(f"  NaN/Inf metrics at epoch {epoch+1} — EMERGENCY stop.")
            break

        # --- Epoch-level metadata ---
        epoch_time = time.time() - epoch_start
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        current_lr = optimizer.param_groups[0]["lr"]

        logger.log_epoch(
            epoch=epoch,
            val_mse=val_mse,
            val_mae=val_mae,
            horizon_maes=metrics.horizon_maes,
            epoch_time=epoch_time,
            peak_vram_mb=peak_vram,
            train_loss_avg=float(np.mean(logger.step_losses[-len(train_loader):])) if logger.step_losses else 0.0,
        )

        scheduler.step()

        # --- Progress ---
        hz = " | ".join(f"{h}h:{m:.3f}" for h, m in zip(config.horizons, metrics.horizon_maes))
        print(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"Val MSE:{val_mse:.6f} MAE:{val_mae:.6f} RMSE:{metrics.rmse:.3f} "
            f"MAPE:{metrics.mape:.2f}% | Horizons [{hz}] | "
            f"LR:{current_lr:.2e} | {epoch_time:.1f}s VRAM:{peak_vram:.0f}MB"
        )

        # --- Early stopping + best checkpoint ---
        if val_mse < best_val_mse - config.early_stopping_min_delta:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0
            state = _checkpoint_state(
                epoch, global_step, model, optimizer, scheduler, config,
                scaler_info or {}, best_val_mse, val_mse, val_mae,
                metrics.horizon_maes, grad_norm_history,
                fwd_amax_history, bwd_amax_history,
            )
            save_checkpoint(best_ckpt_path, state)
            print(f"  -> New best! Val MSE={val_mse:.6f}")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{config.early_stopping_patience})")
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best: epoch {best_epoch+1})")
                break

        # --- Per-epoch checkpoint (if save_every_n_epochs > 0) ---
        if config.save_every_n_epochs > 0 and (epoch + 1) % config.save_every_n_epochs == 0:
            path = ckpt_dir / f"{config.run_name}_epoch{epoch+1}.pt"
            state = _checkpoint_state(
                epoch, global_step, model, optimizer, scheduler, config,
                scaler_info or {}, best_val_mse, val_mse, val_mae,
                metrics.horizon_maes, grad_norm_history,
                fwd_amax_history, bwd_amax_history,
            )
            save_checkpoint(path, state)

    # --- Always save last checkpoint ---
    last_state = _checkpoint_state(
        epoch, global_step, model, optimizer, scheduler, config,
        scaler_info or {}, best_val_mse,
        metrics.mse if metrics is not None and not nan_abort else float("nan"),
        metrics.mae if metrics is not None and not nan_abort else float("nan"),
        metrics.horizon_maes if metrics is not None and not nan_abort else [],
        grad_norm_history, fwd_amax_history, bwd_amax_history,
    )
    save_checkpoint(last_ckpt_path, last_state)

    total_time = time.time() - total_start
    logger.close()

    # --- Final test evaluation ---
    test_results: Dict[str, Any] = {}
    if test_loader is not None and not nan_abort:
        print(f"\n{'='*60}")
        print(f"Final test evaluation (best checkpoint from epoch {best_epoch+1})")
        print(f"{'='*60}")
        if best_ckpt_path.exists():
            load_checkpoint(best_ckpt_path, model)
        test_metrics = evaluate(model, test_loader, device, scaler_mean, scaler_std, config.horizons, train_targets_for_mase)
        test_results = {
            "test_mse": test_metrics.mse,
            "test_mae": test_metrics.mae,
            "test_rmse": test_metrics.rmse,
            "test_mape": test_metrics.mape,
            "test_smape": test_metrics.smape,
            "test_mase": test_metrics.mase,
            "test_horizon_maes": test_metrics.horizon_maes,
        }
        hz = " | ".join(f"{h}h:{m:.3f}" for h, m in zip(config.horizons, test_metrics.horizon_maes))
        print(f"Test MSE:{test_metrics.mse:.6f} RMSE:{test_metrics.rmse:.3f} "
              f"MAE:{test_metrics.mae:.3f} MAPE:{test_metrics.mape:.2f}%")
        print(f"Test Horizons: [{hz}]")

    print(f"\nTraining complete. Best Val MSE: {best_val_mse:.6f} (epoch {best_epoch+1})")
    print(f"Total train time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Config hash: {config.config_hash}")

    return {
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "total_train_time": total_time,
        "num_params": params,
        "config_hash": config.config_hash,
        "nan_abort": nan_abort,
        **test_results,
    }
