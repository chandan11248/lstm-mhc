"""Centralized metrics logger for all experiment metrics.

Logs to CSV files for post-hoc analysis and plotting.
Tracks:
    - Step-level: train_loss, grad_norm, fwd_amax, bwd_amax, spectral_norm
    - Epoch-level: val_mse, val_mae, horizon_maes, time, vram
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional


class MetricsLogger:
    """CSV logger for step-level and epoch-level metrics.

    Files created:
        - ``{run_name}_step.csv``: one row per training step.
        - ``{run_name}_epoch.csv``: one row per completed epoch.

    Existing viz code reads the columns ``step, train_loss, grad_norm,
    fwd_amax, bwd_amax`` — the new ``spectral_norm`` column is additive.
    """

    def __init__(self, run_name: str, output_dir: str):
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Step-level log.
        self.step_path = self.output_dir / f"{run_name}_step.csv"
        self.step_file = open(self.step_path, "w", newline="")
        self.step_writer = csv.writer(self.step_file)
        self.step_writer.writerow([
            "step", "train_loss", "grad_norm",
            "fwd_amax", "bwd_amax", "spectral_norm",
        ])

        # Epoch-level log.
        self.epoch_path = self.output_dir / f"{run_name}_epoch.csv"
        self.epoch_file = open(self.epoch_path, "w", newline="")
        self.epoch_writer = csv.writer(self.epoch_file)
        self.epoch_writer.writerow([
            "epoch", "val_mse", "val_mae",
            "horizon_6h_mae", "horizon_12h_mae", "horizon_24h_mae",
            "horizon_48h_mae", "horizon_72h_mae",
            "epoch_time_s", "peak_vram_mb", "train_loss_avg",
        ])

        # In-memory accumulators for quick in-process access.
        self.step_losses: List[float] = []
        self.grad_norms: List[float] = []
        self.epoch_val_mse: List[float] = []

    def log_step(
        self,
        step: int,
        train_loss: float,
        grad_norm: float,
        fwd_amax: Optional[float] = None,
        bwd_amax: Optional[float] = None,
        spectral_norm: Optional[float] = None,
    ):
        """Log metrics for a single training step."""
        self.step_writer.writerow([
            step, train_loss, grad_norm,
            "" if fwd_amax is None else fwd_amax,
            "" if bwd_amax is None else bwd_amax,
            "" if spectral_norm is None else spectral_norm,
        ])
        self.step_losses.append(train_loss)
        self.grad_norms.append(grad_norm)

    def log_epoch(
        self,
        epoch: int,
        val_mse: float,
        val_mae: float,
        horizon_maes: List[float],
        epoch_time: float,
        peak_vram_mb: float,
        train_loss_avg: float = 0.0,
    ):
        """Log metrics for a completed epoch."""
        self.epoch_writer.writerow([
            epoch, val_mse, val_mae,
            *horizon_maes,
            epoch_time, peak_vram_mb, train_loss_avg,
        ])
        self.epoch_val_mse.append(val_mse)
        # Flush every epoch to prevent data loss on Kaggle session kill.
        self.epoch_file.flush()
        self.step_file.flush()

    def close(self):
        """Close all file handles."""
        for f in (self.step_file, self.epoch_file):
            try:
                f.close()
            except Exception:
                pass

    def begin_epoch(self) -> None:
        """Reset per-epoch accumulators.

        Call this at the start of each training epoch. The trainer previously
        sliced ``step_losses[-len(train_loader):]`` to recover per-epoch losses,
        but ``step_losses`` was a class-level list that accumulated forever —
        the slice lied after resume, after ``drop_last=True`` discarded a
        batch, or after any NaN-abort that re-used the list. Clearing the
        list at the start of each epoch makes the per-epoch slice trivially
        correct.
        """
        self.step_losses.clear()
        self.grad_norms.clear()

    def __del__(self):
        self.close()
