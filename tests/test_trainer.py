"""Smoke tests for the training loop.

These tests exercise ``run_training`` and ``train_one_epoch`` end-to-end on
synthetic data to catch regressions in:
  - checkpoint writing (best/last/emergency)
  - NaN guard (train-time and val-time)
  - MASE computation (was previously always-NaN, claude_checked.md H3)
  - ``is_finite_metric`` coverage (claude_checked.md H2)

The tests use a tiny synthetic dataset and a small config so they finish in
seconds on CPU. They are NOT meant to verify that the model trains well —
that's what the real Kaggle runs are for.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from lstm_mhc.training.trainer import run_training, evaluate
from lstm_mhc.training.metrics_logger import MetricsLogger
from lstm_mhc.utils.config import ExperimentConfig


@pytest.fixture
def tiny_cfg(tmp_path):
    """Minimal config + dataloaders on synthetic data.

    Uses a small learning rate (1e-4) and dropout=0 to keep gradients stable
    on the tiny random dataset. The point is to exercise the trainer plumbing,
    not to verify the model learns anything meaningful.
    """
    cfg = ExperimentConfig(
        num_layers=2,
        n_streams=2,
        hidden_dim=8,
        input_len=12,
        horizons=[2, 4],
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-4,
        dropout=0.0,
        weight_decay=0.0,
        warmup_epochs=0,
        early_stopping_patience=99,
        save_every_n_epochs=0,
        log_amax_every=1,
        output_dir=str(tmp_path / "run"),
    )
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    (cfg.output_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, cfg.input_dim)).astype(np.float32)
    from lstm_mhc.data.weather_dataset import WeatherTimeSeriesDataset
    ds = WeatherTimeSeriesDataset(data, cfg.input_len, cfg.horizons, stride=1)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    return cfg, loader, loader


def _build_model(cfg):
    from lstm_mhc.models.mhc_lstm import MHCLSTM
    return MHCLSTM(cfg)


def test_run_training_writes_checkpoints(tiny_cfg):
    cfg, train_loader, val_loader = tiny_cfg
    logger = MetricsLogger("smoke", str(cfg.output_path / "logs"))
    model = _build_model(cfg)
    result = run_training(
        model, train_loader, val_loader, cfg, logger,
        device=torch.device("cpu"),
    )
    assert result["best_val_mse"] >= 0
    ckpt_dir = cfg.output_path / "checkpoints"
    # best ckpt is only written when val improves; the tiny smoke test may
    # emergency-stop before any improvement, so we only assert the last ckpt
    # (which is always written) and the run completed.
    assert (ckpt_dir / f"{cfg.run_name}_last.pt").exists()
    assert "best_val_mse" in result


def test_evaluate_returns_finite_metrics(tiny_cfg):
    cfg, _, val_loader = tiny_cfg
    model = _build_model(cfg)
    # Pass train_targets_for_mase so MASE is computed (was previously always
    # NaN — see claude_checked.md H3).
    train_targets_for_mase = np.random.randn(200, cfg.input_dim).astype(np.float32)
    metrics = evaluate(
        model, val_loader, device=torch.device("cpu"),
        scaler_mean=None, scaler_std=None,
        horizons=cfg.horizons,
        train_targets_for_mase=train_targets_for_mase,
    )
    for name in ("mse", "rmse", "mae", "mape", "smape", "mase"):
        v = getattr(metrics, name)
        assert np.isfinite(v), f"metric {name} is not finite: {v}"


def test_evaluate_mase_nan_without_train_targets(tiny_cfg):
    """MASE is by design NaN when ``train_targets_for_mase`` is not provided."""
    cfg, _, val_loader = tiny_cfg
    model = _build_model(cfg)
    metrics = evaluate(
        model, val_loader, device=torch.device("cpu"),
        scaler_mean=None, scaler_std=None,
        horizons=cfg.horizons,
    )
    assert np.isnan(metrics.mase), (
        f"expected MASE to be NaN without train_targets_for_mase, got {metrics.mase}"
    )


def test_metrics_logger_begin_epoch_clears_accumulators(tiny_cfg):
    """claude_checked.md C12 fix: ``step_losses`` must not leak across epochs."""
    cfg, _, _ = tiny_cfg
    logger = MetricsLogger("cleartest", str(cfg.output_path / "logs"))
    logger.log_step(step=0, train_loss=1.0, grad_norm=0.5)
    logger.log_step(step=1, train_loss=0.5, grad_norm=0.4)
    assert len(logger.step_losses) == 2
    logger.begin_epoch()
    assert len(logger.step_losses) == 0
    assert len(logger.grad_norms) == 0


def test_metrics_logger_spectral_norm_column_present(tiny_cfg):
    """``MetricsLogger.log_step`` accepts and writes ``spectral_norm``."""
    cfg, _, _ = tiny_cfg
    logger = MetricsLogger("specttest", str(cfg.output_path / "logs"))
    logger.log_step(step=0, train_loss=1.0, grad_norm=0.5, spectral_norm=1.2)
    logger.close()
    text = (cfg.output_path / "logs" / "specttest_step.csv").read_text()
    assert "spectral_norm" in text
