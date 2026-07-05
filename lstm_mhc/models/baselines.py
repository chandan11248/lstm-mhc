"""Forecasting baselines for LSTM-µHC paper.

All models implement: ``forward(x) -> (predictions, None)`` where predictions
shape is ``(B, num_horizons, input_dim)``.

Baselines:
    - PersistenceForecaster: last-value (no training)
    - SeasonalNaiveForecaster: 24h lag for hourly data (no training)
    - RidgeForecaster: linear regression with L2 regularization
    - DLinearForecaster: trend-seasonal decomposition linear model (Zeng 2023)
    - NLinearForecaster: channel-independent normalised linear model (Zeng 2023)
    - GRUBaseline: single-stream GRU baseline (NOT parameter-matched)
    - TCNBaseline: temporal convolutional network
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# Untrained baselines (sanity checks)
# =====================================================================

class PersistenceForecaster(nn.Module):
    """Last-value baseline. No trainable parameters."""

    def __init__(self, config):
        super().__init__()
        self.num_horizons = config.num_horizons
        self.input_dim = config.input_dim

    def forward(self, x: torch.Tensor):
        last = x[:, -1, :].unsqueeze(1).expand(-1, self.num_horizons, -1)
        return last, None


class SeasonalNaiveForecaster(nn.Module):
    """Seasonal naive: predicts value at same hour yesterday (lag=24). No training."""

    def __init__(self, config, season_lag: int = 24):
        super().__init__()
        self.horizons = config.horizons
        self.input_dim = config.input_dim
        self.season_lag = season_lag

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        preds = []
        for h in self.horizons:
            idx = max(0, T - h - (self.season_lag * ((h - 1) // self.season_lag + 1)))
            preds.append(x[:, min(idx, T - 1), :])
        return torch.stack(preds, dim=1), None


# =====================================================================
# Linear baselines
# =====================================================================

class RidgeForecaster(nn.Module):
    """Linear autoregressive baseline with L2 regularization (lookback → horizons)."""

    def __init__(self, config, lookback: int = 168):
        super().__init__()
        self.input_dim = config.input_dim
        self.num_horizons = config.num_horizons
        self.lookback = lookback
        self.linear = nn.Linear(lookback * config.input_dim,
                                config.num_horizons * config.input_dim)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        lb = min(self.lookback, T)
        x_flat = x[:, -lb:, :].reshape(B, -1)
        if lb < self.lookback:
            pad = torch.zeros(B, (self.lookback - lb) * self.input_dim, device=x.device)
            x_flat = torch.cat([pad, x_flat], dim=1)
        pred = self.linear(x_flat).view(B, self.num_horizons, self.input_dim)
        return pred, None


class DLinearForecaster(nn.Module):
    """DLinear: Decomposition + Linear baseline (Zeng et al., AAAI 2023).

    Decomposes input into trend (moving average) + seasonal (residual),
    then applies per-channel linear layers.

    Trend extraction uses an asymmetric ``padding=(kernel_size - 1, 0)`` so
    the AvgPool output length is exactly ``T`` (no trim needed). The
    previous symmetric padding produced ``T + (kernel_size - 1)`` entries
    and then right-trimmed, which silently biased the trend toward the
    right side of the window (claude_checked.md H17).
    """

    def __init__(self, config, kernel_size: int = 25):
        super().__init__()
        self.input_dim = config.input_dim
        self.num_horizons = config.num_horizons
        self.seq_len = config.input_len
        self.trend_linear = nn.Linear(self.seq_len, config.num_horizons)
        self.seasonal_linear = nn.Linear(self.seq_len, config.num_horizons)
        # Asymmetric left-only padding so the trend output length stays at T.
        # nn.AvgPool1d does NOT support tuple padding (RuntimeError), so we use
        # nn.ConstantPad1d for left-only zero-padding followed by stride-1 pool.
        # With kernel=25, left-pad=24 makes the padded length T+24, and the
        # pool output is (T+24) - 25 + 1 = T.
        self.left_pad = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_t = x.permute(0, 2, 1)                               # (B, C, T)
        trend = self.avg_pool(self.left_pad(x_t))               # (B, C, T)
        seasonal = x_t - trend
        t_pred = self.trend_linear(trend.reshape(B * C, T)).view(B, C, self.num_horizons)
        s_pred = self.seasonal_linear(seasonal.reshape(B * C, T)).view(B, C, self.num_horizons)
        return (t_pred + s_pred).permute(0, 2, 1), None         # (B, H, C)


class NLinearForecaster(nn.Module):
    """NLinear: channel-independent normalised linear baseline (Zeng 2023).

    Subtracts the last value per channel, applies a learned linear map,
    then adds the last value back. Per-channel so no cross-variable mixing.
    Strong 2024–2026 baseline for multivariate forecasting.
    """

    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.num_horizons = config.num_horizons
        self.seq_len = config.input_len
        self.linear = nn.Linear(self.seq_len, config.num_horizons)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_t = x.permute(0, 2, 1)                                # (B, C, T)
        last = x_t[:, :, -1:].clone()                            # (B, C, 1)
        normed = x_t - last                                      # per-channel subtract
        pred = self.linear(normed.reshape(B * C, T)).view(B, C, self.num_horizons)
        return (pred + last).permute(0, 2, 1), None              # (B, H, C)


# =====================================================================
# Neural baselines (parameter-matched)
# =====================================================================

class GRUBaseline(nn.Module):
    """GRU baseline (single-stream, NOT parameter-matched to multi-stream Model C).

    Note: This is an intentionally *small* baseline. The previous version set
    ``hidden_dim = n_streams * hidden_dim`` which made GRU ~6× larger than
    Model C's recurrent block per layer — the "parameter-matched" docstring
    was a lie. We now use the same per-stream ``config.hidden_dim`` as Model C,
    so the comparison is "Model C (multi-stream µHC) vs GRU (single-stream,
    same per-stream width)".
    """

    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.num_horizons = config.num_horizons
        self.hidden_dim = config.hidden_dim
        self.gru = nn.GRU(
            input_size=self.input_dim, hidden_size=self.hidden_dim,
            num_layers=config.num_layers, batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(self.hidden_dim, config.num_horizons * config.input_dim)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        out, _ = self.gru(x)
        pred = self.output_head(out[:, -1, :]).view(B, self.num_horizons, self.input_dim)
        return pred, None


class TCNBaseline(nn.Module):
    """Temporal Convolutional Network baseline (causal dilated convs)."""

    def __init__(self, config, num_channels: Optional[int] = None):
        super().__init__()
        self.input_dim = config.input_dim
        self.num_horizons = config.num_horizons
        channels = num_channels or max(16, config.n_streams * config.hidden_dim // 4)
        layers = []
        for i in range(config.num_layers):
            in_ch = self.input_dim if i == 0 else channels
            dilation = 2 ** i
            layers.append(nn.Conv1d(in_ch, channels, kernel_size=3,
                                     dilation=dilation, padding=dilation * (3 - 1) // 2))
            layers.append(nn.GELU())
            layers.append(nn.GroupNorm(min(8, channels), channels))
            layers.append(nn.Dropout(config.dropout))
        self.conv_layers = nn.Sequential(*layers)
        self.output_head = nn.Linear(channels, config.num_horizons * config.input_dim)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        out = self.conv_layers(x.permute(0, 2, 1))              # (B, channels, T)
        pred = self.output_head(out[:, :, -1]).view(B, self.num_horizons, self.input_dim)
        return pred, None


# =====================================================================
# Registry
# =====================================================================

BASELINE_MAP = {
    "persistence": PersistenceForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "ridge": RidgeForecaster,
    "dlinear": DLinearForecaster,
    "nlinear": NLinearForecaster,
    "gru": GRUBaseline,
    "tcn": TCNBaseline,
}


def build_baseline(name: str, config, **kwargs) -> nn.Module:
    """Instantiate a baseline model by name."""
    if name not in BASELINE_MAP:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_MAP.keys())}")
    return BASELINE_MAP[name](config, **kwargs)
