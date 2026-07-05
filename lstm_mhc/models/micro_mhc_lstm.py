"""Micro-level mHC-LSTM — ablation model for the macro-vs-micro comparison.

This is NOT a replacement for Model C (macro mHC). It is a controlled ablation
that answers one question (see micro_mhc_lstm_architecture.md):

    Does applying the µHC highway *inside every recurrent step* (micro) change
    accuracy/stability enough to justify the sequential overhead, versus the
    macro design that applies µHC *between* cuDNN-backed LSTM layers?

The head math is IDENTICAL to Model C — we reuse ``MuHCHeads`` and ``RMSNorm``
from components.py. The ONLY difference is placement: here Sinkhorn-Knopp runs
T times per layer (inside an ``nn.LSTMCell`` loop) instead of once per layer.

Interface matches MHCLSTM exactly::

    pred, h_res_matrices = model(x)
    x:    (B, T, input_dim)
    pred: (B, num_horizons, input_dim)
    h_res_matrices: list of length L, each (B, T, n, n)  (for composite_gain)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import ExperimentConfig
from .components import MuHCHeads, RMSNorm


class MicroMHCLSTMCell(nn.Module):
    """One time-step constrained µHC + LSTMCell update (Micro-Input variant).

    Given multi-stream state ``X_t`` (B, n, d) and recurrent state ``(h, c)``:
      1. RMSNorm + flatten -> compute constrained heads (H_pre, H_post, H_res)
      2. aggregate streams -> single (B, d) cell input
      3. advance nn.LSTMCell
      4. distribute hidden state back to streams via H_post
      5. doubly-stochastic residual mixing via H_res
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.n = config.n_streams
        self.d = config.hidden_dim
        self.nd = config.n_streams * config.hidden_dim

        self.rmsnorm = RMSNorm(self.nd, eps=config.rmsnorm_eps)
        # constrained=True => Sinkhorn-projected H_res (same as Model C).
        self.heads = MuHCHeads.from_config(config, constrained=True)
        self.cell = nn.LSTMCell(self.d, self.d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X_t: torch.Tensor, state):
        # X_t: (B, n, d); state: (h, c) each (B, d)
        B, n, d = X_t.shape
        X_flat = X_t.reshape(B, 1, n * d)          # fake T=1 for MuHCHeads
        X_norm = self.rmsnorm(X_flat)
        H_pre, H_post, H_res = self.heads(X_norm)  # (B,1,n,1),(B,1,n,1),(B,1,n,n)
        H_pre = H_pre.squeeze(1)                   # (B, n, 1)
        H_post = H_post.squeeze(1)                 # (B, n, 1)
        H_res = H_res.squeeze(1)                   # (B, n, n)

        x_cell = (H_pre * X_t).sum(dim=1)          # (B, d)
        h_new, c_new = self.cell(self.dropout(x_cell), state)
        X_new = H_post * h_new.unsqueeze(1)        # (B,n,1) * (B,1,d) -> (B,n,d)
        X_res = torch.einsum("bnm,bmd->bnd", H_res, X_t)
        return X_res + X_new, (h_new, c_new), H_res


class MicroMHCLSTMLayer(nn.Module):
    """Runs one micro-µHC recurrent layer over all T steps."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.cell = MicroMHCLSTMCell(config)
        self.d = config.hidden_dim

    def forward(self, X: torch.Tensor):
        # X: (B, T, n, d) -> X_out: (B, T, n, d), H_res_seq: (B, T, n, n)
        B, T, n, d = X.shape
        h = X.new_zeros(B, self.d)
        c = X.new_zeros(B, self.d)
        outs, hres = [], []
        for t in range(T):
            X_out_t, (h, c), H_res_t = self.cell(X[:, t], (h, c))
            outs.append(X_out_t)
            hres.append(H_res_t)
        X_out = torch.stack(outs, dim=1)           # (B, T, n, d)
        H_res_seq = torch.stack(hres, dim=1)       # (B, T, n, n)
        return X_out, H_res_seq


class MicroMHCLSTM(nn.Module):
    """Top-level micro-placement mHC-LSTM (macro-compatible interface)."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.n = config.n_streams
        self.d = config.hidden_dim
        self.input_dim = config.input_dim
        self.num_layers = config.num_layers
        self.num_horizons = config.num_horizons

        self.input_proj = nn.Linear(self.input_dim, self.n * self.d)
        self.layers = nn.ModuleList([MicroMHCLSTMLayer(config) for _ in range(self.num_layers)])
        self.output_proj = nn.Linear(self.n * self.d, self.num_horizons * self.input_dim)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        X = self.input_proj(x).view(B, T, self.n, self.d)
        h_res_matrices = []
        for layer in self.layers:
            X, H_res_seq = layer(X)
            h_res_matrices.append(H_res_seq)
        last_hidden = X.reshape(B, T, self.n * self.d)[:, -1, :]
        pred = self.output_proj(last_hidden).view(B, self.num_horizons, self.input_dim)
        return pred, h_res_matrices
