"""Model C: mHC-LSTM (Manifold-Constrained Hyper-Connections).

Our method — uses multi-stream residual connections WITH the Birkhoff
polytope projection (Sinkhorn-Knopp algorithm). The doubly stochastic
constraint on H_res ensures gradient stability at any depth.
"""

import torch
import torch.nn as nn

from ..utils.config import ExperimentConfig
from .components import MuHCHeads, RMSNorm


class MHCLSTMBlock(nn.Module):
    """One inter-layer µHC block WITH manifold constraints.

    Flow::

        X_l (B, T, n, d)
        → RMSNorm + Flatten → X_norm (B, T, nd)
        → Compute H_pre (sigmoid), H_post (2*sigmoid), H_res (Sinkhorn)
        → H_pre · X_l → squeeze → LSTM → H_post · LSTM_out
        → H_res · X_l + above = X_{l+1}
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.n = config.n_streams
        self.d = config.hidden_dim
        self.nd = config.n_streams * config.hidden_dim

        self.rmsnorm = RMSNorm(self.nd, eps=config.rmsnorm_eps)
        self.heads = MuHCHeads.from_config(config, constrained=True)

        self.lstm = nn.LSTM(
            input_size=self.d, hidden_size=self.d,
            num_layers=1, batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X: torch.Tensor):
        B, T, n, d = X.shape
        X_flat = X.reshape(B, T, n * d)
        X_norm = self.rmsnorm(X_flat)
        H_pre, H_post, H_res = self.heads(X_norm)

        # Stream aggregation.
        X_agg = (H_pre * X).sum(dim=2)                             # (B, T, d)
        lstm_out, _ = self.lstm(self.dropout(X_agg))                # (B, T, d)
        X_new = H_post * lstm_out.unsqueeze(2)                     # (B, T, n, d)

        # Doubly stochastic residual mixing (guaranteed stable).
        X_res = torch.einsum("btnm,btmd->btnd", H_res, X)
        return X_res + X_new, H_res


class MHCLSTM(nn.Module):
    """Model C: mHC-LSTM — our full method.

    Multi-stream residual architecture WITH the Birkhoff polytope projection.
    The doubly stochastic constraint on H_res ensures gradient stability at
    any depth and bounded signal propagation.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.n = config.n_streams
        self.d = config.hidden_dim
        self.input_dim = config.input_dim
        self.num_layers = config.num_layers
        self.num_horizons = config.num_horizons

        self.input_proj = nn.Linear(self.input_dim, self.n * self.d)
        self.blocks = nn.ModuleList([MHCLSTMBlock(config) for _ in range(self.num_layers)])
        self.output_proj = nn.Linear(self.n * self.d, self.num_horizons * self.input_dim)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        X = self.input_proj(x).view(B, T, self.n, self.d)
        h_res_matrices = []
        for block in self.blocks:
            X, H_res = block(X)
            h_res_matrices.append(H_res)
        last_hidden = X.reshape(B, T, self.n * self.d)[:, -1, :]
        pred = self.output_proj(last_hidden).view(B, self.num_horizons, self.input_dim)
        return pred, h_res_matrices
