"""Model A: Standard stacked LSTM (parameter-matched baseline).

No highway, no streams, no manifold constraint. Hidden dim is solved at
config time so Model A's total parameter count matches Model C's to within
``config.match_tolerance`` (default 5%). This ensures reviewers can verify
that Model C's advantage comes from the architecture, not from extra capacity.
"""

import torch
import torch.nn as nn

from ..utils.config import ExperimentConfig


class StandardLSTM(nn.Module):
    """Parameter-matched stacked LSTM baseline.

    Architecture::

        Input (B, T, input_dim)
        → Dropout
        → Stacked LSTM (num_layers, vanilla_hidden_dim)
        → Dropout
        → Linear head → Predictions (B, num_horizons, input_dim)

    Uses standard PyTorch LSTM with cuDNN optimisation. The hidden dim is
    determined at config time via :func:`match_vanilla` so total params ≈ Model C.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.vanilla_hidden_dim or (config.n_streams * config.hidden_dim)
        self.num_layers = config.num_layers
        self.num_horizons = config.num_horizons

        self.input_dropout = nn.Dropout(config.dropout)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.dropout if self.num_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(config.dropout)
        self.output_head = nn.Linear(
            self.hidden_dim, self.num_horizons * self.input_dim,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, input_dim).
        Returns:
            predictions: (B, num_horizons, input_dim).
            h_res_matrices: None (no highway).
        """
        B, T, _ = x.shape
        lstm_out, _ = self.lstm(self.input_dropout(x))  # (B, T, hidden_dim)
        last = self.output_dropout(lstm_out[:, -1, :])
        pred = self.output_head(last).view(B, self.num_horizons, self.input_dim)
        return pred, None
