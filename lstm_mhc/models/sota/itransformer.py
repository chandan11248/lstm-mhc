"""iTransformer (Liu et al., ICLR 2024), adapted to our multi-horizon interface.

Core idea: invert the data so each *variate* (channel) becomes a token. The
whole time series of one variate is embedded into a d_model vector; a standard
Transformer encoder attends across variates; a linear head maps each variate
token to the forecast.

Reference: Y. Liu et al., "iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting," ICLR 2024.

Adaptation: the prediction length is our set of horizons (default 5), so the
output head maps d_model -> num_horizons for each variate, giving (B, H, F).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ITransformer(nn.Module):
    def __init__(self, config, d_model: int = 128, n_heads: int = 8,
                 e_layers: int = 3, d_ff: int = 256):
        super().__init__()
        self.input_len = config.input_len
        self.input_dim = config.input_dim
        self.num_horizons = config.num_horizons
        dropout = config.dropout

        # Variate embedding: each variate's length-T series -> d_model.
        self.embed = nn.Linear(self.input_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        # Project each variate token to the forecast horizons.
        self.head = nn.Linear(d_model, self.num_horizons)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F) -> invert to variate tokens (B, F, T)
        B, T, F = x.shape
        z = x.permute(0, 2, 1)                 # (B, F, T)
        tokens = self.embed_drop(self.embed(z))  # (B, F, d_model)
        enc = self.encoder(tokens)             # (B, F, d_model)
        out = self.head(enc)                   # (B, F, H)
        pred = out.permute(0, 2, 1)            # (B, H, F)
        return pred, None
