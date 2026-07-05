"""PatchTST (Nie et al., ICLR 2023), adapted to our multi-horizon interface.

Core ideas: (1) channel independence — each variate is processed by the same
Transformer with shared weights; (2) patching — each variate's series is split
into overlapping patches that become tokens.

Reference: Y. Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers," ICLR 2023.

Adaptation: the flattened patch representation is mapped to our num_horizons
forecast per variate, giving (B, H, F).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, config, d_model: int = 128, n_heads: int = 8,
                 e_layers: int = 3, d_ff: int = 256,
                 patch_len: int = 24, stride: int = 12):
        super().__init__()
        self.input_len = config.input_len
        self.input_dim = config.input_dim
        self.num_horizons = config.num_horizons
        self.patch_len = patch_len
        self.stride = stride
        dropout = config.dropout

        # Number of patches: pad the end by `stride`, then unfold.
        padded_len = self.input_len + stride
        self.num_patches = (padded_len - patch_len) // stride + 1
        self.pad = nn.ReplicationPad1d((0, stride))  # pad end so the tail is covered

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.embed_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.head = nn.Linear(self.num_patches * d_model, self.num_horizons)

    def _patchify(self, z: torch.Tensor) -> torch.Tensor:
        # z: (BF, T) -> (BF, num_patches+?, patch_len) via unfold on padded series
        z = self.pad(z.unsqueeze(1)).squeeze(1)           # (BF, T+stride)
        patches = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches                                     # (BF, n_patches', patch_len)

    def forward(self, x: torch.Tensor):
        B, T, F = x.shape
        # Channel independence: fold variates into the batch.
        z = x.permute(0, 2, 1).reshape(B * F, T)          # (B*F, T)
        patches = self._patchify(z)                        # (B*F, P, patch_len)
        tok = self.embed_drop(self.patch_embed(patches))   # (B*F, P, d_model)
        tok = tok + self.pos[:, : tok.size(1), :]
        enc = self.encoder(tok)                            # (B*F, P, d_model)
        flat = enc.reshape(enc.size(0), -1)                # (B*F, P*d_model)
        out = self.head(flat)                              # (B*F, H)
        pred = out.reshape(B, F, self.num_horizons).permute(0, 2, 1)  # (B, H, F)
        return pred, None
