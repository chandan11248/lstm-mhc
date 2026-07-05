"""TimesNet (Wu et al., ICLR 2023), adapted to our multi-horizon interface.

Core idea: detect the dominant periods of the series via FFT, reshape the 1D
sequence into a 2D tensor (period x cycles), and model intra-/inter-period
variation with an Inception-style 2D convolution. Outputs from the top-k periods
are aggregated by their (softmax) amplitudes.

Reference: H. Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
Time Series Analysis," ICLR 2023. Implementation follows the official structure
(FFT_for_Period, Inception_Block_V1, TimesBlock); the prediction head is adapted
to emit our num_horizons forecast per variate -> (B, H, F).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_for_period(x: torch.Tensor, k: int = 2):
    # x: (B, T, d). Amplitude spectrum averaged over batch & channels.
    xf = torch.fft.rfft(x, dim=1)
    freq_amp = abs(xf).mean(0).mean(-1)        # (T//2+1,)
    freq_amp[0] = 0                            # ignore DC
    k = min(k, freq_amp.shape[0] - 1) if freq_amp.shape[0] > 1 else 1
    _, top = torch.topk(freq_amp, k)
    top = top.detach().cpu().numpy()
    T = x.shape[1]
    periods = [max(T // int(f), 1) for f in top]
    return periods, abs(xf).mean(-1)[:, top]   # periods, (B, k) amplitudes


class InceptionBlockV1(nn.Module):
    def __init__(self, in_ch, out_ch, num_kernels: int = 6):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])

    def forward(self, x):
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k=2, num_kernels=6):
        super().__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels),
        )

    def forward(self, x):
        B, T, d = x.shape
        periods, amps = fft_for_period(x, self.k)
        outs = []
        for p in periods:
            # pad time so it is a multiple of p
            if T % p != 0:
                pad = ((T // p) + 1) * p - T
                xp = torch.cat([x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
                length = T + pad
            else:
                xp = x
                length = T
            # reshape to 2D image: (B, d, cycles, period)
            img = xp.reshape(B, length // p, p, d).permute(0, 3, 1, 2)
            img = self.conv(img)
            back = img.permute(0, 2, 3, 1).reshape(B, length, d)[:, :T, :]
            outs.append(back)
        stacked = torch.stack(outs, dim=-1)             # (B, T, d, k)
        weights = torch.softmax(amps, dim=1)            # (B, k)
        weights = weights.unsqueeze(1).unsqueeze(1)     # (B,1,1,k)
        agg = (stacked * weights).sum(-1)
        return agg + x                                  # residual


class TimesNet(nn.Module):
    def __init__(self, config, d_model: int = 64, d_ff: int = 64,
                 e_layers: int = 2, top_k: int = 2, num_kernels: int = 6):
        super().__init__()
        self.input_len = config.input_len
        self.input_dim = config.input_dim
        self.num_horizons = config.num_horizons
        dropout = config.dropout

        self.embed = nn.Linear(self.input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.input_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TimesBlock(self.input_len, d_model, d_ff, top_k, num_kernels)
            for _ in range(e_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(e_layers)])
        self.projection = nn.Linear(d_model, self.input_dim)
        # map time dimension to the forecast horizons
        self.time_head = nn.Linear(self.input_len, self.num_horizons)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        z = self.drop(self.embed(x) + self.pos)         # (B, T, d_model)
        for block, norm in zip(self.blocks, self.norms):
            z = norm(block(z))
        proj = self.projection(z)                        # (B, T, F)
        out = self.time_head(proj.permute(0, 2, 1))      # (B, F, H)
        return out.permute(0, 2, 1), None                # (B, H, F)
