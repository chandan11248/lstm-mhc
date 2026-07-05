#!/usr/bin/env python3
"""Measure parameter count and FLOPs/MACs for each model (CPU, deterministic).

Runs locally on CPU (one forward pass per model) — no GPU, no training. Produces
outputs/tables/efficiency.csv with columns:
    model, depth, params, macs, gflops

FLOP coverage note: thop provides hooks for nn.LSTM / nn.GRU / nn.Linear / conv,
which dominate cost here. The small custom ops in the muHC heads (exp, row/col
normalization in Sinkhorn-Knopp, einsum mixing on n=4 streams) are not counted by
thop and are negligible relative to the LSTM matmuls. We report the thop count and
state this caveat in the paper rather than hand-deriving uncounted ops.

Inference latency and peak VRAM are intentionally NOT measured here: they must be
taken on a single fixed GPU to be comparable, via a separate Kaggle notebook.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lstm_mhc.utils.config import ExperimentConfig
from lstm_mhc.models.vanilla_lstm import StandardLSTM
from lstm_mhc.models.hc_lstm import HCLSTM
from lstm_mhc.models.mhc_lstm import MHCLSTM
from lstm_mhc.models.baselines import build_baseline
from lstm_mhc.models.sota import build_sota, SOTA_MAP

from thop import profile

TABLES = ROOT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# (model_type, depth) combinations to profile. Baselines are depth-agnostic in
# their own way but we profile at the depths used in the paper.
CORE = [("mhc", d) for d in (4, 8, 16, 32)] + \
       [("vanilla", d) for d in (4, 8, 16, 32)] + \
       [("hc", d) for d in (4, 8, 16)]
BASELINES = [("gru", d) for d in (4, 8, 16)] + \
            [("tcn", d) for d in (4, 8, 16)] + \
            [("dlinear", 4), ("nlinear", 4), ("ridge", 4)]
# SOTA transformer/CNN forecasters use their own depth (paper Table tab:efficiency,
# L=4 row is the only one reported, but the model itself is not "depth 4" in the
# LSTM sense -- we keep the label "4" only to align with the table's L column).
SOTA = [(name, 4) for name in SOTA_MAP]


def build(model_type, depth):
    cfg = ExperimentConfig(model_type=model_type, num_layers=depth,
                           n_streams=4, hidden_dim=128, input_dim=4,
                           input_len=724, horizons=[6, 12, 24, 48, 72],
                           output_dir="outputs")
    if model_type == "vanilla":
        return StandardLSTM(cfg), cfg
    if model_type == "hc":
        return HCLSTM(cfg), cfg
    if model_type == "mhc":
        return MHCLSTM(cfg), cfg
    if model_type in SOTA_MAP:
        return build_sota(model_type, cfg), cfg
    return build_baseline(model_type, cfg), cfg


def main():
    device = torch.device("cpu")
    rows = []
    for model_type, depth in CORE + BASELINES + SOTA:
        try:
            model, cfg = build(model_type, depth)
            model.eval().to(device)
            x = torch.randn(1, cfg.input_len, cfg.input_dim, device=device)
            params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # thop returns (macs, params); some models return (pred, aux).
                macs, _ = profile(model, inputs=(x,), verbose=False)
            gflops = 2.0 * macs / 1e9
            rows.append({"model": model_type, "depth": depth, "params": params,
                         "macs": int(macs), "gflops": round(gflops, 4)})
            print(f"{model_type:8s} L={depth:<2d}  params={params:>10,}  GFLOPs={gflops:8.3f}")
        except Exception as exc:
            print(f"{model_type:8s} L={depth:<2d}  ERROR: {exc}")
            rows.append({"model": model_type, "depth": depth, "params": float("nan"),
                         "macs": float("nan"), "gflops": float("nan")})

    df = pd.DataFrame(rows)
    out = TABLES / "efficiency.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
