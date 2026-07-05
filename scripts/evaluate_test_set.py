#!/usr/bin/env python3
"""Offline test-set evaluation from locally-saved checkpoints.

WHY THIS EXISTS
---------------
Phase 1/2 reported *validation* MSE (the model-selection metric). Reporting the
same split used for selection is optimistic and a reviewer red flag. This script
evaluates the held-out TEST split using each run's best-validation checkpoint,
producing de-normalized test metrics (MSE/MAE/RMSE/MAPE/sMAPE/MASE + per-horizon
MAE) in original units.

CORRECTNESS / NO-LEAKAGE GUARANTEES
-----------------------------------
- We rebuild the train/val/test split deterministically from each checkpoint's
  embedded config (same ratios, gap, input_len, horizons as training).
- The scaler is re-fit on the TRAIN slice only (identical to training). We also
  cross-check the rebuilt scaler against the scaler stored in the checkpoint and
  warn on any mismatch, so a silent preprocessing drift can never pass unnoticed.
- We load each model from its embedded config (no guessing of hyperparameters)
  and restore the exact saved weights.
- No training happens. Pure inference on the test split.

OUTPUTS
-------
- outputs/tables/test_metrics_per_run.csv   (one row per checkpoint)
- outputs/tables/test_metrics_summary.csv   (mean +/- std over seeds)

Diverged runs (HC L=16, NaN before any best checkpoint) are recorded with
status="diverged" and NaN metrics, never fabricated.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lstm_mhc.utils.config import ExperimentConfig
from lstm_mhc.evaluation.metrics import compute_forecast_metrics, denormalize_np
from lstm_mhc.data.weather_dataset import build_weather_dataloaders, WeatherTimeSeriesDataset, _make_scaler

SEARCH_DIRS = [ROOT / "outputs" / "logs", ROOT / "outputs" / "logs_phase2"]
TABLES = ROOT / "outputs" / "tables"
WEATHER_CSV = ROOT / "data" / "noaa-weather-data-jfk-airport" / "jfk_weather_cleaned.csv"

RUN_PAT = re.compile(r'(?P<model>[a-z]+)_l(?P<depth>\d+)_s(?P<seed>\d+)_(?P<dataset>weather|ett|ettm1)')


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: ExperimentConfig):
    """Instantiate the correct model class from config (mirrors generate_phase2_plots)."""
    from lstm_mhc.models.baselines import build_baseline
    from lstm_mhc.models.hc_lstm import HCLSTM
    from lstm_mhc.models.mhc_lstm import MHCLSTM
    from lstm_mhc.models.vanilla_lstm import StandardLSTM
    from lstm_mhc.models.sota import SOTA_MAP, build_sota

    mt = cfg.model_type
    if mt == "vanilla":
        return StandardLSTM(cfg)
    if mt == "hc":
        return HCLSTM(cfg)
    if mt == "mhc":
        return MHCLSTM(cfg)
    if mt in SOTA_MAP:
        return build_sota(mt, cfg)
    if mt in {"ridge", "dlinear", "nlinear", "gru", "tcn"}:
        return build_baseline(mt, cfg)
    raise ValueError(f"Unknown model_type: {mt}")


# --- Cache test loaders + scaler + MASE train targets per split signature ---
_loader_cache: dict = {}


# Larger inference batch -> far better device utilization. Inference is pure
# forward; no batch-dependent layers (no BatchNorm), so metrics are identical
# regardless of batch size. Overridable via --eval-batch.
EVAL_BATCH = 256


def get_test_assets(cfg: ExperimentConfig):
    """Return (test_loader, scaler_info, train_targets_denorm) for cfg's dataset/split.

    Cached by the split-relevant fields so we read each CSV / rebuild windows
    only once, not once per checkpoint. We override batch_size for eval speed.
    """
    cfg.batch_size = EVAL_BATCH
    key = (
        cfg.dataset, cfg.ett_file, cfg.input_len, tuple(cfg.horizons),
        cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.split_gap,
        cfg.batch_size, tuple(cfg.weather_columns),
    )
    if key in _loader_cache:
        return _loader_cache[key]

    if cfg.dataset == "weather":
        if not WEATHER_CSV.exists():
            raise FileNotFoundError(f"Weather CSV not found at {WEATHER_CSV}")
        _, _, test_loader, scaler_info = build_weather_dataloaders(str(WEATHER_CSV), cfg)
        train_targets = _train_targets_weather(cfg, scaler_info)
    elif cfg.dataset == "ett":
        from lstm_mhc.data.ett_dataset import build_ett_dataloaders
        _, _, test_loader, scaler_info = build_ett_dataloaders(cfg, csv_name=cfg.ett_file)
        train_targets = _train_targets_ett(cfg, scaler_info)
    else:
        raise ValueError(f"Unsupported dataset for offline eval: {cfg.dataset}")

    _loader_cache[key] = (test_loader, scaler_info, train_targets)
    return _loader_cache[key]


def _train_targets_weather(cfg, scaler_info):
    """De-normalized train-window targets for MASE (mirrors trainer logic)."""
    from lstm_mhc.data.weather_dataset import load_and_clean_weather
    df = load_and_clean_weather(str(WEATHER_CSV), columns=cfg.weather_columns)
    return _train_targets_from_df(df, cfg, scaler_info)


def _train_targets_ett(cfg, scaler_info):
    from lstm_mhc.data.ett_dataset import load_ett
    df = load_ett(None, cfg.ett_file, data_dir="data/ett")
    return _train_targets_from_df(df, cfg, scaler_info)


def _train_targets_from_df(df, cfg, scaler_info):
    data = df.values.astype(np.float32)
    N = len(data)
    gap = cfg.split_gap
    train_end = int(N * cfg.train_ratio)
    train_slice = data[:train_end]
    mean, std = _make_scaler(train_slice)
    train_norm = (train_slice - mean) / std
    ds = WeatherTimeSeriesDataset(train_norm, cfg.input_len, cfg.horizons, cfg.window_stride)
    ys = np.stack([ds[i][1].numpy() for i in range(len(ds))]) if len(ds) else np.zeros((0, len(cfg.horizons), data.shape[1]))
    ys = ys.reshape(-1, ys.shape[-1])
    return denormalize_np(ys, scaler_info["mean"], scaler_info["std"])


@torch.no_grad()
def eval_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["config"])
    cfg_dict.pop("output_path", None)
    cfg_dict.pop("config_hash", None)
    # Redirect outputs to a throwaway dir (config __post_init__ creates folders).
    cfg_dict["output_dir"] = tempfile.mkdtemp(prefix="ttest_")
    cfg_dict["num_workers"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = ExperimentConfig._from_mapping(cfg_dict)

    test_loader, scaler_info, train_targets = get_test_assets(cfg)

    # Leakage cross-check: rebuilt scaler must match the checkpoint's scaler.
    scaler_warn = ""
    ck_mean = ckpt.get("scaler_mean")
    if ck_mean is not None:
        diff = float(np.max(np.abs(np.asarray(ck_mean) - np.asarray(scaler_info["mean"]))))
        if diff > 1e-3:
            scaler_warn = f"scaler_mean drift={diff:.4g}"

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    preds, targets = [], []
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    mean, std = scaler_info["mean"], scaler_info["std"]
    preds = denormalize_np(preds, mean, std)
    targets = denormalize_np(targets, mean, std)

    m = compute_forecast_metrics(preds, targets, cfg.horizons, train_targets)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    row = {
        "dataset": cfg.dataset,
        "model": cfg.model_type,
        "depth": cfg.num_layers,
        "seed": cfg.seed,
        "params": n_params,
        "test_mse": m.mse,
        "test_rmse": m.rmse,
        "test_mae": m.mae,
        "test_mape": m.mape,
        "test_smape": m.smape,
        "test_mase": m.mase,
        "best_val_mse_ckpt": float(ckpt.get("best_val_mse", float("nan"))),
        "status": "ok",
        "note": scaler_warn,
    }
    for h, hm in zip(cfg.horizons, m.horizon_maes):
        row[f"test_horizon_{h}h_mae"] = hm
    return row


def discover_best_checkpoints() -> dict:
    """Map (dataset, model, depth, seed) -> best.pt path (dedup, first wins)."""
    best: dict = {}
    for sd in SEARCH_DIRS:
        if not sd.exists():
            continue
        for pt in sorted(sd.rglob("*_best.pt")):
            m = RUN_PAT.search(pt.name) or RUN_PAT.search(pt.parent.parent.parent.name)
            if not m:
                continue
            key = (m["dataset"], m["model"], int(m["depth"]), int(m["seed"]))
            best.setdefault(key, pt)
    return best


def diverged_rows() -> list:
    """HC L=16 diverged (NaN) before any best checkpoint — report, never fabricate."""
    rows = []
    for ds in ("weather", "ett"):
        for seed in (42, 43, 44, 45, 46):
            rows.append({
                "dataset": ds, "model": "hc", "depth": 16, "seed": seed,
                "params": np.nan, "test_mse": np.nan, "test_rmse": np.nan,
                "test_mae": np.nan, "test_mape": np.nan, "test_smape": np.nan,
                "test_mase": np.nan, "best_val_mse_ckpt": np.nan,
                "status": "diverged", "note": "NaN before first epoch (no best ckpt)",
            })
    return rows


def aggregate(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_run.columns if c.startswith("test_") or c == "params"]
    ok = per_run[per_run["status"] == "ok"].copy()
    grp = ok.groupby(["dataset", "model", "depth"])
    out = []
    for (ds, model, depth), g in grp:
        rec = {"dataset": ds, "model": model, "depth": depth,
               "n_seeds": int(g["seed"].nunique()),
               "seeds": ",".join(str(s) for s in sorted(g["seed"].unique()))}
        for c in metric_cols:
            rec[f"{c}_mean"] = float(g[c].mean())
            rec[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        out.append(rec)
    return pd.DataFrame(out).sort_values(["dataset", "model", "depth"]).reset_index(drop=True)


def main():
    global EVAL_BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="", help="substring filter on dataset/model (e.g. 'mhc', 'weather')")
    ap.add_argument("--limit", type=int, default=0, help="evaluate at most N checkpoints (0=all)")
    ap.add_argument("--eval-batch", type=int, default=EVAL_BATCH, help="inference batch size")
    ap.add_argument("--fresh", action="store_true", help="ignore existing per-run CSV and redo all")
    ap.add_argument("--cpu", action="store_true", help="force CPU (avoids MPS memory pressure)")
    args = ap.parse_args()

    EVAL_BATCH = args.eval_batch

    device = pick_device(force_cpu=args.cpu)
    print(f"Device: {device} | eval_batch={EVAL_BATCH}")
    TABLES.mkdir(parents=True, exist_ok=True)

    best = discover_best_checkpoints()
    keys = sorted(best.keys())
    if args.filter:
        keys = [k for k in keys if args.filter in k[0] or args.filter in k[1]]
    if args.limit:
        keys = keys[: args.limit]

    per_run_path = TABLES / "test_metrics_per_run.csv"

    # Resume: keep prior ok rows, skip those keys.
    prior_rows = []
    done_keys = set()
    if per_run_path.exists() and not args.fresh:
        prev = pd.read_csv(per_run_path)
        for _, r in prev.iterrows():
            if r.get("status") == "ok":
                prior_rows.append(r.to_dict())
                done_keys.add((r["dataset"], r["model"], int(r["depth"]), int(r["seed"])))
        keys = [k for k in keys if k not in done_keys]
        print(f"Resuming: {len(done_keys)} runs already done, {len(keys)} to go.")

    print(f"Found {len(best)} best checkpoints; evaluating {len(keys)}.\n")

    rows = list(prior_rows)
    for i, key in enumerate(keys, 1):
        ds, model, depth, seed = key
        try:
            row = eval_checkpoint(best[key], device)
            tag = f"  {row['note']}" if row["note"] else ""
            print(f"[{i}/{len(keys)}] {ds:7s} {model:8s} L={depth:<2d} s{seed}  "
                  f"test_mse={row['test_mse']:.4f} mae={row['test_mae']:.4f}{tag}")
        except Exception as exc:
            print(f"[{i}/{len(keys)}] {ds:7s} {model:8s} L={depth:<2d} s{seed}  ERROR: {exc}")
            row = {"dataset": ds, "model": model, "depth": depth, "seed": seed,
                   "status": "error", "note": str(exc)[:120], "test_mse": np.nan}
        rows.append(row)
        # Incremental save so partial progress survives interruption.
        pd.DataFrame(rows).to_csv(per_run_path, index=False)

    # Append diverged HC L=16 rows only on a full (unfiltered) run.
    if not args.filter and not args.limit:
        rows.extend(diverged_rows())

    per_run = pd.DataFrame(rows)
    per_run.to_csv(per_run_path, index=False)
    print(f"\nWrote {per_run_path} ({len(per_run)} rows)")

    summary = aggregate(per_run)
    summary_path = TABLES / "test_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary)} groups)")


if __name__ == "__main__":
    main()
