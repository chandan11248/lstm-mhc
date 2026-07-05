#!/usr/bin/env python3
"""Consolidate all test-set metrics (L=4/8/16 + L=32 + ETTm1) into final tables.

Inputs (all measured):
  - outputs/tables/test_metrics_per_run.csv      (L=4/8/16, computed offline)
  - outputs/tables/test_metrics_depth32.csv      (L=32 weather + initial ETTm1 seed, computed offline)
  - completed ETTm1 L=32 test metrics parsed from raw Kaggle run logs
    (outputs/logs_ettm1/.../*.log and outputs/logs/*ettm1*/*.log)
  - vanilla weather L=32 test metrics parsed from the ORIGINAL Kaggle run logs
    (outputs/logs_depth32/.../*.log) — these were printed by the trainer's final
    test evaluation on Kaggle; seeds 45/46 hit the 12h session limit and are omitted.

Outputs:
  - outputs/tables/test_metrics_all_per_run.csv  (everything, one row per run)
  - outputs/tables/test_metrics_final_summary.csv (mean/std over seeds per config)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
LOGDIR = ROOT / "outputs" / "logs_depth32"
ETTM1_LOGDIRS = [ROOT / "outputs" / "logs_ettm1", ROOT / "outputs" / "logs"]

TEST_RE = re.compile(r"Test MSE:([0-9.]+) RMSE:([0-9.]+) MAE:([0-9.]+)")
TEST_FULL_RE = re.compile(r"Test MSE:([0-9.]+) RMSE:([0-9.]+) MAE:([0-9.]+) MAPE:([0-9.]+)%")
HORIZON_RE = re.compile(
    r"Test Horizons: \[6h:([0-9.]+) \| 12h:([0-9.]+) \| 24h:([0-9.]+) \| "
    r"48h:([0-9.]+) \| 72h:([0-9.]+)\]"
)
MODEL_RE = re.compile(r"Model: ([a-z_]+) \| L=(\d+)")
SEED_RE = re.compile(r"Seed: (\d+)")
PARAM_RE = re.compile(r"(?:num_params:|Parameters:|Params:)\s*([0-9,]+)")
EXACT_METRIC_RE = re.compile(r"\b(test_mse|test_rmse|test_mae|test_mape|test_smape):\s*([0-9.eE+-]+)")
EXACT_HORIZON_RE = re.compile(r"test_horizon_maes:\s*\[([^\]]+)\]")


def parse_ettm1_l32_logs():
    """Pull completed ETTm1 L=32 test metrics from measured Kaggle logs.

    A run is counted only if the trainer printed the final ``Test MSE`` line.
    Partial/cancelled runs (for example vanilla seed 43) are skipped rather than
    evaluated from intermediate checkpoints or imputed.
    """
    rows = {}
    for log_dir in ETTM1_LOGDIRS:
        if not log_dir.exists():
            continue
        for path in sorted(log_dir.rglob("*ettm1*.log")):
            text = path.read_text(errors="ignore")
            tests = TEST_FULL_RE.findall(text)
            if not tests:
                continue
            model_matches = MODEL_RE.findall(text)
            seed_matches = SEED_RE.findall(text)
            param_matches = PARAM_RE.findall(text)
            if not model_matches or not seed_matches:
                print(f"skip ETTm1 log with missing model/seed metadata: {path}")
                continue
            model, depth_s = model_matches[-1]
            depth = int(depth_s)
            seed = int(seed_matches[-1])
            if depth != 32 or model not in {"mhc", "vanilla"}:
                continue
            mse, rmse, mae, mape = map(float, tests[-1])
            exact = {name: float(value) for name, value in EXACT_METRIC_RE.findall(text)}
            row = {
                "dataset": "ettm1", "model": model, "depth": depth, "seed": seed,
                "params": float(param_matches[-1].replace(",", "")) if param_matches else np.nan,
                "test_mse": exact.get("test_mse", mse),
                "test_rmse": exact.get("test_rmse", rmse),
                "test_mae": exact.get("test_mae", mae),
                "test_mape": exact.get("test_mape", mape),
                "test_smape": exact.get("test_smape", np.nan), "test_mase": np.nan,
                "best_val_mse_ckpt": np.nan, "status": "ok",
                "note": f"from Kaggle ETTm1 run log: {path.relative_to(ROOT)}",
            }
            exact_horizons = EXACT_HORIZON_RE.findall(text)
            if exact_horizons:
                vals = [float(v.strip()) for v in exact_horizons[-1].split(",")]
                for h, val in zip((6, 12, 24, 48, 72), vals):
                    row[f"test_horizon_{h}h_mae"] = val
            else:
                horizons = HORIZON_RE.findall(text)
                if horizons:
                    for h, val in zip((6, 12, 24, 48, 72), horizons[-1]):
                        row[f"test_horizon_{h}h_mae"] = float(val)
            rows[("ettm1", model, depth, seed)] = row
    return list(rows.values())


def parse_vanilla_weather_l32():
    """Pull vanilla weather L=32 test metrics from original Kaggle logs (measured)."""
    rows = []
    for seed in (42, 43, 44, 45, 46):
        logs = list(LOGDIR.rglob(f"*vanilla_l32_s{seed}_weather*/*.log"))
        if not logs:
            continue
        text = logs[0].read_text(errors="ignore")
        m = TEST_RE.search(text)
        if not m:
            continue  # seeds 45/46: session timeout, no final test eval
        mse, rmse, mae = map(float, m.groups())
        rows.append({
            "dataset": "weather", "model": "vanilla", "depth": 32, "seed": seed,
            "params": np.nan, "test_mse": mse, "test_rmse": rmse, "test_mae": mae,
            "test_mape": np.nan, "test_smape": np.nan, "test_mase": np.nan,
            "best_val_mse_ckpt": np.nan, "status": "ok",
            "note": "from Kaggle run log (final test eval)",
        })
    return rows


def main():
    frames = []
    main_csv = TABLES / "test_metrics_per_run.csv"
    if main_csv.exists():
        frames.append(pd.read_csv(main_csv))
    ettm1 = parse_ettm1_l32_logs()
    if ettm1:
        frames.append(pd.DataFrame(ettm1))
    d32_csv = TABLES / "test_metrics_depth32.csv"
    if d32_csv.exists():
        d32 = pd.read_csv(d32_csv)
        # Prefer the raw completed-run ETTm1 log rows above; the offline depth32
        # table may contain only the initial seed-42 ETTm1 runs.
        if ettm1 and "dataset" in d32.columns:
            d32 = d32[d32["dataset"] != "ettm1"]
        frames.append(d32)
    van = parse_vanilla_weather_l32()
    if van:
        frames.append(pd.DataFrame(van))

    if not frames:
        raise FileNotFoundError("No measured test metric inputs found")
    allrows = pd.concat(frames, ignore_index=True, sort=False)
    # Drop duplicate (dataset,model,depth,seed), keeping first occurrence.
    allrows = allrows.drop_duplicates(subset=["dataset", "model", "depth", "seed"], keep="first")
    allrows.to_csv(TABLES / "test_metrics_all_per_run.csv", index=False)
    print(f"Wrote test_metrics_all_per_run.csv ({len(allrows)} rows)")

    ok = allrows[allrows["status"] == "ok"].copy()
    metric_cols = [c for c in ok.columns if c.startswith("test_") or c == "params"]
    out = []
    for (ds, model, depth), g in ok.groupby(["dataset", "model", "depth"]):
        rec = {"dataset": ds, "model": model, "depth": depth,
               "n_seeds": int(g["seed"].nunique()),
               "seeds": ",".join(str(s) for s in sorted(g["seed"].unique()))}
        for c in metric_cols:
            rec[f"{c}_mean"] = float(g[c].mean())
            rec[f"{c}_std"] = float(g[c].std(ddof=1)) if g[c].notna().sum() > 1 else 0.0
        out.append(rec)
    summary = pd.DataFrame(out).sort_values(["dataset", "model", "depth"]).reset_index(drop=True)
    summary.to_csv(TABLES / "test_metrics_final_summary.csv", index=False)
    print(f"Wrote test_metrics_final_summary.csv ({len(summary)} configs)")

    # Quick depth-scaling view.
    print("\nDepth-scaling test MSE (mean):")
    for ds in sorted(summary.dataset.unique()):
        print(f"  --- {ds} ---")
        for m in ["mhc", "vanilla", "gru", "tcn", "hc"]:
            r = summary[(summary.dataset == ds) & (summary.model == m)].sort_values("depth")
            if len(r):
                pts = "  ".join(f"L{int(x.depth)}={x.test_mse_mean:.1f}(n{int(x.n_seeds)})" for x in r.itertuples())
                print(f"    {m:8s}: {pts}")


if __name__ == "__main__":
    main()
