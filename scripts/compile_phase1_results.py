#!/usr/bin/env python3
"""Compile Phase 1 results from downloaded Kaggle epoch/step CSVs.

Discovers all epoch CSVs in ``outputs/logs/`` (nested layout), extracts
per-run metrics, groups by (dataset, depth, model), and writes summary
tables to ``outputs/tables/``.

Outputs:
    - phase1_summary.csv     (one row per run)
    - phase1_aggregate.csv   (one row per dataset/depth/model, mean +/- std)
    - phase1_diverged.csv    (runs with empty CSVs or all-NaN metrics)
    - phase1_stability.csv   (step-level: max grad norm, spectral norm, amax)

Usage:
    python scripts/compile_phase1_results.py
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "outputs" / "logs"
TABLES_DIR = ROOT / "outputs" / "tables"

# Pattern: {model}_l{depth}_s{seed}_{dataset}[_h{hidden_dim}]
RUN_RE = re.compile(
    r"^(?P<model>vanilla|hc|mhc)_l(?P<depth>\d+)_s(?P<seed>\d+)_"
    r"(?P<dataset>weather|ett)(?:_h\d+)?$"
)


def strip_hidden_suffix(name: str) -> str:
    """Strip the _h{dim} suffix from vanilla run names for master_status matching."""
    return re.sub(r"_h\d+$", "", name)


def discover_epoch_csvs() -> list[tuple[Path, str, str]]:
    """Find all epoch CSVs and return [(path, canonical_name, base_name)].

    canonical_name includes _h{dim} (matches the CSV filename).
    base_name strips _h{dim} (matches master_status.csv).
    """
    results = []
    for csv_path in sorted(LOGS_DIR.rglob("*_epoch*.csv")):
        stem = csv_path.stem.replace("_epoch", "")
        m = RUN_RE.match(stem)
        if m is None:
            continue
        base_name = strip_hidden_suffix(stem)
        results.append((csv_path, stem, base_name))
    return results


def discover_step_csv(canonical_name: str) -> Path | None:
    """Find the step CSV for a given canonical run name (searches all account dirs)."""
    for csv_path in LOGS_DIR.rglob(f"{canonical_name}_step.csv"):
        return csv_path
    return None


def parse_epoch_csv(csv_path: Path) -> dict | None:
    """Parse an epoch CSV and extract best-epoch metrics.

    Returns None if the CSV is empty (header only) or all val_mse are NaN.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ERROR reading {csv_path}: {e}")
        return None

    if df.empty or "val_mse" not in df.columns:
        return None

    # Drop rows with NaN val_mse
    df = df.dropna(subset=["val_mse"])
    if df.empty:
        return None

    best_idx = df["val_mse"].idxmin()
    best = df.loc[best_idx]

    result = {
        "best_epoch": int(best["epoch"]),
        "best_val_mse": float(best["val_mse"]),
        "best_val_mae": float(best["val_mae"]) if "val_mae" in df.columns else float("nan"),
        "total_epochs": len(df),
        "avg_epoch_time_s": float(df["epoch_time_s"].mean()) if "epoch_time_s" in df.columns else float("nan"),
        "peak_vram_mb": float(df["peak_vram_mb"].max()) if "peak_vram_mb" in df.columns else float("nan"),
    }

    # Horizon MAEs at best epoch
    for h in [6, 12, 24, 48, 72]:
        col = f"horizon_{h}h_mae"
        result[f"horizon_{h}h_mae"] = float(best[col]) if col in df.columns else float("nan")

    return result


def parse_step_csv(csv_path: Path) -> dict:
    """Parse a step CSV and extract stability metrics (max values)."""
    result = {
        "max_grad_norm": float("nan"),
        "max_spectral_norm": float("nan"),
        "max_fwd_amax": float("nan"),
        "max_bwd_amax": float("nan"),
    }
    if csv_path is None or not csv_path.exists():
        return result

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return result

    if df.empty:
        return result

    if "grad_norm" in df.columns:
        gn = pd.to_numeric(df["grad_norm"], errors="coerce").dropna()
        if len(gn) > 0:
            result["max_grad_norm"] = float(gn.max())

    if "spectral_norm" in df.columns:
        sn = pd.to_numeric(df["spectral_norm"], errors="coerce").dropna()
        if len(sn) > 0:
            result["max_spectral_norm"] = float(sn.max())

    if "fwd_amax" in df.columns:
        fa = pd.to_numeric(df["fwd_amax"], errors="coerce").dropna()
        if len(fa) > 0:
            result["max_fwd_amax"] = float(fa.max())

    if "bwd_amax" in df.columns:
        ba = pd.to_numeric(df["bwd_amax"], errors="coerce").dropna()
        if len(ba) > 0:
            result["max_bwd_amax"] = float(ba.max())

    return result


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {LOGS_DIR} for epoch CSVs...")
    epoch_csvs = discover_epoch_csvs()
    print(f"Found {len(epoch_csvs)} epoch CSVs\n")

    # --- Parse all runs ---
    rows = []
    diverged = []
    seen_runs = set()      # canonical names (with _h suffix)
    seen_base_runs = set()  # base names (without _h suffix, matches master_status)

    for csv_path, canonical_name, base_name in epoch_csvs:
        m = RUN_RE.match(canonical_name)
        model = m.group("model")
        depth = int(m.group("depth"))
        seed = int(m.group("seed"))
        dataset = m.group("dataset")

        # Dedup: keep the first CSV found per base run name
        if base_name in seen_base_runs:
            continue
        seen_runs.add(canonical_name)
        seen_base_runs.add(base_name)

        epoch_data = parse_epoch_csv(csv_path)

        # Step CSV (for stability) — search by canonical name
        step_path = discover_step_csv(canonical_name)
        step_data = parse_step_csv(step_path)

        if epoch_data is None:
            diverged.append({
                "run_name": base_name,
                "dataset": dataset,
                "depth": depth,
                "seed": seed,
                "model": model,
                "reason": "empty or all-NaN epoch CSV (model diverged before completing any epoch)",
                "csv_path": str(csv_path),
            })
            continue

        row = {
            "run_name": base_name,
            "dataset": dataset,
            "depth": depth,
            "seed": seed,
            "model": model,
            **epoch_data,
            **step_data,
        }
        rows.append(row)

    # --- Also check for runs in master_status that have NO epoch CSV at all ---
    status_path = ROOT / "kaggle" / "master_status.csv"
    if status_path.exists():
        status_df = pd.read_csv(status_path)
        done_runs = set(status_df[status_df["status"] == "done"]["run_name"])
        missing = done_runs - seen_base_runs
        for run_name in sorted(missing):
            m = RUN_RE.match(run_name)
            if m is None:
                continue
            diverged.append({
                "run_name": run_name,
                "dataset": m.group("dataset"),
                "depth": int(m.group("depth")),
                "seed": int(m.group("seed")),
                "model": m.group("model"),
                "reason": "no epoch CSV found (download may have failed or run diverged immediately)",
                "csv_path": "",
            })

    print(f"Parsed: {len(rows)} runs with valid data")
    print(f"Diverged/missing: {len(diverged)} runs\n")

    # --- Write per-run summary ---
    summary_cols = [
        "run_name", "dataset", "depth", "seed", "model",
        "best_epoch", "best_val_mse", "best_val_mae",
        "horizon_6h_mae", "horizon_12h_mae", "horizon_24h_mae",
        "horizon_48h_mae", "horizon_72h_mae",
        "total_epochs", "avg_epoch_time_s", "peak_vram_mb",
        "max_grad_norm", "max_spectral_norm", "max_fwd_amax", "max_bwd_amax",
    ]
    summary_df = pd.DataFrame(rows, columns=summary_cols)
    summary_df = summary_df.sort_values(["dataset", "depth", "model", "seed"])
    summary_df.to_csv(TABLES_DIR / "phase1_summary.csv", index=False)
    print(f"Wrote: {TABLES_DIR / 'phase1_summary.csv'} ({len(summary_df)} rows)")

    # --- Write aggregate (mean +/- std per group) ---
    agg_rows = []
    for (dataset, depth, model), group in summary_df.groupby(["dataset", "depth", "model"]):
        n = len(group)
        agg = {
            "dataset": dataset,
            "depth": depth,
            "model": model,
            "n_seeds": n,
            "best_val_mse_mean": group["best_val_mse"].mean(),
            "best_val_mse_std": group["best_val_mse"].std(),
            "best_val_mae_mean": group["best_val_mae"].mean(),
            "best_val_mae_std": group["best_val_mae"].std(),
            "horizon_72h_mae_mean": group["horizon_72h_mae"].mean(),
            "horizon_72h_mae_std": group["horizon_72h_mae"].std(),
            "total_epochs_mean": group["total_epochs"].mean(),
            "avg_epoch_time_s_mean": group["avg_epoch_time_s"].mean(),
            "peak_vram_mb_max": group["peak_vram_mb"].max(),
            "max_grad_norm_max": group["max_grad_norm"].max(),
            "max_spectral_norm_max": group["max_spectral_norm"].max(),
            "max_fwd_amax_max": group["max_fwd_amax"].max(),
        }
        agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows)
    agg_df = agg_df.sort_values(["dataset", "depth", "model"])
    agg_df.to_csv(TABLES_DIR / "phase1_aggregate.csv", index=False)
    print(f"Wrote: {TABLES_DIR / 'phase1_aggregate.csv'} ({len(agg_df)} rows)")

    # --- Write diverged ---
    if diverged:
        div_df = pd.DataFrame(diverged)
        div_df.to_csv(TABLES_DIR / "phase1_diverged.csv", index=False)
        print(f"Wrote: {TABLES_DIR / 'phase1_diverged.csv'} ({len(div_df)} rows)")
    else:
        print("No diverged runs detected.")

    # --- Write stability ---
    stability_cols = [
        "run_name", "dataset", "depth", "seed", "model",
        "max_grad_norm", "max_spectral_norm", "max_fwd_amax", "max_bwd_amax",
    ]
    stability_df = summary_df[stability_cols].copy()
    stability_df.to_csv(TABLES_DIR / "phase1_stability.csv", index=False)
    print(f"Wrote: {TABLES_DIR / 'phase1_stability.csv'} ({len(stability_df)} rows)")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"PHASE 1 RESULTS SUMMARY")
    print(f"{'='*70}")

    for dataset in ["weather", "ett"]:
        print(f"\n--- {dataset.upper()} ---")
        ds_agg = agg_df[agg_df["dataset"] == dataset]
        for _, row in ds_agg.iterrows():
            print(
                f"  L={int(row['depth']):2d} {row['model']:8s} | "
                f"Val MSE: {row['best_val_mse_mean']:10.4f} +/- {row['best_val_mse_std']:8.4f} | "
                f"72h MAE: {row['horizon_72h_mae_mean']:8.4f} +/- {row['horizon_72h_mae_std']:6.4f} | "
                f"n={int(row['n_seeds'])}"
            )

    if diverged:
        print(f"\n--- DIVERGED / MISSING ({len(diverged)}) ---")
        for d in diverged:
            print(f"  {d['run_name']:30s} | {d['reason']}")

    print(f"\nDone. Tables written to {TABLES_DIR}/")


if __name__ == "__main__":
    main()
