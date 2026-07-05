#!/usr/bin/env python3
"""Run statistical significance tests on Phase 1 results.

Reads ``outputs/tables/phase1_summary.csv`` (produced by compile_phase1_results.py)
and runs paired comparisons between models at each (dataset, depth).

Comparisons:
    - mHC vs HC (all depths where both have 5 seeds)
    - mHC vs Vanilla (all depths where both have 5 seeds)

Uses ``lstm_mhc.utils.stats.paired_significance`` for paired t-test,
Wilcoxon signed-rank, bootstrap CI, and Cohen's d.

Output: ``outputs/tables/phase1_significance.csv``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lstm_mhc.utils.stats import paired_significance

SUMMARY_CSV = ROOT / "outputs" / "tables" / "phase1_summary.csv"
OUTPUT_CSV = ROOT / "outputs" / "tables" / "phase1_significance.csv"


def main():
    if not SUMMARY_CSV.exists():
        print(f"ERROR: {SUMMARY_CSV} not found. Run compile_phase1_results.py first.")
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} runs from {SUMMARY_CSV}\n")

    results = []
    comparisons = [
        ("mhc", "hc", "mHC vs HC"),
        ("mhc", "vanilla", "mHC vs Vanilla"),
    ]

    for dataset in ["weather", "ett"]:
        ds_df = df[df["dataset"] == dataset]
        depths = sorted(ds_df["depth"].unique())

        for depth in depths:
            depth_df = ds_df[ds_df["depth"] == depth]

            for model_a, model_b, label in comparisons:
                a_df = depth_df[depth_df["model"] == model_a]
                b_df = depth_df[depth_df["model"] == model_b]

                if len(a_df) < 2 or len(b_df) < 2:
                    print(f"  SKIP {dataset} L={depth} {label}: "
                          f"need >=2 seeds each, got {len(a_df)}/{len(b_df)}")
                    continue

                # Align by seed
                seeds = sorted(set(a_df["seed"]) & set(b_df["seed"]))
                if len(seeds) < 2:
                    print(f"  SKIP {dataset} L={depth} {label}: "
                          f"only {len(seeds)} common seeds")
                    continue

                a_mse = np.array([a_df[a_df["seed"] == s]["best_val_mse"].values[0]
                                  for s in seeds])
                b_mse = np.array([b_df[b_df["seed"] == s]["best_val_mse"].values[0]
                                  for s in seeds])

                # Also extract MAE for secondary comparison
                a_mae = np.array([a_df[a_df["seed"] == s]["best_val_mae"].values[0]
                                  for s in seeds])
                b_mae = np.array([b_df[b_df["seed"] == s]["best_val_mae"].values[0]
                                  for s in seeds])

                # Run significance test on MSE
                sig_mse = paired_significance(a_mse, b_mse)

                # Run significance test on MAE
                sig_mae = paired_significance(a_mae, b_mae)

                result_row = {
                    "dataset": dataset,
                    "depth": depth,
                    "comparison": label,
                    "n": len(seeds),
                    "seeds": str(seeds),
                    # MSE results
                    "mse_mean_a": sig_mse.mean_a,
                    "mse_mean_b": sig_mse.mean_b,
                    "mse_std_a": sig_mse.std_a,
                    "mse_std_b": sig_mse.std_b,
                    "mse_diff": sig_mse.mean_diff,
                    "mse_ci_lo": sig_mse.ci_95_lo,
                    "mse_ci_hi": sig_mse.ci_95_hi,
                    "mse_p_ttest": sig_mse.p_value_t,
                    "mse_p_wilcoxon": sig_mse.p_value_wilcoxon,
                    "mse_cohens_d": sig_mse.cohens_d,
                    "mse_effect": sig_mse.effect_size,
                    "mse_significant": sig_mse.significant_at_005,
                    # MAE results
                    "mae_mean_a": sig_mae.mean_a,
                    "mae_mean_b": sig_mae.mean_b,
                    "mae_diff": sig_mae.mean_diff,
                    "mae_p_ttest": sig_mae.p_value_t,
                    "mae_p_wilcoxon": sig_mae.p_value_wilcoxon,
                    "mae_cohens_d": sig_mae.cohens_d,
                    "mae_effect": sig_mae.effect_size,
                    "mae_significant": sig_mae.significant_at_005,
                }
                results.append(result_row)

                sig_mark = "***" if sig_mse.significant_at_005 else ""
                print(
                    f"  {dataset:8s} L={depth:2d} {label:15s} | "
                    f"MSE: {sig_mse.mean_a:.4f} vs {sig_mse.mean_b:.4f} "
                    f"(diff={sig_mse.mean_diff:+.4f}, d={sig_mse.cohens_d:.2f} "
                    f"{sig_mse.effect_size}) p_t={sig_mse.p_value_t:.4f} "
                    f"p_w={sig_mse.p_value_wilcoxon:.4f} {sig_mark}"
                )

    # Write results
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote: {OUTPUT_CSV} ({len(out_df)} comparisons)")

    # Summary
    sig_count = sum(1 for r in results if r["mse_significant"])
    print(f"\nSignificant MSE comparisons: {sig_count}/{len(results)}")
    for r in results:
        if r["mse_significant"]:
            winner = r["comparison"].split(" vs ")[0] if r["mse_diff"] < 0 else r["comparison"].split(" vs ")[1]
            print(
                f"  {r['dataset']:8s} L={r['depth']:2d} | {r['comparison']:15s} | "
                f"Winner: {winner} (d={r['mse_cohens_d']:.2f}, p={r['mse_p_ttest']:.4f})"
            )


if __name__ == "__main__":
    main()
