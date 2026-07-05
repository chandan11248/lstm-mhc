#!/usr/bin/env python3
"""Generate LaTeX (booktabs) tables from MEASURED test + stability data.

Outputs -> paper/tables/*.tex  (each \\input-able from main.tex)
  - tab_main_results.tex     : test MSE/MAE/RMSE per dataset x depth x model
  - tab_significance.tex      : paired significance (mHC vs HC / Vanilla) on test
  - tab_stability.tex         : max grad norm + spectral norm per depth x model
  - tab_depth_scaling.tex     : compact test-MSE vs depth (incl. L=32 + ETTm1)

No fabrication: diverged/missing entries render as "diverged" / "--".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = {"mhc": "mHC-LSTM (ours)", "vanilla": "Standard LSTM", "hc": "HC-LSTM",
              "gru": "GRU", "tcn": "TCN", "ridge": "Ridge", "dlinear": "DLinear", "nlinear": "NLinear",
              "patchtst": "PatchTST", "itransformer": "iTransformer", "timesnet": "TimesNet"}
DS_NAME = {"weather": "Weather", "ett": "ETTh1", "ettm1": "ETTm1"}


def ms(mean, std, d=2):
    if mean is None or not np.isfinite(mean):
        return "--"
    if std is None or not np.isfinite(std) or std == 0:
        return f"{mean:.{d}f}"
    return f"{mean:.{d}f}\\,$\\pm$\\,{std:.{d}f}"


def main_results():
    s = pd.read_csv(TABLES / "test_metrics_final_summary.csv")
    order = ["mhc", "vanilla", "hc", "gru", "tcn", "ridge", "dlinear", "nlinear"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Held-out \textbf{test}-set forecasting error (original units, mean$\pm$std over seeds). "
        r"Lower is better. HC diverges (NaN) at $L{=}16$; vanilla collapses at $L{\geq}16$.}",
        r"\label{tab:main}", r"\small", r"\begin{tabular}{llrrrc}", r"\toprule",
        r"Dataset & Model & $L$ & Test MSE & Test MAE & $n$ \\", r"\midrule",
    ]
    for ds in ["weather", "ett", "ettm1"]:
        sd = s[s.dataset == ds]
        if sd.empty:
            continue
        sd = sd.copy()
        sd["ord"] = sd["model"].map({m: i for i, m in enumerate(order)}).fillna(99)
        sd = sd.sort_values(["depth", "ord"])
        for _, r in sd.iterrows():
            lines.append(
                f"{DS_NAME.get(ds, ds)} & {MODEL_NAME.get(r['model'], r['model'])} & "
                f"{int(r['depth'])} & {ms(r['test_mse_mean'], r['test_mse_std'])} & "
                f"{ms(r['test_mae_mean'], r['test_mae_std'], 3)} & {int(r['n_seeds'])} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_main_results.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_main_results.tex")


def significance():
    sig = pd.read_csv(TABLES / "test_significance.csv")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Paired significance on \textbf{test} MSE for matched five-seed comparisons: "
        r"Cohen's $d$ and two-sided $t$-test $p$. Negative diff favors mHC.}",
        r"\label{tab:sig}", r"\small", r"\begin{tabular}{llrrrc}", r"\toprule",
        r"Dataset & Comparison & $L$ & $\Delta$MSE & Cohen's $d$ & $p$ \\", r"\midrule",
    ]
    for _, r in sig.iterrows():
        p = r["mse_p_ttest"]
        pstr = f"{p:.3f}" + (r"$^{*}$" if r["mse_significant"] else "")
        lines.append(
            f"{DS_NAME.get(r['dataset'], r['dataset'])} & {r['comparison']} & {int(r['depth'])} & "
            f"{r['mse_diff']:+.2f} & {r['mse_cohens_d']:.2f} & {pstr} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
              r"\footnotesize $^{*}$ significant at $p<0.05$.", r"\end{table}"]
    (OUT / "tab_significance.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_significance.tex")


def stability():
    st = pd.read_csv(TABLES / "phase1_stability.csv")
    g = st.groupby(["dataset", "depth", "model"]).agg(
        grad=("max_grad_norm", "max"), spec=("max_spectral_norm", "max")).reset_index()
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Training stability (max over all steps/seeds). mHC holds the "
        r"composite spectral norm near $1.0$ at every depth; unconstrained HC explodes.}",
        r"\label{tab:stability}", r"\small", r"\begin{tabular}{llrrr}", r"\toprule",
        r"Dataset & Model & $L$ & Max grad norm & Max spec. norm \\", r"\midrule",
    ]
    for ds in ["weather", "ett"]:
        for depth in [4, 8, 16]:
            for model in ["mhc", "hc", "vanilla"]:
                r = g[(g.dataset == ds) & (g.depth == depth) & (g.model == model)]
                if not len(r):
                    if model == "hc" and depth == 16:
                        lines.append(f"{DS_NAME[ds]} & HC-LSTM & 16 & \\multicolumn{{2}}{{c}}{{diverged}} \\\\")
                    continue
                gv = r["grad"].iloc[0]; sv = r["spec"].iloc[0]
                gstr = f"{gv:.2e}" if np.isfinite(gv) else "--"
                sstr = f"{sv:.4f}" if np.isfinite(sv) else "--"
                lines.append(f"{DS_NAME[ds]} & {MODEL_NAME.get(model, model)} & {depth} & {gstr} & {sstr} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_stability.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_stability.tex")


def depth_scaling():
    s = pd.read_csv(TABLES / "test_metrics_final_summary.csv")
    depths = [4, 8, 16, 32]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Test MSE vs.\ depth. mHC remains stable across the evaluated depth range; "
        r"Standard LSTM and TCN collapse at depth.}",
        r"\label{tab:depth}", r"\small", r"\begin{tabular}{ll" + "r" * len(depths) + "}", r"\toprule",
        "Dataset & Model & " + " & ".join(f"$L{{=}}{d}$" for d in depths) + r" \\", r"\midrule",
    ]
    for ds in ["weather", "ett"]:
        for model in ["mhc", "vanilla", "gru", "tcn"]:
            cells = []
            any_val = False
            for d in depths:
                r = s[(s.dataset == ds) & (s.model == model) & (s.depth == d)]
                if len(r):
                    cells.append(f"{r['test_mse_mean'].iloc[0]:.1f}")
                    any_val = True
                else:
                    cells.append("--")
            if any_val:
                lines.append(f"{DS_NAME[ds]} & {MODEL_NAME.get(model, model)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_depth_scaling.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_depth_scaling.tex")


def efficiency():
    """Efficiency table: params + GFLOPs (measured) + peak VRAM + train time/epoch
    + CPU inference latency. Weather, core models across depth."""
    eff_p = TABLES / "efficiency.csv"
    agg_p = TABLES / "phase1_aggregate.csv"
    lat_p = TABLES / "gpu_latency.csv"
    if not eff_p.exists() or not agg_p.exists():
        print("skip efficiency (missing efficiency.csv or phase1_aggregate.csv)")
        return
    eff = pd.read_csv(eff_p)
    agg = pd.read_csv(agg_p)
    aggw = agg[agg.dataset == "weather"]
    lat = pd.read_csv(lat_p) if lat_p.exists() else None
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Compute efficiency (weather). Params and FLOPs measured on a "
        r"$1{\times}724{\times}4$ input; peak VRAM and train time from training logs; "
        r"inference latency on CPU (batch=32, 100 runs, mean). "
        r"mHC is parameter- and FLOP-matched to the standard LSTM, at higher "
        r"activation memory and CPU latency from the $n$-stream expansion.}",
        r"\label{tab:efficiency}", r"\footnotesize", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Model & $L$ & Params & GFLOPs & VRAM (MB) & Train/epoch (s) & Latency (ms) \\", r"\midrule",
    ]
    for model in ["mhc", "vanilla", "hc", "gru", "tcn", "patchtst", "itransformer", "timesnet", "dlinear", "nlinear"]:
        for depth in [4]:
            e = eff[(eff.model == model) & (eff.depth == depth)]
            if not len(e):
                continue
            a = aggw[(aggw.model == model) & (aggw.depth == depth)]
            vram = f"{a['peak_vram_mb_max'].iloc[0]:.0f}" if len(a) and pd.notna(a['peak_vram_mb_max'].iloc[0]) else "--"
            t = f"{a['avg_epoch_time_s_mean'].iloc[0]:.1f}" if len(a) and pd.notna(a['avg_epoch_time_s_mean'].iloc[0]) else "--"
            if lat is not None:
                lrow = lat[(lat.model == model) & (lat.depth == depth)]
                latency = f"{lrow['latency_mean_ms'].iloc[0]:.1f}" if len(lrow) else "--"
            else:
                latency = "--"
            lines.append(f"{MODEL_NAME.get(model, model)} & {depth} & "
                         f"{int(e['params'].iloc[0]):,} & {e['gflops'].iloc[0]:.3f} & {vram} & {t} & {latency} \\\\")
    lines.append(r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    (OUT / "tab_efficiency.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_efficiency.tex")


def sota_comparison():
    """Shallow comparison vs modern SOTA forecasters (test MSE/MAE, 5 seeds)."""
    fs = pd.read_csv(TABLES / "test_metrics_final_summary.csv")
    sota = pd.read_csv(TABLES / "sota_test_offline.csv")
    sota_ok = sota[sota.status == "ok"]
    sa = sota_ok.groupby(["dataset", "model"]).agg(
        test_mse_mean=("test_mse", "mean"), test_mse_std=("test_mse", "std"),
        test_mae_mean=("test_mae", "mean"), test_mae_std=("test_mae", "std"),
        n=("seed", "nunique")).reset_index()

    def get(ds, model):
        r = sa[(sa.dataset == ds) & (sa.model == model)]
        if len(r):
            return (r.test_mse_mean.iloc[0], r.test_mse_std.iloc[0],
                    r.test_mae_mean.iloc[0], r.test_mae_std.iloc[0])
        r = fs[(fs.dataset == ds) & (fs.model == model) & (fs.depth == 4)]
        if len(r):
            return (r.test_mse_mean.iloc[0], r.test_mse_std.iloc[0],
                    r.test_mae_mean.iloc[0], r.test_mae_std.iloc[0])
        return None

    rows_models = ["nlinear", "dlinear", "ridge", "tcn", "gru", "patchtst", "itransformer", "timesnet", "mhc"]
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Comparison against modern forecasters (held-out \textbf{test} MSE/MAE, "
        r"mean$\pm$std over five seeds). Our LSTM models are at $L{=}4$; "
        r"transformer baselines use their own depth. mHC is best on weather; "
        r"PatchTST/NLinear lead on ETTh1.}",
        r"\label{tab:sota}", r"\footnotesize", r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Model & ETTh1 MSE & ETTh1 MAE & Weather MSE & Weather MAE \\", r"\midrule",
    ]
    for m in rows_models:
        e = get("ett", m)
        w = get("weather", m)
        if e is None and w is None:
            continue
        ev = ms(e[0], e[1]) + " & " + ms(e[2], e[3], 3) if e else "-- & --"
        wv = ms(w[0], w[1]) + " & " + ms(w[2], w[3], 3) if w else "-- & --"
        name = MODEL_NAME.get(m, m)
        if m == "mhc":
            name = r"\textbf{" + name + "}"
        lines.append(f"{name} & {ev} & {wv} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "tab_sota.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_sota.tex")


def sota_significance():
    """Paired significance: mHC (L=4) vs PatchTST and iTransformer on test MSE."""
    import sys
    sys.path.insert(0, str(ROOT))
    from lstm_mhc.utils.stats import paired_significance

    per = pd.read_csv(TABLES / "test_metrics_all_per_run.csv")
    sota = pd.read_csv(TABLES / "sota_test_offline.csv")
    mhc = per[(per.model == "mhc") & (per.depth == 4) & (per.status == "ok")]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Paired significance of mHC ($L{=}4$) vs.\ SOTA on \textbf{test} MSE "
        r"(five matched seeds). Negative diff favors mHC.}",
        r"\label{tab:sotasig}", r"\footnotesize", r"\begin{tabular}{llrrc}", r"\toprule",
        r"Dataset & vs.\ & $\Delta$MSE & Cohen's $d$ & $p$ \\", r"\midrule",
    ]
    for ds in ["weather", "ett"]:
        a = mhc[mhc.dataset == ds]
        for other in ["patchtst", "itransformer", "timesnet"]:
            b = sota[(sota.dataset == ds) & (sota.model == other) & (sota.status == "ok")]
            seeds = sorted(set(a.seed) & set(b.seed))
            if len(seeds) < 2:
                continue
            av = np.array([a[a.seed == s]["test_mse"].values[0] for s in seeds])
            bv = np.array([b[b.seed == s]["test_mse"].values[0] for s in seeds])
            sg = paired_significance(av, bv)
            star = r"$^{*}$" if sg.significant_at_005 else ""
            lines.append(f"{DS_NAME.get(ds, ds)} & {MODEL_NAME.get(other, other)} & "
                         f"{sg.mean_diff:+.2f} & {sg.cohens_d:.2f} & {sg.p_value_t:.3f}{star} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
              r"\footnotesize $^{*}$ significant at $p<0.05$.", r"\end{table}"]
    (OUT / "tab_sota_sig.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_sota_sig.tex")


if __name__ == "__main__":
    main_results()
    significance()
    stability()
    depth_scaling()
    efficiency()
    sota_comparison()
    sota_significance()
    # Post-process: make all tables full-width floats (table*) at footnotesize so
    # they don't overflow the narrow IEEE two-column width.
    for f in OUT.glob("tab_*.tex"):
        t = f.read_text()
        t = t.replace(r"\begin{table}[t]", r"\begin{table*}[t]")
        t = t.replace(r"\end{table}", r"\end{table*}")
        t = t.replace(r"\small", r"\footnotesize")
        f.write_text(t)
    print(f"\nLaTeX tables (full-width) in {OUT}")
