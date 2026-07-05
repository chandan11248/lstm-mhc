#!/usr/bin/env python3
"""Generate Phase 1 Kaggle notebooks: weather L=4, 3 models × 5 seeds.

Each notebook is self-contained (all lstm_mhc code inlined) and runs one
training job on real NOAA weather data with the P100 GPU.

Output: kaggle/phase1/{model}_l4_s{seed}.ipynb + kernel-metadata.json
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Notebook helpers ──────────────────────────────────────────────────

def code_cell(source: str) -> dict:
    lines = source.strip().split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}

def md_cell(source: str) -> dict:
    lines = source.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}

def make_notebook(cells: list) -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                         "language": "python", "name": "python3"},
                         "language_info": {"name": "python",
                         "version": "3.10.12"},
                         "accelerator": "GPU",
                         "gpu": True},
            "cells": cells}


# ── Inline source code ────────────────────────────────────────────────
# Read all .py files from the lstm_mhc package and embed them.

def collect_source_files() -> dict[str, str]:
    """Return {module_path: source_code} for all .py files in lstm_mhc/."""
    sources = {}
    pkg = ROOT / "lstm_mhc"
    for py in sorted(pkg.rglob("*.py")):
        rel = py.relative_to(ROOT)
        sources[str(rel)] = py.read_text()
    return sources


def inline_setup_cell(sources: dict[str, str]) -> str:
    """Generate a cell that writes all source files to /kaggle/working/.

    Uses base64 encoding to avoid all string-escaping issues with triple-quoted
    strings (which previously broke f-strings containing \\n and other escape
    sequences in the inlined source code).
    """
    import base64
    lines = [
        "# === Install dependencies (pin PyTorch 2.3 for P100 sm_60 support) ===",
        "!pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 -q",
        "!pip install pyyaml numpy pandas matplotlib seaborn scikit-learn -q",
        "",
        "# === Write lstm_mhc package to disk (base64-encoded to preserve source) ===",
        "import os, base64",
        "",
        "_FILES = {",
    ]
    for path, code in sources.items():
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        lines.append(f'    "{path}":')
        # Split base64 into 200-char chunks for readability
        for i in range(0, len(encoded), 200):
            chunk = encoded[i:i+200]
            lines.append(f'        "{chunk}"')
        lines.append("        ,")
    lines.append("}")
    lines.append("")
    lines.append("for _path, _b64 in _FILES.items():")
    lines.append("    _full = os.path.join('/kaggle/working', _path)")
    lines.append("    os.makedirs(os.path.dirname(_full), exist_ok=True)")
    lines.append("    with open(_full, 'w') as f:")
    lines.append("        f.write(base64.b64decode(_b64).decode('utf-8'))")
    lines.append("")
    lines.append("import sys")
    lines.append("sys.path.insert(0, '/kaggle/working')")
    lines.append("print('✅ Package written to /kaggle/working/lstm_mhc')")
    return "\n".join(lines)


# ── Training cell ──────────────────────────────────────────────────────

def training_cell(model_type: str, depth: int, seed: int, dataset: str = "weather") -> str:
    if dataset == "ett":
        data_block = """
# Data — ETT (auto-downloads ETTh1.csv)
from lstm_mhc.data.ett_dataset import build_ett_dataloaders
train_loader, val_loader, test_loader, scaler = build_ett_dataloaders(config, csv_name="ETTh1.csv")
print("Using dataset: ETT (ETTh1)")
"""
    else:
        data_block = """
# Data — NOAA Weather
from lstm_mhc.data.weather_dataset import build_weather_dataloaders
csv_candidates = [
    "/kaggle/input/datasets/zhaodianwen/noaaweatherdatajfkairport/noaa-weather-data-jfk-airport/jfk_weather_cleaned.csv",
    "/kaggle/input/noaa-weather-data-jfk-airport/noaa-weather-data-jfk-airport/jfk_weather_cleaned.csv",
    "/kaggle/input/noaa-weather-data-jfk-airport/jfk_weather_cleaned.csv",
    "/kaggle/input/noaaweatherdatajfkairport/jfk_weather_cleaned.csv",
    "/kaggle/working/data/jfk_weather_cleaned.csv",
]
csv_path = None
for candidate in csv_candidates:
    if os.path.exists(candidate):
        csv_path = candidate
        break
if csv_path is None:
    import glob
    found = glob.glob("/kaggle/input/**/jfk_weather_cleaned.csv", recursive=True)
    if found:
        csv_path = found[0]
    else:
        raise FileNotFoundError(
            "jfk_weather_cleaned.csv not found. Checked: " + str(csv_candidates)
            + " | Available: " + str(os.listdir("/kaggle/input/") if os.path.exists("/kaggle/input/") else "no /kaggle/input/")
        )
print(f"Using data: {{csv_path}}")
train_loader, val_loader, test_loader, scaler = build_weather_dataloaders(csv_path, config)
"""

    return f"""
# === Training: {model_type} L={depth} seed={seed} dataset={dataset} ===
import os
import torch
import sys
sys.path.insert(0, '/kaggle/working')

from lstm_mhc.utils.config import ExperimentConfig
from lstm_mhc.utils.seed import seed_everything
from lstm_mhc.models.vanilla_lstm import StandardLSTM
from lstm_mhc.models.hc_lstm import HCLSTM
from lstm_mhc.models.mhc_lstm import MHCLSTM
from lstm_mhc.models.micro_mhc_lstm import MicroMHCLSTM
from lstm_mhc.models.sota import build_sota, SOTA_MAP
from lstm_mhc.training.trainer import run_training
from lstm_mhc.training.metrics_logger import MetricsLogger

# Config
config = ExperimentConfig(
    model_type="{model_type}",
    dataset="{dataset}",
    num_layers={depth},
    seed={seed},
    num_epochs=50,
    batch_size=32,
    output_dir="/kaggle/working/outputs",
    kaggle_user=os.environ.get("KAGGLE_USERNAME", "unknown"),
    save_every_n_epochs=5,
    resume=True,
)

print(f"Run: {{config.run_name}}")
print(f"Config hash: {{config.config_hash}}")

# Seed
seed_everything({seed})

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}")
if torch.cuda.is_available():
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}} GB")
{data_block}

# Model
MODEL_MAP = {{"vanilla": StandardLSTM, "hc": HCLSTM, "mhc": MHCLSTM, "micro_mhc": MicroMHCLSTM}}
if "{model_type}" in SOTA_MAP:
    model = build_sota("{model_type}", config)
else:
    model = MODEL_MAP["{model_type}"](config)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {{params:,}}")

# Logger
logger = MetricsLogger(config.run_name, str(config.output_path / "logs"))

# Train
result = run_training(
    model, train_loader, val_loader, config, logger, device,
    scaler_info=scaler, test_loader=test_loader,
)

# Print final results
print("\\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
for k, v in result.items():
    print(f"  {{k}}: {{v}}")

# Save config
config.dump()

print(f"\\n✅ Run complete: {{config.run_name}}")
""".strip()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--models", type=str, nargs="+", default=["vanilla", "hc", "mhc"])
    parser.add_argument("--dataset", type=str, default="weather", choices=["weather", "ett"])
    parser.add_argument("--outdir", type=str, default="kaggle/phase1")
    args = parser.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sources = collect_source_files()
    print(f"Collected {len(sources)} source files from lstm_mhc/")

    ds_tag = args.dataset
    ds_label = "NOAA weather" if ds_tag == "weather" else "ETT (ETTh1)"
    # ETT auto-downloads, so no Kaggle dataset source needed
    dataset_sources = ["zhaodianwen/noaaweatherdatajfkairport"] if ds_tag == "weather" else []

    generated = []
    for model_type in args.models:
        for seed in args.seeds:
            run_name = f"{model_type}_l{args.depth}_s{seed}_{ds_tag}"
            cells = [
                md_cell(f"# LSTM-µHC Phase 1: {model_type.upper()} L={args.depth} seed={seed}\n"
                        f"{ds_label} data, leakage-free splits, de-normalized metrics."),
                code_cell("import os\nos.environ['KAGGLE_USERNAME'] = os.environ.get('KAGGLE_USERNAME', 'unknown')"),
                code_cell(inline_setup_cell(sources)),
                code_cell(f"import os\nos.environ.get('KAGGLE_USERNAME', 'unknown')"),
                code_cell(training_cell(model_type, args.depth, seed, dataset=ds_tag)),
            ]
            nb = make_notebook(cells)
            nb_path = outdir / f"{run_name}.ipynb"
            with open(nb_path, "w") as f:
                json.dump(nb, f, indent=1)

            # kernel-metadata.json for Kaggle API push
            meta = {
                "id": f"USERNAME/{run_name}",
                "title": f"LSTM-uHC {model_type} L={args.depth} s{seed} {ds_tag}",
                "code_file": f"{run_name}.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": dataset_sources,
                "kernel_sources": [],
                "model_sources": [],
            }
            meta_path = outdir / f"{run_name}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            generated.append((run_name, nb_path, meta_path))
            print(f"  ✅ {run_name}.ipynb")

    print(f"\nGenerated {len(generated)} notebooks in {outdir}/")
    print(f"\nNext: dispatch to Kaggle accounts using scripts/push_phase1.py")


if __name__ == "__main__":
    main()
