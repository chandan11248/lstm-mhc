# LSTM-µHC: Manifold-Constrained Hyper-Connections for Stable Depth Scaling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of **LSTM-µHC**, a macro-level manifold-constrained Hyper-Connection highway for stacked LSTMs in multivariate time-series forecasting.

## Overview

Deep stacked LSTMs suffer from a depth ceiling: gradients vanish or explode and deep variants collapse to near-constant predictions. Hyper-Connections (HC) widen the residual stream with learnable mixing matrices but are unstable. We adapt **manifold-constrained Hyper-Connections (mHC)** to recurrent forecasting, constraining the residual mixing matrix to the Birkhoff polytope (doubly stochastic) via the Sinkhorn–Knopp algorithm. The highway is placed **between LSTM layers** (macro), keeping the cuDNN-optimized sequence kernel intact.

Key result: mHC preserves stable test error and bounded spectral norms across depths $L \in \{4, 8, 16, 32\}$, while Standard LSTM and TCN collapse at depth and unconstrained HC diverges.

## Installation

```bash
git clone <repo-url>
cd lstm-mhc
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Run a single training run (weather, L=4, mHC):

```bash
python scripts/run_experiment.py \
  --model mhc \
  --depth 4 \
  --dataset weather \
  --seed 42 \
  --output-dir outputs/mhc_l4_s42
```

Run the full test suite:

```bash
pytest tests/test_core.py -v
```

## Reproducing the paper

1. **Data**: the NOAA JFK weather CSV and ETTh1/ETTm1 are downloaded automatically by the data loaders on first run.
2. **Phase 1 (main results)**: use `scripts/generate_phase1_notebooks.py` to generate Kaggle notebooks, or run `scripts/run_experiment.py` locally.
3. **Tables/figures**: after training, run `scripts/make_latex_tables.py` and `scripts/make_final_plots.py` from the measured logs.
4. **Paper**: the camera-ready LaTeX source is in `paper/`, including generated tables and final plots.

See `LSTM-µHC-Architecture.md` and `µHC-Concept-Breakdown.md` for the architectural and mathematical details.

## Project structure

```
.
├── lstm_mhc/          # Core implementation
│   ├── models/        # Vanilla LSTM, HC-LSTM, mHC-LSTM, baselines
│   ├── data/          # Weather and ETT loaders
│   ├── training/      # Trainer and metrics logger
│   ├── evaluation/    # Forecast and stability metrics
│   ├── utils/         # Config, seeding, parameter matching, stats
│   └── visualization/ # Plotting utilities
├── tests/             # Unit tests
├── scripts/           # Reproduction scripts
├── configs/           # YAML experiment configs
├── paper/             # LaTeX paper source and PDF
└── data/              # Data loaders (datasets auto-download)
```

## Citation

If you use this code, please cite:

```bibtex
@article{shah2026lstmumhc,
  title={Manifold-Constrained Hyper-Connections for Stable Depth Scaling in LSTM Time-Series Forecasting},
  author={Shah, Chandan Kumar and Shrestha, Abhaya and Khanal, Asmit and Regmi, Kushal and Thapa, Nabina},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds on mHC (Xie et al., arXiv:2512.24880) and Hyper-Connections (Zhu et al., ICLR 2025).
