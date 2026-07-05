"""Pytest config: add project root to sys.path so ``import lstm_mhc`` works.

The active notebook generator (``scripts/generate_phase1_notebooks.py``)
inlines the package source into the notebooks at generation time, so the
notebooks are self-contained on Kaggle and don't need this path fix.
But local test runs and the trainer need ``import lstm_mhc`` to resolve
from the project root.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
