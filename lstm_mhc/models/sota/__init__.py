"""State-of-the-art transformer forecasting baselines.

Faithful, compact implementations of recent long-term-forecasting models,
adapted to the LSTM-µHC interface so they train under the *identical* protocol
(same splits, scaler, horizons, optimizer, seeds, checkpoint selection):

    forward(x) -> (pred, None)
    x:    (B, T, F)            input window
    pred: (B, num_horizons, F) multi-horizon point forecast

These are re-trained on our data; no numbers are copied from their papers.

Models:
    - iTransformer (Liu et al., ICLR 2024)
    - PatchTST     (Nie et al., ICLR 2023)
    - TimesNet     (Wu et al., ICLR 2023)   [added separately]
"""

from .itransformer import ITransformer
from .patchtst import PatchTST
from .timesnet import TimesNet

SOTA_MAP = {
    "itransformer": ITransformer,
    "patchtst": PatchTST,
    "timesnet": TimesNet,
}


def build_sota(name: str, config):
    if name not in SOTA_MAP:
        raise ValueError(f"Unknown SOTA model: {name}. Available: {list(SOTA_MAP)}")
    return SOTA_MAP[name](config)
