"""Experiment configuration dataclasses and YAML (de)serialization.

A single ``ExperimentConfig`` fully describes one training run. It is the unit
of reproducibility: it is (a) dumped to ``{output_dir}/{run_name}_config.yaml``
next to every checkpoint so any number in a results table can be traced back to
the exact config that produced it, and (b) stored inside every checkpoint.

Design notes
------------
- All tunables live here as dataclass fields — no magic numbers in the trainer.
- Numerical-stability epsilons default to safe values (RMSNorm ``1e-6``,
  AdamW ``1e-8``) rather than the DeepSeek LLM-pretraining values (``1e-20``)
  which are unsafe for small RNNs / mixed precision. The paper values can be
  re-enabled via YAML for an exact replication study.
- Fairness knobs (``dropout``, gradient clipping per model) are first-class so
  Models A/B/C are compared under identical regularization.
- Ablation switches (``use_pre / use_post / use_res``) enable the mHC paper's
  Table-1 head-ablation directly from config.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class ExperimentConfig:
    """Configuration for a single training run."""

    # ===== Identity / provenance =====
    model_type: str = "mhc"               # "vanilla" | "hc" | "mhc" | baseline name
    dataset: str = "weather"              # "weather" | "ett"
    ett_file: str = "ETTh1.csv"           # used only when dataset == "ett"
    kaggle_user: str = "local"            # which Kaggle account ran this
    seed: int = 42

    # ===== Architecture =====
    num_layers: int = 4                   # LSTM depth L (4, 8, 16, ...)
    n_streams: int = 4                    # number of parallel streams (n)
    hidden_dim: int = 128                 # per-stream feature dim (d) for B/C
    input_dim: int = 4                    # number of input variables
    sinkhorn_iterations: int = 20         # Sinkhorn-Knopp iterations (t_max)
    alpha_init: float = 0.01              # head gating-factor init

    # Parameter matching for Model A (vanilla). When ``match_params=True`` the
    # vanilla hidden dim is solved so Model A's *recurrent* parameter count is
    # within ``match_tolerance`` of Model C's. Otherwise ``vanilla_hidden_dim``
    # (if set) or ``n_streams * hidden_dim`` is used.
    match_params: bool = True
    match_tolerance: float = 0.05
    vanilla_hidden_dim: Optional[int] = None

    # mHC head ablation switches (paper Table 1). Only affect hc/mhc models.
    use_pre: bool = True
    use_post: bool = True
    use_res: bool = True

    # ===== Numerical stability =====
    rmsnorm_eps: float = 1e-6             # was 1e-20 (unsafe); see module docstring
    sinkhorn_clamp: float = 10.0
    sinkhorn_norm_eps: float = 1e-8

    # ===== Training =====
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 4e-4
    warmup_epochs: int = 5                # linear warmup before cosine decay
    min_lr_ratio: float = 0.0             # cosine floor (0 = decay to 0)
    weight_decay: float = 0.1
    grad_clip: float = 1.0                # global L2 grad-norm clip (all models)
    dropout: float = 0.1                  # uniform dropout across all models
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8               # was 1e-20 (unsafe)
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # ===== Data =====
    input_len: int = 724                  # input window length (T)
    horizons: List[int] = field(default_factory=lambda: [6, 12, 24, 48, 72])
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    window_stride: int = 1                # sliding-window stride (train)
    split_gap: int = -1                   # guard gap between splits; -1 => max(horizons)
    num_workers: int = 0
    weather_columns: List[str] = field(default_factory=lambda: [
        "temperature", "humidity", "pressure", "wind_speed",
    ])

    # ===== Evaluation / metrics =====
    # MSE/MAE/etc. are always computed in *original* (de-normalized) units when
    # a scaler is provided (this is the correct, comparable default). The
    # previous `denormalize_metrics: bool = True` config flag was a no-op
    # (never read anywhere); removed (claude_checked.md H6).

    # ===== Logging =====
    log_amax_every: int = 100
    save_every_n_epochs: int = 0          # 0 = only best + last; >0 = also per-N epoch ckpt

    # ===== Output =====
    output_dir: str = "outputs"

    # ===== Runtime (not part of config equality) =====
    resume: bool = False

    def __post_init__(self):
        self.output_path = Path(self.output_dir)
        (self.output_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_path / "logs").mkdir(parents=True, exist_ok=True)
        # Resolve gap default to the forecasting horizon.
        if self.split_gap < 0:
            object.__setattr__(self, "split_gap", int(max(self.horizons)))
        # Validate ratios.
        if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) > 1e-3:
            raise ValueError(
                f"train+val+test ratios must sum to 1.0, got "
                f"{self.train_ratio + self.val_ratio + self.test_ratio}"
            )
        # Resolve Model A hidden dim.
        if self.model_type == "vanilla" and self.vanilla_hidden_dim is None:
            object.__setattr__(self, "vanilla_hidden_dim", self._solve_vanilla_hidden_dim())

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def num_horizons(self) -> int:
        return len(self.horizons)

    @property
    def run_name(self) -> str:
        """Stable, descriptive run identifier encoding the salient knobs.

        Includes model, depth, seed, dataset, and (for vanilla) the matched
        hidden dim — enough to disambiguate any two runs in a results table.
        """
        parts = [self.model_type, f"l{self.num_layers}", f"s{self.seed}",
                 self.dataset]
        if self.model_type == "vanilla":
            vh = self.vanilla_hidden_dim or (self.n_streams * self.hidden_dim)
            parts.append(f"h{vh}")
        if not self.use_pre or not self.use_post or not self.use_res:
            tag = []
            if not self.use_pre:
                tag.append("nopre")
            if not self.use_post:
                tag.append("nopost")
            if not self.use_res:
                tag.append("nores")
            parts.append("-".join(tag))
        return "_".join(str(p) for p in parts)

    @property
    def config_hash(self) -> str:
        """Short hash of the config (excluding output_dir / resume) for provenance."""
        d = self.to_dict()
        for k in ("output_dir", "output_path", "resume"):
            d.pop(k, None)
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:10]

    # ------------------------------------------------------------------
    # Model A parameter matching
    # ------------------------------------------------------------------
    def _solve_vanilla_hidden_dim(self) -> int:
        """Pick a vanilla-LSTM hidden dim whose *total* params match Model C.

        Delegates to :func:`lstm_mhc.utils.matching.match_vanilla`, which does
        an exact integer scan against the closed-form parameter counts of
        StandardLSTM vs MHCLSTM (a coarse quadratic approximation was off by
        60-90% in practice). If ``match_params`` is False, falls back to
        ``n_streams * hidden_dim``.
        """
        # Local import avoids a circular import (matching -> config).
        from .matching import match_vanilla
        return match_vanilla(self)

    # ------------------------------------------------------------------
    # (De)serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output_path"] = str(self.output_path)
        return d

    def dump(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Write this config as YAML next to outputs (or to ``path``)."""
        if path is None:
            path = self.output_path / f"{self.run_name}_config.yaml"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        d["config_hash"] = self.config_hash  # computed after to_dict to avoid recursion
        with open(path, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False, default_flow_style=False)
        return path

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        data.pop("output_path", None)
        data.pop("config_hash", None)
        return cls._from_mapping(data)

    @classmethod
    def _from_mapping(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Construct, warning on unknown keys (catches YAML typos).

        Previously unknown keys were silently dropped, which meant a typo
        like ``learnin_rate: 1e-3`` was indistinguishable from a successful
        load. We now emit a `warnings.warn` listing every dropped key.
        """
        import warnings
        valid = {f.name for f in fields(cls)}
        unknown = [k for k in data if k not in valid]
        if unknown:
            warnings.warn(
                f"ExperimentConfig: ignoring unknown key(s): {unknown}. "
                f"Valid keys: {sorted(valid)}",
                UserWarning,
                stacklevel=2,
            )
        clean = {k: v for k, v in data.items() if k in valid}
        return cls(**clean)
