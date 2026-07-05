"""Reproducibility utilities — seed everything."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility.

    Notes
    -----
    - ``PYTHONHASHSEED`` must be set *before* the Python interpreter starts
      to make dict ordering deterministic. We set it here as a best-effort
      for child processes; the parent process is unaffected.
    - ``torch.use_deterministic_algorithms(True, warn_only=True)`` enables
      strict determinism where supported (cuDNN, scatter_add, index_add,
      etc.). ``warn_only=True`` lets us see which ops are non-deterministic
      instead of crashing the run. Once phase 2 knows its full op set, the
      flag can be flipped to ``warn_only=False`` or specific non-deterministic
      ops can be whitelisted via ``torch.use_deterministic_algorithms(..., whitelisted=...)``.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
