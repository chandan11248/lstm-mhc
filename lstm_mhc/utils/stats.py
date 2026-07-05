"""Statistical significance tests for comparing model performance.

For n >= 5 seeds, applies paired t-test (parametric) and Wilcoxon
signed-rank test (non-parametric) with bootstrap confidence intervals
and Cohen's d effect size.

Usage::

    from lstm_mhc.utils.stats import paired_significance
    result = paired_significance(model_a_mse_array, model_c_mse_array)
    print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SignificanceResult:
    """Results of a paired comparison."""
    n: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    mean_diff: float
    ci_95_lo: float
    ci_95_hi: float
    t_stat: float
    p_value_t: float          # paired t-test p-value
    p_value_wilcoxon: float   # Wilcoxon signed-rank p-value
    cohens_d: float
    effect_size: str           # "negligible" | "small" | "medium" | "large"
    significant_at_005: bool   # True if p < 0.05 (either test)

    def summary(self) -> str:
        """Human-readable summary suitable for a paper's experimental section."""
        sig = "YES" if self.significant_at_005 else "NO"
        return (
            f"n={self.n} | A={self.mean_a:.6f}±{self.std_a:.6f} vs "
            f"B={self.mean_b:.6f}±{self.std_b:.6f} | "
            f"diff={self.mean_diff:+.6f} [{self.ci_95_lo:+.6f}, {self.ci_95_hi:+.6f}] | "
            f"t={self.t_stat:.3f} p={self.p_value_t:.4f} | "
            f"Wilcoxon p={self.p_value_wilcoxon:.4f} | "
            f"Cohen's d={self.cohens_d:.2f} ({self.effect_size}) | "
            f"Significant: {sig}"
        )


def _cohens_d_category(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def paired_significance(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> SignificanceResult:
    """Paired statistical comparison of two models' performance arrays.

    Args:
        a: shape ``(n_seeds,)`` of Model A's metric (e.g. MSE per seed).
        b: shape ``(n_seeds,)`` of Model B's metric.
        n_bootstrap: number of bootstrap resamples for CI.
        alpha: significance level (default 0.05).

    Returns:
        :class:`SignificanceResult` with both parametric and non-parametric
        tests, effect size, and bootstrap CI.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    assert a.shape == b.shape and a.ndim == 1, f"Shape mismatch: {a.shape} vs {b.shape}"
    n = len(a)
    diff = a - b

    mean_a, mean_b = float(a.mean()), float(b.mean())
    std_a, std_b = float(a.std(ddof=1)), float(b.std(ddof=1))
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    # Paired t-test (with fallback for n < 2).
    if n < 2 or se < 1e-16:
        t_stat, p_t = 0.0, 1.0
    else:
        t_stat = mean_diff / se
        # Two-tailed p via survival function of Student's t (df = n-1).
        try:
            from scipy import stats as sp_stats
            p_t = float(2 * sp_stats.t.sf(abs(t_stat), df=n - 1))
        except ImportError:
            p_t = 1.0

    # Wilcoxon signed-rank test (non-parametric, n >= 6 for scipy to work).
    if n >= 6:
        try:
            from scipy import stats as sp_stats
            if np.allclose(diff, 0):
                p_w = 1.0
            else:
                _, p_w = sp_stats.wilcoxon(diff, alternative="two-sided")
                p_w = float(p_w)
        except ImportError:
            p_w = 1.0
    else:
        p_w = 1.0

    # Cohen's d (paired).
    sd_diff = float(diff.std(ddof=1)) if n > 1 else 1.0
    cohens_d = mean_diff / max(sd_diff, 1e-12)
    effect = _cohens_d_category(cohens_d)

    # Bootstrap 95% CI for the mean difference.
    rng = np.random.default_rng(42)
    boot = np.array([
        diff[rng.integers(0, n, size=n)].mean()
        for _ in range(n_bootstrap)
    ])
    ci_lo = float(np.percentile(boot, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))

    return SignificanceResult(
        n=n, mean_a=mean_a, mean_b=mean_b, std_a=std_a, std_b=std_b,
        mean_diff=mean_diff, ci_95_lo=ci_lo, ci_95_hi=ci_hi,
        t_stat=t_stat, p_value_t=p_t, p_value_wilcoxon=p_w,
        cohens_d=cohens_d, effect_size=effect,
        significant_at_005=(p_t < alpha) or (p_w < alpha),
    )
