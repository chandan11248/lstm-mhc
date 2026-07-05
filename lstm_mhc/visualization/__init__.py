"""LSTM-µHC visualization module.

The four orphan plot modules (``plot_stability``, ``plot_convergence``,
``plot_accuracy``, ``plot_efficiency``) are still on disk in this directory
but are not re-exported here. They are inlined as base64-encoded source in
the phase1b notebooks and must remain on disk for any re-run of phase1b.
Future phase-2 cleanup (after phase1b is archived) can safely delete them.
"""

from .plot_style import setup_style, save_fig, get_style, get_label, get_color, COLORS, LABELS
from .plot_core import (
    plot_gradient_norm_vs_steps,
    plot_amax_gain_vs_steps,
    plot_training_loss_smoothness,
    plot_validation_mse_vs_epochs,
    plot_horizon_degradation,
    generate_all_l4,
)
from .plot_depth import (
    plot_performance_vs_depth,
    plot_gradient_norm_vs_depth,
    plot_amax_gain_vs_depth,
    plot_depth_loss_convergence,
)
from .plot_matrices import plot_h_res_heatmaps, plot_doubly_stochastic_verification
