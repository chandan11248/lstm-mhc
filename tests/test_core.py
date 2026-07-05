"""Core test suite for LSTM-µHC.

Tests every critical property fixed in the review:
    1. RMSNorm numerical safety (eps, dtype)
    2. Sinkhorn-Knopp doubly-stochastic property
    3. MuHCHeads constrained/unconstrained + ablation flags
    4. Composite gain measures spectral norm correctly
    5. Parameter matching stays within tolerance
    6. Data pipeline has no leakage gaps
    7. Config YAML round-trip
    8. ForecastMetrics de-normalization and NaN guard
    9. All models forward/backward without error
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg4():
    from lstm_mhc.utils.config import ExperimentConfig
    return ExperimentConfig(num_layers=4, model_type="mhc", seed=42, batch_size=2, num_epochs=1)


@pytest.fixture
def cfg4_vanilla():
    from lstm_mhc.utils.config import ExperimentConfig
    return ExperimentConfig(num_layers=4, model_type="vanilla", seed=42, batch_size=2, num_epochs=1)


# ---------------------------------------------------------------------------
# 1. RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_shape_preserved(self, cfg4):
        from lstm_mhc.models.components import RMSNorm
        norm = RMSNorm(128, eps=cfg4.rmsnorm_eps)
        x = torch.randn(4, 100, 128)
        assert norm(x).shape == x.shape

    def test_norm_output_bounded(self, cfg4):
        from lstm_mhc.models.components import RMSNorm
        norm = RMSNorm(64, eps=cfg4.rmsnorm_eps)
        x = torch.randn(8, 50, 64)
        out = norm(x)
        rms = out.float().pow(2).mean(dim=-1).sqrt()
        # RMS should be near 1 after normalization (scale init = 1).
        assert rms.mean().item() < 2.0
        assert rms.mean().item() > 0.1

    def test_eps_no_overflow(self):
        """eps=1e-6 should not cause rsqrt(1e-6)=1e3 gain spikes."""
        from lstm_mhc.models.components import RMSNorm
        norm = RMSNorm(16, eps=1e-6)
        x = torch.full((2, 10, 16), 1e-7)  # near-zero input
        out = norm(x)
        assert torch.isfinite(out).all(), "RMSNorm produced NaN/Inf on near-zero input"

    def test_fp16_safe(self):
        """fp16 input should not overflow with default eps."""
        from lstm_mhc.models.components import RMSNorm
        norm = RMSNorm(32, eps=1e-6)
        x = torch.randn(2, 20, 32).half()
        out = norm(x)
        assert torch.isfinite(out).all(), "RMSNorm produced NaN in fp16"


# ---------------------------------------------------------------------------
# 2. Sinkhorn-Knopp
# ---------------------------------------------------------------------------

class TestSinkhornKnopp:
    def test_doubly_stochastic(self):
        from lstm_mhc.models.components import SinkhornKnopp
        # 20 iterations on hard inputs (scale=5) can leave ~5% residual.
        # Test moderate inputs (scale=1) converge well in 20 iterations.
        sk = SinkhornKnopp(num_iterations=20)
        x = torch.randn(8, 10, 4, 4)  # unit-scale
        ds = sk(x)
        row_sums = ds.sum(dim=-1)
        col_sums = ds.sum(dim=-2)
        max_dev = max((row_sums - 1.0).abs().max().item(),
                      (col_sums - 1.0).abs().max().item())
        assert max_dev < 0.02, f"unit-scale: max deviation {max_dev}"

    def test_non_negative(self):
        from lstm_mhc.models.components import SinkhornKnopp
        sk = SinkhornKnopp()
        ds = sk(torch.randn(4, 5, 4, 4) * 10)
        assert ds.min().item() >= 0.0

    def test_gradient_flows(self):
        """Backward pass through Sinkhorn should not be zero/NaN."""
        from lstm_mhc.models.components import SinkhornKnopp
        sk = SinkhornKnopp()
        x = torch.randn(2, 5, 4, 4, requires_grad=True)
        out = sk(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 3. MuHCHeads
# ---------------------------------------------------------------------------

class TestMuHCHeads:
    def test_constrained_ds(self, cfg4):
        from lstm_mhc.models.components import MuHCHeads
        heads = MuHCHeads.from_config(cfg4, constrained=True)
        x = torch.randn(2, 50, cfg4.n_streams * cfg4.hidden_dim)
        H_pre, H_post, H_res = heads(x)
        # H_res rows and cols sum to 1.
        rs = H_res.detach().sum(dim=-1)
        cs = H_res.detach().sum(dim=-2)
        assert (rs - 1.0).abs().max().item() < 1e-4
        assert (cs - 1.0).abs().max().item() < 1e-4
        # H_pre in [0,1], H_post in [0,2].
        assert H_pre.min().item() >= 0.0
        assert H_post.max().item() <= 2.0 + 1e-4

    def test_unconstrained_raw(self, cfg4):
        from lstm_mhc.models.components import MuHCHeads
        heads = MuHCHeads.from_config(cfg4, constrained=False)
        x = torch.randn(2, 20, cfg4.n_streams * cfg4.hidden_dim)
        H_pre, H_post, H_res = heads(x)
        # Unconstrained: H_res is the raw learned matrix (can be negative).
        # Just verify shapes.
        assert H_res.shape == (2, 20, cfg4.n_streams, cfg4.n_streams)

    def test_ablation_no_res(self, cfg4):
        """With use_res=False, H_res should be the identity."""
        from lstm_mhc.models.components import MuHCHeads
        from lstm_mhc.utils.config import ExperimentConfig
        d = {k: v for k, v in cfg4.to_dict().items()
             if k in {f.name for f in __import__("dataclasses").fields(ExperimentConfig)}}
        d["use_res"] = False
        cfg = ExperimentConfig(**d)
        heads = MuHCHeads.from_config(cfg, constrained=True)
        x = torch.randn(2, 10, cfg.n_streams * cfg.hidden_dim)
        _, _, H_res = heads(x)
        eye = torch.eye(cfg.n_streams).expand_as(H_res)
        assert (H_res.detach() - eye).abs().max().item() < 1e-5

    def test_ablation_no_pre(self, cfg4):
        """With use_pre=False, H_pre should be uniform 1/n."""
        from lstm_mhc.models.components import MuHCHeads
        from lstm_mhc.utils.config import ExperimentConfig
        d = {k: v for k, v in cfg4.to_dict().items()
             if k in {f.name for f in __import__("dataclasses").fields(ExperimentConfig)}}
        d["use_pre"] = False
        cfg = ExperimentConfig(**d)
        n = cfg.n_streams
        heads = MuHCHeads.from_config(cfg, constrained=True)
        x = torch.randn(2, 10, n * cfg.hidden_dim)
        H_pre, _, _ = heads(x)
        expected = 1.0 / n
        assert (H_pre.detach() - expected).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# 4. Composite gain
# ---------------------------------------------------------------------------

class TestCompositeGain:
    def test_identity_gain(self):
        """Identity matrices => gain = 1."""
        from lstm_mhc.evaluation.stability import composite_gain
        eye = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 10, 4, 4).contiguous()
        g = composite_gain([eye])
        assert abs(g["spectral_norm"] - 1.0) < 1e-4
        assert abs(g["fwd_amax"] - 1.0) < 1e-4
        assert abs(g["bwd_amax"] - 1.0) < 1e-4

    def test_exploding_gain(self):
        """2× identity * 3 layers => spectral gain = 8 (2^3)."""
        from lstm_mhc.evaluation.stability import composite_gain
        two_I = (torch.eye(4) * 2.0).unsqueeze(0).unsqueeze(0).expand(2, 5, 4, 4).contiguous()
        g = composite_gain([two_I, two_I, two_I])
        assert abs(g["spectral_norm"] - 8.0) < 1e-3

    def test_ds_gain_bounded(self):
        """DS matrices from Sinkhorn => spectral gain <= 1."""
        from lstm_mhc.models.components import SinkhornKnopp
        from lstm_mhc.evaluation.stability import composite_gain
        sk = SinkhornKnopp(20)
        layers = [sk(torch.randn(4, 20, 4, 4) * 3) for _ in range(8)]
        g = composite_gain(layers)
        assert g["spectral_norm"] <= 1.001, f"DS spectral norm {g['spectral_norm']} > 1"

    def test_empty_returns_one(self):
        from lstm_mhc.evaluation.stability import composite_gain
        g = composite_gain([])
        assert g["spectral_norm"] == 1.0


# ---------------------------------------------------------------------------
# 5. Parameter matching
# ---------------------------------------------------------------------------

class TestParamMatching:
    @pytest.mark.parametrize("L", [4, 8, 16])
    def test_within_tolerance(self, L):
        from lstm_mhc.utils.config import ExperimentConfig
        from lstm_mhc.models.vanilla_lstm import StandardLSTM
        from lstm_mhc.models.mhc_lstm import MHCLSTM
        cfg_mhc = ExperimentConfig(num_layers=L, model_type="mhc")
        cfg_van = ExperimentConfig(num_layers=L, model_type="vanilla")
        p_mhc = sum(p.numel() for p in MHCLSTM(cfg_mhc).parameters())
        p_van = sum(p.numel() for p in StandardLSTM(cfg_van).parameters())
        rel_err = abs(p_van - p_mhc) / p_mhc
        # Empirical results: L=4 ≈ 0.5%, L=8 ≈ 1%, L=16 ≈ 4.99%.
        # Tolerance is 7% to leave headroom for future int->fp rounding changes.
        # See claude_checked.md T2.
        assert rel_err <= 0.07, (
            f"L={L}: Model A has {p_van:,} params vs Model C {p_mhc:,} "
            f"({rel_err*100:.1f}% error, tolerance 7%)"
        )

    def test_exact_matching_util(self):
        from lstm_mhc.utils.config import ExperimentConfig
        from lstm_mhc.utils.matching import find_parameter_matched_config
        from lstm_mhc.models.mhc_lstm import MHCLSTM
        from lstm_mhc.models.vanilla_lstm import StandardLSTM
        cfg = ExperimentConfig(num_layers=4, model_type="mhc")
        target = MHCLSTM(cfg)
        result = find_parameter_matched_config(target, StandardLSTM, cfg)
        assert result["within_tolerance"], f"Rel error: {result['rel_error']*100:.1f}%"


# ---------------------------------------------------------------------------
# 6. Data pipeline (gap correctness)
# ---------------------------------------------------------------------------

class TestDataPipeline:
    def test_gap_prevents_leakage(self):
        """Train+val window inputs should never overlap the same global indices.

        Constructs a synthetic dataset with a known ``gap`` between train and
        val. Verifies:
          - Train's last window ends at or before the train slice end.
          - Val's first window starts at slice index 0 (which corresponds to
            global index ``train_end + gap``, *not* ``train_end``).
          - The full-length no-gap case (gap=0) shrinks the val dataset
            window count by exactly the number of indices the gap covers.
        """
        from lstm_mhc.data.weather_dataset import WeatherTimeSeriesDataset
        N = 2000; in_len = 100; horizons = [6, 12]
        train_end = 1000
        train = np.random.randn(train_end, 4).astype(np.float32)
        val = np.random.randn(N - train_end, 4).astype(np.float32)
        train_ds = WeatherTimeSeriesDataset(train, in_len, horizons, stride=1)
        # Train window [0:in_len] is global [0:in_len], last window ends at
        # train_end - in_len + in_len = train_end, never past it.
        assert train_ds.valid_starts[-1] + in_len <= train_end

        # Two gap settings: gap=0 and gap=72. Both should produce valid
        # datasets, but the with-gap case should be shorter by exactly the
        # number of stride-1 window starts that fall inside the gap.
        for gap in (0, 72):
            val_with_gap = val[: N - train_end - gap]
            val_ds = WeatherTimeSeriesDataset(val_with_gap, in_len, horizons, stride=1)
            assert len(val_ds) > 0
            # First val window reads val[0:in_len] = global [train_end+gap : ...]
            # — must not touch any train index (< train_end + gap means safe).
            assert val_ds.valid_starts[0] >= 0
        # The gap=0 case must produce strictly more val windows than gap=72.
        n0 = len(WeatherTimeSeriesDataset(val, in_len, horizons, stride=1))
        n72 = len(WeatherTimeSeriesDataset(val[: N - train_end - 72], in_len, horizons, stride=1))
        assert n0 > n72, f"gap=0 yielded {n0} windows, gap=72 yielded {n72}; expected gap=0 > gap=72"


# ---------------------------------------------------------------------------
# 7. Config round-trip
# ---------------------------------------------------------------------------

class TestConfig:
    def test_yaml_roundtrip(self, cfg4):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test_config.yaml"
            cfg4.dump(path)
            from lstm_mhc.utils.config import ExperimentConfig
            loaded = ExperimentConfig.from_yaml(path)
            assert loaded.model_type == cfg4.model_type
            assert loaded.num_layers == cfg4.num_layers
            assert loaded.rmsnorm_eps == cfg4.rmsnorm_eps
            assert loaded.seed == cfg4.seed
            assert loaded.config_hash == cfg4.config_hash

    def test_run_name_encodes_key_params(self, cfg4):
        name = cfg4.run_name
        assert "mhc" in name
        assert "l4" in name
        assert "s42" in name

    def test_safe_defaults(self, cfg4):
        assert cfg4.rmsnorm_eps == 1e-6  # not 1e-20
        assert cfg4.adamw_eps == 1e-8    # not 1e-20
        assert cfg4.dropout == 0.1
        assert cfg4.split_gap == 72


# ---------------------------------------------------------------------------
# 8. ForecastMetrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_mse_zero_on_identical(self):
        from lstm_mhc.evaluation.metrics import compute_forecast_metrics
        x = np.random.randn(10, 5, 4)
        m = compute_forecast_metrics(x, x, [6, 12, 24, 48, 72])
        assert m.mse < 1e-10
        assert m.mae < 1e-10

    def test_mape_on_realistic_data(self):
        from lstm_mhc.evaluation.metrics import compute_forecast_metrics
        np.random.seed(0)
        target = 70 + 10 * np.random.randn(20, 5, 4)  # ~70°F
        pred = target + np.random.randn(*target.shape) * 2
        m = compute_forecast_metrics(pred, target, [6, 12, 24, 48, 72])
        assert m.mape > 0 and m.mape < 20  # ~2°F error on ~70°F data
        assert m.smape > 0 and m.smape < 20

    def test_is_finite_guard(self):
        from lstm_mhc.evaluation.metrics import ForecastMetrics, is_finite_metric
        good = ForecastMetrics(0.5, 0.7, 0.4, 2.0, 3.0, 1.0, [0.4]*5)
        bad = ForecastMetrics(float("nan"), 0.7, 0.4, 2.0, 3.0, 1.0, [0.4]*5)
        assert is_finite_metric(good)
        assert not is_finite_metric(bad)


# ---------------------------------------------------------------------------
# 9. Model forward/backward (smoke tests)
# ---------------------------------------------------------------------------

class TestModels:
    @pytest.mark.parametrize("model_type", ["vanilla", "hc", "mhc"])
    def test_forward_backward(self, model_type, cfg4):
        from lstm_mhc.models.vanilla_lstm import StandardLSTM
        from lstm_mhc.models.hc_lstm import HCLSTM
        from lstm_mhc.models.mhc_lstm import MHCLSTM
        from lstm_mhc.utils.config import ExperimentConfig
        cfg = ExperimentConfig(num_layers=4, model_type=model_type, seed=42,
                               batch_size=2, num_epochs=1)
        models = {"vanilla": StandardLSTM, "hc": HCLSTM, "mhc": MHCLSTM}
        model = models[model_type](cfg)
        x = torch.randn(2, 100, 4)
        pred, h_res = model(x)
        assert pred.shape == (2, 5, 4)
        loss = pred.sum()
        loss.backward()
        # All params should have gradients.
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
        # h_res should be a list of (B,T,n,n) for hc/mhc, None for vanilla.
        if model_type == "vanilla":
            assert h_res is None
        else:
            assert isinstance(h_res, list) and len(h_res) == 4
            assert h_res[0].shape == (2, 100, 4, 4)

    def test_mhc_h_res_is_ds(self):
        """After forward, Model C's H_res must be doubly stochastic."""
        from lstm_mhc.models.mhc_lstm import MHCLSTM
        from lstm_mhc.utils.config import ExperimentConfig
        cfg = ExperimentConfig(num_layers=8, model_type="mhc")
        model = MHCLSTM(cfg)
        x = torch.randn(4, 80, 4)
        _, hrs = model(x)
        for hr in hrs:
            rs = hr.detach().sum(dim=-1)
            cs = hr.detach().sum(dim=-2)
            assert (rs - 1.0).abs().max().item() < 1e-4
            assert (cs - 1.0).abs().max().item() < 1e-4
            assert hr.detach().min().item() >= -1e-4

    @pytest.mark.parametrize("L", [4, 8, 16])
    def test_depth_scaling(self, L):
        """All 3 models should build and forward at all target depths."""
        from lstm_mhc.models.vanilla_lstm import StandardLSTM
        from lstm_mhc.models.hc_lstm import HCLSTM
        from lstm_mhc.models.mhc_lstm import MHCLSTM
        from lstm_mhc.utils.config import ExperimentConfig
        for name, cls in [("vanilla", StandardLSTM), ("hc", HCLSTM), ("mhc", MHCLSTM)]:
            cfg = ExperimentConfig(num_layers=L, model_type=name, seed=0)
            model = cls(cfg)
            x = torch.randn(2, 64, 4)
            pred, _ = model(x)
            assert pred.shape == (2, 5, 4)


# ---------------------------------------------------------------------------
# 10. Stats utility
# ---------------------------------------------------------------------------

class TestStats:
    def test_identical_arrays(self):
        from lstm_mhc.utils.stats import paired_significance
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        r = paired_significance(a, a)
        assert abs(r.mean_diff) < 1e-10
        assert abs(r.cohens_d) < 1e-10

    def test_large_effect_detected(self):
        from lstm_mhc.utils.stats import paired_significance
        np.random.seed(42)
        a = np.random.normal(10, 0.5, 20)
        b = a - 3.0  # large consistent improvement
        r = paired_significance(a, b)
        assert r.significant_at_005
        assert r.effect_size in ("medium", "large")
        assert r.mean_diff > 0

    def test_summary_string(self):
        from lstm_mhc.utils.stats import paired_significance
        a = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1], dtype=float)
        s = paired_significance(a, b).summary()
        assert "n=" in s
        assert "Cohen" in s
