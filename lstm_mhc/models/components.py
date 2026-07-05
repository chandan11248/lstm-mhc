"""Shared architectural components for LSTM-µHC.

Contains:
    - RMSNorm: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    - SinkhornKnopp: Birkhoff-polytope projection (doubly stochastic matrices).
    - MuHCHeads: pre/post/res head computation for the µHC highway.

Engineering notes
-----------------
- ``eps`` defaults are *numerically safe* (1e-6 / 1e-8), not the DeepSeek LLM
  values (1e-20) which are unsafe for small RNNs / AMP. Replication of the
  paper's exact setup is still possible by passing ``rmsnorm_eps=1e-20``.
- Sinkhorn-Knopp is implemented with an ``exp`` positive map and a hard clamp;
  the ``learned`` (pre-projection) matrix is returned alongside the projected
  matrix so analyses can separate the analytically-guaranteed DS property from
  what the model actually learned (see review §1, C1/C6).
- Head ablation flags (``use_pre/use_post/use_res``) implement the mHC paper's
  Table-1 ablation: a disabled head falls back to a fixed identity / uniform
  mapping rather than the learned one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import ExperimentConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019).

    Normalizes by the root-mean-square of the activations (no mean subtraction),
    then applies a learned per-feature scale. Computed in fp32 for numerical
    stability; output is cast back to the input dtype.

    Args:
        dim: feature dimension to normalize over (last axis).
        eps: small constant added inside the sqrt. Default ``1e-6`` is safe for
            fp16/bf16 autocast; the mHC paper used ``1e-20`` which is unsafe in
            this regime and should only be used for an exact replication study.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Up-cast to fp32 so rsqrt is stable; cast result back.
        xf = x.float()
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        normed = xf * torch.rsqrt(ms + self.eps)
        return normed.to(x.dtype) * self.weight


class SinkhornKnopp(nn.Module):
    """Project matrices onto the Birkhoff polytope (doubly stochastic).

    Given an unconstrained real matrix ``M``::

        1.  ``P = exp(clamp(M, -clamp, clamp))``          (positivity)
        2.  repeat ``num_iterations`` times:
                P <- P / row_sum(P)                       (row normalize)
                P <- P / col_sum(P)                       (col normalize)

    The map is differentiable; gradients flow through the normalizations.
    Converges to a doubly stochastic matrix (Sinkhorn 1967); 20 iterations is
    the practical default and gives the ~1.6 backward-gain deviation reported
    in the mHC paper.

    Args:
        num_iterations: number of row/col normalization sweeps (t_max).
        clamp: clamp applied *before* exp to prevent overflow/underflow.
        eps: numerical stabilizer added to row/col sums before division.
    """

    def __init__(self, num_iterations: int = 20, clamp: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.num_iterations = num_iterations
        self.clamp = clamp
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x : (..., n, n) unconstrained real matrices. Returns DS matrices."""
        p = torch.exp(x.clamp(min=-self.clamp, max=self.clamp))
        for _ in range(self.num_iterations):
            p = p / (p.sum(dim=-1, keepdim=True) + self.eps)   # rows -> 1
            p = p / (p.sum(dim=-2, keepdim=True) + self.eps)   # cols -> 1
        return p


class MuHCHeads(nn.Module):
    """Compute the three µHC heads: H_pre, H_post, H_res.

    Dynamic + static parameterization (mHC paper Eqs. 7-8)::

        H_tilde = alpha * (x_norm @ phi) + b
        H_pre   = sigmoid(H_tilde_pre)              in [0, 1]
        H_post  = 2 * sigmoid(H_tilde_post)         in [0, 2]
        H_res   = SinkhornKnopp(H_tilde_res)        doubly stochastic

    For Model B (unconstrained) the raw ``H_tilde`` values are returned with no
    sigmoid / projection. Head ablation (``use_pre/use_post/use_res``) replaces
    a head's output with a fixed identity/uniform mapping.

    Note on asymmetric ablation values (``H_pre=1/n`` vs ``H_post=1``)
    ------------------------------------------------------------------
    The two ablations look asymmetric but are correct for their operations:
    H_pre is a *sum-reduction* (aggregation over streams, ``X_agg = (H_pre * X).sum(2)``),
    so uniform ``1/n`` = unweighted mean over streams. H_post is *element-wise
    multiplication* (broadcast, ``X_new = H_post * lstm_out``), so ``1.0`` = full
    magnitude distribution to every stream. The unit-broadcast for H_post is
    the correct "no scaling" fallback; ``1/n`` would silently attenuate the
    LSTM output by 1/n and make the ablation no longer comparable.

    Args:
        n_streams: number of parallel streams (n).
        feature_dim: per-stream feature dim (d); flattened dim is ``n*d``.
        constrained: True => Model C (manifold); False => Model B (raw).
        alpha_init: gating-factor init.
        sinkhorn_iters: Sinkhorn-Knopp iterations.
        use_pre/use_post/use_res: enable (True) or ablate (False) each head.
        rmsnorm_eps/sinkhorn_clamp/sinkhorn_norm_eps: passed through from config.
    """

    def __init__(
        self,
        n_streams: int,
        feature_dim: int,
        constrained: bool = True,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
        use_pre: bool = True,
        use_post: bool = True,
        use_res: bool = True,
        rmsnorm_eps: float = 1e-6,
        sinkhorn_clamp: float = 10.0,
        sinkhorn_norm_eps: float = 1e-8,
    ):
        super().__init__()
        self.n = n_streams
        self.d = feature_dim
        self.nd = n_streams * feature_dim
        self.constrained = constrained
        self.use_pre = use_pre
        self.use_post = use_post
        self.use_res = use_res

        # Gating factors (learnable scalars, small init).
        self.alpha_pre = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(alpha_init)))

        # Dynamic linear projections.
        self.phi_pre = nn.Linear(self.nd, n_streams, bias=False)
        self.phi_post = nn.Linear(self.nd, n_streams, bias=False)
        self.phi_res = nn.Linear(self.nd, n_streams * n_streams, bias=False)

        # Static biases.
        self.b_pre = nn.Parameter(torch.zeros(1, n_streams))
        self.b_post = nn.Parameter(torch.zeros(1, n_streams))
        self.b_res = nn.Parameter(torch.eye(n_streams))

        # Manifold projection.
        if constrained:
            self.sinkhorn = SinkhornKnopp(
                num_iterations=sinkhorn_iters,
                clamp=sinkhorn_clamp,
                eps=sinkhorn_norm_eps,
            )
        else:
            self.sinkhorn = None

        self._init_weights()

    def _init_weights(self):
        """Start near identity mapping (small dynamic contribution).

        Critical fix for b_res: the static bias must be strong enough that
        Sinkhorn(exp(b_res)) ≈ identity at init.  A value of 0.1*I produces
        a near-uniform matrix after Sinkhorn (all entries ≈ 1/n), which
        destroys the per-stream residual signal.  Using 5.0*I gives
        ~0.98*I after Sinkhorn, preserving per-stream information.
        """
        for proj in (self.phi_pre, self.phi_post, self.phi_res):
            nn.init.normal_(proj.weight, mean=0.0, std=0.01)
        # Residual bias: strong diagonal so Sinkhorn ≈ identity at init.
        nn.init.eye_(self.b_res)
        with torch.no_grad():
            self.b_res.mul_(5.0)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: ExperimentConfig, constrained: bool) -> "MuHCHeads":
        return cls(
            n_streams=config.n_streams,
            feature_dim=config.hidden_dim,
            constrained=constrained,
            alpha_init=config.alpha_init,
            sinkhorn_iters=config.sinkhorn_iterations,
            use_pre=config.use_pre,
            use_post=config.use_post,
            use_res=config.use_res,
            rmsnorm_eps=config.rmsnorm_eps,
            sinkhorn_clamp=config.sinkhorn_clamp,
            sinkhorn_norm_eps=config.sinkhorn_norm_eps,
        )

    # ------------------------------------------------------------------
    def forward(self, x_norm: torch.Tensor):
        """Compute heads from the RMSNorm-applied flattened input.

        Args:
            x_norm: (B, T, n*d).
        Returns:
            H_pre:  (B, T, n, 1)  stream-aggregation weights
            H_post: (B, T, n, 1)  stream-distribution weights
            H_res:  (B, T, n, n)  residual mixing matrix
        """
        B, T, _ = x_norm.shape
        n = self.n

        # Dynamic + static mapping (DeepSeek Eq. 7-8: alpha * tanh(theta * x) + b).
        # tanh bounds the dynamic contribution to [-alpha, alpha], preventing
        # unbounded head values that would destabilize training.
        h_pre = self.alpha_pre * torch.tanh(self.phi_pre(x_norm)) + self.b_pre      # (B,T,n)
        h_post = self.alpha_post * torch.tanh(self.phi_post(x_norm)) + self.b_post  # (B,T,n)
        h_res = self.alpha_res * torch.tanh(self.phi_res(x_norm)) + self.b_res.flatten()  # (B,T,n*n)
        h_res = h_res.view(B, T, n, n)

        # ---- H_pre ----
        if not self.use_pre:
            # Uniform aggregation: each stream weighted 1/n.
            H_pre = torch.full((B, T, n), 1.0 / n, device=x_norm.device, dtype=x_norm.dtype)
        elif self.constrained:
            H_pre = torch.sigmoid(h_pre)
        else:
            H_pre = h_pre

        # ---- H_post ----
        if not self.use_post:
            # Uniform distribution: LSTM output broadcast equally to all streams.
            H_post = torch.ones((B, T, n), device=x_norm.device, dtype=x_norm.dtype)
        elif self.constrained:
            H_post = 2.0 * torch.sigmoid(h_post)
        else:
            H_post = h_post

        # ---- H_res ----
        if not self.use_res:
            # Identity residual mixing (skip the highway) — the ablation that
            # isolates H_res's contribution.
            eye = torch.eye(n, device=x_norm.device, dtype=x_norm.dtype)
            H_res = eye.expand(B, T, n, n).contiguous()
        elif self.constrained:
            H_res = self.sinkhorn(h_res)
        else:
            H_res = h_res

        # (B,T,n) -> (B,T,n,1) for broadcasting against (B,T,n,d).
        return H_pre.unsqueeze(3), H_post.unsqueeze(3), H_res


def build_heads(config: ExperimentConfig, constrained: bool) -> MuHCHeads:
    """Convenience factory used by the model classes."""
    return MuHCHeads.from_config(config, constrained)
