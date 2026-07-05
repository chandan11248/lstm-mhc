# µHC: Micro-Convex Highway — Concept Breakdown from mHC

> **Source Paper:** *mHC: Manifold-Constrained Hyper-Connections* (Xie et al., DeepSeek-AI, arXiv:2512.24880v2, Jan 2026)
>
> **Purpose:** This document distills the core concepts, mathematics, and engineering insights from the DeepSeek mHC paper that directly inform and justify our **LSTM-µHC** architecture for time-series forecasting.

---

## Table of Contents

1. [The Problem mHC Solves](#1-the-problem-mhc-solves)
2. [From Residual Connections to Hyper-Connections](#2-from-residual-connections-to-hyper-connections)
3. [Why Unconstrained HC Fails at Scale](#3-why-unconstrained-hc-fails-at-scale)
4. [The Manifold Constraint: Core Innovation](#4-the-manifold-constraint-core-innovation)
5. [Mathematical Formulation](#5-mathematical-formulation)
6. [Sinkhorn-Knopp Algorithm](#6-sinkhorn-knopp-algorithm)
7. [Why Doubly Stochastic Matrices Work](#7-why-doubly-stochastic-matrices-work)
8. [Head Architecture (H_pre, H_post, H_res)](#8-head-architecture-h_pre-h_post-h_res)
9. [Parameterization Details](#9-parameterization-details)
10. [Infrastructure Optimizations](#10-infrastructure-optimizations)
11. [Key Experimental Findings](#11-key-experimental-findings)
12. [Adaptation to LSTM-µHC](#12-adaptation-to-lstm-µhc)

---

## 1. The Problem mHC Solves

### The Identity Mapping Crisis

Standard residual connections (He et al., 2016) preserve signal stability through an **identity mapping**:

$$x_{l+1} = x_l + \mathcal{F}(x_l, W_l)$$

When unrolled across layers $l$ to $L$:

$$x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i)$$

The term $x_l$ maps directly from shallow to deep layers **without modification** — this is the identity mapping property that keeps training stable.

### What HC Breaks

Hyper-Connections (Zhu et al., 2024) expand the residual stream width by a factor $n$ and introduce learnable matrices to route information. This improves expressivity but **destroys the identity mapping**, causing:

1. **Signal explosion/vanishing** across deep layers
2. **Training instability** (loss surges, gradient norm spikes)
3. **Restricted scalability** — cannot train beyond moderate depth

### What mHC Restores

mHC projects the HC residual matrices onto a **constrained manifold** (the Birkhoff polytope of doubly stochastic matrices), which:

- Restores the identity mapping property
- Enables stable training at scale
- Adds only **6.7% training overhead** (with $n = 4$)

---

## 2. From Residual Connections to Hyper-Connections

### Standard Residual Connection

```
x_l ──────────────────────→ x_{l+1}
 │                            ↑
 └──→ Layer F(x_l, W_l) ────┘
```

- Single stream of dimension $C$
- Identity mapping preserves signal
- No cross-stream mixing possible

### Hyper-Connections (HC)

```
x_l (n×C) ──→ H_res · x_l ──────────────────────────→ x_{l+1} (n×C)
                │                                        ↑
                └──→ H_pre · x_l → Layer F → H_post ───┘
```

- **$n$ parallel streams** of dimension $C$ (expanded residual width)
- Three learnable matrices: $\mathcal{H}^{pre}_l \in \mathbb{R}^{1 \times n}$, $\mathcal{H}^{post}_l \in \mathbb{R}^{1 \times n}$, $\mathcal{H}^{res}_l \in \mathbb{R}^{n \times n}$
- **Unconstrained** learnable matrices — this is the source of instability

### Manifold-Constrained HC (mHC)

```
x_l (n×C) ──→ P_M(H_res) · x_l ──────────────────────────→ x_{l+1} (n×C)
                │                                            ↑
                └──→ σ(H_pre) · x_l → Layer F → 2σ(H_post) ┘
```

- Same $n$-stream structure
- $\mathcal{H}^{res}_l$ projected onto the **Birkhoff polytope** (doubly stochastic)
- $\mathcal{H}^{pre}_l$ and $\mathcal{H}^{post}_l$ constrained to **non-negative** via sigmoid

---

## 3. Why Unconstrained HC Fails at Scale

### The Composite Mapping Problem

When HC is extended across multiple layers, the signal propagation from layer $l$ to $L$ is governed by:

$$x_L = \left(\prod_{i=1}^{L-l} \mathcal{H}^{res}_{L-i}\right) x_l + \sum_{i=l}^{L-1} \left(\prod_{j=1}^{L-1-i} \mathcal{H}^{res}_{L-j}\right) \mathcal{H}^{post\top}_i \mathcal{F}(\mathcal{H}^{pre}_i x_i, W_i)$$

The composite mapping $\prod_{i=1}^{L-l} \mathcal{H}^{res}_{L-i}$ is the problem:

| Metric | HC (Unconstrained) | mHC (Doubly Stochastic) |
|---|---|---|
| Max Amax Gain (forward) | **~3000** (exploding) | **~1.6** (bounded) |
| Max Amax Gain (backward) | **~3000** (exploding) | **~1.6** (bounded) |
| Loss surge observed | Yes, around step 12k | No |
| Gradient norm stability | Unstable spikes | Stable, comparable to baseline |

### Empirical Evidence from the Paper

- **Figure 2:** HC shows an unexpected loss surge around step 12k in 27B models, correlated with gradient norm instability
- **Figure 3:** The composite mapping in HC produces extreme gain values peaking at 3000 — a stark divergence from the ideal value of 1
- **Figure 8:** HC residual matrices contain large negative values (e.g., -489.0, -259.2) that cause destructive signal interference

---

## 4. The Manifold Constraint: Core Innovation

### The Birkhoff Polytope

The set of all $n \times n$ doubly stochastic matrices forms the **Birkhoff polytope** $\mathcal{M}^{res}$:

$$\mathcal{M}^{res} = \left\{ H \in \mathbb{R}^{n \times n} \;\middle|\; H \mathbf{1}_n = \mathbf{1}_n, \;\; \mathbf{1}^\top_n H = \mathbf{1}^\top_n, \;\; H \geq 0 \right\}$$

This means:
- Every **row sums to 1**
- Every **column sums to 1**
- All entries are **non-negative**

### The Projection

mHC constrains $\mathcal{H}^{res}_l$ to lie on this manifold:

$$\mathcal{P}_{\mathcal{M}^{res}}(\mathcal{H}^{res}_l) \triangleq \text{Sinkhorn-Knopp}\left(\exp(\tilde{\mathcal{H}}^{res}_l)\right)$$

### Degenerate Case

When $n = 1$, the doubly stochastic condition degenerates to the scalar **1**, recovering the original identity mapping $x_{l+1} = x_l + \mathcal{F}(x_l)$.

---

## 5. Mathematical Formulation

### Single-Layer Propagation (mHC)

Given input $x_l \in \mathbb{R}^{n \times C}$ at layer $l$:

$$x_{l+1} = \mathcal{H}^{res}_l \cdot x_l + \mathcal{H}^{post\top}_l \cdot \mathcal{F}(\mathcal{H}^{pre}_l \cdot x_l, W_l)$$

where:
- $x_l, x_{l+1} \in \mathbb{R}^{n \times C}$ — $n$-stream residual with $C$-dim features
- $\mathcal{H}^{res}_l \in \mathbb{R}^{n \times n}$ — doubly stochastic residual mixing
- $\mathcal{H}^{pre}_l \in \mathbb{R}^{1 \times n}$ — stream aggregation (sigmoid-gated)
- $\mathcal{H}^{post}_l \in \mathbb{R}^{1 \times n}$ — stream distribution (sigmoid-gated, scaled by 2)
- $\mathcal{F}$ — the layer function (Attention, FFN, or in our case, LSTM)

### Multi-Layer Unrolling

$$x_L = \left(\prod_{i=1}^{L-l} \mathcal{P}_{\mathcal{M}^{res}}(\mathcal{H}^{res}_{L-i})\right) x_l + \sum_{i=l}^{L-1} \left(\prod_{j=1}^{L-1-i} \mathcal{P}_{\mathcal{M}^{res}}(\mathcal{H}^{res}_{L-j})\right) \mathcal{H}^{post\top}_i \mathcal{F}(\mathcal{H}^{pre}_i x_i, W_i)$$

Because doubly stochastic matrices are **closed under multiplication**, the composite mapping remains doubly stochastic, preserving stability throughout the entire depth.

---

## 6. Sinkhorn-Knopp Algorithm

### Purpose

Project an arbitrary matrix onto the Birkhoff polytope (make it doubly stochastic).

### Procedure

Given an unconstrained matrix $\tilde{\mathcal{H}}^{res}_l$:

**Step 1:** Make all entries positive via exponentiation:

$$M^{(0)} = \exp(\tilde{\mathcal{H}}^{res}_l)$$

**Step 2:** Iteratively normalize rows and columns for $t = 1, 2, \ldots, t_{max}$:

$$M^{(t)} = \mathcal{T}_r\left(\mathcal{T}_c(M^{(t-1)})\right)$$

where:
- $\mathcal{T}_c$: column normalization — divide each column by its sum
- $\mathcal{T}_r$: row normalization — divide each row by its sum

**Step 3:** The result converges to a doubly stochastic matrix:

$$\mathcal{H}^{res}_l = M^{(t_{max})}$$

### Practical Settings

| Parameter | Value | Notes |
|---|---|---|
| $t_{max}$ (iterations) | **20** | Practical convergence; paper's default |
| Convergence | Guaranteed as $t_{max} \to \infty$ | 20 iterations gives backward gradient gain deviation of ~1.6 max |

### Why This Matters for LSTM-µHC

In our architecture, Sinkhorn-Knopp runs **outside the time loop** at the inter-layer level, operating on $(n \times n)$ matrices batched over $B \times T$. This makes it trivially parallelizable via `torch.bmm()` and adds minimal overhead.

---

## 7. Why Doubly Stochastic Matrices Work

Three rigorous theoretical properties make this choice principled:

### Property 1: Norm Preservation (Non-Expansiveness)

$$\|\mathcal{H}^{res}_l\|_2 \leq 1$$

The spectral norm is bounded by 1, so the mapping is **non-expansive**. Signals cannot grow in magnitude — gradient explosion is mathematically impossible.

### Property 2: Compositional Closure

If $A$ and $B$ are doubly stochastic, then $A \cdot B$ is also doubly stochastic. Therefore:

$$\prod_{i=1}^{L-l} \mathcal{H}^{res}_{L-i} \in \mathcal{M}^{res}$$

Stability is preserved across **arbitrary depth** — not just single layers.

### Property 3: Geometric Interpretation (Convex Combination of Permutations)

The Birkhoff polytope is the **convex hull of permutation matrices**. This means $\mathcal{H}^{res}_l$ acts as a **soft permutation** — a convex combination of different ways to route information between streams.

Repeated application monotonically increases stream mixing, functioning as a **robust feature fusion mechanism**.

### Visual Comparison (from Paper Figure 8)

| | HC (Unconstrained) | mHC (Doubly Stochastic) |
|---|---|---|
| Single-layer $\mathcal{H}^{res}_1$ | Values like -6.81, 18.73, -15.29 | Values like 0.83, 0.73, 0.66, 0.75 |
| Composite $\prod_{i=1}^{60} \mathcal{H}^{res}_{61-i}$ | Values like -489.0, 273.3, -259.2 | Values like 0.88, 1.03, 1.00, 1.11 |
| Row/column sums | Wildly divergent from 1 | All approximately 1 |

---

## 8. Head Architecture (H_pre, H_post, H_res)

### Role of Each Head

| Head | Shape | Function | Constraint |
|---|---|---|---|
| $\mathcal{H}^{pre}_l$ | $\mathbb{R}^{1 \times n}$ | **Aggregates** $n$ streams into a single $C$-dim input for the layer function | $\sigma$ (sigmoid, non-negative) |
| $\mathcal{H}^{post}_l$ | $\mathbb{R}^{1 \times n}$ | **Distributes** the layer output back onto $n$ streams | $2\sigma$ (scaled sigmoid, range [0,2]) |
| $\mathcal{H}^{res}_l$ | $\mathbb{R}^{n \times n}$ | **Mixes** features within the residual stream (convex combination) | Sinkhorn-Knopp (doubly stochastic) |

### Ablation Study (Paper Table 1)

When individual heads are disabled (replaced with fixed mappings):

| Configuration | Absolute Loss Gap |
|---|---|
| All heads active | **-0.027** (best) |
| Without $\mathcal{H}^{res}_l$ (identity matrix) | -0.022 |
| Without $\mathcal{H}^{pre}_l$ (uniform 1/n weights) | -0.025 |
| Without $\mathcal{H}^{post}_l$ (uniform ones) | -0.025 |

**Key insight:** $\mathcal{H}^{res}_l$ contributes the most to performance, underscoring the importance of effective inter-stream information exchange.

---

## 9. Parameterization Details

### Input Processing

The input $x_l \in \mathbb{R}^{n \times C}$ is flattened and normalized:

$$\vec{x}_l = \text{vec}(x_l) \in \mathbb{R}^{1 \times nC}$$

$$\vec{x}'_l = \text{RMSNorm}(\vec{x}_l)$$

### Dynamic + Static Mapping Computation

Following the HC formulation, each head combines **input-dependent** (dynamic) and **learned** (static) components:

$$\tilde{\mathcal{H}}^{pre}_l = \alpha^{pre}_l \cdot (\vec{x}'_l \varphi^{pre}_l) + b^{pre}_l$$

$$\tilde{\mathcal{H}}^{post}_l = \alpha^{post}_l \cdot (\vec{x}'_l \varphi^{post}_l) + b^{post}_l$$

$$\tilde{\mathcal{H}}^{res}_l = \alpha^{res}_l \cdot \text{mat}(\vec{x}'_l \varphi^{res}_l) + b^{res}_l$$

where:
- $\alpha^{pre}_l, \alpha^{post}_l, \alpha^{res}_l \in \mathbb{R}$ — learnable **gating factors**, initialized to **0.01**
- $\varphi^{pre}_l, \varphi^{post}_l \in \mathbb{R}^{nC \times n}$ — linear projections for dynamic mappings
- $\varphi^{res}_l \in \mathbb{R}^{nC \times n^2}$ — linear projection for residual mapping
- $b^{pre}_l, b^{post}_l \in \mathbb{R}^{1 \times n}$ — learnable static biases
- $b^{res}_l \in \mathbb{R}^{n \times n}$ — learnable static bias matrix
- $\text{mat}(\cdot)$ reshapes $\mathbb{R}^{1 \times n^2}$ to $\mathbb{R}^{n \times n}$

### Final Constrained Mappings

$$\mathcal{H}^{pre}_l = \sigma(\tilde{\mathcal{H}}^{pre}_l)$$

$$\mathcal{H}^{post}_l = 2\sigma(\tilde{\mathcal{H}}^{post}_l)$$

$$\mathcal{H}^{res}_l = \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}^{res}_l)$$

### Note on $2\sigma$ for $\mathcal{H}^{post}$

The scaling factor of 2 allows the post-distribution head to amplify signals (range $[0, 2]$), compensating for the norm-reducing effect of the doubly stochastic residual mixing.

### Computational Overhead

Since $n$ (typically 4) is much smaller than $C$ (the model dimension), the overhead of computing these heads is **negligible** relative to the layer function $\mathcal{F}$.

---

## 10. Infrastructure Optimizations

### Memory Access Analysis (Paper Table 2)

| Method | Read (elements) | Write (elements) |
|---|---|---|
| Standard Residual | $2C$ | $C$ |
| Hyper-Connections | $(5n+1)C + n^2 + 2n$ | $(3n+1)C + n^2 + 2n$ |

HC increases memory access by a factor proportional to $n$ — the "memory wall" problem.

### Kernel Fusion Strategy

The paper implements three specialized fused kernels:

1. **Unified Head Computation Kernel** (Eqs. 14–15): Fuses two scans on $\vec{x}_l$ into a single matrix multiplication, maximizing memory bandwidth utilization

2. **Lightweight Coefficient Kernel** (Eqs. 16–18): Fuses sigmoid, scaling, and gating operations into a single kernel to reduce launch overhead

3. **Sinkhorn-Knopp Kernel** (Eq. 19): Implements the entire iteration within a single kernel with custom backward pass that recomputes intermediates on-chip

### Recomputing Strategy

- Discard intermediate activations after forward pass
- Recompute on-the-fly during backward pass
- Optimal block size: $L^*_r \approx \sqrt{\frac{nL}{n+2}}$
- Synchronized with pipeline stage boundaries

### DualPipe Communication Overlap

- $\mathcal{F}_{post,res}$ kernels of MLP layers run on a **dedicated high-priority compute stream**
- Prevents blocking of communication
- Enables preemption of overlapped attention computations

### Result: 6.7% Overhead

With all optimizations and $n = 4$, mHC introduces only **6.7% additional training time**.

---

## 11. Key Experimental Findings

### Training Stability (27B model)

| Metric | Baseline | HC | mHC |
|---|---|---|---|
| Final loss gap | 0.000 | Unstable (surge at 12k) | **-0.021** |
| Gradient norm | Stable | Unstable spikes | **Stable** |
| Max composite gain | ~1 | **~3000** | **~1.6** |

### Downstream Performance (Paper Table 4)

| Benchmark | Baseline | HC | mHC |
|---|---|---|---|
| BBH (3-shot) | 43.8 | 48.9 | **51.0** |
| DROP (3-shot) | 47.0 | 51.6 | **53.9** |
| GSM8K (8-shot) | 46.7 | 53.2 | **53.8** |
| MATH (4-shot) | 22.0 | **26.4** | 26.0 |
| MMLU (5-shot) | 59.0 | 63.0 | **63.4** |
| TriviaQA (5-shot) | 54.3 | 56.3 | **57.6** |

### Scaling Behavior

- **Compute scaling** (3B → 9B → 27B): Performance advantage is robustly maintained at higher budgets
- **Token scaling** (3B, 1T tokens): Advantage persists throughout training

### Stability Metrics (Figure 7)

- Single-layer mapping: forward/backward gains tightly clustered around 1.0
- Composite mapping (60 layers): maximum gain bounded at ~1.6 (vs. ~3000 for HC)
- Three orders of magnitude improvement in signal stability

---

## 12. Adaptation to LSTM-µHC

### What We Adopt

| mHC Concept | Our LSTM-µHC Adaptation |
|---|---|
| $n$-stream residual expansion | $n = 4$ continuous physical streams for weather variables |
| Birkhoff polytope projection | Sinkhorn-Knopp on $\mathcal{H}^{res}$ for convex feature fusion |
| Three-head architecture | $\mathcal{H}^{pre}$, $\mathcal{H}^{post}$, $\mathcal{H}^{res}$ at each inter-layer boundary |
| Non-negative constraints | Sigmoid gating on $\mathcal{H}^{pre}$ and $\mathcal{H}^{post}$ |
| RMSNorm normalization | Applied to flattened multi-stream input before head computation |
| Doubly stochastic mixing | Prevents exploding weather variables from dominating the network |

### What We Modify

| mHC (Transformer Domain) | LSTM-µHC (Time-Series Domain) |
|---|---|
| Layer function $\mathcal{F}$ = Attention/FFN | Layer function $\mathcal{F}$ = **LSTM** (cuDNN-optimized) |
| Per-token processing ($x_l \in \mathbb{R}^{n \times C}$) | **Full sequence processing** ($X_l \in \mathbb{R}^{B \times T \times n \times d}$) |
| Sinkhorn-Knopp per token | Sinkhorn-Knopp **vectorized over $B \times T$** via `torch.bmm()` |
| Inside Transformer block | **Inter-layer** (between LSTM layers, outside time loop) |
| Language model pretraining | **Time-series forecasting** (NOAA Weather, ETT) |
| Kernel fusion for memory wall | Standard PyTorch ops sufficient (cuDNN handles LSTM) |

### Why the Macro Shift Works

The original mHC applies the Sinkhorn-Knopp projection per token within each Transformer block. In our LSTM setting:

1. **LSTM processes the entire sequence** $T$ in one cuDNN-optimized pass
2. **µHC heads operate inter-layer**, not per-timestep
3. Sinkhorn-Knopp is batched over $(B \times T)$, making it $O(L)$ not $O(T \times L)$
4. The doubly stochastic constraint still prevents signal explosion across LSTM layers

### Formal LSTM-µHC Block (Layer $l$)

$$\vec{X}_{l,\text{norm}} = \text{RMSNorm}(\text{Flatten}(X_l)) \quad \in \mathbb{R}^{B \times T \times nd}$$

$$\mathcal{H}_l^{pre} = \sigma(\vec{X}_{l,\text{norm}} W_l^{pre} + b_l^{pre}) \quad \in \mathbb{R}^{B \times T \times 1 \times n}$$

$$\mathcal{H}_l^{post} = \sigma(\vec{X}_{l,\text{norm}} W_l^{post} + b_l^{post}) \quad \in \mathbb{R}^{B \times T \times n \times 1}$$

$$\mathcal{H}_l^{res} = \text{Sinkhorn-Knopp}(\exp(\text{Reshape}(\vec{X}_{l,\text{norm}} W_l^{res} + b_l^{res}))) \quad \in \mathbb{R}^{B \times T \times n \times n}$$

$$\tilde{X}_l = \mathcal{H}_l^{pre} \cdot X_l \xrightarrow{\text{squeeze}} \mathbb{R}^{B \times T \times d}$$

$$\hat{X}_{l+1} = \text{LSTM}_l(\tilde{X}_l)$$

$$\tilde{\hat{X}}_{l+1} = \mathcal{H}_l^{post} \cdot \hat{X}_{l+1} \quad \in \mathbb{R}^{B \times T \times n \times d}$$

$$X_{l+1} = \mathcal{H}_l^{res} \cdot X_l + \tilde{\hat{X}}_{l+1} \quad \in \mathbb{R}^{B \times T \times n \times d}$$

---

## Quick Reference: Hyperparameters from mHC Paper

| Parameter | Value | Source |
|---|---|---|
| Expansion rate $n$ | 4 | Table 5 |
| Sinkhorn-Knopp iterations $t_{max}$ | 20 | Table 5 |
| Gating factor initialization $\alpha$ | 0.01 | Table 5 |
| Activation function $\sigma$ | Sigmoid | Eq. (8) |
| $\mathcal{H}^{post}$ scaling factor | 2 | Eq. (8) |
| Normalization | RMSNorm | Eq. (7) |
| Optimizer | AdamW | Table 5 |
| AdamW betas | (0.9, 0.95) | Table 5 |
| AdamW $\epsilon$ | 1e-20 | Table 5 |
| Weight decay | 0.1 | Table 5 |
| RMSNorm $\epsilon$ | 1e-20 | Table 5 |

---

## Key Citations

- **mHC (this paper):** Xie et al., "mHC: Manifold-Constrained Hyper-Connections," arXiv:2512.24880v2, Jan 2026
- **Hyper-Connections (HC):** Zhu et al., "Hyper-Connections," arXiv:2409.19606, 2024
- **Residual Networks:** He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
- **Identity Mappings:** He et al., "Identity Mappings in Deep Residual Networks," ECCV 2016
- **Sinkhorn-Knopp:** Sinkhorn and Knopp, "Concerning nonnegative matrices and doubly stochastic matrices," Pacific J. Math, 1967
- **RMSNorm:** Zhang and Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019
- **DeepSeek-V3 (base architecture):** Liu et al., "DeepSeek-V3 Technical Report," arXiv:2412.19437, 2024
