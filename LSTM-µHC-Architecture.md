# LSTM-µHC: Macro-Level Micro-Convex Highway Architecture

## Architectural Overview

The LSTM-µHC architecture is built on a **macro-level inter-layer residual highway** design, decoupling spatial feature mixing from temporal recurrence. Rather than embedding the µHC (Micro-Convex Highway) mechanics inside the sequential time loop of an LSTM cell, the highway operates as an **inter-layer connection** that preserves hardware-optimized temporal processing while enabling mathematically principled cross-stream feature stability.

### Core Design Principle

> **Decoupling spatial feature mixing from temporal recurrence.**
> The LSTM layer handles 100% of the temporal context processing, while the µHC highway handles 100% of the cross-stream feature stability and depth scaling.

---

## 1. Hardware & Parallelization Justification

Standard LSTM layers in PyTorch rely on highly optimized CUDA/cuDNN kernels that process entire sequences efficiently on GPU. By pulling the µHC mechanics out of the temporal loop and making them an inter-layer connection, we preserve this speed.

### Execution Pipeline

1. **LSTM Layer** processes the whole sequence window (e.g., $T = 724$ steps) in one highly optimized pass, outputting a feature tensor of shape `(Batch, Time, Hidden_Dim)`.

2. **µHC Layer** then takes over. When projecting the expanded streams onto the Birkhoff Polytope via Sinkhorn-Knopp, the operation is **not** performed step-by-step. Instead, the Batch and Time dimensions are treated as parallel batches.

3. **Sinkhorn-Knopp matrix multiplications** are fully vectorized across the entire sequence length using `torch.bmm()`, keeping training overhead at approximately **~6–7%** rather than incurring a ~500% slowdown from sequential per-step execution.

### Complexity Comparison

| Approach | Sequential Dependency | Sinkhorn-Knopp Calls | Parallelization |
|---|---|---|---|
| Micro-level (inside time loop) | $O(T)$ sequential | $T$ calls per layer | None — fully sequential |
| **Macro-level (inter-layer)** | **$O(L)$ sequential** | **$L$ calls total** | **Full — vectorized over $B \times T$** |

---

## 2. Information Routing Justification (Weather Domain)

Weather data consists of heavily entangled physical variables: temperature affects pressure, pressure changes wind speed, wind speed alters humidity.

### The Problem with Deep Stacked LSTMs

In a standard deep stacked LSTM:
- **Layer 1** mixes physical variables
- **Layer 2** mixes them further
- **By Layer 4**, distinct physical boundaries are lost — known as **feature degradation** or **spatial blur**

### How the Macro-Highway Solves This

| Component | Role |
|---|---|
| **Continuous Streams** ($n = 4$) | Long-term, high-dimensional physical channels running through the entire depth of the network |
| **$\mathcal{H}^{pre}$** (Pre-mixing) | Blends streams into a single representation so the LSTM layer can focus purely on temporal patterns (e.g., tracking how a variable drops over a 24-hour window) |
| **$\mathcal{H}^{post}$** (Post-distribution) | Distributes the LSTM's temporal insights back into the highway streams |
| **$\mathcal{H}^{res}$** (Residual mixing) | Performs convex feature fusion on historical highway tracks. Being doubly stochastic (norm $\leq 1$), it prevents any single exploding variable from dominating the network |

---

## 3. Formal Architecture Definition

### Notation

Let the input to layer $l$ be a multi-stream sequence tensor:

$$X_l \in \mathbb{R}^{B \times T \times n \times d}$$

where:
- $B$ = batch size
- $T$ = sequence length
- $n$ = number of parallel streams
- $d$ = feature dimension

### Step 1: Head Computation

**Normalization and Flattening:**

$$\vec{X}_{l,\text{norm}} = \text{RMSNorm}\left(\text{Flatten}(X_l)\right) \quad \in \mathbb{R}^{B \times T \times nd}$$

**Pre-mixing Head:**

$$\mathcal{H}_l^{pre} = \sigma\left(\vec{X}_{l,\text{norm}} \, W_l^{pre} + b_l^{pre}\right) \quad \in \mathbb{R}^{B \times T \times 1 \times n}$$

**Post-distribution Head:**

$$\mathcal{H}_l^{post} = 2\,\sigma\left(\vec{X}_{l,\text{norm}} \, W_l^{post} + b_l^{post}\right) \quad \in \mathbb{R}^{B \times T \times n \times 1}$$

**Residual Mixing Head (Birkhoff Polytope Projection):**

$$\mathcal{H}_l^{res} = \text{Sinkhorn-Knopp}\left(\exp\left(\text{Reshape}(\vec{X}_{l,\text{norm}} \, W_l^{res} + b_l^{res})\right)\right) \quad \in \mathbb{R}^{B \times T \times n \times n}$$

### Step 2: Temporal Aggregation and Recurrence

**Stream Aggregation:**

$$\tilde{X}_l = \mathcal{H}_l^{pre} \cdot X_l \quad \in \mathbb{R}^{B \times T \times 1 \times d} \xrightarrow{\text{squeeze}} \mathbb{R}^{B \times T \times d}$$

**LSTM Processing:**

$$\hat{X}_{l+1} = \text{LSTM}_{\text{Layer } l}(\tilde{X}_l) \quad \in \mathbb{R}^{B \times T \times d}$$

### Step 3: Distribution and Convex Mixing

**Stream Distribution:**

$$\tilde{\hat{X}}_{l+1} = \mathcal{H}_l^{post} \cdot \hat{X}_{l+1} \quad \in \mathbb{R}^{B \times T \times n \times d}$$

**Convex Residual Fusion:**

$$X_{l+1} = \mathcal{H}_l^{res} \cdot X_l + \tilde{\hat{X}}_{l+1} \quad \in \mathbb{R}^{B \times T \times n \times d}$$

---

## 4. Architectural Claims

### Claim 1: Decoupled Spatial-Temporal Processing

> The LSTM layer handles 100% of the temporal context processing, while the µHC highway handles 100% of the cross-stream feature stability and depth scaling.

This separation ensures that temporal recurrence and spatial feature mixing are independently optimized, avoiding the entanglement that causes feature degradation in deep stacked LSTMs.

### Claim 2: $O(1)$ Parallelization for Manifold Projection

> By moving Sinkhorn-Knopp outside of the sequential time loop, the projection step scales with the number of layers $L$, not the sequence length $T$, ensuring optimal hardware utilization.

The Sinkhorn-Knopp algorithm (20 iterations) operates on matrices of shape $(n \times n)$ for each element in the $B \times T$ batch, which is trivially parallelizable via `torch.bmm()`.

---

## 5. Data Flow Diagram

```
Input: X_l ∈ ℝ^(B×T×n×d)
         │
         ▼
┌─────────────────────────┐
│  RMSNorm + Flatten      │──→ X_norm ∈ ℝ^(B×T×nd)
└─────────────────────────┘
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
 H_pre (σ)              H_res (Sinkhorn-Knopp)
 ℝ^(B×T×1×n)           ℝ^(B×T×n×n)
    │                         │
    ▼                         │
 Stream Aggregation           │
 H_pre · X_l                  │
    │                         │
    ▼                         │
┌─────────────────┐           │
│   LSTM Layer    │           │
│  (cuDNN opt.)   │           │
└─────────────────┘           │
    │                         │
    ▼                         │
 H_post (σ)                   │
 ℝ^(B×T×n×1)                  │
    │                         │
    ▼                         │
 Stream Distribution          │
 H_post · X̂_(l+1)            │
    │                         │
    ▼                         ▼
┌──────────────────────────────────┐
│   Convex Residual Fusion         │
│   X_(l+1) = H_res · X_l + X̃     │
└──────────────────────────────────┘
         │
         ▼
Output: X_(l+1) ∈ ℝ^(B×T×n×d)
```

---

## 6. Hyperparameter Configuration

| Parameter | Symbol | Default Value | Notes |
|---|---|---|---|
| Number of streams | $n$ | 4 | Physical variable channels |
| Feature dimension | $d$ | Task-dependent | Per-stream hidden size |
| Sequence length | $T$ | 724 | Weather data window |
| Sinkhorn-Knopp iterations | $K$ | 20 | Convergence tolerance for doubly stochastic projection |
| LSTM layers | $L$ | 4 | Depth of the network |
| Activation function | $\sigma$ | Sigmoid | Head output gating |
| Normalization | — | RMSNorm | Applied before head computation |

---

## 7. Target Benchmarks

- **NOAA Weather Dataset** — Multivariate meteorological time series with entangled physical variables
- **ETT (Electricity Transformer Temperature)** — Long-horizon energy forecasting benchmark
- Additional time-series forecasting benchmarks for generalization evaluation
