# OK-DMD: Online Koopman Dynamic Mode Decomposition for KV-Cache Eviction

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**OK-DMD** is a mathematically grounded, query-agnostic KV-cache compression and eviction framework for Long-Context Large Language Models (LLMs). By treating the sequence of attention Key vectors as a physical trajectory in a semantic space, OK-DMD tracks and preserves the stable mathematical "attractors" of the text, preventing catastrophic context forgetting during long generations.

---

## The Core Concept: Language as a River

Traditional KV-cache eviction policies (like H2O or local windowing) treat attention keys as static, isolated vectors. They evaluate importance based on rolling attention scores. 

However, attention scores are highly volatile and subject to **attention drift**—a token ignored now might be vital 100 steps later when the conversation shifts back.

**OK-DMD shifts this paradigm:**
Think of the text flow as a river. Every new word is a drop of water following a current. Instead of analyzing individual drops, OK-DMD estimates the **underlying current (the semantic attractor)** using **Koopman Operator Theory**. 

We assume that the transition from one Key state ($k_{t-1}$) to the next ($k_t$) is governed locally by a linear transition operator $A_t$:

$$k_t \approx A_t k_{t-1}$$

By calculating the persistent eigenvalues of $A_t$ in real-time, we identify which historical tokens represent the global structure of the document (stable modes) and which ones are transient noise (local syntax or conversational tangents) that can be safely evicted.

---

## Domain Shift Resilience: The "Lasagna Trap"

Standard eviction heuristics (like H2O) suffer from catastrophic forgetting during domain shifts. 

If your LLM is processing a long document about **Astronomy**, and the user suddenly inserts a transient **Lasagna Recipe**, H2O's local attention focus shifts entirely to cooking terms. To fit the strict memory budget, H2O purges the older Astronomy keys. When the conversation returns to Astronomy, the model's memory of the primary domain is gone, leading to hallucinations.

Because **OK-DMD** tracks the spectral footprint (the persistent eigenvalues) of the global trajectory, the "Astronomy" subspace remains stored in the background of the Koopman operator. Transient noise is allowed to pass through and get evicted, while the primary domain is shielded.

---

## Installation

You can install the package locally in editable mode:

```bash
git clone https://github.com/UnK-Center-Inc/ok-dmd.git
cd ok-dmd
pip install -e .
```

---

## Quick Start & Benchmarking

To reproduce our multi-domain stress test (Astronomy $\rightarrow$ Lasagna $\rightarrow$ Astronomy Recall) and verify how OK-DMD preserves the global attention manifold compared to H2O:

```bash
python benchmarks/run_stress_test.py
```

### Expected Output:
Under an aggressive **92.2% compression budget** (restricting the KV-cache of Qwen-2.5-0.5B from 817 down to just 64 active tokens), OK-DMD consistently outperforms attention-accumulating baselines:

```text
------------------------------------------------------------
 FINAL PERFORMANCE REPORT: DOMAIN SHIFT STRESS TEST
 Total Sequence Length:    817 tokens
 KV-Cache Budget Limit:    64 tokens (Aggressive Compression)
------------------------------------------------------------
 -> Mean Attention Coherence (H2O Baseline): 0.997470
 -> Mean Attention Coherence (OK-DMD Ours):  0.998259
------------------------------------------------------------
[SUCCESS] OK-DMD outperformed H2O! The dynamic attractor preserved the old Astronomy context.
```

*(You can save your generated plot as `assets/benchmark_results.png` and reference it here to visually showcase the stability of the OK-DMD curve).*

---

## Mathematical Architecture

OK-DMD runs on three integrated mathematical components:

### 1. Causal Recursive Update (O(d²))
To update the Koopman matrix $A_t$ at every generation step without expensive matrix inversions ($O(d^3)$), we apply the **Sherman-Morrison** formula to update the inverse covariance matrix $P_t$ recursively:

$$u_t = P_{t-1} k_{t-1}$$

$$g_t = \frac{u_t}{\lambda + k_{t-1}^T u_t + \eta}$$

$$P_t = \frac{P_{\text{next}} + P_{\text{next}}^T}{2} + \sigma I$$

$$A_t = A_{t-1} + e_t g_t^T$$

*   $\lambda = 0.995$ acts as an exponential forgetting factor.
*   $\sigma I$ is a Tikhonov diagonal loading step that guarantees numerical stability under **FP16** or **BF16** GPU precisions.

### 2. Warm-Started QR Subspace Iteration
Instead of performing complex eigendecompositions, we extract the orthonormal basis $Q_t \in \mathbb{R}^{d \times k}$ representing the $k$ most persistent Schur modes using a single step of QR iteration initialized by the previous step's subspace:

$$Y = A_t Q_{t-1}$$

$$Q_t, R = \text{QR}(Y)$$

### 3. Topological Eviction Scoring
The survival score $S_i$ for any token $i$ in the cache represents the energy of its key projection onto the persistent subspace:

$$S_i = \| Q_t^T k_i \|_2^2$$

---

## Memory Break-Even Analysis

Maintaining the state matrices $P$ and $A$ introduces a tiny memory overhead of **41 MB** per active sequence (for Llama-3-70B with 80 layers and 8 KV heads). 

By analyzing the memory trade-offs, we determine the exact sequence length $N$ where OK-DMD starts saving VRAM under a compression rate of $1-C$ (where we evict $1-C$ of tokens):

$$N > \frac{40,960 \text{ KB}}{(1 - C) \cdot 320 \text{ KB}}$$

*   **50% Eviction:** OK-DMD is memory-profitable for any sequence longer than **256 tokens**.
*   **80% Eviction:** OK-DMD is memory-profitable for any sequence longer than **160 tokens**.

---

## Limitations

As a mathematically rigorous project, we explicitly outline the core limitation of OK-DMD:

*   **The Needle in a Haystack (NIAH) Constraint:** OK-DMD is designed to extract global semantic trends. If a completely out-of-context semantic anomaly (a needle) is mentioned briefly and never recalled, OK-DMD will mathematically classify it as a transient perturbation (noise) and evict it under tight budgets.
*   **The Production Solution:** In enterprise RAG architectures, OK-DMD should not compress the cache step-by-step during prompt prefilling. The prompt and the final instruction should be parsed in parallel, and a **Prompt-Aware filter** (like SnapKV) should lock the needle tokens. OK-DMD is then deployed during the long answer generation phase (*Decode*) to prevent semantic drift.

---

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

Developed with ☕ by **UnK Center Inc.**
