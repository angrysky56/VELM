# POC Findings: Gradient-Free Training of the VELM Backbone

**Date:** 2026-07-04
**Experiment:** `notebooks/poc_eggroll_vs_backprop.ipynb` (v1–v4)
**Question:** Can EGGROLL-style evolution strategies train the Miras backbone to match a backprop baseline on identical model, data, and loss?
**Answer:** **No — not at consumer scale.** ES learns, scales with population, and refines from good basins, but is 1–2 orders of magnitude less efficient than backprop. The defensible VELM architecture is a **hybrid**: backprop-trained core, ES reserved for GEA self-improvement.

---

## 1 · Experimental design

All four iterations shared the same skeleton: a tiny VELM backbone (real `src/` code, ~1.5M params: 2 Miras + 2 SWA blocks, dim 128) plus a linear head, trained two ways from an *identical* initialization on *identical* data with an *identical* MSE loss:

- **Backprop**: AdamW, lr 1e-3, grad-clip 1.0
- **EGGROLL**: antithetic rank-1 low-rank ES (repo `perturb_pytree`), Adam on the z-scored ES gradient, adaptive σ ∈ [3e-4, 5e-3], grad-clip 1.0

Data: TinyStories, K=4 token chunks, 128-chunk sequences. The pop-512 implementation used two-pass chunked evaluation with **noise regenerated from RNG keys** (the EGGROLL paper's memory design), verified bit-equivalent to naive single-pass evaluation.

## 2 · Iteration history — how the task was debugged

| Version | Task / change | Backprop | EGGROLL | Lesson |
|---|---|---|---|---|
| v1 | Predict next chunk's pooled *random* embedding | 0.0038 (= mean baseline 0.0037) | 0.167 (diverged after σ collapse) | Target was ~pure irreducible entropy: mean baseline equals target variance 1/(e·K). Task failure, not optimizer failure. |
| v2 | Teacher distillation: distilgpt2 contextual hidden states at chunk boundaries, R² = 1−MSE | **0.63** after escaping a ~900-step plateau | 1.00 — stuck at trivial mean solution (pop 32) | Task learnable. Pop 32 in a 1.5M-dim space is starvation (paper uses up to 262,144). |
| v3 | Pop 512 (1024 evals/step, 3000 steps) + warm-start diagnostic | 0.985 — *failed to escape the plateau this time* | **0.9795 — below backprop**, still descending; warm-start captured 92% of post-plateau gain | Plateau escape is bifurcation-flaky → the plateau is a task artifact. First positive ES signals: population scaling works; ES follows local signal. |
| v4 | Student inputs = teacher wte embeddings (semantic geometry in); linear-probe baseline; 2 backprop seeds; 4000 steps | **0.370, R² 0.628** (probe: 0.765, R² 0.229) | 0.912, R² 0.082; warm-start 0.591→0.532 (27% of gain) then stalled | Clean verdict — see below. |

## 3 · v4 headline numbers

```
mean baseline          : 0.9927
linear probe (no mem)  : 0.7652   R² 0.229
backprop  best eval    : 0.3697   R² 0.628   (best of 2 seeds)
ES-XL     best eval    : 0.9117   R² 0.082   (pop 512, 4000 steps, ~4M evals)
R² ratio (ES-XL/BP)    : 0.130
ES-warm from plateau   : 0.591 → 0.532 (27% of backprop's post-plateau gain, then stalled)
```

## 4 · Findings

**F1 — The Miras backbone trains fine with backprop.** R² 0.628, nearly 3× the no-memory linear probe: the backbone demonstrably uses sequence memory, and JAX autodiff handles the nonlinear recurrence without difficulty at this scale. This *falsifies the founding premise* that the architecture needs gradient-free training ("EGGROLL eliminates BPTT" — BPTT was never the binding constraint here).

**F2 — ES population scaling is real but brutally sublinear.** Pop 32 → 512 (16×) moved ES from hard-stuck-at-baseline to R² 0.082. Extrapolating the trend, parity with backprop plausibly requires the paper's 10⁴–10⁵ populations — datacenter compute, not a T4 or an RTX 3060. Chronic σ-ceiling pinning throughout all runs shows fitness diversity (signal-to-noise), not step size, is the binding constraint.

**F3 — ES refines but cannot explore.** Warm-started from backprop's plateau checkpoint, ES immediately captured 27–92% (v4/v3) of the available local gain, then stalled at a noise floor. ES is usable as a *local* optimizer where gradients are unavailable; it is not competitive as a *pretraining* optimizer at accessible population sizes.

## 5 · Architectural consequence: hybrid VELM

| Component | Original plan | Revised plan | Status |
|---|---|---|---|
| CALM autoencoder | Backprop | Backprop (unchanged) | ✅ Done — 99.9% reconstruction |
| Miras backbone + energy head | EGGROLL | **Backprop distillation / energy loss** | Proven viable by POC (minutes, not hours) |
| qTTT long-context | Inference-time | Unchanged | Planned |
| CIB reasoning compression | ES fitness term | Backprop loss term | Planned |
| GEA self-improvement | EGGROLL populations | **ES stays here** — fitness (task success, workflow quality) is genuinely non-differentiable, and F3 shows ES works given a good basin | Planned |

This is a *stronger* story than the original: each optimizer is used exactly where it is the right tool, and the claim is backed by a four-iteration ablation.

## 6 · Threats to validity

- Single architecture size (~1.5M params); ES/backprop gap could differ at other scales (though the direction of the population-starvation argument worsens with scale).
- The true EGGROLL forward-pass trick (`Wx + σA(Bᵀx)`) was not implemented; at 1.5M params two-pass regeneration is compute-equivalent, but a pop-16k+ run remains untested for completeness.
- MSE distillation objective, not the energy score; chosen deliberately to give both optimizers a smooth deterministic loss.
- TinyStories only; K=4 only.

## 7 · Reproduction

Open `notebooks/poc_eggroll_vs_backprop.ipynb` in Colab (badge at top), Runtime → T4 GPU, Run All (~1.5–2h). The jupytext `.py` is the source of truth; regenerate the notebook with `jupytext --to ipynb notebooks/poc_eggroll_vs_backprop.py`.
