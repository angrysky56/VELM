# Phase 3 Findings: Training the Hybrid VELM Loop

**Dates:** 2026-07-04 → 2026-07-05
**Notebooks:** `phase3_hybrid_train.ipynb`, `phase3_input_ablation.ipynb`, `ae_latent_diagnostics.ipynb`
**Companion:** `poc_findings.md` (the EGGROLL go/no-go that produced the hybrid architecture)

**Result:** the full hybrid VELM loop — frozen CALM AE → Miras backbone → conditional energy head — trains end-to-end on consumer/Colab hardware and learns genuine contextual next-latent prediction: centered cosine **0.218** (vs 0.130 linear-oracle bar), decoded token accuracy **3×** unconditional, and a conditional energy head that beats the unconditional-sampler baseline. Getting there took **nine consecutive failed runs**, and the reasons they failed are the useful part of this document.

---

## 1 · The setup

Frozen CALM AE (99.9% reconstruction) encodes K=4 token chunks into 128-d latents `z`. A Miras backbone (dim 256, 4+4 blocks, ~15M params) reads token embeddings compressed per chunk and predicts `z_{t+1}` — via a deterministic linear "direct head" and a stochastic energy head trained with the energy score (CALM Eq. 10). Data: TinyStories, 2–4M tokens. Primary metric: **centered cosine** on a fixed eval set — the cosine after removing the global mean latent, on which any unconditional predictor scores exactly 0. Reference bar: a ridge regression from the previous 4 true latents ("linear oracle", ≈ 0.13).

## 2 · The nine-run plateau, and what each failure taught

| Run(s) | Change tested | Centered cos | Lesson |
|---|---|---|---|
| 1 | energy + aux-cosine loss, raw latents | (metric didn't exist) | Beat the *copy* baseline while being a near-unconditional predictor; identical decoded text across contexts exposed it. **Always include the unconditional baseline for every metric.** |
| 2 | + aux loss, collapse detector, 2× data | 0.074 → later shown irreproducible | Anti-collapse aux worked, but the gain was a **rare stochastic escape**, not a recipe (cf. the same bifurcation flakiness in `poc_findings.md` v2-vs-v3). |
| 3 | resumed weights, more steps | 0.005 | **Weights-only resume is destructive**: fresh Adam + re-warmed LR bulldozed the contextual features (≈1% of loss — first destroyed, last relearned). Multi-session training needs *full* state: params + optimizer moments + schedule step. |
| 4–5 | standardized latent targets | 0.074 → regressed | Standardization was necessary (see §3) but not sufficient; the resume bug ate the gains. |
| 6–8 | TF32 precision, LR horizon, batch size, weight decay | 0.003–0.017, flat | **All hyperparameter hypotheses falsified.** Eight runs converging on one plateau means the bug is structural, not a knob. |

## 3 · Diagnostics that localized the bug

**AE exoneration** (`ae_latent_diagnostics`): the latent space is smooth (1-token-changed chunks keep cosine 0.65 vs 0.15 for random pairs) and linearly predictable (ridge oracle 0.125 vs chance −0.001). The autoencoder was never the problem — but the oracle finding *25× more signal than the trained 15M-param model* proved the signal existed and training wasn't reaching it.

**Anisotropy, the named villain**: raw CALM latents carry a dominant shared mean direction that soaks up nearly all cosine/energy gradient — the rank-1 case of *representation anisotropy* (Ethayarajh 2019; LatentGate, Ratnakar et al. ACL 2026). Standardizing targets ((z−μ)/σ, frozen stats) removes it; conveniently, standardized 128-d latents have norm ≈ √128, matching the energy head's output sphere.

**Boundary result — whitening rejected**: LatentGate's stronger remedy (full PCA/ZCA whitening, +17.2 pts in their routing ablation vs +8.8 for standardization) *hurts* here: ridge oracle 0.125 standardized → 0.075 whitened. Our latents are only mildly anisotropic (top-3 PCs: 16% of variance), and the predictable structure concentrates in **high-variance** components, which whitening deamplifies relative to noise dims. Anisotropy correction transfers from classification to regression; the full-whitening dose does not.

**The input ablation** (`phase3_input_ablation`) — four models, one identical pure-MSE loop:

| Arm | Input → model | Centered cos |
|---|---|---|
| A | prev-4 true latents → linear | 0.109 (≈ oracle — loop sound) |
| B | prev-4 true latents → MLP | **0.229** (2× oracle — rich nonlinear structure) |
| C | true latent sequence → Miras backbone | 0.005 (dead; RMS-norm fix also dead) |
| D | tokens → `compress_input` → Miras backbone | **0.128, climbing** |

Arm D is the same architecture that flatlined for eight runs — under **pure MSE at lr 1e-3** it reaches the oracle in 3k steps. **Verdict: the energy score's stochastic 8-sample gradient was drowning the ~1% contextual signal.** Not featurization, not the backbone, not the data.

**Open anomaly (arm C):** the backbone learns from high-dimensional categorical inputs but not from compact continuous inputs in the target space — even normalized, and even though an MLP extracts 0.229 from the same vectors. Suspects: LTI anchor injection (raw input mixed unnormalized every loop), memory write dynamics on temporally correlated low-dim streams. Parked; not on the critical path because VELM's inference loop round-trips through decoded *tokens* (the working D pathway).

## 4 · The staged recipe that works

1. **Stage 1 — representation learning (`objective: "mse"`)**: backbone + direct head, pure MSE on standardized latents, lr 1e-3 warmup-cosine, wd 0, full-state checkpointing, fixed eval set. Result: centered cos 0.167 @ 8k steps → **0.218 @ 16k** (saturated with the schedule); token acc 0.075 vs 0.025 unconditional; healthy diversity 0.52.
2. **Stage 2a — conditional head (`objective: "energy_frozen"`)**: energy head only, gradients stopped at the hidden states (representations untouchable), own LR cycle and state file. Result: eval energy **1.045 < 1.123** unconditional-sampler bar; head cosine 0.002 → **0.161**; NN-snapped samples become story-shaped and position-appropriate.
3. **Stage 2b — joint finetune (`objective: "energy+aux"`, optional)**: low LR, to close the 0.161 → 0.218 head-representation gap. Best attempted after data scaling.

Engineering rules learned the hard way: full training-state serialization (params + Adam moments + schedule step) per stage; fixed stream-order eval set so baselines and oracle bars are stable across runs and data scales; frozen standardization stats; per-stage LR cycles (never resume trained weights into a fresh warmup).

## 5 · Honest caveats

- **Context parity, not context advantage**: the backbone with 63 chunks of context ≈ the 4-chunk MLP (0.218 vs 0.229). On TinyStories at 4M tokens, extractable next-chunk signal is mostly local. Whether Miras memory pays for itself is *the* open scaling question.
- **Raw decode is still token soup** — expected: predictions must land on the latent manifold before the AE decoder emits coherent text, and CALM-quality generation sits orders of magnitude away in data/params. NN-snap (project prediction to nearest training latent) is the honest qualitative probe at this scale.
- Single dataset, single AE, K=4, one model size. The arm-C anomaly is unexplained.

## 6 · What's next

Data scaling (16M+ tokens, optional wikitext mix) with a fresh v2 LR cycle; autoregressive rollout demo measuring per-step quality decay (compounding error — CALM's known hard part); stage 2b; then qTTT + CIB, and GEA with ES where fitness is non-differentiable (per `poc_findings.md`).
