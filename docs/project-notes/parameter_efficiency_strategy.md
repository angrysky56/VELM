# VELM Parameter Efficiency Strategy: Proving the Architecture

## The Honest Assessment

A 7M parameter model will not beat a 7B model on general benchmarks.
That's not the claim, and pretending otherwise wastes time.

The claim VELM CAN make is architectural: **the same parameter budget,
organized differently, yields measurably better results on specific axes.**
Bruce Lee didn't outbox Ali. He created Jeet Kune Do — a different paradigm
that exploited axes Ali's style couldn't reach.

VELM's axes:
- Compute efficiency (fewer FLOPs per quality unit)
- State tracking (nonlinear memory vs linear approximations)
- Information routing (d-stream topology vs single residual)
- Inference adaptation (test-time compute instead of training-time params)
- Continual learning (retention gates prevent catastrophic forgetting)
- Reasoning density (CIB: same accuracy, fewer tokens)

## What "Punching Above Weight" Actually Means

The CALM paper already proved this at scale:
- CALM-M (371M) matches Transformer-S (281M) at **44% fewer training FLOPs**
- Same quality, dramatically less compute
- The lever: 4x fewer autoregressive steps via K=4 chunk compression

VELM stacks additional efficiency mechanisms on top of CALM:
- Miras deep memory: O(1) state per step (vs O(T²) attention)
- go-mHC routing: d-axis capacity without parameter explosion
- CIB: 32% fewer reasoning tokens at <1% accuracy loss
- qTTT: inference-time adaptation instead of training-time scale

## Formal Efficiency Claims to Prove

### Claim 1: FLOP Efficiency (CALM Compression)
**Hypothesis:** VELM at N params achieves BrierLM comparable to
Transformer at N params while using ~K/2 × fewer inference FLOPs.

**Mechanism:** K=4 chunk compression → 4x fewer autoregressive steps.
Not quite 4x FLOP reduction (autoencoder overhead), but CALM paper
shows 34% inference FLOP reduction at matched quality.

**Benchmark:** BrierLM per inference FLOP, compared against standard
Transformer of same parameter count, same training data.

**Falsification:** If VELM's BrierLM/FLOP is worse than Transformer at
matched scale, the compression overhead exceeds the step reduction benefit.

### Claim 2: State Tracking (Miras vs Linear RNN)
**Hypothesis:** On synthetic state tracking tasks (multi-step arithmetic,
variable binding, bracket matching), Miras-based model matches or exceeds
linear RNN (Mamba, GLA, DeltaNet) at 2-4x the parameter count.

**Mechanism:** Miras uses nonlinear MLP memory with learned retention.
Linear RNNs compress state linearly — they provably cannot track state
that requires multiplicative interactions.

**Benchmark:** Accuracy on MQAR (multi-query associative recall),
state tracking synthetic tasks from Miras paper (Table 4).

**Falsification:** If linear RNNs match Miras at same param count on
state tracking tasks, the nonlinear memory adds complexity without benefit.

### Claim 3: Routing Expressivity (go-mHC d-axis)
**Hypothesis:** At fixed parameter count, d=4 go-mHC streams yield
lower next-vector prediction loss than d=1 (standard residual).

**Mechanism:** go-mHC adds d(d-1)/2 × s² ≈ 28 params per layer for
d=4, s=2. The doubly stochastic routing provides information flow
patterns (cyclic permutations, directed advection) that single-stream
residuals cannot express.

**Benchmark:** Energy loss on held-out data, VELM-d=4 vs VELM-d=1,
same total params (adjust hidden_dim to compensate for HC overhead).

**Ablation:** Remove go-mHC (set hc_streams=1), retrain, compare.
This isolates the d-axis contribution from other architecture changes.

### Claim 4: Continual Learning (Miras Retention Gates)
**Hypothesis:** Miras retention gates (KL divergence regularization)
reduce catastrophic forgetting compared to standard Transformer
fine-tuning, without requiring replay buffers or EWC.

**Mechanism:** The update rule W_t = Softmax(α_t log(W_{t-1}) - η_t ∇ℓ)
inherently balances new learning (η_t term) with retention (α_t term).
This is structurally analogous to Elastic Weight Consolidation but
learned per-channel, not per-parameter.

**Benchmark:** Train on domain A (math), then domain B (narrative).
Measure accuracy retention on domain A after domain B training.
Compare VELM-Miras vs Transformer baseline vs Mamba baseline.

**Falsification:** If Transformer with EWC matches Miras retention
at same param count, the architectural advantage is illusory.

### Claim 5: Reasoning Density (CIB Compression)
**Hypothesis:** CIB-trained VELM produces reasoning chains that are
30-40% shorter while maintaining >98% accuracy, because chunk-level
compression (K=4) amplifies CIB's per-token savings.

**Benchmark:** MATH500, GSM8K — measure accuracy × compression factor.
Compare CIB on VELM vs CIB on standard Transformer at same scale.

## The Teacher Model Lever: Knowledge Distillation via CALM

Here's the key tactical insight most people miss: **we're already using
Qwen3.5-0.8B as our tokenizer.** That 0.8B model can also serve as
a teacher for knowledge distillation into VELM's backbone.

Standard distillation: student learns soft labels from teacher.
VELM distillation: student learns **continuous vector representations**
from teacher, because CALM operates in continuous latent space.

Formally, the distillation objective becomes:

  L_distill = E_z~teacher[ ||f_VELM(z_{1:i-1}) - z_i^teacher||² ]

where z_i^teacher = AE_encode(Qwen_hidden_state_i) — we encode the
teacher's hidden states through our autoencoder to get target vectors
in VELM's latent space.

This is uniquely possible because VELM doesn't predict discrete tokens
during backbone training — it predicts continuous vectors. The teacher's
internal representations become the training signal directly. No soft
labels, no KL divergence over vocabularies, no temperature scaling.

**Practical efficiency:** The 0.8B Qwen model fits on the same T4 GPU.
We run it once to generate target vectors for the training data, save
them, then train VELM's backbone to predict those vectors. This gives
VELM the benefit of 0.8B parameters worth of learned representations
distilled into a 7M parameter backbone + head.

## The MoE Lever: Sparse Activation for Future Scaling

For future versions beyond the current proof-of-concept: Mixture of
Experts (MoE) is the single most effective "punch above weight" technique.

A model with 4 experts of 7M params each = 28M total params but only
7M active per forward pass. This gives the model access to 4x more
learned capacity while keeping inference cost constant.

MoE composes naturally with VELM:
- EGGROLL can evolve expert selection (gradient-free routing decisions)
- go-mHC d-streams could map to expert groups (stream 1→expert 1, etc.)
- GEA can evolve diverse expert specializations

## Proving It: Concrete Experimental Protocol

### Phase A: Architecture Validation (Current Scale, 7M backbone)

**A1. Complete EGGROLL Training**
Let the current progressive EGGROLL run finish. This validates:
- EGGROLL can train VELM's architecture (gradient-free on nonlinear RNN)
- go-mHC d=4 doesn't break convergence
- Progressive unfreezing works at small scale

**A2. Ablation Grid**
Train 4 variants, same data, same total param budget:
| Variant | Backbone | Residual | Training |
|---------|----------|----------|----------|
| baseline | Transformer (8L, 256d) | standard | Adam |
| velm-miras | Miras hybrid (4M+4S) | standard | EGGROLL |
| velm-hc | Miras hybrid | go-mHC d=4 | EGGROLL |
| velm-full | Miras hybrid | go-mHC d=4 | EGGROLL+GEA |

Compare: BrierLM, energy loss, inference FLOPs, training time.
The Transformer baseline uses standard backprop — this is the fair
comparison to show EGGROLL produces competitive results without gradients.

**A3. State Tracking Synthetic Tasks**
MQAR (multi-query associative recall), bracket matching, variable binding.
These tasks specifically test nonlinear state tracking — where Miras
should dominate linear RNNs. Use synthetic data, measure accuracy vs
sequence length. Include Mamba/GLA baselines at same parameter count.

### Phase B: Knowledge Distillation (0.8B → 7M)

**B1. Teacher Vector Generation**
Run Qwen3.5-0.8B on training data. For each K=4 chunk, extract
hidden states from the teacher's last layer. Encode through VELM's
frozen autoencoder to get target vectors in the latent space.

**B2. Distillation Training**
Train VELM backbone to predict teacher's vectors (not just
next-chunk vectors from the AE). The loss becomes:
  L = α × L_energy(self-predicted) + β × L_distill(teacher-predicted)

**B3. Compare Distilled vs Self-Supervised**
The key metric: does distillation from 0.8B teacher give VELM's 7M
backbone quality that would normally require a 50-100M backbone?
If yes, VELM achieves "effective" 0.8B quality at 7M inference cost
(because CALM's K=4 compression further reduces inference FLOPs).

### Phase C: Continual Learning Validation

**C1. Sequential Domain Training**
Train on 3 domains in sequence: math → narrative → code.
After each domain, evaluate retention on all previous domains.
Compare VELM's inherent retention (Miras gates) vs:
- Transformer + naive fine-tuning (catastrophic forgetting baseline)
- Transformer + EWC (state of the art continual learning)
- Mamba + naive fine-tuning (linear RNN baseline)

**C2. The Retention Gate Hypothesis**
Miras's α_t gate controls how much of the previous memory state W_{t-1}
is retained. During continual learning, if the model encounters data from
a previously-learned domain, the retention gate should increase (preserve
what was learned). On new domain data, the gate should decrease (learn
new associations). This is testable: log α_t values during training and
verify this pattern.

## What This Achieves for Edge/Commodity GPU

The compound efficiency gains:
1. CALM K=4: 4x fewer autoregressive steps → ~34% inference FLOP reduction
2. go-mHC d=4: richer routing at ~0.1% param overhead
3. CIB: 30-40% shorter reasoning chains → proportional FLOP savings
4. qTTT: test-time adaptation instead of training-time param scale
5. Distillation: 0.8B quality at 7M inference cost
6. EGGROLL int8: 2-4x memory reduction, 4x energy reduction

Multiplicative: 0.34 × 0.65 × (7M/800M inference) = radically lower
cost per quality unit than standard approaches.

A 7M VELM model running in int8 on an RTX 3060 processes language at
a fraction of the cost of serving a 0.8B Qwen model — while potentially
matching its quality on domain-specific tasks via distillation.

## What We're NOT Claiming

- VELM does NOT match frontier models on general benchmarks
- VELM does NOT replace large models for open-ended generation
- VELM DOES demonstrate that architectural innovation yields
  measurable efficiency gains on specific, testable axes
- VELM DOES provide a proof-of-concept for combining 7 recent
  innovations into a coherent system
- Each claim has a falsification condition — if the ablation shows
  no benefit, the component is bloat and gets removed

## Priority Order

1. **Finish current EGGROLL training** — validate the training pipeline works
2. **Run ablations** (A2) — isolate which components actually help
3. **Knowledge distillation** (B1-B3) — the single biggest quality lever
4. **Continual learning eval** (C1-C2) — the strongest differentiation story
5. **Publish findings** — positive or negative, the ablations ARE the paper
