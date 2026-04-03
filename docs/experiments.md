# VELM Experimental Plan

## Overview

Experiments are structured in 4 phases, each building on the previous.
Phase 1 can run on a single RTX 3060 12GB. Later phases require more compute.

## Phase 1: Foundation (Single GPU — Your RTX 3060)

### Experiment 1.1: CALM Autoencoder Reproduction
**Goal:** Train and validate the CALM autoencoder independently.

**Setup:**
- Dataset: OpenWebMath (streamed, ~1M tokens for constrained hardware)
- Tokenizer: Qwen3.5 (248K vocab, 201 languages)
- K = 4 (token chunk size)
- Latent dimension l = 64 (gpu_12gb config), l = 128 (full-scale configs)
- Regularization: KL clipping (β=0.001), DropToken, DropLatent
- Stability: logvar clamped to [-20, 2], gradient clipping (norm=1.0),
  logit scaling by 1/√d (required for 248K vocab numerical stability)

**Metrics:**
- Reconstruction accuracy (target: >99.9%)
- Latent space smoothness (KL divergence distribution)
- Downstream BrierLM score correlation

**Compute:** ~10 GPU-hours on RTX 3060

### Experiment 1.2: Miras Backbone Validation (Backprop)
**Goal:** Verify Miras deep memory works over continuous CALM vectors.

**Setup:**
- Frozen CALM autoencoder from 1.1
- Miras-Memora variant (MLP memory, ℓ2 bias, KL retention)
- 12 layers, dim 768, 16 heads (170M params)
- Standard backprop training (SGD/AdamW)
- Hybrid: 6 Miras + 6 SWA layers

**Metrics:**
- BrierLM on WikiText-103
- Comparison: pure SWA vs pure Miras vs hybrid
- Memory utilization analysis (what does the MLP memory learn?)

**Compute:** ~50 GPU-hours on RTX 3060

### Experiment 1.3: EGGROLL Training of Miras (No Backprop)
**Goal:** Demonstrate EGGROLL can train the Miras backbone without gradients.

**Setup:**
- Same architecture as 1.2
- EGGROLL: rank r=1, σ=0.001
- Population sizes: 2^10, 2^12, 2^14, 2^16
- Fitness: BrierLM on validation set
- Compare against backprop baseline from 1.2

**Key question:** At what population size does EGGROLL match backprop?
(EGGROLL paper found pop=2^20 beat backprop for EGG, but Miras is more complex)

**Compute:** ~200 GPU-hours on RTX 3060 (pop=2^16, larger needs multi-GPU)

### Experiment 1.4: Int8 VELM-Tiny
**Goal:** Train VELM entirely in int8 datatypes.

**Setup:**
- Miras backbone with all weights in int8
- Activation functions removed (int8 saturation = implicit nonlinearity)
- EGGROLL with discretized Adam updates
- Character-level prediction on minipile (following EGG methodology)

**Metrics:**
- Test loss (bits/byte) vs EGG baseline
- Test loss vs fp32 Transformer backprop baseline
- Training speed (tokens/sec)

**Compute:** ~100 GPU-hours on RTX 3060

## Phase 2: Integration (Multi-GPU)

### Experiment 2.1: Full VELM-Small with CIB
**Goal:** Train complete VELM with compression-aware fitness.

**Setup:**
- VELM-Small: 24 layers, dim 1024, 340M params
- EGGROLL pop=2^18, rank r=1
- Fitness = α×BrierLM + β×CIB_compression_score
- Training data: 15B tokens from Pile
- CIB: λ sweep [0.01, 0.05, 0.1, 0.2] to find quality-compression Pareto

**Metrics:**
- BrierLM vs Transformer-S, Mamba2-S, GatedDeltaNet-S (matched params)
- Reasoning benchmarks: MATH500, GSM8K
- Token efficiency: accuracy vs reasoning tokens used

**Compute:** ~2000 GPU-hours (4x A100 or equivalent)

### Experiment 2.2: qTTT on VELM
**Goal:** Validate query-only TTT for VELM's hybrid architecture.

**Setup:**
- Trained VELM-Small from 2.1
- Long-context eval: LongBench-v2, ZeroScrolls
- qTTT: N_qTTT=32, k=128, only on SWA layers
- Baselines: vanilla inference, 8K thinking tokens (FLOP-matched)

**Metrics:**
- Accuracy across all LongBench-v2 subsets
- Attention mass on target tokens (with/without qTTT)
- Comparison: VELM-qTTT vs pure Transformer-qTTT vs Miras-only

**Key question:** Does the Miras recurrent state complement qTTT's attention
fix? Hypothesis: hybrid VELM + qTTT > pure Transformer + qTTT because the
Miras state captures global patterns that attention misses entirely.

**Compute:** ~500 GPU-hours (inference-heavy, can run on fewer GPUs)

## Phase 3: Scaling (Large Compute)

### Experiment 3.1: VELM-Medium Full Training
**Goal:** Validate VELM at meaningful scale.

**Setup:**
- VELM-Medium: 24 layers, dim 1536, 780M params
- EGGROLL pop=2^20, 30B tokens
- Full pipeline: CALM → Miras+SWA → Energy Head → CIB fitness
- Compare against Transformer++ 780M, Mamba2 780M, CALM-L (original)

**Metrics:**
- Full benchmark suite: WikiText-103, LAMBADA, HellaSwag, WinoGrande,
  PIQA, ARC-E, ARC-C, BoolQ
- Long-context: LongBench-v2, ZeroScrolls, RULER
- Reasoning: MATH500, AIME24, OlympiadBench
- Needle-in-haystack at various context lengths

**Compute:** ~10,000 GPU-hours (8x H100)

### Experiment 3.2: Scaling Laws
**Goal:** Characterize VELM's scaling behavior along three axes.

**Axes:**
1. Model size: 170M, 340M, 780M, 1.5B (4 points)
2. Population size: 2^14, 2^16, 2^18, 2^20 (4 points)
3. Chunk size K: 1, 2, 4, 8 (4 points)

**Analysis:**
- VELM scaling law: perf = f(params, pop_size, K, data)
- Compare with Chinchilla scaling (params × data)
- Identify the compute-optimal (params, pop_size, K) triple
- Does semantic bandwidth K interact with model size?

**Compute:** ~30,000 GPU-hours total (many configurations)

## Phase 4: Self-Improvement (Exploratory)

### Experiment 4.1: GEA-EGGROLL Integration
**Goal:** Demonstrate continuous self-improvement via group evolution.

**Setup:**
- Pre-trained VELM-Small from Phase 2
- GEA group size K=5 agents (EGGROLL population subsets)
- Evolution iterations: 30 (following GEA paper methodology)
- Evaluation: reasoning benchmarks after each iteration

**Experience sharing mechanism (adapted from GEA):**
1. Each agent = VELM with different EGGROLL weight perturbation history
2. Evolutionary traces = (perturbation directions, fitness trajectories,
   reasoning patterns that solved hard problems)
3. Reflection: analyze which directions improved which capabilities
4. Evolution: bias next-generation perturbations toward promising subspaces
5. Selection: Performance-Novelty criterion across agents

**Metrics:**
- Fitness trajectory over iterations (monotone improvement?)
- Diversity maintenance (novelty scores over time)
- Transfer: does improvement on math transfer to code? To QA?
- Comparison: GEA group evolution vs independent EGGROLL populations

**Compute:** ~5,000 GPU-hours (many parallel model evaluations)

## Ablation Study Matrix

The core ablation incrementally adds each component to measure marginal value:

| # | Configuration | Components | What it tests |
|---|--------------|------------|---------------|
| A0 | Transformer baseline | Standard LM | Reference point |
| A1 | CALM-Transformer | + CALM AE | Value of continuous vectors |
| A2 | CALM-Miras (backprop) | + Miras backbone | Value of deep memory |
| A3 | CALM-Miras (EGGROLL) | + gradient-free training | Can ES match backprop? |
| A4 | A3 + int8 | + native quantization | Quality cost of int8 |
| A5 | A3 + qTTT | + inference adaptation | Long-context gains |
| A6 | A3 + CIB | + reasoning compression | Efficiency gains |
| A7 | Full VELM | All components | Complete system |
| A8 | A7 + GEA | + self-improvement | Continuous improvement |

Each row is evaluated on the same benchmark suite to enable fair comparison.

## Benchmark Suite

### Language Modeling
- WikiText-103 (BrierLM + perplexity)
- LAMBADA (accuracy)

### Common Sense Reasoning
- HellaSwag, WinoGrande, PIQA, ARC-Easy, ARC-Challenge, BoolQ

### Long Context
- LongBench-v2 (6 subsets: Code, Dialogue, Structured, In-Context, MultiDoc QA, SingleDoc QA)
- ZeroScrolls (GovReport, MuSiQue, NarrativeQA, QASPER, QuALITY, SQuALITY)
- RULER (synthetic needle-in-haystack at various lengths)

### Mathematical Reasoning
- MATH500, GSM8K, AIME24, AMC-23, OlympiadBench

### Efficiency Metrics
- Tokens/second (inference throughput)
- Reasoning tokens per correct answer (CIB efficiency)
- GPU memory usage (int8 vs fp16 vs fp32)
- Training cost (GPU-hours to reach target performance)

## Implementation Priority

For a resource-constrained setting (1x RTX 3060 12GB), the recommended
execution order is:

1. **CALM autoencoder** — Small model, standard backprop, validates the
   representation layer independently. ~10 hours.

2. **Miras backbone (backprop)** — Validates deep memory over continuous
   vectors. Uses existing training infrastructure. ~50 hours.

3. **EGGROLL integration** — The critical experiment: can ES train Miras
   without backprop? Start with small populations and scale up. ~200 hours.

4. **Int8 path** — Once EGGROLL works, try pure int8. High risk, high reward.

5. **qTTT inference** — Requires a trained model. Fast to validate. ~10 hours.

6. **CIB** — Modify the fitness function. Requires re-training. ~200 hours.

7. **GEA** — Requires multiple model instances. Most compute-intensive.

Phase 1 (items 1-4) is achievable on your hardware. Phase 2+ needs cloud
compute or institutional access.
