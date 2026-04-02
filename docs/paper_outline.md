# VELM Paper Outline

## Title (working)

**VELM: Toward Self-Evolving Language Models via Continuous-Latent
Gradient-Free Architecture**

## Authors
- TBD

## Abstract (~250 words)

We present VELM (Vector-Evolution Language Model), a composite architecture
that unifies six recent advances — continuous next-vector prediction, deep
nonlinear associative memory, gradient-free evolution strategies, query-only
test-time training, reasoning compression, and group-based self-improvement —
into a single coherent system. The key insight is that these innovations are
not merely additive: gradient-free training (EGGROLL) enables nonlinear
recurrent memory (Miras) that is impossible under backpropagation through
time; continuous-latent prediction (CALM) compresses sequences by Kx, making
deep memory tractable; and query-only test-time training (qTTT) provides
inference-time adaptation that compensates for the recurrent backbone's
limited attention. We describe the architecture, analyze the theoretical
synergies between components, and propose an experimental roadmap for
validation. Preliminary analysis suggests VELM can achieve competitive
language modeling performance with fundamentally different scaling properties:
compute-only scaling via population size (no gradient infrastructure),
native int8 training, and continuous self-improvement without human
intervention.

## 1. Introduction (~2 pages)

### Motivation
- LLMs are bottlenecked by sequential token generation, quadratic attention,
  and gradient-dependent training
- Recent work addresses these independently but no one has composed them
- The composition is non-obvious: each paper assumes constraints the others lift

### Contributions
1. A unified architecture combining CALM, Miras, EGGROLL, qTTT, CIB, and GEA
2. Theoretical analysis of why these six innovations compose synergistically
3. Identification of the "gradient-freedom" principle as the architectural
   linchpin (EGGROLL enables Miras; Miras enables deep memory over CALM vectors)
4. A concrete experimental roadmap with proposed ablations
5. Discussion of a new scaling axis: population size as compute lever

## 2. Background and Related Work (~3 pages)

### 2.1 Continuous Autoregressive Models
- CALM (Shao et al. 2025): next-vector prediction, autoencoder, energy head
- Large Concept Models, MegaByte, text diffusion
- Information-theoretic limits of discrete vocabularies

### 2.2 Deep Associative Memory
- Miras framework (Behrouz et al. 2025): 4-choice memory design
- Connection to meta-learning and online optimization (FTRL viewpoint)
- Linear recurrent models: RetNet, GLA, Mamba, DeltaNet
- Why nonlinear memory matters: state tracking failures in linear models

### 2.3 Gradient-Free Training at Scale
- EGGROLL (Sarkar et al. 2025): low-rank ES, int8 training, GRPO-competitive
- Prior work: zeroth-order fine-tuning, CMA-ES for LoRA
- The linearisation theorem: ES converges to gradient in high dimensions
- Why ES enables architectures impossible under backprop

### 2.4 Long-Context Adaptation
- qTTT (Bansal et al. 2025): query-only test-time training
- Score dilution: logarithmic margin requirement
- Why thinking tokens fail: they use the same static attention
- Connection to TTT and inference-time compute scaling

### 2.5 Efficient Reasoning
- CIB: conditional information bottleneck for reasoning compression
- Budget forcing, ThinkPrune, HAPO, and other compression methods
- Information-theoretic view of chain-of-thought

### 2.6 Self-Improving Systems
- GEA (Weng et al. 2025): group-evolving agents
- DGM, Gödel Agent, open-ended evolution
- Why group evolution outperforms tree-structured evolution

## 3. The VELM Architecture (~4 pages)

### 3.1 Design Principles
- No backpropagation required (gradient-freedom principle)
- Continuous latent space (information density principle)
- Inference-time adaptation (context-specific principle)

### 3.2 Representation Layer
- CALM autoencoder: training, architecture, hyperparameters
- Extension to context-aware encoding (future work direction)
- Latent space properties: smoothness, semantic structure

### 3.3 Miras Deep Memory Backbone
- Choice of Memora variant (MLP memory, ℓ2 bias, KL retention, GD learning)
- Hybrid architecture: Miras + Sliding Window Attention
- Why nonlinear memory over continuous vectors (vs discrete tokens)
- Channel-wise parameterization and low-rank projections

### 3.4 Energy-Based Generative Head
- Architecture: residual MLP blocks with SwiGLU
- Energy loss: strictly proper scoring rule
- Likelihood-free temperature sampling algorithm
- Computational overhead: ~10% of total model params

### 3.5 EGGROLL Training
- Low-rank perturbations for Miras + head parameters
- Fitness function: quality × compression (CIB-aware)
- Int8 training path: discretized Adam updates
- Population sizing and compute-performance tradeoff
- Comparison with backprop baselines at matched compute

### 3.6 qTTT Inference Adaptation
- Single-prefill KV caching for SWA layers
- Query-only gradient updates on sampled spans
- Margin improvement theorem (adapted for VELM's hybrid architecture)
- FLOP equivalence with thinking tokens

### 3.7 CIB Reasoning Compression
- Dual role: training objective + inference budget controller
- Integration with CALM chunks (compress at chunk granularity)
- Adaptive budget allocation per query complexity

### 3.8 GEA Self-Improvement
- Group evolution over EGGROLL populations
- Experience sharing: perturbation directions, reasoning traces
- Performance-Novelty selection criterion
- Convergence analysis

## 4. Theoretical Analysis (~2 pages)

### 4.1 Synergy Analysis
**Theorem sketch:** EGGROLL + Miras unlocks strictly greater model classes
than backprop + Miras (nonlinear RNNs without BPTT)

**Theorem sketch:** CALM + Miras reduces effective sequence length by K while
maintaining associative memory capacity (memory operates over semantic units)

**Theorem sketch:** qTTT margin improvement extends to hybrid Miras+SWA
(query updates in SWA layers propagate through Miras state)

### 4.2 Scaling Properties
- New scaling axis: population size (EGGROLL)
- Semantic bandwidth K as third variable (CALM)
- Compute-only scaling: more GPUs = larger populations = better ES gradients
- Comparison with standard scaling laws (Chinchilla, etc.)

### 4.3 Information-Theoretic Analysis
- Bits per token vs bits per vector (CALM information density)
- CIB compression bound: minimal sufficient statistics for reasoning
- Score dilution recovery rate under qTTT

## 5. Proposed Experiments (~3 pages)

### 5.1 Language Modeling (Validation)
- Dataset: Pile uncopyrighted (following CALM), WikiText-103 eval
- Metric: BrierLM (following CALM), perplexity (standard)
- Baselines: Transformer++, Mamba2, Gated DeltaNet, CALM (original)
- Ablations: each component added incrementally

### 5.2 Ablation Studies
| Experiment | Tests | Expected outcome |
|-----------|-------|-----------------|
| CALM only (Transformer backbone) | Baseline continuous LM | Match CALM paper |
| + Miras backbone (backprop) | Deep memory benefit | Small improvement |
| + EGGROLL (no backprop) | Gradient-free training | Match backprop |
| + Int8 training | Native quantization | Slight quality loss, big efficiency gain |
| + qTTT at inference | Long-context adaptation | Large gain on RULER/LongBench |
| + CIB loss | Reasoning compression | Same accuracy, ~78% fewer tokens |
| + GEA self-improvement | Open-ended evolution | Continuous improvement over time |

### 5.3 Long-Context Evaluation
- Benchmarks: LongBench-v2, ZeroScrolls, RULER
- Compare: vanilla inference, thinking tokens, qTTT
- Attention mass analysis (following qTTT paper methodology)

### 5.4 Reasoning Efficiency
- Benchmarks: MATH500, AIME24, OlympiadBench
- Compare: standard CoT, CIB-compressed CoT, VELM reasoning
- Metric: accuracy vs token count (Pareto frontier)

### 5.5 Scaling Experiments
- Population size scaling: 2^10 to 2^20
- Model size scaling: 170M to 1.5B
- K (chunk size) scaling: K=1,2,4,8
- Interaction: does larger K benefit from larger populations?

### 5.6 Self-Improvement Trajectory
- Run GEA evolution for N iterations
- Track: fitness, reasoning quality, long-context accuracy
- Compare: GEA group evolution vs isolated ES evolution
- Measure: experience transfer rate, convergence speed

## 6. Discussion (~1.5 pages)

### 6.1 Limitations
- EGGROLL compute overhead (180x GPU-hours vs backprop at max population)
- CALM autoencoder is context-free (limits chunk boundary awareness)
- qTTT requires SWA layers (limits pure-recurrent configurations)
- GEA evaluation is compute-intensive (many parallel model instances)

### 6.2 Broader Impact
- Gradient-free training democratizes architecture exploration
- Int8 native training reduces energy consumption
- Self-improving systems require careful alignment considerations

### 6.3 Future Directions
- Context-aware CALM autoencoder (condition on prior vectors)
- Integrated end-to-end generative Transformer (remove AE separation)
- Neurosymbolic components (EGGROLL trains non-differentiable modules)
- Multi-modal extension (continuous vectors naturally accommodate vision)
- Distributed GEA across heterogeneous hardware
- Formal verification of self-improvement alignment

## 7. Conclusion (~0.5 pages)

VELM demonstrates that recent innovations in language modeling are not
isolated advances but composable building blocks for a fundamentally new
architecture paradigm. The gradient-freedom principle — enabled by EGGROLL —
is the linchpin that unlocks deep nonlinear memory, native int8 training,
and self-improving systems. Combined with CALM's continuous-latent
representation and qTTT's inference-time adaptation, VELM charts a path
toward language models that scale via population rather than gradient
infrastructure, adapt to each context without retraining, and continuously
improve without human intervention.

---

## Appendices

### A. Detailed Miras Variant Derivations
### B. EGGROLL Convergence Proofs for VELM Architecture
### C. qTTT FLOP Calculations for Hybrid Miras+SWA
### D. CIB Theoretical Compression Bounds
### E. Full Experimental Hyperparameters
### F. GEA Evolution Trajectory Visualizations

## Estimated Length: ~16 pages + appendices (NeurIPS/ICML format)
