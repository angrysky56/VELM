# VELM: Vector-Evolution Language Model

## A Self-Evolving, Continuous-Latent, Gradient-Free Language Model Architecture

**Status:** Research Design Phase

### Abstract

VELM proposes a composite language model architecture that integrates six recent
advances into a unified system: continuous next-vector prediction (CALM), deep
nonlinear associative memory (Miras), gradient-free evolution strategies at scale
(EGGROLL), query-only test-time training for long context (qTTT), reasoning
compression via conditional information bottleneck (CIB), and group-based
open-ended self-improvement (GEA).

The key insight is that these innovations are not merely additive — they unlock
capabilities impossible under any single paradigm:

- **Nonlinear RNNs become trainable** because EGGROLL eliminates the need for
  backpropagation through time
- **Continuous-latent generation** becomes efficient because CALM compresses K
  tokens into single vectors, reducing autoregressive steps by Kx
- **Deep memory** over continuous vectors replaces shallow linear recurrence with
  MLP-based associative memory that can actually track state
- **Long-context failures** are fixed at inference time via query-only TTT,
  without retraining or growing the KV cache
- **Reasoning efficiency** is enforced structurally via CIB, compressing
  chain-of-thought by ~78% without accuracy loss
- **The entire system self-improves** through group evolution: populations of
  models share experience and evolve weights + workflows without human intervention

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ REPRESENTATION: CALM Autoencoder → Miras Deep Memory → EBM  │
│ K tokens → 1 continuous vector → MLP memory backbone → head  │
├─────────────────────────────────────────────────────────────┤
│ TRAINING: EGGROLL (gradient-free, int8-native) + CIB loss    │
│ Low-rank ES populations, fitness = quality × compression     │
├─────────────────────────────────────────────────────────────┤
│ INFERENCE: qTTT query adaptation + CIB budget control        │
│ Frozen KV cache, adaptive query projections, efficient CoT   │
├─────────────────────────────────────────────────────────────┤
│ SELF-IMPROVEMENT: GEA group evolution + EGGROLL populations  │
│ Experience sharing across model variants, open-ended search  │
└─────────────────────────────────────────────────────────────┘
```

### Source Papers

| Paper | Contribution | Key Innovation |
|-------|-------------|----------------|
| CALM (Shao et al., 2025) | Representation | Compress K tokens → 1 continuous vector, energy-based decode |
| Miras (Behrouz et al., 2025) | Backbone | Unified framework: 4-choice deep associative memory |
| EGGROLL (Sarkar et al., 2025) | Training | Gradient-free ES at billion-param scale, int8 native |
| qTTT (Bansal et al., 2025) | Inference | Query-only TTT fixes score dilution in long context |
| CIB (2025) | Efficiency | Reasoning as compression via conditional info bottleneck |
| GEA (Weng et al., 2025) | Self-improvement | Group-evolving agents with experience sharing |

### Project Structure

```
VELM/
├── README.md                     # This file
├── docs/
│   ├── paper_outline.md          # Full paper structure
│   ├── architecture.md           # Detailed architecture specification
│   ├── synergies.md              # Cross-paper integration analysis
│   ├── experiments.md            # Proposed experimental plan
│   └── figures/                  # Diagrams and visualizations
├── src/
│   ├── model/                    # Model architecture (Miras + CALM)
│   ├── training/                 # EGGROLL optimizer + CIB loss
│   ├── inference/                # qTTT adaptation + CIB budget
│   └── evolution/                # GEA self-improvement loop
└── references/                   # Paper citations and notes
```

### Getting Started

This project is in the research design phase. See:
- `docs/architecture.md` for the full technical specification
- `docs/paper_outline.md` for the paper structure
- `docs/synergies.md` for how the six papers compose
