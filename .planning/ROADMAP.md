# ROADMAP

## Milestone 1: Research & Foundation Optimization

- \[x\] Phase 1: Optimization & Architecture Audit (Completed)
- \[x\] Phase 1.1: Mythos-Enhanced Architecture Implementation (Completed)
- \[x\] Phase 2: Foundation Experiments — go/no-go POC (Completed 2026-07-04)
  - CALM Autoencoder trained (99.9% recon)
  - EGGROLL vs backprop head-to-head, 4 iterations (`docs/poc_findings.md`)
  - **Verdict: hybrid** — backprop pretraining, ES scoped to GEA

## Milestone 2: Hybrid VELM

- \[ \] Phase 3: Hybrid Integration (CURRENT)
  - Backprop backbone + energy head on CALM latents (`notebooks/phase3_hybrid_train.ipynb`)
  - Decode demo, then qTTT and CIB-as-loss
- \[ \] Phase 4: Scaling & GEA Self-Improvement
  - VELM-Small (340M) on Colab/rented compute
  - GEA loop: ES on non-differentiable fitness, warm-started from backprop core

## Milestone 3: Write-up

- \[ \] Paper/blog: "When does gradient-free training pay?" — POC methodology + hybrid architecture
