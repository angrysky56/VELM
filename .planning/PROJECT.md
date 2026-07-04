# PROJECT: VELM (Vector-Evolution Language Model)

## 🎯 Vision (revised 2026-07-04 — hybrid)

VELM is a continuous-latent LLM architecture with a **hybrid training strategy**: backprop where gradients exist (CALM AE, Miras backbone, energy head), evolution strategies where they don't (GEA self-improvement). The original all-gradient-free thesis was tested and falsified at consumer scale by the go/no-go POC (`docs/poc_findings.md`): backprop trains the nonlinear Miras backbone easily (R² 0.628 vs linear-probe 0.229), while pop-512 EGGROLL reached only R² 0.082 in the identical setting, and warm-started ES stalls at a noise floor after capturing local gains.

## 🏗️ Core Architecture (The "Six Synergies", hybrid revision)

1. **CALM Autoencoder**: Compresses tokens into continuous latent vectors ($K=4$). *(backprop — done, 99.9%)*
2. **Miras Backbone**: Deep non-linear associative memory (MLP-based recurrence). *(backprop — POC-proven)*
3. **EGGROLL**: Zero-order (ES) optimizer — **rescoped to GEA phase only** (POC: not viable for pretraining at accessible population sizes).
4. **qTTT**: Inference-time adaptation via query updates for infinite context.
5. **CIB Integration**: Reasoning compression as a backprop loss term (was: fitness term).
6. **GEA**: Group-Evolving Agents for population-level self-improvement — **where ES earns its keep** (non-differentiable fitness; ES refines well from good basins).

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Backend**: JAX / Equinox (Functional ML)
- **Precision**: Aiming for Native Int8 (Initial validation in BFloat16)
- **Infrastructure**: Local (Logic/Dev) + Google Colab (Training/Compute)

## 📈 Roadmap & Milestones

### Phase 1: Research & Viability (Completed)

- \[x\] **Optimization Audit**: Evaluate EGGROLL vs MeZO.
- \[x\] **Architecture Audit**: Validate OpenMythos RDT patterns.
- \[x\] **Blocker Fix**: Resolve EGGROLL RecursionError.

### Phase 1.1: Mythos-Enhanced Architecture (COMPLETED)

- \[x\] **Core Implementation**: Add LTI Injection & Loop Embeddings.
- \[x\] **Backbone Refactor**: Support n_loops and Input Injection.
- \[x\] **Stability Verification**: Unit tests for spectral radius.
- \[x\] **Notebook Integration**: Mythos-RDT training workflow in Colab.

### Phase 2: Foundation — go/no-go POC (COMPLETED 2026-07-04)

- \[x\] Train & Validate CALM Autoencoder (99.9% recon, `checkpoints/calm_ae_best.json`).
- \[x\] Train Miras Backbone via Backprop (R² 0.628, beats no-memory linear probe 3× — backbone demonstrably uses sequence memory).
- \[x\] Train Miras Backbone via EGGROLL (POC v1–v4: **fails at consumer scale** — pop 512, ~4M evals → R² 0.082; ES refines from good basins, cannot explore. Full analysis: `docs/poc_findings.md`).
- \[x\] **Decision: hybrid architecture** — backprop pretraining, ES scoped to GEA.

### Phase 3: Hybrid Integration (CURRENT)

- \[ \] Backprop-train backbone + energy head on real CALM AE latents (energy score loss) — `notebooks/phase3_hybrid_train.ipynb`.
- \[ \] Qualitative decode demo: predicted latent → AE decoder → text.
- \[ \] Implement qTTT for hybrid SWA/Miras layers.
- \[ \] CIB reasoning compression as a backprop loss term.

### Phase 4: Scaling & Self-Improvement

- \[ \] Scale to VELM-Small (340M params) on Colab/rented compute.
- \[ \] Implement GEA group evolution loop — ES applied to non-differentiable workflow/task fitness, warm-started from the backprop-trained core.

## ⚠️ Known Risks & "Windmills" (updated post-POC)

- ~~**Dimensionality**: Can EGGROLL's population-based gradients scale to 1.5B parameters?~~ **Answered: no at consumer scale** (see `docs/poc_findings.md` §F2).
- **Continuous prediction quality**: Does the energy head learn sharp next-latent distributions via backprop at small scale? (Phase 3's key question.)
- **Hardware**: Ty's RTX 3060 is thermally damaged — all training runs on Colab; local machine is for logic/dev only.
