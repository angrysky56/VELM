# PROJECT: VELM (Vector-Evolution Language Model)

## 🎯 Vision
VELM is a next-generation LLM architecture designed for **gradient-freedom** and **hardware-native efficiency**. By synthesizing continuous-latent representations with evolutionary optimization, VELM aims to break the memory and compute bottlenecks of standard backpropagation-based Transformers.

## 🏗️ Core Architecture (The "Six Synergies")
1.  **CALM Autoencoder**: Compresses tokens into continuous latent vectors ($K=4$).
2.  **Miras Backbone**: Deep non-linear associative memory (MLP-based recurrence).
3.  **EGGROLL**: Zero-order (ES) optimizer enabling native Int8 training without backprop.
4.  **qTTT**: Inference-time adaptation via query updates for infinite context.
5.  **CIB Integration**: Reasoning compression regularized via fitness loss.
6.  **GEA**: Group-Evolving Agents for population-level self-improvement.

## 🛠️ Tech Stack
*   **Language**: Python 3.10+
*   **Backend**: JAX / Equinox (Functional ML)
*   **Precision**: Aiming for Native Int8 (Initial validation in BFloat16)
*   **Infrastructure**: Local (Logic/Dev) + Google Colab (Training/Compute)

## 📈 Roadmap & Milestones

### Phase 1: Research & Viability (CURRENT)
*   [ ] **Optimization Audit**: Evaluate EGGROLL vs MeZO for billion-param scaling.
*   [ ] **Architecture Audit**: Validate CALM+Miras stability on WikiText-103.
*   [ ] **Hardware Audit**: Profile Int8 JAX performance on RTX 3060/T4.
*   [x] **Blocker Fix**: Resolve EGGROLL RecursionError in JAX tracing.

### Phase 2: Foundation (Foundation Experiments)
*   [ ] Train & Validate CALM Autoencoder (>99.9% recon).
*   [ ] Train Miras Backbone via Backprop (Baseline).
*   [ ] Train Miras Backbone via EGGROLL (Proof of Concept).

### Phase 3: Integration
*   [ ] Implement qTTT for hybrid SWA/Miras layers.
*   [ ] Integrate CIB reasoning compression into the fitness function.
*   [ ] First end-to-end VELM-Tiny training run.

### Phase 4: Scaling & Self-Improvement
*   [ ] Scale to VELM-Small (340M params).
*   [ ] Implement GEA group evolution loop.

## ⚠️ Known Risks & "Windmills"
*   **Dimensionality**: Can EGGROLL's population-based gradients scale to 1.5B parameters?
*   **Stability**: Will deep non-linear memory (Miras) collapse under low-rank perturbations?
*   **Hardware**: Balancing ES compute intensity with consumer GPU thermal limits (Colab focus).
