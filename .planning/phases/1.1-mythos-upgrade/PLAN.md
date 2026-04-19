# PLAN: Phase 1.1 - Mythos-Enhanced Architecture Implementation

## 🎯 Goal
Integrate stability and reasoning improvements from **OpenMythos/RDT** research into the **VELM** backbone. This transforms VELM from a standard hybrid-recurrent model into a **Recurrent Depth Transformer (RDT)** with guaranteed stability and adaptive compute.

## 🏗️ Technical Implementation

### 1. Stability: LTI-Stable Injection
- Implement `LTIInjection` module in `src/model/miras_backbone.py`.
- Enforce $ρ(A) < 1$ using log-parameterization for the retention diagonal.
- This prevents the "exploding memory" problem during deep recurrent loops.

### 2. Identity: Loop-Index Embeddings
- Add `LoopIndexEmbedding` to provide the model with "temporal awareness" within the recurrence.
- Use sinusoidal embeddings (inspired by RoPE) to ensure generalization to unseen loop depths.
- Inject these embeddings at the start of each backbone loop.

### 3. Consistency: Input Injection
- Implement "Anchor Injection": The original compressed latent $e$ is injected at every recurrent step.
- Prevents signal decay and ensures the reasoning process remains grounded in the original input.

### 4. Efficiency: ACT Halting (Optional/Experimental)
- Add a learned `HaltingHead` to `VELMBackbone`.
- Allow the model to decide when to stop looping based on a learned probability threshold.

## 📝 Steps

1.  **[x] Core Modules**: 
    - Add `LTIInjection`, `LoopEmbedding`, and `ACTHalting` to `src/model/miras_backbone.py`.
2.  **[x] Backbone Refactor**: 
    - Update `VELMBackbone` to accept `n_loops`.
    - Modify `__call__` to implement the RDT update loop.
3.  **[x] Unit Testing**: 
    - Verify spectral radius of `LTIInjection` is always $< 1$.
    - Verify `LoopEmbedding` produces distinct vectors for different indices.
    - Test `VELMBackbone` with `n_loops > 1` for NaN-free execution.
4.  **[x] Notebook Integration**: 
    - Update `velm_colab.ipynb` to reflect the new architecture.
    - Set `n_loops=1` by default for Phase 2 training, enabling Phase 3/4 depth.

## ✅ Verification
- `pytest tests/test_model_stability.py`
- Manual check: Latent drift across 100 loops should be bounded.
