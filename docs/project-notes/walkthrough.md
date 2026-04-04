# Fix Phase 2 Hang + Missing Checkpoints + Single-Dataset Bug

## Bug 1: Phase 2 Infinite Recompilation Loop 🔴 → Fixed

**Root cause**: `fitness_fn` closed over a mutable global `_current_batch`. Inside `eggroll_step`, `jax.lax.map` traced this as a compile-time constant. When the batch changed on the next step, JAX recompiled the *entire* backbone + AE + energy scoring. Each compilation took ~30 min on T4. Zero training ever happened.

**Fix**: Complete rewrite of Phase 2:
1. Removed `_current_batch` global
2. Wrote `evaluate_member(base_params, member_key, batch)` with `@eqx.filter_jit` — batch is an explicit argument, not a closure
3. Replaced `jax.lax.map` (single massive XLA program) with a Python loop over population members (each calls the pre-compiled function)
4. Added **compilation warmup** before the loop with user feedback: `"Compiling backbone evaluation (one-time, may take 2-5 min on T4)..."`

> [!IMPORTANT]
> The warmup ensures the user sees progress immediately instead of a 30-minute blank screen.

## Bug 2: No Checkpoint Saving During Phase 2 → Fixed

Phase 1 saved every 2,000 steps. Phase 2 saved **only after all steps completed**. If Colab timed out, all progress was lost.

**Fix**: Checkpoint every 500 steps with best-fitness tracking:
- Saves `backbone_step{N}.eqx` + `head_step{N}.eqx` at each checkpoint
- Tracks best fitness and saves `backbone_eggroll_best.eqx` when improved

## Bug 3: Only OpenWebMath Dataset → Fixed

The `CurriculumDataLoader` in `data_loader.py` was never instantiated. Only OpenWebMath streamed.

**Fix**: Section 4 now loads 3 datasets with curriculum weighting:
- **OpenWebMath** (50%) — mathematical reasoning
- **WikiText-103** (30%) — general knowledge  
- **TinyStories** (20%) — narrative structure

Each chunk gets a domain label (`chunk_labels` list) used by GEA Phase 3 for domain-specific evaluation. Fallback to random tokens if any dataset fails.

Section 9 now uses `chunk_labels` directly (via `domain_indices` dict) instead of the nonexistent `data_loader` object.

## Files Changed

| File | Changes |
|---|---|
| [velm_colab.py](file:///home/ty/Repositories/ai_workspace/VELM/notebooks/velm_colab.py) | All 3 bug fixes |
| [velm_colab.ipynb](file:///home/ty/Repositories/ai_workspace/VELM/notebooks/velm_colab.ipynb) | Regenerated from .py |

## Verification

```
VELM Smoke Tests: 11 passed, 0 failed
All tests passed!
```

## What to Expect in Colab Now

1. **Section 4**: ~5-15 min streaming from 3 datasets (progress bars per domain)
2. **Phase 1 (AE)**: Same as before — reaches 97%+ rapidly
3. **Phase 2 (EGGROLL)**: "Compiling..." message → 2-5 min warmup → steady ~0.5-2 steps/sec depending on GPU. Checkpoints saved every 500 steps.
4. **Phase 3 (GEA)**: Domain-aware evolution across math/general/narrative tasks
