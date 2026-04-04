# Fix Phase 2 Hang + Missing Checkpoints + Single-Dataset Bug

## Root Cause Analysis

### Bug 1: Phase 2 Hangs — JIT Recompilation Every Step 🔴

The `fitness_fn` (line 277) closes over a **mutable global** `_current_batch`:

```python
_current_batch = None  # global

def fitness_fn(params):
    batch_tokens = _current_batch  # ← captured closure variable
    ...
```

Then in the loop (line 321): `_current_batch = jnp.array(all_chunks[idx])` **changes every step**.

Inside `eggroll_step`, `jax.lax.map(eval_member, keys)` traces `fitness_fn` for XLA compilation. JAX treats the closed-over `_current_batch` as a **compile-time constant**. When it changes next step, **JAX recompiles the entire program from scratch** — the full backbone (4 Miras scan layers + 4 SWA attention layers + AE encode + energy scoring) for 32 population members.

Each compilation takes **several minutes on a T4**. With 5,000 steps, each triggering recompilation, Phase 2 would never finish.

> [!CAUTION]
> This is an **infinite recompilation loop**. The first step compiles for ~30 min, then step 2 recompiles for ~30 min, ad infinitum. Zero training ever happens.

**Fix**: Rewrite `eggroll_step` and `fitness_fn` so the batch is passed as a **function argument** rather than closed over. Since we can't easily modify the `jax.lax.map` calling convention inside `eggroll_step` (it's a library function), we restructure the notebook's Phase 2 to use a **custom training loop** that: (a) compiles the per-member evaluation once, and (b) passes the batch through the compiled function each step.

### Bug 2: No Checkpoint Saving During Phase 2

Phase 1 saves every 2,000 steps (line 193). Phase 2 saves **only after all steps complete** (line 343). If training hangs, OOMs, or the Colab session times out, all progress is lost.

**Fix**: Add periodic checkpoint saving every 500 steps during Phase 2.

### Bug 3: Only OpenWebMath Dataset

Section 4 (line 88-115) streams **only** from OpenWebMath. The `CurriculumDataLoader` from `data_loader.py` is never instantiated. Phase 3's `data_loader` reference (line 436) always hits the `NameError` fallback, collapsing to `[{"type": "default"}]`.

**Fix**: Replace the single-dataset streaming with `CurriculumDataLoader` for multi-domain data (OpenWebMath 50%, WikiText 30%, TinyStories 20%).

---

## Proposed Changes

### Section 4: Replace Single-Dataset Streaming With CurriculumDataLoader

#### [MODIFY] velm_colab.py — Section 4

Replace the 27-line OpenWebMath-only loading block with a `CurriculumDataLoader` call that streams from all 3 datasets. Fall back to OpenWebMath-only if the loader fails (HuggingFace rate limits, etc).

---

### Section 7: Fix Phase 2 Training Loop

#### [MODIFY] velm_colab.py — Section 7 (EGGROLL training)

1. **Remove global `_current_batch`** — this is the root cause of infinite recompilation
2. **Rewrite fitness evaluation** as a JIT-compiled function that takes `(params, batch)` as arguments
3. **Write a custom ES step** that avoids `jax.lax.map` (which tries to compile the entire population evaluation into one XLA program). Instead, use a Python loop over population members with JIT-compiled per-member evaluation. This is slightly slower per-step but avoids the massive compilation overhead.
4. **Add checkpoint saving** every 500 steps with best-fitness tracking
5. **Add compilation warmup** — run one dummy evaluation before the loop starts and print "Compiling..." so the user knows what's happening

---

### Section 9: Fix GEA Phase 3

#### [MODIFY] velm_colab.py — Section 9

Wire `data_loader` from Section 4 into the GEA fitness function so domain-specific evaluation works correctly.

---

## Verification Plan

### Automated Tests
- Run `python tests/test_smoke.py` — all 11 tests pass
- Local sanity check: run first 3 steps of Phase 2 on CPU to verify no recompilation (should finish in seconds, not minutes)

### Manual Verification  
- Regenerate `velm_colab.ipynb` from the fixed `.py`
- User runs on Colab: Phase 2 should show output within ~2-5 minutes (first compilation) then proceed steadily
