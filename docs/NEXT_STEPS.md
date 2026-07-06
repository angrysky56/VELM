# VELM — Next Steps

_The session-start reference. Updated 2026-07-06. History and rationale live in
`poc_findings.md`, `phase3_findings.md`, `blooms_ladder.md`; this doc is only
about what to do next and how to decide._

---

## 1 · Where things stand

| Component                           | Status                           | Number                                                                                                              |
| ----------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| CALM AE (frozen)                    | ✅ done                          | 99.9% recon (`calm_ae_best.eqx`)                                                                                    |
| Miras backbone (understand)         | ✅ trained, scaling measured     | centered cos **0.314** (v3, 64M tokens); curve: 0.167 → 0.218 → 0.306 → 0.314 → **saturated: capacity-limited now** |
| Energy head (conditional sampler)   | ✅ trained (v3, frozen-backbone) | head cos 0.227, energy 1.000 < 1.123 bar                                                                            |
| Free-running generation             | ❌ scale-gated                   | horizon ~2 chunks; snap mode is the honest generator (judge NLL 4.74 vs truth 3.33)                                 |
| **Active run: v3l2 loops ablation** | 🔄 in progress                   | ccos **0.320 @ gs 16k** (baseline to beat: 0.314)                                                                   |

Optimizer verdict (settled): backprop pretrains; ES scoped to GEA.
Objective verdict (settled): stage rung-pure — MSE first, energy on frozen reps second.

## 2 · The queue

### Now — finish v3l2 (loops / "thinking" ablation)

Re-run `phase3_hybrid_train.ipynb` until the log prints _stage budget
exhausted_. No config changes between sessions.

**Decision rule at completion:**

- final best ccos **≥ 0.33** → loops pay. Result: latent iteration adds capability at fixed params (System-2 headline for the paper). Next run: try `n_loops: 3` (`run_version: "v3l3"`, `init_from_version: "v3l2"`) to see if it stacks.
- **0.315–0.33** → marginal. Note it, move to width (below); revisit loops after.
- **≤ v3's 0.314** → loops don't pay at this scale. Clean negative; go to width.

### Next — width run (v4, dim 384)

When loops are settled. Config changes in `phase3_hybrid_train.ipynb`:

```
"run_version": "v4",  "n_loops": <winner from ablation>,
"init_from_version": None,        # width change = new arrays, fresh start
"data_version": "v3",             # reuse caches
"dim": 384, "ffn_intermediate": 768, "head_ffn": 768,
"total_steps": 64000, "session_steps": 8000, "peak_lr": 1e-3,
```

~2× params (≈35M). 8 sessions. **Decision rule:** ccos ≥ 0.38 → capacity was
the constraint, consider dim 512 later; ≤ 0.33 → diminishing — pivot to
capabilities (qTTT/CIB) rather than more scale.

### After any stage-1 winner — the standard measurement pass

1. Flip `objective: "energy_frozen"`, run one session (trains sampler on the
   new backbone; expect head cos to track ~0.7× of direct cos).
2. Run `phase3_rollout.ipynb` (ckpt search already knows all version names).
   Record: step-1 cos, horizon, judge NLL + distinct-bigram for all modes.
3. Append the numbers to `phase3_findings.md` §5 table.

### Then — capabilities (new work, order of preference)

1. **qTTT** (`src/inference/qttt.py` exists, unintegrated): query-only
   test-time adaptation on the SWA layers. Natural eval: long-context
   degradation curve with/without qTTT. Thinking-as-adaptation.
2. **Latent chain-of-thought**: roll extra never-decoded "thought latents"
   before each prediction (Coconut-style). VELM's continuous latents are the
   natural home; `n_loops` results inform whether iteration helps.
3. **CIB** as a backprop loss term (reasoning compression).
4. **GEA** (`src/evolution/gea_eggroll.py`): ES on non-differentiable fitness,
   warm-started from the backprop core — where ES belongs per the POC.
5. **Arm-C anomaly** (backbone dead on latent inputs; MLP fine on same
   vectors): unexplained; blocks direct latent-feed inference; a focused
   debugging notebook would make a good paper appendix.

### In parallel, whenever — the write-up

Skeleton already exists across `poc_findings.md` + `phase3_findings.md` +
`blooms_ladder.md`. Assembly order: scaling figure (4-point curve + loops/width
points) → EGGROLL verdict → staged-training recipe → imposter-baseline
methodology → exposure-bias measurements → negatives (whitening, scheduled
sampling, arm C). Working title: _"Hybrid training of a continuous-latent
language model at consumer scale."_

## 3 · Session mechanics (the how-to)

- **Continue a run**: open notebook from the Colab badge → Runtime = A100
  **High-RAM** → Run All. State resumes automatically; sessions are ~20 min
  cache-load + ~1–1.5h train.
- **Caches on Drive** (`MyDrive/VELM_checkpoints/`): `poc_tokens_v3.npz`,
  `poc_latents_v3.npy` (~9GB), `latent_stats.npz` (frozen — never delete),
  `train_state_<ver>[.eqx|.json]`, component ckpts `*_hybrid_<ver>_<tag>.eqx`.
- **Start a new run**: change `run_version` (never reuse one) + the intended
  knobs; set `init_from_version` to warm-start or `None` for fresh;
  `data_version: "v3"` reuses data. Delete nothing.
- **Switch stage on a finished run**: flip `objective` only. Each stage keeps
  its own state file; the budget-exhausted message tells you when.

## 4 · Hard-won rules (violate at your peril)

1. Never weights-only-resume into a fresh optimizer mid-schedule (killed run 5). The notebook's state files handle this; don't hand-load checkpoints for training.
2. Never trust a capability metric without its imposter baseline (uncond cosine, linear-oracle bar, marginal sampler, distinct-bigrams).
3. Cosine-to-truth is meaningless beyond ~2 rollout steps (branching ≠ failure); use judge NLL + distinct-bigrams for generation.
4. Diversity ≈ 0.5 is _correct_ for the mean head; 1.0 is the sampler's target.
5. Warm-starting across `n_loops` works only via the params-partition path (the notebook's); full-module deserialise silently clobbers the loop count.
6. Frozen artifacts stay frozen: `latent_stats.npz`, the eval set (first-128 stream order), the AE. They are what makes runs comparable.
7. Objectives stay rung-pure (see `blooms_ladder.md` §9).

## 5 · Metric glossary (30-second refresher)

- **centered cos** — contextual prediction quality; 0 = knows nothing beyond the marginal; the headline number. Bars: linear oracle ≈ 0.12, v3 = 0.314.
- **decoded token acc** — predictions pushed through the AE decoder, token match vs truth; uncond imposter ≈ 0.025.
- **diversity** — pred spread ÷ target spread across contexts; <0.2 = collapsed; ~0.5 = healthy mean head; →1.0 = calibrated sampler.
- **energy score** — proper scoring rule for the sampler; beat the marginal-sampler bar (1.123).
- **judge NLL / distinct-2gram** — generation plausibility and degeneracy; the pair is required (repetition fools NLL, gibberish fools distinct).
- **horizon** — free-running steps before quality halves vs teacher-forced.
