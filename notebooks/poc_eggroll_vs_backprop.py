# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VELM POC v3: Can EGGROLL match backprop?
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angrysky56/VELM/blob/main/notebooks/poc_eggroll_vs_backprop.ipynb)
#
# **The single go/no-go experiment for VELM**, third iteration.
#
# **History:**
# - **v1 FAIL (task)** — next-chunk pooled-random-embedding targets were ~pure
#   irreducible entropy; even backprop converged to the mean predictor.
# - **v2 FAIL (population starvation?)** — teacher-distillation targets fixed the
#   task (backprop broke through its ~900-step plateau → MSE 0.63 and falling).
#   EGGROLL at pop=32 learned only the trivial mean solution. But 32 antithetic
#   directions in a 1.5M-dim space is a 50,000:1 ratio; the EGGROLL paper used
#   populations up to **262,144**. v2 never tested the paper's actual claim.
#
# **v3 tests two hypotheses at once:**
#
# 1. **Population starvation** (Run B): pop **512** antithetic (1024 evals/step),
#    3000 steps. Faithful to the paper's design: fitness is evaluated in
#    GPU-parallel chunks, and perturbations are **regenerated from RNG keys**
#    in a second pass rather than stored — the paper's memory trick (at 1.5M
#    params the FLOP-side matmul reordering is irrelevant; noise-on-demand +
#    large population is the algorithmic content).
# 2. **Plateau escape vs refinement** (Run C): warm-start ES from backprop's
#    step-900 plateau checkpoint. If ES can descend where backprop found signal,
#    ES can *refine* but not *escape* — which argues for a hybrid VELM
#    (backprop pretraining, ES for GEA self-improvement only).
#
# **Combined verdict** (automated, final cell):
#
# | Run B (pop 512) | Run C (warm start) | Reading |
# |---|---|---|
# | ✅ matches backprop | — | v2 was population starvation → wager holds, proceed |
# | ❌ | ✅ descends | ES refines but can't escape plateaus → hybrid VELM |
# | ❌ | ❌ | wager dead at this scale → pivot |
#
# Runtime: ~1–1.5 h on a free Colab T4. No Google Drive needed.

# %% [markdown]
# ## 1 · Setup

# %%
import os
import subprocess
import sys

IN_COLAB = os.path.exists("/content")
if IN_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "jax[cuda12]", "equinox", "jaxtyping", "optax", "einops",
         "datasets", "transformers"],
        check=True,
    )

# %%
if IN_COLAB:
    if not os.path.exists("/content/VELM"):
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/angrysky56/VELM.git", "/content/VELM"],
            check=True,
        )
    VELM_DIR = "/content/VELM"
else:
    VELM_DIR = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()

sys.path.insert(0, VELM_DIR)
print("VELM_DIR:", VELM_DIR)

# %%
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.model.miras_backbone import VELMBackbone
from src.training.eggroll import SigmaAdaptor, perturb_pytree

print("JAX devices:", jax.devices())
if not any(d.platform == "gpu" for d in jax.devices()):
    print("⚠️  No GPU detected — pop-512 ES will be very slow on CPU. "
          "Colab: Runtime → Change runtime type → T4 GPU.")

# %% [markdown]
# ## 2 · Config

# %%
CFG = {
    "seed": 42,
    # ── data / teacher (unchanged from v2 → target cache reusable) ──
    "teacher_id": "distilgpt2",
    "chunk_k": 4,
    "seq_len": 128,
    "num_train_seqs": 2048,
    "num_eval_seqs": 128,
    "batch_size": 8,
    "teacher_batch": 32,
    # ── model ─────────────────────────────────────────────
    "dim": 128,
    "num_heads": 4,
    "miras_layers": 2,
    "swa_layers": 2,
    "ffn_intermediate": 256,
    "embed_dim": 64,
    # ── Run A: backprop ───────────────────────────────────
    "bp_steps": 1500,
    "bp_lr": 1e-3,
    "bp_weight_decay": 0.01,
    "bp_plateau_step": 900,       # checkpoint for the warm-start diagnostic
    # ── Run B: EGGROLL-XL from scratch ────────────────────
    "es_steps": 3000,
    "es_pop": 512,                # antithetic directions → 1024 evals/step
    "es_chunk": 64,               # members vmapped per GPU chunk
    "es_rank": 1,
    "es_sigma": 1e-3,
    "es_sigma_min": 3e-4,
    "es_sigma_max": 5e-3,
    "es_lr": 3e-4,
    "es_adaptive_sigma": True,
    # ── Run C: warm-start ES from backprop plateau ────────
    "warm_steps": 1000,
    # ── shared ────────────────────────────────────────────
    "grad_clip": 1.0,
    "eval_every": 50,
}
assert CFG["es_pop"] % CFG["es_chunk"] == 0
key = jax.random.PRNGKey(CFG["seed"])

# %% [markdown]
# ## 3 · Data (identical to v2)

# %%
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CFG["teacher_id"])

K, T = CFG["chunk_k"], CFG["seq_len"]
N_SEQS = CFG["num_train_seqs"] + CFG["num_eval_seqs"]
tokens_needed = N_SEQS * T * K

stream, buf = load_dataset("roneneldan/TinyStories", split="train", streaming=True), []
total = 0
for ex in stream:
    text = ex.get("text", "")
    if len(text) < 50:
        continue
    ids = tokenizer.encode(text, max_length=512, truncation=True)
    buf.append(np.asarray(ids, dtype=np.int32))
    total += len(ids)
    if total >= tokens_needed:
        break

flat = np.concatenate(buf)[: (tokens_needed // (T * K)) * T * K]
seqs_np = flat.reshape(-1, T, K)
rng = np.random.default_rng(CFG["seed"])
seqs_np = seqs_np[rng.permutation(seqs_np.shape[0])]
print(f"sequences: {seqs_np.shape}  ({seqs_np.size:,} tokens)")

# %% [markdown]
# ## 4 · Targets: contextual teacher hidden states (identical to v2)

# %%
TARGET_CACHE = os.path.join(VELM_DIR, "checkpoints", "poc_teacher_targets.npy")
os.makedirs(os.path.dirname(TARGET_CACHE), exist_ok=True)

if os.path.exists(TARGET_CACHE):
    targets_np = np.load(TARGET_CACHE)
    assert targets_np.shape[:2] == seqs_np.shape[:2], "stale cache — delete and re-run"
    print(f"✓ loaded cached targets {targets_np.shape}")
else:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = (
        AutoModelForCausalLM.from_pretrained(
            CFG["teacher_id"],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
    )
    t_dim = teacher.config.hidden_size
    boundary_pos = np.arange(K - 1, T * K, K)

    proj_rng = np.random.default_rng(CFG["seed"] + 1)
    JL = proj_rng.standard_normal((t_dim, CFG["embed_dim"])).astype(np.float32)
    JL /= np.sqrt(CFG["embed_dim"])

    chunks_out = []
    flat_seqs = seqs_np.reshape(-1, T * K)
    with torch.no_grad():
        for start in tqdm(range(0, flat_seqs.shape[0], CFG["teacher_batch"]),
                          desc="teacher"):
            batch = torch.tensor(
                flat_seqs[start: start + CFG["teacher_batch"]],
                dtype=torch.long, device=device,
            )
            out = teacher(input_ids=batch, output_hidden_states=True)
            hid = out.hidden_states[-1][:, boundary_pos, :].float().cpu().numpy()
            chunks_out.append(hid @ JL)

    targets_np = np.concatenate(chunks_out, axis=0)
    targets_np = np.nan_to_num(targets_np, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(TARGET_CACHE, targets_np)
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"✓ extracted targets {targets_np.shape}")

n_tr = CFG["num_train_seqs"]
mu = targets_np[:n_tr].reshape(-1, CFG["embed_dim"]).mean(axis=0)
sd = targets_np[:n_tr].reshape(-1, CFG["embed_dim"]).std(axis=0) + 1e-6
targets_np = (targets_np - mu) / sd

train_seqs = jnp.asarray(seqs_np[:n_tr])
eval_seqs = jnp.asarray(seqs_np[n_tr:])
train_tgts = jnp.asarray(targets_np[:n_tr])
eval_tgts = jnp.asarray(targets_np[n_tr:])
baseline_mean = float(jnp.mean(eval_tgts ** 2))
print(f"mean-predictor baseline (eval): {baseline_mean:.4f}")

# %% [markdown]
# ## 5 · Model + loss (shared)

# %%
key, ek = jax.random.split(key)
EMBED = jax.random.normal(ek, (len(tokenizer), CFG["embed_dim"])) / jnp.sqrt(CFG["embed_dim"])


def make_model(k):
    kb, kh = jax.random.split(k)
    backbone = VELMBackbone(
        dim=CFG["dim"],
        num_heads=CFG["num_heads"],
        num_miras_layers=CFG["miras_layers"],
        num_swa_layers=CFG["swa_layers"],
        ffn_intermediate=CFG["ffn_intermediate"],
        chunk_size=K,
        ae_hidden_dim=CFG["embed_dim"],
        key=kb,
    )
    head = eqx.nn.Linear(CFG["dim"], CFG["embed_dim"], key=kh)
    return {"backbone": backbone, "head": head}


key, mk = jax.random.split(key)
model0 = make_model(mk)
params0, static = eqx.partition(model0, eqx.is_inexact_array)
n_params = sum(x.size for x in jax.tree.leaves(params0))
print(f"trainable params: {n_params:,}")


def seq_loss(params, seq_tokens, seq_tgt):
    model = eqx.combine(params, static)
    bb, head = model["backbone"], model["head"]
    embs = EMBED[seq_tokens]
    inp = jax.vmap(bb.compress_input)(embs)
    hid, _ = bb(inp)
    pred = jax.vmap(head)(hid)
    return jnp.mean((pred - seq_tgt) ** 2)


def batch_loss(params, batch_tokens, batch_tgts):
    return jnp.mean(jax.vmap(lambda s, t: seq_loss(params, s, t))(batch_tokens, batch_tgts))


@eqx.filter_jit
def eval_loss(params, seqs, tgts):
    return batch_loss(params, seqs, tgts)


init_loss = float(eval_loss(params0, eval_seqs, eval_tgts))
print(f"untrained eval MSE: {init_loss:.4f}")


def sample_batch(k):
    idx = jax.random.randint(k, (CFG["batch_size"],), 0, train_seqs.shape[0])
    return train_seqs[idx], train_tgts[idx]


# %% [markdown]
# ## 6 · Run A — AdamW backprop (+ plateau checkpoint for Run C)

# %%
bp_opt = optax.chain(
    optax.clip_by_global_norm(CFG["grad_clip"]),
    optax.adamw(CFG["bp_lr"], weight_decay=CFG["bp_weight_decay"]),
)
bp_state = bp_opt.init(params0)


@eqx.filter_jit
def bp_step(params, opt_state, batch, tgts):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(params, batch, tgts)
    updates, opt_state = bp_opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


bp_params, bp_best, bp_best_loss = params0, params0, float("inf")
plateau_params = None
bp_eval_hist = []
dk = jax.random.PRNGKey(CFG["seed"] + 100)
t0 = time.time()
for step in range(CFG["bp_steps"]):
    dk, bk = jax.random.split(dk)
    batch, tgts = sample_batch(bk)
    bp_params, bp_state, loss = bp_step(bp_params, bp_state, batch, tgts)
    if step == CFG["bp_plateau_step"]:
        plateau_params = bp_params  # snapshot for the warm-start diagnostic
    if step % CFG["eval_every"] == 0 or step == CFG["bp_steps"] - 1:
        ev = float(eval_loss(bp_params, eval_seqs, eval_tgts))
        bp_eval_hist.append((step, time.time() - t0, ev))
        if ev < bp_best_loss:
            bp_best_loss, bp_best = ev, bp_params
        print(f"[backprop {step:5d}] train {float(loss):.4f}  eval {ev:.4f}")
bp_time = time.time() - t0
bp_r2 = 1.0 - bp_best_loss / baseline_mean
plateau_eval = float(eval_loss(plateau_params, eval_seqs, eval_tgts))
print(f"\nbackprop best eval {bp_best_loss:.4f}  R² {bp_r2:.3f}  ({bp_time:.0f}s)")
print(f"plateau checkpoint (step {CFG['bp_plateau_step']}) eval: {plateau_eval:.4f}")

# %% [markdown]
# ## 7 · EGGROLL-XL machinery: chunked population, noise regenerated on demand
#
# Two passes per step, exactly the paper's memory design:
#
# 1. **Fitness pass** — members evaluated in vmapped chunks of `es_chunk`;
#    only the 2×pop fitness scalars are kept. Perturbations are *discarded*.
# 2. **Gradient pass** — perturbations are **regenerated from the same RNG
#    keys** and accumulated into the weighted ES gradient chunk by chunk.
#    Peak memory: one chunk of perturbations, never the full population.
#
# Fitness diffs are z-scored (scale-invariant update), gradient clipped at the
# same norm as backprop, σ adaptive within [3e-4, 5e-3].

# %%
N_CHUNKS = CFG["es_pop"] // CFG["es_chunk"]


def make_es_step(opt):
    @eqx.filter_jit
    def es_step(params, opt_state, batch, tgts, step_key, sigma):
        member_keys = jax.random.split(step_key, CFG["es_pop"])
        keys_chunked = member_keys.reshape(N_CHUNKS, CFG["es_chunk"], 2)

        # ── pass 1: fitness only (perturbations discarded) ──────────
        def member_fitness(mk):
            _, pert = perturb_pytree(params, mk, sigma, CFG["es_rank"])
            pos = jax.tree.map(lambda p, e: p + sigma * e, params, pert)
            neg = jax.tree.map(lambda p, e: p - sigma * e, params, pert)
            return -batch_loss(pos, batch, tgts), -batch_loss(neg, batch, tgts)

        def fitness_chunk(chunk_keys):
            return jax.vmap(member_fitness)(chunk_keys)

        f_pos, f_neg = jax.lax.map(fitness_chunk, keys_chunked)  # (C, ch)
        f_pos, f_neg = f_pos.reshape(-1), f_neg.reshape(-1)      # (pop,)

        diffs = f_pos - f_neg
        diffs = (diffs - jnp.mean(diffs)) / (jnp.std(diffs) + 1e-8)
        diffs_chunked = diffs.reshape(N_CHUNKS, CFG["es_chunk"])

        # ── pass 2: regenerate noise, accumulate weighted gradient ──
        def grad_chunk(carry, xs):
            chunk_keys, chunk_w = xs

            def weighted_pert(mk, w):
                _, pert = perturb_pytree(params, mk, sigma, CFG["es_rank"])
                return jax.tree.map(lambda e: e * w, pert)

            perts = jax.vmap(weighted_pert)(chunk_keys, chunk_w)
            chunk_sum = jax.tree.map(lambda s: jnp.sum(s, axis=0), perts)
            return jax.tree.map(jnp.add, carry, chunk_sum), None

        zero = jax.tree.map(jnp.zeros_like, params)
        es_grad, _ = jax.lax.scan(grad_chunk, zero, (keys_chunked, diffs_chunked))
        es_grad = jax.tree.map(lambda g: g / CFG["es_pop"], es_grad)
        es_grad = jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), es_grad)

        neg_grad = jax.tree.map(lambda g: -g, es_grad)
        updates, opt_state = opt.update(neg_grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        fits = (f_pos + f_neg) / 2.0
        diversity = jnp.std(fits) / (jnp.abs(jnp.mean(fits)) + 1e-8)
        return params, opt_state, -jnp.mean(fits), diversity

    return es_step


def run_eggroll(start_params, steps, seed_offset, label):
    """Run EGGROLL-XL from start_params; returns (best_params, best_loss, hist, sigmas)."""
    opt = optax.chain(
        optax.clip_by_global_norm(CFG["grad_clip"]),
        optax.adam(CFG["es_lr"]),
    )
    opt_state = opt.init(start_params)
    es_step = make_es_step(opt)

    params, best, best_loss = start_params, start_params, float("inf")
    hist, sigmas = [], []
    adaptor = SigmaAdaptor(
        initial_sigma=CFG["es_sigma"],
        min_sigma=CFG["es_sigma_min"],
        max_sigma=CFG["es_sigma_max"],
    )
    sigma = CFG["es_sigma"]
    dk = jax.random.PRNGKey(CFG["seed"] + seed_offset)
    t0 = time.time()
    for step in range(steps):
        dk, bk = jax.random.split(dk)
        sk = jax.random.fold_in(bk, 7)
        batch, tgts = sample_batch(bk)
        params, opt_state, loss, diversity = es_step(
            params, opt_state, batch, tgts, sk, jnp.asarray(sigma)
        )
        sigmas.append(sigma)
        if CFG["es_adaptive_sigma"]:
            sigma = adaptor.update(float(diversity))
        if step % CFG["eval_every"] == 0 or step == steps - 1:
            ev = float(eval_loss(params, eval_seqs, eval_tgts))
            hist.append((step, time.time() - t0, ev))
            if ev < best_loss:
                best_loss, best = ev, params
            print(f"[{label} {step:5d}] train {float(loss):.4f}  eval {ev:.4f}  σ {sigma:.2e}")
    print(f"\n{label} best eval {best_loss:.4f}  ({time.time() - t0:.0f}s)")
    return best, best_loss, hist, sigmas


# %% [markdown]
# ## 8 · Run B — EGGROLL-XL from scratch (pop 512, 3000 steps)

# %%
es_best, es_best_loss, es_eval_hist, sigma_hist = run_eggroll(
    params0, CFG["es_steps"], seed_offset=100, label="ES-XL "
)
es_r2 = 1.0 - es_best_loss / baseline_mean
print(f"ES-XL R² {es_r2:.3f}")

# %% [markdown]
# ## 9 · Run C — warm-start ES from the backprop plateau
#
# Starts where backprop had already found the breakthrough region
# (step-900 checkpoint, eval ≈ plateau). If ES descends from here, it can
# *refine* with local gradient signal present; if it stalls, ES lacks even
# local signal at this population — not just plateau-escape ability.

# %%
warm_best, warm_best_loss, warm_eval_hist, warm_sigma_hist = run_eggroll(
    plateau_params, CFG["warm_steps"], seed_offset=200, label="ES-warm"
)
warm_r2 = 1.0 - warm_best_loss / baseline_mean
print(f"ES-warm R² {warm_r2:.3f}  (started from plateau eval {plateau_eval:.4f})")

# %% [markdown]
# ## 10 · Results

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

ax = axes[0]
ax.plot(*zip(*[(s, v) for s, _, v in bp_eval_hist]), label="backprop (AdamW)", lw=2)
ax.plot(*zip(*[(s, v) for s, _, v in es_eval_hist]), label=f"EGGROLL pop {CFG['es_pop']}", lw=2)
ax.axhline(baseline_mean, color="gray", ls="--", label="mean baseline")
ax.set(xlabel="optimizer step", ylabel="eval MSE", title="Run A vs Run B (from scratch)")
ax.legend(); ax.set_yscale("log")

ax = axes[1]
ax.plot(*zip(*[(s, v) for s, _, v in warm_eval_hist]),
        label="ES-warm (from plateau)", lw=2, color="tab:red")
ax.axhline(plateau_eval, color="tab:blue", ls="--", label=f"plateau start ({plateau_eval:.3f})")
ax.axhline(bp_best_loss, color="tab:blue", ls=":", label=f"backprop best ({bp_best_loss:.3f})")
ax.set(xlabel="ES step", ylabel="eval MSE", title="Run C: warm-start diagnostic")
ax.legend(); ax.set_yscale("log")

ax = axes[2]
ax.plot(sigma_hist, lw=2, label="ES-XL σ")
ax.plot(warm_sigma_hist, lw=2, label="ES-warm σ")
ax.set(xlabel="ES step", ylabel="σ", title="Adaptive σ (clamped)")
ax.legend(); ax.set_yscale("log")

plt.tight_layout(); plt.show()

# %% [markdown]
# ## 11 · Verdict

# %%
ratio = es_r2 / bp_r2 if bp_r2 > 0 else 0.0
# warm-start success: recovers a meaningful share of backprop's post-plateau gain
post_plateau_gain = plateau_eval - bp_best_loss
warm_gain = plateau_eval - warm_best_loss
warm_frac = warm_gain / post_plateau_gain if post_plateau_gain > 0 else 0.0
warm_ok = warm_frac >= 0.25

print("=" * 66)
print(f"  mean baseline          : {baseline_mean:.4f}")
print(f"  backprop  best eval    : {bp_best_loss:.4f}   R² {bp_r2:.3f}   ({bp_time:.0f}s)")
print(f"  ES-XL     best eval    : {es_best_loss:.4f}   R² {es_r2:.3f}   "
      f"(pop {CFG['es_pop']}, {CFG['es_steps']} steps)")
print(f"  R² ratio (ES-XL/BP)    : {ratio:.3f}")
print(f"  plateau start          : {plateau_eval:.4f}")
print(f"  ES-warm   best eval    : {warm_best_loss:.4f}   "
      f"({warm_frac * 100:.0f}% of backprop's post-plateau gain)")
print("=" * 66)

if bp_best_loss > 0.8:
    print("❌ TASK SANITY FAIL — backprop did not clearly beat the mean baseline.")
elif ratio >= 0.90:
    print("✅ PASS — pop-512 EGGROLL matches backprop. v2 was population "
          "starvation → the wager holds; population scale is the engineering "
          "requirement. Proceed to Phase 3 with large-pop EGGROLL.")
elif ratio >= 0.40:
    print("🟡 PARTIAL — large-population ES is learning but trails backprop. "
          "Population/budget scaling trend matters: compare against v2's pop-32 "
          "result before deciding.")
elif warm_ok:
    print("🔵 HYBRID SIGNAL — ES can refine from a good basin (warm-start "
          "descended) but cannot escape the plateau from scratch at this "
          "population. Recommended pivot: backprop pretraining + ES only for "
          "GEA self-improvement (non-differentiable fitness), where ES is "
          "actually needed.")
else:
    print("❌ FAIL — even at pop 512 with local gradient signal available, ES "
          "makes no progress on this backbone. The gradient-free wager is "
          "dead at consumer scale; pivot VELM to a backprop-trained core.")
