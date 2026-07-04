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
# # VELM POC v4: Can EGGROLL match backprop?
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angrysky56/VELM/blob/main/notebooks/poc_eggroll_vs_backprop.ipynb)
#
# **The single go/no-go experiment for VELM**, fourth iteration.
#
# **History:**
# - **v1 FAIL (task)** — next-chunk pooled-random-embedding targets were ~pure
#   irreducible entropy; even backprop converged to the mean predictor.
# - **v2 FAIL (population starvation)** — teacher-distillation targets fixed the
#   task; backprop escaped its ~900-step plateau → MSE 0.63. EGGROLL pop=32
#   stuck at the mean solution.
# - **v3 SANITY FAIL — but with the first positive ES signals.** Identical
#   backprop code failed to escape the plateau this time (escape is
#   bifurcation-flaky). Meanwhile pop-512 ES went *below* backprop (0.9795 vs
#   0.9846, still descending at step 3000) and warm-started ES captured 92% of
#   backprop's post-plateau gain. Population scaling works; the task keeps
#   failing us.
#
# **v4 diagnosis:** the plateau is an artifact of **random student input
# embeddings** — the model must first organize a random hash of 50k tokens
# before any sequence learning can begin. That barrier is ours, not VELM's
# (the real pipeline feeds the backbone *trained CALM-AE embeddings*).
#
# **v4 changes:**
#
# 1. Student inputs = the **teacher's own input embeddings** (wte,
#    JL-projected 768→64, standardized). Tokens arrive with semantic geometry;
#    the test becomes sequence dynamics — what we actually care about.
# 2. **Linear-probe baseline**: ridge regression from the current chunk's
#    embeddings to the target. Quantifies the lexical shortcut; the backbone
#    must beat it to demonstrate sequence memory.
# 3. **Two backprop seeds** (best kept) to de-flake the sanity gate.
# 4. Budgets: backprop 4000 steps × 2 seeds, ES 4000 steps (was still
#    descending at v3's 3000).
#
# **Combined verdict** (automated, final cell):
#
# | Run B (pop 512) | Run C (warm start) | Reading |
# |---|---|---|
# | ✅ matches backprop | — | wager holds → proceed to Phase 3 with large-pop EGGROLL |
# | ❌ | ✅ descends | ES refines but can't explore → hybrid VELM (backprop core, ES for GEA) |
# | ❌ | ❌ | wager dead at consumer scale → pivot |
#
# Runtime: ~1.5–2 h on a free Colab T4. No Google Drive needed.

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
    # ── data / teacher ────────────────────────────────────
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
    # ── Run A: backprop (2 seeds, best kept) ──────────────
    "bp_steps": 4000,
    "bp_seed_offsets": [100, 300],
    "bp_lr": 1e-3,
    "bp_weight_decay": 0.01,
    "bp_plateau_step": 900,
    # ── Run B: EGGROLL-XL from scratch ────────────────────
    "es_steps": 4000,
    "es_pop": 512,
    "es_chunk": 64,
    "es_rank": 1,
    "es_sigma": 1e-3,
    "es_sigma_min": 3e-4,
    "es_sigma_max": 5e-3,
    "es_lr": 3e-4,
    "es_adaptive_sigma": True,
    # ── Run C: warm-start ES from backprop plateau ────────
    "warm_steps": 1000,
    # ── probe baseline ────────────────────────────────────
    "probe_lambda": 1e-2,
    "probe_train_seqs": 512,
    # ── shared ────────────────────────────────────────────
    "grad_clip": 1.0,
    "eval_every": 50,
}
assert CFG["es_pop"] % CFG["es_chunk"] == 0
key = jax.random.PRNGKey(CFG["seed"])

# %% [markdown]
# ## 3 · Data (identical to v2/v3)

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
# ## 4 · Teacher targets + student input embeddings
#
# Targets (unchanged from v2/v3): distilgpt2 *contextual* hidden states at
# chunk boundaries, JL-projected 768→64, standardized (R² = 1 − MSE).
#
# **New in v4:** student input embeddings are the teacher's input embedding
# table (wte), JL-projected 768→64 and standardized — semantic geometry in,
# so the experiment measures sequence modeling, not lexicon reconstruction.

# %%
CKPT_DIR = os.path.join(VELM_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
TARGET_CACHE = os.path.join(CKPT_DIR, "poc_teacher_targets.npy")
EMBED_CACHE = os.path.join(CKPT_DIR, "poc_student_embed.npy")

need_targets = not os.path.exists(TARGET_CACHE)
need_embed = not os.path.exists(EMBED_CACHE)

if need_targets or need_embed:
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

    if need_embed:
        wte = teacher.get_input_embeddings().weight.detach().float().cpu().numpy()
        proj_rng = np.random.default_rng(CFG["seed"] + 2)
        JL_in = proj_rng.standard_normal((t_dim, CFG["embed_dim"])).astype(np.float32)
        JL_in /= np.sqrt(CFG["embed_dim"])
        emb = wte @ JL_in                                   # (vocab, 64)
        emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)     # standardize
        emb /= np.sqrt(CFG["embed_dim"])                    # unit-ish norms
        np.save(EMBED_CACHE, emb.astype(np.float32))
        print(f"✓ built student embeddings {emb.shape}")

    if need_targets:
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
        print(f"✓ extracted targets {targets_np.shape}")

    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()

targets_np = np.load(TARGET_CACHE)
assert targets_np.shape[:2] == seqs_np.shape[:2], "stale cache — delete and re-run"
EMBED = jnp.asarray(np.load(EMBED_CACHE))
print(f"targets {targets_np.shape}, student embed {EMBED.shape}")

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
# ## 5 · Linear-probe baseline
#
# Ridge regression from the *current chunk's* embeddings (K·64 = 256 features)
# to the target. This captures everything achievable without sequence memory.
# The backbone only demonstrates value beyond a lexical lookup if it beats
# this number.

# %%
Xp = np.asarray(EMBED)[np.asarray(train_seqs[: CFG["probe_train_seqs"]])]  # (S,T,K,e)
Xp = Xp.reshape(-1, K * CFG["embed_dim"])
Yp = np.asarray(train_tgts[: CFG["probe_train_seqs"]]).reshape(-1, CFG["embed_dim"])
XtX = Xp.T @ Xp + CFG["probe_lambda"] * Xp.shape[0] * np.eye(Xp.shape[1], dtype=np.float32)
W_probe = np.linalg.solve(XtX, Xp.T @ Yp)
b_probe = Yp.mean(0) - Xp.mean(0) @ W_probe

Xe = np.asarray(EMBED)[np.asarray(eval_seqs)].reshape(-1, K * CFG["embed_dim"])
Ye = np.asarray(eval_tgts).reshape(-1, CFG["embed_dim"])
probe_mse = float(np.mean((Xe @ W_probe + b_probe - Ye) ** 2))
probe_r2 = 1.0 - probe_mse / baseline_mean
print(f"linear probe (current chunk only): MSE {probe_mse:.4f}  R² {probe_r2:.3f}")

# %% [markdown]
# ## 6 · Model + loss (shared)

# %%
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
# ## 7 · Run A — AdamW backprop, 2 seeds, best kept

# %%
bp_opt = optax.chain(
    optax.clip_by_global_norm(CFG["grad_clip"]),
    optax.adamw(CFG["bp_lr"], weight_decay=CFG["bp_weight_decay"]),
)


@eqx.filter_jit
def bp_step(params, opt_state, batch, tgts):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(params, batch, tgts)
    updates, opt_state = bp_opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


def run_backprop(seed_offset):
    params, opt_state = params0, bp_opt.init(params0)
    best, best_loss, plateau = params0, float("inf"), None
    hist = []
    dk = jax.random.PRNGKey(CFG["seed"] + seed_offset)
    t0 = time.time()
    for step in range(CFG["bp_steps"]):
        dk, bk = jax.random.split(dk)
        batch, tgts = sample_batch(bk)
        params, opt_state, loss = bp_step(params, opt_state, batch, tgts)
        if step == CFG["bp_plateau_step"]:
            plateau = params
        if step % CFG["eval_every"] == 0 or step == CFG["bp_steps"] - 1:
            ev = float(eval_loss(params, eval_seqs, eval_tgts))
            hist.append((step, time.time() - t0, ev))
            if ev < best_loss:
                best_loss, best = ev, params
            if step % 500 == 0 or step == CFG["bp_steps"] - 1:
                print(f"[bp seed+{seed_offset} {step:5d}] eval {ev:.4f}")
    return {"best": best, "best_loss": best_loss, "plateau": plateau,
            "hist": hist, "time": time.time() - t0}


bp_runs = [run_backprop(o) for o in CFG["bp_seed_offsets"]]
bp_run = min(bp_runs, key=lambda r: r["best_loss"])
bp_best, bp_best_loss = bp_run["best"], bp_run["best_loss"]
bp_eval_hist, bp_time = bp_run["hist"], sum(r["time"] for r in bp_runs)
plateau_params = bp_run["plateau"]
plateau_eval = float(eval_loss(plateau_params, eval_seqs, eval_tgts))
bp_r2 = 1.0 - bp_best_loss / baseline_mean
print(f"\nbackprop best-of-{len(bp_runs)} eval {bp_best_loss:.4f}  R² {bp_r2:.3f}  "
      f"(all seeds: {[round(r['best_loss'], 4) for r in bp_runs]})")
print(f"plateau checkpoint eval: {plateau_eval:.4f}")

# %% [markdown]
# ## 8 · EGGROLL-XL machinery (unchanged from v3, verified two-pass)
#
# Fitness evaluated in vmapped chunks; perturbations regenerated from RNG keys
# in a second gradient pass (the paper's memory design). Fitness diffs
# z-scored, gradient clipped, σ adaptive within [3e-4, 5e-3].

# %%
N_CHUNKS = CFG["es_pop"] // CFG["es_chunk"]


def make_es_step(opt):
    @eqx.filter_jit
    def es_step(params, opt_state, batch, tgts, step_key, sigma):
        member_keys = jax.random.split(step_key, CFG["es_pop"])
        keys_chunked = member_keys.reshape(N_CHUNKS, CFG["es_chunk"], 2)

        def member_fitness(mk):
            _, pert = perturb_pytree(params, mk, sigma, CFG["es_rank"])
            pos = jax.tree.map(lambda p, e: p + sigma * e, params, pert)
            neg = jax.tree.map(lambda p, e: p - sigma * e, params, pert)
            return -batch_loss(pos, batch, tgts), -batch_loss(neg, batch, tgts)

        f_pos, f_neg = jax.lax.map(
            lambda ck: jax.vmap(member_fitness)(ck), keys_chunked
        )
        f_pos, f_neg = f_pos.reshape(-1), f_neg.reshape(-1)

        diffs = f_pos - f_neg
        diffs = (diffs - jnp.mean(diffs)) / (jnp.std(diffs) + 1e-8)
        diffs_chunked = diffs.reshape(N_CHUNKS, CFG["es_chunk"])

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
            if step % 500 == 0 or step == steps - 1:
                print(f"[{label} {step:5d}] train {float(loss):.4f}  eval {ev:.4f}  σ {sigma:.2e}")
    print(f"{label} best eval {best_loss:.4f}  ({time.time() - t0:.0f}s)")
    return best, best_loss, hist, sigmas


# %% [markdown]
# ## 9 · Run B — EGGROLL-XL from scratch (pop 512, 4000 steps)

# %%
es_best, es_best_loss, es_eval_hist, sigma_hist = run_eggroll(
    params0, CFG["es_steps"], seed_offset=100, label="ES-XL "
)
es_r2 = 1.0 - es_best_loss / baseline_mean
print(f"ES-XL R² {es_r2:.3f}")

# %% [markdown]
# ## 10 · Run C — warm-start ES from the backprop plateau

# %%
warm_best, warm_best_loss, warm_eval_hist, warm_sigma_hist = run_eggroll(
    plateau_params, CFG["warm_steps"], seed_offset=200, label="ES-warm"
)
warm_r2 = 1.0 - warm_best_loss / baseline_mean
print(f"ES-warm R² {warm_r2:.3f}  (started from plateau eval {plateau_eval:.4f})")

# %% [markdown]
# ## 11 · Results

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

ax = axes[0]
for i, r in enumerate(bp_runs):
    ax.plot(*zip(*[(s, v) for s, _, v in r["hist"]]),
            lw=2, alpha=0.8, label=f"backprop seed {i}")
ax.plot(*zip(*[(s, v) for s, _, v in es_eval_hist]),
        label=f"EGGROLL pop {CFG['es_pop']}", lw=2, color="tab:orange")
ax.axhline(baseline_mean, color="gray", ls="--", label="mean baseline")
ax.axhline(probe_mse, color="purple", ls="-.", label=f"linear probe ({probe_mse:.3f})")
ax.set(xlabel="optimizer step", ylabel="eval MSE", title="Run A vs Run B (from scratch)")
ax.legend(); ax.set_yscale("log")

ax = axes[1]
ax.plot(*zip(*[(s, v) for s, _, v in warm_eval_hist]),
        label="ES-warm (from plateau)", lw=2, color="tab:red")
ax.axhline(plateau_eval, color="tab:blue", ls="--", label=f"plateau start ({plateau_eval:.3f})")
ax.axhline(bp_best_loss, color="tab:blue", ls=":", label=f"backprop best ({bp_best_loss:.3f})")
ax.axhline(probe_mse, color="purple", ls="-.", label="linear probe")
ax.set(xlabel="ES step", ylabel="eval MSE", title="Run C: warm-start diagnostic")
ax.legend(); ax.set_yscale("log")

ax = axes[2]
ax.plot(sigma_hist, lw=2, label="ES-XL σ")
ax.plot(warm_sigma_hist, lw=2, label="ES-warm σ")
ax.set(xlabel="ES step", ylabel="σ", title="Adaptive σ (clamped)")
ax.legend(); ax.set_yscale("log")

plt.tight_layout(); plt.show()

# %% [markdown]
# ## 12 · Verdict

# %%
ratio = es_r2 / bp_r2 if bp_r2 > 0 else 0.0
post_plateau_gain = plateau_eval - bp_best_loss
warm_gain = plateau_eval - warm_best_loss
warm_frac = warm_gain / post_plateau_gain if post_plateau_gain > 0 else 0.0
warm_ok = warm_frac >= 0.25
uses_memory = bp_best_loss < probe_mse  # beats the no-memory lexical shortcut

print("=" * 66)
print(f"  mean baseline          : {baseline_mean:.4f}")
print(f"  linear probe (no mem)  : {probe_mse:.4f}   R² {probe_r2:.3f}")
print(f"  backprop  best eval    : {bp_best_loss:.4f}   R² {bp_r2:.3f}   "
      f"(best of {len(bp_runs)} seeds)")
print(f"  ES-XL     best eval    : {es_best_loss:.4f}   R² {es_r2:.3f}   "
      f"(pop {CFG['es_pop']}, {CFG['es_steps']} steps)")
print(f"  R² ratio (ES-XL/BP)    : {ratio:.3f}")
print(f"  plateau start          : {plateau_eval:.4f}")
print(f"  ES-warm   best eval    : {warm_best_loss:.4f}   "
      f"({warm_frac * 100:.0f}% of backprop's post-plateau gain)")
print("=" * 66)

if bp_best_loss > 0.8 * baseline_mean:
    print("❌ TASK SANITY FAIL — backprop did not clearly beat the mean "
          "baseline even with semantic input embeddings and 2 seeds × "
          f"{CFG['bp_steps']} steps. The bottleneck is model capacity or task "
          "design, not the optimizer. Investigate before judging EGGROLL.")
else:
    if not uses_memory:
        print("⚠️  NOTE — backprop beat the mean baseline but NOT the linear "
              "probe: the model is exploiting the lexical shortcut, not "
              "sequence memory. Optimizer comparison below is still valid, "
              "but the backbone's memory value is unproven on this task.")
    if ratio >= 0.90:
        print("✅ PASS — pop-512 EGGROLL matches backprop. The wager holds; "
              "population scale is the engineering requirement. Proceed to "
              "Phase 3 with large-pop EGGROLL.")
    elif ratio >= 0.40:
        print("🟡 PARTIAL — large-population ES is learning but trails "
              "backprop. Compare the pop-32 → pop-512 scaling trend and "
              "consider a bigger population or longer budget before deciding.")
    elif warm_ok:
        print("🔵 HYBRID SIGNAL — ES refines from a good basin but cannot "
              "match backprop from scratch. Recommended pivot: backprop "
              "pretraining + ES only for GEA self-improvement.")
    else:
        print("❌ FAIL — ES makes no meaningful progress even at pop 512. "
              "The gradient-free wager is dead at consumer scale; pivot VELM "
              "to a backprop-trained core.")
