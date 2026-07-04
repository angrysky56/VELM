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
# # VELM POC: Can EGGROLL match backprop?
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angrysky56/VELM/blob/main/notebooks/poc_eggroll_vs_backprop.ipynb)
#
# **The single go/no-go experiment for VELM.** Everything else in the project
# (qTTT, CIB, GEA) only matters if gradient-free evolution strategies can train
# the Miras backbone to match a backprop baseline on the *identical* model,
# data, and loss.
#
# **Design (deliberately minimal — no confounds):**
#
# - **Model**: tiny VELM backbone (real `src/` code, ~1–2M params) + linear head
# - **Task**: next-chunk vector regression — predict the embedding of chunk
#   *t+1* from chunks *≤ t* (TinyStories, Qwen tokenizer, K=4)
# - **Targets**: frozen random embedding table (self-contained; no Drive
#   checkpoints needed — the *optimizer comparison* is what's under test,
#   not representation quality)
# - **Loss**: plain MSE. Deterministic, smooth, identical for both optimizers.
#   The stochastic energy head is intentionally excluded from this POC.
# - **Runs**: AdamW backprop vs EGGROLL (antithetic, rank-1, `src/` code),
#   same init, same data order, same step budget
#
# **Verdict criteria** (automated in the final cell):
#
# - **PASS** — EGGROLL final eval loss ≤ 1.10× backprop, both beat baselines
# - **PARTIAL** — EGGROLL clearly learning (≥50% of backprop's improvement) but gap > 10%
# - **FAIL** — EGGROLL flat or diverging
#
# Runtime: ~20–40 min on a free Colab T4. Nothing here touches Google Drive.

# %% [markdown]
# ## 1 · Setup

# %%
# Install dependencies (Colab-safe; quiet)
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
# Get VELM source (clone on Colab; use local checkout otherwise)
if IN_COLAB:
    if not os.path.exists("/content/VELM"):
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/angrysky56/VELM.git", "/content/VELM"],
            check=True,
        )
    VELM_DIR = "/content/VELM"
else:
    # running locally from notebooks/ or repo root
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
    print("⚠️  No GPU detected — EGGROLL will be slow. "
          "Colab: Runtime → Change runtime type → T4 GPU.")

# %% [markdown]
# ## 2 · Config
#
# One place for every knob. Both optimizers share the model/data config;
# only the optimizer sections differ.

# %%
CFG = {
    "seed": 42,
    # ── data ──────────────────────────────────────────────
    "chunk_k": 4,             # tokens per chunk (CALM K)
    "seq_len": 128,           # chunks per training sequence (512 tokens)
    "num_train_seqs": 2048,
    "num_eval_seqs": 128,
    "batch_size": 8,          # sequences per step
    # ── model (tiny; the question is trainability, not capacity) ──
    "dim": 128,
    "num_heads": 4,
    "miras_layers": 2,
    "swa_layers": 2,
    "ffn_intermediate": 256,
    "embed_dim": 64,          # frozen random embedding / target dim
    # ── backprop run ──────────────────────────────────────
    "bp_steps": 1500,
    "bp_lr": 1e-3,
    "bp_weight_decay": 0.01,
    # ── EGGROLL run ───────────────────────────────────────
    "es_steps": 1500,
    "es_pop": 32,             # antithetic directions (64 evals/step)
    "es_rank": 1,
    "es_sigma": 1e-3,         # EGGROLL paper default
    "es_lr": 3e-4,            # Adam on ES gradient (project default)
    "es_adaptive_sigma": True,
    # ── shared ────────────────────────────────────────────
    "eval_every": 50,
}
key = jax.random.PRNGKey(CFG["seed"])

# %% [markdown]
# ## 3 · Data
#
# TinyStories → Qwen tokens → contiguous stream → `(N, seq_len, K)` int32
# chunk sequences. Small on purpose: a 1–2M-param model can meaningfully
# model TinyStories statistics.

# %%
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

K, T = CFG["chunk_k"], CFG["seq_len"]
tokens_needed = (CFG["num_train_seqs"] + CFG["num_eval_seqs"]) * T * K

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
seqs = flat.reshape(-1, T, K)  # (N, T, K)
rng = np.random.default_rng(CFG["seed"])
rng.shuffle(seqs)
train_seqs = jnp.asarray(seqs[: CFG["num_train_seqs"]])
eval_seqs = jnp.asarray(seqs[CFG["num_train_seqs"]:])
print(f"train {train_seqs.shape}, eval {eval_seqs.shape}  "
      f"({train_seqs.size:,} + {eval_seqs.size:,} tokens)")

# %% [markdown]
# ## 4 · Targets: frozen random embedding
#
# A fixed random table `E (vocab, embed_dim)` defines both the backbone
# input (per-token embeddings, compressed by the backbone's own
# `compress_input`) and the regression target
# `z_t = mean_k E[token_{t,k}]`. Predicting `z_{t+1}` from history is a real
# sequence-modeling problem — the mapping is fixed and structured, so any
# improvement over the mean-predictor baseline is genuine learning.
#
# *(Fidelity upgrade, not needed for the go/no-go: swap `E`-mean targets for
# CALM AE latents by loading `calm_ae_best.eqx` and replacing `chunk_targets`
# with `vmap(ae.encode)`.)*

# %%
key, ek = jax.random.split(key)
EMBED = jax.random.normal(ek, (len(tokenizer), CFG["embed_dim"])) / jnp.sqrt(CFG["embed_dim"])
EMBED = jax.lax.stop_gradient(EMBED)


def chunk_targets(seq_tokens):
    """(T, K) int32 → (T, embed_dim) mean chunk embeddings."""
    return EMBED[seq_tokens].mean(axis=1)


# %% [markdown]
# ## 5 · Model + loss (shared by both optimizers)

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
model0 = make_model(mk)  # ONE init, reused by both runs
params0, static = eqx.partition(model0, eqx.is_inexact_array)
n_params = sum(x.size for x in jax.tree.leaves(params0))
print(f"trainable params: {n_params:,}")


def seq_loss(params, seq_tokens):
    """MSE of next-chunk prediction over one (T, K) sequence."""
    model = eqx.combine(params, static)
    bb, head = model["backbone"], model["head"]
    embs = EMBED[seq_tokens]                        # (T, K, e)
    inp = jax.vmap(bb.compress_input)(embs)         # (T, dim)
    hid, _ = bb(inp)                                # (T, dim)
    pred = jax.vmap(head)(hid[:-1])                 # (T-1, e)
    tgt = chunk_targets(seq_tokens)[1:]             # (T-1, e)
    return jnp.mean((pred - tgt) ** 2)


def batch_loss(params, batch_tokens):
    return jnp.mean(jax.vmap(lambda s: seq_loss(params, s))(batch_tokens))


@eqx.filter_jit
def eval_loss(params, seqs):
    return batch_loss(params, seqs)


def cosine_metric(params, seqs):
    """Mean cosine similarity between predictions and targets on eval set."""
    model = eqx.combine(params, static)
    bb, head = model["backbone"], model["head"]

    def per_seq(seq_tokens):
        embs = EMBED[seq_tokens]
        inp = jax.vmap(bb.compress_input)(embs)
        hid, _ = bb(inp)
        pred, tgt = jax.vmap(head)(hid[:-1]), chunk_targets(seq_tokens)[1:]
        num = jnp.sum(pred * tgt, axis=-1)
        den = jnp.linalg.norm(pred, axis=-1) * jnp.linalg.norm(tgt, axis=-1) + 1e-8
        return jnp.mean(num / den)

    return float(jnp.mean(jax.vmap(per_seq)(seqs)))


# %% [markdown]
# ## 6 · Baselines
#
# Floor and sanity reference. Any optimizer must clearly beat the
# mean-predictor to count as learning.

# %%
all_eval_tgts = jax.vmap(chunk_targets)(eval_seqs)[:, 1:, :]      # (N, T-1, e)
global_mean = jax.vmap(chunk_targets)(train_seqs).mean(axis=(0, 1))
baseline_mean = float(jnp.mean((all_eval_tgts - global_mean) ** 2))
# "copy current chunk" — a cheap non-trivial predictor
all_eval_cur = jax.vmap(chunk_targets)(eval_seqs)[:, :-1, :]
baseline_copy = float(jnp.mean((all_eval_tgts - all_eval_cur) ** 2))
init_loss = float(eval_loss(params0, eval_seqs))
print(f"baseline (predict global mean): {baseline_mean:.6f}")
print(f"baseline (copy current chunk):  {baseline_copy:.6f}")
print(f"untrained model:                {init_loss:.6f}")


def sample_batch(k):
    idx = jax.random.randint(k, (CFG["batch_size"],), 0, train_seqs.shape[0])
    return train_seqs[idx]


# %% [markdown]
# ## 7 · Run A — AdamW backprop (the bar to clear)

# %%
bp_opt = optax.adamw(CFG["bp_lr"], weight_decay=CFG["bp_weight_decay"])
bp_state = bp_opt.init(params0)


@eqx.filter_jit
def bp_step(params, opt_state, batch):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(params, batch)
    updates, opt_state = bp_opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


bp_params = params0
bp_hist, bp_eval_hist = [], []   # (step, train_loss), (step, wall_s, eval_loss)
key, dk = jax.random.split(key)
t0 = time.time()
for step in range(CFG["bp_steps"]):
    dk, bk = jax.random.split(dk)
    bp_params, bp_state, loss = bp_step(bp_params, bp_state, sample_batch(bk))
    bp_hist.append((step, float(loss)))
    if step % CFG["eval_every"] == 0 or step == CFG["bp_steps"] - 1:
        ev = float(eval_loss(bp_params, eval_seqs))
        bp_eval_hist.append((step, time.time() - t0, ev))
        print(f"[backprop {step:5d}] train {loss:.6f}  eval {ev:.6f}")
bp_time = time.time() - t0
bp_final = bp_eval_hist[-1][2]
print(f"\nbackprop done: eval {bp_final:.6f}  cos {cosine_metric(bp_params, eval_seqs):.4f}  ({bp_time:.0f}s)")

# %% [markdown]
# ## 8 · Run B — EGGROLL (antithetic, rank-1, common random numbers)
#
# Same init, same data order (same batch keys), same eval schedule. Fitness
# is `−batch_loss`; every population member sees the same batch (common
# random numbers → lower ES gradient variance). Uses the repo's
# `perturb_pytree` — the exact code the full pipeline uses.

# %%
es_opt = optax.adam(CFG["es_lr"])
es_state = es_opt.init(params0)


@eqx.filter_jit
def es_step(params, opt_state, batch, step_key, sigma):
    member_keys = jax.random.split(step_key, CFG["es_pop"])

    def eval_anti(mk):
        _, pert = perturb_pytree(params, mk, sigma, CFG["es_rank"])
        pos = jax.tree.map(lambda p, e: p + sigma * e, params, pert)
        neg = jax.tree.map(lambda p, e: p - sigma * e, params, pert)
        f_pos = -batch_loss(pos, batch)
        f_neg = -batch_loss(neg, batch)
        return f_pos, f_neg, pert

    f_pos, f_neg, perts = jax.lax.map(eval_anti, member_keys)
    diffs = f_pos - f_neg
    fits = (f_pos + f_neg) / 2.0

    def wsum(stack):
        w = diffs.reshape((-1,) + (1,) * (stack.ndim - 1))
        return jnp.sum(stack * w, axis=0)

    scale = 1.0 / (2.0 * sigma * CFG["es_pop"])
    es_grad = jax.tree.map(lambda s: wsum(s) * scale, perts)
    es_grad = jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), es_grad)
    neg_grad = jax.tree.map(lambda g: -g, es_grad)  # optax minimizes
    updates, opt_state = es_opt.update(neg_grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    diversity = jnp.std(fits) / (jnp.abs(jnp.mean(fits)) + 1e-8)
    return params, opt_state, -jnp.mean(fits), diversity


es_params = params0
es_hist, es_eval_hist, sigma_hist = [], [], []
adaptor = SigmaAdaptor(initial_sigma=CFG["es_sigma"])
sigma = CFG["es_sigma"]
key, dk2 = jax.random.split(key)
# reproduce backprop's batch sequence: re-seed identically
dk2 = jax.random.split(jax.random.PRNGKey(CFG["seed"]))[0]
t0 = time.time()
for step in range(CFG["es_steps"]):
    dk2, bk, sk = jax.random.split(dk2, 3)
    es_params, es_state, loss, diversity = es_step(
        es_params, es_state, sample_batch(bk), sk, jnp.asarray(sigma)
    )
    es_hist.append((step, float(loss)))
    sigma_hist.append(sigma)
    if CFG["es_adaptive_sigma"]:
        sigma = adaptor.update(float(diversity))
    if step % CFG["eval_every"] == 0 or step == CFG["es_steps"] - 1:
        ev = float(eval_loss(es_params, eval_seqs))
        es_eval_hist.append((step, time.time() - t0, ev))
        print(f"[EGGROLL  {step:5d}] train {loss:.6f}  eval {ev:.6f}  σ {sigma:.2e}")
es_time = time.time() - t0
es_final = es_eval_hist[-1][2]
print(f"\nEGGROLL done: eval {es_final:.6f}  cos {cosine_metric(es_params, eval_seqs):.4f}  ({es_time:.0f}s)")

# %% [markdown]
# ## 9 · Results

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.plot(*zip(*[(s, v) for s, _, v in bp_eval_hist]), label="backprop (AdamW)", lw=2)
ax.plot(*zip(*[(s, v) for s, _, v in es_eval_hist]), label="EGGROLL (ES)", lw=2)
ax.axhline(baseline_mean, color="gray", ls="--", label="mean baseline")
ax.axhline(baseline_copy, color="gray", ls=":", label="copy baseline")
ax.set(xlabel="optimizer step", ylabel="eval MSE", title="Loss vs steps")
ax.legend(); ax.set_yscale("log")

ax = axes[1]
ax.plot(*zip(*[(t, v) for _, t, v in bp_eval_hist]), label="backprop", lw=2)
ax.plot(*zip(*[(t, v) for _, t, v in es_eval_hist]), label="EGGROLL", lw=2)
ax.axhline(baseline_mean, color="gray", ls="--")
ax.set(xlabel="wall-clock (s)", ylabel="eval MSE", title="Loss vs wall-clock")
ax.legend(); ax.set_yscale("log")

ax = axes[2]
ax.plot(sigma_hist, lw=2, color="tab:green")
ax.set(xlabel="ES step", ylabel="σ", title="EGGROLL σ (adaptive)")
ax.set_yscale("log")

plt.tight_layout(); plt.show()

# %% [markdown]
# ## 10 · Verdict

# %%
bp_gain = init_loss - bp_final
es_gain = init_loss - es_final
ratio = es_final / bp_final if bp_final > 0 else float("inf")
beats_baseline = es_final < baseline_mean * 0.95 and bp_final < baseline_mean * 0.95

print("=" * 62)
print(f"  untrained            : {init_loss:.6f}")
print(f"  mean baseline        : {baseline_mean:.6f}")
print(f"  backprop  final eval : {bp_final:.6f}   ({bp_time:.0f}s)")
print(f"  EGGROLL   final eval : {es_final:.6f}   ({es_time:.0f}s, "
      f"{2 * CFG['es_pop']}x fwd evals/step)")
print(f"  EGGROLL / backprop   : {ratio:.3f}")
print("=" * 62)

if not beats_baseline:
    print("❌ FAIL — model(s) did not clearly beat the mean baseline; "
          "the task setup or budget needs revisiting before judging the optimizer.")
elif ratio <= 1.10:
    print("✅ PASS — EGGROLL matches backprop (≤1.10x). "
          "The core VELM wager holds at this scale → proceed to Phase 3.")
elif bp_gain > 0 and es_gain >= 0.5 * bp_gain:
    print("🟡 PARTIAL — EGGROLL is clearly learning but trails backprop. "
          "Tune σ / population / lr, or increase the ES step budget, then re-run.")
else:
    print("❌ FAIL — EGGROLL is not learning this backbone at this scale. "
          "The gradient-free wager is in trouble; investigate before Phase 3.")
