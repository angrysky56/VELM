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
# # VELM POC v2: Can EGGROLL match backprop?
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angrysky56/VELM/blob/main/notebooks/poc_eggroll_vs_backprop.ipynb)
#
# **The single go/no-go experiment for VELM.** Everything else in the project
# (qTTT, CIB, GEA) only matters if gradient-free evolution strategies can train
# the Miras backbone to match a backprop baseline on the *identical* model,
# data, and loss.
#
# **Why v2:** v1 used next-chunk prediction of pooled random embeddings. That
# target is almost pure irreducible entropy — the mean baseline equaled the
# target variance (1/(embed_dim·K)), and even backprop converged *to* the mean
# predictor. Diagnosis: task failure, not optimizer failure. v2 switches to
# **teacher distillation** — the target is a *deterministic function of the
# input* (zero irreducible entropy), and it is exactly VELM Phase 2's real
# objective. Backprop can demonstrably win this task, so the ES parity ratio
# becomes meaningful.
#
# **Design:**
#
# - **Model**: tiny VELM backbone (real `src/` code, ~1–2M params) + linear head
# - **Task**: match the teacher's *contextual* hidden state at each chunk
#   boundary. Teacher = distilgpt2 run over the full sequence, hidden states
#   taken at the last token of each K=4 chunk, JL-projected 768→64,
#   standardized. Contextual targets force the student to use sequence memory —
#   a per-chunk shortcut cannot solve it.
# - **Loss**: plain MSE on standardized targets (mean-predictor baseline ≈ 1.0,
#   so R² = 1 − MSE). Deterministic, smooth, identical for both optimizers.
# - **Runs**: AdamW backprop vs EGGROLL (antithetic, rank-1, `src/` code),
#   same init, same data order, same step budget, both with grad-clip 1.0.
# - **v2 ES stabilizers** (v1 destabilized when σ collapsed): z-scored fitness
#   diffs, σ floor 3e-4 / ceiling 5e-3, best-eval snapshotting.
#
# **Verdict criteria** (automated in the final cell):
#
# - **Sanity** — backprop MSE ≤ 0.8 (clearly beats mean baseline), else the
#   task/budget needs revisiting and the optimizer question stays open
# - **PASS** — EGGROLL R² ≥ 0.90 × backprop R²
# - **PARTIAL** — EGGROLL R² ≥ 0.40 × backprop R² (learning, but trailing)
# - **FAIL** — below that: the gradient-free wager is in trouble
#
# Runtime: ~25–45 min on a free Colab T4. Nothing here touches Google Drive.

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

# %%
CFG = {
    "seed": 42,
    # ── data / teacher ────────────────────────────────────
    "teacher_id": "distilgpt2",   # small, fast, own tokenizer
    "chunk_k": 4,                 # tokens per chunk (CALM K)
    "seq_len": 128,               # chunks per sequence (512 tokens)
    "num_train_seqs": 2048,
    "num_eval_seqs": 128,
    "batch_size": 8,              # sequences per step
    "teacher_batch": 32,          # sequences per teacher forward
    # ── model (tiny; the question is trainability, not capacity) ──
    "dim": 128,
    "num_heads": 4,
    "miras_layers": 2,
    "swa_layers": 2,
    "ffn_intermediate": 256,
    "embed_dim": 64,              # student input embedding & target dim
    # ── backprop run ──────────────────────────────────────
    "bp_steps": 1500,
    "bp_lr": 1e-3,
    "bp_weight_decay": 0.01,
    # ── EGGROLL run ───────────────────────────────────────
    "es_steps": 1500,
    "es_pop": 32,                 # antithetic directions (64 evals/step)
    "es_rank": 1,
    "es_sigma": 1e-3,             # EGGROLL paper default
    "es_sigma_min": 3e-4,         # v2: floor — v1 collapse to 2.5e-4 caused regression
    "es_sigma_max": 5e-3,
    "es_lr": 3e-4,
    "es_adaptive_sigma": True,
    # ── shared ────────────────────────────────────────────
    "grad_clip": 1.0,
    "eval_every": 50,
}
key = jax.random.PRNGKey(CFG["seed"])

# %% [markdown]
# ## 3 · Data
#
# TinyStories → distilgpt2 tokens → contiguous stream → `(N, seq_len, K)`
# int32 chunk sequences.

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
seqs_np = flat.reshape(-1, T, K)  # (N, T, K)
rng = np.random.default_rng(CFG["seed"])
perm = rng.permutation(seqs_np.shape[0])
seqs_np = seqs_np[perm]
print(f"sequences: {seqs_np.shape}  ({seqs_np.size:,} tokens)")

# %% [markdown]
# ## 4 · Targets: contextual teacher hidden states
#
# distilgpt2 runs over each full 512-token sequence; we keep the last-layer
# hidden state at the **last token of every chunk** (positions K−1, 2K−1, …).
# Because the teacher is causal, target *t* is a deterministic function of
# exactly the tokens the student has seen through chunk *t* — zero
# irreducible entropy, but rich sequence structure. JL-project 768→64
# (fixed Gaussian, geometry-preserving), then standardize per-dim on the
# train split so the mean-predictor baseline is ≈ 1.0 and R² = 1 − MSE.

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
    boundary_pos = np.arange(K - 1, T * K, K)  # last token of each chunk

    # fixed JL projection 768 → embed_dim
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
            chunks_out.append(hid @ JL)  # (B, T, embed_dim)

    targets_np = np.concatenate(chunks_out, axis=0)
    targets_np = np.nan_to_num(targets_np, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(TARGET_CACHE, targets_np)
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"✓ extracted targets {targets_np.shape}")

# standardize on the train split only
n_tr = CFG["num_train_seqs"]
mu = targets_np[:n_tr].reshape(-1, CFG["embed_dim"]).mean(axis=0)
sd = targets_np[:n_tr].reshape(-1, CFG["embed_dim"]).std(axis=0) + 1e-6
targets_np = (targets_np - mu) / sd

train_seqs = jnp.asarray(seqs_np[:n_tr])
eval_seqs = jnp.asarray(seqs_np[n_tr:])
train_tgts = jnp.asarray(targets_np[:n_tr])
eval_tgts = jnp.asarray(targets_np[n_tr:])
baseline_mean = float(jnp.mean(eval_tgts ** 2))  # mean predictor ≈ 1.0
print(f"mean-predictor baseline (eval): {baseline_mean:.4f}")

# %% [markdown]
# ## 5 · Model + loss (shared by both optimizers)
#
# Student inputs are frozen random token embeddings — all information enters
# through token identity, none through the teacher.

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
model0 = make_model(mk)  # ONE init, reused by both runs
params0, static = eqx.partition(model0, eqx.is_inexact_array)
n_params = sum(x.size for x in jax.tree.leaves(params0))
print(f"trainable params: {n_params:,}")


def seq_loss(params, seq_tokens, seq_tgt):
    """MSE of same-position teacher matching over one (T, K) sequence."""
    model = eqx.combine(params, static)
    bb, head = model["backbone"], model["head"]
    embs = EMBED[seq_tokens]                        # (T, K, e)
    inp = jax.vmap(bb.compress_input)(embs)         # (T, dim)
    hid, _ = bb(inp)                                # (T, dim)
    pred = jax.vmap(head)(hid)                      # (T, e)
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
# ## 6 · Run A — AdamW backprop (the bar to clear)

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
bp_eval_hist = []  # (step, wall_s, eval_loss)
dk = jax.random.PRNGKey(CFG["seed"] + 100)
t0 = time.time()
for step in range(CFG["bp_steps"]):
    dk, bk = jax.random.split(dk)
    batch, tgts = sample_batch(bk)
    bp_params, bp_state, loss = bp_step(bp_params, bp_state, batch, tgts)
    if step % CFG["eval_every"] == 0 or step == CFG["bp_steps"] - 1:
        ev = float(eval_loss(bp_params, eval_seqs, eval_tgts))
        bp_eval_hist.append((step, time.time() - t0, ev))
        if ev < bp_best_loss:
            bp_best_loss, bp_best = ev, bp_params
        print(f"[backprop {step:5d}] train {float(loss):.4f}  eval {ev:.4f}")
bp_time = time.time() - t0
bp_r2 = 1.0 - bp_best_loss / baseline_mean
print(f"\nbackprop best eval {bp_best_loss:.4f}  R² {bp_r2:.3f}  ({bp_time:.0f}s)")

# %% [markdown]
# ## 7 · Run B — EGGROLL (antithetic, rank-1, common random numbers)
#
# Same init, same batch keys, same eval schedule. v2 stabilizers: fitness
# diffs are z-scored before the ES gradient (scale-invariant updates —
# prevents the blowup v1 showed when σ shrank), gradient clipped at the same
# norm as backprop, σ clamped to [3e-4, 5e-3].

# %%
es_opt = optax.chain(
    optax.clip_by_global_norm(CFG["grad_clip"]),
    optax.adam(CFG["es_lr"]),
)
es_state = es_opt.init(params0)


@eqx.filter_jit
def es_step(params, opt_state, batch, tgts, step_key, sigma):
    member_keys = jax.random.split(step_key, CFG["es_pop"])

    def eval_anti(mk):
        _, pert = perturb_pytree(params, mk, sigma, CFG["es_rank"])
        pos = jax.tree.map(lambda p, e: p + sigma * e, params, pert)
        neg = jax.tree.map(lambda p, e: p - sigma * e, params, pert)
        f_pos = -batch_loss(pos, batch, tgts)
        f_neg = -batch_loss(neg, batch, tgts)
        return f_pos, f_neg, pert

    f_pos, f_neg, perts = jax.lax.map(eval_anti, member_keys)
    diffs = f_pos - f_neg
    # z-score the fitness diffs: scale-invariant ES gradient (OpenAI-ES style)
    diffs = (diffs - jnp.mean(diffs)) / (jnp.std(diffs) + 1e-8)
    fits = (f_pos + f_neg) / 2.0

    def wsum(stack):
        w = diffs.reshape((-1,) + (1,) * (stack.ndim - 1))
        return jnp.sum(stack * w, axis=0)

    es_grad = jax.tree.map(lambda s: wsum(s) / CFG["es_pop"], perts)
    es_grad = jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), es_grad)
    neg_grad = jax.tree.map(lambda g: -g, es_grad)  # optax minimizes
    updates, opt_state = es_opt.update(neg_grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    diversity = jnp.std(fits) / (jnp.abs(jnp.mean(fits)) + 1e-8)
    return params, opt_state, -jnp.mean(fits), diversity


es_params, es_best, es_best_loss = params0, params0, float("inf")
es_eval_hist, sigma_hist = [], []
adaptor = SigmaAdaptor(
    initial_sigma=CFG["es_sigma"],
    min_sigma=CFG["es_sigma_min"],
    max_sigma=CFG["es_sigma_max"],
)
sigma = CFG["es_sigma"]
dk = jax.random.PRNGKey(CFG["seed"] + 100)  # same batch sequence as backprop
t0 = time.time()
for step in range(CFG["es_steps"]):
    dk, bk = jax.random.split(dk)
    sk = jax.random.fold_in(bk, 7)
    batch, tgts = sample_batch(bk)
    es_params, es_state, loss, diversity = es_step(
        es_params, es_state, batch, tgts, sk, jnp.asarray(sigma)
    )
    sigma_hist.append(sigma)
    if CFG["es_adaptive_sigma"]:
        sigma = adaptor.update(float(diversity))
    if step % CFG["eval_every"] == 0 or step == CFG["es_steps"] - 1:
        ev = float(eval_loss(es_params, eval_seqs, eval_tgts))
        es_eval_hist.append((step, time.time() - t0, ev))
        if ev < es_best_loss:
            es_best_loss, es_best = ev, es_params
        print(f"[EGGROLL  {step:5d}] train {float(loss):.4f}  eval {ev:.4f}  σ {sigma:.2e}")
es_time = time.time() - t0
es_r2 = 1.0 - es_best_loss / baseline_mean
print(f"\nEGGROLL best eval {es_best_loss:.4f}  R² {es_r2:.3f}  ({es_time:.0f}s)")

# %% [markdown]
# ## 8 · Results

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.plot(*zip(*[(s, v) for s, _, v in bp_eval_hist]), label="backprop (AdamW)", lw=2)
ax.plot(*zip(*[(s, v) for s, _, v in es_eval_hist]), label="EGGROLL (ES)", lw=2)
ax.axhline(baseline_mean, color="gray", ls="--", label="mean baseline (≈1.0)")
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
ax.set(xlabel="ES step", ylabel="σ", title="EGGROLL σ (adaptive, clamped)")
ax.set_yscale("log")

plt.tight_layout(); plt.show()

# %% [markdown]
# ## 9 · Verdict
#
# On standardized targets, MSE against a ≈1.0 mean-baseline gives
# R² = 1 − MSE: the fraction of teacher-state variance the student explains.

# %%
ratio = es_r2 / bp_r2 if bp_r2 > 0 else 0.0

print("=" * 62)
print(f"  untrained eval MSE   : {init_loss:.4f}")
print(f"  mean baseline        : {baseline_mean:.4f}")
print(f"  backprop  best eval  : {bp_best_loss:.4f}   R² {bp_r2:.3f}   ({bp_time:.0f}s)")
print(f"  EGGROLL   best eval  : {es_best_loss:.4f}   R² {es_r2:.3f}   ({es_time:.0f}s, "
      f"{2 * CFG['es_pop']}x fwd evals/step)")
print(f"  R² ratio (ES/BP)     : {ratio:.3f}")
print("=" * 62)

if bp_best_loss > 0.8:
    print("❌ TASK SANITY FAIL — backprop did not clearly beat the mean baseline. "
          "Fix task/budget before judging the optimizer.")
elif ratio >= 0.90:
    print("✅ PASS — EGGROLL matches backprop (R² ratio ≥ 0.90). "
          "The core VELM wager holds at this scale → proceed to Phase 3.")
elif ratio >= 0.40:
    print("🟡 PARTIAL — EGGROLL is learning but trails backprop. "
          "Tune σ / population / lr or extend the ES budget, then re-run.")
else:
    print("❌ FAIL — EGGROLL is not learning this backbone at this scale. "
          "The gradient-free wager is in trouble; investigate before Phase 3.")
