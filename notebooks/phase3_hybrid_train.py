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
# # VELM Phase 3: Hybrid training — backbone + energy head on CALM latents
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angrysky56/VELM/blob/main/notebooks/phase3_hybrid_train.ipynb)
#
# **The first real VELM training run under the hybrid architecture**
# (POC verdict: `docs/poc_findings.md` — backprop for pretraining, ES scoped
# to GEA).
#
# Pipeline (all real `src/` components):
#
# 1. **Frozen CALM AE** (99.9% recon, `calm_ae_best.eqx`) encodes K=4 token
#    chunks → 128-d latents `z`.
# 2. **Miras backbone** (dim 256, 4+4 blocks) reads AE token embeddings,
#    compressed per chunk, and produces hidden states `h_t`.
# 3. **Energy head** generates next-latent samples from `h_t`; trained with the
#    **energy score** (strictly proper scoring rule, CALM Eq. 10) against
#    `z_{t+1}`.
# 4. Everything except the AE is trained end-to-end with **AdamW backprop**
#    (warmup + cosine decay).
#
# Success criteria:
#
# - eval energy loss clearly below the **copy baseline** (predict `z_t` for
#   `z_{t+1}`) and the **unconditional mean** baseline
# - cosine similarity of predicted vs target latent above the copy baseline
# - qualitative: decoded predicted chunks are plausible continuations
#
# Requirements: `calm_ae_best.eqx` — either in the repo's `checkpoints/`
# (local) or on Drive at `MyDrive/VELM_checkpoints/` (Colab mounts
# automatically). Runtime: ~45–90 min on a T4 at default budget.

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

from src.model.autoencoder import CALMAutoencoder
from src.model.energy_head import EnergyHead, energy_score
from src.model.miras_backbone import VELMBackbone

print("JAX devices:", jax.devices())

# %% [markdown]
# ## 2 · Config

# %%
CFG = {
    "seed": 42,
    # ── AE checkpoint (frozen) — must match calm_ae_best training config ──
    "tokenizer_id": "Qwen/Qwen3.5-0.8B",
    "vocab_size": 248077,
    "chunk_k": 4,
    "ae_hidden_dim": 384,          # gpu_12gb_v2
    "ae_ffn_intermediate": 768,
    "latent_dim": 128,
    # ── data ──────────────────────────────────────────────
    "seq_len": 64,                 # chunks per sequence (256 tokens)
    "num_train_seqs": 4096,
    "num_eval_seqs": 128,
    "batch_size": 4,
    # ── trainable model ───────────────────────────────────
    "dim": 256,
    "num_heads": 8,
    "miras_layers": 4,
    "swa_layers": 4,
    "ffn_intermediate": 512,
    "head_blocks": 2,
    "head_ffn": 512,
    "energy_samples": 8,           # N for the MC energy-score estimator
    # ── optimization ──────────────────────────────────────
    "steps": 5000,
    "peak_lr": 3e-4,
    "warmup_steps": 200,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "eval_every": 100,
    "ckpt_every": 1000,
}
K, T = CFG["chunk_k"], CFG["seq_len"]
key = jax.random.PRNGKey(CFG["seed"])

# %% [markdown]
# ## 3 · Load the frozen CALM autoencoder

# %%
AE_LOCAL = os.path.join(VELM_DIR, "checkpoints", "calm_ae_best.eqx")
AE_DRIVE = "/content/drive/MyDrive/VELM_checkpoints/calm_ae_best.eqx"

ae_path = AE_LOCAL if os.path.exists(AE_LOCAL) else None
if ae_path is None and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    if os.path.exists(AE_DRIVE):
        ae_path = AE_DRIVE
assert ae_path, "calm_ae_best.eqx not found in checkpoints/ or Drive"

key, ak = jax.random.split(key)
ae_skeleton = CALMAutoencoder(
    vocab_size=CFG["vocab_size"],
    chunk_size=K,
    hidden_dim=CFG["ae_hidden_dim"],
    latent_dim=CFG["latent_dim"],
    ffn_intermediate=CFG["ae_ffn_intermediate"],
    key=ak,
)
frozen_ae = eqx.tree_deserialise_leaves(ae_path, ae_skeleton)
print(f"✓ loaded AE from {ae_path}")

# where trained checkpoints go
CKPT_OUT = ("/content/drive/MyDrive/VELM_checkpoints"
            if IN_COLAB and os.path.exists("/content/drive/MyDrive")
            else os.path.join(VELM_DIR, "checkpoints"))
os.makedirs(CKPT_OUT, exist_ok=True)
print("checkpoints →", CKPT_OUT)

# %% [markdown]
# ## 4 · Data: TinyStories → Qwen tokens → sequences of chunks

# %%
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CFG["tokenizer_id"], trust_remote_code=True)

N_SEQS = CFG["num_train_seqs"] + CFG["num_eval_seqs"]
tokens_needed = N_SEQS * T * K

stream, buf = load_dataset("roneneldan/TinyStories", split="train", streaming=True), []
total = 0
for ex in stream:
    text = ex.get("text", "")
    if len(text) < 50:
        continue
    ids = tokenizer.encode(text, max_length=512, truncation=True)
    ids = [i for i in ids if i < CFG["vocab_size"]]
    buf.append(np.asarray(ids, dtype=np.int32))
    total += len(ids)
    if total >= tokens_needed:
        break

flat = np.concatenate(buf)[: (tokens_needed // (T * K)) * T * K]
seqs_np = flat.reshape(-1, T, K)
rng = np.random.default_rng(CFG["seed"])
seqs_np = seqs_np[rng.permutation(seqs_np.shape[0])]
n_tr = CFG["num_train_seqs"]
train_seqs, eval_seqs = jnp.asarray(seqs_np[:n_tr]), jnp.asarray(seqs_np[n_tr:])
print(f"train {train_seqs.shape}, eval {eval_seqs.shape}")

# %% [markdown]
# ## 5 · AE sanity + precompute latents
#
# The AE is frozen, so every chunk's target latent is encoded exactly once.
# Sanity gate: reconstruction accuracy on a sample must be ≥99% —
# otherwise the checkpoint/config mismatch would poison everything downstream.

# %%
@eqx.filter_jit
def encode_seq(seq_tokens):
    """(T, K) → (T, latent) via frozen AE (deterministic)."""
    return jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(seq_tokens)


# sanity: roundtrip a sample
sample = train_seqs[0]
recon = jax.vmap(frozen_ae.reconstruct)(sample)
acc = float(jnp.mean(recon == sample))
print(f"AE roundtrip accuracy on sample: {acc:.4f}")
assert acc >= 0.99, "AE checkpoint/config mismatch — check ae dims in CFG"

def encode_all(seqs, bs=64):
    outs = []
    for i in range(0, seqs.shape[0], bs):
        outs.append(jax.vmap(encode_seq)(seqs[i: i + bs]))
    return jnp.concatenate(outs, axis=0)

t0 = time.time()
train_z = encode_all(train_seqs)   # (N, T, latent)
eval_z = encode_all(eval_seqs)
print(f"latents: train {train_z.shape}, eval {eval_z.shape}  ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 6 · Baselines
#
# Energy score of two trivial predictors on the eval set (lower is better):
# **copy** (predict `z_t` for `z_{t+1}`) and **unconditional mean latent**.
# The model must beat both — copy is surprisingly strong when consecutive
# chunks are topically similar.

# %%
def degenerate_energy(pred, tgt):
    """Energy score when all N samples equal `pred`: pairwise term = 0."""
    return energy_score(jnp.tile(pred[None, :], (2, 1)), tgt)


z_mean = train_z.reshape(-1, CFG["latent_dim"]).mean(axis=0)
copy_scores = jax.vmap(
    lambda zs: jnp.mean(jax.vmap(degenerate_energy)(zs[:-1], zs[1:]))
)(eval_z)
mean_scores = jax.vmap(
    lambda zs: jnp.mean(jax.vmap(lambda t: degenerate_energy(z_mean, t))(zs[1:]))
)(eval_z)
baseline_copy = float(jnp.mean(copy_scores))
baseline_uncond = float(jnp.mean(mean_scores))
print(f"copy baseline energy:   {baseline_copy:.4f}")
print(f"uncond-mean baseline:   {baseline_uncond:.4f}")

# %% [markdown]
# ## 7 · Trainable model: backbone + energy head

# %%
key, kb, kh = jax.random.split(key, 3)
backbone = VELMBackbone(
    dim=CFG["dim"],
    num_heads=CFG["num_heads"],
    num_miras_layers=CFG["miras_layers"],
    num_swa_layers=CFG["swa_layers"],
    ffn_intermediate=CFG["ffn_intermediate"],
    chunk_size=K,
    ae_hidden_dim=CFG["ae_hidden_dim"],
    key=kb,
)
head = EnergyHead(
    hidden_dim=CFG["dim"],
    latent_dim=CFG["latent_dim"],
    num_blocks=CFG["head_blocks"],
    ffn_intermediate=CFG["head_ffn"],
    key=kh,
)
model = {"backbone": backbone, "head": head}
params, static = eqx.partition(model, eqx.is_inexact_array)
n_params = sum(x.size for x in jax.tree.leaves(params))
print(f"trainable params: {n_params:,}")

AE_EMB = frozen_ae.embedding.weight  # (vocab, ae_hidden) — frozen lookup table


def seq_energy_loss(p, seq_tokens, seq_z, loss_key):
    """Mean energy score of next-latent prediction over one sequence."""
    m = eqx.combine(p, static)
    bb, hd = m["backbone"], m["head"]
    embs = AE_EMB[seq_tokens]                       # (T, K, ae_hidden)
    inp = jax.vmap(bb.compress_input)(embs)         # (T, dim)
    hid, _ = bb(inp)                                # (T, dim)
    hid_in, z_tgt = hid[:-1], seq_z[1:]
    keys = jax.random.split(loss_key, hid_in.shape[0])

    def pos_loss(h, z_t, k):
        samples = hd(h, key=k, num_samples=CFG["energy_samples"])
        return energy_score(samples, z_t)

    return jnp.mean(jax.vmap(pos_loss)(hid_in, z_tgt, keys))


def batch_energy_loss(p, batch_tokens, batch_z, loss_key):
    keys = jax.random.split(loss_key, batch_tokens.shape[0])
    return jnp.mean(
        jax.vmap(lambda s, z, k: seq_energy_loss(p, s, z, k))(batch_tokens, batch_z, keys)
    )


@eqx.filter_jit
def eval_metrics(p, seqs, zs, mkey):
    """Eval energy loss + cosine(prediction-mean, target) vs copy-cosine."""
    m = eqx.combine(p, static)
    bb, hd = m["backbone"], m["head"]

    def per_seq(seq_tokens, seq_z, k):
        embs = AE_EMB[seq_tokens]
        inp = jax.vmap(bb.compress_input)(embs)
        hid, _ = bb(inp)
        hid_in, z_tgt = hid[:-1], seq_z[1:]
        keys = jax.random.split(k, hid_in.shape[0])

        def pos(h, z_t, kk):
            samples = hd(h, key=kk, num_samples=CFG["energy_samples"])
            es = energy_score(samples, z_t)
            pred = samples.mean(axis=0)
            cos = jnp.sum(pred * z_t) / (
                jnp.linalg.norm(pred) * jnp.linalg.norm(z_t) + 1e-8
            )
            return es, cos

        es, cos = jax.vmap(pos)(hid_in, z_tgt, keys)
        copy_cos = jnp.sum(seq_z[:-1] * z_tgt, axis=-1) / (
            jnp.linalg.norm(seq_z[:-1], axis=-1) * jnp.linalg.norm(z_tgt, axis=-1) + 1e-8
        )
        return jnp.mean(es), jnp.mean(cos), jnp.mean(copy_cos)

    keys = jax.random.split(mkey, seqs.shape[0])
    es, cos, ccos = jax.vmap(per_seq)(seqs, zs, keys)
    return jnp.mean(es), jnp.mean(cos), jnp.mean(ccos)


# %% [markdown]
# ## 8 · Train (AdamW, warmup + cosine)

# %%
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=CFG["peak_lr"],
    warmup_steps=CFG["warmup_steps"],
    decay_steps=CFG["steps"],
)
opt = optax.chain(
    optax.clip_by_global_norm(CFG["grad_clip"]),
    optax.adamw(schedule, weight_decay=CFG["weight_decay"]),
)
opt_state = opt.init(params)


@eqx.filter_jit
def train_step(p, o, batch, bz, skey):
    loss, grads = eqx.filter_value_and_grad(batch_energy_loss)(p, batch, bz, skey)
    updates, o = opt.update(grads, o, p)
    return optax.apply_updates(p, updates), o, loss


def save_ckpt(p, tag):
    m = eqx.combine(p, static)
    eqx.tree_serialise_leaves(os.path.join(CKPT_OUT, f"backbone_hybrid_{tag}.eqx"), m["backbone"])
    eqx.tree_serialise_leaves(os.path.join(CKPT_OUT, f"energy_head_hybrid_{tag}.eqx"), m["head"])


best_loss, hist = float("inf"), []
dk = jax.random.PRNGKey(CFG["seed"] + 100)
t0 = time.time()
for step in range(CFG["steps"]):
    dk, bk, sk = jax.random.split(dk, 3)
    idx = jax.random.randint(bk, (CFG["batch_size"],), 0, train_seqs.shape[0])
    params, opt_state, loss = train_step(params, opt_state, train_seqs[idx], train_z[idx], sk)
    if step % CFG["eval_every"] == 0 or step == CFG["steps"] - 1:
        dk, mk = jax.random.split(dk)
        es, cos, ccos = eval_metrics(params, eval_seqs, eval_z, mk)
        es, cos, ccos = float(es), float(cos), float(ccos)
        hist.append((step, time.time() - t0, es, cos))
        marker = ""
        if es < best_loss:
            best_loss = es
            save_ckpt(params, "best")
            marker = "  ← best (saved)"
        print(f"[{step:5d}] train {float(loss):.4f}  eval {es:.4f}  "
              f"cos {cos:.3f} (copy {ccos:.3f}){marker}")
    if step and step % CFG["ckpt_every"] == 0:
        save_ckpt(params, "latest")
save_ckpt(params, "final")
print(f"\ndone in {time.time() - t0:.0f}s — best eval energy {best_loss:.4f} "
      f"(copy {baseline_copy:.4f}, uncond {baseline_uncond:.4f})")

# %% [markdown]
# ## 9 · Results

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
steps_h = [h[0] for h in hist]
axes[0].plot(steps_h, [h[2] for h in hist], lw=2, label="model (eval)")
axes[0].axhline(baseline_copy, color="gray", ls="--", label="copy baseline")
axes[0].axhline(baseline_uncond, color="gray", ls=":", label="uncond-mean baseline")
axes[0].set(xlabel="step", ylabel="energy score", title="Eval energy loss")
axes[0].legend()
axes[1].plot(steps_h, [h[3] for h in hist], lw=2, label="cos(pred, target)")
axes[1].axhline(float(jnp.mean(jnp.sum(eval_z[:, :-1] * eval_z[:, 1:], axis=-1) /
                     (jnp.linalg.norm(eval_z[:, :-1], axis=-1) *
                      jnp.linalg.norm(eval_z[:, 1:], axis=-1) + 1e-8))),
                color="gray", ls="--", label="copy cosine")
axes[1].set(xlabel="step", ylabel="cosine similarity", title="Latent prediction quality")
axes[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 10 · Decode demo: predicted latents → text
#
# For a few eval positions: generate a next-latent sample, push it through the
# frozen AE decoder, and compare the decoded chunk with the ground truth.
# Early in training expect gibberish-adjacent text; watch for topical/local
# coherence emerging.

# %%
m = eqx.combine(params, static)
bb, hd = m["backbone"], m["head"]
demo_seq = eval_seqs[0]
demo_z = eval_z[0]
embs = AE_EMB[demo_seq]
inp = jax.vmap(bb.compress_input)(embs)
hid, _ = bb(inp)

demo_key = jax.random.PRNGKey(0)
for t in [8, 24, 48]:
    dk1, demo_key = jax.random.split(demo_key)
    z_pred = hd.predict(hid[t], key=dk1)
    pred_tokens = jnp.argmax(frozen_ae.decode(z_pred), axis=-1)
    ctx = tokenizer.decode(np.asarray(demo_seq[max(0, t - 3): t + 1]).reshape(-1))
    truth = tokenizer.decode(np.asarray(demo_seq[t + 1]))
    pred = tokenizer.decode(np.asarray(pred_tokens))
    print(f"── position {t} ────────────────────────────")
    print(f"  context …{ctx!r}")
    print(f"  truth    {truth!r}")
    print(f"  predicted{pred!r}\n")

# %% [markdown]
# ## 11 · Verdict

# %%
final_cos = hist[-1][3]
copy_cos_val = float(jnp.mean(jnp.sum(eval_z[:, :-1] * eval_z[:, 1:], axis=-1) /
                     (jnp.linalg.norm(eval_z[:, :-1], axis=-1) *
                      jnp.linalg.norm(eval_z[:, 1:], axis=-1) + 1e-8)))
print("=" * 60)
print(f"  best eval energy : {best_loss:.4f}")
print(f"  copy baseline    : {baseline_copy:.4f}")
print(f"  uncond baseline  : {baseline_uncond:.4f}")
print(f"  final cos(pred,z): {final_cos:.3f}  (copy cos {copy_cos_val:.3f})")
print("=" * 60)
if best_loss < min(baseline_copy, baseline_uncond) and final_cos > copy_cos_val:
    print("✅ Phase 3 core objective met — the hybrid pipeline learns genuine "
          "next-latent prediction. Next: scale (dim/layers/data), then qTTT + CIB.")
elif best_loss < min(baseline_copy, baseline_uncond):
    print("🟡 Beats baselines on energy but not on cosine — likely modeling "
          "distribution spread more than the mode. Try more steps or larger "
          "energy_samples.")
else:
    print("❌ Not beating trivial baselines yet — extend budget, check LR, "
          "or reduce seq_len before scaling.")
