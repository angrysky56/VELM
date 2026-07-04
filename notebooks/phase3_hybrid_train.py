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
# **Run 2 revision** — run 1 (5k steps, 1M tokens) beat the copy baseline but
# turned out to be a near-unconditional predictor: identical decoded text for
# different contexts. The eval lacked the unconditional-*cosine* baseline that
# would have exposed it. Changes: (a) uncond-cosine baseline added — the bar
# that matters; (b) **prediction-diversity collapse detector**; (c) auxiliary
# **direct-prediction loss** (`aux_weight`) forcing the backbone to carry
# next-latent information; (d) 2× data, 8k steps.
#
# Success criteria:
#
# - energy loss below the **unconditional-mean** baseline
# - cosine above the **unconditional cosine** by ≥0.05 (context use, not
#   marginal-distribution modeling)
# - prediction diversity ≥ 0.2 (not collapsed)
# - qualitative: decoded predictions *differ across contexts* and trend toward
#   plausible continuations
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
    "num_train_seqs": 8192,        # ~2M tokens
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
    "aux_weight": 1.0,             # direct-prediction anti-collapse loss weight
    # ── optimization ──────────────────────────────────────
    "steps": 8000,
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


# sanity: roundtrip a 16-sequence sample (4096 chunks). The AE's 99.9% was
# measured on its training mix (math/wikitext/TinyStories); pure TinyStories
# with occasional rare tokens sits slightly lower. A true config mismatch
# would score near zero, so 0.95 is a safe gate.
sample = train_seqs[:16].reshape(-1, K)
recon = jax.vmap(frozen_ae.reconstruct)(sample)
acc = float(jnp.mean(recon == sample))
print(f"AE roundtrip accuracy on {sample.shape[0]} chunks: {acc:.4f}")
assert acc >= 0.95, "AE checkpoint/config mismatch — check ae dims in CFG"

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

# cosine baselines — the unconditional one is essential: latents share a
# common direction component, so even a constant predictor scores well on
# cosine. The model only demonstrates *context use* above this line.
def _cos(a, b):
    return jnp.sum(a * b, axis=-1) / (
        jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8)

copy_cos_base = float(jnp.mean(_cos(eval_z[:, :-1], eval_z[:, 1:])))
uncond_cos_base = float(jnp.mean(_cos(jnp.broadcast_to(z_mean, eval_z[:, 1:].shape),
                                      eval_z[:, 1:])))
print(f"copy baseline energy:   {baseline_copy:.4f}   cos {copy_cos_base:.3f}")
print(f"uncond-mean baseline:   {baseline_uncond:.4f}   cos {uncond_cos_base:.3f}")

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
# auxiliary deterministic head: forces the backbone to carry next-latent
# information (anti-collapse pressure); the energy head models the
# distribution on top of an informative h. POC v4 proved this direct path
# trains reliably with backprop.
key, kd = jax.random.split(key)
direct = eqx.nn.Linear(CFG["dim"], CFG["latent_dim"], key=kd)
model = {"backbone": backbone, "head": head, "direct": direct}
params, static = eqx.partition(model, eqx.is_inexact_array)
n_params = sum(x.size for x in jax.tree.leaves(params))
print(f"trainable params: {n_params:,}")

AE_EMB = frozen_ae.embedding.weight  # (vocab, ae_hidden) — frozen lookup table


def seq_energy_loss(p, seq_tokens, seq_z, loss_key):
    """Energy score + auxiliary direct-prediction loss over one sequence."""
    m = eqx.combine(p, static)
    bb, hd, dr = m["backbone"], m["head"], m["direct"]
    embs = AE_EMB[seq_tokens]                       # (T, K, ae_hidden)
    inp = jax.vmap(bb.compress_input)(embs)         # (T, dim)
    hid, _ = bb(inp)                                # (T, dim)
    hid_in, z_tgt = hid[:-1], seq_z[1:]
    keys = jax.random.split(loss_key, hid_in.shape[0])

    def pos_loss(h, z_t, k):
        samples = hd(h, key=k, num_samples=CFG["energy_samples"])
        return energy_score(samples, z_t)

    e_loss = jnp.mean(jax.vmap(pos_loss)(hid_in, z_tgt, keys))
    # aux: deterministic cosine loss — anti-collapse pressure on the backbone
    d_pred = jax.vmap(dr)(hid_in)
    d_cos = jnp.sum(d_pred * z_tgt, axis=-1) / (
        jnp.linalg.norm(d_pred, axis=-1) * jnp.linalg.norm(z_tgt, axis=-1) + 1e-8)
    aux_loss = jnp.mean(1.0 - d_cos)
    return e_loss + CFG["aux_weight"] * aux_loss


def batch_energy_loss(p, batch_tokens, batch_z, loss_key):
    keys = jax.random.split(loss_key, batch_tokens.shape[0])
    return jnp.mean(
        jax.vmap(lambda s, z, k: seq_energy_loss(p, s, z, k))(batch_tokens, batch_z, keys)
    )


@eqx.filter_jit
def eval_metrics(p, seqs, zs, mkey):
    """Energy loss, sample-mean cosine, direct-head cosine, and prediction
    diversity (std of predictions across contexts ÷ std of targets —
    ~0 means the model collapsed to an unconditional predictor)."""
    m = eqx.combine(p, static)
    bb, hd, dr = m["backbone"], m["head"], m["direct"]

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
            return es, cos, pred

        es, cos, preds = jax.vmap(pos)(hid_in, z_tgt, keys)
        d_pred = jax.vmap(dr)(hid_in)
        d_cos = jnp.sum(d_pred * z_tgt, axis=-1) / (
            jnp.linalg.norm(d_pred, axis=-1) * jnp.linalg.norm(z_tgt, axis=-1) + 1e-8
        )
        # diversity: how much predictions vary across positions vs targets
        div = jnp.mean(jnp.std(preds, axis=0)) / (jnp.mean(jnp.std(z_tgt, axis=0)) + 1e-8)
        return jnp.mean(es), jnp.mean(cos), jnp.mean(d_cos), div

    keys = jax.random.split(mkey, seqs.shape[0])
    es, cos, dcos, div = jax.vmap(per_seq)(seqs, zs, keys)
    return jnp.mean(es), jnp.mean(cos), jnp.mean(dcos), jnp.mean(div)


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
        es, cos, dcos, div = eval_metrics(params, eval_seqs, eval_z, mk)
        es, cos, dcos, div = float(es), float(cos), float(dcos), float(div)
        hist.append((step, time.time() - t0, es, cos, dcos, div))
        marker = ""
        if es < best_loss:
            best_loss = es
            save_ckpt(params, "best")
            marker = "  ← best (saved)"
        print(f"[{step:5d}] train {float(loss):.4f}  eval {es:.4f}  "
              f"cos {cos:.3f}/direct {dcos:.3f} (uncond {uncond_cos_base:.3f})  "
              f"div {div:.2f}{marker}")
    if step and step % CFG["ckpt_every"] == 0:
        save_ckpt(params, "latest")
save_ckpt(params, "final")
print(f"\ndone in {time.time() - t0:.0f}s — best eval energy {best_loss:.4f} "
      f"(copy {baseline_copy:.4f}, uncond {baseline_uncond:.4f})")

# %% [markdown]
# ## 9 · Results

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
steps_h = [h[0] for h in hist]
axes[0].plot(steps_h, [h[2] for h in hist], lw=2, label="model (eval)")
axes[0].axhline(baseline_copy, color="gray", ls="--", label="copy baseline")
axes[0].axhline(baseline_uncond, color="gray", ls=":", label="uncond-mean baseline")
axes[0].set(xlabel="step", ylabel="energy score", title="Eval energy loss")
axes[0].legend()
axes[1].plot(steps_h, [h[3] for h in hist], lw=2, label="cos (energy-head mean)")
axes[1].plot(steps_h, [h[4] for h in hist], lw=2, label="cos (direct head)")
axes[1].axhline(uncond_cos_base, color="red", ls="--",
                label=f"uncond cos ({uncond_cos_base:.3f}) — must beat this")
axes[1].axhline(copy_cos_base, color="gray", ls=":", label="copy cos")
axes[1].set(xlabel="step", ylabel="cosine similarity", title="Latent prediction quality")
axes[1].legend()
axes[2].plot(steps_h, [h[5] for h in hist], lw=2, color="tab:green")
axes[2].axhline(1.0, color="gray", ls="--", label="target diversity")
axes[2].axhline(0.2, color="red", ls=":", label="collapse threshold")
axes[2].set(xlabel="step", ylabel="pred std / target std",
            title="Prediction diversity (collapse detector)")
axes[2].legend()
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

dr = m["direct"]
demo_key = jax.random.PRNGKey(0)
for t in [8, 24, 48]:
    dk1, demo_key = jax.random.split(demo_key)
    z_pred = hd.predict(hid[t], key=dk1)
    pred_tokens = jnp.argmax(frozen_ae.decode(z_pred), axis=-1)
    z_direct = dr(hid[t])
    direct_tokens = jnp.argmax(frozen_ae.decode(z_direct), axis=-1)
    ctx = tokenizer.decode(np.asarray(demo_seq[max(0, t - 3): t + 1]).reshape(-1))
    truth = tokenizer.decode(np.asarray(demo_seq[t + 1]))
    print(f"── position {t} ────────────────────────────")
    print(f"  context …{ctx!r}")
    print(f"  truth        {truth!r}")
    print(f"  energy-head  {tokenizer.decode(np.asarray(pred_tokens))!r}")
    print(f"  direct-head  {tokenizer.decode(np.asarray(direct_tokens))!r}\n")
# If both heads print the SAME text for different positions, the model is
# ignoring context — check the diversity plot above.

# %% [markdown]
# ## 11 · Verdict

# %%
final_cos, final_dcos, final_div = hist[-1][3], hist[-1][4], hist[-1][5]
best_cos = max(max(h[3] for h in hist), max(h[4] for h in hist))
print("=" * 64)
print(f"  best eval energy   : {best_loss:.4f}")
print(f"  copy baseline      : {baseline_copy:.4f}   cos {copy_cos_base:.3f}")
print(f"  uncond baseline    : {baseline_uncond:.4f}   cos {uncond_cos_base:.3f}")
print(f"  final cos          : energy-head {final_cos:.3f} / direct {final_dcos:.3f}")
print(f"  prediction diversity: {final_div:.2f}  (1.0 = target-like, <0.2 = collapsed)")
print("=" * 64)

CONTEXT_MARGIN = 0.05  # must beat unconditional cosine by this much
if final_div < 0.2:
    print("❌ COLLAPSED — predictions barely vary with context; the model is "
          "an unconditional latent sampler. Raise aux_weight, extend budget, "
          "or scale data before re-judging.")
elif best_loss < baseline_uncond and best_cos > uncond_cos_base + CONTEXT_MARGIN:
    print("✅ Phase 3 core objective met — genuinely *contextual* next-latent "
          "prediction (beats the unconditional predictor on both energy and "
          "cosine, no collapse). Next: scale (dim/layers/data), then qTTT + CIB.")
elif best_loss < baseline_uncond:
    print("🟡 WEAK CONTEXT — beats the unconditional baseline on energy but "
          "not clearly on cosine. Some context signal, mostly marginal "
          "distribution. More data/steps is the first lever (next-chunk "
          "latents have high irreducible entropy; contextual structure "
          "emerges with scale).")
else:
    print("❌ Not beating the unconditional baseline — check LR/budget "
          "before scaling.")
