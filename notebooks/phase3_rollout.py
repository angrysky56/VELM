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
# # VELM Rollout: the first autoregressive generation
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angrysky56/VELM/blob/main/notebooks/phase3_rollout.ipynb)
#
# Uses the trained phase-3 checkpoints to run VELM's actual inference loop:
#
# ```
# seed chunks → backbone → predict ẑ → AE-decode → tokens → append → repeat
# ```
#
# This measures the thing CALM identifies as the hard part of continuous-
# latent generation: **compounding error**. Each generated chunk feeds the
# next prediction, so errors accumulate. We quantify it with the per-step
# quality decay curve:
#
# - **teacher-forced** (upper bound): predict every position from TRUE history
# - **free-running (direct head)**: deterministic rollout
# - **free-running (energy head)**: stochastic rollout (single sample/step)
#
# Reading the curve: where free-running crosses ~50% of teacher-forced
# quality is the model's usable horizon. Expect rough text at this scale —
# the *shape of the decay*, not prose quality, is the result. Runtime:
# ~10 min after setup.

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

import equinox as eqx
import jax

jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np

from src.model.autoencoder import CALMAutoencoder
from src.model.energy_head import EnergyHead
from src.model.miras_backbone import VELMBackbone

CFG = {
    "seed": 42,
    "chunk_k": 4, "seq_len": 64, "latent_dim": 128,
    "dim": 256, "num_heads": 8, "miras_layers": 4, "swa_layers": 4,
    "ffn_intermediate": 512, "head_blocks": 2, "head_ffn": 512,
    "ae_hidden_dim": 384, "vocab_size": 248077,
    "seed_chunks": 16,      # context given before rollout starts
    "rollout_steps": 32,    # chunks generated free-running
    "num_eval_rollouts": 32,  # sequences for the decay curve
    # which checkpoint set to load (newest first)
    "ckpt_versions": ["v2_best", "v2_final", "best", "final"],
}
K, T, LAT = CFG["chunk_k"], CFG["seq_len"], CFG["latent_dim"]

# %% [markdown]
# ## 2 · Load AE, checkpoints, stats, eval data

# %%
AE_LOCAL = os.path.join(VELM_DIR, "checkpoints", "calm_ae_best.eqx")
AE_DRIVE = "/content/drive/MyDrive/VELM_checkpoints/calm_ae_best.eqx"
ae_path = AE_LOCAL if os.path.exists(AE_LOCAL) else None
if ae_path is None and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ae_path = AE_DRIVE if os.path.exists(AE_DRIVE) else None
assert ae_path, "calm_ae_best.eqx not found"
CKPT_OUT = ("/content/drive/MyDrive/VELM_checkpoints"
            if IN_COLAB and os.path.exists("/content/drive/MyDrive")
            else os.path.join(VELM_DIR, "checkpoints"))

key = jax.random.PRNGKey(CFG["seed"])
key, ak, kb, kh, kd = jax.random.split(key, 5)
frozen_ae = eqx.tree_deserialise_leaves(
    ae_path,
    CALMAutoencoder(vocab_size=CFG["vocab_size"], chunk_size=K,
                    hidden_dim=CFG["ae_hidden_dim"], latent_dim=LAT,
                    ffn_intermediate=768, key=ak))
AE_EMB = frozen_ae.embedding.weight

backbone = VELMBackbone(dim=CFG["dim"], num_heads=CFG["num_heads"],
                        num_miras_layers=CFG["miras_layers"],
                        num_swa_layers=CFG["swa_layers"],
                        ffn_intermediate=CFG["ffn_intermediate"],
                        chunk_size=K, ae_hidden_dim=CFG["ae_hidden_dim"], key=kb)
head = EnergyHead(hidden_dim=CFG["dim"], latent_dim=LAT,
                  num_blocks=CFG["head_blocks"],
                  ffn_intermediate=CFG["head_ffn"], key=kh)
direct = eqx.nn.Linear(CFG["dim"], LAT, key=kd)

loaded = None
for ver in CFG["ckpt_versions"]:
    bb_p = os.path.join(CKPT_OUT, f"backbone_hybrid_{ver}.eqx")
    if os.path.exists(bb_p):
        backbone = eqx.tree_deserialise_leaves(bb_p, backbone)
        direct = eqx.tree_deserialise_leaves(
            os.path.join(CKPT_OUT, f"direct_head_hybrid_{ver}.eqx"), direct)
        eh_p = os.path.join(CKPT_OUT, f"energy_head_hybrid_{ver}.eqx")
        if os.path.exists(eh_p):
            head = eqx.tree_deserialise_leaves(eh_p, head)
        loaded = ver
        break
assert loaded, f"no checkpoints found in {CKPT_OUT}"
print(f"✓ loaded checkpoint set: {loaded}")

_s = np.load(os.path.join(CKPT_OUT, "latent_stats.npz"))
z_mu, z_sd = jnp.asarray(_s["mu"]), jnp.asarray(_s["sd"])

from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
need = CFG["num_eval_rollouts"] * T * K
stream, buf, total = load_dataset("roneneldan/TinyStories", split="train",
                                  streaming=True), [], 0
for ex in stream:
    text = ex.get("text", "")
    if len(text) < 50:
        continue
    ids = [i for i in tokenizer.encode(text, max_length=512, truncation=True)
           if i < CFG["vocab_size"]]
    buf.append(np.asarray(ids, dtype=np.int32))
    total += len(ids)
    if total >= need:
        break
flat = np.concatenate(buf)[: (need // (T * K)) * T * K]
eval_seqs = jnp.asarray(flat.reshape(-1, T, K))  # same fixed stream-order eval
print(f"eval sequences: {eval_seqs.shape}")

encode_seq = eqx.filter_jit(jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0]))

# %% [markdown]
# ## 3 · Rollout machinery

# %%
def hidden_states(tokens_TK):
    """(t, K) tokens → (t, dim) causal hidden states."""
    inp = jax.vmap(backbone.compress_input)(AE_EMB[tokens_TK])
    hid, _ = backbone(inp)
    return hid


def z_to_tokens(z_std):
    """standardized latent → K decoded tokens (argmax)."""
    return jnp.argmax(frozen_ae.decode(z_std * z_sd + z_mu), axis=-1)


def rollout(seq_tokens, mode, rkey):
    """Free-run from seed_chunks; return predicted std-latents (rollout_steps, LAT).

    seq_tokens: (T, K) ground-truth sequence (first seed_chunks used as seed).
    mode: "direct" | "energy"
    """
    S, R = CFG["seed_chunks"], CFG["rollout_steps"]
    toks = np.asarray(seq_tokens[:S])  # growing (t, K) context
    preds = []
    for step in range(R):
        hid = hidden_states(jnp.asarray(toks))
        h = hid[-1]
        if mode == "direct":
            z_hat = direct(h)
        else:
            rkey, sk = jax.random.split(rkey)
            z_hat = head.predict(h, key=sk)
        preds.append(z_hat)
        next_toks = np.asarray(z_to_tokens(z_hat))[None, :]  # decode → re-feed
        toks = np.concatenate([toks, next_toks], axis=0)
    return jnp.stack(preds), toks  # (R, LAT), (S+R, K)


def teacher_forced(seq_tokens):
    """One-step predictions from TRUE history at every rollout position."""
    S, R = CFG["seed_chunks"], CFG["rollout_steps"]
    hid = hidden_states(seq_tokens[: S + R])
    return jax.vmap(direct)(hid[S - 1: S + R - 1])  # (R, LAT)


def cos_per_step(preds, true_z):
    return np.asarray(jnp.sum(preds * true_z, -1) / (
        jnp.linalg.norm(preds, axis=-1) * jnp.linalg.norm(true_z, axis=-1) + 1e-8))


# %% [markdown]
# ## 4 · Decay curves

# %%
S, R = CFG["seed_chunks"], CFG["rollout_steps"]
curves = {"teacher-forced": [], "direct rollout": [], "energy rollout": []}
rkey = jax.random.PRNGKey(7)

for i in range(CFG["num_eval_rollouts"]):
    seq = eval_seqs[i]
    true_z = (encode_seq(seq[S: S + R]) - z_mu) / z_sd  # (R, LAT) targets
    curves["teacher-forced"].append(cos_per_step(teacher_forced(seq), true_z))
    p_d, _ = rollout(seq, "direct", rkey)
    curves["direct rollout"].append(cos_per_step(p_d, true_z))
    rkey, rk2 = jax.random.split(rkey)
    p_e, _ = rollout(seq, "energy", rk2)
    curves["energy rollout"].append(cos_per_step(p_e, true_z))
    if (i + 1) % 8 == 0:
        print(f"  {i + 1}/{CFG['num_eval_rollouts']} rollouts")

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))
for label, cs in curves.items():
    mean = np.mean(np.stack(cs), axis=0)
    plt.plot(range(1, R + 1), mean, lw=2, label=label)
plt.axhline(0, color="black", lw=0.5)
plt.xlabel("rollout step (chunks after seed)")
plt.ylabel("cosine to true next latent (standardized)")
plt.title("Compounding error: free-running vs teacher-forced")
plt.legend(); plt.show()

tf = np.mean(np.stack(curves["teacher-forced"]), axis=0)
dr = np.mean(np.stack(curves["direct rollout"]), axis=0)
half = tf * 0.5
horizon = next((i + 1 for i in range(R) if dr[i] < half[i]), R)
print(f"teacher-forced mean cos: {tf.mean():.3f}")
print(f"direct rollout step-1 cos: {dr[0]:.3f}, step-{R} cos: {dr[-1]:.3f}")
print(f"usable horizon (direct ≥ 50% of teacher-forced): ~{horizon} chunks "
      f"({horizon * K} tokens)")

# %% [markdown]
# ## 5 · Read a generation
#
# Raw decoded text from one direct-head rollout. Expectation management: at
# this scale the value is watching *whether topical drift is gradual or
# instant* — coherent prose needs orders of magnitude more training (see
# `docs/phase3_findings.md`).

# %%
demo = eval_seqs[0]
_, toks_d = rollout(demo, "direct", jax.random.PRNGKey(0))
_, toks_e = rollout(demo, "energy", jax.random.PRNGKey(1))
seed_text = tokenizer.decode(np.asarray(demo[:S]).reshape(-1))
true_cont = tokenizer.decode(np.asarray(demo[S: S + R]).reshape(-1))
print("── SEED ─────────────────────────────────────")
print(seed_text)
print("\n── TRUE CONTINUATION ────────────────────────")
print(true_cont)
print("\n── VELM (direct head, free-running) ─────────")
print(tokenizer.decode(toks_d[S:].reshape(-1)))
print("\n── VELM (energy head, free-running) ─────────")
print(tokenizer.decode(toks_e[S:].reshape(-1)))
