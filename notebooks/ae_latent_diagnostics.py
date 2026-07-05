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
# # CALM AE Latent-Space Diagnostics: is the latent space *learnable*?
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angrysky56/VELM/blob/main/notebooks/ae_latent_diagnostics.ipynb)
#
# **Motivation.** Phase 3 run 3 showed the backbone extracts *zero* contextual
# signal from next-latent prediction (centered cos 0.005), while POC v4 proved
# the identical backbone easily learns contextual structure in a *semantic*
# target space (R² 0.628). Hypothesis: the CALM AE — trained for reconstruction
# only (99.9%, β_KL = 0.001) — produced a **hash-like latent space**: perfect
# for storing 4 tokens, but geometrically unstructured, so E[z_next | context]
# ≈ global mean and *no* downstream predictor can win. This is exactly the
# reconstruction-vs-learnability tension the CALM paper designs around.
#
# Two tests, no training, ~10 min on CPU or any GPU:
#
# - **Test A — smoothness**: does changing ONE token of a chunk move its latent
#   nearly as far as swapping to a completely different chunk? (hash signature)
# - **Test B — oracle predictability**: can the *best linear probe* from the
#   previous W latents predict the next latent above centered-cos ≈ 0? If even
#   the oracle fails, the space is unpredictable — the AE is the blocker,
#   independent of backbone, optimizer, or scale.
#
# Verdict guides the fix: retrain the AE with learnability regularization
# (stronger KL, latent noise) and adopt "oracle predictability" as a Phase 2
# acceptance criterion alongside reconstruction.

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
import jax.numpy as jnp
import numpy as np

from src.model.autoencoder import CALMAutoencoder

K, LATENT = 4, 128

# %% [markdown]
# ## 2 · Load AE + data

# %%
AE_LOCAL = os.path.join(VELM_DIR, "checkpoints", "calm_ae_best.eqx")
AE_DRIVE = "/content/drive/MyDrive/VELM_checkpoints/calm_ae_best.eqx"
ae_path = AE_LOCAL if os.path.exists(AE_LOCAL) else None
if ae_path is None and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ae_path = AE_DRIVE if os.path.exists(AE_DRIVE) else None
assert ae_path, "calm_ae_best.eqx not found"

ae = CALMAutoencoder(vocab_size=248077, chunk_size=K, hidden_dim=384,
                     latent_dim=LATENT, ffn_intermediate=768,
                     key=jax.random.PRNGKey(0))
ae = eqx.tree_deserialise_leaves(ae_path, ae)
print(f"✓ AE loaded from {ae_path}")

from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

N_CHUNKS = 30000
stream, buf, total = load_dataset("roneneldan/TinyStories", split="train",
                                  streaming=True), [], 0
for ex in stream:
    text = ex.get("text", "")
    if len(text) < 50:
        continue
    ids = [i for i in tokenizer.encode(text, max_length=512, truncation=True)
           if i < 248077]
    usable = len(ids) - len(ids) % K
    if usable >= K:
        buf.append(np.array(ids[:usable], dtype=np.int32).reshape(-1, K))
        total += usable // K
    if total >= N_CHUNKS:
        break
chunks = jnp.asarray(np.concatenate(buf)[:N_CHUNKS])  # contiguous within docs
print(f"chunks: {chunks.shape}")

encode = eqx.filter_jit(
    jax.vmap(lambda c: ae.encode(c, training=False)[0])
)
def encode_all(x, bs=1024):
    return jnp.concatenate([encode(x[i:i + bs]) for i in range(0, x.shape[0], bs)])

Z = encode_all(chunks)  # (N, latent)
print(f"latents: {Z.shape}")

def cos(a, b):
    return jnp.sum(a * b, axis=-1) / (
        jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8)

# %% [markdown]
# ## 3 · Test A — smoothness
#
# For 2,000 chunks: replace ONE random position with a random *common* token,
# re-encode, and measure latent cosine to the original. Compare against
# (a) identical chunks (control, cos = 1) and (b) random chunk pairs (floor).
#
# Reading: a smooth, learnable space keeps 1-token-changed chunks much closer
# than random pairs (they share 3/4 of their content). A hash-like space
# scatters them to the random-pair floor — the signature that killed Phase 3.

# %%
rng = np.random.default_rng(0)
n_test = 2000
idx = rng.choice(N_CHUNKS, n_test, replace=False)
orig = np.asarray(chunks[idx])
# perturb one position with a random common token (ids 1000–20000 ≈ frequent)
pert = orig.copy()
pos = rng.integers(0, K, n_test)
pert[np.arange(n_test), pos] = rng.integers(1000, 20000, n_test)

z_orig = encode_all(jnp.asarray(orig))
z_pert = encode_all(jnp.asarray(pert))
cos_pert = cos(z_orig, z_pert)                                  # 1-token change
cos_rand = cos(z_orig, z_orig[rng.permutation(n_test)])         # random pairs

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.hist(np.asarray(cos_pert), bins=60, alpha=0.6, label="1-token changed", density=True)
plt.hist(np.asarray(cos_rand), bins=60, alpha=0.6, label="random pairs", density=True)
plt.xlabel("latent cosine similarity"); plt.ylabel("density")
plt.title("Test A — latent smoothness"); plt.legend(); plt.show()

m_pert, m_rand = float(jnp.mean(cos_pert)), float(jnp.mean(cos_rand))
# smoothness margin: how much closer a 75%-identical chunk stays vs random
smooth_margin = (m_pert - m_rand) / (1.0 - m_rand + 1e-8)
print(f"mean cos — 1-token changed: {m_pert:.3f}   random pairs: {m_rand:.3f}")
print(f"smoothness margin: {smooth_margin:.3f}  (1 = perfectly smooth, 0 = hash-like)")

# %% [markdown]
# ## 4 · Test B — oracle predictability
#
# Ridge probe: previous W=4 latents (concatenated, 512 features) → next latent,
# fit on 80% of positions, scored on 20% with **centered cosine** (the same
# metric Phase 3 uses; unconditional predictor = 0). This is a *best-case
# linear oracle*: if it can't beat ~0, no sequence model can be expected to —
# the latent space itself carries no exploitable next-step structure.

# %%
W = 4
Xo = np.concatenate([np.asarray(Z[i:N_CHUNKS - W + i]) for i in range(W)], axis=1)
Yo = np.asarray(Z[W:])
n = Xo.shape[0]
split = int(n * 0.8)
mu_y = Yo[:split].mean(0)

lam = 1e-1
XtX = Xo[:split].T @ Xo[:split] + lam * split * np.eye(Xo.shape[1], dtype=np.float32)
Wp = np.linalg.solve(XtX, Xo[:split].T @ (Yo[:split] - mu_y))
pred = Xo[split:] @ Wp + mu_y

pc, tc = pred - mu_y, Yo[split:] - mu_y
oracle_ccos = float(np.mean(
    np.sum(pc * tc, -1) /
    (np.linalg.norm(pc, axis=-1) * np.linalg.norm(tc, axis=-1) + 1e-8)))
# reference: same probe, shuffled targets (chance floor)
sh = rng.permutation(tc.shape[0])
chance_ccos = float(np.mean(
    np.sum(pc * tc[sh], -1) /
    (np.linalg.norm(pc, axis=-1) * np.linalg.norm(tc[sh], axis=-1) + 1e-8)))
print(f"oracle centered cosine: {oracle_ccos:.4f}   (chance: {chance_ccos:.4f})")

# %% [markdown]
# ## 5 · Test C — anisotropy & PCA-whitening (LatentGate, ACL 2026)
#
# LatentGate (Ratnakar et al., 2026) shows causal-LM representations collapse
# into a narrow cone ("representation anisotropy") and that **PCA-whitening**
# (decorrelation + variance normalization) — not mere per-dim standardization
# — restores discriminative geometry (+17.2 pts in their ablation vs +8.8 for
# StandardScaler alone). Our phase-3 standardization fix was exactly their
# StandardScaler row. This test asks whether full whitening buys more:
#
# 1. eigenspectrum of the *standardized* latents (residual anisotropy)
# 2. the same ridge oracle, recomputed on **whitened** latents
#
# If the whitened oracle clearly beats the standardized one, whitening is the
# next stage-1 upgrade (a purely linear change to the target space).

# %%
Zs = np.asarray(Z)  # raw latents (N, LAT)
mu_all = Zs.mean(0)
sd_all = Zs.std(0) + 1e-6
Z_std = (Zs - mu_all) / sd_all

# eigenspectrum of standardized latents
cov = np.cov(Z_std.T)
eig = np.linalg.eigvalsh(cov)[::-1]
frac = eig / eig.sum()
print(f"variance in top 3 / 8 / 16 PCs: "
      f"{frac[:3].sum():.1%} / {frac[:8].sum():.1%} / {frac[:16].sum():.1%}")

plt.figure(figsize=(7, 3.5))
plt.bar(range(32), frac[:32])
plt.xlabel("principal component"); plt.ylabel("variance fraction")
plt.title("Eigenspectrum of standardized CALM latents")
plt.show()

# PCA-whitening transform (full rank, ε-regularized)
evals, evecs = np.linalg.eigh(cov)
W_wh = evecs @ np.diag(1.0 / np.sqrt(evals + 1e-5)) @ evecs.T  # ZCA whitening
Z_wh = Z_std @ W_wh


def window_oracle(Zmat, w=4, lam=1e-1, split_frac=0.8):
    X = np.concatenate([Zmat[i: Zmat.shape[0] - w + i] for i in range(w)], axis=1)
    Y = Zmat[w:]
    s = int(X.shape[0] * split_frac)
    m = Y[:s].mean(0)
    XtX = X[:s].T @ X[:s] + lam * s * np.eye(X.shape[1], dtype=np.float32)
    Wp = np.linalg.solve(XtX, X[:s].T @ (Y[:s] - m))
    p = X[s:] @ Wp
    pc, tc = p, Y[s:] - m
    return float(np.mean(np.sum(pc * tc, -1) /
                 (np.linalg.norm(pc, axis=-1) * np.linalg.norm(tc, axis=-1) + 1e-8)))


oracle_std = window_oracle(Z_std)
oracle_wh = window_oracle(Z_wh)
print(f"ridge oracle — standardized: {oracle_std:.4f}   whitened: {oracle_wh:.4f}")
if oracle_wh > oracle_std * 1.15:
    print("✅ WHITENING PAYS — residual anisotropy is hiding predictive "
          "structure. Stage-1b: train with ZCA-whitened targets (freeze W_wh "
          "with the stats; de-whiten before AE decode). Existing stage-1 "
          "models can be re-scored in whitened space linearly, no retrain.")
else:
    print("— whitening ≈ standardization here; per-dim scaling already "
          "captured the gain. No change needed.")

# %% [markdown]
# ## 6 · Verdict

# %%
print("=" * 64)
print(f"  Test A smoothness margin : {smooth_margin:.3f}  (0 = hash-like)")
print(f"  Test B oracle centered cos: {oracle_ccos:.4f}  (chance {chance_ccos:.4f})")
print("=" * 64)

if smooth_margin < 0.25 and oracle_ccos < 0.05:
    print("❌ HASH-LIKE LATENT SPACE CONFIRMED — the AE is the Phase 3 "
          "blocker. Latents are unpredictable regardless of downstream model. "
          "Fix: retrain the AE for *learnability*, not just reconstruction:")
    print("   • kl_weight 0.001 → 0.01–0.05 (stronger information bottleneck)")
    print("   • add latent-space noise during training (robust codes)")
    print("   • acceptance criterion: recon ≥ 99% AND oracle centered-cos ≥ 0.15")
elif oracle_ccos < 0.05:
    print("⚠️  Space is locally smooth but next-step structure is still "
          "linear-invisible; a nonlinear oracle (small MLP probe) is worth "
          "one check before retraining the AE.")
elif oracle_ccos >= 0.15:
    print("✅ LATENT SPACE IS PREDICTABLE — the oracle finds real structure "
          f"(centered cos {oracle_ccos:.3f}). The Phase 3 failure lies in the "
          "backbone/training config after all: revisit capacity, LR, and "
          "aux_weight; the target space is not the excuse.")
else:
    print("🟡 MARGINAL — weak but nonzero oracle signal. AE retraining with "
          "stronger regularization will likely pay, but a Phase 3 rerun with "
          "10× data could also expose it. Cheapest decisive move: retrain AE "
          "at kl_weight=0.01 and re-run this diagnostic.")
