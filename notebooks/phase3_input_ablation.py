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
# # Phase 3 Input Ablation: where does the signal die?
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angrysky56/VELM/blob/main/notebooks/phase3_input_ablation.ipynb)
#
# Eight Phase 3 runs (standardization, TF32, LR schedules, batch size, weight
# decay) all converged to the same plateau: centered cos ≈ 0.005–0.02 against
# a linear-oracle bar of ≈ 0.13. That is no longer a tuning problem. This
# notebook trains **four models with one identical loop** (pure MSE, no energy
# head, no aux weighting) and localizes the failure by where the ladder breaks:
#
# | Arm | Input | Model | Tests |
# |---|---|---|---|
# | **A** | prev-4 true latents (oracle's input) | Linear | the training loop itself — must ≈ match ridge (~0.13) |
# | **B** | prev-4 true latents | MLP (2×512) | nonlinear capacity on oracle-grade input |
# | **C** | true latent sequence | **Miras backbone** (thin z→dim proj, bypassing `compress_input`) | the backbone/memory mechanics |
# | **D** | tokens → frozen AE embeddings → `compress_input` | Miras backbone (current pipeline) | the featurization path (expected flat) |
#
# Reading: A fails → loop bug. A✓ C✗ → backbone can't express even a ridge
# regression → architecture/optimization pathology in Miras/LTI. C✓ D✗ →
# `compress_input` over frozen embeddings is the bottleneck; feed the backbone
# AE latents (or a trained encoder) instead. Runtime: ~20–35 min on A100/T4.

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

import time

import equinox as eqx
import jax

jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np
import optax

from src.model.autoencoder import CALMAutoencoder
from src.model.miras_backbone import VELMBackbone

print("JAX devices:", jax.devices())

CFG = {
    "seed": 42,
    "chunk_k": 4,
    "seq_len": 64,
    "num_train_seqs": 8192,
    "num_eval_seqs": 128,
    "latent_dim": 128,
    "window": 4,            # oracle context width for arms A/B
    # shared training loop
    "steps": 3000,
    "batch_windows": 256,   # windows per step for A/B
    "batch_seqs": 8,        # sequences per step for C/D
    "lr": 1e-3,
    "grad_clip": 1.0,
    "eval_every": 100,
    # backbone (same as phase3)
    "dim": 256,
    "num_heads": 8,
    "miras_layers": 4,
    "swa_layers": 4,
    "ffn_intermediate": 512,
    "ae_hidden_dim": 384,
}
K, T, LAT, W = CFG["chunk_k"], CFG["seq_len"], CFG["latent_dim"], CFG["window"]
key = jax.random.PRNGKey(CFG["seed"])

# %% [markdown]
# ## 2 · AE, data, frozen-stat latents (identical to phase3)

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

key, ak = jax.random.split(key)
frozen_ae = eqx.tree_deserialise_leaves(
    ae_path,
    CALMAutoencoder(vocab_size=248077, chunk_size=K, hidden_dim=CFG["ae_hidden_dim"],
                    latent_dim=LAT, ffn_intermediate=768, key=ak),
)
AE_EMB = frozen_ae.embedding.weight

from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
tokens_needed = (CFG["num_train_seqs"] + CFG["num_eval_seqs"]) * T * K
stream, buf, total = load_dataset("roneneldan/TinyStories", split="train",
                                  streaming=True), [], 0
for ex in stream:
    text = ex.get("text", "")
    if len(text) < 50:
        continue
    ids = [i for i in tokenizer.encode(text, max_length=512, truncation=True)
           if i < 248077]
    buf.append(np.asarray(ids, dtype=np.int32))
    total += len(ids)
    if total >= tokens_needed:
        break
flat = np.concatenate(buf)[: (tokens_needed // (T * K)) * T * K]
seqs_np = flat.reshape(-1, T, K)
n_ev = CFG["num_eval_seqs"]
eval_seqs = jnp.asarray(seqs_np[:n_ev])                       # fixed, stream-order
rng = np.random.default_rng(CFG["seed"])
rest = seqs_np[n_ev:]
train_seqs = jnp.asarray(rest[rng.permutation(rest.shape[0])][: CFG["num_train_seqs"]])

encode_seq = eqx.filter_jit(jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0]))
def encode_all(seqs, bs=64):
    return jnp.concatenate(
        [jax.vmap(encode_seq)(seqs[i:i + bs]) for i in range(0, seqs.shape[0], bs)])

train_z, eval_z = encode_all(train_seqs), encode_all(eval_seqs)

STATS_FILE = os.path.join(CKPT_OUT, "latent_stats.npz")
if os.path.exists(STATS_FILE):
    _s = np.load(STATS_FILE)
    z_mu, z_sd = jnp.asarray(_s["mu"]), jnp.asarray(_s["sd"])
    print("✓ frozen stats loaded")
else:
    zf = train_z.reshape(-1, LAT)
    z_mu, z_sd = zf.mean(0), zf.std(0) + 1e-6
train_z = (train_z - z_mu) / z_sd
eval_z = (eval_z - z_mu) / z_sd
print(f"train {train_z.shape}, eval {eval_z.shape}")

# %% [markdown]
# ## 3 · Oracle bar (ridge, prev-4 latents)

# %%
Zo = np.asarray(train_z[:512]).reshape(-1, LAT)
Xo = np.concatenate([Zo[i:Zo.shape[0] - W + i] for i in range(W)], axis=1)
Yo = Zo[W:]
XtX = Xo.T @ Xo + 1e-1 * Xo.shape[0] * np.eye(Xo.shape[1], dtype=np.float32)
Wp = np.linalg.solve(XtX, Xo.T @ Yo)
Ze = np.asarray(eval_z).reshape(-1, LAT)
Xe = np.concatenate([Ze[i:Ze.shape[0] - W + i] for i in range(W)], axis=1)
pe, Ye = Xe @ Wp, Ze[W:]
oracle_ccos = float(np.mean(np.sum(pe * Ye, -1) /
                    (np.linalg.norm(pe, axis=-1) * np.linalg.norm(Ye, axis=-1) + 1e-8)))
print(f"oracle bar: {oracle_ccos:.4f}")

# %% [markdown]
# ## 4 · Four arms, one loop

# %%
def ccos(pred, tgt):
    return jnp.mean(jnp.sum(pred * tgt, -1) / (
        jnp.linalg.norm(pred, axis=-1) * jnp.linalg.norm(tgt, axis=-1) + 1e-8))


# ── window data for A/B: X (N, W*LAT), Y (N, LAT) — within-sequence windows ──
def make_windows(zs):
    xs = jnp.concatenate([zs[:, i: zs.shape[1] - W + i, :] for i in range(W)], axis=-1)
    ys = zs[:, W:, :]
    return xs.reshape(-1, W * LAT), ys.reshape(-1, LAT)

Xtr, Ytr = make_windows(train_z)
Xev, Yev = make_windows(eval_z)

key, ka, kb2, kc1, kc2, kc3, kd1, kd2 = jax.random.split(key, 8)

# A: linear
arm_A = eqx.nn.Linear(W * LAT, LAT, key=ka)

# B: MLP
arm_B = eqx.nn.MLP(W * LAT, LAT, width_size=512, depth=2, key=kb2)


def make_backbone(k):
    return VELMBackbone(
        dim=CFG["dim"], num_heads=CFG["num_heads"],
        num_miras_layers=CFG["miras_layers"], num_swa_layers=CFG["swa_layers"],
        ffn_intermediate=CFG["ffn_intermediate"], chunk_size=K,
        ae_hidden_dim=CFG["ae_hidden_dim"], key=k,
    )

# C: backbone fed TRUE latents via thin projection (bypasses compress_input)
arm_C = {"proj": eqx.nn.Linear(LAT, CFG["dim"], key=kc1),
         "bb": make_backbone(kc2),
         "head": eqx.nn.Linear(CFG["dim"], LAT, key=kc3)}

# D: current pipeline (tokens → frozen AE embeddings → compress_input)
arm_D = {"bb": make_backbone(kd1),
         "head": eqx.nn.Linear(CFG["dim"], LAT, key=kd2)}


def loss_A(model, xb, yb):
    return jnp.mean((jax.vmap(model)(xb) - yb) ** 2)


loss_B = loss_A


def loss_C(model, seq_z_batch, _unused):
    def per_seq(zs):
        inp = jax.vmap(model["proj"])(zs[:-1])           # (T-1, dim)
        hid, _ = model["bb"](inp)
        pred = jax.vmap(model["head"])(hid)
        return jnp.mean((pred - zs[1:]) ** 2)
    return jnp.mean(jax.vmap(per_seq)(seq_z_batch))


def loss_D(model, seq_tok_batch, seq_z_batch):
    def per_seq(toks, zs):
        inp = jax.vmap(model["bb"].compress_input)(AE_EMB[toks])
        hid, _ = model["bb"](inp)
        pred = jax.vmap(model["head"])(hid[:-1])
        return jnp.mean((pred - zs[1:]) ** 2)
    return jnp.mean(jax.vmap(per_seq)(seq_tok_batch, seq_z_batch))


def eval_arm(name, model):
    if name in ("A", "B"):
        return float(ccos(jax.vmap(model)(Xev), Yev))
    if name == "C":
        def per_seq(zs):
            inp = jax.vmap(model["proj"])(zs[:-1])
            hid, _ = model["bb"](inp)
            return jax.vmap(model["head"])(hid)
        preds = jax.vmap(per_seq)(eval_z)
        return float(ccos(preds.reshape(-1, LAT), eval_z[:, 1:].reshape(-1, LAT)))
    def per_seq(toks, zs):
        inp = jax.vmap(model["bb"].compress_input)(AE_EMB[toks])
        hid, _ = model["bb"](inp)
        return jax.vmap(model["head"])(hid[:-1])
    preds = jax.vmap(per_seq)(eval_seqs, eval_z)
    return float(ccos(preds.reshape(-1, LAT), eval_z[:, 1:].reshape(-1, LAT)))


def train_arm(name, model, loss_fn):
    opt = optax.chain(optax.clip_by_global_norm(CFG["grad_clip"]),
                      optax.adamw(CFG["lr"], weight_decay=0.0))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step(p, o, xb, yb):
        def _l(pp, xb, yb):
            return loss_fn(eqx.combine(pp, static), xb, yb)
        l, g = eqx.filter_value_and_grad(_l)(p, xb, yb)
        up, o = opt.update(g, o, p)
        return optax.apply_updates(p, up), o, l

    hist, dk = [], jax.random.PRNGKey(CFG["seed"] + 7)
    t0 = time.time()
    for i in range(CFG["steps"]):
        dk, bk = jax.random.split(dk)
        if name in ("A", "B"):
            idx = jax.random.randint(bk, (CFG["batch_windows"],), 0, Xtr.shape[0])
            xb, yb = Xtr[idx], Ytr[idx]
        else:
            idx = jax.random.randint(bk, (CFG["batch_seqs"],), 0, train_seqs.shape[0])
            xb = train_z[idx] if name == "C" else train_seqs[idx]
            yb = train_z[idx]
        params, opt_state, l = step(params, opt_state, xb, yb)
        if i % CFG["eval_every"] == 0 or i == CFG["steps"] - 1:
            c = eval_arm(name, eqx.combine(params, static))
            hist.append((i, c))
            if i % 500 == 0 or i == CFG["steps"] - 1:
                print(f"  [{name} {i:5d}] loss {float(l):.4f}  eval ccos {c:.4f}")
    print(f"  arm {name}: final ccos {hist[-1][1]:.4f}  ({time.time() - t0:.0f}s)")
    return hist


print("Arm A — linear on prev-4 latents (loop sanity; must ≈ oracle):")
hist_A = train_arm("A", arm_A, loss_A)
print("\nArm B — MLP on prev-4 latents:")
hist_B = train_arm("B", arm_B, loss_B)
print("\nArm C — Miras backbone on TRUE latents (bypasses compress_input):")
hist_C = train_arm("C", arm_C, loss_C)
print("\nArm D — current token→embedding→compress_input pipeline:")
hist_D = train_arm("D", arm_D, loss_D)

# %% [markdown]
# ## 5 · Results + verdict

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))
for h, lbl in [(hist_A, "A linear (latents)"), (hist_B, "B MLP (latents)"),
               (hist_C, "C backbone (latents)"), (hist_D, "D backbone (tokens)")]:
    plt.plot(*zip(*h), lw=2, label=lbl)
plt.axhline(oracle_ccos, color="red", ls="--", label=f"ridge oracle ({oracle_ccos:.3f})")
plt.axhline(0, color="black", lw=0.5)
plt.xlabel("step"); plt.ylabel("eval centered cosine")
plt.title("Where does the signal die?"); plt.legend(); plt.show()

fA, fB, fC, fD = (h[-1][1] for h in (hist_A, hist_B, hist_C, hist_D))
print("=" * 60)
print(f"  oracle : {oracle_ccos:.4f}")
print(f"  A linear/latents   : {fA:.4f}")
print(f"  B MLP/latents      : {fB:.4f}")
print(f"  C backbone/latents : {fC:.4f}")
print(f"  D backbone/tokens  : {fD:.4f}")
print("=" * 60)

ok = lambda v: v >= 0.75 * oracle_ccos
if not ok(fA):
    print("❌ ARM A FAILED — the training loop/eval itself is buggy: a linear "
          "model with the oracle's exact inputs must match ridge. Fix the "
          "loop before believing anything else.")
elif not ok(fC):
    print("❌ ARCHITECTURE — loop is sound (A ≈ oracle) but the Miras "
          "backbone can't express what a ridge regression does even on "
          "perfect inputs. Suspects: LTI injection scaling, Miras block "
          "read/write dynamics, gradient attenuation through the recurrence. "
          "Debug the backbone before touching data or features.")
elif not ok(fD):
    print("✅ FEATURIZATION CONVICTED — the backbone learns fine on true "
          "latents (C ≈ oracle) but starves on token→frozen-embedding→"
          "compress_input inputs. Fix: feed the backbone the frozen AE's own "
          "*encoded latents* of past chunks (teacher-forced z_≤t) instead of "
          "raw embedding compression — i.e., autoregress in latent space "
          "exactly as CALM does at inference.")
else:
    print("✅ ALL ARMS PASS — with this loop even the token pipeline works; "
          "the phase3 notebook's loss composition (energy + aux) or its "
          "hyperparameters were the difference. Diff against this setup.")
