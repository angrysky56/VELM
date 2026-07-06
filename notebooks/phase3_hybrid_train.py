# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
# **Run 4 revision (standardized latent space).** Runs 1–3 trained on RAW
# latents and extracted zero contextual signal (centered cos 0.005), while the
# AE-diagnostic exonerated the latent space: smooth (margin 0.59) and linearly
# predictable (oracle centered-cos 0.125 — 25× the trained model). Root cause:
# raw latents carry a dominant shared mean direction that soaks up nearly all
# cosine/energy gradient; the model converges to the marginal and stops.
# Fix: train **entirely in standardized latent space** ((z−μ)/σ, de-standardize
# before AE decode), aux loss = MSE (POC-v4-proven), unconditional baseline =
# marginal *sampler*, and the **linear-oracle bar** computed in-notebook.
#
# Success criteria:
#
# - centered cosine ≥ the linear-oracle bar (~0.125) — the backbone sees
#   strictly more than the oracle, so matching it is the minimum
# - decoded token accuracy > unconditional
# - prediction diversity ≥ 0.2 (not collapsed)
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

# CRITICAL on A100/H100: JAX defaults matmuls to TF32 (10-bit mantissa).
# The contextual gradient here is ~1% of the loss — near/below TF32 rounding.
# T4s (where run 4 learned) do true FP32; force it everywhere.
jax.config.update("jax_default_matmul_precision", "highest")

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
    # ── RUN: v3l2 — the "thinking" ablation ───────────────
    # v3 result: scaling curve bent (0.167→0.218→0.306→0.314): the model is
    # now capacity-limited, not data-limited. This run tests RECURRENT DEPTH
    # (Mythos RDT n_loops=2): same parameters, twice the sequential compute
    # per chunk. v3 weights load directly into the 2-loop skeleton
    # (init_from_version) and finetune for one session. If ccos moves,
    # latent-space iteration pays at fixed params; if not, width (dim 384)
    # is the next lever.
    "run_version": "v3l2",         # versions state/ckpt files
    "n_loops": 2,                  # recurrent depth (v1–v3 ran 1)
    "init_from_version": "v3",     # seed stage-1 params from this run's state
    "data_version": "v3",          # reuse v3's token/latent caches (same data)
    "seq_len": 64,                 # chunks per sequence (256 tokens)
    "num_train_seqs": 262144,      # ~64M tokens (4× v2, 16× v1)
    "num_eval_seqs": 128,
    "wiki_fraction": 0.25,         # train-mix fraction from wikitext-103
    #   (eval stays PURE TinyStories first-128 stream order, so baselines,
    #    oracle bar, and centered-cos stay comparable to v1/v2 runs)
    "batch_size": 8,
    "cache_data": True,            # cache tokens+latents to Drive (~9GB);
    #   falls back to recompute if the save fails (Drive quota)
    # v3 NOTE: train tokens+latents now live in HOST RAM (≈9GB numpy) and
    # batches are moved to device per step — the arrays no longer fit
    # comfortably GPU-resident. Requires a High-RAM runtime.
    # ── trainable model ───────────────────────────────────
    "dim": 256,
    "num_heads": 8,
    "miras_layers": 4,
    "swa_layers": 4,
    "ffn_intermediate": 512,
    "head_blocks": 2,
    "head_ffn": 512,
    "energy_samples": 8,           # N for the MC energy-score estimator
    "aux_weight": 1.0,             # direct-prediction loss weight (stage 2)
    # ── STAGING (input-ablation finding, 2026-07-05) ──────
    # Arm D proved the full token→compress_input→Miras pipeline reaches the
    # oracle bar in 3k steps under pure MSE @ lr 1e-3 / wd 0 — the energy
    # score's stochastic gradient was drowning the contextual signal all
    # along. Stage 1 CONFIRMED 2026-07-05: centered-cos 0.167 > oracle 0.130.
    # Stages:
    #   "mse"           — stage 1: representation learning, direct head only
    #   "energy_frozen" — stage 2a: train ONLY the energy head; gradients are
    #                     stopped at the backbone hidden states, so stage-1
    #                     representations cannot be damaged
    #   "mse_ss"        — stage 1c: scheduled sampling — exposure-bias fix.
    #                     Two passes: teacher-forced predictions are decoded
    #                     to tokens and swapped into a fraction of input
    #                     positions; loss is MSE vs TRUE targets on the
    #                     corrupted context. Trains inference-time robustness.
    #   "energy+aux"    — stage 2b (optional): joint finetune at low LR
    "objective": "mse",
    # stage-1c scheduled sampling
    "ss_fraction": 0.25,           # prob a position's input is model-generated
    "ss_steps": 8000,
    "ss_lr": 3e-4,
    # stage-2a optimization (fresh head → its own schedule + state file;
    # stage-1 params are loaded but its optimizer state is NOT — safe because
    # stop_gradient freezes the backbone, so only virgin head params train)
    "stage2_steps": 8000,
    "stage2_lr": 3e-4,
    # ── optimization (multi-session) ──────────────────────
    # The LR schedule spans total_steps GLOBALLY; each Colab session runs
    # session_steps and saves the FULL training state (params + optimizer
    # moments + global step), so resuming continues the schedule instead of
    # re-warming up over trained weights (run 5's fatal mistake).
    # v3l2: finetune cycle — one session, warm-started from v3
    "total_steps": 8000,
    "session_steps": 8000,
    "peak_lr": 3e-4,               # finetune LR (1e-3 for from-scratch runs)
    "warmup_steps": 200,
    "weight_decay": 0.0,           # arm D ran wd 0; decay added nothing
    "grad_clip": 1.0,
    "eval_every": 100,
    "ckpt_every": 1000,
}
K, T = CFG["chunk_k"], CFG["seq_len"]
RUNV = CFG["run_version"]
DATAV = CFG.get("data_version", RUNV)
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

TOK_CACHE = os.path.join(CKPT_OUT, f"poc_tokens_{DATAV}.npz")
if os.path.exists(TOK_CACHE):
    _t = np.load(TOK_CACHE)
    train_seqs, eval_seqs = _t["train"], jnp.asarray(_t["evals"])
    print(f"✓ loaded cached tokens: train {train_seqs.shape} (host RAM)")


def stream_seqs(name, config, text_field, n_seqs, min_len=50):
    """Stream a HF dataset into (n_seqs, T, K) int32 chunk sequences."""
    kwargs = {"split": "train", "streaming": True}
    if config:
        kwargs["name"] = config
    ds, buf, total = load_dataset(name, **kwargs), [], 0
    need = n_seqs * T * K
    for ex in ds:
        text = ex.get(text_field, "")
        if not text or len(text) < min_len:
            continue
        ids = [i for i in tokenizer.encode(text, max_length=512, truncation=True)
               if i < CFG["vocab_size"]]
        buf.append(np.asarray(ids, dtype=np.int32))
        total += len(ids)
        if total >= need:
            break
    flat = np.concatenate(buf)[: (need // (T * K)) * T * K]
    return flat.reshape(-1, T, K)


if not os.path.exists(TOK_CACHE):
    n_ev = CFG["num_eval_seqs"]
    n_wiki = int(CFG["num_train_seqs"] * CFG["wiki_fraction"])
    n_ts = CFG["num_train_seqs"] - n_wiki + n_ev

    ts_seqs = stream_seqs("roneneldan/TinyStories", None, "text", n_ts)
    # FIXED eval set: the first num_eval_seqs TinyStories sequences in stream
    # order, BEFORE any permutation or mixing — identical across sessions,
    # data scales, and mix ratios, so baselines/oracle bars stay comparable.
    eval_seqs = jnp.asarray(ts_seqs[: n_ev])

    train_pool = ts_seqs[n_ev:]
    if n_wiki > 0:
        wiki_seqs = stream_seqs("Salesforce/wikitext", "wikitext-103-raw-v1",
                                "text", n_wiki)
        train_pool = np.concatenate([train_pool, wiki_seqs], axis=0)
        print(f"mix: {train_pool.shape[0] - wiki_seqs.shape[0]:,} TinyStories + "
              f"{wiki_seqs.shape[0]:,} wikitext sequences")

    rng = np.random.default_rng(CFG["seed"])
    # v3: train tokens stay in HOST RAM (numpy); batches device-put per step
    train_seqs = np.ascontiguousarray(
        train_pool[rng.permutation(train_pool.shape[0])][: CFG["num_train_seqs"]])
    if CFG["cache_data"]:
        try:
            np.savez(TOK_CACHE, train=train_seqs, evals=np.asarray(eval_seqs))
            print(f"✓ cached tokens → {TOK_CACHE}")
        except Exception as e:
            print(f"⚠️  token cache save failed ({type(e).__name__}) — continuing")

print(f"train {train_seqs.shape} (~{train_seqs.size:,} tokens, host RAM), "
      f"eval {eval_seqs.shape} (fixed, pure TinyStories)")

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
sample = jnp.asarray(train_seqs[:16].reshape(-1, K))
recon = jax.vmap(frozen_ae.reconstruct)(sample)
acc = float(jnp.mean(recon == sample))
print(f"AE roundtrip accuracy on {sample.shape[0]} chunks: {acc:.4f}")
assert acc >= 0.95, "AE checkpoint/config mismatch — check ae dims in CFG"


def encode_all_np(seqs_np, bs=256):
    """Encode to HOST-RAM numpy in device-sized bites (v3: ~9GB output)."""
    outs = []
    for i in range(0, seqs_np.shape[0], bs):
        z = jax.vmap(encode_seq)(jnp.asarray(seqs_np[i: i + bs]))
        outs.append(np.asarray(z))
        if (i // bs) % 200 == 0:
            print(f"  encoding… {i:,}/{seqs_np.shape[0]:,}", end="\r")
    return np.concatenate(outs, axis=0)


LAT_CACHE = os.path.join(CKPT_OUT, f"poc_latents_{DATAV}.npy")
t0 = time.time()
if os.path.exists(LAT_CACHE):
    train_z = np.load(LAT_CACHE)  # raw latents, host RAM
    print(f"✓ loaded cached latents {train_z.shape}")
else:
    train_z = encode_all_np(np.asarray(train_seqs))   # (N, T, latent) host
    if CFG["cache_data"]:
        try:
            np.save(LAT_CACHE, train_z)
            print(f"✓ cached latents → {LAT_CACHE}")
        except Exception as e:
            print(f"⚠️  latent cache save failed ({type(e).__name__}) — continuing")
eval_z = jax.vmap(encode_seq)(eval_seqs)              # small, device-resident
print(f"latents: train {train_z.shape} (host), eval {eval_z.shape}  "
      f"({time.time() - t0:.0f}s)")

# ── STANDARDIZE the latent space (run-4 fix) ──────────────────────────
# Raw latents carry a dominant shared mean direction; cosine/energy losses
# spend nearly all gradient reproducing it, drowning contextual signal
# (runs 1–3 pinned at the unconditional cosine; the AE-diagnostic oracle
# found 25× more structure than the trained model). In standardized space
# the mean is gone: every unit of cosine is contextual, and per-dim unit
# variance gives ||z|| ≈ √latent_dim — matching the energy head's output
# sphere. De-standardize with z_mu/z_sd before AE-decoding.
# standardization stats are FROZEN on first computation and reloaded ever
# after — recomputing them per session drifts the target space under a
# resumed model.
STATS_FILE = os.path.join(CKPT_OUT, "latent_stats.npz")
if os.path.exists(STATS_FILE):
    _s = np.load(STATS_FILE)
    mu_np, sd_np = _s["mu"], _s["sd"]
    print("✓ loaded frozen latent stats")
else:
    z_flat = train_z.reshape(-1, CFG["latent_dim"])
    mu_np = z_flat.mean(axis=0)
    sd_np = z_flat.std(axis=0) + 1e-6
    np.savez(STATS_FILE, mu=mu_np, sd=sd_np)
    print("✓ computed + froze latent stats")
z_mu, z_sd = jnp.asarray(mu_np), jnp.asarray(sd_np)
# standardize IN PLACE on host (v3: never materialize a 9GB copy on device)
train_z -= mu_np
train_z /= sd_np
eval_z = (eval_z - z_mu) / z_sd
print(f"standardized: mean|z| = {float(jnp.mean(jnp.linalg.norm(eval_z, axis=-1))):.2f} "
      f"(sphere radius √{CFG['latent_dim']} = {np.sqrt(CFG['latent_dim']):.2f})")

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


def _cos(a, b):
    return jnp.sum(a * b, axis=-1) / (
        jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8)


z_mean = jnp.asarray(
    train_z[:4096].reshape(-1, CFG["latent_dim"]).mean(axis=0))  # ≈ 0 (standardized)

# copy: predict z_t for z_{t+1}
copy_scores = jax.vmap(
    lambda zs: jnp.mean(jax.vmap(degenerate_energy)(zs[:-1], zs[1:]))
)(eval_z)
baseline_copy = float(jnp.mean(copy_scores))
copy_cos_base = float(jnp.mean(_cos(eval_z[:, :-1], eval_z[:, 1:])))

# unconditional SAMPLER: N random latents from the marginal pool per position.
# In standardized space the mean predictor is degenerate (zero vector), so a
# calibrated marginal sampler is the honest unconditional bar for the energy
# score. Its cosine is ≈ 0 by construction.
pool = eval_z.reshape(-1, CFG["latent_dim"])
bkey = jax.random.PRNGKey(123)


def uncond_energy(tgt, k):
    idx = jax.random.randint(k, (CFG["energy_samples"],), 0, pool.shape[0])
    return energy_score(pool[idx], tgt)


tgts_flat = eval_z[:, 1:].reshape(-1, CFG["latent_dim"])
ukeys = jax.random.split(bkey, tgts_flat.shape[0])
baseline_uncond = float(jnp.mean(jax.vmap(uncond_energy)(tgts_flat, ukeys)))
uncond_cos_base = 0.0  # by construction in standardized space
print(f"copy baseline energy:    {baseline_copy:.4f}   cos {copy_cos_base:.3f}")
print(f"uncond-sampler baseline: {baseline_uncond:.4f}   cos ≈ 0")

# %% [markdown]
# ## 6b · Oracle bar: linear probe from previous latents
#
# The AE-diagnostic showed a ridge probe from the previous W latents reaches
# centered-cos ≈ 0.125. The backbone sees strictly more (raw tokens), so this
# is the **minimum bar**: a healthy run must at least match the oracle.

# %%
W_ORACLE, LAM = 4, 1e-1
Zo = np.asarray(train_z[:512]).reshape(-1, CFG["latent_dim"])
Xo = np.concatenate([Zo[i:Zo.shape[0] - W_ORACLE + i] for i in range(W_ORACLE)], axis=1)
Yo = Zo[W_ORACLE:]
XtX = Xo.T @ Xo + LAM * Xo.shape[0] * np.eye(Xo.shape[1], dtype=np.float32)
Wp = np.linalg.solve(XtX, Xo.T @ Yo)

Ze = np.asarray(eval_z).reshape(-1, CFG["latent_dim"])
Xe = np.concatenate([Ze[i:Ze.shape[0] - W_ORACLE + i] for i in range(W_ORACLE)], axis=1)
Ye = Ze[W_ORACLE:]
pe = Xe @ Wp
oracle_ccos = float(np.mean(
    np.sum(pe * Ye, -1) /
    (np.linalg.norm(pe, axis=-1) * np.linalg.norm(Ye, axis=-1) + 1e-8)))
print(f"oracle (linear, prev-{W_ORACLE} latents) cos: {oracle_ccos:.4f}  ← minimum bar")

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
    n_loops=CFG["n_loops"],   # recurrent depth — static, so weights from an
    #                           n_loops=1 run load directly into this skeleton
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

# NOTE: weights-only resume is gone. Run 5 proved it destructive: loading
# trained weights into a FRESH optimizer + re-warmed LR schedule bulldozed
# run 4's contextual features (worth ~1% of the loss, first thing destroyed,
# last thing relearned). Resume now restores the FULL training state —
# params + Adam moments + global step — in section 8. Old per-component
# best-checkpoints remain for inference only.
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

    # stage 1 ("mse"): pure direct-head MSE — the arm-D-proven objective.
    # The energy head is untouched (zero grads) until stage 2.
    d_pred = jax.vmap(dr)(hid_in)
    aux_loss = jnp.mean((d_pred - z_tgt) ** 2)
    if CFG["objective"] == "mse":
        return aux_loss
    if CFG["objective"] == "mse_ss":
        # stage 1c: scheduled sampling. Decode pass-1 predictions to tokens,
        # swap them into a random fraction of input positions, then require
        # the TRUE targets from the corrupted context (pass 2). Gradients
        # flow only through pass 2 (argmax + stop_gradient sever pass 1).
        pred_tok = jnp.argmax(
            jax.vmap(frozen_ae.decode)(
                jax.lax.stop_gradient(d_pred) * z_sd + z_mu),
            axis=-1)                                        # (T-1, K)
        lk1, lk2 = jax.random.split(loss_key)
        swap = jax.random.bernoulli(
            lk1, CFG["ss_fraction"], (pred_tok.shape[0],))  # positions 1..T-1
        mixed = seq_tokens.at[1:].set(
            jnp.where(swap[:, None], pred_tok, seq_tokens[1:]))
        embs2 = AE_EMB[mixed]
        inp2 = jax.vmap(bb.compress_input)(embs2)
        hid2, _ = bb(inp2)
        d2 = jax.vmap(dr)(hid2[:-1])
        return jnp.mean((d2 - z_tgt) ** 2)
    if CFG["objective"] == "energy_frozen":
        # stage 2a: energy head only — backbone representations are frozen
        # via stop_gradient so the energy score's noise can't erode them
        hid_in = jax.lax.stop_gradient(hid_in)
        keys = jax.random.split(loss_key, hid_in.shape[0])
        return jnp.mean(jax.vmap(pos_loss)(hid_in, z_tgt, keys))
    e_loss = jnp.mean(jax.vmap(pos_loss)(hid_in, z_tgt, keys))
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
        # centered cosine: subtract the global mean latent first. All latents
        # share a dominant direction, so plain cosine is ~blind to context;
        # in centered space the unconditional predictor scores exactly 0.
        pc, tc = d_pred - z_mean, z_tgt - z_mean
        c_cos = jnp.sum(pc * tc, axis=-1) / (
            jnp.linalg.norm(pc, axis=-1) * jnp.linalg.norm(tc, axis=-1) + 1e-8
        )
        # diversity: how much predictions vary across positions vs targets
        div = jnp.mean(jnp.std(preds, axis=0)) / (jnp.mean(jnp.std(z_tgt, axis=0)) + 1e-8)
        return jnp.mean(es), jnp.mean(cos), jnp.mean(d_cos), jnp.mean(c_cos), div

    keys = jax.random.split(mkey, seqs.shape[0])
    es, cos, dcos, ccos, div = jax.vmap(per_seq)(seqs, zs, keys)
    return jnp.mean(es), jnp.mean(cos), jnp.mean(dcos), jnp.mean(ccos), jnp.mean(div)


# %% [markdown]
# ## 8 · Train (AdamW, warmup + cosine)

# %%
# LR schedule spans the stage's step budget GLOBALLY. Adam's step count lives
# inside opt_state, so restoring opt_state continues the schedule — no
# re-warmup over trained weights. Each stage has its OWN schedule + state
# file: a fresh energy head must not inherit stage 1's decayed-to-floor LR.
import json

RUNV = CFG["run_version"]
# per-stage budgets, peaks, and file suffixes; every non-"mse" stage seeds
# its params from the stage-1 state file when its own state doesn't exist
_STAGES = {
    "mse":           (CFG["total_steps"], CFG["peak_lr"], ""),
    "mse_ss":        (CFG["ss_steps"], CFG["ss_lr"], "_ss"),
    "energy_frozen": (CFG["stage2_steps"], CFG["stage2_lr"], "_s2"),
    "energy+aux":    (CFG["stage2_steps"], CFG["stage2_lr"], "_s2b"),
}
stage_total, stage_peak, _suffix = _STAGES[CFG["objective"]]
STAGE2 = CFG["objective"] != "mse"
stage_tag = f"_{RUNV}{_suffix}"

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=stage_peak,
    warmup_steps=CFG["warmup_steps"],
    decay_steps=stage_total,
    end_value=stage_peak * 0.05,
)
opt = optax.chain(
    optax.clip_by_global_norm(CFG["grad_clip"]),
    optax.adamw(schedule, weight_decay=CFG["weight_decay"]),
)
opt_state = opt.init(params)

# ── full training-state resume ────────────────────────────────────────
STATE_FILE = os.path.join(CKPT_OUT, f"train_state{stage_tag}.eqx")
STATE_META = os.path.join(CKPT_OUT, f"train_state{stage_tag}.json")
S1_STATE = os.path.join(CKPT_OUT, f"train_state_{RUNV}.eqx")
global_step = 0
if os.path.exists(STATE_FILE) and os.path.exists(STATE_META):
    try:
        restored = eqx.tree_deserialise_leaves(
            STATE_FILE, {"params": params, "opt_state": opt_state})
        params, opt_state = restored["params"], restored["opt_state"]
        with open(STATE_META) as f:
            global_step = json.load(f)["global_step"]
        print(f"✓ resumed stage state at step {global_step:,} ({STATE_FILE})")
    except Exception as e:
        print(f"⚠️  train_state incompatible ({type(e).__name__}) — fresh start")
elif STAGE2 and os.path.exists(S1_STATE):
    # first stage-2 session: load stage-1 PARAMS (backbone/direct/head),
    # keep the fresh optimizer + schedule for the head-only training
    restored = eqx.tree_deserialise_leaves(
        S1_STATE, {"params": params, "opt_state": opt_state})
    params = restored["params"]
    print(f"✓ {CFG['objective']} start: seeded from stage-1 params, "
          "fresh optimizer + LR cycle")
elif not STAGE2 and CFG.get("init_from_version"):
    # warm-start a NEW stage-1 run (e.g., the n_loops ablation) from another
    # run's trained params; fresh optimizer + LR cycle
    src_state = os.path.join(
        CKPT_OUT, f"train_state_{CFG['init_from_version']}.eqx")
    if os.path.exists(src_state):
        restored = eqx.tree_deserialise_leaves(
            src_state, {"params": params, "opt_state": opt_state})
        params = restored["params"]
        print(f"✓ warm-started params from {CFG['init_from_version']} "
              f"(n_loops now {CFG['n_loops']}), fresh optimizer")
    else:
        print(f"⚠️  init_from state {src_state} not found — fresh start")
else:
    print("no train_state found — fresh start (global step 0)")


@eqx.filter_jit
def train_step(p, o, batch, bz, skey):
    loss, grads = eqx.filter_value_and_grad(batch_energy_loss)(p, batch, bz, skey)
    updates, o = opt.update(grads, o, p)
    return optax.apply_updates(p, updates), o, loss


def save_ckpt(p, tag):
    m = eqx.combine(p, static)
    for comp, fname in [("backbone", "backbone_hybrid"),
                        ("head", "energy_head_hybrid"),
                        ("direct", "direct_head_hybrid")]:
        eqx.tree_serialise_leaves(
            os.path.join(CKPT_OUT, f"{fname}_{RUNV}_{tag}.eqx"), m[comp])


def save_state(p, o, gs):
    eqx.tree_serialise_leaves(STATE_FILE, {"params": p, "opt_state": o})
    with open(STATE_META, "w") as f:
        json.dump({"global_step": gs}, f)


best_loss, best_ccos, hist = float("inf"), -float("inf"), []
# batch stream keyed by global step → different data order each session
dk = jax.random.fold_in(jax.random.PRNGKey(CFG["seed"] + 100), global_step)
t0 = time.time()
n_sess = min(CFG["session_steps"], stage_total - global_step)
if n_sess <= 0:
    print(f"stage budget exhausted ({global_step:,}/{stage_total:,}) — "
          "flip CFG['objective'] for the next stage or raise the budget.")
for i in range(n_sess):
    gs = global_step + i
    dk, bk, sk = jax.random.split(dk, 3)
    idx = np.asarray(
        jax.random.randint(bk, (CFG["batch_size"],), 0, train_seqs.shape[0]))
    params, opt_state, loss = train_step(
        params, opt_state, jnp.asarray(train_seqs[idx]), jnp.asarray(train_z[idx]), sk)
    if i % CFG["eval_every"] == 0 or i == n_sess - 1:
        dk, mk = jax.random.split(dk)
        es, cos, dcos, ccos, div = eval_metrics(params, eval_seqs, eval_z, mk)
        es, cos, dcos, ccos, div = (float(es), float(cos), float(dcos),
                                    float(ccos), float(div))
        hist.append((gs, time.time() - t0, es, cos, dcos, div, ccos))
        best_loss = min(best_loss, es)
        marker = ""
        # checkpoint criterion follows the stage: representation stages track
        # centered cos; energy stages track eval energy (ccos is frozen in 2a)
        if CFG["objective"] in ("mse", "mse_ss"):
            if ccos > best_ccos:
                best_ccos = ccos
                save_ckpt(params, "best")
                marker = "  ← best ccos (saved)"
        else:
            if es <= best_loss:
                save_ckpt(params, "best")
                marker = "  ← best energy (saved)"
        print(f"[gs {gs:6d}] train {float(loss):.4f}  eval {es:.4f}  "
              f"cos {cos:.3f}/direct {dcos:.3f}  centered {ccos:.3f}  "
              f"div {div:.2f}{marker}")
    if i and i % CFG["ckpt_every"] == 0:
        save_state(params, opt_state, gs + 1)
global_step += n_sess
save_state(params, opt_state, global_step)
save_ckpt(params, "final")
print(f"\nsession done in {time.time() - t0:.0f}s — global step {global_step:,}"
      f"/{stage_total:,}, best eval energy this session {best_loss:.4f}")
print("Re-run this notebook (fresh Colab session is fine) to continue training.")

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
axes[1].axhline(oracle_ccos, color="red", ls="--",
                label=f"linear-oracle bar ({oracle_ccos:.3f})")
axes[1].axhline(copy_cos_base, color="gray", ls=":", label="copy cos")
axes[1].set(xlabel="step", ylabel="cosine similarity", title="Latent prediction quality")
axes[1].legend()
ax2b = axes[2]
ax2b.plot(steps_h, [h[5] for h in hist], lw=2, color="tab:green", label="diversity")
ax2b.plot(steps_h, [h[6] for h in hist], lw=2, color="tab:purple",
          label="centered cos (0 = no context)")
ax2b.axhline(1.0, color="gray", ls="--", label="target diversity")
ax2b.axhline(0.2, color="red", ls=":", label="collapse threshold")
ax2b.axhline(0.0, color="black", lw=0.5)
ax2b.set(xlabel="step", title="Context signal: diversity + centered cosine")
ax2b.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 10 · Decoded token accuracy
#
# Direct-head predictions decoded through the AE, token-level top-1 accuracy
# vs the true next chunk. Random ≈ 0; the marginal-mean predictor lands on
# frequent-token soup. Any stable value above the uncond row is real
# contextual signal in *token* space — a much stricter test than cosine.

# %%
m = eqx.combine(params, static)
bb, hd, dr = m["backbone"], m["head"], m["direct"]

N_ACC = 32  # eval sequences to score (decode is vocab-wide, keep it bounded)

@eqx.filter_jit
def decoded_token_acc(seq_tokens, seq_z):
    embs = AE_EMB[seq_tokens]
    inp = jax.vmap(bb.compress_input)(embs)
    hid, _ = bb(inp)
    z_pred = jax.vmap(dr)(hid[:-1]) * z_sd + z_mu         # de-standardize
    logits = jax.vmap(frozen_ae.decode)(z_pred)           # (T-1, K, vocab)
    pred_tok = jnp.argmax(logits, axis=-1)
    return jnp.mean(pred_tok == seq_tokens[1:])

acc_model = float(jnp.mean(jax.vmap(decoded_token_acc)(eval_seqs[:N_ACC], eval_z[:N_ACC])))
uncond_tok = jnp.argmax(frozen_ae.decode(z_mu), axis=-1)  # raw-space mean latent
acc_uncond = float(jnp.mean(uncond_tok[None, None, :] == eval_seqs[:N_ACC, 1:]))
print(f"decoded token accuracy — model: {acc_model:.4f}   uncond-mean: {acc_uncond:.4f}")

# %% [markdown]
# ## 11 · Decode demo: raw + nearest-neighbor snap
#
# Raw AE decode of an off-manifold prediction is expected to be token soup at
# this compute scale — CALM-quality text sits ~5 orders of magnitude away in
# data/params. The **NN-snap** row projects the prediction onto the manifold
# (nearest training latent by cosine) and decodes *that* chunk's actual text:
# it shows what the prediction points at semantically, even when the raw
# decode is unreadable. Watch NN-snap for topical relevance first.

# %%
# latent bank for NN-snap (subsample of train chunks)
BANK_N = 40000
bank_z = jnp.asarray(train_z[: BANK_N // T + 1].reshape(-1, CFG["latent_dim"])[:BANK_N])
bank_tokens = jnp.asarray(train_seqs[: BANK_N // T + 1].reshape(-1, K)[:BANK_N])
bank_norm = bank_z / (jnp.linalg.norm(bank_z, axis=-1, keepdims=True) + 1e-8)

def nn_snap(z):
    zn = z / (jnp.linalg.norm(z) + 1e-8)
    idx = int(jnp.argmax(bank_norm @ zn))
    return tokenizer.decode(np.asarray(bank_tokens[idx]))


demo_seq = eval_seqs[0]
embs = AE_EMB[demo_seq]
inp = jax.vmap(bb.compress_input)(embs)
hid, _ = bb(inp)

demo_key = jax.random.PRNGKey(0)
for t in [8, 24, 48]:
    dk1, demo_key = jax.random.split(demo_key)
    z_pred = hd.predict(hid[t], key=dk1)
    z_direct = dr(hid[t])
    # de-standardize before AE decode; NN-snap stays in standardized space
    pred_tokens = jnp.argmax(frozen_ae.decode(z_pred * z_sd + z_mu), axis=-1)
    direct_tokens = jnp.argmax(frozen_ae.decode(z_direct * z_sd + z_mu), axis=-1)
    ctx = tokenizer.decode(np.asarray(demo_seq[max(0, t - 3): t + 1]).reshape(-1))
    truth = tokenizer.decode(np.asarray(demo_seq[t + 1]))
    print(f"── position {t} ────────────────────────────")
    print(f"  context …{ctx!r}")
    print(f"  truth        {truth!r}")
    print(f"  energy raw   {tokenizer.decode(np.asarray(pred_tokens))!r}")
    print(f"  direct raw   {tokenizer.decode(np.asarray(direct_tokens))!r}")
    print(f"  energy NN⇒   {nn_snap(z_pred)!r}")
    print(f"  direct NN⇒   {nn_snap(z_direct)!r}\n")
# If rows repeat verbatim across positions, the model is ignoring context —
# check the diversity/centered-cosine plot above.

# %% [markdown]
# ## 12 · Verdict
#
# Judged on **context signal**, not text quality — readable decode at this
# compute scale is not a fair criterion (CALM operates ~5 orders of magnitude
# higher). Centered cosine and decoded token accuracy are the honest yardsticks:
# both are exactly-zero-ish for any unconditional predictor.

# %%
final_cos, final_dcos, final_div = hist[-1][3], hist[-1][4], hist[-1][5]
final_ccos = hist[-1][6]
best_ccos = max(h[6] for h in hist)
print("=" * 64)
print(f"  best eval energy   : {best_loss:.4f}")
print(f"  copy baseline      : {baseline_copy:.4f}   cos {copy_cos_base:.3f}")
print(f"  uncond baseline    : {baseline_uncond:.4f}   cos {uncond_cos_base:.3f}")
print(f"  final cos          : energy-head {final_cos:.3f} / direct {final_dcos:.3f}")
print(f"  centered cos       : {final_ccos:.3f}  (best {best_ccos:.3f}; uncond = 0)")
print(f"  decoded token acc  : model {acc_model:.4f} vs uncond {acc_uncond:.4f}")
print(f"  prediction diversity: {final_div:.2f}  (1.0 = target-like, <0.2 = collapsed)")
print("=" * 64)

print(f"  oracle bar (linear)  : {oracle_ccos:.3f}  — model must ≥ this")
if final_div < 0.2:
    print("❌ COLLAPSED — predictions barely vary with context. Raise "
          "aux_weight, extend budget, or scale data before re-judging.")
elif best_ccos >= oracle_ccos and acc_model > acc_uncond:
    print("✅ CONTEXTUAL SIGNAL CONFIRMED — the model matches/beats the "
          "linear oracle and wins on decoded tokens. The hybrid VELM loop "
          "works; quality is now a scaling question (data → params → steps). "
          "Then qTTT + CIB.")
elif best_ccos >= 0.5 * oracle_ccos:
    print("🟡 PARTIAL — real contextual signal but below the linear-oracle "
          "bar despite seeing strictly more information. Levers: more "
          "steps/data, larger dim, or lower aux_weight late in training.")
else:
    print("❌ Below half the oracle bar — the training setup is still not "
          "extracting the structure a ridge regression finds. Re-examine "
          "objective/LR/capacity before scaling.")
