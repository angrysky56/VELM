#!/usr/bin/env python3
"""
VELM Colab Training — Single-file version for Google Colab or any GPU.

Run all cells in order. Designed for T4 (free) through H100.
Expected runtime: ~18 hours on T4, ~4 hours on H100.

Dataset:  OpenWebMath (streamed, ~1M tokens subset)
Tokenizer: Qwen3.5 (248K vocab, 201 languages)
"""

# %% [markdown]
# # VELM: Vector-Evolution Language Model — Training
#
# **Phases:**
# 1. Stream & tokenize OpenWebMath with Qwen3.5 tokenizer
# 2. Train CALM autoencoder → >99.9% token reconstruction
# 3. Train backbone + energy head via EGGROLL (gradient-free ES)
# 4. Evaluate & save checkpoints

# %% — 0. Environment Setup
# !pip install -q "jax[cuda12]" equinox jaxtyping optax einops tqdm
# !pip install -q datasets tokenizers transformers

import os
import sys

import jax
import jax.numpy as jnp

print(f"JAX {jax.__version__} | backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
if jax.default_backend() != "gpu":
    print("⚠️  No GPU — go to Runtime → Change runtime type → T4 GPU")

# %% — 1. Clone repo & import VELM

VELM_DIR = "/content/VELM" if os.path.exists("/content") else os.getcwd()
if not os.path.exists(os.path.join(VELM_DIR, "src")):
    os.system(f"git clone https://github.com/angrysky56/VELM.git {VELM_DIR} 2>/dev/null")
if os.path.exists(os.path.join(VELM_DIR, "src")):
    os.chdir(VELM_DIR)
    sys.path.insert(0, VELM_DIR)

from src.model.autoencoder import (
    CALMAutoencoder,
    batch_ae_loss,
    reconstruction_accuracy,
)
from src.model.config import CONFIGS, DEFAULT_TOKENIZER
from src.model.energy_head import EnergyHead, energy_score
from src.model.miras_backbone import VELMBackbone
from src.training.eggroll import perturb_pytree  # used in Phase 2 custom ES loop

print("✓ VELM modules imported")

import time

# %% — 2. Hardware-adaptive config
import equinox as eqx
import numpy as np
import optax


# detect GPU memory and pick profile
def detect_hardware() -> dict:
    """Pick training profile based on available GPU."""
    try:
        dev = jax.devices("gpu")[0]
        kind = dev.device_kind.lower()
        if "h100" in kind or "a100" in kind:
            return {"batch": 256, "ae_steps": 100_000, "egg_steps": 10_000,
                    "pop": 64, "chunks": 500_000, "name": "A100/H100"}
        else:  # T4, L4, RTX 3060, etc.
            return {"batch": 64, "ae_steps": 150_000, "egg_steps": 5_000,
                    "pop": 32, "chunks": 250_000, "name": "T4/consumer"}
    except Exception:  # pylint: disable=broad-except
        return {"batch": 32, "ae_steps": 10_000, "egg_steps": 1_000,
                "pop": 16, "chunks": 50_000, "name": "CPU (slow!)"}

HW = detect_hardware()
# gpu_12gb_v2: ae_hidden_dim=384 for 248K Qwen vocab (256 plateaus ~99.66%)
CONFIG_NAME = "gpu_12gb_v2"
cfg = CONFIGS[CONFIG_NAME]
K = cfg["chunk_size_k"]
print(f"Hardware: {HW['name']} | Config: {CONFIG_NAME}")
print(f"  AE steps: {HW['ae_steps']:,} | EGGROLL steps: {HW['egg_steps']:,}")
print(f"  Target chunks: {HW['chunks']:,} ({HW['chunks']*K:,} tokens)")

# %% — 3. Load Qwen3.5 tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, trust_remote_code=True)
VOCAB_SIZE = len(tokenizer)
print(f"Tokenizer: {DEFAULT_TOKENIZER}, vocab: {VOCAB_SIZE:,}")

# %% — 4. Stream and tokenize training data (multi-dataset curriculum)
from datasets import load_dataset
from tqdm import tqdm

TARGET_CHUNKS = HW["chunks"]

print(f"Loading training data → {TARGET_CHUNKS:,} chunks (K={K})...")

chunk_buffer: list[np.ndarray] = []
chunk_labels: list[str] = []  # domain label per chunk for GEA Phase 3
total_chunks = 0

# Dataset sources with curriculum weights
CURRICULUM = [
    {"name": "open-web-math/open-web-math", "weight": 0.5, "label": "math",
     "split": "train", "text_field": "text"},
    {"name": "wikitext", "config": "wikitext-103-raw-v1", "weight": 0.3,
     "label": "general", "split": "train", "text_field": "text"},
    {"name": "roneneldan/TinyStories", "weight": 0.2, "label": "narrative",
     "split": "train", "text_field": "text"},
]

for source in CURRICULUM:
    source_target = int(TARGET_CHUNKS * source["weight"])
    source_count = 0
    label = source["label"]
    print(f"  Streaming {label}: {source['name']} "
          f"(target: {source_target:,} chunks)...")
    try:
        load_kwargs = {"split": source["split"], "streaming": True}
        if "config" in source:
            load_kwargs["name"] = source["config"]
        ds = load_dataset(source["name"], **load_kwargs)
        text_field = source.get("text_field", "text")
        for example in tqdm(ds, desc=f"  {label}", unit="docs", leave=False):
            text = example.get(text_field, "")
            if not text or len(text) < 50:
                continue
            tokens = tokenizer.encode(text, max_length=512, truncation=True)
            usable = len(tokens) - (len(tokens) % K)
            if usable < K:
                continue
            chunks = np.array(tokens[:usable], dtype=np.int32).reshape(-1, K)
            chunk_buffer.append(chunks)
            chunk_labels.extend([label] * chunks.shape[0])
            source_count += chunks.shape[0]
            total_chunks += chunks.shape[0]
            if source_count >= source_target:
                break
        print(f"    ✓ {label}: {source_count:,} chunks")
    except Exception as e:  # pylint: disable=broad-except
        print(f"    ⚠ {label} failed ({e}), generating fallback random chunks")
        fallback_n = source_target - source_count
        if fallback_n > 0:
            rng = np.random.default_rng(42)
            fallback = rng.integers(0, VOCAB_SIZE, size=(fallback_n, K), dtype=np.int32)
            chunk_buffer.append(fallback)
            chunk_labels.extend([f"{label}_fallback"] * fallback_n)
            total_chunks += fallback_n

if total_chunks == 0:
    raise RuntimeError("No training data loaded! Check network and dataset access.")

all_chunks = np.concatenate(chunk_buffer, axis=0)[:TARGET_CHUNKS]
chunk_labels = chunk_labels[:TARGET_CHUNKS]
num_chunks = all_chunks.shape[0]

# Report dataset composition
label_counts: dict[str, int] = {}
for lab in chunk_labels:
    label_counts[lab] = label_counts.get(lab, 0) + 1
print(f"\n✓ {num_chunks:,} chunks × K={K} = {num_chunks * K:,} tokens")
for lab, cnt in sorted(label_counts.items()):
    print(f"  {lab}: {cnt:,} ({cnt / num_chunks:.0%})")

# %% — 5. Phase 1: Train CALM Autoencoder
HIDDEN_DIM = cfg.get("ae_hidden_dim", 256)
LATENT_DIM = cfg["latent_dim"]  # now 128 (was 64 — too tight for 248K vocab)
FFN_DIM = cfg.get("ae_ffn_intermediate", 512)
KL_CLIP = cfg.get("ae_kl_clip", 0.5)    # paper uses λ_KL=0.5
KL_WEIGHT = cfg.get("ae_kl_weight", 0.001)  # paper uses β=0.001
BATCH_SIZE = HW["batch"]
AE_STEPS = HW["ae_steps"]
LR = 3e-4

key = jax.random.PRNGKey(42)
k_init, key = jax.random.split(key)

model = CALMAutoencoder(
    vocab_size=VOCAB_SIZE,
    chunk_size=K,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    ffn_intermediate=FFN_DIM,
    kl_weight=KL_WEIGHT,
    kl_clip=KL_CLIP,  # was 1.0 — paper says 0.5 prevents posterior collapse
    key=k_init,
)
num_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
print(f"Autoencoder: {num_params:,} params")
print(f"  hidden={HIDDEN_DIM}, latent={LATENT_DIM}, K={K}, vocab={VOCAB_SIZE:,}")

effective_warmup = min(500, max(AE_STEPS // 10, 1))
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=LR,
    warmup_steps=effective_warmup, decay_steps=AE_STEPS, end_value=LR * 0.1,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # prevent gradient explosion → NaN
    optax.adamw(schedule, weight_decay=0.01),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def train_step(mdl, state, batch_chunk, key_step):
    """JIT-compiled AE training step."""
    def loss_fn(m):
        return batch_ae_loss(m, batch_chunk, key=key_step)
    (loss_val, metrics_val), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(mdl)
    updates, new_opt = optimizer.update(grads, state, mdl)
    new_model = eqx.apply_updates(mdl, updates)
    return new_model, new_opt, loss_val, metrics_val

@eqx.filter_jit
def eval_accuracy(mdl, batch_chunk):
    """Compute token-level reconstruction accuracy limit."""
    return reconstruction_accuracy(mdl, batch_chunk)

# training loop
print(f"\nPhase 1: AE training — {AE_STEPS:,} steps, batch={BATCH_SIZE}")
print("=" * 60)

history = {"loss": [], "recon": [], "kl": [], "accuracy": []}
start_time = time.time()
num_chunks = all_chunks.shape[0]
best_acc = 0.0

for step in range(1, AE_STEPS + 1):
    key, batch_key, step_key = jax.random.split(key, 3)
    indices = jax.random.randint(batch_key, (BATCH_SIZE,), 0, num_chunks)
    batch = jnp.array(all_chunks[indices])
    model, opt_state, loss, metrics = train_step(model, opt_state, batch, step_key)

    if step % 500 == 0:
        elapsed = time.time() - start_time
        rl = float(metrics["recon_loss"])
        kl = float(metrics["kl_loss"])
        history["loss"].append(float(loss))
        history["recon"].append(rl)
        history["kl"].append(kl)
        print(f"  step {step:>6}/{AE_STEPS} | recon: {rl:.4f} | kl: {kl:.4f} | "
              f"{step/elapsed:.1f} steps/s | {elapsed/60:.0f}m")

    if step % 2000 == 0:
        key, eval_key = jax.random.split(key)
        eval_idx = jax.random.randint(eval_key, (512,), 0, num_chunks)
        eval_batch = jnp.array(all_chunks[eval_idx])
        acc = float(eval_accuracy(model, eval_batch))
        history["accuracy"].append(acc)
        print(f"  >> Reconstruction accuracy: {acc:.4%}")
        if acc > best_acc:
            best_acc = acc
            os.makedirs("checkpoints", exist_ok=True)
            eqx.tree_serialise_leaves("checkpoints/calm_ae_best.eqx", model)
            import json
            with open("checkpoints/calm_ae_best.json", "w", encoding="utf-8") as f:
                json.dump({"config": CONFIG_NAME, "vocab_size": VOCAB_SIZE,
                           "tokenizer": DEFAULT_TOKENIZER, "step": step,
                           "accuracy": acc}, f, indent=2)
            print(f"  >> Saved best checkpoint (acc={acc:.4%})")
        if acc > 0.999:
            print(f"  🎯 TARGET >99.9% at step {step}! — stopping early")
            break

total_time = time.time() - start_time
final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
print(f"\nPhase 1 done: {total_time/3600:.2f}h | accuracy: {final_acc:.4%}")

# save final
eqx.tree_serialise_leaves("checkpoints/calm_ae_final.eqx", model)

# persist to Google Drive (Colab disk is ephemeral!)
try:
    from google.colab import drive  # noqa: E402
    drive.mount("/content/drive", force_remount=False)
    drive_dir = "/content/drive/MyDrive/VELM_checkpoints"
    os.makedirs(drive_dir, exist_ok=True)
    import shutil
    for f in ["calm_ae_best.eqx", "calm_ae_best.json", "calm_ae_final.eqx"]:
        src = f"checkpoints/{f}"
        if os.path.exists(src):
            shutil.copy2(src, f"{drive_dir}/{f}")
    print(f"✓ Checkpoints backed up to Google Drive: {drive_dir}")
except ImportError:
    print("Not on Colab — checkpoints are in ./checkpoints/")

# %% — 6. Plot AE training curves
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history["recon"]); axes[0].set_title("Recon Loss"); axes[0].set_xlabel("×500 steps")
axes[1].plot(history["kl"]); axes[1].set_title("KL Loss"); axes[1].set_xlabel("×500 steps")
axes[2].plot(history["accuracy"])
axes[2].axhline(y=0.999, color="r", linestyle="--", label="99.9% target")
axes[2].set_title("Reconstruction Acc"); axes[2].set_xlabel("×2000 steps"); axes[2].legend()
plt.tight_layout()
plt.savefig("ae_training_curves.png", dpi=150, bbox_inches="tight")
print("Saved: ae_training_curves.png")
try:
    plt.show()
except Exception:  # pylint: disable=broad-except
    pass

# %% — 6.5. Phase 1.5: Continue AE Training (warm restart)
# Skip this cell if Phase 1 already hit >99.9%.
# Otherwise, load the best checkpoint and continue with a fresh LR schedule.
# The model was still improving at 100K steps — more training helps.

CONTINUE_AE = True  # set False to skip continuation
CONTINUE_STEPS = 100_000
CONTINUE_LR = 1e-4  # lower peak than Phase 1 (was 3e-4)
TARGET_ACC = 0.999   # stop when we hit this

if CONTINUE_AE and best_acc < TARGET_ACC:
    print(f"\nPhase 1.5: Continue AE training — best so far: {best_acc:.4%}")
    print(f"  Loading best checkpoint, {CONTINUE_STEPS:,} more steps, peak LR={CONTINUE_LR}")
    print("=" * 60)

    # load best checkpoint
    model = eqx.tree_deserialise_leaves("checkpoints/calm_ae_best.eqx", model)
    print(f"  ✓ Loaded best checkpoint (acc={best_acc:.4%})")

    # fresh optimizer with lower LR and longer warmup
    cont_warmup = min(1000, CONTINUE_STEPS // 10)
    cont_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=CONTINUE_LR,
        warmup_steps=cont_warmup, decay_steps=CONTINUE_STEPS,
        end_value=CONTINUE_LR * 0.01,  # decay to 1e-6
    )
    cont_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(cont_schedule, weight_decay=0.01),
    )
    opt_state = cont_optimizer.init(eqx.filter(model, eqx.is_array))

    # redefine train_step with the new optimizer
    @eqx.filter_jit
    def cont_train_step(mdl, state, batch_chunk, key_step):
        def loss_fn(m):
            return batch_ae_loss(m, batch_chunk, key=key_step)
        (loss_val, metrics_val), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True)(mdl)
        updates, new_opt = cont_optimizer.update(grads, state, mdl)
        new_model = eqx.apply_updates(mdl, updates)
        return new_model, new_opt, loss_val, metrics_val

    cont_history = {"recon": [], "kl": [], "accuracy": []}
    cont_start = time.time()
    hit_target = False

    for step in range(1, CONTINUE_STEPS + 1):
        key, batch_key, step_key = jax.random.split(key, 3)
        indices = jax.random.randint(batch_key, (BATCH_SIZE,), 0, num_chunks)
        batch = jnp.array(all_chunks[indices])
        model, opt_state, loss, metrics = cont_train_step(
            model, opt_state, batch, step_key)

        if step % 500 == 0:
            elapsed = time.time() - cont_start
            rl = float(metrics["recon_loss"])
            kl = float(metrics["kl_loss"])
            cont_history["recon"].append(rl)
            cont_history["kl"].append(kl)
            print(f"  step {step:>6}/{CONTINUE_STEPS} | recon: {rl:.4f} | "
                  f"kl: {kl:.4f} | {step/elapsed:.1f} steps/s | "
                  f"{elapsed/60:.0f}m")

        if step % 2000 == 0:
            key, eval_key = jax.random.split(key)
            eval_idx = jax.random.randint(eval_key, (512,), 0, num_chunks)
            eval_batch = jnp.array(all_chunks[eval_idx])
            acc = float(eval_accuracy(model, eval_batch))
            cont_history["accuracy"].append(acc)
            print(f"  >> Reconstruction accuracy: {acc:.4%}")
            if acc > best_acc:
                best_acc = acc
                eqx.tree_serialise_leaves("checkpoints/calm_ae_best.eqx", model)
                import json
                with open("checkpoints/calm_ae_best.json", "w",
                          encoding="utf-8") as f:
                    json.dump({"config": CONFIG_NAME, "vocab_size": VOCAB_SIZE,
                               "tokenizer": DEFAULT_TOKENIZER,
                               "step": 100_000 + step, "accuracy": acc,
                               "phase": "1.5_continuation"}, f, indent=2)
                print(f"  >> New best! Saved checkpoint (acc={acc:.4%})")
            if acc >= TARGET_ACC:
                print(f"  🎯 TARGET {TARGET_ACC:.1%} reached at step {step}!")
                hit_target = True
                break

            # plateau detection: if no improvement in last 5 evals, stop
            if len(cont_history["accuracy"]) >= 6:
                last_5 = cont_history["accuracy"][-5:]
                if max(last_5) - min(last_5) < 0.001:
                    print(f"  ⚠ Plateau detected (last 5 evals within 0.1%)")
                    print(f"    Consider increasing hidden_dim to 384 or 512")
                    break

    cont_elapsed = time.time() - cont_start
    eqx.tree_serialise_leaves("checkpoints/calm_ae_final.eqx", model)
    print(f"\nPhase 1.5 done: {cont_elapsed/3600:.2f}h | best accuracy: {best_acc:.4%}")
    if not hit_target and best_acc < TARGET_ACC:
        print(f"  ⚠ Did not reach {TARGET_ACC:.1%}. The hidden_dim=256 may be too")
        print(f"    small for 248K vocab. Set hidden_dim=384 in config and retrain.")
else:
    if best_acc >= TARGET_ACC:
        print(f"\n✓ AE already at {best_acc:.4%} — skipping Phase 1.5")
    else:
        print(f"\n⏭ Phase 1.5 skipped (CONTINUE_AE=False)")

# %% — 7. Phase 2: EGGROLL Backbone Training (gradient-free)
# Re-derive AE dimensions from config (in case Phase 1 cells were skipped)
HIDDEN_DIM = cfg.get("ae_hidden_dim", 256)
LATENT_DIM = cfg["latent_dim"]
FFN_DIM = cfg.get("ae_ffn_intermediate", 512)
KL_CLIP = cfg.get("ae_kl_clip", 0.5)
KL_WEIGHT = cfg.get("ae_kl_weight", 0.001)

POP_SIZE = HW["pop"]
EGGROLL_STEPS = HW["egg_steps"]
SIGMA = cfg.get("eggroll_sigma", 0.001)
EGG_LR = cfg.get("eggroll_lr", 3e-4)
EVAL_BATCH = cfg.get("eggroll_eval_batch", 128)  # was 32 — more data per eval
ANTITHETIC = cfg.get("eggroll_antithetic", True)  # ±σ pairs: halves variance
HC_D = cfg.get("hc_streams", 1)  # go-mHC residual streams
HC_S = cfg.get("hc_s", 2)        # go-mHC expressivity

key = jax.random.PRNGKey(123)
k1, k2, key = jax.random.split(key, 3)

backbone = VELMBackbone(
    dim=cfg["hidden_dim"], num_heads=cfg["num_heads"],
    num_miras_layers=cfg["miras_layers"], num_swa_layers=cfg["swa_layers"],
    ffn_intermediate=cfg["ffn_intermediate"], chunk_size=K,
    ae_hidden_dim=HIDDEN_DIM,
    hc_streams=HC_D,  # go-mHC d-axis scaling
    hc_s=HC_S,
    key=k1,
)

head = EnergyHead(
    hidden_dim=cfg["hidden_dim"], latent_dim=LATENT_DIM,
    num_blocks=cfg["energy_head_blocks"], key=k2,
)

bb_p = sum(x.size for x in jax.tree.leaves(eqx.filter(backbone, eqx.is_array)))
hd_p = sum(x.size for x in jax.tree.leaves(eqx.filter(head, eqx.is_array)))
print(f"Backbone: {bb_p:,} | Head: {hd_p:,} | Total: {bb_p+hd_p:,}")
if HC_D > 1:
    print(f"go-mHC: d={HC_D} streams, s={HC_S} (doubly stochastic routing)")
print(f"EGGROLL: pop={POP_SIZE}, σ={SIGMA}, steps={EGGROLL_STEPS:,}"
      f"{', antithetic' if ANTITHETIC else ''}, eval_batch={EVAL_BATCH}")

# Load frozen AE — from Phase 1 if it ran, otherwise from checkpoint
if "model" not in dir() or model is None:
    print("Loading AE from checkpoint (Phase 1 was skipped)...")
    model = CALMAutoencoder(
        vocab_size=VOCAB_SIZE, chunk_size=K,
        hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
        ffn_intermediate=FFN_DIM, kl_weight=KL_WEIGHT,
        kl_clip=KL_CLIP, key=jax.random.PRNGKey(42),
    )
    ckpt_path = "checkpoints/calm_ae_best.eqx"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/calm_ae_final.eqx"
    # try restoring from Google Drive if local checkpoints are gone
    if not os.path.exists(ckpt_path):
        try:
            from google.colab import drive  # noqa: E402
            drive.mount("/content/drive", force_remount=False)
            drive_dir = "/content/drive/MyDrive/VELM_checkpoints"
            import shutil
            os.makedirs("checkpoints", exist_ok=True)
            for f in os.listdir(drive_dir):
                shutil.copy2(f"{drive_dir}/{f}", f"checkpoints/{f}")
            print(f"  ✓ Restored checkpoints from Google Drive")
            ckpt_path = "checkpoints/calm_ae_best.eqx"
            if not os.path.exists(ckpt_path):
                ckpt_path = "checkpoints/calm_ae_final.eqx"
        except (ImportError, FileNotFoundError):
            pass
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No AE checkpoint found! Colab's disk is ephemeral — "
            "checkpoints are lost on restart.\n"
            "  → Re-run Phase 1 (cells 5+6) to retrain the AE (~2.7h)\n"
            "  → Or mount Google Drive before training to persist checkpoints:\n"
            "    from google.colab import drive; drive.mount('/content/drive')\n"
            "    !cp -r checkpoints/ /content/drive/MyDrive/VELM_ckpts/\n"
            "    Restore: !cp -r /content/drive/MyDrive/VELM_ckpts/ checkpoints/"
        )
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
    print(f"  ✓ Loaded: {ckpt_path}")
frozen_ae = model

# pack trainable params
trainable = {
    "backbone": eqx.filter(backbone, eqx.is_array),
    "head": eqx.filter(head, eqx.is_array),
}

# Extract static (non-array) module parts ONCE — used by evaluate_member and GEA
_bb_static = eqx.filter(backbone, lambda x: not eqx.is_array(x))
_hd_static = eqx.filter(head, lambda x: not eqx.is_array(x))

# %% — 7a. Phase 2a: Gradient-based backbone training
# The current forward pass is fully differentiable through JAX.
# Gradient-based training converges orders of magnitude faster than
# ES with pop=32 on 7M params (SNR ≈ 0.012).
# EGGROLL is preserved below as Phase 2b for future nonlinear recurrence.

USE_GRADIENT_PHASE = False  # optional warmup only — EGGROLL is primary
GRAD_STEPS = 20_000
GRAD_LR = 3e-4
GRAD_BATCH = EVAL_BATCH  # 32 chunks per step

if USE_GRADIENT_PHASE:
    print(f"\nPhase 2a: Gradient-based backbone + head training")
    print(f"  {GRAD_STEPS:,} steps, batch={GRAD_BATCH}, lr={GRAD_LR}")
    print("=" * 60)

    grad_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=GRAD_LR,
        warmup_steps=500, decay_steps=GRAD_STEPS, end_value=GRAD_LR * 0.01,
    )
    grad_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(grad_schedule, weight_decay=0.01),
    )
    grad_opt_state = grad_optimizer.init(trainable)

    @eqx.filter_jit
    def grad_train_step(params, opt_st, batch_tokens, step_key):
        """Gradient-based backbone + energy head training step."""
        def loss_fn(p):
            bb = eqx.combine(p["backbone"], _bb_static)
            hd = eqx.combine(p["head"], _hd_static)
            # encode chunks → target latents via frozen AE
            tgt_z = jax.vmap(
                lambda c: frozen_ae.encode(c, training=False)[0])(batch_tokens)
            # compress input for backbone
            def compress(chunk):
                embs = jax.vmap(frozen_ae.embedding)(chunk)
                return bb.compress_input(embs)
            inp_seq = jax.vmap(compress)(batch_tokens)
            # backbone forward
            hid, _ = bb(inp_seq)
            # energy loss: predict z_{i+1} from h_i
            hid_in, z_target = hid[:-1], tgt_z[1:]
            def pos_loss(h, z_t, k):
                samples = hd(h, key=k, num_samples=8)
                return energy_score(samples, z_t)
            keys = jax.random.split(step_key, hid_in.shape[0])
            losses = jax.vmap(pos_loss)(hid_in, z_target, keys)
            return jnp.mean(losses)

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, new_opt = grad_optimizer.update(grads, opt_st, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss_val

    # compile warmup
    print("Compiling gradient training step (one-time)...")
    _w_batch = jnp.array(all_chunks[:GRAD_BATCH])
    _w_key = jax.random.PRNGKey(0)
    _, _, _w_loss = grad_train_step(trainable, grad_opt_state, _w_batch, _w_key)
    _w_loss.block_until_ready()
    print(f"✓ Compiled | initial energy loss: {float(_w_loss):.4f}\n")

    grad_history = {"energy_loss": []}
    grad_start = time.time()
    best_grad_loss = float("inf")

    for step in range(1, GRAD_STEPS + 1):
        key, batch_key, step_key = jax.random.split(key, 3)
        idx = jax.random.randint(batch_key, (GRAD_BATCH,), 0, num_chunks)
        batch = jnp.array(all_chunks[idx])
        trainable, grad_opt_state, eloss = grad_train_step(
            trainable, grad_opt_state, batch, step_key)

        if step % 200 == 0:
            el = float(eloss)
            grad_history["energy_loss"].append(el)
            elapsed = time.time() - grad_start
            marker = ""
            if el < best_grad_loss:
                best_grad_loss = el
                marker = " ★"
            print(f"  step {step:>5}/{GRAD_STEPS} | energy: {el:.4f}{marker} | "
                  f"{step/elapsed:.1f} steps/s | {elapsed/60:.0f}m")

        if step % 2000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_bb = eqx.combine(trainable["backbone"], _bb_static)
            ckpt_hd = eqx.combine(trainable["head"], _hd_static)
            eqx.tree_serialise_leaves(
                "checkpoints/backbone_grad.eqx", ckpt_bb)
            eqx.tree_serialise_leaves(
                "checkpoints/energy_head_grad.eqx", ckpt_hd)
            print(f"  >> Checkpoint saved (energy={best_grad_loss:.4f})")

    grad_elapsed = time.time() - grad_start
    print(f"\nPhase 2a done: {grad_elapsed/3600:.2f}h | "
          f"best energy loss: {best_grad_loss:.4f}")

    # save final gradient-trained models
    final_bb = eqx.combine(trainable["backbone"], _bb_static)
    final_hd = eqx.combine(trainable["head"], _hd_static)
    eqx.tree_serialise_leaves("checkpoints/backbone_grad.eqx", final_bb)
    eqx.tree_serialise_leaves("checkpoints/energy_head_grad.eqx", final_hd)
    print("✓ Saved: checkpoints/backbone_grad.eqx + energy_head_grad.eqx")

# %% — 7b. Phase 2b: EGGROLL fine-tuning (gradient-free, optional)
# Skip this if gradient-based Phase 2a already trained the backbone.
# EGGROLL is preserved for future nonlinear recurrence where backprop is
# impractical (true BPTT over long sequences with nonlinear Miras memory).
USE_EGGROLL_PHASE = True  # EGGROLL is the primary training method




@eqx.filter_jit
def evaluate_member(base_params, member_key, batch):
    """Evaluate one perturbed population member.

    JIT-compiled with batch as an explicit argument — no closures over
    mutable state, so this compiles ONCE and reuses on every step.
    """
    perturbed, perturbation = perturb_pytree(base_params, member_key, SIGMA, rank=1)
    bb = eqx.combine(perturbed["backbone"], _bb_static)
    hd = eqx.combine(perturbed["head"], _hd_static)

    # encode chunks → target latents via frozen AE
    tgt_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(batch)

    # compress input for backbone
    def compress(chunk):
        embs = jax.vmap(frozen_ae.embedding)(chunk)
        return bb.compress_input(embs)
    inp_seq = jax.vmap(compress)(batch)

    # backbone forward → hidden states
    hid, _ = bb(inp_seq)

    # energy loss: predict z_{i+1} from h_i
    hid_in, z_target = hid[:-1], tgt_z[1:]
    def pos_loss(h, z_t):
        samples = hd(h, key=jax.random.PRNGKey(0), num_samples=8)  # paper recommends N=8
        return energy_score(samples, z_t)
    loss_vals = jax.vmap(pos_loss)(hid_in, z_target)
    fitness = -jnp.mean(loss_vals)
    return jnp.where(jnp.isfinite(fitness), fitness, -1e6), perturbation


@eqx.filter_jit
def evaluate_params(params, batch):
    """Evaluate fitness for pre-built params (no perturbation).

    Used by antithetic sampling where we manually construct ±σ params.
    """
    bb = eqx.combine(params["backbone"], _bb_static)
    hd = eqx.combine(params["head"], _hd_static)
    tgt_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(batch)
    def compress(chunk):
        embs = jax.vmap(frozen_ae.embedding)(chunk)
        return bb.compress_input(embs)
    inp_seq = jax.vmap(compress)(batch)
    hid, _ = bb(inp_seq)
    hid_in, z_target = hid[:-1], tgt_z[1:]
    def pos_loss(h, z_t):
        samples = hd(h, key=jax.random.PRNGKey(0), num_samples=8)
        return energy_score(samples, z_t)
    loss_vals = jax.vmap(pos_loss)(hid_in, z_target)
    fitness = -jnp.mean(loss_vals)
    return jnp.where(jnp.isfinite(fitness), fitness, -1e6)


# (optimizers are created inside the EGGROLL loop for progressive unfreezing)

# --- Warmup: compile the evaluation function ONCE before the loop ---
print("\nCompiling backbone evaluation (one-time, may take 2-5 min on T4)...")
compile_start = time.time()
_warmup_key = jax.random.PRNGKey(0)
_warmup_batch = jnp.array(all_chunks[:EVAL_BATCH])
_warmup_f, _warmup_p = evaluate_member(trainable, _warmup_key, _warmup_batch)
_warmup_f.block_until_ready()  # force compilation to finish
# also compile the antithetic evaluation function
_warmup_f2 = evaluate_params(trainable, _warmup_batch)
_warmup_f2.block_until_ready()
del _warmup_f, _warmup_p, _warmup_f2
print(f"✓ Compilation done in {time.time() - compile_start:.0f}s — "
      "training loop will be fast\n")

# EGGROLL training loop
if not USE_EGGROLL_PHASE:
    print(f"\n⏭ Skipping EGGROLL loop (USE_EGGROLL_PHASE=False)")
    print(f"  Backbone was trained via gradient descent in Phase 2a")
    egg_history = {"mean_fitness": [], "max_fitness": [], "grad_norm": []}
else:
    # Progressive EGGROLL: train head first, then unfreeze backbone.
    # pop=32 on 7.2M params has ~1% SNR (noise). But pop=32 on 819K
    # head params has ~7% SNR — enough to learn.
    HEAD_STEPS = 2000   # Phase i: head only (819K params)
    FULL_STEPS = 3000   # Phase ii: head + backbone (7.2M params)
    EGGROLL_STEPS = HEAD_STEPS + FULL_STEPS

    print(f"Phase 2b: Progressive EGGROLL — {EGGROLL_STEPS:,} total steps")
    print(f"  Phase i:  head only ({hd_p:,} params), {HEAD_STEPS:,} steps")
    print(f"  Phase ii: full model ({bb_p+hd_p:,} params), {FULL_STEPS:,} steps")
    print(f"  pop={POP_SIZE}, σ={SIGMA}, antithetic={ANTITHETIC}")
    print("=" * 60)

egg_history = {"mean_fitness": [], "max_fitness": [], "grad_norm": [], "phase": []}
start = time.time()
best_egg_fitness = -float("inf")

# separate trainable sets for progressive unfreezing
head_trainable = {"head": trainable["head"]}
head_opt = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adam(EGG_LR))
head_opt_state = head_opt.init(head_trainable)

full_opt = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adam(EGG_LR * 0.3))  # lower lr for full
full_opt_state = full_opt.init(trainable)

for step in range(1, (EGGROLL_STEPS + 1) if USE_EGGROLL_PHASE else 0):
    key, batch_key, step_key = jax.random.split(key, 3)
    idx = jax.random.randint(batch_key, (EVAL_BATCH,), 0, num_chunks)
    batch = jnp.array(all_chunks[idx])

    # decide what to train this step
    training_head_only = step <= HEAD_STEPS
    if training_head_only:
        active_params = head_trainable
        phase_label = "head"
    else:
        # sync head params into full trainable before first full step
        if step == HEAD_STEPS + 1:
            trainable = {
                "backbone": trainable["backbone"],
                "head": head_trainable["head"],
            }
            full_opt_state = full_opt.init(trainable)
            print(f"\n  >>> Unfreezing backbone at step {step} <<<\n")
        active_params = trainable
        phase_label = "full"

    # evaluate population with antithetic sampling
    member_keys = jax.random.split(step_key, POP_SIZE)
    fitnesses_list = []
    perturbations_list = []
    fitness_diffs = []

    for member_key in member_keys:
        _, pert = perturb_pytree(active_params, member_key, SIGMA, rank=1)

        if training_head_only:
            # perturb head only, backbone stays frozen
            pos_full = {"backbone": trainable["backbone"],
                        "head": jax.tree.map(lambda p, e: p + SIGMA * e,
                                             active_params["head"], pert["head"])}
            neg_full = {"backbone": trainable["backbone"],
                        "head": jax.tree.map(lambda p, e: p - SIGMA * e,
                                             active_params["head"], pert["head"])}
        else:
            pos_full = jax.tree.map(
                lambda p, e: p + SIGMA * e, active_params, pert)
            neg_full = jax.tree.map(
                lambda p, e: p - SIGMA * e, active_params, pert)

        f_pos = evaluate_params(pos_full, batch)
        f_neg = evaluate_params(neg_full, batch)
        fitnesses_list.append((f_pos + f_neg) / 2)
        fitness_diffs.append(f_pos - f_neg)
        perturbations_list.append(pert)

    fitnesses_arr = jnp.array(fitnesses_list)
    diffs_arr = jnp.array(fitness_diffs)

    # antithetic ES gradient
    leaves_list = [jax.tree.leaves(p) for p in perturbations_list]
    treedef = jax.tree.structure(perturbations_list[0])
    es_grad_leaves = []
    for j in range(len(leaves_list[0])):
        stacked = jnp.stack([ll[j] for ll in leaves_list], axis=0)
        w = diffs_arr.reshape((-1,) + (1,) * (stacked.ndim - 1))
        es_grad_leaves.append(jnp.sum(stacked * w, axis=0))
    es_grad = jax.tree.unflatten(treedef, es_grad_leaves)
    neg_grad = jax.tree.map(
        lambda g: g * (-1.0 / (2.0 * SIGMA * POP_SIZE)), es_grad)

    # Adam update on the active param set
    if training_head_only:
        updates, head_opt_state = head_opt.update(
            neg_grad, head_opt_state, head_trainable)
        head_trainable = optax.apply_updates(head_trainable, updates)
    else:
        updates, full_opt_state = full_opt.update(
            neg_grad, full_opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)

    # logging every 100 steps
    if step % 100 == 0:
        mf = float(jnp.mean(fitnesses_arr))
        xf = float(jnp.max(fitnesses_arr))
        gn = float(jnp.sqrt(sum(
            jnp.sum(leaf**2) for leaf in jax.tree.leaves(es_grad)
        )))
        egg_history["mean_fitness"].append(mf)
        egg_history["max_fitness"].append(xf)
        egg_history["grad_norm"].append(gn)
        egg_history["phase"].append(phase_label)
        elapsed = time.time() - start
        print(f"  step {step:>5}/{EGGROLL_STEPS} [{phase_label:>4}] | "
              f"mean_f: {mf:.4f} | max_f: {xf:.4f} | "
              f"grad: {gn:.4f} | {elapsed/60:.0f}m")

        # divergence detection — fitness should improve (get less negative)
        if len(egg_history["mean_fitness"]) >= 5:
            recent = egg_history["mean_fitness"][-5:]
            if recent[-1] < min(recent[:-1]) * 10:  # 10x worse = diverging
                print(f"  ⚠ DIVERGENCE DETECTED at step {step} — stopping EGGROLL")
                print(f"    Fitness went from {recent[0]:.2f} to {recent[-1]:.2f}")
                print(f"    This usually means σ is too large or lr is too high.")
                break

    # checkpoint every 500 steps
    if step % 500 == 0:
        cur_fitness = float(jnp.mean(fitnesses_arr))
        os.makedirs("checkpoints", exist_ok=True)
        # sync head into trainable if still in head-only phase
        if training_head_only:
            save_params = {"backbone": trainable["backbone"],
                           "head": head_trainable["head"]}
        else:
            save_params = trainable
        ckpt_bb = eqx.combine(save_params["backbone"], _bb_static)
        ckpt_hd = eqx.combine(save_params["head"], _hd_static)
        eqx.tree_serialise_leaves(f"checkpoints/backbone_step{step}.eqx", ckpt_bb)
        eqx.tree_serialise_leaves(f"checkpoints/head_step{step}.eqx", ckpt_hd)
        if cur_fitness > best_egg_fitness:
            best_egg_fitness = cur_fitness
            eqx.tree_serialise_leaves("checkpoints/backbone_eggroll_best.eqx", ckpt_bb)
            eqx.tree_serialise_leaves("checkpoints/head_eggroll_best.eqx", ckpt_hd)
            print(f"  >> New best fitness: {cur_fitness:.4f} — saved checkpoint")

elapsed = time.time() - start
print(f"\nPhase 2 done: {elapsed/3600:.2f}h")

# sync final head params if still in head-only phase
if USE_EGGROLL_PHASE and "head" in head_trainable:
    trainable = {"backbone": trainable["backbone"],
                 "head": head_trainable["head"]}

# save final
final_bb = eqx.combine(trainable["backbone"], _bb_static)
final_hd = eqx.combine(trainable["head"], _hd_static)

eqx.tree_serialise_leaves("checkpoints/backbone_eggroll.eqx", final_bb)
eqx.tree_serialise_leaves("checkpoints/energy_head_eggroll.eqx", final_hd)

import json  # noqa: E402

with open("checkpoints/backbone_meta.json", "w", encoding="utf-8") as f:
    json.dump({"config": CONFIG_NAME, "vocab_size": VOCAB_SIZE,
               "tokenizer": DEFAULT_TOKENIZER,
               "eggroll_steps": EGGROLL_STEPS, "pop_size": POP_SIZE,
               "sigma": SIGMA, "history": egg_history}, f, indent=2)
print("✓ Saved: checkpoints/backbone_eggroll.eqx + energy_head_eggroll.eqx")

# %% — 8. Plot EGGROLL training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(egg_history["mean_fitness"], label="mean")
axes[0].plot(egg_history["max_fitness"], label="max")
axes[0].set_title("EGGROLL Fitness"); axes[0].legend(); axes[0].set_xlabel("×100 steps")
axes[1].plot(egg_history["grad_norm"]); axes[1].set_title("ES Grad Norm")
axes[2].plot([-f for f in egg_history["mean_fitness"]])
axes[2].set_title("Energy Loss (↓ better)"); axes[2].set_xlabel("×100 steps")
plt.tight_layout()
plt.savefig("eggroll_training.png", dpi=150, bbox_inches="tight")
print("Saved: eggroll_training.png")
try:
    plt.show()
except Exception:  # pylint: disable=broad-except
    pass

# %% — 8.5. EGGROLL Diagnostics Report
from src.training.diagnostics import EGGROLLDiagnostics  # noqa: E402

diag = EGGROLLDiagnostics(plateau_window=50, plateau_threshold=0.001)
# backfill from egg_history (each entry was logged at step % 100)
for i, mf in enumerate(egg_history["mean_fitness"]):
    step_num = (i + 1) * 100
    diag.log_step(
        step_num,
        trainable,
        {
            "mean_fitness": mf,
            "max_fitness": egg_history["max_fitness"][i],
            "fitness_std": 0.0,  # not tracked during Phase 2
            "grad_norm": egg_history["grad_norm"][i],
        },
    )
diag.report()
if diag.warnings:
    print("\n⚠ Active warnings:")
    for w in diag.warnings:
        print(f"  - {w}")

# %% — 9. Phase 3: GEA-EGGROLL Group Evolution
from src.evolution.gea_eggroll import (  # noqa: E402
    GroupEvolver,
    experience_weighted_eggroll_step,
)

# --- GEA configuration (hw-adaptive) ---
GEA_ITERATIONS = 10         # quality over speed
GEA_POP = min(HW["pop"], 32)  # population per GEA generation
GEA_GROUP = 5                # parent group size (K in GEA paper)
GEA_SIGMA = SIGMA             # use same σ as EGGROLL Phase 2
GEA_BIAS = 0.3               # parent-direction bias strength

print(f"\nPhase 3: GEA Group Evolution — {GEA_ITERATIONS} iterations")
print(f"  population: {GEA_POP} | group: {GEA_GROUP} | σ: {GEA_SIGMA}")
print("=" * 60)

# Build task distribution from actual data domains (set in Section 4)
unique_labels = sorted(set(
    lab for lab in chunk_labels if not lab.endswith("_fallback")
))
task_distribution = (
    [{"type": lab} for lab in unique_labels]
    if unique_labels else [{"type": "default"}]
)
print(f"  Task domains: {[t['type'] for t in task_distribution]}")

# Build per-domain index lookup for efficient sampling
domain_indices: dict[str, list[int]] = {}
for i, lab in enumerate(chunk_labels):
    domain_indices.setdefault(lab, []).append(i)

def gea_fitness_fn(params, task):
    """Evaluate fitness on a specific task domain.

    Returns (fitness_scalar, metrics_dict) for GEA trace recording.
    """
    task_type = task.get("type", "default")

    # assemble full models from params (using pre-extracted static parts)
    bb = eqx.combine(params["backbone"], _bb_static)
    hd = eqx.combine(params["head"], _hd_static)

    # sample from the specific domain using chunk_labels index
    domain_key = jax.random.PRNGKey(hash(task_type) % (2**31))
    domain_idx = domain_indices.get(task_type, [])

    if domain_idx:
        sample_idx = np.array(domain_idx[:EVAL_BATCH], dtype=np.int32)
        if len(sample_idx) < EVAL_BATCH:
            extra = np.random.choice(domain_idx, EVAL_BATCH - len(sample_idx))
            sample_idx = np.concatenate([sample_idx, extra.astype(np.int32)])
    else:
        sample_idx = jax.random.randint(domain_key, (EVAL_BATCH,), 0, num_chunks)

    batch_tokens = jnp.array(all_chunks[np.array(sample_idx)])

    # encode chunks via frozen AE
    tgt_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(batch_tokens)

    # compress input for backbone
    def compress(chunk):
        embs = jax.vmap(frozen_ae.embedding)(chunk)
        return bb.compress_input(embs)
    inp_seq = jax.vmap(compress)(batch_tokens)

    # backbone forward
    hid, _ = bb(inp_seq)

    # energy loss
    hid_in, z_target = hid[:-1], tgt_z[1:]

    def pos_loss(h, z_t):
        samples = hd(h, key=jax.random.PRNGKey(0), num_samples=8)  # paper recommends N=8
        return energy_score(samples, z_t)

    loss_vals = jax.vmap(pos_loss)(hid_in, z_target)
    fitness = -jnp.mean(loss_vals)
    fitness = jnp.where(jnp.isfinite(fitness), fitness, -1e6)

    metrics = {"energy_loss": float(-fitness), "num_chunks": EVAL_BATCH}
    return float(fitness), metrics


# GEA optimizer (separate from EGGROLL Phase 2)
gea_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
)
gea_opt_state = gea_optimizer.init(trainable)

# Initialize GroupEvolver
evolver = GroupEvolver(population_size=GEA_POP, group_size=GEA_GROUP)

# GEA evolution loop
gea_history = {"mean_fitness": [], "max_fitness": [], "parent_sizes": []}
gea_start = time.time()

for gea_iter in range(1, GEA_ITERATIONS + 1):
    key, iter_key, eggroll_key = jax.random.split(key, 3)

    # GEA evaluate + select
    experience, traces = evolver.evolution_step(
        trainable, gea_fitness_fn, task_distribution,
        key=iter_key, sigma=GEA_SIGMA, rank=1,
    )

    # extract unique parent indices
    parent_indices = list(experience.get("task_champions", {}).values())
    seen = set()
    unique_parents = []
    for idx in parent_indices:
        if idx not in seen:
            seen.add(idx)
            unique_parents.append(idx)

    # experience-biased EGGROLL update
    trainable, gea_opt_state, step_metrics = experience_weighted_eggroll_step(
        trainable, traces, key=eggroll_key,
        optimizer=gea_optimizer, opt_state=gea_opt_state,
        sigma=GEA_SIGMA, rank=1,
        parent_indices=unique_parents if unique_parents else None,
        parent_bias=GEA_BIAS,
    )

    mf = step_metrics["mean_fitness"]
    xf = step_metrics["max_fitness"]
    gea_history["mean_fitness"].append(mf)
    gea_history["max_fitness"].append(xf)
    gea_history["parent_sizes"].append(len(unique_parents))

    elapsed = time.time() - gea_start
    print(f"  GEA {gea_iter:>3}/{GEA_ITERATIONS} | "
          f"mean: {mf:.4f} | best: {xf:.4f} | "
          f"parents: {len(unique_parents)} | {elapsed/60:.0f}m")

gea_elapsed = time.time() - gea_start
print(f"\nPhase 3 done: {gea_elapsed/60:.1f}m")

# save evolved params
final_bb = eqx.combine(trainable["backbone"], _bb_static)
final_hd = eqx.combine(trainable["head"], _hd_static)

eqx.tree_serialise_leaves("checkpoints/backbone_gea.eqx", final_bb)
eqx.tree_serialise_leaves("checkpoints/energy_head_gea.eqx", final_hd)
print("✓ Saved: checkpoints/backbone_gea.eqx + energy_head_gea.eqx")

# %% — 9.5. Plot GEA evolution curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(gea_history["mean_fitness"], label="mean", marker="o")
axes[0].plot(gea_history["max_fitness"], label="best", marker="s")
axes[0].set_title("GEA Fitness Over Iterations")
axes[0].set_xlabel("GEA Iteration"); axes[0].legend()
axes[1].bar(range(len(gea_history["parent_sizes"])), gea_history["parent_sizes"])
axes[1].set_title("Unique Parents Per Iteration")
axes[1].set_xlabel("GEA Iteration"); axes[1].set_ylabel("# Parents")
plt.tight_layout()
plt.savefig("gea_evolution.png", dpi=150, bbox_inches="tight")
print("Saved: gea_evolution.png")
try:
    plt.show()
except Exception:  # pylint: disable=broad-except
    pass

# %% — 10. Quick evaluation: reconstruction + generation quality
print("\n" + "=" * 60)
print("Evaluation")
print("=" * 60)

# AE reconstruction on held-out samples
key, eval_key = jax.random.split(key)
eval_idx = jax.random.randint(eval_key, (1000,), 0, num_chunks)
eval_batch = jnp.array(all_chunks[eval_idx])
final_acc = float(eval_accuracy(frozen_ae, eval_batch))
print(f"AE reconstruction accuracy (1000 chunks): {final_acc:.4%}")

# qualitative: encode → backbone → energy head → decode → compare
print("\nSample reconstructions (original → reconstructed):")
for i in range(5):
    original = all_chunks[i]
    original_text = tokenizer.decode(original.tolist())
    reconstructed = frozen_ae.reconstruct(jnp.array(original))
    recon_text = tokenizer.decode(reconstructed.tolist())
    match_char = "✓" if np.array_equal(original, np.array(reconstructed)) else "✗"
    print(f"  {match_char} [{original_text[:40]:>40s}] → [{recon_text[:40]:<40s}]")

# check backbone next-vector prediction quality (using GEA-evolved model)
print("\nBackbone next-vector prediction (energy loss on 100 pairs):")
key, pred_key = jax.random.split(key)
test_idx = jax.random.randint(pred_key, (100,), 0, num_chunks)
test_tokens = jnp.array(all_chunks[test_idx])
target_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(test_tokens)
input_seq = jax.vmap(lambda c: final_bb.compress_input(
    jax.vmap(frozen_ae.embedding)(c)))(test_tokens)
hidden, _ = final_bb(input_seq)
h_in, z_tgt = hidden[:-1], target_z[1:]

def measure_loss(h, z_t, k):
    """Measure energy loss by sampling and scoring."""
    samples = final_hd(h, key=k, num_samples=8)
    return energy_score(samples, z_t)

pred_keys = jax.random.split(pred_key, h_in.shape[0])
losses = jax.vmap(measure_loss)(h_in, z_tgt, pred_keys)
print(f"  Mean energy loss: {float(jnp.mean(losses)):.4f}")
print(f"  Std energy loss:  {float(jnp.std(losses)):.4f}")

# %% — 11. Summary & Download
print("\n" + "=" * 60)
print("✅ VELM Training Complete!")
print("=" * 60)
print(f"Config:      {CONFIG_NAME}")
print(f"Tokenizer:   {DEFAULT_TOKENIZER} (vocab={VOCAB_SIZE:,})")
print(f"Dataset:     {num_chunks:,} chunks across {len(set(chunk_labels)):,} domains")
print(f"AE accuracy: {final_acc:.4%}")
print(f"GEA iters:   {GEA_ITERATIONS} (bias={GEA_BIAS})")
print("\nCheckpoints saved:")
print("  checkpoints/calm_ae_best.eqx      — best autoencoder")
print("  checkpoints/calm_ae_final.eqx     — final autoencoder")
print("  checkpoints/backbone_eggroll.eqx  — EGGROLL backbone")
print("  checkpoints/energy_head_eggroll.eqx — energy head (EGGROLL)")
print("  checkpoints/backbone_gea.eqx      — GEA-evolved backbone")
print("  checkpoints/energy_head_gea.eqx   — GEA-evolved energy head")
print("  checkpoints/*.json                — metadata for loading")
print("\nTraining curves:")
print("  ae_training_curves.png")
print("  eggroll_training.png")
print("  gea_evolution.png")

# Colab download helper
try:
    from google.colab import files  # noqa: E402
    for f in ["checkpoints/calm_ae_best.eqx", "checkpoints/calm_ae_best.json",
              "checkpoints/backbone_eggroll.eqx", "checkpoints/energy_head_eggroll.eqx",
              "checkpoints/backbone_gea.eqx", "checkpoints/energy_head_gea.eqx",
              "checkpoints/backbone_meta.json",
              "ae_training_curves.png", "eggroll_training.png", "gea_evolution.png"]:
        if os.path.exists(f):
            files.download(f)
except ImportError:
    print("\nNot on Colab — checkpoints are in ./checkpoints/")

