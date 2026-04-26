#!/usr/bin/env python3
# %%
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
# ## RUNBOOK: What to run and when
#
# **Fresh start (no checkpoints):** Run ALL cells 0→11 in order.
#
# **After Colab restart (AE already trained):**
# 1. Run cells 0-4 (setup, imports, config, tokenizer, data) — always needed
# 2. Skip cells 5, 6, 6.5 (AE training) — loads from Google Drive automatically
# 3. Run cell 7 (Phase 2 setup) — creates **Mythos-Enhanced RDT** backbone+head
# 4. Run cell 7.5 (teacher vectors) — loads from Drive if cached, else extracts
# 5. Skip cell 7a (gradient training) — disabled, contradicts EGGROLL thesis
# 6. Run cell 7b (EGGROLL) — **Atomic Resumption** supported; run to continue
# 7. Run cells 9+ (GEA, evaluation) — optional
#
# **Mythos-Enhanced RDT Features (v1.1):**
# - **LTI-Stable Injection**: Spectral radius $ρ(A) < 1$ prevents memory explosion.
# - **Loop-Index Embeddings**: Sinusoidal awareness for reasoning depth.
# - **Anchor Injection**: Original latent $e$ injected at every step for stability.
# - **ACT Halting**: Learned halting probabilities (Experimental).
#
# **All artifacts persist to Google Drive automatically:**
# - `VELM_checkpoints/calm_ae_best.eqx` — trained autoencoder
# - `VELM_checkpoints/teacher_vectors.npy` — extracted teacher hidden states
# - `VELM_checkpoints/backbone_stepX.eqx` — EGGROLL-trained backbone (checkpoints)
# - `VELM_checkpoints/energy_head_stepX.eqx` — EGGROLL-trained energy head (checkpoints)
#
# **Phases:**
# 1. Stream & tokenize OpenWebMath with Qwen3.5 tokenizer
# 2. Train CALM autoencoder → >99.9% token reconstruction
# 3. Extract teacher vectors from Qwen3.5-0.8B (cached to Drive)
# 4. EGGROLL progressive unfreezing with **Mythos stability guarantees**
# 5. GEA group evolution across task domains
# 6. Evaluate: energy loss distributions, per-domain, cosine similarity

# %%
# !pip install -q "jax[cuda12]" equinox jaxtyping optax einops tqdm datasets
# !pip install -q "transformers>=5.6.2"

import glob
import json
import os
import shutil
import sys
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

# Detect environment and set up persistent checkpoint directory.
# On Colab: mount Drive once and sync any prior checkpoints to local disk.
# Off Colab: use ./checkpoints/ as the persistent root.
try:
    from google.colab import drive  # noqa: E402

    drive.mount("/content/drive", force_remount=False)
    DRIVE_DIR = "/content/drive/MyDrive/VELM_checkpoints"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    for f in os.listdir(DRIVE_DIR):
        if (
            f.endswith(".eqx") or f.endswith(".json") or f.endswith(".npy")
        ) and not os.path.exists(f"checkpoints/{f}"):
            shutil.copy2(f"{DRIVE_DIR}/{f}", f"checkpoints/{f}")
    print(f"🚀 Google Drive mounted. Persistence enabled at {DRIVE_DIR}")
except ImportError:
    DRIVE_DIR = "checkpoints"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print("⚠ Not on Colab. Persistence using local 'checkpoints/' directory.")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

print(
    f"JAX {jax.__version__} | backend: {jax.default_backend()} | devices: {jax.devices()}"
)


# %% — 1. Clone repo & import VELM

VELM_DIR = "/content/VELM" if os.path.exists("/content") else os.getcwd()
if not os.path.exists(os.path.join(VELM_DIR, "src")):
    os.system(
        f"git clone https://github.com/angrysky56/VELM.git {VELM_DIR} 2>/dev/null"
    )
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

# %%
import optax


def detect_hardware() -> dict:
    """Pick training profile based on available GPU."""
    try:
        dev = jax.devices("gpu")[0]
        kind = dev.device_kind.lower()
        # data point order matters: H100/A100 first, then mid-tier 16GB
        # (T4/L4), then 12 GB consumer (3060/4070), then CPU fallback.
        if "h100" in kind or "a100" in kind:
            return {
                "batch": 256,
                "ae_steps": 100_000,
                "egg_steps": 10_000,
                "pop": 64,
                "chunks": 500_000,
                "name": f"A100/H100 ({kind})",
            }
        if "t4" in kind or "l4" in kind or "v100" in kind:
            return {
                "batch": 64,
                "ae_steps": 150_000,
                "egg_steps": 5_000,
                "pop": 32,
                "chunks": 250_000,
                "name": f"T4-class ({kind})",
            }
        if "3060" in kind or "rtx" in kind or "4060" in kind or "4070" in kind:
            # 12 GB consumer cards: smaller AE batch, same EGGROLL pop
            return {
                "batch": 48,
                "ae_steps": 150_000,
                "egg_steps": 5_000,
                "pop": 32,
                "chunks": 200_000,
                "name": f"3060/4070 ({kind})",
            }
        # unknown GPU — be conservative
        return {
            "batch": 32,
            "ae_steps": 150_000,
            "egg_steps": 5_000,
            "pop": 24,
            "chunks": 200_000,
            "name": f"Unknown GPU ({kind})",
        }
    except Exception:
        return {
            "batch": 32,
            "ae_steps": 10_000,
            "egg_steps": 1_000,
            "pop": 16,
            "chunks": 50_000,
            "name": "CPU (slow!)",
        }


HW = detect_hardware()
CONFIG_NAME = "gpu_12gb_v2"
cfg = CONFIGS[CONFIG_NAME]
cfg["n_loops"] = cfg.get("n_loops", 1)
cfg["use_act"] = cfg.get("use_act", False)
print(f"  Mythos RDT: n_loops={cfg['n_loops']} | use_act={cfg['use_act']}")

K = cfg["chunk_size_k"]
print(f"Hardware: {HW['name']} | Config: {CONFIG_NAME}")
print(f"  AE steps: {HW['ae_steps']:,} | EGGROLL steps: {HW['egg_steps']:,}")
print(f"  Target chunks: {HW['chunks']:,} ({HW['chunks']*K:,} tokens)")


# %% — 3. Load Qwen3.5 tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, trust_remote_code=True)
VOCAB_SIZE = len(tokenizer)
print(f"Tokenizer: {DEFAULT_TOKENIZER}, vocab: {VOCAB_SIZE:,}")

# %%
DATA_CACHE_BIN = f"{DRIVE_DIR}/all_chunks.npy"
DATA_CACHE_LABELS = f"{DRIVE_DIR}/chunk_labels.json"

if os.path.exists(DATA_CACHE_BIN) and os.path.exists(DATA_CACHE_LABELS):
    print("🚀 Loading cached dataset from", DRIVE_DIR)
    all_chunks = np.load(DATA_CACHE_BIN)
    with open(DATA_CACHE_LABELS, "r") as f:
        chunk_labels = json.load(f)
    num_chunks = all_chunks.shape[0]
    print(f"✓ Loaded {num_chunks:,} chunks ({num_chunks*K:,} tokens) from cache.")
else:
    from datasets import load_dataset
    from tqdm import tqdm

    TARGET_CHUNKS = HW["chunks"]
    print(f"📥 Cache not found. Streaming → {TARGET_CHUNKS:,} chunks (K={K})...")

    chunk_buffer: list[np.ndarray] = []
    chunk_labels = []
    total_chunks = 0

    CURRICULUM = [
        {
            "name": "open-web-math/open-web-math",
            "weight": 0.5,
            "label": "math",
            "split": "train",
            "text_field": "text",
        },
        {
            "name": "wikitext",
            "config": "wikitext-103-raw-v1",
            "weight": 0.3,
            "label": "general",
            "split": "train",
            "text_field": "text",
        },
        {
            "name": "roneneldan/TinyStories",
            "weight": 0.2,
            "label": "narrative",
            "split": "train",
            "text_field": "text",
        },
    ]

    for source in CURRICULUM:
        source_target = int(TARGET_CHUNKS * source["weight"])
        source_count = 0
        label = source["label"]
        print(
            f"  Streaming {label}: {source['name']} (target: {source_target:,} chunks)..."
        )
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
        except Exception as e:
            print(f"    ⚠ {label} failed ({e}), generating fallback random chunks")
            fallback_n = source_target - source_count
            if fallback_n > 0:
                rng = np.random.default_rng(42)
                fallback = rng.integers(
                    0, VOCAB_SIZE, size=(fallback_n, K), dtype=np.int32
                )
                chunk_buffer.append(fallback)
                chunk_labels.extend([f"{label}_fallback"] * fallback_n)
                total_chunks += fallback_n

    if total_chunks == 0:
        raise RuntimeError("No training data loaded! Check network and dataset access.")

    # CRITICAL: actually concatenate the buffer (this was missing before)
    all_chunks = np.concatenate(chunk_buffer, axis=0)[:TARGET_CHUNKS]
    chunk_labels = chunk_labels[:TARGET_CHUNKS]
    num_chunks = all_chunks.shape[0]

    label_counts: dict[str, int] = {}
    for lab in chunk_labels:
        label_counts[lab] = label_counts.get(lab, 0) + 1
    print(f"\n✓ {num_chunks:,} chunks × K={K} = {num_chunks * K:,} tokens")
    for lab, cnt in sorted(label_counts.items()):
        print(f"  {lab}: {cnt:,} ({cnt / num_chunks:.0%})")

    print(f"💾 Caching dataset to {DATA_CACHE_BIN}...")
    np.save(DATA_CACHE_BIN, all_chunks)
    with open(DATA_CACHE_LABELS, "w") as f:
        json.dump(chunk_labels, f)
    print("✓ Dataset cached.")

# host→device once: avoid per-step copy in tight training loops
all_chunks_jnp = jnp.asarray(all_chunks)


# %% — 5. Phase 1: Train CALM Autoencoder
HIDDEN_DIM = cfg.get("ae_hidden_dim", 256)
LATENT_DIM = cfg["latent_dim"]  # now 128 (was 64 — too tight for 248K vocab)
FFN_DIM = cfg.get("ae_ffn_intermediate", 512)
KL_CLIP = cfg.get("ae_kl_clip", 0.5)  # paper uses λ_KL=0.5
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
    init_value=0.0,
    peak_value=LR,
    warmup_steps=effective_warmup,
    decay_steps=AE_STEPS,
    end_value=LR * 0.1,
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

    (loss_val, metrics_val), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        mdl
    )
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

# Try restoring checkpoints from Google Drive first (Colab wipes local disk)
if not os.path.exists("checkpoints/calm_ae_best.eqx"):
    try:
        from google.colab import drive  # noqa: E402

        drive.mount("/content/drive", force_remount=False)
        drive_dir = "/content/drive/MyDrive/VELM_checkpoints"
        if os.path.isdir(drive_dir):
            import shutil

            os.makedirs("checkpoints", exist_ok=True)
            for fname in os.listdir(drive_dir):
                shutil.copy2(f"{drive_dir}/{fname}", f"checkpoints/{fname}")
            print("  ✓ Restored checkpoints from Google Drive")
    except (ImportError, FileNotFoundError, OSError):
        pass  # Not on Colab or no Drive checkpoints

SKIP_AE = False
if os.path.exists("checkpoints/calm_ae_best.json") and os.path.exists(
    "checkpoints/calm_ae_best.eqx"
):
    try:
        import json

        with open("checkpoints/calm_ae_best.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("accuracy", 0.0) >= 0.999:
            print(
                f"\n✓ Found fully trained AE checkpoint! (accuracy: {meta['accuracy']:.4%})"
            )
            print("  Skipping Phase 1 training.")
            model = eqx.tree_deserialise_leaves("checkpoints/calm_ae_best.eqx", model)
            SKIP_AE = True
            best_acc = meta["accuracy"]
            history["accuracy"].append(best_acc)
    except Exception as e:
        print("  Could not read checkpoint metadata, training normally:", e)

if SKIP_AE:
    AE_STEPS = 0  # Skip the loop entirely

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
        print(
            f"  step {step:>6}/{AE_STEPS} | recon: {rl:.4f} | kl: {kl:.4f} | "
            f"{step/elapsed:.1f} steps/s | {elapsed/60:.0f}m"
        )

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

            with open("checkpoints/calm_ae_best.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "config": CONFIG_NAME,
                        "vocab_size": VOCAB_SIZE,
                        "tokenizer": DEFAULT_TOKENIZER,
                        "step": step,
                        "accuracy": acc,
                    },
                    f,
                    indent=2,
                )
            print(f"  >> Saved best checkpoint (acc={acc:.4%})")
        if acc > 0.999:
            print(f"  ✅ TARGET >99.9% at step {step}! — stopping early")
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
axes[0].plot(history["recon"])
axes[0].set_title("Recon Loss")
axes[0].set_xlabel("×500 steps")
axes[1].plot(history["kl"])
axes[1].set_title("KL Loss")
axes[1].set_xlabel("×500 steps")
axes[2].plot(history["accuracy"])
axes[2].axhline(y=0.999, color="r", linestyle="--", label="99.9% target")
axes[2].set_title("Reconstruction Acc")
axes[2].set_xlabel("×2000 steps")
axes[2].legend()
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
TARGET_ACC = 0.999  # stop when we hit this

if CONTINUE_AE and best_acc < TARGET_ACC:
    print(f"\nPhase 1.5: Continue AE training — best so far: {best_acc:.4%}")
    print(
        f"  Loading best checkpoint, {CONTINUE_STEPS:,} more steps, peak LR={CONTINUE_LR}"
    )
    print("=" * 60)

    # load best checkpoint
    model = eqx.tree_deserialise_leaves("checkpoints/calm_ae_best.eqx", model)
    print(f"  ✓ Loaded best checkpoint (acc={best_acc:.4%})")

    # fresh optimizer with lower LR and longer warmup
    cont_warmup = min(1000, CONTINUE_STEPS // 10)
    cont_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=CONTINUE_LR,
        warmup_steps=cont_warmup,
        decay_steps=CONTINUE_STEPS,
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
            loss_fn, has_aux=True
        )(mdl)
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
            model, opt_state, batch, step_key
        )

        if step % 500 == 0:
            elapsed = time.time() - cont_start
            rl = float(metrics["recon_loss"])
            kl = float(metrics["kl_loss"])
            cont_history["recon"].append(rl)
            cont_history["kl"].append(kl)
            print(
                f"  step {step:>6}/{CONTINUE_STEPS} | recon: {rl:.4f} | "
                f"kl: {kl:.4f} | {step/elapsed:.1f} steps/s | "
                f"{elapsed/60:.0f}m"
            )

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

                with open("checkpoints/calm_ae_best.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "config": CONFIG_NAME,
                            "vocab_size": VOCAB_SIZE,
                            "tokenizer": DEFAULT_TOKENIZER,
                            "step": 100_000 + step,
                            "accuracy": acc,
                            "phase": "1.5_continuation",
                        },
                        f,
                        indent=2,
                    )
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

# %%
# Re-derive AE dimensions from config (in case Phase 1 cells were skipped)
HIDDEN_DIM = cfg.get("ae_hidden_dim", 256)
LATENT_DIM = cfg["latent_dim"]
FFN_DIM = cfg.get("ae_ffn_intermediate", 512)
KL_CLIP = cfg.get("ae_kl_clip", 0.5)
KL_WEIGHT = cfg.get("ae_kl_weight", 0.001)

POP_SIZE = HW["pop"]
EGGROLL_STEPS = HW["egg_steps"]
HEAD_MAX_STEPS = 3000  # phase-1 head-only steps (hard cap)
USE_EGGROLL_PHASE = True
SIGMA = cfg.get("eggroll_sigma", 0.001)
EGG_LR = cfg.get("eggroll_lr", 3e-4)
EVAL_BATCH = cfg.get("eggroll_eval_batch", 64)
ANTITHETIC = cfg.get("eggroll_antithetic", True)
HC_D = cfg.get("hc_streams", 1)
HC_S = cfg.get("hc_s", 2)

key = jax.random.PRNGKey(123)
k1, k2, key = jax.random.split(key, 3)

backbone = VELMBackbone(
    dim=cfg["hidden_dim"],
    num_heads=cfg["num_heads"],
    num_miras_layers=cfg["miras_layers"],
    num_swa_layers=cfg["swa_layers"],
    ffn_intermediate=cfg["ffn_intermediate"],
    chunk_size=K,
    ae_hidden_dim=HIDDEN_DIM,
    hc_streams=HC_D,
    hc_s=HC_S,
    n_loops=cfg["n_loops"],
    use_act=cfg["use_act"],
    key=k1,
)
head = EnergyHead(
    hidden_dim=cfg["hidden_dim"],
    latent_dim=LATENT_DIM,
    num_blocks=cfg["energy_head_blocks"],
    key=k2,
)

bb_p = sum(x.size for x in jax.tree.leaves(eqx.filter(backbone, eqx.is_array)))
hd_p = sum(x.size for x in jax.tree.leaves(eqx.filter(head, eqx.is_array)))
print(f"Backbone: {bb_p:,} | Head: {hd_p:,} | Total: {bb_p+hd_p:,}")
if HC_D > 1:
    print(f"go-mHC: d={HC_D} streams, s={HC_S} (doubly stochastic routing)")
print(
    f"EGGROLL: pop={POP_SIZE}, σ={SIGMA}, steps={EGGROLL_STEPS:,}"
    f"{', antithetic' if ANTITHETIC else ''}, eval_batch={EVAL_BATCH}"
)

# Load frozen AE — from Phase 1 if it ran in this session, otherwise from disk.
# Cell 2 already synced any Drive checkpoints to ./checkpoints/.
if "model" not in dir() or model is None:
    print("Loading AE from checkpoint (Phase 1 was skipped)...")
    model = CALMAutoencoder(
        vocab_size=VOCAB_SIZE,
        chunk_size=K,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        ffn_intermediate=FFN_DIM,
        kl_weight=KL_WEIGHT,
        kl_clip=KL_CLIP,
        key=jax.random.PRNGKey(42),
    )
    ckpt_path = "checkpoints/calm_ae_best.eqx"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/calm_ae_final.eqx"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No AE checkpoint found in ./checkpoints/. "
            "If you've trained before, ensure DRIVE_DIR is mounted (re-run cell 2). "
            "Otherwise, re-run Phase 1 to train the AE (~2.7h on T4-class)."
        )
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
    print(f"  ✓ Loaded: {ckpt_path}")
frozen_ae = model


# %%
# Extract teacher hidden states (Qwen3.5) and cache them.
# Cell 2 already restored teacher_vectors.npy from Drive if present.

TEACHER_CACHE = "checkpoints/teacher_vectors.npy"
_loaded_teacher = False

if os.path.exists(TEACHER_CACHE):
    all_teacher_vecs = np.load(TEACHER_CACHE)
    TEACHER_DIM = all_teacher_vecs.shape[1]
    print(f"✓ Loaded cached teacher vectors: {all_teacher_vecs.shape}")
    _loaded_teacher = True

if not _loaded_teacher:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM

    print(f"\nExtracting teacher vectors from {DEFAULT_TOKENIZER}...")
    teacher_device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = (
        AutoModelForCausalLM.from_pretrained(
            DEFAULT_TOKENIZER,
            trust_remote_code=True,
            torch_dtype=torch.float16 if teacher_device == "cuda" else torch.float32,
        )
        .to(teacher_device)
        .eval()
    )
    TEACHER_DIM = teacher_model.config.hidden_size
    print(f"  Teacher: dim={TEACHER_DIM} | device={teacher_device}")

    TEACHER_BATCH = 64
    teacher_hiddens = []
    for start in tqdm(range(0, num_chunks, TEACHER_BATCH), desc="  Extracting"):
        end = min(start + TEACHER_BATCH, num_chunks)
        with torch.no_grad():
            input_ids = torch.tensor(
                np.asarray(all_chunks[start:end]),
                dtype=torch.long,
                device=teacher_device,
            )
            outputs = teacher_model(input_ids=input_ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1].float().mean(dim=1)
            teacher_hiddens.append(last_hidden.cpu().numpy())

    all_teacher_vecs = np.concatenate(teacher_hiddens, axis=0)
    nan_count = int(np.sum(~np.isfinite(all_teacher_vecs)))
    if nan_count:
        print(f"  ⚠ {nan_count} non-finite values — replacing with 0")
        all_teacher_vecs = np.nan_to_num(
            all_teacher_vecs, nan=0.0, posinf=0.0, neginf=0.0
        )
    teacher_std = float(np.std(all_teacher_vecs))
    if teacher_std > 0:
        all_teacher_vecs = all_teacher_vecs / teacher_std
        print(f"  Normalized (std was {teacher_std:.2f})")

    os.makedirs("checkpoints", exist_ok=True)
    np.save(TEACHER_CACHE, all_teacher_vecs)
    if DRIVE_DIR != "checkpoints":
        shutil.copy2(TEACHER_CACHE, f"{DRIVE_DIR}/teacher_vectors.npy")
        print(f"  ✓ Saved to Drive: {DRIVE_DIR}/teacher_vectors.npy")
    del teacher_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  ✓ {all_teacher_vecs.shape[0]:,} vectors ({all_teacher_vecs.shape[1]}d)")

# Pack trainable params and split out static (non-array) module structure.
# These are used by the JIT-compiled fitness step in src/training/velm_fitness.py.
trainable = {
    "backbone": eqx.filter(backbone, eqx.is_array),
    "head": eqx.filter(head, eqx.is_array),
}
_bb_static = eqx.filter(backbone, lambda x: not eqx.is_array(x))
_hd_static = eqx.filter(head, lambda x: not eqx.is_array(x))

# Teacher projection (only used if Phase 2a gradient distillation is enabled).
from src.training.distillation import TeacherProjection  # noqa: E402

k_proj, key = jax.random.split(key)
teacher_proj = TeacherProjection(TEACHER_DIM, cfg["hidden_dim"], key=k_proj)
_tp_static = eqx.filter(teacher_proj, lambda x: not eqx.is_array(x))


# %%
# Phase 2a (gradient + teacher distillation) is intentionally DISABLED.
#
# The project's thesis is gradient-free training via EGGROLL — using
# backprop where backprop works contradicts that goal. The code path is
# preserved in src/training/distillation.py for ablation studies, but
# the notebook flow goes straight from Phase 1 (AE) → Phase 2 (EGGROLL).
USE_GRADIENT_PHASE = False
print("Phase 2a (gradient distillation): SKIPPED — proceeding to EGGROLL.")


# %%
# Phase 2: EGGROLL training with progressive head-only → full unfreeze.
#
# Uses the JIT-compiled antithetic ES step from src/training/velm_fitness.
# Three correctness fixes vs the previous hand-rolled loop:
#   1. Per-step random keys for the energy head (was constant PRNGKey(0))
#   2. Frozen-AE encoding done once per step, not once per member
#   3. jax.lax.map over population (single JIT compile, GPU-resident)

from src.training.diagnostics import EGGROLLDiagnostics
from src.training.eggroll import SigmaAdaptor
from src.training.velm_fitness import make_velm_eggroll_step

diag = EGGROLLDiagnostics(plateau_window=50, plateau_threshold=0.001)
adaptor = SigmaAdaptor(initial_sigma=SIGMA, target_diversity=0.02, max_sigma=0.01)

# History buffers — fed into the plot cell and convergence checks.
egg_history = {
    "mean_fitness": [],
    "max_fitness": [],
    "fitness_std": [],
    "grad_norm": [],
    "phase": [],
    "sigma": [],
    "diversity": [],
    "step": [],
}

# Two compiled step functions: head-only (rank=1, fast warmup) and full
# (rank=2, lower LR). Each compiles once and is reused for the whole phase.
head_opt = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adamw(EGG_LR, weight_decay=0.01)
)
head_opt_state = head_opt.init({"head": trainable["head"]})
step_head = make_velm_eggroll_step(
    frozen_ae,
    _bb_static,
    _hd_static,
    head_opt,
    pop_size=POP_SIZE,
    rank=1,
    num_samples=8,
    perturb_head_only=True,
)

full_opt = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adamw(EGG_LR * 0.3, weight_decay=0.01)
)
full_opt_state = full_opt.init(trainable)
step_full = make_velm_eggroll_step(
    frozen_ae,
    _bb_static,
    _hd_static,
    full_opt,
    pop_size=POP_SIZE,
    rank=2,
    num_samples=8,
    perturb_head_only=False,
)

# Resume from latest checkpoint if any exist locally.
RESUME_STEP = 0
existing = sorted(
    glob.glob("checkpoints/backbone_step*.eqx"),
    key=lambda p: int(p.rsplit("step", 1)[1].rsplit(".", 1)[0]),
)
if existing:
    latest = existing[-1]
    RESUME_STEP = int(latest.rsplit("step", 1)[1].rsplit(".", 1)[0])
    print(f"🔄 Resuming EGGROLL from step {RESUME_STEP} ({latest})")
    try:
        trainable["backbone"] = eqx.tree_deserialise_leaves(
            latest, trainable["backbone"]
        )
    except RuntimeError:
        _tmp = eqx.tree_deserialise_leaves(
            latest, eqx.combine(trainable["backbone"], _bb_static)
        )
        trainable["backbone"] = eqx.filter(_tmp, eqx.is_array)
        print("  (loaded old-format combined checkpoint)")
    hd_ckpt = latest.replace("backbone", "head")
    if os.path.exists(hd_ckpt):
        try:
            trainable["head"] = eqx.tree_deserialise_leaves(hd_ckpt, trainable["head"])
        except RuntimeError:
            _tmp = eqx.tree_deserialise_leaves(
                hd_ckpt, eqx.combine(trainable["head"], _hd_static)
            )
            trainable["head"] = eqx.filter(_tmp, eqx.is_array)
    # re-init opt states after loading params
    head_opt_state = head_opt.init({"head": trainable["head"]})
    full_opt_state = full_opt.init(trainable)

UNFREEZE_PLATEAU = 0.01
CONVERGENCE_PLATEAU = 0.005
UNFREEZE_WINDOW = 10
CONVERGENCE_WINDOW = 20

# Stability gate: did head-only training already plateau?
already_unfrozen = RESUME_STEP >= HEAD_MAX_STEPS

start = time.time()
print(
    f"🚀 EGGROLL: starting at step {RESUME_STEP+1}/{EGGROLL_STEPS}, "
    f"target diversity={adaptor.target_diversity}"
)

for step in range(RESUME_STEP + 1, EGGROLL_STEPS + 1):
    key, batch_key, step_key = jax.random.split(key, 3)

    # contiguous slice → backbone sees sequential context
    start_idx = int(
        jax.random.randint(batch_key, (), 0, max(1, num_chunks - EVAL_BATCH))
    )
    batch = all_chunks_jnp[start_idx : start_idx + EVAL_BATCH]

    # phase decision: head-only until either hard cap or stability gate
    in_head_phase = (not already_unfrozen) and (step <= HEAD_MAX_STEPS)
    sigma_jax = jnp.asarray(adaptor.sigma, dtype=jnp.float32)

    if in_head_phase:
        trainable, head_opt_state, m = step_head(
            trainable,
            head_opt_state,
            batch,
            step_key,
            sigma_jax,
        )
        phase_label = "head"
    else:
        if not already_unfrozen:
            already_unfrozen = True
            full_opt_state = full_opt.init(trainable)  # fresh state on transition
            print(f"\n  >>> Unfreezing backbone at step {step} <<<\n")
        trainable, full_opt_state, m = step_full(
            trainable,
            full_opt_state,
            batch,
            step_key,
            sigma_jax,
        )
        phase_label = "full"

    # cheap metrics: read out floats only every 100 steps to avoid syncs
    if step % 100 == 0:
        mf = float(m["mean_fitness"])
        xf = float(m["max_fitness"])
        fs = float(m["fitness_std"])
        gn = float(m["grad_norm"])

        # NaN guard: stop if fitness went non-finite
        if not (np.isfinite(mf) and np.isfinite(gn)):
            print(f"\n  ⚠ NaN/Inf detected at step {step} — stopping EGGROLL")
            break

        entry = diag.log_step(
            step,
            trainable,
            {
                "mean_fitness": mf,
                "max_fitness": xf,
                "fitness_std": fs,
                "grad_norm": gn,
            },
        )
        adaptor.update(entry["diversity"])
        egg_history["mean_fitness"].append(mf)
        egg_history["max_fitness"].append(xf)
        egg_history["fitness_std"].append(fs)
        egg_history["grad_norm"].append(gn)
        egg_history["phase"].append(phase_label)
        egg_history["sigma"].append(adaptor.sigma)
        egg_history["diversity"].append(entry["diversity"])
        egg_history["step"].append(step)
        elapsed = time.time() - start
        rate = (step - RESUME_STEP) / elapsed if elapsed > 0 else 0.0
        print(
            f"  step {step:>5} [{phase_label}] | mean_f: {mf:+.4f} | "
            f"max_f: {xf:+.4f} | grad: {gn:.2f} | σ: {adaptor.sigma:.4f} | "
            f"div: {entry['diversity']:.4f} | {rate:.1f} st/s | "
            f"{elapsed/60:.0f}m"
        )

        # stability-gated auto-unfreeze
        if in_head_phase and len(egg_history["mean_fitness"]) >= UNFREEZE_WINDOW:
            recent = [
                v for v in egg_history["mean_fitness"][-UNFREEZE_WINDOW:] if v == v
            ]  # filter NaN
            if len(recent) == UNFREEZE_WINDOW:
                rel = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-8)
                if rel < UNFREEZE_PLATEAU:
                    print(
                        f"\n  🔓 head-only fitness plateaued (Δ={rel:.4f}) — unfreezing now\n"
                    )
                    already_unfrozen = True
                    full_opt_state = full_opt.init(trainable)

        # convergence detection in full phase
        if not in_head_phase and len(egg_history["mean_fitness"]) >= CONVERGENCE_WINDOW:
            recent = [
                v for v in egg_history["mean_fitness"][-CONVERGENCE_WINDOW:] if v == v
            ]
            if len(recent) >= CONVERGENCE_WINDOW // 2:
                rel = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-8)
                if rel < CONVERGENCE_PLATEAU:
                    print(f"\n  🎯 full-model converged (Δ={rel:.4f}) — stopping early")
                    break

        # divergence guard
        if len(egg_history["mean_fitness"]) >= 5:
            tail = [v for v in egg_history["mean_fitness"][-5:] if v == v]
            if len(tail) >= 2 and tail[-1] < min(tail[:-1]) * 10:
                print(f"\n  ⚠ DIVERGENCE at step {step} — stopping EGGROLL")
                break

    # checkpoint every 500 steps (save filtered arrays — must match resume tree)
    if step % 500 == 0:
        bb_ckpt = f"checkpoints/backbone_step{step}.eqx"
        hd_ckpt = f"checkpoints/head_step{step}.eqx"
        eqx.tree_serialise_leaves(bb_ckpt, trainable["backbone"])
        eqx.tree_serialise_leaves(hd_ckpt, trainable["head"])
        if DRIVE_DIR != "checkpoints":
            shutil.copy2(bb_ckpt, f"{DRIVE_DIR}/backbone_step{step}.eqx")
            shutil.copy2(hd_ckpt, f"{DRIVE_DIR}/head_step{step}.eqx")

elapsed = time.time() - start
print(f"\nPhase 2 done: {elapsed/3600:.2f}h")

# Final save of best params (filtered arrays — matches resume tree structure)
final_bb = eqx.combine(trainable["backbone"], _bb_static)
final_hd = eqx.combine(trainable["head"], _hd_static)
eqx.tree_serialise_leaves("checkpoints/backbone_eggroll.eqx", trainable["backbone"])
eqx.tree_serialise_leaves("checkpoints/energy_head_eggroll.eqx", trainable["head"])
if DRIVE_DIR != "checkpoints":
    shutil.copy2(
        "checkpoints/backbone_eggroll.eqx", f"{DRIVE_DIR}/backbone_eggroll.eqx"
    )
    shutil.copy2(
        "checkpoints/energy_head_eggroll.eqx", f"{DRIVE_DIR}/energy_head_eggroll.eqx"
    )

with open("checkpoints/backbone_meta.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "config": CONFIG_NAME,
            "vocab_size": VOCAB_SIZE,
            "tokenizer": DEFAULT_TOKENIZER,
            "eggroll_steps": EGGROLL_STEPS,
            "pop_size": POP_SIZE,
            "sigma": SIGMA,
            "history": egg_history,
        },
        f,
        indent=2,
    )
print("✓ Saved: checkpoints/backbone_eggroll.eqx + energy_head_eggroll.eqx + meta.json")


# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

mf = egg_history["mean_fitness"]
xf = egg_history["max_fitness"]
phases = egg_history.get("phase", [])

if mf:
    x = list(range(len(mf)))
    head_idx = [i for i, p in enumerate(phases) if p == "head"]
    full_idx = [i for i, p in enumerate(phases) if p == "full"]
    if head_idx:
        axes[0, 0].plot(
            head_idx, [mf[i] for i in head_idx], "b-", label="head only", alpha=0.85
        )
    if full_idx:
        axes[0, 0].plot(
            full_idx, [mf[i] for i in full_idx], "r-", label="full model", alpha=0.85
        )
    axes[0, 0].plot(x, xf, "g--", label="max", alpha=0.5)
    if head_idx and full_idx:
        axes[0, 0].axvline(x=full_idx[0], color="k", linestyle=":", label="unfreeze")
    axes[0, 0].legend(fontsize=8)
else:
    axes[0, 0].text(0.5, 0.5, "No data", ha="center", va="center")
axes[0, 0].set_title("EGGROLL Fitness (higher=better)")
axes[0, 0].set_xlabel("×100 steps")

# Energy loss = -mean_fitness
if mf:
    valid = [(i, -v) for i, v in enumerate(mf) if v == v]
    if valid:
        axes[0, 1].plot([v[0] for v in valid], [v[1] for v in valid], "b-")
axes[0, 1].set_title("Energy Loss (↓ better)")
axes[0, 1].set_xlabel("×100 steps")

gn = egg_history["grad_norm"]
if gn:
    axes[1, 0].semilogy(gn)
axes[1, 0].set_title("ES Grad Norm (log scale)")
axes[1, 0].set_xlabel("×100 steps")

# Sigma + diversity over time
sg = egg_history.get("sigma", [])
dv = egg_history.get("diversity", [])
if sg:
    ax = axes[1, 1]
    ax.plot(sg, "b-", label="σ")
    ax.set_ylabel("σ", color="b")
    ax2 = ax.twinx()
    ax2.plot(dv, "r-", label="diversity")
    ax2.set_ylabel("diversity", color="r")
axes[1, 1].set_title("Adaptive σ vs population diversity")
axes[1, 1].set_xlabel("×100 steps")

plt.tight_layout()
plt.savefig("eggroll_training.png", dpi=150, bbox_inches="tight")
print("Saved: eggroll_training.png")
try:
    plt.show()
except Exception:
    pass
plt.close(fig)


# %% — 8.5. EGGROLL Diagnostics Report
print("Diagnostics were tracked and logged during the EGGROLL training loop.")

try:
    diag.report()
    if diag.warnings:
        print("\n⚠ Active warnings:")
        for w in diag.warnings:
            print(f"  - {w}")
except NameError:
    print("Diagnostics not available. Please run the EGGROLL training cell first.")

# %% [markdown]
# ### 🔄 Resume Checkpoint (Run this to skip Phase 2)
# Run this cell after restarting Colab to load the Phase 2 EGGROLL checkpoints into memory. This allows you to skip the 15-hour training and proceed directly to Phase 3 or Evaluation.

import os

# %%
import equinox as eqx

bb_path = "checkpoints/backbone_eggroll.eqx"
hd_path = "checkpoints/energy_head_eggroll.eqx"

# Check if we need to copy from Drive first
if not os.path.exists(bb_path):
    try:
        import shutil

        drive_dir = "/content/drive/MyDrive/VELM_checkpoints"
        os.makedirs("checkpoints", exist_ok=True)
        if os.path.exists(f"{drive_dir}/backbone_eggroll.eqx"):
            shutil.copy2(f"{drive_dir}/backbone_eggroll.eqx", bb_path)
            shutil.copy2(f"{drive_dir}/energy_head_eggroll.eqx", hd_path)
            print("Restored EGGROLL checkpoints from Google Drive.")
    except Exception:
        pass

if os.path.exists(bb_path) and os.path.exists(hd_path):
    print("Loading EGGROLL trained parameters into memory...")
    try:
        trainable["backbone"] = eqx.tree_deserialise_leaves(
            bb_path, trainable["backbone"]
        )
    except RuntimeError:
        print("  Old-format checkpoint detected, loading via full module...")
        _tmp_bb = eqx.tree_deserialise_leaves(
            bb_path, eqx.combine(trainable["backbone"], _bb_static)
        )
        trainable["backbone"] = eqx.filter(_tmp_bb, eqx.is_array)
    try:
        trainable["head"] = eqx.tree_deserialise_leaves(hd_path, trainable["head"])
    except RuntimeError:
        _tmp_hd = eqx.tree_deserialise_leaves(
            hd_path, eqx.combine(trainable["head"], _hd_static)
        )
        trainable["head"] = eqx.filter(_tmp_hd, eqx.is_array)

    # Prepare final_bb and final_hd so Evaluation works even if Phase 3 is skipped
    final_bb = eqx.combine(trainable["backbone"], _bb_static)
    final_hd = eqx.combine(trainable["head"], _hd_static)
    print(
        "\u2713 Successfully loaded 15-hour EGGROLL checkpoints! You can now run Evaluation."
    )
else:
    print(
        "\u26a0 EGGROLL checkpoints not found locally or in Drive. You may need to train first."
    )

# %%
from src.evolution.gea_eggroll import (  # noqa: E402
    GroupEvolver,
    experience_weighted_eggroll_step,
)
from src.training.velm_fitness import make_velm_fitness_eval  # noqa: E402

GEA_ITERATIONS = 10
GEA_POP = min(HW["pop"], 32)
GEA_GROUP = 5
GEA_SIGMA = SIGMA
GEA_BIAS = 0.3

print(f"\nPhase 3: GEA Group Evolution — {GEA_ITERATIONS} iterations")
print(f"  population: {GEA_POP} | group: {GEA_GROUP} | σ: {GEA_SIGMA}")
print("=" * 60)

# Task domains derived from data labels (excluding fallbacks)
unique_labels = sorted(
    set(lab for lab in chunk_labels if not lab.endswith("_fallback"))
)
task_distribution = (
    [{"type": lab} for lab in unique_labels] if unique_labels else [{"type": "default"}]
)
print(f"  Task domains: {[t['type'] for t in task_distribution]}")

# Per-domain index lookup
domain_indices: dict[str, list[int]] = {}
for i, lab in enumerate(chunk_labels):
    domain_indices.setdefault(lab, []).append(i)

# JIT-compiled evaluator (replaces the slow non-JIT in-cell version)
gea_eval = make_velm_fitness_eval(frozen_ae, _bb_static, _hd_static, num_samples=8)


def gea_fitness_fn(params, task):
    """Evaluate one perturbed candidate on one task domain."""
    task_type = task.get("type", "default")
    domain_idx = domain_indices.get(task_type, [])
    if domain_idx:
        # random sample from domain (was deterministic first-N, causing bias)
        sample_idx = np.random.choice(
            domain_idx,
            size=min(EVAL_BATCH, len(domain_idx)),
            replace=False,
        ).astype(np.int32)
        if len(sample_idx) < EVAL_BATCH:
            extra = np.random.choice(domain_idx, EVAL_BATCH - len(sample_idx))
            sample_idx = np.concatenate([sample_idx, extra.astype(np.int32)])
    else:
        sample_idx = np.random.choice(num_chunks, EVAL_BATCH, replace=False).astype(
            np.int32
        )

    batch = all_chunks_jnp[sample_idx]
    # real per-call key — was hard-coded to PRNGKey(0) before
    eval_key = jax.random.PRNGKey(hash((task_type, int(sample_idx.sum()))) & 0x7FFFFFFF)
    fitness = float(gea_eval(params, batch, eval_key))
    return fitness, {"energy_loss": -fitness, "num_chunks": EVAL_BATCH}


gea_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adamw(1e-3, weight_decay=0.01)
)
gea_opt_state = gea_optimizer.init(trainable)
evolver = GroupEvolver(population_size=GEA_POP, group_size=GEA_GROUP)

gea_history = {"mean_fitness": [], "max_fitness": [], "parent_sizes": []}
gea_start = time.time()

for gea_iter in range(1, GEA_ITERATIONS + 1):
    key, iter_key, eggroll_key = jax.random.split(key, 3)

    experience, traces = evolver.evolution_step(
        trainable,
        gea_fitness_fn,
        task_distribution,
        key=iter_key,
        sigma=GEA_SIGMA,
        rank=1,
    )

    parent_indices = list(experience.get("task_champions", {}).values())
    seen = set()
    unique_parents = [i for i in parent_indices if not (i in seen or seen.add(i))]

    trainable, gea_opt_state, step_metrics = experience_weighted_eggroll_step(
        trainable,
        traces,
        key=eggroll_key,
        optimizer=gea_optimizer,
        opt_state=gea_opt_state,
        sigma=GEA_SIGMA,
        rank=1,
        parent_indices=unique_parents if unique_parents else None,
        parent_bias=GEA_BIAS,
    )

    mf = step_metrics["mean_fitness"]
    xf = step_metrics["max_fitness"]
    gea_history["mean_fitness"].append(mf)
    gea_history["max_fitness"].append(xf)
    gea_history["parent_sizes"].append(len(unique_parents))
    elapsed = time.time() - gea_start
    print(
        f"  GEA {gea_iter:>3}/{GEA_ITERATIONS} | mean: {mf:.4f} | "
        f"best: {xf:.4f} | parents: {len(unique_parents)} | {elapsed/60:.0f}m"
    )

print(f"\nPhase 3 done: {(time.time()-gea_start)/60:.1f}m")

final_bb = eqx.combine(trainable["backbone"], _bb_static)
final_hd = eqx.combine(trainable["head"], _hd_static)
eqx.tree_serialise_leaves("checkpoints/backbone_gea.eqx", trainable["backbone"])
eqx.tree_serialise_leaves("checkpoints/energy_head_gea.eqx", trainable["head"])
if DRIVE_DIR != "checkpoints":
    shutil.copy2("checkpoints/backbone_gea.eqx", f"{DRIVE_DIR}/backbone_gea.eqx")
    shutil.copy2("checkpoints/energy_head_gea.eqx", f"{DRIVE_DIR}/energy_head_gea.eqx")
print("✓ Saved: backbone_gea.eqx + energy_head_gea.eqx")


# %% — 9.5. Plot GEA evolution curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(gea_history["mean_fitness"], label="mean", marker="o")
axes[0].plot(gea_history["max_fitness"], label="best", marker="s")
axes[0].set_title("GEA Fitness Over Iterations")
axes[0].set_xlabel("GEA Iteration")
axes[0].legend()
axes[1].bar(range(len(gea_history["parent_sizes"])), gea_history["parent_sizes"])
axes[1].set_title("Unique Parents Per Iteration")
axes[1].set_xlabel("GEA Iteration")
axes[1].set_ylabel("# Parents")
plt.tight_layout()
plt.savefig("gea_evolution.png", dpi=150, bbox_inches="tight")
print("Saved: gea_evolution.png")
try:
    plt.show()
except Exception:  # pylint: disable=broad-except
    pass
plt.close(fig)

# %%
print("\n" + "=" * 60)
print("Evaluation")
print("=" * 60)

# AE reconstruction on held-out samples
key, eval_key = jax.random.split(key)
eval_idx = jax.random.randint(eval_key, (1000,), 0, num_chunks)
eval_batch = all_chunks_jnp[eval_idx]
final_acc = float(eval_accuracy(frozen_ae, eval_batch))
print(f"AE reconstruction accuracy (1000 chunks): {final_acc:.4%}")

print("\nSample reconstructions (original → reconstructed):")
for i in range(5):
    original = all_chunks[i]
    original_text = tokenizer.decode(original.tolist())
    reconstructed = frozen_ae.reconstruct(jnp.array(original))
    recon_text = tokenizer.decode(reconstructed.tolist())
    match_char = "✓" if np.array_equal(original, np.array(reconstructed)) else "✗"
    print(f"  {match_char} [{original_text[:40]:>40s}] → [{recon_text[:40]:<40s}]")

# Backbone next-vector prediction: energy loss DISTRIBUTION
print("\n— Energy Loss Distribution (500 pairs) —")
key, pred_key = jax.random.split(key)
N_EVAL = 500
test_idx = jax.random.randint(pred_key, (N_EVAL,), 0, num_chunks)
test_tokens = all_chunks_jnp[test_idx]
target_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(test_tokens)
input_seq = jax.vmap(
    lambda c: final_bb.compress_input(jax.vmap(frozen_ae.embedding)(c))
)(test_tokens)
hidden, _ = final_bb(input_seq)
h_in, z_tgt = hidden[:-1], target_z[1:]


def measure_loss(h, z_t, k):
    samples = final_hd(h, key=k, num_samples=8)
    return energy_score(samples, z_t)


pred_keys = jax.random.split(pred_key, h_in.shape[0])
losses = jax.vmap(measure_loss)(h_in, z_tgt, pred_keys)
losses_np = np.array(losses)
finite_losses = losses_np[np.isfinite(losses_np)]

print(
    f"  Total pairs: {len(losses_np)} | Finite: {len(finite_losses)} | "
    f"NaN/Inf: {len(losses_np) - len(finite_losses)}"
)
if len(finite_losses) > 0:
    pcts = np.percentile(finite_losses, [5, 25, 50, 75, 95])
    print(f"  Mean:   {np.mean(finite_losses):.4f}")
    print(f"  Median: {pcts[2]:.4f}")
    print(
        f"  P5/P25/P75/P95: {pcts[0]:.4f} / {pcts[1]:.4f} / {pcts[3]:.4f} / {pcts[4]:.4f}"
    )
    print(f"  Std:    {np.std(finite_losses):.4f}")

# Per-domain breakdown — FIX: was [:len(losses_np)+1] → off-by-one
print("\n— Per-Domain Energy Loss —")
test_labels = [chunk_labels[int(idx)] for idx in test_idx[: len(losses_np)]]
domain_losses: dict[str, list] = {}
# losses array maps to pairs (i, i+1); we use shifted labels so loss_i pairs to label_{i+1}
shift_labels = test_labels[1:] + ["unknown"]
for i, loss_val in enumerate(losses_np):
    if not np.isfinite(loss_val):
        continue
    lab = shift_labels[i] if i < len(shift_labels) else "unknown"
    domain_losses.setdefault(lab, []).append(float(loss_val))
for lab in sorted(domain_losses):
    vals = domain_losses[lab]
    print(
        f"  {lab:>12s}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} n={len(vals)}"
    )

# Cosine similarity between predictions and targets
print("\n— Prediction-Target Cosine Similarity —")


def cos_sim(h, z_t, k):
    pred = final_hd.predict(h, key=k)
    p_n = pred / (jnp.linalg.norm(pred) + 1e-8)
    t_n = z_t / (jnp.linalg.norm(z_t) + 1e-8)
    return jnp.sum(p_n * t_n)


sims = jax.vmap(cos_sim)(h_in, z_tgt, pred_keys)
sims_np = np.array(sims)
finite_sims = sims_np[np.isfinite(sims_np)]
if len(finite_sims) > 0:
    pcts_s = np.percentile(finite_sims, [5, 25, 50, 75, 95])
    print(f"  Mean sim:   {np.mean(finite_sims):.4f} (1.0=perfect, 0=random)")
    print(f"  Median sim: {pcts_s[2]:.4f}")
    print(f"  P5/P95:     {pcts_s[0]:.4f} / {pcts_s[4]:.4f}")

# Miras memory state norm sanity check
print("\n— Miras Memory State Norms —")
for i, block in enumerate(final_bb.miras_blocks):
    cap = float(block.miras.frobenius_capacity)
    print(f"  Layer {i}: frobenius_capacity = {cap:.1f}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
if len(finite_losses) > 0:
    axes[0].hist(finite_losses, bins=50, alpha=0.7, color="steelblue")
    axes[0].axvline(np.median(finite_losses), color="r", linestyle="--", label="median")
    axes[0].legend()
axes[0].set_title("Energy Loss Distribution")
axes[0].set_xlabel("Energy Score")

if len(finite_sims) > 0:
    axes[1].hist(finite_sims, bins=50, alpha=0.7, color="seagreen")
    axes[1].axvline(0, color="gray", linestyle=":", alpha=0.5)
    axes[1].axvline(np.mean(finite_sims), color="r", linestyle="--", label="mean")
    axes[1].legend()
axes[1].set_title("Pred-Target Cosine Similarity")
axes[1].set_xlabel("Cosine Similarity")

if domain_losses:
    labels = sorted(domain_losses.keys())
    means = [np.mean(domain_losses[l]) for l in labels]
    stds = [np.std(domain_losses[l]) for l in labels]
    axes[2].bar(labels, means, yerr=stds, capsize=5, alpha=0.7, color="coral")
axes[2].set_title("Energy Loss by Domain")
axes[2].set_ylabel("Mean Energy Loss")

plt.tight_layout()
plt.savefig("evaluation_distributions.png", dpi=150, bbox_inches="tight")
print("\nSaved: evaluation_distributions.png")
try:
    plt.show()
except Exception:
    pass
plt.close(fig)


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
print("  checkpoints/backbone_eggroll.eqx   — EGGROLL backbone")
print("  checkpoints/energy_head_eggroll.eqx — energy head (EGGROLL)")
print("  checkpoints/backbone_gea.eqx       — GEA-evolved backbone")
print("  checkpoints/energy_head_gea.eqx    — GEA-evolved energy head")
print("  checkpoints/*.json                 — metadata for loading")
print("\nTraining curves:")
print("  ae_training_curves.png")
print("  eggroll_training.png")
print("  gea_evolution.png")
print("  evaluation_distributions.png")

# Colab download helper
try:
    from google.colab import files  # noqa: E402

    for f in [
        "checkpoints/calm_ae_best.eqx",
        "checkpoints/calm_ae_best.json",
        "checkpoints/backbone_eggroll.eqx",
        "checkpoints/energy_head_eggroll.eqx",
        "checkpoints/backbone_gea.eqx",
        "checkpoints/energy_head_gea.eqx",
        "checkpoints/backbone_meta.json",
        "ae_training_curves.png",
        "eggroll_training.png",
        "gea_evolution.png",
        "evaluation_distributions.png",
    ]:
        if os.path.exists(f):
            files.download(f)
except ImportError:
    print("\nNot on Colab — checkpoints are in ./checkpoints/")
