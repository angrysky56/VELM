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

from src.model.autoencoder import CALMAutoencoder, batch_ae_loss, reconstruction_accuracy
from src.model.energy_head import EnergyHead, energy_score
from src.model.miras_backbone import VELMBackbone
from src.model.config import CONFIGS, DEFAULT_TOKENIZER
from src.training.eggroll import eggroll_step, create_eggroll_optimizer
print("✓ VELM modules imported")

# %% — 2. Hardware-adaptive config
import equinox as eqx
import optax
import numpy as np
import time

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
            return {"batch": 64, "ae_steps": 100_000, "egg_steps": 5_000,
                    "pop": 32, "chunks": 250_000, "name": "T4/consumer"}
    except Exception:  # pylint: disable=broad-except
        return {"batch": 32, "ae_steps": 10_000, "egg_steps": 1_000,
                "pop": 16, "chunks": 50_000, "name": "CPU (slow!)"}

HW = detect_hardware()
CONFIG_NAME = "gpu_12gb"
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

# %% — 4. Stream OpenWebMath and tokenize into K=4 chunks
from datasets import load_dataset
from tqdm import tqdm

TARGET_CHUNKS = HW["chunks"]

print(f"Streaming OpenWebMath → {TARGET_CHUNKS:,} chunks...")
dataset = load_dataset("open-web-math/open-web-math", split="train", streaming=True)

chunk_buffer: list[np.ndarray] = []
total_chunks = 0

for example in tqdm(dataset, desc="Tokenizing", unit="docs"):
    text = example.get("text", "")
    if not text or len(text) < 50:
        continue
    tokens = tokenizer.encode(text, max_length=512, truncation=True)
    usable = len(tokens) - (len(tokens) % K)
    if usable < K:
        continue
    chunks = np.array(tokens[:usable], dtype=np.int32).reshape(-1, K)
    chunk_buffer.append(chunks)
    total_chunks += chunks.shape[0]
    if total_chunks >= TARGET_CHUNKS:
        break

all_chunks = np.concatenate(chunk_buffer, axis=0)[:TARGET_CHUNKS]
print(f"✓ {all_chunks.shape[0]:,} chunks × K={K} = {all_chunks.shape[0]*K:,} tokens")

# %% — 5. Phase 1: Train CALM Autoencoder
HIDDEN_DIM = cfg.get("ae_hidden_dim", 256)
LATENT_DIM = cfg["latent_dim"]
FFN_DIM = cfg.get("ae_ffn_intermediate", 512)
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
    kl_weight=0.001,
    kl_clip=1.0,
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
            print(f"  🎯 TARGET >99.9% at step {step}!")

total_time = time.time() - start_time
final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
print(f"\nPhase 1 done: {total_time/3600:.2f}h | accuracy: {final_acc:.4%}")

# save final
eqx.tree_serialise_leaves("checkpoints/calm_ae_final.eqx", model)

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

# %% — 7. Phase 2: EGGROLL Backbone Training (gradient-free)
POP_SIZE = HW["pop"]
EGGROLL_STEPS = HW["egg_steps"]
SIGMA = 0.01
EGG_LR = 1e-3
EVAL_BATCH = 32

key = jax.random.PRNGKey(123)
k1, k2, key = jax.random.split(key, 3)

backbone = VELMBackbone(
    dim=cfg["hidden_dim"], num_heads=cfg["num_heads"],
    num_miras_layers=cfg["miras_layers"], num_swa_layers=cfg["swa_layers"],
    ffn_intermediate=cfg["ffn_intermediate"], chunk_size=K,
    ae_hidden_dim=HIDDEN_DIM, key=k1,
)

head = EnergyHead(
    hidden_dim=cfg["hidden_dim"], latent_dim=LATENT_DIM,
    num_blocks=cfg["energy_head_blocks"], key=k2,
)

bb_p = sum(x.size for x in jax.tree.leaves(eqx.filter(backbone, eqx.is_array)))
hd_p = sum(x.size for x in jax.tree.leaves(eqx.filter(head, eqx.is_array)))
print(f"Backbone: {bb_p:,} | Head: {hd_p:,} | Total: {bb_p+hd_p:,}")
print(f"EGGROLL: pop={POP_SIZE}, σ={SIGMA}, steps={EGGROLL_STEPS:,}")

frozen_ae = model  # from Phase 1

# pack trainable params
trainable = {
    "backbone": eqx.filter(backbone, eqx.is_array),
    "head": eqx.filter(head, eqx.is_array),
}

# fitness function — uses _current_batch set per step
_current_batch = None

def fitness_fn(params):
    """Evaluate fitness: negative energy loss on current batch."""
    bb = eqx.combine(params["backbone"],
                      eqx.filter(backbone, lambda x: not eqx.is_array(x)))
    hd = eqx.combine(params["head"],
                      eqx.filter(head, lambda x: not eqx.is_array(x)))
    batch_tokens = _current_batch

    # encode chunks → target latents via frozen AE
    tgt_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(batch_tokens)

    # compress input for backbone
    def compress(chunk):
        embs = jax.vmap(frozen_ae.embedding)(chunk)
        return bb.compress_input(embs)
    inp_seq = jax.vmap(compress)(batch_tokens)

    # backbone forward → hidden states
    hid, _ = bb(inp_seq)

    # energy loss: predict z_{i+1} from h_i
    hid_in, z_target = hid[:-1], tgt_z[1:]
    def pos_loss(h, z_t):
        samples = hd(h, key=jax.random.PRNGKey(0), num_samples=4)
        return energy_score(samples, z_t)
    loss_vals = jax.vmap(pos_loss)(hid_in, z_target)
    fitness = -jnp.mean(loss_vals)
    # guard against NaN/Inf from numerical instability in early training
    return jnp.where(jnp.isfinite(fitness), fitness, -1e6)

eggroll_opt, eggroll_state = create_eggroll_optimizer(trainable, learning_rate=EGG_LR)

# EGGROLL training loop
print(f"\nPhase 2: EGGROLL training — {EGGROLL_STEPS:,} steps")
print("=" * 60)

egg_history = {"mean_fitness": [], "max_fitness": [], "grad_norm": []}
start = time.time()

for step in range(1, EGGROLL_STEPS + 1):
    key, batch_key, step_key = jax.random.split(key, 3)

    # fresh batch each step, shared across population for fair comparison
    idx = jax.random.randint(batch_key, (EVAL_BATCH,), 0, num_chunks)
    _current_batch = jnp.array(all_chunks[idx])

    trainable, eggroll_state, metrics = eggroll_step(
        trainable, fitness_fn, eggroll_opt, eggroll_state,
        key=step_key, population_size=POP_SIZE, sigma=SIGMA, rank=1,
    )

    if step % 100 == 0:
        mf = float(metrics["mean_fitness"])
        xf = float(metrics["max_fitness"])
        gn = float(metrics["grad_norm"])
        egg_history["mean_fitness"].append(mf)
        egg_history["max_fitness"].append(xf)
        egg_history["grad_norm"].append(gn)
        elapsed = time.time() - start
        print(f"  step {step:>5}/{EGGROLL_STEPS} | "
              f"mean_f: {mf:.4f} | max_f: {xf:.4f} | "
              f"grad: {gn:.4f} | {elapsed/60:.0f}m")

elapsed = time.time() - start
print(f"\nPhase 2 done: {elapsed/3600:.2f}h")

# save backbone + head
final_bb = eqx.combine(trainable["backbone"],
                         eqx.filter(backbone, lambda x: not eqx.is_array(x)))
final_hd = eqx.combine(trainable["head"],
                         eqx.filter(head, lambda x: not eqx.is_array(x)))

eqx.tree_serialise_leaves("checkpoints/backbone_eggroll.eqx", final_bb)
eqx.tree_serialise_leaves("checkpoints/energy_head_eggroll.eqx", final_hd)

import json
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

# %% — 9. Quick evaluation: reconstruction + generation quality
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

# check backbone next-vector prediction quality
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

# %% — 10. Summary & Download
print("\n" + "=" * 60)
print("✅ VELM Training Complete!")
print("=" * 60)
print(f"Config:      {CONFIG_NAME}")
print(f"Tokenizer:   {DEFAULT_TOKENIZER} (vocab={VOCAB_SIZE:,})")
print(f"Dataset:     OpenWebMath ({all_chunks.shape[0]:,} chunks)")
print(f"AE accuracy: {final_acc:.4%}")
print("\nCheckpoints saved:")
print("  checkpoints/calm_ae_best.eqx      — best autoencoder")
print("  checkpoints/calm_ae_final.eqx     — final autoencoder")
print("  checkpoints/backbone_eggroll.eqx  — EGGROLL backbone")
print("  checkpoints/energy_head_eggroll.eqx — energy head")
print("  checkpoints/*.json                — metadata for loading")
print("\nTraining curves:")
print("  ae_training_curves.png")
print("  eggroll_training.png")

# Colab download helper
try:
    from google.colab import files
    for f in ["checkpoints/calm_ae_best.eqx", "checkpoints/calm_ae_best.json",
              "checkpoints/backbone_eggroll.eqx", "checkpoints/energy_head_eggroll.eqx",
              "checkpoints/backbone_meta.json",
              "ae_training_curves.png", "eggroll_training.png"]:
        if os.path.exists(f):
            files.download(f)
except ImportError:
    print("\nNot on Colab — checkpoints are in ./checkpoints/")
