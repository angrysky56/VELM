#!/usr/bin/env python3
"""
VELM Training Script — Phase 1: CALM Autoencoder + Phase 2: EGGROLL Backbone

Trains on a streaming subset of OpenWebMath using the Qwen3.5 tokenizer.
Designed to complete in <2 days on a T4/G4 or <5 hours on H100.

Usage:
    python scripts/train_velm.py --config gpu_12gb --target-chunks 250000
    python scripts/train_velm.py --config gpu_12gb --target-chunks 1000000 --device-type h100
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from tqdm import tqdm

# ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.autoencoder import CALMAutoencoder, batch_ae_loss, reconstruction_accuracy
from src.model.config import CONFIGS, QWEN35_VOCAB_SIZE, DEFAULT_TOKENIZER


# ─── Device / Hardware Profiles ───────────────────────────────────────────────

HARDWARE_PROFILES: dict[str, dict] = {
    "t4": {  # Colab free / G4 instance
        "batch_size": 64,
        "ae_steps": 15_000,
        "eggroll_steps": 3_000,
        "pop_size": 32,
        "target_chunks": 250_000,
    },
    "a100": {  # Colab Pro / P4 instance
        "batch_size": 256,
        "ae_steps": 20_000,
        "eggroll_steps": 5_000,
        "pop_size": 64,
        "target_chunks": 500_000,
    },
    "h100": {  # H100 80GB
        "batch_size": 512,
        "ae_steps": 30_000,
        "eggroll_steps": 10_000,
        "pop_size": 128,
        "target_chunks": 1_000_000,
    },
}


# ─── Data Pipeline ────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str = DEFAULT_TOKENIZER):
    """Load Qwen3.5 tokenizer from HuggingFace."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"Tokenizer: {model_id}, vocab size: {vocab_size}")
    return tokenizer, vocab_size


def stream_openwebmath_chunks(
    tokenizer,
    chunk_size: int = 4,
    target_chunks: int = 250_000,
    max_seq_len: int = 512,
) -> np.ndarray:
    """Stream and tokenize OpenWebMath into K-token chunks.

    Uses HuggingFace streaming to avoid downloading the full 14.7B token dataset.
    We only pull enough documents to fill our target chunk count.

    Args:
        tokenizer: HuggingFace tokenizer instance
        chunk_size: K tokens per chunk (default 4, per CALM paper)
        target_chunks: number of chunks to collect
        max_seq_len: max tokens per document before truncation

    Returns:
        np.ndarray of shape (num_chunks, K) with token IDs
    """
    from datasets import load_dataset

    print(f"Streaming OpenWebMath → {target_chunks:,} chunks of K={chunk_size}...")
    dataset = load_dataset(
        "open-web-math/open-web-math",
        split="train",
        streaming=True,
    )

    chunk_buffer: list[np.ndarray] = []
    total_chunks = 0
    docs_processed = 0

    for example in tqdm(dataset, desc="Tokenizing", unit="docs"):
        text = example.get("text", "")
        if not text or len(text) < 50:
            continue

        tokens = tokenizer.encode(text, max_length=max_seq_len, truncation=True)
        # trim to multiple of K
        usable = len(tokens) - (len(tokens) % chunk_size)
        if usable < chunk_size:
            continue

        chunks = np.array(tokens[:usable], dtype=np.int32).reshape(-1, chunk_size)
        chunk_buffer.append(chunks)
        total_chunks += chunks.shape[0]
        docs_processed += 1

        if total_chunks >= target_chunks:
            break

        if docs_processed % 5000 == 0:
            print(f"  {docs_processed:,} docs → {total_chunks:,} chunks")

    all_chunks = np.concatenate(chunk_buffer, axis=0)[:target_chunks]
    print(f"✓ Collected {all_chunks.shape[0]:,} chunks × K={chunk_size}")
    print(f"  = {all_chunks.shape[0] * chunk_size:,} tokens from {docs_processed:,} docs")
    return all_chunks


# ─── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(
    model: eqx.Module,
    path: str | Path,
    config_name: str,
    vocab_size: int,
    tokenizer_id: str,
    step: int,
    metrics: dict | None = None,
) -> None:
    """Save model weights + metadata for reproducible loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # save weights
    eqx.tree_serialise_leaves(str(path), model)

    # save metadata alongside
    meta = {
        "config_name": config_name,
        "vocab_size": vocab_size,
        "tokenizer_id": tokenizer_id,
        "step": step,
        "metrics": metrics or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved: {path} + {meta_path}")


# ─── Phase 1: Autoencoder Training ───────────────────────────────────────────

def train_autoencoder(
    all_chunks: np.ndarray,
    config_name: str,
    vocab_size: int,
    tokenizer_id: str,
    batch_size: int = 64,
    total_steps: int = 15_000,
    lr: float = 3e-4,
    warmup: int = 500,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
) -> CALMAutoencoder:
    """Train CALM autoencoder to >99.9% token reconstruction accuracy.

    Args:
        all_chunks: (N, K) array of token IDs
        config_name: model config key for dims
        vocab_size: tokenizer vocabulary size
        tokenizer_id: HuggingFace tokenizer model ID
        batch_size: training batch size
        total_steps: number of gradient steps
        lr: peak learning rate
        warmup: warmup steps for cosine schedule
        checkpoint_dir: where to save weights
        seed: PRNG seed

    Returns:
        Trained CALMAutoencoder
    """
    cfg = CONFIGS[config_name]
    K = cfg["chunk_size_k"]
    hidden_dim = cfg.get("ae_hidden_dim", 256)
    latent_dim = cfg["latent_dim"]
    ffn_dim = cfg.get("ae_ffn_intermediate", 512)

    print(f"\n{'='*60}")
    print(f"Phase 1: CALM Autoencoder Training")
    print(f"  Config: {config_name} | vocab: {vocab_size:,}")
    print(f"  dims: hidden={hidden_dim}, latent={latent_dim}, K={K}")
    print(f"  steps: {total_steps:,} | batch: {batch_size} | lr: {lr}")
    print(f"{'='*60}\n")

    key = jax.random.PRNGKey(seed)
    k_init, key = jax.random.split(key)

    model = CALMAutoencoder(
        vocab_size=vocab_size,
        chunk_size=K,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        ffn_intermediate=ffn_dim,
        kl_weight=0.001,
        kl_clip=1.0,
        key=k_init,
    )

    num_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Autoencoder parameters: {num_params:,}")

    # optimizer: warmup cosine with AdamW
    # cap warmup to at most 10% of total steps (handles small smoke tests)
    effective_warmup = min(warmup, max(total_steps // 10, 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=effective_warmup,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )
    optimizer = optax.adamw(schedule, weight_decay=0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # JIT-compiled training step
    @eqx.filter_jit
    def train_step(model, opt_state, batch, step_key):
        def loss_fn(m):
            return batch_ae_loss(m, batch, key=step_key)
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt, loss, metrics

    @eqx.filter_jit
    def eval_accuracy(model, batch):
        return reconstruction_accuracy(model, batch)

    # training loop
    history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": [], "accuracy": []}
    start_time = time.time()
    num_chunks = all_chunks.shape[0]
    best_acc = 0.0

    for step in range(1, total_steps + 1):
        key, batch_key, step_key = jax.random.split(key, 3)

        # sample random batch of chunks
        indices = jax.random.randint(batch_key, (batch_size,), 0, num_chunks)
        batch = jnp.array(all_chunks[indices])

        model, opt_state, loss, metrics = train_step(model, opt_state, batch, step_key)

        # log every 500 steps
        if step % 500 == 0:
            elapsed = time.time() - start_time
            sps = step / elapsed
            rl = float(metrics["recon_loss"])
            kl = float(metrics["kl_loss"])
            history["loss"].append(float(loss))
            history["recon"].append(rl)
            history["kl"].append(kl)
            print(f"  step {step:>6}/{total_steps} | "
                  f"recon: {rl:.4f} | kl: {kl:.4f} | "
                  f"{sps:.1f} steps/s | {elapsed/60:.0f}m elapsed")

        # eval accuracy every 2000 steps
        if step % 2000 == 0:
            key, eval_key = jax.random.split(key)
            eval_idx = jax.random.randint(eval_key, (512,), 0, num_chunks)
            eval_batch = jnp.array(all_chunks[eval_idx])
            acc = float(eval_accuracy(model, eval_batch))
            history["accuracy"].append(acc)
            print(f"  >> Reconstruction accuracy: {acc:.4%}")

            if acc > best_acc:
                best_acc = acc
                save_checkpoint(
                    model,
                    Path(checkpoint_dir) / "calm_ae_best.eqx",
                    config_name, vocab_size, tokenizer_id,
                    step=step,
                    metrics={"accuracy": acc, "recon_loss": float(metrics["recon_loss"])},
                )

            if acc > 0.999:
                print(f"  🎯 TARGET >99.9% REACHED at step {step}!")

    total_time = time.time() - start_time
    final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
    print(f"\nPhase 1 complete: {total_time/3600:.2f}h, accuracy={final_acc:.4%}")

    # save final
    save_checkpoint(
        model,
        Path(checkpoint_dir) / "calm_ae_final.eqx",
        config_name, vocab_size, tokenizer_id,
        step=total_steps,
        metrics={"accuracy": final_acc, "history": history},
    )

    return model


# ─── Phase 2: EGGROLL Backbone Training ──────────────────────────────────────

def train_backbone_eggroll(
    frozen_ae: CALMAutoencoder,
    all_chunks: np.ndarray,
    config_name: str,
    vocab_size: int,
    tokenizer_id: str,
    pop_size: int = 32,
    total_steps: int = 3_000,
    sigma: float = 0.01,
    lr: float = 1e-3,
    eval_batch_size: int = 32,
    checkpoint_dir: str = "checkpoints",
    seed: int = 123,
):
    """Train backbone + energy head using EGGROLL (gradient-free ES).

    The autoencoder is frozen. EGGROLL perturbs backbone + head weights,
    evaluates fitness (negative energy loss), and applies Adam on the
    ES gradient estimate.

    Args:
        frozen_ae: trained autoencoder (frozen, used for encoding targets)
        all_chunks: (N, K) token chunks
        config_name: model config key
        vocab_size: tokenizer vocab size
        tokenizer_id: HuggingFace tokenizer model ID
        pop_size: EGGROLL population size
        total_steps: number of ES update steps
        sigma: perturbation scale σ
        lr: Adam learning rate for ES gradient
        eval_batch_size: chunks per fitness evaluation
        checkpoint_dir: where to save weights
        seed: PRNG seed
    """
    from src.model.miras_backbone import VELMBackbone
    from src.model.energy_head import EnergyHead, energy_score
    from src.training.eggroll import eggroll_step, create_eggroll_optimizer

    cfg = CONFIGS[config_name]
    K = cfg["chunk_size_k"]
    ae_hdim = cfg.get("ae_hidden_dim", 256)

    print(f"\n{'='*60}")
    print(f"Phase 2: EGGROLL Backbone Training (gradient-free)")
    print(f"  Config: {config_name} | dim={cfg['hidden_dim']}")
    print(f"  pop_size={pop_size} | σ={sigma} | steps={total_steps:,}")
    print(f"{'='*60}\n")

    key = jax.random.PRNGKey(seed)
    k1, k2, key = jax.random.split(key, 3)

    backbone = VELMBackbone(
        dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_miras_layers=cfg["miras_layers"],
        num_swa_layers=cfg["swa_layers"],
        ffn_intermediate=cfg["ffn_intermediate"],
        chunk_size=K,
        ae_hidden_dim=ae_hdim,
        key=k1,
    )

    head = EnergyHead(
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_blocks=cfg["energy_head_blocks"],
        key=k2,
    )

    bb_params = sum(x.size for x in jax.tree.leaves(eqx.filter(backbone, eqx.is_array)))
    hd_params = sum(x.size for x in jax.tree.leaves(eqx.filter(head, eqx.is_array)))
    print(f"Backbone: {bb_params:,} | Head: {hd_params:,} | Total: {bb_params+hd_params:,}")

    # pack trainable params for EGGROLL
    trainable = {
        "backbone": eqx.filter(backbone, eqx.is_array),
        "head": eqx.filter(head, eqx.is_array),
    }

    num_chunks = all_chunks.shape[0]

    def fitness_fn(params):
        """Evaluate fitness: negative energy loss on a random batch.

        Uses the step key (captured via closure) to vary batches across
        steps while keeping all population members on the same batch
        for fair comparison within a generation.
        """
        # reconstruct modules from param pytree
        bb = eqx.combine(
            params["backbone"],
            eqx.filter(backbone, lambda x: not eqx.is_array(x)),
        )
        hd = eqx.combine(
            params["head"],
            eqx.filter(head, lambda x: not eqx.is_array(x)),
        )

        # use a fixed batch per step (set in the loop below via closure)
        batch_tokens = _current_batch  # noqa: F821 — set by training loop

        # encode chunks → target latents via frozen AE
        target_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(batch_tokens)

        # compress input for backbone
        def compress(chunk):
            embs = jax.vmap(frozen_ae.embedding)(chunk)
            return bb.compress_input(embs)
        input_seq = jax.vmap(compress)(batch_tokens)

        # backbone forward → hidden states
        hidden, _ = bb(input_seq)

        # energy loss: predict z_{i+1} from h_i (shifted pairs)
        h_in = hidden[:-1]
        z_tgt = target_z[1:]

        def pos_loss(h, z_t):
            samples = hd(h, key=jax.random.PRNGKey(0), num_samples=4)
            return energy_score(samples, z_t)

        losses = jax.vmap(pos_loss)(h_in, z_tgt)
        return -jnp.mean(losses)  # negate: EGGROLL maximizes fitness

    # EGGROLL optimizer
    eggroll_opt, eggroll_state = create_eggroll_optimizer(trainable, learning_rate=lr)

    # training loop
    history = {"mean_fitness": [], "max_fitness": [], "grad_norm": []}
    start = time.time()
    best_fitness = -float("inf")

    for step in range(1, total_steps + 1):
        key, batch_key, step_key = jax.random.split(key, 3)

        # sample a fresh batch each step (varied across steps, shared within pop)
        idx = jax.random.randint(batch_key, (eval_batch_size,), 0, num_chunks)
        _current_batch = jnp.array(all_chunks[idx])  # noqa: F841 — used by fitness_fn closure

        trainable, eggroll_state, metrics = eggroll_step(
            trainable, fitness_fn, eggroll_opt, eggroll_state,
            key=step_key, population_size=pop_size, sigma=sigma, rank=1,
        )

        if step % 100 == 0:
            mf = float(metrics["mean_fitness"])
            xf = float(metrics["max_fitness"])
            gn = float(metrics["grad_norm"])
            history["mean_fitness"].append(mf)
            history["max_fitness"].append(xf)
            history["grad_norm"].append(gn)

            elapsed = time.time() - start
            print(f"  step {step:>5}/{total_steps} | "
                  f"mean_f: {mf:.4f} | max_f: {xf:.4f} | "
                  f"grad: {gn:.4f} | {elapsed/60:.0f}m elapsed")

            if xf > best_fitness:
                best_fitness = xf

    elapsed = time.time() - start
    print(f"\nPhase 2 complete: {elapsed/3600:.2f}h")
    print(f"Best fitness: {best_fitness:.4f}")

    # reconstruct and save backbone + head
    final_bb = eqx.combine(
        trainable["backbone"],
        eqx.filter(backbone, lambda x: not eqx.is_array(x)),
    )
    final_hd = eqx.combine(
        trainable["head"],
        eqx.filter(head, lambda x: not eqx.is_array(x)),
    )

    save_checkpoint(
        final_bb,
        Path(checkpoint_dir) / "backbone_eggroll.eqx",
        config_name, vocab_size, tokenizer_id,
        step=total_steps,
        metrics={"best_fitness": best_fitness, "history": history},
    )
    save_checkpoint(
        final_hd,
        Path(checkpoint_dir) / "energy_head_eggroll.eqx",
        config_name, vocab_size, tokenizer_id,
        step=total_steps,
        metrics={"best_fitness": best_fitness},
    )

    return final_bb, final_hd


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VELM Training Pipeline")
    parser.add_argument("--config", default="gpu_12gb",
                        choices=list(CONFIGS.keys()),
                        help="Model config (default: gpu_12gb)")
    parser.add_argument("--device-type", default="t4",
                        choices=list(HARDWARE_PROFILES.keys()),
                        help="Hardware profile for batch/step sizing")
    parser.add_argument("--target-chunks", type=int, default=None,
                        help="Override: number of text chunks to collect")
    parser.add_argument("--ae-steps", type=int, default=None,
                        help="Override: autoencoder training steps")
    parser.add_argument("--eggroll-steps", type=int, default=None,
                        help="Override: EGGROLL backbone training steps")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for saving model weights")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="HuggingFace tokenizer model ID")
    parser.add_argument("--skip-ae", action="store_true",
                        help="Skip AE training, load from checkpoint")
    parser.add_argument("--skip-eggroll", action="store_true",
                        help="Skip EGGROLL training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hw = HARDWARE_PROFILES[args.device_type]
    target_chunks = args.target_chunks or hw["target_chunks"]
    ae_steps = args.ae_steps or hw["ae_steps"]
    eggroll_steps = args.eggroll_steps or hw["eggroll_steps"]
    batch_size = hw["batch_size"]
    pop_size = hw["pop_size"]

    # device info
    print(f"\n{'='*60}")
    print(f"VELM Training Pipeline")
    print(f"  JAX {jax.__version__} | backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    print(f"  Config: {args.config} | HW profile: {args.device_type}")
    print(f"  Target chunks: {target_chunks:,} | AE steps: {ae_steps:,}")
    print(f"  EGGROLL steps: {eggroll_steps:,} | pop: {pop_size}")
    print(f"{'='*60}")

    if jax.default_backend() != "gpu":
        print("⚠️  No GPU detected! Training will be very slow.")

    # load tokenizer
    tokenizer, vocab_size = load_tokenizer(args.tokenizer)

    # load data
    all_chunks = stream_openwebmath_chunks(
        tokenizer,
        chunk_size=CONFIGS[args.config]["chunk_size_k"],
        target_chunks=target_chunks,
    )

    # Phase 1: Autoencoder
    if not args.skip_ae:
        ae_model = train_autoencoder(
            all_chunks,
            config_name=args.config,
            vocab_size=vocab_size,
            tokenizer_id=args.tokenizer,
            batch_size=batch_size,
            total_steps=ae_steps,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
        )
    else:
        # load from checkpoint
        ae_path = Path(args.checkpoint_dir) / "calm_ae_best.eqx"
        print(f"Loading AE from {ae_path}...")
        cfg = CONFIGS[args.config]
        ae_model = CALMAutoencoder(
            vocab_size=vocab_size,
            chunk_size=cfg["chunk_size_k"],
            hidden_dim=cfg.get("ae_hidden_dim", 256),
            latent_dim=cfg["latent_dim"],
            ffn_intermediate=cfg.get("ae_ffn_intermediate", 512),
            key=jax.random.PRNGKey(0),
        )
        ae_model = eqx.tree_deserialise_leaves(str(ae_path), ae_model)
        print("✓ AE loaded")


    # Phase 2: EGGROLL Backbone
    if not args.skip_eggroll:
        train_backbone_eggroll(
            frozen_ae=ae_model,
            all_chunks=all_chunks,
            config_name=args.config,
            vocab_size=vocab_size,
            tokenizer_id=args.tokenizer,
            pop_size=pop_size,
            total_steps=eggroll_steps,
            sigma=0.01,
            lr=1e-3,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed + 81,
        )

    print(f"\n{'='*60}")
    print(f"✅ VELM training complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
