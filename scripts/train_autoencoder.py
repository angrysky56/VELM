"""
VELM Phase 1.1 — Train CALM Autoencoder

Trains the token-chunk → latent-vector autoencoder independently.
This is the first experiment and foundation for all subsequent work.

Target: >99.9% token reconstruction accuracy.

Usage:
  python scripts/train_autoencoder.py [--config tiny] [--steps 50000]

Hardware: Single RTX 3060 12GB (~10 GPU-hours)
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.autoencoder import CALMAutoencoder, batch_ae_loss, reconstruction_accuracy
from src.model.config import CONFIGS


def create_dummy_batch(
    key: jax.Array,
    batch_size: int,
    chunk_size: int,
    vocab_size: int,
) -> jax.Array:
    """Create a random batch of token chunks for testing.

    Replace with real data loading for actual training.
    """
    return jax.random.randint(
        key, shape=(batch_size, chunk_size), minval=0, maxval=vocab_size
    )


@eqx.filter_jit
def train_step(
    model: CALMAutoencoder,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: jax.Array,
    key: jax.Array,
) -> tuple[CALMAutoencoder, optax.OptState, dict]:
    """Single training step: forward + backward + update."""

    def loss_fn(model):
        return batch_ae_loss(model, batch, key=key)

    (_, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_opt_state, metrics


@eqx.filter_jit
def eval_step(
    model: CALMAutoencoder,
    batch: jax.Array,
) -> float:
    """Compute reconstruction accuracy on a batch."""
    return reconstruction_accuracy(model, batch)


def main():
    parser = argparse.ArgumentParser(description="Train CALM Autoencoder")
    parser.add_argument("--config", default="tiny", choices=CONFIGS.keys())
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-dir", default="checkpoints/autoencoder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== VELM Autoencoder Training ===")
    print(f"Config: {args.config}")
    print(f"Chunk size K: {cfg['chunk_size_k']}")
    print(f"Latent dim: {cfg['latent_dim']}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {jax.default_backend()}")
    print()

    # initialize model
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    model = CALMAutoencoder(
        vocab_size=128_256,
        chunk_size=cfg["chunk_size_k"],
        hidden_dim=512,
        latent_dim=cfg["latent_dim"],
        ffn_intermediate=1024,
        kl_weight=0.001,
        kl_clip=1.0,
        key=init_key,
    )

    # count parameters
    num_params = sum(
        x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))
    )
    print(f"Autoencoder parameters: {num_params:,}")

    # optimizer: AdamW with warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
        end_value=args.lr * 0.1,
    )
    optimizer = optax.adamw(schedule, weight_decay=0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # training loop
    print("Starting training...")
    start_time = time.time()

    for step in range(1, args.steps + 1):
        key, batch_key, step_key = jax.random.split(key, 3)

        # TODO: replace with real data loader
        batch = create_dummy_batch(
            batch_key, args.batch_size, cfg["chunk_size_k"], 128_256
        )

        model, opt_state, metrics = train_step(
            model, opt_state, optimizer, batch, step_key
        )

        # logging
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            print(
                f"Step {step:>6d}/{args.steps} | "
                f"loss: {metrics['recon_loss']:.4f} | "
                f"kl: {metrics['kl_loss']:.4f} | "
                f"kl_raw: {metrics['kl_raw']:.2f} | "
                f"{steps_per_sec:.1f} steps/s"
            )

        # evaluation
        if step % args.eval_every == 0:
            key, eval_key = jax.random.split(key)
            eval_batch = create_dummy_batch(
                eval_key, args.batch_size, cfg["chunk_size_k"], 128_256
            )
            acc = eval_step(model, eval_batch)
            print(f"  >> Reconstruction accuracy: {acc:.4%}")

            # save checkpoint
            ckpt_path = save_dir / f"ae_step_{step}.eqx"
            eqx.tree_serialise_leaves(str(ckpt_path), model)
            print(f"  >> Saved checkpoint: {ckpt_path}")

    # final save
    final_path = save_dir / "ae_final.eqx"
    eqx.tree_serialise_leaves(str(final_path), model)
    print(f"\nTraining complete. Final model saved to {final_path}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()
