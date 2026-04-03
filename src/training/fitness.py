"""
VELM Training — Multi-objective Fitness Function

Combines quality and compression objectives for EGGROLL training:

  fitness(θ) = quality_score(θ) - λ × compression_penalty(θ)

Components:
  - quality_score: negative energy loss on next-vector prediction
  - compression_penalty: CIB-inspired penalty for verbose reasoning

The fitness is a black-box scalar — no differentiability required.
"""

# ruff: noqa: F722, F821

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


def quality_fitness(
    model_forward: Callable,
    params: PyTree,
    batch: dict,
    *,
    key: jax.Array,
) -> Float[Array, ""]:
    """Compute quality fitness from next-vector prediction.

    Higher fitness = better model. Uses negative energy loss
    so that EGGROLL (which maximizes fitness) improves quality.

    Args:
        model_forward: function(params, batch, key) → energy_loss
        params: model parameters being evaluated
        batch: dict with "token_ids" key → (S, K) array
        key: PRNG key

    Returns:
        Scalar quality fitness (higher = better)
    """
    energy_loss = model_forward(params, batch, key=key)
    return -energy_loss  # negate: lower loss = higher fitness


def compression_penalty(
    num_reasoning_chunks: int,
    max_chunks: int,
    target_ratio: float = 0.3,
) -> Float[Array, ""]:
    """CIB-inspired compression penalty.

    Penalizes using more reasoning chunks than needed.
    Target: use only target_ratio of the maximum budget.

    Args:
        num_reasoning_chunks: actual chunks used for reasoning
        max_chunks: maximum allowed reasoning chunks
        target_ratio: target compression ratio (default 0.3 = 70% compression)

    Returns:
        Scalar penalty (higher = worse, more verbose)
    """
    ratio = num_reasoning_chunks / max_chunks
    # penalty increases quadratically above target ratio
    excess = jnp.maximum(ratio - target_ratio, 0.0)
    return excess**2


def combined_fitness(
    model_forward: Callable,
    params: PyTree,
    batch: dict,
    *,
    key: jax.Array,
    compression_weight: float = 0.1,
    max_reasoning_chunks: int = 64,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Combined quality + compression fitness for EGGROLL.

    fitness = quality - λ × compression_penalty

    Args:
        model_forward: function(params, batch, key) → (loss, num_chunks)
        params: model parameters
        batch: training batch
        key: PRNG key
        compression_weight: λ — weight on compression penalty
        max_reasoning_chunks: maximum reasoning budget

    Returns:
        (fitness_scalar, metrics_dict)
    """
    energy_loss, num_chunks = model_forward(params, batch, key=key)

    q_fitness = -energy_loss
    c_penalty = compression_penalty(num_chunks, max_reasoning_chunks)

    total_fitness = q_fitness - compression_weight * c_penalty

    metrics = {
        "total_fitness": total_fitness,
        "quality_fitness": q_fitness,
        "compression_penalty": c_penalty,
        "energy_loss": energy_loss,
        "num_reasoning_chunks": jnp.array(num_chunks, dtype=jnp.float32),
    }
    return total_fitness, metrics
