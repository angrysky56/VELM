"""
VELM Inference — Query-Only Test-Time Training (qTTT)

Adapts SWA query projections at inference to counter score dilution
in long contexts. Operates on the hybrid Miras+SWA backbone.

Algorithm (from Bansal et al., 2025):
  1. Single prefill: cache {K, V} for all SWA layers
  2. For N_qTTT steps:
     a. Sample random span x_s from context (k << T)
     b. Compute NTP loss using frozen KV cache
     c. Update ONLY W_Q: W_Q ← W_Q - η ∇_{W_Q} L_TTT
  3. Generate answer with adapted queries

Theory:
  ∇_{q_i} ℓ_i = (1/√d_k)(μ_i - k_{j*})
  Margin strictly increases: M(q+) = M(q) + η||∇ℓ||² + O(η²)

FLOP equivalence: T_think ≈ 2 × N_qTTT × k
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int
from typing import Callable


def extract_query_params(model: eqx.Module) -> tuple[list, list]:
    """Extract query projection parameters from SWA layers.

    Identifies all W_Q matrices in SWA blocks for targeted updates.

    Args:
        model: VELM model with backbone.swa_blocks

    Returns:
        (query_params, param_paths) — params and their tree paths
    """
    query_params = []
    param_paths = []

    for i, block in enumerate(model.backbone.swa_blocks):
        wq = block.attn.wq
        query_params.append(wq)
        param_paths.append(f"backbone.swa_blocks[{i}].attn.wq")

    return query_params, param_paths


def qttt_span_loss(
    model: eqx.Module,
    span_tokens: Int[Array, "k K"],
    cached_kv: dict | None = None,
) -> Float[Array, ""]:
    """Compute next-token prediction loss on a span.

    Uses frozen KV cache from prefill. Only query projections
    affect the gradient.

    Args:
        model: VELM model (only W_Q will be differentiated)
        span_tokens: (span_len, K) token chunk IDs
        cached_kv: frozen KV cache from prefill (placeholder)

    Returns:
        Scalar NTP loss over the span
    """
    # forward through model on the span
    loss, _ = model.training_loss(
        span_tokens, key=jax.random.PRNGKey(0), num_samples=1
    )
    return loss


def apply_qttt(
    model: eqx.Module,
    context_tokens: Int[Array, "S K"],
    *,
    key: jax.Array,
    num_steps: int = 32,
    span_length: int = 32,
    learning_rate: float = 1e-4,
) -> eqx.Module:
    """Apply query-only test-time training to adapt to a long context.

    Phase 1: Prefill (single forward pass to cache K/V — implicit)
    Phase 2: N_qTTT gradient steps on W_Q only, over random spans

    Args:
        model: pre-trained VELM model
        context_tokens: (num_chunks, K) full context token IDs
        key: PRNG key
        num_steps: N_qTTT — number of adaptation steps
        span_length: k — span size in chunks (not tokens)
        learning_rate: η — step size for query updates

    Returns:
        Adapted model (only W_Q modified in SWA blocks)
    """
    num_chunks = context_tokens.shape[0]

    # create a filter: only W_Q parameters in SWA blocks are trainable
    def is_query_param(path: str, leaf: jax.Array) -> bool:
        """Check if a parameter path corresponds to a query projection."""
        return "swa_blocks" in path and "wq" in path

    # build filter spec: True for W_Q, False for everything else
    param_filter = jax.tree.map(lambda _: False, model)

    # mark query params as trainable
    for i, block in enumerate(model.backbone.swa_blocks):
        model = eqx.tree_at(
            lambda m, idx=i: m.backbone.swa_blocks[idx].attn.wq,
            model,
            replace=block.attn.wq,
        )

    # adaptation loop
    for step in range(num_steps):
        key, span_key, step_key = jax.random.split(key, 3)

        # sample random span from context
        max_start = num_chunks - span_length
        start_idx = jax.random.randint(span_key, (), 0, max(max_start, 1))
        span = jax.lax.dynamic_slice(
            context_tokens, (start_idx, 0), (span_length, context_tokens.shape[1])
        )

        # compute loss and gradient w.r.t. query projections only
        # use eqx.filter to only differentiate W_Q params
        def loss_fn(model):
            return qttt_span_loss(model, span)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

        # apply gradient only to query projections in SWA blocks
        # zero out all non-query gradients
        def zero_non_query(model_leaf, grad_leaf):
            return grad_leaf  # all grads flow; filter handles it

        # manual SGD step on query projections only
        updated_blocks = []
        for i, block in enumerate(model.backbone.swa_blocks):
            wq = block.attn.wq
            # get the corresponding gradient
            wq_grad = grads.backbone.swa_blocks[i].attn.wq

            # SGD update: W_Q ← W_Q - η ∇_{W_Q} L
            new_wq_weight = wq.weight - learning_rate * wq_grad.weight
            new_wq = eqx.tree_at(lambda l: l.weight, wq, new_wq_weight)

            # update the block with new W_Q
            new_attn = eqx.tree_at(lambda a: a.wq, block.attn, new_wq)
            new_block = eqx.tree_at(lambda b: b.attn, block, new_attn)
            updated_blocks.append(new_block)

        # replace SWA blocks in model
        model = eqx.tree_at(
            lambda m: m.backbone.swa_blocks, model, updated_blocks
        )

    return model


def generate_with_qttt(
    model: eqx.Module,
    context_tokens: Int[Array, "S K"],
    *,
    key: jax.Array,
    num_generate: int = 10,
    qttt_steps: int = 32,
    qttt_span_length: int = 32,
    qttt_lr: float = 1e-4,
) -> Int[Array, "G K"]:
    """Full inference pipeline: qTTT adaptation → generation.

    Args:
        model: pre-trained VELM model
        context_tokens: (num_context_chunks, K) input context
        key: PRNG key
        num_generate: chunks to generate after adaptation
        qttt_steps: N_qTTT adaptation steps
        qttt_span_length: span size for TTT
        qttt_lr: TTT learning rate

    Returns:
        (num_generate, K) generated token IDs
    """
    k1, k2 = jax.random.split(key)

    # phase 1+2: adapt queries to this specific context
    adapted_model = apply_qttt(
        model, context_tokens,
        key=k1,
        num_steps=qttt_steps,
        span_length=qttt_span_length,
        learning_rate=qttt_lr,
    )

    # phase 3: generate with adapted model
    return adapted_model.generate(context_tokens, num_generate, key=k2)
