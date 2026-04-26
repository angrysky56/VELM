"""
VELM Training — JIT-compiled fitness module for EGGROLL Phase 2.

Replaces the hand-rolled, un-JIT'd Python loop that lived in the Colab
notebook. Three correctness fixes from the previous version:

  1. Energy-head noise key was hard-coded to PRNGKey(0), making every
     fitness evaluation see identical noise. ES gradient signal was
     therefore dominated by weight differences amplified by whatever
     the 8 fixed noise samples happened to be sensitive to. Now we
     thread a real per-step key through.

  2. The frozen-AE batch encoding ran inside every fitness call,
     redoing the same work for every population member (~63x waste with
     antithetic POP=32). The new factory pre-computes (tgt_z, inp_seq)
     once per step and passes them into a JIT-compiled inner step.

  3. Population evaluation used a Python `for` loop instead of
     `jax.lax.map`, retracing per member and serializing host->device.
     The new step uses `jax.lax.map` for a single GPU-resident scan.

Public API:

    make_velm_eggroll_step(frozen_ae, bb_static, hd_static, optimizer, *,
                           pop_size, rank=1, num_samples=8,
                           perturb_head_only=False)
        → step(base_params, opt_state, batch_tokens, step_key, sigma)
          → (new_params, new_opt_state, metrics)

The factory bakes `pop_size`, `rank`, `num_samples`, and the
`perturb_head_only` flag into the JIT cache, so two compilations per
training run are typical (head-only phase, then full-model phase).
"""

# ruff: noqa: F722, F821

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PyTree

from src.model.energy_head import energy_score
from src.training.eggroll import perturb_pytree


def prepare_batch(
    frozen_ae,
    batch_tokens: Int[Array, "B K"],
) -> tuple[Float[Array, "B latent"], Float[Array, "B dim"]]:
    """Pre-compute fixed quantities used by every population member.

    The frozen AE is the same for every member, so we only need to
    encode each chunk to its target latent and to compress its token
    embeddings to the backbone input ONCE per step, not once per member.

    Args:
        frozen_ae: trained CALMAutoencoder (frozen, no weight updates)
        batch_tokens: (B, K) int token IDs

    Returns:
        (tgt_z, batch_embs) where
          tgt_z: (B, latent_dim) AE-encoded target latents
          batch_embs: (B, K, ae_hidden_dim) embedded tokens, ready
                      to be compressed by each candidate backbone

    Note: we return embeddings rather than the compressed input vector
    because `compress_input` lives on the backbone — and the backbone
    is what gets perturbed per member. So compression must happen
    inside the per-member fitness call. Embeddings are tied to the
    frozen AE only, hence safe to compute once.
    """
    tgt_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(batch_tokens)
    batch_embs = jax.vmap(jax.vmap(frozen_ae.embedding))(batch_tokens)
    return tgt_z, batch_embs


def make_velm_eggroll_step(
    frozen_ae,
    bb_static: PyTree,
    hd_static: PyTree,
    optimizer: optax.GradientTransformation,
    *,
    pop_size: int,
    rank: int = 1,
    num_samples: int = 8,
    perturb_head_only: bool = False,
) -> Callable:
    """Build a JIT-compiled antithetic-EGGROLL step for VELM.

    Args:
        frozen_ae: trained CALMAutoencoder, frozen
        bb_static: backbone non-array structure (from eqx.filter)
        hd_static: head non-array structure (from eqx.filter)
        optimizer: optax GradientTransformation already initialized
        pop_size: N — number of antithetic ES directions per step.
                  Total fitness evaluations per step = 2N.
        rank: low-rank perturbation rank (1 is the EGGROLL paper default
              and the most memory-efficient).
        num_samples: N for the energy-score Monte Carlo estimator (CALM
                     paper recommends N=8).
        perturb_head_only: when True, only the "head" subtree is
                           perturbed; the "backbone" subtree of
                           base_params is held fixed (used for the
                           Phase i head-only warmup).

    Returns:
        step: a JIT-compiled callable

            step(base_params, opt_state, batch_tokens, step_key, sigma)
                → (new_params, new_opt_state, metrics_dict)

        where:
            base_params is a {"backbone": ..., "head": ...} dict of
            arrays (the "trainable" pytree from the notebook).
            sigma is a JAX scalar (not a Python float) so it can be
            adapted at run time without recompilation.
    """

    @eqx.filter_jit
    def step(
        base_params: PyTree,
        opt_state: optax.OptState,
        batch_tokens: Int[Array, "B K"],
        step_key: jax.Array,
        sigma: Float[Array, ""],
    ) -> tuple[PyTree, optax.OptState, dict[str, jax.Array]]:
        # ── 1. Pre-compute frozen-AE outputs once per step ─────────
        tgt_z, batch_embs = prepare_batch(frozen_ae, batch_tokens)
        z_target = tgt_z[1:]  # predict z_{i+1} from h_i

        # ── 2. Inner fitness function (negative energy loss) ───────
        def fitness_fn(perturbed: PyTree) -> Float[Array, ""]:
            bb = eqx.combine(perturbed["backbone"], bb_static)
            hd = eqx.combine(perturbed["head"], hd_static)

            # compress per-chunk embeddings into a single backbone input
            inp_seq = jax.vmap(bb.compress_input)(batch_embs)  # (B, dim)

            hid, _ = bb(inp_seq)  # (B, dim)
            hid_in = hid[:-1]

            # use a real per-position key so members see distinct noise
            keys = jax.random.split(step_key, hid_in.shape[0])

            def pos_loss(h, z_t, k):
                samples = hd(h, key=k, num_samples=num_samples)
                return energy_score(samples, z_t)

            losses = jax.vmap(pos_loss)(hid_in, z_target, keys)
            mean_loss = jnp.mean(losses)
            # NaN guard: turn pathological evals into very-low fitness
            # (rather than NaN-poisoning the entire ES gradient).
            return jnp.where(jnp.isfinite(mean_loss), -mean_loss, -1e6)

        # ── 3. Antithetic population evaluation via jax.lax.map ────
        member_keys = jax.random.split(step_key, pop_size)

        def eval_anti(m_key: jax.Array):
            """Generate one perturbation, evaluate W±σE, return diff + pert."""
            if perturb_head_only:
                # perturb only the head subtree; backbone stays at base
                _, head_pert = perturb_pytree(
                    base_params["head"], m_key, sigma, rank
                )
                pos_p = {
                    "backbone": base_params["backbone"],
                    "head": jax.tree.map(
                        lambda p, e: p + sigma * e,
                        base_params["head"],
                        head_pert,
                    ),
                }
                neg_p = {
                    "backbone": base_params["backbone"],
                    "head": jax.tree.map(
                        lambda p, e: p - sigma * e,
                        base_params["head"],
                        head_pert,
                    ),
                }
                pert = {"head": head_pert}
            else:
                _, pert = perturb_pytree(base_params, m_key, sigma, rank)
                pos_p = jax.tree.map(
                    lambda p, e: p + sigma * e, base_params, pert
                )
                neg_p = jax.tree.map(
                    lambda p, e: p - sigma * e, base_params, pert
                )

            f_pos = fitness_fn(pos_p)
            f_neg = fitness_fn(neg_p)
            return f_pos, f_neg, pert

        f_pos_arr, f_neg_arr, perts = jax.lax.map(eval_anti, member_keys)

        # ── 4. Antithetic ES gradient ──────────────────────────────
        diffs = f_pos_arr - f_neg_arr  # (N,)
        fits = (f_pos_arr + f_neg_arr) / 2  # (N,) for metrics

        def weighted_sum(leaf_stack: jax.Array) -> jax.Array:
            w = diffs.reshape((-1,) + (1,) * (leaf_stack.ndim - 1))
            return jnp.sum(leaf_stack * w, axis=0)

        es_grad = jax.tree.map(weighted_sum, perts)
        # 1/(2σN) for the antithetic ES estimator
        scale = 1.0 / (2.0 * sigma * pop_size)
        es_grad = jax.tree.map(lambda g: g * scale, es_grad)

        # NaN-guard the gradient before handing it to the optimizer
        es_grad = jax.tree.map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), es_grad
        )

        # ── 5. Adam update (negate: ES maximizes, optax minimizes) ─
        if perturb_head_only:
            # Gradient lives only on the head subtree; the optimizer
            # state was initialized over {"head": ...} so updates must
            # be applied to a same-shaped pytree, then re-merged.
            head_neg_grad = jax.tree.map(lambda g: -g, es_grad["head"])
            head_params = {"head": base_params["head"]}
            updates, new_opt_state = optimizer.update(
                {"head": head_neg_grad}, opt_state, head_params
            )
            new_head_params = optax.apply_updates(head_params, updates)
            new_params = {
                "backbone": base_params["backbone"],
                "head": new_head_params["head"],
            }
        else:
            neg_grad = jax.tree.map(lambda g: -g, es_grad)
            updates, new_opt_state = optimizer.update(
                neg_grad, opt_state, base_params
            )
            new_params = optax.apply_updates(base_params, updates)

        # ── 6. Metrics ─────────────────────────────────────────────
        es_grad_leaves = jax.tree.leaves(es_grad)
        grad_norm = jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in es_grad_leaves))
        metrics = {
            "mean_fitness": jnp.mean(fits),
            "max_fitness": jnp.max(fits),
            "min_fitness": jnp.min(fits),
            "fitness_std": jnp.std(fits),
            "grad_norm": grad_norm,
        }
        return new_params, new_opt_state, metrics

    return step


def make_velm_fitness_eval(
    frozen_ae,
    bb_static: PyTree,
    hd_static: PyTree,
    *,
    num_samples: int = 8,
) -> Callable:
    """Build a JIT-compiled fitness evaluator (no perturbation).

    Used for: GEA Phase 3 task-domain evaluation, smoke checks,
    and final evaluation. Reuses the same energy-loss machinery as
    `make_velm_eggroll_step` but for a single param set.

    Returns:
        evaluate(params, batch_tokens, key) -> scalar fitness
    """

    @eqx.filter_jit
    def evaluate(
        params: PyTree,
        batch_tokens: Int[Array, "B K"],
        key: jax.Array,
    ) -> Float[Array, ""]:
        bb = eqx.combine(params["backbone"], bb_static)
        hd = eqx.combine(params["head"], hd_static)

        tgt_z, batch_embs = prepare_batch(frozen_ae, batch_tokens)
        inp_seq = jax.vmap(bb.compress_input)(batch_embs)

        hid, _ = bb(inp_seq)
        hid_in, z_target = hid[:-1], tgt_z[1:]

        keys = jax.random.split(key, hid_in.shape[0])

        def pos_loss(h, z_t, k):
            samples = hd(h, key=k, num_samples=num_samples)
            return energy_score(samples, z_t)

        losses = jax.vmap(pos_loss)(hid_in, z_target, keys)
        mean_loss = jnp.mean(losses)
        return jnp.where(jnp.isfinite(mean_loss), -mean_loss, -1e6)

    return evaluate
