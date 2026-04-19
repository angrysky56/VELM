"""
VELM Training — EGGROLL Optimizer

Low-rank evolution strategies for gradient-free training of VELM.

Key insight from EGGROLL paper: structure perturbations as rank-r matrices
E_i = (1/√r) A_i B_i^T. This makes fitness evaluation as fast as batch
inference (91% throughput), enabling population sizes up to 2^20.

Algorithm:
  1. For each member i in population:
     a. Generate A_i, B_i from counter-based RNG (no storage needed)
     b. Perturbation E_i = (1/√r) A_i B_i^T
     c. Apply to base weights: W_i = M + σ E_i
     d. Evaluate fitness: f_i = fitness(W_i)
  2. ES gradient: ∇M ≈ (1/σN) Σ E_i f_i
  3. Update base weights via Adam on the ES gradient
"""

# ruff: noqa: F722, F821

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree


def generate_low_rank_perturbation(
    key: jax.Array,
    shape: tuple[int, ...],
    rank: int = 1,
) -> Float[Array, "..."]:
    """Generate a single low-rank perturbation matrix.

    E = (1/√r) A B^T where A ∈ R^{m×r}, B ∈ R^{n×r}

    For rank=1, this is an outer product of two vectors,
    which is the most memory-efficient configuration.

    Args:
        key: PRNG key (counter-based for reproducibility)
        shape: target parameter shape (m, n) for matrices, (d,) for vectors
        rank: perturbation rank (default 1)

    Returns:
        Perturbation with same shape as target parameter
    """
    if len(shape) == 0:  # Scalar
        return jax.random.normal(key, ())

    # Use static calculation for the flat size to avoid JAX tracing issues
    total_size = math.prod(shape)
    if total_size == 0:
        return jnp.zeros(shape)

    # For low-rank, we interpret the shape as a matrix (Rows, Cols)
    # If it's a vector, Rows=1. If it's a tensor, we flatten to (Prod(dims[:-1]), last_dim).
    if len(shape) == 1:
        rows, cols = 1, shape[0]
    else:
        rows = math.prod(shape[:-1])
        cols = shape[-1]

    # If the rank is higher than dimensions, just do full-rank
    if rank >= min(rows, cols):
        return jax.random.normal(key, shape)

    k1, k2 = jax.random.split(key)
    u = jax.random.normal(k1, (rows, rank))
    v = jax.random.normal(k2, (rank, cols))

    # Scale to maintain Unit Variance: E = (U @ V) / sqrt(rank)
    # Var((U@V)_ij) = sum_{k=1}^rank Var(U_ik * V_kj) = rank
    perturb_2d = (u @ v) / jnp.sqrt(float(rank))
    return perturb_2d.reshape(shape)


def perturb_pytree(
    params: PyTree,
    key: jax.Array,
    sigma: float,
    rank: int = 1,
) -> tuple[PyTree, PyTree]:
    """Generate a low-rank perturbation for all leaves of a PyTree.

    Args:
        params: model parameters (Equinox PyTree of arrays)
        key: PRNG key
        sigma: perturbation scale
        rank: perturbation rank

    Returns:
        (perturbed_params, perturbation_tree)
    """
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))

    perturbations = []
    perturbed = []
    for leaf, k in zip(leaves, keys):
        e = generate_low_rank_perturbation(k, leaf.shape, rank)
        perturbations.append(e)
        perturbed.append(leaf + sigma * e)

    return (
        jax.tree.unflatten(treedef, perturbed),
        jax.tree.unflatten(treedef, perturbations),
    )


class EGGROLLState:
    """Optimizer state for EGGROLL.

    Wraps an optax optimizer state for applying Adam-like updates
    to the ES gradient estimates.

    Attributes:
        opt_state: underlying optax optimizer state
        step: current training step
        base_params: the mean parameter matrix M
    """

    def __init__(
        self,
        opt_state: optax.OptState,
        step: int = 0,
    ) -> None:
        self.opt_state = opt_state
        self.step = step


def create_eggroll_optimizer(
    params: PyTree,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> tuple[optax.GradientTransformation, EGGROLLState]:
    """Create an EGGROLL optimizer with Adam on ES gradients.

    Args:
        params: initial model parameters
        learning_rate: Adam learning rate for ES gradient updates
        weight_decay: optional weight decay

    Returns:
        (optimizer, initial_state)
    """
    optimizer = optax.adam(learning_rate)
    if weight_decay > 0:
        optimizer = optax.chain(
            optax.adam(learning_rate),
            optax.add_decayed_weights(weight_decay),
        )
    opt_state = optimizer.init(params)
    return optimizer, EGGROLLState(opt_state)


def eggroll_step(
    base_params: PyTree,
    fitness_fn: Callable[[PyTree], Float[Array, ""]],
    optimizer: optax.GradientTransformation,
    state: EGGROLLState,
    *,
    key: jax.Array,
    population_size: int = 64,
    sigma: float = 0.001,
    rank: int = 1,
    antithetic: bool = True,
) -> tuple[PyTree, EGGROLLState, dict]:
    """Single EGGROLL update step with antithetic sampling.

    Evaluates a population of perturbed models, computes the ES
    gradient estimate, and applies an Adam update to base params.

    When antithetic=True (default), each perturbation direction E_i
    is evaluated at BOTH W+σE_i and W-σE_i. This doubles effective
    population size at minimal compute cost, halving gradient variance.
    Critical for small populations (pop=32 on 7M params).

    Args:
        base_params: current mean parameters M
        fitness_fn: f(params) → scalar fitness (higher = better)
        optimizer: optax optimizer for ES gradient
        state: current EGGROLL state
        key: PRNG key
        population_size: N directions (actual evals = 2N if antithetic)
        sigma: perturbation scale σ
        rank: perturbation rank r
        antithetic: use ±σ pairs (halves variance, recommended)

    Returns:
        (updated_params, new_state, metrics_dict)
    """
    keys = jax.random.split(key, population_size)

    # ── Evaluate population using jax.lax.map (JIT-friendly) ──────────
    # jax.lax.map applies a function sequentially over a leading axis
    # but unlike a Python for-loop, it compiles inside JIT and avoids
    # Python-level overhead and re-tracing per member.

    def eval_member(member_key: jax.Array):
        """Generate perturbation, evaluate fitness, return both."""
        perturbed, perturbation = perturb_pytree(base_params, member_key, sigma, rank)
        fitness = fitness_fn(perturbed)
        return fitness, perturbation

    if antithetic:
        # Antithetic sampling: evaluate both W+σE and W-σE per direction.
        # This halves gradient variance — critical for small populations.
        def eval_antithetic(member_key: jax.Array):
            """Evaluate ±σ pair, return fitness difference and perturbation."""
            _, perturbation = perturb_pytree(base_params, member_key, sigma, rank)
            # positive direction: W + σE
            pos_params = jax.tree.map(
                lambda p, e: p + sigma * e, base_params, perturbation
            )
            f_pos = fitness_fn(pos_params)
            # negative direction: W - σE
            neg_params = jax.tree.map(
                lambda p, e: p - sigma * e, base_params, perturbation
            )
            f_neg = fitness_fn(neg_params)
            return f_pos, f_neg, perturbation

        f_pos_arr, f_neg_arr, perturbations_stacked = jax.lax.map(eval_antithetic, keys)

        # antithetic ES gradient: (1/2σN) Σ (f+ - f-) * E
        diffs = f_pos_arr - f_neg_arr  # (N,)
        fitnesses_arr = (f_pos_arr + f_neg_arr) / 2  # for metrics

        def weighted_leaf_sum_anti(leaf_stack: jax.Array) -> jax.Array:
            w = diffs.reshape((-1,) + (1,) * (leaf_stack.ndim - 1))
            return jnp.sum(leaf_stack * w, axis=0)

        es_grad = jax.tree.map(weighted_leaf_sum_anti, perturbations_stacked)
        scale = 1.0 / (2.0 * sigma * population_size)
        es_grad = jax.tree.map(lambda g: g * scale, es_grad)
    else:
        # Original rank-based ES gradient (less efficient for small pop)
        fitnesses_arr, perturbations_stacked = jax.lax.map(eval_member, keys)

        ranks = jnp.argsort(jnp.argsort(fitnesses_arr)).astype(jnp.float32)
        normalized = 0.5 - ranks / (population_size - 1)

        def weighted_leaf_sum(leaf_stack: jax.Array) -> jax.Array:
            w = normalized.reshape((-1,) + (1,) * (leaf_stack.ndim - 1))
            return jnp.sum(leaf_stack * w, axis=0)

        es_grad = jax.tree.map(weighted_leaf_sum, perturbations_stacked)
        scale = 1.0 / (sigma * population_size)
        es_grad = jax.tree.map(lambda g: g * scale, es_grad)

    # negate gradient (ES maximizes fitness, optimizer minimizes)
    neg_grad = jax.tree.map(lambda g: -g, es_grad)

    # apply Adam update
    updates, new_opt_state = optimizer.update(neg_grad, state.opt_state, base_params)
    new_params = optax.apply_updates(base_params, updates)

    # metrics
    es_grad_leaves = jax.tree.leaves(es_grad)
    metrics = {
        "mean_fitness": jnp.mean(fitnesses_arr),
        "max_fitness": jnp.max(fitnesses_arr),
        "min_fitness": jnp.min(fitnesses_arr),
        "fitness_std": jnp.std(fitnesses_arr),
        "grad_norm": jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in es_grad_leaves)),
        "step": state.step,
    }

    new_state = EGGROLLState(new_opt_state, state.step + 1)
    return new_params, new_state, metrics


class SigmaAdaptor:
    """Dynamic adaptation for ES perturbation scale (sigma).

    Adjusts sigma to maintain a target population diversity
    (fitness_std / |mean_fitness|).

    Similar to the 1/5th rule in evolution strategies:
    - If diversity < target/2: increase sigma (more exploration)
    - If diversity > target*2: decrease sigma (more precision)
    """

    def __init__(
        self,
        initial_sigma: float = 0.001,
        target_diversity: float = 0.02,
        adjustment_rate: float = 1.05,
        min_sigma: float = 1e-5,
        max_sigma: float = 0.1,
    ) -> None:
        self.sigma = initial_sigma
        self.target_diversity = target_diversity
        self.rate = adjustment_rate
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def update(self, current_diversity: float) -> float:
        """Update and return the new sigma."""
        if current_diversity < self.target_diversity / 2.0:
            self.sigma = min(self.max_sigma, self.sigma * self.rate)
        elif current_diversity > self.target_diversity * 2.0:
            self.sigma = max(self.min_sigma, self.sigma / self.rate)
        return self.sigma


def discretize_update_int8(
    params_int8: PyTree,
    es_gradient: PyTree,
    threshold: float = 1.5,
) -> PyTree:
    """Apply discretized int8 update from EGGROLL paper.

    For integer parameters, Adam produces a real-valued proposal.
    We convert to sparse unit-step updates via z-score thresholding:

      z = (u - mean(u)) / (std(u) + 1e-8)
      Δ = sign(z) · 1{|z| ≥ τ} ∈ {-1, 0, +1}
      Q ← clip(Q + Δ, -127, 127)

    Args:
        params_int8: current int8 parameters
        es_gradient: real-valued ES gradient estimate
        threshold: z-score threshold τ (default 1.5)

    Returns:
        Updated int8 parameters
    """

    def update_leaf(param, grad):
        # z-score normalization
        mu = jnp.mean(grad)
        std = jnp.std(grad) + 1e-8
        z = (grad - mu) / std

        # sparse unit-step: only update where |z| exceeds threshold
        delta = jnp.sign(z) * (jnp.abs(z) >= threshold).astype(jnp.float32)

        # apply and clip to int8 range
        updated = param.astype(jnp.float32) + delta
        return jnp.clip(updated, -127, 127).astype(jnp.int8)

    return jax.tree.map(update_leaf, params_int8, es_gradient)
