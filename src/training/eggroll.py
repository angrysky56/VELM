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

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, PyTree
from typing import Callable


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
    if len(shape) == 1:
        # vector parameter: simple Gaussian perturbation
        return jax.random.normal(key, shape)

    elif len(shape) == 2:
        # matrix parameter: low-rank perturbation
        m, n = shape
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, (m, rank))
        b = jax.random.normal(k2, (n, rank))
        return (1.0 / jnp.sqrt(rank)) * (a @ b.T)
    else:
        # higher-order tensor: reshape → perturb → reshape back
        flat_shape = (shape[0], -1)
        total = 1
        for s in shape[1:]:
            total *= s
        flat = generate_low_rank_perturbation(key, (shape[0], total), rank)
        return flat.reshape(shape)


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
) -> tuple[PyTree, EGGROLLState, dict]:
    """Single EGGROLL update step.

    Evaluates a population of perturbed models, computes the ES
    gradient estimate, and applies an Adam update to base params.

    Args:
        base_params: current mean parameters M
        fitness_fn: f(params) → scalar fitness (higher = better)
        optimizer: optax optimizer for ES gradient
        state: current EGGROLL state
        key: PRNG key
        population_size: N workers
        sigma: perturbation scale σ
        rank: perturbation rank r

    Returns:
        (updated_params, new_state, metrics_dict)
    """
    keys = jax.random.split(key, population_size)

    # evaluate population: generate perturbations and compute fitness
    def eval_member(member_key: jax.Array):
        perturbed, perturbation = perturb_pytree(
            base_params, member_key, sigma, rank
        )
        fitness = fitness_fn(perturbed)
        return fitness, perturbation

    # sequential evaluation (vmap requires fixed structure)
    # for large populations on GPU, this should be parallelized
    # via the batched low-rank trick from EGGROLL paper
    fitnesses = []
    perturbations = []
    for i in range(population_size):
        f, e = eval_member(keys[i])
        fitnesses.append(f)
        perturbations.append(e)

    fitnesses_arr = jnp.array(fitnesses)  # (N,)

    # fitness normalization: rank-based (more robust than raw fitness)
    # map fitnesses to [-0.5, 0.5] based on rank
    ranks = jnp.argsort(jnp.argsort(fitnesses_arr)).astype(jnp.float32)
    normalized = 0.5 - ranks / (population_size - 1)  # best=0.5, worst=-0.5

    # ES gradient estimate: ∇M ≈ (1/σN) Σ E_i * f_i
    # weighted sum of perturbations
    def weighted_sum(leaves_list: list, weights: jax.Array) -> list:
        """Compute weighted sum across population for each leaf."""
        result = []
        num_leaves = len(leaves_list[0])
        for j in range(num_leaves):
            leaf_sum = jnp.zeros_like(leaves_list[0][j])
            for i in range(len(leaves_list)):
                leaf_sum = leaf_sum + weights[i] * leaves_list[i][j]
            result.append(leaf_sum)
        return result

    # flatten all perturbations for weighted sum
    perturbation_leaves = [jax.tree.leaves(p) for p in perturbations]

    # compute weighted sum → ES gradient estimate
    es_grad_leaves = weighted_sum(perturbation_leaves, normalized)
    scale = 1.0 / (sigma * population_size)
    es_grad_leaves = [leaf * scale for leaf in es_grad_leaves]

    # reconstruct gradient pytree
    _, treedef = jax.tree.flatten(base_params)
    es_grad = jax.tree.unflatten(treedef, es_grad_leaves)

    # negate gradient (ES maximizes fitness, optimizer minimizes)
    neg_grad = jax.tree.map(lambda g: -g, es_grad)

    # apply Adam update
    updates, new_opt_state = optimizer.update(neg_grad, state.opt_state, base_params)
    new_params = optax.apply_updates(base_params, updates)

    # metrics
    metrics = {
        "mean_fitness": jnp.mean(fitnesses_arr),
        "max_fitness": jnp.max(fitnesses_arr),
        "min_fitness": jnp.min(fitnesses_arr),
        "fitness_std": jnp.std(fitnesses_arr),
        "grad_norm": jnp.sqrt(sum(jnp.sum(l**2) for l in es_grad_leaves)),
        "step": state.step,
    }

    new_state = EGGROLLState(new_opt_state, state.step + 1)
    return new_params, new_state, metrics


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
