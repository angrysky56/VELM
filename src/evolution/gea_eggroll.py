"""
VELM Evolution — GEA-EGGROLL Integration

Group-Evolving Agents adapted for weight-level evolution of VELM populations.

Bridges two frameworks:
  - EGGROLL: maintains population of weight perturbations, evaluates fitness
  - GEA: enables experience sharing across population members

Evolution cycle:
  1. EGGROLL evaluates population on diverse task distribution
  2. Collect evolutionary traces per member
  3. GEA reflection: analyze traces across the group
  4. GEA evolution: bias next-generation perturbations
  5. Select parent group for next iteration

Selection criterion:
  score(i) = α_i × √nov(i)
"""

# ruff: noqa: F722, F821

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from ..training.eggroll import perturb_pytree


@dataclass
class EvolutionTrace:
    """Records what happened during one population member's evaluation.

    Captures per-member performance, reasoning behavior, and the
    perturbation seed needed to reconstruct the weight delta.
    """

    member_id: int
    perturbation_seed: int
    fitness_scores: dict[str, float] = field(default_factory=dict)
    reasoning_lengths: dict[str, float] = field(default_factory=dict)
    attention_mass: dict[str, float] = field(default_factory=dict)

    @property
    def mean_fitness(self) -> float:
        """Average fitness across all task types."""
        if not self.fitness_scores:
            return 0.0
        return sum(self.fitness_scores.values()) / len(self.fitness_scores)

    @property
    def mean_reasoning_length(self) -> float:
        """Average reasoning chunks used across tasks."""
        if not self.reasoning_lengths:
            return 0.0
        vals = self.reasoning_lengths.values()
        return sum(vals) / len(vals)


def compute_novelty(
    embeddings: Float[Array, "N dim"],
    num_neighbors: int = 5,
) -> Float[Array, "N"]:
    """Compute novelty score for each member via k-nearest neighbor distance.

    Novelty = mean cosine distance to M nearest neighbors.
    Higher novelty = more unique perturbation direction.

    Args:
        embeddings: (N, dim) embedding per population member
        num_neighbors: M nearest neighbors

    Returns:
        (N,) novelty scores
    """
    # cosine similarity matrix
    norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    sim_matrix = normed @ normed.T  # (N, N)

    # convert to distance
    dist_matrix = 1.0 - sim_matrix  # (N, N)

    # for each member, find M nearest neighbors (excluding self)
    # set self-distance to infinity
    dist_matrix = dist_matrix + jnp.eye(dist_matrix.shape[0]) * 1e9

    # sort distances and take mean of M smallest
    sorted_dists = jnp.sort(dist_matrix, axis=1)
    neighbor_dists = sorted_dists[:, :num_neighbors]  # (N, M)

    return jnp.mean(neighbor_dists, axis=1)  # (N,)


def performance_novelty_selection(
    fitness_scores: Float[Array, "N"],
    novelty_scores: Float[Array, "N"],
    group_size: int,
) -> list[int]:
    """Select parent group using Performance-Novelty criterion.

    score(i) = fitness(i) × √novelty(i)

    Performance is the primary criterion; novelty provides a mild
    exploration bias without dominating.

    Args:
        fitness_scores: (N,) per-member fitness
        novelty_scores: (N,) per-member novelty
        group_size: K — number of parents to select

    Returns:
        List of selected member indices
    """
    combined = fitness_scores * jnp.sqrt(novelty_scores + 1e-8)
    top_k = jnp.argsort(combined)[::-1][:group_size]
    return top_k.tolist()


@dataclass
class GroupEvolver:
    """Orchestrates GEA-style group evolution over EGGROLL populations.

    At each iteration:
      1. Evaluate all population members on task distribution
      2. Collect evolutionary traces
      3. Select parent group via Performance-Novelty
      4. Share experience: aggregate traces from parent group
      5. Generate offspring: bias perturbations toward promising directions
    """

    population_size: int = 64
    group_size: int = 5
    novelty_neighbors: int = 5
    archive: list[EvolutionTrace] = field(default_factory=list)
    iteration: int = 0

    def evaluate_population(
        self,
        base_params: PyTree,
        fitness_fn: Callable,
        task_distribution: list[dict],
        *,
        key: jax.Array,
        sigma: float = 0.001,
        rank: int = 1,
    ) -> list[EvolutionTrace]:
        """Evaluate all population members on the task distribution.

        Args:
            base_params: current mean parameters M
            fitness_fn: f(params, task) → (fitness, metrics)
            task_distribution: list of task dicts to evaluate on
            key: PRNG key
            sigma: EGGROLL perturbation scale
            rank: perturbation rank

        Returns:
            List of EvolutionTrace for each member
        """
        traces = []
        member_keys = jax.random.split(key, self.population_size)

        for i in range(self.population_size):
            seed = int(member_keys[i][0])
            perturbed, _ = perturb_pytree(base_params, member_keys[i], sigma, rank)

            trace = EvolutionTrace(member_id=i, perturbation_seed=seed)

            # evaluate on each task type
            for task in task_distribution:
                task_type = task.get("type", "default")
                fitness, metrics = fitness_fn(perturbed, task)
                trace.fitness_scores[task_type] = float(fitness)
                if "num_chunks" in metrics:
                    trace.reasoning_lengths[task_type] = float(metrics["num_chunks"])

            traces.append(trace)

        self.archive.extend(traces)
        return traces

    def select_parents(
        self,
        traces: list[EvolutionTrace],
        embeddings: Float[Array, "N dim"],
    ) -> list[int]:
        """Select parent group from current population.

        Args:
            traces: evaluation traces for current population
            embeddings: (N, dim) parameter embeddings for novelty

        Returns:
            Indices of selected parent members
        """
        fitnesses = jnp.array([t.mean_fitness for t in traces])
        novelties = compute_novelty(embeddings, self.novelty_neighbors)

        return performance_novelty_selection(fitnesses, novelties, self.group_size)

    def aggregate_experience(
        self,
        traces: list[EvolutionTrace],
        parent_indices: list[int],
    ) -> dict:
        """Aggregate evolutionary experience from parent group.

        Collects insights from the selected parents:
          - Which task types each parent excels at
          - How much reasoning budget each uses
          - Which perturbation seeds led to improvements

        Args:
            traces: all population traces
            parent_indices: indices of selected parents

        Returns:
            Aggregated experience dict
        """
        parent_traces = [traces[i] for i in parent_indices]

        # find best-performing member per task type
        task_champions: dict[str, int] = {}
        for trace in parent_traces:
            for task_type, fitness in trace.fitness_scores.items():
                if task_type not in task_champions:
                    task_champions[task_type] = trace.member_id
                else:
                    champ_trace = traces[task_champions[task_type]]
                    if fitness > champ_trace.fitness_scores.get(
                        task_type, -float("inf")
                    ):
                        task_champions[task_type] = trace.member_id

        # compute average reasoning budget across parents
        avg_reasoning = {}
        for trace in parent_traces:
            for task_type, length in trace.reasoning_lengths.items():
                if task_type not in avg_reasoning:
                    avg_reasoning[task_type] = []
                avg_reasoning[task_type].append(length)

        avg_reasoning = {k: sum(v) / len(v) for k, v in avg_reasoning.items()}

        # collect successful perturbation seeds
        successful_seeds = [t.perturbation_seed for t in parent_traces]

        return {
            "task_champions": task_champions,
            "avg_reasoning_budget": avg_reasoning,
            "successful_seeds": successful_seeds,
            "parent_fitness": [t.mean_fitness for t in parent_traces],
            "iteration": self.iteration,
        }

    def evolution_step(
        self,
        base_params: PyTree,
        fitness_fn: Callable,
        task_distribution: list[dict],
        *,
        key: jax.Array,
        sigma: float = 0.001,
        rank: int = 1,
    ) -> tuple[dict, list[EvolutionTrace]]:
        """Run one full GEA evolution iteration.

        Args:
            base_params: current mean parameters
            fitness_fn: f(params, task) → (fitness, metrics)
            task_distribution: tasks to evaluate on
            key: PRNG key
            sigma: EGGROLL perturbation scale
            rank: perturbation rank

        Returns:
            (experience_dict, traces)
        """
        k1, _ = jax.random.split(key)

        # 1. evaluate population
        traces = self.evaluate_population(
            base_params,
            fitness_fn,
            task_distribution,
            key=k1,
            sigma=sigma,
            rank=rank,
        )

        # 2. compute embeddings for novelty (use fitness profile)
        # simple embedding: concatenate fitness scores across tasks
        task_types = sorted(set().union(*(t.fitness_scores.keys() for t in traces)))

        embeddings = jnp.array(
            [[t.fitness_scores.get(tt, 0.0) for tt in task_types] for t in traces]
        )

        # 3. select parent group
        parent_indices = self.select_parents(traces, embeddings)

        # 4. aggregate experience from parents
        experience = self.aggregate_experience(traces, parent_indices)

        self.iteration += 1
        return experience, traces


def experience_weighted_eggroll_step(
    base_params: PyTree,
    traces: list[EvolutionTrace],
    *,
    key: jax.Array,
    optimizer: "optax.GradientTransformation",
    opt_state: "optax.OptState",
    sigma: float = 0.001,
    rank: int = 1,
    parent_indices: list[int] | None = None,
    parent_bias: float = 0.3,
) -> tuple[PyTree, "optax.OptState", dict]:
    """Bridge between GEA traces and EGGROLL weight updates.

    Reconstructs perturbation directions from population seeds,
    computes ES gradient weighted by fitness, and optionally biases
    the gradient toward parent-group directions.

    Algorithm:
      1. Reconstruct each member's perturbation from their seed
      2. Compute fitness-weighted ES gradient (standard EGGROLL)
      3. If parent_indices given, add a bias term toward parent
         perturbation directions (experience-guided exploration)
      4. Apply optax Adam update

    Args:
        base_params: current mean parameters M
        traces: EvolutionTrace list from GEA evaluation
        key: PRNG key
        optimizer: optax optimizer (e.g., Adam)
        opt_state: optimizer state
        sigma: perturbation scale (must match evaluation)
        rank: perturbation rank (must match evaluation)
        parent_indices: indices of GEA-selected parents (for bias)
        parent_bias: strength of bias toward parent directions [0, 1]

    Returns:
        (updated_params, new_opt_state, metrics_dict)
    """
    import optax  # deferred to avoid circular imports

    population_size = len(traces)

    # 1. reconstruct perturbation directions from seeds
    fitnesses = []
    perturbations = []

    for trace in traces:
        member_key = jax.random.PRNGKey(trace.perturbation_seed)
        _, perturbation = perturb_pytree(base_params, member_key, sigma, rank)
        perturbations.append(perturbation)
        fitnesses.append(trace.mean_fitness)

    fitnesses_arr = jnp.array(fitnesses)

    # guard against all-NaN fitness
    valid_mask = jnp.isfinite(fitnesses_arr)
    safe_fitnesses = jnp.where(valid_mask, fitnesses_arr, -1e6)

    # 2. rank-based fitness normalization (robust to outliers)
    ranks = jnp.argsort(jnp.argsort(safe_fitnesses)).astype(jnp.float32)
    normalized = 0.5 - ranks / (population_size - 1)  # best → 0.5, worst → -0.5

    # 3. ES gradient: fitness-weighted sum of perturbation directions
    def weighted_sum_leaf(*leaves):
        """Sum perturbation leaves weighted by normalized fitness."""
        stacked = jnp.stack(leaves, axis=0)  # (N, *shape)
        w = normalized.reshape((-1,) + (1,) * (stacked.ndim - 1))
        return jnp.sum(stacked * w, axis=0)

    leaf_lists = [jax.tree.leaves(p) for p in perturbations]
    # transpose: list-of-pytrees → list-of-leaf-lists, then zip
    es_grad_leaves = [
        weighted_sum_leaf(*[ll[j] for ll in leaf_lists])
        for j in range(len(leaf_lists[0]))
    ]
    es_grad = jax.tree.unflatten(
        jax.tree.structure(perturbations[0]), es_grad_leaves
    )

    # scale by 1/(σ·N)
    scale = 1.0 / (sigma * population_size)
    es_grad = jax.tree.map(lambda g: g * scale, es_grad)

    # 4. parent bias: add a gentle pull toward parent directions
    if parent_indices and parent_bias > 0:
        parent_perturbations = [perturbations[i] for i in parent_indices]

        def mean_leaf(*leaves):
            return jnp.mean(jnp.stack(leaves, axis=0), axis=0)

        parent_leaves = [jax.tree.leaves(p) for p in parent_perturbations]
        parent_mean_leaves = [
            mean_leaf(*[pl[j] for pl in parent_leaves])
            for j in range(len(parent_leaves[0]))
        ]
        parent_direction = jax.tree.unflatten(
            jax.tree.structure(parent_perturbations[0]), parent_mean_leaves
        )

        # blend: grad = (1 - bias) * es_grad + bias * parent_direction
        es_grad = jax.tree.map(
            lambda g, p: (1.0 - parent_bias) * g + parent_bias * p,
            es_grad, parent_direction,
        )

    # 5. negate (ES maximizes fitness, optimizer minimizes)
    neg_grad = jax.tree.map(lambda g: -g, es_grad)

    # 6. apply Adam update
    updates, new_opt_state = optimizer.update(neg_grad, opt_state, base_params)
    new_params = optax.apply_updates(base_params, updates)

    # metrics
    es_grad_leaves_flat = jax.tree.leaves(es_grad)
    metrics = {
        "mean_fitness": float(jnp.mean(safe_fitnesses)),
        "max_fitness": float(jnp.max(safe_fitnesses)),
        "min_fitness": float(jnp.min(safe_fitnesses)),
        "fitness_std": float(jnp.std(safe_fitnesses)),
        "grad_norm": float(
            jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in es_grad_leaves_flat))
        ),
        "valid_members": int(jnp.sum(valid_mask)),
        "parent_bias_applied": parent_indices is not None,
    }
    return new_params, new_opt_state, metrics


def run_evolution(
    base_params: PyTree,
    fitness_fn: Callable,
    task_distribution: list[dict],
    *,
    key: jax.Array,
    optimizer: "optax.GradientTransformation",
    opt_state: "optax.OptState",
    num_iterations: int = 30,
    population_size: int = 64,
    group_size: int = 5,
    sigma: float = 0.001,
    rank: int = 1,
    parent_bias: float = 0.3,
) -> tuple[PyTree, "optax.OptState", list[dict]]:
    """Run the full GEA-EGGROLL evolution loop.

    At each iteration:
      1. GEA evaluates population and selects parents
      2. Experience is aggregated from the parent group
      3. EGGROLL applies an experience-biased ES update
      4. Base params are updated via Adam

    Args:
        base_params: initial mean parameters
        fitness_fn: f(params, task) → (fitness, metrics)
        task_distribution: tasks to evaluate on
        key: PRNG key
        optimizer: optax optimizer for ES gradient
        opt_state: optimizer state
        num_iterations: number of evolution iterations
        population_size: members per generation
        group_size: parents per group (K in GEA)
        sigma: EGGROLL perturbation scale
        rank: perturbation rank
        parent_bias: strength of parent-direction bias [0, 1]

    Returns:
        (final_params, final_opt_state, history_of_experiences)
    """
    evolver = GroupEvolver(
        population_size=population_size,
        group_size=group_size,
    )

    history = []
    params = base_params

    for i in range(num_iterations):
        key, iter_key, eggroll_key = jax.random.split(key, 3)

        # GEA: evaluate, select, aggregate
        experience, traces = evolver.evolution_step(
            params,
            fitness_fn,
            task_distribution,
            key=iter_key,
            sigma=sigma,
            rank=rank,
        )

        # extract parent indices from experience
        parent_indices = list(experience.get("task_champions", {}).values())
        # deduplicate while preserving order
        seen = set()
        unique_parents = []
        for idx in parent_indices:
            if idx not in seen:
                seen.add(idx)
                unique_parents.append(idx)

        # EGGROLL: experience-biased weight update
        params, opt_state, step_metrics = experience_weighted_eggroll_step(
            params,
            traces,
            key=eggroll_key,
            optimizer=optimizer,
            opt_state=opt_state,
            sigma=sigma,
            rank=rank,
            parent_indices=unique_parents if unique_parents else None,
            parent_bias=parent_bias,
        )

        experience["step_metrics"] = step_metrics
        history.append(experience)

        # log progress
        mean_fit = step_metrics["mean_fitness"]
        best_fit = step_metrics["max_fitness"]
        n_valid = step_metrics["valid_members"]
        print(
            f"GEA iteration {i + 1}/{num_iterations} | "
            f"mean: {mean_fit:.4f} | best: {best_fit:.4f} | "
            f"valid: {n_valid}/{population_size} | "
            f"parents: {unique_parents}"
        )

    return params, opt_state, history
