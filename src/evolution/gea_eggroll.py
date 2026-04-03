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


def run_evolution(
    base_params: PyTree,
    fitness_fn: Callable,
    task_distribution: list[dict],
    eggroll_step_fn: Callable,
    *,
    key: jax.Array,
    num_iterations: int = 30,
    population_size: int = 64,
    group_size: int = 5,
    sigma: float = 0.001,
    rank: int = 1,
) -> tuple[PyTree, list[dict]]:
    """Run the full GEA-EGGROLL evolution loop.

    At each iteration:
      1. GEA evaluates population and selects parents
      2. Experience is aggregated from the parent group
      3. EGGROLL uses the experience to bias the next ES update
      4. Base params are updated via EGGROLL step

    Args:
        base_params: initial mean parameters
        fitness_fn: f(params, task) → (fitness, metrics)
        task_distribution: tasks to evaluate on
        eggroll_step_fn: EGGROLL weight update function
        key: PRNG key
        num_iterations: number of evolution iterations
        population_size: members per generation
        group_size: parents per group (K in GEA)
        sigma: EGGROLL perturbation scale
        rank: perturbation rank

    Returns:
        (final_params, history_of_experiences)
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
        history.append(experience)

        # EGGROLL: update base params using population fitness
        params = eggroll_step_fn(params, traces, key=eggroll_key, sigma=sigma)

        # log progress
        mean_fit = sum(t.mean_fitness for t in traces) / len(traces)
        best_fit = max(t.mean_fitness for t in traces)
        print(
            f"GEA iteration {i + 1}/{num_iterations} | "
            f"mean fitness: {mean_fit:.4f} | "
            f"best fitness: {best_fit:.4f} | "
            f"parents: {experience['parent_fitness']}"
        )

    return params, history
