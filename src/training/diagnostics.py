"""
VELM Training — EGGROLL Diagnostics & Observability

Tracking tools for debugging the black-box evolutionary optimizer.
Since EGGROLL has no gradients to inspect, we monitor:
  - Fitness trajectories and plateau detection
  - Parameter drift magnitude per pytree leaf
  - Population diversity (fitness spread, parameter spread)
  - NaN/Inf sentinels

Usage:
    diag = EGGROLLDiagnostics(plateau_window=50, plateau_threshold=0.001)
    for step in range(num_steps):
        params, state, metrics = eggroll_step(...)
        diag.log_step(step, params, metrics)
        if diag.is_plateaued:
            print("WARNING: fitness plateau detected")
    diag.report()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jaxtyping import PyTree


@dataclass
class EGGROLLDiagnostics:
    """Observability harness for EGGROLL training runs.

    Tracks fitness curves, parameter drift, population health,
    and detects pathological conditions (NaN, plateaus, collapse).

    Attributes:
        plateau_window: number of steps to look back for plateau detection
        plateau_threshold: minimum fitness improvement to not be "plateaued"
        history: per-step metrics log
        is_plateaued: whether a plateau has been detected
        nan_count: number of NaN fitness values encountered
    """

    plateau_window: int = 50
    plateau_threshold: float = 0.001
    history: list[dict] = field(default_factory=list)
    _initial_params: PyTree | None = None
    is_plateaued: bool = False
    nan_count: int = 0

    def log_step(
        self,
        step: int,
        params: PyTree,
        metrics: dict,
    ) -> dict:
        """Record diagnostics for one EGGROLL step.

        Args:
            step: current training step
            params: current mean parameters
            metrics: dict from eggroll_step (mean_fitness, max_fitness, etc.)

        Returns:
            Diagnostics dict with computed health indicators
        """
        mean_fit = float(metrics.get("mean_fitness", 0.0))
        max_fit = float(metrics.get("max_fitness", 0.0))
        min_fit = float(metrics.get("min_fitness", 0.0))
        fit_std = float(metrics.get("fitness_std", 0.0))
        grad_norm = float(metrics.get("grad_norm", 0.0))

        # NaN detection
        has_nan = not (
            jnp.isfinite(jnp.array(mean_fit))
            and jnp.isfinite(jnp.array(max_fit))
        )
        if has_nan:
            self.nan_count += 1

        # parameter norms per leaf
        param_norms = _compute_param_norms(params)

        # store initial parameters for true displacement computation
        if step == 1 or not self._initial_params:
            self._initial_params = params

        # parameter drift: Euclidean distance from initialization normalized by initial norm
        drift = {}
        curr_leaves = jax.tree.leaves(params)
        init_leaves = jax.tree.leaves(self._initial_params)
        paths = _get_leaf_paths(params)

        for path, curr, init in zip(paths, curr_leaves, init_leaves):
            if hasattr(curr, "shape"):
                dist = float(jnp.sqrt(jnp.sum((curr - init)**2)))
                init_norm = float(jnp.sqrt(jnp.sum(init**2)))
                drift[path] = dist / (init_norm + 1e-8)

        # population diversity: fitness spread
        diversity = fit_std / (abs(mean_fit) + 1e-8)

        entry = {
            "step": step,
            "mean_fitness": mean_fit,
            "max_fitness": max_fit,
            "min_fitness": min_fit,
            "fitness_std": fit_std,
            "grad_norm": grad_norm,
            "has_nan": has_nan,
            "diversity": diversity,
            "max_drift": max(drift.values()) if drift else 0.0,
            "param_drift": drift,
        }
        self.history.append(entry)

        # plateau detection
        self._check_plateau()

        return entry

    def _check_plateau(self) -> None:
        """Detect if fitness has stalled over the plateau window."""
        if len(self.history) < self.plateau_window:
            self.is_plateaued = False
            return

        recent = self.history[-self.plateau_window:]
        fitnesses = [e["mean_fitness"] for e in recent]
        improvement = max(fitnesses) - min(fitnesses)
        self.is_plateaued = improvement < self.plateau_threshold

    def report(self, last_n: int = 10) -> str:
        """Generate a diagnostic summary report.

        Args:
            last_n: number of recent steps to show in detail

        Returns:
            Formatted report string (also printed)
        """
        if not self.history:
            msg = "No steps logged yet."
            print(msg)
            return msg

        lines = []
        lines.append("=" * 70)
        lines.append("EGGROLL Diagnostics Report")
        lines.append("=" * 70)

        total_steps = len(self.history)
        lines.append(f"Total steps logged: {total_steps}")
        lines.append(f"NaN events: {self.nan_count}")
        lines.append(f"Plateau detected: {self.is_plateaued}")

        # fitness trajectory summary
        all_fits = [e["mean_fitness"] for e in self.history]
        lines.append("\nFitness trajectory:")
        lines.append(f"  First:  {all_fits[0]:.6f}")
        lines.append(f"  Last:   {all_fits[-1]:.6f}")
        lines.append(f"  Best:   {max(all_fits):.6f}")
        lines.append(f"  Worst:  {min(all_fits):.6f}")
        lines.append(f"  Change: {all_fits[-1] - all_fits[0]:+.6f}")

        # diversity trend
        all_div = [e["diversity"] for e in self.history]
        lines.append("\nPopulation diversity (fitness_std / |mean|):")
        lines.append(f"  First:  {all_div[0]:.4f}")
        lines.append(f"  Last:   {all_div[-1]:.4f}")
        if all_div[-1] < 0.01:
            lines.append("  ⚠ LOW DIVERSITY — population may have collapsed")

        # parameter drift
        last = self.history[-1]
        if last["param_drift"]:
            lines.append("\nParameter drift from initialization:")
            sorted_drift = sorted(
                last["param_drift"].items(), key=lambda x: x[1], reverse=True
            )
            for name, d in sorted_drift[:5]:
                bar = "█" * min(int(d * 50), 50)
                lines.append(f"  {name:40s} {d:8.4f} {bar}")

        # recent steps detail
        lines.append(f"\nLast {min(last_n, total_steps)} steps:")
        lines.append(
            f"  {'step':>6s}  {'mean_f':>10s}  {'max_f':>10s}  "
            f"{'grad':>10s}  {'div':>8s}  {'drift':>8s}"
        )
        for entry in self.history[-last_n:]:
            lines.append(
                f"  {entry['step']:6d}  {entry['mean_fitness']:10.4f}  "
                f"{entry['max_fitness']:10.4f}  {entry['grad_norm']:10.4f}  "
                f"{entry['diversity']:8.4f}  {entry['max_drift']:8.4f}"
            )

        report_text = "\n".join(lines)
        print(report_text)
        return report_text

    @property
    def fitness_trend(self) -> list[float]:
        """Return the fitness trajectory as a simple list."""
        return [e["mean_fitness"] for e in self.history]

    @property
    def warnings(self) -> list[str]:
        """Return active warning messages."""
        warns = []
        if self.nan_count > 0:
            warns.append(f"NaN fitness detected {self.nan_count} times")
        if self.is_plateaued:
            warns.append(
                f"Fitness plateaued (Δ < {self.plateau_threshold} "
                f"over last {self.plateau_window} steps)"
            )
        if self.history:
            last = self.history[-1]
            if last["diversity"] < 0.01:
                warns.append("Population diversity critically low — possible collapse")
            if last["max_drift"] > 10.0:
                warns.append(
                    f"Parameter drift very large ({last['max_drift']:.1f}x) "
                    f"— possible instability"
                )
        return warns


def _compute_param_norms(params: PyTree) -> dict[str, float]:
    """Compute L2 norm for each leaf in a parameter pytree.

    Returns a dict mapping leaf path → L2 norm.
    """
    leaves, treedef = jax.tree.flatten(params)
    # reconstruct paths from treedef
    paths = _get_leaf_paths(params)

    norms = {}
    for path, leaf in zip(paths, leaves):
        if hasattr(leaf, "shape"):
            norms[path] = float(jnp.sqrt(jnp.sum(leaf**2)))
    return norms


def _get_leaf_paths(tree: PyTree, prefix: str = "") -> list[str]:
    """Extract string paths for each leaf in a pytree."""
    paths = []
    if isinstance(tree, dict):
        for key in sorted(tree.keys()):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_get_leaf_paths(tree[key], child_prefix))
    elif isinstance(tree, (list, tuple)):
        for i, child in enumerate(tree):
            child_prefix = f"{prefix}[{i}]"
            paths.extend(_get_leaf_paths(child, child_prefix))
    else:
        # leaf node
        leaf_name = prefix if prefix else "param"
        paths.append(leaf_name)
    return paths
