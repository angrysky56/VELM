"""
VELM Inference — CIB Budget Controller

Adaptive reasoning token budget based on Conditional Information Bottleneck.

Core idea: treat chain-of-thought as a compression problem. Monitor the
information gain per reasoning chunk and terminate early when marginal
gain drops below a threshold.

In VELM, each reasoning step generates K=4 tokens (one chunk), so budget
control is K× more impactful than in standard token-level models.

Two modes:
  - Static: fixed budget based on estimated problem difficulty
  - Dynamic: monitor info gain per step, terminate when diminishing
"""

# ruff: noqa: F722, F821

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class CIBBudgetController:
    """Adaptive reasoning budget based on information gain monitoring.

    Tracks the marginal information gain per reasoning chunk and
    terminates reasoning early when gain drops below threshold.

    In VELM, each step = K=4 tokens, so early termination saves
    K tokens per skipped step (4× more efficient than token-level).

    Attributes:
        max_chunks: hard ceiling on reasoning steps
        gain_threshold: minimum info gain to continue reasoning
        warmup_chunks: minimum steps before early termination allowed
        ema_decay: exponential moving average decay for gain tracking
    """

    max_chunks: int
    gain_threshold: float
    warmup_chunks: int
    ema_decay: float

    def __init__(
        self,
        max_chunks: int = 64,
        gain_threshold: float = 0.01,
        warmup_chunks: int = 4,
        ema_decay: float = 0.9,
    ) -> None:
        self.max_chunks = max_chunks
        self.gain_threshold = gain_threshold
        self.warmup_chunks = warmup_chunks
        self.ema_decay = ema_decay


def estimate_difficulty(
    hidden_state: Float[Array, "dim"],
) -> Float[Array, ""]:
    """Estimate query difficulty from the initial hidden state.

    Simple heuristic: higher norm = more complex input = more budget.
    Can be replaced with a learned estimator.

    Args:
        hidden_state: backbone hidden state from prompt encoding

    Returns:
        Scalar difficulty estimate in [0, 1]
    """
    norm = jnp.linalg.norm(hidden_state)
    # sigmoid normalization centered at typical norm
    return jax.nn.sigmoid((norm - 10.0) / 5.0)


def compute_info_gain(
    prev_hidden: Float[Array, "dim"],
    curr_hidden: Float[Array, "dim"],
) -> Float[Array, ""]:
    """Compute information gain between consecutive hidden states.

    Uses cosine distance as a proxy for new information introduced
    by the latest reasoning step. High similarity = low gain.

    Args:
        prev_hidden: hidden state from previous step
        curr_hidden: hidden state from current step

    Returns:
        Scalar info gain in [0, 2] (cosine distance)
    """
    cos_sim = jnp.dot(prev_hidden, curr_hidden) / (
        jnp.linalg.norm(prev_hidden) * jnp.linalg.norm(curr_hidden) + 1e-8
    )
    return 1.0 - cos_sim  # distance: 0 = identical, 2 = opposite


def should_continue_reasoning(
    controller: CIBBudgetController,
    step: int,
    info_gains: list[float],
) -> bool:
    """Decide whether to continue generating reasoning chunks.

    Args:
        controller: CIB budget controller configuration
        step: current reasoning step (0-indexed)
        info_gains: list of info gains from previous steps

    Returns:
        True if reasoning should continue, False to stop
    """
    # hard ceiling
    if step >= controller.max_chunks:
        return False

    # warmup: always continue for minimum steps
    if step < controller.warmup_chunks:
        return True

    # no gains recorded yet
    if len(info_gains) == 0:
        return True

    # compute EMA of recent info gains
    ema = info_gains[-1]
    for g in reversed(info_gains[:-1]):
        ema = controller.ema_decay * ema + (1 - controller.ema_decay) * g

    # stop if smoothed gain drops below threshold
    return ema > controller.gain_threshold


def allocate_static_budget(
    controller: CIBBudgetController,
    difficulty: float,
) -> int:
    """Allocate a fixed reasoning budget based on estimated difficulty.

    Simple linear scaling: harder problems get more budget.

    Args:
        controller: CIB budget controller
        difficulty: estimated difficulty in [0, 1]

    Returns:
        Number of reasoning chunks to allocate
    """
    min_budget = controller.warmup_chunks
    budget_range = controller.max_chunks - min_budget
    budget = int(min_budget + difficulty * budget_range)
    return min(budget, controller.max_chunks)
