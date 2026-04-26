# Concerns

- **EGGROLL Recursion Bug**: `generate_low_rank_perturbation` in `src/training/eggroll.py` has a recursive call for higher-order tensors that triggers `RecursionError` during JAX tracing. A fix using `math.prod` for static shape calculation is currently only in the Colab notebooks.
- **Source/Notebook Drift**: Critical training patches are often applied in the Colab environment first, leading to drift with the `src/` directory.
- **Hardware Profile Hardcoding**: Batch sizes and step counts are often hardcoded for T4/A100 profiles, which may not generalize well to other GPUs without manual adjustment.
- **Population Memory Usage**: Scaling EGGROLL to very large populations (&gt;2^10) risks OOM in JAX if not carefully managed with `jax.lax.map`.
- **Data Latency**: High reliance on HuggingFace streaming means training speed is coupled to network performance.
