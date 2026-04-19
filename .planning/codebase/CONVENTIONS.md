# Conventions

- **Immutable State**: Use `equinox.Module` for all model components. Parameters are immutable; updates produce new PyTrees.
- **Functional JAX**: Prefer pure functions and `jax.vmap`/`jax.lax.map` over Python loops.
- **Strict Typing**: Use `jaxtyping` for tensor shapes (e.g., `Float[Array, "batch dim"]`) and standard type hints for all signatures.
- **Randomness**: Explicitly manage `jax.random.PRNGKey` state. Split keys immediately before use.
- **Error Handling**: Use `jnp.clip` and stability-gated logic to prevent `NaN` in evolutionary loops.
- **Documentation**: All core modules must include a summary of the architecture or algorithm they implement at the top of the file.
- **Linting**: Follow Ruff defaults with Python 3.12 target.
