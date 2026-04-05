"""
VELM Model Architecture — go-mHC Hyper-Connections

Manifold-Constrained Hyper-Connections via generalized orthostochastic
matrices (Dandachi & Diggs-Galligan, 2026).

Replaces single-stream residual connections with d-stream parallel
routing using exactly doubly stochastic mixing matrices.

Pipeline per layer:
  1. Learned skew-symmetric params X = -X^T ∈ R^{ds×ds}
  2. Cayley transform: Q = (I - X)(I + X)^{-1} → orthogonal Q
  3. Block Frobenius projection: B_ij = (1/s) ||Q_ij||_F^2 → doubly stochastic
  4. Multi-stream residual: x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

The doubly stochastic constraint is exact by construction — no iterations,
no approximation gap. Compatible with EGGROLL: any perturbation of the
skew-symmetric params produces a valid doubly stochastic matrix.

Reference: arXiv 2604.02309v1
"""

# ruff: noqa: F722, F821

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def skew_symmetric(params: Float[Array, "n_params"], n: int) -> Float[Array, "n n"]:
    """Map a flat vector to a skew-symmetric matrix.

    Args:
        params: flat vector of n*(n-1)/2 free parameters
        n: matrix dimension (ds in go-mHC)

    Returns:
        X = -X^T ∈ R^{n×n}
    """
    X = jnp.zeros((n, n))
    # fill upper triangle
    idx = jnp.triu_indices(n, k=1)
    X = X.at[idx].set(params)
    # skew-symmetric: lower triangle = -upper triangle
    return X - X.T


def cayley_transform(X: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Cayley transform: maps skew-symmetric → special orthogonal.

    Q = (I - X)(I + X)^{-1}

    Since X is skew-symmetric, (I + X) is always invertible and
    det(Q) = +1 (special orthogonal, pure rotation).

    Args:
        X: skew-symmetric matrix (X = -X^T)

    Returns:
        Q ∈ SO(n) orthogonal matrix
    """
    n = X.shape[0]
    I = jnp.eye(n)
    return jnp.linalg.solve(I + X, I - X)


def frobenius_block_projection(
    Q: Float[Array, "ds ds"], d: int, s: int,
) -> Float[Array, "d d"]:
    """Block-wise Frobenius norm projection → doubly stochastic.

    Φ_{d,s}(Q)_ij = (1/s) ||Q_ij||_F^2

    where Q_ij is the (i,j)-th s×s block of Q.

    Args:
        Q: orthogonal matrix in R^{ds×ds}
        d: number of residual streams
        s: block size (expressivity parameter)

    Returns:
        B ∈ B_d: exactly doubly stochastic d×d matrix
    """
    # reshape Q into d×d grid of s×s blocks
    Q_blocks = Q.reshape(d, s, d, s)  # (d, s, d, s)
    # ||Q_ij||_F^2 = sum of squared elements in each s×s block
    B = jnp.sum(Q_blocks ** 2, axis=(1, 3)) / s  # (d, d)
    return B


class GoMHCProjection(eqx.Module):
    """Generalized orthostochastic projection: params → doubly stochastic.

    Maps ds(ds-1)/2 free parameters to an exactly doubly stochastic d×d
    matrix via skew-symmetric → Cayley → block Frobenius.

    The projection is exact, differentiable, and any perturbation of the
    free parameters produces a valid doubly stochastic matrix — making it
    inherently compatible with EGGROLL's gradient-free perturbations.
    """

    d: int   # number of residual streams
    s: int   # expressivity parameter (block size)

    def __call__(
        self, params: Float[Array, "n_params"],
    ) -> Float[Array, "d d"]:
        """Project free parameters to doubly stochastic matrix.

        Args:
            params: flat vector of ds*(ds-1)/2 free parameters

        Returns:
            B ∈ B_d: exactly doubly stochastic d×d matrix
        """
        n = self.d * self.s
        X = skew_symmetric(params, n)   # skew-symmetric
        Q = cayley_transform(X)          # orthogonal
        B = frobenius_block_projection(Q, self.d, self.s)  # doubly stochastic
        return B

    @property
    def num_params(self) -> int:
        """Number of free parameters for this projection."""
        n = self.d * self.s
        return n * (n - 1) // 2


class HyperConnectionBlock(eqx.Module):
    """Wraps any backbone block with go-mHC d-stream routing.

    Replaces the standard residual connection:
      x = x + block(x)

    With d-stream hyper-connection:
      x_streams = H_res @ x_streams + H_post^T @ block(H_pre @ x_streams)

    where H_res is exactly doubly stochastic via go-mHC,
    H_pre aggregates d streams → 1 for the block input,
    H_post distributes block output → d streams.
    """

    # go-mHC projection for H_res
    projection: GoMHCProjection

    # learned skew-symmetric parameters for H_res
    res_params: Float[Array, "n_params"]

    # H_pre: (d,) weights for aggregating streams → 1
    pre_weights: Float[Array, "d"]

    # H_post: (d,) weights for distributing output → streams
    post_weights: Float[Array, "d"]

    d: int  # number of streams

    def __init__(self, d: int, s: int = 2, *, key: jax.Array) -> None:
        """Initialize hyper-connection block.

        Args:
            d: number of residual streams
            s: go-mHC expressivity parameter
            key: PRNG key
        """
        k1, k2, k3 = jax.random.split(key, 3)
        self.d = d
        self.projection = GoMHCProjection(d=d, s=s)

        # initialize res_params near zero → H_res starts near identity
        # (barycenter = 1/d * J_d, small params → near identity behavior)
        n_params = self.projection.num_params
        self.res_params = jax.random.normal(k1, (n_params,)) * 0.01

        # H_pre: start with uniform aggregation (1/d each)
        self.pre_weights = jnp.ones(d) / d

        # H_post: start with uniform distribution (2/d each, scaled by 2
        # following go-mHC paper Eq. 4 for post-mapping)
        self.post_weights = jnp.ones(d) * (2.0 / d)

    def mix_residual(
        self,
        x_streams: Float[Array, "d T dim"],
    ) -> Float[Array, "d T dim"]:
        """Apply H_res mixing across d streams.

        H_res is exactly doubly stochastic: information is conserved.
        """
        H_res = self.projection(self.res_params)  # (d, d)
        # mix across stream dimension: each (T, dim) slice gets mixed
        # H_res @ x_streams: matrix multiply on the d axis
        return jnp.einsum("ij,jTD->iTD", H_res, x_streams)

    def aggregate_for_block(
        self,
        x_streams: Float[Array, "d T dim"],
    ) -> Float[Array, "T dim"]:
        """H_pre: aggregate d streams → 1 for block input.

        Uses sigmoid-gated weights (Eq. 3 in go-mHC paper).
        """
        weights = jax.nn.sigmoid(self.pre_weights)  # (d,)
        # weighted sum across streams
        return jnp.einsum("i,iTD->TD", weights, x_streams)

    def distribute_from_block(
        self,
        block_output: Float[Array, "T dim"],
    ) -> Float[Array, "d T dim"]:
        """H_post: distribute block output → d streams.

        Uses 2*sigmoid-gated weights (Eq. 4 in go-mHC paper).
        """
        weights = 2.0 * jax.nn.sigmoid(self.post_weights)  # (d,)
        # broadcast block output to all d streams, weighted
        return jnp.einsum("i,TD->iTD", weights, block_output)


def init_streams(
    x: Float[Array, "T dim"], d: int,
) -> Float[Array, "d T dim"]:
    """Initialize d residual streams by replicating the input.

    Following Zhu et al. (2025a): x_0 = d repetitions of initial input.
    """
    return jnp.broadcast_to(x[None, :, :], (d, x.shape[0], x.shape[1]))


def collapse_streams(
    x_streams: Float[Array, "d T dim"],
) -> Float[Array, "T dim"]:
    """Collapse d streams to 1 by averaging (for final output).

    The doubly stochastic mixing ensures streams are balanced,
    so simple averaging is appropriate.
    """
    return jnp.mean(x_streams, axis=0)
