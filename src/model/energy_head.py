"""
VELM Model Architecture — Energy-Based Generative Head

Predicts the next continuous vector z_i from the backbone hidden state h_{i-1}.

Architecture (from CALM):
  Input: hidden state h ∈ R^d + random noise ε ~ U[-0.5, 0.5]^{d_noise}
  Both projected via independent linear layers to internal dim d
  Stack of L residual MLP blocks (SwiGLU + residual)
  Each block fuses current representation ε_l with hidden state h
  Final linear projection to latent dim l

Trained with energy loss (strictly proper scoring rule).
The head accounts for ~10% of total model parameters.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class ResidualMLPBlock(eqx.Module):
    """Single residual block in the generative head.

    Fuses the current noise representation ε_l with the hidden state h,
    then applies SwiGLU + residual connection.
    """

    proj_eps: eqx.nn.Linear
    proj_h: eqx.nn.Linear
    w_gate: eqx.nn.Linear
    w_up: eqx.nn.Linear
    w_down: eqx.nn.Linear

    def __init__(self, dim: int, intermediate: int, *, key: jax.Array) -> None:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        # fusion projections: bring eps and h into the same space
        self.proj_eps = eqx.nn.Linear(dim, dim, use_bias=False, key=k1)
        self.proj_h = eqx.nn.Linear(dim, dim, use_bias=False, key=k2)
        # SwiGLU layers
        self.w_gate = eqx.nn.Linear(dim, intermediate, use_bias=False, key=k3)
        self.w_up = eqx.nn.Linear(dim, intermediate, use_bias=False, key=k4)
        self.w_down = eqx.nn.Linear(intermediate, dim, use_bias=False, key=k5)

    def __call__(
        self,
        eps_l: Float[Array, "dim"],
        h: Float[Array, "dim"],
    ) -> Float[Array, "dim"]:
        """Fuse noise representation with hidden state, apply SwiGLU + residual.

        Args:
            eps_l: current noise representation at this block
            h: backbone hidden state (constant across blocks)

        Returns:
            eps_{l+1}: refined noise representation
        """
        # fuse eps_l and h via two independent linear projections + addition
        fused = self.proj_eps(eps_l) + self.proj_h(h)

        # SwiGLU
        out = self.w_down(jax.nn.silu(self.w_gate(fused)) * self.w_up(fused))

        # residual connection
        return eps_l + out


class EnergyHead(eqx.Module):
    """Energy-based generative head for continuous next-vector prediction.

    Takes a backbone hidden state h and generates a continuous vector z
    by progressively refining random noise through L residual MLP blocks.

    The head is lightweight: L = num_backbone_layers / 4, accounting for
    ~10% of total model parameters.
    """

    proj_noise: eqx.nn.Linear
    proj_hidden: eqx.nn.Linear
    blocks: list[ResidualMLPBlock]
    to_latent: eqx.nn.Linear
    num_blocks: int
    noise_dim: int
    internal_dim: int
    latent_dim: int

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        num_blocks: int = 3,
        noise_dim: int | None = None,
        ffn_intermediate: int | None = None,
        *,
        key: jax.Array,
    ) -> None:
        """Initialize the energy-based generative head.

        Args:
            hidden_dim: backbone hidden dimension (d)
            latent_dim: output latent dimension (l)
            num_blocks: number of residual MLP blocks (L)
            noise_dim: dimension of input noise (default: hidden_dim)
            ffn_intermediate: SwiGLU intermediate dim (default: hidden_dim)
            key: PRNG key
        """
        noise_dim = noise_dim or hidden_dim
        ffn_intermediate = ffn_intermediate or hidden_dim
        self.num_blocks = num_blocks
        self.noise_dim = noise_dim
        self.internal_dim = hidden_dim
        self.latent_dim = latent_dim

        keys = jax.random.split(key, num_blocks + 3)

        # project noise and hidden state to internal dim
        self.proj_noise = eqx.nn.Linear(noise_dim, hidden_dim, use_bias=False, key=keys[0])
        self.proj_hidden = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=keys[1])

        # stack of residual MLP blocks
        self.blocks = [
            ResidualMLPBlock(hidden_dim, ffn_intermediate, key=keys[i + 2])
            for i in range(num_blocks)
        ]

        # final projection to latent dim
        self.to_latent = eqx.nn.Linear(hidden_dim, latent_dim, use_bias=False, key=keys[-1])

    def __call__(
        self,
        h: Float[Array, "dim"],
        *,
        key: jax.Array,
        num_samples: int = 1,
    ) -> Float[Array, "S latent"]:
        """Generate continuous vector samples from hidden state.

        Args:
            h: (hidden_dim,) backbone hidden state
            key: PRNG key for noise sampling
            num_samples: number of samples to generate (N in energy loss)

        Returns:
            (num_samples, latent_dim) generated continuous vectors
        """

        def _single_sample(noise_key: jax.Array) -> Float[Array, "latent"]:
            # sample noise: ε ~ U[-0.5, 0.5]
            eps = jax.random.uniform(
                noise_key, shape=(self.noise_dim,), minval=-0.5, maxval=0.5
            )

            # project noise and hidden state to internal dim
            eps_0 = self.proj_noise(eps)
            h_proj = self.proj_hidden(h)

            # initial fusion
            eps_l = eps_0 + h_proj

            # refine through L residual MLP blocks
            for block in self.blocks:
                eps_l = block(eps_l, h_proj)

            # project to latent dim
            return self.to_latent(eps_l)

        # generate num_samples in parallel
        sample_keys = jax.random.split(key, num_samples)
        return jax.vmap(_single_sample)(sample_keys)  # (S, l)

    def predict(
        self,
        h: Float[Array, "dim"],
        *,
        key: jax.Array,
    ) -> Float[Array, "latent"]:
        """Generate a single continuous vector (inference mode)."""
        samples = self(h, key=key, num_samples=1)
        return samples[0]


def energy_score(
    samples: Float[Array, "N latent"],
    target: Float[Array, "latent"],
    alpha: float = 1.0,
) -> Float[Array, ""]:
    """Compute the energy score (strictly proper scoring rule).

    From CALM paper Equation 10:
      ES(P, z) = (2/N) Σ_i ||z_i - z*||^α - (1/(N²)) Σ_{i,j} ||z_i - z_j||^α

    where z_i are model samples and z* is the target.

    Args:
        samples: (N, l) model-generated samples
        target: (l,) target continuous vector from autoencoder
        alpha: exponent (must be in (0, 2), default 1.0)

    Returns:
        Scalar energy score (lower is better)
    """
    n = samples.shape[0]

    # term 1: mean distance from samples to target
    # (2/N) Σ_i ||z_i - z*||^α
    diffs_to_target = jnp.linalg.norm(samples - target[None, :], axis=-1)  # (N,)
    term1 = (2.0 / n) * jnp.sum(diffs_to_target ** alpha)

    # term 2: mean pairwise distance between samples
    # (1/N²) Σ_{i,j} ||z_i - z_j||^α
    # compute pairwise distances efficiently
    diffs_pairwise = samples[:, None, :] - samples[None, :, :]  # (N, N, l)
    pairwise_dists = jnp.linalg.norm(diffs_pairwise, axis=-1)  # (N, N)
    term2 = (1.0 / (n * n)) * jnp.sum(pairwise_dists ** alpha)

    return term1 - term2


def energy_loss(
    head: EnergyHead,
    h: Float[Array, "dim"],
    target_z: Float[Array, "latent"],
    *,
    key: jax.Array,
    num_model_samples: int = 8,
    alpha: float = 1.0,
) -> Float[Array, ""]:
    """Compute energy loss for training the generative head.

    Uses Monte Carlo estimation with N model samples.
    Default: N=8 (from CALM paper ablations).

    Args:
        head: EnergyHead instance
        h: (hidden_dim,) backbone hidden state
        target_z: (latent_dim,) target latent from CALM encoder
        key: PRNG key
        num_model_samples: N — model-generated samples
        alpha: energy score exponent in (0, 2)

    Returns:
        Scalar energy loss
    """
    samples = head(h, key=key, num_samples=num_model_samples)  # (N, l)
    return energy_score(samples, target_z, alpha=alpha)
