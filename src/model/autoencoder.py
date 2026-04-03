"""
VELM Model Architecture — CALM Autoencoder

Implements the high-fidelity autoencoder from CALM (Shao et al., 2025).
Compresses K discrete tokens into a single continuous vector z ∈ R^l.

Architecture:
  Encoder: token embeddings → position-wise FFN → flatten(K×d → d) → FFN → z ∈ R^l
  Decoder: z → FFN → expand(d → K×d) → FFN → vocab logits → argmax

Trained with cross-entropy reconstruction + KL clipping regularization.
Achieves >99.9% token reconstruction accuracy.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int


class FFN(eqx.Module):
    """Position-wise feed-forward network with SwiGLU activation."""

    w_gate: eqx.nn.Linear
    w_up: eqx.nn.Linear
    w_down: eqx.nn.Linear

    def __init__(self, dim: int, intermediate: int, *, key: jax.Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.w_gate = eqx.nn.Linear(dim, intermediate, use_bias=False, key=k1)
        self.w_up = eqx.nn.Linear(dim, intermediate, use_bias=False, key=k2)
        self.w_down = eqx.nn.Linear(intermediate, dim, use_bias=False, key=k3)

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """SwiGLU: down(silu(gate(x)) * up(x))."""
        return self.w_down(jax.nn.silu(self.w_gate(x)) * self.w_up(x))


class Encoder(eqx.Module):
    """CALM encoder: K token embeddings → single latent vector z.

    Pipeline:
      1. Embed each of K tokens → K vectors of dim d
      2. Position-wise FFN on each embedding independently
      3. Flatten K×d → d via linear projection
      4. FFN refinement
      5. Linear project to latent dim l
    """

    token_ffn: FFN
    flatten_proj: eqx.nn.Linear
    refine_ffn: FFN
    to_mu: eqx.nn.Linear
    to_logvar: eqx.nn.Linear
    chunk_size: int
    hidden_dim: int
    latent_dim: int

    def __init__(
        self,
        chunk_size: int,
        hidden_dim: int,
        latent_dim: int,
        ffn_intermediate: int,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # position-wise FFN applied to each token embedding
        self.token_ffn = FFN(hidden_dim, ffn_intermediate, key=k1)
        # flatten K*d → d
        self.flatten_proj = eqx.nn.Linear(
            chunk_size * hidden_dim, hidden_dim, use_bias=False, key=k2
        )
        # refine the flattened representation
        self.refine_ffn = FFN(hidden_dim, ffn_intermediate, key=k3)
        # VAE-style: project to mu and logvar for KL regularization
        self.to_mu = eqx.nn.Linear(hidden_dim, latent_dim, use_bias=False, key=k4)
        self.to_logvar = eqx.nn.Linear(hidden_dim, latent_dim, use_bias=False, key=k5)

    def __call__(
        self,
        token_embeddings: Float[Array, "K dim"],
        *,
        key: jax.Array | None = None,
        training: bool = False,
    ) -> tuple[Float[Array, "latent"], Float[Array, "latent"], Float[Array, "latent"]]:
        """Encode K token embeddings into a latent vector.

        Args:
            token_embeddings: (K, hidden_dim) embedded tokens
            key: PRNG key for dropout/reparameterization during training
            training: whether to apply regularization

        Returns:
            (z, mu, logvar) — z is the sampled latent, mu/logvar for KL loss
        """
        k = self.chunk_size

        # 1. position-wise FFN on each token embedding
        h = jax.vmap(self.token_ffn)(token_embeddings)  # (K, d)

        # 2. DropToken: randomly zero out token embeddings during training
        if training and key is not None:
            key, drop_key = jax.random.split(key)
            token_mask = jax.random.bernoulli(drop_key, 0.9, shape=(k, 1))
            h = h * token_mask

        # 3. flatten K×d → d
        h_flat = h.reshape(-1)  # (K*d,)
        h_compressed = self.flatten_proj(h_flat)  # (d,)

        # 4. FFN refinement
        h_refined = self.refine_ffn(h_compressed)  # (d,)

        # 5. project to latent: mu and logvar
        mu = self.to_mu(h_refined)  # (l,)
        logvar_raw = self.to_logvar(h_refined)  # (l,)
        # clamp logvar to prevent exp() overflow — critical for stability
        logvar = jnp.clip(logvar_raw, -20.0, 2.0)

        # reparameterization trick during training
        if training and key is not None:
            key, z_key = jax.random.split(key)
            eps = jax.random.normal(z_key, shape=mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps

            # DropLatent: randomly zero out latent dimensions
            key, lat_key = jax.random.split(key)
            lat_mask = jax.random.bernoulli(lat_key, 0.85, shape=z.shape)
            z = z * lat_mask
        else:
            z = mu

        return z, mu, logvar


class Decoder(eqx.Module):
    """CALM decoder: latent vector z → K token logits.

    Pipeline:
      1. Linear project z from latent dim l to hidden dim d
      2. FFN refinement
      3. Expand d → K×d via linear projection
      4. Reshape to K separate hidden states
      5. Position-wise FFN on each
      6. Project to vocabulary logits (using tied embeddings)
    """

    from_latent: eqx.nn.Linear
    expand_ffn: FFN
    expand_proj: eqx.nn.Linear
    token_ffn: FFN
    chunk_size: int
    hidden_dim: int

    def __init__(
        self,
        chunk_size: int,
        hidden_dim: int,
        latent_dim: int,
        ffn_intermediate: int,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        self.from_latent = eqx.nn.Linear(latent_dim, hidden_dim, use_bias=False, key=k1)
        self.expand_ffn = FFN(hidden_dim, ffn_intermediate, key=k2)
        self.expand_proj = eqx.nn.Linear(
            hidden_dim, chunk_size * hidden_dim, use_bias=False, key=k3
        )
        self.token_ffn = FFN(hidden_dim, ffn_intermediate, key=k4)

    def __call__(
        self,
        z: Float[Array, "latent"],
        embedding_matrix: Float[Array, "vocab dim"],
    ) -> Float[Array, "K vocab"]:
        """Decode latent vector into K token logits.

        Args:
            z: (latent_dim,) latent vector
            embedding_matrix: (vocab_size, hidden_dim) tied embedding weights

        Returns:
            (K, vocab_size) logits for each token position
        """
        # 1. project from latent to hidden dim
        h = self.from_latent(z)  # (d,)

        # 2. FFN refinement
        h = self.expand_ffn(h)  # (d,)

        # 3. expand d → K*d
        h_expanded = self.expand_proj(h)  # (K*d,)

        # 4. reshape to K separate hidden states
        h_tokens = h_expanded.reshape(self.chunk_size, self.hidden_dim)  # (K, d)

        # 5. position-wise FFN on each
        h_tokens = jax.vmap(self.token_ffn)(h_tokens)  # (K, d)

        # 6. project to vocab logits using tied embeddings
        # scale by 1/sqrt(d) to prevent large logits with 248K vocab
        scale = jnp.sqrt(jnp.float32(self.hidden_dim))
        logits = (h_tokens @ embedding_matrix.T) / scale  # (K, vocab)

        return logits


class CALMAutoencoder(eqx.Module):
    """Complete CALM autoencoder: tokens ↔ continuous latent vectors.

    Compresses K discrete tokens into a single continuous vector z ∈ R^l
    with >99.9% reconstruction accuracy.
    """

    embedding: eqx.nn.Embedding
    encoder: Encoder
    decoder: Decoder
    vocab_size: int
    chunk_size: int
    hidden_dim: int
    latent_dim: int
    kl_weight: float
    kl_clip: float

    def __init__(
        self,
        vocab_size: int = 248077,  # Qwen3.5 tokenizer (actual token count)
        chunk_size: int = 4,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        ffn_intermediate: int = 1024,
        kl_weight: float = 0.001,
        kl_clip: float = 1.0,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.kl_clip = kl_clip

        self.embedding = eqx.nn.Embedding(vocab_size, hidden_dim, key=k1)
        self.encoder = Encoder(
            chunk_size, hidden_dim, latent_dim, ffn_intermediate, key=k2
        )
        self.decoder = Decoder(
            chunk_size, hidden_dim, latent_dim, ffn_intermediate, key=k3
        )

    def encode(
        self,
        token_ids: Int[Array, "K"],
        *,
        key: jax.Array | None = None,
        training: bool = False,
    ) -> tuple[Float[Array, "latent"], Float[Array, "latent"], Float[Array, "latent"]]:
        """Encode K token IDs into latent vector z.

        Returns: (z, mu, logvar)
        """
        embeddings = jax.vmap(self.embedding)(token_ids)  # (K, d)
        return self.encoder(embeddings, key=key, training=training)

    def decode(self, z: Float[Array, "latent"]) -> Float[Array, "K vocab"]:
        """Decode latent vector z into K token logits."""
        emb_matrix = self.embedding.weight  # (vocab, d)
        return self.decoder(z, emb_matrix)

    def reconstruct(
        self, token_ids: Int[Array, "K"]
    ) -> Int[Array, "K"]:
        """Full encode → decode → argmax reconstruction."""
        z, _, _ = self.encode(token_ids, training=False)
        logits = self.decode(z)
        return jnp.argmax(logits, axis=-1)

    def loss(
        self,
        token_ids: Int[Array, "K"],
        *,
        key: jax.Array,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        """Compute autoencoder loss: reconstruction CE + clipped KL.

        Args:
            token_ids: (K,) token IDs for one chunk
            key: PRNG key for training stochasticity

        Returns:
            (total_loss, {"recon_loss": ..., "kl_loss": ..., "kl_raw": ...})
        """
        z, mu, logvar = self.encode(token_ids, key=key, training=True)
        logits = self.decode(z)  # (K, vocab)

        # reconstruction loss: cross-entropy across K positions
        # logits: (K, vocab), token_ids: (K,)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (K, vocab)
        recon_loss = -jnp.mean(
            jnp.take_along_axis(log_probs, token_ids[:, None], axis=-1)
        )

        # KL divergence with clipping (from CALM paper)
        # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # logvar is already clamped in encoder, but belt-and-suspenders:
        safe_logvar = jnp.clip(logvar, -20.0, 2.0)
        kl_per_dim = -0.5 * (1.0 + safe_logvar - mu**2 - jnp.exp(safe_logvar))
        kl_raw = jnp.sum(kl_per_dim)

        # clip KL: only penalize dimensions that exceed the clip threshold
        # this prevents posterior collapse while maintaining regularization
        kl_clipped = jnp.sum(jnp.maximum(kl_per_dim, self.kl_clip)) - (
            self.latent_dim * self.kl_clip
        )
        kl_loss = self.kl_weight * jnp.maximum(kl_clipped, 0.0)

        total_loss = recon_loss + kl_loss

        metrics = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_raw": kl_raw,
        }
        return total_loss, metrics


def batch_ae_loss(
    model: CALMAutoencoder,
    batch_token_ids: Int[Array, "B K"],
    *,
    key: jax.Array,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Compute mean autoencoder loss over a batch of chunks.

    Args:
        model: CALMAutoencoder instance
        batch_token_ids: (batch_size, K) token IDs
        key: PRNG key

    Returns:
        (mean_loss, aggregated_metrics)
    """
    batch_size = batch_token_ids.shape[0]
    keys = jax.random.split(key, batch_size)

    def single_loss(tokens: Int[Array, "K"], k: jax.Array):
        return model.loss(tokens, key=k)

    losses, metrics = jax.vmap(single_loss)(batch_token_ids, keys)
    mean_loss = jnp.mean(losses)
    mean_metrics = jax.tree.map(jnp.mean, metrics)
    return mean_loss, mean_metrics


def reconstruction_accuracy(
    model: CALMAutoencoder,
    batch_token_ids: Int[Array, "B K"],
) -> Float[Array, ""]:
    """Compute token-level reconstruction accuracy over a batch.

    Args:
        model: CALMAutoencoder instance
        batch_token_ids: (batch_size, K) token IDs

    Returns:
        Scalar accuracy in [0, 1] — target is >0.999
    """
    reconstructed = jax.vmap(model.reconstruct)(batch_token_ids)  # (B, K)
    correct = jnp.sum(reconstructed == batch_token_ids)
    total = batch_token_ids.size
    return correct / total
