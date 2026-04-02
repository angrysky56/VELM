"""
VELM — Full model composition

Composes CALM autoencoder + Miras backbone + Energy head into the
complete VELM model for both training and inference.

Forward pass (training):
  1. Chunk input tokens into groups of K
  2. Encode each chunk → latent z_i via CALM encoder
  3. Process latent sequence through backbone → hidden states
  4. Energy head predicts next-vector from each hidden state
  5. Loss = energy_score(predicted, target) over all positions

Forward pass (inference):
  1. Encode previous K tokens → compressed input
  2. Backbone processes input → hidden state
  3. Energy head generates z_i from hidden state
  4. CALM decoder: z_i → K predicted tokens
  5. Loop with predicted tokens as next input
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int

from .autoencoder import CALMAutoencoder
from .miras_backbone import VELMBackbone
from .energy_head import EnergyHead, energy_score
from .config import CONFIGS, QWEN35_VOCAB_SIZE


class VELM(eqx.Module):
    """Vector-Evolution Language Model.

    Complete model composing:
      - CALMAutoencoder: tokens ↔ continuous latent vectors
      - VELMBackbone: Miras deep memory + SWA (interleaved)
      - EnergyHead: continuous next-vector prediction
    """

    autoencoder: CALMAutoencoder
    backbone: VELMBackbone
    head: EnergyHead
    chunk_size: int

    def __init__(
        self,
        config_name: str = "gpu_12gb",
        vocab_size: int = QWEN35_VOCAB_SIZE,
        ae_hidden_dim: int | None = None,
        ae_ffn_intermediate: int | None = None,
        *,
        key: jax.Array,
    ) -> None:
        """Initialize VELM from a named configuration.

        Args:
            config_name: one of "smoke", "gpu_12gb", "tiny", "small", "medium", "large"
            vocab_size: tokenizer vocabulary size (default: Qwen3.5 248320)
            ae_hidden_dim: autoencoder hidden dim (overrides config if set)
            ae_ffn_intermediate: autoencoder FFN dim (overrides config if set)
            key: PRNG key
        """
        cfg = CONFIGS[config_name]
        self.chunk_size = cfg["chunk_size_k"]
        k1, k2, k3 = jax.random.split(key, 3)

        # use config values unless explicitly overridden
        _ae_hdim = ae_hidden_dim or cfg.get("ae_hidden_dim", 512)
        _ae_ffn = ae_ffn_intermediate or cfg.get("ae_ffn_intermediate", 1024)

        self.autoencoder = CALMAutoencoder(
            vocab_size=vocab_size,
            chunk_size=cfg["chunk_size_k"],
            hidden_dim=_ae_hdim,
            latent_dim=cfg["latent_dim"],
            ffn_intermediate=_ae_ffn,
            key=k1,
        )

        self.backbone = VELMBackbone(
            dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_miras_layers=cfg["miras_layers"],
            num_swa_layers=cfg["swa_layers"],
            ffn_intermediate=cfg["ffn_intermediate"],
            chunk_size=cfg["chunk_size_k"],
            ae_hidden_dim=_ae_hdim,
            key=k2,
        )

        self.head = EnergyHead(
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            num_blocks=cfg["energy_head_blocks"],
            key=k3,
        )

    def encode_chunks(
        self,
        token_ids: Int[Array, "S K"],
    ) -> Float[Array, "S latent"]:
        """Encode a sequence of token chunks into latent vectors.

        Args:
            token_ids: (num_chunks, K) token IDs

        Returns:
            (num_chunks, latent_dim) latent vectors
        """
        # encode each chunk (no training noise — just get targets)
        def encode_one(chunk: Int[Array, "K"]) -> Float[Array, "latent"]:
            z, _, _ = self.autoencoder.encode(chunk, training=False)
            return z

        return jax.vmap(encode_one)(token_ids)

    def training_loss(
        self,
        token_ids: Int[Array, "S K"],
        *,
        key: jax.Array,
        num_samples: int = 8,
        alpha: float = 1.0,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        """Compute backbone + head training loss over a sequence.

        The autoencoder is assumed frozen. We compute:
          1. Target latent vectors z_i from autoencoder
          2. Input representations from token embeddings
          3. Backbone hidden states from input sequence
          4. Energy loss: head predictions vs target z_{i+1}

        Args:
            token_ids: (num_chunks, K) token IDs
            key: PRNG key
            num_samples: N for energy score Monte Carlo
            alpha: energy score exponent

        Returns:
            (mean_energy_loss, metrics_dict)
        """

        # 1. encode all chunks → target latent vectors
        target_z = self.encode_chunks(token_ids)  # (S, l)

        # 2. build input representations for backbone
        # for each chunk, compress K token embeddings → single vector
        def compress_chunk(chunk_ids: Int[Array, "K"]) -> Float[Array, "dim"]:
            embs = jax.vmap(self.autoencoder.embedding)(chunk_ids)  # (K, ae_d)
            return self.backbone.compress_input(embs)

        input_seq = jax.vmap(compress_chunk)(token_ids)  # (S, dim)

        # 3. backbone forward → hidden states
        hidden_states, _ = self.backbone(input_seq)  # (S, dim)

        # 4. energy loss: predict z_{i+1} from h_i
        # shift: hidden_states[:-1] predicts target_z[1:]
        h_input = hidden_states[:-1]  # (S-1, dim)
        z_target = target_z[1:]       # (S-1, l)
        num_positions = h_input.shape[0]

        keys = jax.random.split(key, num_positions)

        def position_loss(
            h: Float[Array, "dim"],
            target: Float[Array, "latent"],
            k: jax.Array,
        ) -> Float[Array, ""]:
            samples = self.head(h, key=k, num_samples=num_samples)
            return energy_score(samples, target, alpha=alpha)

        losses = jax.vmap(position_loss)(h_input, z_target, keys)
        mean_loss = jnp.mean(losses)

        metrics = {
            "energy_loss": mean_loss,
            "num_positions": jnp.array(num_positions, dtype=jnp.float32),
        }
        return mean_loss, metrics

    def generate_step(
        self,
        prev_token_ids: Int[Array, "K"],
        backbone_states: list[Float[Array, "dim dim"]] | None,
        *,
        key: jax.Array,
    ) -> tuple[Int[Array, "K"], list[Float[Array, "dim dim"]]]:
        """Generate one chunk of K tokens autoregressively.

        Args:
            prev_token_ids: (K,) previous chunk's token IDs
            backbone_states: Miras memory states from previous step
            key: PRNG key

        Returns:
            (predicted_token_ids, new_backbone_states)
        """

        # 1. compress previous tokens → input representation
        embs = jax.vmap(self.autoencoder.embedding)(prev_token_ids)
        input_vec = self.backbone.compress_input(embs)  # (dim,)

        # 2. backbone forward (single step)
        input_seq = input_vec[None, :]  # (1, dim)
        hidden_seq, new_states = self.backbone(input_seq, backbone_states)
        h = hidden_seq[0]  # (dim,)

        # 3. energy head → predicted latent vector
        z_pred = self.head.predict(h, key=key)  # (latent,)

        # 4. decode latent → K token logits → argmax
        logits = self.autoencoder.decode(z_pred)  # (K, vocab)
        predicted_ids = jnp.argmax(logits, axis=-1)  # (K,)

        return predicted_ids, new_states

    def generate(
        self,
        prompt_token_ids: Int[Array, "S K"],
        num_steps: int = 10,
        *,
        key: jax.Array,
    ) -> Int[Array, "G K"]:
        """Generate multiple chunks autoregressively from a prompt.

        Args:
            prompt_token_ids: (num_prompt_chunks, K) prompt tokens
            num_steps: number of chunks to generate
            key: PRNG key

        Returns:
            (num_steps, K) generated token IDs
        """
        # encode prompt through backbone to get initial states
        def compress_chunk(chunk_ids):
            embs = jax.vmap(self.autoencoder.embedding)(chunk_ids)
            return self.backbone.compress_input(embs)

        prompt_seq = jax.vmap(compress_chunk)(prompt_token_ids)
        _, states = self.backbone(prompt_seq)

        # autoregressive generation loop
        last_chunk = prompt_token_ids[-1]
        generated = []
        keys = jax.random.split(key, num_steps)

        for i in range(num_steps):
            next_chunk, states = self.generate_step(
                last_chunk, states, key=keys[i]
            )
            generated.append(next_chunk)
            last_chunk = next_chunk

        return jnp.stack(generated)  # (G, K)
