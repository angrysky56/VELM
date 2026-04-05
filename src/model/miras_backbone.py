"""
VELM Model Architecture — Miras Deep Memory Backbone

Implements the Memora variant from the Miras framework (Behrouz et al., 2025):
  - Memory architecture: MLP (parameterized by W)
  - Attentional bias: ℓ2 regression loss
  - Retention gate: KL divergence (elastic net regularization)
  - Learning algorithm: Gradient descent

Combined with Sliding Window Attention (SWA) in a hybrid architecture
following the Samba pattern for qTTT compatibility.

The Miras layer processes sequences recurrently with deep associative memory,
while SWA layers provide local attention for precise retrieval.
"""

# ruff: noqa: F722, F821
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.model.hyper_connections import (
    HyperConnectionBlock,
    collapse_streams,
    init_streams,
)


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization."""

    weight: Float[Array, "dim"]
    eps: float

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.weight = jnp.ones(dim)
        self.eps = eps

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        rms = jnp.sqrt(jnp.mean(x**2) + self.eps)
        return (x / rms) * self.weight


class DepthwiseConv1d(eqx.Module):
    """1D depthwise-separable convolution (kernel size 4).

    Applied after Q, K, V projections for local feature mixing,
    following Miras paper architecture.
    """

    weight: Float[Array, "dim kernel"]
    bias: Float[Array, "dim"]
    kernel_size: int

    def __init__(self, dim: int, kernel_size: int = 4, *, key: jax.Array) -> None:
        self.kernel_size = kernel_size
        self.weight = jax.random.normal(key, (dim, kernel_size)) * 0.02
        self.bias = jnp.zeros(dim)

    def __call__(self, x: Float[Array, "T dim"]) -> Float[Array, "T dim"]:
        """Apply causal depthwise conv1d.

        Args:
            x: (seq_len, dim) input sequence

        Returns:
            (seq_len, dim) convolved output
        """
        # pad causally: (kernel_size - 1) zeros on the left
        padded = jnp.pad(x, ((self.kernel_size - 1, 0), (0, 0)))  # (T+K-1, d)

        # depthwise conv: each channel independently
        def conv_channel(channel_idx: int) -> Float[Array, "T"]:
            w = self.weight[channel_idx]  # (K,)
            signal = padded[:, channel_idx]  # (T+K-1,)
            return jnp.convolve(signal, w[::-1], mode="valid")  # (T,)

        out = jax.vmap(conv_channel)(jnp.arange(x.shape[1])).T  # (T, d)
        return out + self.bias[None, :]


class MirasMemoryLayer(eqx.Module):
    """Single Miras layer: deep associative memory with MLP structure.

    Implements the Memora variant:
      W_t = Softmax(α_t * log(W_{t-1}) - η_t * ∇ℓ2(W_{t-1}; k_t, v_t))

    where:
      - W is the MLP memory (key→value mapping)
      - α_t is the retention gate (channel-wise, via low-rank projection)
      - η_t is the learning rate (channel-wise, via low-rank projection)
      - ℓ2 is the attentional bias (regression loss)
    """

    # projections for Q, K, V
    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear

    # depthwise conv after Q, K, V
    conv_q: DepthwiseConv1d
    conv_k: DepthwiseConv1d
    conv_v: DepthwiseConv1d

    # low-rank projections for channel-wise parameters
    # η (learning rate) and α (retention gate): input → R^k → R^d
    eta_down: eqx.nn.Linear
    eta_up: eqx.nn.Linear
    alpha_down: eqx.nn.Linear
    alpha_up: eqx.nn.Linear

    # output gate
    gate_proj: eqx.nn.Linear

    # layer norm
    q_norm: RMSNorm
    k_norm: RMSNorm

    dim: int
    head_dim: int
    num_heads: int
    low_rank_dim: int

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        low_rank_dim: int = 32,
        *,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, 12)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.low_rank_dim = low_rank_dim

        # QKV projections
        self.wq = eqx.nn.Linear(dim, dim, use_bias=False, key=keys[0])
        self.wk = eqx.nn.Linear(dim, dim, use_bias=False, key=keys[1])
        self.wv = eqx.nn.Linear(dim, dim, use_bias=False, key=keys[2])
        self.wo = eqx.nn.Linear(dim, dim, use_bias=False, key=keys[3])

        # depthwise convolutions (kernel size 4)
        self.conv_q = DepthwiseConv1d(dim, 4, key=keys[4])
        self.conv_k = DepthwiseConv1d(dim, 4, key=keys[5])
        self.conv_v = DepthwiseConv1d(dim, 4, key=keys[6])

        # ℓ2 normalization for q, k (training stability)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # low-rank projections for η (learning rate): dim → k → dim
        self.eta_down = eqx.nn.Linear(dim, low_rank_dim, use_bias=False, key=keys[7])
        self.eta_up = eqx.nn.Linear(low_rank_dim, dim, use_bias=False, key=keys[8])

        # low-rank projections for α (retention gate): dim → k → dim
        self.alpha_down = eqx.nn.Linear(dim, low_rank_dim, use_bias=False, key=keys[9])
        self.alpha_up = eqx.nn.Linear(low_rank_dim, dim, use_bias=False, key=keys[10])

        # output gate
        self.gate_proj = eqx.nn.Linear(dim, dim, use_bias=False, key=keys[11])

    def _compute_eta(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """Channel-wise learning rate via low-rank projection + sigmoid."""
        return jax.nn.sigmoid(self.eta_up(self.eta_down(x)))

    def _compute_alpha(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """Channel-wise retention gate via low-rank projection + sigmoid."""
        return jax.nn.sigmoid(self.alpha_up(self.alpha_down(x)))

    def _recurrent_step(
        self,
        state: Float[Array, "dim dim"],
        k_t: Float[Array, "dim"],
        v_t: Float[Array, "dim"],
        x_t: Float[Array, "dim"],
    ) -> tuple[Float[Array, "dim dim"], Float[Array, "dim"]]:
        """Single recurrent memory update step (Memora variant).

        Performs:
          η_t = sigmoid(low_rank(x_t))
          α_t = sigmoid(low_rank(x_t))
          memory_pred = state @ k_t  (linear memory readout)
          grad = 2 * (memory_pred - v_t) ⊗ k_t  (ℓ2 gradient)
          state_new = α_t * state - η_t * grad  (retention + update)
          output = state_new @ k_t  (readout after update)

        Args:
            state: (dim, dim) current memory state W_t
            k_t: (dim,) key for this timestep
            v_t: (dim,) value for this timestep
            x_t: (dim,) input for computing gates

        Returns:
            (new_state, output) tuple
        """
        # compute channel-wise gates
        eta = self._compute_eta(x_t)  # (dim,) learning rate
        alpha = self._compute_alpha(x_t)  # (dim,) retention

        # memory readout: predict value from key
        memory_pred = state @ k_t  # (dim,)

        # ℓ2 gradient: ∇_W ||W@k - v||² = 2(W@k - v) @ k^T
        error = memory_pred - v_t  # (dim,)
        grad = jnp.outer(error, k_t)  # (dim, dim)

        # retention + update with channel-wise gating
        # apply α and η as diagonal scaling on rows
        new_state = alpha[:, None] * state - eta[:, None] * grad

        # numerical stability: clamp state to prevent explosion in long scans
        new_state = jnp.clip(new_state, -10.0, 10.0)

        # readout from updated memory
        output = new_state @ k_t  # (dim,)

        return new_state, output

    def __call__(
        self,
        x: Float[Array, "T dim"],
        state: Float[Array, "dim dim"] | None = None,
    ) -> tuple[Float[Array, "T dim"], Float[Array, "dim dim"]]:
        """Process a sequence through the Miras memory layer.

        Args:
            x: (seq_len, dim) input sequence
            state: optional initial memory state, zeros if None

        Returns:
            (output_sequence, final_state)
        """
        seq_len = x.shape[0]

        # project Q, K, V with depthwise conv
        q_seq = self.conv_q(jax.vmap(self.wq)(x))  # (T, dim)
        k_seq = self.conv_k(jax.vmap(self.wk)(x))  # (T, dim)
        v_seq = self.conv_v(jax.vmap(self.wv)(x))  # (T, dim)

        # ℓ2 normalize q and k per head for stability
        def norm_heads(seq, norm_fn):
            heads = seq.reshape(seq_len, self.num_heads, self.head_dim)
            normed = jax.vmap(jax.vmap(norm_fn))(heads)
            return normed.reshape(seq_len, self.dim)

        q_seq = norm_heads(q_seq, self.q_norm)
        k_seq = norm_heads(k_seq, self.k_norm)

        # init memory state
        if state is None:
            state = jnp.zeros((self.dim, self.dim))

        # sequential scan through the sequence
        def scan_fn(carry, inputs):
            st = carry
            k_t, v_t, x_t = inputs
            new_st, out_t = self._recurrent_step(st, k_t, v_t, x_t)
            return new_st, out_t

        final_state, outputs = jax.lax.scan(
            scan_fn, state, (k_seq, v_seq, x)
        )  # outputs: (T, dim)

        # output gate and projection
        gate = jax.nn.silu(jax.vmap(self.gate_proj)(x))  # (T, dim)
        gated = outputs * gate
        projected = jax.vmap(self.wo)(gated)  # (T, dim)

        return projected, final_state


class SlidingWindowAttention(eqx.Module):
    """Sliding Window Attention (SWA) layer.

    Provides local attention for precise retrieval within a window.
    Compatible with qTTT: query projections can be adapted at inference.
    """

    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear
    dim: int
    num_heads: int
    head_dim: int
    window_size: int

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        window_size: int = 512,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.wq = eqx.nn.Linear(dim, dim, use_bias=False, key=k1)
        self.wk = eqx.nn.Linear(dim, dim, use_bias=False, key=k2)
        self.wv = eqx.nn.Linear(dim, dim, use_bias=False, key=k3)
        self.wo = eqx.nn.Linear(dim, dim, use_bias=False, key=k4)

    def __call__(
        self,
        x: Float[Array, "T dim"],
    ) -> Float[Array, "T dim"]:
        """Apply sliding window causal attention.

        Args:
            x: (seq_len, dim) input sequence

        Returns:
            (seq_len, dim) attention output
        """
        seq_len = x.shape[0]
        scale = self.head_dim**-0.5

        # project Q, K, V
        q = jax.vmap(self.wq)(x)  # (T, dim)
        k = jax.vmap(self.wk)(x)
        v = jax.vmap(self.wv)(x)

        # reshape to multi-head: (T, H, head_dim)
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_heads, self.head_dim)

        # compute attention scores: (H, T, T)
        # transpose to (H, T, d) for batched matmul
        q_h = jnp.transpose(q, (1, 0, 2))  # (H, T, d)
        k_h = jnp.transpose(k, (1, 0, 2))
        v_h = jnp.transpose(v, (1, 0, 2))

        scores = jnp.matmul(q_h, jnp.transpose(k_h, (0, 2, 1))) * scale  # (H, T, T)

        # causal mask + sliding window mask
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        window = jnp.triu(
            jnp.ones((seq_len, seq_len), dtype=jnp.bool_), k=-(self.window_size - 1)
        )
        mask = causal & window  # (T, T)

        scores = jnp.where(mask[None, :, :], scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)  # (H, T, T)

        # weighted sum of values
        attn_out = jnp.matmul(weights, v_h)  # (H, T, d)

        # reshape back: (H, T, d) → (T, H, d) → (T, dim)
        attn_out = jnp.transpose(attn_out, (1, 0, 2))  # (T, H, d)
        attn_out = attn_out.reshape(seq_len, self.dim)

        # output projection
        return jax.vmap(self.wo)(attn_out)


class SwiGLUFFN(eqx.Module):
    """SwiGLU feed-forward network for backbone blocks."""

    w_gate: eqx.nn.Linear
    w_up: eqx.nn.Linear
    w_down: eqx.nn.Linear

    def __init__(self, dim: int, intermediate: int, *, key: jax.Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.w_gate = eqx.nn.Linear(dim, intermediate, use_bias=False, key=k1)
        self.w_up = eqx.nn.Linear(dim, intermediate, use_bias=False, key=k2)
        self.w_down = eqx.nn.Linear(intermediate, dim, use_bias=False, key=k3)

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        return self.w_down(jax.nn.silu(self.w_gate(x)) * self.w_up(x))


class MirasBlock(eqx.Module):
    """Single Miras block: pre-norm → Miras memory → residual → pre-norm → FFN → residual."""

    norm1: RMSNorm
    miras: MirasMemoryLayer
    norm2: RMSNorm
    ffn: SwiGLUFFN

    def __init__(
        self, dim: int, num_heads: int, ffn_intermediate: int, *, key: jax.Array
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.norm1 = RMSNorm(dim)
        self.miras = MirasMemoryLayer(dim, num_heads, key=k1)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_intermediate, key=k2)

    def __call__(
        self,
        x: Float[Array, "T dim"],
        state: Float[Array, "dim dim"] | None = None,
    ) -> tuple[Float[Array, "T dim"], Float[Array, "dim dim"]]:
        """Forward pass with residual connections.

        Returns: (output, new_memory_state)
        """
        # pre-norm → miras → residual
        normed = jax.vmap(self.norm1)(x)
        miras_out, new_state = self.miras(normed, state)
        x = x + miras_out

        # pre-norm → FFN → residual
        normed = jax.vmap(self.norm2)(x)
        ffn_out = jax.vmap(self.ffn)(normed)
        x = x + ffn_out

        return x, new_state


class SWABlock(eqx.Module):
    """Single SWA block: pre-norm → SWA → residual → pre-norm → FFN → residual.

    These blocks are the qTTT adaptation targets — their W_Q matrices
    get updated at inference via query-only gradient steps.
    """

    norm1: RMSNorm
    attn: SlidingWindowAttention
    norm2: RMSNorm
    ffn: SwiGLUFFN

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_intermediate: int,
        window_size: int = 512,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.norm1 = RMSNorm(dim)
        self.attn = SlidingWindowAttention(dim, num_heads, window_size, key=k1)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_intermediate, key=k2)

    def __call__(self, x: Float[Array, "T dim"]) -> Float[Array, "T dim"]:
        """Forward pass with residual connections."""
        # pre-norm → attention → residual
        normed = jax.vmap(self.norm1)(x)
        attn_out = self.attn(normed)
        x = x + attn_out

        # pre-norm → FFN → residual
        normed = jax.vmap(self.norm2)(x)
        ffn_out = jax.vmap(self.ffn)(normed)
        x = x + ffn_out

        return x


class VELMBackbone(eqx.Module):
    """Full VELM backbone: interleaved Miras + SWA blocks.

    Follows Samba-style hybrid architecture:
      - Miras blocks provide long-range recurrent memory
      - SWA blocks provide local attention (qTTT-compatible)
      - Blocks alternate: [Miras, SWA, Miras, SWA, ...]

    The input compression MLP maps K token embeddings → single vector
    for each autoregressive step.
    """

    input_compress: eqx.nn.Linear
    input_ffn: SwiGLUFFN
    miras_blocks: list[MirasBlock]
    swa_blocks: list[SWABlock]
    hc_blocks: list[HyperConnectionBlock]  # go-mHC per layer
    final_norm: RMSNorm
    block_order: list[str]  # ["miras", "swa", "miras", "swa", ...]
    dim: int
    hc_streams: int  # d: number of residual streams (1 = no HC)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_miras_layers: int,
        num_swa_layers: int,
        ffn_intermediate: int,
        chunk_size: int = 4,
        window_size: int = 512,
        ae_hidden_dim: int | None = None,
        hc_streams: int = 1,
        hc_s: int = 2,
        *,
        key: jax.Array,
    ) -> None:
        """Initialize the VELM backbone.

        Args:
            dim: hidden dimension
            num_heads: attention heads (shared by Miras and SWA)
            num_miras_layers: number of Miras memory blocks
            num_swa_layers: number of SWA blocks
            ffn_intermediate: SwiGLU intermediate dim
            chunk_size: K tokens per autoregressive step
            window_size: SWA window size
            ae_hidden_dim: autoencoder embedding dimension (defaults to dim)
            hc_streams: d — number of residual streams (1=standard, >1=go-mHC)
            hc_s: s — go-mHC expressivity parameter
            key: PRNG key
        """
        self.dim = dim
        self.hc_streams = hc_streams
        ae_dim = ae_hidden_dim if ae_hidden_dim is not None else dim
        total_layers = num_miras_layers + num_swa_layers
        keys = jax.random.split(key, total_layers + 2)

        # input compression: K embeddings → single vector
        # CALM paper uses 2-layer MLP for this
        self.input_compress = eqx.nn.Linear(
            chunk_size * ae_dim, dim, use_bias=False, key=keys[0]
        )
        self.input_ffn = SwiGLUFFN(dim, ffn_intermediate, key=keys[1])

        # build interleaved block order
        self.block_order = []
        miras_idx, swa_idx = 0, 0
        m_blocks, s_blocks = [], []

        for i in range(total_layers):
            if i % 2 == 0 and miras_idx < num_miras_layers:
                m_blocks.append(
                    MirasBlock(dim, num_heads, ffn_intermediate, key=keys[i + 2])
                )
                self.block_order.append("miras")
                miras_idx += 1
            elif swa_idx < num_swa_layers:
                s_blocks.append(
                    SWABlock(
                        dim, num_heads, ffn_intermediate, window_size, key=keys[i + 2]
                    )
                )
                self.block_order.append("swa")
                swa_idx += 1
            else:
                # fill remaining with whichever has slots left
                if miras_idx < num_miras_layers:
                    m_blocks.append(
                        MirasBlock(dim, num_heads, ffn_intermediate, key=keys[i + 2])
                    )
                    self.block_order.append("miras")
                    miras_idx += 1
                else:
                    s_blocks.append(
                        SWABlock(
                            dim,
                            num_heads,
                            ffn_intermediate,
                            window_size,
                            key=keys[i + 2],
                        )
                    )
                    self.block_order.append("swa")
                    swa_idx += 1

        self.miras_blocks = m_blocks
        self.swa_blocks = s_blocks
        self.final_norm = RMSNorm(dim)

        # go-mHC hyper-connection blocks (one per layer)
        # when hc_streams=1, these are no-ops (standard residual)
        if hc_streams > 1:
            hc_keys = jax.random.split(
                jax.random.PRNGKey(999), total_layers)
            self.hc_blocks = [
                HyperConnectionBlock(d=hc_streams, s=hc_s, key=hc_keys[i])
                for i in range(total_layers)
            ]
        else:
            self.hc_blocks = []

    def compress_input(
        self,
        chunk_embeddings: Float[Array, "K dim"],
    ) -> Float[Array, "dim"]:
        """Compress K token embeddings into a single input vector.

        Used at each autoregressive step to create the input representation
        from the previous K predicted tokens.
        """
        flat = chunk_embeddings.reshape(-1)  # (K*dim,)
        h = self.input_compress(flat)  # (dim,)
        return self.input_ffn(h)  # (dim,)

    def __call__(
        self,
        x: Float[Array, "T dim"],
        miras_states: list[Float[Array, "dim dim"]] | None = None,
    ) -> tuple[Float[Array, "T dim"], list[Float[Array, "dim dim"]]]:
        """Process sequence through interleaved Miras + SWA blocks.

        Args:
            x: (seq_len, dim) input sequence of compressed representations
            miras_states: optional list of initial memory states per Miras block

        Returns:
            (output_sequence, list_of_final_miras_states)
        """
        num_miras = len(self.miras_blocks)
        if miras_states is None:
            miras_states = [None] * num_miras

        d = self.hc_streams
        use_hc = d > 1 and len(self.hc_blocks) > 0

        # initialize d residual streams if using hyper-connections
        if use_hc:
            x_streams = init_streams(x, d)  # (d, T, dim)

        new_states = []
        m_idx, s_idx = 0, 0

        for layer_idx, block_type in enumerate(self.block_order):
            if use_hc:
                hc = self.hc_blocks[layer_idx]
                # H_res mixing across d streams
                x_mixed = hc.mix_residual(x_streams)
                # H_pre: aggregate streams → single input for the block
                block_input = hc.aggregate_for_block(x_streams)

                # run the actual block on aggregated input
                if block_type == "miras":
                    block_out, state = self.miras_blocks[m_idx](
                        block_input, miras_states[m_idx])
                    new_states.append(state)
                    m_idx += 1
                else:
                    block_out = self.swa_blocks[s_idx](block_input)
                    s_idx += 1

                # H_post: distribute output → d streams + add to mixed
                x_streams = x_mixed + hc.distribute_from_block(block_out)
            else:
                # standard single-stream residuals (hc_streams=1)
                if block_type == "miras":
                    x, state = self.miras_blocks[m_idx](x, miras_states[m_idx])
                    new_states.append(state)
                    m_idx += 1
                else:
                    x = self.swa_blocks[s_idx](x)
                    s_idx += 1

        # collapse streams and normalize
        if use_hc:
            x = collapse_streams(x_streams)  # (d, T, dim) → (T, dim)

        x = jax.vmap(self.final_norm)(x)

        return x, new_states
