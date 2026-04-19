"""
VELM Training — Knowledge Distillation from Teacher Model

Distills representations from a larger teacher (Qwen3.5-0.8B) into
VELM's continuous latent backbone. Uniquely enabled by CALM's
continuous latent space: we distill representations directly,
not soft logits.

Pipeline:
  1. Run teacher on training chunks → extract hidden states
  2. Project teacher hidden states → VELM latent space via linear probe
  3. Train backbone + head via backprop to predict teacher vectors
  4. The backbone learns 0.8B-quality representations at 7M inference cost

This is NOT abandoning EGGROLL's thesis. EGGROLL enables:
  - Int8 native training (future)
  - Truly nonlinear recurrence without BPTT (at scale)
  - GEA population evolution (gradient-free by nature)

Distillation uses backprop because the current forward pass IS
differentiable. Using ES where backprop works is like using a
hammer where a screwdriver fits — the wrong tool.
"""

# ruff: noqa: F722, F821

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def extract_teacher_vectors(
    teacher_model,
    chunks: jnp.ndarray,
    ae_model,
    batch_size: int = 64,
) -> jnp.ndarray:
    """Extract teacher hidden states and project to VELM latent space.

    Pipeline:
      1. Run teacher model to get hidden states
      2. Pool hidden states per chunk (mean over K token positions)
      3. Project to VELM latent dim via frozen AE encoder

    This runs ONCE on the dataset, producing target vectors that
    are saved and reused across training.

    Args:
        teacher_model: HuggingFace teacher model (e.g., Qwen3.5-0.8B)
        chunks: (N, K) int32 token IDs
        ae_model: frozen CALM autoencoder
        batch_size: processing batch size

    Returns:
        (N, latent_dim) target vectors in VELM's latent space
    """
    import numpy as np
    import torch

    num_chunks = chunks.shape[0]
    all_teacher_hiddens = []

    teacher_model.eval()
    device = next(teacher_model.parameters()).device

    for start in range(0, num_chunks, batch_size):
        end = min(start + batch_size, num_chunks)
        batch_ids = chunks[start:end]  # (B, K)

        with torch.no_grad():
            input_ids = torch.tensor(
                np.array(batch_ids), dtype=torch.long, device=device
            )
            # teacher forward → get last hidden states
            outputs = teacher_model(input_ids=input_ids, output_hidden_states=True)
            # last layer hidden: (B, K, teacher_dim)
            last_hidden = outputs.hidden_states[-1]
            # mean-pool over K positions → (B, teacher_dim)
            pooled = last_hidden.mean(dim=1)
            all_teacher_hiddens.append(pooled.cpu().numpy())

    # (N, teacher_dim) numpy
    teacher_vecs = np.concatenate(all_teacher_hiddens, axis=0)
    return teacher_vecs


class TeacherProjection(eqx.Module):
    """Projects teacher hidden states to VELM latent dimension.

    Linear projection: R^{teacher_dim} → R^{latent_dim}
    Trained alongside the backbone during distillation.
    """

    proj: eqx.nn.Linear

    def __init__(self, teacher_dim: int, latent_dim: int, *, key: jax.Array) -> None:
        self.proj = eqx.nn.Linear(teacher_dim, latent_dim, use_bias=False, key=key)

    def __call__(
        self,
        teacher_vec: Float[Array, "teacher_dim"],
    ) -> Float[Array, "latent_dim"]:
        return self.proj(teacher_vec)


def distillation_loss(
    backbone_hidden: Float[Array, "dim"],
    teacher_vec: Float[Array, "latent_dim"],
    proj: TeacherProjection | None = None,
) -> Float[Array, ""]:
    """L2 distillation loss between backbone hidden and teacher target.

    If teacher_vec is in teacher_dim, proj maps it to latent_dim.
    If teacher_vec is already in latent_dim, proj can be None.

    Args:
        backbone_hidden: (dim,) backbone output for this position
        teacher_vec: (teacher_dim,) or (latent_dim,) teacher target
        proj: optional projection layer (None if dims already match)

    Returns:
        scalar L2 loss
    """
    if proj is not None:
        target = proj(teacher_vec)
    else:
        target = teacher_vec
    # L2 distance in latent space
    return jnp.mean((backbone_hidden - target) ** 2)


def combined_training_loss(
    backbone_params,
    head_params,
    bb_static,
    hd_static,
    frozen_ae,
    batch_tokens,
    teacher_vecs,
    teacher_proj,
    step_key,
    alpha_energy: float = 1.0,
    alpha_distill: float = 0.5,
):
    """Combined energy + distillation loss for backbone training.

    L = α_energy × L_energy(next-vector prediction)
      + α_distill × L_distill(backbone hidden ↔ teacher vector)

    The energy loss trains the head to generate accurate next-vectors.
    The distillation loss trains the backbone to produce hidden states
    that carry 0.8B-quality semantic representations.

    Args:
        backbone_params: backbone array leaves
        head_params: head array leaves
        bb_static: backbone non-array structure
        hd_static: head non-array structure
        frozen_ae: frozen CALM autoencoder
        batch_tokens: (B, K) token chunks
        teacher_vecs: (B, teacher_dim) teacher hidden states
        teacher_proj: projection teacher_dim → backbone_dim
        step_key: PRNG key
        alpha_energy: weight for energy loss
        alpha_distill: weight for distillation loss

    Returns:
        (total_loss, metrics_dict)
    """
    from src.model.energy_head import energy_score

    bb = eqx.combine(backbone_params, bb_static)
    hd = eqx.combine(head_params, hd_static)

    # encode chunks → target latents via frozen AE
    tgt_z = jax.vmap(lambda c: frozen_ae.encode(c, training=False)[0])(batch_tokens)

    # compress input for backbone
    def compress(chunk):
        embs = jax.vmap(frozen_ae.embedding)(chunk)
        return bb.compress_input(embs)

    inp_seq = jax.vmap(compress)(batch_tokens)

    # backbone forward → hidden states
    hid, _ = bb(inp_seq)

    # --- Energy loss: predict z_{i+1} from h_i ---
    hid_in, z_target = hid[:-1], tgt_z[1:]

    def pos_energy(h, z_t, k):
        samples = hd(h, key=k, num_samples=8)
        return energy_score(samples, z_t)

    keys = jax.random.split(step_key, hid_in.shape[0])
    energy_losses = jax.vmap(pos_energy)(hid_in, z_target, keys)
    e_loss = jnp.mean(energy_losses)

    # --- Distillation loss: align backbone hidden with teacher ---
    # project teacher vectors to backbone dim
    if teacher_vecs is not None and teacher_proj is not None:
        teacher_targets = jax.vmap(teacher_proj)(teacher_vecs)  # (B, dim)
        # align backbone hidden states with teacher targets
        # use all positions (not shifted like energy loss)
        d_losses = jax.vmap(lambda h, t: jnp.mean((h - t) ** 2))(
            hid, teacher_targets[: hid.shape[0]]
        )
        d_loss = jnp.mean(d_losses)
    else:
        d_loss = 0.0

    total = alpha_energy * e_loss + alpha_distill * d_loss
    metrics = {
        "energy_loss": e_loss,
        "distill_loss": d_loss,
        "total_loss": total,
    }
    return total, metrics
