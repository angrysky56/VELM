"""
VELM Smoke Test — verify all modules compile and JIT-trace cleanly.

Run with: python -m pytest tests/test_smoke.py -v
Or standalone: python tests/test_smoke.py

Tests use tiny dimensions to run fast on any hardware.
"""

import sys
from pathlib import Path

# ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp


# use tiny dimensions for fast testing
VOCAB = 256
K = 4
DIM = 32
LATENT = 16
FFN = 64
HEADS = 4
SEQ = 8


def test_autoencoder_encode_decode():
    """Autoencoder forward pass: encode → decode → correct shapes."""
    from src.model.autoencoder import CALMAutoencoder

    key = jax.random.PRNGKey(0)
    model = CALMAutoencoder(
        vocab_size=VOCAB, chunk_size=K, hidden_dim=DIM,
        latent_dim=LATENT, ffn_intermediate=FFN, key=key,
    )

    tokens = jax.random.randint(key, (K,), 0, VOCAB)

    # encode
    z, mu, logvar = model.encode(tokens, training=False)
    assert z.shape == (LATENT,), f"Expected ({LATENT},), got {z.shape}"
    assert mu.shape == (LATENT,)
    assert logvar.shape == (LATENT,)

    # decode
    logits = model.decode(z)
    assert logits.shape == (K, VOCAB), f"Expected ({K},{VOCAB}), got {logits.shape}"

    # reconstruct
    recon = model.reconstruct(tokens)
    assert recon.shape == (K,)
    print("  autoencoder encode/decode: OK")


def test_autoencoder_loss():
    """Autoencoder loss computes and returns metrics."""
    from src.model.autoencoder import CALMAutoencoder, batch_ae_loss

    key = jax.random.PRNGKey(1)
    model = CALMAutoencoder(
        vocab_size=VOCAB, chunk_size=K, hidden_dim=DIM,
        latent_dim=LATENT, ffn_intermediate=FFN, key=key,
    )

    k1, k2 = jax.random.split(key)
    tokens = jax.random.randint(k1, (K,), 0, VOCAB)
    loss, metrics = model.loss(tokens, key=k2)

    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert "recon_loss" in metrics
    assert "kl_loss" in metrics
    print("  autoencoder loss: OK")


def test_energy_head():
    """Energy head generates samples and computes energy score."""
    from src.model.energy_head import EnergyHead, energy_score

    key = jax.random.PRNGKey(2)
    head = EnergyHead(
        hidden_dim=DIM, latent_dim=LATENT, num_blocks=2, key=key,
    )

    h = jax.random.normal(key, (DIM,))
    samples = head(h, key=key, num_samples=4)
    assert samples.shape == (4, LATENT), f"Expected (4,{LATENT}), got {samples.shape}"

    # single predict
    z = head.predict(h, key=key)
    assert z.shape == (LATENT,)

    # energy score
    target = jax.random.normal(key, (LATENT,))
    score = energy_score(samples, target, alpha=1.0)
    assert score.shape == (), "Energy score should be scalar"
    print("  energy head: OK")


def test_miras_layer():
    """Single Miras memory layer processes a sequence."""
    from src.model.miras_backbone import MirasMemoryLayer

    key = jax.random.PRNGKey(3)
    layer = MirasMemoryLayer(dim=DIM, num_heads=HEADS, key=key)

    x = jax.random.normal(key, (SEQ, DIM))
    out, state = layer(x)
    assert out.shape == (SEQ, DIM), f"Expected ({SEQ},{DIM}), got {out.shape}"
    assert state.shape == (DIM, DIM), f"State should be ({DIM},{DIM})"
    print("  miras layer: OK")


def test_swa_layer():
    """Sliding window attention processes a sequence."""
    from src.model.miras_backbone import SlidingWindowAttention

    key = jax.random.PRNGKey(4)
    attn = SlidingWindowAttention(
        dim=DIM, num_heads=HEADS, window_size=4, key=key,
    )

    x = jax.random.normal(key, (SEQ, DIM))
    out = attn(x)
    assert out.shape == (SEQ, DIM)
    print("  SWA layer: OK")


def test_backbone():
    """Full VELM backbone: interleaved Miras + SWA."""
    from src.model.miras_backbone import VELMBackbone

    key = jax.random.PRNGKey(5)
    backbone = VELMBackbone(
        dim=DIM, num_heads=HEADS, num_miras_layers=2,
        num_swa_layers=2, ffn_intermediate=FFN,
        chunk_size=K, window_size=4, key=key,
    )

    x = jax.random.normal(key, (SEQ, DIM))
    out, states = backbone(x)
    assert out.shape == (SEQ, DIM)
    assert len(states) == 2, "Should have 2 Miras states"
    print("  backbone: OK")


def test_eggroll_perturbation():
    """EGGROLL low-rank perturbation generation."""
    from src.training.eggroll import (
        generate_low_rank_perturbation, perturb_pytree,
    )

    key = jax.random.PRNGKey(6)

    # vector perturbation
    e_vec = generate_low_rank_perturbation(key, (DIM,), rank=1)
    assert e_vec.shape == (DIM,)

    # matrix perturbation (rank 1 = outer product)
    e_mat = generate_low_rank_perturbation(key, (DIM, DIM), rank=1)
    assert e_mat.shape == (DIM, DIM)
    # rank-1 matrix should have rank 1
    _, s, _ = jnp.linalg.svd(e_mat)
    # most singular values should be ~0
    assert jnp.sum(s > 0.01) <= 1, "Rank-1 perturbation should have rank 1"

    # pytree perturbation
    params = {"w1": jnp.zeros((DIM, DIM)), "b1": jnp.zeros((DIM,))}
    perturbed, perturbation = perturb_pytree(params, key, sigma=0.01, rank=1)
    assert perturbed["w1"].shape == (DIM, DIM)
    assert perturbed["b1"].shape == (DIM,)
    # perturbed should differ from original
    assert jnp.any(perturbed["w1"] != 0.0)
    print("  EGGROLL perturbation: OK")


def test_eggroll_step():
    """EGGROLL optimizer step with tiny population."""
    from src.training.eggroll import (
        eggroll_step, create_eggroll_optimizer,
    )

    key = jax.random.PRNGKey(7)
    params = {"w": jax.random.normal(key, (DIM, DIM))}

    def dummy_fitness(p):
        return -jnp.sum(p["w"] ** 2)  # minimize norm

    optimizer, state = create_eggroll_optimizer(params, learning_rate=1e-2)

    new_params, new_state, metrics = eggroll_step(
        params, dummy_fitness, optimizer, state,
        key=key, population_size=8, sigma=0.01, rank=1,
    )

    assert new_params["w"].shape == (DIM, DIM)
    assert "mean_fitness" in metrics
    assert "max_fitness" in metrics
    assert new_state.step == 1
    print("  EGGROLL step: OK")


def test_cib_budget():
    """CIB budget controller decisions."""
    from src.inference.cib_budget import (
        CIBBudgetController, should_continue_reasoning,
        estimate_difficulty, compute_info_gain,
    )

    controller = CIBBudgetController(
        max_chunks=64, gain_threshold=0.01, warmup_chunks=4,
    )

    # during warmup, always continue
    assert should_continue_reasoning(controller, step=1, info_gains=[])
    assert should_continue_reasoning(controller, step=3, info_gains=[0.5])

    # after warmup with low gains, should stop
    assert not should_continue_reasoning(
        controller, step=10, info_gains=[0.001, 0.001, 0.001]
    )

    # at max chunks, always stop
    assert not should_continue_reasoning(controller, step=64, info_gains=[1.0])

    # difficulty estimate
    h = jnp.ones(DIM) * 15.0
    diff = estimate_difficulty(h)
    assert 0.0 <= float(diff) <= 1.0

    # info gain
    h1 = jnp.ones(DIM)
    h2 = -jnp.ones(DIM)
    gain = compute_info_gain(h1, h2)
    assert float(gain) > 1.0, "Opposite vectors should have high gain"
    print("  CIB budget: OK")


def test_gea_novelty():
    """GEA novelty computation and selection."""
    from src.evolution.gea_eggroll import (
        compute_novelty, performance_novelty_selection,
    )

    key = jax.random.PRNGKey(8)
    embeddings = jax.random.normal(key, (10, DIM))

    novelty = compute_novelty(embeddings, num_neighbors=3)
    assert novelty.shape == (10,)
    assert jnp.all(novelty >= 0.0)

    fitness = jax.random.uniform(key, (10,))
    selected = performance_novelty_selection(fitness, novelty, group_size=3)
    assert len(selected) == 3
    assert len(set(selected)) == 3, "Selected indices should be unique"
    print("  GEA novelty/selection: OK")


def test_gea_eggroll_bridge():
    """GEA-EGGROLL integration: traces → experience-weighted update."""
    import optax

    from src.evolution.gea_eggroll import (
        EvolutionTrace,
        experience_weighted_eggroll_step,
    )

    key = jax.random.PRNGKey(9)
    params = {"w": jax.random.normal(key, (DIM, DIM))}

    # create mock traces (simulating GEA population evaluation)
    traces = []
    for i in range(8):
        trace = EvolutionTrace(
            member_id=i,
            perturbation_seed=42 + i,
            fitness_scores={"math": float(i * 0.1), "general": float(i * 0.05)},
        )
        traces.append(trace)

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    # run the bridge function
    new_params, new_opt_state, metrics = experience_weighted_eggroll_step(
        params,
        traces,
        key=key,
        optimizer=optimizer,
        opt_state=opt_state,
        sigma=0.01,
        rank=1,
        parent_indices=[6, 7],  # best performers
        parent_bias=0.3,
    )

    assert new_params["w"].shape == (DIM, DIM)
    assert "mean_fitness" in metrics
    assert "max_fitness" in metrics
    assert "valid_members" in metrics
    assert metrics["parent_bias_applied"] is True
    # params should have changed
    assert not jnp.allclose(params["w"], new_params["w"])
    print("  GEA-EGGROLL bridge: OK")


def run_all():

    """Run all smoke tests and report results."""
    tests = [
        ("Autoencoder encode/decode", test_autoencoder_encode_decode),
        ("Autoencoder loss", test_autoencoder_loss),
        ("Energy head", test_energy_head),
        ("Miras memory layer", test_miras_layer),
        ("Sliding window attention", test_swa_layer),
        ("VELM backbone", test_backbone),
        ("EGGROLL perturbation", test_eggroll_perturbation),
        ("EGGROLL step", test_eggroll_step),
        ("CIB budget controller", test_cib_budget),
        ("GEA novelty/selection", test_gea_novelty),
        ("GEA-EGGROLL bridge", test_gea_eggroll_bridge),
    ]

    print("=" * 60)
    print("VELM Smoke Tests")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  {name}: FAILED — {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    else:
        print("All tests passed!")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
