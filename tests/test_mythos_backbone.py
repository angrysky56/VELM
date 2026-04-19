import jax
import jax.numpy as jnp
from src.model.miras_backbone import VELMBackbone

def test_mythos_backbone_looping():
    key = jax.random.PRNGKey(0)
    dim = 128
    num_heads = 8
    num_miras = 2
    num_swa = 2
    ffn_dim = 512
    T = 16
    
    # Initialize backbone with Mythos looping enabled
    backbone = VELMBackbone(
        dim=dim,
        num_heads=num_heads,
        num_miras_layers=num_miras,
        num_swa_layers=num_swa,
        ffn_intermediate=ffn_dim,
        n_loops=3,
        key=key
    )
    
    x = jax.random.normal(key, (T, dim))
    
    # Run with default n_loops=3
    out1, states1 = backbone(x)
    
    # Run with override n_loops=1
    out2, states2 = backbone(x, n_loops=1)
    
    # Outputs should be different for different loop counts
    assert not jnp.allclose(out1, out2)
    assert out1.shape == (T, dim)
    assert out2.shape == (T, dim)
    
    print("Mythos Looping Test Passed!")

def test_lti_stability():
    key = jax.random.PRNGKey(0)
    dim = 64
    from src.model.miras_backbone import LTIInjection
    
    lti = LTIInjection(dim, key=key)
    
    # Get the A matrix
    A = lti.get_a()
    
    # Check that all diagonal elements are < 1
    assert jnp.all(A < 1.0)
    assert jnp.all(A > 0.0)
    
    print("LTI Stability Test Passed!")

if __name__ == "__main__":
    test_mythos_backbone_looping()
    test_lti_stability()
