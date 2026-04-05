# go-mHC Integration Notes for VELM

## Paper: go-mHC (Dandachi & Diggs-Galligan, April 2026)
- arXiv: 2604.02309v1
- Repo: https://github.com/itstorque/go-mHC

## What It Does
Replaces standard single-stream residual connections with d-stream
parallel residual mixing using exactly doubly stochastic matrices.
Parameterized via generalized orthostochastic matrices (Cayley
transform → block Frobenius projection), achieving:
- Exact double stochasticity (no Sinkhorn-Knopp iterations)
- O(d³) scaling (vs factorial for mHC-lite)
- Full Birkhoff polytope coverage at s≥2
- 10x faster convergence than prior methods on synthetic tasks
- No custom CUDA kernels needed

## Key Equations (for implementation)
```
# Per layer l:
A_l = skew(learned_params)        # ds(ds-1)/2 params → ds×ds skew-symmetric
Q_l = (I - A_l)(I + A_l)^{-1}    # Cayley transform → orthogonal matrix
H_res_l = Φ_{d,s}(Q_l)           # Block Frobenius projection → doubly stochastic

# Residual update:
x_{l+1} = H_res_l @ x_l + H_post_l^T @ F(H_pre_l @ x_l, W_l)
```

## Integration Points in VELM

### Where to modify:
- `src/model/miras_backbone.py` → MirasBlock and SWABlock residual connections
- Currently: `output = x + block(x)` (single stream)
- go-mHC: `output = H_res @ x_streams + H_post @ block(H_pre @ x_streams)`

### Recommended config:
- d=4 streams (good balance of expressivity vs overhead)
- s=2 (covers most of Birkhoff polytope, default in paper)
- Per-layer params: ds(ds-1)/2 = 28 skew-symmetric params for d=4,s=2

### Architecture change:
```
# Current VELM backbone forward pass (simplified):
for block in blocks:
    x = x + block(x)  # single residual stream

# With go-mHC (d=4 streams):
x_streams = repeat(x, d=4)  # initialize d copies
for block in blocks:
    H_res = go_mhc_projection(learned_params)  # 4×4 doubly stochastic
    H_pre = learned_readout(x_streams)          # aggregate streams → 1
    H_post = learned_writeback(x_streams)       # distribute 1 → streams
    x_streams = H_res @ x_streams + H_post @ block(H_pre @ x_streams)
output = x_streams[0]  # or learned combination
```

### Interaction with EGGROLL:
- go-mHC adds ~28 params per layer for d=4,s=2 (trivial vs 7M backbone)
- These params are differentiable but also small enough for ES perturbation
- The Cayley transform is a smooth manifold → good for gradient-free search
- Doubly stochastic constraint is maintained by construction (no constraint
  violations from ES perturbations of the skew-symmetric params)

### Interaction with Miras memory:
- Miras layers already learn dynamic memory associations
- go-mHC adds dynamic *routing* between streams on top of that
- The two are complementary: Miras handles what to remember,
  go-mHC handles where information flows between layers

## Priority: Phase 2 (after EGGROLL + GEA work end-to-end)
- Don't integrate during current AE/EGGROLL debugging
- Implement as a separate branch after full pipeline validates
- Start with d=4 on the smoke config for testing
- Compare backbone next-vector prediction quality with/without

## Key Results from Paper (for reference)
- 30M GPT model: go-mHC(s=2) best on grammar(6.63), creativity(5.88),
  consistency(5.22) vs all baselines (Table 3)
- Synthetic tasks: reaches theoretical minimum loss, 10x faster convergence
- Gradient stability: all mHC variants similar, major improvement over
  unconstrained HC (Figure 9)
- Spectral reach: go-mHC(s=2) fills Karpelevič region much more than
  KromHC (Figure 4), approaching mHC-lite expressivity
