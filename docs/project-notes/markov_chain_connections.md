# Doubly Stochastic Matrices as Markov Chains: Theoretical Connections

## The Mathematical Identity

A doubly stochastic matrix isn't merely analogous to a Markov chain transition
matrix — it *is* one. Specifically, it is the transition matrix of a reversible
Markov chain whose unique stationary distribution is uniform: π = (1/d, ..., 1/d).

When go-mHC computes `x_{l+1} = H_res_l @ x_l`, this is a literal Markov chain
step. The d residual streams are states, H_res_ij is the transition probability
from stream j to stream i, and the doubly stochastic constraint guarantees:
- Row sums = 1: probability is conserved (each state transitions somewhere)
- Column sums = 1: the uniform distribution is stationary (information balance)

This isn't a loose metaphor. The eigenstructure, mixing time analysis, and
convergence theorems of Markov chain theory apply directly.

## Depth as Mixing Time

The go-mHC paper's Appendix B.5 demonstrates a core result from Markov chain
theory without explicitly naming it: the product of independent doubly stochastic
matrices across L layers converges to (1/d)J_d (the uniform mixing matrix).

This is precisely the **mixing time** result: after sufficient transitions, any
initial distribution converges to the stationary distribution. The rate of
convergence is governed by the **spectral gap** (1 - |λ₂|), where λ₂ is the
second-largest eigenvalue magnitude.

Their Figure A9 tracks |λ₂| decaying toward zero across composed layers — this
is the standard diagnostic for Markov chain mixing. Larger spectral gaps mean
faster mixing, meaning distant layers become effectively independent in their
routing decisions.

### Practical Consequence: Depth Decoupling

Because the composed transition matrix H_res_{l→L} converges to the barycenter,
layers far apart in the network become effectively independent. Each layer can
learn its mixing pattern without long-range backward dependencies through the
residual stream. This is a free architectural benefit inherited directly from
Markov chain ergodicity — the go-mHC paper notes it but doesn't emphasize the
theoretical origin.

## Why Discrete State Aggregation Fails

### The Lumpability Problem

When a Markov chain has too many states to be tractable, the natural move is to
aggregate ("lump") states into macro-states. But preserving the Markov property
under aggregation is brutally restrictive:

- **Strong lumpability** requires that transition probabilities between macro-states
  are identical regardless of the micro-state distribution within them. For
  heterogeneous neural network representations, finding a perfectly strong
  lumpable partition is effectively impossible.

- **Weak lumpability** preserves the Markov property only for a specific initial
  distribution. Any perturbation during training or inference shatters the
  property and voids all predictive guarantees.

This is the fundamental reason you can't simply discretize the latent space into
a small number of states and run standard Markov chain algorithms on it. The
representation distributions inside a neural network are far too heterogeneous
and non-stationary for lumpability conditions to hold.

### The ε-Machine Problem

Computational mechanics (Crutchfield & Shalizi) offers the theoretically optimal
approach: ε-machines identify "causal states" by grouping all historical pasts
that yield identical conditional future distributions. These are the minimal
sufficient statistics for prediction.

Why ε-machines are intractable for neural network internals:

1. **Infinite horizons**: True causal states require observing semi-infinite pasts
   and projecting semi-infinite futures. Inside a network with finite depth, this
   is a conceptual mismatch.

2. **Statistical complexity explosion**: Discovering the equivalence classes and
   computing statistical complexity (Shannon entropy of causal states) scales
   combinatorially. For dynamic, non-stationary latent representations, real-time
   ε-machine construction is computationally impossible.

3. **Non-stationarity**: The latent distributions shift during training. Any
   ε-machine computed at step t is invalid at step t+1, requiring continuous
   reconstruction of the entire causal state partition.

## How Continuous Geometric Constraints Sidestep the Problem

go-mHC avoids the discrete aggregation nightmare entirely by working in
continuous geometry:

- Instead of discovering the "right" macro-states (lumpability) or causal
  equivalence classes (ε-machines), it fixes d continuous streams and constrains
  their mixing to the Birkhoff polytope interior.

- The Cayley transform parameterization (skew-symmetric → orthogonal → doubly
  stochastic) ensures the transition matrix is always exactly balanced while
  remaining fully differentiable. No iterative projection, no approximation gap.

- The model learns smooth probability trajectories via gradient descent rather
  than fighting combinatorial state-space enumeration.

This is a shift from "find the right discrete structure" to "learn within a
continuous structure that has the right mathematical properties by construction."

## Not MCTS, But Closer to Particle Filtering

The MCTS comparison is tempting but mechanistically wrong. MCTS actively
searches at runtime — it pauses, simulates discrete rollouts, evaluates rewards,
and backtracks. go-mHC is deterministic soft-routing computed in a single
forward pass with no lookahead or backtracking.

The better analogy is **particle filtering** (sequential Monte Carlo). The d
streams function as d parallel hypotheses about the representation. At each
layer, the doubly stochastic mixing:
- Redistributes belief mass across hypotheses (reweighting)
- Spreads useful content to other streams (resampling)
- Overwrites less useful streams (particle death/rebirth)

All paths are explored simultaneously in continuous superposition, not searched
sequentially. The doubly stochastic constraint ensures no hypothesis accumulates
unbounded weight (no particle degeneracy).

## Connections to VELM Architecture

### Miras Memory + go-mHC Routing: Complementary Geometric Constraints

VELM's Miras backbone already operates on constrained geometric objects:

- **Memora variant** (Eq. 21 in Miras paper): Memory updates are constrained to
  a scaled probability simplex via KL divergence retention gates. The update
  rule `W_t = Softmax(α_t log(W_{t-1}) - η_t ∇ℓ)` keeps memory weights on the
  simplex. This constrains *what* gets stored.

- **go-mHC** would add: Inter-layer routing constrained to the Birkhoff polytope
  (the set of doubly stochastic matrices). This constrains *where* information
  flows between layers.

These are different geometric objects (simplex for memory content, Birkhoff
polytope for routing) but both enforce conservation-like properties:
- Simplex: memory weights sum to a constant (bounded capacity)
- Birkhoff: transition probabilities sum to 1 in both directions (information
  neither created nor destroyed across streams)

Both prevent the gradient pathologies (vanishing/exploding) that plague
unconstrained deep networks — the simplex bounds memory magnitudes, the
Birkhoff polytope bounds the spectral radius of the composed routing matrix.

### EGGROLL Compatibility

The skew-symmetric parameters of go-mHC are particularly well-suited to
gradient-free evolution strategies:
- They're a small parameter set (ds(ds-1)/2 ≈ 28 for d=4, s=2)
- Any perturbation of skew-symmetric params produces a valid skew-symmetric
  matrix (the constraint is linear)
- The Cayley transform then guarantees the output is exactly orthogonal
- The Frobenius projection guarantees exactly doubly stochastic
- So EGGROLL perturbations can *never* violate the manifold constraint

This is stronger than what you'd get perturbing raw weight matrices — there's
no need for projection or clipping after perturbation.

## References

- Crutchfield, J.P. & Shalizi, C.R. — Computational Mechanics (ε-machines)
- Kemeny & Snell — Finite Markov Chains (lumpability, mixing times)
- Dandachi & Diggs-Galligan (2604.02309) — go-mHC paper
- Behrouz et al. (2504.13173) — Miras framework
- Nechita et al. (2310.03436) — Generalized orthostochastic matrices
