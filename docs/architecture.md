# VELM Architecture Specification

## 1. Design Principles

VELM is designed around three non-negotiable constraints that distinguish it from
standard LLM architectures:

1. **No backpropagation required.** The architecture must be trainable via
   zeroth-order methods (EGGROLL). This enables nonlinear RNNs, int8 native
   weights, and eliminates gradient infrastructure entirely.

2. **Continuous latent space.** The model operates over continuous vectors rather
   than discrete tokens. This breaks the information bottleneck of vocabulary-
   sized softmax (15-18 bits per token) and reduces autoregressive steps by Kx.

3. **Inference-time adaptation.** The model adapts to each input at inference via
   lightweight query-only gradient updates (qTTT), counteracting score dilution
   without retraining.

## 2. Component Architecture

### 2.1 CALM Autoencoder (Representation Layer)

**Purpose:** Bijective mapping between K discrete tokens and a single continuous
vector z ∈ R^l.

**Architecture (from CALM paper):**
- Encoder: Token embeddings → position-wise FFN → flatten (K×d → d) → FFN → linear → z ∈ R^l
- Decoder: z → linear → FFN → expand (d → K×d) → FFN → vocab logits → argmax
- Trained with cross-entropy reconstruction loss, >99.9% token accuracy
- Context-free (processes chunks independently)
- Lightweight: l=128 latent dimension optimal, autoencoder is ~negligible compute

**Key hyperparameters:**
- K = 4 (token chunk size, best performance-compute tradeoff)
- l = 128 (latent dimension)
- β = 0.001 (KL divergence weight for latent regularization)
- DropToken + DropLatent regularization for robust latent manifold

**VELM adaptation:**
- Train autoencoder first (standard backprop, it's small and simple)
- Freeze autoencoder, train backbone + head with EGGROLL
- Future: context-aware autoencoder conditioning on prior vectors

### 2.2 Miras Backbone (Sequence Processing)

**Purpose:** Replace Transformer self-attention with deep associative memory
that processes the sequence of continuous vectors from CALM.

**Why Miras over standard Transformer:**
- Miras provides a unified framework with 4 design choices
- Supports MLP-based deep memory (not just linear recurrence)
- Nonlinear RNN variants handle state tracking (impossible for SSMs/linear RNNs)
- EGGROLL makes nonlinear backprop-through-time unnecessary

**Miras 4-choice framework for VELM:**

1. **Memory Architecture:** MLP (same as Memora variant)
   - W_t ∈ R^{d_k × d_v} parameterizes the memory
   - More expressive than linear (matrix) memory used in DeltaNet/GLA
   - Can approximate arbitrary key-value mappings

2. **Attentional Bias:** ℓ2 regression loss
   - L(M(W; k_t), v_t) = ||M(W; k_t) - v_t||^2
   - Simple, stable, works well with gradient descent learning algo

3. **Retention Gate:** KL divergence (elastic net regularization)
   - Balances learning new associations vs retaining past state
   - W_t = Softmax(α_t log(W_{t-1}) - η_t ∇ℓ2(W_{t-1}; k_t, v_t))
   - Channel-wise parameterization with low-rank projections

4. **Memory Learning Algorithm:** Gradient descent
   - Standard SGD on the attentional bias objective
   - Parallelizable via chunk-wise computation

**Hybrid configuration (recommended):**
- Miras layer (deep memory) + Sliding Window Attention (SWA)
- SWA provides local context, Miras provides long-range state
- SWA layers enable qTTT query adaptation at inference
- Architecture follows Samba-style sequential combination

**Macro architecture (Llama-style):**
- RMSNorm, SwiGLU MLPs, RoPE (on SWA layers)
- 1D depthwise-separable convolution after Q, K, V projections
- ℓ2 normalization on q and k for stability
- Output: gated with linear layer

### 2.3 Energy-Based Generative Head

**Purpose:** Single-step prediction of the next continuous vector z_i from
the backbone's hidden state h_{i-1}.

**Architecture (from CALM):**
- Input: hidden state h ∈ R^d + random noise ε ~ U[-0.5, 0.5]^{d_noise}
- Both projected by independent linear layers to match head dimension d
- Stack of L residual MLP blocks (SwiGLU, residual connections)
- Each block fuses current representation with hidden state
- Final linear projection to latent dimension l
- L = num_transformer_layers / 4 (head is ~10% of total params)

**Training objective:** Energy loss (strictly proper scoring rule)
- Likelihood-free: no explicit probability computation needed
- Compatible with EGGROLL's fitness-based optimization
- Supports temperature sampling via rejection sampling algorithm

### 2.4 EGGROLL Training System

**Purpose:** Train the entire Miras + generative head without backpropagation.

**How EGGROLL works:**
- Low-rank perturbations E_i = (1/√r) A_i B_i^T where A ∈ R^{m×r}, B ∈ R^{n×r}
- Rank r=1 is sufficient (minimal performance loss vs full-rank)
- Population members share base weights, only differ by low-rank adapters
- Fitness evaluation is pure forward pass (inference-only)
- Counter-based RNG eliminates need to store perturbation matrices
- Achieves 91% of pure batch inference throughput

**VELM-specific fitness function:**

```
fitness(θ) = α × quality_score(θ) + β × compression_score(θ)

where:
  quality_score = BrierLM on next-vector prediction
  compression_score = CIB objective (reward shorter, accurate reasoning)
```

**Training pipeline:**
1. Pre-train CALM autoencoder (standard backprop, small model)
2. Freeze autoencoder
3. Initialize Miras backbone + energy head
4. EGGROLL training with population size 2^16 - 2^20
   - Each member generates text, evaluated on quality + compression
   - Low-rank updates (rank 1) with σ ≈ 0.001
   - Adam-like moment tracking on ES gradient estimates
5. For int8 training: discretized Adam updates (z-score threshold τ)
   - Sparse unit-step updates: ∆ = sign(z) · 1{|z| ≥ τ} ∈ {-1, 0, +1}
   - Clip to valid int8 range [-127, 127]

### 2.5 qTTT Inference Adaptation

**Purpose:** Fix long-context score dilution at inference time without
retraining or modifying the KV cache.

**The problem qTTT solves:**
- As context length T grows, attention mass on target tokens collapses
- Thinking tokens cannot fix this: they use the same static attention
- The margin must scale as Ω(log T) — impossible with frozen weights

**qTTT procedure for VELM:**
1. Single prefill pass: compute and cache {K^(ℓ), V^(ℓ)} for all layers
2. For N_qTTT steps:
   a. Sample random span x_s = x_{t:t+k} from context (k ≪ T)
   b. Compute next-token loss using frozen KV cache
   c. Update ONLY query projections: W_Q ← W_Q - η ∇_{W_Q} L_TTT
3. Generate answer with adapted queries

**Why this works (Proposition 3.1 from paper):**
- Gradient ∇_{q_i} ℓ_i = (1/√d_k)(μ_i - k_{j*})
- Descent moves q_i toward target key k_{j*} and away from mean μ_i
- Margin strictly increases: M(q_i^+) = M(q_i) + η||∇ℓ_i||² + O(η²)
- Improvement is largest when attention is most diffuse

**FLOP equivalence:** T_think ≈ 2 × N_qTTT × k
- 8K thinking tokens ≈ 16 qTTT steps on 128-token spans
- qTTT reshapes queries; thinking just grows the cache

### 2.6 CIB Reasoning Compression

**Purpose:** Treat chain-of-thought reasoning as a compression problem.
Minimize reasoning tokens while preserving answer accuracy.

**Integration with VELM:**
- CIB regularization during EGGROLL training: penalize verbose reasoning
- At inference: adaptive token budget per query complexity
- Reduces compute cost of reasoning by ~78% (from CIB paper benchmarks)
- Naturally compatible with CALM's K-token chunks (compress reasoning chunks)

**Dual role:**
- Training: CIB loss term in EGGROLL fitness function
- Inference: Budget controller deciding how many reasoning steps to execute

### 2.7 GEA Self-Improvement Loop

**Purpose:** Continuous autonomous improvement of the VELM system through
group-based evolution of model populations.

**How GEA integrates with EGGROLL:**
- EGGROLL already maintains a population of weight perturbations
- GEA upgrades this from isolated fitness evaluation to group experience sharing
- At each evolution iteration:
  1. Select parent group via Performance-Novelty criterion
  2. Share evolutionary traces across group members:
     - Weight perturbation histories that improved fitness
     - Reasoning patterns that solved difficult problems
     - Tool usage / workflow improvements
  3. Each member evolves using shared group experience
  4. Successful innovations propagate to the whole population

**What GEA evolves in VELM:**
- Weight perturbation directions (which low-rank subspaces improve fitness)
- Reasoning strategies (which CoT patterns solve which problem types)
- qTTT hyperparameters (N_qTTT, k, η per task type)
- CIB budget allocation (how aggressively to compress per difficulty)

**Performance-Novelty selection:**
- score(i) = α_i × √nov(i)
- Performance = fitness on downstream tasks
- Novelty = cosine distance to M nearest neighbors in embedding space
- Balances exploitation (high-performing variants) with exploration

## 3. Full System Integration

### 3.1 Data Flow (Inference)

```
Input text: "The quick brown fox jumps over the lazy dog"
    ↓
CALM Encoder: chunk into K=4 tokens each
  ["The quick brown fox"] → z₁ ∈ R^128
  ["jumps over the lazy"]  → z₂ ∈ R^128
  ["dog"]                  → z₃ ∈ R^128
    ↓
Miras Backbone: process sequence [z₁, z₂, z₃]
  - Deep MLP memory updates state at each step
  - SWA layers attend locally (for qTTT-compatible attention)
  - Output: hidden states [h₁, h₂, h₃]
    ↓
```
```
Energy Head: predict next vector from h₃
  - Inject noise ε ~ U[-0.5, 0.5]
  - Refine through L residual MLP blocks
  - Output: z₄ ∈ R^128
    ↓
CALM Decoder: z₄ → 4 tokens ["..." predicted continuation]
    ↓
(Loop: feed predicted tokens back as next input chunk)
```

### 3.2 Data Flow (Long Context with qTTT)

```
Long input: 100K tokens → 25K continuous vectors (K=4)
    ↓
Phase 1: Single prefill through full model
  - Cache all K^(ℓ), V^(ℓ) from SWA layers
  - Miras recurrent state captures global context
  - Cost: O(T²) for SWA, O(T) for Miras layers
    ↓
Phase 2: qTTT adaptation (16 steps, k=128 spans)
  - Sample random spans from the cached context
  - Update only W_Q in SWA layers
  - Margin improvement: queries move toward target keys
  - Cost: equivalent to ~8K thinking tokens
    ↓
Phase 3: Generate answer with adapted model
  - SWA queries now target relevant context
  - Miras state provides complementary long-range signal
  - CIB budget controls reasoning length
```

### 3.3 Data Flow (Training with EGGROLL)

```
For each training step:
  1. Sample batch of text sequences
  2. For each population member i ∈ {1, ..., N_workers}:
     a. Generate low-rank perturbation: E_i = (1/√r) A_i B_i^T
     b. Apply to base weights: W_i = M + σE_i
     c. Forward pass through CALM encoder → Miras → Energy Head
     d. Evaluate fitness:
        f_i = quality(next-vector prediction) + λ × compression(CIB)
  3. Workers share scalar fitness values
  4. Update base weights: M ← M + α × (1/N) Σ E_i f_i
```

### 3.4 Model Size Configurations

| Config | Layers | Dim  | Heads | Miras:SWA | Params | Notes |
|--------|--------|------|-------|-----------|--------|-------|
| Tiny   | 12     | 768  | 16    | 6:6       | ~170M  | Proof of concept |
| Small  | 24     | 1024 | 16    | 12:12     | ~340M  | Ablation studies |
| Medium | 24     | 1536 | 16    | 12:12     | ~780M  | Main experiments |
| Large  | 32     | 2048 | 32    | 16:16     | ~1.5B  | Scaling validation |

Energy head: L = num_layers / 4, ~10% of total params
CALM autoencoder: ~50M params (fixed across all configs)

## 4. Hardware Considerations

### 4.1 Int8 Native Training Path

EGGROLL enables pure int8 training (demonstrated with EGG architecture):
- All weights in int8, all activations in integer formats
- No floating point required at any point during training
- int8 matmul with int32 accumulation = fastest tensor core op on modern GPUs
- Saturated int8 addition provides implicit nonlinearity (no activation functions)

**Implication for VELM:**
- The Miras backbone can be designed for int8 from the ground up
- Energy savings: int8 ops are ~4x more energy efficient than fp16
- Memory savings: 2x reduction vs fp16, 4x vs fp32
- Your RTX 3060 12GB could hold a ~1.5B int8 model in memory

### 4.2 Compute Requirements (Estimates)

| Stage | Config | GPU Hours (est) | Hardware |
|-------|--------|----------------|----------|
| AE training | All | ~10 | 1x RTX 3060 |
| VELM-Tiny EGGROLL | pop=2^16 | ~200 | 1x RTX 3060 |
| VELM-Small EGGROLL | pop=2^18 | ~2000 | 4x A100 |
| VELM-Medium EGGROLL | pop=2^20 | ~10000 | 8x H100 |
| qTTT inference | Any | ~2x prefill | Same as inference |

Note: EGGROLL achieves 91% of pure batch inference throughput.
Population scaling is embarrassingly parallel — more GPUs = larger populations.

## 5. Open Research Questions

1. **CALM + Miras interaction:** Does continuous-vector input change the optimal
   memory architecture choice? The Miras paper assumes discrete token inputs.

2. **EGGROLL scaling for VELM:** EGGROLL demonstrated int8 GRU pretraining at
   small scale (6L-256D). Does it scale to the deeper Miras architectures?

3. **qTTT over continuous vectors:** The qTTT paper operates on standard token
   attention. Does query adaptation work as well when keys/values represent
   compressed token chunks?

4. **CIB + CALM chunk alignment:** CIB compresses reasoning tokens. When reasoning
   operates over K-token chunks, does compression become more or less efficient?

5. **GEA convergence:** Does group evolution converge faster when populations share
   EGGROLL perturbation directions rather than just fitness evaluations?

6. **Semantic bandwidth scaling:** CALM shows K=4 is optimal for their setup.
   Does the Miras backbone change this? Can deeper memory support larger K?

7. **Hybrid ratio:** What's the optimal Miras:SWA layer ratio for different
   tasks? More Miras layers help state tracking; more SWA layers help qTTT.

8. **Int8 + continuous latents:** CALM's autoencoder uses float latent vectors.
   Can the backbone operate on quantized latent representations?
