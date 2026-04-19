"""
VELM Model — Configuration

Model size configurations ranging from GPU-constrained experiments
to full research-scale models.

Tokenizer: Qwen3.5 BPE (vocab_size=248320, 201 languages)
"""

# Qwen3.5 tokenizer vocabulary size
# len(tokenizer) = 248077 actual tokens; Qwen pads to 248320 for GPU alignment
# We use the actual count to avoid wasted embedding parameters
QWEN35_VOCAB_SIZE = 248077

# Default tokenizer model ID for loading from HuggingFace
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B"

# Model size configurations
CONFIGS: dict[str, dict] = {
    # RTX 3060 12GB / T4 16GB — functional experiments
    # latent_dim=128 matches CALM paper optimal (Fig 7); 64 was too tight
    # for 248K vocab, causing ~97% accuracy ceiling instead of >99.9%
    "gpu_12gb": {
        "num_layers": 8,
        "hidden_dim": 256,
        "num_heads": 8,
        "miras_layers": 4,
        "swa_layers": 4,
        "ffn_intermediate": 512,
        "energy_head_blocks": 2,
        "chunk_size_k": 4,
        "latent_dim": 128,         # was 64 — paper optimal is 128 for K=4
        "ae_hidden_dim": 256,
        "ae_ffn_intermediate": 512,
        "ae_kl_clip": 0.5,         # paper uses λ_KL=0.5, not 1.0
        "ae_kl_weight": 0.001,     # paper uses β=0.001
        "eggroll_sigma": 0.001,    # paper uses σ=0.001, NOT 0.01
        "eggroll_lr": 3e-4,        # conservative for noisy ES gradients
    },
    # RTX 3060 12GB / T4 16GB — higher capacity AE for 248K vocab
    # Use this if gpu_12gb plateaus below 99.9% reconstruction accuracy.
    # The 248K Qwen vocab needs more hidden_dim than CALM's 32K Llama vocab.
    "gpu_12gb_v2": {
        "num_layers": 8,
        "hidden_dim": 256,
        "num_heads": 8,
        "miras_layers": 4,
        "swa_layers": 4,
        "ffn_intermediate": 512,
        "energy_head_blocks": 2,
        "chunk_size_k": 4,
        "latent_dim": 128,
        "ae_hidden_dim": 384,      # 256→384: more capacity for 248K vocab logits
        "ae_ffn_intermediate": 768, # scale FFN proportionally
        "ae_kl_clip": 0.5,
        "ae_kl_weight": 0.001,
        "eggroll_sigma": 0.001,
        "eggroll_lr": 3e-4,
        "eggroll_antithetic": True,   # ±σ pairs: halves variance for small pop
        "eggroll_eval_batch": 64,     # balance speed vs stability on T4
        "hc_streams": 4,             # d: number of residual streams (go-mHC)
        "hc_s": 2,                   # s: orthostochastic expressivity param
        "n_loops": 1,                # Recurrent depth (Mythos-Enhanced RDT)
        "use_act": False,            # Adaptive Compute Time (ACT)
    },
    # Smoke test / CI — minimal viable forward pass
    "smoke": {
        "num_layers": 4,
        "hidden_dim": 64,
        "num_heads": 4,
        "miras_layers": 2,
        "swa_layers": 2,
        "ffn_intermediate": 128,
        "energy_head_blocks": 1,
        "chunk_size_k": 4,
        "latent_dim": 16,
        "ae_hidden_dim": 64,
        "ae_ffn_intermediate": 128,
    },
    # Research scale configs (multi-GPU)
    "tiny": {
        "num_layers": 12,
        "hidden_dim": 768,
        "num_heads": 16,
        "miras_layers": 6,
        "swa_layers": 6,
        "ffn_intermediate": 2048,
        "energy_head_blocks": 3,
        "chunk_size_k": 4,
        "latent_dim": 128,
        "ae_hidden_dim": 512,
        "ae_ffn_intermediate": 1024,
    },
    "small": {
        "num_layers": 24,
        "hidden_dim": 1024,
        "num_heads": 16,
        "miras_layers": 12,
        "swa_layers": 12,
        "ffn_intermediate": 2816,
        "energy_head_blocks": 6,
        "chunk_size_k": 4,
        "latent_dim": 128,
        "ae_hidden_dim": 512,
        "ae_ffn_intermediate": 1024,
    },
    "medium": {
        "num_layers": 24,
        "hidden_dim": 1536,
        "num_heads": 16,
        "miras_layers": 12,
        "swa_layers": 12,
        "ffn_intermediate": 4096,
        "energy_head_blocks": 6,
        "chunk_size_k": 4,
        "latent_dim": 128,
        "ae_hidden_dim": 512,
        "ae_ffn_intermediate": 1024,
    },
    "large": {
        "num_layers": 32,
        "hidden_dim": 2048,
        "num_heads": 32,
        "miras_layers": 16,
        "swa_layers": 16,
        "ffn_intermediate": 5504,
        "energy_head_blocks": 8,
        "chunk_size_k": 4,
        "latent_dim": 128,
        "ae_hidden_dim": 512,
        "ae_ffn_intermediate": 1024,
    },
}
