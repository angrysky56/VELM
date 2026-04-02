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
    "gpu_12gb": {
        "num_layers": 8,
        "hidden_dim": 256,
        "num_heads": 8,
        "miras_layers": 4,
        "swa_layers": 4,
        "ffn_intermediate": 512,
        "energy_head_blocks": 2,
        "chunk_size_k": 4,
        "latent_dim": 64,
        "ae_hidden_dim": 256,
        "ae_ffn_intermediate": 512,
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
