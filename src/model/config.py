"""
VELM Model — Full model composition

Composes CALM autoencoder + Miras backbone + Energy head into the
complete VELM model for both training and inference.
"""

# Model size configurations (from architecture.md §3.4)
CONFIGS: dict[str, dict] = {
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
    },
}
