# Tech Stack

- **Language**: Python 3.12+
- **Compute**: [JAX](https://github.com/google/jax) (High-performance numerical computing)
- **Model Framework**: [Equinox](https://github.com/patrick-kidger/equinox) (Neural networks as PyTrees)
- **Optimization**: [Optax](https://github.com/google-deepmind/optax) (JAX-native gradient processing and optimization)
- **Data Processing**: 
  - HuggingFace `datasets` (Streaming large-scale data)
  - HuggingFace `tokenizers`
  - HuggingFace `transformers` (Tokenizer management)
- **Tensor Manipulations**: `einops`
- **Visualization & Logging**:
  - `matplotlib`
  - `tqdm`
  - `wandb` (Weights & Biases)
- **Build & Environment**:
  - `uv` (Fast Python package management)
  - `hatchling` (Build backend)
  - `ruff` (Fast linting and formatting)
  - `pytest` (Testing)
