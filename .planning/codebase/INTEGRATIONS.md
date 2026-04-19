# Integrations

- **HuggingFace Hub**:
  - **Datasets**: Integrated for streaming the `OpenWebMath` dataset to handle large-scale math data without full local storage.
  - **Tokenizers**: Uses the Qwen3.5 tokenizer (248K vocab) for multilingual and technical text support.
- **Weights & Biases (WandB)**:
  - Primary integration for experiment tracking, hyperparameter sweeps, and artifact versioning.
- **Google Colab**:
  - Specialized workflows for training in ephemeral GPU environments (T4, A100, H100).
  - Automated backup and synchronization logic between `.ipynb` and `.py` files.
- **JAX/XLA**:
  - Deep integration with XLA for hardware-accelerated computation.
  - Custom memory management via `XLA_PYTHON_CLIENT_MEM_FRACTION`.
- **Equinox**:
  - Used for parameter serialization and model state management using PyTrees.
