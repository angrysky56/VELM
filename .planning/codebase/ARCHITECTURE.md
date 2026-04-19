# Architecture

VELM (Vector-Evolution Language Model) is a continuous-latent, gradient-free language model architecture.

## Core Components

- **CALM Autoencoder (`src/model/autoencoder.py`)**:
  - Encodes discrete token chunks (size K) into a continuous latent vector `z`.
  - High-fidelity reconstruction (>99.9% accuracy).
- **Miras Backbone (`src/model/miras_backbone.py`)**:
  - Memory-augmented or structural backbone that predicts the next latent vector in the sequence.
- **Energy Head (`src/model/energy_head.py`)**:
  - An energy-based prediction head for refining latent vector transitions.
- **Hyper-connections (`src/model/hyper_connections.py`)**:
  - Advanced routing or weight generation mechanisms for dynamic architecture adjustment.

## Training Pipeline

- **EGGROLL (`src/training/eggroll.py`)**:
  - "Evolutionary Gradient-free Global Regression with Optimal Low-rank Learning".
  - Structured low-rank perturbations for high-throughput population-based training.
- **GEA (`src/evolution/gea_eggroll.py`)**:
  - Genetic Evolutionary Algorithm applied to the EGGROLL optimization process.
- **Curriculum Learning**:
  - Data loading strategy that adjusts complexity over time.
