"""
VELM: Vector-Evolution Language Model

A self-evolving, continuous-latent, gradient-free language model architecture.

Modules:
  model/       - Architecture (CALM autoencoder, Miras backbone, energy head)
  training/    - EGGROLL optimizer and fitness functions
  inference/   - qTTT adaptation and CIB budget control
  evolution/   - GEA-EGGROLL self-improvement loop
"""

__version__ = "0.1.0"
