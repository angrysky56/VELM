# Testing

- **Framework**: `pytest`.
- **Smoke Tests**: `tests/test_smoke.py` verifies model instantiation and basic forward passes across all components.
- **Config Tests**: `tests/test_config.py` ensures model hyperparameters are valid for the given vocab and hardware profiles.
- **Validation Gates**:
  - Autoencoder must achieve >99.9% reconstruction accuracy on test sets.
  - EGGROLL fitness must show monotonic improvement in stability tests.
- **Environment**: All tests are designed to run in a `uv`-managed virtual environment.
