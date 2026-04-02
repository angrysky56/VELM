"""VELM test suite — placeholder tests for initial project structure."""


def test_config_exists() -> None:
    """Verify model configs are importable."""
    from src.model.config import CONFIGS
    assert "tiny" in CONFIGS
    assert "small" in CONFIGS
    assert "medium" in CONFIGS
    assert "large" in CONFIGS


def test_config_chunk_size() -> None:
    """All configs should use K=4 as default chunk size."""
    from src.model.config import CONFIGS
    for name, cfg in CONFIGS.items():
        assert cfg["chunk_size_k"] == 4, f"{name} has unexpected K"
