"""Tests for config loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from event_graph_generation.config import Config


def test_from_yaml_defaults() -> None:
    """Config loads with defaults when YAML is minimal."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({}, f)
        f.flush()
        config = Config.from_yaml(f.name)

    assert config.data.batch_size == 32
    assert config.training.learning_rate == 1e-3
    assert config.training.optimizer == "adamw"


def test_from_yaml_override() -> None:
    """Config respects YAML overrides."""
    data = {"data": {"batch_size": 64}, "training": {"learning_rate": 5e-4}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        config = Config.from_yaml(f.name)

    assert config.data.batch_size == 64
    assert config.training.learning_rate == 5e-4
    assert config.training.optimizer == "adamw"  # default preserved


def test_merge() -> None:
    """Config.merge applies overrides from a second YAML."""
    base_data = {"training": {"epochs": 100}}
    override_data = {"training": {"epochs": 50, "learning_rate": 1e-4}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(base_data, f)
        base_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(override_data, f)
        override_path = f.name

    config = Config.from_yaml(base_path).merge(override_path)
    assert config.training.epochs == 50
    assert config.training.learning_rate == 1e-4


def test_to_yaml_roundtrip() -> None:
    """Config survives a YAML roundtrip."""
    config = Config()
    config.data.batch_size = 128

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        out_path = f.name

    config.to_yaml(out_path)
    loaded = Config.from_yaml(out_path)
    assert loaded.data.batch_size == 128
