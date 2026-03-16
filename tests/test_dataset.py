"""Tests for dataset module."""

from __future__ import annotations

import pytest

from event_graph_generation.data.dataset import BaseDataset


def test_base_dataset_not_implemented() -> None:
    """BaseDataset raises NotImplementedError for abstract methods."""
    with pytest.raises(NotImplementedError):
        BaseDataset(data_dir="/tmp", split="train")
