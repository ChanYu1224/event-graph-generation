"""Tests for metrics."""

from __future__ import annotations

import torch
import pytest

from event_graph_generation.evaluation.metrics import accuracy, get_metric


def test_accuracy_perfect() -> None:
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 0])
    assert accuracy(preds, targets) == 1.0


def test_accuracy_zero() -> None:
    preds = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([1, 0])
    assert accuracy(preds, targets) == 0.0


def test_get_metric() -> None:
    fn = get_metric("accuracy")
    assert fn is accuracy


def test_get_metric_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("nonexistent")
