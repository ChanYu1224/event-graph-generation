"""Metric computation functions."""

from __future__ import annotations

import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy."""
    pred_labels = predictions.argmax(dim=-1)
    return (pred_labels == targets).float().mean().item()


METRIC_REGISTRY: dict[str, callable] = {
    "accuracy": accuracy,
}


def get_metric(name: str) -> callable:
    """Get a metric function by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]
