"""Metric computation functions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy."""
    pred_labels = predictions.argmax(dim=-1)
    return (pred_labels == targets).float().mean().item()


def event_detection_map(predictions: dict, targets: dict) -> float:
    """Compute average precision for event detection based on interaction scores.

    Args:
        predictions: Dict with 'interaction' scores (list of floats per event),
            and 'matched' boolean flags.
        targets: Dict with 'num_events' per sample.

    Returns:
        Mean average precision across the batch.
    """
    interaction_scores = predictions.get("interaction", [])
    gt_labels = targets.get("labels", [])

    if len(interaction_scores) == 0 or len(gt_labels) == 0:
        return 0.0

    scores = np.array(interaction_scores, dtype=np.float64)
    labels = np.array(gt_labels, dtype=np.float64)

    # Sort by descending score
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    # Compute precision-recall curve
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (labels.sum() + 1e-8)

    # AP via trapezoidal rule
    ap = np.trapezoid(precision * sorted_labels, x=recall)
    return float(ap)


def action_accuracy(predictions: dict, targets: dict) -> float:
    """Accuracy of action classification for matched events.

    Args:
        predictions: Dict with 'action_classes' (list of predicted class indices).
        targets: Dict with 'action_classes' (list of ground-truth class indices).

    Returns:
        Fraction of correctly classified actions.
    """
    pred_actions = predictions.get("action_classes", [])
    gt_actions = targets.get("action_classes", [])

    if len(pred_actions) == 0 or len(gt_actions) == 0:
        return 0.0

    n = min(len(pred_actions), len(gt_actions))
    correct = sum(1 for i in range(n) if pred_actions[i] == gt_actions[i])
    return correct / n


def pointer_accuracy(predictions: dict, targets: dict) -> float:
    """Accuracy of agent/target pointer predictions.

    Args:
        predictions: Dict with 'agent_ptrs' and 'target_ptrs' lists.
        targets: Dict with 'agent_ptrs' and 'target_ptrs' lists.

    Returns:
        Average accuracy across agent and target pointers.
    """
    total_correct = 0
    total_count = 0

    for key in ("agent_ptrs", "target_ptrs"):
        pred = predictions.get(key, [])
        gt = targets.get(key, [])
        n = min(len(pred), len(gt))
        total_correct += sum(1 for i in range(n) if pred[i] == gt[i])
        total_count += n

    return total_correct / max(total_count, 1)


def frame_mae(predictions: dict, targets: dict) -> float:
    """Mean absolute error of frame predictions.

    Args:
        predictions: Dict with 'frame_indices' (list of predicted frame indices).
        targets: Dict with 'frame_indices' (list of ground-truth frame indices).

    Returns:
        Mean absolute error in frame indices.
    """
    pred_frames = predictions.get("frame_indices", [])
    gt_frames = targets.get("frame_indices", [])

    if len(pred_frames) == 0 or len(gt_frames) == 0:
        return 0.0

    n = min(len(pred_frames), len(gt_frames))
    errors = [abs(pred_frames[i] - gt_frames[i]) for i in range(n)]
    return sum(errors) / n


def graph_f1(predictions: dict, targets: dict) -> float:
    """F1 score comparing predicted vs GT event graphs edge-by-edge.

    Each event is represented as an edge tuple (agent, action, target).
    Computes precision, recall, and F1 between predicted and GT edge sets.

    Args:
        predictions: Dict with 'edges' as list of (agent, action, target) tuples.
        targets: Dict with 'edges' as list of (agent, action, target) tuples.

    Returns:
        F1 score.
    """
    pred_edges = set(tuple(e) for e in predictions.get("edges", []))
    gt_edges = set(tuple(e) for e in targets.get("edges", []))

    if len(pred_edges) == 0 and len(gt_edges) == 0:
        return 1.0
    if len(pred_edges) == 0 or len(gt_edges) == 0:
        return 0.0

    tp = len(pred_edges & gt_edges)
    precision = tp / len(pred_edges) if len(pred_edges) > 0 else 0.0
    recall = tp / len(gt_edges) if len(gt_edges) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


METRIC_REGISTRY: dict[str, Callable] = {
    "accuracy": accuracy,
    "event_detection_map": event_detection_map,
    "action_accuracy": action_accuracy,
    "pointer_accuracy": pointer_accuracy,
    "frame_mae": frame_mae,
    "graph_f1": graph_f1,
}


def get_metric(name: str) -> Callable:
    """Get a metric function by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]
