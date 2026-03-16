"""Evaluation loop."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import EvaluationConfig
from .metrics import get_metric


class Evaluator:
    """Runs evaluation on a dataset."""

    def __init__(self, config: EvaluationConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self.metrics = {name: get_metric(name) for name in config.metrics}

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on the given dataloader."""
        model.eval()
        all_predictions: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in dataloader:
            outputs = model(batch.inputs.to(self.device))
            all_predictions.append(outputs.cpu())
            all_targets.append(batch.targets)

        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        results = {}
        for name, metric_fn in self.metrics.items():
            results[name] = metric_fn(predictions, targets)

        return results
