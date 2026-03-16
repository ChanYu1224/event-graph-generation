"""Optimizer and scheduler factory functions."""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR

from ..config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Build optimizer from config."""
    name = config.optimizer.lower()
    params = model.parameters()

    if name == "adam":
        return optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif name == "sgd":
        return optim.SGD(
            params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: optim.Optimizer, config: TrainingConfig
) -> LRScheduler | None:
    """Build learning rate scheduler from config."""
    name = config.scheduler.lower()

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.scheduler_params.T_max)
    elif name == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")
