"""Base model definition."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class BaseModel(nn.Module):
    """Base model skeleton. Replace with your architecture."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # Define layers here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_model(config: ModelConfig) -> nn.Module:
    """Factory function to build a model from config."""
    return BaseModel(config)
