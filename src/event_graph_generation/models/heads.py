"""Prediction head modules for the Event Decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """Simple MLP prediction head: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(
        self, d_input: int, d_hidden: int, d_output: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., d_input).

        Returns:
            Logits tensor of shape (..., d_output).
        """
        return self.mlp(x)
