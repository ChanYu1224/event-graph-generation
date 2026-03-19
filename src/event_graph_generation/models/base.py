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
    if config.name == "event_decoder":
        from .event_decoder import EventDecoder

        return EventDecoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_object_encoder_layers=config.num_object_encoder_layers,
            num_context_encoder_layers=config.num_context_encoder_layers,
            num_event_decoder_layers=config.num_event_decoder_layers,
            num_event_queries=config.num_event_queries,
            max_objects=config.max_objects,
            dropout=config.dropout,
            d_geo=config.d_geo,
            d_pair=config.d_pair,
            num_actions=config.num_actions,
            embedding_dim=config.embedding_dim,
        )
    return BaseModel(config)
