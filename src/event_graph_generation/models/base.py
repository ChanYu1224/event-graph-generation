"""Base model definition."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig, VJEPAConfig


class BaseModel(nn.Module):
    """Base model skeleton. Replace with your architecture."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # Define layers here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_model(config: ModelConfig, vjepa_config: VJEPAConfig | None = None) -> nn.Module:
    """Factory function to build a model from config.

    Args:
        config: ModelConfig with model parameters.
        vjepa_config: Optional VJEPAConfig, required for vjepa_pipeline.

    Returns:
        Constructed model.
    """
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
    elif config.name == "vjepa_pipeline":
        from .event_decoder import VJEPAEventDecoder
        from .object_pooling import ObjectPoolingModule
        from .vjepa_pipeline import VJEPAPipeline

        op = config.object_pooling
        # Resolve V-JEPA dimensions from vjepa_config if provided
        if vjepa_config is not None:
            input_dim = vjepa_config.hidden_size
            temporal_tokens = vjepa_config.temporal_tokens
            spatial_tokens = vjepa_config.spatial_tokens
            frames_per_clip = vjepa_config.frames_per_clip
        else:
            input_dim = 1024
            temporal_tokens = 8
            spatial_tokens = 196
            frames_per_clip = 16

        object_pooling = ObjectPoolingModule(
            input_dim=input_dim,
            d_model=op.d_model,
            num_slots=op.num_slots,
            num_iterations=op.num_iterations,
            num_refinement_layers=op.num_refinement_layers,
            nhead=op.nhead,
            n_categories=op.n_categories,
            temporal_tokens=temporal_tokens,
            spatial_tokens=spatial_tokens,
            dropout=op.dropout,
        )
        event_decoder = VJEPAEventDecoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_context_encoder_layers=config.num_context_encoder_layers,
            num_event_decoder_layers=config.num_event_decoder_layers,
            num_event_queries=config.num_event_queries,
            num_slots=op.num_slots,
            dropout=config.dropout,
            num_actions=config.num_actions,
            temporal_window=frames_per_clip,
        )
        return VJEPAPipeline(object_pooling, event_decoder)
    return BaseModel(config)
