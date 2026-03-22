"""V-JEPA Pipeline: Object Pooling + Event Decoder end-to-end model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .event_decoder import EventPredictions, VJEPAEventDecoder
from .object_pooling import ObjectPoolingModule, ObjectRepresentation


class VJEPAPipeline(nn.Module):
    """End-to-end model combining Object Pooling and Event Decoder.

    Takes V-JEPA spatiotemporal tokens and produces event predictions.

    Args:
        object_pooling: ObjectPoolingModule instance.
        event_decoder: VJEPAEventDecoder instance.
    """

    def __init__(
        self,
        object_pooling: ObjectPoolingModule,
        event_decoder: VJEPAEventDecoder,
    ) -> None:
        super().__init__()
        self.object_pooling = object_pooling
        self.event_decoder = event_decoder

    def forward(
        self, vjepa_tokens: torch.Tensor
    ) -> tuple[ObjectRepresentation, EventPredictions]:
        """Forward pass.

        Args:
            vjepa_tokens: (B, S, input_dim) V-JEPA spatiotemporal tokens.

        Returns:
            Tuple of (ObjectRepresentation, EventPredictions).
        """
        obj_repr = self.object_pooling(vjepa_tokens)
        event_preds = self.event_decoder(obj_repr)
        return obj_repr, event_preds
