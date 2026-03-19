"""DETR-style Event Decoder for set prediction of events."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .heads import PredictionHead


@dataclass
class EventPredictions:
    """Predictions from the Event Decoder."""

    interaction: torch.Tensor  # (B, M, 1)
    action: torch.Tensor  # (B, M, A)
    agent_ptr: torch.Tensor  # (B, M, K)
    target_ptr: torch.Tensor  # (B, M, K)
    source_ptr: torch.Tensor  # (B, M, K+1)
    dest_ptr: torch.Tensor  # (B, M, K+1)
    frame: torch.Tensor  # (B, M, T)


class EventDecoder(nn.Module):
    """DETR-style set prediction model for event graph generation.

    Encodes object representations from SAM embeddings and geometric features,
    then decodes a fixed set of event queries via cross-attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_object_encoder_layers: int = 3,
        num_context_encoder_layers: int = 3,
        num_event_decoder_layers: int = 4,
        num_event_queries: int = 20,
        max_objects: int = 30,
        dropout: float = 0.1,
        d_geo: int = 12,
        d_pair: int = 7,
        num_actions: int = 13,
        embedding_dim: int = 256,
        temporal_window: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_event_queries = num_event_queries
        self.max_objects = max_objects
        self.temporal_window = temporal_window

        # Projection layers
        self.geo_proj = nn.Linear(d_geo, d_model)
        self.pair_proj = nn.Linear(d_pair, d_model)
        self.emb_proj = nn.Linear(embedding_dim, d_model)

        # Temporal encoder: self-attention over objects
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_object_encoder_layers
        )

        # Context encoder: self-attention with object mask
        context_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer, num_layers=num_context_encoder_layers
        )

        # Learnable event queries (scaled by 1/sqrt(d_model) for stable init)
        self.event_queries = nn.Parameter(
            torch.randn(num_event_queries, d_model) / (d_model ** 0.5)
        )

        # Event decoder: cross-attention between queries and object slots
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.event_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_event_decoder_layers
        )

        # Prediction heads
        d_hidden = d_model
        self.interaction_head = PredictionHead(d_model, d_hidden, 1, dropout=dropout)
        self.action_head = PredictionHead(d_model, d_hidden, num_actions, dropout=dropout)
        self.agent_ptr_head = PredictionHead(d_model, d_hidden, max_objects, dropout=dropout)
        self.target_ptr_head = PredictionHead(d_model, d_hidden, max_objects, dropout=dropout)
        self.source_ptr_head = PredictionHead(d_model, d_hidden, max_objects + 1, dropout=dropout)
        self.dest_ptr_head = PredictionHead(d_model, d_hidden, max_objects + 1, dropout=dropout)
        self.frame_head = PredictionHead(d_model, d_hidden, temporal_window, dropout=dropout)

    def forward(
        self,
        object_embeddings: torch.Tensor,
        object_temporal: torch.Tensor,
        pairwise: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> EventPredictions:
        """Forward pass.

        Args:
            object_embeddings: (B, K, D_emb) SAM object embeddings.
            object_temporal: (B, K, T, D_geo) per-object geometric features over time.
            pairwise: (B, K, K, T, D_pair) pairwise features over time.
            object_mask: (B, K) boolean mask, True for valid objects.

        Returns:
            EventPredictions with all prediction head outputs.
        """
        B, K, T, D_geo = object_temporal.shape

        # 1. Project geometric temporal features and mean-pool over T
        geo_projected = self.geo_proj(object_temporal)  # (B, K, T, d_model)
        geo_repr = geo_projected.mean(dim=2)  # (B, K, d_model)

        # 2. Project SAM embeddings
        emb_repr = self.emb_proj(object_embeddings)  # (B, K, d_model)

        # 3. Combine
        object_repr = emb_repr + geo_repr  # (B, K, d_model)

        # 4. Temporal encoder: self-attention over objects
        # Use inverted mask for transformer (True = ignore)
        src_key_padding_mask = ~object_mask  # (B, K)
        object_repr = self.temporal_encoder(
            object_repr, src_key_padding_mask=src_key_padding_mask
        )  # (B, K, d_model)

        # 5. Pairwise: project, pool over T, and add to object repr
        # pairwise: (B, K, K, T, D_pair) -> (B, K, K, T, d_model) -> mean over T and K_j
        pair_projected = self.pair_proj(pairwise)  # (B, K, K, T, d_model)
        pair_pooled = pair_projected.mean(dim=3)  # (B, K, K, d_model)
        # Mask invalid objects in pairwise dimension (K_j axis = dim 2)
        pair_mask = object_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, K, 1)
        pair_pooled = pair_pooled * pair_mask  # (B, K, K, d_model)
        # Sum over K_j (dim=2) and normalize
        pair_count = object_mask.float().sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)  # (B, 1, 1)
        pair_agg = pair_pooled.sum(dim=2) / pair_count  # (B, K, d_model)
        object_repr = object_repr + pair_agg

        # 6. Context encoder: self-attention with object mask
        object_slots = self.context_encoder(
            object_repr, src_key_padding_mask=src_key_padding_mask
        )  # (B, K, d_model)

        # 7. Event decoder: cross-attention between event_queries and object_slots
        queries = self.event_queries.unsqueeze(0).expand(B, -1, -1)  # (B, M, d_model)
        event_repr = self.event_decoder(
            queries,
            object_slots,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (B, M, d_model)

        # 8. Apply prediction heads
        interaction = self.interaction_head(event_repr)  # (B, M, 1)
        action = self.action_head(event_repr)  # (B, M, A)
        agent_ptr = self.agent_ptr_head(event_repr)  # (B, M, K)
        target_ptr = self.target_ptr_head(event_repr)  # (B, M, K)
        source_ptr = self.source_ptr_head(event_repr)  # (B, M, K+1)
        dest_ptr = self.dest_ptr_head(event_repr)  # (B, M, K+1)
        frame = self.frame_head(event_repr)  # (B, M, T)

        # 9. Mask invalid objects in pointer logits
        # object_mask: (B, K) -> expand to (B, M, K)
        ptr_mask = object_mask.unsqueeze(1).expand(-1, self.num_event_queries, -1)
        neg_inf = float("-inf")

        agent_ptr = agent_ptr.masked_fill(~ptr_mask, neg_inf)
        target_ptr = target_ptr.masked_fill(~ptr_mask, neg_inf)
        # source/dest have K+1 slots; last slot is "none", always valid
        src_dest_mask = torch.cat(
            [ptr_mask, torch.ones(B, self.num_event_queries, 1, device=object_mask.device, dtype=torch.bool)],
            dim=2,
        )  # (B, M, K+1)
        source_ptr = source_ptr.masked_fill(~src_dest_mask, neg_inf)
        dest_ptr = dest_ptr.masked_fill(~src_dest_mask, neg_inf)

        return EventPredictions(
            interaction=interaction,
            action=action,
            agent_ptr=agent_ptr,
            target_ptr=target_ptr,
            source_ptr=source_ptr,
            dest_ptr=dest_ptr,
            frame=frame,
        )
