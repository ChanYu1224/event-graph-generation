"""Batch collation for EventGraphDataset."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EventBatch:
    """Collated batch for event graph generation."""

    object_embeddings: torch.Tensor  # (B, K, D_emb)
    object_temporal: torch.Tensor  # (B, K, T, D_geo)
    pairwise: torch.Tensor  # (B, K, K, T, D_pair)
    object_mask: torch.Tensor  # (B, K)
    gt_events: list[list[dict]]  # B x variable-length event lists
    num_objects: list[int]

    def to(self, device: str | torch.device) -> "EventBatch":
        """Move all tensor fields to the given device."""
        return EventBatch(
            object_embeddings=self.object_embeddings.to(device),
            object_temporal=self.object_temporal.to(device),
            pairwise=self.pairwise.to(device),
            object_mask=self.object_mask.to(device),
            gt_events=self.gt_events,
            num_objects=self.num_objects,
        )


def event_collate_fn(samples: list[dict]) -> EventBatch:
    """Collate a list of sample dicts into an EventBatch.

    Args:
        samples: List of dicts from EventGraphDataset.__getitem__.

    Returns:
        EventBatch with stacked tensors and collected event lists.
    """
    object_embeddings = torch.stack([s["object_embeddings"] for s in samples])
    object_temporal = torch.stack([s["object_temporal"] for s in samples])
    pairwise = torch.stack([s["pairwise"] for s in samples])
    object_mask = torch.stack([s["object_mask"] for s in samples])
    gt_events = [s["gt_events"] for s in samples]
    num_objects = [s["num_objects"] for s in samples]

    return EventBatch(
        object_embeddings=object_embeddings,
        object_temporal=object_temporal,
        pairwise=pairwise,
        object_mask=object_mask,
        gt_events=gt_events,
        num_objects=num_objects,
    )
