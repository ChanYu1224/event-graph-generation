"""Event Graph dataset for training the Event Decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from event_graph_generation.data.dataset import BaseDataset


class EventGraphDataset(BaseDataset):
    """Dataset for event graph generation training.

    Each sample is a .pt file containing object embeddings, temporal features,
    pairwise features, and ground-truth event annotations.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_objects: int = 30,
    ) -> None:
        self.max_objects = max_objects
        super().__init__(data_dir=data_dir, split=split)

    def _load_samples(self) -> None:
        """Read sample IDs from the split file."""
        split_file = self.data_dir / "splits" / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load a single sample and pad to max_objects.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
                - object_embeddings: (K, D_emb) tensor
                - object_temporal: (K, T, D_geo) tensor
                - pairwise: (K, K, T, D_pair) tensor
                - object_mask: (K,) bool tensor
                - gt_events: list[dict] with event annotations
                - num_objects: int
        """
        sample_id = self.samples[idx]
        sample_path = self.data_dir / "samples" / f"{sample_id}.pt"
        data = torch.load(sample_path, map_location="cpu", weights_only=True)

        # Extract raw tensors
        obj_emb = data["object_embeddings"]  # (N, D_emb)
        obj_temp = data["object_temporal"]  # (N, T, D_geo)
        pairwise = data["pairwise"]  # (N, N, T, D_pair)
        gt_events = data.get("gt_events", [])

        N = obj_emb.shape[0]
        K = self.max_objects
        D_emb = obj_emb.shape[1]
        T = obj_temp.shape[1]
        D_geo = obj_temp.shape[2]
        D_pair = pairwise.shape[3]

        num_objects = min(N, K)

        # Pad to max_objects
        padded_emb = torch.zeros(K, D_emb)
        padded_temp = torch.zeros(K, T, D_geo)
        padded_pair = torch.zeros(K, K, T, D_pair)
        mask = torch.zeros(K, dtype=torch.bool)

        padded_emb[:num_objects] = obj_emb[:num_objects]
        padded_temp[:num_objects] = obj_temp[:num_objects]
        padded_pair[:num_objects, :num_objects] = pairwise[:num_objects, :num_objects]
        mask[:num_objects] = True

        return {
            "object_embeddings": padded_emb,
            "object_temporal": padded_temp,
            "pairwise": padded_pair,
            "object_mask": mask,
            "gt_events": gt_events,
            "num_objects": num_objects,
        }
