"""Feature extraction from SAM 3 tracking results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from event_graph_generation.tracking.sam3_tracker import FrameTrackingResult

logger = logging.getLogger(__name__)


@dataclass
class ObjectFeatures:
    """Per-object features aggregated over time."""

    track_id: int
    category_id: int
    embedding: torch.Tensor  # (D_emb,)
    bbox_seq: torch.Tensor  # (T, 4) normalized cx/cy/w/h
    centroid_seq: torch.Tensor  # (T, 2)
    area_seq: torch.Tensor  # (T, 1)
    presence_seq: torch.Tensor  # (T, 1)
    delta_centroid_seq: torch.Tensor  # (T-1, 2)
    delta_area_seq: torch.Tensor  # (T-1, 1)
    velocity_seq: torch.Tensor  # (T-1, 1)


@dataclass
class PairwiseFeatures:
    """Pairwise features between two tracked objects over time."""

    track_id_i: int
    track_id_j: int
    iou_seq: torch.Tensor  # (T, 1)
    distance_seq: torch.Tensor  # (T, 1)
    containment_ij_seq: torch.Tensor  # (T, 1)
    containment_ji_seq: torch.Tensor  # (T, 1)
    relative_position_seq: torch.Tensor  # (T, 2)


class FeatureExtractor:
    """Extract temporal object and pairwise features from tracking results.

    Converts raw tracking outputs (bboxes, masks, embeddings) into fixed-length
    temporal feature sequences suitable for downstream graph neural networks.
    """

    def __init__(
        self,
        temporal_window: int = 16,
        normalize_coords: bool = True,
        image_size: tuple[int, int] = (566, 1008),
    ) -> None:
        """Initialize the feature extractor.

        Args:
            temporal_window: Number of time steps T for feature sequences.
            normalize_coords: Whether to normalize coordinates by image size.
            image_size: (height, width) of the input frames.
        """
        self.temporal_window = temporal_window
        self.normalize_coords = normalize_coords
        self.image_h, self.image_w = image_size

    def extract(
        self,
        tracking_results: list[FrameTrackingResult],
    ) -> tuple[dict[int, ObjectFeatures], list[PairwiseFeatures]]:
        """Extract object and pairwise features from tracking results.

        Args:
            tracking_results: List of per-frame tracking results.

        Returns:
            A tuple of (object_features_dict, pairwise_features_list).
            object_features_dict maps track_id -> ObjectFeatures.
            pairwise_features_list contains PairwiseFeatures for all pairs.
        """
        T = self.temporal_window

        # Collect all track IDs and their per-frame data
        track_ids: set[int] = set()
        # track_id -> category string
        track_categories: dict[int, str] = {}
        # track_id -> list of (time_index, TrackedObject)
        track_data: dict[int, list[tuple[int, object]]] = {}
        # track_id -> list of embeddings
        track_embeddings: dict[int, list[torch.Tensor]] = {}

        for t_idx, frame_result in enumerate(tracking_results):
            if t_idx >= T:
                break
            for obj in frame_result.objects:
                tid = obj.track_id
                track_ids.add(tid)
                track_categories.setdefault(tid, obj.category)
                track_data.setdefault(tid, []).append((t_idx, obj))
                track_embeddings.setdefault(tid, []).append(obj.embedding)

        # Assign category IDs based on sorted unique categories
        all_categories = sorted(set(track_categories.values()))
        cat_to_id = {cat: idx for idx, cat in enumerate(all_categories)}

        # Build per-object features
        object_features: dict[int, ObjectFeatures] = {}
        bbox_xyxy_dict: dict[int, torch.Tensor] = {}

        for tid in sorted(track_ids):
            bbox_seq = torch.zeros(T, 4)
            bbox_xyxy_seq = torch.zeros(T, 4)
            centroid_seq = torch.zeros(T, 2)
            area_seq = torch.zeros(T, 1)
            presence_seq = torch.zeros(T, 1)

            for t_idx, obj in track_data[tid]:
                bbox_xyxy_seq[t_idx] = torch.tensor(obj.bbox, dtype=torch.float32)
                cxcywh = self._bbox_xyxy_to_cxcywh(
                    obj.bbox, self.image_h, self.image_w
                )
                bbox_seq[t_idx] = torch.tensor(cxcywh, dtype=torch.float32)
                centroid_seq[t_idx] = torch.tensor(
                    cxcywh[:2], dtype=torch.float32
                )
                area_seq[t_idx, 0] = cxcywh[2] * cxcywh[3]
                presence_seq[t_idx, 0] = 1.0

            bbox_xyxy_dict[tid] = bbox_xyxy_seq

            # Temporal deltas — only valid when both current and previous frame
            # have the object present; otherwise zero to avoid misleading jumps.
            delta_centroid_seq = centroid_seq[1:] - centroid_seq[:-1]  # (T-1, 2)
            delta_area_seq = area_seq[1:] - area_seq[:-1]  # (T-1, 1)
            # Mask: both frames must have presence=1 for a valid delta
            both_present = (presence_seq[:-1, 0] * presence_seq[1:, 0]).unsqueeze(1)  # (T-1, 1)
            delta_centroid_seq = delta_centroid_seq * both_present  # zero out invalid
            delta_area_seq = delta_area_seq * both_present
            velocity_seq = torch.norm(delta_centroid_seq, dim=1, keepdim=True)  # (T-1, 1)

            # Average embedding across frames
            embs = track_embeddings[tid]
            embedding = torch.stack(embs).mean(dim=0)  # (D_emb,)

            object_features[tid] = ObjectFeatures(
                track_id=tid,
                category_id=cat_to_id[track_categories[tid]],
                embedding=embedding,
                bbox_seq=bbox_seq,
                centroid_seq=centroid_seq,
                area_seq=area_seq,
                presence_seq=presence_seq,
                delta_centroid_seq=delta_centroid_seq,
                delta_area_seq=delta_area_seq,
                velocity_seq=velocity_seq,
            )

        # Build pairwise features (vectorized)
        pairwise_features: list[PairwiseFeatures] = []
        sorted_ids = sorted(track_ids)

        if len(sorted_ids) >= 2:
            all_bboxes_xyxy = torch.stack(
                [bbox_xyxy_dict[tid] for tid in sorted_ids]
            )  # (K, T, 4)
            all_centroids = torch.stack(
                [object_features[tid].centroid_seq for tid in sorted_ids]
            )  # (K, T, 2)
            all_presence = torch.stack(
                [object_features[tid].presence_seq for tid in sorted_ids]
            )  # (K, T, 1)

            # Mask: both objects must be present for valid pairwise features
            presence_i = all_presence.unsqueeze(1)  # (K, 1, T, 1)
            presence_j = all_presence.unsqueeze(0)  # (1, K, T, 1)
            both_present = (presence_i * presence_j).squeeze(-1)  # (K, K, T)

            iou_matrix = self._compute_iou_matrix(all_bboxes_xyxy) * both_present
            containment_matrix = (
                self._compute_containment_matrix(all_bboxes_xyxy) * both_present
            )

            diff = (
                all_centroids.unsqueeze(1) - all_centroids.unsqueeze(0)
            )  # (K, K, T, 2)
            distance_matrix = diff.norm(dim=-1) * both_present  # (K, K, T)
            # Zero out relative position where not both present
            rel_pos = diff * both_present.unsqueeze(-1)  # (K, K, T, 2)

            for idx_i, idx_j in combinations(range(len(sorted_ids)), 2):
                pairwise_features.append(
                    PairwiseFeatures(
                        track_id_i=sorted_ids[idx_i],
                        track_id_j=sorted_ids[idx_j],
                        iou_seq=iou_matrix[idx_i, idx_j].unsqueeze(-1),
                        distance_seq=distance_matrix[idx_i, idx_j].unsqueeze(-1),
                        containment_ij_seq=containment_matrix[
                            idx_i, idx_j
                        ].unsqueeze(-1),
                        containment_ji_seq=containment_matrix[
                            idx_j, idx_i
                        ].unsqueeze(-1),
                        relative_position_seq=rel_pos[idx_i, idx_j],
                    )
                )

        logger.info(
            "Extracted features: %d objects, %d pairs",
            len(object_features),
            len(pairwise_features),
        )
        return object_features, pairwise_features

    @staticmethod
    def _compute_iou_matrix(bboxes: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix from batched bounding boxes.

        Args:
            bboxes: (K, T, 4) tensor in xyxy format.

        Returns:
            (K, K, T) IoU matrix.
        """
        b1 = bboxes.unsqueeze(1)  # (K, 1, T, 4)
        b2 = bboxes.unsqueeze(0)  # (1, K, T, 4)
        inter_x1 = torch.max(b1[..., 0], b2[..., 0])
        inter_y1 = torch.max(b1[..., 1], b2[..., 1])
        inter_x2 = torch.min(b1[..., 2], b2[..., 2])
        inter_y2 = torch.min(b1[..., 3], b2[..., 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (
            inter_y2 - inter_y1
        ).clamp(min=0)
        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union = area1 + area2 - inter
        return inter / union.clamp(min=1e-6)

    @staticmethod
    def _compute_containment_matrix(bboxes: torch.Tensor) -> torch.Tensor:
        """Compute pairwise containment matrix from batched bounding boxes.

        containment[i, j, t] = intersection(i, j) / area(i) at frame t.

        Args:
            bboxes: (K, T, 4) tensor in xyxy format.

        Returns:
            (K, K, T) containment matrix.
        """
        b1 = bboxes.unsqueeze(1)  # (K, 1, T, 4)
        b2 = bboxes.unsqueeze(0)  # (1, K, T, 4)
        inter_x1 = torch.max(b1[..., 0], b2[..., 0])
        inter_y1 = torch.max(b1[..., 1], b2[..., 1])
        inter_x2 = torch.min(b1[..., 2], b2[..., 2])
        inter_y2 = torch.min(b1[..., 3], b2[..., 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (
            inter_y2 - inter_y1
        ).clamp(min=0)
        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        return inter / area1.clamp(min=1e-6)

    @staticmethod
    def _bbox_xyxy_to_cxcywh(
        bbox: np.ndarray, image_h: int, image_w: int
    ) -> tuple[float, float, float, float]:
        """Convert [x1, y1, x2, y2] bbox to normalized (cx, cy, w, h).

        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format.
            image_h: Image height for normalization.
            image_w: Image width for normalization.

        Returns:
            Tuple of (cx, cy, w, h) normalized by image dimensions.
        """
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = (x1 + x2) / 2.0 / image_w
        cy = (y1 + y2) / 2.0 / image_h
        w = (x2 - x1) / image_w
        h = (y2 - y1) / image_h
        return cx, cy, w, h
