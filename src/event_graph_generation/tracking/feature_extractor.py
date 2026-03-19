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
        image_size: tuple[int, int] = (480, 640),
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

        for tid in sorted(track_ids):
            bbox_seq = torch.zeros(T, 4)
            centroid_seq = torch.zeros(T, 2)
            area_seq = torch.zeros(T, 1)
            presence_seq = torch.zeros(T, 1)

            for t_idx, obj in track_data[tid]:
                cxcywh = self._bbox_xyxy_to_cxcywh(
                    obj.bbox, self.image_h, self.image_w
                )
                bbox_seq[t_idx] = torch.tensor(cxcywh, dtype=torch.float32)
                centroid_seq[t_idx] = torch.tensor(
                    cxcywh[:2], dtype=torch.float32
                )
                area_seq[t_idx, 0] = cxcywh[2] * cxcywh[3]
                presence_seq[t_idx, 0] = 1.0

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

        # Build pairwise features
        pairwise_features: list[PairwiseFeatures] = []
        sorted_ids = sorted(track_ids)

        for tid_i, tid_j in combinations(sorted_ids, 2):
            iou_seq = torch.zeros(T, 1)
            distance_seq = torch.zeros(T, 1)
            containment_ij_seq = torch.zeros(T, 1)
            containment_ji_seq = torch.zeros(T, 1)
            relative_position_seq = torch.zeros(T, 2)

            # Build frame-level lookup for each track
            frames_i = {t_idx: obj for t_idx, obj in track_data[tid_i]}
            frames_j = {t_idx: obj for t_idx, obj in track_data[tid_j]}

            for t in range(T):
                if t in frames_i and t in frames_j:
                    bbox_i = frames_i[t].bbox
                    bbox_j = frames_j[t].bbox

                    iou_seq[t, 0] = self._compute_iou(bbox_i, bbox_j)
                    containment_ij_seq[t, 0] = self._compute_containment(
                        bbox_i, bbox_j
                    )
                    containment_ji_seq[t, 0] = self._compute_containment(
                        bbox_j, bbox_i
                    )

                    # Centroid distance (normalized)
                    ci = self._bbox_xyxy_to_cxcywh(
                        bbox_i, self.image_h, self.image_w
                    )
                    cj = self._bbox_xyxy_to_cxcywh(
                        bbox_j, self.image_h, self.image_w
                    )
                    diff = torch.tensor(ci[:2]) - torch.tensor(cj[:2])
                    distance_seq[t, 0] = torch.norm(diff).item()
                    relative_position_seq[t, 0] = diff[0].item()
                    relative_position_seq[t, 1] = diff[1].item()

            pairwise_features.append(
                PairwiseFeatures(
                    track_id_i=tid_i,
                    track_id_j=tid_j,
                    iou_seq=iou_seq,
                    distance_seq=distance_seq,
                    containment_ij_seq=containment_ij_seq,
                    containment_ji_seq=containment_ji_seq,
                    relative_position_seq=relative_position_seq,
                )
            )

        logger.info(
            "Extracted features: %d objects, %d pairs",
            len(object_features),
            len(pairwise_features),
        )
        return object_features, pairwise_features

    @staticmethod
    def _compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        """Compute Intersection over Union between two bboxes.

        Args:
            bbox_a: Bounding box [x1, y1, x2, y2].
            bbox_b: Bounding box [x1, y1, x2, y2].

        Returns:
            IoU value in [0, 1].
        """
        x1 = max(float(bbox_a[0]), float(bbox_b[0]))
        y1 = max(float(bbox_a[1]), float(bbox_b[1]))
        x2 = min(float(bbox_a[2]), float(bbox_b[2]))
        y2 = min(float(bbox_a[3]), float(bbox_b[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = (float(bbox_a[2]) - float(bbox_a[0])) * (
            float(bbox_a[3]) - float(bbox_a[1])
        )
        area_b = (float(bbox_b[2]) - float(bbox_b[0])) * (
            float(bbox_b[3]) - float(bbox_b[1])
        )

        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def _compute_containment(
        bbox_inner: np.ndarray, bbox_outer: np.ndarray
    ) -> float:
        """Compute containment ratio of bbox_inner within bbox_outer.

        Containment = intersection_area / area_inner.

        Args:
            bbox_inner: Inner bounding box [x1, y1, x2, y2].
            bbox_outer: Outer bounding box [x1, y1, x2, y2].

        Returns:
            Containment ratio in [0, 1].
        """
        x1 = max(float(bbox_inner[0]), float(bbox_outer[0]))
        y1 = max(float(bbox_inner[1]), float(bbox_outer[1]))
        x2 = min(float(bbox_inner[2]), float(bbox_outer[2]))
        y2 = min(float(bbox_inner[3]), float(bbox_outer[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area_inner = (float(bbox_inner[2]) - float(bbox_inner[0])) * (
            float(bbox_inner[3]) - float(bbox_inner[1])
        )
        if area_inner <= 0:
            return 0.0

        return inter_area / area_inner

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
