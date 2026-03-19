"""Alignment between VLM annotations and SAM 3 tracking results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from event_graph_generation.schemas.vlm_output import VLMAnnotation, VLMObject
from event_graph_generation.tracking.sam3_tracker import FrameTrackingResult, TrackedObject

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of aligning VLM objects to SAM 3 tracks."""

    mapping: dict[str, int]  # vlm_obj_id -> sam3_track_id
    unmatched_vlm: list[str] = field(default_factory=list)
    unmatched_sam3: list[int] = field(default_factory=list)
    confidence: float = 0.0


class Aligner:
    """Align VLM-annotated objects to SAM 3 tracked objects using IoU matching."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold

    def align(
        self,
        tracking_results: list[FrameTrackingResult],
        vlm_annotation: VLMAnnotation,
    ) -> AlignmentResult:
        """Match VLM objects to SAM 3 tracks.

        For each VLM object:
        1. Filter SAM 3 tracks by matching category name.
        2. At the VLM object's first_seen_frame, compute bbox IoU between candidates.
        3. Use Hungarian matching for optimal assignment.
        4. Apply iou_threshold filter.

        Args:
            tracking_results: Per-frame tracking results from SAM 3.
            vlm_annotation: VLM annotation with objects and events.

        Returns:
            AlignmentResult with mapping, unmatched lists, and confidence.
        """
        # Build frame index -> FrameTrackingResult lookup
        frame_lookup: dict[int, FrameTrackingResult] = {
            r.frame_index: r for r in tracking_results
        }

        # Collect all unique SAM 3 track IDs and their per-frame data
        all_sam3_tracks: dict[int, dict[int, TrackedObject]] = {}
        sam3_track_categories: dict[int, str] = {}
        for fr in tracking_results:
            for obj in fr.objects:
                all_sam3_tracks.setdefault(obj.track_id, {})[fr.frame_index] = obj
                sam3_track_categories.setdefault(obj.track_id, obj.category)

        vlm_objects = vlm_annotation.objects
        if not vlm_objects:
            return AlignmentResult(
                mapping={},
                unmatched_vlm=[],
                unmatched_sam3=list(all_sam3_tracks.keys()),
                confidence=1.0,
            )

        sam3_ids = sorted(all_sam3_tracks.keys())
        if not sam3_ids:
            return AlignmentResult(
                mapping={},
                unmatched_vlm=[o.obj_id for o in vlm_objects],
                unmatched_sam3=[],
                confidence=0.0,
            )

        # Build cost matrix: (num_vlm_objects, num_sam3_tracks)
        num_vlm = len(vlm_objects)
        num_sam3 = len(sam3_ids)
        cost_matrix = np.ones((num_vlm, num_sam3), dtype=np.float64)

        for i, vlm_obj in enumerate(vlm_objects):
            frame_idx = vlm_obj.first_seen_frame
            vlm_category = vlm_obj.category.lower()

            for j, sam3_id in enumerate(sam3_ids):
                sam3_category = sam3_track_categories[sam3_id].lower()

                # Category must match
                if vlm_category != sam3_category:
                    cost_matrix[i, j] = 1.0
                    continue

                # Check if this SAM 3 track is present at the target frame
                if frame_idx not in all_sam3_tracks[sam3_id]:
                    cost_matrix[i, j] = 1.0
                    continue

                sam3_obj = all_sam3_tracks[sam3_id][frame_idx]

                # Compute IoU using bboxes
                # VLM objects don't have bboxes directly, so we use SAM3 bbox
                # and match by category + frame presence.
                # If there is a frame result, find VLM bbox proxy from all SAM3
                # objects of same category in that frame.
                iou = self._compute_bbox_iou_for_vlm(
                    vlm_obj, sam3_obj, frame_lookup.get(frame_idx)
                )
                cost_matrix[i, j] = 1.0 - iou

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        mapping: dict[str, int] = {}
        matched_vlm_indices: set[int] = set()
        matched_sam3_indices: set[int] = set()
        iou_scores: list[float] = []

        for r, c in zip(row_ind, col_ind):
            iou = 1.0 - cost_matrix[r, c]
            if iou >= self.iou_threshold:
                vlm_obj_id = vlm_objects[r].obj_id
                sam3_track_id = sam3_ids[c]
                mapping[vlm_obj_id] = sam3_track_id
                matched_vlm_indices.add(r)
                matched_sam3_indices.add(c)
                iou_scores.append(iou)

        unmatched_vlm = [
            vlm_objects[i].obj_id
            for i in range(num_vlm)
            if i not in matched_vlm_indices
        ]
        unmatched_sam3 = [
            sam3_ids[j]
            for j in range(num_sam3)
            if j not in matched_sam3_indices
        ]

        confidence = float(np.mean(iou_scores)) if iou_scores else 0.0

        logger.info(
            "Alignment: %d matched, %d unmatched VLM, %d unmatched SAM3, confidence=%.3f",
            len(mapping),
            len(unmatched_vlm),
            len(unmatched_sam3),
            confidence,
        )

        return AlignmentResult(
            mapping=mapping,
            unmatched_vlm=unmatched_vlm,
            unmatched_sam3=unmatched_sam3,
            confidence=confidence,
        )

    @staticmethod
    def _compute_bbox_iou_for_vlm(
        vlm_obj: VLMObject,
        sam3_obj: TrackedObject,
        frame_result: FrameTrackingResult | None,
    ) -> float:
        """Compute a match score between a VLM object and a SAM 3 tracked object.

        Since VLM objects don't have explicit bboxes, we use a heuristic:
        - If the candidate is the only same-category object, return 1.0
        - If multiple same-category objects exist, use spatial separation:
          compute the candidate's IoU against all other same-category objects.
          A well-separated (low IoU with others) candidate is preferred, as it
          is easier to unambiguously match. The score is 1.0 - max_iou_with_others,
          weighted by detection confidence.

        Args:
            vlm_obj: The VLM object being matched.
            sam3_obj: The candidate SAM 3 tracked object.
            frame_result: Full tracking result for this frame.

        Returns:
            Match score in [0, 1]. Higher means better match.
        """
        if frame_result is None:
            return 0.0

        # Find all SAM 3 objects with same category in this frame
        same_cat_objs = [
            o for o in frame_result.objects
            if o.category.lower() == vlm_obj.category.lower()
        ]

        if not same_cat_objs:
            return 0.0

        # If only one object of this category, it's a clear match
        if len(same_cat_objs) == 1 and same_cat_objs[0].track_id == sam3_obj.track_id:
            return 1.0

        bbox = sam3_obj.bbox
        if bbox is None:
            return 0.0

        # Multiple same-category objects: compute IoU of candidate against
        # all other same-category objects to measure spatial separation.
        max_iou_with_others = 0.0
        for other in same_cat_objs:
            if other.track_id == sam3_obj.track_id:
                continue
            if other.bbox is None:
                continue
            iou = Aligner._bbox_iou(bbox, other.bbox)
            max_iou_with_others = max(max_iou_with_others, iou)

        # Well-separated objects score higher; weight by detection confidence
        separation_score = 1.0 - max_iou_with_others
        confidence_weight = max(float(sam3_obj.score), 0.1)
        return separation_score * confidence_weight

    @staticmethod
    def _bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
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
