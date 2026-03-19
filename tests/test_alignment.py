"""Tests for VLM-to-SAM3 alignment."""

from __future__ import annotations

import numpy as np
import torch

from event_graph_generation.annotation.alignment import Aligner, AlignmentResult
from event_graph_generation.schemas.vlm_output import VLMAnnotation, VLMObject
from event_graph_generation.tracking.sam3_tracker import (
    FrameTrackingResult,
    TrackedObject,
)


def _make_tracked_object(
    track_id: int,
    category: str,
    bbox: list[float],
    score: float = 0.9,
) -> TrackedObject:
    """Helper to create a TrackedObject."""
    return TrackedObject(
        track_id=track_id,
        category=category,
        mask=np.zeros((10, 10), dtype=bool),
        bbox=np.array(bbox, dtype=np.float32),
        score=score,
        embedding=torch.randn(256),
    )


def test_perfect_alignment():
    """Test alignment with clear 1:1 matches by category and frame."""
    tracking_results = [
        FrameTrackingResult(
            frame_index=0,
            objects=[
                _make_tracked_object(1, "person", [10, 10, 50, 50]),
                _make_tracked_object(2, "wrench", [100, 100, 150, 150]),
            ],
        ),
        FrameTrackingResult(
            frame_index=1,
            objects=[
                _make_tracked_object(1, "person", [12, 12, 52, 52]),
                _make_tracked_object(2, "wrench", [102, 102, 152, 152]),
            ],
        ),
    ]

    vlm_annotation = VLMAnnotation(
        objects=[
            VLMObject(obj_id="person_0", category="person", first_seen_frame=0),
            VLMObject(obj_id="wrench_0", category="wrench", first_seen_frame=0),
        ],
        events=[],
    )

    aligner = Aligner(iou_threshold=0.3)
    result = aligner.align(tracking_results, vlm_annotation)

    assert isinstance(result, AlignmentResult)
    assert len(result.mapping) == 2
    assert result.mapping["person_0"] == 1
    assert result.mapping["wrench_0"] == 2
    assert len(result.unmatched_vlm) == 0
    assert len(result.unmatched_sam3) == 0
    assert result.confidence > 0.0


def test_no_match():
    """Test alignment with completely different categories -> empty mapping."""
    tracking_results = [
        FrameTrackingResult(
            frame_index=0,
            objects=[
                _make_tracked_object(1, "car", [10, 10, 50, 50]),
            ],
        ),
    ]

    vlm_annotation = VLMAnnotation(
        objects=[
            VLMObject(obj_id="person_0", category="person", first_seen_frame=0),
        ],
        events=[],
    )

    aligner = Aligner(iou_threshold=0.3)
    result = aligner.align(tracking_results, vlm_annotation)

    assert len(result.mapping) == 0
    assert "person_0" in result.unmatched_vlm
    assert 1 in result.unmatched_sam3
    assert result.confidence == 0.0


def test_partial_match():
    """Test alignment where some objects match and some don't."""
    tracking_results = [
        FrameTrackingResult(
            frame_index=0,
            objects=[
                _make_tracked_object(1, "person", [10, 10, 50, 50]),
                _make_tracked_object(2, "drawer", [200, 200, 300, 300]),
            ],
        ),
    ]

    vlm_annotation = VLMAnnotation(
        objects=[
            VLMObject(obj_id="person_0", category="person", first_seen_frame=0),
            VLMObject(obj_id="wrench_0", category="wrench", first_seen_frame=0),
        ],
        events=[],
    )

    aligner = Aligner(iou_threshold=0.3)
    result = aligner.align(tracking_results, vlm_annotation)

    # person should match, wrench should not
    assert "person_0" in result.mapping
    assert result.mapping["person_0"] == 1
    assert "wrench_0" in result.unmatched_vlm
    assert 2 in result.unmatched_sam3
    assert result.confidence > 0.0
