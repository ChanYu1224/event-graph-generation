"""Tests for FeatureExtractor (no GPU needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from event_graph_generation.tracking.feature_extractor import (
    FeatureExtractor,
    ObjectFeatures,
    PairwiseFeatures,
)
from event_graph_generation.tracking.sam3_tracker import (
    FrameTrackingResult,
    TrackedObject,
)


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor(
        temporal_window=8,
        normalize_coords=True,
        image_size=(480, 640),
    )


def _make_tracked_object(
    track_id: int = 0,
    category: str = "person",
    bbox: list[float] | None = None,
    mask_shape: tuple[int, int] = (480, 640),
) -> TrackedObject:
    """Create a synthetic TrackedObject for testing."""
    if bbox is None:
        bbox = [100.0, 100.0, 200.0, 200.0]
    return TrackedObject(
        track_id=track_id,
        category=category,
        mask=np.zeros(mask_shape, dtype=bool),
        bbox=np.array(bbox, dtype=np.float32),
        score=0.9,
        embedding=torch.randn(256),
    )


class TestBboxConversion:
    def test_bbox_xyxy_to_cxcywh(self, extractor: FeatureExtractor) -> None:
        """Verify normalization of xyxy to cx/cy/w/h."""
        bbox = np.array([0.0, 0.0, 640.0, 480.0])
        cx, cy, w, h = extractor._bbox_xyxy_to_cxcywh(bbox, 480, 640)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(1.0)
        assert h == pytest.approx(1.0)

    def test_bbox_xyxy_to_cxcywh_partial(self, extractor: FeatureExtractor) -> None:
        """Test normalization for a bbox in the top-left quadrant."""
        bbox = np.array([0.0, 0.0, 320.0, 240.0])
        cx, cy, w, h = extractor._bbox_xyxy_to_cxcywh(bbox, 480, 640)
        assert cx == pytest.approx(0.25)
        assert cy == pytest.approx(0.25)
        assert w == pytest.approx(0.5)
        assert h == pytest.approx(0.5)


class TestIoU:
    def test_compute_iou(self, extractor: FeatureExtractor) -> None:
        """Test IoU with known overlapping boxes."""
        bbox_a = np.array([0, 0, 10, 10], dtype=np.float32)
        bbox_b = np.array([5, 5, 15, 15], dtype=np.float32)
        # Intersection: [5,5,10,10] = 5*5 = 25
        # Union: 100 + 100 - 25 = 175
        iou = extractor._compute_iou(bbox_a, bbox_b)
        assert iou == pytest.approx(25.0 / 175.0, abs=1e-6)

    def test_compute_iou_perfect_overlap(self, extractor: FeatureExtractor) -> None:
        """Same box should give IoU = 1."""
        bbox = np.array([10, 20, 50, 60], dtype=np.float32)
        iou = extractor._compute_iou(bbox, bbox)
        assert iou == pytest.approx(1.0, abs=1e-6)

    def test_compute_iou_no_overlap(self, extractor: FeatureExtractor) -> None:
        """Non-overlapping boxes should give IoU = 0."""
        bbox_a = np.array([0, 0, 10, 10], dtype=np.float32)
        bbox_b = np.array([20, 20, 30, 30], dtype=np.float32)
        iou = extractor._compute_iou(bbox_a, bbox_b)
        assert iou == pytest.approx(0.0, abs=1e-6)


class TestContainment:
    def test_full_containment(self, extractor: FeatureExtractor) -> None:
        """Inner box fully inside outer should give containment = 1."""
        inner = np.array([10, 10, 20, 20], dtype=np.float32)
        outer = np.array([0, 0, 30, 30], dtype=np.float32)
        c = extractor._compute_containment(inner, outer)
        assert c == pytest.approx(1.0, abs=1e-6)

    def test_no_containment(self, extractor: FeatureExtractor) -> None:
        """Non-overlapping boxes should give containment = 0."""
        inner = np.array([0, 0, 10, 10], dtype=np.float32)
        outer = np.array([20, 20, 30, 30], dtype=np.float32)
        c = extractor._compute_containment(inner, outer)
        assert c == pytest.approx(0.0, abs=1e-6)


class TestExtractWithSyntheticData:
    def test_extract_with_synthetic_data(self, extractor: FeatureExtractor) -> None:
        """Create synthetic tracking results and verify output shapes."""
        T = extractor.temporal_window  # 8

        # Create 2 objects tracked over 4 frames
        tracking_results: list[FrameTrackingResult] = []
        for t in range(4):
            objects = [
                _make_tracked_object(
                    track_id=0,
                    category="person",
                    bbox=[10.0 + t, 20.0, 100.0 + t, 120.0],
                ),
                _make_tracked_object(
                    track_id=1,
                    category="car",
                    bbox=[200.0, 200.0, 400.0, 400.0],
                ),
            ]
            tracking_results.append(
                FrameTrackingResult(frame_index=t, objects=objects)
            )

        obj_feats, pair_feats = extractor.extract(tracking_results)

        # Check object features
        assert len(obj_feats) == 2
        assert 0 in obj_feats
        assert 1 in obj_feats

        for tid in [0, 1]:
            of = obj_feats[tid]
            assert isinstance(of, ObjectFeatures)
            assert of.embedding.shape == (256,)
            assert of.bbox_seq.shape == (T, 4)
            assert of.centroid_seq.shape == (T, 2)
            assert of.area_seq.shape == (T, 1)
            assert of.presence_seq.shape == (T, 1)
            assert of.delta_centroid_seq.shape == (T - 1, 2)
            assert of.delta_area_seq.shape == (T - 1, 1)
            assert of.velocity_seq.shape == (T - 1, 1)

        # Check presence: first 4 frames should be present
        assert obj_feats[0].presence_seq[:4, 0].sum().item() == 4.0
        assert obj_feats[0].presence_seq[4:, 0].sum().item() == 0.0

    def test_extract_single_object_no_pairs(
        self, extractor: FeatureExtractor
    ) -> None:
        """Single object should produce no pairwise features."""
        tracking_results = [
            FrameTrackingResult(
                frame_index=0,
                objects=[_make_tracked_object(track_id=0)],
            )
        ]
        obj_feats, pair_feats = extractor.extract(tracking_results)
        assert len(obj_feats) == 1
        assert len(pair_feats) == 0


class TestPairwiseFeaturesShape:
    def test_pairwise_features_shape(self, extractor: FeatureExtractor) -> None:
        """Verify pairwise feature tensor dimensions."""
        T = extractor.temporal_window

        # 3 objects -> 3 pairs
        tracking_results = []
        for t in range(3):
            objects = [
                _make_tracked_object(track_id=0, bbox=[0, 0, 50, 50]),
                _make_tracked_object(track_id=1, bbox=[30, 30, 80, 80]),
                _make_tracked_object(track_id=2, bbox=[200, 200, 300, 300]),
            ]
            tracking_results.append(
                FrameTrackingResult(frame_index=t, objects=objects)
            )

        _, pair_feats = extractor.extract(tracking_results)

        # C(3,2) = 3 pairs
        assert len(pair_feats) == 3

        for pf in pair_feats:
            assert isinstance(pf, PairwiseFeatures)
            assert pf.iou_seq.shape == (T, 1)
            assert pf.distance_seq.shape == (T, 1)
            assert pf.containment_ij_seq.shape == (T, 1)
            assert pf.containment_ji_seq.shape == (T, 1)
            assert pf.relative_position_seq.shape == (T, 2)
