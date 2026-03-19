"""End-to-end inference pipeline tests (CPU, mocked components)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from event_graph_generation.inference.pipeline import InferencePipeline
from event_graph_generation.inference.postprocess import (
    build_event_graph,
    predictions_to_events,
)
from event_graph_generation.models.event_decoder import EventPredictions
from event_graph_generation.schemas.event_graph import EventGraph


class TestPredictionsToEvents:
    """Test conversion of model tensor outputs to event dicts."""

    def test_predictions_to_events_basic(self):
        """Events above threshold are returned with correct fields."""
        M = 4  # event queries
        K = 3  # objects
        A = 5  # actions
        T = 8  # frames

        # Create mock predictions (batch size 1)
        preds = EventPredictions(
            interaction=torch.tensor([[[2.0], [-2.0], [1.5], [-1.0]]]),  # (1, M, 1)
            action=torch.zeros(1, M, A),
            agent_ptr=torch.zeros(1, M, K),
            target_ptr=torch.zeros(1, M, K),
            source_ptr=torch.zeros(1, M, K + 1),
            dest_ptr=torch.zeros(1, M, K + 1),
            frame=torch.zeros(1, M, T),
        )

        # Set specific action for query 0: action index 2
        preds.action[0, 0, 2] = 5.0
        # Agent slot 0, target slot 1
        preds.agent_ptr[0, 0, 0] = 5.0
        preds.target_ptr[0, 0, 1] = 5.0
        # Source: "none" (index K=3)
        preds.source_ptr[0, 0, K] = 5.0
        # Dest: slot 2
        preds.dest_ptr[0, 0, 2] = 5.0
        # Frame index 3
        preds.frame[0, 0, 3] = 5.0

        # Set action for query 2 (also above threshold)
        preds.action[0, 2, 1] = 5.0
        preds.agent_ptr[0, 2, 1] = 5.0
        preds.target_ptr[0, 2, 2] = 5.0
        preds.source_ptr[0, 2, K] = 5.0
        preds.dest_ptr[0, 2, K] = 5.0
        preds.frame[0, 2, 5] = 5.0

        track_id_map = {0: 10, 1: 20, 2: 30}
        action_names = ["pick", "place", "open", "close", "use"]
        frame_indices = [0, 5, 10, 15, 20, 25, 30, 35]

        events = predictions_to_events(
            predictions=preds,
            track_id_map=track_id_map,
            action_names=action_names,
            frame_indices=frame_indices,
            confidence_threshold=0.5,
        )

        # Queries 0 and 2 should pass threshold (sigmoid(2.0)~0.88, sigmoid(1.5)~0.82)
        # Queries 1 and 3 should not (sigmoid(-2.0)~0.12, sigmoid(-1.0)~0.27)
        assert len(events) == 2

        # Check first event (query 0)
        evt0 = events[0]
        assert evt0["action"] == "open"  # action index 2
        assert evt0["agent_track_id"] == 10
        assert evt0["target_track_id"] == 20
        assert evt0["source_track_id"] is None  # "none" slot
        assert evt0["destination_track_id"] == 30
        assert evt0["frame"] == 15  # frame_indices[3]
        assert evt0["confidence"] > 0.5

        # Check second event (query 2)
        evt1 = events[1]
        assert evt1["action"] == "place"  # action index 1
        assert evt1["agent_track_id"] == 20
        assert evt1["target_track_id"] == 30
        assert evt1["frame"] == 25  # frame_indices[5]

    def test_predictions_to_events_all_below_threshold(self):
        """No events returned when all interaction scores are below threshold."""
        preds = EventPredictions(
            interaction=torch.full((1, 3, 1), -5.0),
            action=torch.zeros(1, 3, 4),
            agent_ptr=torch.zeros(1, 3, 2),
            target_ptr=torch.zeros(1, 3, 2),
            source_ptr=torch.zeros(1, 3, 3),
            dest_ptr=torch.zeros(1, 3, 3),
            frame=torch.zeros(1, 3, 8),
        )

        events = predictions_to_events(
            predictions=preds,
            track_id_map={0: 1, 1: 2},
            action_names=["a", "b", "c", "d"],
            frame_indices=list(range(8)),
            confidence_threshold=0.5,
        )
        assert len(events) == 0


class TestBuildEventGraph:
    """Test EventGraph construction from tracked objects and events."""

    def test_build_event_graph(self):
        tracked_objects = [
            {"track_id": 1, "category": "person", "first_seen_frame": 0, "last_seen_frame": 30, "confidence": 0.95},
            {"track_id": 2, "category": "wrench", "first_seen_frame": 5, "last_seen_frame": 25, "confidence": 0.80},
        ]
        events = [
            {
                "action": "pick",
                "agent_track_id": 1,
                "target_track_id": 2,
                "source_track_id": None,
                "destination_track_id": None,
                "frame": 10,
                "confidence": 0.9,
            },
        ]

        graph = build_event_graph(
            video_id="test_video",
            tracked_objects=tracked_objects,
            events=events,
            metadata={"note": "test"},
        )

        assert isinstance(graph, EventGraph)
        assert graph.video_id == "test_video"
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.nodes[0].track_id == 1
        assert graph.nodes[0].category == "person"
        assert graph.edges[0].action == "pick"
        assert graph.edges[0].event_id == "evt_0000"
        assert graph.metadata["note"] == "test"

    def test_build_event_graph_empty(self):
        graph = build_event_graph(
            video_id="empty",
            tracked_objects=[],
            events=[],
        )
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_event_graph_serialization(self):
        """Verify the graph can be serialized to JSON."""
        tracked_objects = [
            {"track_id": 1, "category": "person", "first_seen_frame": 0, "last_seen_frame": 10},
        ]
        events = [
            {
                "action": "open",
                "agent_track_id": 1,
                "target_track_id": 1,
                "frame": 5,
                "confidence": 0.8,
            },
        ]
        graph = build_event_graph("vid", tracked_objects, events)
        json_str = graph.to_json()
        assert "open" in json_str
        assert "person" in json_str


class TestDeduplicateEvents:
    """Test deduplication of overlapping events."""

    def _make_pipeline(self) -> InferencePipeline:
        """Create a minimal pipeline for testing deduplication."""
        return InferencePipeline(
            sam3_tracker=MagicMock(),
            feature_extractor=MagicMock(),
            event_decoder=MagicMock(),
            frame_sampler=MagicMock(),
            config={"device": "cpu"},
        )

    def test_deduplicate_keeps_higher_confidence(self):
        pipeline = self._make_pipeline()
        raw_events = [
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 10, "confidence": 0.7},
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 12, "confidence": 0.9},
        ]
        result = pipeline._deduplicate_events(raw_events, frame_threshold=3)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_deduplicate_different_actions_kept(self):
        pipeline = self._make_pipeline()
        raw_events = [
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 10, "confidence": 0.8},
            {"action": "place", "agent_track_id": 1, "target_track_id": 2, "frame": 10, "confidence": 0.8},
        ]
        result = pipeline._deduplicate_events(raw_events, frame_threshold=3)
        assert len(result) == 2

    def test_deduplicate_far_apart_frames_kept(self):
        pipeline = self._make_pipeline()
        raw_events = [
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 10, "confidence": 0.8},
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 50, "confidence": 0.7},
        ]
        result = pipeline._deduplicate_events(raw_events, frame_threshold=3)
        assert len(result) == 2

    def test_deduplicate_empty(self):
        pipeline = self._make_pipeline()
        assert pipeline._deduplicate_events([], frame_threshold=3) == []

    def test_deduplicate_multiple_clusters(self):
        pipeline = self._make_pipeline()
        raw_events = [
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 10, "confidence": 0.6},
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 11, "confidence": 0.9},
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 12, "confidence": 0.7},
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 50, "confidence": 0.8},
            {"action": "pick", "agent_track_id": 1, "target_track_id": 2, "frame": 51, "confidence": 0.6},
        ]
        result = pipeline._deduplicate_events(raw_events, frame_threshold=3)
        # Cluster 1: frames 10,11,12 -> keep conf=0.9
        # Cluster 2: frames 50,51 -> keep conf=0.8
        assert len(result) == 2
        confs = sorted([e["confidence"] for e in result], reverse=True)
        assert confs == [0.9, 0.8]


class TestPipelineIntegration:
    """Integration test with mocked components."""

    def test_pipeline_produces_event_graph(self):
        """Mock all components and verify pipeline produces an EventGraph."""
        import numpy as np

        from event_graph_generation.data.frame_sampler import SampledFrame
        from event_graph_generation.tracking.feature_extractor import (
            ObjectFeatures,
            PairwiseFeatures,
        )
        from event_graph_generation.tracking.sam3_tracker import (
            FrameTrackingResult,
            TrackedObject,
        )

        T = 8
        D_emb = 256

        # Mock frame sampler
        mock_frame_sampler = MagicMock()
        mock_frame_sampler.sample.return_value = [
            SampledFrame(
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                frame_index=i * 5,
                timestamp_sec=i * 0.5,
            )
            for i in range(T)
        ]

        # Mock SAM3 tracker
        mock_tracker = MagicMock()
        tracking_results = []
        for i in range(T):
            tracking_results.append(
                FrameTrackingResult(
                    frame_index=i * 5,
                    objects=[
                        TrackedObject(
                            track_id=0,
                            category="person",
                            mask=np.zeros((480, 640), dtype=bool),
                            bbox=np.array([10, 20, 100, 200], dtype=np.float32),
                            score=0.95,
                            embedding=torch.randn(D_emb),
                        ),
                        TrackedObject(
                            track_id=1,
                            category="wrench",
                            mask=np.zeros((480, 640), dtype=bool),
                            bbox=np.array([150, 100, 200, 150], dtype=np.float32),
                            score=0.85,
                            embedding=torch.randn(D_emb),
                        ),
                    ],
                )
            )
        mock_tracker.track_video.return_value = tracking_results

        # Mock feature extractor
        mock_feature_extractor = MagicMock()
        obj_features = {}
        for tid in [0, 1]:
            obj_features[tid] = ObjectFeatures(
                track_id=tid,
                category_id=tid,
                embedding=torch.randn(D_emb),
                bbox_seq=torch.randn(T, 4),
                centroid_seq=torch.randn(T, 2),
                area_seq=torch.randn(T, 1),
                presence_seq=torch.ones(T, 1),
                delta_centroid_seq=torch.randn(T - 1, 2),
                delta_area_seq=torch.randn(T - 1, 1),
                velocity_seq=torch.randn(T - 1, 1),
            )

        pair_features = [
            PairwiseFeatures(
                track_id_i=0,
                track_id_j=1,
                iou_seq=torch.randn(T, 1),
                distance_seq=torch.randn(T, 1),
                containment_ij_seq=torch.randn(T, 1),
                containment_ji_seq=torch.randn(T, 1),
                relative_position_seq=torch.randn(T, 2),
            ),
        ]
        mock_feature_extractor.extract.return_value = (obj_features, pair_features)

        # Real EventDecoder on CPU
        from event_graph_generation.models.event_decoder import EventDecoder

        event_decoder = EventDecoder(
            d_model=64,
            nhead=4,
            num_object_encoder_layers=1,
            num_context_encoder_layers=1,
            num_event_decoder_layers=1,
            num_event_queries=4,
            max_objects=5,
            dropout=0.0,
            d_geo=12,
            d_pair=7,
            num_actions=3,
            embedding_dim=D_emb,
            temporal_window=T,
        )
        event_decoder.eval()

        config = {
            "device": "cpu",
            "clip_length": T,
            "clip_stride": T,
            "max_objects": 5,
            "d_geo": 12,
            "d_pair": 7,
            "action_names": ["pick", "place", "open"],
            "dedup_frame_threshold": 3,
        }

        pipeline = InferencePipeline(
            sam3_tracker=mock_tracker,
            feature_extractor=mock_feature_extractor,
            event_decoder=event_decoder,
            frame_sampler=mock_frame_sampler,
            config=config,
        )

        graph = pipeline.process_video(
            video_path="fake_video.mp4",
            concept_prompts=["person", "wrench"],
            confidence_threshold=0.3,
        )

        assert isinstance(graph, EventGraph)
        assert len(graph.nodes) == 2
        assert graph.nodes[0].category in ("person", "wrench")
        # Events may or may not be present depending on random weights
        assert isinstance(graph.edges, list)
        assert graph.video_id == "fake_video.mp4"
        assert graph.metadata["num_frames_sampled"] == T

        # Verify JSON serialization works
        json_str = graph.to_json()
        assert "fake_video.mp4" in json_str
