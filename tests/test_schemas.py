"""Tests for event graph schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from event_graph_generation.schemas.event_graph import EventEdge, EventGraph, ObjectNode
from event_graph_generation.schemas.vlm_output import VLMAnnotation, VLMEvent, VLMObject


class TestObjectNode:
    def test_create(self):
        node = ObjectNode(
            track_id=1,
            category="wrench",
            first_seen_frame=0,
            last_seen_frame=10,
            confidence=0.95,
        )
        assert node.track_id == 1
        assert node.category == "wrench"
        assert node.attributes == {}

    def test_with_attributes(self):
        node = ObjectNode(
            track_id=1,
            category="person",
            first_seen_frame=0,
            last_seen_frame=100,
            attributes={"color": "blue"},
        )
        assert node.attributes["color"] == "blue"


class TestEventEdge:
    def test_create_minimal(self):
        edge = EventEdge(
            event_id="evt_001",
            agent_track_id=1,
            action="take_out",
            target_track_id=2,
        )
        assert edge.source_track_id is None
        assert edge.destination_track_id is None
        assert edge.confidence == 0.0

    def test_create_full(self):
        edge = EventEdge(
            event_id="evt_002",
            agent_track_id=1,
            action="move",
            target_track_id=2,
            source_track_id=3,
            destination_track_id=4,
            frame=10,
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            confidence=0.9,
        )
        assert edge.source_track_id == 3
        assert edge.destination_track_id == 4


class TestEventGraph:
    def _make_graph(self):
        nodes = [
            ObjectNode(1, "person", 0, 20, 0.99),
            ObjectNode(2, "wrench", 5, 20, 0.95),
            ObjectNode(3, "drawer", 0, 20, 0.98),
        ]
        edges = [
            EventEdge("evt_001", 1, "open", 3, frame=3),
            EventEdge("evt_002", 1, "take_out", 2, source_track_id=3, frame=5),
            EventEdge("evt_003", 1, "place_on", 2, destination_track_id=4, frame=12),
        ]
        return EventGraph(video_id="test_video", nodes=nodes, edges=edges)

    def test_to_dict(self):
        g = self._make_graph()
        d = g.to_dict()
        assert d["video_id"] == "test_video"
        assert len(d["nodes"]) == 3
        assert len(d["edges"]) == 3

    def test_to_json(self):
        g = self._make_graph()
        text = g.to_json()
        assert '"video_id": "test_video"' in text

    def test_get_object_timeline(self):
        g = self._make_graph()
        timeline = g.get_object_timeline(2)
        assert len(timeline) == 2
        assert timeline[0].frame < timeline[1].frame

    def test_get_events_in_range(self):
        g = self._make_graph()
        events = g.get_events_in_range(0, 5)
        assert len(events) == 2  # frame 3 and frame 5

    def test_empty_graph(self):
        g = EventGraph(video_id="empty")
        assert g.to_dict()["nodes"] == []
        assert g.get_object_timeline(1) == []


class TestVLMObject:
    def test_valid(self):
        obj = VLMObject(
            obj_id="person_01",
            category="person",
            first_seen_frame=0,
            attributes=["blue_gloves"],
        )
        assert obj.obj_id == "person_01"

    def test_invalid_obj_id(self):
        with pytest.raises(ValidationError):
            VLMObject(
                obj_id="invalid-id",
                category="person",
                first_seen_frame=0,
            )

    def test_negative_frame(self):
        with pytest.raises(ValidationError):
            VLMObject(
                obj_id="person_01",
                category="person",
                first_seen_frame=-1,
            )


class TestVLMEvent:
    def test_valid(self):
        event = VLMEvent(
            event_id="evt_001",
            frame=5,
            action="take_out",
            agent="person_01",
            target="wrench_01",
            source="drawer_01",
        )
        assert event.action == "take_out"
        assert event.destination is None

    def test_invalid_event_id(self):
        with pytest.raises(ValidationError):
            VLMEvent(
                event_id="bad_id",
                frame=5,
                action="take_out",
                agent="person_01",
                target="wrench_01",
            )


class TestVLMAnnotation:
    def test_valid(self):
        annotation = VLMAnnotation(
            objects=[
                VLMObject(obj_id="person_01", category="person", first_seen_frame=0),
                VLMObject(obj_id="wrench_01", category="wrench", first_seen_frame=5),
            ],
            events=[
                VLMEvent(
                    event_id="evt_001",
                    frame=5,
                    action="take_out",
                    agent="person_01",
                    target="wrench_01",
                    source="drawer_01",
                ),
            ],
        )
        assert len(annotation.objects) == 2
        assert len(annotation.events) == 1

    def test_empty(self):
        annotation = VLMAnnotation()
        assert annotation.objects == []
        assert annotation.events == []

    def test_from_dict(self):
        data = {
            "objects": [
                {"obj_id": "person_01", "category": "person", "first_seen_frame": 0}
            ],
            "events": [
                {
                    "event_id": "evt_001",
                    "frame": 3,
                    "action": "open",
                    "agent": "person_01",
                    "target": "drawer_01",
                }
            ],
        }
        annotation = VLMAnnotation.model_validate(data)
        assert annotation.objects[0].category == "person"
