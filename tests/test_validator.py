"""Tests for AnnotationValidator."""

from __future__ import annotations

import pytest

from event_graph_generation.annotation.validator import AnnotationValidator
from event_graph_generation.schemas.vlm_output import (
    VLMAnnotation,
    VLMEvent,
    VLMObject,
)

ALLOWED_ACTIONS = [
    "take_out",
    "put_in",
    "place_on",
    "pick_up",
    "hand_over",
    "open",
    "close",
    "use",
    "move",
    "attach",
    "detach",
    "inspect",
    "no_event",
]

ALLOWED_CATEGORIES = ["person", "cup", "table", "shelf", "box", "drawer"]

ACTION_CONFIG = [
    {"name": "take_out", "requires_source": True, "requires_destination": False},
    {"name": "put_in", "requires_source": False, "requires_destination": True},
    {"name": "place_on", "requires_source": False, "requires_destination": True},
    {"name": "pick_up", "requires_source": True, "requires_destination": False},
    {"name": "hand_over", "requires_source": True, "requires_destination": True},
    {"name": "open", "requires_source": False, "requires_destination": False},
    {"name": "close", "requires_source": False, "requires_destination": False},
    {"name": "use", "requires_source": False, "requires_destination": False},
    {"name": "move", "requires_source": True, "requires_destination": True},
    {"name": "attach", "requires_source": False, "requires_destination": True},
    {"name": "detach", "requires_source": True, "requires_destination": False},
    {"name": "inspect", "requires_source": False, "requires_destination": False},
    {"name": "no_event", "requires_source": False, "requires_destination": False},
]


@pytest.fixture
def validator() -> AnnotationValidator:
    return AnnotationValidator(
        allowed_actions=ALLOWED_ACTIONS,
        allowed_categories=ALLOWED_CATEGORIES,
        action_config=ACTION_CONFIG,
    )


def _make_valid_annotation() -> VLMAnnotation:
    """Create a fully valid annotation for testing."""
    return VLMAnnotation(
        objects=[
            VLMObject(
                obj_id="person_01",
                category="person",
                first_seen_frame=0,
                attributes=["standing"],
            ),
            VLMObject(
                obj_id="cup_01",
                category="cup",
                first_seen_frame=3,
                attributes=["red"],
            ),
            VLMObject(
                obj_id="table_01",
                category="table",
                first_seen_frame=0,
                attributes=["wooden"],
            ),
        ],
        events=[
            VLMEvent(
                event_id="evt_001",
                frame=5,
                action="pick_up",
                agent="person_01",
                target="cup_01",
                source="table_01",
                destination=None,
            ),
            VLMEvent(
                event_id="evt_002",
                frame=10,
                action="inspect",
                agent="person_01",
                target="cup_01",
            ),
        ],
    )


class TestAnnotationValidator:
    def test_valid_annotation(self, validator: AnnotationValidator) -> None:
        """Fully valid annotation passes without event removal."""
        ann = _make_valid_annotation()
        corrected, warnings = validator.validate(ann)

        assert len(corrected.events) == 2
        assert len(corrected.objects) == 3
        # Only possible warning is source/dest logic, but our example is correct
        # No events should be discarded
        assert corrected.events[0].event_id == "evt_001"
        assert corrected.events[1].event_id == "evt_002"

    def test_invalid_action(self, validator: AnnotationValidator) -> None:
        """Event with action not in allowed list gets removed."""
        ann = _make_valid_annotation()
        ann.events.append(
            VLMEvent(
                event_id="evt_003",
                frame=15,
                action="fly",  # not in allowed actions
                agent="person_01",
                target="cup_01",
            )
        )

        corrected, warnings = validator.validate(ann)

        assert len(corrected.events) == 2  # evt_003 removed
        event_ids = {e.event_id for e in corrected.events}
        assert "evt_003" not in event_ids
        assert any("unknown action" in w for w in warnings)

    def test_invalid_reference(self, validator: AnnotationValidator) -> None:
        """Event referencing non-existent obj_id gets removed."""
        ann = _make_valid_annotation()
        ann.events.append(
            VLMEvent(
                event_id="evt_003",
                frame=15,
                action="inspect",
                agent="ghost_01",  # does not exist
                target="cup_01",
            )
        )

        corrected, warnings = validator.validate(ann)

        assert len(corrected.events) == 2  # evt_003 removed
        event_ids = {e.event_id for e in corrected.events}
        assert "evt_003" not in event_ids
        assert any("not in objects" in w for w in warnings)

    def test_temporal_ordering(self, validator: AnnotationValidator) -> None:
        """Out-of-order events get sorted by frame."""
        ann = VLMAnnotation(
            objects=[
                VLMObject(
                    obj_id="person_01",
                    category="person",
                    first_seen_frame=0,
                    attributes=[],
                ),
                VLMObject(
                    obj_id="cup_01",
                    category="cup",
                    first_seen_frame=0,
                    attributes=[],
                ),
            ],
            events=[
                VLMEvent(
                    event_id="evt_002",
                    frame=10,
                    action="inspect",
                    agent="person_01",
                    target="cup_01",
                ),
                VLMEvent(
                    event_id="evt_001",
                    frame=5,
                    action="inspect",
                    agent="person_01",
                    target="cup_01",
                ),
            ],
        )

        corrected, warnings = validator.validate(ann)

        assert len(corrected.events) == 2
        assert corrected.events[0].frame == 5
        assert corrected.events[1].frame == 10
        assert any("re-sorted" in w for w in warnings)

    def test_source_destination_logic(self, validator: AnnotationValidator) -> None:
        """Events with missing required source/dest get flagged."""
        ann = VLMAnnotation(
            objects=[
                VLMObject(
                    obj_id="person_01",
                    category="person",
                    first_seen_frame=0,
                    attributes=[],
                ),
                VLMObject(
                    obj_id="cup_01",
                    category="cup",
                    first_seen_frame=0,
                    attributes=[],
                ),
            ],
            events=[
                VLMEvent(
                    event_id="evt_001",
                    frame=5,
                    action="pick_up",  # requires_source=True
                    agent="person_01",
                    target="cup_01",
                    source=None,  # missing required source
                    destination=None,
                ),
                VLMEvent(
                    event_id="evt_002",
                    frame=10,
                    action="put_in",  # requires_destination=True
                    agent="person_01",
                    target="cup_01",
                    source=None,
                    destination=None,  # missing required destination
                ),
            ],
        )

        corrected, warnings = validator.validate(ann)

        # Events are kept but warnings are generated for missing source/dest
        assert len(corrected.events) == 2
        assert any("requires source" in w for w in warnings)
        assert any("requires destination" in w for w in warnings)

    def test_empty_annotation(self, validator: AnnotationValidator) -> None:
        """Empty annotation passes without errors."""
        ann = VLMAnnotation()

        corrected, warnings = validator.validate(ann)

        assert len(corrected.objects) == 0
        assert len(corrected.events) == 0
        assert len(warnings) == 0

    def test_validate_batch(self, validator: AnnotationValidator) -> None:
        """Batch validation returns correct aggregate stats."""
        good_ann = _make_valid_annotation()

        # Annotation with one bad event
        bad_ann = VLMAnnotation(
            objects=[
                VLMObject(
                    obj_id="person_01",
                    category="person",
                    first_seen_frame=0,
                    attributes=[],
                ),
            ],
            events=[
                VLMEvent(
                    event_id="evt_001",
                    frame=5,
                    action="inspect",
                    agent="person_01",
                    target="person_01",
                ),
                VLMEvent(
                    event_id="evt_002",
                    frame=10,
                    action="fly",  # invalid action -> discard
                    agent="person_01",
                    target="person_01",
                ),
            ],
        )

        validated, stats = validator.validate_batch([good_ann, bad_ann])

        assert len(validated) == 2
        assert stats["total_annotations"] == 2
        assert stats["total_events_input"] == 4  # 2 + 2
        assert stats["total_events_output"] == 3  # 2 + 1
        assert stats["total_events_discarded"] == 1
        assert stats["discard_rate"] == 0.25
        assert stats["total_warnings"] > 0
