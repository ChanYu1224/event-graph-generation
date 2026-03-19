"""Annotation validator for VLM output quality assurance."""

from __future__ import annotations

import logging

from event_graph_generation.schemas.vlm_output import VLMAnnotation, VLMEvent

logger = logging.getLogger(__name__)


class AnnotationValidator:
    """Validate and optionally auto-fix VLM annotations."""

    def __init__(
        self,
        allowed_actions: list[str],
        allowed_categories: list[str],
        action_config: list[dict] | None = None,
    ) -> None:
        """Initialize validator.

        Args:
            allowed_actions: List of valid action names.
            allowed_categories: List of valid object category names.
            action_config: List of action config dicts from actions.yaml.
                Each dict should have 'name', 'requires_source', 'requires_destination'.
        """
        self.allowed_actions = set(allowed_actions)
        self.allowed_categories = set(allowed_categories)

        # Build action requirements lookup
        self.action_requirements: dict[str, dict[str, bool]] = {}
        if action_config:
            for entry in action_config:
                name = entry.get("name", "")
                self.action_requirements[name] = {
                    "requires_source": entry.get("requires_source", False),
                    "requires_destination": entry.get("requires_destination", False),
                }

    def validate(
        self, annotation: VLMAnnotation
    ) -> tuple[VLMAnnotation, list[str]]:
        """Validate and auto-fix an annotation.

        Checks performed:
        1. Schema validation (already done by Pydantic on construction)
        2. Vocabulary: action/category must be in allowed lists
        3. Reference integrity: agent/target/source/destination reference valid obj_ids
        4. Temporal ordering: events sorted by frame (auto-fix)
        5. Logic: source/destination presence matches action requirements

        For minor issues (ordering): auto-fix in place.
        For major issues (bad references, invalid vocabulary): remove the event and log.

        Args:
            annotation: The VLMAnnotation to validate.

        Returns:
            Tuple of (possibly corrected VLMAnnotation, list of warning messages).
        """
        annotation = annotation.model_copy(deep=True)
        warnings: list[str] = []

        # Collect valid object IDs
        valid_obj_ids: set[str] = set()
        valid_objects = []
        for obj in annotation.objects:
            # Check category vocabulary
            if obj.category not in self.allowed_categories:
                msg = f"Object {obj.obj_id}: unknown category '{obj.category}'"
                warnings.append(msg)
                logger.warning(msg)
                # Keep the object but warn (it's still referenceable)
            valid_obj_ids.add(obj.obj_id)
            valid_objects.append(obj)

        annotation.objects = valid_objects

        # Validate events
        valid_events: list[VLMEvent] = []
        for event in annotation.events:
            event_warnings: list[str] = []
            discard = False

            # Check action vocabulary
            if event.action not in self.allowed_actions:
                msg = f"Event {event.event_id}: unknown action '{event.action}'"
                event_warnings.append(msg)
                discard = True

            # Check reference integrity
            if event.agent not in valid_obj_ids:
                msg = f"Event {event.event_id}: agent '{event.agent}' not in objects"
                event_warnings.append(msg)
                discard = True

            if event.target not in valid_obj_ids:
                msg = f"Event {event.event_id}: target '{event.target}' not in objects"
                event_warnings.append(msg)
                discard = True

            if event.source is not None and event.source not in valid_obj_ids:
                msg = f"Event {event.event_id}: source '{event.source}' not in objects"
                event_warnings.append(msg)
                discard = True

            if event.destination is not None and event.destination not in valid_obj_ids:
                msg = (
                    f"Event {event.event_id}: destination '{event.destination}' "
                    f"not in objects"
                )
                event_warnings.append(msg)
                discard = True

            # Check source/destination logic against action requirements
            if event.action in self.action_requirements and not discard:
                reqs = self.action_requirements[event.action]
                if reqs["requires_source"] and event.source is None:
                    msg = (
                        f"Event {event.event_id}: action '{event.action}' "
                        f"requires source but none provided"
                    )
                    event_warnings.append(msg)
                if reqs["requires_destination"] and event.destination is None:
                    msg = (
                        f"Event {event.event_id}: action '{event.action}' "
                        f"requires destination but none provided"
                    )
                    event_warnings.append(msg)

            warnings.extend(event_warnings)
            if discard:
                logger.warning(
                    "Discarding event %s: %s",
                    event.event_id,
                    "; ".join(event_warnings),
                )
            else:
                valid_events.append(event)

        # Auto-fix: sort events by frame (temporal ordering)
        sorted_events = sorted(valid_events, key=lambda e: e.frame)
        if valid_events != sorted_events:
            warnings.append("Events were re-sorted by frame for temporal ordering")
        annotation.events = sorted_events

        return annotation, warnings

    def validate_batch(
        self, annotations: list[VLMAnnotation]
    ) -> tuple[list[VLMAnnotation], dict]:
        """Validate a batch of annotations and return aggregate stats.

        Args:
            annotations: List of VLMAnnotation to validate.

        Returns:
            Tuple of (list of validated annotations, stats dict).
        """
        validated: list[VLMAnnotation] = []
        total_events_in = 0
        total_events_out = 0
        total_warnings = 0

        for ann in annotations:
            events_before = len(ann.events)
            corrected, warns = self.validate(ann)
            events_after = len(corrected.events)

            total_events_in += events_before
            total_events_out += events_after
            total_warnings += len(warns)

            validated.append(corrected)

        discarded = total_events_in - total_events_out
        discard_rate = discarded / total_events_in if total_events_in > 0 else 0.0

        stats = {
            "total_annotations": len(annotations),
            "total_events_input": total_events_in,
            "total_events_output": total_events_out,
            "total_events_discarded": discarded,
            "discard_rate": round(discard_rate, 4),
            "total_warnings": total_warnings,
        }

        return validated, stats
