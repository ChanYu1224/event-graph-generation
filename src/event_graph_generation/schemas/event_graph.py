"""EventGraph data structures: nodes (objects) and edges (events)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class ObjectNode:
    """Graph node: a tracked object."""

    track_id: int
    category: str
    first_seen_frame: int
    last_seen_frame: int
    confidence: float = 0.0
    attributes: dict = field(default_factory=dict)


@dataclass
class EventEdge:
    """Graph edge: an interaction event between objects."""

    event_id: str
    agent_track_id: int
    action: str
    target_track_id: int
    source_track_id: int | None = None
    destination_track_id: int | None = None
    frame: int = 0
    timestamp: datetime | None = None
    confidence: float = 0.0


@dataclass
class EventGraph:
    """Complete event graph for a video."""

    video_id: str
    nodes: list[ObjectNode] = field(default_factory=list)
    edges: list[EventEdge] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert datetime objects to ISO strings
        for edge in d["edges"]:
            if edge["timestamp"] is not None:
                edge["timestamp"] = edge["timestamp"].isoformat()
        return d

    def to_json(self, path: str | None = None) -> str:
        """Serialize to JSON string, optionally saving to file."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def get_object_timeline(self, track_id: int) -> list[EventEdge]:
        """Return events involving a specific object, sorted by frame."""
        relevant = [
            e
            for e in self.edges
            if e.agent_track_id == track_id
            or e.target_track_id == track_id
            or e.source_track_id == track_id
            or e.destination_track_id == track_id
        ]
        return sorted(relevant, key=lambda e: e.frame)

    def get_events_in_range(self, start_frame: int, end_frame: int) -> list[EventEdge]:
        """Return events within a frame range."""
        return sorted(
            [e for e in self.edges if start_frame <= e.frame <= end_frame],
            key=lambda e: e.frame,
        )
