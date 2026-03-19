"""Convert model tensor outputs to structured EventGraph."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from event_graph_generation.schemas.event_graph import EventEdge, EventGraph, ObjectNode

if TYPE_CHECKING:
    from event_graph_generation.models.event_decoder import EventPredictions

logger = logging.getLogger(__name__)


def predictions_to_events(
    predictions: EventPredictions,
    track_id_map: dict[int, int],
    action_names: list[str],
    frame_indices: list[int],
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """Convert EventPredictions tensors to list of event dicts.

    For each event query slot:
    1. Check interaction score > threshold (sigmoid of interaction logit)
    2. Get action via argmax of action logits
    3. Get agent/target via argmax of pointer logits
    4. Get source/dest via argmax (index K means "none")
    5. Get frame via argmax of frame logits, map to actual frame number
    6. Build event dict with confidence score

    Args:
        predictions: EventPredictions from the model (batch size 1).
        track_id_map: Mapping from slot_index -> track_id.
        action_names: Action vocabulary list.
        frame_indices: Frame numbers for this clip.
        confidence_threshold: Minimum interaction score to keep an event.

    Returns:
        List of event dicts ready for EventGraph construction.
    """
    # Work with batch index 0
    interaction_logits = predictions.interaction[0]  # (M, 1)
    action_logits = predictions.action[0]  # (M, A)
    agent_ptr_logits = predictions.agent_ptr[0]  # (M, K)
    target_ptr_logits = predictions.target_ptr[0]  # (M, K)
    source_ptr_logits = predictions.source_ptr[0]  # (M, K+1)
    dest_ptr_logits = predictions.dest_ptr[0]  # (M, K+1)
    frame_logits = predictions.frame[0]  # (M, T)

    interaction_scores = torch.sigmoid(interaction_logits).squeeze(-1)  # (M,)
    num_queries = interaction_scores.shape[0]
    num_objects = agent_ptr_logits.shape[1]

    events: list[dict] = []

    for q in range(num_queries):
        score = float(interaction_scores[q])
        if score < confidence_threshold:
            continue

        # Action
        action_idx = int(torch.argmax(action_logits[q]))
        action_name = action_names[action_idx] if action_idx < len(action_names) else f"action_{action_idx}"

        # Agent and target
        agent_slot = int(torch.argmax(agent_ptr_logits[q]))
        target_slot = int(torch.argmax(target_ptr_logits[q]))

        agent_track_id = track_id_map.get(agent_slot, agent_slot)
        target_track_id = track_id_map.get(target_slot, target_slot)

        # Source and dest (last index K means "none")
        source_slot = int(torch.argmax(source_ptr_logits[q]))
        dest_slot = int(torch.argmax(dest_ptr_logits[q]))

        source_track_id = track_id_map.get(source_slot) if source_slot < num_objects else None
        dest_track_id = track_id_map.get(dest_slot) if dest_slot < num_objects else None

        # Frame
        frame_slot = int(torch.argmax(frame_logits[q]))
        frame_number = frame_indices[frame_slot] if frame_slot < len(frame_indices) else frame_slot

        events.append({
            "action": action_name,
            "agent_track_id": agent_track_id,
            "target_track_id": target_track_id,
            "source_track_id": source_track_id,
            "destination_track_id": dest_track_id,
            "frame": frame_number,
            "confidence": score,
        })

    logger.debug("Converted predictions to %d events (threshold=%.2f)", len(events), confidence_threshold)
    return events


def build_event_graph(
    video_id: str,
    tracked_objects: list[dict],
    events: list[dict],
    metadata: dict | None = None,
) -> EventGraph:
    """Build EventGraph from tracked objects and predicted events.

    Args:
        video_id: Identifier for the video.
        tracked_objects: List of dicts with keys: track_id, category,
            first_seen_frame, last_seen_frame, confidence.
        events: List of event dicts from predictions_to_events.
        metadata: Optional metadata dict.

    Returns:
        Constructed EventGraph.
    """
    nodes = []
    for obj in tracked_objects:
        nodes.append(
            ObjectNode(
                track_id=obj["track_id"],
                category=obj["category"],
                first_seen_frame=obj.get("first_seen_frame", 0),
                last_seen_frame=obj.get("last_seen_frame", 0),
                confidence=obj.get("confidence", 0.0),
                attributes=obj.get("attributes", {}),
            )
        )

    edges = []
    for idx, evt in enumerate(events):
        edges.append(
            EventEdge(
                event_id=f"evt_{idx:04d}",
                agent_track_id=evt["agent_track_id"],
                action=evt["action"],
                target_track_id=evt["target_track_id"],
                source_track_id=evt.get("source_track_id"),
                destination_track_id=evt.get("destination_track_id"),
                frame=evt.get("frame", 0),
                confidence=evt.get("confidence", 0.0),
            )
        )

    graph = EventGraph(
        video_id=video_id,
        nodes=nodes,
        edges=edges,
        metadata=metadata or {},
    )

    logger.info(
        "Built EventGraph for %s: %d nodes, %d edges",
        video_id,
        len(nodes),
        len(edges),
    )
    return graph
