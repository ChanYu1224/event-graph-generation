"""Tests for EventGraphLoss."""

from __future__ import annotations

import torch

from event_graph_generation.models.event_decoder import EventDecoder, EventPredictions
from event_graph_generation.models.losses import EventGraphLoss


def _make_predictions(B: int, M: int, K: int, A: int, T: int) -> EventPredictions:
    """Create random predictions for testing."""
    return EventPredictions(
        interaction=torch.randn(B, M, 1),
        action=torch.randn(B, M, A),
        agent_ptr=torch.randn(B, M, K),
        target_ptr=torch.randn(B, M, K),
        source_ptr=torch.randn(B, M, K + 1),
        dest_ptr=torch.randn(B, M, K + 1),
        frame=torch.randn(B, M, T),
    )


def test_hungarian_matching():
    """Verify Hungarian matching finds optimal assignment with known optimal."""
    B, M, K, A, T = 1, 4, 5, 3, 8

    loss_fn = EventGraphLoss(num_actions=A)

    # Create predictions where slot 0 strongly predicts action 0, agent 0, target 1, frame 2
    # and slot 1 strongly predicts action 1, agent 2, target 3, frame 4
    preds = EventPredictions(
        interaction=torch.zeros(B, M, 1),
        action=torch.full((B, M, A), -10.0),
        agent_ptr=torch.full((B, M, K), -10.0),
        target_ptr=torch.full((B, M, K), -10.0),
        source_ptr=torch.zeros(B, M, K + 1),
        dest_ptr=torch.zeros(B, M, K + 1),
        frame=torch.full((B, M, T), -10.0),
    )

    # Slot 0: action=0, agent=0, target=1, frame=2
    preds.action[0, 0, 0] = 10.0
    preds.agent_ptr[0, 0, 0] = 10.0
    preds.target_ptr[0, 0, 1] = 10.0
    preds.frame[0, 0, 2] = 10.0

    # Slot 1: action=1, agent=2, target=3, frame=4
    preds.action[0, 1, 1] = 10.0
    preds.agent_ptr[0, 1, 2] = 10.0
    preds.target_ptr[0, 1, 3] = 10.0
    preds.frame[0, 1, 4] = 10.0

    gt_events = [
        [
            {
                "agent_track_id": 0,
                "action_class": 0,
                "target_track_id": 1,
                "source_track_id": None,
                "dest_track_id": None,
                "event_frame_index": 2,
            },
            {
                "agent_track_id": 2,
                "action_class": 1,
                "target_track_id": 3,
                "source_track_id": None,
                "dest_track_id": None,
                "event_frame_index": 4,
            },
        ]
    ]

    object_mask = torch.ones(B, K, dtype=torch.bool)

    # Verify matching: slot 0 -> GT 0, slot 1 -> GT 1
    matched_pred, matched_gt = loss_fn._hungarian_match(preds, 0, gt_events[0], object_mask[0])

    # The optimal matching should pair slot 0 with GT 0 and slot 1 with GT 1
    match_pairs = dict(zip(matched_pred, matched_gt))
    assert match_pairs[0] == 0, f"Expected slot 0 -> GT 0, got slot 0 -> GT {match_pairs.get(0)}"
    assert match_pairs[1] == 1, f"Expected slot 1 -> GT 1, got slot 1 -> GT {match_pairs.get(1)}"


def test_loss_computation():
    """Verify loss is non-negative and finite."""
    B, M, K, A, T = 2, 10, 8, 5, 16

    loss_fn = EventGraphLoss(num_actions=A)
    preds = _make_predictions(B, M, K, A, T)

    gt_events = [
        [
            {
                "agent_track_id": 0,
                "action_class": 1,
                "target_track_id": 2,
                "source_track_id": 3,
                "dest_track_id": 4,
                "event_frame_index": 5,
            },
            {
                "agent_track_id": 1,
                "action_class": 0,
                "target_track_id": 3,
                "source_track_id": None,
                "dest_track_id": None,
                "event_frame_index": 8,
            },
        ],
        [
            {
                "agent_track_id": 0,
                "action_class": 2,
                "target_track_id": 1,
                "source_track_id": None,
                "dest_track_id": 2,
                "event_frame_index": 3,
            },
        ],
    ]

    object_mask = torch.ones(B, K, dtype=torch.bool)

    total_loss, loss_dict = loss_fn(preds, gt_events, object_mask)

    assert torch.isfinite(total_loss), "Total loss should be finite"
    assert total_loss.item() >= 0, "Total loss should be non-negative"

    for key in ["interaction", "action", "agent_ptr", "target_ptr", "source_ptr", "dest_ptr", "frame"]:
        assert key in loss_dict, f"Missing loss key: {key}"
        assert loss_dict[key] >= 0, f"Loss '{key}' should be non-negative"


def test_no_events():
    """Batch with no GT events should still compute interaction loss."""
    B, M, K, A, T = 1, 10, 5, 3, 8

    loss_fn = EventGraphLoss(num_actions=A)
    preds = _make_predictions(B, M, K, A, T)

    gt_events: list[list[dict]] = [[]]  # No events for the single batch item
    object_mask = torch.ones(B, K, dtype=torch.bool)

    total_loss, loss_dict = loss_fn(preds, gt_events, object_mask)

    assert torch.isfinite(total_loss), "Total loss should be finite even with no events"
    assert total_loss.item() >= 0, "Total loss should be non-negative"
    # Interaction loss should be non-zero (predicting all zeros)
    assert loss_dict["interaction"] >= 0
    # Other losses should be zero since there are no matched events
    assert loss_dict["action"] == 0.0
    assert loss_dict["agent_ptr"] == 0.0
    assert loss_dict["target_ptr"] == 0.0
