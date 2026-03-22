"""Tests for VJEPAPipeline end-to-end and VJEPAEventGraphLoss."""

from __future__ import annotations

import torch

from event_graph_generation.models.object_pooling import ObjectPoolingModule, ObjectRepresentation
from event_graph_generation.models.event_decoder import VJEPAEventDecoder, EventPredictions
from event_graph_generation.models.vjepa_pipeline import VJEPAPipeline
from event_graph_generation.models.losses import VJEPAEventGraphLoss
from event_graph_generation.models.base import build_model
from event_graph_generation.config import ModelConfig, ObjectPoolingConfig


# Small dims for CPU tests
D_INPUT = 64
D_MODEL = 32
K = 6
T_ENC = 4
SP = 16
S = T_ENC * SP
N_CAT = 10
M = 8
A = 5
T_WIN = 8
B = 2


def _make_pipeline() -> VJEPAPipeline:
    op = ObjectPoolingModule(
        input_dim=D_INPUT,
        d_model=D_MODEL,
        num_slots=K,
        num_iterations=2,
        num_refinement_layers=1,
        nhead=4,
        n_categories=N_CAT,
        temporal_tokens=T_ENC,
        spatial_tokens=SP,
        dropout=0.0,
    )
    ed = VJEPAEventDecoder(
        d_model=D_MODEL,
        nhead=4,
        num_context_encoder_layers=1,
        num_event_decoder_layers=1,
        num_event_queries=M,
        num_slots=K,
        dropout=0.0,
        num_actions=A,
        temporal_window=T_WIN,
    )
    return VJEPAPipeline(op, ed)


def _make_gt_events() -> list[list[dict]]:
    """Create sample GT events for 2 batch items."""
    return [
        [
            {
                "agent_track_id": 0,
                "action_class": 1,
                "target_track_id": 1,
                "source_track_id": None,
                "dest_track_id": None,
                "event_frame_index": 3,
            },
        ],
        [
            {
                "agent_track_id": 0,
                "action_class": 2,
                "target_track_id": 2,
                "source_track_id": 1,
                "dest_track_id": None,
                "event_frame_index": 5,
            },
        ],
    ]


def _make_gt_categories() -> list[list[int]]:
    """Create sample GT object categories for 2 batch items."""
    return [
        [0, 1, 2],  # 3 objects
        [0, 3, 1, 4],  # 4 objects
    ]


class TestVJEPAPipeline:
    def test_forward_output_types(self):
        pipeline = _make_pipeline()
        pipeline.eval()
        x = torch.randn(B, S, D_INPUT)
        with torch.no_grad():
            obj_repr, preds = pipeline(x)
        assert isinstance(obj_repr, ObjectRepresentation)
        assert isinstance(preds, EventPredictions)

    def test_forward_shapes(self):
        pipeline = _make_pipeline()
        pipeline.eval()
        x = torch.randn(B, S, D_INPUT)
        with torch.no_grad():
            obj_repr, preds = pipeline(x)
        assert obj_repr.identity.shape == (B, K, D_MODEL)
        assert preds.interaction.shape == (B, M, 1)
        assert preds.action.shape == (B, M, A)
        assert preds.agent_ptr.shape == (B, M, K)

    def test_end_to_end_gradient_flow(self):
        pipeline = _make_pipeline()
        pipeline.train()
        x = torch.randn(1, S, D_INPUT)
        obj_repr, preds = pipeline(x)
        # Include all output tensors to ensure all parameters get gradients
        loss = (
            preds.interaction.sum()
            + preds.action.sum()
            + preds.agent_ptr.sum()
            + preds.target_ptr.sum()
            + preds.source_ptr.sum()
            + preds.dest_ptr.sum()
            + preds.frame.sum()
            + obj_repr.categories.sum()
            + obj_repr.existence.sum()
        )
        loss.backward()
        # Gradients should flow to both object_pooling and event_decoder
        for name, param in pipeline.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestVJEPAEventGraphLoss:
    def _make_loss_fn(self) -> VJEPAEventGraphLoss:
        return VJEPAEventGraphLoss(num_actions=A)

    def test_loss_computation(self):
        pipeline = _make_pipeline()
        pipeline.train()
        loss_fn = self._make_loss_fn()

        x = torch.randn(B, S, D_INPUT)
        obj_repr, preds = pipeline(x)

        gt_events = _make_gt_events()
        gt_cats = _make_gt_categories()

        total_loss, loss_dict = loss_fn(obj_repr, preds, gt_events, gt_cats)

        assert torch.isfinite(total_loss)
        assert total_loss.item() >= 0
        assert "category" in loss_dict
        assert "existence" in loss_dict
        assert "interaction" in loss_dict
        assert "action" in loss_dict

    def test_loss_backward(self):
        pipeline = _make_pipeline()
        pipeline.train()
        loss_fn = self._make_loss_fn()

        x = torch.randn(B, S, D_INPUT)
        obj_repr, preds = pipeline(x)

        gt_events = _make_gt_events()
        gt_cats = _make_gt_categories()

        total_loss, _ = loss_fn(obj_repr, preds, gt_events, gt_cats)
        total_loss.backward()

        # At least the object pooling parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in pipeline.object_pooling.parameters()
        )
        assert has_grad, "Object pooling should receive gradients through loss"

    def test_no_events(self):
        pipeline = _make_pipeline()
        pipeline.eval()
        loss_fn = self._make_loss_fn()

        x = torch.randn(B, S, D_INPUT)
        with torch.no_grad():
            obj_repr, preds = pipeline(x)

        gt_events: list[list[dict]] = [[], []]
        gt_cats: list[list[int]] = [[], []]

        total_loss, loss_dict = loss_fn(obj_repr, preds, gt_events, gt_cats)

        assert torch.isfinite(total_loss)
        assert loss_dict["action"] == 0.0

    def test_slot_object_matching(self):
        loss_fn = self._make_loss_fn()
        # Create categories where slot 0 predicts cat 2, slot 1 predicts cat 0
        slot_cats = torch.full((K, N_CAT), -10.0)
        slot_cats[0, 2] = 10.0
        slot_cats[1, 0] = 10.0
        existence = torch.ones(K) * 0.9

        gt_cats = [2, 0]  # obj 0 = cat 2, obj 1 = cat 0
        matched_slots, matched_objs = loss_fn._slot_object_match(
            slot_cats, existence, gt_cats
        )
        match_pairs = dict(zip(matched_slots, matched_objs))
        # Slot 0 should match obj 0 (both cat 2), slot 1 match obj 1 (both cat 0)
        assert match_pairs[0] == 0
        assert match_pairs[1] == 1


class TestBuildModel:
    def test_build_vjepa_pipeline(self):
        config = ModelConfig(
            name="vjepa_pipeline",
            d_model=D_MODEL,
            nhead=4,
            num_context_encoder_layers=1,
            num_event_decoder_layers=1,
            num_event_queries=M,
            dropout=0.0,
            num_actions=A,
            object_pooling=ObjectPoolingConfig(
                num_slots=K,
                d_model=D_MODEL,
                num_iterations=2,
                num_refinement_layers=1,
                nhead=4,
                n_categories=N_CAT,
                dropout=0.0,
            ),
        )
        model = build_model(config)
        assert isinstance(model, VJEPAPipeline)
