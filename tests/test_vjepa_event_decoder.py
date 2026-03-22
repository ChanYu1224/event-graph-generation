"""Tests for VJEPAEventDecoder."""

from __future__ import annotations

import torch

from event_graph_generation.models.event_decoder import EventPredictions, VJEPAEventDecoder
from event_graph_generation.models.object_pooling import ObjectRepresentation


# Small dims for CPU tests
D_MODEL = 32
K = 6
T_ENC = 4
M = 8
A = 5
T_WIN = 8
N_CAT = 10
B = 2


def _make_obj_repr(batch_size: int = B, existence_val: float = 0.9) -> ObjectRepresentation:
    """Create a synthetic ObjectRepresentation."""
    return ObjectRepresentation(
        identity=torch.randn(batch_size, K, D_MODEL),
        trajectory=torch.randn(batch_size, K, T_ENC, D_MODEL),
        existence=torch.full((batch_size, K), existence_val),
        categories=torch.randn(batch_size, K, N_CAT),
        attn_maps=torch.randn(batch_size, K, T_ENC * 16).softmax(dim=-1),
    )


def _make_decoder(**kwargs) -> VJEPAEventDecoder:
    defaults = dict(
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
    defaults.update(kwargs)
    return VJEPAEventDecoder(**defaults)


class TestVJEPAEventDecoder:
    def test_forward_shape(self):
        model = _make_decoder()
        model.eval()
        obj_repr = _make_obj_repr()
        with torch.no_grad():
            preds = model(obj_repr)
        assert isinstance(preds, EventPredictions)
        assert preds.interaction.shape == (B, M, 1)
        assert preds.action.shape == (B, M, A)
        assert preds.agent_ptr.shape == (B, M, K)
        assert preds.target_ptr.shape == (B, M, K)
        assert preds.source_ptr.shape == (B, M, K + 1)
        assert preds.dest_ptr.shape == (B, M, K + 1)
        assert preds.frame.shape == (B, M, T_WIN)

    def test_gradient_flow(self):
        model = _make_decoder()
        model.train()
        obj_repr = _make_obj_repr(batch_size=1)
        preds = model(obj_repr)
        loss = (
            preds.interaction.sum()
            + preds.action.sum()
            + preds.agent_ptr.sum()
            + preds.target_ptr.sum()
            + preds.source_ptr.sum()
            + preds.dest_ptr.sum()
            + preds.frame.sum()
        )
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"

    def test_masked_objects(self):
        """When object_mask is provided, masked slots get -inf in pointer logits."""
        model = _make_decoder()
        model.eval()
        # First 3 slots exist, rest don't
        existence = torch.zeros(1, K)
        existence[0, :3] = 0.9
        object_mask = torch.zeros(1, K, dtype=torch.bool)
        object_mask[0, :3] = True
        obj_repr = ObjectRepresentation(
            identity=torch.randn(1, K, D_MODEL),
            trajectory=torch.randn(1, K, T_ENC, D_MODEL),
            existence=existence,
            categories=torch.randn(1, K, N_CAT),
            attn_maps=torch.randn(1, K, T_ENC * 16).softmax(dim=-1),
        )
        with torch.no_grad():
            preds = model(obj_repr, object_mask=object_mask)

        # Masked slots (3-5) should be -inf for agent/target
        assert (preds.agent_ptr[0, :, 3:K] == float("-inf")).all()
        assert (preds.target_ptr[0, :, 3:K] == float("-inf")).all()
        # "none" slot (K) in source/dest should be finite
        assert torch.isfinite(preds.source_ptr[0, :, K]).all()
        assert torch.isfinite(preds.dest_ptr[0, :, K]).all()

    def test_no_mask_no_inf(self):
        """Without object_mask, no pointer logits should be -inf."""
        model = _make_decoder()
        model.eval()
        obj_repr = _make_obj_repr(batch_size=1)
        with torch.no_grad():
            preds = model(obj_repr)
        assert torch.isfinite(preds.agent_ptr).all()
        assert torch.isfinite(preds.target_ptr).all()
