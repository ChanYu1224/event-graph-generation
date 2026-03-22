"""Tests for the Object Pooling Module and its subcomponents."""

from __future__ import annotations

import torch

from event_graph_generation.models.object_pooling import (
    InputProjection,
    ObjectPoolingModule,
    ObjectRepresentation,
    SlotRefinement,
    SpatiotemporalSlotAttention,
    TemporalTrajectoryExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use small dimensions for CPU-friendly tests
D_INPUT = 64
D_MODEL = 32
K = 6
T_ENC = 4
SP = 16
S = T_ENC * SP  # total tokens
N_CAT = 10
B = 2


def _make_tokens(batch_size: int = B) -> torch.Tensor:
    return torch.randn(batch_size, S, D_INPUT)


# ---------------------------------------------------------------------------
# InputProjection tests
# ---------------------------------------------------------------------------

class TestInputProjection:
    def test_output_shape(self):
        proj = InputProjection(D_INPUT, D_MODEL, T_ENC, SP)
        x = _make_tokens()
        out = proj(x)
        assert out.shape == (B, S, D_MODEL)

    def test_positional_encoding_added(self):
        proj = InputProjection(D_INPUT, D_MODEL, T_ENC, SP)
        x = torch.zeros(1, S, D_INPUT)
        out = proj(x)
        # Output should not be all zeros (pos encoding contributes)
        assert not torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# SpatiotemporalSlotAttention tests
# ---------------------------------------------------------------------------

class TestSpatiotemporalSlotAttention:
    def test_output_shapes(self):
        sa = SpatiotemporalSlotAttention(D_MODEL, K, num_iterations=2)
        tokens = torch.randn(B, S, D_MODEL)
        slots, attn = sa(tokens)
        assert slots.shape == (B, K, D_MODEL)
        assert attn.shape == (B, K, S)

    def test_attention_sums_to_one_over_slots(self):
        """Slot-axis softmax: each token's attn sums to 1 across slots."""
        sa = SpatiotemporalSlotAttention(D_MODEL, K, num_iterations=2)
        tokens = torch.randn(B, S, D_MODEL)
        _, attn = sa(tokens)
        sums = attn.sum(dim=1)  # (B, S)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_different_initializations(self):
        """Two forward passes in train mode produce different slot initializations."""
        sa = SpatiotemporalSlotAttention(D_MODEL, K, num_iterations=1)
        sa.train()
        tokens = torch.randn(1, S, D_MODEL)
        slots1, _ = sa(tokens)
        slots2, _ = sa(tokens)
        # Due to stochastic initialization, outputs should differ
        assert not torch.allclose(slots1, slots2)

    def test_deterministic_eval(self):
        """Eval mode should produce deterministic outputs."""
        sa = SpatiotemporalSlotAttention(D_MODEL, K, num_iterations=2)
        sa.eval()
        tokens = torch.randn(1, S, D_MODEL)
        with torch.no_grad():
            slots1, _ = sa(tokens)
            slots2, _ = sa(tokens)
        assert torch.allclose(slots1, slots2)


# ---------------------------------------------------------------------------
# SlotRefinement tests
# ---------------------------------------------------------------------------

class TestSlotRefinement:
    def test_output_shape(self):
        refine = SlotRefinement(D_MODEL, nhead=4, num_layers=1, dropout=0.0)
        slots = torch.randn(B, K, D_MODEL)
        out = refine(slots)
        assert out.shape == (B, K, D_MODEL)


# ---------------------------------------------------------------------------
# TemporalTrajectoryExtractor tests
# ---------------------------------------------------------------------------

class TestTemporalTrajectoryExtractor:
    def test_output_shapes(self):
        ext = TemporalTrajectoryExtractor(D_MODEL, T_ENC, SP)
        slots = torch.randn(B, K, D_MODEL)
        attn = torch.randn(B, K, S).softmax(dim=-1)
        tokens = torch.randn(B, S, D_MODEL)
        identity, trajectory, existence = ext(slots, attn, tokens)
        assert identity.shape == (B, K, D_MODEL)
        assert trajectory.shape == (B, K, T_ENC, D_MODEL)
        assert existence.shape == (B, K)

    def test_existence_in_zero_one(self):
        ext = TemporalTrajectoryExtractor(D_MODEL, T_ENC, SP)
        slots = torch.randn(B, K, D_MODEL)
        attn = torch.randn(B, K, S).softmax(dim=-1)
        tokens = torch.randn(B, S, D_MODEL)
        _, _, existence = ext(slots, attn, tokens)
        assert (existence >= 0).all() and (existence <= 1).all()


# ---------------------------------------------------------------------------
# ObjectPoolingModule (full pipeline) tests
# ---------------------------------------------------------------------------

class TestObjectPoolingModule:
    def _make_module(self) -> ObjectPoolingModule:
        return ObjectPoolingModule(
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

    def test_output_types_and_shapes(self):
        module = self._make_module()
        module.eval()
        x = _make_tokens()
        with torch.no_grad():
            out = module(x)
        assert isinstance(out, ObjectRepresentation)
        assert out.identity.shape == (B, K, D_MODEL)
        assert out.trajectory.shape == (B, K, T_ENC, D_MODEL)
        assert out.existence.shape == (B, K)
        assert out.categories.shape == (B, K, N_CAT)
        assert out.attn_maps.shape == (B, K, S)

    def test_gradient_flow(self):
        module = self._make_module()
        module.train()
        x = _make_tokens(batch_size=1)
        out = module(x)
        loss = (
            out.identity.sum()
            + out.trajectory.sum()
            + out.existence.sum()
            + out.categories.sum()
        )
        loss.backward()
        for name, param in module.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_existence_probabilities(self):
        module = self._make_module()
        module.eval()
        x = _make_tokens()
        with torch.no_grad():
            out = module(x)
        assert (out.existence >= 0).all()
        assert (out.existence <= 1).all()
