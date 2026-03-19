"""Tests for the EventDecoder model."""

from __future__ import annotations

import torch

from event_graph_generation.models.event_decoder import EventDecoder, EventPredictions


def _make_decoder(**kwargs) -> EventDecoder:
    """Create an EventDecoder with test-friendly defaults."""
    defaults = dict(
        d_model=256,
        nhead=8,
        num_object_encoder_layers=3,
        num_context_encoder_layers=3,
        num_event_decoder_layers=4,
        num_event_queries=20,
        max_objects=30,
        dropout=0.1,
        d_geo=12,
        d_pair=7,
        num_actions=13,
        embedding_dim=256,
        temporal_window=16,
    )
    defaults.update(kwargs)
    return EventDecoder(**defaults)


def test_forward_shape():
    """Verify output shapes match expected dimensions."""
    B, K, T = 2, 10, 16
    D_geo, D_pair = 12, 7
    M, A = 20, 13
    D_emb = 256

    model = _make_decoder(max_objects=K, temporal_window=T)
    model.eval()

    object_embeddings = torch.randn(B, K, D_emb)
    object_temporal = torch.randn(B, K, T, D_geo)
    pairwise = torch.randn(B, K, K, T, D_pair)
    object_mask = torch.ones(B, K, dtype=torch.bool)

    with torch.no_grad():
        preds = model(object_embeddings, object_temporal, pairwise, object_mask)

    assert isinstance(preds, EventPredictions)
    assert preds.interaction.shape == (B, M, 1)
    assert preds.action.shape == (B, M, A)
    assert preds.agent_ptr.shape == (B, M, K)
    assert preds.target_ptr.shape == (B, M, K)
    assert preds.source_ptr.shape == (B, M, K + 1)
    assert preds.dest_ptr.shape == (B, M, K + 1)
    assert preds.frame.shape == (B, M, T)


def test_gradient_flow():
    """Verify gradients flow to all parameters."""
    B, K, T = 1, 5, 8
    D_geo, D_pair = 12, 7
    D_emb = 256

    model = _make_decoder(
        max_objects=K,
        temporal_window=T,
        num_object_encoder_layers=1,
        num_context_encoder_layers=1,
        num_event_decoder_layers=1,
        num_event_queries=4,
    )
    model.train()

    object_embeddings = torch.randn(B, K, D_emb)
    object_temporal = torch.randn(B, K, T, D_geo)
    pairwise = torch.randn(B, K, K, T, D_pair)
    object_mask = torch.ones(B, K, dtype=torch.bool)

    preds = model(object_embeddings, object_temporal, pairwise, object_mask)

    # Sum all outputs to create a scalar loss
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
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_masked_objects():
    """Verify masked object slots get -inf in pointer logits."""
    B, K, T = 1, 10, 8
    D_geo, D_pair = 12, 7
    D_emb = 256

    model = _make_decoder(
        max_objects=K,
        temporal_window=T,
        num_object_encoder_layers=1,
        num_context_encoder_layers=1,
        num_event_decoder_layers=1,
        num_event_queries=4,
    )
    model.eval()

    object_embeddings = torch.randn(B, K, D_emb)
    object_temporal = torch.randn(B, K, T, D_geo)
    pairwise = torch.randn(B, K, K, T, D_pair)

    # Only first 5 objects are valid
    object_mask = torch.zeros(B, K, dtype=torch.bool)
    object_mask[:, :5] = True

    with torch.no_grad():
        preds = model(object_embeddings, object_temporal, pairwise, object_mask)

    # Masked positions (indices 5-9) should be -inf in agent and target pointer logits
    assert (preds.agent_ptr[0, :, 5:] == float("-inf")).all(), (
        "Masked objects should have -inf in agent_ptr"
    )
    assert (preds.target_ptr[0, :, 5:] == float("-inf")).all(), (
        "Masked objects should have -inf in target_ptr"
    )

    # source/dest: masked positions (5-9) should be -inf, but last slot (none) should not
    assert (preds.source_ptr[0, :, 5:K] == float("-inf")).all(), (
        "Masked objects should have -inf in source_ptr"
    )
    assert (preds.dest_ptr[0, :, 5:K] == float("-inf")).all(), (
        "Masked objects should have -inf in dest_ptr"
    )
    # "none" slot (index K) should be finite
    assert torch.isfinite(preds.source_ptr[0, :, K]).all(), (
        "None slot in source_ptr should be finite"
    )
    assert torch.isfinite(preds.dest_ptr[0, :, K]).all(), (
        "None slot in dest_ptr should be finite"
    )
