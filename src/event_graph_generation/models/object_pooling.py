"""Object Pooling Module using Spatiotemporal Slot Attention.

Extracts object-centric representations from V-JEPA 2.1 spatiotemporal tokens
via competitive slot attention, slot refinement, and temporal trajectory extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObjectRepresentation:
    """Object-centric representation produced by ObjectPoolingModule.

    Args:
        identity: Per-slot identity features, shape (B, K, d_model).
        trajectory: Per-slot per-timestep features, shape (B, K, T_enc, d_model).
        existence: Per-slot existence probability (sigmoid), shape (B, K).
        categories: Per-slot category logits, shape (B, K, n_categories).
        attn_maps: Slot-to-token attention weights, shape (B, K, S).
    """

    identity: torch.Tensor
    trajectory: torch.Tensor
    existence: torch.Tensor
    categories: torch.Tensor
    attn_maps: torch.Tensor


class InputProjection(nn.Module):
    """Project V-JEPA tokens and add spatiotemporal positional encodings.

    Args:
        input_dim: V-JEPA hidden size (e.g. 1024).
        d_model: Output projection dimension.
        temporal_tokens: Number of temporal positions (e.g. 8).
        spatial_tokens: Number of spatial positions per timestep (e.g. 196).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        temporal_tokens: int,
        spatial_tokens: int,
    ) -> None:
        super().__init__()
        self.temporal_tokens = temporal_tokens
        self.spatial_tokens = spatial_tokens
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.temporal_pos = nn.Parameter(torch.randn(temporal_tokens, d_model) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(spatial_tokens, d_model) * 0.02)

    def forward(self, vjepa_tokens: torch.Tensor) -> torch.Tensor:
        """Project and add positional encodings.

        Args:
            vjepa_tokens: (B, S, input_dim) where S = temporal_tokens * spatial_tokens.

        Returns:
            Projected tokens with positional encoding, shape (B, S, d_model).
        """
        B, S, _ = vjepa_tokens.shape
        x = self.norm(self.proj(vjepa_tokens))  # (B, S, d_model)

        # Build (S, d_model) positional encoding by broadcasting temporal + spatial
        # Tokens are ordered as [t0_s0, t0_s1, ..., t0_s195, t1_s0, ..., t7_s195]
        temporal_pe = self.temporal_pos.unsqueeze(1).expand(
            -1, self.spatial_tokens, -1
        )  # (T, Sp, d_model)
        spatial_pe = self.spatial_pos.unsqueeze(0).expand(
            self.temporal_tokens, -1, -1
        )  # (T, Sp, d_model)
        pos = (temporal_pe + spatial_pe).reshape(S, -1)  # (S, d_model)

        return x + pos.unsqueeze(0)  # (B, S, d_model)


class SpatiotemporalSlotAttention(nn.Module):
    """Iterative slot attention with competitive binding.

    Args:
        d_model: Slot and token dimension.
        num_slots: Number of object slots K.
        num_iterations: Number of attention iterations.
    """

    def __init__(
        self,
        d_model: int,
        num_slots: int,
        num_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        # Learnable slot initialization parameters
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, d_model))

        # Attention projections
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)

        # GRU for slot updates
        self.gru = nn.GRUCell(d_model, d_model)

        # Residual MLP after GRU
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

        self._scale = d_model ** -0.5

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run iterative slot attention.

        Args:
            tokens: (B, S, d_model) projected V-JEPA tokens.

        Returns:
            slots: (B, K, d_model) final slot representations.
            attn_maps: (B, K, S) last-iteration attention weights (slot-softmax).
        """
        B, S, D = tokens.shape

        # Initialize slots via reparameterization (noise only during training)
        mu = self.slot_mu.expand(B, -1, -1)
        if self.training:
            sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
            slots = mu + sigma * torch.randn_like(mu)  # (B, K, D)
        else:
            slots = mu.clone()  # deterministic at eval

        # Pre-compute keys and values from tokens (shared across iterations)
        k = self.to_k(tokens)  # (B, S, D)
        v = self.to_v(tokens)  # (B, S, D)

        attn_weights = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.to_q(slots)  # (B, K, D)

            # Attention: (B, K, S)
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * self._scale

            # Slot-axis softmax for competitive binding (each token distributes
            # its attention weight across slots)
            attn_weights = F.softmax(attn_logits, dim=1)  # (B, K, S)

            # Normalize per slot (so each slot gets a weighted average)
            attn_norm = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)

            # Weighted aggregate of values
            updates = torch.bmm(attn_norm, v)  # (B, K, D)

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, D),
                slots_prev.reshape(B * self.num_slots, D),
            ).reshape(B, self.num_slots, D)

            # Residual MLP
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_weights  # type: ignore[return-value]


class SlotRefinement(nn.Module):
    """Self-attention refinement over slots.

    Args:
        d_model: Slot dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Refine slots via self-attention.

        Args:
            slots: (B, K, d_model).

        Returns:
            Refined slots, shape (B, K, d_model).
        """
        return self.encoder(slots)


class TemporalTrajectoryExtractor(nn.Module):
    """Extract per-timestep trajectory features and existence scores from slots.

    Args:
        d_model: Feature dimension.
        temporal_tokens: Number of temporal positions T_enc.
        spatial_tokens: Number of spatial positions per timestep.
    """

    def __init__(
        self,
        d_model: int,
        temporal_tokens: int,
        spatial_tokens: int,
    ) -> None:
        super().__init__()
        self.temporal_tokens = temporal_tokens
        self.spatial_tokens = spatial_tokens
        self.trajectory_proj = nn.Linear(d_model, d_model)
        self.existence_head = nn.Linear(d_model, 1)

    def forward(
        self,
        slots: torch.Tensor,
        attn_maps: torch.Tensor,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract trajectories and existence from slot attention maps.

        Args:
            slots: (B, K, d_model) refined slot features.
            attn_maps: (B, K, S) slot attention weights.
            tokens: (B, S, d_model) projected input tokens.

        Returns:
            identity: (B, K, d_model) slot identity features.
            trajectory: (B, K, T_enc, d_model) per-timestep features.
            existence: (B, K) existence probabilities.
        """
        B, K, S = attn_maps.shape
        T = self.temporal_tokens
        Sp = self.spatial_tokens

        # Reshape attention maps: (B, K, T, Sp)
        attn_3d = attn_maps.reshape(B, K, T, Sp)

        # Reshape tokens: (B, T, Sp, d_model)
        tokens_3d = tokens.reshape(B, T, Sp, -1)

        # Per-timestep attention-weighted pooling
        # attn_3d: (B, K, T, Sp) -> normalize per timestep
        attn_per_t = attn_3d / (attn_3d.sum(dim=3, keepdim=True) + 1e-8)

        # (B, K, T, Sp) x (B, 1, T, Sp, D) -> (B, K, T, D)
        trajectory = torch.einsum("bkts,btsd->bktd", attn_per_t, tokens_3d)
        trajectory = self.trajectory_proj(trajectory)  # (B, K, T, d_model)

        # Identity: slot features directly
        identity = slots  # (B, K, d_model)

        # Existence: from slot mean features
        existence = self.existence_head(slots).squeeze(-1)  # (B, K)
        existence = torch.sigmoid(existence)

        return identity, trajectory, existence


class ObjectPoolingModule(nn.Module):
    """Full Object Pooling pipeline: project -> slot attention -> refine -> extract.

    Args:
        input_dim: V-JEPA hidden size (default 1024).
        d_model: Internal feature dimension (default 256).
        num_slots: Number of object slots K (default 24).
        num_iterations: Slot attention iterations (default 3).
        num_refinement_layers: Slot refinement transformer layers (default 2).
        nhead: Number of attention heads (default 8).
        n_categories: Number of object categories including unknown (default 29).
        temporal_tokens: Number of temporal positions (default 8).
        spatial_tokens: Number of spatial positions per timestep (default 196).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        d_model: int = 256,
        num_slots: int = 24,
        num_iterations: int = 3,
        num_refinement_layers: int = 2,
        nhead: int = 8,
        n_categories: int = 29,
        temporal_tokens: int = 8,
        spatial_tokens: int = 196,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = InputProjection(
            input_dim=input_dim,
            d_model=d_model,
            temporal_tokens=temporal_tokens,
            spatial_tokens=spatial_tokens,
        )
        self.slot_attn = SpatiotemporalSlotAttention(
            d_model=d_model,
            num_slots=num_slots,
            num_iterations=num_iterations,
        )
        self.slot_refine = SlotRefinement(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_refinement_layers,
            dropout=dropout,
        )
        self.traj_ext = TemporalTrajectoryExtractor(
            d_model=d_model,
            temporal_tokens=temporal_tokens,
            spatial_tokens=spatial_tokens,
        )
        self.category_head = nn.Linear(d_model, n_categories)

    def forward(self, vjepa_tokens: torch.Tensor) -> ObjectRepresentation:
        """Forward pass: V-JEPA tokens -> object representations.

        Args:
            vjepa_tokens: (B, S, input_dim) V-JEPA spatiotemporal tokens
                where S = temporal_tokens * spatial_tokens.

        Returns:
            ObjectRepresentation with identity, trajectory, existence,
            categories, and attention maps.
        """
        tokens = self.input_proj(vjepa_tokens)  # (B, S, d_model)
        slots, attn = self.slot_attn(tokens)  # (B, K, d_model), (B, K, S)
        slots = self.slot_refine(slots)  # (B, K, d_model)
        identity, trajectory, existence = self.traj_ext(slots, attn, tokens)
        categories = self.category_head(slots)  # (B, K, n_categories)
        return ObjectRepresentation(
            identity=identity,
            trajectory=trajectory,
            existence=existence,
            categories=categories,
            attn_maps=attn,
        )
