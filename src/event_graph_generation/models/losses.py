"""Loss functions for the Event Decoder with Hungarian matching."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .event_decoder import EventPredictions


class EventGraphLoss(nn.Module):
    """Combined loss for event graph prediction using Hungarian matching.

    For each batch item, computes optimal matching between predicted event
    slots and ground-truth events, then computes per-head losses on matched pairs.
    """

    def __init__(
        self,
        loss_weights: dict[str, float] | None = None,
        num_actions: int = 13,
    ) -> None:
        """Initialize loss module.

        Args:
            loss_weights: Dict with keys: interaction, action, agent_ptr,
                target_ptr, source_ptr, dest_ptr, frame.
            num_actions: Number of action classes.
        """
        super().__init__()
        self.loss_weights = loss_weights or {
            "interaction": 2.0,
            "action": 1.0,
            "agent_ptr": 1.0,
            "target_ptr": 1.0,
            "source_ptr": 0.5,
            "dest_ptr": 0.5,
            "frame": 0.5,
        }
        self.num_actions = num_actions

    def forward(
        self,
        predictions: EventPredictions,
        gt_events: list[list[dict]],
        object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss with Hungarian matching.

        Args:
            predictions: EventPredictions from the decoder.
            gt_events: List of list of GT event dicts per batch item.
                Each dict has keys: agent_track_id, action_class,
                target_track_id, source_track_id, dest_track_id,
                event_frame_index.
            object_mask: (B, K) boolean mask for valid objects.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict has per-head losses.
        """
        device = predictions.interaction.device
        B, M, _ = predictions.interaction.shape

        total_interaction_loss = torch.tensor(0.0, device=device)
        total_action_loss = torch.tensor(0.0, device=device)
        total_agent_loss = torch.tensor(0.0, device=device)
        total_target_loss = torch.tensor(0.0, device=device)
        total_source_loss = torch.tensor(0.0, device=device)
        total_dest_loss = torch.tensor(0.0, device=device)
        total_frame_loss = torch.tensor(0.0, device=device)
        total_matched = 0  # Count matched events for per-head normalization

        for b in range(B):
            gt_list = gt_events[b]
            N_gt = len(gt_list)

            # Interaction targets: 1 for matched, 0 for unmatched
            interaction_target = torch.zeros(M, 1, device=device)

            if N_gt == 0:
                # No GT events: all predictions should be inactive
                interaction_loss = F.binary_cross_entropy_with_logits(
                    predictions.interaction[b], interaction_target
                )
                total_interaction_loss = total_interaction_loss + interaction_loss
                continue

            # Hungarian matching
            matched_pred, matched_gt = self._hungarian_match(
                predictions, b, gt_list, object_mask[b]
            )

            # Mark matched predictions as active
            for pred_idx in matched_pred:
                interaction_target[pred_idx, 0] = 1.0

            # Interaction loss (all M predictions)
            interaction_loss = F.binary_cross_entropy_with_logits(
                predictions.interaction[b], interaction_target
            )
            total_interaction_loss = total_interaction_loss + interaction_loss

            if len(matched_pred) == 0:
                continue

            total_matched += len(matched_pred)

            # Pre-build target tensors for matched GT events on device
            none_source_idx = predictions.source_ptr.shape[-1] - 1
            none_dest_idx = predictions.dest_ptr.shape[-1] - 1

            gt_matched = [gt_list[gi] for gi in matched_gt]
            action_targets = torch.tensor(
                [e["action_class"] for e in gt_matched], device=device, dtype=torch.long
            )
            agent_targets = torch.tensor(
                [e["agent_track_id"] for e in gt_matched], device=device, dtype=torch.long
            )
            target_targets = torch.tensor(
                [e["target_track_id"] for e in gt_matched], device=device, dtype=torch.long
            )
            source_ids = [
                e["source_track_id"] if e.get("source_track_id") is not None else none_source_idx
                for e in gt_matched
            ]
            source_targets = torch.tensor(source_ids, device=device, dtype=torch.long)
            dest_ids = [
                e["dest_track_id"] if e.get("dest_track_id") is not None else none_dest_idx
                for e in gt_matched
            ]
            dest_targets = torch.tensor(dest_ids, device=device, dtype=torch.long)
            frame_targets = torch.tensor(
                [e["event_frame_index"] for e in gt_matched], device=device, dtype=torch.long
            )

            # Losses for matched pairs
            for i, pred_idx in enumerate(matched_pred):
                action_loss = F.cross_entropy(
                    predictions.action[b, pred_idx].unsqueeze(0),
                    action_targets[i].unsqueeze(0),
                )
                total_action_loss = total_action_loss + action_loss

                agent_loss = F.cross_entropy(
                    predictions.agent_ptr[b, pred_idx].unsqueeze(0),
                    agent_targets[i].unsqueeze(0),
                )
                total_agent_loss = total_agent_loss + agent_loss

                target_loss = F.cross_entropy(
                    predictions.target_ptr[b, pred_idx].unsqueeze(0),
                    target_targets[i].unsqueeze(0),
                )
                total_target_loss = total_target_loss + target_loss

                source_loss = F.cross_entropy(
                    predictions.source_ptr[b, pred_idx].unsqueeze(0),
                    source_targets[i].unsqueeze(0),
                )
                total_source_loss = total_source_loss + source_loss

                dest_loss = F.cross_entropy(
                    predictions.dest_ptr[b, pred_idx].unsqueeze(0),
                    dest_targets[i].unsqueeze(0),
                )
                total_dest_loss = total_dest_loss + dest_loss

                frame_loss = F.cross_entropy(
                    predictions.frame[b, pred_idx].unsqueeze(0),
                    frame_targets[i].unsqueeze(0),
                )
                total_frame_loss = total_frame_loss + frame_loss

        # Interaction loss: normalize by batch size (computed for all M slots per item)
        total_interaction_loss = total_interaction_loss / B
        # Per-head losses: normalize by total matched events (not batch size)
        n_matched = max(total_matched, 1)
        total_action_loss = total_action_loss / n_matched
        total_agent_loss = total_agent_loss / n_matched
        total_target_loss = total_target_loss / n_matched
        total_source_loss = total_source_loss / n_matched
        total_dest_loss = total_dest_loss / n_matched
        total_frame_loss = total_frame_loss / n_matched

        # Weighted total
        total_loss = (
            self.loss_weights["interaction"] * total_interaction_loss
            + self.loss_weights["action"] * total_action_loss
            + self.loss_weights["agent_ptr"] * total_agent_loss
            + self.loss_weights["target_ptr"] * total_target_loss
            + self.loss_weights["source_ptr"] * total_source_loss
            + self.loss_weights["dest_ptr"] * total_dest_loss
            + self.loss_weights["frame"] * total_frame_loss
        )

        loss_dict = {
            "interaction": total_interaction_loss.item(),
            "action": total_action_loss.item(),
            "agent_ptr": total_agent_loss.item(),
            "target_ptr": total_target_loss.item(),
            "source_ptr": total_source_loss.item(),
            "dest_ptr": total_dest_loss.item(),
            "frame": total_frame_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict

    def _hungarian_match(
        self,
        predictions: EventPredictions,
        batch_idx: int,
        gt_list: list[dict],
        obj_mask: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        """Compute Hungarian matching between predictions and GT events.

        Args:
            predictions: Full batch predictions.
            batch_idx: Index into the batch.
            gt_list: List of GT event dicts for this batch item.
            obj_mask: (K,) boolean mask for valid objects.

        Returns:
            Tuple of (matched_pred_indices, matched_gt_indices).
        """
        M = predictions.interaction.shape[1]
        N_gt = len(gt_list)

        if N_gt == 0:
            return [], []

        # Build cost matrix (M x N_gt)
        cost = torch.zeros(M, N_gt)

        with torch.no_grad():
            action_probs = F.softmax(predictions.action[batch_idx], dim=-1)  # (M, A)
            agent_probs = F.softmax(predictions.agent_ptr[batch_idx], dim=-1)  # (M, K)
            target_probs = F.softmax(predictions.target_ptr[batch_idx], dim=-1)  # (M, K)
            frame_probs = F.softmax(predictions.frame[batch_idx], dim=-1)  # (M, T)

            for j, gt_evt in enumerate(gt_list):
                action_cls = gt_evt["action_class"]
                agent_id = gt_evt["agent_track_id"]
                target_id = gt_evt["target_track_id"]
                frame_idx = gt_evt["event_frame_index"]

                # Cost = negative log probability (lower is better)
                action_cost = -torch.log(action_probs[:, action_cls].clamp(min=1e-8))
                agent_cost = -torch.log(agent_probs[:, agent_id].clamp(min=1e-8))
                target_cost = -torch.log(target_probs[:, target_id].clamp(min=1e-8))
                frame_cost = -torch.log(frame_probs[:, frame_idx].clamp(min=1e-8))

                cost[:, j] = action_cost + agent_cost + target_cost + frame_cost

        cost_np = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        return list(row_ind), list(col_ind)
