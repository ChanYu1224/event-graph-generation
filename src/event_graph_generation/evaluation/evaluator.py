"""Evaluation loop."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from ..config import EvaluationConfig
from .metrics import get_metric


class Evaluator:
    """Runs evaluation on a dataset."""

    def __init__(self, config: EvaluationConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self.metrics = {name: get_metric(name) for name in config.metrics}

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on the given dataloader."""
        model.eval()

        # Detect batch type from first batch and process all in one pass
        is_event_decoder = None
        all_predictions: list = []
        all_gt_events: list = []
        all_tensor_preds: list[torch.Tensor] = []
        all_tensor_targets: list[torch.Tensor] = []

        for batch in dataloader:
            if is_event_decoder is None:
                is_event_decoder = hasattr(batch, "object_embeddings")

            if is_event_decoder:
                self._process_event_batch(model, batch, all_predictions, all_gt_events)
            else:
                outputs = model(batch.inputs.to(self.device))
                all_tensor_preds.append(outputs.cpu())
                all_tensor_targets.append(batch.targets)

        if is_event_decoder is None:
            return {}

        if is_event_decoder:
            return self._finalize_event_decoder(all_predictions, all_gt_events)
        else:
            return self._finalize_legacy(all_tensor_preds, all_tensor_targets)

    def _finalize_event_decoder(
        self, all_predictions: list, all_gt_events: list
    ) -> dict[str, float]:
        """Compute metrics for EventDecoder evaluation."""

        # Build aggregated predictions and targets for metrics
        pred_dict, target_dict = self._build_metric_dicts(all_predictions, all_gt_events)

        results = {}
        for name, metric_fn in self.metrics.items():
            if name == "accuracy":
                # accuracy metric requires tensors, skip for event decoder
                continue
            results[name] = metric_fn(pred_dict, target_dict)

        return results

    def _process_event_batch(
        self,
        model: nn.Module,
        batch: object,
        all_predictions: list,
        all_gt_events: list,
    ) -> None:
        """Run model on a single EventBatch and collect results."""
        batch = batch.to(self.device)
        predictions = model(
            batch.object_embeddings,
            batch.object_temporal,
            batch.pairwise,
            batch.object_mask,
        )
        all_predictions.append(predictions)
        all_gt_events.extend(batch.gt_events)

    def _build_metric_dicts(
        self, all_predictions: list, all_gt_events: list[list[dict]]
    ) -> tuple[dict, dict]:
        """Build aggregated prediction and target dicts for metrics."""
        interaction_scores = []
        labels = []
        pred_action_classes = []
        gt_action_classes = []
        pred_agent_ptrs = []
        gt_agent_ptrs = []
        pred_target_ptrs = []
        gt_target_ptrs = []
        pred_frame_indices = []
        gt_frame_indices = []
        pred_edges = []
        gt_edges = []

        for preds in all_predictions:
            B = preds.interaction.shape[0]
            M = preds.interaction.shape[1]

            for b in range(B):
                # Interaction scores
                scores = torch.sigmoid(preds.interaction[b, :, 0]).cpu().tolist()
                interaction_scores.extend(scores)

                # Determine which predictions are "active" (score > 0.5)
                active_mask = [s > 0.5 for s in scores]

                # Predicted actions, pointers, frames for active slots
                action_preds = preds.action[b].argmax(dim=-1).cpu().tolist()
                agent_preds = preds.agent_ptr[b].argmax(dim=-1).cpu().tolist()
                target_preds = preds.target_ptr[b].argmax(dim=-1).cpu().tolist()
                frame_preds = preds.frame[b].argmax(dim=-1).cpu().tolist()

                for m in range(M):
                    if active_mask[m]:
                        pred_action_classes.append(action_preds[m])
                        pred_agent_ptrs.append(agent_preds[m])
                        pred_target_ptrs.append(target_preds[m])
                        pred_frame_indices.append(frame_preds[m])
                        pred_edges.append(
                            (agent_preds[m], action_preds[m], target_preds[m])
                        )

        # Process GT events — use Hungarian matching to assign labels
        sample_idx = 0
        for preds in all_predictions:
            B = preds.interaction.shape[0]
            M = preds.interaction.shape[1]
            for b in range(B):
                gt_list = all_gt_events[sample_idx]
                sample_idx += 1
                N_gt = len(gt_list)

                # Compute Hungarian matching to determine which slots are true positives
                matched_slots: set[int] = set()
                if N_gt > 0:
                    with torch.no_grad():
                        action_probs = F.softmax(preds.action[b], dim=-1)
                        agent_probs = F.softmax(preds.agent_ptr[b], dim=-1)
                        target_probs = F.softmax(preds.target_ptr[b], dim=-1)
                        cost = torch.zeros(M, N_gt)
                        for j, gt_evt in enumerate(gt_list):
                            ac = gt_evt.get("action_class", 0)
                            ag = gt_evt.get("agent_track_id", 0)
                            tg = gt_evt.get("target_track_id", 0)
                            cost[:, j] = (
                                -torch.log(action_probs[:, ac].clamp(min=1e-8))
                                - torch.log(agent_probs[:, ag].clamp(min=1e-8))
                                - torch.log(target_probs[:, tg].clamp(min=1e-8))
                            )
                    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
                    matched_slots = set(int(r) for r in row_ind)

                # Labels: 1.0 for matched slots, 0.0 for unmatched
                for m in range(M):
                    labels.append(1.0 if m in matched_slots else 0.0)

                for gt_evt in gt_list:
                    gt_action_classes.append(gt_evt.get("action_class", 0))
                    gt_agent_ptrs.append(gt_evt.get("agent_track_id", 0))
                    gt_target_ptrs.append(gt_evt.get("target_track_id", 0))
                    gt_frame_indices.append(gt_evt.get("event_frame_index", 0))
                    gt_edges.append(
                        (
                            gt_evt.get("agent_track_id", 0),
                            gt_evt.get("action_class", 0),
                            gt_evt.get("target_track_id", 0),
                        )
                    )

        pred_dict = {
            "interaction": interaction_scores,
            "action_classes": pred_action_classes,
            "agent_ptrs": pred_agent_ptrs,
            "target_ptrs": pred_target_ptrs,
            "frame_indices": pred_frame_indices,
            "edges": pred_edges,
        }
        target_dict = {
            "labels": labels,
            "action_classes": gt_action_classes,
            "agent_ptrs": gt_agent_ptrs,
            "target_ptrs": gt_target_ptrs,
            "frame_indices": gt_frame_indices,
            "edges": gt_edges,
        }
        return pred_dict, target_dict

    def _finalize_legacy(
        self, all_predictions: list[torch.Tensor], all_targets: list[torch.Tensor]
    ) -> dict[str, float]:
        """Compute metrics for legacy tensor-based evaluation."""
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        results = {}
        for name, metric_fn in self.metrics.items():
            results[name] = metric_fn(predictions, targets)

        return results
