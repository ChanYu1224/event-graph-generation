"""End-to-end inference pipeline for event graph generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from event_graph_generation.data.frame_sampler import FrameSampler
from event_graph_generation.inference.postprocess import (
    build_event_graph,
    predictions_to_events,
)
from event_graph_generation.schemas.event_graph import EventGraph
from event_graph_generation.tracking.feature_extractor import FeatureExtractor
from event_graph_generation.tracking.sam3_tracker import SAM3Tracker

if TYPE_CHECKING:
    from event_graph_generation.models.event_decoder import EventDecoder

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end video to EventGraph inference pipeline.

    Orchestrates frame sampling, object tracking, feature extraction,
    sliding-window event decoding, deduplication, and graph construction.
    """

    def __init__(
        self,
        sam3_tracker: SAM3Tracker,
        feature_extractor: FeatureExtractor,
        event_decoder: EventDecoder,
        frame_sampler: FrameSampler,
        config: dict,
    ) -> None:
        self.sam3_tracker = sam3_tracker
        self.feature_extractor = feature_extractor
        self.event_decoder = event_decoder
        self.frame_sampler = frame_sampler
        self.config = config
        self.device = config.get("device", "cuda")

    def process_video(
        self,
        video_path: str,
        concept_prompts: list[str],
        confidence_threshold: float = 0.5,
    ) -> EventGraph:
        """End-to-end: video -> EventGraph.

        1. Sample frames using FrameSampler
        2. Track objects with SAM3Tracker
        3. Extract features with FeatureExtractor
        4. Run sliding window inference with EventDecoder
        5. Deduplicate overlapping events (NMS-like)
        6. Build EventGraph

        Args:
            video_path: Path to input video file.
            concept_prompts: Text prompts for object detection
                (e.g. ["person", "wrench", "drawer"]).
            confidence_threshold: Minimum confidence to keep an event.

        Returns:
            Constructed EventGraph.
        """
        # 1. Sample frames
        logger.info("Sampling frames from %s", video_path)
        sampled_frames = self.frame_sampler.sample(video_path)
        frames = [sf.image for sf in sampled_frames]
        frame_indices = [sf.frame_index for sf in sampled_frames]

        # 2. Track objects
        logger.info("Tracking objects with %d concept prompts", len(concept_prompts))
        self.sam3_tracker.set_concept_prompts(concept_prompts)
        tracking_results = self.sam3_tracker.track_video(frames, frame_indices)

        # 3. Extract features
        logger.info("Extracting features")
        object_features, pairwise_features = self.feature_extractor.extract(
            tracking_results
        )

        # Build track_id -> slot_index mapping and reverse
        sorted_track_ids = sorted(object_features.keys())
        slot_to_track = {i: tid for i, tid in enumerate(sorted_track_ids)}

        # Build category_id -> category_name mapping.
        # FeatureExtractor assigns category_ids based on sorted unique category names
        # seen during extraction, NOT based on concept_prompts order.
        unique_categories = sorted({
            obj.category
            for fr in tracking_results
            for obj in fr.objects
        })
        cat_id_to_name = {i: name for i, name in enumerate(unique_categories)}

        # Collect tracked object info for graph construction
        tracked_objects = []
        for tid in sorted_track_ids:
            obj_feat = object_features[tid]
            # Find first/last frame where object is present
            presence = obj_feat.presence_seq.squeeze(-1)  # (T,)
            present_frames = [
                frame_indices[t] for t in range(len(presence))
                if t < len(frame_indices) and presence[t] > 0.5
            ]
            first_frame = min(present_frames) if present_frames else 0
            last_frame = max(present_frames) if present_frames else 0

            # Map category_id using the same sorted-unique mapping as FeatureExtractor
            cat_id = obj_feat.category_id
            category = cat_id_to_name.get(cat_id, f"object_{cat_id}")

            tracked_objects.append({
                "track_id": tid,
                "category": category,
                "first_seen_frame": first_frame,
                "last_seen_frame": last_frame,
                "confidence": 1.0,
            })

        # 4. Sliding window inference
        clip_length = self.config.get("clip_length", 16)
        clip_stride = self.config.get("clip_stride", 8)
        action_names = self.config.get("action_names", [])

        raw_events = self._sliding_window_inference(
            object_features=object_features,
            pairwise_features=pairwise_features,
            clip_length=clip_length,
            clip_stride=clip_stride,
            frame_indices=frame_indices,
            slot_to_track=slot_to_track,
            action_names=action_names,
            confidence_threshold=confidence_threshold,
        )

        # 5. Deduplicate
        frame_threshold = self.config.get("dedup_frame_threshold", 3)
        events = self._deduplicate_events(raw_events, frame_threshold=frame_threshold)

        # 6. Build EventGraph
        video_id = str(video_path)
        graph = build_event_graph(
            video_id=video_id,
            tracked_objects=tracked_objects,
            events=events,
            metadata={
                "num_frames_sampled": len(sampled_frames),
                "concept_prompts": concept_prompts,
                "confidence_threshold": confidence_threshold,
            },
        )

        logger.info(
            "Pipeline complete: %d nodes, %d edges",
            len(graph.nodes),
            len(graph.edges),
        )
        return graph

    def _sliding_window_inference(
        self,
        object_features: dict,
        pairwise_features: list,
        clip_length: int,
        clip_stride: int,
        frame_indices: list[int] | None = None,
        slot_to_track: dict[int, int] | None = None,
        action_names: list[str] | None = None,
        confidence_threshold: float = 0.5,
    ) -> list[dict]:
        """Run EventDecoder on sliding windows over the temporal dimension.

        Args:
            object_features: Dict mapping track_id -> ObjectFeatures.
            pairwise_features: List of PairwiseFeatures.
            clip_length: Number of frames per clip window.
            clip_stride: Stride between windows.
            frame_indices: Original frame numbers.
            slot_to_track: Mapping from slot index to track ID.
            action_names: Action vocabulary.
            confidence_threshold: Minimum confidence for events.

        Returns:
            List of raw event dicts from all windows.
        """
        if frame_indices is None:
            frame_indices = list(range(clip_length))
        if slot_to_track is None:
            slot_to_track = {}
        if action_names is None:
            action_names = []

        sorted_track_ids = sorted(object_features.keys())
        K = len(sorted_track_ids)
        T_total = len(frame_indices)

        if K == 0:
            logger.warning("No tracked objects, returning empty events")
            return []

        all_events: list[dict] = []

        # Determine number of windows
        max_objects = self.config.get("max_objects", 30)

        for win_start in range(0, max(1, T_total - clip_length + 1), clip_stride):
            win_end = min(win_start + clip_length, T_total)
            win_frame_indices = frame_indices[win_start:win_end]
            T_win = len(win_frame_indices)

            # Build tensors for this window
            # object_embeddings: (1, K, D_emb)
            embeddings = []
            for tid in sorted_track_ids:
                embeddings.append(object_features[tid].embedding)
            object_embeddings = torch.stack(embeddings).unsqueeze(0)  # (1, K, D_emb)

            # object_temporal: (1, K, T_win, D_geo)
            d_geo = self.config.get("d_geo", 12)
            object_temporal = torch.zeros(1, K, T_win, d_geo)
            for slot_idx, tid in enumerate(sorted_track_ids):
                feat = object_features[tid]
                # Pack geometric features: bbox(4) + centroid(2) + area(1) + presence(1) + delta_centroid(2) + delta_area(1) + velocity(1) = 12
                for t_local in range(T_win):
                    t_global = win_start + t_local
                    if t_global < feat.bbox_seq.shape[0]:
                        geo = torch.cat([
                            feat.bbox_seq[t_global],           # 4
                            feat.centroid_seq[t_global],       # 2
                            feat.area_seq[t_global],           # 1
                            feat.presence_seq[t_global],       # 1
                        ])
                        # Delta features (use t_global - 1 for deltas)
                        if t_global > 0 and (t_global - 1) < feat.delta_centroid_seq.shape[0]:
                            delta_geo = torch.cat([
                                feat.delta_centroid_seq[t_global - 1],  # 2
                                feat.delta_area_seq[t_global - 1],      # 1
                                feat.velocity_seq[t_global - 1],        # 1
                            ])
                        else:
                            delta_geo = torch.zeros(4)
                        full_geo = torch.cat([geo, delta_geo])  # 12
                        object_temporal[0, slot_idx, t_local, :full_geo.shape[0]] = full_geo

            # pairwise: (1, K, K, T_win, D_pair)
            d_pair = self.config.get("d_pair", 7)
            pairwise_tensor = torch.zeros(1, K, K, T_win, d_pair)
            # Build lookup for pairwise features
            pair_lookup = {}
            for pf in pairwise_features:
                pair_lookup[(pf.track_id_i, pf.track_id_j)] = pf

            for si, tid_i in enumerate(sorted_track_ids):
                for sj, tid_j in enumerate(sorted_track_ids):
                    if si == sj:
                        continue
                    key = (min(tid_i, tid_j), max(tid_i, tid_j))
                    pf = pair_lookup.get(key)
                    if pf is None:
                        continue
                    for t_local in range(T_win):
                        t_global = win_start + t_local
                        if t_global < pf.iou_seq.shape[0]:
                            pair_vec = torch.cat([
                                pf.iou_seq[t_global],
                                pf.distance_seq[t_global],
                                pf.containment_ij_seq[t_global],
                                pf.containment_ji_seq[t_global],
                                pf.relative_position_seq[t_global],
                            ])  # 7
                            pairwise_tensor[0, si, sj, t_local, :pair_vec.shape[0]] = pair_vec

            # object_mask: (1, K) padded to max_objects
            object_mask = torch.zeros(1, max_objects, dtype=torch.bool)
            object_mask[0, :K] = True

            # Pad tensors to max_objects
            D_emb = object_embeddings.shape[2]
            padded_emb = torch.zeros(1, max_objects, D_emb)
            padded_emb[0, :K] = object_embeddings[0]

            padded_temporal = torch.zeros(1, max_objects, T_win, d_geo)
            padded_temporal[0, :K] = object_temporal[0]

            padded_pairwise = torch.zeros(1, max_objects, max_objects, T_win, d_pair)
            padded_pairwise[0, :K, :K] = pairwise_tensor[0]

            # Move to device
            padded_emb = padded_emb.to(self.device)
            padded_temporal = padded_temporal.to(self.device)
            padded_pairwise = padded_pairwise.to(self.device)
            object_mask = object_mask.to(self.device)

            # Run model
            with torch.no_grad():
                preds = self.event_decoder(
                    object_embeddings=padded_emb,
                    object_temporal=padded_temporal,
                    pairwise=padded_pairwise,
                    object_mask=object_mask,
                )

            # Convert predictions to events
            window_events = predictions_to_events(
                predictions=preds,
                track_id_map=slot_to_track,
                action_names=action_names,
                frame_indices=win_frame_indices,
                confidence_threshold=confidence_threshold,
            )
            all_events.extend(window_events)

        logger.info("Sliding window produced %d raw events", len(all_events))
        return all_events

    def _deduplicate_events(
        self,
        raw_events: list[dict],
        frame_threshold: int = 3,
    ) -> list[dict]:
        """Remove duplicate events from overlapping windows.

        Two events are duplicates if they have the same action, agent, target
        and frame difference <= threshold. The one with higher confidence is kept.

        Args:
            raw_events: List of event dicts, possibly with duplicates.
            frame_threshold: Maximum frame difference to consider as duplicate.

        Returns:
            Deduplicated list of event dicts.
        """
        if not raw_events:
            return []

        # Sort by confidence descending (keep higher confidence first)
        sorted_events = sorted(raw_events, key=lambda e: e["confidence"], reverse=True)

        kept: list[dict] = []

        for event in sorted_events:
            is_duplicate = False
            for kept_event in kept:
                if (
                    event["action"] == kept_event["action"]
                    and event["agent_track_id"] == kept_event["agent_track_id"]
                    and event["target_track_id"] == kept_event["target_track_id"]
                    and abs(event["frame"] - kept_event["frame"]) <= frame_threshold
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(event)

        logger.info(
            "Deduplicated %d -> %d events (frame_threshold=%d)",
            len(raw_events),
            len(kept),
            frame_threshold,
        )
        return kept
