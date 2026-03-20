"""SAM 3 wrapper for object tracking and segmentation."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    _SAM3_AVAILABLE = True
except ImportError:
    _SAM3_AVAILABLE = False
    logger.warning(
        "sam3 is not installed. SAM3Tracker will not be functional. "
        "Install sam3 to enable tracking capabilities."
    )


@dataclass
class TrackedObject:
    """A single tracked object in a frame."""

    track_id: int
    category: str
    mask: np.ndarray  # (H, W) bool
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    embedding: torch.Tensor  # (256,)


@dataclass
class FrameTrackingResult:
    """Tracking results for a single frame."""

    frame_index: int
    objects: list[TrackedObject] = field(default_factory=list)


class SAM3Tracker:
    """Wrapper around SAM 3 for video object tracking and segmentation.

    Uses Sam3VideoPredictor's text-prompted grounding to detect and track
    objects across video frames.
    """

    def __init__(self, model_size: str = "large", device: str = "cuda") -> None:
        self.model_size = model_size
        self.device = device
        self.concept_prompts: list[str] = []
        self.predictor: Sam3VideoPredictor | None = None

        if _SAM3_AVAILABLE:
            self.predictor = Sam3VideoPredictor()
            logger.info("Sam3VideoPredictor loaded")
        else:
            logger.warning("SAM 3 not available; model not loaded.")

    def set_concept_prompts(self, prompts: list[str]) -> None:
        """Store concept text prompts for tracking."""
        self.concept_prompts = list(prompts)
        logger.info("Set %d concept prompts: %s", len(prompts), prompts)

    def _build_text_prompt(self) -> str:
        """Join concept prompts into a single grounding text string."""
        return " . ".join(self.concept_prompts)

    def _start_session(self, resource_path) -> str:
        """Start a SAM3 session with offload_video_to_cpu=True to avoid OOM."""
        inference_state = self.predictor.model.init_state(
            resource_path=resource_path,
            offload_video_to_cpu=True,
            async_loading_frames=self.predictor.async_loading_frames,
            video_loader_type=self.predictor.video_loader_type,
        )
        session_id = str(uuid.uuid4())
        self.predictor._ALL_INFERENCE_STATES[session_id] = {
            "state": inference_state,
            "session_id": session_id,
        }
        return session_id

    def _run_session(
        self,
        session_id: str,
        frame_indices: list[int] | None,
    ) -> list[FrameTrackingResult]:
        """Add prompt, propagate, and collect results for a session."""
        text_prompt = self._build_text_prompt()
        try:
            self.predictor.add_prompt(
                session_id=session_id,
                frame_idx=0,
                text=text_prompt,
            )

            frame_outputs: dict[int, dict] = {}
            for result in self.predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=None,
            ):
                frame_outputs[result["frame_index"]] = result["outputs"]
        finally:
            self.predictor.close_session(session_id)
            torch.cuda.empty_cache()

        num_frames = len(frame_outputs)
        if frame_indices is None:
            frame_indices = list(range(num_frames))

        sorted_frame_idxs = sorted(frame_outputs.keys())
        results: list[FrameTrackingResult] = []
        for i, sam_frame_idx in enumerate(sorted_frame_idxs):
            outputs = frame_outputs[sam_frame_idx]
            original_frame_idx = frame_indices[i] if i < len(frame_indices) else sam_frame_idx
            tracked_objects = self._parse_frame_output(outputs)
            results.append(
                FrameTrackingResult(
                    frame_index=original_frame_idx,
                    objects=tracked_objects,
                )
            )

        logger.info(
            "Tracked %d frames, objects per frame: %s",
            len(results),
            [len(r.objects) for r in results[:5]],
        )
        return results

    def track_video(
        self,
        frames: Sequence[np.ndarray],
        frame_indices: list[int] | None = None,
    ) -> list[FrameTrackingResult]:
        """Run SAM 3 tracking on pre-loaded video frames.

        Args:
            frames: Video frames as numpy arrays (H, W, 3) in BGR format.
            frame_indices: Original frame indices. If None, 0..N-1.

        Returns:
            List of FrameTrackingResult, one per frame.
        """
        if not _SAM3_AVAILABLE or self.predictor is None:
            raise RuntimeError(
                "sam3 is not installed. Cannot run tracking. "
                "Install sam3 with: pip install sam3"
            )
        if not self.concept_prompts:
            logger.warning("No concept prompts set. Call set_concept_prompts() first.")
            return []

        # Convert BGR numpy frames to RGB PIL images for sam3
        pil_frames = [Image.fromarray(f[:, :, ::-1]) for f in frames]
        session_id = self._start_session(pil_frames)
        return self._run_session(session_id, frame_indices)

    def track_frame_dir(
        self,
        frame_dir: str | Path,
        frame_indices: list[int] | None = None,
    ) -> list[FrameTrackingResult]:
        """Run SAM 3 tracking on a directory of pre-extracted frames.

        Args:
            frame_dir: Directory containing JPEG frames (sorted lexicographically).
            frame_indices: Optional list of original frame indices. If None,
                sequential indices 0..N-1 are used.

        Returns:
            List of FrameTrackingResult, one per frame.
        """
        if not _SAM3_AVAILABLE or self.predictor is None:
            raise RuntimeError(
                "sam3 is not installed. Cannot run tracking. "
                "Install sam3 with: pip install sam3"
            )
        if not self.concept_prompts:
            logger.warning("No concept prompts set. Call set_concept_prompts() first.")
            return []

        session_id = self._start_session(str(Path(frame_dir)))
        return self._run_session(session_id, frame_indices)

    @staticmethod
    def _parse_frame_output(outputs: dict) -> list[TrackedObject]:
        """Convert Sam3VideoPredictor output dict to TrackedObject list."""
        obj_ids = outputs.get("out_obj_ids", np.array([]))
        probs = outputs.get("out_probs", np.array([]))
        boxes_xywh = outputs.get("out_boxes_xywh", np.zeros((0, 4)))
        masks = outputs.get("out_binary_masks", np.zeros((0, 0, 0), dtype=bool))

        if len(obj_ids) == 0:
            return []

        # Get image dimensions from masks for denormalization
        if masks.ndim == 3 and masks.shape[0] > 0:
            img_h, img_w = masks.shape[1], masks.shape[2]
        else:
            img_h, img_w = 1, 1

        tracked_objects: list[TrackedObject] = []
        for idx in range(len(obj_ids)):
            # Convert normalized xywh to pixel xyxy
            x, y, w, h = boxes_xywh[idx]
            x1 = x * img_w
            y1 = y * img_h
            x2 = (x + w) * img_w
            y2 = (y + h) * img_h
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            mask = masks[idx] if idx < len(masks) else np.zeros((img_h, img_w), dtype=bool)

            tracked_objects.append(
                TrackedObject(
                    track_id=int(obj_ids[idx]),
                    category="object",
                    mask=mask,
                    bbox=bbox,
                    score=float(probs[idx]) if idx < len(probs) else 0.0,
                    embedding=torch.zeros(256),
                )
            )

        return tracked_objects
