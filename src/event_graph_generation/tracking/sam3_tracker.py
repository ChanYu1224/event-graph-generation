"""SAM 3 wrapper for object tracking and segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from sam3 import SAM3

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

    Uses SAM 3's text-prompted segmentation to detect and track objects
    across video frames.
    """

    def __init__(self, model_size: str = "large", device: str = "cuda") -> None:
        """Initialize SAM 3 tracker.

        Args:
            model_size: SAM 3 model size ('large', 'base', 'small').
            device: Device to run the model on.
        """
        self.model_size = model_size
        self.device = device
        self.concept_prompts: list[str] = []
        self.model: SAM3 | None = None

        if _SAM3_AVAILABLE:
            self.model = SAM3(model_size=model_size, device=device)
            logger.info("SAM 3 model (%s) loaded on %s", model_size, device)
        else:
            logger.warning("SAM 3 not available; model not loaded.")

    def set_concept_prompts(self, prompts: list[str]) -> None:
        """Store concept text prompts for tracking.

        Args:
            prompts: List of text prompts describing object categories to track
                (e.g., ["person", "car", "dog"]).
        """
        self.concept_prompts = list(prompts)
        logger.info("Set %d concept prompts: %s", len(prompts), prompts)

    def track_video(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
    ) -> list[FrameTrackingResult]:
        """Process video frames through SAM 3 for object tracking.

        Processes one frame at a time to manage GPU memory. For each frame,
        detects objects matching the configured concept prompts.

        Args:
            frames: List of video frames as numpy arrays (H, W, 3) in BGR or RGB.
            frame_indices: Corresponding frame indices in the original video.

        Returns:
            List of FrameTrackingResult, one per input frame.

        Raises:
            RuntimeError: If sam3 is not installed.
        """
        if not _SAM3_AVAILABLE or self.model is None:
            raise RuntimeError(
                "sam3 is not installed. Cannot run tracking. "
                "Install sam3 with: pip install sam3"
            )

        if not self.concept_prompts:
            logger.warning("No concept prompts set. Call set_concept_prompts() first.")
            return [
                FrameTrackingResult(frame_index=idx, objects=[])
                for idx in frame_indices
            ]

        results: list[FrameTrackingResult] = []

        # Initialize SAM 3 with text prompts
        self.model.set_text_prompts(self.concept_prompts)

        for frame, frame_idx in zip(frames, frame_indices):
            logger.debug("Processing frame %d", frame_idx)

            # Run SAM 3 inference on a single frame
            model_output = self.model.predict(frame)

            tracked_objects: list[TrackedObject] = []

            # Parse SAM 3 output: iterate over detected objects
            for det_idx, detection in enumerate(model_output.detections):
                track_id = int(detection.track_id)
                category = str(detection.category)
                mask = detection.mask.astype(bool)  # (H, W)
                bbox = np.array(detection.bbox, dtype=np.float32)  # [x1, y1, x2, y2]
                score = float(detection.score)

                # Extract embedding from SAM 3 internals
                embedding = self._extract_embedding(detection)

                tracked_objects.append(
                    TrackedObject(
                        track_id=track_id,
                        category=category,
                        mask=mask,
                        bbox=bbox,
                        score=score,
                        embedding=embedding,
                    )
                )

            results.append(
                FrameTrackingResult(frame_index=frame_idx, objects=tracked_objects)
            )

            # Free intermediate GPU tensors
            if hasattr(model_output, "clear_cache"):
                model_output.clear_cache()

        logger.info(
            "Tracked %d frames, found objects: %s",
            len(results),
            [len(r.objects) for r in results],
        )
        return results

    @staticmethod
    def _extract_embedding(model_output) -> torch.Tensor:
        """Extract object embedding from SAM 3's internal representation.

        Attempts to extract the DETR decoder output embedding. Falls back
        to a zero vector if extraction fails.

        Args:
            model_output: A single detection output from SAM 3.

        Returns:
            A 256-dimensional embedding tensor.
        """
        try:
            # SAM 3 stores DETR decoder output in the detection object
            if hasattr(model_output, "decoder_embedding"):
                emb = model_output.decoder_embedding
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                emb = emb.detach().cpu().float()
                # Project or truncate/pad to 256 dims
                if emb.numel() > 256:
                    emb = emb.flatten()[:256]
                elif emb.numel() < 256:
                    padded = torch.zeros(256)
                    padded[: emb.numel()] = emb.flatten()
                    emb = padded
                else:
                    emb = emb.flatten()
                return emb
            elif hasattr(model_output, "embedding"):
                emb = model_output.embedding
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                emb = emb.detach().cpu().float().flatten()
                if emb.shape[0] != 256:
                    padded = torch.zeros(256)
                    n = min(emb.shape[0], 256)
                    padded[:n] = emb[:n]
                    emb = padded
                return emb
        except Exception:
            logger.debug("Failed to extract embedding, returning zeros.", exc_info=True)

        return torch.zeros(256)
