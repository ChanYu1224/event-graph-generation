"""VLM annotator for local Qwen 3.5 inference on video clips."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from event_graph_generation.annotation.prompts import build_prompt
from event_graph_generation.data.frame_sampler import FrameSampler
from event_graph_generation.schemas.vlm_output import VLMAnnotation

logger = logging.getLogger(__name__)


class VLMAnnotator:
    """Local VLM annotator using Qwen 3.5 for video event extraction."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        thinking: bool = False,
        quantization: str = "none",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ) -> None:
        """Initialize VLM annotator with model and processor.

        Args:
            model_name: HuggingFace model identifier.
            device_map: Device placement strategy.
            torch_dtype: Torch data type string (e.g. "bfloat16", "float16").
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            thinking: Whether to enable Qwen thinking mode.
            quantization: Quantization mode ("none", "4bit", "8bit").
            bnb_4bit_quant_type: 4-bit quantization type (e.g. "nf4", "fp4").
            bnb_4bit_use_double_quant: Whether to use double quantization.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.thinking = thinking

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        model_kwargs: dict = {
            "dtype": resolved_dtype,
            "device_map": device_map,
            "attn_implementation": "sdpa",
        }

        if quantization == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=resolved_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        logger.info("Loading model: %s (dtype=%s, quantization=%s)", model_name, torch_dtype, quantization)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **model_kwargs,
        )
        logger.info("Model loaded successfully")

    def annotate_clip(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        video_id: str,
        categories: list[str],
        actions: list[str],
        fps: float = 1.0,
    ) -> VLMAnnotation:
        """Annotate a clip of frames.

        Args:
            frames: List of BGR numpy arrays (H, W, 3).
            frame_indices: Corresponding frame indices in the original video.
            video_id: Identifier for the source video.
            categories: Allowed object categories.
            actions: Allowed action labels.

        Returns:
            VLMAnnotation with detected objects and events.
        """
        # Overlay frame numbers and convert to PIL
        pil_images = []
        for frame, idx in zip(frames, frame_indices):
            overlaid = self._overlay_frame_number(frame, idx)
            # Convert BGR -> RGB -> PIL
            rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        # Build prompts
        system_prompt, user_prompt = build_prompt(
            categories=categories,
            actions=actions,
            n_frames=len(frames),
            fps=fps,
        )

        # Build messages with images
        content: list[dict] = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Tokenize
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        inputs = self.processor(
            text=[text],
            images=pil_images,
            return_tensors="pt",
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        # Decode generated tokens only
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        raw_output = self.processor.decode(generated_ids, skip_special_tokens=True)

        logger.debug("Raw VLM output for %s: %s", video_id, raw_output[:200])

        # Parse and validate
        parsed = self._parse_output(raw_output)
        if parsed is None:
            logger.warning(
                "Failed to parse VLM output for %s, returning empty annotation",
                video_id,
            )
            return VLMAnnotation()

        try:
            annotation = VLMAnnotation.model_validate(parsed)
        except Exception as e:
            logger.warning(
                "VLM output validation failed for %s: %s. Returning empty annotation.",
                video_id,
                e,
            )
            annotation = VLMAnnotation()

        return annotation

    def annotate_video(
        self,
        video_path: str,
        fps: float = 1.0,
        clip_length: int = 16,
        clip_stride: int = 8,
        categories: list[str] | None = None,
        actions: list[str] | None = None,
    ) -> list[VLMAnnotation]:
        """Annotate an entire video using sliding window over sampled frames.

        Args:
            video_path: Path to the video file.
            fps: Target sampling FPS.
            clip_length: Number of frames per clip window.
            clip_stride: Stride between clip windows.
            categories: Allowed categories. Defaults to empty list.
            actions: Allowed actions. Defaults to empty list.

        Returns:
            List of VLMAnnotation, one per clip window.
        """
        categories = categories or []
        actions = actions or []
        video_id = Path(video_path).stem

        # Sample all frames
        sampler = FrameSampler(target_fps=fps)
        sampled_frames = sampler.sample(video_path)

        if not sampled_frames:
            logger.warning("No frames sampled from %s", video_path)
            return []

        all_images = [sf.image for sf in sampled_frames]
        all_indices = [sf.frame_index for sf in sampled_frames]

        # Sliding window
        annotations: list[VLMAnnotation] = []
        n_frames = len(all_images)

        for start in range(0, n_frames, clip_stride):
            end = min(start + clip_length, n_frames)
            clip_images = all_images[start:end]
            clip_indices = all_indices[start:end]

            if not clip_images:
                break

            logger.info(
                "Annotating %s clip [%d:%d] (%d frames)",
                video_id,
                start,
                end,
                len(clip_images),
            )

            annotation = self.annotate_clip(
                frames=clip_images,
                frame_indices=clip_indices,
                video_id=video_id,
                categories=categories,
                actions=actions,
                fps=fps,
            )
            annotations.append(annotation)

            # Stop if we've reached the end
            if end >= n_frames:
                break

        return annotations

    @staticmethod
    def _overlay_frame_number(image: np.ndarray, frame_number: int) -> np.ndarray:
        """Draw frame number on top-left corner of the image.

        Args:
            image: BGR numpy array (H, W, 3).
            frame_number: Frame number to overlay.

        Returns:
            Copy of image with frame number drawn.
        """
        img = image.copy()
        text = f"F{frame_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)  # Green in BGR
        bg_color = (0, 0, 0)  # Black background

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Draw background rectangle
        cv2.rectangle(img, (0, 0), (text_w + 10, text_h + baseline + 10), bg_color, -1)
        # Draw text
        cv2.putText(
            img,
            text,
            (5, text_h + 5),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return img

    @staticmethod
    def _parse_output(raw_output: str) -> dict | None:
        """Parse model output, handling </think> blocks and ```json blocks.

        Args:
            raw_output: Raw text output from the model.

        Returns:
            Parsed dict or None if parsing fails.
        """
        text = raw_output.strip()

        # Remove </think> block (Qwen3.5 thinking output)
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()

        # Handle ```json ... ``` blocks
        if "```json" in text:
            text = text.split("```json", 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0]
            text = text.strip()
        elif text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
