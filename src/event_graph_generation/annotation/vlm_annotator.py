"""VLM annotator for local Qwen 3.5 inference on video clips."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

try:
    from openai import AsyncOpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

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
        backend: str = "transformers",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
        max_num_seqs: int = 5,
        limit_mm_per_prompt: int = 16,
        api_base: str = "http://localhost:8000/v1",
        max_concurrent_requests: int = 8,
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
            backend: Inference backend ("transformers", "vllm", or "vllm-server").
            tensor_parallel_size: Number of GPUs for VLLM tensor parallelism.
            gpu_memory_utilization: GPU memory fraction for VLLM.
            max_model_len: Maximum model context length for VLLM.
            max_num_seqs: Maximum concurrent sequences for VLLM.
            limit_mm_per_prompt: Maximum images per prompt for VLLM.
            api_base: Base URL for VLLM server API.
            max_concurrent_requests: Maximum concurrent async requests for vllm-server.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.backend = backend

        if backend not in ("transformers", "vllm", "vllm-server"):
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                "Must be 'transformers', 'vllm', or 'vllm-server'."
            )

        if backend == "vllm-server":
            if not _OPENAI_AVAILABLE:
                raise ImportError(
                    "openai is required for the 'vllm-server' backend. "
                    "Install with: pip install openai"
                )
            self.async_client = AsyncOpenAI(base_url=api_base, api_key="dummy")
            self.max_concurrent_requests = max_concurrent_requests
            logger.info("Using VLLM server at %s", api_base)
            return

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        if backend == "vllm":
            if not _VLLM_AVAILABLE:
                raise ImportError(
                    "vllm is required for the 'vllm' backend. "
                    "Install with: pip install vllm"
                )
            logger.info(
                "Loading model with VLLM: %s (dtype=%s, tp=%d)",
                model_name, torch_dtype, tensor_parallel_size,
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                limit_mm_per_prompt={"image": limit_mm_per_prompt},
                dtype=resolved_dtype,
            )
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            logger.info("VLLM model loaded successfully")
        else:
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
        attribute_vocab: dict[str, list[str]] | None = None,
    ) -> VLMAnnotation:
        """Annotate a clip of frames.

        Args:
            frames: List of BGR numpy arrays (H, W, 3).
            frame_indices: Corresponding frame indices in the original video.
            video_id: Identifier for the source video.
            categories: Allowed object categories.
            actions: Allowed action labels.
            fps: Frames per second at which the clip was sampled.
            attribute_vocab: Mapping of axis name to list of allowed values.

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
            attribute_vocab=attribute_vocab,
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

        # Generate text via selected backend
        if self.backend == "vllm":
            raw_output = self._generate_vllm(messages, pil_images)
        else:
            raw_output = self._generate_transformers(messages, pil_images)

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

    def _generate_transformers(
        self, messages: list[dict], pil_images: list[Image.Image],
    ) -> str:
        """Generate text using the transformers backend.

        Args:
            messages: Chat messages for the model.
            pil_images: List of PIL images for the prompt.

        Returns:
            Raw decoded text from the model.
        """
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

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)

    def _generate_vllm(
        self, messages: list[dict], pil_images: list[Image.Image],
    ) -> str:
        """Generate text using the VLLM backend.

        Args:
            messages: Chat messages for the model.
            pil_images: List of PIL images for the prompt.

        Returns:
            Raw decoded text from the model.
        """
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        outputs = self.llm.generate(
            {"prompt": text, "multi_modal_data": {"image": pil_images}},
            sampling_params=self.sampling_params,
        )
        return outputs[0].outputs[0].text

    @staticmethod
    def _encode_image_base64(image: Image.Image) -> str:
        """Encode PIL image to base64 JPEG string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64-encoded JPEG string.
        """
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_openai_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        pil_images: list[Image.Image],
    ) -> list[dict]:
        """Build OpenAI API format messages with base64-encoded images.

        Args:
            system_prompt: System prompt text.
            user_prompt: User prompt text.
            pil_images: List of PIL images to include.

        Returns:
            List of message dicts in OpenAI chat format.
        """
        content: list[dict] = []
        for img in pil_images:
            b64 = self._encode_image_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    async def _annotate_clips_server_batch(
        self,
        clip_data: list[tuple[list[np.ndarray], list[int]]],
        video_id: str,
        categories: list[str],
        actions: list[str],
        fps: float,
        attribute_vocab: dict[str, list[str]] | None = None,
    ) -> list[VLMAnnotation]:
        """Batch-annotate clips via VLLM server with concurrent async requests.

        Args:
            clip_data: List of (frames, frame_indices) tuples per clip.
            video_id: Identifier for the source video.
            categories: Allowed object categories.
            actions: Allowed action labels.
            fps: Target FPS used for prompt generation.
            attribute_vocab: Mapping of axis name to list of allowed values.

        Returns:
            List of VLMAnnotation, one per clip.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def _process_clip(
            clip_idx: int,
            frames: list[np.ndarray],
            frame_indices: list[int],
        ) -> VLMAnnotation:
            # Build per-clip prompt (n_frames may differ for the last clip)
            system_prompt, user_prompt = build_prompt(
                categories=categories,
                actions=actions,
                n_frames=len(frames),
                fps=fps,
                attribute_vocab=attribute_vocab,
            )

            # Overlay frame numbers and convert to PIL
            pil_images = []
            for frame, idx in zip(frames, frame_indices):
                overlaid = self._overlay_frame_number(frame, idx)
                rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))

            messages = self._build_openai_messages(
                system_prompt, user_prompt, pil_images,
            )

            async with semaphore:
                logger.info(
                    "Sending %s clip %d (%d frames) to VLLM server",
                    video_id, clip_idx, len(frames),
                )
                extra_body = {}
                if not self.thinking:
                    extra_body["chat_template_kwargs"] = {
                        "enable_thinking": False,
                    }
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    extra_body=extra_body or None,
                )

            raw_output = response.choices[0].message.content
            logger.debug("Raw VLLM server output for %s clip %d: %s",
                         video_id, clip_idx, raw_output[:200])

            parsed = self._parse_output(raw_output)
            if parsed is None:
                logger.warning(
                    "Failed to parse VLLM server output for %s clip %d",
                    video_id, clip_idx,
                )
                return VLMAnnotation()

            try:
                return VLMAnnotation.model_validate(parsed)
            except Exception as e:
                logger.warning(
                    "VLLM server output validation failed for %s clip %d: %s",
                    video_id, clip_idx, e,
                )
                return VLMAnnotation()

        tasks = [
            _process_clip(i, frames, indices)
            for i, (frames, indices) in enumerate(clip_data)
        ]
        return list(await asyncio.gather(*tasks))

    def annotate_video(
        self,
        video_path: str,
        fps: float = 1.0,
        clip_length: int = 16,
        clip_stride: int = 8,
        categories: list[str] | None = None,
        actions: list[str] | None = None,
        attribute_vocab: dict[str, list[str]] | None = None,
        motion_filter_enabled: bool = False,
        motion_threshold: float = 3.0,
    ) -> list[VLMAnnotation]:
        """Annotate an entire video using sliding window over sampled frames.

        Args:
            video_path: Path to the video file.
            fps: Target sampling FPS.
            clip_length: Number of frames per clip window.
            clip_stride: Stride between clip windows.
            categories: Allowed categories. Defaults to empty list.
            actions: Allowed actions. Defaults to empty list.
            attribute_vocab: Mapping of axis name to list of allowed values.
            motion_filter_enabled: Whether to skip static clips via motion detection.
            motion_threshold: Minimum motion score to keep a clip.

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

        # Build clip windows (common to all backends)
        clip_data: list[tuple[list[np.ndarray], list[int]]] = []
        n_frames = len(all_images)

        for start in range(0, n_frames, clip_stride):
            end = min(start + clip_length, n_frames)
            clip_images = all_images[start:end]
            clip_indices = all_indices[start:end]
            if not clip_images:
                break
            clip_data.append((clip_images, clip_indices))
            if end >= n_frames:
                break

        # Motion filter: identify which clips are static
        if motion_filter_enabled:
            from event_graph_generation.utils.motion import compute_clip_motion_score

            static_mask = [
                compute_clip_motion_score(images) < motion_threshold
                for images, _indices in clip_data
            ]
            skipped = sum(static_mask)
            if skipped > 0:
                logger.info(
                    "Motion filter: skipped %d/%d static clips for %s",
                    skipped, len(clip_data), video_id,
                )
        else:
            static_mask = [False] * len(clip_data)

        # Separate active clips for backend processing
        active_clips = [
            (i, images, indices)
            for i, ((images, indices), is_static) in enumerate(
                zip(clip_data, static_mask)
            )
            if not is_static
        ]

        if not active_clips:
            logger.info("All clips filtered out for %s", video_id)
            return [VLMAnnotation() for _ in clip_data]

        # Backend dispatch on active clips only
        if self.backend == "vllm-server":
            active_data = [(images, indices) for _, images, indices in active_clips]
            active_results = asyncio.run(self._annotate_clips_server_batch(
                active_data, video_id, categories, actions, fps,
                attribute_vocab=attribute_vocab,
            ))
        else:
            # transformers / vllm: sequential processing
            active_results = []
            for _, clip_images, clip_indices in active_clips:
                logger.info(
                    "Annotating %s clip (%d frames)",
                    video_id,
                    len(clip_images),
                )
                annotation = self.annotate_clip(
                    frames=clip_images,
                    frame_indices=clip_indices,
                    video_id=video_id,
                    categories=categories,
                    actions=actions,
                    fps=fps,
                    attribute_vocab=attribute_vocab,
                )
                active_results.append(annotation)

        # Reconstruct full list: empty annotations for static clips
        annotations: list[VLMAnnotation] = [VLMAnnotation()] * len(clip_data)
        for (orig_idx, _, _), result in zip(active_clips, active_results):
            annotations[orig_idx] = result

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
