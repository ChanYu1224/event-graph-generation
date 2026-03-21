"""Tests for VLMAnnotator backend selection logic."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from PIL import Image


class TestBackendValidation:
    """Test backend parameter validation in VLMAnnotator.__init__."""

    def test_invalid_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown backend.*'typo'"):
            from event_graph_generation.annotation.vlm_annotator import VLMAnnotator

            VLMAnnotator(backend="typo")

    def test_vllm_backend_raises_import_error_when_unavailable(self):
        with mock.patch(
            "event_graph_generation.annotation.vlm_annotator._VLLM_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="vllm is required"):
                from event_graph_generation.annotation.vlm_annotator import (
                    VLMAnnotator,
                )

                VLMAnnotator(backend="vllm")


class TestVLLMServerBackend:
    """Test vllm-server backend initialization and helpers."""

    def test_vllm_server_backend_raises_import_error_when_unavailable(self):
        with mock.patch(
            "event_graph_generation.annotation.vlm_annotator._OPENAI_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="openai is required"):
                from event_graph_generation.annotation.vlm_annotator import (
                    VLMAnnotator,
                )

                VLMAnnotator(backend="vllm-server")

    def test_vllm_server_backend_valid(self):
        mock_client = mock.MagicMock()
        with mock.patch(
            "event_graph_generation.annotation.vlm_annotator._OPENAI_AVAILABLE", True
        ), mock.patch(
            "event_graph_generation.annotation.vlm_annotator.AsyncOpenAI",
            return_value=mock_client,
        ):
            from event_graph_generation.annotation.vlm_annotator import VLMAnnotator

            annotator = VLMAnnotator(backend="vllm-server")
            assert annotator.backend == "vllm-server"
            assert annotator.async_client is mock_client
            assert annotator.max_concurrent_requests == 8

    def test_encode_image_base64_returns_string(self):
        from event_graph_generation.annotation.vlm_annotator import VLMAnnotator

        img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
        result = VLMAnnotator._encode_image_base64(img)
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it's valid base64
        import base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_build_openai_messages_format(self):
        mock_client = mock.MagicMock()
        with mock.patch(
            "event_graph_generation.annotation.vlm_annotator._OPENAI_AVAILABLE", True
        ), mock.patch(
            "event_graph_generation.annotation.vlm_annotator.AsyncOpenAI",
            return_value=mock_client,
        ):
            from event_graph_generation.annotation.vlm_annotator import VLMAnnotator

            annotator = VLMAnnotator(backend="vllm-server")

        images = [
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)),
            Image.fromarray(np.ones((16, 16, 3), dtype=np.uint8)),
        ]
        messages = annotator._build_openai_messages(
            system_prompt="System",
            user_prompt="User prompt",
            pil_images=images,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System"
        assert messages[1]["role"] == "user"
        # User content: 2 images + 1 text
        user_content = messages[1]["content"]
        assert len(user_content) == 3
        assert user_content[0]["type"] == "image_url"
        assert user_content[1]["type"] == "image_url"
        assert user_content[2]["type"] == "text"
        assert user_content[2]["text"] == "User prompt"
        # Check image_url format
        assert user_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
