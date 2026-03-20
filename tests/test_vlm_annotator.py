"""Tests for VLMAnnotator backend selection logic."""

from __future__ import annotations

from unittest import mock

import pytest


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
