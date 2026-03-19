"""VLM annotation module for event graph generation."""

from event_graph_generation.annotation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from event_graph_generation.annotation.validator import AnnotationValidator
from event_graph_generation.annotation.vlm_annotator import VLMAnnotator

__all__ = [
    "VLMAnnotator",
    "AnnotationValidator",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
]
