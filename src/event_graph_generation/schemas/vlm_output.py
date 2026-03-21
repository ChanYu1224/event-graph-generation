"""Pydantic schemas for VLM output JSON validation."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class VLMObject(BaseModel):
    """A single object detected by the VLM."""

    obj_id: str = Field(..., pattern=r"^[a-z_]+_\d+$", description="Object ID in category_NN format")
    category: str
    first_seen_frame: int = Field(..., ge=0)
    attributes: dict[str, str | None] = Field(default_factory=dict)

    @field_validator("attributes", mode="before")
    @classmethod
    def _coerce_attributes(cls, v: object) -> dict[str, str | None]:
        """Accept legacy list[str] format and convert to empty dict.

        Args:
            v: Raw value for the attributes field.

        Returns:
            Dict of attribute axes to values.
        """
        if isinstance(v, list):
            logger.debug("Coercing legacy list attributes %s to empty dict", v)
            return {}
        return v  # type: ignore[return-value]


class VLMEvent(BaseModel):
    """A single event annotated by the VLM."""

    event_id: str = Field(..., pattern=r"^evt_\d+$")
    frame: int = Field(..., ge=0)
    action: str
    agent: str
    target: str | None = None
    source: str | None = None
    destination: str | None = None


class VLMAnnotation(BaseModel):
    """Complete VLM annotation for a video clip."""

    objects: list[VLMObject] = Field(default_factory=list)
    events: list[VLMEvent] = Field(default_factory=list)
