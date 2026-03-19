"""Pydantic schemas for VLM output JSON validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VLMObject(BaseModel):
    """A single object detected by the VLM."""

    obj_id: str = Field(..., pattern=r"^[a-z_]+_\d+$", description="Object ID in category_NN format")
    category: str
    first_seen_frame: int = Field(..., ge=0)
    attributes: list[str] = Field(default_factory=list)


class VLMEvent(BaseModel):
    """A single event annotated by the VLM."""

    event_id: str = Field(..., pattern=r"^evt_\d+$")
    frame: int = Field(..., ge=0)
    action: str
    agent: str
    target: str
    source: str | None = None
    destination: str | None = None


class VLMAnnotation(BaseModel):
    """Complete VLM annotation for a video clip."""

    objects: list[VLMObject] = Field(default_factory=list)
    events: list[VLMEvent] = Field(default_factory=list)
