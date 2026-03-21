"""Prompt templates for VLM-based event annotation."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert video event annotator. Given a sequence of video frames, \
your task is to produce a structured JSON annotation.

Follow these two steps exactly:

**Step 1 — Object Listing**
List every distinct object that appears across the frames. For each object provide:
- obj_id: a unique identifier in "category_NN" format (e.g. "cup_01", "person_02").
  The category part must come from the allowed category list below.
- category: the object category (must be one of the allowed categories).
- first_seen_frame: the frame number where the object first appears.
- attributes: a dict of attribute axes. Each axis key maps to a single allowed value \
or null if not applicable. Only use the allowed values listed below for each axis.

Attribute axes and allowed values:
{attribute_vocab_text}

**Step 2 — Event Listing**
List every event (action) that occurs. For each event provide:
- event_id: a unique identifier in "evt_NNN" format (e.g. "evt_001", "evt_002").
- frame: the frame number where the event occurs.
- action: the action label (must be one of the allowed actions below).
- agent: the obj_id of the entity performing the action.
- target: the obj_id of the entity being acted upon.
- source (optional): the obj_id of the origin location/container, if applicable.
- destination (optional): the obj_id of the destination location/container, if applicable.

Allowed categories: {category_list}
Allowed actions: {action_list}

Output ONLY valid JSON matching the schema shown in the user message. \
Do not include any text before or after the JSON."""


USER_PROMPT_TEMPLATE = """\
The following {n_frames} frames were sampled at {fps} FPS. \
Each frame has its frame number overlaid in the top-left corner.

Analyze the frames and produce a JSON annotation with the following schema:

{json_schema_example}

Output ONLY the JSON object."""


_EVENTS_EXAMPLE = """\
    {
      "event_id": "evt_001",
      "frame": 5,
      "action": "pick_up",
      "agent": "person_01",
      "target": "cup_01",
      "source": "table_01",
      "destination": null
    },
    {
      "event_id": "evt_002",
      "frame": 10,
      "action": "place_on",
      "agent": "person_01",
      "target": "cup_01",
      "source": null,
      "destination": "shelf_01"
    }"""


def _build_attributes_example(axes: list[str]) -> str:
    """Build example attributes dicts for person and cup objects.

    Args:
        axes: Ordered list of attribute axis names.

    Returns:
        Formatted JSON snippet for two example objects with attributes.
    """
    # Example values: person has pose/orientation, cup has color/material/position/size/state
    person_values: dict[str, str | None] = {
        "pose": "standing",
        "orientation": "upright",
    }
    cup_values: dict[str, str | None] = {
        "color": "red",
        "material": "ceramic",
        "position": "on_desk",
        "size": "small",
        "state": "filled",
    }

    def _fmt_attrs(values: dict[str, str | None]) -> str:
        lines = []
        for axis in axes:
            val = values.get(axis)
            val_str = f'"{val}"' if val else "null"
            lines.append(f'          "{axis}": {val_str}')
        return ",\n".join(lines)

    person_attrs = _fmt_attrs(person_values)
    cup_attrs = _fmt_attrs(cup_values)

    return f"""\
    {{
      "obj_id": "person_01",
      "category": "person",
      "first_seen_frame": 0,
      "attributes": {{
{person_attrs}
      }}
    }},
    {{
      "obj_id": "cup_01",
      "category": "cup",
      "first_seen_frame": 3,
      "attributes": {{
{cup_attrs}
      }}
    }}"""


def _build_json_schema_example(attribute_vocab: dict[str, list[str]]) -> str:
    """Build the full JSON schema example with dynamic attributes.

    Args:
        attribute_vocab: Mapping of axis name to list of allowed values.

    Returns:
        Complete JSON schema example string.
    """
    axes = list(attribute_vocab.keys()) if attribute_vocab else []
    objects_part = _build_attributes_example(axes)
    return f"""\
{{
  "objects": [
{objects_part}
  ],
  "events": [
{_EVENTS_EXAMPLE}
  ]
}}"""


def _format_attribute_vocab(attribute_vocab: dict[str, list[str]]) -> str:
    """Format the attribute vocabulary for inclusion in the system prompt.

    Args:
        attribute_vocab: Mapping of axis name to list of allowed values.

    Returns:
        Formatted string listing each axis and its values.
    """
    lines = []
    for axis, values in attribute_vocab.items():
        lines.append(f"  - {axis}: {', '.join(values)}")
    return "\n".join(lines)


def build_prompt(
    categories: list[str],
    actions: list[str],
    n_frames: int,
    fps: float,
    attribute_vocab: dict[str, list[str]] | None = None,
) -> tuple[str, str]:
    """Build the system and user prompts with filled-in templates.

    Args:
        categories: List of allowed object category names.
        actions: List of allowed action names.
        n_frames: Number of frames in the clip.
        fps: Frames per second at which the clip was sampled.
        attribute_vocab: Mapping of axis name to list of allowed values.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    vocab = attribute_vocab or {}
    system_prompt = SYSTEM_PROMPT.format(
        category_list=", ".join(categories),
        action_list=", ".join(actions),
        attribute_vocab_text=_format_attribute_vocab(vocab),
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        n_frames=n_frames,
        fps=fps,
        json_schema_example=_build_json_schema_example(vocab),
    )
    return system_prompt, user_prompt
