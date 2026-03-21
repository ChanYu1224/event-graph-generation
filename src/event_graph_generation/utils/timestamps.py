"""Timestamp utilities for video annotation enrichment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta


_VIDEO_ID_PATTERN = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_")


@dataclass
class ClipTimestamp:
    """Timestamp metadata for a single clip.

    Args:
        clip_index: Zero-based clip index.
        frame_indices: List of source-video frame indices in this clip.
        start_sec: Start offset in seconds from video beginning.
        end_sec: End offset in seconds from video beginning.
        start_time: Absolute start time (ISO 8601).
        end_time: Absolute end time (ISO 8601).
    """

    clip_index: int
    frame_indices: list[int]
    start_sec: float
    end_sec: float
    start_time: str
    end_time: str


def parse_video_start_time(video_id: str) -> datetime:
    """Parse recording start time from a video ID.

    Args:
        video_id: Video identifier like ``20260316_130406_tp00001``.

    Returns:
        Parsed datetime.

    Raises:
        ValueError: If the video ID does not match the expected format.
    """
    m = _VIDEO_ID_PATTERN.match(video_id)
    if m is None:
        raise ValueError(
            f"Cannot parse start time from video_id: {video_id!r}. "
            "Expected format: YYYYMMDD_HHMMSS_..."
        )
    year, month, day, hour, minute, second = (int(g) for g in m.groups())
    return datetime(year, month, day, hour, minute, second)


def compute_clip_timestamps(
    source_fps: float,
    target_fps: float,
    total_frames: int,
    clip_length: int,
    clip_stride: int,
    video_start_time: datetime,
) -> list[ClipTimestamp]:
    """Compute frame indices and absolute timestamps for every clip.

    Reproduces the sliding-window logic used by ``FrameSampler`` /
    ``VLMAnnotator``:  each clip consists of *clip_length* frames sampled
    at *target_fps* from the source video whose native rate is *source_fps*.

    Args:
        source_fps: Native FPS of the source video.
        target_fps: Sampling FPS (typically 1.0).
        total_frames: Total number of frames in the source video.
        clip_length: Number of sampled frames per clip.
        clip_stride: Stride in sampled-frame space between clips.
        video_start_time: Absolute start time of the recording.

    Returns:
        List of :class:`ClipTimestamp`, one per clip.
    """
    frame_interval = max(1, round(source_fps / target_fps))

    # All sampled frame indices in source-video space
    sampled_indices = list(range(0, total_frames, frame_interval))
    num_sampled = len(sampled_indices)

    clips: list[ClipTimestamp] = []
    clip_index = 0
    start = 0
    while start + clip_length <= num_sampled:
        indices = sampled_indices[start : start + clip_length]
        start_sec = indices[0] / source_fps
        end_sec = indices[-1] / source_fps
        clips.append(
            ClipTimestamp(
                clip_index=clip_index,
                frame_indices=indices,
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                start_time=offset_to_iso(video_start_time, start_sec),
                end_time=offset_to_iso(video_start_time, end_sec),
            )
        )
        clip_index += 1
        start += clip_stride

    return clips


def frame_to_absolute_time(
    frame_index: int,
    source_fps: float,
    video_start_time: datetime,
) -> datetime:
    """Convert a single source-video frame index to an absolute datetime.

    Args:
        frame_index: Frame index in the source video.
        source_fps: Native FPS of the source video.
        video_start_time: Absolute start time of the recording.

    Returns:
        Absolute datetime for the given frame.
    """
    offset_sec = frame_index / source_fps
    return video_start_time + timedelta(seconds=offset_sec)


def offset_to_iso(base: datetime, offset_sec: float) -> str:
    """Return ISO 8601 string for *base* + *offset_sec*.

    Args:
        base: Base datetime.
        offset_sec: Offset in seconds.

    Returns:
        ISO 8601 formatted string (without microseconds).
    """
    dt = base + timedelta(seconds=offset_sec)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def clip_status(clip: dict) -> str:
    """Determine clip status from its content.

    Args:
        clip: Clip dict with ``objects`` and ``events`` keys.

    Returns:
        ``"motion_filtered"`` if both lists are empty, else ``"annotated"``.
    """
    if clip.get("objects", []) == [] and clip.get("events", []) == []:
        return "motion_filtered"
    return "annotated"


def find_motion_filtered_ranges(
    clips: list[dict],
    clip_timestamps: list[ClipTimestamp],
) -> list[dict[str, str]]:
    """Identify contiguous ranges of motion-filtered clips.

    Args:
        clips: Clip dicts from the annotation JSON.
        clip_timestamps: Corresponding :class:`ClipTimestamp` objects.

    Returns:
        List of dicts with ``start`` and ``end`` ISO 8601 strings.
    """
    ranges: list[dict[str, str]] = []
    i = 0
    n = min(len(clips), len(clip_timestamps))
    while i < n:
        if clip_status(clips[i]) == "motion_filtered":
            start_idx = i
            while i < n and clip_status(clips[i]) == "motion_filtered":
                i += 1
            end_idx = i - 1
            ranges.append({
                "start": clip_timestamps[start_idx].start_time,
                "end": clip_timestamps[end_idx].end_time,
            })
        else:
            i += 1
    return ranges


@dataclass
class ClipEnrichment:
    """Result of enriching clips with timestamp metadata.

    Args:
        clips: Clip dicts with ``clip_metadata`` added.
        annotated_clips: Number of annotated clips.
        motion_filtered_clips: Number of motion-filtered clips.
        clips_with_events: Number of clips that contain events.
        motion_filtered_ranges: Contiguous motion-filtered time ranges.
    """

    clips: list[dict]
    annotated_clips: int
    motion_filtered_clips: int
    clips_with_events: int
    motion_filtered_ranges: list[dict[str, str]]


def enrich_clips(
    clips: list[dict],
    clip_timestamps: list[ClipTimestamp],
) -> ClipEnrichment:
    """Add ``clip_metadata`` to each clip and compute coverage statistics.

    Args:
        clips: List of clip dicts, each with ``objects`` and ``events`` keys.
        clip_timestamps: :class:`ClipTimestamp` objects from
            :func:`compute_clip_timestamps`.

    Returns:
        :class:`ClipEnrichment` with enriched clips and summary statistics.
    """
    enriched: list[dict] = []
    annotated_count = 0
    motion_filtered_count = 0
    events_count = 0

    for i, clip in enumerate(clips):
        status = clip_status(clip)
        if status == "annotated":
            annotated_count += 1
        else:
            motion_filtered_count += 1
        if clip.get("events", []):
            events_count += 1

        if i < len(clip_timestamps):
            ct = clip_timestamps[i]
            meta = {
                "clip_index": ct.clip_index,
                "frame_indices": ct.frame_indices,
                "start_time": ct.start_time,
                "end_time": ct.end_time,
                "start_offset_sec": ct.start_sec,
                "end_offset_sec": ct.end_sec,
                "status": status,
            }
        else:
            meta = {
                "clip_index": i,
                "frame_indices": [],
                "start_time": "",
                "end_time": "",
                "start_offset_sec": 0.0,
                "end_offset_sec": 0.0,
                "status": status,
            }

        enriched.append({**clip, "clip_metadata": meta})

    mf_ranges = find_motion_filtered_ranges(clips, clip_timestamps)

    return ClipEnrichment(
        clips=enriched,
        annotated_clips=annotated_count,
        motion_filtered_clips=motion_filtered_count,
        clips_with_events=events_count,
        motion_filtered_ranges=mf_ranges,
    )
