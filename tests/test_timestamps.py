"""Tests for timestamp utilities."""

from __future__ import annotations

from datetime import datetime

import pytest

from event_graph_generation.utils.timestamps import (
    ClipTimestamp,
    clip_status,
    compute_clip_timestamps,
    enrich_clips,
    find_motion_filtered_ranges,
    frame_to_absolute_time,
    offset_to_iso,
    parse_video_start_time,
)


class TestParseVideoStartTime:
    def test_standard_id(self) -> None:
        dt = parse_video_start_time("20260316_130406_tp00001")
        assert dt == datetime(2026, 3, 16, 13, 4, 6)

    def test_2025_data(self) -> None:
        dt = parse_video_start_time("20251209_152712_tp00000")
        assert dt == datetime(2025, 12, 9, 15, 27, 12)

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse start time"):
            parse_video_start_time("bad_video_name")

    def test_incomplete_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse start time"):
            parse_video_start_time("2026031_130406_tp00001")


class TestComputeClipTimestamps:
    """Tests with a small synthetic video."""

    # source_fps=20, target_fps=1 → frame_interval=20
    # 100 frames → sampled indices [0, 20, 40, 60, 80] (5 sampled frames)
    # clip_length=3, clip_stride=2:
    #   clip 0: sampled[0:3] = [0, 20, 40]
    #   clip 1: sampled[2:5] = [40, 60, 80]
    #   clip 2: start=4, 4+3=7 > 5 → stop

    def _make_clips(self) -> list[ClipTimestamp]:
        return compute_clip_timestamps(
            source_fps=20.0,
            target_fps=1.0,
            total_frames=100,
            clip_length=3,
            clip_stride=2,
            video_start_time=datetime(2026, 1, 1, 12, 0, 0),
        )

    def test_correct_clip_count(self) -> None:
        clips = self._make_clips()
        assert len(clips) == 2

    def test_frame_indices(self) -> None:
        clips = self._make_clips()
        assert clips[0].frame_indices == [0, 20, 40]
        assert clips[1].frame_indices == [40, 60, 80]

    def test_clip_index_sequential(self) -> None:
        clips = self._make_clips()
        assert [c.clip_index for c in clips] == [0, 1]

    def test_start_before_end(self) -> None:
        clips = self._make_clips()
        for c in clips:
            assert c.start_sec <= c.end_sec

    def test_start_time_format(self) -> None:
        clips = self._make_clips()
        # First clip starts at frame 0 → offset 0s → base time
        assert clips[0].start_time == "2026-01-01T12:00:00"
        # end at frame 40 → 2.0s
        assert clips[0].end_time == "2026-01-01T12:00:02"

    def test_second_clip_times(self) -> None:
        clips = self._make_clips()
        # clip 1: frames [40, 60, 80] → start 2.0s, end 4.0s
        assert clips[1].start_sec == pytest.approx(2.0)
        assert clips[1].end_sec == pytest.approx(4.0)

    def test_larger_video(self) -> None:
        """Realistic parameters: 54200 frames, fps=20, clip_length=16, stride=8."""
        clips = compute_clip_timestamps(
            source_fps=20.0,
            target_fps=1.0,
            total_frames=54200,
            clip_length=16,
            clip_stride=8,
            video_start_time=datetime(2026, 3, 16, 13, 4, 6),
        )
        # 54200 / 20 = 2710 sampled frames
        # (2710 - 16) / 8 + 1 = 337.25 → 337 clips
        # But with floor: start goes 0,8,...,2704 → (2710-16)//8 +1
        num_sampled = 54200 // 20  # 2710
        expected = (num_sampled - 16) // 8 + 1
        assert len(clips) == expected
        # First clip starts at video start
        assert clips[0].start_time == "2026-03-16T13:04:06"


class TestFrameToAbsoluteTime:
    def test_frame_zero(self) -> None:
        base = datetime(2026, 3, 16, 13, 4, 6)
        assert frame_to_absolute_time(0, 20.0, base) == base

    def test_frame_600(self) -> None:
        """Frame 600 at 20fps = 30 seconds offset."""
        base = datetime(2026, 3, 16, 13, 4, 6)
        result = frame_to_absolute_time(600, 20.0, base)
        assert result == datetime(2026, 3, 16, 13, 4, 36)

    def test_fractional_second(self) -> None:
        base = datetime(2026, 1, 1, 0, 0, 0)
        result = frame_to_absolute_time(10, 20.0, base)
        # 10 / 20 = 0.5s
        assert result == datetime(2026, 1, 1, 0, 0, 0, 500000)


class TestOffsetToIso:
    def test_zero_offset(self) -> None:
        base = datetime(2026, 3, 16, 13, 4, 6)
        assert offset_to_iso(base, 0.0) == "2026-03-16T13:04:06"

    def test_positive_offset(self) -> None:
        base = datetime(2026, 3, 16, 13, 4, 6)
        assert offset_to_iso(base, 90.0) == "2026-03-16T13:05:36"

    def test_truncates_microseconds(self) -> None:
        base = datetime(2026, 1, 1, 0, 0, 0)
        # 0.5s → no microseconds in output
        assert offset_to_iso(base, 0.5) == "2026-01-01T00:00:00"


class TestClipStatus:
    def test_annotated_with_objects_and_events(self) -> None:
        clip = {"objects": [{"obj_id": 1}], "events": [{"event_id": 1}]}
        assert clip_status(clip) == "annotated"

    def test_annotated_with_objects_only(self) -> None:
        clip = {"objects": [{"obj_id": 1}], "events": []}
        assert clip_status(clip) == "annotated"

    def test_motion_filtered_empty(self) -> None:
        clip = {"objects": [], "events": []}
        assert clip_status(clip) == "motion_filtered"

    def test_motion_filtered_missing_keys(self) -> None:
        clip: dict = {}
        assert clip_status(clip) == "motion_filtered"


class TestFindMotionFilteredRanges:
    def _make_timestamps(self) -> list[ClipTimestamp]:
        """Create 5 clips for testing."""
        return compute_clip_timestamps(
            source_fps=20.0,
            target_fps=1.0,
            total_frames=200,
            clip_length=3,
            clip_stride=1,
            video_start_time=datetime(2026, 1, 1, 12, 0, 0),
        )

    def test_no_filtered_clips(self) -> None:
        cts = self._make_timestamps()
        clips = [{"objects": [{"id": 1}], "events": []} for _ in cts]
        assert find_motion_filtered_ranges(clips, cts) == []

    def test_single_filtered_clip(self) -> None:
        cts = self._make_timestamps()
        clips = [{"objects": [{"id": 1}], "events": []} for _ in cts]
        clips[2] = {"objects": [], "events": []}
        ranges = find_motion_filtered_ranges(clips, cts)
        assert len(ranges) == 1
        assert ranges[0]["start"] == cts[2].start_time
        assert ranges[0]["end"] == cts[2].end_time

    def test_contiguous_filtered_clips_merged(self) -> None:
        cts = self._make_timestamps()
        clips = [{"objects": [{"id": 1}], "events": []} for _ in cts]
        clips[1] = {"objects": [], "events": []}
        clips[2] = {"objects": [], "events": []}
        clips[3] = {"objects": [], "events": []}
        ranges = find_motion_filtered_ranges(clips, cts)
        assert len(ranges) == 1
        assert ranges[0]["start"] == cts[1].start_time
        assert ranges[0]["end"] == cts[3].end_time

    def test_non_contiguous_filtered_clips_separate(self) -> None:
        cts = self._make_timestamps()
        clips = [{"objects": [{"id": 1}], "events": []} for _ in cts]
        clips[0] = {"objects": [], "events": []}
        clips[3] = {"objects": [], "events": []}
        ranges = find_motion_filtered_ranges(clips, cts)
        assert len(ranges) == 2


class TestEnrichClips:
    def _make_timestamps(self) -> list[ClipTimestamp]:
        return compute_clip_timestamps(
            source_fps=20.0,
            target_fps=1.0,
            total_frames=140,
            clip_length=3,
            clip_stride=2,
            video_start_time=datetime(2026, 1, 1, 12, 0, 0),
        )

    def test_all_annotated(self) -> None:
        cts = self._make_timestamps()
        clips = [
            {"objects": [{"id": i}], "events": [{"eid": i}]}
            for i in range(len(cts))
        ]
        result = enrich_clips(clips, cts)
        assert result.annotated_clips == len(cts)
        assert result.motion_filtered_clips == 0
        assert result.clips_with_events == len(cts)
        assert result.motion_filtered_ranges == []
        assert len(result.clips) == len(cts)

    def test_mixed_statuses(self) -> None:
        cts = self._make_timestamps()
        clips = [
            {"objects": [{"id": 0}], "events": [{"eid": 0}]},
            {"objects": [], "events": []},
            {"objects": [{"id": 2}], "events": []},
        ]
        # Pad to match timestamps count
        while len(clips) < len(cts):
            clips.append({"objects": [{"id": 99}], "events": []})
        result = enrich_clips(clips, cts)
        assert result.annotated_clips == len(cts) - 1
        assert result.motion_filtered_clips == 1
        assert result.clips_with_events == 1
        assert len(result.motion_filtered_ranges) == 1

    def test_clip_metadata_fields(self) -> None:
        cts = self._make_timestamps()
        clips = [{"objects": [{"id": 0}], "events": []} for _ in cts]
        result = enrich_clips(clips, cts)
        meta = result.clips[0]["clip_metadata"]
        assert meta["clip_index"] == 0
        assert meta["frame_indices"] == cts[0].frame_indices
        assert meta["start_time"] == cts[0].start_time
        assert meta["end_time"] == cts[0].end_time
        assert meta["start_offset_sec"] == cts[0].start_sec
        assert meta["end_offset_sec"] == cts[0].end_sec
        assert meta["status"] == "annotated"

    def test_preserves_original_clip_data(self) -> None:
        cts = self._make_timestamps()
        clips = [{"objects": [{"id": 0}], "events": [{"eid": 0}]}]
        while len(clips) < len(cts):
            clips.append({"objects": [], "events": []})
        result = enrich_clips(clips, cts)
        # Original fields preserved alongside clip_metadata
        assert result.clips[0]["objects"] == [{"id": 0}]
        assert result.clips[0]["events"] == [{"eid": 0}]
        assert "clip_metadata" in result.clips[0]

    def test_clips_exceed_timestamps(self) -> None:
        """Clips beyond timestamp range get fallback metadata."""
        cts = self._make_timestamps()
        clips = [{"objects": [{"id": i}], "events": []} for i in range(len(cts) + 2)]
        result = enrich_clips(clips, cts)
        assert len(result.clips) == len(cts) + 2
        extra = result.clips[-1]["clip_metadata"]
        assert extra["frame_indices"] == []
        assert extra["start_time"] == ""
