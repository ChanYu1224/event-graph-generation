"""Tests for frame sampler."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from event_graph_generation.data.frame_sampler import FrameSampler, SampledFrame


def _create_test_video(path: str, num_frames: int = 30, fps: float = 30.0) -> None:
    """Create a minimal test video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (64, 64))
    for i in range(num_frames):
        frame = np.full((64, 64, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestFrameSampler:
    def test_init_valid(self):
        sampler = FrameSampler(target_fps=1.0)
        assert sampler.target_fps == 1.0

    def test_init_invalid_fps(self):
        with pytest.raises(ValueError):
            FrameSampler(target_fps=0.0)
        with pytest.raises(ValueError):
            FrameSampler(target_fps=-1.0)

    def test_sample_file_not_found(self):
        sampler = FrameSampler()
        with pytest.raises(FileNotFoundError):
            sampler.sample("/nonexistent/video.mp4")

    def test_sample_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            _create_test_video(str(video_path), num_frames=60, fps=30.0)

            sampler = FrameSampler(target_fps=1.0)
            frames = sampler.sample(video_path)

            # At 30fps source and 1fps target, interval=30, so ~2 frames from 60 total
            assert len(frames) >= 1
            assert all(isinstance(f, SampledFrame) for f in frames)
            assert all(f.image.shape == (64, 64, 3) for f in frames)
            # Frame indices should be monotonically increasing
            indices = [f.frame_index for f in frames]
            assert indices == sorted(indices)

    def test_sample_high_fps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            _create_test_video(str(video_path), num_frames=30, fps=30.0)

            sampler = FrameSampler(target_fps=10.0)
            frames = sampler.sample(video_path)

            # At 30fps source and 10fps target, interval=3, so ~10 frames
            assert len(frames) >= 5

    def test_sample_clip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            _create_test_video(str(video_path), num_frames=90, fps=30.0)

            sampler = FrameSampler(target_fps=1.0)
            frames = sampler.sample_clip(video_path, start_frame=0, num_frames=3)

            assert len(frames) == 3

    def test_get_video_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            _create_test_video(str(video_path), num_frames=60, fps=30.0)

            info = FrameSampler.get_video_info(video_path)
            assert info["width"] == 64
            assert info["height"] == 64
            assert info["total_frames"] == 60
            assert abs(info["fps"] - 30.0) < 1.0

    def test_timestamp_increases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            _create_test_video(str(video_path), num_frames=90, fps=30.0)

            sampler = FrameSampler(target_fps=1.0)
            frames = sampler.sample(video_path)

            timestamps = [f.timestamp_sec for f in frames]
            assert timestamps == sorted(timestamps)
            if len(timestamps) > 1:
                assert timestamps[-1] > timestamps[0]

    def test_sample_sequential_read_correctness(self):
        """Verify sequential read produces correct pixel values per frame index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            # Each frame has pixel value = i % 256
            _create_test_video(str(video_path), num_frames=90, fps=30.0)

            sampler = FrameSampler(target_fps=1.0)
            frames = sampler.sample(video_path)

            # At 30fps source and 1fps target, interval=30
            # Expected frame indices: 0, 30, 60
            assert len(frames) == 3
            expected_indices = [0, 30, 60]
            for frame, expected_idx in zip(frames, expected_indices):
                assert frame.frame_index == expected_idx
                # Pixel value should be expected_idx % 256
                # Use mean to be robust to codec compression artifacts
                expected_val = expected_idx % 256
                actual_mean = frame.image.mean()
                assert abs(actual_mean - expected_val) < 10, (
                    f"Frame {expected_idx}: expected pixel ~{expected_val}, "
                    f"got mean {actual_mean:.1f}"
                )
