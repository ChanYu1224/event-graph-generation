"""Tests for VJEPAEventDataset and vjepa_collate_fn."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from event_graph_generation.data.vjepa_dataset import VJEPAEventDataset
from event_graph_generation.data.vjepa_collator import VJEPAEventBatch, vjepa_collate_fn


S = 64  # small token count for tests
D = 32


def _make_sample(
    video_id: str = "test_video",
    clip_index: int = 0,
    n_objects: int = 3,
    n_events: int = 1,
) -> dict:
    """Create a synthetic sample dict."""
    gt_events = []
    for i in range(n_events):
        gt_events.append({
            "agent_track_id": 0,
            "action_class": i % 5,
            "target_track_id": min(1, n_objects - 1),
            "source_track_id": None,
            "dest_track_id": None,
            "event_frame_index": i,
        })
    return {
        "vjepa_tokens": torch.randn(S, D),
        "gt_events": gt_events,
        "gt_object_categories": list(range(n_objects)),
        "num_objects": n_objects,
        "video_id": video_id,
        "clip_index": clip_index,
    }


class TestVJEPAEventDataset:
    def test_load_and_getitem(self, tmp_path: Path):
        """Test dataset loads samples from split file."""
        samples_dir = tmp_path / "samples"
        splits_dir = tmp_path / "splits"
        samples_dir.mkdir()
        splits_dir.mkdir()

        # Create samples
        sample_ids = ["video_a_clip_0000", "video_a_clip_0001"]
        for sid in sample_ids:
            sample = _make_sample()
            torch.save(sample, samples_dir / f"{sid}.pt")

        # Create split file
        with open(splits_dir / "train.txt", "w") as f:
            for sid in sample_ids:
                f.write(f"{sid}\n")

        dataset = VJEPAEventDataset(data_dir=tmp_path, split="train")
        assert len(dataset) == 2

        item = dataset[0]
        assert "vjepa_tokens" in item
        assert item["vjepa_tokens"].shape == (S, D)
        assert isinstance(item["gt_events"], list)
        assert isinstance(item["gt_object_categories"], list)


class TestVJEPACollator:
    def test_collate_shapes(self):
        samples = [_make_sample(clip_index=i) for i in range(3)]
        batch = vjepa_collate_fn(samples)

        assert isinstance(batch, VJEPAEventBatch)
        assert batch.vjepa_tokens.shape == (3, S, D)
        assert len(batch.gt_events) == 3
        assert len(batch.gt_object_categories) == 3
        assert len(batch.num_objects) == 3

    def test_to_device(self):
        samples = [_make_sample()]
        batch = vjepa_collate_fn(samples)
        batch_cpu = batch.to("cpu")
        assert batch_cpu.vjepa_tokens.device == torch.device("cpu")
        # Non-tensor fields preserved
        assert batch_cpu.gt_events == batch.gt_events
