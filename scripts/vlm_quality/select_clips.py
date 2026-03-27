"""Select clips for VLM annotation quality evaluation via stratified sampling.

Selects ~25 clips: 15 with events (stratified by action type) + 10 without events
(5 annotated-no-event + 5 motion_filtered) for false positive/negative analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_action_names(actions_path: str) -> list[str]:
    """Load action names from actions.yaml."""
    import yaml

    with open(actions_path) as f:
        data = yaml.safe_load(f)
    return [a["name"] for a in data["actions"] if a["name"] != "no_event"]


def _scan_annotations(annotations_dir: Path) -> list[dict]:
    """Scan all annotation files and build clip index.

    Returns:
        List of clip info dicts with video_id, clip_index, frame_indices,
        status, actions, n_events, n_objects, video_path.
    """
    clips = []
    for ann_path in sorted(annotations_dir.glob("*.json")):
        with open(ann_path) as f:
            ann = json.load(f)

        video_id = ann.get("video_id", ann_path.stem)
        video_path = ann.get("video_path", "")

        for clip in ann.get("clips", []):
            meta = clip.get("clip_metadata", {})
            events = clip.get("events", [])
            objects = clip.get("objects", [])
            actions = [
                e.get("action", "")
                for e in events
                if e.get("action", "") != "no_event"
            ]

            clips.append({
                "video_id": video_id,
                "clip_index": meta.get("clip_index", 0),
                "frame_indices": meta.get("frame_indices", []),
                "status": meta.get("status", "annotated"),
                "actions": actions,
                "n_events": len(events),
                "n_objects": len(objects),
                "video_path": video_path,
            })

    return clips


def select_clips(
    annotations_dir: Path,
    action_names: list[str],
    seed: int = 42,
    n_event_clips: int = 15,
    n_no_event_annotated: int = 5,
    n_motion_filtered: int = 5,
) -> list[dict]:
    """Select clips using stratified sampling.

    Args:
        annotations_dir: Path to annotation JSON files.
        action_names: List of action type names (excluding no_event).
        seed: Random seed for reproducibility.
        n_event_clips: Number of event clips to select.
        n_no_event_annotated: Number of annotated-but-no-event clips.
        n_motion_filtered: Number of motion_filtered clips.

    Returns:
        List of selected clip info dicts with selection_reason.
    """
    rng = random.Random(seed)
    all_clips = _scan_annotations(annotations_dir)
    logger.info("Scanned %d total clips", len(all_clips))

    # Partition clips
    event_clips = [c for c in all_clips if c["actions"]]
    no_event_annotated = [
        c for c in all_clips
        if c["status"] == "annotated" and not c["actions"] and c["n_events"] == 0
    ]
    motion_filtered = [c for c in all_clips if c["status"] == "motion_filtered"]

    logger.info(
        "Event clips: %d, No-event annotated: %d, Motion filtered: %d",
        len(event_clips), len(no_event_annotated), len(motion_filtered),
    )

    selected: list[dict] = []
    used_videos: set[str] = set()

    # Phase 1: one clip per action type (ensure coverage)
    action_to_clips: dict[str, list[dict]] = {}
    for c in event_clips:
        for action in set(c["actions"]):
            action_to_clips.setdefault(action, []).append(c)

    for action in action_names:
        candidates = action_to_clips.get(action, [])
        if not candidates:
            continue

        # Prefer clips from unseen videos, with multiple diverse actions
        def _score(c: dict) -> tuple[int, int]:
            video_novelty = 0 if c["video_id"] in used_videos else 1
            action_diversity = len(set(c["actions"]))
            return (video_novelty, action_diversity)

        candidates.sort(key=_score, reverse=True)
        # Pick from top candidates randomly
        top = [c for c in candidates if _score(c) == _score(candidates[0])]
        pick = rng.choice(top)

        # Avoid duplicates
        clip_key = (pick["video_id"], pick["clip_index"])
        if any((s["video_id"], s["clip_index"]) == clip_key for s in selected):
            # Try next best
            for alt in candidates:
                alt_key = (alt["video_id"], alt["clip_index"])
                if not any((s["video_id"], s["clip_index"]) == alt_key for s in selected):
                    pick = alt
                    break

        pick_copy = dict(pick)
        pick_copy["selection_reason"] = f"action_type:{action}"
        selected.append(pick_copy)
        used_videos.add(pick["video_id"])

    # Phase 2: fill remaining event slots
    remaining = n_event_clips - len(selected)
    if remaining > 0:
        selected_keys = {(s["video_id"], s["clip_index"]) for s in selected}
        pool = [
            c for c in event_clips
            if (c["video_id"], c["clip_index"]) not in selected_keys
        ]
        # Prefer unseen videos, more events
        pool.sort(
            key=lambda c: (0 if c["video_id"] in used_videos else 1, c["n_events"]),
            reverse=True,
        )
        for pick in pool[:remaining]:
            pick_copy = dict(pick)
            pick_copy["selection_reason"] = "fill_event"
            selected.append(pick_copy)
            used_videos.add(pick["video_id"])

    # Phase 3: no-event annotated clips
    rng.shuffle(no_event_annotated)
    # Prefer unseen videos
    no_event_annotated.sort(
        key=lambda c: 0 if c["video_id"] in used_videos else 1, reverse=True,
    )
    for pick in no_event_annotated[:n_no_event_annotated]:
        pick_copy = dict(pick)
        pick_copy["selection_reason"] = "no_event_annotated"
        selected.append(pick_copy)
        used_videos.add(pick["video_id"])

    # Phase 4: motion_filtered clips
    rng.shuffle(motion_filtered)
    motion_filtered.sort(
        key=lambda c: 0 if c["video_id"] in used_videos else 1, reverse=True,
    )
    for pick in motion_filtered[:n_motion_filtered]:
        pick_copy = dict(pick)
        pick_copy["selection_reason"] = "motion_filtered"
        selected.append(pick_copy)
        used_videos.add(pick["video_id"])

    logger.info("Selected %d clips from %d videos", len(selected), len(used_videos))
    return selected


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Select clips for VLM quality eval")
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--actions", default="configs/actions.yaml")
    parser.add_argument("--output", type=Path, default=Path("data/vlm_quality/selected_clips.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    action_names = _load_action_names(args.actions)
    logger.info("Action types: %s", action_names)

    selected = select_clips(args.annotations_dir, action_names, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "n_clips": len(selected),
        "clips": selected,
    }
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary
    event_clips = [c for c in selected if c["actions"]]
    no_event = [c for c in selected if not c["actions"]]
    all_actions = set()
    for c in event_clips:
        all_actions.update(c["actions"])

    print(f"\nSelected {len(selected)} clips:")
    print(f"  Event clips: {len(event_clips)}")
    print(f"  No-event clips: {len(no_event)}")
    print(f"  Action types covered: {sorted(all_actions)}")
    print(f"  Videos: {len(set(c['video_id'] for c in selected))}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
