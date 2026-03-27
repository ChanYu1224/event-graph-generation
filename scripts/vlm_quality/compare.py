"""Compare VLM annotations against human ground truth.

Computes object detection and event detection metrics using Hungarian matching.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_frame(frame: int, frame_indices: list[int]) -> int:
    """Normalize event frame to local 0-15 index.

    Args:
        frame: Frame value (may be source frame index or local index).
        frame_indices: Source frame indices for the clip.

    Returns:
        Local frame index (0-15).
    """
    if frame <= 15:
        return frame
    if frame in frame_indices:
        return frame_indices.index(frame)
    diffs = [abs(fi - frame) for fi in frame_indices]
    return diffs.index(min(diffs))


def _load_vlm_clip(
    video_id: str, clip_index: int, annotations_dir: Path,
) -> dict | None:
    """Load VLM annotation for a specific clip.

    Args:
        video_id: Video identifier.
        clip_index: Clip index within the video.
        annotations_dir: Directory with annotation JSON files.

    Returns:
        Clip dict from VLM annotation, or None if not found.
    """
    ann_path = annotations_dir / f"{video_id}.json"
    if not ann_path.exists():
        return None
    with open(ann_path) as f:
        ann = json.load(f)
    for clip in ann.get("clips", []):
        meta = clip.get("clip_metadata", {})
        if meta.get("clip_index") == clip_index:
            return clip
    return None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_objects(
    gt_objects: list[dict], vlm_objects: list[dict],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match GT objects to VLM objects using category-based Hungarian matching.

    Args:
        gt_objects: List of GT object dicts.
        vlm_objects: List of VLM object dicts.

    Returns:
        Tuple of (matched pairs, unmatched GT indices, unmatched VLM indices).
    """
    if not gt_objects or not vlm_objects:
        return [], list(range(len(gt_objects))), list(range(len(vlm_objects)))

    n_gt = len(gt_objects)
    n_vlm = len(vlm_objects)
    cost = np.ones((n_gt, n_vlm), dtype=np.float64)

    for i, go in enumerate(gt_objects):
        for j, vo in enumerate(vlm_objects):
            if go["category"] == vo.get("category", ""):
                cost[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)

    matched = []
    matched_gt = set()
    matched_vlm = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 0.5:  # Only match same-category
            matched.append((r, c))
            matched_gt.add(r)
            matched_vlm.add(c)

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_vlm = [j for j in range(n_vlm) if j not in matched_vlm]

    return matched, unmatched_gt, unmatched_vlm


def match_events(
    gt_events: list[dict],
    vlm_events: list[dict],
    gt_obj_map: dict[str, dict],
    vlm_obj_map: dict[str, dict],
    frame_indices: list[int],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match GT events to VLM events using multi-factor Hungarian matching.

    Args:
        gt_events: List of GT event dicts.
        vlm_events: List of VLM event dicts (excluding no_event).
        gt_obj_map: GT obj_id -> object dict.
        vlm_obj_map: VLM obj_id -> object dict.
        frame_indices: Source frame indices for normalization.

    Returns:
        Tuple of (matched pairs, unmatched GT indices, unmatched VLM indices).
    """
    if not gt_events or not vlm_events:
        return [], list(range(len(gt_events))), list(range(len(vlm_events)))

    n_gt = len(gt_events)
    n_vlm = len(vlm_events)
    cost = np.full((n_gt, n_vlm), 3.0, dtype=np.float64)

    for i, ge in enumerate(gt_events):
        gt_frame = _normalize_frame(ge.get("frame", 0), frame_indices)
        gt_agent_cat = gt_obj_map.get(ge.get("agent", ""), {}).get("category", "")
        gt_target_cat = gt_obj_map.get(ge.get("target", ""), {}).get("category", "")

        for j, ve in enumerate(vlm_events):
            vlm_frame = _normalize_frame(ve.get("frame", 0), frame_indices)
            vlm_agent_cat = vlm_obj_map.get(ve.get("agent", ""), {}).get("category", "")
            vlm_target_cat = vlm_obj_map.get(ve.get("target", ""), {}).get("category", "")

            c = 0.0
            c += 0.0 if ge.get("action") == ve.get("action") else 1.0
            c += min(abs(gt_frame - vlm_frame) / 16.0, 1.0)
            c += 0.0 if gt_agent_cat == vlm_agent_cat else 0.5
            c += 0.0 if gt_target_cat == vlm_target_cat else 0.5
            cost[i, j] = c

    row_ind, col_ind = linear_sum_assignment(cost)

    matched = []
    matched_gt = set()
    matched_vlm = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 2.0:
            matched.append((r, c))
            matched_gt.add(r)
            matched_vlm.add(c)

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_vlm = [j for j in range(n_vlm) if j not in matched_vlm]

    return matched, unmatched_gt, unmatched_vlm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> dict:
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compare_all(
    gt_data: dict,
    annotations_dir: Path,
    selected_clips: list[dict],
) -> dict:
    """Compare all clips and compute aggregate metrics.

    Args:
        gt_data: Ground truth data with clips list.
        annotations_dir: Directory with VLM annotation JSON files.
        selected_clips: List of selected clip manifests.

    Returns:
        Comparison report dict.
    """
    gt_clips = gt_data["clips"]

    # Build frame_indices lookup from selected_clips
    fi_lookup: dict[tuple[str, int], list[int]] = {}
    for sc in selected_clips:
        fi_lookup[(sc["video_id"], sc["clip_index"])] = sc.get("frame_indices", [])

    # Aggregate counters
    obj_tp = obj_fp = obj_fn = 0
    evt_tp = evt_fp = evt_fn = 0
    action_correct = action_total = 0
    agent_correct = agent_total = 0
    target_correct = target_total = 0

    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_action: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    attr_correct: dict[str, int] = defaultdict(int)
    attr_total: dict[str, int] = defaultdict(int)

    clip_details = []
    fp_events_in_no_event_clips = 0
    n_no_event_clips = 0

    for gt_clip in gt_clips:
        vid = gt_clip["video_id"]
        ci = gt_clip["clip_index"]
        frame_indices = fi_lookup.get((vid, ci), list(range(16)))

        vlm_clip = _load_vlm_clip(vid, ci, annotations_dir)
        if vlm_clip is None:
            logger.warning("VLM clip not found: %s clip %d", vid, ci)
            continue

        gt_objects = gt_clip.get("objects", [])
        vlm_objects = vlm_clip.get("objects", [])
        gt_events = gt_clip.get("events", [])
        vlm_events = [
            e for e in vlm_clip.get("events", [])
            if e.get("action", "") != "no_event"
        ]

        # Object matching
        obj_matched, obj_unmatched_gt, obj_unmatched_vlm = match_objects(gt_objects, vlm_objects)
        obj_tp += len(obj_matched)
        obj_fn += len(obj_unmatched_gt)
        obj_fp += len(obj_unmatched_vlm)

        for gi, vi in obj_matched:
            cat = gt_objects[gi]["category"]
            per_category[cat]["tp"] += 1
        for gi in obj_unmatched_gt:
            per_category[gt_objects[gi]["category"]]["fn"] += 1
        for vi in obj_unmatched_vlm:
            per_category[vlm_objects[vi].get("category", "unknown")]["fp"] += 1

        # Attribute comparison for matched objects
        for gi, vi in obj_matched:
            gt_attrs = gt_objects[gi].get("attributes", {})
            vlm_attrs = vlm_objects[vi].get("attributes", {})
            for key in ("color", "material", "position", "size", "state", "orientation", "pose"):
                gt_val = gt_attrs.get(key)
                vlm_val = vlm_attrs.get(key)
                if gt_val is not None or vlm_val is not None:
                    attr_total[key] += 1
                    if gt_val == vlm_val:
                        attr_correct[key] += 1

        # Event matching
        gt_obj_map = {o["obj_id"]: o for o in gt_objects}
        vlm_obj_map = {o.get("obj_id", o.get("id", "")): o for o in vlm_objects}

        evt_matched, evt_unmatched_gt, evt_unmatched_vlm = match_events(
            gt_events, vlm_events, gt_obj_map, vlm_obj_map, frame_indices,
        )
        evt_tp += len(evt_matched)
        evt_fn += len(evt_unmatched_gt)
        evt_fp += len(evt_unmatched_vlm)

        # No-event clip analysis
        if not gt_events:
            n_no_event_clips += 1
            if vlm_events:
                fp_events_in_no_event_clips += len(vlm_events)

        # Per-action metrics
        for gi, vi in evt_matched:
            gt_action = gt_events[gi].get("action", "")
            vlm_action = vlm_events[vi].get("action", "")
            per_action[gt_action]["tp"] += 1
            action_total += 1
            if gt_action == vlm_action:
                action_correct += 1

            # Agent/target accuracy
            gt_agent_cat = gt_obj_map.get(gt_events[gi].get("agent", ""), {}).get("category", "")
            vlm_agent_cat = vlm_obj_map.get(vlm_events[vi].get("agent", ""), {}).get("category", "")
            agent_total += 1
            if gt_agent_cat == vlm_agent_cat:
                agent_correct += 1

            gt_target_cat = gt_obj_map.get(gt_events[gi].get("target"), {}).get("category", "")
            vlm_target_cat = vlm_obj_map.get(vlm_events[vi].get("target"), {}).get("category", "")
            target_total += 1
            if gt_target_cat == vlm_target_cat:
                target_correct += 1

        for gi in evt_unmatched_gt:
            per_action[gt_events[gi].get("action", "")]["fn"] += 1
        for vi in evt_unmatched_vlm:
            per_action[vlm_events[vi].get("action", "")]["fp"] += 1

        clip_details.append({
            "video_id": vid,
            "clip_index": ci,
            "gt_objects": len(gt_objects),
            "vlm_objects": len(vlm_objects),
            "obj_matched": len(obj_matched),
            "gt_events": len(gt_events),
            "vlm_events": len(vlm_events),
            "evt_matched": len(evt_matched),
            "gt_event_actions": [e.get("action") for e in gt_events],
            "vlm_event_actions": [e.get("action") for e in vlm_events],
        })

    # Assemble report
    report = {
        "summary": {
            "n_clips": len(gt_clips),
            "n_clips_with_gt_events": sum(1 for c in gt_clips if c.get("events")),
            "n_clips_no_gt_events": sum(1 for c in gt_clips if not c.get("events")),
        },
        "object_detection": _prf(obj_tp, obj_fp, obj_fn),
        "object_per_category": {
            cat: _prf(v["tp"], v["fp"], v["fn"])
            for cat, v in sorted(per_category.items())
        },
        "event_detection": _prf(evt_tp, evt_fp, evt_fn),
        "event_per_action": {
            act: _prf(v["tp"], v["fp"], v["fn"])
            for act, v in sorted(per_action.items())
        },
        "action_classification": {
            "accuracy": action_correct / action_total if action_total > 0 else 0.0,
            "correct": action_correct,
            "total": action_total,
        },
        "pointer_accuracy": {
            "agent_accuracy": agent_correct / agent_total if agent_total > 0 else 0.0,
            "target_accuracy": target_correct / target_total if target_total > 0 else 0.0,
        },
        "attribute_accuracy": {
            key: attr_correct[key] / attr_total[key] if attr_total[key] > 0 else 0.0
            for key in ("color", "material", "position", "size", "state", "orientation", "pose")
        },
        "false_positive_analysis": {
            "no_event_clips_total": n_no_event_clips,
            "no_event_clips_with_vlm_events": sum(
                1 for d in clip_details
                if d["gt_events"] == 0 and d["vlm_events"] > 0
            ),
            "total_spurious_events": fp_events_in_no_event_clips,
        },
        "per_clip_details": clip_details,
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(report: dict) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("VLM ANNOTATION QUALITY REPORT")
    print("=" * 70)

    s = report["summary"]
    print(f"\nClips: {s['n_clips']} total "
          f"({s['n_clips_with_gt_events']} with events, "
          f"{s['n_clips_no_gt_events']} without)")

    od = report["object_detection"]
    print(f"\n--- Object Detection ---")
    print(f"  Precision: {od['precision']:.3f}  Recall: {od['recall']:.3f}  F1: {od['f1']:.3f}")
    print(f"  TP: {od['tp']}  FP: {od['fp']}  FN: {od['fn']}")

    print(f"\n  Per-category:")
    for cat, m in report["object_per_category"].items():
        if m["tp"] + m["fp"] + m["fn"] > 0:
            print(f"    {cat:<15s}  P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}  "
                  f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    ed = report["event_detection"]
    print(f"\n--- Event Detection ---")
    print(f"  Precision: {ed['precision']:.3f}  Recall: {ed['recall']:.3f}  F1: {ed['f1']:.3f}")
    print(f"  TP: {ed['tp']}  FP: {ed['fp']}  FN: {ed['fn']}")

    print(f"\n  Per-action:")
    for act, m in report["event_per_action"].items():
        if m["tp"] + m["fp"] + m["fn"] > 0:
            print(f"    {act:<15s}  P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}  "
                  f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    ac = report["action_classification"]
    print(f"\n--- Action Classification ---")
    print(f"  Accuracy: {ac['accuracy']:.3f} ({ac['correct']}/{ac['total']})")

    pa = report["pointer_accuracy"]
    print(f"\n--- Pointer Accuracy ---")
    print(f"  Agent:  {pa['agent_accuracy']:.3f}")
    print(f"  Target: {pa['target_accuracy']:.3f}")

    aa = report["attribute_accuracy"]
    print(f"\n--- Attribute Accuracy ---")
    for key, val in aa.items():
        print(f"  {key:<15s}  {val:.3f}")

    fp = report["false_positive_analysis"]
    print(f"\n--- False Positive Analysis (no-event clips) ---")
    print(f"  No-event clips: {fp['no_event_clips_total']}")
    print(f"  With spurious VLM events: {fp['no_event_clips_with_vlm_events']}")
    print(f"  Total spurious events: {fp['total_spurious_events']}")

    print("\n" + "=" * 70)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compare VLM vs human annotations")
    parser.add_argument("--ground-truth", type=Path,
                        default=Path("data/vlm_quality/ground_truth.json"))
    parser.add_argument("--annotations-dir", type=Path,
                        default=Path("data/annotations"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/vlm_quality/selected_clips.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/vlm_quality/comparison_report.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(args.ground_truth) as f:
        gt_data = json.load(f)
    with open(args.manifest) as f:
        manifest = json.load(f)

    report = compare_all(gt_data, args.annotations_dir, manifest["clips"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _print_report(report)
    logger.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
