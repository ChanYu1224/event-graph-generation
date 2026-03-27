"""Compress annotation JSON files by eliminating structural redundancy.

Lossless structural compression: object deduplication, frame_indices formula,
short keys, null omission, minified output. Round-trip verifiable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

ATTRIBUTE_KEYS = ("color", "material", "position", "size", "state", "orientation", "pose")
STATUS_MAP = {"annotated": "a", "motion_filtered": "m"}
STATUS_RMAP = {v: k for k, v in STATUS_MAP.items()}


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _build_object_defs(clips: list[dict]) -> dict[str, str]:
    """Build obj_id -> category mapping from all clips.

    Args:
        clips: List of clip dicts from the original annotation.

    Returns:
        Dict mapping obj_id to category (first occurrence wins).
    """
    defs: dict[str, str] = {}
    for clip in clips:
        for obj in clip.get("objects", []):
            oid = obj["obj_id"]
            if oid not in defs:
                defs[oid] = obj["category"]
    return defs


def _build_attribute_profiles(clips: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Build deduplicated attribute profile table.

    Args:
        clips: List of clip dicts from the original annotation.

    Returns:
        Tuple of (profiles list, hash-to-index mapping).
        Each profile contains only non-null attribute key-value pairs.
    """
    profiles: list[dict] = []
    hash_to_idx: dict[str, int] = {}

    for clip in clips:
        for obj in clip.get("objects", []):
            attrs = obj.get("attributes", {})
            stripped = {k: v for k, v in attrs.items() if v is not None}
            h = json.dumps(stripped, sort_keys=True)
            if h not in hash_to_idx:
                hash_to_idx[h] = len(profiles)
                profiles.append(stripped)

    return profiles, hash_to_idx


def _compute_frame_index_params(clips: list[dict]) -> dict | None:
    """Detect frame_indices generation parameters.

    Args:
        clips: List of clip dicts from the original annotation.

    Returns:
        Dict with step, stride, length or None if no valid clips.
    """
    # Find first two clips with non-empty frame_indices
    samples = []
    for clip in clips:
        fi = clip.get("clip_metadata", {}).get("frame_indices", [])
        if fi:
            samples.append((clip["clip_metadata"]["clip_index"], fi))
        if len(samples) >= 2:
            break

    if not samples:
        return None

    fi0 = samples[0][1]
    length = len(fi0)
    step = fi0[1] - fi0[0] if length >= 2 else 0

    if len(samples) >= 2:
        stride = samples[1][1][0] - samples[0][1][0]
    else:
        stride = step * (length // 2) if step else 0

    return {"step": step, "stride": stride, "length": length}


def _reconstruct_frame_indices(clip_index: int, params: dict) -> list[int]:
    """Reconstruct frame_indices from clip_index and params.

    Args:
        clip_index: The clip's sequential index.
        params: Dict with step, stride, length.

    Returns:
        List of frame indices.
    """
    start = clip_index * params["stride"]
    return [start + j * params["step"] for j in range(params["length"])]


def _find_irregular_clips(clips: list[dict], params: dict | None) -> dict[int, list[int]]:
    """Find clips whose frame_indices don't match the formula.

    Args:
        clips: List of clip dicts.
        params: Frame index params (may be None).

    Returns:
        Dict mapping clip_index to actual frame_indices for irregular clips.
    """
    irregular: dict[int, list[int]] = {}
    if params is None:
        return irregular

    for clip in clips:
        meta = clip.get("clip_metadata", {})
        ci = meta.get("clip_index", 0)
        fi = meta.get("frame_indices", [])
        expected = _reconstruct_frame_indices(ci, params)

        if fi != expected:
            irregular[ci] = fi

    return irregular


def _compress_event(evt: dict) -> list:
    """Compress a single event dict to a positional array.

    Args:
        evt: Event dict with event_id, frame, action, agent, target, source, destination.

    Returns:
        Positional array with trailing nulls stripped.
    """
    arr = [
        evt["event_id"],
        evt["frame"],
        evt["action"],
        evt["agent"],
        evt.get("target"),
        evt.get("source"),
        evt.get("destination"),
    ]
    # Strip trailing nulls
    while arr and arr[-1] is None:
        arr.pop()
    return arr


def compress_file(data: dict) -> dict:
    """Compress a single annotation file.

    Args:
        data: Original annotation dict.

    Returns:
        Compact annotation dict.
    """
    clips = data.get("clips", [])

    object_defs = _build_object_defs(clips)
    attr_profiles, attr_hash_to_idx = _build_attribute_profiles(clips)
    fi_params = _compute_frame_index_params(clips)
    irregular_clips = _find_irregular_clips(clips, fi_params)

    compact_clips = []
    for clip in clips:
        meta = clip.get("clip_metadata", {})
        ci = meta.get("clip_index", 0)
        status = meta.get("status", "annotated")

        cc: dict = {
            "i": ci,
            "s": STATUS_MAP.get(status, "a"),
        }

        objects = clip.get("objects", [])
        events = clip.get("events", [])

        if objects:
            compact_objs = []
            for obj in objects:
                attrs = obj.get("attributes", {})
                stripped = {k: v for k, v in attrs.items() if v is not None}
                h = json.dumps(stripped, sort_keys=True)
                profile_idx = attr_hash_to_idx[h]
                entry = [
                    obj["obj_id"],
                    obj.get("first_seen_frame", 0),
                    profile_idx,
                ]
                # Store category override when it differs from object_defs
                oid = obj["obj_id"]
                if obj.get("category", "") != object_defs.get(oid, ""):
                    entry.append(obj["category"])
                compact_objs.append(entry)
            cc["o"] = compact_objs

        if events:
            cc["e"] = [_compress_event(evt) for evt in events]

        # Store irregular frame_indices explicitly
        if ci in irregular_clips:
            cc["fi"] = irregular_clips[ci]

        compact_clips.append(cc)

    compact = {
        "format_version": 1,
        "video_id": data.get("video_id", ""),
        "video_path": data.get("video_path", ""),
        "video_metadata": data.get("video_metadata", {}),
        "coverage": data.get("coverage", {}),
        "num_clips": data.get("num_clips", len(clips)),
        "validation_stats": data.get("validation_stats", {}),
        "frame_index_params": fi_params,
        "object_defs": object_defs,
        "attribute_profiles": attr_profiles,
        "clips": compact_clips,
    }
    return compact


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------


def _expand_attributes(profile: dict) -> dict:
    """Expand an attribute profile to the full 7-key dict with nulls.

    Args:
        profile: Sparse attribute dict (non-null keys only).

    Returns:
        Full attribute dict with all 7 keys.
    """
    return {k: profile.get(k) for k in ATTRIBUTE_KEYS}


def _expand_event(arr: list) -> dict:
    """Expand a positional event array back to a dict.

    Args:
        arr: Positional array [event_id, frame, action, agent, target?, source?, dest?].

    Returns:
        Full event dict with all 7 keys.
    """
    # Pad to 7 elements with None
    padded = arr + [None] * (7 - len(arr))
    return {
        "event_id": padded[0],
        "frame": padded[1],
        "action": padded[2],
        "agent": padded[3],
        "target": padded[4],
        "source": padded[5],
        "destination": padded[6],
    }


def _reconstruct_clip_metadata(
    clip_index: int,
    status: str,
    fi_params: dict | None,
    video_metadata: dict,
    irregular_fi: list[int] | None = None,
) -> dict:
    """Reconstruct full clip_metadata from compact representation.

    Args:
        clip_index: Clip sequential index.
        status: Full status string ("annotated" or "motion_filtered").
        fi_params: Frame index generation params.
        video_metadata: Video-level metadata for timestamp reconstruction.
        irregular_fi: Explicit frame_indices for irregular clips.

    Returns:
        Full clip_metadata dict.
    """
    if irregular_fi is not None:
        frame_indices = irregular_fi
    elif fi_params:
        frame_indices = _reconstruct_frame_indices(clip_index, fi_params)
    else:
        frame_indices = []

    source_fps = video_metadata.get("source_fps", 20.0)
    video_start_str = video_metadata.get("video_start_time", "")

    meta: dict = {
        "clip_index": clip_index,
        "frame_indices": frame_indices,
    }

    if video_start_str and frame_indices:
        video_start = datetime.fromisoformat(video_start_str)
        start_offset = frame_indices[0] / source_fps
        end_offset = frame_indices[-1] / source_fps
        start_time = video_start + timedelta(seconds=start_offset)
        end_time = video_start + timedelta(seconds=end_offset)
        meta["start_time"] = start_time.isoformat()
        meta["end_time"] = end_time.isoformat()
        meta["start_offset_sec"] = start_offset
        meta["end_offset_sec"] = end_offset
    elif video_start_str:
        # Empty frame_indices edge case
        meta["start_time"] = video_start_str
        meta["end_time"] = video_start_str
        meta["start_offset_sec"] = 0.0
        meta["end_offset_sec"] = 0.0

    meta["status"] = status
    return meta


def decompress_file(compact: dict) -> dict:
    """Decompress a compact annotation file to original format.

    Args:
        compact: Compact annotation dict.

    Returns:
        Original-format annotation dict.
    """
    object_defs = compact.get("object_defs", {})
    attr_profiles = compact.get("attribute_profiles", [])
    fi_params = compact.get("frame_index_params")
    video_metadata = compact.get("video_metadata", {})

    expanded_clips = []
    for cc in compact.get("clips", []):
        ci = cc["i"]
        status = STATUS_RMAP.get(cc["s"], "annotated")
        irregular_fi = cc.get("fi")

        # Expand objects
        objects = []
        for obj_arr in cc.get("o", []):
            oid, first_seen_frame, profile_idx = obj_arr[0], obj_arr[1], obj_arr[2]
            # 4th element is category override (when VLM assigned different category)
            category = obj_arr[3] if len(obj_arr) > 3 else object_defs.get(oid, oid.rsplit("_", 1)[0])
            objects.append({
                "obj_id": oid,
                "category": category,
                "first_seen_frame": first_seen_frame,
                "attributes": _expand_attributes(attr_profiles[profile_idx]),
            })

        # Expand events
        events = [_expand_event(e) for e in cc.get("e", [])]

        # Reconstruct clip_metadata
        clip_metadata = _reconstruct_clip_metadata(
            ci, status, fi_params, video_metadata, irregular_fi
        )

        expanded_clips.append({
            "objects": objects,
            "events": events,
            "clip_metadata": clip_metadata,
        })

    return {
        "video_id": compact.get("video_id", ""),
        "video_path": compact.get("video_path", ""),
        "video_metadata": video_metadata,
        "coverage": compact.get("coverage", {}),
        "num_clips": compact.get("num_clips", 0),
        "validation_stats": compact.get("validation_stats", {}),
        "clips": expanded_clips,
    }


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------


def verify_roundtrip(original: dict, decompressed: dict) -> None:
    """Verify that decompressed data matches original on all critical fields.

    Args:
        original: Original annotation dict.
        decompressed: Decompressed annotation dict.

    Raises:
        AssertionError: If any field mismatch is found.
    """
    assert original["video_id"] == decompressed["video_id"], "video_id mismatch"

    orig_clips = original.get("clips", [])
    dec_clips = decompressed.get("clips", [])
    assert len(orig_clips) == len(dec_clips), (
        f"Clip count mismatch: {len(orig_clips)} vs {len(dec_clips)}"
    )

    for idx, (oc, dc) in enumerate(zip(orig_clips, dec_clips)):
        ometa = oc.get("clip_metadata", {})
        dmeta = dc.get("clip_metadata", {})

        # clip_index and status
        assert ometa.get("clip_index") == dmeta.get("clip_index"), (
            f"Clip {idx}: clip_index mismatch"
        )
        assert ometa.get("status") == dmeta.get("status"), (
            f"Clip {idx}: status mismatch"
        )

        # frame_indices
        assert ometa.get("frame_indices", []) == dmeta.get("frame_indices", []), (
            f"Clip {idx}: frame_indices mismatch: "
            f"{ometa.get('frame_indices', [])[:3]}... vs {dmeta.get('frame_indices', [])[:3]}..."
        )

        # Objects
        oobjs = oc.get("objects", [])
        dobjs = dc.get("objects", [])
        assert len(oobjs) == len(dobjs), (
            f"Clip {idx}: object count mismatch {len(oobjs)} vs {len(dobjs)}"
        )
        for j, (oo, do) in enumerate(zip(oobjs, dobjs)):
            assert oo["obj_id"] == do["obj_id"], (
                f"Clip {idx} obj {j}: obj_id mismatch"
            )
            assert oo["category"] == do["category"], (
                f"Clip {idx} obj {j}: category mismatch "
                f"'{oo['category']}' vs '{do['category']}'"
            )
            assert oo.get("first_seen_frame", 0) == do.get("first_seen_frame", 0), (
                f"Clip {idx} obj {j}: first_seen_frame mismatch"
            )
            oattrs = oo.get("attributes", {})
            dattrs = do.get("attributes", {})
            for key in ATTRIBUTE_KEYS:
                assert oattrs.get(key) == dattrs.get(key), (
                    f"Clip {idx} obj {j}: attribute '{key}' mismatch "
                    f"'{oattrs.get(key)}' vs '{dattrs.get(key)}'"
                )

        # Events
        oevts = oc.get("events", [])
        devts = dc.get("events", [])
        assert len(oevts) == len(devts), (
            f"Clip {idx}: event count mismatch {len(oevts)} vs {len(devts)}"
        )
        for j, (oe, de) in enumerate(zip(oevts, devts)):
            for key in ("event_id", "frame", "action", "agent", "target", "source", "destination"):
                assert oe.get(key) == de.get(key), (
                    f"Clip {idx} evt {j}: '{key}' mismatch "
                    f"'{oe.get(key)}' vs '{de.get(key)}'"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for annotation compression."""
    parser = argparse.ArgumentParser(
        description="Compress/decompress annotation JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Input directory containing JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations_compact"),
        help="Output directory for processed files.",
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Run in decompress mode (compact -> original format).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify round-trip correctness after compression (default: True).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip round-trip verification.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.no_verify:
        args.verify = False

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.error("No JSON files found in %s", input_dir)
        sys.exit(1)

    logger.info("Processing %d files from %s", len(json_files), input_dir)

    total_input_size = 0
    total_output_size = 0
    errors = 0

    for fpath in json_files:
        input_size = fpath.stat().st_size
        total_input_size += input_size

        with open(fpath) as f:
            data = json.load(f)

        if args.decompress:
            result = decompress_file(data)
            out_path = output_dir / fpath.name
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            compact = compress_file(data)
            out_path = output_dir / fpath.name
            with open(out_path, "w") as f:
                json.dump(compact, f, ensure_ascii=False)

            if args.verify:
                try:
                    decompressed = decompress_file(compact)
                    verify_roundtrip(data, decompressed)
                except AssertionError as e:
                    logger.error("Round-trip FAILED for %s: %s", fpath.name, e)
                    errors += 1
                    continue

        output_size = out_path.stat().st_size
        total_output_size += output_size
        ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0
        logger.info(
            "  %s: %s -> %s (%.1f%% reduction)",
            fpath.name,
            _fmt_size(input_size),
            _fmt_size(output_size),
            ratio,
        )

    logger.info("=" * 60)
    mode = "Decompression" if args.decompress else "Compression"
    logger.info("%s complete: %d files", mode, len(json_files))
    logger.info(
        "Total: %s -> %s (%.1f%% reduction)",
        _fmt_size(total_input_size),
        _fmt_size(total_output_size),
        (1 - total_output_size / total_input_size) * 100 if total_input_size > 0 else 0,
    )
    if errors:
        logger.error("Round-trip verification failed for %d files!", errors)
        sys.exit(1)
    elif not args.decompress and args.verify:
        logger.info("Round-trip verification passed for all files.")


def _fmt_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    if nbytes >= 1024 * 1024:
        return f"{nbytes / 1024 / 1024:.1f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes} B"


if __name__ == "__main__":
    main()
