"""Upload V-JEPA features and VLM annotations to Hugging Face as a dataset."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

MIT_LICENSE = """MIT License

Copyright (c) 2026 Yuchn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _discover_videos(
    features_dir: Path,
    annotations_dir: Path,
    compact_dir: Path,
) -> list[dict]:
    """Discover videos that have both features and annotations.

    Args:
        features_dir: Directory containing V-JEPA feature subdirectories.
        annotations_dir: Directory containing annotation JSON files.
        compact_dir: Directory containing compact annotation JSON files.

    Returns:
        List of video info dicts sorted by clip count (ascending).
    """
    feature_videos = {
        d.name for d in features_dir.iterdir() if d.is_dir()
    }
    annotation_videos = {
        f.stem for f in annotations_dir.glob("*.json")
    }
    compact_videos = {
        f.stem for f in compact_dir.glob("*.json")
    }

    common = sorted(feature_videos & annotation_videos)
    videos = []
    for video_id in common:
        clip_files = sorted((features_dir / video_id).glob("clip_*.pt"))
        feature_size = sum(f.stat().st_size for f in clip_files)
        annotation_size = (annotations_dir / f"{video_id}.json").stat().st_size

        # Load annotation to get coverage stats
        with open(annotations_dir / f"{video_id}.json") as f:
            ann = json.load(f)
        coverage = ann.get("coverage", {})

        videos.append({
            "video_id": video_id,
            "num_clips": len(clip_files),
            "feature_size_bytes": feature_size,
            "annotation_size_bytes": annotation_size,
            "has_compact": video_id in compact_videos,
            "total_clips": coverage.get("total_clips", len(clip_files)),
            "annotated_clips": coverage.get("annotated_clips", 0),
            "clips_with_events": coverage.get("clips_with_events", 0),
        })

    videos.sort(key=lambda v: v["num_clips"])
    return videos


def _build_metadata(
    selected_videos: list[dict],
    total_available: int,
) -> dict:
    """Build dataset-level metadata.

    Args:
        selected_videos: List of selected video info dicts.
        total_available: Total number of available videos.

    Returns:
        Metadata dict for metadata.json.
    """
    total_clips = sum(v["num_clips"] for v in selected_videos)
    total_feature_bytes = sum(v["feature_size_bytes"] for v in selected_videos)

    return {
        "dataset_name": "event-graph-vjepa-vitl-dataset",
        "description": "V-JEPA 2.1 ViT-L features and VLM synthetic annotations for event graph generation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": "MIT",
        "vjepa_model": "vjepa2_1_vit_large_384",
        "vjepa_backbone": "V-JEPA 2.1 ViT-L (Meta, frozen)",
        "feature_dim": 1024,
        "num_tokens_per_clip": 4608,
        "temporal_tokens": 8,
        "spatial_tokens": 576,
        "image_size": 384,
        "clip_length_frames": 16,
        "sampling_fps": 1.0,
        "num_object_categories": 28,
        "num_action_classes": 13,
        "subset": {
            "is_subset": len(selected_videos) < total_available,
            "total_videos_available": total_available,
            "included_videos": len(selected_videos),
            "included_clips": total_clips,
            "total_feature_size_gb": round(total_feature_bytes / 1e9, 2),
        },
        "videos": [
            {
                "video_id": v["video_id"],
                "num_clips": v["num_clips"],
                "annotated_clips": v["annotated_clips"],
                "clips_with_events": v["clips_with_events"],
                "feature_size_mb": round(v["feature_size_bytes"] / 1e6, 1),
            }
            for v in selected_videos
        ],
    }


def _generate_dataset_card(metadata: dict, repo_id: str) -> str:
    """Generate HuggingFace dataset card README.md.

    Args:
        metadata: Dataset metadata dict.
        repo_id: Full HF repo ID (namespace/repo-name).

    Returns:
        Markdown string for README.md.
    """
    subset = metadata["subset"]
    videos_table = "\n".join(
        f"| `{v['video_id']}` | {v['num_clips']} | {v['annotated_clips']} "
        f"| {v['clips_with_events']} | {v['feature_size_mb']:.0f} MB |"
        for v in metadata["videos"]
    )

    return f"""---
license: mit
tags:
  - video-understanding
  - event-graph
  - vjepa
  - slot-attention
  - pytorch
  - synthetic-annotations
task_categories:
  - video-classification
language:
  - ja
size_categories:
  - 1K<n<10K
---

# Event Graph Generation Dataset — V-JEPA 2.1 ViT-L Features + VLM Annotations

## 1. データセット概要

動画から構造化されたイベントグラフ（**誰が**・**何を**・**どこから**・**どこへ**）を予測するための学習データセットです。

室内環境（デスク・キッチン・部屋）の録画動画を対象に、以下の 2 種類のデータを含みます：

1. **映像特徴量（Features）**: [V-JEPA 2.1](https://github.com/facebookresearch/vjepa) ViT-L（frozen）で抽出した時空間トークン（PyTorch `.pt` 形式）
2. **合成アノテーション（Annotations）**: [Qwen 3.5](https://huggingface.co/Qwen) VLM による自動生成アノテーション（JSON 形式）。オブジェクト検出・イベント（行動）記述を含む

> **Note**: このデータセットは全体の一部（{subset['included_videos']}/{subset['total_videos_available']} 動画、{subset['included_clips']} クリップ）です。完全版が必要な場合はリポジトリから再生成できます。

## 2. データの内容・件数

| 項目 | 値 |
|---|---|
| 動画数 | {subset['included_videos']} / {subset['total_videos_available']} |
| クリップ数（16フレーム/クリップ） | {subset['included_clips']} |
| 特徴量サイズ合計 | {subset['total_feature_size_gb']} GB |
| アノテーション JSON | {subset['included_videos']} ファイル |
| V-JEPA backbone | `{metadata['vjepa_model']}` |
| 特徴量次元 | {metadata['feature_dim']} |
| トークン数/クリップ | {metadata['num_tokens_per_clip']}（{metadata['temporal_tokens']} temporal x {metadata['spatial_tokens']} spatial） |
| 入力画像サイズ | {metadata['image_size']}px |
| クリップ長 | {metadata['clip_length_frames']} frames @ {metadata['sampling_fps']} FPS |
| オブジェクトカテゴリ数 | {metadata['num_object_categories']} |
| アクションクラス数 | {metadata['num_action_classes']} |

### 収録動画一覧

| Video ID | クリップ数 | アノテーション済み | イベントあり | 特徴量サイズ |
|---|---|---|---|---|
{videos_table}

## 3. フォーマット・カラム説明

### 3.1 特徴量ファイル（`features/<video_id>/clip_XXXX.pt`）

**フォーマット**: PyTorch `.pt` ファイル（`torch.load()` で読み込み）

| キー | 型 | Shape / 値 | 説明 |
|---|---|---|---|
| `vjepa_tokens` | `Tensor` (float32) | `(4608, 1024)` | V-JEPA 2.1 ViT-L が出力した時空間トークン。8 temporal x 576 spatial tokens |
| `video_id` | `str` | `"20260316_130406_tp00001"` | 動画の一意識別子（録画日時 + 連番） |
| `clip_index` | `int` | `0` ~ | クリップの連番（動画内での順序） |
| `frame_indices` | `list[int]` | 長さ 16 | 元動画のフレームインデックス（例: `[0, 20, 40, ...]`） |

### 3.2 アノテーションファイル（`annotations/<video_id>.json`）

**フォーマット**: JSON

**トップレベル構造**:

| キー | 型 | 説明 |
|---|---|---|
| `video_id` | `str` | 動画 ID |
| `video_path` | `str` | 元動画のパス |
| `video_metadata` | `object` | `source_fps`, `target_fps`, `duration_sec`, `video_start_time`, `video_end_time` 等 |
| `coverage` | `object` | `total_clips`, `annotated_clips`, `motion_filtered_clips`, `clips_with_events` 等の統計 |
| `num_clips` | `int` | 総クリップ数 |
| `validation_stats` | `object` | VLM 出力のバリデーション統計（`discard_rate` 等） |
| `clips` | `list[object]` | 各クリップのアノテーション |

**`clips[i]` の構造**:

| キー | 型 | 説明 |
|---|---|---|
| `objects` | `list[object]` | 検出されたオブジェクト一覧 |
| `events` | `list[object]` | イベント（行動）一覧 |
| `clip_metadata` | `object` | `clip_index`, `frame_indices`, `start_time`, `end_time`, `status` |

**`objects[j]` のフィールド**:

| キー | 型 | 説明 |
|---|---|---|
| `obj_id` | `str` | オブジェクト ID（例: `"person_01"`, `"laptop_01"`） |
| `category` | `str` | カテゴリ名（28 クラスのいずれか） |
| `first_seen_frame` | `int` | オブジェクトが初めて出現するフレーム (0-15) |
| `attributes` | `object` | `color`, `material`, `position`, `size`, `state`, `orientation`, `pose`（null 許容） |

**`events[k]` のフィールド**:

| キー | 型 | 説明 |
|---|---|---|
| `event_id` | `str` | イベント ID（例: `"evt_001"`） |
| `frame` | `int` | イベント発生フレーム (0-15) |
| `action` | `str` | アクション名（13 クラスのいずれか） |
| `agent` | `str` | 行為者の `obj_id` |
| `target` | `str` | 対象物の `obj_id` |
| `source` | `str` or `null` | 取り出し元の `obj_id`（アクションに応じて任意） |
| `destination` | `str` or `null` | 格納先の `obj_id`（アクションに応じて任意） |

### 3.3 圧縮アノテーション（`annotations_compact/<video_id>.json`）

フルアノテーションと同一内容を構造的に圧縮したバージョン。オブジェクト定義の重複排除、短縮キー、null 省略により 93% のサイズ削減を実現。ロスレスで相互変換可能。

### 3.4 設定ファイル（`configs/`）

| ファイル | 説明 |
|---|---|
| `vocab.yaml` | 28 オブジェクトカテゴリと属性語彙の定義 |
| `actions.yaml` | 13 アクションクラスの定義（`source`/`destination` 要否フラグ付き） |

### アクション語彙 (13 classes)

| Action | Description | Source | Destination |
|---|---|---|---|
| take_out | 物体を容器/収納から取り出す | required | - |
| put_in | 物体を容器/収納に入れる | - | required |
| place_on | 物体を面の上に置く | - | required |
| pick_up | 面の上から物体を持ち上げる | required | - |
| hand_over | 人から人へ物体を渡す | required | required |
| open | 容器/引き出し/蓋を開ける | - | - |
| close | 容器/引き出し/蓋を閉める | - | - |
| use | 工具/道具を使用する | - | - |
| move | 物体を場所Aから場所Bへ移動する | required | required |
| attach | 物体を別の物体に取り付ける | - | required |
| detach | 物体を別の物体から取り外す | required | - |
| inspect | 物体を視認/確認する | - | - |
| no_event | イベントなし (negative class) | - | - |

### オブジェクトカテゴリ (28 classes)

person, hand, chair, desk, laptop, monitor, phone, keyboard, mouse, tablet, pen, notebook, book, bookshelf, shelf, cup, drawer, curtain, jacket, backpack, box, speaker, microphone, stool, pc_case, earbuds, smartphone, case

## 4. データの取得元・作成方法

### 出典

室内環境（オフィスデスク・キッチン・部屋）で撮影された録画動画。固定カメラ（20 FPS）で撮影。

### 特徴量の抽出方法

1. 元動画を 1 FPS でフレームサンプリング
2. 16 フレーム単位のクリップに分割（50% オーバーラップ、stride=8）
3. 各クリップを [V-JEPA 2.1](https://github.com/facebookresearch/vjepa) ViT-L（384px、frozen）に入力
4. 出力の時空間トークン（4608 tokens x 1024 dim）を `.pt` ファイルとして保存

### アノテーション方法

**VLM 合成アノテーション**（人手ラベルなし）:

1. 各クリップの 16 フレーム画像を [Qwen 3.5 VLM](https://huggingface.co/Qwen) に入力
2. 構造化プロンプトにより、オブジェクト（カテゴリ・属性）とイベント（アクション・行為者・対象）を JSON で出力させる
3. スキーマバリデーションにより不正な出力を除外
4. **モーションフィルター**: 静止シーン（オブジェクト・イベントなし）はフィルタリング

### 前処理

- フレームは 384x384 px にリサイズ（V-JEPA 入力サイズ）
- V-JEPA トークンは float32 で保存（抽出時は bfloat16、保存時に float32 変換）
- アノテーション JSON はバリデーション済み（カテゴリ・アクション語彙外の値は除外）

## 5. 想定ユースケース

- **イベントグラフ予測モデルの学習**: 本データセットの主目的。V-JEPA トークンを入力とし、構造化イベントグラフを予測する Event Decoder の学習に使用
- **動画理解の研究**: 時空間トークンとイベントアノテーションを用いた動画理解手法の開発・評価
- **合成アノテーションの品質評価**: VLM 自動生成アノテーションの精度・特性の分析
- **Transfer Learning**: V-JEPA 特徴量をファインチューニングなしで下流タスクに転用する研究

## 6. 注意点・制約

### バイアス

- **ドメインバイアス**: 学習データは室内環境（オフィス・キッチン・部屋）に限定。屋外・工場・医療等の環境には未対応
- **VLM バイアス**: アノテーションは Qwen 3.5 VLM の出力に依存。VLM 自体のバイアス（認識しやすい/しにくいオブジェクト、文化的偏り）を継承する可能性がある
- **カテゴリバイアス**: 28 カテゴリに限定されており、語彙外のオブジェクトは検出されない
- **アクションバイアス**: 13 アクションクラスは室内の日常行動に特化。製造作業や屋外行動は対象外

### 欠損

- **モーションフィルタリング**: 静止シーン（動きがないクリップ）は `objects=[], events=[]` としてアノテーションされる（`status: "motion_filtered"`）。これらのクリップには特徴量は存在するがイベント情報がない
- **VLM 出力の除外**: スキーマバリデーションで不正と判定された VLM 出力は除外される（`validation_stats.discard_rate` で確認可能）
- **部分データセット**: 全 {subset['total_videos_available']} 動画のうち {subset['included_videos']} 動画のみ収録

### 利用上の注意

- `.pt` ファイルの読み込みには **PyTorch** が必要（`torch.load(..., weights_only=False)` を使用）
- 特徴量は V-JEPA 2.1 ViT-L 固有のもの。他のバックボーン（ViT-B, ViT-g 等）の特徴量とは互換性がない
- アノテーションは**人手レビューを経ていない**合成データ。研究・開発目的の利用を推奨
- 1 FPS サンプリングのため、1 秒未満の高速なイベントは記録されていない可能性がある
- 動画には人物が映っている。プライバシーに配慮した利用が求められる

## 7. 使い方

### ダウンロード

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    local_dir="data/hf_dataset",
)
```

### 特徴量 + アノテーションの読み込み

```python
import torch
import json

# V-JEPA 特徴量の読み込み（1 クリップ分）
features = torch.load(
    "data/hf_dataset/features/20260316_130406_tp00001/clip_0000.pt",
    map_location="cpu",
    weights_only=False,
)
vjepa_tokens = features["vjepa_tokens"]  # shape: (4608, 1024)
video_id = features["video_id"]          # "20260316_130406_tp00001"
clip_index = features["clip_index"]      # 0

# 対応するアノテーションの読み込み
with open("data/hf_dataset/annotations/20260316_130406_tp00001.json") as f:
    annotation = json.load(f)

clip = annotation["clips"][clip_index]
print(clip["objects"])   # 検出オブジェクト一覧
print(clip["events"])    # イベントグラフ（行動記述）
```

### Event Graph Generation で学習

```bash
git clone https://github.com/ChanYu1224/event-graph-generation.git
cd event-graph-generation
uv sync

# ダウンロードしたデータから学習用データを構築
uv run python scripts/4b_build_vjepa_dataset.py \\
    --features-dir data/hf_dataset/features \\
    --annotations-dir data/hf_dataset/annotations \\
    --output-dir data/vjepa_aligned

# 学習
uv run python scripts/5_train.py --config configs/vjepa_training.yaml
```

## 8. ライセンス

MIT License

Copyright (c) 2026 Yuchn

本データセットは MIT ライセンスの下で公開されています。商用・非商用を問わず、自由に使用・改変・再配布が可能です。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 9. 引用

```bibtex
@software{{event_graph_generation_2026,
  title = {{Event Graph Generation: Structured Event Prediction from Video}},
  author = {{Yuchn}},
  year = {{2026}},
  url = {{https://github.com/ChanYu1224/event-graph-generation}},
  license = {{MIT}}
}}
```

## 10. 関連リンク

- **Repository**: [ChanYu1224/event-graph-generation](https://github.com/ChanYu1224/event-graph-generation)
- **V-JEPA**: [facebookresearch/vjepa](https://github.com/facebookresearch/vjepa)
- **Qwen VLM**: [Qwen](https://huggingface.co/Qwen)
"""


def _prepare_staging_dir(
    tmpdir: Path,
    selected_videos: list[dict],
    annotations_dir: Path,
    compact_dir: Path,
    configs_dir: Path,
    metadata: dict,
    repo_id: str,
) -> None:
    """Prepare staging directory with lightweight files.

    Args:
        tmpdir: Temporary directory to stage files.
        selected_videos: List of selected video info dicts.
        annotations_dir: Source annotation directory.
        compact_dir: Source compact annotation directory.
        configs_dir: Source configs directory.
        metadata: Dataset metadata dict.
        repo_id: Full HF repo ID.
    """
    # README
    readme = _generate_dataset_card(metadata, repo_id)
    (tmpdir / "README.md").write_text(readme)

    # LICENSE
    (tmpdir / "LICENSE").write_text(MIT_LICENSE)

    # metadata.json
    with open(tmpdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # configs
    cfg_dir = tmpdir / "configs"
    cfg_dir.mkdir()
    for name in ("vocab.yaml", "actions.yaml"):
        src = configs_dir / name
        if src.exists():
            shutil.copy2(src, cfg_dir / name)

    # annotations
    ann_dir = tmpdir / "annotations"
    ann_dir.mkdir()
    for v in selected_videos:
        src = annotations_dir / f"{v['video_id']}.json"
        if src.exists():
            shutil.copy2(src, ann_dir / src.name)

    # compact annotations
    compact_out = tmpdir / "annotations_compact"
    compact_out.mkdir()
    for v in selected_videos:
        src = compact_dir / f"{v['video_id']}.json"
        if src.exists():
            shutil.copy2(src, compact_out / src.name)


def _human_size(nbytes: int) -> str:
    """Format byte count as human-readable string.

    Args:
        nbytes: Number of bytes.

    Returns:
        Human-readable size string.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def upload_dataset(
    namespace: str,
    repo_name: str,
    features_dir: Path,
    annotations_dir: Path,
    compact_dir: Path,
    configs_dir: Path,
    max_videos: int,
    dry_run: bool = False,
    private: bool = False,
) -> str:
    """Upload dataset to HuggingFace Hub.

    Args:
        namespace: HuggingFace namespace (user or org).
        repo_name: Repository name.
        features_dir: Path to V-JEPA features directory.
        annotations_dir: Path to annotation JSON directory.
        compact_dir: Path to compact annotation directory.
        configs_dir: Path to configs directory.
        max_videos: Maximum number of videos to include.
        dry_run: If True, only print what would be uploaded.
        private: If True, create a private repository.

    Returns:
        URL of the created dataset repo (or dry-run message).
    """
    repo_id = f"{namespace}/{repo_name}"

    # Discover and select videos
    all_videos = _discover_videos(features_dir, annotations_dir, compact_dir)
    logger.info(
        "Discovered %d videos with both features and annotations",
        len(all_videos),
    )

    selected = all_videos[:max_videos]
    logger.info("Selected %d videos for upload", len(selected))

    # Build metadata
    metadata = _build_metadata(selected, total_available=len(all_videos))

    # Print summary
    total_feature_size = sum(v["feature_size_bytes"] for v in selected)
    total_ann_size = sum(v["annotation_size_bytes"] for v in selected)
    total_clips = sum(v["num_clips"] for v in selected)

    print("\n=== Dataset Upload Summary ===")
    print(f"  Repository: {repo_id}")
    print(f"  Videos: {len(selected)} / {len(all_videos)}")
    print(f"  Total clips: {total_clips}")
    print(f"  Feature size: {_human_size(total_feature_size)}")
    print(f"  Annotation size: {_human_size(total_ann_size)}")
    print()

    for v in selected:
        print(
            f"  {v['video_id']}: {v['num_clips']} clips, "
            f"{_human_size(v['feature_size_bytes'])}, "
            f"{v['clips_with_events']} clips with events"
        )

    if dry_run:
        print("\n(dry-run) No files uploaded.")
        return f"(dry-run) {repo_id}"

    # Phase 1: Prepare and upload staging directory (annotations, configs, README)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _prepare_staging_dir(
            tmpdir, selected, annotations_dir, compact_dir, configs_dir,
            metadata, repo_id,
        )

        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=private,
        )
        logger.info("Created dataset repo: %s", repo_id)

        logger.info("Uploading metadata, annotations, and configs...")
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card, annotations, and configs",
        )
        logger.info("Phase 1 complete: metadata uploaded")

    # Phase 2: Upload features
    allow_patterns = [f"{v['video_id']}/*" for v in selected]
    logger.info(
        "Uploading features for %d videos (%s)...",
        len(selected),
        _human_size(total_feature_size),
    )
    api = HfApi()
    api.upload_folder(
        folder_path=str(features_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        path_in_repo="features",
        commit_message=f"Add V-JEPA ViT-L features ({len(selected)} videos)",
    )
    logger.info("Phase 2 complete: features uploaded")

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Dataset uploaded to %s", url)
    return url


def main() -> None:
    """CLI entrypoint for dataset upload."""
    parser = argparse.ArgumentParser(
        description="Upload V-JEPA features and VLM annotations to Hugging Face"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="Yuchn",
        help="HuggingFace namespace (user or org)",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="event-graph-vjepa-vitl-dataset",
        help="Dataset repository name",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/vjepa_features_v21_vitl",
        help="Directory containing V-JEPA feature subdirectories",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/annotations",
        help="Directory containing annotation JSON files",
    )
    parser.add_argument(
        "--compact-dir",
        type=str,
        default="data/annotations_compact",
        help="Directory containing compact annotation JSON files",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs",
        help="Directory containing vocab.yaml and actions.yaml",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=5,
        help="Maximum number of videos to upload (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset repository",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    url = upload_dataset(
        namespace=args.namespace,
        repo_name=args.repo_name,
        features_dir=Path(args.features_dir),
        annotations_dir=Path(args.annotations_dir),
        compact_dir=Path(args.compact_dir),
        configs_dir=Path(args.configs_dir),
        max_videos=args.max_videos,
        dry_run=args.dry_run,
        private=args.private,
    )

    print(f"\n=== Result ===\n  {url}")


if __name__ == "__main__":
    main()
