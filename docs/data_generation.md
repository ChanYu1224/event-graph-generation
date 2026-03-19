# データ生成手順

学習データの構築は 3 つのステップで行う。各ステップは独立したスクリプトとして実行し、中間成果物はファイルに保存される。

## 全体フロー

```
Step 1: SAM 3 トラッキング
  入力: data/videos/*.mp4
  出力: data/sam3_outputs/*.pt (FrameTrackingResult + ObjectFeatures + PairwiseFeatures)

Step 2: VLM アノテーション
  入力: data/videos/*.mp4
  出力: data/annotations/*.json (VLMAnnotation per video)

Step 3: アライメント & データセット構築
  入力: data/sam3_outputs/*.pt + data/annotations/*.json
  出力: data/aligned/samples/*.pt + data/aligned/splits/{train,val,test}.txt
```

## Step 1: SAM 3 トラッキング

動画中のオブジェクトを SAM 3 で検出・追跡し、特徴量を抽出する。

### 実行

```bash
uv run python scripts/2_run_sam3_tracking.py \
  --config configs/sam3.yaml \
  --video-dir data/videos \
  --output-dir data/sam3_outputs \
  --resume
```

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--config` | `configs/sam3.yaml` | SAM 3 設定（model_size, concept_prompts 等） |
| `--video-dir` | `data/videos` | 動画ディレクトリ |
| `--output-dir` | `data/sam3_outputs` | 出力ディレクトリ |
| `--resume` | - | 処理済み動画をスキップ |
| `--shard-id` | - | マルチ GPU 並列処理のシャード ID |
| `--num-shards` | - | シャード総数 |

### 処理内容

1. 指定ディレクトリ内の動画ファイル (`.mp4`, `.avi`, `.mov`) を発見
2. 各動画のフレームを読み込み（`sample_rate` でサブサンプリング可能）
3. `SAM3Tracker` で `concept_prompts` に基づくオブジェクト追跡を実行
4. `FeatureExtractor` で `ObjectFeatures` と `PairwiseFeatures` を計算
5. 追跡結果と特徴量を `.pt` ファイルとして保存

### 出力ファイル

各動画につき 1 つの `.pt` ファイルを生成:

```python
{
    "video_id": "assembly_001",
    "video_path": "data/videos/assembly_001.mp4",
    "frame_indices": [0, 30, 60, ...],
    "tracking_results": [FrameTrackingResult, ...],
    "object_features": {track_id: ObjectFeatures, ...},
    "pairwise_features": [PairwiseFeatures, ...],
}
```

### マルチ GPU 並列処理

`--shard-id` と `--num-shards` で動画を分割して複数 GPU で並列処理可能:

```bash
# GPU 0
uv run python scripts/2_run_sam3_tracking.py --shard-id 0 --num-shards 4
# GPU 1
uv run python scripts/2_run_sam3_tracking.py --shard-id 1 --num-shards 4
# ...
```

### concept_prompts の設定

`configs/sam3.yaml` で検出対象を指定:

```yaml
sam3:
  concept_prompts:
    - "person"
    - "wrench"
    - "screwdriver"
    - "drawer"
    - "workbench"
    - "toolbox"
    - "hammer"
    - "pliers"
```

## Step 2: VLM アノテーション

ローカルの Qwen 3.5 で動画フレームからイベントアノテーションを生成する。

### 実行

```bash
uv run python scripts/3_generate_annotations.py \
  --config configs/vlm.yaml \
  --video-dir data/videos \
  --output-dir data/annotations \
  --actions-config configs/actions.yaml \
  --resume
```

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--config` | `configs/vlm.yaml` | VLM 設定 |
| `--video-dir` | `data/videos` | 動画ディレクトリ |
| `--output-dir` | `data/annotations` | アノテーション出力先 |
| `--actions-config` | `configs/actions.yaml` | アクション語彙 |
| `--resume` | - | 処理済み動画をスキップ |

### 処理内容

1. 動画ファイルを発見
2. `VLMAnnotator` を初期化（Qwen 3.5 モデルの読み込み）
3. 各動画に対してスライディングウィンドウでアノテーション:
   - `FrameSampler` で `target_fps` でフレーム抽出
   - `clip_length` / `clip_stride` でクリップに分割
   - 各クリップに対して VLM を実行
4. `AnnotationValidator` で品質チェック:
   - 語彙のバリデーション
   - 参照整合性チェック
   - 不正なイベントの除去
5. JSON ファイルとして保存

### 出力ファイル

各動画につき 1 つの `.json` ファイルを生成:

```json
{
  "video_id": "assembly_001",
  "video_path": "data/videos/assembly_001.mp4",
  "num_clips": 5,
  "validation_stats": {
    "total_annotations": 5,
    "total_events_input": 23,
    "total_events_output": 20,
    "total_events_discarded": 3,
    "discard_rate": 0.1304,
    "total_warnings": 5
  },
  "clips": [
    {
      "objects": [
        {"obj_id": "person_01", "category": "person", "first_seen_frame": 0, "attributes": ["standing"]},
        {"obj_id": "wrench_01", "category": "wrench", "first_seen_frame": 3, "attributes": ["metal"]}
      ],
      "events": [
        {"event_id": "evt_001", "frame": 5, "action": "pick_up", "agent": "person_01", "target": "wrench_01", "source": "drawer_01"}
      ]
    }
  ]
}
```

### VLM 設定の調整

`configs/vlm.yaml` の主要パラメータ:

```yaml
vlm:
  model_name: "Qwen/Qwen3.5-9B"
  torch_dtype: "bfloat16"     # VRAM 節約のため bfloat16
  temperature: 0.1            # 低めで安定した出力
  clip_length: 16             # 1クリップあたりのフレーム数
  clip_stride: 8              # オーバーラップ 50%
  target_fps: 1.0             # 1秒1フレーム
```

## Step 3: アライメント & データセット構築

SAM 3 の追跡結果と VLM のアノテーションを結合し、学習用データセットを構築する。

### 実行

```bash
uv run python scripts/4_build_dataset.py \
  --sam3-dir data/sam3_outputs \
  --annotations-dir data/annotations \
  --output-dir data/aligned \
  --actions-config configs/actions.yaml \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --iou-threshold 0.3 \
  --temporal-window 16
```

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--sam3-dir` | `data/sam3_outputs` | SAM 3 出力 |
| `--annotations-dir` | `data/annotations` | VLM アノテーション |
| `--output-dir` | `data/aligned` | 出力先 |
| `--actions-config` | `configs/actions.yaml` | アクション語彙 |
| `--val-ratio` | 0.1 | 検証セットの割合 |
| `--test-ratio` | 0.1 | テストセットの割合 |
| `--seed` | 42 | 分割の乱数シード |
| `--iou-threshold` | 0.3 | アライメントの IoU 閾値 |
| `--temporal-window` | 16 | 時系列ウィンドウ長 |

### 処理内容

1. SAM 3 `.pt` と VLM `.json` をファイル名でマッチング
2. 各ペアに対して:
   a. SAM 3 の追跡結果を読み込み
   b. VLM のアノテーションを読み込み
   c. `Aligner` で VLM オブジェクト → SAM 3 トラック ID のマッピングを計算
   d. `FeatureExtractor` で特徴量を抽出
   e. VLM イベントの agent/target を SAM 3 のスロット index に変換
   f. アクション名を `actions.yaml` のクラス index に変換
   g. `.pt` ファイルとして保存
3. ランダムに train/val/test に分割して `splits/` に保存
4. 統計情報を `meta.json` に保存

### 出力ディレクトリ構造

```
data/aligned/
├── samples/
│   ├── assembly_001.pt     # 学習サンプル
│   ├── assembly_002.pt
│   └── ...
├── splits/
│   ├── train.txt           # sample ID のリスト
│   ├── val.txt
│   └── test.txt
└── meta.json               # 統計情報
```

### GT イベントの構造

アライメント後の GT イベント:

```python
{
    "agent_track_id": 0,       # スロット index（sorted track_ids 内の位置）
    "action_class": 3,          # actions.yaml でのクラス index
    "target_track_id": 1,      # スロット index
    "source_track_id": 2,      # スロット index or None
    "dest_track_id": None,     # スロット index or None
    "event_frame_index": 5,    # temporal_window 内でのフレーム index (0-15)
}
```

**注意点:**
- `agent_track_id` 等は SAM 3 の `track_id` ではなく、ソート済みトラック ID リスト内のスロット index
- `event_frame_index` は `temporal_window` の範囲に clamp される（`min(frame, T-1)`）
- `actions.yaml` にないアクションのイベントはスキップされる

### meta.json

```json
{
  "total_videos": 100,
  "successful_alignments": 92,
  "total_events": 450,
  "total_objects": 680,
  "skipped": 8,
  "train_count": 74,
  "val_count": 9,
  "test_count": 9
}
```

## トラブルシューティング

### SAM 3 が利用できない

`sam3` パッケージが未インストールの場合、`SAM3Tracker` は警告を出して初期化される。`track_video()` の呼び出し時に `RuntimeError` が発生する。

### VLM の出力パースに失敗

`VLMAnnotator._parse_output()` がパースに失敗した場合、空の `VLMAnnotation` が返される。`temperature` を下げるか、`max_new_tokens` を増やすことで改善する場合がある。

### アライメントが見つからない

VLM のカテゴリ名と SAM 3 の concept_prompts が一致していない場合、アライメントに失敗する。`concept_prompts` と VLM のプロンプトに渡す `categories` リストを揃える必要がある。

### VRAM 不足

- SAM 3: `model_size` を `"base"` や `"small"` に変更
- VLM: `quantization` を `"4bit"` に設定、または `torch_dtype` を `"float16"` に変更
- 学習: `batch_size` を削減
