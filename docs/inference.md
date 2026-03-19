# 推論モジュール (`inference/`)

動画ファイルから EventGraph JSON を生成する End-to-End 推論パイプライン。

## InferencePipeline (`pipeline.py`)

フレームサンプリング → SAM 3 追跡 → 特徴抽出 → Event Decoder 推論 → 重複排除 → EventGraph 構築を一貫して実行する。

### 初期化

```python
pipeline = InferencePipeline(
    sam3_tracker=sam3_tracker,
    feature_extractor=feature_extractor,
    event_decoder=event_decoder,
    frame_sampler=frame_sampler,
    config={
        "clip_length": 16,
        "clip_stride": 8,
        "max_objects": 30,
        "d_geo": 12,
        "d_pair": 7,
        "device": "cuda",
        "action_names": ["take_out", "put_in", ...],
        "dedup_frame_threshold": 3,
    },
)
```

### 推論実行

```python
event_graph = pipeline.process_video(
    video_path="data/raw/video.mp4",
    concept_prompts=["person", "wrench", "screwdriver", "drawer"],
    confidence_threshold=0.5,
)
event_graph.to_json("output/event_graph.json")
```

### 処理フロー

#### 1. フレームサンプリング

`FrameSampler` で `target_fps`（デフォルト 1.0）でフレームを抽出。

#### 2. オブジェクト追跡

`SAM3Tracker` に `concept_prompts` を設定し、全フレームを追跡。各フレームで `FrameTrackingResult`（オブジェクトの bbox, mask, embedding）を得る。

#### 3. 特徴抽出

`FeatureExtractor` で `ObjectFeatures`（per-track）と `PairwiseFeatures`（per-pair）を計算。

#### 4. スライディングウィンドウ推論

時間方向をスライディングウィンドウで分割し、各ウィンドウで Event Decoder を実行。

```
全フレーム: |-------- T_total --------|
Window 0:   |-- clip_length --|
Window 1:        |-- clip_length --|
Window 2:             |-- clip_length --|
            <-- clip_stride -->
```

各ウィンドウの処理:
1. `ObjectFeatures` から `object_embeddings` テンソルを構築
2. 幾何特徴量を `object_temporal` テンソルにパック（12 次元: bbox4 + centroid2 + area1 + presence1 + deltas4）
3. `PairwiseFeatures` から `pairwise` テンソルを構築
4. `max_objects` にパディング、デバイスに転送
5. `torch.no_grad()` で Event Decoder を実行
6. `predictions_to_events()` で event dict に変換

#### 5. 重複排除

スライディングウィンドウの重なりから生じる重複イベントを除去。

**重複の判定条件:**
- 同じ `action`
- 同じ `agent_track_id`
- 同じ `target_track_id`
- フレーム差が `frame_threshold` 以下

信頼度の高い方を優先的に残す（greedy NMS 方式）。

#### 6. EventGraph 構築

追跡されたオブジェクト情報とイベントから `EventGraph` を構築。

### カテゴリ ID のマッピング

`FeatureExtractor` はカテゴリをソート済みユニーク名で ID に変換する。推論パイプラインも同じ方式でカテゴリ名を復元する:

```python
unique_categories = sorted({obj.category for fr in tracking_results for obj in fr.objects})
cat_id_to_name = {i: name for i, name in enumerate(unique_categories)}
```

## postprocess.py

モデルのテンソル出力をイベント dict と EventGraph に変換する。

### predictions_to_events

```python
events = predictions_to_events(
    predictions=preds,          # EventPredictions (batch_size=1)
    track_id_map={0: 5, 1: 12, ...},  # slot_index → track_id
    action_names=["take_out", "put_in", ...],
    frame_indices=[0, 30, 60, ...],
    confidence_threshold=0.5,
)
```

各 event query スロットに対して:
1. `sigmoid(interaction)` が閾値以上かチェック
2. `argmax(action)` でアクション決定
3. `argmax(agent_ptr)`, `argmax(target_ptr)` でポインタ決定
4. `argmax(source_ptr)`, `argmax(dest_ptr)` でオプショナルポインタ決定（index K = "none"）
5. `argmax(frame)` でフレーム決定、実際のフレーム番号にマッピング

### build_event_graph

```python
graph = build_event_graph(
    video_id="video_001",
    tracked_objects=[{"track_id": 0, "category": "person", ...}, ...],
    events=[{"action": "pick_up", "agent_track_id": 0, ...}, ...],
    metadata={"num_frames_sampled": 120},
)
```

dict リストから `ObjectNode` と `EventEdge` を構築し、`EventGraph` を返す。

## CLI エントリポイント (`scripts/7_run_inference.py`)

```bash
uv run python scripts/7_run_inference.py \
  --config configs/inference.yaml \
  --video data/raw/video.mp4 \
  --checkpoint data/checkpoints/best.pt \
  --output output/event_graph.json \
  --concept-prompts person wrench drawer \
  --confidence-threshold 0.5 \
  --actions-config configs/actions.yaml
```

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--config` | `configs/inference.yaml` | 推論設定 |
| `--video` | (必須) | 入力動画 |
| `--checkpoint` | (必須) | 学習済みチェックポイント |
| `--output` | `output/event_graph.json` | 出力 JSON パス |
| `--concept-prompts` | 設定ファイルから | SAM 3 検出プロンプト |
| `--confidence-threshold` | 0.5 | 信頼度閾値 |
| `--actions-config` | `configs/actions.yaml` | アクション語彙 |

実行後、検出されたオブジェクトとイベントのサマリーがコンソールに出力される。
