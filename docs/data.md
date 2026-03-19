# データモジュール (`data/`)

フレームサンプリング、データセット管理、バッチ collation を行う。

## FrameSampler (`frame_sampler.py`)

OpenCV を使って動画から指定 FPS でフレームを抽出する。

### 使い方

```python
sampler = FrameSampler(target_fps=1.0)

# 動画全体からサンプリング
frames = sampler.sample("data/raw/video.mp4")
# frames: list[SampledFrame]

# 特定位置からクリップをサンプリング
clip = sampler.sample_clip("data/raw/video.mp4", start_frame=100, num_frames=16)

# 動画のメタ情報を取得
info = FrameSampler.get_video_info("data/raw/video.mp4")
# {"fps": 30.0, "total_frames": 9000, "width": 1920, "height": 1080, "duration_sec": 300.0}
```

### SampledFrame

| フィールド | 型 | 説明 |
|---|---|---|
| `image` | `np.ndarray (H,W,3)` | BGR 画像 |
| `frame_index` | `int` | 元動画でのフレーム番号 |
| `timestamp_sec` | `float` | 秒単位のタイムスタンプ |

### サンプリングロジック

```
frame_interval = round(source_fps / target_fps)
```

例: source_fps=30, target_fps=1.0 → 30 フレームに 1 枚抽出

## EventGraphDataset (`event_dataset.py`)

学習データ（`.pt` ファイル）を読み込む PyTorch Dataset。`BaseDataset` を継承。

### データディレクトリ構造

```
data/aligned/
├── samples/
│   ├── video_001.pt
│   ├── video_002.pt
│   └── ...
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── meta.json
```

### 初期化

```python
dataset = EventGraphDataset(
    data_dir="data/aligned",
    split="train",       # train / val / test
    max_objects=30,      # パディングサイズ K
)
```

`splits/train.txt` から sample ID を読み込み、`samples/{id}.pt` からデータをロードする。

### `__getitem__` の出力

| キー | 形状 | 説明 |
|---|---|---|
| `object_embeddings` | `(K, D_emb)` | SAM 3 埋め込み（K にパディング） |
| `object_temporal` | `(K, T, D_geo)` | 幾何特徴量（K にパディング） |
| `pairwise` | `(K, K, T, D_pair)` | ペアワイズ特徴量（K にパディング） |
| `object_mask` | `(K,)` bool | 有効オブジェクトのマスク |
| `gt_events` | `list[dict]` | 正解イベントのリスト |
| `num_objects` | `int` | 実際のオブジェクト数 N |

**パディング**: 実際のオブジェクト数 N が `max_objects` (K=30) より少ない場合、テンソルをゼロパディングし、`object_mask` で有効範囲を示す。

### .pt ファイルの構造

`build_dataset.py` が生成するサンプルファイルの内容:

```python
{
    "video_id": "video_001",
    "object_embeddings": Tensor (N, 256),
    "object_temporal": Tensor (N, 16, 12),
    "pairwise": Tensor (N, N, 16, 7),
    "gt_events": [
        {
            "agent_track_id": 0,      # スロット index（track_id ではない）
            "action_class": 3,         # actions.yaml での index
            "target_track_id": 1,
            "source_track_id": 2,      # None の場合あり
            "dest_track_id": None,
            "event_frame_index": 5,    # T 内での frame index
        },
        ...
    ],
    "track_ids": [0, 3, 5, 7],        # スロット index → 実際の track_id
    "alignment": {
        "mapping": {"person_01": 0, "wrench_01": 3},
        "confidence": 0.85,
    },
}
```

**注意**: `gt_events` 内の `agent_track_id` 等はスロット index（0 から N-1）であり、SAM 3 の `track_id` とは異なる。`track_ids` リストで変換可能。

## EventBatch / event_collate_fn (`event_collator.py`)

`EventGraphDataset` の出力をバッチにまとめる。

### EventBatch

| フィールド | 形状 | 説明 |
|---|---|---|
| `object_embeddings` | `(B, K, D_emb)` | バッチ化された埋め込み |
| `object_temporal` | `(B, K, T, D_geo)` | バッチ化された幾何特徴 |
| `pairwise` | `(B, K, K, T, D_pair)` | バッチ化されたペアワイズ特徴 |
| `object_mask` | `(B, K)` | バッチ化されたマスク |
| `gt_events` | `list[list[dict]]` | B 個のイベントリスト |
| `num_objects` | `list[int]` | 各サンプルのオブジェクト数 |

**`to(device)`** メソッドでテンソルフィールドをデバイスに移動。`gt_events` と `num_objects` は CPU に残る。

### DataLoader での使用

```python
from torch.utils.data import DataLoader
from event_graph_generation.data.event_collator import event_collate_fn

loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=event_collate_fn,
    shuffle=True,
)
```
