# トラッキングモジュール (`tracking/`)

SAM 3 によるオブジェクト検出・追跡と、学習用特徴量の抽出を行う。

## SAM3Tracker (`sam3_tracker.py`)

SAM 3 のラッパークラス。テキストプロンプトによるオブジェクト検出・セグメンテーション・追跡を行う。

### 初期化

```python
tracker = SAM3Tracker(model_size="large", device="cuda")
tracker.set_concept_prompts(["person", "wrench", "screwdriver", "drawer"])
```

- `model_size`: `"large"`, `"base"`, `"small"` から選択
- `concept_prompts`: 検出対象のカテゴリをテキストで指定

### 推論

```python
tracking_results = tracker.track_video(frames, frame_indices)
# tracking_results: list[FrameTrackingResult]
```

- GPU メモリ管理のため、フレームを 1 枚ずつ処理
- 各フレームの出力をキャッシュクリアして VRAM を解放

### データ構造

**TrackedObject** — 1 フレーム内の 1 つの追跡オブジェクト:

| フィールド | 型 | 説明 |
|---|---|---|
| `track_id` | `int` | SAM 3 が割り当てたトラック ID（フレーム間で一貫） |
| `category` | `str` | カテゴリ名 |
| `mask` | `np.ndarray (H,W)` | セグメンテーションマスク（bool） |
| `bbox` | `np.ndarray [x1,y1,x2,y2]` | バウンディングボックス |
| `score` | `float` | 検出スコア |
| `embedding` | `torch.Tensor (256,)` | SAM 3 の DETR decoder 出力埋め込み |

**FrameTrackingResult** — 1 フレームの追跡結果:

| フィールド | 型 | 説明 |
|---|---|---|
| `frame_index` | `int` | 元動画でのフレーム番号 |
| `objects` | `list[TrackedObject]` | そのフレームで検出されたオブジェクト |

### 埋め込みの抽出

`_extract_embedding()` は SAM 3 の内部表現から 256 次元のオブジェクト埋め込みを抽出する。`decoder_embedding` → `embedding` の順で属性を探索し、256 次元にパディング/トランケートする。抽出失敗時はゼロベクトルを返す。

## FeatureExtractor (`feature_extractor.py`)

追跡結果から時系列特徴量とペアワイズ特徴量を抽出し、Event Decoder の入力形式に変換する。

### 初期化

```python
extractor = FeatureExtractor(
    temporal_window=16,      # 時間ステップ数 T
    normalize_coords=True,   # 座標を画像サイズで正規化
    image_size=(480, 640),   # (height, width)
)
```

### 抽出

```python
object_features, pairwise_features = extractor.extract(tracking_results)
# object_features: dict[track_id, ObjectFeatures]
# pairwise_features: list[PairwiseFeatures]
```

### ObjectFeatures

各トラックの時系列特徴量を `temporal_window` ステップ分格納する。

| フィールド | 形状 | 説明 |
|---|---|---|
| `track_id` | `int` | トラック ID |
| `category_id` | `int` | カテゴリの整数 ID（ソート済みユニークカテゴリの index） |
| `embedding` | `(D_emb,)` | 全フレームの埋め込みの平均 |
| `bbox_seq` | `(T, 4)` | 正規化 bbox (cx, cy, w, h) |
| `centroid_seq` | `(T, 2)` | 正規化重心 (cx, cy) |
| `area_seq` | `(T, 1)` | 正規化面積 (w × h) |
| `presence_seq` | `(T, 1)` | 存在フラグ (0 or 1) |
| `delta_centroid_seq` | `(T-1, 2)` | フレーム間の重心差分 |
| `delta_area_seq` | `(T-1, 1)` | フレーム間の面積差分 |
| `velocity_seq` | `(T-1, 1)` | 重心差分の L2 ノルム |

**差分特徴量の処理:**
- 両フレームにオブジェクトが存在する場合のみ有効な差分を計算
- 片方でも不在の場合はゼロにマスク（不正なジャンプを回避）

### PairwiseFeatures

全オブジェクトペア `(N choose 2)` について計算する時系列特徴量。

| フィールド | 形状 | 説明 |
|---|---|---|
| `track_id_i`, `track_id_j` | `int` | ペアのトラック ID（常に i < j） |
| `iou_seq` | `(T, 1)` | bbox の IoU |
| `distance_seq` | `(T, 1)` | 正規化重心間距離 |
| `containment_ij_seq` | `(T, 1)` | i の j に対する包含率 |
| `containment_ji_seq` | `(T, 1)` | j の i に対する包含率 |
| `relative_position_seq` | `(T, 2)` | 相対位置 (dx, dy) |

### 座標の正規化

- `bbox_xyxy_to_cxcywh`: `[x1,y1,x2,y2]` → 正規化 `(cx,cy,w,h)`
- cx, w は `image_w` で割り、cy, h は `image_h` で割る
- 全特徴量が [0, 1] 範囲に収まる

### 包含率 (Containment)

`containment(inner, outer) = intersection_area / area_inner`

- 1.0 に近いほど inner が outer に完全に含まれている
- 工具が引き出しの中にある場合などの空間関係を捉える
