# モデル (`models/`)

DETR 風のセット予測モデルで、オブジェクトの追跡特徴量からイベントグラフを直接予測する。

## EventDecoder (`event_decoder.py`)

固定数の learnable event queries を使い、クロスアテンションでオブジェクト表現からイベントを一括予測する。自己回帰デコーディングは行わない。

### アーキテクチャ

```
入力テンソル
  object_embeddings: (B, K, D_emb=256)    ← SAM 3 の埋め込み
  object_temporal:   (B, K, T=16, D_geo=12) ← 幾何特徴量の時系列
  pairwise:          (B, K, K, T, D_pair=7) ← ペアワイズ特徴量
  object_mask:       (B, K)               ← 有効オブジェクトのマスク

処理フロー:
  1. geo_proj:  object_temporal → (B, K, T, d_model) → T方向mean pool → (B, K, d_model)
  2. emb_proj:  object_embeddings → (B, K, d_model)
  3. object_repr = emb_repr + geo_repr
  4. temporal_encoder:  TransformerEncoder (self-attention over K objects)
  5. pair_proj:  pairwise → mean pool → aggregate → object_repr に加算
  6. context_encoder:  TransformerEncoder (self-attention with mask)
  7. event_decoder:  TransformerDecoder (cross-attention: queries × object_slots)
  8. 7つの予測ヘッドで出力

出力: EventPredictions
```

### ハイパーパラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `d_model` | 256 | Transformer の隠れ次元 |
| `nhead` | 8 | マルチヘッドアテンションのヘッド数 |
| `num_object_encoder_layers` | 3 | Temporal Encoder の層数 |
| `num_context_encoder_layers` | 3 | Context Encoder の層数 |
| `num_event_decoder_layers` | 4 | Event Decoder の層数 |
| `num_event_queries` | 20 (M) | 学習可能な event query の数 |
| `max_objects` | 30 (K) | 最大オブジェクト数 |
| `dropout` | 0.1 | ドロップアウト率 |
| `d_geo` | 12 | 幾何特徴量の次元 |
| `d_pair` | 7 | ペアワイズ特徴量の次元 |
| `num_actions` | 13 (A) | アクションクラス数 |
| `embedding_dim` | 256 | SAM 3 埋め込みの次元 |
| `temporal_window` | 16 (T) | 時間ステップ数 |

### Event Queries の初期化

```python
self.event_queries = nn.Parameter(
    torch.randn(M, d_model) / (d_model ** 0.5)
)
```

`1/sqrt(d_model)` でスケーリングし、学習初期の安定性を確保。

### ポインタマスキング

無効なオブジェクトスロット（パディング部分）をポインタヘッドの logit で `-inf` に設定し、softmax 後に確率 0 となるようにする。`source_ptr` と `dest_ptr` は K+1 番目のスロットが "none" を表し、常に有効。

## EventPredictions

EventDecoder の出力を格納するデータクラス。

| フィールド | 形状 | 損失関数 | 説明 |
|---|---|---|---|
| `interaction` | `(B, M, 1)` | BCE | 有効なイベントか否か |
| `action` | `(B, M, A)` | CE | アクション分類（13 クラス） |
| `agent_ptr` | `(B, M, K)` | CE | 行為者オブジェクトへのポインタ |
| `target_ptr` | `(B, M, K)` | CE | 対象オブジェクトへのポインタ |
| `source_ptr` | `(B, M, K+1)` | CE | 取り出し元へのポインタ（K="none"） |
| `dest_ptr` | `(B, M, K+1)` | CE | 格納先へのポインタ（K="none"） |
| `frame` | `(B, M, T)` | CE | 発生フレームの分類 |

## PredictionHead (`heads.py`)

全 7 ヘッドで共通の 2 層 MLP。

```
Linear(d_input, d_hidden) → ReLU → Dropout → Linear(d_hidden, d_output)
```

## EventGraphLoss (`losses.py`)

Hungarian Matching ベースの損失関数。DETR と同様に、予測スロットと GT イベントの最適割当を求めてからロスを計算する。

### Hungarian Matching

コスト行列 `(M × N_gt)` の各要素は以下の負対数確率の和:

```
cost(i, j) = -log P(action_j | pred_i)
           + -log P(agent_j | pred_i)
           + -log P(target_j | pred_i)
           + -log P(frame_j | pred_i)
```

`scipy.optimize.linear_sum_assignment` で最適割当を計算。

### 損失の計算

| ロス | 対象 | 関数 | 正規化 |
|---|---|---|---|
| interaction | 全 M スロット | BCE with logits | バッチサイズ B |
| action | マッチしたスロットのみ | Cross Entropy | マッチ数 |
| agent_ptr | マッチしたスロットのみ | Cross Entropy | マッチ数 |
| target_ptr | マッチしたスロットのみ | Cross Entropy | マッチ数 |
| source_ptr | マッチしたスロットのみ | Cross Entropy | マッチ数 |
| dest_ptr | マッチしたスロットのみ | Cross Entropy | マッチ数 |
| frame | マッチしたスロットのみ | Cross Entropy | マッチ数 |

- **マッチしたスロット**: interaction target = 1.0、各ヘッドのロスを計算
- **未マッチのスロット**: interaction target = 0.0、interaction loss のみ

### デフォルトの損失重み

| ヘッド | 重み | 根拠 |
|---|---|---|
| interaction | 2.0 | イベント検出を重視 |
| action | 1.0 | - |
| agent_ptr | 1.0 | - |
| target_ptr | 1.0 | - |
| source_ptr | 0.5 | 任意フィールド（全アクションに必須ではない） |
| dest_ptr | 0.5 | 任意フィールド |
| frame | 0.5 | 粗い時間推定で十分 |

### source/dest の "none" ラベル

GT で `source_track_id` が `None` の場合、ターゲットクラスは K+1 番目のスロット（= `source_ptr.shape[-1] - 1`）に設定される。これにより、source/destination が不要なアクション（`open`, `close` 等）でも CE loss が適切に計算される。
