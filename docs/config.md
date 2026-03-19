# 設定管理 (`config.py`)

Python dataclass ベースの YAML 設定管理。型安全で深いマージをサポートする。

## Config 構造

```python
@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    wandb: WandbConfig
    sam3: SAM3Config
    vlm: VLMConfig
    inference: InferenceConfig
```

### DataConfig

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `raw_dir` | `str` | `"data/raw"` | 生動画ディレクトリ |
| `processed_dir` | `str` | `"data/processed"` | 処理済みデータ |
| `splits_dir` | `str` | `"data/splits"` | train/val/test 分割 |
| `batch_size` | `int` | 32 | バッチサイズ |
| `num_workers` | `int` | 4 | DataLoader のワーカー数 |
| `pin_memory` | `bool` | True | ピンメモリ |

### ModelConfig

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `name` | `str` | `"base_model"` | モデル名（`"event_decoder"` で EventDecoder を使用） |
| `d_model` | `int` | 256 | Transformer 隠れ次元 |
| `nhead` | `int` | 8 | アテンションヘッド数 |
| `num_object_encoder_layers` | `int` | 3 | Temporal Encoder 層数 |
| `num_context_encoder_layers` | `int` | 3 | Context Encoder 層数 |
| `num_event_decoder_layers` | `int` | 4 | Event Decoder 層数 |
| `num_event_queries` | `int` | 20 | event query 数 (M) |
| `max_objects` | `int` | 30 | 最大オブジェクト数 (K) |
| `dropout` | `float` | 0.1 | ドロップアウト率 |
| `d_geo` | `int` | 12 | 幾何特徴量次元 |
| `d_pair` | `int` | 7 | ペアワイズ特徴量次元 |
| `num_actions` | `int` | 13 | アクションクラス数 |
| `embedding_dim` | `int` | 256 | SAM 3 埋め込み次元 |

### SAM3Config

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `model_size` | `str` | `"large"` | モデルサイズ |
| `device` | `str` | `"cuda"` | デバイス |
| `score_threshold` | `float` | 0.3 | 検出スコア閾値 |
| `embedding_dim` | `int` | 256 | 埋め込み次元 |
| `concept_prompts` | `list[str]` | `["person", "wrench", ...]` | テキストプロンプト |
| `output_dir` | `str` | `"data/sam3_outputs"` | 出力先 |

### VLMConfig

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `model_name` | `str` | `"Qwen/Qwen3.5-9B"` | HuggingFace モデル名 |
| `device_map` | `str` | `"auto"` | デバイス配置 |
| `torch_dtype` | `str` | `"bfloat16"` | データ型 |
| `max_new_tokens` | `int` | 4096 | 最大生成トークン数 |
| `temperature` | `float` | 0.1 | サンプリング温度 |
| `thinking` | `bool` | False | Qwen 思考モード |
| `clip_length` | `int` | 16 | クリップ長 |
| `clip_stride` | `int` | 8 | クリップストライド |
| `target_fps` | `float` | 1.0 | サンプリング FPS |
| `max_retries` | `int` | 3 | パース失敗時のリトライ回数 |

### TrainingConfig

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `epochs` | `int` | 100 | 学習エポック数 |
| `learning_rate` | `float` | 1e-3 | 学習率 |
| `weight_decay` | `float` | 1e-4 | 重み減衰 |
| `optimizer` | `str` | `"adam"` | adam / adamw / sgd |
| `scheduler` | `str` | `"cosine"` | cosine / cosine_warmup / step / none |
| `scheduler_params` | `SchedulerParams` | - | T_max, warmup_epochs |
| `grad_clip_norm` | `float` | 1.0 | 勾配クリッピング |
| `seed` | `int` | 42 | 乱数シード |
| `device` | `str` | `"cuda"` | デバイス |
| `checkpoint_dir` | `str` | `"checkpoints"` | チェックポイント保存先 |
| `save_every_n_epochs` | `int` | 10 | 定期保存間隔 |
| `early_stopping_patience` | `int` | 0 | Early stopping（0 = 無効） |
| `loss_weights` | `EventDecoderConfig` | - | 各ヘッドの損失重み |

### InferenceConfig

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `sam3` | `SAM3Config` | - | SAM 3 設定 |
| `target_fps` | `float` | 1.0 | サンプリング FPS |
| `clip_length` | `int` | 16 | クリップ長 |
| `clip_stride` | `int` | 8 | クリップストライド |
| `checkpoint_path` | `str` | `"data/checkpoints/best.pt"` | チェックポイント |
| `confidence_threshold` | `float` | 0.5 | 信頼度閾値 |
| `nms_frame_threshold` | `int` | 3 | 重複排除のフレーム閾値 |

## 使い方

### 基本的な読み込み

```python
from event_graph_generation.config import Config

config = Config.from_yaml("configs/training.yaml")
```

### 実験オーバーライド

```python
config = Config.from_yaml("configs/training.yaml")
config = config.merge("configs/experiment/event_decoder_v1.yaml")
```

`merge()` は deep merge を行う。オーバーライドファイルには変更したいフィールドのみを記述すればよい:

```yaml
# configs/experiment/event_decoder_v1.yaml
model:
  num_event_queries: 30
  dropout: 0.2
training:
  learning_rate: 5.0e-5
```

### dict / YAML への変換

```python
d = config.to_dict()                        # 再帰的に dict に変換
config.to_yaml("output/resolved_config.yaml") # YAML ファイルに保存
```

## 設定ファイル一覧

| ファイル | 用途 |
|---|---|
| `configs/training.yaml` | Event Decoder 学習のベース設定 |
| `configs/inference.yaml` | 推論パイプライン設定 |
| `configs/vlm.yaml` | VLM アノテーション設定 |
| `configs/sam3.yaml` | SAM 3 トラッキング設定 |
| `configs/actions.yaml` | アクション語彙定義（13 クラス + requires_source/destination フラグ） |
| `configs/experiment/*.yaml` | 実験ごとのオーバーライド |
