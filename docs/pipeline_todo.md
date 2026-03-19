# Event Graph Generation Pipeline — 実装指示書

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [アーキテクチャ全体図](#2-アーキテクチャ全体図)
3. [ディレクトリ構成](#3-ディレクトリ構成)
4. [Phase 0: 環境構築](#4-phase-0-環境構築)
5. [Phase 1: SAM 3 物体追跡モジュール](#5-phase-1-sam-3-物体追跡モジュール)
6. [Phase 2: VLM合成データ生成モジュール](#6-phase-2-vlm合成データ生成モジュール)
7. [Phase 3: アラインメント & データセット構築](#7-phase-3-アラインメント--データセット構築)
8. [Phase 4: Event Decoder モデル](#8-phase-4-event-decoder-モデル)
9. [Phase 5: 学習パイプライン](#9-phase-5-学習パイプライン)
10. [Phase 6: 推論 & DB格納パイプライン](#10-phase-6-推論--db格納パイプライン)
11. [Phase 7: 検索API](#11-phase-7-検索api)
12. [テスト方針](#12-テスト方針)
13. [補足: 設計判断の根拠](#13-補足-設計判断の根拠)

---

## 1. プロジェクト概要

### 目的

動画から「いつ・誰が・何を・どこで・どうしたか」を構造化したEvent Graphを自動生成するパイプラインを構築する。

### ユースケース

作業場における用具のトレース。例:
- レンチがいつ引き出しから取り出されたか
- 誰がどの工具をどこに置いたか
- 各用具の現在の所在地

### 技術戦略

1. **SAM 3** (Meta, 848M params) を凍結状態で使い、動画中の物体の検出・セグメンテーション・追跡を行う
2. **VLM** (GPT-4o, Gemini 2.5 Pro 等) に複数フレームを入力し、Event Graphの合成教師データを生成する
3. SAM 3の追跡結果とVLMの教師データをアラインメントし、**軽量なEvent Decoder** (~5-15M params) を蒸留学習する
4. 推論時は SAM 3 → Event Decoder のパイプラインでリアルタイムにEvent Graphを生成し、DBに格納する

### 重要な設計原則

- **Event Decoderの出力はテキストではなくグラフ構造体**とする（自己回帰デコードは行わない）
- SAM 3のVision Encoderは凍結し、学習対象はEvent Decoderのみとする
- 全モジュールをPythonで実装し、PyTorchをフレームワークとする

---

## 2. アーキテクチャ全体図

```
動画ファイル (.mp4)
    │
    ├──→ [Frame Sampler] ── 1fps でフレーム抽出
    │         │
    │         ├──→ [SAM 3 Tracker] (凍結)
    │         │        │
    │         │        ├── tracked objects (mask, bbox, embedding, track_id)
    │         │        └── per-frame instance segmentation
    │         │
    │         └──→ [VLM Annotator] (合成データ生成時のみ)
    │                  │
    │                  └── event annotations (JSON)
    │
    ├──→ [Alignment Module]
    │         │
    │         └── (SAM 3 features, VLM event labels) のペアデータセット
    │
    ├──→ [Event Decoder Training]
    │         │
    │         └── 学習済み Event Decoder チェックポイント
    │
    └──→ [Inference Pipeline] (運用時)
              │
              ├── SAM 3 Tracker → Feature Extractor → Event Decoder
              │
              └── EventGraph → DB Writer → PostgreSQL / Neo4j
```

---

## 3. ディレクトリ構成

```
event-graph-pipeline/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml              # デフォルト設定
│   ├── sam3.yaml                 # SAM 3 関連設定
│   ├── vlm.yaml                  # VLM API 設定
│   ├── training.yaml             # 学習ハイパーパラメータ
│   ├── inference.yaml            # 推論パイプライン設定
│   └── actions.yaml              # アクション語彙定義
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── frame_sampler.py      # 動画からフレーム抽出
│   │   ├── dataset.py            # PyTorch Dataset (学習用)
│   │   └── collate.py            # カスタム collate_fn
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── sam3_tracker.py       # SAM 3 ラッパー
│   │   └── feature_extractor.py  # SAM 3出力 → 特徴量ベクトル
│   │
│   ├── annotation/
│   │   ├── __init__.py
│   │   ├── vlm_annotator.py      # VLM API呼び出し + パース
│   │   ├── prompts.py            # VLM用プロンプトテンプレート
│   │   ├── alignment.py          # SAM 3 track_id ↔ VLM obj_id 対応付け
│   │   └── validator.py          # 生成データのバリデーション
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── event_decoder.py      # Event Decoder 本体
│   │   ├── heads.py              # 予測ヘッド群 (action, pointer, frame, etc.)
│   │   └── losses.py             # Hungarian Matching + 各損失関数
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # 学習ループ
│   │   └── evaluator.py          # 評価メトリクス
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── pipeline.py           # 推論パイプライン (SAM3 → Event Decoder → EventGraph)
│   │   └── postprocess.py        # Event Decoder出力 → EventGraph構造体
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── db_writer.py          # EventGraph → DB書き込み
│   │   ├── models.py             # SQLAlchemy / ORM モデル定義
│   │   └── query_api.py          # 検索クエリAPI
│   │
│   └── schemas/
│       ├── __init__.py
│       ├── event_graph.py        # EventGraph, ObjectNode, EventEdge のデータクラス
│       └── vlm_output.py         # VLM出力JSONのPydanticスキーマ
│
├── scripts/
│   ├── generate_annotations.py   # VLM合成データ一括生成
│   ├── build_dataset.py          # アラインメント + データセット構築
│   ├── train.py                  # 学習実行
│   ├── evaluate.py               # 評価実行
│   └── run_inference.py          # 推論デモ
│
├── tests/
│   ├── test_frame_sampler.py
│   ├── test_sam3_tracker.py
│   ├── test_vlm_annotator.py
│   ├── test_alignment.py
│   ├── test_event_decoder.py
│   ├── test_pipeline.py
│   └── test_db_writer.py
│
└── data/
    ├── videos/                   # 入力動画
    ├── frames/                   # 抽出フレーム (中間生成物)
    ├── annotations/              # VLM生成アノテーション (JSON)
    ├── sam3_outputs/             # SAM 3 追跡結果 (pickle/pt)
    ├── aligned/                  # アラインメント済みデータセット
    └── checkpoints/              # 学習済みモデル
```

---

## 4. Phase 0: 環境構築

### pyproject.toml の依存関係

```toml
[project]
name = "event-graph-pipeline"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "torchvision>=0.16",
    "sam3",                      # pip install sam3 (Meta公式)
    "ultralytics>=8.3.237",      # SAM 3統合済み (代替手段)
    "opencv-python>=4.8",
    "Pillow>=10.0",
    "pydantic>=2.0",
    "sqlalchemy>=2.0",
    "psycopg2-binary",           # PostgreSQL用
    "scipy",                     # Hungarian matching
    "omegaconf",                 # 設定管理
    "openai",                    # VLM API (GPT-4o)
    "google-generativeai",       # VLM API (Gemini)
    "tqdm",
    "wandb",                     # 実験追跡
]
```

### configs/actions.yaml — アクション語彙定義

```yaml
# 作業場用途のアクション語彙
# Event Decoderの分類ヘッドの出力クラス数を決定する
actions:
  - name: "take_out"
    description: "物体を容器/収納から取り出す"
    requires_source: true
    requires_destination: false

  - name: "put_in"
    description: "物体を容器/収納に入れる"
    requires_source: false
    requires_destination: true

  - name: "place_on"
    description: "物体を面の上に置く"
    requires_source: false
    requires_destination: true

  - name: "pick_up"
    description: "面の上から物体を持ち上げる"
    requires_source: true
    requires_destination: false

  - name: "hand_over"
    description: "人から人へ物体を渡す"
    requires_source: true      # source = 渡す人
    requires_destination: true  # destination = 受け取る人

  - name: "open"
    description: "容器/引き出し/蓋を開ける"
    requires_source: false
    requires_destination: false

  - name: "close"
    description: "容器/引き出し/蓋を閉める"
    requires_source: false
    requires_destination: false

  - name: "use"
    description: "工具/道具を使用する"
    requires_source: false
    requires_destination: false

  - name: "move"
    description: "物体を場所Aから場所Bへ移動する"
    requires_source: true
    requires_destination: true

  - name: "attach"
    description: "物体を別の物体に取り付ける"
    requires_source: false
    requires_destination: true

  - name: "detach"
    description: "物体を別の物体から取り外す"
    requires_source: true
    requires_destination: false

  - name: "inspect"
    description: "物体を視認/確認する"
    requires_source: false
    requires_destination: false

  - name: "no_event"
    description: "イベントなし (negative class)"
    requires_source: false
    requires_destination: false

# 用途に応じて語彙を拡張可能
# 新しいアクションを追加する場合はここに定義を追加するだけで良い
```

---

## 5. Phase 1: SAM 3 物体追跡モジュール

### 5.1 `src/tracking/sam3_tracker.py`

SAM 3を動画に適用し、指定したコンセプト（用具名）の全インスタンスを検出・追跡する。

#### 要件

- SAM 3モデルを初期化し、テキストプロンプト（用具名リスト）を受け取る
- 動画のフレーム列を入力し、フレームごとに以下を返す:
  - `track_id`: フレーム間で一貫した物体ID (int)
  - `category`: テキストプロンプトと対応するカテゴリ名 (str)
  - `mask`: セグメンテーションマスク (np.ndarray, H×W, bool)
  - `bbox`: バウンディングボックス [x1, y1, x2, y2] (np.ndarray)
  - `score`: 検出信頼度 (float)
  - `object_embedding`: SAM 3内部のobject embedding (torch.Tensor, dim=256)
- バッチ処理に対応し、複数フレームをまとめて処理できること

#### インターフェース

```python
class SAM3Tracker:
    def __init__(self, model_size: str = "large", device: str = "cuda"):
        """
        Args:
            model_size: SAM 3モデルサイズ ("large" | "base")
            device: 推論デバイス
        """
        ...

    def set_concept_prompts(self, prompts: list[str]) -> None:
        """
        追跡対象のコンセプトプロンプトを設定する。
        例: ["person", "wrench", "drawer", "workbench", "screwdriver"]
        """
        ...

    def track_video(
        self,
        frames: list[np.ndarray],   # (H, W, 3) BGR images
        frame_indices: list[int],    # 元動画でのフレーム番号
    ) -> list[FrameTrackingResult]:
        """
        フレーム列を入力し、各フレームの追跡結果を返す。
        SAM 3のメモリベーストラッカーを利用してフレーム間の同一性を維持する。
        """
        ...

@dataclass
class TrackedObject:
    track_id: int
    category: str
    mask: np.ndarray          # (H, W) bool
    bbox: np.ndarray          # [x1, y1, x2, y2]
    score: float
    embedding: torch.Tensor   # SAM 3 object embedding (凍結済み特徴)

@dataclass
class FrameTrackingResult:
    frame_index: int
    objects: list[TrackedObject]
```

#### 実装上の注意

- SAM 3の公式リポジトリ (facebookresearch/sam3) の API に従って実装する
- Ultralytics経由 (`from ultralytics import SAM`) でも利用可能。どちらでも良いが、object embeddingを取り出せる方を選ぶこと
- **object_embedding の取得**: SAM 3のDetectorはDETRベースなので、decoder layerの出力(object query)を抽出する。Ultralytics経由の場合はモデル内部にhookを仕掛ける必要がある可能性あり。公式リポジトリ経由の方が内部アクセスしやすい
- GPUメモリ管理: 1フレームずつ処理し、不要なテンソルは即座に解放する

### 5.2 `src/tracking/feature_extractor.py`

SAM 3の追跡結果から、Event Decoderの入力となる特徴量ベクトルを構成する。

#### 要件

各tracked objectについて、以下の特徴量を計算してconcatする:

```python
@dataclass
class ObjectFeatures:
    """1つのtracked objectの、1つの時間窓における特徴量"""
    track_id: int
    category_id: int                          # カテゴリの整数ID

    # SAM 3由来 (意味特徴)
    embedding: torch.Tensor                   # (D_emb,) SAM 3 object embedding

    # 幾何特徴の時系列 (時間窓内の各フレーム)
    bbox_seq: torch.Tensor                    # (T, 4) 正規化 bbox [cx, cy, w, h]
    centroid_seq: torch.Tensor                # (T, 2) 正規化 centroid [cx, cy]
    area_seq: torch.Tensor                    # (T, 1) 正規化面積
    presence_seq: torch.Tensor                # (T, 1) そのフレームに存在するか (0/1)

    # 時間差分特徴 (変化の手がかり)
    delta_centroid_seq: torch.Tensor          # (T-1, 2) フレーム間の位置変化
    delta_area_seq: torch.Tensor              # (T-1, 1) フレーム間の面積変化
    velocity_seq: torch.Tensor                # (T-1, 1) 移動速度 (L2 norm of delta)

@dataclass
class PairwiseFeatures:
    """2つのtracked object間の関係特徴"""
    track_id_i: int
    track_id_j: int

    iou_seq: torch.Tensor                     # (T, 1) フレームごとのIoU
    distance_seq: torch.Tensor                # (T, 1) centroid間距離
    containment_ij_seq: torch.Tensor          # (T, 1) iがjに含まれる割合
    containment_ji_seq: torch.Tensor          # (T, 1) jがiに含まれる割合
    relative_position_seq: torch.Tensor       # (T, 2) i→jの相対位置ベクトル
```

#### インターフェース

```python
class FeatureExtractor:
    def __init__(self, config):
        """
        Args:
            config: 特徴量抽出の設定
                - temporal_window: 時間窓サイズ (フレーム数, default: 16)
                - normalize_coords: 座標を [0, 1] に正規化するか (default: True)
                - image_size: 正規化の基準サイズ (H, W)
        """
        ...

    def extract(
        self,
        tracking_results: list[FrameTrackingResult],
    ) -> tuple[dict[int, ObjectFeatures], list[PairwiseFeatures]]:
        """
        追跡結果から全objectの特徴量とペアワイズ特徴量を抽出する。

        Returns:
            object_features: {track_id: ObjectFeatures}
            pairwise_features: 全ペアのPairwiseFeatures
        """
        ...
```

#### 実装上の注意

- bbox, centroidは画像サイズで正規化して [0, 1] の範囲にする
- 存在しないフレームでは presence=0 とし、他の特徴量はゼロ埋めする
- ペアワイズ特徴量は全ペア (K*(K-1)/2) を計算する。物体数Kが多い場合は空間的に近いペアのみに絞るオプションを設ける
- maskからbbox/centroid/areaへの変換はOpenCVの `cv2.boundingRect` / `cv2.moments` を使う

---

## 6. Phase 2: VLM合成データ生成モジュール

### 6.1 `src/annotation/prompts.py`

VLMに送るプロンプトテンプレートを定義する。

#### プロンプト設計方針

VLMには**2段階の出力**を要求する:
1. まず全フレームに映る物体をリストアップし、一貫したIDを割り当てる
2. 次に、物体間で発生したイベント（状態変化）を時系列順に記述する

#### プロンプトテンプレート

```python
SYSTEM_PROMPT = """
あなたは動画フレーム列から物体のインタラクションイベントを抽出する専門のアノテーターです。
以下の規則に厳密に従ってJSON形式で出力してください。

## 出力規則

1. まず `objects` に、全フレームを通じて登場する物体を列挙する
   - `obj_id`: "category_NN" 形式 (例: "person_01", "wrench_01", "drawer_01")
   - `category`: 物体カテゴリ (下記の許可カテゴリのみ使用)
   - `first_seen_frame`: 最初に出現するフレーム番号
   - `attributes`: 外観的特徴 (色、サイズ、位置など)

2. 次に `events` に、状態変化が起きたイベントのみを記述する
   - 変化のないフレームは記述しない
   - `event_id`: "evt_NNN" 形式の連番
   - `frame`: イベントが発生したフレーム番号
   - `action`: 下記の許可アクションのみ使用
   - `agent`: 動作主体の obj_id
   - `target`: 動作対象の obj_id
   - `source`: 物体の移動元 (該当する場合のみ)
   - `destination`: 物体の移動先 (該当する場合のみ)

## 許可カテゴリ
{category_list}

## 許可アクション
{action_list}

## 重要
- 同一物体には必ず同じ obj_id を使うこと
- 推測や不確かな情報は含めない
- フレーム番号は入力画像に付記されたものを使用する
"""

USER_PROMPT_TEMPLATE = """
以下の {n_frames} 枚のフレーム画像は、動画から {fps}fps でサンプリングしたものです。
各フレーム画像の左上にフレーム番号が表示されています。

これらのフレームを時系列順に観察し、物体のインタラクションイベントを抽出してください。

出力はJSON形式のみで、他のテキストは含めないでください。
```json
{json_schema_example}
```
"""
```

#### JSONスキーマ例 (VLM出力の期待形式)

```json
{
  "objects": [
    {
      "obj_id": "person_01",
      "category": "person",
      "first_seen_frame": 0,
      "attributes": ["blue_gloves", "standing"]
    },
    {
      "obj_id": "wrench_01",
      "category": "wrench",
      "first_seen_frame": 5,
      "attributes": ["silver", "medium"]
    },
    {
      "obj_id": "drawer_01",
      "category": "drawer",
      "first_seen_frame": 0,
      "attributes": ["wooden", "top_left"]
    }
  ],
  "events": [
    {
      "event_id": "evt_001",
      "frame": 3,
      "action": "open",
      "agent": "person_01",
      "target": "drawer_01"
    },
    {
      "event_id": "evt_002",
      "frame": 5,
      "action": "take_out",
      "agent": "person_01",
      "target": "wrench_01",
      "source": "drawer_01"
    },
    {
      "event_id": "evt_003",
      "frame": 12,
      "action": "place_on",
      "agent": "person_01",
      "target": "wrench_01",
      "destination": "workbench_01"
    }
  ]
}
```

### 6.2 `src/annotation/vlm_annotator.py`

#### 要件

- 複数のVLM APIに対応する (GPT-4o, Gemini 2.5 Pro)
- フレーム画像を base64 エンコードしてAPI に送信する
- 各フレーム画像の左上にフレーム番号を描画してから送信する
- レスポンスのJSONをパースし、Pydanticモデルでバリデーションする
- API失敗時のリトライ (exponential backoff, max 3回)
- レート制限の遵守 (configurable な sleep interval)

#### インターフェース

```python
class VLMAnnotator:
    def __init__(self, provider: str, model: str, api_key: str, config: dict):
        """
        Args:
            provider: "openai" | "google"
            model: モデル名 (例: "gpt-4o", "gemini-2.5-pro")
            api_key: APIキー
            config: プロンプト設定 (カテゴリリスト、アクションリスト等)
        """
        ...

    def annotate_clip(
        self,
        frames: list[np.ndarray],     # フレーム画像列 (BGR)
        frame_indices: list[int],      # フレーム番号
        video_id: str,
    ) -> VLMAnnotation:
        """
        フレーム列をVLMに送信し、アノテーションを取得する。

        Returns:
            VLMAnnotation: パース・バリデーション済みのアノテーション
        """
        ...

    def annotate_video(
        self,
        video_path: str,
        fps: float = 1.0,
        clip_length: int = 16,
        clip_stride: int = 8,
    ) -> list[VLMAnnotation]:
        """
        動画全体をスライディングウィンドウでアノテーションする。

        Args:
            video_path: 動画ファイルパス
            fps: サンプリングFPS
            clip_length: 1回のVLM呼び出しに含めるフレーム数
            clip_stride: スライドのストライド (重複を許可)

        Returns:
            各クリップのアノテーションのリスト
        """
        ...
```

### 6.3 `src/annotation/validator.py`

VLM出力のバリデーションと正規化を行う。

#### バリデーション項目

1. **スキーマ検証**: Pydantic で JSON構造を検証
2. **語彙検証**: action, category が許可リストに含まれるか
3. **参照整合性**: events内の agent, target, source, destination が objects内の obj_id を正しく参照しているか
4. **時系列整合性**: events が frame 番号の昇順になっているか
5. **論理整合性**: source/destination が action の requires_source / requires_destination と一致するか

#### バリデーション失敗時の処理

- 軽微な問題 (語彙の表記ゆれ等): 自動修正してログに記録
- 重大な問題 (参照不整合等): そのクリップのアノテーションを破棄してログに記録
- 破棄率が閾値 (default: 30%) を超えたらプロンプト修正を提案するログを出力

---

## 7. Phase 3: アラインメント & データセット構築

### 7.1 `src/annotation/alignment.py`

SAM 3の追跡結果とVLMのアノテーションを対応付ける。

#### アラインメント手順

```
各クリップについて:
1. SAM 3 の追跡結果から、各 track_id のフレームごとの mask/bbox を取得
2. VLM の各 obj_id について、first_seen_frame の情報と attributes を取得
3. マッチング:
   a. 同一フレームにおける SAM 3 の各 track の bbox と、
      VLM の obj_id のカテゴリ名でフィルタした SAM 3 track を突合
   b. 同一カテゴリの track が複数ある場合:
      - VLM の attributes (位置情報等) を手がかりに、空間的に最も近いものを選択
      - それでも曖昧な場合: そのオブジェクトペアのイベントは破棄
4. 結果: { vlm_obj_id → sam3_track_id } のマッピング辞書
```

#### インターフェース

```python
class Aligner:
    def __init__(self, iou_threshold: float = 0.3):
        ...

    def align(
        self,
        tracking_results: list[FrameTrackingResult],
        vlm_annotation: VLMAnnotation,
    ) -> AlignmentResult:
        """
        Returns:
            AlignmentResult:
                mapping: dict[str, int]  # vlm_obj_id → sam3_track_id
                unmatched_vlm: list[str]  # マッチできなかったVLM obj_id
                unmatched_sam3: list[int]  # マッチできなかったSAM 3 track_id
                confidence: float          # アラインメント全体の信頼度
        """
        ...
```

### 7.2 `scripts/4_build_dataset.py` — データセット構築スクリプト

#### 処理フロー

```
入力:
  - data/sam3_outputs/{video_id}.pt     (SAM 3追跡結果)
  - data/annotations/{video_id}.json   (VLMアノテーション)

処理:
  1. アラインメント実行
  2. 各イベントについて、教師ラベルを構成:
     - agent_track_id: SAM 3の track_id (pointer target)
     - action_class: アクションの整数ID
     - target_track_id: SAM 3の track_id (pointer target)
     - source_track_id: (optional) SAM 3の track_id
     - dest_track_id: (optional) SAM 3の track_id
     - event_frame: イベント発生フレームのインデックス
  3. ネガティブサンプル生成:
     - イベントが発生していない時間窓も一定割合でデータセットに含める
     - ラベルは全スロット "no_event"
  4. 保存形式: 各サンプルを個別の .pt ファイルとして保存

出力:
  data/aligned/
    ├── meta.json                   # データセット統計情報
    ├── samples/
    │   ├── {video_id}_{clip_id}_000.pt
    │   ├── {video_id}_{clip_id}_001.pt
    │   └── ...
    └── splits/
        ├── train.txt               # 学習用サンプルIDリスト
        ├── val.txt                  # 検証用サンプルIDリスト
        └── test.txt                # テスト用サンプルIDリスト
```

#### 各 .pt ファイルの中身

```python
{
    "video_id": str,
    "clip_id": str,
    "frame_indices": list[int],           # このクリップのフレーム番号列

    # SAM 3 特徴量
    "object_features": {
        track_id: {
            "embedding": Tensor,          # (D_emb,)
            "bbox_seq": Tensor,           # (T, 4)
            "centroid_seq": Tensor,       # (T, 2)
            "area_seq": Tensor,           # (T, 1)
            "presence_seq": Tensor,       # (T, 1)
            "delta_centroid_seq": Tensor, # (T-1, 2)
            "delta_area_seq": Tensor,     # (T-1, 1)
            "velocity_seq": Tensor,       # (T-1, 1)
            "category_id": int,
        }
    },

    "pairwise_features": [
        {
            "track_id_i": int,
            "track_id_j": int,
            "iou_seq": Tensor,            # (T, 1)
            "distance_seq": Tensor,       # (T, 1)
            "containment_ij": Tensor,     # (T, 1)
            "containment_ji": Tensor,     # (T, 1)
            "relative_position": Tensor,  # (T, 2)
        }
    ],

    # 教師ラベル (イベント)
    "gt_events": [
        {
            "agent_track_id": int,
            "action_class": int,
            "target_track_id": int,
            "source_track_id": int | None,
            "dest_track_id": int | None,
            "event_frame_index": int,     # clip内でのフレームインデックス (0〜T-1)
        }
    ],

    # メタデータ
    "num_objects": int,
    "num_events": int,
}
```

### 7.3 `src/data/dataset.py`

#### 要件

```python
class EventGraphDataset(torch.utils.data.Dataset):
    """
    build_dataset.py で生成された .pt ファイルを読み込む。

    __getitem__ の返り値:
        object_embeddings: Tensor (K, D_emb)          # K = パディング後の最大物体数
        object_temporal:   Tensor (K, T, D_geo)       # 幾何特徴の時系列
        pairwise:          Tensor (K, K, T, D_pair)   # ペアワイズ特徴
        object_mask:       Tensor (K,)                # 有効な物体スロット (bool)
        gt_events:         list[dict]                 # 教師イベント
        num_objects:       int                        # 実際の物体数

    Notes:
        - 物体数が最大数 K_max に満たない場合はゼロパディングし、object_mask で管理
        - collate_fn で batch 化する際にも K_max でパディング
    """
```

---

## 8. Phase 4: Event Decoder モデル

### 8.1 `src/model/event_decoder.py`

DETR系のSet Predictionパラダイムに基づく。自己回帰デコードは行わない。

#### アーキテクチャ

```
入力:
  object_embeddings: (B, K, D_emb)    # SAM 3 embedding
  object_temporal:   (B, K, T, D_geo) # 幾何特徴時系列
  pairwise:          (B, K, K, T, D_pair) # ペアワイズ特徴
  object_mask:       (B, K)           # 有効スロットマスク

処理:
  1. Object Temporal Encoder:
     各物体の時系列特徴を temporal transformer で集約
     → (B, K, D_model)

  2. Object Context Encoder:
     SAM 3 embedding + temporal encoding + category embedding を結合
     self-attention で物体間の文脈を学習
     → (B, K, D_model)  ← これが object slots

  3. Event Decoder:
     M個の learnable event queries: (M, D_model)
     cross-attention で object slots を参照
     → (B, M, D_model)

  4. Prediction Heads:
     各 event query から:
     - interaction_head → (B, M, 1)   このスロットが有効か
     - action_head      → (B, M, A)   アクション分類 (A = アクション語彙数)
     - agent_ptr_head   → (B, M, K)   agent pointer (object slot上のsoftmax)
     - target_ptr_head  → (B, M, K)   target pointer
     - source_ptr_head  → (B, M, K+1) source pointer (+1 for "none")
     - dest_ptr_head    → (B, M, K+1) destination pointer (+1 for "none")
     - frame_head       → (B, M, T)   event frame 分類 (離散)

出力:
  EventPredictions dataclass
```

#### モデル設定のデフォルト値

```yaml
# configs/training.yaml
model:
  d_model: 256
  nhead: 8
  num_object_encoder_layers: 3    # Object Temporal Encoder
  num_context_encoder_layers: 3   # Object Context Encoder
  num_event_decoder_layers: 4     # Event Decoder
  num_event_queries: 20           # M (1クリップ内の最大イベント数)
  max_objects: 30                 # K (1クリップ内の最大物体数)
  dropout: 0.1
  d_geo: 12                      # 幾何特徴の次元 (bbox4 + centroid2 + area1 + delta_cent2 + delta_area1 + velocity1 + presence1)
  d_pair: 7                      # ペアワイズ特徴の次元 (iou1 + dist1 + cont2 + relpos2 + pad1)
```

#### 実装上の注意

- **Pointer機構**: agent_ptr_head / target_ptr_head は object slots の数 K に対する softmax。object_mask が False のスロットには -inf を設定してマスクすること
- **source/dest pointer**: "該当なし" を表す +1 スロットを追加。action の requires_source/requires_destination が False の場合、学習時にこのスロットの選択を強制する
- **Positional encoding**: event queries は learnable。object slots には物体のcategory embedding を加算する
- **パラメータ数目安**: d_model=256, 上記レイヤー数で約 10-15M params

### 8.2 `src/model/heads.py`

予測ヘッドは全て独立した MLP (2層, ReLU) とする。

```python
class PredictionHead(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, d_output: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_output),
        )

    def forward(self, x):
        return self.mlp(x)
```

### 8.3 `src/model/losses.py`

#### Hungarian Matching

1クリップ内の M 個の event query 予測と、GT の N 個のイベント (N ≤ M) を Hungarian Algorithm で 1対1マッチングする。

#### マッチングコスト

```python
def compute_matching_cost(pred, gt):
    """
    pred: M個の予測
    gt: N個のGTイベント

    コスト行列 C: (M, N)
    C[i,j] = λ_action * (1 - action_prob[i, gt_action[j]])
           + λ_agent  * (1 - agent_ptr_prob[i, gt_agent_slot[j]])
           + λ_target * (1 - target_ptr_prob[i, gt_target_slot[j]])
           + λ_frame  * |pred_frame[i] - gt_frame[j]| / T
    """
```

#### 損失関数

```
L_total = λ_inter * L_interaction    # BCE: event query が有効かどうか
        + λ_act   * L_action         # CE:  アクション分類
        + λ_agent * L_agent_ptr      # CE:  agent pointer
        + λ_tgt   * L_target_ptr     # CE:  target pointer
        + λ_src   * L_source_ptr     # CE:  source pointer (該当イベントのみ)
        + λ_dst   * L_dest_ptr       # CE:  dest pointer (該当イベントのみ)
        + λ_frm   * L_frame          # CE:  event frame分類

損失係数のデフォルト:
  λ_inter = 2.0  (interaction の正負比率が偏るため高めに)
  λ_act   = 1.0
  λ_agent = 1.0
  λ_tgt   = 1.0
  λ_src   = 0.5
  λ_dst   = 0.5
  λ_frm   = 0.5
```

#### 実装上の注意

- マッチされなかった event query は "no_event" クラスとして interaction loss のみ計算
- scipy.optimize.linear_sum_assignment を使って Hungarian Matching を実行
- source/dest pointer の損失は、当該アクションが requires_source/requires_destination の場合のみ計算する

---

## 9. Phase 5: 学習パイプライン

### 9.1 `src/training/trainer.py`

#### 学習設定

```yaml
# configs/training.yaml
training:
  batch_size: 16
  lr: 1e-4
  weight_decay: 1e-4
  max_epochs: 100
  warmup_epochs: 5
  scheduler: cosine_annealing
  gradient_clip_norm: 1.0
  early_stopping_patience: 10
  checkpoint_dir: "data/checkpoints"
  log_interval: 50        # steps
  val_interval: 1          # epochs
  seed: 42
```

#### 学習ループの要件

- 標準的な PyTorch 学習ループ
- WandB でメトリクス記録 (loss, accuracy per head, val metrics)
- early stopping (val loss ベース)
- best model checkpoint の自動保存
- gradient clipping
- mixed precision training (torch.cuda.amp) 対応

### 9.2 `src/training/evaluator.py`

#### 評価メトリクス

1. **Event Detection mAP**: interaction_head の AP (IoU 的な基準は frame 距離)
2. **Action Accuracy**: マッチされたイベントでの action 分類精度
3. **Pointer Accuracy**: agent / target pointer の正解率
4. **Frame MAE**: 予測フレームと正解フレームの平均絶対誤差
5. **End-to-End Graph F1**: 予測 EventGraph と GT EventGraph をエッジ単位で比較した F1
   - エッジの一致条件: agent, action, target が全て一致 かつ frame 差が閾値以内

---

## 10. Phase 6: 推論 & DB格納パイプライン

### 10.1 `src/inference/pipeline.py`

#### 要件

- 動画ファイルパスを受け取り、EventGraph を返すエンドツーエンドのパイプライン
- スライディングウィンドウで処理し、重複区間のイベントは重複除去する
- 推論時に confidence threshold でフィルタリング

#### インターフェース

```python
class InferencePipeline:
    def __init__(
        self,
        sam3_tracker: SAM3Tracker,
        feature_extractor: FeatureExtractor,
        event_decoder: EventDecoder,      # 学習済み
        config: dict,
    ):
        ...

    def process_video(
        self,
        video_path: str,
        concept_prompts: list[str],
        confidence_threshold: float = 0.5,
    ) -> EventGraph:
        """
        動画全体を処理し、EventGraph を返す。

        処理フロー:
        1. Frame Sampler でフレーム抽出 (1fps)
        2. SAM 3 Tracker で物体追跡
        3. Feature Extractor で特徴量抽出
        4. スライディングウィンドウで Event Decoder 推論
        5. 重複イベントの除去 (NMS的な処理)
        6. EventGraph 構築

        Returns:
            EventGraph: ノード (物体) とエッジ (イベント) のグラフ構造
        """
        ...
```

### 10.2 `src/schemas/event_graph.py`

#### データ構造定義

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ObjectNode:
    """グラフのノード: 追跡された物体"""
    track_id: int
    category: str
    first_seen_frame: int
    last_seen_frame: int
    confidence: float
    attributes: dict = field(default_factory=dict)  # 将来の拡張用

@dataclass
class EventEdge:
    """グラフのエッジ: 物体間のインタラクションイベント"""
    event_id: str
    agent_track_id: int
    action: str
    target_track_id: int
    source_track_id: int | None = None
    destination_track_id: int | None = None
    frame: int = 0
    timestamp: datetime | None = None  # フレーム番号からFPS経由で算出
    confidence: float = 0.0

@dataclass
class EventGraph:
    """イベントグラフ全体"""
    video_id: str
    nodes: list[ObjectNode]
    edges: list[EventEdge]
    metadata: dict = field(default_factory=dict)  # fps, duration, camera_id 等

    def to_dict(self) -> dict:
        """JSON serializable な辞書に変換"""
        ...

    def get_object_timeline(self, track_id: int) -> list[EventEdge]:
        """特定物体に関連するイベントを時系列順に返す"""
        ...

    def get_events_in_range(self, start_frame: int, end_frame: int) -> list[EventEdge]:
        """指定フレーム範囲内のイベントを返す"""
        ...
```

### 10.3 `src/storage/models.py` — DBスキーマ

#### PostgreSQL テーブル定義

```python
# SQLAlchemy ORM モデル

class Video(Base):
    __tablename__ = "videos"
    id = Column(String, primary_key=True)         # video_id
    source_path = Column(String, nullable=False)
    fps = Column(Float, nullable=False)
    duration_sec = Column(Float)
    camera_id = Column(String)
    created_at = Column(DateTime, default=func.now())

class TrackedObject(Base):
    __tablename__ = "tracked_objects"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    category = Column(String, nullable=False)
    first_seen_frame = Column(Integer, nullable=False)
    last_seen_frame = Column(Integer, nullable=False)
    first_seen_time = Column(DateTime)
    last_seen_time = Column(DateTime)
    confidence = Column(Float)

    __table_args__ = (
        UniqueConstraint("video_id", "track_id"),
        Index("idx_category", "category"),
        Index("idx_video_category", "video_id", "category"),
    )

class Event(Base):
    __tablename__ = "events"
    id = Column(String, primary_key=True)          # event_id
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    frame = Column(Integer, nullable=False)
    timestamp = Column(DateTime)
    action = Column(String, nullable=False)
    agent_track_id = Column(Integer, nullable=False)
    target_track_id = Column(Integer, nullable=False)
    source_track_id = Column(Integer, nullable=True)
    destination_track_id = Column(Integer, nullable=True)
    confidence = Column(Float)

    __table_args__ = (
        Index("idx_video_frame", "video_id", "frame"),
        Index("idx_action", "action"),
        Index("idx_target", "video_id", "target_track_id"),
        Index("idx_timestamp", "timestamp"),
    )
```

#### インデックス設計の意図

- `idx_category`: 「全てのwrenchの履歴」検索用
- `idx_video_frame`: 「この動画のこの時間帯の全イベント」検索用
- `idx_action`: 「take_outイベントの全検索」用
- `idx_target`: 「特定物体に対する全アクション」検索用
- `idx_timestamp`: 時系列レンジクエリ用

### 10.4 `src/storage/db_writer.py`

```python
class DBWriter:
    def __init__(self, db_url: str):
        """
        Args:
            db_url: "postgresql://user:pass@host:port/dbname"
        """
        ...

    def write_event_graph(self, event_graph: EventGraph) -> None:
        """
        EventGraph を DB に書き込む。
        トランザクション内で Video → TrackedObject → Event の順に書き込む。
        """
        ...
```

---

## 11. Phase 7: 検索API

### 11.1 `src/storage/query_api.py`

#### 主要クエリ

```python
class EventQueryAPI:
    def __init__(self, db_url: str):
        ...

    def get_object_history(
        self,
        category: str,
        object_attributes: dict | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[Event]:
        """
        特定カテゴリの物体に関する全イベント履歴を時系列順に返す。
        例: "wrench" の全移動履歴
        """
        ...

    def get_object_current_location(
        self,
        category: str,
        video_id: str,
    ) -> dict:
        """
        特定物体の最新の所在地を返す。
        最後の place_on / put_in / move イベントの destination を返す。
        """
        ...

    def get_events_by_action(
        self,
        action: str,
        time_range: tuple[datetime, datetime] | None = None,
        agent_category: str | None = None,
    ) -> list[Event]:
        """
        特定アクションの全イベントを検索する。
        例: 過去1時間の全 "take_out" イベント
        """
        ...

    def get_location_contents(
        self,
        location_category: str,
        location_track_id: int,
        video_id: str,
        at_frame: int | None = None,
    ) -> list[TrackedObject]:
        """
        特定の場所（引き出し、棚など）に現在ある物体のリストを返す。
        put_in / take_out イベントの差分から算出する。
        """
        ...

    def search_events(
        self,
        query: dict,
    ) -> list[Event]:
        """
        汎用検索。query は以下のフィールドの任意の組み合わせ:
        {
            "action": str,
            "agent_category": str,
            "target_category": str,
            "time_range": [start, end],
            "video_id": str,
            "min_confidence": float,
        }
        """
        ...
```

---

## 12. テスト方針

### ユニットテスト

各モジュールに対して以下のテストを作成する:

| テスト対象 | テスト内容 |
|---|---|
| `frame_sampler` | FPS指定通りにフレーム抽出されるか、フレーム番号が正しいか |
| `sam3_tracker` | ダミー入力でインターフェースが正しく動くか、出力形式の検証 |
| `feature_extractor` | bbox/centroid/area の計算が正しいか、パディング処理 |
| `vlm_annotator` | プロンプト生成、JSONパース、バリデーション |
| `alignment` | 既知のGTマッピングと一致するか (合成テストケース) |
| `event_decoder` | forward passの出力shape検証、gradient flow確認 |
| `losses` | Hungarian matchingの正しさ、各損失値の妥当性 |
| `pipeline` | エンドツーエンドで EventGraph が生成されるか (小規模テスト) |
| `db_writer` | SQLite (テスト用) への書き込みと読み出しの整合性 |
| `query_api` | 各クエリメソッドが期待通りの結果を返すか |

### 統合テスト

- 短い動画 (10秒程度) を用いたエンドツーエンドテスト
- VLMアノテーション → アラインメント → 学習 → 推論 → DB格納 → 検索 の全工程

---

## 13. 補足: 設計判断の根拠

### なぜテキスト出力ではなくグラフ出力か

- 自己回帰デコードが不要 → 推論速度が速い
- 構造の整合性がアーキテクチャレベルで保証される（不正なJSONが生成されることがない）
- pointer機構により同一カテゴリの複数物体を正しく区別できる
- DB格納時の変換コストがゼロ（構造体をそのままORMに渡せる）

### なぜSAM 3をObject検出/追跡に使うか

- 物体認識・追跡は最も困難なサブタスクであり、SAM 3の848Mパラメータがこれを解決済み
- テキストプロンプトで追跡対象を指定できるため、新しい用具の追加に再学習不要
- メモリベーストラッカーにより、フレーム間の同一性追跡が高精度
- 学習対象をEvent Decoderのみ (~10M params) に絞れる → 少量データで蒸留可能

### なぜVLMで合成データを作るか

- 動画のイベントアノテーションは人手では非常にコストが高い
- VLMは複数フレーム入力で「何が起きたか」を自然言語で説明する能力がある
- 2段階プロンプト (objects → events) で構造化出力の品質を担保
- VLMの出力にはノイズがあるが、バリデーション + アラインメントフィルタで品質を維持

### Hungarian Matching を採用する理由

- イベント数は事前に未知（0〜M個）
- DETR/RelTRで実績のあるSet Prediction手法
- 順序に依存しない予測が可能（自己回帰の順序バイアスを回避）

---

## 実装優先順序

最小動作可能なパイプラインを最速で構築するための推奨順序:

1. **schemas** (event_graph.py, vlm_output.py) — データ構造の確定
2. **frame_sampler** — 動画入力の基盤
3. **sam3_tracker** + **feature_extractor** — SAM 3パイプラインの動作確認
4. **vlm_annotator** + **prompts** + **validator** — 合成データ生成
5. **alignment** + **build_dataset** — 学習データ構築
6. **event_decoder** + **heads** + **losses** — モデル実装
7. **trainer** + **evaluator** — 学習・評価
8. **pipeline** + **postprocess** — 推論パイプライン
9. **db_writer** + **models** + **query_api** — DB連携
