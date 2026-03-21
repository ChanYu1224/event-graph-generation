# アノテーションモジュール (`annotation/`)

VLM（Qwen 3.5）を使った合成アノテーションの生成、バリデーション、SAM 3 追跡結果とのアライメントを行う。

## 関連設定ファイル

| 設定 | ファイル | 内容 |
|---|---|---|
| オブジェクトカテゴリ・属性語彙 | [`configs/vocab.yaml`](../configs/vocab.yaml) | 27 カテゴリ（person, hand, chair, …）＋ 属性軸 6 種（color, material, position, size, state, orientation, pose） |
| アクション語彙 | [`configs/actions.yaml`](../configs/actions.yaml) | 13 アクション（take_out, put_in, pick_up, …, no_event）＋ source/destination 要否フラグ |
| VLM (transformers) | [`configs/vlm.yaml`](../configs/vlm.yaml) | Qwen3.5-9B, clip_length=16, clip_stride=8, target_fps=1.0 |
| VLM (vllm 直接) | [`configs/vlm_vllm.yaml`](../configs/vlm_vllm.yaml) | テンソル並列バックエンド |
| VLM (vllm-server) | [`configs/vlm_vllm_server.yaml`](../configs/vlm_vllm_server.yaml) | Qwen3.5-35B-A3B, OpenAI 互換 API, motion_filter 有効 |

## VLMAnnotator (`vlm_annotator.py`)

ローカルで Qwen 3.5 を推論し、動画フレーム列からイベントアノテーションを JSON として生成する。

### 初期化

```python
annotator = VLMAnnotator(
    model_name="Qwen/Qwen3.5-9B",
    device_map="auto",
    torch_dtype="bfloat16",
    max_new_tokens=4096,
    temperature=0.1,
    thinking=False,         # Qwen の思考モード
    quantization="none",    # "none", "4bit", "8bit"
)
```

- `AutoModelForImageTextToText` + `AutoProcessor` で HuggingFace transformers から読み込み
- `sdpa` アテンション実装を使用
- 量子化は bitsandbytes の `BitsAndBytesConfig` で設定

### クリップのアノテーション

```python
annotation = annotator.annotate_clip(
    frames=bgr_frames,         # list[np.ndarray] BGR (H,W,3)
    frame_indices=[0, 30, 60], # 元動画でのフレーム番号
    video_id="video_001",
    categories=["person", "wrench", "drawer"],
    actions=["pick_up", "put_in", "use"],
)
# annotation: VLMAnnotation
```

**処理フロー:**
1. 各フレームの左上にフレーム番号をオーバーレイ（`_overlay_frame_number`）
2. BGR → RGB → PIL 変換
3. 2 段階プロンプト（System + User）を構築
4. `apply_chat_template` でトークナイズ
5. `model.generate()` で推論
6. 生成トークンのみデコード（入力部分を除外）
7. JSON パース → Pydantic バリデーション

### 動画全体のアノテーション

```python
annotations = annotator.annotate_video(
    video_path="data/raw/video.mp4",
    fps=1.0,
    clip_length=16,     # 1クリップあたりのフレーム数
    clip_stride=8,      # クリップ間のストライド
    categories=[...],
    actions=[...],
)
# annotations: list[VLMAnnotation]
```

内部で `FrameSampler` を使いフレームを抽出し、スライディングウィンドウで `annotate_clip` を繰り返す。

### 出力パース

`_parse_output()` は以下のケースを処理する:
- `</think>` ブロックの除去（Qwen 3.5 の思考出力）
- ` ```json ... ``` ` コードブロックの抽出
- 素の JSON テキストのパース

## プロンプト (`prompts.py`)

VLM に渡すプロンプトテンプレートを管理する。

### 2 段階プロンプト構成

**System Prompt** — VLM の役割を定義:
1. **Step 1 (Object Listing)**: フレーム中のすべてのオブジェクトを列挙
   - `obj_id`: `category_NN` 形式（例: `person_01`, `cup_02`）
   - `category`: 許可リストから選択
   - `first_seen_frame`: 初出フレーム
   - `attributes`: 属性リスト
2. **Step 2 (Event Listing)**: 発生したイベントを列挙
   - `event_id`: `evt_NNN` 形式
   - `frame`, `action`, `agent`, `target`, `source`, `destination`

**User Prompt** — クリップの具体情報:
- フレーム数、サンプリング FPS
- JSON スキーマの例示（Few-Shot）

```python
system_prompt, user_prompt = build_prompt(
    categories=["person", "wrench"],
    actions=["pick_up", "put_in"],
    n_frames=16,
    fps=1.0,
)
```

## AnnotationValidator (`validator.py`)

VLM 出力の品質を保証するバリデーター。

### バリデーション項目

| チェック | レベル | 対応 |
|---|---|---|
| スキーマ | Pydantic | 構築時に自動検証 |
| 語彙（action, category） | 警告/破棄 | 不正な action → イベント破棄 |
| 参照整合性 | 破棄 | agent/target が objects に存在しない → イベント破棄 |
| 時系列順序 | 自動修正 | イベントをフレーム順にソート |
| source/destination 論理 | 警告 | action が要求するのに未指定 → 警告のみ |

### 使い方

```python
validator = AnnotationValidator(
    allowed_actions=["pick_up", "put_in", ...],
    allowed_categories=["person", "wrench", ...],
    action_config=action_entries,  # actions.yaml の dict リスト
)

corrected, warnings = validator.validate(annotation)
# corrected: VLMAnnotation（修正済み）
# warnings: list[str]（警告メッセージ）
```

**バッチバリデーション:**

```python
validated_list, stats = validator.validate_batch(annotations)
# stats: {"total_events_input": 50, "total_events_output": 45, "discard_rate": 0.1, ...}
```

## Aligner (`alignment.py`)

VLM が生成したオブジェクト（`obj_id`）と SAM 3 のトラック（`track_id`）を対応付ける。

### アルゴリズム

1. 各 VLM オブジェクトに対し、カテゴリ名が一致する SAM 3 トラックを候補として抽出
2. VLM オブジェクトの `first_seen_frame` で、候補の bbox IoU に基づくコスト行列を構築
3. `scipy.optimize.linear_sum_assignment`（Hungarian Matching）で最適割当
4. IoU 閾値以下の割当をフィルタリング

### マッチングヒューリスティック

VLM オブジェクトには明示的な bbox がないため、以下のヒューリスティックを使用:

- **同カテゴリのオブジェクトが 1 つだけ**: スコア 1.0（明確なマッチ）
- **複数の同カテゴリオブジェクト**: 空間的分離度を計算
  - 候補の bbox と他の同カテゴリオブジェクトの IoU を計算
  - 分離度 = `1.0 - max_iou_with_others`
  - 検出スコアで重み付け

### 出力

```python
aligner = Aligner(iou_threshold=0.3)
result = aligner.align(tracking_results, vlm_annotation)

result.mapping       # {"person_01": 0, "wrench_01": 3, ...}  (vlm_obj_id → sam3_track_id)
result.unmatched_vlm # ["cup_02"]  マッチしなかった VLM オブジェクト
result.unmatched_sam3 # [5, 7]  マッチしなかった SAM 3 トラック
result.confidence    # 0.85  マッチした割当の平均 IoU
```
