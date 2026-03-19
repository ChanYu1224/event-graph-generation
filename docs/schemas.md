# スキーマ (`schemas/`)

フレームワーク全体で使用されるデータ構造の定義。

## EventGraph (`event_graph.py`)

推論パイプラインの最終出力。動画中の「誰が・何を・どこへ」をグラフ構造で表現する。

### ObjectNode

追跡されたオブジェクト（グラフのノード）。

| フィールド | 型 | 説明 |
|---|---|---|
| `track_id` | `int` | SAM 3 が割り当てたトラック ID |
| `category` | `str` | オブジェクトカテゴリ（例: "person", "wrench"） |
| `first_seen_frame` | `int` | 最初に検出されたフレーム番号 |
| `last_seen_frame` | `int` | 最後に検出されたフレーム番号 |
| `confidence` | `float` | 検出信頼度 |
| `attributes` | `dict` | 追加属性（色、サイズ等） |

### EventEdge

オブジェクト間のインタラクション（グラフのエッジ）。

| フィールド | 型 | 説明 |
|---|---|---|
| `event_id` | `str` | イベント ID（`evt_NNNN` 形式） |
| `agent_track_id` | `int` | 行為者の track_id |
| `action` | `str` | アクションラベル（13 種類） |
| `target_track_id` | `int` | 対象物の track_id |
| `source_track_id` | `int \| None` | 取り出し元の track_id（任意） |
| `destination_track_id` | `int \| None` | 格納先の track_id（任意） |
| `frame` | `int` | イベント発生フレーム |
| `timestamp` | `datetime \| None` | タイムスタンプ |
| `confidence` | `float` | 予測信頼度 |

### EventGraph

動画全体の完全なイベントグラフ。

| フィールド | 型 | 説明 |
|---|---|---|
| `video_id` | `str` | 動画識別子 |
| `nodes` | `list[ObjectNode]` | オブジェクトノードのリスト |
| `edges` | `list[EventEdge]` | イベントエッジのリスト |
| `metadata` | `dict` | メタデータ（フレーム数等） |

**主要メソッド:**

- `to_dict()` — JSON シリアライズ可能な dict に変換
- `to_json(path)` — JSON 文字列に変換、ファイル保存も可能
- `get_object_timeline(track_id)` — 特定オブジェクトに関連するイベントをフレーム順で取得
- `get_events_in_range(start, end)` — フレーム範囲内のイベントを取得

### 出力例

```json
{
  "video_id": "assembly_001",
  "nodes": [
    {"track_id": 0, "category": "person", "first_seen_frame": 0, "last_seen_frame": 120, "confidence": 1.0, "attributes": {}},
    {"track_id": 1, "category": "wrench", "first_seen_frame": 10, "last_seen_frame": 95, "confidence": 0.95, "attributes": {}}
  ],
  "edges": [
    {"event_id": "evt_0000", "agent_track_id": 0, "action": "pick_up", "target_track_id": 1, "source_track_id": 2, "destination_track_id": null, "frame": 15, "confidence": 0.92}
  ],
  "metadata": {"num_frames_sampled": 120, "concept_prompts": ["person", "wrench", "drawer"]}
}
```

## VLM 出力スキーマ (`vlm_output.py`)

VLM（Qwen 3.5）が生成する JSON アノテーションの Pydantic バリデーションモデル。

### VLMObject

| フィールド | 型 | 制約 | 説明 |
|---|---|---|---|
| `obj_id` | `str` | 正規表現: `^[a-z_]+_\d+$` | オブジェクト ID（`category_NN` 形式） |
| `category` | `str` | - | カテゴリ名 |
| `first_seen_frame` | `int` | >= 0 | 初出フレーム |
| `attributes` | `list[str]` | - | 属性リスト（例: `["red", "metal"]`） |

### VLMEvent

| フィールド | 型 | 制約 | 説明 |
|---|---|---|---|
| `event_id` | `str` | 正規表現: `^evt_\d+$` | イベント ID |
| `frame` | `int` | >= 0 | 発生フレーム |
| `action` | `str` | - | アクション名 |
| `agent` | `str` | - | 行為者の obj_id |
| `target` | `str` | - | 対象の obj_id |
| `source` | `str \| None` | - | 取り出し元（任意） |
| `destination` | `str \| None` | - | 格納先（任意） |

### VLMAnnotation

| フィールド | 型 | 説明 |
|---|---|---|
| `objects` | `list[VLMObject]` | 検出オブジェクトのリスト |
| `events` | `list[VLMEvent]` | イベントのリスト |

`VLMAnnotation` は Pydantic `BaseModel` を継承しており、`model_validate(dict)` で JSON dict からインスタンスを生成し、自動的にスキーマバリデーションが行われる。
