# 評価モジュール (`evaluation/`)

モデルの性能を測定する評価器とメトリクスレジストリ。

## Evaluator (`evaluator.py`)

検証用 DataLoader に対してモデルを評価し、設定されたメトリクスを計算する。

### 初期化

```python
evaluator = Evaluator(
    config=evaluation_config,  # metrics: ["event_detection_map", "action_accuracy", ...]
    device="cuda",
)
```

### 評価フロー

```python
metrics = evaluator.evaluate(model, val_loader)
# metrics: {"event_detection_map": 0.72, "action_accuracy": 0.85, ...}
```

**EventDecoder の場合:**
1. 各バッチの `EventBatch` をモデルに通す
2. 全バッチの predictions と gt_events を収集
3. Hungarian Matching で予測スロットと GT の対応を計算
4. 各メトリクス関数に aggregated な pred_dict / target_dict を渡す

**Legacy モデルの場合:**
- テンソルベースの predictions と targets を結合し、メトリクスを計算

### pred_dict / target_dict の構造

Evaluator が内部で構築する集約済みデータ:

**pred_dict:**
- `interaction`: 全スロットの sigmoid スコアのリスト
- `action_classes`: 活性スロット（score > 0.5）の argmax アクションクラス
- `agent_ptrs`, `target_ptrs`: 活性スロットのポインタ argmax
- `frame_indices`: 活性スロットのフレーム argmax
- `edges`: `(agent, action, target)` タプルのリスト

**target_dict:**
- `labels`: 各スロットが TP かどうか（1.0 or 0.0）
- `action_classes`, `agent_ptrs`, `target_ptrs`, `frame_indices`: GT 値
- `edges`: GT の `(agent, action, target)` タプルのリスト

## メトリクスレジストリ (`metrics.py`)

文字列名でメトリクス関数を取得する Factory。

```python
metric_fn = get_metric("graph_f1")
score = metric_fn(pred_dict, target_dict)
```

### 利用可能なメトリクス

| 名前 | 入力 | 説明 |
|---|---|---|
| `event_detection_map` | dict | イベント検出の平均精度（interaction score ベース） |
| `action_accuracy` | dict | マッチしたイベントのアクション分類精度 |
| `pointer_accuracy` | dict | agent/target ポインタの精度（平均） |
| `frame_mae` | dict | フレーム予測の平均絶対誤差 |
| `graph_f1` | dict | (agent, action, target) エッジの F1 スコア |
| `accuracy` | tensor | 汎用の分類精度（Legacy モデル用） |

### 各メトリクスの詳細

#### event_detection_map

interaction score を降順にソートし、precision-recall 曲線を計算して平均精度 (AP) を求める。

```
AP = ∫ precision(r) × is_positive(r) dr  (trapezoidal rule)
```

#### action_accuracy

マッチした予測スロットの action class を GT と比較した正答率。

#### pointer_accuracy

agent_ptr と target_ptr の正答率の平均。

```
pointer_accuracy = (agent_correct + target_correct) / (agent_total + target_total)
```

#### frame_mae

マッチした予測のフレーム index と GT フレーム index の平均絶対誤差。値が小さいほど良い。

#### graph_f1

予測エッジ集合と GT エッジ集合の F1 スコア。各エッジは `(agent, action, target)` のタプルで表現。

```
precision = |pred ∩ gt| / |pred|
recall = |pred ∩ gt| / |gt|
F1 = 2 × precision × recall / (precision + recall)
```

### レジストリの拡張

新しいメトリクスを追加するには:

```python
# metrics.py に関数を追加
def my_metric(predictions: dict, targets: dict) -> float:
    ...

# レジストリに登録
METRIC_REGISTRY["my_metric"] = my_metric
```

設定で使用:

```yaml
evaluation:
  metrics:
    - event_detection_map
    - my_metric
```
