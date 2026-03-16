---
name: scaffold-training
description: 深層学習モデル学習用のディレクトリ構成とテンプレートファイルを生成する
disable-model-invocation: true
allowed-tools: Bash(mkdir *), Bash(touch *), Write, Read, Edit, Glob
---

# scaffold-training

深層学習モデル学習用のプロジェクト scaffold を生成するスキルです。

## 引数

`$ARGUMENTS` でパッケージ名を受け取ります。省略時は `pyproject.toml` の `name` フィールドから取得し、ハイフンをアンダースコアに変換します。

## 実行手順

### Step 1: パッケージ名の決定

1. `$ARGUMENTS` が指定されている場合はそれをパッケージ名として使用
2. 指定されていない場合は `pyproject.toml` の `[project] name` を読み取る
3. パッケージ名のハイフン (`-`) をアンダースコア (`_`) に変換する

パッケージ名を `<pkg>` として以降の手順で使用する。

### Step 2: ディレクトリ構造の作成

以下のコマンドでディレクトリを一括作成する：

```bash
mkdir -p configs/experiment \
         data/raw data/processed data/splits \
         src/<pkg>/data src/<pkg>/models src/<pkg>/training src/<pkg>/evaluation src/<pkg>/utils \
         tests \
         notebooks checkpoints experiments
```

### Step 3: .gitkeep ファイルの作成

```bash
touch configs/experiment/.gitkeep notebooks/.gitkeep checkpoints/.gitkeep experiments/.gitkeep
```

### Step 4: テンプレートファイルの生成

`.claude/skills/scaffold-training/reference/structure-reference.md` を読み込み、各セクションの内容を対応するファイルパスに Write する。

**重要**: テンプレート内の `<pkg>` プレースホルダーをすべて実際のパッケージ名に置換すること。これはファイルパスだけでなく、ファイル内容（import文、docstring等）にも適用する。

生成するファイル一覧：

| ファイルパス | referenceのセクション |
|---|---|
| `configs/default.yaml` | configs/default.yaml |
| `src/<pkg>/__init__.py` | src/\<pkg\>/\_\_init\_\_.py |
| `src/<pkg>/config.py` | src/\<pkg\>/config.py |
| `src/<pkg>/data/__init__.py` | src/\<pkg\>/data/\_\_init\_\_.py |
| `src/<pkg>/data/dataset.py` | src/\<pkg\>/data/dataset.py |
| `src/<pkg>/data/collator.py` | src/\<pkg\>/data/collator.py |
| `src/<pkg>/data/transforms.py` | src/\<pkg\>/data/transforms.py |
| `src/<pkg>/models/__init__.py` | src/\<pkg\>/models/\_\_init\_\_.py |
| `src/<pkg>/models/base.py` | src/\<pkg\>/models/base.py |
| `src/<pkg>/training/__init__.py` | src/\<pkg\>/training/\_\_init\_\_.py |
| `src/<pkg>/training/trainer.py` | src/\<pkg\>/training/trainer.py |
| `src/<pkg>/training/optimizer.py` | src/\<pkg\>/training/optimizer.py |
| `src/<pkg>/evaluation/__init__.py` | src/\<pkg\>/evaluation/\_\_init\_\_.py |
| `src/<pkg>/evaluation/metrics.py` | src/\<pkg\>/evaluation/metrics.py |
| `src/<pkg>/evaluation/evaluator.py` | src/\<pkg\>/evaluation/evaluator.py |
| `src/<pkg>/utils/__init__.py` | src/\<pkg\>/utils/\_\_init\_\_.py |
| `src/<pkg>/utils/seed.py` | src/\<pkg\>/utils/seed.py |
| `src/<pkg>/utils/logging.py` | src/\<pkg\>/utils/logging.py |
| `src/<pkg>/utils/io.py` | src/\<pkg\>/utils/io.py |
| `scripts/train.py` | scripts/train.py |
| `scripts/evaluate.py` | scripts/evaluate.py |
| `scripts/preprocess.py` | scripts/preprocess.py |
| `scripts/predict.py` | scripts/predict.py |
| `tests/__init__.py` | tests/\_\_init\_\_.py |
| `tests/test_config.py` | tests/test_config.py |
| `tests/test_dataset.py` | tests/test_dataset.py |
| `tests/test_metrics.py` | tests/test_metrics.py |

### Step 5: .gitignore の更新

`.gitignore` に以下のエントリを追加する（既に存在するものはスキップ）：

```
# Training data
data/raw/
data/processed/

# Experiment artifacts
checkpoints/
experiments/
notebooks/
```

### Step 6: 既存ファイルの保護

**絶対に上書きしてはいけないファイル：**
- `scripts/vision_understanding.py`
- `data/images/` 以下のファイル
- `main.py`
- その他、生成対象でない既存ファイル

Write する前に、対象パスに既存ファイルがないか確認し、既存ファイルがある場合はスキップして警告を出す。
ただし `scripts/` ディレクトリ内の新規スクリプト（`train.py`, `evaluate.py`, `preprocess.py`, `predict.py`）は生成して良い。

### Step 7: 完了サマリー

作成されたディレクトリとファイルの一覧を表示する。以下の形式：

```
## Scaffold 生成完了

パッケージ名: <pkg>

### 作成されたディレクトリ
- configs/experiment/
- data/raw/, data/processed/, data/splits/
- src/<pkg>/ (data, models, training, evaluation, utils)
- tests/
- notebooks/, checkpoints/, experiments/

### 作成されたファイル
- configs/default.yaml
- src/<pkg>/config.py (+ 各サブモジュール)
- scripts/train.py, evaluate.py, preprocess.py, predict.py
- tests/test_config.py, test_dataset.py, test_metrics.py

### 次のステップ
1. `src/<pkg>/data/dataset.py` を実装（データの読み込みロジック）
2. `src/<pkg>/models/base.py` を実装（モデルアーキテクチャ）
3. `configs/default.yaml` を実験に合わせて調整
4. `scripts/train.py --config configs/default.yaml` で学習開始
```
