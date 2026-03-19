# 学習モジュール (`training/`)

Event Decoder の学習ループと、optimizer/scheduler の構築を担当する。

## Trainer (`trainer.py`)

PyTorch の学習ループを実装するクラス。AMP、early stopping、per-head loss logging、WandB 連携をサポート。

### 初期化

```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    evaluator=evaluator,
    criterion=criterion,   # EventGraphLoss
)
```

### 機能

| 機能 | 説明 |
|---|---|
| **AMP** | CUDA 環境で自動的に有効化。`torch.amp.GradScaler` を使用 |
| **Gradient Clipping** | `grad_clip_norm` で最大勾配ノルムを制限（デフォルト 1.0） |
| **Early Stopping** | `early_stopping_patience` エポック改善なしで学習を停止 |
| **チェックポイント** | best モデルの保存 + `save_every_n_epochs` で定期保存 |
| **WandB** | wandb がインストール済みかつ `config.wandb.enabled=True` の場合に有効 |
| **Per-head loss logging** | interaction, action, agent_ptr 等のヘッドごとの損失を記録 |

### 学習ループの流れ

```
for epoch in range(epochs):
    1. _train_one_epoch()
       - バッチごとに forward → loss → backward → optimizer.step()
       - EventBatch の場合: model(emb, temporal, pairwise, mask) → criterion(preds, gt, mask)
       - AMP の場合: scaler で scale/unscale/step
    2. scheduler.step()
    3. evaluator.evaluate() (eval_every_n_epochs ごと)
    4. wandb.log() (全メトリクスを 1 回の呼び出しで記録)
    5. Early stopping チェック
    6. チェックポイント保存
```

### 学習の再開

```python
trainer.resume("data/checkpoints/epoch_0050.pt")
trainer.train()
```

`resume()` はモデルの `state_dict`、optimizer の状態、エポック番号を復元する。

### チェックポイントの保存先

```
{checkpoint_dir}/
├── best.pt              # val_loss が最良のモデル
├── epoch_0010.pt        # 定期保存（save_every_n_epochs=10）
├── epoch_0020.pt
└── ...
```

## Optimizer / Scheduler (`optimizer.py`)

### build_optimizer

```python
optimizer = build_optimizer(model, training_config)
```

| 名前 | 実装 |
|---|---|
| `adam` | `torch.optim.Adam` |
| `adamw` | `torch.optim.AdamW` |
| `sgd` | `torch.optim.SGD` |

### build_scheduler

```python
scheduler = build_scheduler(optimizer, training_config)
```

| 名前 | 実装 | 備考 |
|---|---|---|
| `cosine` | `CosineAnnealingLR` | `T_max` で周期を設定 |
| `cosine_warmup` | `CosineAnnealingLR` | warmup 付き |
| `step` | `StepLR` | - |
| `none` | `None` | スケジューラなし |

warmup は `scheduler_params.warmup_epochs` で指定し、学習率を線形に上昇させる。

## 設定パラメータ

`configs/training.yaml` のデフォルト値:

```yaml
training:
  epochs: 100
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  optimizer: adamw
  scheduler: cosine_warmup
  scheduler_params:
    T_max: 100
    warmup_epochs: 5
  grad_clip_norm: 1.0
  seed: 42
  device: cuda
  checkpoint_dir: "data/checkpoints"
  save_every_n_epochs: 10
  early_stopping_patience: 10

  loss_weights:
    interaction: 2.0
    action: 1.0
    agent_ptr: 1.0
    target_ptr: 1.0
    source_ptr: 0.5
    dest_ptr: 0.5
    frame: 0.5
```
