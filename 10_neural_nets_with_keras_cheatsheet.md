# Ch10 速查表：Neural Networks with Keras

> **核心主旨**：Keras 的三種建模 API + 訓練工作流程 —— `model.compile` / `model.fit` / Callbacks 是每次建模的核心。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Sequential API | 層的線性堆疊 | 標準 MLP/CNN，最常用 |
| Functional API | 可定義多輸入/多輸出/分支架構 | 複雜網路拓撲 |
| Subclassing API | 繼承 `tf.keras.Model`，完全自訂 | 研究用非標準架構 |
| `model.compile()` | 設定 optimizer、loss、metrics | 訓練前必做 |
| `model.fit()` | 執行訓練，傳回 history | 支援 validation_split/data |
| `EarlyStopping` | 監控 val_loss，提前停止防 overfit | 幾乎必用 |
| `ModelCheckpoint` | 自動儲存最佳模型 | 長時間訓練保險 |
| `TensorBoard` | 可視化訓練過程 | 除錯和比較實驗 |
| `keras_tuner` | 自動超參數搜索 | 調網路結構和 lr |

---

## 2. 關鍵 API 速查

| Keras Class / Function | 重點參數 | 用途 |
|------------------------|---------|------|
| `tf.keras.Sequential` | `layers=[...]` | 序列模型 |
| `tf.keras.layers.Dense` | `units=128`, `activation="relu"` | 全連接層 |
| `tf.keras.layers.Dropout` | `rate=0.2` | 正則化 |
| `tf.keras.layers.BatchNormalization` | – | 批次歸一化 |
| `model.compile()` | `optimizer`, `loss`, `metrics` | 設定訓練配置 |
| `model.fit()` | `epochs`, `batch_size`, `validation_split`, `callbacks` | 執行訓練 |
| `model.evaluate()` | – | 評估測試集 |
| `model.predict()` | – | 取得預測 |
| `model.save("model.keras")` | – | 儲存完整模型 |
| `tf.keras.models.load_model()` | – | 載入模型 |
| `EarlyStopping` | `patience=10`, `restore_best_weights=True` | 提前停止 |
| `ModelCheckpoint` | `save_best_only=True` | 儲存最佳 checkpoint |
| `TensorBoard` | `log_dir="logs/"` | TensorBoard 日誌 |

---

## 3. 必備代碼片段

```python
import tensorflow as tf

# Sequential API（最常用）
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[28, 28]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir="logs/")
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks
)

# Functional API（多輸入/多輸出）
input_ = tf.keras.layers.Input(shape=[28, 28])
hidden1 = tf.keras.layers.Dense(300, activation="relu")(input_)
hidden2 = tf.keras.layers.Dense(100, activation="relu")(hidden1)
output = tf.keras.layers.Dense(10, activation="softmax")(hidden2)
model_func = tf.keras.Model(inputs=input_, outputs=output)

# Subclassing API（研究用途）
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(300, activation="relu")
        self.dense2 = tf.keras.layers.Dense(100, activation="relu")
        self.out = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

# 儲存與載入
model.save("my_model.keras")
loaded_model = tf.keras.models.load_model("my_model.keras")
```

---

## 4. 常見陷阱

- **`loss="sparse_categorical_crossentropy"` vs `"categorical_crossentropy"`**：標籤是整數 → `sparse`；標籤是 one-hot → 不加 `sparse`。
- **`validation_split` 是從資料尾端切**：若資料有時間順序必須手動 shuffle 或用 `validation_data`。
- **`EarlyStopping(restore_best_weights=True)`**：不加這個選項，停止時的模型不是最佳的。
- **`BatchNormalization` 在 `training=True/False` 的行為不同**：`model.fit()` 自動處理，但在自訂訓練迴圈中要手動傳。
- **`model.save()` 的格式**：推薦 `.keras` 格式（新版），舊版 SavedModel 格式用 `model.save("dir/")`。

---

## 5. 決策指南

```
選哪種 Keras API？
├── 標準架構（MLP/CNN/RNN）   → Sequential
├── 多輸入/多輸出/殘差連接    → Functional API
└── 非標準前向傳播邏輯        → Subclassing API

Loss Function 選擇：
├── 二元分類      → binary_crossentropy
├── 多類別分類    → sparse_categorical_crossentropy（整數標籤）
├── 連續值迴歸    → mse 或 mae（有 outlier 用 mae）
└── 多標籤分類    → binary_crossentropy（每個輸出獨立 sigmoid）

Optimizer 建議：
├── 通用起點   → Adam(learning_rate=1e-3)
├── Fine-tuning → SGD(learning_rate=1e-4, momentum=0.9)
└── 研究       → AdamW 或 SGD with cosine annealing
```
