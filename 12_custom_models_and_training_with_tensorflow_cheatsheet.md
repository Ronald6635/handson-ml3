# Ch12 速查表：Custom Models & Training with TensorFlow

> **核心主旨**：打開 Keras 的黑盒子 —— `GradientTape` 自訂訓練迴圈、自訂 Layer/Loss/Metric 讓你掌控每一步。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| `tf.Tensor` | 不可變的多維陣列（類 numpy），支援 GPU / 自動微分 | TF 的基本計算單元 |
| `tf.Variable` | 可變的張量，用於儲存模型權重 | 自訂層的參數 |
| `tf.GradientTape` | 記錄前向計算，用於計算梯度（自動微分） | 自訂訓練迴圈 |
| Custom Loss Function | 接受 `(y_true, y_pred)` 並返回純量張量的函數 | Huber Loss、Focal Loss 等 |
| Custom Metric | 繼承 `tf.keras.metrics.Metric`，支援跨批次累積 | Precision@K 等複雜指標 |
| Custom Layer | 繼承 `tf.keras.layers.Layer`，定義 `build()` + `call()` | 殘差塊、自注意力層 |
| Custom Model | 繼承 `tf.keras.Model`，自訂 `call()` 的資料流 | 非標準架構 |
| `@tf.function` | 將 Python 函數編譯為 TF Graph，大幅加速 | 高頻呼叫的函數 |

---

## 2. 關鍵 API 速查

| TF / Keras API | 重點說明 | 用途 |
|----------------|---------|------|
| `tf.constant(...)` | 建立不可變張量 | 建立資料 |
| `tf.Variable(...)` | 建立可變張量（權重） | 自訂層的權重 |
| `tf.cast(x, tf.float32)` | 類型轉換 | 型別不符時必用 |
| `tf.GradientTape()` | 前向計算錄影帶 | 計算梯度 |
| `tape.gradient(loss, variables)` | 計算梯度 | 取得 dLoss/dW |
| `optimizer.apply_gradients(zip(grads, vars))` | 更新權重 | 手動梯度下降 |
| `tf.keras.losses.Huber()` | Huber loss（內建） | 對 outlier 魯棒的迴歸 |
| `tf.keras.layers.Layer` | 自訂層的基類 | 定義 `build()` + `call()` |
| `tf.keras.Model` | 自訂模型的基類 | 定義複雜資料流 |
| `@tf.function` | Graph 模式加速裝飾器 | 加速純 TF 函數 |

---

## 3. 必備代碼片段

```python
import tensorflow as tf

# 基本張量操作
t = tf.constant([[1., 2.], [3., 4.]])
v = tf.Variable([[1., 2.], [3., 4.]])
v.assign(v + 1)      # 原地更新
v.assign_add([[1, 1], [1, 1]])

# 自訂 Loss Function
def huber_fn(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < delta
    squared_loss = tf.square(error) / 2
    linear_loss = delta * tf.abs(error) - delta**2 / 2
    return tf.where(is_small_error, squared_loss, linear_loss)

model.compile(loss=huber_fn, optimizer="adam")

# 自訂 Metric（跨批次累積）
class HuberMetric(tf.keras.metrics.Metric):
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.huber_fn = huber_fn
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_metrics = self.huber_fn(y_true, y_pred, self.delta)
        self.total.assign_add(tf.reduce_sum(sample_metrics))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

# 自訂 Layer
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):  # 在知道輸入形狀後才建立權重
        self.W = self.add_weight(name="W",
                                  shape=(input_shape[-1], self.units),
                                  initializer="glorot_normal")
        self.b = self.add_weight(name="b", shape=(self.units,),
                                  initializer="zeros")
        super().build(input_shape)

    def call(self, X):
        return self.activation(X @ self.W + self.b)

# GradientTape 自訂訓練迴圈
@tf.function  # 編譯為 Graph，加速執行
def train_step(X_batch, y_batch, model, optimizer, loss_fn, metric):
    with tf.GradientTape() as tape:
        y_pred = model(X_batch, training=True)
        main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
        loss = tf.add_n([main_loss] + model.losses)  # 含正則化損失
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metric.update_state(y_batch, y_pred)
    return loss

# 完整自訂訓練迴圈
n_epochs = 5
batch_size = 32
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metric = tf.keras.metrics.MeanAbsoluteError()

for epoch in range(n_epochs):
    for step in range(len(X_train) // batch_size):
        X_batch = X_train[step*batch_size:(step+1)*batch_size]
        y_batch = y_train[step*batch_size:(step+1)*batch_size]
        loss = train_step(X_batch, y_batch, model, optimizer, loss_fn, metric)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, MAE: {metric.result():.4f}")
    metric.reset_state()
```

---

## 4. 常見陷阱

- **`tf.Variable` 需要在 `tape` 的 watch 範圍內**：`tape` 預設只錄 `tf.Variable`，非 variable 張量需手動 `tape.watch(tensor)`。
- **`tf.function` 的 tracing**：第一次呼叫會追蹤（建立 Graph），後續呼叫用快取。傳入不同形狀的張量會觸發重新追蹤，影響速度。
- **型別轉換**：TF 不做隱式型別轉換，`float32` 和 `float64` 混合會報錯，用 `tf.cast()` 手動轉換。
- **`model.losses`**：Layer 用 `self.add_loss()` 新增的正則化損失，在自訂迴圈中要手動加進總損失。

---

## 5. 決策指南

```
何時用自訂訓練迴圈？
├── 需要不同層不同 lr      → 是（GradientTape 分開計算）
├── GAN、MAML 等複雜訓練   → 是
├── 標準訓練流程           → 否，用 model.fit()（更快更穩）

自訂 Layer 的結構：
├── __init__: 儲存超參數（units, activation...）
├── build: 建立權重（self.add_weight()），延遲到知道 input_shape
└── call: 前向傳播邏輯，接受 training=False 參數
```
