<!-- meta-title: TensorFlow 自訂模型與訓練迴圈完整指南：GradientTape、自訂層與損失函數 -->
<!-- meta-description: 深入 TensorFlow 的自訂開發：自訂損失函數、自訂層（含可訓練參數）、自訂訓練迴圈（GradientTape）、自訂評估指標，以及如何儲存和載入含自訂物件的模型。 -->
<!-- meta-keywords: Python, TensorFlow, Keras, GradientTape, 自訂層, 自訂損失, 自訂訓練迴圈, 自動微分, 深度學習 -->
<!-- meta-hashtags: #Python #TensorFlow #Keras #GradientTape #自訂層 #自訂訓練 #深度學習 #程式設計 #教學 #DataScience -->

# 🐍 TensorFlow 自訂模型與訓練迴圈：GradientTape 完全指南

Keras 的高階 API 讓建模變得容易，但當你需要**非標準損失函數**、**特殊層結構**或**精細控制訓練過程**時，就需要深入 TensorFlow 的底層。本教學帶你掌握 TensorFlow 的自訂開發能力，讓你突破 Keras 預設 API 的限制。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🔢 TensorFlow 張量基礎](#tensors)
- [🎯 自訂損失函數](#custom-loss)
- [🏗️ 自訂層](#custom-layers)
- [📊 自訂評估指標](#custom-metrics)
- [🔄 自訂訓練迴圈 (GradientTape)](#custom-training-loop)
- [⚡ 自動微分深入](#autodiff)
- [💾 儲存與載入自訂模型](#saving)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **自訂損失函數**：繼承 `keras.losses.Loss` 或直接傳入函數，`sample_weight` 自動支援
- **自訂層**：繼承 `keras.layers.Layer`，在 `build()` 中建立可學習參數，在 `call()` 中定義前向傳播
- **GradientTape** 是 TF 的自動微分引擎，讓你完全控制梯度計算和參數更新
- **`@tf.function`** 將 Python 函數編譯成 TF 計算圖，加速推論（通常 2-5 倍）
- 自訂物件的模型儲存：必須提供 `get_config()` 方法才能正確反序列化

---

## <a id="tensors"></a>🔢 TensorFlow 張量基礎

💡 **實際應用情境：** 了解 TF 張量操作是進行自訂開發的前提，特別是在撰寫自訂損失函數時需要用 TF 操作（而非 NumPy）確保支援自動微分。

### 範例 1: 張量建立與操作

```python
import tensorflow as tf
import numpy as np

# 建立張量
t_const  = tf.constant([[1., 2., 3.],
                         [4., 5., 6.]])   # 不可變張量

t_var    = tf.Variable([[1., 2., 3.],
                         [4., 5., 6.]])   # 可變張量（可學習的參數）

print(f"形狀: {t_const.shape}, 資料類型: {t_const.dtype}")

# 基本操作（類似 NumPy，但在 GPU 上執行）
t_sq = tf.square(t_const)              # 元素平方
t_matmul = t_const @ tf.transpose(t_const)  # 矩陣乘法
t_concat = tf.concat([t_const, t_const], axis=0)  # 沿行方向串接

# 與 NumPy 互轉
np_array = t_const.numpy()             # Tensor → NumPy
tf_from_np = tf.constant(np_array)     # NumPy → Tensor

# 稀疏張量（NLP 中常用）
sparse = tf.SparseTensor(
    indices=[[0, 0], [1, 2]],
    values=[1., 2.],
    dense_shape=[3, 4]
)
print(tf.sparse.to_dense(sparse))
```

**✅ 程式碼逐行解析：**

1. `tf.constant`: 不可變，一旦建立就不能更改值（適合固定資料）
2. `tf.Variable`: 可變，訓練時 GradientTape 會追蹤其梯度（適合模型參數）
3. `t_const.numpy()`: 將張量轉回 NumPy 陣列（需要在 Eager Mode 下）

**🎯 重點摘要:**

- TF 的操作語法與 NumPy 非常相似，但在 GPU/TPU 上執行
- 自訂函數中必須使用 TF 操作（如 `tf.reduce_mean`）而非 NumPy，確保自動微分有效

---

## <a id="custom-loss"></a>🎯 自訂損失函數

💡 **實際應用情境：** 預測房價時，低估（預測值低於實際）比高估代價更高（銀行損失更大）。自訂非對稱損失函數讓你將業務邏輯編碼到訓練過程中。

### 範例 2: 自訂 Huber 損失函數

```python
# 方法 1：簡單函數（適合無超參數的損失）
def huber_loss(y_true, y_pred, threshold=1.0):
    """Huber 損失：小誤差用 MSE，大誤差用 MAE（對離群值更健壯）"""
    error = y_true - y_pred
    is_small_error = tf.abs(error) < threshold
    squared_loss = tf.square(error) / 2
    linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
    return tf.where(is_small_error, squared_loss, linear_loss)

# 方法 2：繼承 Loss 類別（支援超參數、序列化）
class HuberLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        """支援序列化（儲存/載入模型時必需）"""
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

# 使用自訂損失
model_huber = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(1)
])
model_huber.compile(
    optimizer="adam",
    loss=HuberLoss(threshold=1.5)  # 自訂損失函數
)
```

**✅ 程式碼逐行解析：**

1. `tf.where(condition, x, y)`: 元素級條件選擇（類似 `np.where`）
2. 繼承 `keras.losses.Loss` 的優點：自動支援 `sample_weight`、`reduction` 策略
3. `get_config()`: 返回超參數字典，讓模型可以正確儲存和載入

---

## <a id="custom-layers"></a>🏗️ 自訂層

### 範例 3: 含可訓練參數的自訂層

```python
class ResidualBlock(tf.keras.layers.Layer):
    """殘差塊（ResNet 的核心組件）"""

    def __init__(self, n_layers: int, n_neurons: int, **kwargs):
        super().__init__(**kwargs)
        # 在 __init__ 建立子層（但不建立權重，因為不知道輸入維度）
        self.hidden = [
            tf.keras.layers.Dense(n_neurons, activation="relu",
                                   kernel_initializer="he_normal")
            for _ in range(n_layers)
        ]

    def call(self, inputs):
        """定義前向傳播（建議在此用 @tf.function 加速）"""
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z  # 殘差連接（shortcut）

    def get_config(self):
        config = super().get_config()
        config.update({"n_layers": len(self.hidden),
                        "n_neurons": self.hidden[0].units})
        return config


class NormalizationLayer(tf.keras.layers.Layer):
    """自訂標準化層（示範 build 的用法）"""

    def build(self, batch_input_shape):
        """在第一次呼叫時根據輸入形狀建立權重"""
        n_inputs = batch_input_shape[-1]
        self.scale = self.add_weight(
            name="scale",
            shape=[n_inputs],
            initializer="ones"     # 初始化為全 1
        )
        self.offset = self.add_weight(
            name="offset",
            shape=[n_inputs],
            initializer="zeros"    # 初始化為全 0
        )
        super().build(batch_input_shape)

    def call(self, X):
        mean, variance = tf.nn.moments(X, axes=[0])
        return (X - mean) / (tf.sqrt(variance) + 1e-8) * self.scale + self.offset


# 使用自訂層
model_custom = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    NormalizationLayer(),
    ResidualBlock(n_layers=2, n_neurons=64),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

**✅ 程式碼逐行解析：**

1. `build(batch_input_shape)`: 在第一次 `call()` 時自動被呼叫，此時才知道輸入維度
2. `self.add_weight(...)`: 建立可學習的張量（會被 GradientTape 追蹤）
3. `inputs + Z`: 殘差連接——讓梯度繞過中間層直接傳回（解決深網路的梯度消失）

**🎯 重點摘要:**

- `build()` vs `__init__()` 的分工：`__init__` 建立子層（不知道維度）；`build` 建立需要輸入維度的參數
- 自訂層繼承後可以完全整合到 `model.fit()` / `model.save()` 流程中

---

## <a id="custom-metrics"></a>📊 自訂評估指標

### 範例 4: 自訂 Streaming Metric

```python
class RMSEMetric(tf.keras.metrics.Metric):
    """自訂 RMSE 評估指標（支援 streaming 累積計算）"""

    def __init__(self, name="RMSE", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_sq_error = self.add_weight(name="sum_sq_error", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """每個 batch 後更新累積統計量"""
        sq_error = tf.reduce_sum(tf.square(y_pred - y_true))
        self.sum_sq_error.assign_add(sq_error)
        self.total_samples.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        """計算最終指標值"""
        return tf.sqrt(self.sum_sq_error / self.total_samples)

    def reset_state(self):
        """每個 epoch 後重置"""
        self.sum_sq_error.assign(0.)
        self.total_samples.assign(0.)


# 使用自訂指標
model_metric = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])
model_metric.compile(optimizer="adam", loss="mse", metrics=[RMSEMetric()])
```

---

## <a id="custom-training-loop"></a>🔄 自訂訓練迴圈 (GradientTape)

### 範例 5: 完整的自訂訓練迴圈

```python
import tensorflow as tf
import numpy as np

# 準備資料
X_train = np.random.randn(1000, 8).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(1000).batch(32)

# 模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 優化器和損失
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function  # 編譯為計算圖（加速）
def train_step(X_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred = model(X_batch, training=True)  # 前向傳播
        loss   = loss_fn(y_batch, y_pred)        # 計算損失

    # 計算梯度
    gradients = tape.gradient(loss, model.trainable_variables)
    # 應用梯度更新模型參數
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy_metric.update_state(y_batch, y_pred)
    return loss

# 訓練迴圈
for epoch in range(5):
    epoch_losses = []
    accuracy_metric.reset_state()

    for X_batch, y_batch in dataset:
        batch_loss = train_step(X_batch, y_batch)
        epoch_losses.append(batch_loss.numpy())

    epoch_acc = accuracy_metric.result().numpy()
    print(f"Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}, acc={epoch_acc:.4f}")
```

**✅ 程式碼逐行解析：**

1. `with tf.GradientTape() as tape`: 進入 tape 上下文後，所有對 `trainable_variables` 的操作都被記錄
2. `tape.gradient(loss, model.trainable_variables)`: 對 `trainable_variables` 中每個變數計算損失的偏導數
3. `optimizer.apply_gradients(zip(gradients, variables))`: 根據梯度和優化器策略更新參數
4. `@tf.function`: 第一次呼叫時追蹤計算圖，後續呼叫直接執行圖（速度快 2-5 倍）

**🎯 重點摘要:**

- 自訂訓練迴圈給你**完全的控制**：自訂梯度（如梯度裁剪）、多個優化器、多個損失
- `model(X, training=True)` vs `model(X, training=False)`: BN 和 Dropout 在訓練/推論模式行為不同

---

## <a id="autodiff"></a>⚡ 自動微分深入

### 範例 6: GradientTape 高階用法

```python
# 計算二階梯度（梯度的梯度）
x = tf.Variable(3.0)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        y = x**3          # y = x³
    dy_dx = inner_tape.gradient(y, x)       # dy/dx = 3x² = 27
d2y_dx2 = outer_tape.gradient(dy_dx, x)    # d²y/dx² = 6x = 18

print(f"dy/dx   = {dy_dx.numpy():.1f}")    # 27
print(f"d²y/dx² = {d2y_dx2.numpy():.1f}") # 18

# 自訂梯度（如梯度裁剪）
@tf.custom_gradient
def clip_gradients(y):
    def backward(dy):
        return tf.clip_by_value(dy, -1.0, 1.0)  # 裁剪梯度
    return y, backward
```

---

## <a id="saving"></a>💾 儲存與載入自訂模型

### 範例 7: 序列化自訂物件

```python
# 方法 1：SavedModel 格式（推薦，保存計算圖）
model.save("my_model")  # 儲存為 SavedModel 目錄格式

# 方法 2：.keras 格式（新格式，需要自訂物件可序列化）
# 確保自訂損失/層有 get_config() 方法
model.save("my_model.keras")

# 載入含自訂物件的模型
loaded_model = tf.keras.models.load_model(
    "my_model.keras",
    custom_objects={
        "HuberLoss": HuberLoss,
        "ResidualBlock": ResidualBlock
    }
)

# 方法 3：只儲存權重（最輕量）
model.save_weights("my_weights.weights.h5")
# 需要重新建立模型結構才能載入
model.load_weights("my_weights.weights.h5")
```

**🎯 重點摘要:**

- `.keras` 格式是 Keras 3 推薦的格式，跨後端（TF/JAX/PyTorch）
- 自訂物件**必須**實作 `get_config()` 才能被 `load_model` 正確重建

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: 為什麼 GradientTape 只追蹤 `tf.Variable`？**

A: GradientTape 預設只追蹤 `tf.Variable`（可學習的參數）。若要計算對 `tf.Tensor` 的梯度，需要顯式呼叫 `tape.watch(tensor)`。

**Q2: `@tf.function` 有什麼限制？**

A: Python 的動態行為（如 `print`、Python 迴圈、列表操作）在 `@tf.function` 中可能不如預期——需要改用 TF 操作（`tf.print`、`tf.while_loop`）。初期不確定時，先不加 `@tf.function` 確保邏輯正確。

**Q3: 自訂訓練迴圈什麼時候比 `model.fit()` 更好？**

A: 需要以下功能時：(1) 多個損失函數分別優化不同部分（如 GAN）；(2) 動態調整訓練策略；(3) 自訂梯度累積；(4) 教學用途（理解訓練細節）。

**Q4: `build()` 和 `__init__()` 的差別？**

A: `__init__` 在實例化時呼叫，但此時不知道輸入維度（無法建立需要維度的權重）；`build()` 在第一次 `call()` 時自動觸發，此時輸入形狀已知，適合建立需要輸入維度的 `add_weight`。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #TensorFlow #Keras #GradientTape #自訂層 #自訂訓練 #深度學習 #自動微分 #自訂損失 #程式設計 #教學 #DataScience #MachineLearning #AI
