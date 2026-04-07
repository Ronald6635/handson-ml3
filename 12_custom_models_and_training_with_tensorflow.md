<!-- meta-title: TensorFlow 自訂模型與訓練：從基礎到進階實作 -->
<!-- meta-description: 深入探討 TensorFlow 自訂模型、損失函數、層、指標與訓練迴圈的完整指南，包含實務範例與最佳實踐。 -->
<!-- meta-keywords: TensorFlow, 自訂模型, 訓練迴圈, 損失函數, 機器學習, Python, 深度學習 -->
<!-- meta-hashtags: #TensorFlow #自訂模型 #機器學習 #Python #深度學習 -->

# 🐍 TensorFlow 自訂模型與訓練：從基礎到進階實作

這份教學基於《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》第三版第12章，深入探討如何在 **TensorFlow** 中建立**自訂模型 (Custom Models)**、**損失函數 (Loss Functions)**、**層 (Layers)**、**指標 (Metrics)** 以及**訓練迴圈 (Training Loops)**。無論你是初學者還是進階使用者，這份指南將帶你從基礎概念逐步建構複雜的機器學習應用。

## 📝 本文目錄

- [像 NumPy 一樣使用 TensorFlow](#tensorflow-numpy)
- [自訂損失函數 (Custom Loss Functions)](#custom-loss)
- [儲存與載入自訂物件的模型 (Saving/Loading Models with Custom Objects)](#saving-loading)
- [其他自訂函數 (Other Custom Functions)](#other-custom-functions)
- [自訂指標 (Custom Metrics)](#custom-metrics)
- [自訂層 (Custom Layers)](#custom-layers)
- [自訂模型 (Custom Models)](#custom-models)
- [基於模型內部的損失 (Losses Based on Model Internals)](#losses-internal)
- [使用 Autodiff 計算梯度 (Autodiff)](#autodiff)
- [自訂訓練迴圈 (Custom Training Loops)](#custom-training)
- [TensorFlow 函數 (TF Functions)](#tf-functions)
- [與 tf.keras 一起使用 TF 函數 (Using TF Functions with tf.keras)](#tf-keras)
- [習題解答 (Exercises)](#exercises)

## 🎯 關鍵重點 (Key Takeaways)

- 了解 TensorFlow **張量 (Tensor)** 操作與 NumPy 的差異。
- 實作自訂損失函數與指標，提升模型彈性。
- 建立自訂層與模型架構，打造獨特網路。
- 使用 **GradientTape (梯度帶)** 進行**自動微分 (Autodiff)**，精確控制梯度計算。
- 設計自訂訓練迴圈，掌握訓練過程的每一個細節。

---

## 像 NumPy 一樣使用 TensorFlow {#tensorflow-numpy}

💡 **實際應用情境：** **TensorFlow (TF)** 的核心是**張量操作 (Tensor Operations)**，類似 NumPy 但支援 GPU 運算與**自動微分 (Autodiff)**。

### 張量與操作

#### 張量
```python
t = tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
t
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 使用 `tf.constant` 建立一個 $2 \times 3$ 的常數張量矩陣，包含浮點數。
    - 注意：`tf.constant` 建立的張量是**不可變 (immutable)** 的，無法直接修改其值。
    - `dtype` 依輸入值自動推斷：浮點數會選 `tf.float32`，整數會選 `tf.int32`。例如 `tf.constant([[1, 2, 3], [4, 5, 6]])` 的預設 dtype 是 `tf.int32`；若要硬性指定，請使用 `dtype=tf.float32`（或其他支援類型）。

**🎯 重點摘要:**

-   **核心功能**: 建立不可變的張量。
-   **潛在問題**: 張量不可變，若需修改值，應使用 `tf.Variable`。
-   **最佳使用情境**: 定義模型權重或輸入資料。

#### 索引
```python
t[:, 1:]
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 使用 NumPy 風格的索引方式 `[:, 1:]` 取得張量的子集，表示所有行、從第二列到最後一列。

#### 操作
```python
t + 10
tf.square(t)
t @ tf.transpose(t)
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 對張量 `t` 中的每個元素執行加 10 的操作。
2.  `第 2 行`: 使用 `tf.square` 計算張量 `t` 中每個元素的平方。
3.  `第 3 行`: 使用 `@` 符號執行矩陣乘法，將 `t` 與其**轉置 (transpose)** `tf.transpose(t)` 相乘。

### 張量與 NumPy 互轉
```python
import numpy as np

a = np.array([2., 4., 5.])
tf.constant(a)
t.numpy()
np.array(t)
tf.square(a)
np.square(t)
```

**✅ 程式碼逐行解析：**

1.  `第 3 行`: 將 NumPy 陣列 `a` 轉換為 TensorFlow 張量。
2.  `第 4 行`: 將 TensorFlow 張量 `t` 轉換為 NumPy 陣列。
3.  `第 5-6 行`: 展示 TensorFlow 操作可以直接應用於 NumPy 陣列 (`tf.square(a)`)，以及 NumPy 操作可以直接應用於 TensorFlow 張量 (`np.square(t)`)，顯示兩者良好的互通性。

### 型別轉換
> 這一節示範當兩個張量的 dtype 不一致時，TensorFlow 會產生錯誤，並介紹如何使用 `tf.cast()` 做型別統一。

```python
try:
    tf.constant(2.0) + tf.constant(40)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
```

**✅ 程式碼逐行解析：**

1.  `第 2 行`: 嘗試將 `float32` 類型的 `tf.constant(2.0)` 與 `int32` 類型的 `tf.constant(40)` 相加。
2.  `第 3-4 行`: 預期會捕獲 `tf.errors.InvalidArgumentError`，因為 TensorFlow 不會自動進行不同資料型別的隱式轉換。

**解決方法：**

-   `tf.constant(2.0)` 的 dtype 為 `tf.float32`。
-   `tf.constant(40)` 的 dtype 為 `tf.int32`。
-   這兩者直接相加會報錯，因為 TensorFlow 要求運算張量 dtype 一致。

**調整方法：**

```python
try:
    tf.constant(2.0) + tf.constant(40)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
    val_float = tf.cast(tf.constant(40), dtype=tf.float32)
    print("After casting to float32:", tf.constant(2.0) + val_float)
```

**🎯 重點摘要:**

-   TensorFlow 張量運算需要相容的 dtype。
-   使用 `tf.cast()` 做顯式型別轉換可以避免錯誤。
-   這是處理不同型別張量時的推薦模式。

### 變數

這一節說明 `tf.Variable` 如何建立可變張量，並示範如何用 `assign` 和 `scatter_nd_update` 等 API 逐元素更新內容。

```python
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
v
v.assign(2 * v)
v[0, 1].assign(42)
v[:, 2].assign([0., 1.])
v.scatter_nd_update(
    indices=[[0, 0], [1, 2]], updates=[100., 200.])
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 使用 `tf.Variable` 建立一個可變張量 `v`，常用於模型權重。
2.  `第 2 行`: 顯示變數 `v` 的當前值。
3.  `第 3 行`: 使用 `assign` 方法將 `v` 的所有元素更新為其兩倍的值。
4.  `第 4 行`: 將 `v` 中索引為 `[0, 1]` 的元素（第一列第二行）更新為 `42`。
5.  `第 5 行`: 將 `v` 中所有列、第三行的元素更新為 `[0., 1.]`。
6.  `第 6-7 行`: 使用 `scatter_nd_update` 根據提供的 `indices` 和 `updates` 進行稀疏更新，將 `[0, 0]` 更新為 `100.`，`[1, 2]` 更新為 `200.`。

### 字串

這一節展示 TensorFlow 如何處理字串資料，包括位元組字串、Unicode 字串、編碼與解碼操作。

```python
tf.constant(b"hello world")
tf.constant("café")
u = tf.constant([ord(c) for c in "café"])
b = tf.strings.unicode_encode(u, "UTF-8")
tf.strings.length(b, unit="UTF8_CHAR")
tf.strings.unicode_decode(b, "UTF-8")
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 建立一個位元組字串張量 `b"hello world"`。
2.  `第 2 行`: 建立一個 Unicode 字串張量 `"café"`。
3.  `第 3 行`: 將字串 `"café"` 轉換為 Unicode 編碼的整數張量 `u`。
    - `ord(c)` 會返回字元 `c` 的 Unicode 代碼點（例如 `ord('c')` 返回 `99`）。
    - 因此 `u` 會包含 `[99, 97, 102, 233]`，分別對應於 `c`, `a`, `f`, `é` 的 Unicode 代碼點。
4.  `第 4 行`: 使用 `tf.strings.unicode_encode` 將 `u` 編碼為 UTF-8 位元組字串張量 `b`。
    - `b` 的值將是 `b'caf\xc3\xa9'`，其中 `\xc3\xa9` 是 `é` 的 UTF-8 編碼。
5.  `第 5 行`: 使用 `tf.strings.length` 計算 `b` 中每個字串的 UTF-8 字元長度。
6.  `第 6 行`: 使用 `tf.strings.unicode_decode` 將 `b` 解碼回 Unicode 字串張量。
    - 解碼後的張量將包含 `[99, 97, 102, 233]` 等 Unicode 代碼點；若要再轉回 `'café'` 字串，可搭配 `tf.strings.unicode_encode` 或其他字串處理函式進一步轉換。

---

## 自訂損失函數 (Custom Loss Functions) {#custom-loss}

💡 **實際應用情境：** **Huber 損失函數 (Huber Loss Function)** 對於處理**離群值 (Outliers)** 比**均方誤差 (Mean Squared Error, MSE)** 更穩定。

```python
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
```

**✅ 程式碼逐行解析：**

1.  `第 2 行`: 計算真實值 `y_true` 與預測值 `y_pred` 之間的誤差 `error`。
2.  `第 3 行`: 判斷誤差的絕對值是否小於 `1`，儲存為布林張量 `is_small_error`。
3.  `第 4 行`: 如果誤差小，計算平方損失 ($error^2 / 2$)。
4.  `第 5 行`: 如果誤差大，計算線性損失 ($|error| - 0.5$)。
5.  `第 6 行`: 使用 `tf.where` 根據 `is_small_error` 的條件，選擇性地返回 `squared_loss` 或 `linear_loss`。

**🎯 重點摘要:**

-   **核心功能**: 實作 Huber 損失，能減少離群值對訓練的影響。
-   **潛在問題**: 閾值 (threshold) 的選擇會影響損失函數的行為。
-   **最佳使用情境**: 對離群值敏感的迴歸任務。

```python
# visualize the Huber loss function
y_pred = tf.linspace(-3, 3, 100)
y_true = tf.constant(0., dtype=y_pred.dtype)
import matplotlib.pyplot as plt
plt.plot(y_pred, huber_fn(y_true, y_pred), label="Huber Loss (delta=1)", linewidth=2)
plt.axvline(x=1, color='r', linestyle='--', alpha=0.5, label="Threshold (delta)")
plt.axvline(x=-1, color='r', linestyle='--', alpha=0.5)
plt.title("Huber Loss Visualization")
plt.xlabel("Error (y_true - y_pred)")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()
```

---

## 儲存與載入自訂物件的模型 (Saving/Loading Models with Custom Objects) {#saving-loading}

💡 **實際應用情境：** 儲存包含**自訂元件 (Custom Components)** 的模型時，需要特別處理，以便在載入時 TensorFlow 能夠識別這些自訂物件。

```python
model.save("my_model_with_a_custom_loss.keras")
model = tf.keras.models.load_model("my_model_with_a_custom_loss.keras",
                                   custom_objects={"huber_fn": huber_fn})
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 使用 `model.save` 將訓練好的模型儲存為 Keras 格式 (`.keras` 副檔名)。
2.  `第 2-3 行`: 使用 `tf.keras.models.load_model` 載入模型，並透過 `custom_objects` 參數明確告知 TensorFlow 如何解析自訂的 `huber_fn` 損失函數。

**🎯 重點摘要:**

-   **核心功能**：保存自訂元件時需明確告知載入器對應的函數/類別。
-   **潛在問題**：未提供 `custom_objects` 會導致 `ValueError: Unknown loss function` 等錯誤。
-   **最佳使用情境**：部署時將模型搬到不同環境（例如雲端服務），必須確保所有自訂程式碼可用。

---

## 其他自訂函數 (Other Custom Functions) {#other-custom-functions}

💡 **實際應用情境：** 除了損失函數，你也可以自訂**活化函數 (Activation Functions)**、**權重初始化器 (Weight Initializers)**、**正規化器 (Regularizers)** 和**約束條件 (Constraints)**。

```python
def my_softplus(z):
    return tf.math.log(1.0 + tf.exp(z))

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)
```

**✅ 程式碼逐行解析：**

1.  `第 1-2 行`: 定義一個自訂的 **Softplus 活化函數 (Softplus Activation Function)**，計算 $log(1 + e^z)$。
2.  `第 4-6 行`: 定義一個自訂的 **Glorot 權重初始化器 (Glorot Weight Initializer)**，根據輸入和輸出單元數計算標準差，然後生成常態分佈的隨機值。
3.  `第 8-9 行`: 定義一個自訂的 **L1 正規化器 (L1 Regularizer)**，計算權重絕對值的總和並乘以一個因子。
4.  `第 11-12 行`: 定義一個自訂的 **正權重約束 (Positive Weight Constraint)**，將所有負權重設置為零。

---

## 自訂指標 (Custom Metrics) {#custom-metrics}

💡 **實際應用情境：** 自訂指標用於追蹤訓練過程中的特定效能測量。
自訂指標（Custom Metrics）讓你能夠在訓練過程中追蹤特定的效能指標，除了內建的準確率、損失等，也可以根據任務需求設計專屬的評估方式。例如在回歸問題中，除了均方誤差（MSE），你可能會希望追蹤 Huber 損失或其他自訂指標。自訂指標通常繼承自 `tf.keras.metrics.Metric`，並實作 `update_state()`、`result()` 及 `get_config()` 方法，確保在訓練、驗證與模型儲存時都能正確運作。

```python
class HuberMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_metrics = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(sample_metrics))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```

**✅ 程式碼逐行解析：**

1.  `第 1-10 行` (`__init__`): 初始化指標，設定 `threshold`，並用 `add_weight` 建立 `total`（累計損失）與 `count`（樣本計數）兩個狀態變數，確保跨批次累加。
2.  `第 12-15 行` (`update_state`): 每次 `update_state()` 被呼叫（batch 內），計算該 batch 的 Huber 損失：`total += batch_loss_sum`，`count += batch_size`。
    - 注意：此實作值用 `tf.shape(y_true)[0]` 取得樣本數（而不是 `tf.size(y_true)` 元素數），避免在多維輸出時產生元素平均誤差。
3.  `第 17-18 行` (`result`): 回傳 `tf.math.divide_no_nan(self.total, self.count)`（平均 Huber 損失），避免 `count=0` 分母錯誤。
4.  `第 20-22 行` (`get_config`): 實作序列化邏輯，確保 `threshold` 儲存在 model.save()/load_model 的 custom_objects 設定裡。

**🎯 重點摘要:**

-   **核心功能**：可在訓練過程中保持指標狀態（累積損失、樣本計數）。
-   **注意**：如果手動訓練迴圈，需在每個 epoch 結束呼叫 `metric.reset_state()`；`model.fit()` 內建會自動處理。
-   **最佳使用情境**：對 epoch 平均值/整體訓練趨勢進行跟蹤（如 PR 曲線、累計誤差）。

---

## 自訂層 (Custom Layers) {#custom-layers}

💡 **實際應用情境：** 自訂層允許實作特殊的網路架構元件。

```python
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="he_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation)}
```

**✅ 程式碼逐行解析：**

1.  `第 1-5 行` (`__init__`): 接收超參數（如 `units`）並將 `activation` 字串轉換為對應的活化函數。同時呼叫父類別 `super().__init__(**kwargs)` 以處理標準 Keras 參數（如 `name` 或 `trainable`）。
2.  `第 7-12 行` (`build`): 建立層的權重 (`kernel` 與 `bias`)。這是在層第一次被使用時調用的，我們可以在此時取得輸入形狀 (`batch_input_shape[-1]`) 來動態決定權重矩陣的大小。
3.  `第 14-15 行` (`call`): 定義前向傳播邏輯，執行矩陣乘法並加上偏差項，最後應用活化函數。
4.  `第 17-19 行` (`get_config`): 儲存超參數。這對於模型的序列化至關重要，確保載入模型時能正確重建該自訂層。

**🎯 重點摘要:**

-   **核心功能**：自訂層可以實現複雜功能（例如自注意力、自訂正則化／約束）。
-   **潛在問題**：若未實作 `get_config()`，模型儲存/載入會失敗。
-   **最佳使用情境**：需要複用自訂動作、或希望透過 `model.save()` 向他人分享層時。

---

## 自訂模型 (Custom Models) {#custom-models}

💡 **實際應用情境：** 自訂模型提供對網路架構的完全控制。

```python
class ResidualRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden1 = tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "output_dim": self.output_dim}
```

**✅ 程式碼逐行解析：**

1.  `第 1-10 行` (`__init__`): 初始化模型元件，包括設定輸出維度 `output_dim`，定義一個標準的 `Dense` 隱藏層 `hidden1`，以及兩個 `ResidualBlock` (假設已定義的自訂層)。
2.  `第 12-17 行` (`call`): 定義模型的前向傳播邏輯。輸入 `inputs` 首先通過 `hidden1` 層，然後依序通過多個 `ResidualBlock`。這個範例中，`block1` 被重複使用了 4 次 (1 + 3)，接著通過 `block2`，最後輸出到 `out` 層。
3.  `第 19-21 行` (`get_config`): 實作序列化，確保 `output_dim` 超參數在模型儲存與載入時能夠被正確地保存與重建。

**🎯 重點摘要:**

-   **核心功能**：自訂模型讓你自由定義資料流與計算邏輯（例如多分支、條件分支、共用權重）。
-   **潛在問題**：若在 `call()` 內建立變數，會導致重複建立或 tracing 失敗。
-   **最佳使用情境**：當 `Sequential` / functional API 不夠時（例如多任務模型、連續記憶模型）。

---

## 基於模型內部的損失 (Losses Based on Model Internals) {#losses-internal}

💡 **實際應用情境：** 將重建損失加入主要損失以改善模型泛化。

```python
class ReconstructingRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
                       for _ in range(5)]
        self.out = tf.keras.layers.Dense(output_dim)

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = tf.keras.layers.Dense(n_inputs)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)
```

**✅ 程式碼逐行解析：**

1.  `第 1-8 行` (`__init__`): 初始化模型結構，建立一系列隱藏層並設定輸出層 `out`，為後續 `build` 與 `call` 做準備。
2.  `第 10-12 行` (`build`): 根據輸入形狀動態建立 `reconstruct` 層（輸入維度為 `n_inputs`），這裡在第一次呼叫模型時被執行以確保權重形狀正確。
3.  `第 14-20 行` (`call`): 前向傳播中計算重建輸出 `reconstruction`，接著計算重建損失 `recon_loss`，使用 `add_loss()` 將加權重建損失併入模型總損失，最後回傳主輸出 `out(Z)`。

**🎯 重點摘要:**

-   **核心功能**：透過 `add_loss()` 將自訂損失併入模型總損失，讓 `model.compile()` / `model.fit()` 一併優化。
-   **潛在問題**：若在 `call()` 中多次調用 `add_loss()` 可能導致損失重複計算。
-   **最佳使用情境**：多任務訓練、或在網路內部加入自監督損失（如重建、對比學習）。

---

## 使用 Autodiff 計算梯度 (Autodiff) {#autodiff}

💡 **實際應用情境：** `tf.GradientTape` 允許精確控制梯度計算。

```python
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
```

**✅ 程式碼逐行解析：**

1. `第 1-2 行` (`f`, `w1`, `w2`): 定義函數 `f(w1, w2)`，並建立兩個可訓練的變數 `w1` 與 `w2`（使用 `tf.Variable`）。
2. `第 4-6 行` (`GradientTape` 上下文與梯度計算): 使用 `with tf.GradientTape() as tape:` 來追蹤變數上的運算，於上下文內計算標量輸出 `z = f(w1, w2)`，然後呼叫 `tape.gradient(z, [w1, w2])` 取得 `z` 對 `w1` 與 `w2` 的梯度（偏導數）。

**🎯 重點摘要:**

-   **核心功能**：`GradientTape` 允許你手動控制梯度計算範圍與更新時機。
-   **潛在問題**：若不妥善管理 `tape` 的生命週期（例如不釋放 persistent tape），會造成記憶體爆增。
-   **最佳使用情境**：自訂優化流程、混合優化器、多任務優化、進階梯度裁剪。

---

## 自訂訓練迴圈 (Custom Training Loops) {#custom-training}

💡 **實際應用情境：** 自訂訓練迴圈提供對訓練過程的完全控制。

本節示範如何從頭實作自訂訓練迴圈，包含批次抽樣、手動前向/反向傳播、整合模型內部損失（如 `add_loss()`）、以及更新與重置度量。透過這些範例，你將能在多優化器、對抗訓練或進階梯度處理等非標準情境下精細掌控訓練流程。

```python
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

def print_status_bar(step, total, loss, metrics=None):
    metrics = " - ".join([f"{m.name}: {m.result():.4f}"
                          for m in [loss] + (metrics or [])])
    end = "" if step < total else "\n"
    print(f"\r{step}/{total} - " + metrics, end=end)

n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.MeanAbsoluteError()]

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train_scaled, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)

        print_status_bar(step, n_steps, mean_loss, metrics)

    for metric in [mean_loss] + metrics:
        metric.reset_state()
```

**✅ 程式碼逐行解析：**

1. `第 1-3 行` (`random_batch`): 隨機抽樣一個小批次，回傳 `(X_batch, y_batch)`，用於在訓練迴圈中取樣訓練資料。
2. `第 5-9 行` (`print_status_bar`): 列印訓練進度與目前指標（例如 loss、accuracy），以便即時監控每個 step 的狀態。
3. `第 11-17 行` (訓練參數與度量初始化): 設定 `n_epochs`、`batch_size`、`optimizer`、`loss_fn`，並建立要追蹤的度量（例如 `mean_loss`、`metrics` 列表）。
4. `第 19-35 行` (訓練迴圈主體): 主要訓練流程：從 `random_batch` 取得批次、使用 `tf.GradientTape()` 計算 forward 與總損失（`main_loss` + `model.losses`）、計算並套用梯度、更新並記錄度量，最後在每個 epoch 結束時重置指標狀態。

**🎯 重點摘要:**

-   **核心功能**：自訂訓練迴圈可讓你整合非標準訓練邏輯（例如對抗訓練、教師學生、分段優化）。
-   **潛在問題**：必須手動照顧所有指標和狀態重置，容易忘記 `metric.reset_state()`。
-   **最佳使用情境**：需要實作多優化器、梯度累積、或精細的批次內追蹤。

---

## TensorFlow 函數 (TF Functions) {#tf-functions}

💡 **實際應用情境：** `@tf.function` 可將 Python 函數轉換為高效可重用的計算圖 (Computation Graph)，對於追求效能或部署至 TensorFlow Serving 等環境時非常重要。

```python
@tf.function
def cube(x):
    return x ** 3

# 第一次呼叫會觸發 tracing (追蹤)，之後會重用相同計算圖
tf_cube = cube
print(tf_cube(tf.constant(2.0)))
```

**✅ 程式碼逐行解析：**

1. `第 1 行`: 使用 `@tf.function` 將 `cube` 函數轉換為 TF 函數，這會在第一次執行時建立計算圖。
2. `第 3 行`: 定義函數的實際運算內容。
3. `第 7 行`: 呼叫 `tf_cube`（即 `cube`），觸發 tracing 並執行圖。

**🎯 重點摘要:**

-   **核心功能**：將 Python 代碼轉換為高效計算圖，改善執行效能。
-   **潛在問題**：若在 TF 函數內動態建立變數或使用 Python 控制流，可能導致頻繁 tracing 或錯誤。
-   **最佳使用情境**：模型推論、批次處理、需要跨平台執行的情境。

### Tracing 行為觀察

```python
@tf.function
def traced_add(x, y):
    print("Tracing...")
    return x + y

# 第一次執行會看到 Tracing 訊息
traced_add(tf.constant(1), tf.constant(2))
# 第二次執行不會再 tracing（使用同樣的 input signature）
traced_add(tf.constant(3), tf.constant(4))
```

### `run_eagerly=True`（禁用計算圖）

若你需要逐步偵錯或必須執行動態 Python 控制流程，可在 `model.compile()` 中啟用：

```python
model.compile(optimizer="nadam", loss="mse", run_eagerly=True)
```

> ⚠️ 注意：`run_eagerly=True` 會降低效能，但可在開發/偵錯階段提升可讀性。

---

## 與 tf.keras 一起使用 TF 函數 (Using TF Functions with tf.keras) {#tf-keras}

💡 **實際應用情境：** TensorFlow/Keras 預設會將自訂 layer、loss、metric 等轉成 TF 函數，以優化訓練與推論效能。

**🎯 重點摘要:**

-   Keras 會自動將大多數 `call()` 轉成 TF 函數，因此請遵守 TF 函數規則：避免在 `call()` 裡建立新變數、避免使用 Python side-effect（例如列表/字典的 append、print、random 等），盡量只使用 TensorFlow 運算。
-   若你需要混用任意 Python 代碼（例如第三方套件、條件邏輯、除錯輸出），可以使用 `tf.py_function()`（但會降低效能且降低模型可移植性），或在訓練/推論時啟用 `run_eagerly=True` 以強制以 eager 模式執行。

    ```python
    def numpy_op(x):
        import numpy as np
        return np.log(x)

    @tf.function
    def wrapped(x):
        # tf.py_function 會將 numpy_op 包裝為 TensorFlow op
        y = tf.py_function(func=numpy_op, inp=[x], Tout=tf.float32)
        return y + 1.0

    print(wrapped(tf.constant([1.0, 2.0, 3.0])))
    ```

-   在 Keras 3 之後，`dynamic` 參數已不再存在；如果你仍需「動態模型」行為，可透過 `run_eagerly=True` 或在模型/層級別使用 `tf.keras.layers.Layer`/`tf.keras.Model` 時自行控制 Python 執行流程（但這會關閉圖優化並影響效能）。
-   若在 `@tf.function` 內要 log 訊息，請使用 `tf.print()` 以確保在 graph/已編譯模式下也能正確輸出。

---

## 習題解答 (Exercises) {#exercises}

### 12. 實作 Layer Normalization 層

```python
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, eps=0.001, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, batch_input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=batch_input_shape[-1:],
            initializer="ones")
        self.beta = self.add_weight(
            name="beta", shape=batch_input_shape[-1:],
            initializer="zeros")

    def call(self, X):
        mean, variance = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.alpha * (X - mean) / (tf.sqrt(variance + self.eps)) + self.beta

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "eps": self.eps}
```

### 13. 使用自訂訓練迴圈訓練 Fashion MNIST

```python
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test.astype(np.float32) / 255.

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = tf.keras.losses.sparse_categorical_crossentropy
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        status = OrderedDict()
        mean_loss(loss)
        status["loss"] = mean_loss.result().numpy()
        for metric in metrics:
            metric(y_batch, y_pred)
            status[metric.name] = metric.result().numpy()
        
    y_pred = model(X_valid)
    status["val_loss"] = np.mean(loss_fn(y_valid, y_pred))
    status["val_accuracy"] = np.mean(tf.keras.metrics.sparse_categorical_accuracy(
        tf.constant(y_valid, dtype=np.float32), y_pred))
    
    for metric in [mean_loss] + metrics:
        metric.reset_state()
```

---

## 💡 總結與最佳實踐

TensorFlow 的自訂功能提供了極大的彈性，讓你可以針對特定問題打造專屬架構與訓練流程。記住以下原則：

1. **從簡單開始**：先使用內建元件，必要時再抽換自訂邏輯。
2. **測試 thoroughly**：自訂元件應有對應單元測試與數值驗證。
3. **效能考量**：`@tf.function` 可提高效能，但請優先在穩定後啟用。
4. **可重用性**：讓自訂層/模型可序列化 (`get_config`) 會大幅提升跨專案使用性。
5. **版本管理**：訓練環境、TensorFlow 版本與依賴需要同步，避免儲存後載入失敗。

---

## ❓ 常見問答 (FAQ)

**Q: 自訂層與內建層的差異？**
A: 內建層是經過優化、測試與廣泛支援的元件，適合大多數情境。自訂層則提供你對計算流程與變數管理的完全控制，適合需要特殊行為或自訂梯度的案例。

**Q: 何時使用自訂訓練迴圈 (Custom Training Loop)？**
A: 當你需要異常流程（如多優化器、策略梯度、對抗訓練、分段 loss）或要在每個 step 中插入複雜邏輯（如動態梯度裁剪、梯度累積）時，自訂訓練迴圈是最靈活的方式。

**Q: 為什麼我的 `tf.function` 一直 retrace（重複 tracing）？**
A: 常見原因包含輸入 shape 或 dtype 變化、在函數內建立新變量、或使用 Python list/dict 每次傳入不同結構。解法：使用 `input_signature` 固定形狀，避免在函數內建立變量，或改用 `tf.constant`/`tf.Tensor`。

**Q: 儲存自訂模型時出現 `Unknown loss function`？**
A: 這表示載入時未提供對應的 `custom_objects` 或自訂類別未能正確序列化。確保你的自訂 Loss/Metric/Layer 有 `get_config()`，並在 `load_model(..., custom_objects={...})` 中註冊。

**Q: 怎麼在 tf.keras 中 debug TF 函數？**
A: 可使用 `run_eagerly=True`（會降低效能），或用 `tf.print()` 取代 `print()`，並搭配 `tf.autograph.to_code()` 查看轉換後的圖。

---

## 🏷️ 推薦標籤 (Suggested Hashtags)

#TensorFlow #自訂模型 #機器學習 #Python #深度學習 #訓練迴圈 #神經網路 #程式設計
