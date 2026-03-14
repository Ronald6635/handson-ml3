# TensorFlow 自訂模型與訓練教學

<!-- meta-title: TensorFlow 自訂模型與訓練：從基礎到進階實作 -->
<!-- meta-description: 深入探討 TensorFlow 自訂模型、損失函數、層、指標與訓練迴圈的完整指南，包含實務範例與最佳實踐。 -->
<!-- meta-keywords: TensorFlow, 自訂模型, 訓練迴圈, 損失函數, 機器學習, Python -->
<!-- meta-hashtags: #TensorFlow #自訂模型 #機器學習 #Python #深度學習 -->

## 🐍 TensorFlow 自訂模型與訓練：從基礎到進階實作

這份教學基於《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》第三版第12章，深入探討如何在 TensorFlow 中建立自訂模型、損失函數、層、指標以及訓練迴圈。無論你是初學者還是進階使用者，這份指南將帶你從基礎概念逐步建構複雜的機器學習應用。

## 📝 本文目錄
- 像 NumPy 一樣使用 TensorFlow
- 自訂損失函數
- 儲存與載入自訂物件的模型
- 其他自訂函數
- 自訂指標
- 自訂層
- 自訂模型
- 基於模型內部的損失
- 使用 Autodiff 計算梯度
- 自訂訓練迴圈
- TensorFlow 函數
- 與 tf.keras 一起使用 TF 函數
- 習題解答

## 🎯 關鍵重點 (Key Takeaways)
- 了解 TensorFlow 張量操作與 NumPy 的差異
- 實作自訂損失函數與指標
- 建立自訂層與模型架構
- 使用 GradientTape 進行自動微分
- 設計自訂訓練迴圈以獲得更多控制

## <a id="tensorflow-numpy"></a> 像 NumPy 一樣使用 TensorFlow

💡 **實際應用情境：** TensorFlow 的核心是張量操作，類似 NumPy 但支援 GPU 與自動微分。

### 張量與操作

#### 張量
```python
t = tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
t
```

**✅ 程式碼逐行解析：**

1. `第 1 行`: 建立一個 2x3 的常數張量矩陣

**🎯 重點摘要:**

- **核心功能**: 建立張量
- **潛在問題**: 張量不可變，需使用 Variable 進行修改
- **最佳使用情境**: 定義模型權重或輸入資料

#### 索引
```python
t[:, 1:]
```

**✅ 程式碼逐行解析：**

1. `第 1 行`: 使用 NumPy 風格索引取得張量的子集

#### 操作
```python
t + 10
tf.square(t)
t @ tf.transpose(t)
```

**✅ 程式碼逐行解析：**

1. `第 1 行`: 張量加法
2. `第 2 行`: 元素平方
3. `第 3 行`: 矩陣乘法

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

1. `第 3 行`: NumPy 陣列轉 TensorFlow 張量
2. `第 4 行`: TensorFlow 張量轉 NumPy 陣列
3. `第 5-6 行`: 展示兩者操作的互通性

### 型別轉換
```python
try:
    tf.constant(2.0) + tf.constant(40)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
```

**✅ 程式碼逐行解析：**

1. `第 2 行`: 嘗試混合型別運算，會引發錯誤

### 變數
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

1. `第 1 行`: 建立可變張量
2. `第 2-5 行`: 各種賦值操作

### 字串
```python
tf.constant(b"hello world")
tf.constant("café")
u = tf.constant([ord(c) for c in "café"])
b = tf.strings.unicode_encode(u, "UTF-8")
tf.strings.length(b, unit="UTF8_CHAR")
tf.strings.unicode_decode(b, "UTF-8")
```

**✅ 程式碼逐行解析：**

1. `第 1-2 行`: 建立字串張量
3. `第 3-6 行`: Unicode 編解碼操作

## <a id="custom-loss"></a> 自訂損失函數

💡 **實際應用情境：** Huber 損失函數對於處理離群值比 MSE 更穩定。

```python
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
```

**✅ 程式碼逐行解析：**

1. `第 2 行`: 計算預測誤差
2. `第 3 行`: 判斷是否為小誤差
3. `第 4-5 行`: 計算平方與線性損失
4. `第 6 行`: 根據條件選擇損失

**🎯 重點摘要:**

- **核心功能**: 實作 Huber 損失
- **潛在問題**: 閾值選擇影響效能
- **最佳使用情境**: 對離群值敏感的迴歸任務

## <a id="saving-loading"></a> 儲存與載入自訂物件的模型

💡 **實際應用情境：** 儲存包含自訂元件的模型需要特殊處理。

```python
model.save("my_model_with_a_custom_loss.keras")
model = tf.keras.models.load_model("my_model_with_a_custom_loss.keras",
                                   custom_objects={"huber_fn": huber_fn})
```

**✅ 程式碼逐行解析：**

1. `第 1 行`: 儲存模型
2. `第 2-3 行`: 載入時指定自訂物件

## <a id="other-custom-functions"></a> 其他自訂函數

💡 **實際應用情境：** 自訂活化函數、正規化器與約束條件。

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

1. `第 1-2 行`: 自訂 Softplus 活化函數
2. `第 4-6 行`: 自訂 Glorot 初始化器
3. `第 8-9 行`: 自訂 L1 正規化器
4. `第 11-12 行`: 自訂正權重約束

## <a id="custom-metrics"></a> 自訂指標

💡 **實際應用情境：** 自訂指標用於追蹤訓練過程中的特定效能測量。

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
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```

**✅ 程式碼逐行解析：**

1. `第 1-10 行`: 初始化與權重設定
2. `第 12-15 行`: 更新狀態
3. `第 17-18 行`: 計算結果
4. `第 20-22 行`: 配置方法

## <a id="custom-layers"></a> 自訂層

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

1. `第 1-5 行`: 初始化
2. `第 7-12 行`: 建構權重
3. `第 14-15 行`: 前向傳播
4. `第 17-19 行`: 配置序列化

## <a id="custom-models"></a> 自訂模型

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

1. `第 1-10 行`: 初始化元件
2. `第 12-17 行`: 前向傳播邏輯
3. `第 19-21 行`: 配置方法

## <a id="losses-internal"></a> 基於模型內部的損失

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

1. `第 1-8 行`: 初始化
2. `第 10-12 行`: 建構重建層
3. `第 14-20 行`: 前向傳播與損失計算

## <a id="autodiff"></a> 使用 Autodiff 計算梯度

💡 **實際應用情境：** GradientTape 允許精確控制梯度計算。

```python
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
```

**✅ 程式碼逐行解析：**

1. `第 1-2 行`: 定義函數
2. `第 4-6 行`: 使用 GradientTape 計算梯度

## <a id="custom-training"></a> 自訂訓練迴圈

💡 **實際應用情境：** 自訂訓練迴圈提供對訓練過程的完全控制。

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

1. `第 1-3 行`: 隨機批次抽樣函數
2. `第 5-9 行`: 狀態列印函數
3. `第 11-17 行`: 初始化訓練參數
4. `第 19-35 行`: 訓練迴圈

## <a id="tf-functions"></a> TensorFlow 函數

💡 **實際應用情境：** TF 函數將 Python 函數轉為圖形化執行以提升效能。

```python
def cube(x):
    return x ** 3

tf_cube = tf.function(cube)
```

**✅ 程式碼逐行解析：**

1. `第 1-2 行`: 定義函數
2. `第 4 行`: 轉為 TF 函數

## <a id="tf-keras"></a> 與 tf.keras 一起使用 TF 函數

💡 **實際應用情境：** 預設情況下，tf.keras 會自動將自訂元件轉為 TF 函數。

## <a id="exercises"></a> 習題解答

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

## 💡 總結與最佳實踐

TensorFlow 的自訂功能提供了極大的靈活性，讓你可以根據特定問題量身打造模型與訓練過程。記住以下原則：

1. **從簡單開始**：先使用內建元件，逐步加入自訂功能
2. **測試 thoroughly**：自訂元件需要額外的驗證
3. **效能考量**：TF 函數可以大幅提升效能
4. **可重用性**：設計元件時考慮重用性

## ❓ 常見問答 (FAQ)

**Q: 自訂層與內建層的差異？**
A: 自訂層提供完全控制，但需要實作更多細節。

**Q: 何時使用自訂訓練迴圈？**
A: 當需要特殊優化策略或對訓練過程有特定要求時。

## 🏷️ 推薦標籤 (Suggested Hashtags)

#TensorFlow #自訂模型 #機器學習 #Python #深度學習 #訓練迴圈 #神經網路 #程式設計
