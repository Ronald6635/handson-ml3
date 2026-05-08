# 課程講義：深度電腦視覺與卷積神經網路 (Chapter 14)

## 導論

本章聚焦於深度電腦視覺的核心技術：**卷積神經網路（Convolutional Neural
Networks, CNNs）**。在機器學習體系中，CNN 是處理影像資料的基礎架構，
能夠從原始像素中自動學習局部特徵、空間結構與層次表示。

你將理解卷積 (Convolution) 運算的數學原理、池化 (Pooling) 的設計哲學、ResNet 殘差單元 (Residual Unit) 的動機，
以及如何將預訓練模型應用於轉移學習（Transfer Learning）與微調。

本章內容是後續目標檢測、語義分割與視覺 Transformer 的重要基礎。

---

## 1. 卷積層 (Convolutional Layer, Conv)：從影像到特徵圖

### 1.1 為什麼重要？

卷積層是 CNN 的基本構建模組。它透過**局部感受野（local receptive field）**
與**共享權重（weight sharing）**，將影像中的邊緣、紋理和形狀轉換為特徵圖
（feature maps），大幅降低參數數量並強化平移不變性。

相較於全連接層，卷積層在影像任務中有兩大優勢：

- **稀疏連接**：每個神經元只連接輸入的一個局部區域，減少參數量。
- **特徵可重用**：同一濾波器在影像各位置共享權重，能偵測同一特徵。

### 1.2 輸出形狀公式

在卷積神經網路中，**卷積核（kernel）** 在輸入特徵圖上滑動，產生輸出特徵圖。這個過程會受到 **補零（padding）** 策略與 **步幅（stride）** 的影響，進而決定輸出尺寸。

首先，讓我們先明確公式中的符號：
*   $H$: 輸入特徵圖的高度（由於卷積運算通常在寬度 $W$ 方向也應用相同的邏輯，因此這些公式也適用於寬度）。
*   $k$: 卷積核的尺寸。
*   $s$: 卷積運算的步幅，即卷積核每次移動的像素數。

### valid padding（不補零）

當我們採用 **valid padding** 時，表示在輸入特徵圖的邊界不進行任何補零操作。卷積核只會在其能夠*完全覆蓋輸入區域的位置*進行計算。這意味著卷積核在輸入特徵圖的邊緣部分無法進行運算，因此輸出特徵圖的尺寸通常會比輸入特徵圖小。其輸出高度（或寬度）的計算公式為：

$$
\text{output} = \left\lfloor \frac{H - k}{s} \right\rfloor + 1
$$

這個公式可以這樣理解：$H - k$ 代表在不考慮步幅的情況下，卷積核可以從最左上角移動到最右下角的「有效滑動範圍」。將此範圍除以步幅 $s$，再使用**地板函數 (floor function)** $ \lfloor \cdot \rfloor $ 向下取整，得到卷積核可以滑動的「步數」。最後加上 $1$，是因為起始位置也算一個輸出。這種方式能確保每個輸出像素都完全由原始輸入數據計算而來，不會引入補零帶來的額外資訊。

### same padding（補零以維持尺寸）

相對地，**same padding** 的目標是透過在輸入特徵圖的周圍增加零值像素（補零），使得輸出特徵圖的空間尺寸能夠盡可能地與輸入特徵圖保持「相同」或按比例縮小。當步幅 $s=1$ 時，輸出尺寸將與輸入尺寸完全相同；當步幅 $s>1$ 時，輸出尺寸會約為輸入尺寸除以步幅。

對於最常見的步幅 $s=1$ 且卷積核尺寸 $k$ 為奇數的情況，在輸入特徵圖的每個空間維度（例如高度或寬度）上，每一側（頂/底或左/右）的補零量 $P$ 通常計算為：

$$
P = \frac{k-1}{2}
$$

這表示在每個空間維度上，總共會補上 $k-1$ 個零。例如，當卷積核尺寸 $k=3$ 時，每一側會補上 $(3-1)/2 = 1$ 個零。當卷積核尺寸 $k=7$ 時，每一側會補上 $(7-1)/2 = 3$ 個零。

如果卷積核尺寸 $k$ 為偶數，或者步幅 $s>1$，補零的數量可能會不對稱（例如，某一側比另一側多一個像素），以精確達成目標輸出尺寸。然而，這些細節通常由深度學習框架自動處理，你只需要指定 `padding="same"` 即可。

其輸出高度（或寬度）的計算公式為：

$$
\text{output} = \left\lceil \frac{H}{s} \right\rceil
$$

這個公式相對簡潔，它直接將輸入高度 $H$ 除以步幅 $s$，並使用**天花板函數 (ceiling function)** $ \lceil \cdot \rceil $ 向上取整。這確保了即使 $H$ 無法被 $s$ 整除，也能夠透過足夠的補零來覆蓋整個輸入區域，使卷積核能夠完成所有的運算。在深度學習模型中，`same padding` 常用於保持特徵圖的空間尺寸，以便於構建更深層次的網路，避免過快地丟失邊緣資訊或使特徵圖尺寸歸零。

### 1.3 核心代碼

這段程式碼示範如何在 TensorFlow/Keras 中使用 `Conv2D` 層，並比較 `valid` 與 `same` padding 以及步幅對輸出特徵圖大小的影響。

```python
import tensorflow as tf
from sklearn.datasets import load_sample_images

images = load_sample_images()["images"]
images = tf.keras.layers.CenterCrop(height=70, width=120)(images) # Shape (2, 70, 120, 3)
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)

# valid padding（預設）
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
fmaps = conv_layer(images)
print(f'Shape (valid padding): {fmaps.shape[1:3].as_list()}') # 輸出特徵圖的空間尺寸

# same padding
conv_layer_same = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="same")
fmaps_same = conv_layer_same(images)
print(f'Shape (same padding): {fmaps_same.shape[1:3].as_list()}')

# same + stride=2（下採樣）
conv_layer_stride = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="same", strides=2)
fmaps_stride = conv_layer_stride(images)
print(f'Shape (same padding, stride=2): {fmaps_stride.shape[1:3].as_list()}')
```

### 1.4 手動濾波器示範

```python
import numpy as np

filters = np.zeros([7, 7, 3, 2])
filters[:, 3, :, 0] = 1   # 第 1 個濾波器：偵測垂直線
filters[3, :, :, 1] = 1   # 第 2 個濾波器：偵測水平線
biases = tf.zeros([2])
fmaps = tf.nn.conv2d(
    images, filters, strides=1, padding="SAME") + biases
print(f'Shape (manual filters): {fmaps.shape[1:3].as_list()}')
```

### ⚡ 補充練習 1

- **理論題**：比較 `padding="same"` 與 `padding="valid"` 的輸出形狀與
  資訊保留差異。為何深度 CNN 中常用 `same`？
- **實作題**：使用 7×7 卷積層，分別列印 `strides=1` 和 `strides=2`
  的輸出大小，說明改變步幅對計算量與感受野(receptive field)的影響。

--- 

## 2. 池化層 (Pooling Layer)：下採樣(Downsampling)與特徵聚合(Feature Aggregation)

### 2.1 為什麼重要？

池化層用於減少空間維度、擴大感受野並提取不變特徵，
降低計算成本同時保留最顯著的局部特徵。

### 2.2 如何運作？

- **最大池化（Max Pooling）**：在池化窗口中選取最大值，保留最強激活。
- **深度池化（Depth-wise Pooling）**：沿通道方向進行最大化，壓縮通道維度。
- **全域平均池化（Global Average Pooling）**：對每個通道的空間維度(高度和寬度)取平均，
  將特徵圖壓縮為單一標量值，常接於分類層前。換句話說，它把輸入特徵圖從`[height, width, channels]` 壓縮成 `[channels]`。
    - 在 CNN 中，全域平均池化常用於卷積層與**分類層**之間，因為它能夠：
        - 去除空間資訊後保留每個通道的整體響應強度
        - 減少參數數量，避免 Flatten() 後過度擴展
        - 提升模型對平移的魯棒性，適合作為全連接層前的最後一步

### 2.3 核心代碼

```python
print(f'Input shape: {images.shape[1:3].as_list()}')
# 最大池化
max_pool = tf.keras.layers.MaxPool2D(pool_size=2)
output = max_pool(images)
print(f'Shape (max pooling): {output.shape[1:3].as_list()}')

# 全域平均池化
print(f'Input shape: {images.shape.as_list()}')
global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
global_avg_pool(images)
print(f'Shape (global average pooling): {global_avg_pool(images).shape.as_list()}')

# 等價 Lambda 寫法
global_avg_pool_lam = tf.keras.layers.Lambda(
    lambda X: tf.reduce_mean(X, axis=[1, 2]))
global_avg_pool_lam(images)
print(f'Shape (global average pooling, Lambda): {global_avg_pool_lam(images).shape.as_list()}')
```

### 2.4 自訂深度池化層

```python
class DepthPool(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        shape = tf.shape(inputs)
        groups = shape[-1] // self.pool_size
        new_shape = tf.concat(
            [shape[:-1], [groups, self.pool_size]], axis=0)
        return tf.reduce_max(
            tf.reshape(inputs, new_shape), axis=-1)
```

### ⚡ 補充練習 2

- **理論題**：為何全域平均池化常用於 CNN 的最後一層？它相比
  `Flatten()` + `Dense()` 有哪些優缺點？
- **實作題**：使用 `DepthPool(pool_size=3)` 對隨機特徵圖進行運算，
  並與 `tf.nn.max_pool` 的結果比較，確認是否一致。

---

## 3. CNN 架構：LeNet-5 與 ResNet 殘差單元

### 3.1 為什麼重要？

CNN 架構決定模型的表示能力與訓練穩定性。本節從經典的 LeNet-5 出發，
進而介紹 Fashion MNIST 範例 CNN，再到 ResNet 殘差單元（Residual Unit），
反映了視覺模型從淺到深的演進。

### 3.2 LeNet-5 架構

LeNet-5 是最早成功應用於手寫數字辨識的 CNN，架構如下：

| 層   | 類型       | 特徵圖 | 大小       | 核尺寸  | 步幅 | 活化函數 |
|------|-----------|-------|-----------|--------|------|---------|
| In   | 輸入       | 1     | 32×32     | –      | –    | –       |
| C1   | 卷積       | 6     | 28×28     | 5×5    | 1    | tanh    |
| S2   | 平均池化   | 6     | 14×14     | 2×2    | 2    | tanh    |
| C3   | 卷積       | 16    | 10×10     | 5×5    | 1    | tanh    |
| S4   | 平均池化   | 16    | 5×5       | 2×2    | 2    | tanh    |
| C5   | 卷積       | 120   | 1×1       | 5×5    | 1    | tanh    |
| F6   | 全連接     | –     | 84        | –      | –    | tanh    |
| Out  | 全連接     | –     | 10        | –      | –    | RBF     |

### 3.3 Fashion MNIST CNN 範例

```python
from functools import partial

DefaultConv2D = partial(
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer="he_normal")

model = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7,
                  input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

model.summary()
```

### 3.4 ResNet 殘差單元與殘差連接

**殘差連接（skip connection）** 讓輸入直接加到主幹輸出，
緩解深度網路的退化（degradation）問題，使梯度更容易傳播。

```python
import tensorflow as tf
from functools import partial
DefaultConv2D = partial(
    tf.keras.layers.Conv2D,
    kernel_size=3, strides=1,
    padding="same",
    kernel_initializer="he_normal",
    use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", skip_connection=True, **kwargs):
        super().__init__(**kwargs)
        self.skip_connection = skip_connection # 新增開關參數
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        
        # 實作捷徑路徑
        if self.skip_connection:
            skip_Z = inputs
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)
        else:
            skip_Z = tf.zeros_like(Z) # 若關閉則設為全零張量

        return self.activation(Z + skip_Z)
```

### 3.5 ResNet-34 架構

```python
model = tf.keras.Sequential([
    DefaultConv2D(64, kernel_size=7, strides=2,
                  input_shape=[224, 224, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                              padding="same"),
])
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides, skip_connection=True))
    prev_filters = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

### ⚡ 補充練習 3

- **理論題**：為何殘差連接能讓更深的網路比純粹堆疊卷積層更容易訓練？
  請說明殘差單元的梯度流動機制。
- **實作題**：在 `ResidualUnit.call()` 中移除捷徑路徑（令
  `skip_Z = 0`），比較訓練 Fashion MNIST 前後的驗證準確度差異。

### ⚡ 深度解析：殘差連接的優勢

1. **參數重構（Re-parameterization）**：
   傳統網路旨在學習一個潛在映射 $H(x)$。而殘差單元將學習目標重構為 $y = F(x, \{W_i\}) + x$。若將最終映射定義為 $H(x) = F(x) + x$，則學習「殘差」$F(x) = H(x) - x$ 在優化上更具優勢。當底層特徵已足夠時，$F(x)$ 僅需逼近零，讓單元退化為**恒等映射（Identity Mapping）**，有效解決了深度網路中的「退化（Degradation）」問題。

2. **梯度傳播的資訊高速公路（Grandient Highway）**：
   根據連鎖律，損失函數 $L$ 對輸入 $x$ 的梯度可表示為：
   $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left( \frac{\partial F}{\partial x} + 1 \right)$$
   這項公式揭示了關鍵機制：梯度項中的「$+ 1$」確保了即使主幹路徑 $\frac{\partial F}{\partial x}$ 的梯度極小或消失，訊號仍能經由跳接路徑（Skip Connection）毫無阻礙地流向淺層。這種**梯度加法（Gradient Addition）** 性質打破了深度網路梯度連乘導致的指數級衰減，讓訓練數百層的網路成為可能。

```python
import tensorflow as tf

# load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train[:-5000], X_train[-5000:]
y_train, y_valid = y_train[:-5000], y_train[-5000:]
X_train = X_train[..., tf.newaxis] / 255.0
X_valid = X_valid[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).prefetch(tf.data.AUTOTUNE)
valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(64).prefetch(tf.data.AUTOTUNE)
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.AUTOTUNE)

# Preprocessing layers
# preprocess = tf.keras.layers.Resizing(height=224, width=224)

# build ResNet-34 with and without skip connections
def build_resnet(skip_connection=True):
    model = tf.keras.Sequential([
        # preprocess,
        # DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 1]),
        DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[28, 28, 1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
    ])
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides, skip_connection=skip_connection))
        prev_filters = filters

    model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

# 建立實驗模型
model_with_skip = build_resnet(skip_connection=True)
model_without_skip = build_resnet(skip_connection=False)

# 編譯與訓練模型
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model_with_skip.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
model_without_skip.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
history_with_skip = model_with_skip.fit(train_set, validation_data=valid_set, epochs=10)
history_without_skip = model_without_skip.fit(train_set, validation_data=valid_set, epochs=10)

import matplotlib.pyplot as plt
plt.plot(history_with_skip.history['val_accuracy'], label='With Skip Connection')
plt.plot(history_without_skip.history['val_accuracy'], label='Without Skip Connection')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
```

## 4. 預訓練模型 (Pre-trained Models) 與轉移學習 (Transfer Learning)

### 4.1 為什麼重要？

轉移學習（Transfer Learning）可將大型資料集（如 ImageNet）
上預訓練的特徵提取器，*應用到新的小資料集*，節省訓練時間並提升效能。
在資料量有限的視覺任務中，這是最常採用的實務策略。
在`tf.keras.applications` 模組中，提供了多種預訓練模型（如 VGG、ResNet、Xception 等），這些模型在 ImageNet 上訓練，能夠提取通用的視覺特徵，適用於各種下游任務。
> ImageNet 是一個包含超過 1400 萬張標註圖像的資料集，涵蓋了 1000 個類別。預訓練模型在 ImageNet 上學習到的特徵（如邊緣、紋理、形狀等）具有高度的泛化 (generalization) 能力，能夠在其他視覺任務中有效地提取有用資訊，即使這些任務的資料集規模較小。

### 4.2 轉移學習流程

1. **載入基底模型**：`include_top=False` 移除原始分類器。
2. **加入自訂輸出層**：`GlobalAveragePooling2D()` + `Dense(n_classes)`。
3. **凍結基底層**：`layer.trainable = False`，僅訓練頂部新層。
4. **微調（Fine-tuning）**：頂層收斂後，解凍部分基底層並以低學習率繼續訓練。

### 4.3 核心代碼

```python
n_classes = 10  # 有 10 個類別的 Fashion MNIST 資料集

# 1. 載入 Xception 基底模型，排除頂部原有的分類層 (GlobalAvgPool + Dense)
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=False)

# 2. 加入自訂分類頭部：全域平均池化將 3D 特徵圖壓縮為 1D 向量
avg = tf.keras.layers.GlobalAveragePooling2D()(
    base_model.output)
output = tf.keras.layers.Dense(
    n_classes, activation="softmax")(avg)

# 3. 建立最終模型：定義輸入與自訂輸出
model = tf.keras.Model(
    inputs=base_model.input, outputs=output)

# 4. 凍結基底層：防止預訓練的權重在初始訓練階段被破壞
for layer in base_model.layers:
    layer.trainable = False

# 5. 編譯模型：使用帶動量的 SGD，並設定適合分類的損失函數
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.1, momentum=0.9)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

# 6. 訓練新層：通常只需少數 epoch 即可使頂部分類器收斂
history = model.fit(train_set, validation_data=valid_set, epochs=3)
```

檢視基底模型與頂部分類器的結構，確認 `include_top=False` 的效果：

```python
xception_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=True)

# 列印 Xception 模型的前 10 層名稱與輸出形狀，了解基底模型結構
for n in range(10):
    print(f"Layer {n}: its name is {xception_model.layers[n].name} with output {xception_model.layers[n].output}")

# 列印 Xception 模型的後 10 層名稱與輸出形狀，確認頂部分類器結構
for n in range(-1, -11, -1):
    print(f"Layer {n}: its name is {xception_model.layers[n].name} with output {xception_model.layers[n].output}")

# 同樣的列印基底模型（不包含頂部分類器）的前後 10 層，確認 `include_top=False` 的效果
for n in range(10):
    print(f"Layer {n}: its name is {base_model.layers[n].name} with output {base_model.layers[n].output}")

# 列印基底模型的後 10 層，確認頂部分類器已被移除
for n in range(-1, -11, -1):
    print(f"Layer {n}: its name is {base_model.layers[n].name} with output {base_model.layers[n].output}")
```

>Xception 模型結構圖：

![Xception 模型結構圖](xception_model_structure.png)

>精準地列印每層的名稱與類型，確認基底模型與頂部分類器的結構差異：

```python
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.__class__.__name__)

for i, layer in enumerate(xception_model.layers):
    print(i, layer.name, layer.__class__.__name__)
```

### 4.4 資料前處理與資料增強

由於預訓練模型對輸入影像有特定的格式規範（如尺寸與數值範圍），我們將介紹如何使用 Keras Preprocessing Layers 來建立高效的影像處理流水線，包括：
- **尺寸調整 (Resizing)**：將影像統一縮放至模型預期的維度。
- **預處理函數 (Preprocessing Function)**：執行模型專屬的數值標準化（例如將像素縮放至 [−1,1]）。
- **資料增強 (Data Augmentation)**：透過隨機翻轉、旋轉等技巧人工擴張訓練集，以強化模型的泛化能力並防止過度擬合（overfitting）。

這對於處理像 Fashion MNIST 這種小尺寸、單通道的資料集尤其關鍵，因為我們必須先將其轉換為模型可識別的格式，才能有效發揮轉移學習的威力。

```python
batch_size = 32
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(
        height=224, width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(
        tf.keras.applications.xception.preprocess_input)
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.05, seed=42),
    tf.keras.layers.RandomContrast(factor=0.2, seed=42)
])
```

### 4.5 微調範例

解凍 (unfreeze) 模型的部分層，並以較低的學習率繼續訓練，以微調預訓練權重適應新任務。

```python
print(f"模型的總層數: {len(model.layers)}")

for layer in model.layers[56:]:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set,
                    epochs=10)
```

```python
# 列印模型的每一層名稱、類型與是否可訓練，確認微調設定
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({layer.__class__.__name__}) - Trainable: {layer.trainable}")
```

### 4.6 分類與定位（多輸出模型）

實際應用中，我們不僅需要知道物體的類別，還需要精確標定其在影像中的位置，這就是「影像定位」（Image Localization）。
透過多輸出模型，我們可以同時訓練網路進行兩個任務：一個負責預測類別（分類），另一個負責預測物體的 **邊界框 (Bounding Box)** 座標（定位）。
這不僅節省了重新計算特徵的時間，更為隨後更複雜的 **「物件偵測」（Object Detection）** 奠定了基礎。

```python
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=False)

avg = tf.keras.layers.GlobalAveragePooling2D()(
    base_model.output)

# 分類輸出：10 類別的 softmax 預測
class_output = tf.keras.layers.Dense(
    n_classes, activation="softmax")(avg)

# 定位輸出：4 個座標值的線性預測（x_min, y_min, x_max, y_max）
loc_output = tf.keras.layers.Dense(4)(avg)

# 建立多輸出模型：同時輸出分類與定位結果
model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[class_output, loc_output])

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9)
model.compile(
    loss=["sparse_categorical_crossentropy", "mse"], # 分別為分類與定位的損失函數
    loss_weights=[0.8, 0.2], # 給予分類損失較高的權重，因為它通常更重要且更難學習
    optimizer=optimizer,
    metrics=["accuracy", "mse"]) # 分別評估分類準確度與定位誤差
```

### ⚡ 補充練習 4

- **理論題**：解釋為何轉移學習階段通常先凍結基底模型再訓練新頂層。
  微調時為何改用更低的學習率？
- **實作題**：使用
  `tf.keras.applications.ResNet50(weights="imagenet", include_top=False)`
  作為基底模型，以 `tf_flowers` 資料集（224×224）建立分類器並訓練。

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# 載入 tf_flowers 資料集
(train_ds, valid_ds), ds_info = tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True)

# 定義資料前處理與增強流水線
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(
        tf.keras.applications.resnet.preprocess_input)
])
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.05, seed=42),
    tf.keras.layers.RandomContrast(factor=0.2, seed=42)
])

# 建立轉移學習模型
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(ds_info.features["label"].num_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# 凍結基底模型
for layer in base_model.layers:
    layer.trainable = False

# 編譯模型
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

# 訓練資料集：先 map (調整尺寸與增強)，再 batch
training_data = train_ds.map(
    lambda x, y: (
        tf.squeeze(data_augmentation(preprocess(tf.expand_dims(x, 0)), training=True), 0), 
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
).batch(32).prefetch(tf.data.AUTOTUNE)

# 驗證資料集：同樣先 map，再 batch
validation_data = valid_ds.map(
    lambda x, y: (
        tf.squeeze(preprocess(tf.expand_dims(x, 0)), 0), 
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
).batch(32).prefetch(tf.data.AUTOTUNE)

# 執行訓練
history = model.fit(
    training_data,
    validation_data=validation_data,
    epochs=5)
```

---

## 結論

本章提供了電腦視覺的核心工具：**卷積層、池化層、LeNet-5 與 ResNet
殘差架構，以及轉移學習技術**。你應該能理解 CNN 為何在影像任務中
比全連接網路更有效，並能將預訓練特徵提取器應用於新的分類任務。

接下來的章節可進一步延伸至**目標檢測、語義分割與生成對抗網路**，
也可探索更高階的視覺 Transformer（ViT）與弱監督學習方法。

---

## 課後作業

1. **實作題**：依據第 3.3 節的 Fashion MNIST CNN 範例，
   建立自己的卷積神經網路，並在 `X_train`／`X_valid`／`X_test`
   分割後完成訓練。請記錄：
   - 使用 `Conv2D` 與 `MaxPool2D` 的層結構
   - 每次訓練後的驗證準確度
   - 有無 `Dropout` 的比較結果

```python
import tensorflow as tf

# Define parameters
BATCH_SIZE = 64
DROPOUT_RATE = 0.2
EPOCHS = 30

# load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train[:-5000], X_train[-5000:]
y_train, y_valid = y_train[:-5000], y_train[-5000:]
X_train = X_train[..., tf.newaxis] / 255.0
X_valid = X_valid[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 建立 CNN 模型
model_with_spatial_dropout = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1/255.0, input_shape=[28, 28, 1]),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.SpatialDropout2D(DROPOUT_RATE),  # 在卷積層後使用 SpatialDropout2D
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.SpatialDropout2D(DROPOUT_RATE),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
])
model_without_dropout = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1/255.0, input_shape=[28, 28, 1]),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

# 編譯模型
model_with_spatial_dropout.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"])

model_without_dropout.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"])

# 訓練模型
print("訓練帶有 SpatialDropout2D 的模型...")
history_with_spatial_dropout = model_with_spatial_dropout.fit(train_set, validation_data=valid_set, epochs=EPOCHS)
print("\n訓練不帶 Dropout 的模型...")
history_without_dropout = model_without_dropout.fit(train_set, validation_data=valid_set, epochs=EPOCHS)

import matplotlib.pyplot as plt
plt.plot(history_with_spatial_dropout.history['accuracy'], 'r--', label='With SpatialDropout2D (training)')
plt.plot(history_without_dropout.history['accuracy'], 'b--', label='Without Dropout (training)')
plt.plot(history_with_spatial_dropout.history['val_accuracy'], 'r', label='With SpatialDropout2D (validation)')
plt.plot(history_without_dropout.history['val_accuracy'], 'b', label='Without Dropout (validation)')
plt.title('Training and Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

2. **轉移學習題**：使用
   `tf.keras.applications.Xception(include_top=False)`
   對 `tf_flowers` 資料集進行轉移學習。請完成以下步驟：
   - 影像尺寸調整為 224×224
   - 使用 `xception.preprocess_input` 處理輸入
   - 先凍結基底模型訓練新頂層，再解凍部分基底層進行微調
   - 比較微調前後的驗證準確度變化

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# 0. Define parameters
BATCH_SIZE = 32
UNFREEZE_FROM_LAYER = 100
EPOCHS_TOP = 5
EPOCHS_FINE_TUNE = 5
SPLIT_RATIO = 0.8

# 1. 載入 tf_flowers 資料集
(train_ds, valid_ds), ds_info = tfds.load(
    "tf_flowers",
    split=[f"train[:{int(SPLIT_RATIO*100)}%]", f"train[{int(SPLIT_RATIO*100)}%:]"],
    as_supervised=True,
    with_info=True)

n_classes = ds_info.features["label"].num_classes

# 2. 定義前處理流水線 (尺寸調整與 Xception 專屬預處理)
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])

def preprocess_fn(image, label):
    return preprocess(image), label

train_set = train_ds.map(preprocess_fn).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_set = valid_ds.map(preprocess_fn).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 3. 建立轉移學習模型
base_model = tf.keras.applications.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# 4. 第一階段：凍結基底模型權重，僅訓練新加入的頂層
for layer in base_model.layers:
    layer.trainable = False

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("--- 階段 1：訓練新頂層 ---")
history_top = model.fit(train_set, validation_data=valid_set, epochs=EPOCHS_TOP)

# 5. 第二階段：解凍部分基底層進行微調 (Fine-tuning)
# 解凍最後端的部分層 (例如從第 100 層開始)
for layer in base_model.layers[UNFREEZE_FROM_LAYER:]:
    layer.trainable = True

# 微調時必須調低學習率，避免破壞已訓練好的特徵
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("\n--- 階段 2：微調基底模型 ---")
history_fine_tune = model.fit(train_set, validation_data=valid_set, epochs=EPOCHS_FINE_TUNE)

# 6. 比較準確度
val_acc_before = history_top.history["val_accuracy"][-1]
val_acc_after = history_fine_tune.history["val_accuracy"][-1]
print(f"\n微調前驗證準確度: {val_acc_before:.4f}")
print(f"微調後驗證準確度: {val_acc_after:.4f}")

# 微調前驗證準確度: 0.8733
# 微調後驗證準確度: 0.9401
```