<!-- meta-title: 深度電腦視覺與 CNN 實作教學｜TensorFlow Keras -->
<!-- meta-description: 從卷積層、池化層、ResNet 殘差單元到轉移學習，解析第 14 章 CNN 核心實作。 -->
<!-- meta-keywords: CNN, 卷積神經網路, 深度學習, TensorFlow, Keras, 轉移學習, ResNet -->
<!-- meta-hashtags: #CNN #深度學習 #TensorFlow #Keras #Python教學 #機器學習 #電腦視覺 -->

# 🐍 深度電腦視覺與卷積神經網路 (CNN) 實作教學

這份教學基於《Hands-On Machine Learning》第 3 版第 14 章，帶你從卷積層
與池化層的基本運作開始，進而學習 CNN 架構、ResNet 殘差單元，
並應用預訓練模型進行轉移學習 (Transfer Learning) 和微調。

## 📝 本文目錄

- 🎯 [關鍵重點](#key-takeaways)
- 🧠 [卷積層 (Convolutional Layer) 與特徵圖](#conv-layer)
- 📉 [池化層 (Pooling Layer) 與全域平均池化](#pooling-layer)
- 👗 [Fashion MNIST CNN 範例](#fashion-mnist-cnn)
- 🔗 [ResNet 殘差單元與深度架構](#resnet)
- 🚀 [預訓練模型與轉移學習 (Transfer Learning)](#transfer-learning)
- 🧩 [進階微調與多輸出分類](#fine-tuning)
- ❓ [常見問答](#faq)
- 🏷️ [推薦標籤](#hashtags)

## <a id="key-takeaways"></a> 🎯 關鍵重點

- **卷積層 (Convolutional Layer)** 能從影像中提取局部特徵，並透過共享
  權重大幅降低參數量。
- `padding="same"` 或 `padding="valid"` 與步幅 (stride) 影響*特徵圖 (feature map)* 的空間大小。
- **池化層 (Pooling Layer)** 可進行下採樣 (downsampling)、擴大感受野 (receptive field)、並減少計算成本。
- **ResNet 的殘差連接 (skip connection)** 讓深層網路更容易訓練。
- **轉移學習 (Transfer Learning)** 可將預訓練特徵快速應用到新任務。

---

## <a id="conv-layer"></a> 🧠 卷積層 (Convolutional Layer) 與特徵圖

卷積層是 CNN 的核心。它透過濾波器 (filter) 在影像上滑動，對局部區域做
加權求和，產生特徵圖 (feature map)。這個過程可以理解為「局部感受野
(receptive field)」加上「權重共享 (weight sharing)」的組合。

### <a id="conv-1"></a> 1. 讀取範例影像與標準化

```python
from sklearn.datasets import load_sample_images
import tensorflow as tf

img_dataset = load_sample_images()
images = img_dataset["images"]
images = tf.keras.layers.CenterCrop(height=70, width=120)(images)
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)
```

✅ 程式碼逐行解析

1. `load_sample_images()`: 載入 Scikit-Learn 的內建範例影像資料集。
2. `import tensorflow as tf`: 匯入 TensorFlow，用於建構 CNN 層。
3. `["images"]`: 取得內建的兩張示例影像。
4. `CenterCrop(...)`: 將影像裁剪為 70×120，標準化後續卷積示範。
5. `Rescaling(scale=1 / 255)`: 將像素值縮放到 [0, 1]，提高訓練穩定性。

🎯 重點摘要

- 先行資料預處理可保持模型輸入一致性。
- 將像素歸一化有助於梯度穩定。
- 中心裁剪是視覺實作中常見的尺寸標準化方式。

```python
# inspect this dataset
print("Dataset keys:", img_dataset.keys())

for i, img in enumerate(img_dataset["images"]):
    print(f"Image {i} : {img_dataset['filenames'][i]}")
    print(f"Shape: {img.shape}, dtype: {img.dtype}")
    print(f"min: {img.min()}, max: {img.max()}, mean: {img.mean():.4f}")
    print("\n")
```

### <a id="conv-2"></a> 2. 卷積層基本實作

```python
tf.random.set_seed(42)
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
fmaps = conv_layer(images)
```

✅ 程式碼逐行解析

1. `tf.random.set_seed(42)`: 設定隨機種子，使範例可重現。
2. `Conv2D(filters=32, kernel_size=7)`: 建立 32 個 7×7 的卷積核 (kernel)。
3. `fmaps = conv_layer(images)`: 將卷積層套用到影像，得到輸出特徵圖。

🎯 重點摘要

- 32 個濾波器會產生 32 張特徵圖。
- 卷積核大小影響可偵測的局部結構。
- 預設 `padding="valid"` 會讓輸出尺寸縮小。

```python
# inspect the feature maps
print("Feature maps shape:", fmaps.shape)
print("Feature maps dtype:", fmaps.dtype)
print("Feature maps min:", fmaps.numpy().min())
print("Feature maps max:", fmaps.numpy().max())
print("Feature maps mean:", fmaps.numpy().mean())

# inspect the convolutional layer weights
print("Conv layer weights shape:", conv_layer.kernel.shape)
print("Conv layer weights dtype:", conv_layer.kernel.dtype)
print("Conv layer weights min:", conv_layer.kernel.numpy().min())
print("Conv layer weights max:", conv_layer.kernel.numpy().max())
print("Conv layer weights mean:", conv_layer.kernel.numpy().mean())

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(fmaps[0, :, :, i], cmap="viridis")
    plt.axis("off")
plt.suptitle("Feature Maps from Conv2D Layer", fontsize=16)
plt.show()
```

> 從`conv_layer.kernel.shape` 可以看出卷積核的形狀，通常為 `(kernel_height, kernel_width, input_channels, output_channels)`。
> `fmaps.shape` 顯示輸出特徵圖的形狀，通常為 `(batch_size, height, width, output_channels)`。

### <a id="conv-3"></a> 3. Padding 與 Stride 對輸出大小的影響

以下比較不同 `padding` 與 `strides` 設定對特徵圖大小的影響：

```python
conv_layer_same = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="same") # 保持輸入輸出尺寸相同（步幅為 1）
fmaps_same = conv_layer_same(images)

conv_layer_stride = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="same", strides=2) # 步幅為 2，輸出尺寸變為原來的一半
fmaps_stride = conv_layer_stride(images)

conv_layer_valid = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="valid") # 不填充，輸出尺寸會縮小
fmaps_valid = conv_layer_valid(images)

print("Input shape:", images.shape, "\n")
print("Output shape with padding='same':", fmaps_same.shape)
print("Output shape with strides=2:", fmaps_stride.shape, "\n")
print("Output shape with padding='valid':", fmaps_valid.shape)
```

```python
import numpy as np

def conv_output_size(input_size, kernel_size, strides=1, padding="valid"):
    """ 計算卷積層輸出尺寸與被捨去/補零的列/欄數 """
    if padding == "valid":
        z = input_size - kernel_size + strides
        output_size = z // strides
        num_ignored = z % strides
        return output_size, num_ignored
    else:
        output_size = (input_size - 1) // strides + 1
        num_padded = (output_size - 1) * strides + kernel_size - input_size
        return output_size, num_padded

conv_output_size(
    np.array([70, 120]), kernel_size=7, strides=2, padding="same")
```

✅ 程式碼逐行解析

1. `padding="same"`: 在輸入邊緣補零，輸出與輸入空間尺寸相同（步幅為 1）。
2. `strides=2`: 濾波器每次移動 2 個像素，輸出尺寸變為原來的一半。
3. `conv_output_size(...)`: 自訂函數計算卷積後輸出大小。
4. `num_ignored` / `num_padded`: 分別表示被捨去或補零的列/欄數。

🎯 重點摘要

- `same` padding 適合多層卷積網路，能保留空間大小。
- stride 越大，輸出越小，計算成本越低。
- 了解輸出尺寸公式有助於設計合理的 CNN 結構。

### <a id="conv-4"></a> 4. 手動濾波器示範：垂直與水平線檢測

```python
filters = np.zeros([7, 7, 3, 2])
filters[:, 3, :, 0] = 1
filters[3, :, :, 1] = 1
biases = tf.zeros([2])
fmaps = tf.nn.conv2d(
    images, filters, strides=1, padding="SAME") + biases
```

✅ 程式碼逐行解析

1. `np.zeros([7, 7, 3, 2])`: 建立兩個 7×7 濾波器，輸入通道為 3。
2. `filters[:, 3, :, 0] = 1`: 第一個濾波器偵測垂直線。
3. `filters[3, :, :, 1] = 1`: 第二個濾波器偵測水平線。
4. `tf.nn.conv2d(...)`: 使用低階卷積運算直接計算輸出。
5. `+ biases`: 加上零偏差，保持輸出純粹反映濾波器響應。

🎯 重點摘要

- CNN 內部濾波器可以自動學習邊緣與線條特徵。
- 手動設計濾波器有助於理解 CNN 的特徵提取機制。
- `padding="SAME"` 顯示填充對邊緣響應的影響。

---

## <a id="pooling-layer"></a> 📉 池化層 (Pooling Layer) 與全域平均池化

池化層負責下採樣 (downsampling) 與特徵聚合 (feature aggregation)，是 CNN 中常見的空間壓縮方法。

### <a id="max-pool"></a> 1. 最大池化 (Max Pooling)

`MaxPool2D` 是最常見的池化方法，它在每個池化窗口中取最大值，保留最強激活。
- 適合保留邊緣與紋理特徵。
- 減少參數與運算量，提升平移不變性 (translation invariance)。
- 常見的池化窗口大小為 2×2，步幅為 2，會將空間尺寸減半。
    - `pool_size=2` 表示每個 2×2 區塊取最大值。
    - 例如，從 28×28 變為 14×14。

```python
max_pool = tf.keras.layers.MaxPool2D(pool_size=2)
output = max_pool(images)
print("Max pooling output shape:", output.shape)

# Visualize the max pooling output
plt.imshow(output[0].numpy())
plt.title("Max Pooling Output")
plt.axis("off")
plt.show()
```

✅ 程式碼逐行解析

1. `MaxPool2D(pool_size=2)`: 設定 2×2 的池化窗口。
2. `output = max_pool(images)`: 對每個 2×2 區塊取最大值，完成下採樣。

🎯 重點摘要

- 最大池化保留最強激活值，適合保留邊緣與紋理特徵。
- 池化可減少參數與運算量，並提升平移不變性 (translation invariance)。

### <a id="depth-pool"></a> 2. 深度池化 (Depth-wise Pooling)

透過自訂層實現*沿通道維度的池化*，將每個通道分成多個子群組，對每個群組取最大值。

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
        return tf.reduce_max(tf.reshape(inputs, new_shape), axis=-1)
```

✅ 程式碼逐行解析

1. `class DepthPool(...)`: 自訂 Keras Layer，沿通道維度池化。
2. `shape = tf.shape(inputs)`: 取得輸入張量的動態形狀。
3. `groups = shape[-1] // self.pool_size`: 池化後的通道分組數。
4. `tf.reshape(...)`: 重塑張量，使通道可分成多個子群組。
5. `tf.reduce_max(..., axis=-1)`: 在每個通道群組上取最大值。

🎯 重點摘要

- 深度池化提供另一種壓縮通道信息的方法。
- 自訂層展示 TensorFlow 函式庫的彈性。
- 適合在深度特徵抽取後**進行通道壓縮**。

### <a id="gap"></a> 3. 全域平均池化 (Global Average Pooling)

`GlobalAvgPool2D` 將每個通道的空間維度 (高度與寬度) 平均為單一值，適合在分類器之前使用，能顯著降低參數量並減少過擬合風險。

```python
global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
global_avg_pool(images)
```

✅ 程式碼逐行解析

1. `GlobalAvgPool2D()`: 建立全域平均池化層。
2. `global_avg_pool(images)`: 將每個通道的高度與寬度平均為單一值。

🎯 重點摘要

- 全域平均池化可將*每個特徵圖壓縮為一個標量*，適用於分類器之前。
- 它能顯著降低參數數量並減少過擬合風險。
- 與 `Flatten()` 相比，更適合多尺度輸入。

---

## <a id="fashion-mnist-cnn"></a> 👗 Fashion MNIST CNN 範例

這個範例展示如何使用 CNN 進行 Fashion MNIST 影像分類，包含常見的層堆疊、
Dropout 與 Dense 分類器。

```python
from functools import partial

# 定義卷積層的預設參數
default_conv = partial(tf.keras.layers.Conv2D,
                       kernel_size=3,
                       padding="same",
                       activation="relu",
                       kernel_initializer="he_normal")

model = tf.keras.Sequential([
    default_conv(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPool2D(),
    default_conv(filters=128),
    default_conv(filters=128),
    tf.keras.layers.MaxPool2D(),
    default_conv(filters=256),
    default_conv(filters=256),
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
```

✅ 程式碼逐行解析

1. `partial(...)`: 使用 `functools.partial` 簡化 `Conv2D` 建構。
    - 這樣可以在後續呼叫時只需指定 `filters` 和 `input_shape`，其他參數保持一致。
2. `default_conv(filters=64, kernel_size=7, input_shape=[28, 28, 1])`:
   第一層卷積，輸入為 28×28 灰階影像；kernel 大小為 7×7，濾波器數量為 64。
3. `MaxPool2D()`: 每次下採樣 2×2。
4. 重複 128 和 256 通道的卷積堆疊：增加特徵抽取能力。
5. `Flatten()`: 將特徵圖攤平成向量，送入密集層 (Dense layer)。
6. `Dense(128, ...)` / `Dense(64, ...)`: 隱藏層。
7. `Dropout(0.5)`: 隨機失活 50%，避免過擬合 (overfitting)。
8. `Dense(10, activation="softmax")`: 輸出 10 類的機率分佈。

🎯 重點摘要

- 這是典型的影像分類 CNN 架構：卷積 + 池化 + 全連接層。
- `Dropout` 有助於穩定訓練並提升泛化能力。
- `he_normal` 初始化適合 ReLU 活化函數 (activation function)。
- `SpatialDropout` 也可考慮用於卷積層，隨機失活整個特徵圖 (feature map)，進一步提升模型魯棒性。

---

## <a id="resnet"></a> 🔗 ResNet 殘差單元 (Residual Unit) 與深度架構

ResNet 的殘差單元是解決深度網路**退化 (degradation)** 問題的關鍵。它透過
**捷徑連接 (skip connection)** 讓輸入直接疊加到後續層，改善梯度傳播。

```python
DefaultConv2D = partial(tf.keras.layers.Conv2D,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        kernel_initializer="he_normal",
                        use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
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
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
```

✅ 程式碼逐行解析

1. `use_bias=False`: 使用批次歸一化 (Batch Normalization) 時可省略偏差項。
2. `self.main_layers`: 主路徑包含兩次卷積與批次歸一化。
3. `self.skip_layers`: 當 `strides > 1` 時，捷徑路徑也做下採樣。
4. `Z = layer(Z)`: 主路徑逐層計算卷積輸出。
5. `skip_Z = layer(skip_Z)`: 捷徑路徑保留輸入資訊。
6. `return self.activation(Z + skip_Z)`: 主路徑與捷徑相加後再套用激活函數。

🎯 重點摘要

- 殘差連接讓深層網路更容易收斂，也能保留低層特徵。
- `strides > 1` 時，捷徑路徑必須做空間縮放以匹配主路徑輸出。
- 這種設計是 ResNet 成功的核心原因。

### <a id="resnet-34"></a> ResNet-34 結構示範

```python
model = tf.keras.Sequential([
    DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
])
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

🎯 重點摘要

- ResNet-34 使用多個相同濾波器數量的殘差區塊。
- `GlobalAvgPool2D()` 將每個通道壓縮為單一統計量，適合分類器輸入。
- 這樣的架構適合深層影像分類問題。

---

## <a id="transfer-learning"></a> 🚀 預訓練模型與轉移學習 (Transfer Learning)

轉移學習可讓你重用大型資料集（如 ImageNet）上訓練好的特徵，
快速部署到新任務。

```python
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False
```

✅ 程式碼逐行解析

1. `xception.Xception(...)`: 載入 Xception 模型，不包含頂層分類器。
    - `weights="imagenet"` 表示使用在 ImageNet 上預訓練的權重。
    - `include_top=False`: 保留特徵提取器，丟棄原始 ImageNet 分類層。
2. `GlobalAveragePooling2D()`: 將空間特徵聚合成向量。
3. `Dense(n_classes, activation="softmax")`: 新的分類器輸出層。
4. `tf.keras.Model(...)`: 建立完整模型。
5. `layer.trainable = False`: 凍結基底模型，僅訓練新加入的分類層。

🎯 重點摘要

- 凍結基底模型可避免一次訓練過多參數，適合資料量較少的任務。
- 新頂層會學習對應新任務的類別。
- 這是實務上常見的快速部署策略。

### <a id="data-preprocessing"></a> 資料前處理與增強 (Data Augmentation)

```python
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224,
                             crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(
        tf.keras.applications.xception.preprocess_input)
])

train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
```

🎯 重點摘要

- 預訓練模型通常要求固定輸入尺寸，例如 224×224。
- `preprocess_input` 會將像素值轉換到模型預期的區間，例如 [-1, 1]。
- `shuffle()`、`batch()` 和 `prefetch()` 可提升訓練效率。

---

## <a id="fine-tuning"></a> 🧩 進階微調 (Fine-Tuning) 與多輸出分類

當新加入的分類層收斂後，可對基底模型最後幾層進行微調 (fine-tuning)，
讓整體模型更適應新資料集。

```python
for layer in base_model.layers[56:]:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=10)
```

🎯 重點摘要

- **微調時應使用較低學習率**，以避免破壞預訓練權重。
- **只解凍部分基底層** 可以平衡訓練穩定性與適應性。
- 這樣的流程通常先「凍結訓練」、「再解凍微調」。

### <a id="multi-output"></a> 範例：分類與定位的多輸出模型

```python
class_output = tf.keras.layers.Dense(
    n_classes, activation="softmax")(avg)
loc_output = tf.keras.layers.Dense(4)(avg)
model = tf.keras.Model(inputs=base_model.input,
                       outputs=[class_output, loc_output])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
              loss_weights=[0.8, 0.2],
              optimizer=optimizer, metrics=["accuracy", "mse"])
```

🎯 重點摘要

- 多輸出模型同時預測類別與邊界框 (bounding box)，適用於定位任務。
- `loss_weights` 可調整不同目標的影響力。
- 這類架構是目標檢測 (object detection) 任務的基礎概念。

---

## <a id="faq"></a> ❓ 常見問答

### 問：為什麼要用 `padding="same"` 而不是 `padding="valid"`？

`padding="same"` 可以保持輸入與輸出空間尺寸一致，讓卷積網路更容易疊加，
特別適合深層 CNN。`padding="valid"` 則會在每層逐漸縮小空間尺寸，
適合需要快速下採樣的架構。

### 問：為何轉移學習時要先凍結基底模型？

因為預訓練模型已經學到通用特徵，先凍結基底模型可避免新任務資料量不足時
破壞這些特徵。這樣可以只訓練新頂層，降低過擬合風險。

### 問：ResNet 的殘差連接解決了什麼問題？

它解決了深層網路的**梯度退化 (gradient degradation)** 與訓練困難問題，
讓深層網路可以像淺層網路一樣傳遞訊號。**殘差連接 (Residual Connection)** 提供了捷徑路徑，
讓梯度更容易回傳到前層。

---

## <a id="hashtags"></a> 🏷️ 推薦標籤

標籤：#CNN #深度學習 #TensorFlow #Keras #Python教學 #機器學習 #電腦視覺

---

## 🧠 總結與深化學習路徑

### 💡 第一部分：概念理解（What & Why）

這部分是理解「為什麼要這樣做」的基礎。

1.  **CNN 的核心思想 (Feature Hierarchy):**
    *   **理解點：** CNN 的層次結構是模仿人腦視覺皮層的。
        - 淺層（Early Layers）學到的是簡單的特徵（邊緣、角點）；
        - 深層（Deep Layers）則將這些簡單特徵組合起來，學到複雜的特徵（眼睛、車輪、整個物體）。
    *   **關鍵區分：** 
        - **卷積層（Convolution）** 負責提取特徵；
        - **池化層（Pooling）** 負責降維和增加魯棒性（讓模型對輸入的微小位移不敏感）。

2.  **遷移學習 (Transfer Learning) 的必要性：**
    *   **理解點：** 訓練一個大型模型（如 ImageNet 上的 ResNet）需要海量的標籤化數據和巨大的算力。當你的任務數據量小、算力有限時，直接從零開始訓練是不可行的。
    *   **解決方案：** 採用預訓練模型（Pre-trained Model），利用它在大型數據集上學到的通用特徵提取能力，然後只在你的小數據集上進行微調（Fine-tuning）。

3.  **殘差網路 (ResNet) 的突破 (The Vanishing Gradient Problem):**
    *   **理解點：** 隨著網路層數的加深，梯度（Gradient）在反向傳播時可能會變得極小（梯度消失），導致深層網路無法有效學習。
    *   **ResNet 的天才之處：** 它引入了**殘差塊 (Residual Block)**，核心思想是讓網路學習「殘差 $F(x)$」，而不是直接學習完整的映射 $H(x)$。透過 $H(x) = F(x) + x$ 的結構，它讓梯度可以直接「跳過」一些層（Skip Connection），從而有效解決了梯度消失問題，使得模型可以穩定地堆疊到數百層。

---

### 🚀 第二部分：核心概念深化（How）

這部分是深入理解模型結構和訓練策略的關鍵。

#### 1. 關於微調策略 (Fine-Tuning Strategy)
當使用預訓練模型時，切記這三個層級的微調策略：

*   **Feature Extractor (特徵提取器)：** 凍結所有預訓練層的權重，只訓練最後的分類器（Classification Head）。**（適用於：目標任務與源任務差異極大，或數據量極小）**
*   **Fine-Tuning (微調)：** 解凍所有層，但使用非常小的學習率（Learning Rate）。這讓模型在「微調」已學到的知識，而不是「忘記」它。**（適用於：目標任務與源任務相似，數據量中等）**
*   **Full Retraining (完整重訓練)：** 重新訓練所有層。**（適用於：數據量極大，且目標任務與源任務差異極大）**

#### 2. 關於優化器與學習率 (Optimizer & Learning Rate)
*   **優化器：** 雖然 Adam/AdamW 是主流，但理解 **SGD + Momentum** 的物理直覺（慣性）仍然重要。
*   **學習率衰減 (Learning Rate Scheduling)：** 這是提升性能的關鍵。不要用一個固定的學習率。建議使用 **Cosine Annealing** 或 **ReduceLROnPlateau**，讓學習率隨著訓練的進行而逐漸減小，讓模型在後期能精確收斂到最佳點。

---

### 🛠️ 第三部分：實戰建議與進階實作（Best Practices）

這部分是將知識轉化為高效率代碼的技巧。

1.  **數據增強 (Data Augmentation) 的組合拳：**
    *   不要只用隨機翻轉。組合使用：`RandomCrop` (隨機裁剪) + `RandomHorizontalFlip` (水平翻轉) + `ColorJitter` (改變亮度/對比度)。
    *   **進階技巧：** 考慮使用 **Mixup** 或 **CutMix**，這類方法會將多張圖片混合，能極大地提高模型的泛化能力，尤其在數據量較小時。

2.  **模型評估指標的選擇：**
    *   **準確率 (Accuracy)：** 適用於類別分佈均衡的分類任務。
    *   **F1-Score / AUC：** 當類別分佈不均衡（Imbalanced Data）時，**必須**使用 F1-Score 或 ROC-AUC 來評估模型，否則高準確率可能只是因為模型偏向多數類別。

3.  **實作流程總結（Checklist）：**
    1.  **數據預處理：** 統一尺寸 $\rightarrow$ 數據增強 $\rightarrow$ 歸一化 (Normalization)。
    2.  **模型載入：** 載入預訓練權重 $\rightarrow$ 替換分類頭。
    3.  **訓練階段 1 (Warm-up)：** 凍結層 $\rightarrow$ 訓練分類頭 $\rightarrow$ 較高 LR。
    4.  **訓練階段 2 (Fine-tuning)：** 解凍所有層 $\rightarrow$ 訓練所有層 $\rightarrow$ **極低 LR**。
    5.  **優化：** 實施學習率衰減 $\rightarrow$ 監控驗證集性能 $\rightarrow$ 最佳點提前停止 (Early Stopping)。

---

### 總結對照表

| 概念 | 目的 | 關鍵技術/結構 | 實戰應用點 |
| :--- | :--- | :--- | :--- |
| **CNN** | 提取層級特徵 | 卷積層 (Conv) + 池化層 (Pool) | 基礎結構搭建 |
| **ResNet** | 解決梯度消失 | 殘差連接 (Skip Connection) | 堆疊更深層網路 |
| **遷移學習** | 解決數據量不足 | 預訓練權重 (Pre-trained Weights) | 快速入門，提升性能上限 |
| **微調** | 調整通用知識到特定任務 | 凍結/解凍層 + 學習率調整 | 決定訓練的「深度」和「廣度」 |
| **數據增強** | 提高模型泛化能力 | 組合式增強 (Mixup, CutMix) | 數據量小時的「虛擬數據」製造機 |

希望這份結構化的總結能幫助您將知識點串聯起來，從「知道」到「精通」！如果您對某個特定環節（例如：如何手動實現一個殘差塊的代碼）有進一步的疑問，隨時可以提出！