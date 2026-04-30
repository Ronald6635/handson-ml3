# Ch14 速查表：Deep Computer Vision with CNNs

> **核心主旨**：CNN 的三大核心：卷積層、殘差連接、轉移學習 —— `Xception(include_top=False)` + 微調是視覺任務的最快起點。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Conv2D | 用 kernel 掃描影像，提取局部特徵（邊緣、紋理） | 所有影像任務的基礎層 |
| Padding `"same"` | 補零使輸出尺寸 = $\lceil H/s \rceil$ | 保持特徵圖大小 |
| Padding `"valid"` | 不補零，輸出 = $\lfloor (H-k)/s \rfloor + 1$ | 精確控制輸出大小 |
| MaxPool2D | 取窗口最大值，下採樣 × 2 | 降低計算量、擴大感受野 |
| GlobalAvgPool2D | 每個通道取平均，輸出 `[channels]` | 分類頭前的特徵壓縮 |
| ResidualUnit | Skip Connection：$y = F(x) + x$，緩解梯度消失 | 深層網路（20層+） |
| Transfer Learning | 複用 ImageNet 預訓練特徵，凍結底層只訓練頂層 | 小資料集視覺任務 |
| Fine-tuning | 解凍部分底層，以低 lr 繼續訓練 | 轉移學習第二階段 |

---

## 2. 關鍵 API 速查

| Keras API | 重點參數 | 用途 |
|-----------|---------|------|
| `tf.keras.layers.Conv2D` | `filters=64`, `kernel_size=3`, `padding="same"`, `activation="relu"` | 卷積層 |
| `tf.keras.layers.MaxPool2D` | `pool_size=2`, `strides=2` | 最大池化（預設減半） |
| `tf.keras.layers.GlobalAvgPool2D` | – | 全域平均池化 |
| `tf.keras.layers.SpatialDropout2D` | `rate=0.2` | 2D Dropout（整通道遮蔽） |
| `tf.keras.applications.Xception` | `weights="imagenet"`, `include_top=False` | 預訓練基底模型 |
| `tf.keras.applications.ResNet50` | 同上 | ResNet 基底模型 |
| `tf.keras.applications.xception.preprocess_input` | – | Xception 專屬前處理（縮放到 [-1,1]） |
| `tf.keras.applications.resnet.preprocess_input` | – | ResNet 專屬前處理 |
| `tf.keras.layers.RandomFlip` | `mode="horizontal"` | 資料增強：隨機翻轉 |
| `tf.keras.layers.RandomRotation` | `factor=0.05` | 資料增強：隨機旋轉 |

---

## 3. 必備代碼片段

```python
import tensorflow as tf
from functools import partial

# 基礎 CNN（Fashion MNIST）
DefaultConv2D = partial(tf.keras.layers.Conv2D,
    kernel_size=3, padding="same", activation="relu",
    kernel_initializer="he_normal")

model = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 殘差單元（ResNet 核心元件）
class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        Conv = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=strides,
                       padding="same", kernel_initializer="he_normal", use_bias=False)
        self.main_layers = [
            Conv(filters), tf.keras.layers.BatchNormalization(), self.activation,
            Conv(filters), tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:  # 尺寸不同時，skip 也需要卷積對齊
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides=strides, padding="same",
                                        kernel_initializer="he_normal", use_bias=False),
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

# 轉移學習（標準 4 步驟）
import tensorflow_datasets as tfds

# 步驟 1：載入並前處理資料
(train_ds, valid_ds), ds_info = tfds.load("tf_flowers",
    split=["train[:80%]", "train[80%:]"], as_supervised=True, with_info=True)
n_classes = ds_info.features["label"].num_classes

preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(224, 224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=42),
    tf.keras.layers.RandomRotation(0.05, seed=42)
])

train_set = train_ds.map(lambda x, y: (preprocess(x), y)).batch(32).prefetch(tf.data.AUTOTUNE)
valid_set = valid_ds.map(lambda x, y: (preprocess(x), y)).batch(32).prefetch(tf.data.AUTOTUNE)

# 步驟 2：建立模型
base_model = tf.keras.applications.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# 步驟 3：凍結底層，訓練新頂層
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_top = model.fit(train_set, validation_data=valid_set, epochs=5)

# 步驟 4：Fine-tuning（解凍部分層，降低 lr）
for layer in base_model.layers[100:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.SGD(1e-4, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_ft = model.fit(train_set, validation_data=valid_set, epochs=5)
```

---

## 4. 常見陷阱

- **`include_top=False` 移除了什麼**：移除了 `GlobalAvgPool + Dense(1000, softmax)`，保留所有卷積特徵提取層。
- **Fine-tuning 前必須重新 compile**：解凍層後 optimizer 狀態需要重置，且學習率必須降低（通常降低 10 倍）。
- **輸入尺寸要符合預訓練模型**：Xception / ResNet50 需要 224×224，使用 `Resizing` 層統一。
- **`preprocess_input` 不能省**：每個預訓練模型有自己的輸入值域，Xception 是 [-1,1]，ResNet 是 channel-wise 減均值，不正確的前處理會讓模型表現很差。
- **資料增強只在訓練時**：`model.fit()` 自動處理（`training=True`），`model.predict()` 不增強。

---

## 5. 決策指南

```
輸出形狀公式：
├── valid padding: floor((H - k) / s) + 1
└── same padding:  ceil(H / s)

何時用預訓練模型？
└── 幾乎永遠（除非訓練資料 >> 100k 且與 ImageNet 差異極大）

底層凍結 vs 微調：
├── 資料 < 2k 且與 ImageNet 相似     → 只訓練頂層
├── 資料 2k-20k 且與 ImageNet 相似   → 凍結底部 80%，微調頂部 20%
├── 資料 > 20k 或與 ImageNet 差異大  → 微調更多層甚至全部
└── 微調學習率 = 初始訓練學習率 / 10

預訓練模型選擇（由快到慢）：
MobileNetV2 → EfficientNetB0 → ResNet50 → Xception → EfficientNetB7
```
