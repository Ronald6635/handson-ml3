# Ch11 速查表：Training Deep Neural Networks

> **核心主旨**：深層網路訓練的六大工具箱 —— 正確的 initialization + BN + optimizer 組合是穩定訓練的基礎。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| He Initialization | 針對 ReLU 設計，保持各層訊號變異數 | 搭配 ReLU / Leaky ReLU / SELU |
| Glorot (Xavier) Init | 針對 sigmoid/tanh 設計 | 搭配 sigmoid / tanh |
| Batch Normalization | 每批次正規化每層輸出，解決 internal covariate shift | 幾乎所有深層網路 |
| Dropout | 訓練時隨機丟棄神經元，防止 overfitting | 全連接層的正則化 |
| Adam | 自適應學習率，結合 RMSProp + momentum | 通用最佳起點 |
| SELU | 自我正規化活化函數，需搭配 LeCun 初始化 | 全連接網路的 ReLU 替代 |
| Gradient Clipping | 限制梯度的最大範數，防止梯度爆炸 | RNN / 深層網路 |
| Learning Rate Schedule | 訓練過程中動態調整 lr | 精調收斂性 |
| Transfer Learning | 複用預訓練模型的底層，只訓練頂層 | 資料量少的任務 |

---

## 2. 關鍵 API 速查

| Keras Class / Function | 重點參數 | 用途 |
|------------------------|---------|------|
| `kernel_initializer="he_normal"` | – | He 初始化（用於 ReLU） |
| `kernel_initializer="glorot_uniform"` | – | Glorot 初始化（預設） |
| `tf.keras.layers.BatchNormalization` | `momentum=0.99`, `epsilon=0.001` | 批次歸一化 |
| `tf.keras.layers.Dropout` | `rate=0.2` | 標準 Dropout |
| `tf.keras.layers.AlphaDropout` | `rate=0.1` | SELU 專用 Dropout |
| `tf.keras.optimizers.Adam` | `learning_rate=1e-3`, `clipnorm=1.0` | Adam 優化器 |
| `tf.keras.optimizers.SGD` | `learning_rate=0.01`, `momentum=0.9`, `nesterov=True` | SGD + Nesterov momentum |
| `tf.keras.callbacks.ReduceLROnPlateau` | `factor=0.5`, `patience=5` | 自動降低學習率 |
| `tf.keras.callbacks.LearningRateScheduler` | `schedule=lambda epoch, lr: lr * 0.95` | 自訂 LR 排程 |

---

## 3. 必備代碼片段

```python
import tensorflow as tf

# 推薦的深層網路配方（BN + He init + ReLU）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    *[
        layer
        for _ in range(20)
        for layer in [
            tf.keras.layers.Dense(100, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ]
    ],
    tf.keras.layers.Dense(10, activation="softmax")
])

# 如果 BN 加在激活之後（更常見的寫法）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Dropout（用於全連接層）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Adam + Gradient Clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Learning Rate Schedule（指數衰減）
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 1cycle schedule 的近似寫法
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# Transfer Learning：凍結底層，只訓練頂層
base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(10, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Fine-tuning：解凍部分層
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

---

## 4. 常見陷阱

- **BN 的位置**：BN 可放在激活函數前（原論文）或後（實務更常見），兩種都有人用，專案內保持一致即可。
- **Dropout 在推論時關閉**：`model.predict()` 自動關閉 Dropout；在自訂迴圈中要傳 `training=False`。
- **轉移學習重新 compile**：解凍層後 **必須重新 compile**，否則 optimizer 的動量狀態可能造成問題。
- **BN + Dropout 同時用**：效果不一定加乘，有時互相干擾，建議分別實驗。
- **He init 必須搭配 ReLU 類**：若用 sigmoid/tanh，改用 `glorot_uniform`（預設值）。

---

## 5. 決策指南

```
活化函數選擇（由強到弱）：
├── 一般深層網路  → SELU（若全連接且標準化輸入）
├── CNN / 通用    → ELU → Leaky ReLU → ReLU
└── 輸出層        → softmax（多類）/ sigmoid（二元）/ linear（迴歸）

初始化配對規則：
├── ReLU / Leaky ReLU / ELU / SELU → He normal
└── Sigmoid / Tanh                 → Glorot uniform（預設）

正則化策略：
├── 一般過擬合     → Dropout (0.1~0.5)
├── 深層 CNN       → Dropout + BatchNormalization
├── 非常小的資料集 → L2 regularization + Dropout
└── 大資料集       → BatchNormalization 通常足夠
```
