<!-- meta-title: 訓練深度神經網路完整指南：梯度消失、BatchNorm、遷移學習與優化器 -->
<!-- meta-description: 深入深度神經網路訓練技術：梯度消失/爆炸問題、Xavier/He 初始化、Batch Normalization、梯度裁剪、遷移學習、Adam/Nadam 優化器、學習率排程、Dropout 與 Max-Norm 正則化。 -->
<!-- meta-keywords: Python, 深度學習, Keras, TensorFlow, Batch Normalization, 遷移學習, Adam, Dropout, 梯度消失, 神經網路 -->
<!-- meta-hashtags: #Python #深度學習 #Keras #TensorFlow #BatchNorm #遷移學習 #Adam #Dropout #程式設計 #教學 -->

# 🐍 訓練深度神經網路：從梯度消失到 Adam 優化器

為什麼深度神經網路難以訓練？**梯度消失（Vanishing Gradient）** 讓早期層幾乎停止學習；**梯度爆炸（Exploding Gradient）** 讓參數更新失控。本教學帶你掌握現代深度學習中所有關鍵的訓練技術，讓你的神經網路真正學得快、學得好。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [💥 梯度消失/爆炸問題](#vanishing-exploding)
- [🌱 Xavier/He 權重初始化](#weight-init)
- [📊 Batch Normalization](#batch-norm)
- [✂️ 梯度裁剪](#gradient-clipping)
- [🔄 遷移學習](#transfer-learning)
- [⚡ 快速優化器](#optimizers)
- [📅 學習率排程](#lr-schedule)
- [🛡️ 正則化技術](#regularization)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **He 初始化** + **ReLU/ELU 激活函數** 是現代深度網路的標準起點
- **Batch Normalization（批次標準化）** 大幅加速訓練，允許更大學習率
- **Adam/Nadam** 是最常用的優化器，通常無需大幅調整超參數
- **遷移學習（Transfer Learning）** 在小資料集上效果顯著，是實務首選策略
- **Dropout** 是最有效的正則化之一；**MC Dropout** 可獲得不確定性估計

---

## <a id="vanishing-exploding"></a>💥 梯度消失/爆炸問題

💡 **實際應用情境：** 在 2012 年以前，深度網路幾乎無法訓練。當 Sigmoid 的梯度只有 ~0.25，通過 10 層後梯度縮小到 0.25^10 ≈ 0.000001——早期層幾乎不學習。理解這個問題是掌握所有後續技術的基礎。

梯度消失的根本原因：Sigmoid 函數的飽和區梯度趨近於 0，反向傳播時梯度逐層相乘後指數衰減。

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 示範：Sigmoid 的梯度問題
x = tf.Variable(np.linspace(-5, 5, 1000), dtype=tf.float32)

with tf.GradientTape() as tape:
    y_sigmoid = tf.sigmoid(x)
    y_relu = tf.nn.relu(x)

grad_sigmoid = tape.gradient(y_sigmoid, x)  # 最大梯度約 0.25
print(f"Sigmoid 最大梯度: {grad_sigmoid.numpy().max():.4f}")  # ≈ 0.25
```

解決方案演進：

| 技術 | 解決問題 | 出現年份 |
|------|---------|---------|
| ReLU 激活 | 梯度消失 | 2010 |
| Xavier/He 初始化 | 梯度消失/爆炸 | 2010/2015 |
| Batch Normalization | 梯度問題 + 加速訓練 | 2015 |
| 梯度裁剪 | 梯度爆炸（RNN） | 常用 |

---

## <a id="weight-init"></a>🌱 Xavier/He 權重初始化

### 範例 1: 各種初始化方法

```python
from tensorflow import keras

# Xavier 初始化（適合 Sigmoid/tanh）
# 方差 = 2 / (fan_in + fan_out)
layer_xavier = keras.layers.Dense(
    300,
    activation="sigmoid",
    kernel_initializer="glorot_uniform"  # Xavier 均勻分佈（預設）
)

# He 初始化（適合 ReLU 及其變體）
# 方差 = 2 / fan_in
layer_he = keras.layers.Dense(
    300,
    activation="relu",
    kernel_initializer="he_normal"  # He 常態分佈
)

# LeCun 初始化（適合 SELU）
layer_lecun = keras.layers.Dense(
    300,
    activation="selu",
    kernel_initializer="lecun_normal"
)

# 完整網路：He 初始化 + ReLU（現代標準）
model_he = keras.Sequential([
    keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dense(10,  activation="softmax")
])
model_he.summary()
```

**✅ 程式碼逐行解析：**

1. `kernel_initializer="glorot_uniform"`: 從均勻分佈 $U[-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})}]$ 初始化
2. `kernel_initializer="he_normal"`: 從常態分佈 $N(0, \sqrt{2/n_{in}})$ 初始化，為 ReLU 設計
3. 激活函數和初始化的配對規則：Sigmoid/tanh → Xavier；ReLU/Leaky ReLU/ELU → He；SELU → LeCun

**🎯 重點摘要:**

- 不匹配的初始化可能導致梯度在第一個 epoch 就消失或爆炸
- 使用 ReLU + He 初始化是目前最常用的安全起點

---

## <a id="batch-norm"></a>📊 Batch Normalization

💡 **實際應用情境：** 批次正規化大幅降低了「內部協變量偏移（Internal Covariate Shift）」問題——每層輸入的分佈不斷改變，後面的層需要不斷適應。BN 讓每一層的輸入分佈穩定在 N(0,1)。

### 範例 2: 使用 Batch Normalization

```python
# 方法 1：BN 在激活函數之前（原始論文）
model_bn_before = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),   # ← BN 在激活函數之前
    keras.layers.Dense(300, use_bias=False),  # BN 有 bias，Dense 不需要
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(100, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10, activation="softmax")
])

# 方法 2：BN 在激活函數之後（更常見的實務做法）
model_bn_after = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),   # ← BN 在激活函數之後
    keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

model_bn_after.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model_bn_after.summary()
```

**✅ 程式碼逐行解析：**

1. `BatchNormalization()`: 在訓練時計算 batch 的均值/方差正規化；推論時使用訓練期間的移動平均
2. `use_bias=False`: BN 層有自己的偏置參數（beta），所以 Dense 層的 bias 是多餘的
3. BN 有 4 個可學習參數：gamma（縮放）、beta（平移）和用於推論的移動平均/方差

**🎯 重點摘要:**

- BN 的主要優點：允許更大的學習率（加速訓練）、提供一定的正則化效果
- 批次太小（< 16）時 BN 效果不好，可改用 Layer Normalization

---

## <a id="gradient-clipping"></a>✂️ 梯度裁剪

### 範例 3: 梯度裁剪（RNN 的關鍵技術）

```python
# 梯度裁剪：防止梯度爆炸（RNN/LSTM 中特別重要）
optimizer_clipped = keras.optimizers.Adam(
    learning_rate=1e-3,
    clipnorm=1.0    # 梯度的 L2 範數裁剪至 1.0
    # 或 clipvalue=0.5  # 每個梯度值裁剪至 [-0.5, 0.5]
)

model_clip = keras.Sequential([
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10,  activation="softmax")
])
model_clip.compile(
    optimizer=optimizer_clipped,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

**🎯 重點摘要:**

- `clipnorm=1.0`: 若梯度向量的 L2 範數 > 1，等比例縮放整個梯度向量
- `clipvalue=0.5`: 直接裁剪每個梯度值到 [-0.5, 0.5]（會改變梯度方向）
- RNN/LSTM 訓練時梯度裁剪幾乎是標配

---

## <a id="transfer-learning"></a>🔄 遷移學習

💡 **實際應用情境：** 台灣醫療新創的皮膚癌偵測系統——沒有足夠的標記資料從頭訓練 CNN，但可以用 ImageNet 預訓練的 ResNet50 作為特徵提取器，只訓練最後幾層。

### 範例 4: 凍結層遷移學習

```python
import tensorflow as tf
from tensorflow import keras

# 載入預訓練模型（不包含頂部分類層）
base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,          # 不含最後的全連接層
    input_shape=(224, 224, 3)
)

# 凍結預訓練層（不訓練）
base_model.trainable = False

# 添加自訂分類頭
model_tl = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),         # 防止過擬合
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax")  # 10 類自訂任務
])

model_tl.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 訓練第一階段（只訓練新增的分類頭）
# history_1 = model_tl.fit(train_dataset, epochs=10, validation_data=val_dataset)

# 第二階段（Fine-tuning：解凍部分底層）
base_model.trainable = True
# 凍結前 100 層，只 fine-tune 後面的層
for layer in base_model.layers[:100]:
    layer.trainable = False

model_tl.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # 使用極小的學習率！
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# history_2 = model_tl.fit(train_dataset, epochs=10, validation_data=val_dataset)
print(f"可訓練參數: {sum(p.numpy().size for p in model_tl.trainable_variables):,}")
```

**✅ 程式碼逐行解析：**

1. `weights="imagenet"`: 載入在 ImageNet 上預訓練的權重（包含豐富的視覺特徵知識）
2. `base_model.trainable = False`: 凍結所有層，只訓練新增的分類頭（快速收斂）
3. Fine-tuning 階段使用極小學習率（1e-5）：避免破壞預訓練的有價值權重

**🎯 重點摘要:**

- 第一階段（凍結基礎層）：幾個 epoch 即可得到不錯的結果
- 第二階段（Fine-tuning）：使用更小的學習率，逐步調整底層特徵
- 資料越少，凍結的層數應越多（避免過擬合）

---

## <a id="optimizers"></a>⚡ 快速優化器比較

### 範例 5: Adam、Nadam、AdaGrad 配置

```python
# Adam（Adaptive Moment Estimation）：最常用
# 結合 Momentum（第一矩）+ RMSProp（第二矩）
optimizer_adam = keras.optimizers.Adam(
    learning_rate=1e-3,
    beta_1=0.9,    # 第一矩（動量）的衰減率
    beta_2=0.999,  # 第二矩（RMSProp）的衰減率
    epsilon=1e-7
)

# Nadam = Nesterov + Adam（略優於 Adam）
optimizer_nadam = keras.optimizers.Nadam(learning_rate=1e-3)

# AdaGrad（適合稀疏資料，如 NLP）
optimizer_adagrad = keras.optimizers.Adagrad(learning_rate=0.01)

# SGD + Nesterov Momentum（收斂質量通常最好，需要更多調參）
optimizer_sgd = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True  # Nesterov 加速梯度
)
```

| 優化器 | 特點 | 建議場景 |
|--------|------|---------|
| SGD + Momentum | 收斂質量好，但調參難 | 有充足資源調參時 |
| AdaGrad | 學習率自適應，但可能過早衰減 | NLP 稀疏梯度 |
| RMSProp | 解決 AdaGrad 衰減問題 | RNN |
| Adam | 自適應 + 動量，開箱即用 | **大多數情況的預設選擇** |
| Nadam | Adam + Nesterov | 需要略好效果時 |

---

## <a id="lr-schedule"></a>📅 學習率排程

### 範例 6: 指數衰減與 1Cycle 排程

```python
# 指數衰減（每 100 個 step 衰減 10%）
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,         # 每 100 步衰減一次
    decay_rate=0.9,          # 衰減率
    staircase=True           # 階梯式衰減（True）vs 連續衰減（False）
)

# 分段常數衰減
lr_piecewise = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[50000, 100000],  # 在第 50k 和 100k 步衰減
    values=[1e-3, 1e-4, 1e-5]   # 對應的學習率
)

# 使用排程
optimizer_with_schedule = keras.optimizers.Adam(learning_rate=lr_schedule)
```

**🎯 重點摘要:**

- 好的學習率策略：熱身期（線性增加）→ 高學習率期 → 衰減期
- 簡單有效：`ExponentialDecay` 或 `ReduceLROnPlateau`（根據驗證損失自動調整）

---

## <a id="regularization"></a>🛡️ 正則化技術

### 範例 7: Dropout 與 MC Dropout

```python
# 標準 Dropout
model_dropout = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dropout(rate=0.3),  # 訓練時隨機丟棄 30% 的神經元
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(10, activation="softmax")
])

# MC Dropout（Monte Carlo Dropout）：推論時也啟用 Dropout
# 多次預測取平均 → 得到不確定性估計
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)  # 推論時也保持 Dropout

model_mc = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    MCDropout(rate=0.3),
    keras.layers.Dense(100, activation="relu"),
    MCDropout(rate=0.3),
    keras.layers.Dense(10, activation="softmax")
])

# MC Dropout 的不確定性估計（多次預測取均值和方差）
# y_probas = np.stack([model_mc.predict(X_test[:1]) for _ in range(100)])
# y_mean = y_probas.mean(axis=0)
# y_std  = y_probas.std(axis=0)

# L1/L2 正則化（對核權重施加懲罰）
model_reg = keras.Sequential([
    keras.layers.Dense(
        300, activation="relu",
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)  # Elastic Net
    ),
    keras.layers.Dense(
        100, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4)
    ),
    keras.layers.Dense(10, activation="softmax")
])
```

**🎯 重點摘要:**

- Dropout 是最有效的正則化之一，通常 rate=0.2~0.5
- MC Dropout：只需修改 `training=True`，即可獲得**不確定性估計**（對醫療 AI 特別有價值）
- Max-Norm 正則化：`kernel_constraint=keras.constraints.max_norm(3.0)`

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: Batch Normalization 放在激活函數之前還是之後？**

A: 原始論文（2015）建議放在激活函數之前；但實務上很多人放在激活函數之後，效果差異不大。若用 ReLU，放在之前可能導致 BN 後的激活值都為正，減少表達能力；放在之後通常更安全。

**Q2: Adam 和 SGD 哪個更好？**

A: 通常 Adam 收斂更快且更穩定（開箱即用），但 SGD + Nesterov + 仔細調參的最終效能可能更好。實務上：先用 Adam 快速得到 baseline，若需最優效能再切換 SGD 精調。

**Q3: 遷移學習適用於哪些場景？**

A: (1) 訓練資料量少（< 10,000）；(2) 新任務與預訓練任務有相似的底層特徵（如都是視覺任務）；(3) 計算資源有限，需要快速收斂。

**Q4: Dropout 的 rate 如何設定？**

A: 通常 0.2~0.5。小型網路用 0.2；大型網路或容易過擬合時用 0.5。輸入層通常用較小的 rate（0.1~0.2），隱藏層用較大的 rate。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #深度學習 #Keras #TensorFlow #BatchNorm #遷移學習 #Adam #Dropout #梯度消失 #神經網路 #程式設計 #教學 #DataScience #MachineLearning
