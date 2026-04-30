<!-- meta-title: TensorFlow 資料載入與預處理完整指南：tf.data API 與 TFRecord -->
<!-- meta-description: 掌握 TensorFlow 的資料管線：tf.data API（from_tensor_slices、map、batch、prefetch）、TFRecord 格式、Protocol Buffers、Keras 預處理層（Normalization、TextVectorization）與圖像增強。 -->
<!-- meta-keywords: Python, TensorFlow, tf.data, TFRecord, 資料管線, Keras, 預處理, TextVectorization, 圖像增強, 機器學習 -->
<!-- meta-hashtags: #Python #TensorFlow #tfdata #TFRecord #Keras #資料管線 #預處理 #TextVectorization #深度學習 #教學 -->

# 🐍 TensorFlow 資料載入與預處理：tf.data API 完整指南

在深度學習中，**資料管線（Data Pipeline）** 往往是訓練速度的瓶頸。GPU 等待資料的時間比計算時間更長——這就是 tf.data API 誕生的原因。本教學帶你建構高效的 TF 資料管線，從原始資料到模型輸入的每一步都能充分利用多核心 CPU 和磁碟 I/O。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🚀 tf.data API 基礎](#tfdata-basics)
- [🔧 資料轉換與增強](#transformations)
- [📼 TFRecord 格式](#tfrecord)
- [🧩 Keras 預處理層](#keras-preprocessing)
- [📝 文字向量化](#text-vectorization)
- [🖼️ 圖像預處理與增強](#image-preprocessing)
- [⚡ 效能最佳化](#performance)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **`tf.data.Dataset`** 是 TF 的惰性評估資料管線，支援平行預取和多執行緒映射
- **`prefetch(tf.data.AUTOTUNE)`** 讓 CPU 預處理和 GPU 訓練**同步進行**，消除等待時間
- **TFRecord** 是 TF 原生的二進位格式，對大型資料集（圖像、音訊）讀取效率最高
- **Keras 預處理層**（`Normalization`、`TextVectorization`）可以整合到模型中，在推論時自動應用
- **圖像增強**（隨機翻轉、裁切、亮度調整）應只在訓練時使用（`training=True`）

---

## <a id="tfdata-basics"></a>🚀 tf.data API 基礎

💡 **實際應用情境：** 台灣電商平台的推薦系統每天需要處理數 TB 的使用者行為日誌。用 tf.data 的平行映射和預取，可以讓 GPU 的利用率從 20% 提升到 90% 以上——訓練速度翻倍。

### 範例 1: 建立和操作 Dataset

```python
import tensorflow as tf
import numpy as np

# 方法 1：從記憶體資料建立（小型資料集）
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randint(0, 10, 1000)

# from_tensor_slices：沿第一個軸切片（每個樣本一個元素）
dataset = tf.data.Dataset.from_tensor_slices((X, y))
print(f"資料集大小: {len(dataset)}")     # 1000
print(f"元素規格: {dataset.element_spec}")  # 顯示形狀和 dtype

# 查看前 3 個元素
for x_sample, y_sample in dataset.take(3):
    print(f"x shape: {x_sample.shape}, y: {y_sample.numpy()}")

# 方法 2：從生成器建立（適合無法全部載入記憶體的資料）
def data_generator():
    for i in range(1000):
        yield np.random.randn(10), np.random.randint(10)

gen_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(10,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# 方法 3：讀取文字文件（每行一個元素）
# text_dataset = tf.data.TextLineDataset(["train.txt"])
```

**✅ 程式碼逐行解析：**

1. `from_tensor_slices((X, y))`: 建立 (X, y) 對的資料集，每次迭代回傳一對樣本
2. `dataset.element_spec`: 顯示每個元素的 `TensorSpec`（形狀 + dtype），有助於除錯
3. `from_generator`: 惰性評估——只在需要時呼叫生成器，適合大型資料集

**🎯 重點摘要:**

- `Dataset` 物件是**惰性的**——只有在迭代時才真正執行資料讀取和轉換
- `take(n)` 只取前 n 個元素，非常適合除錯時快速驗證管線

---

## <a id="transformations"></a>🔧 資料轉換與增強

### 範例 2: 標準訓練資料管線

```python
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE  # 讓 TF 自動選擇最佳平行度

def preprocess(x, y):
    """資料預處理函數（在 CPU 上平行執行）"""
    # 特徵標準化
    x = (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-8)
    return x, y

# 標準訓練資料管線
train_dataset = (
    dataset
    .shuffle(buffer_size=1000)    # 隨機打亂（buffer_size 越大越隨機）
    .map(preprocess,               # 對每個元素應用預處理函數
         num_parallel_calls=AUTOTUNE)  # 多執行緒平行映射
    .batch(BATCH_SIZE)             # 組合成 batch
    .prefetch(AUTOTUNE)            # 預取下一個 batch（消除 CPU-GPU 等待）
)

# 驗證/測試集管線（不需要 shuffle）
val_dataset = (
    dataset
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
    # 可加 .cache() 緩存到記憶體（如果資料集夠小）
)

print(f"每個 batch 的形狀: {next(iter(train_dataset))[0].shape}")
```

**✅ 程式碼逐行解析：**

1. `shuffle(buffer_size=1000)`: 維持一個大小為 1000 的緩衝區，從中隨機取樣；buffer_size 越大越接近真正隨機但記憶體用量也越大
2. `map(fn, num_parallel_calls=AUTOTUNE)`: 多執行緒平行應用 fn，AUTOTUNE 讓 TF 自動決定最佳執行緒數
3. `prefetch(AUTOTUNE)`: 在 GPU 執行當前 batch 的同時，預先準備下一個 batch——這是最重要的效能優化！

**🎯 重點摘要:**

- **必加** `prefetch(AUTOTUNE)` — 這是零成本的最大效能提升
- 管線順序很重要：`shuffle → map → batch → prefetch`（而非 `batch` 後再 `shuffle`）

---

## <a id="tfrecord"></a>📼 TFRecord 格式

💡 **實際應用情境：** 醫療影像資料集（胸部 X 光，每張幾 MB）直接從 JPEG 讀取效率低下。轉換為 TFRecord 後，I/O 速度可提升 3-5 倍，因為順序讀取大型二進制文件遠快於隨機讀取許多小文件。

### 範例 3: 寫入 TFRecord 文件

```python
import os

# 特徵序列化輔助函數
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(x: np.ndarray, label: int) -> bytes:
    """將一個樣本序列化為 TFRecord 格式"""
    feature = {
        "features": _bytes_feature(x.tobytes()),   # 將陣列轉為 bytes
        "label":    _int64_feature(label),
        "n_features": _int64_feature(len(x))        # 存儲形狀資訊
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()  # 序列化為 bytes

# 寫入 TFRecord
output_path = "my_data.tfrecord"
with tf.io.TFRecordWriter(output_path) as writer:
    for i in range(len(X)):
        serialized = serialize_example(X[i], y[i])
        writer.write(serialized)
print(f"TFRecord 已儲存: {os.path.getsize(output_path)/1024:.1f} KB")
```

### 範例 4: 讀取 TFRecord 文件

```python
# 描述特徵格式（用於反序列化）
feature_description = {
    "features":   tf.io.FixedLenFeature([], tf.string),
    "label":      tf.io.FixedLenFeature([], tf.int64),
    "n_features": tf.io.FixedLenFeature([], tf.int64),
}

def parse_tfrecord(serialized_example):
    """將 bytes 反序列化回張量"""
    parsed = tf.io.parse_single_example(serialized_example, feature_description)

    # 將 bytes 還原為 float32 陣列
    n = parsed["n_features"]
    x = tf.io.decode_raw(parsed["features"], tf.float32)
    x = tf.reshape(x, [n])
    label = parsed["label"]
    return x, label

# 建立 TFRecord 資料集
tfrecord_dataset = (
    tf.data.TFRecordDataset([output_path])
    .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    .shuffle(500)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# 驗證
for x_batch, y_batch in tfrecord_dataset.take(1):
    print(f"從 TFRecord 讀取: x shape={x_batch.shape}, y shape={y_batch.shape}")
```

**✅ 程式碼逐行解析：**

1. `tf.train.Feature(bytes_list=...)`: TFRecord 支援 bytes/float/int64 三種特徵類型
2. `tf.train.Example(features=...)`: 一個 Example 包含多個命名特徵，相當於一行資料
3. `tf.io.parse_single_example`: 根據 `feature_description` 反序列化，類型不符會報錯

---

## <a id="keras-preprocessing"></a>🧩 Keras 預處理層

### 範例 5: Normalization 層（嵌入模型的標準化）

```python
import numpy as np

X_housing = np.random.randn(1000, 8).astype(np.float32)
y_housing = np.random.randn(1000, 1).astype(np.float32)

# Normalization 層：計算並儲存訓練集的均值和方差
normalizer = tf.keras.layers.Normalization(input_shape=[8])
normalizer.adapt(X_housing)  # 在訓練資料上計算統計量（類似 fit）

print(f"均值: {normalizer.mean.numpy().round(3)}")
print(f"方差: {normalizer.variance.numpy().round(3)}")

# 整合進模型（推論時自動標準化）
model_with_norm = tf.keras.Sequential([
    normalizer,                               # 第一層：標準化
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)                  # 回歸輸出
])

model_with_norm.compile(optimizer="adam", loss="mse")
# model_with_norm.fit(X_housing, y_housing, epochs=5)
```

**🎯 重點摘要:**

- 將預處理嵌入模型的**核心優點**：部署時無需單獨執行預處理——用戶只需傳入原始資料
- `adapt()` 類似 `fit()`，需要在訓練資料上呼叫（不能在測試集上）

---

## <a id="text-vectorization"></a>📝 文字向量化

### 範例 6: TextVectorization 層

```python
# 示範文字資料
texts = [
    "台灣的半導體產業很強大",
    "機器學習改變了科技產業",
    "深度學習需要大量資料",
    "GPU 是深度學習的核心硬體",
]

# TextVectorization：字符/詞語 → 整數索引
text_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=50,            # 詞彙表大小
    output_mode="int",        # 輸出整數索引（也可以是 "tf-idf" 或 "binary"）
    output_sequence_length=10  # 固定序列長度（截斷或填充）
)

# 訓練詞彙表
text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(4))
print(f"詞彙表大小: {text_vectorizer.vocabulary_size()}")
print(f"前 10 個詞: {text_vectorizer.get_vocabulary()[:10]}")

# 向量化
vectorized = text_vectorizer(["機器學習改變了科技產業"])
print(f"向量化結果: {vectorized.numpy()}")

# 嵌入到模型中
vocab_size = text_vectorizer.vocabulary_size()
text_model = tf.keras.Sequential([
    text_vectorizer,
    tf.keras.layers.Embedding(vocab_size, 16, mask_zero=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
text_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

**🎯 重點摘要:**

- `output_mode="int"` 適合 RNN/Transformer；`"tf-idf"` 或 `"binary"` 適合傳統 ML 模型
- `mask_zero=True`：讓模型知道哪些位置是填充（padding）

---

## <a id="image-preprocessing"></a>🖼️ 圖像預處理與增強

### 範例 7: 圖像增強管線

```python
from tensorflow import keras

# Keras 內建圖像增強層（只在訓練時啟用）
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),  # 隨機水平/垂直翻轉
    keras.layers.RandomRotation(0.1),    # 隨機旋轉 ±10%（以 2π 為單位）
    keras.layers.RandomZoom(0.1),        # 隨機縮放 ±10%
    keras.layers.RandomBrightness(0.1),  # 隨機調整亮度
    keras.layers.RandomContrast(0.1),    # 隨機調整對比度
    keras.layers.Rescaling(1./255)       # 像素值正規化到 [0, 1]
], name="data_augmentation")

# 整合圖像增強到模型（增強只在 training=True 時生效）
base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs, training=True)  # 訓練時增強，推論時跳過
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

model_img = keras.Model(inputs=inputs, outputs=outputs)
model_img.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
```

---

## <a id="performance"></a>⚡ 效能最佳化

### 範例 8: 測量管線效能

```python
import time

def measure_dataset_performance(dataset, n_batches=100):
    """測量資料管線每秒能處理多少批次"""
    start = time.time()
    for i, _ in enumerate(dataset.take(n_batches)):
        pass
    elapsed = time.time() - start
    return n_batches / elapsed

# 無優化 vs 有優化的管線比較
raw_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
optimized_dataset = (
    tf.data.Dataset.from_tensor_slices((X, y))
    .shuffle(1000)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .cache()         # 緩存到記憶體（第一個 epoch 後免費！）
    .prefetch(AUTOTUNE)
)

# 最佳化 cache() 的使用：在 map 之後
# .map(preprocess).cache().batch().prefetch()
# 這樣 cache 儲存的是預處理後的資料，避免重複計算
```

**🎯 重點摘要:**

- **效能優化黃金法則**：`shuffle → map → cache → batch → prefetch`
- `cache()` 在第一個 epoch 後將資料緩存，後續 epoch 的 I/O 幾乎降為零
- 使用 `tf.data.experimental.AutoShardPolicy` 在多 GPU 環境中自動分片

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: `shuffle()` 的 `buffer_size` 如何設定？**

A: 理想情況下設為資料集大小（完全隨機），但記憶體有限時可以設為 5× batch_size 或更大。關鍵原則：至少設為 batch_size 的幾倍，否則不同 batch 間可能高度相關。

**Q2: 應該在 `batch` 之前還是之後做 `map`？**

A: 通常在 `batch` 之前 `map`（對每個樣本單獨處理），因為更容易寫；但某些增強操作（如 `MixUp`）需要在 batch 後執行。`map` 之後加 `cache()` 效率最高。

**Q3: TFRecord vs TF Datasets（tfds）的選擇？**

A: TFRecord 適合自訂格式的大型資料集（完全控制）；`tensorflow_datasets`（tfds）提供 300+ 標準資料集，一行代碼即可載入。快速實驗用 `tfds`，生產系統用 TFRecord。

**Q4: 為什麼要將預處理層嵌入模型而非在管線中處理？**

A: 嵌入模型的預處理在**部署時自動執行**——只需傳入原始資料，模型內部處理所有預處理。如果在管線中處理，部署時需要額外的預處理代碼，增加出錯風險。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #TensorFlow #tfdata #TFRecord #Keras #資料管線 #預處理 #TextVectorization #深度學習 #圖像增強 #程式設計 #教學 #DataScience #MachineLearning
