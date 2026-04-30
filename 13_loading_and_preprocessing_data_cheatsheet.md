# Ch13 速查表：Loading & Preprocessing Data with TensorFlow

> **核心主旨**：`tf.data` 讓資料管線成為訓練瓶頸的解決方案 —— `map().cache().shuffle().batch().prefetch()` 是黃金公式。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| `tf.data.Dataset` | TF 的資料管線 API，支援惰性求值 | 所有 TF 訓練流程 |
| `from_tensor_slices()` | 從記憶體中的陣列/dict 建立 dataset | 資料可放入記憶體 |
| `from_generator()` | 從 Python generator 建立 dataset | 自訂複雜資料來源 |
| `map()` | 對每個元素套用轉換函數 | 圖片解碼、特徵工程 |
| `cache()` | 快取 dataset 到記憶體（第一個 epoch 後） | 避免重複 I/O |
| `shuffle()` | 隨機打亂（重要！用足夠大的 buffer） | 訓練集必用 |
| `batch()` | 組合成批次 | 每次訓練一批 |
| `prefetch()` | 預取下一批，GPU 訓練時重疊 I/O | 幾乎永遠需要 |
| TFRecord | TF 專用二進位格式，讀取極快 | 大型資料集 |
| Keras Preprocessing Layers | 將前處理嵌入模型，部署更簡單 | 標準化、Token化等 |

---

## 2. 關鍵 API 速查

| TF API | 重點參數 | 用途 |
|--------|---------|------|
| `tf.data.Dataset.from_tensor_slices()` | `tensors` | 從記憶體建立 dataset |
| `dataset.map()` | `map_func`, `num_parallel_calls=tf.data.AUTOTUNE` | 並行轉換 |
| `dataset.cache()` | `filename=""` (空字串=記憶體快取) | 快取 |
| `dataset.shuffle()` | `buffer_size=1000`, `seed=42` | 打亂 |
| `dataset.batch()` | `batch_size=32`, `drop_remainder=False` | 批次化 |
| `dataset.prefetch()` | `buffer_size=tf.data.AUTOTUNE` | 預取 |
| `dataset.repeat()` | `count=None` (無限重複) | 多 epoch 時 |
| `tf.io.read_file()` | – | 讀取原始檔案 |
| `tf.io.decode_jpeg()` | – | 解碼 JPEG 圖片 |
| `tf.image.resize()` | `size=[224, 224]` | 調整圖片大小 |
| `tf.keras.layers.Normalization` | `axis=-1` | 特徵標準化層 |
| `tf.keras.layers.TextVectorization` | `max_tokens=`, `output_sequence_length=` | 文字 Token 化 |

---

## 3. 必備代碼片段

```python
import tensorflow as tf

# 黃金公式：訓練集管線
def create_train_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X), seed=42)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 驗證集管線（不 shuffle）
def create_val_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 圖片資料管線範例
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.xception.preprocess_input(image)
    return image, label

# TFRecord 寫入
def serialize_example(feature_dict):
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()

with tf.io.TFRecordWriter("data.tfrecord") as writer:
    for X, y in zip(X_train, y_train):
        feature = {
            "X": tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
            "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
        }
        writer.write(serialize_example(feature))

# TFRecord 讀取
feature_description = {
    "X": tf.io.FixedLenFeature([n_features], tf.float32),
    "y": tf.io.FixedLenFeature([], tf.int64)
}

def parse_example(serialized):
    example = tf.io.parse_single_example(serialized, feature_description)
    return example["X"], example["y"]

dataset = tf.data.TFRecordDataset("data.tfrecord").map(parse_example)

# Keras 前處理層（嵌入模型，部署一致性）
normalization_layer = tf.keras.layers.Normalization(axis=-1)
normalization_layer.adapt(X_train)  # 計算 mean 和 std

model = tf.keras.Sequential([
    normalization_layer,          # 前處理包在模型裡
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# tfds 載入公開資料集
import tensorflow_datasets as tfds
(train_ds, valid_ds), ds_info = tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)
n_classes = ds_info.features["label"].num_classes
```

---

## 4. 常見陷阱

- **`shuffle` buffer_size 要夠大**：`buffer_size=1000` 只能保證 1000 個樣本的隨機性；理想情況用整個資料集大小，但記憶體有限時用 `min(len(dataset), 10000)`。
- **`cache()` 的位置**：必須放在 `map()` 之後（快取處理後的資料），`shuffle()` 之前（每 epoch 重新打亂）。
- **`prefetch()` 要放最後**：讓 CPU 預取下一批的同時 GPU 正在訓練當前批。
- **`num_parallel_calls=tf.data.AUTOTUNE`**：讓 TF 自動決定並行程度，幾乎每次 `map()` 都應加。
- **前處理層 `adapt()` 只能在訓練集上執行**：和 sklearn 的 `StandardScaler.fit()` 一樣，不能看驗證/測試集。

---

## 5. 決策指南

```
資料大小 vs 策略：
├── 全部放入記憶體 (< ~10GB)
│   → from_tensor_slices + cache() 在 shuffle 前
├── 中型資料（SSD 上的檔案）
│   → 讀取路徑清單 + map(load_file) + cache() + shuffle
└── 超大資料（雲端 / 分散式）
    → TFRecord 格式 + TFRecordDataset + 分片

管線效能調優 checklist：
1. map() 加 num_parallel_calls=AUTOTUNE
2. cache() 避免重複 I/O
3. shuffle() 的 buffer_size 夠大
4. prefetch(AUTOTUNE) 放最後
5. 圖片解碼考慮用 tf.io.decode_image（更通用）
```
