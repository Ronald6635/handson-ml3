# 課程講義：TensorFlow 資料載入與預處理 (Chapter 13)

本章節聚焦於 TensorFlow 資料管線的建構與優化，說明如何使用 `tf.data` 讀取、轉換、批次化與預取資料，並延伸到 TFRecord、Protobuf 以及 Keras 前處理層的實務應用。

## 1. 引言

在深度學習系統中，資料輸入管線往往比模型本身更容易成為效能瓶頸。本講義以第13章教材為基礎，解析 TensorFlow 的資料載入策略，讓你能夠建立可擴充、可重用且具備訓練效率的資料處理流程。

## 2. 核心掌握

### 2.1 tf.data API 的基礎與資料流

**為什麼重要**：`tf.data` 是 TensorFlow 建立輸入管線的核心 API，它能將資料讀取、轉換與批次化串成單一管線，避免在訓練期間反覆進行 Python 層級資料處理。

**如何實作**：

```python
import tensorflow as tf

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
```

- `from_tensor_slices()` 將張量切片成單一元素資料集。
- 這種作法適合資料可放入記憶體的情境。
- 若資料太大，應改用檔案來源（如 CSV、TFRecord、Image）。

**⚡ 補充練習 2.1：**
- 理論：比較 `from_tensor_slices()` 與 `from_tensors()` 的行為差異，並說明在何種情境下選用各自 API。

```python
dataset2 = tf.data.Dataset.from_tensors(X)
```

```python
import tensorflow as tf

# 假設這是你的特徵和標籤
features = tf.constant([[1, 2], [3, 4], [5, 6]]) # 3個樣本，每個樣本有2個特徵
labels = tf.constant([0, 1, 0]) # 3個對應的標籤

dataset_slices = tf.data.Dataset.from_tensor_slices((features, labels))

# 檢視資料集的元素規格
dataset_slices.element_spec

print("from_tensor_slices() 的輸出：")
for element in dataset_slices:
    print(element)

# 假設這是一個完整的張量，你希望它作為資料集的唯一元素
full_tensor = tf.constant([[1, 2], [3, 4], [5, 6]])

dataset_tensors = tf.data.Dataset.from_tensors(full_tensor)

print("\nfrom_tensors() 的輸出：")
for element in dataset_tensors:
    print(element)
```

- 實作：建立一個 `tf.data.Dataset`，然後使用 `batch(3)` 與 `repeat(2)` 觀察輸出形態。

```python
import tensorflow as tf

# 1. 建立一個基礎的 tf.data.Dataset
# 包含從 0 到 9 的 10 個元素
initial_data = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(initial_data)

print("--- 原始資料集 (10 個元素) ---")
for i, element in enumerate(dataset):
    print(f"元素 {i}: {element.numpy()}")

# 2. 應用 repeat(2) 後再應用 batch(3)
# 整個資料集重複 2 次 (邏輯上變成 20 個元素)，然後每 3 個元素組成一個批次
processed_dataset = dataset.repeat(2).batch(3)

print("\n--- 經過 repeat(2).batch(3) 處理後的資料集 ---")
print("資料集的每個元素將是一個批次 (Tensor)")
print("資料集總共有 20 個邏輯元素，會被分成 7 個批次 (6 個批次大小為 3，1 個批次大小為 2)")

for i, batch_element in enumerate(processed_dataset):
    print(f"批次 {i+1}:")
    print(f"  內容: {batch_element.numpy()}")
    print(f"  形狀: {batch_element.shape}") # 觀察每個批次張量的形狀
    print(f"  批次大小: {batch_element.shape[0]}") # 觀察批次大小
```

### 2.2 轉換與交錯讀取：map、filter、interleave

**為什麼重要**：資料前處理通常需要多次轉換，若每次都在 Python 中完成，會嚴重拖慢訓練速度。`tf.data` 提供的 `map()`、`filter()` 與 `interleave()` 可將轉換移到 TensorFlow 執行，並行化處理。

**如何實作**：

```python
dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))
dataset = dataset.repeat(3).batch(7)
dataset = dataset.map(lambda x: x * 2)
```

- `repeat(3)` 代表資料集重複3次。
    - 這會讓資料集的邏輯元素從 10 個變成 30 個。
    - 形成的資料集會是 [0, 1, 2, ..., 9, 0, 1, 2, ..., 9, 0, 1, 2, ..., 9]。
- `batch(7)` 可減少模型呼叫次數，提高 GPU 利用率。
    - 這會將資料集分成批次，每個批次包含 7 個元素。
    - 由於資料集有 30 個元素，會形成 5 個批次（4 個批次大小為 7，1 個批次大小為 2）。
- `map()` 會對每個批次或元素執行轉換，與是否先 `batch()` 密切相關。
    - 若在 `batch()` 前使用 `map()`，則轉換會對每個元素執行。
    - 若在 `batch()` 後使用 `map()`，則轉換會對每個批次執行，這通常更有效率。

```python
filepaths = 'C:\\Users\\yuanh\\Box\\tech\\Python\\side_projects\\twstock\\engine\\robot\\reports\\*.csv'
dataset = tf.data.Dataset.list_files(filepaths, seed=42) # 讀取多個 CSV 檔案

for fp in dataset:
    print(fp)

# 使用 interleave 同時讀取多個檔案，並跳過 CSV 標頭
dataset = dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=5)
```

- `interleave()` 可以同時讀取多個檔案，適合多檔案資料集。
- `skip(1)` 用於跳過 CSV 標頭。

**⚡ 補充練習 2.2：**
- 理論：說明為何在多檔案讀取時，`interleave()` 通常比 `flat_map()` 更適合。
    - `interleave()` 會在多個檔案之間交錯讀取，能更快地提供資料給模型，減少 I/O 等待時間。
    - `flat_map()` 會將每個檔案的資料完全讀取後才開始讀取下一個檔案，可能導致資料供應不均，增加訓練等待時間。
- 實作：建立一個多檔案 `TextLineDataset`，並使用 `interleave()` 同時讀取多個檔案。

### 2.3 CSV、TFRecord 與資料預處理管線

**為什麼重要**：當資料量增長時，單一 CSV 或 NumPy 檔案將無法支援高效訓練。TFRecord 可序列化資料，並搭配 `tf.data` 形成高效管線。

**如何實作**：

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
```

```python
import numpy as np
from pathlib import Path

def save_to_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = Path() / "datasets" / "housing"
    housing_dir.mkdir(parents=True, exist_ok=True)
    chunks = np.array_split(data, n_parts)
    for file_idx, chunk in enumerate(chunks):
        part_csv = housing_dir / f"my_{name_prefix}_{file_idx:02d}.csv"
        np.savetxt(part_csv, chunk, delimiter=",", header=header, comments="")

save_to_csv_files(X_train_full, "train_X", header="MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude")
save_to_csv_files(y_train_full, "train_y", header="MedHouseVal")
```

- 將資料拆成多個 CSV 檔案可提升 I/O 吞吐量。
- 若資料位於遠端儲存或分散式環境，分割檔案更具可擴充性。

```python
def parse_features(line):
    fields = tf.strings.split(line, ",")
    return tf.strings.to_number(fields, out_type=tf.float32)

def parse_labels(line):
    return tf.strings.to_number(line, out_type=tf.float32)

def csv_reader_dataset(filepaths, parser, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=42)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=5)
    dataset = dataset.map(parser, num_parallel_calls=5)
    dataset = dataset.shuffle(10_000, seed=42)
    return dataset.batch(batch_size).prefetch(1)

housing_dir = Path() / "datasets" / "housing"
train_set = csv_reader_dataset(str(housing_dir / "my_train_X_*.csv"), parse_features)
target_set = csv_reader_dataset(str(housing_dir / "my_train_y_*.csv"), parse_labels)
dataset = tf.data.Dataset.zip((train_set, target_set))

```

- `prefetch(1)` 讓資料載入與模型訓練交錯執行。
- `shuffle(10_000)` 提高隨機化效果，減少過擬合風險。

**⚡ 補充練習 2.3：**
- 理論：說明為何 `prefetch()` 能改善 GPU 利用率，以及何時不應該過多預取。
- 實作：加入 `cache()` 於 `tf.data` 管線中，觀察同一資料集不同快取位置的效能改變。

```python
def csv_reader_dataset_cache(filepaths, parser, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=42)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=5)
    dataset = dataset.cache() # 將資料集快取在記憶體中
    dataset = dataset.map(parser, num_parallel_calls=5)
    dataset = dataset.shuffle(10_000, seed=42)
    return dataset.batch(batch_size).prefetch(1)

# housing_dir = Path() / "datasets" / "housing"
train_set = csv_reader_dataset_cache(str(housing_dir / "my_train_X_*.csv"), parse_features)
target_set = csv_reader_dataset_cache(str(housing_dir / "my_train_y_*.csv"), parse_labels)
dataset = tf.data.Dataset.zip((train_set, target_set))
```

### 2.4 Protobuf 與 TFRecord 的實作原理

**為什麼重要**：`tf.train.Example` 透過 Protobuf 描述資料欄位，適合儲存圖片、文字、數值等多樣資料；TFRecord 是序列化此資料的高效格式。

**如何實作**：

```python
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

person_example = Example(
    features=Features(feature={
        "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
        "id": Feature(int64_list=Int64List(value=[123]))
    }))
person_example
person_example.features.feature["name"].bytes_list.value[0] # b"Alice"
person_example.features.feature["id"].int64_list.value[0] # 123
```

- `BytesList`、`FloatList`、`Int64List` 分別對應不同資料型態。
- `Feature` 以 `oneof` 方式封裝實際欄位值。
- `Example` 是序列化後的單筆資料格式。

```python
record_path = str(Path() / "datasets" / "my_data.tfrecord")
with tf.io.TFRecordWriter(record_path) as f:
    f.write(person_example.SerializeToString())
```

- `SerializeToString()` 將 Example 轉成二進位紀錄。
- `TFRecordWriter` 提供一個簡潔寫入介面。

**⚡ 補充練習 2.4：**
- 理論：比較 `tf.io.parse_single_example()` 與 `tf.io.parse_example()` 的差異，說明何時使用 batch 解析更有效率。
- 實作：將 Fashion MNIST 圖片序列化為 TFRecord，並撰寫解析函數還原成原始影像。

```python
# 讀取 TFRecord
raw_dataset = tf.data.TFRecordDataset([record_path])

# 定義解析規格
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string),
    "id": tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(serialized_example):
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    parsed["name"] = tf.cast(parsed["name"], tf.string)
    return parsed

parsed_dataset = raw_dataset.map(parse_example)

# 檢視解析結果
for item in parsed_dataset:
    print("name:", item["name"].numpy().decode("utf-8"))
    print("id:  ", item["id"].numpy())
```

```python
# 解析 Fashion MNIST TFRecord 的範例
import tensorflow as tf

# 1) 讀取 Fashion MNIST
(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()

# 2) helper：將欄位轉成 tf.train.Feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_label_to_example(image, label):
    image_bytes = tf.io.encode_png(image[..., tf.newaxis]).numpy()
    feature = {
        "image": _bytes_feature(image_bytes),
        "label": _int64_feature(int(label)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# 3) 寫入 TFRecord
record_path = str(Path() / "datasets" / "fashion_mnist.tfrecord")
with tf.io.TFRecordWriter(record_path) as writer:
    for image, label in zip(train_images, train_labels):
        example = image_label_to_example(image, label)
        writer.write(example.SerializeToString())

# 4) 讀取與解析 TFRecord
feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(serialized_example):
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_png(parsed["image"], channels=1)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(parsed["label"], tf.int32)
    return image, label

dataset = tf.data.TFRecordDataset([record_path])
dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# 5) 檢視解析後的資料
for images, labels in dataset.take(1):
    print("影像批次形狀:", images.shape) # (32, 28, 28, 1)
    print("標籤批次形狀:", labels.shape) # (32,)
    print("第一張影像的像素值範圍:", tf.reduce_min(images[0]), "到", tf.reduce_max(images[0]))
    print("第一張影像的標籤:", labels[0].numpy())

```

```python
import tensorflow as tf
from pathlib import Path

(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
record_dir = Path() / "datasets" / "fashion_mnist_tfrecords"
record_dir.mkdir(exist_ok=True)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_label_to_example(image, label):
    image_bytes = tf.io.encode_png(image[..., None]).numpy()
    return tf.train.Example(
        features=tf.train.Features(feature={
            "image": _bytes_feature(image_bytes),
            "label": _int64_feature(int(label)),
        })
    )

def write_fashion_mnist_shards(images, labels, record_dir, num_shards=10):
    n = len(images)
    shard_size = (n + num_shards - 1) // num_shards
    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n)
        shard_path = record_dir / f"fashion_mnist_{shard_id:02d}.tfrecord"
        with tf.io.TFRecordWriter(str(shard_path)) as writer:
            for image, label in zip(images[start:end], labels[start:end]):
                writer.write(image_label_to_example(image, label).SerializeToString())

write_fashion_mnist_shards(train_images, train_labels, record_dir, num_shards=10)

files = tf.io.gfile.glob(str(record_dir / "fashion_mnist_*.tfrecord"))
raw_dataset = tf.data.TFRecordDataset(files)
dataset = raw_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# plot the first batch of images
import matplotlib.pyplot as plt

for images, labels in dataset.take(1):
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, :, :, 0], cmap='gray')
        ax.set_title(labels[i].numpy())
        ax.axis('off')
    plt.show()
```

### 2.5 Keras 前處理層的整合與實務

**為什麼重要**：Keras 前處理層可以將輸入標準化、離散化、文字向量化等步驟內建到模型中，使訓練與部署一致。

**如何實作**：
以下是將 `Normalization` 層整合到 Keras 模型中的範例：

```python
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train_full)
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Input(shape=X_train_full.shape[1:]),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])
```

- `Normalization` 會自動計算特徵平均值與變異數。
- `adapt()` 只需對訓練資料執行一次。
- 模型內建前處理層可確保訓練與推論使用相同的轉換邏輯。

這是將 `TextVectorization` 層整合到文本分類模型中的範例：

```python
text_vec_layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")
text_vec_layer.adapt(train_data)
```

- `TextVectorization` 可直接將文字轉成向量表示。
- `output_mode="tf_idf"` 適合文本分類與檢索任務。
    - 若要使用 embedding，則 `output_mode="int"` 並搭配 `Embedding` 層。

**⚡ 補充練習 2.5：**
- 理論：說明 `StringLookup` 與 `Embedding` 的角色差異，並討論何時採用 embedding 取代 one-hot。
- 實作：用 `TextVectorization` 建立 IMDB 影評分類模型，並比較 `tf_idf` 與 `int` + `Embedding` 兩種前處理效果。

## 3. 結論

本章重點在於建立一個從資料讀取到模型輸入的完整資料管線。掌握 `tf.data` 的組合方式、TFRecord/Protobuf 的序列化設計，以及 Keras 前處理層的整合，能讓你在實際專案中同時提升訓練效能與部署穩定性。下一章將深入影像資料與更複雜的前處理策略，進一步拓展資料工程能力。

## 4. 課後作業

1. **實作題**：使用 Fashion MNIST，將資料分割成多個 TFRecord 檔案，編寫 `tf.data` 管線讀取並訓練簡單分類模型。比較直接使用 `tf.data.Dataset.from_tensor_slices()` 與 TFRecord 兩種方式的訓練速度。
2. **思考題**：若資料集包含文字、數值與類別欄位，請設計一個 `tf.data` + Keras 前處理層的完整管線，說明每個步驟為何要放在資料管線中、每個步驟為何要放入模型中。
