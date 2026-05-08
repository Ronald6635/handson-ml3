<!-- meta-title: TensorFlow 資料載入與預處理教學：tf.data、TFRecord、Keras 前處理層 -->
<!-- meta-description: 本文介紹 Chapter 13 的 TensorFlow 資料載入與預處理實務，從 tf.data 管線、CSV/TFRecord、Protobuf，到 Keras 前處理層與 TFDS 全面解析。 -->
<!-- meta-keywords: TensorFlow, tf.data, TFRecord, Keras, 資料預處理, Protobuf, tfds, Python, 教學 -->
<!-- meta-hashtags: #Python #TensorFlow #資料科學 #機器學習 #教學 #開發者社群 -->

# 🐍 Chapter 13：TensorFlow 資料載入與預處理實戰

本章節範例來自 13_loading_and_preprocessing_data.ipynb，我們將用 TensorFlow 的 `tf.data` API 建立高效資料管線，講解從 CSV、TFRecord、Protobuf 到 Keras 前處理層的實作流程，並補充實務觀察與優化建議。

## 📝 本文目錄
- 📦 [tf.data API 基礎](#tf-data-basics)
- 🔀 [資料串接與轉換](#data-transform)
- 🎲 [資料洗牌與交錯讀取](#shuffle-interleave)
- 🧾 [CSV 讀取與前處理](#csv-preprocess)
- 🔧 [Keras 與 Dataset 串接](#keras-with-dataset)
- 📦 [TFRecord 格式與 Protobuf](#tfrecord-protobuf)
- 🧩 [Keras 前處理層](#keras-preprocessing)
- 🌐 [TensorFlow Datasets (TFDS)](#tfds)
- ✅ [章節重點與實務建議](#summary)
- ❓ [常見問答](#faq)

---

## 🎯 關鍵重點 (Key Takeaways)
- `tf.data` 是建立可重用、高效資料管線的核心 API。
- `TFRecord` 與 `Protobuf` 適合序列化大量資料、跨平台部署與分散式訓練。
- Keras 前處理層（如 `Normalization`、`TextVectorization`）可將資料預處理內建到模型中。
- 針對大型資料集，讀取、解析、批次化與預取是性能優化的四大要素。

---

## <a id="tf-data-basics"></a> 📦 tf.data API 基礎

`tf.data` 提供一個統一 API 來處理資料集，從單一張量到結構化嵌套資料都能無縫支援。

```python
import tensorflow as tf

X = tf.range(10)  # any data tensor
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset
```

✅ 程式碼逐行解析:
1. `tf.range(10)` 建立一個從 0 到 9 的整數張量。
2. `tf.data.Dataset.from_tensor_slices(X)` 將張量切片成一個資料集，資料集每次產生一個元素。

🎯 重點摘要:
- `from_tensor_slices()` 適合記憶體中資料。
- 若資料量太大，應改用檔案資料來源或 `TFRecordDataset`。
- 這是 `tf.data` 管線的最簡單入口。

```python
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `for item in dataset:` 逐項遍歷 `tf.data.Dataset`。
2. `print(item)` 印出每個元素。

🎯 重點摘要:
- 直接迭代資料集可以檢查內容。
- 注意：在真實訓練時應避免逐元素 Python 迭代，應使用 `batch()` 批次化。

## <a id="data-transform"></a> 🔀 資料串接與轉換

`tf.data` 的強大之處在於可以串接多個轉換步驟，形成可重用的管線。

```python
dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `Dataset.from_tensor_slices(tf.range(10))` 建立壹個 0-9 的資料集。
2. `.repeat(3)` 讓資料集重複三次。
3. `.batch(7)` 將資料切成每批 7 個元素。
4. `for item in dataset:` 印出每批資料。

🎯 重點摘要:
- `repeat()` 可用於訓練多個 epoch。
- `batch()` 是資料管線中最重要的性能優化步驟。
    - 批次化後的資料可以更有效率地送入 GPU/TPU。
    - 如果最後一個批次太小（例如只有 2 個樣本），有時會導致 Batch Normalization 層出現統計不穩定的問題。這時我們會用 `drop_remainder=True` 來丟棄最後一個不完整的批次。
- 若在交錯讀取前使用 `shuffle()`，可得到更好的隨機化效果。

---

接下來，我們來看看 tf.data 最強大的 **轉換（Transformations）** 功能：

```python
dataset = dataset.map(lambda x: x * 2)  # x is a batch
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `.map(lambda x: x * 2)` 對每個批次進行元素級轉換。
2. `x` 表示整個批次資料，而不是單一元素。

🎯 重點摘要:
- `map()` 可在批次級別或元素級別執行轉換。
- 當資料已 `batch()` 後，`map()` 代表的是批次操作。
    - 效能優化: 在 `map()` 中使用 `num_parallel_calls=tf.data.AUTOTUNE` 可自動調整並行度，提升轉換效率。

## <a id="shuffle-interleave"></a> 🎲 資料洗牌與交錯讀取

資料洗牌與交錯讀取可改善資料隨機性與輸入效率。

```python
dataset = tf.data.Dataset.range(10).repeat(2)
dataset = dataset.shuffle(buffer_size=4, seed=42).batch(7)
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `Dataset.range(10).repeat(2)` 建立 0-9 的資料並重複兩次。
2. `.shuffle(buffer_size=4, seed=42)` 以 4 個緩衝區大小洗牌。
    - 資料集會維持一個最多 4 個元素的緩衝區，從中隨機抽出元素並補入新的元素，這樣可以在串流讀取資料時提供一定程度的隨機性。
    - `buffer_size` 決定洗牌品質與記憶體使用，越大越隨機但越佔記憶體。
3. `.batch(7)` 批次大小 7。
4. `for item in dataset:` 印出每個批次。

🎯 重點摘要:
- `buffer_size` 影響洗牌品質與記憶體使用。
- 若需高品質洗牌，緩衝區應至少等於資料集大小。
- `seed` 保證重現性。

> 工業級的解決方案：多層次打亂 (Multi-stage Shuffling) 🌪️
在處理幾百 GB 甚至 TB 等級的資料時，我們不能只依賴 shuffle()。實務上的標準作法是：
第一步：打亂檔案清單。將資料分散存成多個檔案（Shards），先打亂這些檔案的讀取順序。
第二步：交錯讀取 (interleave)。同時從多個檔案中讀取資料。這樣可以在不增加太多記憶體使用的情況下，達到更好的隨機性。
第三步：使用適度的 buffer_size。在記憶體允許的範圍內（例如 10,000）進行最後的隨機採樣。

## <a id="csv-preprocess"></a> 🧾 CSV 讀取與前處理

本章講解如何從 California housing dataset 分割資料並寫入 CSV，再利用 `tf.data` 讀取。

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
```

✅ 程式碼逐行解析:
1. `fetch_california_housing()` 下載 California housing 資料。
2. `train_test_split(..., random_state=42)` 先切成訓練與測試。
3. 再次 `train_test_split` 取得驗證集。

🎯 重點摘要:
- 先分割資料再做標準化，可避免資料外洩。
- `reshape(-1, 1)` 將標籤調整為二維矩陣，符合 Keras 輸入格式。

```python
def save_to_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = Path() / "datasets" / "housing"
    housing_dir.mkdir(parents=True, exist_ok=True)
    filename_format = "my_{}_{:02d}.csv"

    filepaths = []
    m = len(data)
    chunks = np.array_split(np.arange(m), n_parts)
    for file_idx, row_indices in enumerate(chunks):
        part_csv = housing_dir / filename_format.format(name_prefix, file_idx)
        filepaths.append(str(part_csv))
        with open(part_csv, "w") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([str(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths

HEADER = "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,MedianHouseValue"
train_filepaths = save_to_csv_files(
    np.c_[X_train, y_train], "train", header=HEADER)
valid_filepaths = save_to_csv_files(
    np.c_[X_valid, y_valid], "valid", header=HEADER)
test_filepaths = save_to_csv_files(
    np.c_[X_test, y_test], "test", header=HEADER)

```

✅ 程式碼逐行解析:
1. `Path() / "datasets" / "housing"` 定義輸出資料夾。
2. `mkdir(..., exist_ok=True)` 確保資料夾存在。
3. `np.array_split(np.arange(m), n_parts)` 將索引切成多份。
4. 用 `with open(...)` 寫入 CSV。
5. 如果有 `header`，先寫入欄位名稱。
6. `",".join([str(col) for col in data[row_idx]])` 將每行資料格式化成 CSV。

🎯 重點摘要:
- 多文件存儲可提高 I/O 並行讀取效能。
- 將資料拆成多個 CSV 有助於在雲端或分散式環境讀取。

```python
dataset = tf.data.Dataset.list_files(filepaths, seed=42)
dataset = dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers)
```

✅ 程式碼逐行解析:
1. `list_files(filepaths, seed=42)` 取得檔案路徑資料集並洗牌。
2. `interleave(...)` 並行讀取多個檔案。
    - `lambda filepath: tf.data.TextLineDataset(filepath).skip(1)` 定義每個檔案的讀取方式，使用 `TextLineDataset` 讀取文本行，並用 `.skip(1)` 跳過 CSV 標頭。
    - `cycle_length=n_readers` 定義同時讀取的檔案數量。

🎯 重點摘要:
- `interleave()` 可以同時讀多個檔案，提高磁碟吞吐量。
- 搭配 `skip(1)` 可避免重複標頭被當成資料。

---

## <a id="keras-with-dataset"></a> 🔧 Keras 與 Dataset 串接

建立完整資料管線後，可直接餵給 Keras 模型訓練。

```python
def csv_reader_dataset(filepaths, n_readers=5, n_read_threads=None,
                       n_parse_threads=5, shuffle_buffer_size=10_000, seed=42,
                       batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=seed)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size).prefetch(1)
```

✅ 程式碼逐行解析:
1. `list_files` 建立檔案列表。
2. `interleave(..., num_parallel_calls=n_read_threads)` 以並行方式讀取文件。
3. `map(preprocess, num_parallel_calls=n_parse_threads)` 進行並行解析與前處理。
4. `shuffle()` 增加資料隨機性。
5. `batch()` 批次化。
6. `prefetch(1)` 提前讀取下一批，減少 CPU/GPU 等待。

🎯 重點摘要:
- `prefetch()` 是訓練效率的關鍵。
- `.map()` 與 `.interleave()` 都可以使用 `num_parallel_calls` 加速。
- `shuffle_buffer_size` 要夠大才能得到更均勻的隨機性。

> 注意：定義 `preprocess()` 函數來解析 CSV 行並轉換成模型輸入格式是必要的，這裡實作 `preprocess()` 如下:

```python
# Determine the number of features for record_defaults
n_features = X_train.shape[1]
n_columns = n_features + 1 # Total columns: features + 1 label
record_defaults = [tf.constant(0.0, dtype=tf.float32)] * n_columns

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

def preprocess(line):
    """Parses a CSV line and normalizes features.
    
    Args:
        line: A string tensor representing a single CSV line.
    
    Returns:
        A tuple of (normalized_features, label) where normalized_features
        are scaled using mean and standard deviation, and label is a scalar tensor.
    """
    # Use tf.io.decode_csv for robust parsing
    columns = tf.io.decode_csv(line, record_defaults=record_defaults)
    features = tf.stack(columns[:-1])  # All but the last column are features
    label = columns[-1]                # The last column is the label
    return (features - X_mean) / X_std, label
```

接著，我們就可以直接把 `tf.data.Dataset` 物件傳給 Keras 的 `fit()` 方法：

```python
train_set = csv_reader_dataset(train_filepaths)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train.shape[1:]),
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(1),
])
model.compile(loss="mse", optimizer="sgd")
model.fit(train_set, validation_data=valid_set, epochs=5)
```

✅ 程式碼逐行解析:
1. `csv_reader_dataset(...)` 分別建立訓練、驗證、測試資料集。
2. `Sequential([...])` 定義簡單全連接模型。
3. `compile(loss="mse", optimizer="sgd")` 設定損失與優化器。
4. `fit(train_set, validation_data=valid_set, epochs=5)` 執行訓練。

🎯 重點摘要:
- 將 `tf.data.Dataset` 作為 `fit()` 輸入是最佳實務。
- 這樣模型可以同時使用資料預處理與批次化。
- 若資料集包含不定長度元素，`batch()` 之後可能需要 `padded_batch()`。

---

## <a id="tfrecord-protobuf"></a> 📦 TFRecord 格式與 Protobuf (Protocol Buffers)

TFRecord 讓你把資料序列化為二進位檔案，適合大規模訓練及跨平台部署。

```python
with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")
```

✅ 程式碼逐行解析:
1. `TFRecordWriter("my_data.tfrecord")` 建立 TFRecord 寫入器。
2. `f.write(...)` 逐條寫入二進位紀錄。

🎯 重點摘要:
- TFRecord 本質上是 bytes 的串列。
- 內容可以是任意二進位資料，但常見於序列化 Protobuf。

```python
filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `TFRecordDataset(filepaths)` 建立讀取器。
2. 逐條列印每個二進位紀錄。

🎯 重點摘要:
- TFRecord 播放與 CSV 讀取方式類似，但更適合二進位序列化資料。
- 可以搭配 `num_parallel_reads` 讀多個 TFRecord 檔案。

### Protobuf 基本介紹

TensorFlow 常用 `tf.train.Example` 來表示結構化樣本，這種格式實際上是 Protobuf 序列化的二進位資料。透過 `tf.train.Feature` 與 `tf.train.Example` 建立樣本後，可寫入 TFRecord；在讀取時，使用 `tf.io.parse_single_example()` 將二進位資料解析回張量。

#### 實作範例：寫入 TFRecord 與 Protobuf

```python
import tensorflow as tf

# 建立對應的 Feature helper (Protobuf Helper Functions)
def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """ Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(num_rooms, median_income, median_house_value):
    """ Creates a tf.train.Example message ready to be written to a file."""
    feature = {
        "num_rooms": _float_feature(num_rooms),
        "median_income": _float_feature(median_income),
        "median_house_value": _float_feature(median_house_value),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


record = serialize_example(6.0, 3.1, 265000.0)
with tf.io.TFRecordWriter("my_data.tfrecord") as writer:
    writer.write(record)
```

✅ 程式碼逐行解析:
1. 定義 `_bytes_feature`, `_float_feature`, `_int64_feature` 等輔助函數，將 Python 值轉換為 `tf.train.Feature` 物件，以適應不同資料型別。
2. `serialize_example()` 函式接收原始資料，將其包裝成 `tf.train.Example` 訊息，並序列化為二進位字串。
3. 建立一個 `tf.io.TFRecordWriter` 物件，用於寫入 TFRecord 檔案。
4. 調用 `writer.write(record)` 將序列化後的 `tf.train.Example` 寫入到 `my_data.tfrecord` 檔案中。

#### 實作範例：讀取 TFRecord 與 Protobuf

當我們需要讀取這些資料時，必須先定義一個 `feature_description` 字典，說明每個欄位的資料型別與形狀，然後使用 `tf.io.parse_single_example()` 來解析 TFRecord 中的二進位資料。

```python
feature_description = {
    "num_rooms": tf.io.FixedLenFeature([], tf.float32),
    "median_income": tf.io.FixedLenFeature([], tf.float32),
    "median_house_value": tf.io.FixedLenFeature([], tf.float32),
}


def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


dataset = tf.data.TFRecordDataset(["my_data.tfrecord"])
parsed_dataset = dataset.map(parse_example)
for parsed_record in parsed_dataset.take(1):
    print(parsed_record)
```

✅ 程式碼逐行解析:
1. `feature_description` 定義每個欄位的資料型別與形狀。
2. `parse_single_example()` 解析 TFRecord bytes 成字典張量。
3. `TFRecordDataset` 搭配 `.map(parse_example)` 讀取與反序列化資料。

🎯 重點摘要:
- `tf.train.Example` 與 `TFRecord` 讓資料序列化成單一二進位檔案，適合大規模訓練與分散式讀取。
- 讀取時需先定義 `feature_description`，才能把 Protobuf bytes 轉回可訓練的張量。

> 實務觀察：在實際應用中，使用 TFRecord 與 Protobuf 可以大幅提升資料讀取效率，尤其是在分散式訓練環境中。建議在資料預處理階段就將資料轉換為 TFRecord 格式，並利用 Protobuf 定義清晰的資料結構，以確保訓練過程中的資料一致性與可重用性。
> 以照片為例，直接將圖片序列化為 TFRecord 中的 bytes 欄位，並在讀取時解析回圖片張量，可以避免在訓練過程中頻繁的磁碟 I/O 操作，提升整體訓練效率。

本節將示範如何從磁碟讀取圖片的原始位元組，將其與對應的整數標籤打包成 `tf.train.Example` 訊息，然後序列化並寫入 TFRecord 檔案。

```python
def create_image_example(image_bytes, label):
    """Creates a tf.train.Example message ready to be written to a file."""
    # 1. Map your data to the helper functions
    feature = {
        'image_raw': _bytes_feature(image_bytes), # Storing the compressed JPEG bytes
        'label': _int64_feature(label),           # Storing the class ID
    }
    
    # 2. Wrap it in a Features message, then an Example message
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Example Writing Loop
tfrecord_filename = 'dataset_shard_0001.tfrec'

# tf.io.TFRecordWriter is the object that actually writes the file to disk
with tf.io.TFRecordWriter(tfrecord_filename) as writer:
    
    # Assuming 'image_paths' is a list of file paths and 'labels' are their classes
    for image_path, label in zip(image_paths, labels):
        
        # PRODUCTION RULE: Read raw bytes! Do not use tf.image.decode_jpeg here.
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            
        # Create the Protobuf
        tf_example = create_image_example(image_bytes, label)
        
        # Serialize to binary string and write to file
        writer.write(tf_example.SerializeToString())

print(f"Successfully wrote {tfrecord_filename}")
```

```python
# 1. Create a dictionary describing the features to parse
feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def parse_and_decode(serialized_example):
    """Parses a single tf.train.Example and decodes the image."""
    # Parse the binary string back into a dictionary
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    
    # NOW we decode the JPEG bytes into a mathematical tensor
    image = tf.io.decode_jpeg(parsed['image_raw'], channels=3)
    
    # Resize and normalize for the neural network
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0 
    
    label = parsed['label']
    return image, label

# 2. Build the Production Pipeline
# Use tf.data.TFRecordDataset to read the binary files
dataset = tf.data.TFRecordDataset([tfrecord_filename])

# Apply the parsing function in parallel using AUTOTUNE
dataset = dataset.map(parse_and_decode, num_parallel_calls=tf.data.AUTOTUNE)

# Standard pipeline operations: shuffle, batch, and crucially, prefetch!
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# The dataset is now ready to be passed directly to model.fit()
# model.fit(dataset, epochs=10)
```

> The High-Performance TFRecord Pipeline (batch->map->prefetch) is the gold standard for training efficiency. By reading raw bytes, we minimize disk I/O overhead during training. The parsing and decoding steps are done in parallel, and prefetching ensures that the GPU always has data ready to process, maximizing throughput.

```python
# Parses a BATCH of 32 examples at once!
def parse_batch(serialized_batch):
    # Notice we use parse_example instead of parse_single_example
    parsed = tf.io.parse_example(serialized_batch, features)
    
    # Mathematical augmentations applied to all 32 images instantly
    images = parsed['image_raw'] / 255.0 
    return images, parsed['label']

dataset = dataset.shuffle(1000)
# We batch the raw string bytes!
dataset = dataset.batch(32) 
# Now we map the batch
dataset = dataset.map(parse_batch, num_parallel_calls=tf.data.AUTOTUNE)
```

#### 終極混合方法：真實世界的資料管線優化策略

在實際生產環境中，處理影像資料集時常常會遇到一個常見的障礙：**影像尺寸不一致**。由於壓縮的 JPEG 影像解析度各異，`dataset.batch()` 操作會因元素形狀不一致而崩潰。這意味著，在將影像批次化之前，我們必須將它們全部調整為統一的尺寸。

為了解決這個問題並最大化訓練吞吐量，業界通常會採用一套經過實戰驗證的「終極資料管線」策略，將 `map` 函數巧妙地拆分成兩個階段。這套管線的步驟及其背後的設計理念如下：

1.  **打亂 (Shuffle)：混洗壓縮後的字串**
    *   **目的**：在讀取資料的早期階段，盡快對資料進行初步的隨機混洗。此時，資料通常以輕量的、壓縮的字串形式存在（例如，TFRecord 中的二進位資料或檔案路徑）。對這些「小巧」的元素進行打亂，可以消耗較少的記憶體資源，並為後續的訓練提供良好的隨機性。
    *   **實作**：使用 `tf.data.Dataset.shuffle(buffer_size)` 進行操作，確保資料在進入解碼階段前已被充分打亂。

2.  **映射 (Map)：解碼與調整大小 (decode_and_resize)**
    *   **目的**：將壓縮的影像字串解碼為原始像素資料，並將其調整為統一的形狀（例如，224x224 像素）。這是批次化操作的先決條件，因為只有所有影像都具有相同尺寸，才能正確地形成批次。這個階段通常是計算密集型的，因此會利用多個平行執行緒來加速處理。
    *   **實作**：使用 `tf.data.Dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)`，讓 TensorFlow 自動調整平行處理的效率。

3.  **批次化 (Batch)：將統一尺寸的影像分組**
    *   **目的**：一旦所有影像都被解碼並調整為相同尺寸，就可以將它們分組為固定大小的批次（例如，每批 32 張影像）。批次化對於高效利用 GPU/TPU 至關重要，因為它們擅長處理大型的矩陣運算。
    *   **實作**：呼叫 `dataset.batch(batch_size)`。

4.  **映射 (Map)：向量化資料增強 (vectorized_augmentations)**
    *   **目的**：在影像被批次化之後，對整個批次的影像同時應用各種數學密集型的資料增強操作。這些操作，例如調整亮度、對比度或進行正規化，可以在 GPU 上以高度並行的方式執行，大幅提升效率。將增強操作放在批次化之後，可以最大程度地利用硬體的向量化處理能力。
    *   **實作**：使用 `dataset.map(vectorized_augmentations, num_parallel_calls=tf.data.AUTOTUNE)`。請確保 `vectorized_augmentations` 函數能夠接受並處理整個影像批次。

5.  **預取 (Prefetch)：為 GPU 準備資料**
    *   **目的**：這是資料管線的最終優化步驟。`prefetch()` 允許資料處理（CPU 密集型）與模型訓練（GPU 密集型）並行執行。當 GPU 正在處理當前批次的資料時，CPU 會在後台預先載入並處理下一批資料，從而減少 GPU 的等待時間，最大化整體訓練吞吐量。
    *   **實作**：呼叫 `dataset.prefetch(buffer_size=tf.data.AUTOTUNE)`。

透過理解並實踐這些操作的執行順序及其背後的邏輯，您將能夠像一位專業的 MLOps 工程師一樣，設計並優化高效能的資料管線，從而實現最大化的模型訓練吞吐量！

---

## <a id="keras-preprocessing"></a> 🧩 Keras 前處理層 (Preprocessing Layers)

Keras 前處理層可以把資料規範化、離散化或文字向量化註入模型架構中，確保訓練與推論的一致性。

```python
tf.random.set_seed(42)  # extra code – ensures reproducibility
norm_layer = tf.keras.layers.Normalization()
model = tf.keras.models.Sequential([
    norm_layer,
    tf.keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))
norm_layer.adapt(X_train)  # computes the mean and variance of every feature
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5)
```

✅ 程式碼逐行解析:
1. `tf.random.set_seed(42)` 設定隨機種子，確保重現性。
2. `Normalization()` 建立輸入規範化層。
3. `Sequential([...])` 將規範化層放在模型最前端。
    - `norm_layer` 會在訓練過程中自動將輸入資料標準化，確保訓練與推論的一致性。
4. `norm_layer.adapt(X_train)` 計算訓練資料的平均值與標準差。
5. `fit(...)` 將標準化層與模型一起訓練。

🎯 重點摘要:
- 將前處理層放在模型內能避免訓練/推論不一致。
- `adapt()` 只需呼叫一次，之後相同層可重複使用。

```python
age = tf.constant([[10.], [93.], [57.], [18.], [37.], [5.]])
discretize_layer = tf.keras.layers.Discretization(bin_boundaries=[18., 50.])
age_categories = discretize_layer(age)
age_categories
```

✅ 程式碼逐行解析:
1. 建立年齡張量。
2. `Discretization(...)` 設定分箱邊界。
    - 年齡小於 18 歲的會被分到第一類（0），介於 18-50 歲的會被分到第二類（1），大於 50 歲的會被分到第三類（2）。
3. 呼叫前處理層得到類別索引。

🎯 重點摘要:
- `Discretization` 適合連續數值特徵離散化。
- 若要建立排行榜、年齡段等特徵工程，這是理想工具。

### 將類別索引轉換為 one-hot 編碼

類別索引通常以整數形式表示，但神經網路傾向於 one-hot 編碼。`CategoryEncoding` 層可將類別索引轉換為稀疏向量表示，使模型能更有效地處理類別特徵。

```python
onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3)
onehot_layer(age_categories)
```

✅ 程式碼逐行解析:
1. `CategoryEncoding(num_tokens=3)` 建立 one-hot/多熱編碼層。
2. `onehot_layer(age_categories)` 轉換類別索引為稀疏表示。

🎯 重點摘要:
- `CategoryEncoding` 可用於 `one_hot`、`multi_hot`、`count` 等模式。
- 需要先經過 `StringLookup` 或離散化層將類別轉成索引。

```python
cities = ["Auckland", "Paris", "Paris", "San Francisco"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(cities)
print(str_lookup_layer.get_vocabulary())
str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])
```

✅ 程式碼逐行解析:
1. `StringLookup()` 建立文字索引層。
2. `adapt(cities)` 根據資料建立詞彙表。
3. `get_vocabulary()` 顯示詞彙表內容，此函數回傳實際分配的詞彙表順序。
    - `[UNK]` 是 OOV token，索引為 0。
    - `Paris` 索引為 1，`San Francisco` 索引為 2，`Auckland` 索引為 3。
4. 呼叫該層轉換字串為整數索引。
    - `["Paris"]` 轉為 1，`["Auckland"]` 轉為 3，`["Montreal"]` 轉為 0（OOV）。

🎯 重點摘要:
- `StringLookup` 支援 `num_oov_indices` 來處理未登錄文字。
- 若要直接輸入文字特徵到模型，先做 `StringLookup` 是最佳做法。

---

## <a id="tfds"></a> 🌐 TensorFlow Datasets (TFDS)

TFDS 提供標準資料集載入器，避免手動處理資料集檔案。

```python
import tensorflow_datasets as tfds

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]
```

✅ 程式碼逐行解析:
1. `tfds.load(name="mnist")` 下載並回傳資料集字典。
2. `datasets["train"]` 與 `datasets["test"]` 取得訓練與測試集。

🎯 重點摘要:
- TFDS 內建大量資料集，節省資料準備時間。
- 可直接取得 `as_supervised=True` 的 `(features, label)` 格式資料。

```python
train_set, valid_set, test_set = tfds.load(
    name="mnist",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True
)
train_set = train_set.shuffle(10_000, seed=42).batch(32).prefetch(1)
```

✅ 程式碼逐行解析:
1. `split=["train[:90%]", "train[90%:]", "test"]` 將資料集切成訓練、驗證、測試。
    - `train[:90%]` 取前 90% 作為訓練集，`train[90%:]` 取後 10% 作為驗證集。
    - `test` 直接使用測試集。
2. `.shuffle(10_000, seed=42)` 進行洗牌。
3. `.batch(32).prefetch(1)` 批次化並提前預取。

🎯 重點摘要:
- TFDS 的分割語法可精準控制資料比例。
- 與 `tf.data` 搭配使用可取得高效輸入管線。

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification.cats_vs_dogs import CatsVsDogs, _NAME_RE
import io, zipfile

def fixed_generate_examples(self, archive):
    num_skipped = 0
    for fname, fobj in archive:
        norm_fname = fname.replace("\\", "/")
        res = _NAME_RE.match(norm_fname)
        if not res:
            continue
        label = res.group(1).lower()
        if tf.compat.as_bytes("JFIF") not in fobj.peek(10):
            num_skipped += 1
            continue
        img_data = fobj.read()
        img_tensor = tf.image.decode_image(img_data)
        img_recoded = tf.io.encode_jpeg(img_tensor)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as new_zip:
            new_zip.writestr(norm_fname, img_recoded.numpy())
        new_fobj = zipfile.ZipFile(buffer).open(norm_fname)
        yield norm_fname, {"image": new_fobj, "image/filename": norm_fname, "label": label}

CatsVsDogs._generate_examples = fixed_generate_examples

# 1. Load the dataset (Automatically fetches/builds TFRecords locally)
# as_supervised=True returns a clean (features, label) tuple instead of a dictionary
# with_info=True gives us metadata (like the exact number of examples for shuffling)
(ds_train, ds_test), ds_info = tfds.load(
    'cats_vs_dogs', 
    split=['train[:80%]', 'train[80%:]'], # Native slice support
    as_supervised=True, 
    with_info=True
)

# 2. Define a batch-aware mapping function
def decode_and_resize(image, label):
    # Remember: Keep complex image processing on the CPU to prevent Training-Serving Skew later!
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# 3. Build the highly-optimized asynchronous pipeline
BATCH_SIZE = 32

ds_train = (ds_train
    .cache() # Cache to RAM or disk if the dataset fits
    .shuffle(buffer_size=ds_info.splits['train'].num_examples) # Full random shuffle
    .map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE) # Multi-threaded processing
    .batch(BATCH_SIZE) # Group for vectorized GPU math
    .prefetch(tf.data.AUTOTUNE) # Overlap CPU and GPU work
)
```

✅ 程式碼逐行解析:
1. `tfds.load(...)` 下載並分割 `cats_vs_dogs`
    - `split=['train[:80%]', 'train[80%:]']` 直接在載入階段切分訓練與測試集。
    - `as_supervised=True` 以 `(features, label)` 格式回傳資料。
    - `with_info=True` 同時回傳資料集的元資訊，包含樣本數量等。
2. 定義 `decode_and_resize` 函數，將影像調整為 224x224 並標準化。
3. 建立資料管線：
    - `.cache()` 將資料集緩存在記憶體或磁碟中（如果資料集足夠小）。
    - `.shuffle(buffer_size=ds_info.splits['train'].num_examples)` 進行完全隨機洗牌。
    - `.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)` 多線程處理影像解碼與調整大小。
    - `.batch(BATCH_SIZE)` 批次化資料。
    - `.prefetch(tf.data.AUTOTUNE)` 讓 CPU 與 GPU 工作重疊，提升效率。
    
---

## <a id="summary"></a> ✅ 章節重點與實務建議

- `tf.data` 管線是建立可重用資料處理流程的基礎，建議先從 `list_files()`、`interleave()`、`map()`、`batch()`、`prefetch()` 熟悉起。
- 若資料集無法全部放進記憶體，優先使用檔案格式（CSV、TFRecord）搭配 `TextLineDataset` 或 `TFRecordDataset`。
- Keras 前處理層可提高模型可移植性，尤其在部署時可避免訓練與推論資料不一致。
- `TFRecord + Protobuf` 是大規模訓練與分散式資料處理的首選，但需要額外的序列化與解析邏輯。

---

## <a id="faq"></a> ❓ 常見問答

**Q: 為什麼要用 `prefetch()`？**  
A: `prefetch()` 可以讓資料讀取與模型訓練併行，避免 GPU/TPU 等待資料，提升整體吞吐量。

**Q: CSV 與 TFRecord 哪個比較好？**  
A: 若只是小型資料集，CSV 方便觀察與除錯；若是大規模資料、二進位資料或分散式訓練，TFRecord 與 Protobuf 更穩定。

**Q: `StringLookup` 可以直接當第一層嗎？**  
A: 在某些舊版 Keras 有相容性問題，建議加 `InputLayer(input_shape=[], dtype=tf.string)` 作為第一層做為最佳實務。

**Q: 我該在哪裡做標準化？**  
A: 若希望模型可直接部署，最好把標準化層內置到模型中；若需要更高效訓練，可以在 `tf.data` 管線中先做好數值標準化。

## 🏷️ 建議標籤
#Python #TensorFlow #資料預處理 #機器學習 #教學 #tfdata #TFRecord #Keras #開發者分享