# Ch19 速查表：Training & Deploying at Scale

> **核心主旨**：規模化訓練與生產部署 —— `MirroredStrategy` 多 GPU，TF Serving 提供推論 API，TFLite 縮小模型上邊緣裝置。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| SavedModel | TF 的標準部署格式，包含計算圖與權重 | 所有部署場景 |
| TensorFlow Serving | 為 SavedModel 提供 REST / gRPC API | 伺服器端推論 |
| TFLite | 輕量模型格式，支援量化，用於行動/嵌入式裝置 | IoT、行動端 |
| TensorFlow.js | 在瀏覽器或 Node.js 運行模型 | 前端 ML 應用 |
| MirroredStrategy | 單機多 GPU：每個 GPU 複製一份模型，梯度 all-reduce | 單機多 GPU 訓練 |
| MultiWorkerMirroredStrategy | 多台機器多 GPU 分散式訓練 | 大規模分散式訓練 |
| Vertex AI | Google Cloud 的 ML 平台：訓練、超參數調整、部署 | 雲端 ML 工作流程 |
| Keras Tuner | 自動超參數搜索（Random, Bayesian, Hyperband） | 超參數優化 |
| Quantization | 將 float32 權重轉為 int8，大幅縮小模型 | TFLite 部署前優化 |

---

## 2. 關鍵 API 速查

| TF / Cloud API | 重點參數 | 用途 |
|----------------|---------|------|
| `model.save("model.keras")` | – | 儲存 Keras 格式 |
| `model.save("my_model")` | – | 儲存 SavedModel 格式（目錄） |
| `tf.keras.models.load_model("model.keras")` | – | 載入模型 |
| `tf.distribute.MirroredStrategy()` | – | 單機多 GPU 策略 |
| `tf.distribute.MultiWorkerMirroredStrategy()` | – | 多機多 GPU 策略 |
| `tf.lite.TFLiteConverter.from_keras_model(model)` | – | 轉換為 TFLite |
| `converter.optimizations = [tf.lite.Optimize.DEFAULT]` | – | 啟用 quantization |
| `converter.convert()` | – | 執行轉換 |
| `keras_tuner.RandomSearch` | `hypermodel=`, `objective=`, `max_trials=` | 隨機超參數搜索 |
| `keras_tuner.Hyperband` | `factor=3`, `max_epochs=` | Hyperband 算法（更快） |
| `keras_tuner.BayesianOptimization` | `max_trials=20` | 貝葉斯優化 |

---

## 3. 必備代碼片段

```python
import tensorflow as tf

# -------- 模型儲存與載入 --------
model.save("my_model.keras")  # 推薦格式
loaded = tf.keras.models.load_model("my_model.keras")

# 只儲存權重（快速 checkpoint）
model.save_weights("my_weights.weights.h5")
model.load_weights("my_weights.weights.h5")

# -------- 多 GPU 訓練（MirroredStrategy）--------
strategy = tf.distribute.MirroredStrategy()
print(f"使用 GPU 數量: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu", input_shape=[28*28]),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

# model.fit() 自動分散到各 GPU
# batch_size 應乘以 GPU 數量
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
model.fit(X_train, y_train, batch_size=GLOBAL_BATCH_SIZE, epochs=10)

# -------- TFLite 轉換（行動端部署）--------
# 標準轉換
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# 動態範圍量化（降低模型大小 ~4x，略微降低精度）
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite = converter.convert()

# TFLite 推論
interpreter = tf.lite.Interpreter(model_content=quantized_tflite)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], X_test[:1])
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

# -------- Keras Tuner 超參數搜索 --------
import keras_tuner as kt

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=1, max_value=8)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256, step=16)
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Hyperband（自適應資源分配，最常用）
tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="my_tuner",
    project_name="fashion_mnist"
)
tuner.search(X_train, y_train, epochs=10, validation_split=0.1)
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)

# -------- TF Serving（Docker 方式）--------
# 1. 儲存 SavedModel
tf.saved_model.save(model, "my_model/1")  # 版本號 = 1

# 2. 啟動 TF Serving（在 terminal 執行）
# docker run -p 8501:8501 -v "$(pwd)/my_model:/models/my_model" \
#   -e MODEL_NAME=my_model tensorflow/serving

# 3. 發送 REST 請求
import requests, json
import numpy as np

X_sample = X_test[:3].tolist()
response = requests.post(
    "http://localhost:8501/v1/models/my_model:predict",
    data=json.dumps({"instances": X_sample})
)
predictions = response.json()["predictions"]
```

---

## 4. 常見陷阱

- **`MirroredStrategy` 必須在 `strategy.scope()` 內建立模型**：scope 外建立的模型不會被分散，白費工夫。
- **batch_size 要隨 GPU 數量等比放大**：每個 GPU 分到 `batch_size / n_gpu` 個樣本，等效 batch size 不變，學習率也無需調整。
- **TFLite 量化後要驗證精度**：Dynamic Range Quantization 通常精度損失 < 1%，但某些任務影響較大，建議用測試集驗證。
- **SavedModel 版本目錄**：TF Serving 期望路徑格式為 `model_name/版本號/saved_model.pb`，版本號必須是整數目錄名稱。
- **Keras Tuner 的 `directory`**：每次搜索結果存在這裡，同 `project_name` 的搜索會繼續上次的進度。

---

## 5. 決策指南

```
部署目標 vs 策略：
├── 伺服器端（REST API）    → SavedModel + TF Serving 或 FastAPI
├── 行動端 / 邊緣裝置       → TFLite（+ 量化壓縮）
├── 瀏覽器（JavaScript）    → TensorFlow.js
└── Google Cloud            → Vertex AI Prediction

訓練規模 vs 分散策略：
├── 單機單 GPU              → 直接 model.fit()
├── 單機多 GPU（最常見）    → MirroredStrategy
├── 多機多 GPU（大型訓練）  → MultiWorkerMirroredStrategy
└── 超大模型（模型並行）    → tf.distribute.TPUStrategy（Cloud TPU）

超參數搜索算法比較：
├── 小資源 (< 30 trials)    → RandomSearch（簡單）
├── 中等資源                → Hyperband（最有效率）
└── 大量資源 (> 50 trials)  → BayesianOptimization（樣本效率最高）
```
