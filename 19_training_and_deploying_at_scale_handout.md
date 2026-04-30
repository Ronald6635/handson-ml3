# 課程講義：大規模訓練與部署 (Chapter 19)

模型訓練完畢只是起點，真正的挑戰在於**把模型送到用戶手中**。本章涵蓋模型的完整生命週期：從儲存格式、API 服務、行動/邊緣部署、瀏覽器端執行，到多 GPU 分散式訓練和超參數自動調整。這些技術是工業界 MLOps 工程師的日常必備。

---

## 1. 模型儲存格式與版本管理

### 理論背景

Keras 支援兩種主要儲存格式：

| 格式 | 副檔名 | 說明 | 建議使用場景 |
|------|--------|------|------------|
| **Keras v3** | `.keras` | 新格式（Keras 3+），跨後端 | 一般情況首選 |
| **SavedModel** | 目錄 | TensorFlow 原生格式，包含計算圖 | 部署到 TF Serving、TFLite、TF.js |
| HDF5（舊版） | `.h5` | Keras 舊格式 | 向下相容 |

**SavedModel 的內容**：

```
saved_model/
├── saved_model.pb          # 計算圖定義
├── variables/              # 訓練好的權重
│   ├── variables.index
│   └── variables.data-00000-of-00001
└── assets/                 # 輔助資料（如詞彙表）
```

**自訂物件的儲存**：若模型含自訂層、損失函數，需用 `@tf.keras.utils.register_keras_serializable()` 裝飾器，或在載入時傳入 `custom_objects` 字典。

### 核心代碼

```python
import tensorflow as tf
import numpy as np

# 訓練一個範例模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", input_shape=[8]),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="sgd", loss="mse")
# model.fit(X_train, y_train, ...)

# 儲存為 Keras 格式
model.save("my_model.keras")
model_loaded = tf.keras.models.load_model("my_model.keras")

# 儲存為 SavedModel 格式（用於部署）
model.export("my_saved_model")  # Keras 3 方式
# 或：tf.saved_model.save(model, "my_saved_model")

# 載入 SavedModel
infer = tf.saved_model.load("my_saved_model")

# 只儲存/載入權重（常用於遷移學習）
model.save_weights("my_weights.weights.h5")
model.load_weights("my_weights.weights.h5")

# Callbacks 自動儲存最佳模型
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    save_best_only=True,
    monitor="val_loss"
)
```

### ⚡ 補充練習 1

**理論題：** `.keras` 格式和 SavedModel 格式各有什麼優缺點？若模型需要部署到 TF Serving 的 Docker 容器，應使用哪種格式？若需要讓前端工程師在瀏覽器中執行，又應使用哪種？

**實作題：** 訓練一個含有自訂層的模型（如自訂 `Standardization` 層），分別儲存為 `.keras` 和 `SavedModel` 格式，重新載入後確認預測結果與原始模型完全一致。

---

## 2. TF Serving：生產環境的模型服務

### 理論背景

**TF Serving**：TensorFlow 官方提供的生產級模型服務器，支援：

- **REST API**：標準 HTTP 端點，使用 JSON（易於整合）
- **gRPC API**：二進制協議，速度更快（適合低延遲場景）
- **版本管理**：同時服務多個模型版本，支援灰度發布
- **動態批次 (Dynamic Batching)**：自動將多個請求合併為一個 batch，提升 GPU 利用率

**部署流程**：

1. 將 SavedModel 存至標準目錄結構（`model_name/version_number/`）
2. 啟動 TF Serving Docker 容器
3. 送出 HTTP POST 請求

```bash
# 啟動 TF Serving
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/models,target=/models/my_model \
  -e MODEL_NAME=my_model \
  -t tensorflow/serving
```

**雲端部署**：

- **Vertex AI**（Google Cloud）：托管服務，自動擴縮容
- **SageMaker**（AWS）：Amazon 的機器學習平台
- **Azure ML**：微軟的 MLOps 平台

### 核心代碼

```python
import requests
import json
import numpy as np

# 儲存模型到 TF Serving 期待的目錄結構
import os
model_version = 1
model_name = "my_california_housing_model"
model_path = os.path.join(model_name, str(model_version))
model.export(model_path)

# 呼叫 TF Serving REST API
SERVER_URL = "http://localhost:8501/v1/models/my_california_housing_model:predict"

X_new = np.random.rand(3, 8).tolist()  # 3 筆樣本，8 個特徵
request_data = json.dumps({"instances": X_new})

response = requests.post(SERVER_URL,
                         data=request_data,
                         headers={"content-type": "application/json"})
if response.status_code == 200:
    predictions = json.loads(response.text)["predictions"]
    print(f"預測結果: {predictions}")
else:
    print(f"錯誤: {response.status_code}, {response.text}")
```

### ⚡ 補充練習 2

**理論題：** TF Serving 的「動態批次」如何在服務延遲和吞吐量之間取捨？若設定 `max_batch_delay_millis=5`，意味著服務最多等待 5ms 才將請求合批，這適合哪些應用場景？哪些場景不適合（如自動駕駛實時決策）？

**實作題：** 設計一個完整的部署流程文件（不需要實際部署）：(1) 模型訓練後儲存為 SavedModel；(2) 建立 Docker Compose 配置啟動 TF Serving；(3) 建立 Python 客戶端函式，呼叫 REST API 並解析返回結果。

---

## 3. TFLite：邊緣與行動端部署

### 理論背景

**TFLite (TensorFlow Lite)**：針對行動裝置（iOS、Android）和嵌入式設備（Raspberry Pi）優化的推論框架。

**模型量化 (Quantization)**：將模型的浮點數（float32）參數轉換為低精度（int8 或 float16）：

| 量化類型 | 精度 | 模型大小 | 速度 | 準確率影響 |
|--------|------|---------|------|-----------|
| Dynamic Range | int8（權重）| 縮小 4x | 快 2-3x | 輕微 |
| Full Integer | int8（全部）| 縮小 4x | 最快 | 輕微-中等 |
| Float16 | float16 | 縮小 2x | GPU 加速 | 極輕微 |

**量化感知訓練 (QAT)**：在訓練時模擬量化誤差，讓模型適應低精度，減少準確率下降。

**TF.js**：讓模型在瀏覽器的 JavaScript 環境中運行，支援 WebGL 加速（GPU）。

### 核心代碼

```python
import tensorflow as tf

# 將 SavedModel 轉換為 TFLite（含量化）
converter = tf.lite.TFLiteConverter.from_saved_model("my_saved_model")

# 動態範圍量化（最簡單，推薦先試）
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 儲存
with open("my_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"模型大小: {len(tflite_model) / 1024:.1f} KB")

# 用 TFLite Interpreter 執行推論
interpreter = tf.lite.Interpreter(model_path="my_model.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

X_test_sample = np.array([X_test[0]], dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], X_test_sample)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]["index"])
print(f"TFLite 預測: {prediction}")
```

### ⚡ 補充練習 3

**理論題：** 量化將 float32 的參數轉換為 int8，數值精度從 7 位有效數字降至約 2-3 位。為什麼神經網路對這種大幅精度損失有如此強的魯棒性（準確率下降通常不超過 1%）？

**實作題：** 將一個在 MNIST 上訓練的模型（準確率 > 98%）分別轉換為原始 TFLite（無量化）、動態範圍量化、int8 全量化，比較三者的：(1) 模型檔案大小；(2) 推論時間（1000 次預測的平均）；(3) 測試集準確率。

---

## 4. 分散式訓練策略

### 理論背景

**為何需要分散式訓練？**

- 模型太大，單個 GPU 記憶體不夠
- 訓練資料龐大，單個 GPU 速度太慢
- 縮短訓練時間（從數天縮短到數小時）

**Keras 分散式訓練策略 (`tf.distribute.Strategy`)**：

| 策略 | 適用場景 | 說明 |
|------|---------|------|
| `MirroredStrategy` | 單機多 GPU | 每個 GPU 保存完整模型副本，同步更新 |
| `MultiWorkerMirroredStrategy` | 多機多 GPU | 跨機器同步分散式訓練 |
| `TPUStrategy` | Google TPU | TPU Pod 上的分散式訓練 |
| `ParameterServerStrategy` | 超大規模 | 非同步訓練，適合異質硬體 |

**`MirroredStrategy` 的同步機制**：

1. 每個 GPU 計算自己的梯度
2. **AllReduce** 操作：彙整所有 GPU 的梯度（求平均）
3. 所有 GPU 用相同的梯度更新各自的模型副本（保持同步）

**有效批次大小**：$\text{Batch Size}_{\text{effective}} = \text{Batch Size}_{\text{per GPU}} \times \text{GPU 數量}$

通常需要相應調高學習率（Linear Scaling Rule：學習率 × GPU 數量）。

### 核心代碼

```python
import tensorflow as tf

# 單機多 GPU 訓練
strategy = tf.distribute.MirroredStrategy()
print(f"使用 {strategy.num_replicas_in_sync} 個 GPU")

# 在 Strategy 的 scope 內建立模型
with strategy.scope():
    model_dist = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu", input_shape=[8]),
        tf.keras.layers.Dense(1)
    ])
    model_dist.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3 * strategy.num_replicas_in_sync),
        loss="mse",
        metrics=["mae"]
    )

# 訓練時和一般模型完全相同（Strategy 自動處理分散式）
batch_size_per_gpu = 32
total_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync

# model_dist.fit(X_train, y_train, batch_size=total_batch_size, epochs=10)

# 多機多 Worker
import json, os
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["worker1:12345", "worker2:12345"]
    },
    "task": {"type": "worker", "index": 0}  # 每台機器設不同 index
})
strategy_multi = tf.distribute.MultiWorkerMirroredStrategy()
```

### ⚡ 補充練習 4

**理論題：** 在 `MirroredStrategy` 中，若有 4 個 GPU，每個 GPU 使用 batch_size=32，有效批次大小是多少？為什麼需要相應提高學習率？這個「Linear Scaling Rule」在批次大小超大時為何開始失效？

**實作題：** 用 `tf.distribute.MirroredStrategy` 重新訓練 MNIST CNN 模型（即使只有 1 個 GPU，`MirroredStrategy` 仍可測試 API），確認在 `strategy.scope()` 內外建立模型的行為差異，測量訓練時間是否有提升。

---

## 5. Keras Tuner：自動超參數搜索

### 理論背景

**手動調參的問題**：超參數空間巨大，人工試誤耗時且不系統。

**主要超參數搜索策略**：

| 策略 | 說明 | 優點 | 缺點 |
|------|------|------|------|
| `RandomSearch` | 隨機取樣 | 簡單、不會困於局部最優 | 效率低 |
| `Hyperband` | 早停式 Random Search | 快速剔除差的配置 | 需要可以早停的任務 |
| `BayesianOptimization` | 用代理模型引導搜索 | 樣本效率高 | 初期需要預熱 |
| `GridSearch` | 窮舉網格 | 全面 | 指數級計算量 |

**Keras Tuner 工作流程**：

1. 定義超參數空間（`hp.Int`, `hp.Float`, `hp.Choice`）
2. 建立包含超參數的模型構建函式
3. 執行搜索（自動訓練多個模型配置）
4. 取出最佳超參數，重新訓練最終模型

### 核心代碼

```python
import keras_tuner as kt

# 定義帶超參數的模型構建函式
def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=1, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256, step=16)
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer_type = hp.Choice("optimizer", ["sgd", "adam"])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[8]))
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.get(optimizer_type)
    optimizer.learning_rate = learning_rate
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model

# Hyperband 搜索（速度最快）
tuner = kt.Hyperband(
    build_model,
    objective="val_mae",
    max_epochs=30,
    factor=3,
    hyperband_iterations=2,
    overwrite=True,
    directory="my_kt_results",
    project_name="california_housing_kt"
)

# 執行搜索
tuner.search(X_train, y_train,
             validation_data=(X_valid, y_valid),
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

# 取出最佳超參數
best_params = tuner.get_best_hyperparameters()[0]
print(f"最佳 n_hidden: {best_params['n_hidden']}")
print(f"最佳 n_neurons: {best_params['n_neurons']}")
print(f"最佳 lr: {best_params['lr']:.4f}")

# 重新訓練最終模型
best_model = tuner.hypermodel.build(best_params)
# best_model.fit(X_train_full, y_train_full, epochs=100)
```

### ⚡ 補充練習 5

**理論題：** Hyperband 演算法的「早停」策略為何能大幅減少計算量？它的核心思想是什麼（類比體育賽事的「淘汰制」和「循環賽」）？

**實作題：** 用 `keras_tuner.BayesianOptimization` 在 California Housing 資料集上搜索最佳的神經網路架構（層數 1-5，每層神經元數 16-256，學習率 1e-4 到 1e-2），運行 20 次試驗，繪製每次試驗的驗證 MAE，觀察 Bayesian 搜索是否逐漸收斂到更好的區域。

---

## 結論

工業級機器學習部署的完整工具鏈：

- **模型儲存**：`.keras`（通用）或 SavedModel（部署）
- **TF Serving**：REST/gRPC API，動態批次，多版本管理
- **TFLite**：邊緣設備，量化壓縮（模型縮小 4x，速度提升 2-3x）
- **TF.js**：瀏覽器端推論，零安裝用戶體驗
- **分散式訓練**：`MirroredStrategy`（單機多 GPU）→ `MultiWorkerMirroredStrategy`（多機）
- **Keras Tuner**：Hyperband / Bayesian 自動超參數搜索

恭喜你完成本書！從 Ch01 的 Hello World 到 Ch19 的生產部署，你已建立了完整的機器學習知識體系。下一步：選擇一個真實問題，端到端地實踐！

---

## 課後作業

**作業：端到端生產部署模擬**

1. 在 California Housing 資料集上，用 `keras_tuner.Hyperband` 找到最佳神經網路架構，重新訓練最終模型，確認測試集 MAE < 0.4（以 $100k 為單位，即 MAE < $40,000）。

2. 將模型儲存為 SavedModel 格式，用 `tf.lite.TFLiteConverter` 轉換為 int8 量化的 TFLite 模型，比較轉換前後的模型大小和 MAE。

3. 撰寫一個模擬 TF Serving 的 Python 函式 `serve_prediction(X)`，接受 numpy array 輸入，載入 SavedModel 並返回預測結果（不需要真的啟動 Docker，用 `tf.saved_model.load` 模擬）。
