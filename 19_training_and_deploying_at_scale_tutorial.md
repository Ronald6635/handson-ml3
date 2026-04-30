<!-- meta-title: TensorFlow 大規模訓練與部署完整指南：TF Serving、TFLite、分散式訓練 -->
<!-- meta-description: 深入 TensorFlow 生產環境：SavedModel/Keras 格式儲存、TF Serving REST/gRPC 部署、TFLite 量化、TF.js 瀏覽器推論、MirroredStrategy 多 GPU 訓練、MultiWorkerMirroredStrategy 多機訓練、Keras Tuner 超參數搜索。 -->
<!-- meta-keywords: Python, TensorFlow, TF Serving, TFLite, 分散式訓練, MirroredStrategy, Keras Tuner, 模型部署, 量化, TF.js -->
<!-- meta-hashtags: #Python #TensorFlow #TFServing #TFLite #分散式訓練 #ModelDeploy #KerasTuner #量化 #深度學習 #教學 -->

# 🐍 TensorFlow 大規模訓練與部署：從實驗到生產環境

「模型在 notebook 裡跑得很好，怎麼部署？」這是每個 ML 工程師都會面對的問題。本教學帶你從模型儲存格式、TF Serving API 部署、TFLite 行動端優化，到多 GPU 分散式訓練——完整覆蓋將 TF 模型推向生產的所有關鍵技術。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [💾 模型儲存格式](#saving-formats)
- [🚀 TF Serving 部署](#tf-serving)
- [📱 TFLite 行動端部署](#tflite)
- [🌐 TF.js 瀏覽器部署](#tfjs)
- [⚡ 多 GPU 分散式訓練](#distributed)
- [🔍 Keras Tuner 超參數搜索](#keras-tuner)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **SavedModel** 格式是 TF 生態系統的標準格式：保存計算圖 + 權重，跨語言可用
- **TF Serving** 用 Docker 一行部署 REST/gRPC API，生產效能接近原生 TF
- **TFLite 量化（INT8）** 可以將模型大小減小 4 倍、推論速度提升 2-3 倍，適合行動裝置
- **`MirroredStrategy`** 是單機多 GPU 的標準方案，程式碼改動極小（幾乎只需 2 行）
- **Keras Tuner** 的 `RandomSearch`/`BayesianOptimization` 自動搜索最優超參數

---

## <a id="saving-formats"></a>💾 模型儲存格式

💡 **實際應用情境：** 訓練好的模型需要在不同環境中使用：Python 推論服務、Android App、瀏覽器——每種環境需要不同的格式。

### 範例 1: 各種儲存格式比較

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

# 建立示範模型
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(8,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 格式 1：SavedModel（推薦，生產首選）
# 儲存為目錄（包含計算圖 + 權重 + 簽名）
model.save("my_model_saved")         # 目錄格式
# └── saved_model.pb               ← 計算圖
# └── variables/                   ← 權重
# └── assets/                      ← 額外資源

# 格式 2：.keras 格式（Keras 3 推薦）
model.save("my_model.keras")         # 單一 ZIP 文件

# 格式 3：只儲存架構（不含權重）
model_json = model.to_json()
# model_from_json = keras.models.model_from_json(model_json)

# 格式 4：只儲存權重
model.save_weights("my_weights.weights.h5")

# 格式 5：HDF5（舊格式，仍廣泛使用）
model.save("my_model.h5")            # 需要 h5py

# 載入模型
loaded_model = keras.models.load_model("my_model.keras")

# 驗證載入後的模型等效
X_test = np.random.randn(100, 8).astype(np.float32)
y_original = model.predict(X_test, verbose=0)
y_loaded   = loaded_model.predict(X_test, verbose=0)
print(f"最大預測差異: {np.max(np.abs(y_original - y_loaded)):.2e}")  # ≈ 0
```

**✅ 程式碼逐行解析：**

1. `SavedModel` 格式的優點：儲存完整計算圖，可以用 TF Serving/TFLite 轉換，跨語言（Python/Java/C++）
2. `.keras` 格式：Keras 3 的新格式，更緊湊，支援自訂物件序列化
3. 只儲存權重：最輕量，但需要事先知道模型架構才能載入

---

## <a id="tf-serving"></a>🚀 TF Serving 部署

### 範例 2: 儲存 Serving 格式 + API 客戶端

```python
# 儲存為 TF Serving 可識別的 versioned 目錄結構
# /models/my_model/1/  ← 版本 1
# /models/my_model/2/  ← 版本 2

import os

model_version = 1
serving_path = f"./tf_serving_models/my_model/{model_version}"
model.save(serving_path)
print(f"TF Serving 模型已儲存至: {serving_path}")

# 查看模型簽名（API 規格）
loaded_tf = tf.saved_model.load(serving_path)
print("可用簽名:", list(loaded_tf.signatures.keys()))
# 通常是 ['serving_default']

infer = loaded_tf.signatures["serving_default"]
print("輸入規格:", infer.structured_input_signature)
print("輸出規格:", infer.structured_outputs)

# ── 部署 TF Serving（用 Docker）──
# docker run -p 8501:8501 \
#   -v "$(pwd)/tf_serving_models:/models" \
#   -e MODEL_NAME=my_model \
#   tensorflow/serving

# ── REST API 客戶端 ──
import json

def predict_via_rest_api(data: np.ndarray, server_url: str = "http://localhost:8501") -> np.ndarray:
    """通過 TF Serving REST API 進行推論"""
    import urllib.request

    payload = json.dumps({
        "instances": data.tolist()
    })

    request = urllib.request.Request(
        f"{server_url}/v1/models/my_model:predict",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())
            return np.array(result["predictions"])
    except Exception as e:
        print(f"API 呼叫失敗（服務未啟動）: {e}")
        return None

# 使用 REST API
# predictions = predict_via_rest_api(X_test[:5])
print("\nTF Serving REST API URL: POST /v1/models/{model_name}:predict")
print("TF Serving gRPC Port: 8500（比 REST 更快）")
```

**🎯 重點摘要:**

- TF Serving 支援**版本管理**：可以同時維護多個版本，並進行 A/B 測試
- REST API（8501 port）適合一般應用；gRPC（8500 port）延遲更低，適合高性能場景

---

## <a id="tflite"></a>📱 TFLite 行動端部署

💡 **實際應用情境：** 台灣某醫療 App 需要在使用者手機上離線執行皮膚分析——TFLite INT8 量化後，模型從 50MB 縮小到 12MB，推論速度提升 2.5 倍，同時準確率僅下降 0.3%。

### 範例 3: TFLite 轉換與量化

```python
# ── TFLite 轉換（不量化）──
converter = tf.lite.TFLiteConverter.from_saved_model(serving_path)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"TFLite 模型大小: {len(tflite_model) / 1024:.1f} KB")

# ── TFLite 動態範圍量化（INT8，模型大小 ~4x 縮小）──
converter_quant = tf.lite.TFLiteConverter.from_saved_model(serving_path)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]  # 啟用量化
tflite_quant_model = converter_quant.convert()

with open("model_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)
print(f"量化後大小: {len(tflite_quant_model) / 1024:.1f} KB")

# ── 全整數量化（需要校準資料）──
def representative_dataset():
    """提供少量代表性資料用於量化校準"""
    cal_data = np.random.randn(100, 8).astype(np.float32)
    for sample in cal_data:
        yield [sample[np.newaxis]]

converter_int8 = tf.lite.TFLiteConverter.from_saved_model(serving_path)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
# 強制輸入/輸出也是 INT8（更快的嵌入式推論）
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type  = tf.int8   # 注意：需要應用端轉換資料
converter_int8.inference_output_type = tf.int8
tflite_int8_model = converter_int8.convert()
print(f"INT8 量化後大小: {len(tflite_int8_model) / 1024:.1f} KB")

# ── TFLite 推論（Python 模擬行動端）──
def run_tflite_inference(tflite_model_bytes: bytes,
                          input_data: np.ndarray) -> np.ndarray:
    """在 Python 中模擬 TFLite 推論"""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    results = []
    for i in range(len(input_data)):
        interpreter.set_tensor(
            input_details[0]['index'],
            input_data[i:i+1]
        )
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        results.append(output[0])

    return np.array(results)

tflite_preds = run_tflite_inference(tflite_model, X_test[:10])
print(f"TFLite 推論完成，輸出形狀: {tflite_preds.shape}")
```

**✅ 程式碼逐行解析：**

1. `Optimize.DEFAULT`: 動態範圍量化——將 float32 權重量化為 int8，運行時再反量化（零代價優化）
2. `representative_dataset`: 全整數量化需要 100~200 個代表性樣本來校準量化閾值
3. `tf.lite.Interpreter.invoke()`: 執行推論（需要手動迴圈，每次只能推論一個樣本或 batch）

---

## <a id="tfjs"></a>🌐 TF.js 瀏覽器部署

### 範例 4: 轉換為 TF.js 格式

```python
# 安裝：pip install tensorflowjs
# tensorflowjs_converter --input_format=keras my_model.keras tfjs_model/

# 或在 Python 中轉換
import subprocess

# 方法 1：命令列工具
# subprocess.run([
#     "tensorflowjs_converter",
#     "--input_format=tf_saved_model",
#     "--output_format=tfjs_graph_model",
#     serving_path,
#     "./tfjs_model"
# ])

# ── JavaScript 推論代碼（在 HTML/Node.js 中使用）──
tfjs_inference_code = """
// 載入模型
const model = await tf.loadGraphModel('/tfjs_model/model.json');

// 準備輸入
const inputData = tf.tensor2d([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);

// 推論
const prediction = model.predict(inputData);
const result = await prediction.data();
console.log('預測結果:', result);
"""

print("TF.js 模型可以在瀏覽器中零後端執行推論（用戶資料不離開裝置）！")
print("\nJavaScript 推論代碼:")
print(tfjs_inference_code)
```

---

## <a id="distributed"></a>⚡ 多 GPU 分散式訓練

💡 **實際應用情境：** 大型圖像分類模型在單 GPU 上需要 72 小時；用 4 GPU + `MirroredStrategy` 縮短到 ~18 小時，且程式碼改動極小。

### 範例 5: MirroredStrategy 多 GPU 訓練

```python
# ── 單機多 GPU（MirroredStrategy）──
strategy = tf.distribute.MirroredStrategy()
# 自動偵測可用的 GPU 並在所有 GPU 間同步梯度

print(f"可用 GPU 數量: {strategy.num_replicas_in_sync}")

with strategy.scope():
    # 在 strategy.scope() 內建立的所有變數會自動跨 GPU 複製
    distributed_model = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(8,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    distributed_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

# 分散式訓練時，建議調整 batch_size = 原始 batch_size × GPU 數
# 例如單 GPU 用 32，4 個 GPU 則用 32 × 4 = 128
n_gpus = max(1, strategy.num_replicas_in_sync)
batch_size = 32 * n_gpus
print(f"分散式 batch_size: {batch_size}")

# ── 多機多 GPU（MultiWorkerMirroredStrategy）──
# 環境變數 TF_CONFIG 需要在每台機器上設置
tf_config_example = {
    "cluster": {
        "worker": ["worker0.example.com:2222",
                   "worker1.example.com:2222"]
    },
    "task": {"type": "worker", "index": 0}  # 當前機器的角色和索引
}
# import os
# os.environ["TF_CONFIG"] = json.dumps(tf_config_example)

# multi_worker_strategy = tf.distribute.MultiWorkerMirroredStrategy()
print("MultiWorkerMirroredStrategy 在多台機器間使用 AllReduce 同步梯度")

# ── TPU 訓練（Google Colab 免費 TPU）──
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# tpu_strategy = tf.distribute.TPUStrategy(resolver)
print("TPU 訓練：在 Colab 中免費使用，速度是 GPU 的 5-10 倍！")
```

**✅ 程式碼逐行解析：**

1. `with strategy.scope()`: 作用域內建立的層/變數自動「鏡射」到所有 GPU
2. MirroredStrategy 使用 **AllReduce** 算法（NCCL）在 GPU 間高效同步梯度
3. `num_replicas_in_sync`: 返回實際可用的 GPU 數，用來計算分散式 batch size

---

## <a id="keras-tuner"></a>🔍 Keras Tuner 超參數搜索

### 範例 6: 自動超參數搜索

```python
# 安裝：pip install keras-tuner
import keras_tuner as kt

def build_model_for_tuning(hp: kt.HyperParameters) -> keras.Model:
    """定義超參數搜索空間和模型建構邏輯"""
    model = keras.Sequential()

    # 搜索隱藏層數量（1~3 層）
    for i in range(hp.Int("n_hidden_layers", min_value=1, max_value=3)):
        # 搜索每層神經元數量（32~512，步長 32）
        units = hp.Int(f"units_{i}", min_value=32, max_value=512, step=32)
        # 搜索激活函數
        activation = hp.Choice(f"activation_{i}", values=["relu", "elu", "selu"])

        model.add(keras.layers.Dense(
            units, activation=activation,
            kernel_initializer="he_normal"
        ))
        # 搜索是否加 Dropout
        if hp.Boolean(f"dropout_{i}"):
            model.add(keras.layers.Dropout(
                rate=hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.5)
            ))

    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # 搜索學習率
    lr = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2,
                   sampling="log")  # 對數尺度採樣（因為 LR 在對數空間更均勻）

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# 建立搜索器
tuner = kt.BayesianOptimization(
    build_model_for_tuning,
    objective="val_accuracy",       # 最大化驗證準確率
    max_trials=20,                  # 嘗試 20 種超參數組合
    directory="kt_search",
    project_name="binary_classifier"
)

# 準備資料
X_data = np.random.randn(1000, 8).astype(np.float32)
y_data = (np.random.randn(1000) > 0).astype(np.float32)

X_tr, X_val = X_data[:800], X_data[800:]
y_tr, y_val = y_data[:800], y_data[800:]

# 執行搜索
print("開始超參數搜索（20 個 trial）...")
tuner.search(
    X_tr, y_tr,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[keras.callbacks.EarlyStopping(patience=3)],
    verbose=0
)

# 取得最佳超參數
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\n最佳超參數:")
print(f"  隱藏層數: {best_hps.get('n_hidden_layers')}")
print(f"  學習率: {best_hps.get('learning_rate'):.5f}")

# 用最佳超參數建立最終模型
best_model = tuner.hypermodel.build(best_hps)
# best_model.fit(X_tr, y_tr, epochs=50, validation_data=(X_val, y_val))
```

**✅ 程式碼逐行解析：**

1. `hp.Int/Float/Choice/Boolean`: 定義不同類型的超參數搜索空間
2. `sampling="log"`: 在對數尺度上均勻採樣學習率（因為 1e-4 和 1e-3 的差距比 1e-2 和 2e-2 更重要）
3. `BayesianOptimization`: 根據之前 trial 的結果預測哪些超參數組合更有前途（比 RandomSearch 效率高）

**🎯 重點摘要:**

- Keras Tuner 的其他搜索算法：`RandomSearch`（快速入門）、`Hyperband`（效率最高，基於早停的漏斗篩選）
- 實務上：先用 `RandomSearch` 快速了解超參數重要性，再用 `BayesianOptimization` 精細搜索

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: SavedModel 和 .keras 格式如何選擇？**

A: 需要跨語言部署（Java/C++/TF Serving）→ SavedModel；純 Keras/Python 工作流 → `.keras`（含自訂物件時更方便）；歷史相容性 → `.h5`。TF 2.x 環境建議優先用 SavedModel。

**Q2: TFLite 量化後準確率損失多少？**

A: 動態範圍量化通常損失 < 0.5%；全整數量化（INT8）通常損失 0.5~2%。若精度下降過多，嘗試量化感知訓練（Quantization-Aware Training）——訓練時模擬量化效果，最終準確率可接近浮點模型。

**Q3: MirroredStrategy 需要修改哪些代碼？**

A: 幾乎只需兩行：(1) `strategy = tf.distribute.MirroredStrategy()`；(2) `with strategy.scope():` 包裹模型建立和編譯。`fit()` 不需修改，建議同時調整 `batch_size`（乘以 GPU 數量）。

**Q4: Keras Tuner 搜索時間太長怎麼辦？**

A: 使用 Hyperband：它先用少量 epoch 訓練所有 trial，淘汰差的，逐漸增加存活 trial 的 epoch——通常比 RandomSearch 快 5-10 倍。也可以縮小搜索空間或減少 `max_trials`。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #TensorFlow #TFServing #TFLite #分散式訓練 #MirroredStrategy #KerasTuner #模型部署 #量化 #TFjs #深度學習 #程式設計 #教學 #MachineLearning
