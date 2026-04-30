# Ch15 速查表：Processing Sequences with RNNs & CNNs

> **核心主旨**：時間序列 + 序列模型 —— LSTM/GRU 處理長期依賴，Conv1D 提取局部模式，WaveNet 串接兩者。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| SimpleRNN | 最基本的 RNN，容易梯度消失 | 短序列實驗 |
| LSTM | 有遺忘門/輸入門/輸出門，記憶長期依賴 | 長序列（文字、時間序列） |
| GRU | LSTM 的精簡版（合併遺忘+輸入門），速度更快 | 資源有限時的 LSTM 替代 |
| Deep RNN | 堆疊多層 RNN，`return_sequences=True` | 提升序列表示能力 |
| Bidirectional RNN | 正向 + 反向 RNN 並行，雙向看序列 | NLP、序列分類 |
| Conv1D | 在時間維度做 1D 卷積，提取局部模式 | 時間序列特徵提取 |
| WaveNet | 膨脹因果卷積（dilation），接受野指數增長 | 音訊生成、長序列建模 |
| ARIMA | 統計時間序列模型（差分+自迴歸+移動平均） | 週期性單變數時間序列基準 |

---

## 2. 關鍵 API 速查

| Keras API | 重點參數 | 用途 |
|-----------|---------|------|
| `tf.keras.layers.SimpleRNN` | `units=20`, `return_sequences=True` | 基本 RNN（不推薦用於實務） |
| `tf.keras.layers.LSTM` | `units=20`, `return_sequences=True/False` | 長短期記憶 |
| `tf.keras.layers.GRU` | `units=20`, `return_sequences=True/False` | 閘控循環單元 |
| `tf.keras.layers.Bidirectional` | `layer=LSTM(...)` | 雙向 RNN 包裝器 |
| `tf.keras.layers.Conv1D` | `filters=32`, `kernel_size=5`, `padding="causal"` | 1D 因果卷積 |
| `tf.keras.layers.LayerNormalization` | – | 層歸一化（RNN 常用，代替 BN） |
| `tf.keras.layers.TimeDistributed` | `layer=Dense(...)` | 對每個時間步獨立應用層 |
| `padding="causal"` | – | 確保只看過去資料，不看未來 |
| `dilation_rate=2` | – | 膨脹卷積，擴大感受野 |

---

## 3. 必備代碼片段

```python
import tensorflow as tf
import numpy as np

# 時間序列資料準備（seq2seq 格式）
def to_windows(series, seq_length):
    """將序列切成 (input, target) 滑動視窗"""
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(seq_length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(seq_length + 1))
    dataset = dataset.map(lambda w: (w[:-1], w[-1]))  # 最後一步為 target
    return dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Simple RNN（基礎）
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])

# Deep LSTM（實務推薦）
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=[None, n_features]),
    tf.keras.layers.LSTM(64),          # 最後一層不需要 return_sequences
    tf.keras.layers.Dense(1)           # 迴歸輸出
])

# GRU（比 LSTM 輕量）
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True, input_shape=[None, n_features]),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(n_outputs)
])

# 雙向 LSTM（NLP 任務）
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Conv1D 時間序列（因果填充）
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                            padding="causal", activation="relu",
                            input_shape=[None, n_features]),
    tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="causal", activation="relu"),
    tf.keras.layers.Dense(1)
])

# WaveNet 風格（膨脹因果卷積）
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[None, n_features]))
for dilation_rate in [1, 2, 4, 8, 16, 32, 64, 128]:
    model.add(tf.keras.layers.Conv1D(
        filters=32, kernel_size=2,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu"))
model.add(tf.keras.layers.Conv1D(filters=n_outputs, kernel_size=1))

# 訓練時間序列模型
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,
                                restore_best_weights=True)])
```

---

## 4. 常見陷阱

- **`return_sequences=True` vs `False`**：堆疊多層 RNN 時，除最後一層外全部要 `return_sequences=True`；若需 seq2seq 輸出，最後一層也要 True。
- **因果卷積 `padding="causal"`**：時間序列預測時，**絕對不能**看未來資料，必須用 `"causal"` 而非 `"same"`。
- **LSTM 比 SimpleRNN 慢 3-5 倍**：輕量任務或 CPU 環境可考慮 GRU（速度接近 LSTM，效果略差一點）。
- **LayerNormalization vs BatchNormalization**：RNN 中使用 LayerNorm（對每個樣本的所有特徵歸一化），不用 BatchNorm（對每個批次的樣本歸一化，序列長度不固定時問題很多）。

---

## 5. 決策指南

```
時間序列任務選模型：
├── 統計基準線         → ARIMA/SARIMA（statsmodels）
├── 短期局部模式       → Conv1D（快速）
├── 長期依賴           → LSTM 或 GRU
├── 超長序列（音訊）   → WaveNet（膨脹卷積）
└── 混合               → Conv1D → LSTM（先提取局部，再建模全局）

LSTM vs GRU：
├── 精度優先           → LSTM
├── 速度/記憶體優先    → GRU
└── 實務差異不大       → 兩者都試，選驗證集更好的

多步預測策略：
├── 遞推（1步→多步）→ 誤差累積
├── 直接（一次預測多步）→ 更穩定（推薦）
└── seq2seq Encoder-Decoder → 複雜但靈活
```
