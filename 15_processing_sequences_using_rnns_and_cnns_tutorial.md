<!-- meta-title: RNN 與 CNN 序列處理完整指南：LSTM、GRU、WaveNet 與時間序列預測 -->
<!-- meta-description: 深入序列建模：SimpleRNN、LSTM、GRU 的工作原理與實作、序列到序列預測、TimeDistributed 層、WaveNet 膨脹因果捲積，以及時間序列預測的最佳實踐。 -->
<!-- meta-keywords: Python, RNN, LSTM, GRU, TensorFlow, Keras, 時間序列, WaveNet, 序列建模, 深度學習 -->
<!-- meta-hashtags: #Python #RNN #LSTM #GRU #TensorFlow #Keras #時間序列 #WaveNet #深度學習 #教學 -->

# 🐍 RNN 與 CNN 序列處理：從 SimpleRNN 到 WaveNet

語音識別、股票預測、機器翻譯——這些都是**序列問題（Sequence Problem）**。傳統前饋網路無法處理序列資料，因為它無法利用**時間上下文（Temporal Context）**。本教學帶你從最基礎的 RNN 出發，逐步掌握 LSTM、GRU，最終理解 WaveNet 的膨脹因果捲積架構。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🔄 SimpleRNN 基礎](#simplernn)
- [🧠 LSTM 長短期記憶](#lstm)
- [⚡ GRU 門控遞歸單元](#gru)
- [📊 深層與雙向 RNN](#deep-rnn)
- [🔢 序列到序列預測](#seq2seq)
- [📡 WaveNet 膨脹因果捲積](#wavenet)
- [⏰ 時間序列實戰](#time-series)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **SimpleRNN** 因梯度消失問題，無法捕捉長期依賴（>10~20 步）
- **LSTM** 的三個門（遺忘門、輸入門、輸出門）讓它能記憶/遺忘長期資訊
- **GRU** 是 LSTM 的簡化版，少一個門，速度更快，效能相近
- **`return_sequences=True`**：回傳每個時間步的輸出（給下一層 RNN 或序列輸出）
- **WaveNet** 的膨脹因果捲積可以平行訓練（不像 RNN 必須序列計算），感受野指數增長

---

## <a id="simplernn"></a>🔄 SimpleRNN 基礎

💡 **實際應用情境：** 用感測器數據預測機器故障——感測器每秒回傳一個讀數，我們需要利用過去 24 小時的資料來預測接下來是否會故障。這是一個序列分類問題。

### 範例 1: 建立 SimpleRNN 模型

```python
import tensorflow as tf
import numpy as np

# 產生簡單的序列資料（用於示範）
def generate_time_series(batch_size, n_steps):
    """生成包含兩個不同頻率正弦波的時間序列"""
    freq1, freq2, offsets1, offsets2 = (
        np.random.rand(4, batch_size, 1)
    )
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)  # [batch, steps, 1]

np.random.seed(42)
n_steps = 50
X_train = generate_time_series(7000, n_steps + 1)
X_valid = generate_time_series(2000, n_steps + 1)
X_test  = generate_time_series(500,  n_steps + 1)

# 輸入：前 50 步；輸出：第 51 步（預測下一個值）
y_train, y_valid, y_test = X_train[:, -1], X_valid[:, -1], X_test[:, -1]
X_train, X_valid, X_test = X_train[:, :-1], X_valid[:, :-1], X_test[:, :-1]

print(f"X_train shape: {X_train.shape}")  # (7000, 50, 1)
print(f"y_train shape: {y_train.shape}")  # (7000, 1)

# 最簡單的基準：直接用最後一個值（天真預測）
y_naive = X_valid[:, -1]
baseline_mse = np.mean(tf.keras.losses.mean_squared_error(y_valid, y_naive))
print(f"基準 MSE（天真預測）: {baseline_mse:.4f}")
```

**✅ 程式碼逐行解析：**

1. `[..., np.newaxis]`: 在最後添加維度，讓形狀從 `[batch, steps]` 變為 `[batch, steps, 1]`（RNN 需要特徵維度）
2. `X_train[:, -1]`: 取每個序列的最後一步作為目標值（預測未來一步）
3. 基準測試：用最後已知值直接預測——若你的 RNN 連這個都贏不了，說明模型有問題

---

## <a id="lstm"></a>🧠 LSTM 長短期記憶

💡 **實際應用情境：** 翻譯「我 **非常** 喜歡這部電影」時，LSTM 的遺忘門會根據「非常」修改對情感的記憶強度，而 SimpleRNN 在到達「電影」時早已「忘記」了「非常」的影響。

### 範例 2: LSTM 時間序列預測

```python
# 深層 LSTM 模型
model_lstm = tf.keras.Sequential([
    # 輸入形狀：(時間步數, 特徵數) = (50, 1)
    tf.keras.layers.LSTM(
        64,
        return_sequences=True,   # 回傳每個時間步的隱藏狀態（給下一層用）
        input_shape=[None, 1]    # None 表示可接受任意長度的序列
    ),
    tf.keras.layers.LSTM(32),    # 最後一層：只回傳最後時間步
    tf.keras.layers.Dense(1)     # 輸出一個值（預測下一個時間步）
])

model_lstm.compile(optimizer="adam", loss="mse")
model_lstm.summary()

# 訓練（使用早停回調）
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True  # 恢復最佳權重
    )
]

history = model_lstm.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=callbacks,
    verbose=0  # 靜默訓練（不輸出每個 epoch）
)

# 評估
lstm_mse = model_lstm.evaluate(X_valid, y_valid, verbose=0)
print(f"LSTM 驗證 MSE: {lstm_mse:.4f}")
print(f"基準 MSE: {baseline_mse:.4f}")
print(f"改善幅度: {(baseline_mse - lstm_mse) / baseline_mse * 100:.1f}%")
```

**LSTM 的核心結構（理解門控機制）：**

$$\text{遺忘門}: f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$\text{輸入門}: i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\text{輸出門}: o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$\text{記憶單元}: C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**✅ 程式碼逐行解析：**

1. `return_sequences=True`: LSTM 回傳每個時間步的輸出 `(batch, steps, units)`，讓下一層 LSTM 能看到完整序列
2. `input_shape=[None, 1]`: `None` 允許可變長度的序列（適應不同長度的輸入）
3. `EarlyStopping(restore_best_weights=True)`: 驗證損失不再改善後停止，並恢復最佳時期的權重

**🎯 重點摘要:**

- LSTM 的 `units` 參數是隱藏狀態的維度（不是時間步數）
- 堆疊多層 LSTM 時，中間層需要 `return_sequences=True`，最後一層預設 `return_sequences=False`

---

## <a id="gru"></a>⚡ GRU 門控遞歸單元

### 範例 3: GRU 模型（更快的 LSTM 替代品）

```python
# GRU 只有兩個門（更新門 + 重置門）vs LSTM 的三個門
model_gru = tf.keras.Sequential([
    tf.keras.layers.GRU(
        64,
        return_sequences=True,
        input_shape=[None, 1]
    ),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(1)
])

model_gru.compile(optimizer="adam", loss="mse")

# GRU 參數更少，通常訓練更快
lstm_params = model_lstm.count_params()
gru_params  = model_gru.count_params()
print(f"LSTM 參數量: {lstm_params:,}")
print(f"GRU 參數量: {gru_params:,}")
print(f"GRU 比 LSTM 少 {(lstm_params - gru_params)/lstm_params*100:.1f}% 參數")
```

**🎯 重點摘要:**

- GRU 參數約為 LSTM 的 75%，速度更快，長序列上效能相近
- 實務上：先嘗試 LSTM，若太慢或過擬合嚴重則改用 GRU

---

## <a id="deep-rnn"></a>📊 深層與雙向 RNN

### 範例 4: 雙向 LSTM（適合非實時任務）

```python
# 雙向 RNN：同時從前向後和從後向前處理序列
# 注意：不適合實時預測（需要未來資訊），但適合情感分析等任務
model_bidirectional = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        input_shape=[None, 1]
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32)
    ),
    tf.keras.layers.Dense(1)
])

# 雙向 LSTM 的輸出維度是 2 × units（前向 + 反向）
model_bidirectional.summary()
```

---

## <a id="seq2seq"></a>🔢 序列到序列預測

💡 **實際應用情境：** 氣象站的台灣天氣預測——輸入過去 50 天的氣溫，**一次輸出**未來 10 天的預測（而非每次只輸出一個值）。

### 範例 5: 多步驟序列預測

```python
# 目標：給定前 50 步，同時預測接下來 10 步
n_steps = 50
n_future = 10

# 重新準備資料
X_train_seq = generate_time_series(7000, n_steps + n_future)
y_train_seq = X_train_seq[:, n_steps:, 0]  # 未來 10 步作為目標
X_train_seq = X_train_seq[:, :n_steps]      # 前 50 步作為輸入

# 方法 1：Vector 輸出（最簡單）
model_vector = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(n_future)  # 直接輸出 10 個值
])

# 方法 2：Sequence 輸出（使用 TimeDistributed）
model_seq2seq = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_future)
    )
    # TimeDistributed 對每個時間步都應用 Dense，輸出 (batch, steps, n_future)
])

model_vector.compile(optimizer="adam", loss="mse")
model_vector.summary()
```

**✅ 程式碼逐行解析：**

1. `Dense(n_future)`: 最終 Dense 層直接輸出未來 10 個值——最簡單的多步預測方法
2. `TimeDistributed(Dense(n_future))`: 對 LSTM 每個時間步的輸出都應用 Dense，生成序列輸出

---

## <a id="wavenet"></a>📡 WaveNet 膨脹因果捲積

💡 **實際應用情境：** Google 的語音合成系統（WaveNet）可生成逼真的語音。它的核心是**膨脹因果捲積（Dilated Causal Convolution）**——感受野呈指數增長，且可以完全平行訓練（不像 RNN 必須序列計算）。

### 範例 6: WaveNet 風格架構

```python
# WaveNet 的關鍵：膨脹因果捲積（Dilated Causal Convolution）
# 膨脹率 1, 2, 4, 8 ... → 感受野從 1 指數增長

def build_wavenet(n_steps: int, n_features: int = 1,
                  dilation_rates=(1, 2, 4, 8, 16, 32)) -> tf.keras.Model:
    """建立 WaveNet 風格的膨脹因果捲積模型"""
    inputs = tf.keras.Input(shape=(n_steps, n_features))
    x = inputs

    for dilation_rate in dilation_rates:
        x = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=2,
            dilation_rate=dilation_rate,   # 膨脹率：跳步取樣
            padding="causal",              # 因果填充：只看過去（不看未來）
            activation="relu"
        )(x)

    outputs = tf.keras.layers.Conv1D(filters=1, kernel_size=1)(x)
    # 取最後一個時間步的輸出
    outputs = tf.keras.layers.Lambda(lambda x: x[:, -1:])(outputs)
    outputs = tf.keras.layers.Flatten()(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


model_wavenet = build_wavenet(n_steps=50)
model_wavenet.compile(optimizer="adam", loss="mse")
model_wavenet.summary()

# 膨脹率計算感受野：
# dilation=1:  感受野 = 2 (看前1個時間步)
# dilation=2:  感受野 = 4 (看前2個時間步)
# dilation=4:  感受野 = 8 (看前4個時間步)
# 到 dilation=32: 感受野 = 64 步！
print("WaveNet 感受野大小:")
receptive_field = 1
for d in [1, 2, 4, 8, 16, 32]:
    receptive_field += d
    print(f"  dilation={d:2d}: 累積感受野 = {receptive_field + 1}")
```

**✅ 程式碼逐行解析：**

1. `dilation_rate`: 捲積核元素之間的間距——dilation=4 時，kernel 看的是位置 t 和 t-4（跳過中間）
2. `padding="causal"`: 確保位置 t 的輸出只依賴 t 之前的輸入（不洩露未來資訊）
3. 感受野以 $2^L$ 增長（L 為層數），通過指數增長的感受野捕捉長期依賴

**🎯 重點摘要:**

- WaveNet vs RNN：RNN 必須序列計算（慢）；WaveNet 可完全平行（快）
- 膨脹捲積適合需要大感受野的序列任務（語音、音樂生成）

---

## <a id="time-series"></a>⏰ 時間序列實戰

### 範例 7: 完整預測管線

```python
# 完整的時間序列預測流程
import matplotlib.pyplot as plt

def plot_series(time, series, label=None, y_range=None):
    plt.plot(time, series, label=label)
    if y_range:
        plt.ylim(*y_range)

# 生成測試序列
test_series = generate_time_series(1, n_steps + 10)
X_test_single = test_series[:, :n_steps]
y_test_single = test_series[:, n_steps:, 0]

# 使用已訓練的 LSTM 做預測
# y_pred = model_lstm.predict(X_test_single)

# 多步預測：逐步預測並將預測值作為下一步的輸入
def multi_step_predict(model, X_input: np.ndarray, n_future: int) -> np.ndarray:
    """自迴歸多步預測"""
    current_input = X_input.copy()  # shape: (1, n_steps, 1)
    predictions = []
    for _ in range(n_future):
        pred = model.predict(current_input, verbose=0)  # (1, 1)
        predictions.append(pred[0, 0])
        # 滾動視窗：移除最舊的時間步，加入新預測值
        current_input = np.roll(current_input, shift=-1, axis=1)
        current_input[0, -1, 0] = pred[0, 0]
    return np.array(predictions)

print("多步預測完成！可視化預測結果。")
```

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: LSTM 和 GRU 如何選擇？**

A: GRU 計算更快（參數少），在許多任務上效能相近；LSTM 在極長序列（>100 步）上通常略好。建議先用 GRU 快速實驗，若效能不足再試 LSTM。

**Q2: `return_sequences=True/False` 怎麼選？**

A: 中間 RNN 層用 `return_sequences=True`（傳遞完整序列給下一層）；最後一層根據任務選擇：序列到序列用 `True`，序列到值用 `False`（預設）。

**Q3: 時間序列資料如何正確分割訓練/測試集？**

A: 必須按**時間順序**分割（不能隨機！）：前 70% 訓練，中間 15% 驗證，最後 15% 測試。隨機分割會造成資料洩露（訓練集中包含測試集的未來資訊）。

**Q4: RNN 訓練很慢怎麼辦？**

A: (1) 改用 GRU（比 LSTM 快 ~25%）；(2) 使用 `tf.keras.layers.CuDNNLSTM`（需要 GPU，但有限制）；(3) 考慮 WaveNet（可平行訓練）或 Transformer（現代替代品）。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #RNN #LSTM #GRU #TensorFlow #Keras #時間序列 #WaveNet #序列建模 #深度學習 #程式設計 #教學 #DataScience #MachineLearning
