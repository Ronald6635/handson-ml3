# 課程講義：使用 RNN 與 CNN 處理序列 (Chapter 15)

時間序列、文字、音訊——這些資料的共同特點是**順序很重要**。本章介紹循環神經網路（RNN），一種內建「記憶」的架構：它在處理序列時，隱藏狀態會將過去的資訊傳遞給未來的時間步驟。我們以芝加哥公共交通乘客量預測為主線，從 ARIMA 基準線出發，逐步引入 SimpleRNN、LSTM，最後以 WaveNet 的膨脹卷積收尾。

---

## 1. 時間序列前處理與傳統基準線

### 理論背景

在建立神經網路模型之前，必須先了解資料的統計特性，並建立**傳統基準線**——若神經網路無法超越 ARIMA，那就不值得用神經網路。

**時間序列的關鍵概念**：

- **趨勢 (Trend)**：長期的上升或下降方向
- **季節性 (Seasonality)**：固定週期的規律（如每週、每年）
- **差分 (Differencing)**：$\nabla y_t = y_t - y_{t-k}$ 移除趨勢/季節性，使序列平穩

**ARIMA (Autoregressive Integrated Moving Average)**：

$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t$$

SARIMA = ARIMA + 季節性項目，適合有週期性的時間序列。

**序列到監督式學習**：給定過去 $n$ 步，預測未來 1 步（或多步）。

### 核心代碼

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# 載入芝加哥每日乘客量
df = pd.read_csv("datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv",
                 parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)

rail = df["rail"] / 1e6  # 縮放至百萬人次

# 7 天差分消除季節性（一週週期）
diff_7 = rail.diff(7).dropna()

# ARIMA 基準線（每日重新訓練）
from statsmodels.tsa.arima.model import ARIMA

# 訓練 ARIMA 並預測下一步
model = ARIMA(rail.iloc[:-1], order=(1, 0, 0))  # 簡化版 AR(1)
model = model.fit()
y_pred = model.forecast()  # 預測下一天

# 轉換為監督式學習資料集
def to_supervised(series, seq_length=56):
    """將時間序列轉換為滑動視窗資料集"""
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series.iloc[i:i+seq_length].values)
        y.append(series.iloc[i+seq_length])
    return np.array(X)[..., np.newaxis], np.array(y)

seq_length = 56  # 8 週歷史
X_train_ts, y_train_ts = to_supervised(rail[:int(len(rail)*0.7)], seq_length)
print(f"X_train shape: {X_train_ts.shape}")  # (n_samples, 56, 1)
```

### ⚡ 補充練習 1

**理論題：** 為什麼要用 7 天差分而非 1 天差分？如何用「自相關函數（ACF）」圖判斷一個時間序列的季節性週期？

**實作題：** 計算芝加哥 Rail 乘客量的 7 天和 365 天自相關，繪製 ACF 圖（用 `statsmodels.graphics.tsaplots.plot_acf`），從圖中識別出週週期和年週期。

---

## 2. SimpleRNN 與梯度消失問題

### 理論背景

**RNN 的基本公式**：

$$\mathbf{h}_t = \tanh\left(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}\right)$$

隱藏狀態 $\mathbf{h}_t$ 既是目前時間步的輸出，也傳遞給下一個時間步作為記憶。

**梯度消失 (Vanishing Gradient)**：

反向傳播時梯度需要沿時間步「回傳」，每步乘以 $\mathbf{W}_h^T$。若 $\|\mathbf{W}_h\| < 1$，梯度指數級萎縮 → RNN 難以學習長期依賴關係。

**梯度爆炸**的解法：梯度裁剪 (`clipnorm`, `clipvalue`)

**梯度消失**的解法：LSTM、GRU（門控機制）

### 核心代碼

```python
import tensorflow as tf

tf.random.set_seed(42)

# SimpleRNN 序列預測
model_simple = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])

model_simple.compile(optimizer="adam", loss="huber")
history = model_simple.fit(
    X_train_ts, y_train_ts,
    epochs=20,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# 深層 RNN（堆疊 SimpleRNN）
model_deep = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20, return_sequences=True),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])
```

### ⚡ 補充練習 2

**理論題：** `return_sequences=True` 與 `return_sequences=False`（預設）的差異是什麼？當你需要堆疊多個 RNN 層時，除了最後一層，所有中間層都需要設定什麼？

**實作題：** 比較 `SimpleRNN(20)` 和 `SimpleRNN(20, return_sequences=True) → Dense(1)` 兩種結構在時間序列預測上的 MAE，哪一種對序列預測任務更合適？

---

## 3. LSTM 與 GRU 的門控機制

### 理論背景

**LSTM (Long Short-Term Memory)** 引入**細胞狀態 (Cell State)** $\mathbf{c}_t$，解決梯度消失：

**遺忘閘 (Forget Gate)**：決定丟棄多少舊記憶：

$$\mathbf{f}_t = \sigma\left(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\right)$$

**輸入閘 (Input Gate)**：決定加入多少新資訊：

$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**細胞狀態更新**：

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**輸出閘 (Output Gate)**：

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

**GRU**：LSTM 的簡化版（合併遺忘閘和輸入閘），參數更少，通常速度更快，效果相近。

### 核心代碼

```python
tf.random.set_seed(42)

# LSTM 模型（深層）
model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dense(1)
])
model_lstm.compile(optimizer="adam", loss="huber")

# GRU 模型（更快，效果相近）
model_gru = tf.keras.Sequential([
    tf.keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.GRU(20),
    tf.keras.layers.Dense(1)
])
model_gru.compile(optimizer="adam", loss="huber")

# 多變數時間序列（bus + rail + day_type）
# 輸入形狀：(batch_size, seq_length, n_features)
model_multivar = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),  # 5 個特徵
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1)
])
```

### ⚡ 補充練習 3

**理論題：** LSTM 的細胞狀態 $\mathbf{c}_t$ 如何解決梯度消失問題？從反向傳播的角度，說明為何梯度可以「穿越時間」而不會大幅縮減。

**實作題：** 在芝加哥乘客量資料上，比較 `SimpleRNN`、`LSTM`、`GRU` 三種架構的測試集 MAE 和訓練時間（各 20 epochs），製成比較表。

---

## 4. 多步預測與序列到序列模型

### 理論背景

**多步預測策略**：

1. **迭代預測**：每次只預測 1 步，將預測值加入輸入，再預測下一步（誤差會累積）
2. **直接多步輸出**：最後一層改為 `Dense(n_steps)`，一次輸出多步（不累積誤差）
3. **Seq2Seq**：`TimeDistributed(Dense(n_steps))` 對每個時間步都輸出，訓練更高效

**`TimeDistributed` 層**：將同一個 `Dense` 層應用到序列的每個時間步：

```python
# 等效寫法
TimeDistributed(Dense(14))  ≡  Dense(14)  # 對 LSTM 的 return_sequences=True 輸出
```

### 核心代碼

```python
tf.random.set_seed(42)

# 直接多步輸出（預測未來 14 天）
model_ahead = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(14)  # 一次輸出 14 步
])
model_ahead.compile(optimizer="adam", loss="mae")

# Seq2Seq（TimeDistributed，每步都預測）
model_seq2seq = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),
    tf.keras.layers.LSTM(16, return_sequences=True),
    # 等效：tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(14))
    tf.keras.layers.Dense(14)  # Keras Dense 自動對序列每步應用
])
```

### ⚡ 補充練習 4

**理論題：** 迭代預測（步驟1）的誤差如何「雪球效應」累積？在預測 14 天後，直接多步輸出（策略2）的誤差是否一定優於迭代預測？各有什麼情境下更合適？

**實作題：** 實作迭代預測：訓練一個只預測 1 步的 LSTM，然後循環呼叫 14 次，每次將前一次的預測加入輸入序列。比較迭代預測與直接 `Dense(14)` 輸出在 14 天預測上的 MAE。

---

## 5. WaveNet：膨脹因果卷積

### 理論背景

**WaveNet** 使用 **膨脹因果卷積 (Dilated Causal Convolutions)** 處理長序列：

- **因果 (Causal)**：只看過去的輸入，不洩漏未來資訊（`padding="causal"`）
- **膨脹 (Dilated)**：卷積核的「感受野」以指數速度增長

膨脹率倍增：1, 2, 4, 8, 16, ...

- Dilation=1：感受野 = 2 步
- Dilation=2：感受野 = 3 步（跳隔取樣）
- Dilation=4：感受野 = 5 步
- 10 層（膨脹率 1→512）：感受野 = **1023 步**（$2^{10} - 1$）

優點：不受梯度消失影響、平行計算比 RNN 快、長期依賴性強。

### 核心代碼

```python
tf.random.set_seed(42)

# WaveNet 架構
wavenet_model = tf.keras.Sequential()
wavenet_model.add(tf.keras.layers.Input(shape=[None, 5]))

# 堆疊膨脹因果卷積層（膨脹率：1, 2, 4, 8）
for dilation_rate in (1, 2, 4, 8):
    wavenet_model.add(
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=2,
            padding="causal",      # 因果填充（只看過去）
            activation="relu",
            dilation_rate=dilation_rate  # 膨脹率
        )
    )

wavenet_model.add(tf.keras.layers.Dense(14))
wavenet_model.compile(optimizer="adam", loss="mae")
print(wavenet_model.summary())
```

### ⚡ 補充練習 5

**理論題：** WaveNet 使用 `kernel_size=2`、`dilation_rate=8` 的卷積，這個卷積核能看到多少步之前的資訊（感受野是多少）？計算 4 層（dilation 1, 2, 4, 8）堆疊後的總感受野。

**實作題：** 比較 LSTM 模型和 WaveNet 模型在芝加哥乘客量資料上的訓練速度（每 epoch 時間）和最終 MAE，哪種在此資料集上更有優勢？

---

## 結論

序列模型的演進：

- **SimpleRNN**：概念清晰，但梯度消失使其難以學習長期依賴
- **LSTM/GRU**：門控機制解決梯度消失，是最常用的序列模型
- **多步預測**：直接輸出多步通常優於迭代預測
- **WaveNet**：膨脹因果卷積，長感受野、平行計算，適合長序列

下一章（Ch16）將 RNN 應用於 NLP：文字生成、情感分析、機器翻譯和 Attention 機制。

---

## 課後作業

**作業：時間序列預測競賽**

使用芝加哥乘客量資料集（或自選時間序列）：

1. 實作 ARIMA 基準線和 LSTM 模型，比較兩者在測試集的 MAE（以百萬人次為單位）。

2. 建立**多變數 LSTM**，將 `bus`、`rail`、以及 `day_type` 的 One-Hot 編碼同時作為輸入特徵，是否能提升預測準確率？

3. 嘗試 WaveNet（4 層膨脹卷積），比較其與 LSTM 的訓練速度和最終預測效果，製成對比表（模型 / MAE / 訓練時間 / 參數量）。
