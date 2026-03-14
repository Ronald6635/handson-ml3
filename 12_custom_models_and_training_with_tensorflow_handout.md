# 課程講義：自定義模型與 TensorFlow 訓練 (Chapter 12)

大家早安。在之前的課程中，我們主要使用 Keras 的高階 API 來建構模型。然而，當你面對尖端的學術研究或需要極致的效能優化時，標準的層與損失函數往往不夠用。今天，我們要「打開黑盒子」，學習如何利用 TensorFlow 的低階運算來建構完全自定義的深度學習架構。

---

## 1. TensorFlow 基礎：張量與運算 (Tensors & Ops)

TensorFlow 的核心數據結構是**張量 (Tensors)**。它們與 NumPy 的 `ndarray` 非常相似，但具備兩個關鍵優勢：支援 GPU 加速且對自動微分友好。

- **不可變性 (Immutability)**：與 NumPy 陣列不同，普通張量是不可變的。若需儲存模型權重等可變狀態，必須使用 `tf.Variable`。
- **類型嚴格性**：TensorFlow 不會自動執行隱式類型轉換（例如 `float32` 與 `int32` 相加會報錯）。請務必使用 `tf.cast()` 手動轉換。

* **⚡ 補充練習 1：**
    1. **理論題**：為什麼 TensorFlow 將張量設計為不可變的？這對計算圖（Computation Graph）的優化有什麼幫助？
    2. **實作題**：建立一個隨機的 $3 \times 3$ 張量 $A$，計算 $A^T A$，並將結果轉換為 `tf.float64` 類型。

---

## 2. 自定義損失函數與指標 (Custom Loss & Metrics)

當標準的 MSE 或 Cross-Entropy 無法捕捉問題的特性時（例如需要對離群值具備魯棒性的 Huber Loss），我們需要自定義函數。

- **自定義損失**：通常寫成一個接收 `(y_true, y_pred)` 並返回張量的函數即可。
- **串流指標 (Streaming Metrics)**：與損失函數不同，指標（如 Precision）需要在整個訓練週期中累積狀態。此時應繼承 `tf.keras.metrics.Metric` 並實作 `update_state()` 與 `result()`。

* **⚡ 補充練習 2：**

    1. **理論題**：在實作 Huber Loss 時，為什麼建議計算 $\sqrt{\text{variance} + \epsilon}$ 而非僅僅是 $\sqrt{\text{variance}}$？
    2. **實作題**：請實作 `log_cosh_loss(y_true, y_pred)`，公式為 $L = \sum \log(\cosh(y_{pred} - y_{true}))$。

---

## 3. 自定義層與模型 (Custom Layers & Models)

這是建構如殘差連接（Residual Connections）或自注意力機制（Self-Attention）等複雜架構的基礎。

- **Layer Subclassing**：繼承 `tf.keras.layers.Layer`。
  - `__init__`：儲存超參數。
  - `build()`：定義權重（這是在得知輸入形狀後延遲初始化的最佳時機）。
  - `call()`：定義前向傳播邏輯。
- **Model Subclassing**：繼承 `tf.keras.Model`。通常用於定義模型整體的拓撲結構，這讓你可以自由控制 `call()` 內部的複雜數據流。

* **⚡ 補充練習 3：**

    1. **理論題**：請說明 `tf.keras.layers.Layer` 與 `tf.keras.Model` 的主要差異與各自的使用場景。
    2. **實作題**：建立一個自定義層 `MySoftmax`，其實作標準的 Softmax 運算：$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$。

---

## 4. 自動微分與訓練迴圈 (Autodiff & Custom Loops)

`tf.GradientTape` 是 TensorFlow 實現自動微分的神經中樞。

- **Gradient Tape**：在 `with tf.GradientTape() as tape:` 區塊內執行的運算會被記錄在「磁帶」上，隨後可用 `tape.gradient(target, sources)` 計算梯度。
- **自定義訓練迴圈**：這讓你可以手動控制每一個訓練步驟。雖然比 `model.fit()` 複雜且易出錯，但它允許你在訓練過程中加入非標準邏輯（例如為不同層設定不同的學習率，或手動裁剪梯度）。

* **⚡ 補充練習 4：**

    1. **理論題**：為什麼在計算高階導數（如 Hessian 矩陣）時，我們需要巢狀（Nested）的 `GradientTape`？
    2. **實作題**：撰寫一段虛擬碼（Pseudocode），展示如何結合 `GradientTape` 與 `optimizer.apply_gradients()` 來更新一個簡單模型的權重。

---

## 5. TensorFlow 函數與圖形 (TF Functions & Graphs)

透過 `@tf.function` 裝飾器，TensorFlow 能將 Python 函數轉換為高效的、可移植的**計算圖**。

- **AutoGraph**：自動將 Python 控制流（如 `if`, `for`, `while`）轉換為對應的 TensorFlow 運算節點。
- **效能規則**：避免在 `@tf.function` 內定義變數，且儘量使用張量運算而非純 Python 運算，以避免頻繁的「追蹤（Tracing）」導致效能下降。

* **⚡ 補充練習 5：**

    1. **理論題**：什麼是「多形性（Polymorphism）」在 TensorFlow 函數中的意義？傳入不同形狀的張量會觸發什麼行為？
    2. **實作題**：撰寫一個使用 `@tf.function` 的函數，並使用 `tf.autograph.to_code()` 觀察其生成的圖形代碼。

---

## 結論

掌握了自定義組件與底層 API 後，你將不再受限於框架預設的功能。這不僅是開發新演算法的必經之路，也是深入理解深度學習運作原理的最佳途徑。

---

## 課後作業 (Assignment)

**題目：實作 Layer Normalization 層**
請參考課本練習 12，自行實作一個 `LayerNormalization` 層。該層需包含：

1. 可訓練的權重 $\alpha$ (Scale) 與 $\beta$ (Offset)。
2. 在 `call()` 中計算每個樣本特徵的平均值 $\mu$ 與標準差 $\sigma$。
3. 最終輸出公式為：$Y = \alpha \otimes \frac{X - \mu}{\sigma + \epsilon} + \beta$。
**驗證**：請將你的實作結果與 `tf.keras.layers.LayerNormalization` 進行對比，確保在相同輸入下的誤差極小。
