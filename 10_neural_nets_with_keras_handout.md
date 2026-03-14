# 課程講義：人工神經網路簡介與 Keras 實作 (Chapter 10)

## 導論

各位同學大家好！歡迎來到本章課程，我們將深入探討人工神經網路 (Artificial Neural Networks, ANNs) 的世界。本章旨在為大家介紹 ANNs 的基本概念、它們如何從生物神經元中汲取靈感，並透過業界標準的 Keras 函式庫，實際建構、訓練與評估用於分類和迴歸任務的類神經網路模型。我們將涵蓋感知器、激活函數、多層感知器 (MLP) 的實作，以及模型儲存、回呼函數和 TensorBoard 視覺化等進階主題，最後探討超參數調優的策略。

透過本章的學習，您將能夠理解神經網路的運作原理，並具備使用 Keras 解決實際機器學習問題的能力。

## 1. 從生物神經元到人工神經元

### 為什麼重要？

人工神經網路的靈感源於人腦中的生物神經元，理解這些生物基礎有助於我們掌握 ANNs 的核心建構模塊。本節將探討感知器和激活函數等基本概念，為建構更複雜的神經網路模型奠定基礎。

### 如何運作？

人工神經網路的核心組件是人工神經元，它們模擬生物神經元的資訊處理方式。

* **感知器 (The Perceptron)**
    感知器是最簡單的人工神經元形式，它接收多個二元輸入 $x_i$，為每個輸入分配權重 $w_i$，然後將加權輸入求和，加上一個偏差項 $b$，最後通過一個激活函數（通常是步階函數）產生二元輸出 ($y=0$ 或 $1$)。

    數學上，對於輸入 $x_1, x_2, \dots, x_n$ 和權重 $w_1, w_2, \dots, w_n$，輸出 $y$ 為：
    $$
    y = \begin{cases} 
    1 & \text{if } \sum_{i=1}^n w_i x_i + b > 0 \\
    0 & \text{otherwise}
    \end{cases}
    $$
    感知器可以透過訓練來學習線性可分離的模式，但無法處理非線性問題，例如 XOR 問題。

* **激活函數 (Activation Functions)**
    激活函數引入了非線性，它將層的預激活值（加權和加上偏差）轉換為其輸出。這種非線性使得深度網路能夠學習複雜的非線性模式。常見的激活函數包括：

  * **步階 (Heaviside) 函數**: 根據輸入是否超過閾值輸出 0 或 1，用於早期的感知器。
  * **Sigmoid 函數**: 將輸入映射到 (0, 1) 區間，適用於二元分類，但可能存在梯度飽和問題。
  * **雙曲正切 (tanh) 函數**: 將輸入映射到 (-1, 1) 區間，零中心化，但同樣可能飽和。
  * **整流線性單元 (Rectified Linear Unit, ReLU)**: 輸出 $\max(0, x)$。有助於緩解梯度消失問題，但可能導致「死亡」神經元。
  * **Leaky ReLU / ELU**: ReLU 的變體，在單元不活躍時允許一個小的非零梯度，有助於防止死亡神經元。
  * **Softmax 函數**: 用於多類分類的輸出層，將 logits 轉換為機率分佈。
  * **線性激活 (Linear activation)**: 用於迴歸輸出層，直接傳遞輸入。

    每個激活函數都有不同的特性，會影響梯度的流動和學習動態。

### ⚡ 補充練習 1：

* **理論題**: 比較傳統感知器與羅吉斯迴歸分類器在收斂性、機率估計能力以及處理非線性問題上的差異。在何種情況下，兩者可以被視為等價？
* **實作題**: 參考 `10_neural_nets_with_keras.ipynb` 中感知器的範例程式碼。嘗試將感知器中的激活函數替換為 Sigmoid 函數，並說明這對模型的輸出和決策邊界可能產生什麼影響（無需實際運行，只需概念性說明）。

## 2. 使用 Keras 實作多層感知器

### 為什麼重要？

Keras 提供了一個高階且易於使用的 API，可以快速建構、訓練和評估神經網路。本節將展示如何利用 Keras 實作用於影像分類和迴歸的多層感知器，並介紹 Sequential API、Functional API 和 Subclassing API。

### 如何運作？

Keras 是一個強大的深度學習函式庫，支援多種神經網路架構的建立。

* **建立影像分類器 (Building an Image Classifier Using the Sequential API)**
    我們將使用 Fashion MNIST 資料集來展示影像分類的端到端工作流程。

    1. **資料準備**: 載入 Fashion MNIST 資料集，並將訓練集劃分為訓練集和驗證集。將像素強度縮放到 0-1 範圍以進行正規化。
    2. **模型建構 (Sequential API)**: 使用 `tf.keras.Sequential` 建立模型，層層堆疊：
        * `tf.keras.layers.InputLayer`: 定義輸入形狀 (例如 28x28 影像)。
        * `tf.keras.layers.Flatten()`: 將 2D 輸入轉換為 1D 向量。
        * `tf.keras.layers.Dense(units, activation="relu")`: 全連接層，使用 ReLU 激活。
        * `tf.keras.layers.Dense(10, activation="softmax")`: 輸出層，10 個類別使用 Softmax 激活函數。
    3. **模型編譯**: 使用 `model.compile()` 配置學習過程，指定損失函數 (例如 `sparse_categorical_crossentropy`)、優化器 (例如 `"sgd"`) 和評估指標 (例如 `"accuracy"`)。
    4. **模型訓練與評估**: 使用 `model.fit()` 訓練模型，並使用 `model.evaluate()` 評估模型在測試集上的性能。

* **建立迴歸多層感知器 (Building a Regression MLP Using the Sequential API)**
    針對迴歸任務 (例如預測房價)，我們同樣可以使用 Sequential API 建立 MLP。
    1. **資料準備**: 載入 California Housing 資料集，並進行訓練集、驗證集和測試集的劃分。使用 `tf.keras.layers.Normalization` 或 `StandardScaler` 對特徵進行標準化。
    2. **模型建構**: 定義幾個 `Dense` 隱藏層 (使用 ReLU 激活) 和一個單一線性輸出神經元。
    3. **模型編譯**: 使用迴歸損失函數 (例如 `"mse"`) 和優化器 (例如 `tf.keras.optimizers.Adam`)，並監控 RMSE。
    4. **模型訓練與評估**: 訓練模型並評估其在測試集上的性能。

* **使用 Functional API 建立複雜模型 (Building Complex Models Using the Functional API)**
    對於具有複雜拓撲或多輸入/多輸出的模型，Keras Functional API 提供更大的彈性。我們將建立一個 "Wide & Deep" 迴歸模型：
  * **多輸入**: 定義多個 `tf.keras.layers.Input` 層，每個輸入處理不同的特徵子集。
  * **分支路徑**: 建立「寬路徑」(直接連接輸入到輸出) 和「深路徑」(包含多個隱藏層)。
  * **合併**: 使用 `tf.keras.layers.Concatenate()` 合併不同路徑的輸出。
  * **多輸出**: 為主輸出和輔助輸出定義不同的輸出層，輔助輸出可用於正規化。
  * **模型建構**: 使用 `tf.keras.Model(inputs=[...], outputs=[...])` 建立模型。

* **使用 Subclassing API 建立動態模型 (Using the Subclassing API to Build Dynamic Models)**
    當模型架構需要動態行為或高度客製化時，可以透過繼承 `tf.keras.Model` 並實作 `call()` 方法來建立自定義模型。
  * **`__init__` 方法**: 初始化模型中使用的所有層。
  * **`call()` 方法**: 定義模型的前向傳播邏輯，處理輸入並產生輸出。

### ⚡ 補充練習 2：

* **理論題**: 解釋 `model.compile()` 方法中 `loss`、`optimizer` 和 `metrics` 這三個參數的作用，以及在分類和迴歸任務中，如何根據資料類型和任務目標選擇合適的選項。
* **實作題**: 參考 Functional API 建立 Wide & Deep 模型的範例。設計並實作一個新的 Functional API 模型，該模型具有兩個輸入，其中一個輸入經過一個隱藏層，然後與另一個原始輸入連接，最終產生一個輸出。

## 3. 模型儲存與恢復

### 為什麼重要？

在訓練完神經網路模型後，我們需要將其儲存起來以便日後重新載入進行預測，或繼續訓練。模型儲存確保了模型的可重用性和持久性。

### 如何運作？

Keras 提供多種儲存和恢復模型的方法。

* **儲存與載入整個模型 (Using the .keras format)**
    Keras 建議使用 `.keras` 格式儲存整個模型，它包含模型的架構、權重和優化器狀態。
  * **儲存**: `model.save("my_model.keras")`
  * **載入**: `loaded_model = tf.keras.models.load_model("my_model.keras", custom_objects={"WideAndDeepModel": WideAndDeepModel})` (如果模型包含自定義層或模型，需提供 `custom_objects`)

* **儲存與載入模型權重 (Saving and loading model weights)**
    如果只需要儲存模型的權重，可以使用 HDF5 格式的 `.weights.h5` 擴展名。
  * **儲存**: `model.save_weights("my_weights.weights.h5")`
  * **載入**: `model.load_weights("my_weights.weights.h5")` (需先建立模型架構)

### ⚡ 補充練習 3：

* **理論題**: 說明儲存整個 Keras 模型 (`.keras` 格式) 與只儲存模型權重 (`.weights.h5` 格式) 之間的主要差異。在何種情境下，你會選擇只儲存權重而非整個模型？
* **實作題**: 假設你已經訓練好一個 Keras Sequential 模型。請撰寫程式碼片段，展示如何將這個模型的權重儲存到檔案中，然後建立一個新的相同架構的模型，並從檔案中載入之前儲存的權重。

## 4. 使用回呼函數

### 為什麼重要？

回呼函數 (Callbacks) 是 Keras 提供的一種強大機制，可以在訓練過程中的不同階段執行自定義操作，例如在每個 epoch 結束時。它們對於自動化訓練流程、監控模型性能和防止過度擬合至關重要。

### 如何運作？

回呼函數透過繼承 `tf.keras.callbacks.Callback` 類別並覆寫其方法來實作。

* **模型檢查點 (Model Checkpointing)**
    `tf.keras.callbacks.ModelCheckpoint` 回呼函數允許您在訓練期間定期儲存模型的權重或整個模型。
  * `filepath`: 儲存路徑 (例如 `"my_checkpoints.weights.h5"`)。
  * `save_weights_only=True`: 只儲存權重。
  * `save_best_only=True`: 只儲存驗證集上性能最好的模型。
  * `monitor`: 監控的指標 (例如 `"val_loss"`)。

* **提早停止 (Early Stopping)**
    `tf.keras.callbacks.EarlyStopping` 回呼函數用於在模型性能不再提升時停止訓練，從而防止過度擬合。
  * `patience`: 在多少個 epoch 內性能沒有改善就停止訓練。
  * `restore_best_weights=True`: 訓練結束時恢復到性能最好的 epoch 的權重。

* **自定義回呼函數 (Custom Callbacks)**
    您可以透過繼承 `tf.keras.callbacks.Callback` 來自定義回呼函數，例如在每個 epoch 結束時印出訓練/驗證損失比率。
  * 覆寫 `on_epoch_end(self, epoch, logs)` 等方法來實作自定義邏輯。

### ⚡ 補充練習 4：

* **理論題**: 解釋 `tf.keras.callbacks.EarlyStopping` 中 `patience` 和 `restore_best_weights` 這兩個參數的具體作用，以及它們如何協同工作來防止模型過度擬合。
* **實作題**: 撰寫一個自定義 Keras 回呼函數 `LearningRateLogger`，它應在每個訓練批次 (batch) 結束時記錄並印出當前學習率。然後將此回呼函數添加到一個簡單模型的訓練過程中。

## 5. 使用 TensorBoard 進行視覺化

### 為什麼重要？

TensorBoard 是 TensorFlow 和 Keras 內建的強大視覺化工具。它允許開發者監控和視覺化模型訓練的各個方面，例如損失和準確度曲線、權重和偏差的直方圖以及模型圖。透過將 TensorBoard 整合到訓練流程中，可以深入了解模型的學習過程並識別潛在問題。

### 如何運作？

TensorBoard 透過在訓練期間記錄事件檔案來運作，這些檔案包含訓練指標、圖形定義和其他視覺化資料。

* **TensorBoard 回呼函數 (TensorBoard Callback)**
    `tf.keras.callbacks.TensorBoard` 回呼函數用於在訓練期間自動生成 TensorBoard 日誌。
  * `log_dir`: 指定日誌檔案的儲存目錄。
  * `profile_batch`: 啟用指定批次範圍的性能分析。

* **啟動 TensorBoard 伺服器 (Launching TensorBoard Server)**
    在訓練結束後，您可以使用命令或 Jupyter 筆記本中的 `%tensorboard` 魔術指令來啟動 TensorBoard 伺服器，並在瀏覽器中查看視覺化結果。
  * `%tensorboard --logdir=./my_logs`

### ⚡ 補充練習 5：

* **理論題**: 除了追蹤損失和準確度曲線外，請列舉 TensorBoard 提供的至少三種其他有用的視覺化功能，並簡要說明它們能幫助我們分析模型的哪些方面。
* **實作題**: 修改 TensorBoard 回呼函數的 `profile_batch` 參數，使其對模型訓練的前 50 個批次進行性能分析。說明你預期透過這個設定能夠觀察到什麼樣的性能數據。

## 6. 超參數調優

### 為什麼重要？

超參數調優是優化神經網路性能的關鍵步驟。超參數是控制模型訓練過程和架構的設定，例如學習率、批次大小、層數、每層神經元數量、激活函數和正規化技術。適當地調整這些超參數可以顯著影響模型的學習能力和泛化能力。

### 如何運作？

Keras Tuner 是一個函式庫，專門用於自動化超參數調優過程。

* **Keras Tuner 設定與搜尋策略 (Keras Tuner setup and search strategy)**
  * **`build_model(hp)` 函數**: 定義一個模型建構函數，它接受一個 `kt.HyperParameters` 物件 (`hp`)，並根據 `hp` 中定義的超參數 (例如 `n_hidden`, `n_neurons`, `learning_rate`, `optimizer`) 來建構 Keras 模型。
  * **`kt.RandomSearch`**: 一種隨機搜索策略，在給定的超參數空間中隨機採樣組合。
    * `objective`: 優化目標 (例如 `"val_accuracy"`)。
    * `max_trials`: 最大嘗試次數。
    * `overwrite=True`: 覆蓋之前的搜索結果。

* **Hyperband 演算法 (Hyperband)**
    Hyperband 是一種資源感知型的超參數優化算法，它結合了隨機搜索和自適應早期停止。它以較小的預算評估許多配置，並逐漸將更多資源分配給最有潛力的配置，從而避免在較差的候選方案上浪費計算資源。
  * `hypermodel`: 您的自定義 `HyperModel` (用於建構模型並處理預處理)。
  * `max_epochs`: 任何候選模型訓練的最大 epoch 數。
  * `factor`: 每次連續減半的縮減因子。

* **貝氏優化 (Bayesian Optimization)**
    貝氏優化器使用機率模型來引導超參數搜索。它根據過去的試驗建立目標函數的替代模型 (例如驗證準確度)，並使用該模型選擇接下來要評估的有潛力的超參數配置。這種方法平衡了對超參數空間新區域的探索與對已知良好區域的利用。
  * `alpha`: 控制高斯過程的先驗/噪聲項。
  * `beta`: 控制採集函數中探索與利用的權衡。

* **超參數調優的實用技巧 (Practical tips for hyperparameter tuning)**
  * **逐步優化**: 從粗略搜索 (例如 `RandomSearch`) 開始，然後使用 `BayesianOptimization` 精煉有潛力的區域。
  * **增加資源**: 逐漸增加 `max_trials` 或 `epochs`。
  * **監控**: 使用 `TensorBoard` 監控訓練過程以檢測過度擬合。
  * **回呼函數**: 在搜索過程中應用 `EarlyStopping` 和 `ModelCheckpoint`。
  * **重現性**: 設定隨機種子和目錄以確保實驗的可重現性。

### ⚡ 補充練習 6：

* **理論題**: 比較 Keras Tuner 中 `RandomSearch`、`Hyperband` 和 `BayesianOptimization` 這三種超參數調優方法的優缺點，並說明它們各自最適合在什麼樣的場景下使用。
* **實作題**: 選擇一個你認為最適合 MNIST 分類任務的 Keras Tuner 策略 (例如 `Hyperband` 或 `BayesianOptimization`)，設定其參數，並重新執行 MNIST 資料集的超參數調優過程。完成後，印出最佳的超參數組合。

## 課後作業

請完成 `10_neural_nets_with_keras.ipynb` 筆記本中的第 10 題練習：

* **練習**：在 MNIST 資料集上訓練一個深度 MLP（您可以使用 `tf.keras.datasets.mnist.load_data()` 載入資料集）。嘗試透過手動調整超參數來達到超過 98% 的準確度。嘗試使用本章介紹的方法（即通過指數級增長學習率、繪製損失圖並找到損失飆升的點）來搜尋最佳學習率。接下來，嘗試使用 Keras Tuner 並搭配所有功能——儲存檢查點、使用提早停止以及使用 TensorBoard 繪製學習曲線。

請務必將您的程式碼、訓練結果和對學習率、超參數調優策略選擇的說明，以及您從 TensorBoard 視覺化中獲得的洞見，一併提交。
