<!-- meta-title: 使用 Keras 學習人工神經網路：完整教學指南 -->
<!-- meta-description: 從生物神經元到深度學習的完整 Keras 神經網路教學，包含實作範例、逐行解析與最佳實踐。 -->
<!-- meta-keywords: Keras, 神經網路, 人工智慧, 機器學習, 深度學習, 教學, Python -->
<!-- meta-hashtags: #Keras #神經網路 #人工智慧 #機器學習 #深度學習 #Python教學 #程式設計 -->

# 🐍 使用 Keras 介紹人工神經網路

這份教學基於《Hands-On Machine Learning》第 3 版第 10 章，完整介紹人工神經網路的基本概念與 Keras 實作。從生物神經元開始，逐步學習感知器、多層感知器（MLP）、激活函數，以及使用 Keras 的三種 API 建構神經網路模型。

## 📝 本文目錄
- [🎯 關鍵重點](#關鍵重點)
- [🧠 從生物神經元到人工神經網路](#生物神經元)
- [⚡ 感知器](#感知器)
- [🔄 多層感知器與反向傳播](#多層感知器)
- [📊 使用 Keras 建構神經網路](#keras建構)
- [💾 模型儲存與載入](#模型儲存)
- [📈 訓練神經網路](#訓練神經網路)
- [🔍 視覺化與除錯](#視覺化)
- [🎛️ 超參數調校](#超參數調校)
- [❓ 常見問答](#常見問答)
- [🏷️ 推薦標籤](#推薦標籤)

## 🎯 關鍵重點
- 人工神經網路模仿生物神經系統，透過層層神經元處理資訊
- Keras 提供三種建構模型的 API：Sequential、Functional、Subclassing
- 激活函數決定神經元的輸出，非線性函數讓網路能學習複雜模式
- 反向傳播演算法透過梯度下降優化網路權重
- 正規化技巧如 Dropout 能防止過擬合

## <a id="生物神經元"></a>🧠 從生物神經元到人工神經網路

生物神經元是神經系統的基本單位，接收來自其他神經元的訊號，當累積訊號超過閾值時會發射訊號。人工神經網路模仿這個概念，建構出能學習複雜模式的計算模型。

### 範例 1: 檢查 Python 與套件版本

```python
import sys

# 確保 Python 版本至少 3.7
assert sys.version_info >= (3, 7)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 匯入 sys 模組，用於檢查系統資訊
2. `assert sys.version_info >= (3, 7)`: 斷言 Python 版本必須大於等於 3.7，否則拋出錯誤

**🎯 重點摘要:**

- **核心功能**: 版本檢查確保程式相容性
- **潛在問題**: 舊版 Python 可能缺少某些功能
- **最佳使用情境**: 在程式開始時進行環境檢查

### 範例 2: 檢查 Scikit-Learn 版本

```python
import sklearn

# 檢查 Scikit-Learn 版本
print(sklearn.__version__)
```

**✅ 程式碼逐行解析：**

1. `import sklearn`: 匯入 scikit-learn 機器學習套件
2. `print(sklearn.__version__)`: 印出套件版本號

**🎯 重點摘要:**

- **核心功能**: 確認機器學習套件版本
- **潛在問題**: 不同版本 API 可能有差異
- **最佳使用情境**: 除錯時確認套件版本

## <a id="感知器"></a>⚡ 感知器

感知器是最簡單的人工神經網路，由 Frank Rosenblatt 在 1957 年提出。它是一個單層神經網路，能學習線性可分的模式。

### 範例 3: 使用 Scikit-Learn 實作感知器

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# 載入鳶尾花資料集
iris = load_iris()
X = iris.data[:, (2, 3)]  # 花瓣長度和寬度
y = (iris.target == 0).astype(int)  # 是否為 Setosa

# 建立感知器模型
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

# 預測
y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)
```

**✅ 程式碼逐行解析：**

1. `import numpy as np`: 匯入 NumPy 用於數值運算
2. `from sklearn.datasets import load_iris`: 載入鳶尾花資料集
3. `from sklearn.linear_model import Perceptron`: 匯入感知器類別
4. `iris = load_iris()`: 載入資料集
5. `X = iris.data[:, (2, 3)]`: 選取花瓣特徵
6. `y = (iris.target == 0).astype(int)`: 建立二元分類標籤
7. `per_clf = Perceptron(random_state=42)`: 建立感知器模型
8. `per_clf.fit(X, y)`: 訓練模型
9. `y_pred = per_clf.predict([[2, 0.5]])`: 預測新樣本
10. `print(y_pred)`: 印出預測結果

**🎯 重點摘要:**

- **核心功能**: 實作單層神經網路進行二元分類
- **潛在問題**: 只能學習線性可分問題
- **最佳使用情境**: 簡單分類任務的基準模型

## <a id="多層感知器"></a>🔄 多層感知器與反向傳播

多層感知器（MLP）由多層神經元組成，能學習非線性模式。反向傳播演算法透過計算梯度來更新權重。

### 範例 4: 實作多層感知器

```python
from sklearn.neural_network import MLPClassifier

# 建立 MLP 分類器
mlp_clf = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', 
                       solver='adam', random_state=42, max_iter=1000)
mlp_clf.fit(X, y)

# 預測
y_pred = mlp_clf.predict([[2, 0.5]])
print(y_pred)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.neural_network import MLPClassifier`: 匯入多層感知器
2. `mlp_clf = MLPClassifier(...)`: 設定隱藏層大小、激活函數等參數
3. `mlp_clf.fit(X, y)`: 訓練模型
4. `y_pred = mlp_clf.predict([[2, 0.5]])`: 預測
5. `print(y_pred)`: 輸出結果

**🎯 重點摘要:**

- **核心功能**: 使用多層網路學習非線性模式
- **潛在問題**: 訓練時間較長，可能過擬合
- **最佳使用情境**: 非線性分類與迴歸任務

## <a id="keras建構"></a>📊 使用 Keras 建構神經網路

Keras 是高階神經網路 API，提供三種建構模型的方式：Sequential、Functional、Subclassing。

### 範例 5: 使用 Sequential API 建構模型

```python
import tensorflow as tf
from tensorflow import keras

# 建立 Sequential 模型
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# 編譯模型
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
```

**✅ 程式碼逐行解析：**

1. `import tensorflow as tf`: 匯入 TensorFlow
2. `from tensorflow import keras`: 匯入 Keras
3. `model = keras.models.Sequential([...])`: 建立序列模型
4. `keras.layers.Flatten(input_shape=[28, 28])`: 攤平輸入層
5. `keras.layers.Dense(300, activation="relu")`: 全連接層
6. `model.compile(...)`: 設定損失函數、優化器、評估指標

**🎯 重點摘要:**

- **核心功能**: 簡單直覺的模型建構方式
- **潛在問題**: 無法處理複雜架構如多輸入輸出
- **最佳使用情境**: 標準前饋網路

### 範例 6: 使用 Functional API

```python
# 建立 Functional 模型
input_ = keras.layers.Input(shape=[28, 28])
flatten = keras.layers.Flatten(input_shape=[28, 28])(input_)
hidden1 = keras.layers.Dense(300, activation="relu")(flatten)
hidden2 = keras.layers.Dense(100, activation="relu")(hidden1)
output = keras.layers.Dense(10, activation="softmax")(hidden2)
model = keras.models.Model(inputs=[input_], outputs=[output])
```

**✅ 程式碼逐行解析：**

1. `input_ = keras.layers.Input(shape=[28, 28])`: 定義輸入
2. `flatten = keras.layers.Flatten(...)(input_)`: 連接層
3. `hidden1 = keras.layers.Dense(...)(flatten)`: 第一隱藏層
4. `model = keras.models.Model(...)`: 建立模型物件

**🎯 重點摘要:**

- **核心功能**: 靈活的模型架構設計
- **潛在問題**: 語法較複雜
- **最佳使用情境**: 複雜網路如多分支或殘差網路

### 範例 7: 使用 Subclassing API

```python
class MyModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(300, activation="relu")
        self.dense2 = keras.layers.Dense(100, activation="relu")
        self.dense3 = keras.layers.Dense(10, activation="softmax")
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

model = MyModel()
```

**✅ 程式碼逐行解析：**

1. `class MyModel(keras.models.Model)`: 繼承 Model 類別
2. `def __init__(self)`: 初始化層
3. `def call(self, inputs)`: 定義前向傳播
4. `model = MyModel()`: 實例化模型

**🎯 重點摘要:**

- **核心功能**: 最大靈活性，自訂邏輯
- **潛在問題**: 較難除錯，程式碼較長
- **最佳使用情境**: 研究與自訂網路

## <a id="模型儲存"></a>💾 模型儲存與載入

訓練好的模型可以儲存為 HDF5 格式或 SavedModel 格式。

### 範例 8: 儲存與載入模型

```python
# 儲存模型
model.save("my_model.h5")

# 載入模型
loaded_model = keras.models.load_model("my_model.h5")
```

**✅ 程式碼逐行解析：**

1. `model.save("my_model.h5")`: 儲存為 HDF5 格式
2. `loaded_model = keras.models.load_model("my_model.h5")`: 載入模型

**🎯 重點摘要:**

- **核心功能**: 模型持久化
- **潛在問題**: HDF5 格式即將棄用
- **最佳使用情境**: 模型部署與重用

## <a id="訓練神經網路"></a>📈 訓練神經網路

使用回呼函數（callbacks）可以監控訓練過程並實作早期停止等技巧。

### 範例 9: 使用回呼函數訓練

```python
# 定義回呼
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_model.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# 訓練模型
history = model.fit(X_train, y_train, epochs=100,
                   validation_data=(X_valid, y_valid),
                   callbacks=[checkpoint_cb, early_stopping_cb])
```

**✅ 程式碼逐行解析：**

1. `checkpoint_cb = keras.callbacks.ModelCheckpoint(...)`: 儲存最佳模型
2. `early_stopping_cb = keras.callbacks.EarlyStopping(...)`: 早期停止
3. `model.fit(...)`: 訓練模型並使用回呼

**🎯 重點摘要:**

- **核心功能**: 自動化訓練過程管理
- **潛在問題**: 需要適當設定 patience 參數
- **最佳使用情境**: 防止過擬合與最佳化訓練

## <a id="視覺化"></a>🔍 視覺化與除錯

TensorBoard 提供強大的視覺化功能來監控訓練過程。

### 範例 10: 使用 TensorBoard

```python
import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# 訓練時使用 TensorBoard 回呼
history = model.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid),
                   callbacks=[tensorboard_cb])
```

**✅ 程式碼逐行解析：**

1. `root_logdir = os.path.join(os.curdir, "my_logs")`: 設定日誌目錄
2. `def get_run_logdir()`: 產生唯一執行目錄
3. `tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)`: 建立 TensorBoard 回呼
4. `model.fit(..., callbacks=[tensorboard_cb])`: 訓練時記錄日誌

**🎯 重點摘要:**

- **核心功能**: 即時視覺化訓練指標
- **潛在問題**: 日誌檔案可能佔用大量空間
- **最佳使用情境**: 模型開發與除錯

## <a id="超參數調校"></a>🎛️ 超參數調校

使用 Scikit-Learn 的 RandomizedSearchCV 進行超參數搜尋。

### 範例 11: 超參數調校

```python
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[28, 28]):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
```

**✅ 程式碼逐行解析：**

1. `def build_model(...)`: 定義模型建構函數
2. `keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)`: 包裝為 Scikit-Learn 分類器
3. `param_distribs = {...}`: 定義參數分佈
4. `rnd_search_cv = RandomizedSearchCV(...)`: 建立隨機搜尋物件
5. `rnd_search_cv.fit(...)`: 執行超參數調校

**🎯 重點摘要:**

- **核心功能**: 自動化超參數最佳化
- **潛在問題**: 計算成本高，需要大量時間
- **最佳使用情境**: 模型效能最佳化

## ❓ 常見問答

**Q: Sequential API 與 Functional API 的差異？**
A: Sequential API 適合簡單的線性堆疊架構，而 Functional API 能處理複雜的網路結構如多輸入輸出或共享層。

**Q: 如何避免過擬合？**
A: 使用 Dropout、早期停止、正規化技巧，或增加訓練資料。

**Q: 激活函數選擇的考量？**
A: ReLU 適合隱藏層避免梯度消失，Softmax 適用於多分類輸出層。

## 🏷️ 推薦標籤

#Keras #神經網路 #人工智慧 #機器學習 #深度學習 #Python教學 #程式設計 #TensorFlow #AI #資料科學