<!-- meta-title: 使用 Keras 建構神經網路：從基礎到進階實作 -->
<!-- meta-description: 完整教學：學習使用 Keras 建構神經網路，從生物神經元到人工神經網路，包含分類、回歸、複雜模型設計、超參數調優等實戰技巧。 -->
<!-- meta-keywords: Keras, 神經網路, 機器學習, TensorFlow, 深度學習, 人工智慧, 程式設計, 教學 -->
<!-- meta-hashtags: #Keras #神經網路 #機器學習 #深度學習 #TensorFlow #人工智慧 #程式設計 #教學 #編程 #開發 #技術分享 #學習筆記 #程式開發者 #軟體工程 -->

# 🐍 使用 Keras 建構神經網路：從基礎到進階實作

神經網路是現代人工智慧的核心技術，而 Keras 作為 TensorFlow 的高階 API，讓建構和訓練神經網路變得簡單直觀。本教學將帶您從生物神經元的概念出發，逐步深入人工神經網路的實作，包含分類、回歸、複雜模型設計、超參數調優等完整流程。

## 📝 本文目錄
- [🎯 關鍵重點 (Key Takeaways)](#key-takeaways)
- [🔧 環境設定與準備](#setup)
- [🧠 從生物神經元到人工神經網路](#biological-to-artificial)
  - [感知器 (The Perceptron)](#perceptron)
  - [活化函數 (Activation Functions)](#activation-functions)
- [📊 回歸多層感知器 (Regression MLPs)](#regression-mlps)
- [🏷️ 分類多層感知器 (Classification MLPs)](#classification-mlps)
- [🚀 使用 Keras 實作多層感知器](#implementing-mlps-keras)
  - [使用序列 API 建構影像分類器](#sequential-api-classifier)
  - [使用序列 API 建構回歸模型](#sequential-api-regression)
  - [使用函數式 API 建構複雜模型](#functional-api)
  - [使用子類化 API 建構動態模型](#subclassing-api)
  - [模型的儲存與載入](#saving-models)
  - [回呼函數 (Callbacks)](#callbacks)
  - [使用 TensorBoard 視覺化](#tensorboard)
- [⚙️ 微調神經網路超參數](#hyperparameter-tuning)
- [💡 總結與最佳實踐](#conclusion)
- [❓ 常見問答 (FAQ)](#faq)
- [🏷️ 推薦標籤 (Suggested Hashtags)](#hashtags)

<a id="key-takeaways"></a>
## 🎯 關鍵重點 (Key Takeaways)
- 神經網路由互連的神經元組成，能學習複雜的非線性關係
- Keras 提供三種建構模型的 API：序列、函數式和子類化
- 活化函數決定神經元的輸出行為，ReLU 是隱藏層的常用選擇
- 過擬合是神經網路常見問題，可透過正則化、Dropout 等技術解決
- 超參數調優對於模型效能至關重要，可使用 Keras Tuner 等工具

---

<a id="setup"></a>
## 🔧 環境設定與準備
💡 **實際應用情境：** 在開始建構神經網路之前，需要確保開發環境已正確設定，包括必要的程式庫版本檢查。

### 範例 1: 環境檢查與設定
```python
import sys
import matplotlib.pyplot as plt

# 檢查 Python 版本
assert sys.version_info >= (3, 7)

# 設定圖表字體大小
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 匯入 sys 模組，用於檢查 Python 版本
2. `assert sys.version_info >= (3, 7)`: 確保 Python 版本至少為 3.7
3. `plt.rc('font', size=14)`: 設定 Matplotlib 字體大小為 14
4. `plt.rc('axes', labelsize=14, titlesize=14)`: 設定軸標籤和標題字體大小
5. `plt.rc('legend', fontsize=14)`: 設定圖例字體大小
6. `plt.rc('xtick', labelsize=10)`: 設定 x 軸刻度字體大小
7. `plt.rc('ytick', labelsize=10)`: 設定 y 軸刻度字體大小

**🎯 重點摘要:**

- **核心功能**: 確保開發環境符合最低要求，並設定一致的視覺化樣式
- **潛在問題**: 版本不相容可能導致程式錯誤
- **最佳使用情境**: 在每個機器學習專案開始時執行環境檢查

---

<a id="biological-to-artificial"></a>
## 🧠 從生物神經元到人工神經網路

<a id="perceptron"></a>
### 感知器 (The Perceptron)
💡 **實際應用情境：** 感知器是最簡單的人工神經元，能解決線性可分的二元分類問題，如鳶尾花分類。

### 範例 2: 使用 Scikit-Learn 實作感知器
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

# 載入鳶尾花資料集
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)  # 鳶尾花 setosa

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練感知器
per_clf = Perceptron(random_state=42)
per_clf.fit(X_train, y_train)

# 進行預測
X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import load_iris`: 匯入鳶尾花資料集
2. `iris = load_iris(as_frame=True)`: 載入資料集並轉為 DataFrame 格式
3. `X = iris.data[["petal length (cm)", "petal width (cm)"]].values`: 選取花瓣長度和寬度作為特徵
4. `y = (iris.target == 0)`: 將目標轉為二元分類 (是否為 setosa)
5. `per_clf = Perceptron(random_state=42)`: 建立感知器模型
6. `per_clf.fit(X_train, y_train)`: 訓練模型
7. `y_pred = per_clf.predict(X_new)`: 對新資料進行預測

**🎯 重點摘要:**

- **核心功能**: 實作簡單的線性分類器
- **潛在問題**: 只能處理線性可分資料，無法學習複雜模式
- **最佳使用情境**: 快速原型設計或作為基準模型

---

<a id="activation-functions"></a>
### 活化函數 (Activation Functions)
💡 **實際應用情境：** 活化函數決定神經元的輸出，選擇合適的活化函數對於模型效能至關重要。

### 範例 3: 常見活化函數視覺化
```python
import numpy as np
from scipy.special import expit as sigmoid

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

max_z = 4.5
z = np.linspace(-max_z, max_z, 200)

plt.figure(figsize=(11, 3.1))

plt.subplot(121)
plt.plot([-max_z, 0], [0, 0], "r-", linewidth=2, label="Heaviside")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.plot([0, 0], [0, 1], "r-", linewidth=0.5)
plt.plot([0, max_z], [1, 1], "r-", linewidth=2)
plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, np.tanh(z), "b-", linewidth=1, label="Tanh")
plt.grid(True)
plt.title("活化函數")
plt.axis([-max_z, max_z, -1.65, 2.4])
plt.legend(loc="lower right", fontsize=13)

plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Heaviside")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=1, label="Tanh")
plt.grid(True)
plt.title("導數")
plt.axis([-max_z, max_z, -0.2, 1.2])

plt.show()
```

**✅ 程式碼逐行解析：**

1. `def relu(z): return np.maximum(0, z)`: 定義 ReLU 活化函數
2. `def derivative(f, z, eps=0.000001)`: 定義數值微分函數
3. `z = np.linspace(-max_z, max_z, 200)`: 產生輸入值範圍
4. `plt.subplot(121)`: 建立第一個子圖
5. `plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")`: 繪製 ReLU 函數
6. `plt.subplot(122)`: 建立第二個子圖
7. `plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")`: 繪製 Sigmoid 導數

**🎯 重點摘要:**

- **核心功能**: 視覺化不同活化函數及其導數
- **潛在問題**: 某些函數可能導致梯度消失或爆炸
- **最佳使用情境**: 選擇適合任務的活化函數

---

<a id="regression-mlps"></a>
## 📊 回歸多層感知器 (Regression MLPs)
💡 **實際應用情境：** 回歸 MLP 用於預測連續值，如房價預測。

### 範例 4: 使用 Scikit-Learn 建構回歸 MLP
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 載入加州房價資料集
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

# 建立 MLP 回歸模型
mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)

# 計算 RMSE
rmse = root_mean_squared_error(y_valid, y_pred)
print(f"驗證集 RMSE: {rmse:.4f}")
```

**✅ 程式碼逐行解析：**

1. `housing = fetch_california_housing()`: 載入加州房價資料集
2. `X_train, X_valid, y_train, y_valid = train_test_split(...)`: 分割訓練和驗證資料
3. `mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)`: 建立三層隱藏層的 MLP
4. `pipeline = make_pipeline(StandardScaler(), mlp_reg)`: 建立包含標準化和 MLP 的管道
5. `pipeline.fit(X_train, y_train)`: 訓練模型
6. `y_pred = pipeline.predict(X_valid)`: 進行預測
7. `rmse = root_mean_squared_error(y_valid, y_pred)`: 計算均方根誤差

**🎯 重點摘要:**

- **核心功能**: 使用 MLP 進行回歸預測
- **潛在問題**: 需要適當的資料預處理和超參數調優
- **最佳使用情境**: 處理複雜的非線性回歸問題

---

<a id="classification-mlps"></a>
## 🏷️ 分類多層感知器 (Classification MLPs)
💡 **實際應用情境：** 分類 MLP 用於將輸入分類到不同類別，如影像分類。

### 範例 5: 使用 Scikit-Learn 建構分類 MLP
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 載入鳶尾花資料集
iris = load_iris()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    iris.data, iris.target, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42)

# 建立 MLP 分類器
mlp_clf = MLPClassifier(hidden_layer_sizes=[5], max_iter=10_000, random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_clf)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_valid, y_valid)
print(f"驗證集準確率: {accuracy:.4f}")
```

**✅ 程式碼逐行解析：**

1. `iris = load_iris()`: 載入鳶尾花資料集
2. `X_train, X_valid, y_train, y_valid = train_test_split(...)`: 分割資料
3. `mlp_clf = MLPClassifier(hidden_layer_sizes=[5], max_iter=10_000, random_state=42)`: 建立單隱藏層 MLP
4. `pipeline = make_pipeline(StandardScaler(), mlp_clf)`: 建立管道
5. `pipeline.fit(X_train, y_train)`: 訓練模型
6. `accuracy = pipeline.score(X_valid, y_valid)`: 計算準確率

**🎯 重點摘要:**

- **核心功能**: 多類別分類任務
- **潛在問題**: 過擬合風險，需要適當的正則化
- **最佳使用情境**: 中小型分類資料集

<a id="implementing-mlps-keras"></a>
## 🚀 使用 Keras 實作多層感知器

<a id="sequential-api-classifier"></a>
### 使用序列 API 建構影像分類器
💡 **實際應用情境：** Fashion MNIST 是常見的影像分類基準資料集，用於測試分類演算法。

### 範例 6: 載入和預處理 Fashion MNIST 資料集
```python
import tensorflow as tf

# 載入 Fashion MNIST 資料集
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# 正規化像素值
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

# 類別名稱
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

**✅ 程式碼逐行解析：**

1. `fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()`: 載入資料集
2. `X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]`: 分割訓練資料
3. `X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]`: 分割驗證資料
4. `X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.`: 正規化到 0-1 範圍
5. `class_names = [...]`: 定義類別名稱

**🎯 重點摘要:**

- **核心功能**: 準備影像分類資料集
- **潛在問題**: 忘記正規化可能導致訓練不穩定
- **最佳使用情境**: 任何影像分類任務的起點

---

### 範例 7: 建構和編譯序列模型
```python
tf.random.set_seed(42)

# 建構模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 編譯模型
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# 顯示模型摘要
model.summary()
```

**✅ 程式碼逐行解析：**

1. `tf.random.set_seed(42)`: 設定隨機種子確保重現性
2. `tf.keras.layers.Flatten(input_shape=[28, 28])`: 將 28x28 影像展平為 784 維向量
3. `tf.keras.layers.Dense(300, activation="relu")`: 第一隱藏層，300 個神經元，ReLU 活化
4. `tf.keras.layers.Dense(100, activation="relu")`: 第二隱藏層，100 個神經元
5. `tf.keras.layers.Dense(10, activation="softmax")`: 輸出層，10 個類別，softmax 活化
6. `model.compile(...)`: 編譯模型，指定損失函數、優化器和指標
7. `model.summary()`: 顯示模型結構和參數數量

**🎯 重點摘要:**

- **核心功能**: 建構標準的密集神經網路
- **潛在問題**: 層數和神經元數量需根據任務調整
- **最佳使用情境**: 大多數分類和回歸任務

---

### 範例 8: 訓練和評估模型
```python
# 訓練模型
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

# 評估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"測試準確率: {test_accuracy:.4f}")
```

**✅ 程式碼逐行解析：**

1. `history = model.fit(...)`: 訓練模型 30 個 epoch，使用驗證資料
2. `test_loss, test_accuracy = model.evaluate(X_test, y_test)`: 在測試集上評估
3. `print(f"測試準確率: {test_accuracy:.4f}")`: 輸出測試準確率

**🎯 重點摘要:**

- **核心功能**: 訓練和評估神經網路模型
- **潛在問題**: 過擬合，需監控驗證指標
- **最佳使用情境**: 模型訓練和效能評估

---

<a id="sequential-api-regression"></a>
### 使用序列 API 建構回歸模型
💡 **實際應用情境：** 使用神經網路進行回歸預測，如加州房價預測。

### 範例 9: 建構回歸 MLP
```python
# 載入資料
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

# 建構模型
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)  # 回歸任務無活化函數
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)

# 訓練模型
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

# 進行預測
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
```

**✅ 程式碼逐行解析：**

1. `norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])`: 建立歸一化層
2. `tf.keras.layers.Dense(1)`: 回歸輸出層無活化函數
3. `optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)`: 使用 Adam 優化器
4. `model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])`: 編譯回歸模型
5. `norm_layer.adapt(X_train)`: 適應訓練資料進行歸一化
6. `y_pred = model.predict(X_new)`: 對新資料進行預測

**🎯 重點摘要:**

- **核心功能**: 神經網路回歸預測
- **潛在問題**: 需注意輸出範圍和損失函數選擇
- **最佳使用情境**: 複雜非線性回歸問題

---

<a id="functional-api"></a>
### 使用函數式 API 建構複雜模型
💡 **實際應用情境：** Wide & Deep 模型結合寬路徑和深路徑，能同時學習記憶和泛化。

### 範例 10: Wide & Deep 模型
```python
tf.keras.backend.clear_session()
tf.random.set_seed(42)

# 定義各層
normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

# 建構模型
input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])

# 編譯和訓練
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
normalization_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
```

**✅ 程式碼逐行解析：**

1. `input_ = tf.keras.layers.Input(shape=X_train.shape[1:])`: 定義輸入層
2. `normalized = normalization_layer(input_)`: 正規化輸入
3. `hidden1 = hidden_layer1(normalized)`: 第一隱藏層
4. `concat = concat_layer([normalized, hidden2])`: 串接寬路徑和深路徑
5. `model = tf.keras.Model(inputs=[input_], outputs=[output])`: 建立函數式模型
6. `normalization_layer.adapt(X_train)`: 適應正規化參數

**🎯 重點摘要:**

- **核心功能**: 建構多輸入多輸出複雜模型
- **潛在問題**: 拓撲複雜度增加維護難度
- **最佳使用情境**: 需要特殊架構的進階模型

---

<a id="subclassing-api"></a>
### 使用子類化 API 建構動態模型
💡 **實際應用情境：** 子類化允許完全自訂模型行為，如動態架構或自訂訓練邏輯。

### 範例 11: 自訂 WideAndDeepModel 類別
```python
class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output

# 建立和訓練模型
tf.random.set_seed(42)
model = WideAndDeepModel(30, activation="relu", name="my_cool_model")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=optimizer,
              metrics=["RootMeanSquaredError", "RootMeanSquaredError"])
```

**✅ 程式碼逐行解析：**

1. `class WideAndDeepModel(tf.keras.Model)`: 繼承 tf.keras.Model
2. `def __init__(self, units=30, activation="relu", **kwargs)`: 初始化方法
3. `def call(self, inputs)`: 前向傳播邏輯
4. `input_wide, input_deep = inputs`: 解包輸入
5. `concat = tf.keras.layers.concatenate([norm_wide, hidden2])`: 串接特徵
6. `return output, aux_output`: 返回主輸出和輔助輸出

**🎯 重點摘要:**

- **核心功能**: 完全自訂模型架構和行為
- **潛在問題**: 複雜度高，易出錯
- **最佳使用情境**: 需要高度自訂邏輯的研究性模型

---

<a id="saving-models"></a>
### 模型的儲存與載入
💡 **實際應用情境：** 儲存訓練好的模型以便後續使用或部署。

### 範例 12: 儲存和載入 Keras 模型
```python
# 儲存模型
model.save("my_model.keras")

# 載入模型
loaded_model = tf.keras.models.load_model("my_model.keras")

# 儲存權重
model.save_weights("my_weights.weights.h5")

# 載入權重
model.load_weights("my_weights.weights.h5")
```

**✅ 程式碼逐行解析：**

1. `model.save("my_model.keras")`: 以 Keras 格式儲存完整模型
2. `loaded_model = tf.keras.models.load_model("my_model.keras")`: 載入模型
3. `model.save_weights("my_weights.weights.h5")`: 僅儲存權重
4. `model.load_weights("my_weights.weights.h5")`: 載入權重

**🎯 重點摘要:**

- **核心功能**: 模型持久化
- **潛在問題**: 版本相容性問題
- **最佳使用情境**: 模型部署和重用

---

<a id="callbacks"></a>
### 回呼函數 (Callbacks)
💡 **實際應用情境：** 回呼函數允許在訓練過程中執行自訂邏輯，如早期停止或檢查點儲存。

### 範例 13: 使用常見回呼函數
```python
# 檢查點回呼
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints.weights.h5",
                                                   save_weights_only=True)

# 早期停止回呼
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

# 自訂回呼
class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")

# 在訓練中使用
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=100,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[checkpoint_cb, early_stopping_cb, PrintValTrainRatioCallback()])
```

**✅ 程式碼逐行解析：**

1. `checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(...)`: 定期儲存最佳權重
2. `early_stopping_cb = tf.keras.callbacks.EarlyStopping(...)`: 早期停止訓練
3. `class PrintValTrainRatioCallback(tf.keras.callbacks.Callback)`: 自訂回呼類別
4. `def on_epoch_end(self, epoch, logs)`: 每個 epoch 結束時執行的方法
5. `callbacks=[...]` : 在 fit 方法中指定回呼列表

**🎯 重點摘要:**
回呼函數的使用能有效管理訓練過程，以下為常見回呼函數的重點摘要：
- **ModelCheckpoint**: 定期儲存模型權重，避免訓練中斷導致的損失
- **EarlyStopping**: 監控驗證指標，當指標不再改善時自動停止訓練
- **自訂回呼**: 可根據需求自訂回呼函數，如紀錄特定指標、動態調整學習率等

--- 

### 範例 13b: 指數學習率搜尋 (Learning-rate finder) 回呼
```python
import tensorflow as tf
class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor=1.005):
        super().__init__()
        self.factor = factor
        self.lrs = []
        self.losses = []

    def on_train_begin(self, logs=None):
        self.lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.lrs.append(self.lr)
        self.losses.append(logs.get("loss"))
        self.lr *= self.factor
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
```

**✅ 程式碼逐行解析：**
1. `class ExponentialLearningRate(...)`: 定義自訂回呼類別用於逐批提高學習率  
2. `self.factor = factor`: 每個 batch 增長因子  
3. `on_train_begin`: 取得並儲存優化器初始學習率  
4. `on_batch_end`: 每 batch 記錄目前 lr 與 loss，並將 lr 乘上因子後更新優化器  

**🎯 重點摘要:**
- 快速探索合適學習率範圍，找出 loss 開始急劇上升前的最佳 lr。  
- 使用方法：在短訓練中加入此回呼，訓練結束後繪製 lrs vs losses。

範例使用與繪圖：
```python
# 範例：lr finder 使用方式
lr_cb = ExponentialLearningRate(factor=1.01)
model.compile(... )  # 如前面所示
history = model.fit(X_train, y_train, epochs=1, callbacks=[lr_cb], batch_size=128)
import matplotlib.pyplot as plt
plt.semilogx(lr_cb.lrs, lr_cb.losses)
plt.xlabel("learning rate")
plt.ylabel("loss")
plt.show()
```

--- 

<a id="tensorboard"></a>
### 使用 TensorBoard 視覺化
💡 **實際應用情境：** TensorBoard 提供豐富的視覺化工具來監控訓練過程和模型效能。

### 範例 14: 設定 TensorBoard 回呼
```python
from pathlib import Path
from time import strftime

def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir()

# 建立 TensorBoard 回呼
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir,
                                                profile_batch=(100, 200))

# 在訓練中使用
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])

# 啟動 TensorBoard
%load_ext tensorboard
%tensorboard --logdir=./my_logs
```

**✅ 程式碼逐行解析：**

1. `def get_run_logdir(root_logdir="my_logs")`: 建立唯一的執行日誌目錄
2. `run_logdir = get_run_logdir()`: 取得當前執行的日誌目錄
3. `tensorboard_cb = tf.keras.callbacks.TensorBoard(...)`: 建立 TensorBoard 回呼
4. `%load_ext tensorboard`: 載入 TensorBoard Jupyter 擴充
5. `%tensorboard --logdir=./my_logs`: 啟動 TensorBoard 伺服器

**🎯 重點摘要:**
TensorBoard 回呼的使用重點如下：
- **自動紀錄**: 搭配 Keras 回呼自動紀錄訓練過程中的指標、損失等資訊
- **視覺化**: 提供豐富的視覺化工具，如曲線圖、直方圖、影像等
- **性能分析**: 可視化每層的權重分佈、激活值等，協助除錯和優化模型

--- 

### 範例 14b: 使用 tf.summary 在自訂訓練回圈或回呼中記錄指標
```python
import tensorflow as tf
from pathlib import Path
from time import strftime

def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir()
file_writer = tf.summary.create_file_writer(str(run_logdir))

# 在自訂訓練迴圈或回呼內使用
with file_writer.as_default():
    tf.summary.scalar("train/loss", 0.1234, step=1)
    tf.summary.scalar("val/accuracy", 0.8765, step=1)
```

**✅ 程式碼逐行解析：**
1. `get_run_logdir(...)`: 建立唯一執行日誌路徑  
2. `tf.summary.create_file_writer(...)`: 建立 summary 寫入器（TensorBoard 可讀取）  
3. `tf.summary.scalar(...)`: 在指定 step 寫入 scalar 指標  

**🎯 重點摘要:**
- tf.summary 可與自訂訓練迴圈或回呼無縫整合，提供更細緻的監控（例如 batch 級別指標）。

--- 

<a id="hyperparameter-tuning"></a>
## ⚙️ 微調神經網路超參數
💡 **實際應用情境：** 超參數調優對於提升模型效能至關重要，Keras Tuner 提供自動化工具。

### 範例 15: 使用 Keras Tuner 進行超參數搜尋
```python
import keras_tuner as kt

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model

# 執行隨機搜尋
random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)
random_search_tuner.search(X_train, y_train, epochs=10,
                           validation_data=(X_valid, y_valid))

# 取得最佳模型
top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]
```

**✅ 程式碼逐行解析：**

1. `def build_model(hp)`: 定義模型建構函數，接受超參數物件
2. `n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)`: 定義隱藏層數量搜尋空間
3. `n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)`: 定義神經元數量搜尋空間
4. `learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")`: 定義學習率搜尋空間
5. `optimizer = hp.Choice("optimizer", values=["sgd", "adam"])`: 定義優化器選擇
6. `random_search_tuner = kt.RandomSearch(...)`: 建立隨機搜尋調優器
7. `random_search_tuner.search(...)`: 執行超參數搜尋
8. `top3_models = random_search_tuner.get_best_models(num_models=3)`: 取得前三名模型

**🎯 重點摘要:**
Keras Tuner 超參數調優的重點摘要如下：
- **自動化搜尋**: 可自動搜尋最佳超參數組合，節省時間與計算資源
- **多種搜尋策略**: 提供隨機搜尋、貝葉斯優化等多種策略
- **易於整合**: 與 Keras 模型無縫整合，使用簡單

--- 

### 範例 15b: 使用 HyperModel 與多種搜尋策略 (RandomSearch / Hyperband / Bayesian)
```python
import keras_tuner as kt
import tensorflow as tf

class MyClassificationHyperModel(kt.HyperModel):
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.input_shape))
        for i in range(hp.Int("n_layers", 1, 4, default=2)):
            model.add(tf.keras.layers.Dense(
                units=hp.Int(f"units_{i}", 32, 256, step=32, default=64),
                activation="relu"))
        model.add(tf.keras.layers.Dense(self.n_classes, activation="softmax"))
        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

hm = MyClassificationHyperModel(input_shape=(28,28), n_classes=10)

# Random Search
rnd = kt.RandomSearch(hm, objective="val_accuracy", max_trials=5,
                     directory="kt_dir", project_name="rnd")
rnd.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
best_rnd = rnd.get_best_models(num_models=1)[0]

# Hyperband
hb = kt.Hyperband(hm, objective="val_accuracy", max_epochs=20,
                  directory="kt_dir", project_name="hyperband")
hb.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
best_hb = hb.get_best_models(num_models=1)[0]

# Bayesian Optimization (若安裝)
try:
    bo = kt.BayesianOptimization(hm, objective="val_accuracy", max_trials=10,
                                 directory="kt_dir", project_name="bayes")
    bo.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
    best_bo = bo.get_best_models(num_models=1)[0]
except Exception:
    best_bo = None
```

**✅ 程式碼逐行解析：**

1. `class MyClassificationHyperModel(...)`: 利用 HyperModel 封裝模型建構邏輯與搜尋空間  
2. `hp.Int / hp.Float`: 定義離散與連續超參數空間  
3. `kt.RandomSearch / kt.Hyperband / kt.BayesianOptimization`: 不同搜尋策略，根據資源與需求選擇  
4. `get_best_models(...)`: 取回最佳訓練模型（可進一步微調或評估）

**🎯 重點摘要:**
- HyperModel 能把複雜搜尋空間與建模邏輯封裝起來，便於重複實驗。  
- 選擇搜尋器時需考量計算資源：Hyperband 對早停友好，Bayesian 在樣本效率上較佳。  
- 儲存與對比不同搜尋結果（project_name）有助於實驗管理。

---

<a id="conclusion"></a>
## 💡 總結與最佳實踐

神經網路是強大的機器學習工具，能學習複雜的模式和關係。本教學涵蓋了從基礎概念到進階實作的完整流程。

**最佳實踐：**
- 從簡單模型開始，逐步增加複雜度
- 始終使用驗證集監控過擬合
- 正確預處理資料，包括正規化和特徵工程
- 使用適當的激活函數和優化器
- 利用回呼函數進行訓練管理
- 定期儲存模型檢查點
- 使用 TensorBoard 監控訓練過程
- 進行系統性的超參數調優

---

<a id="faq"></a>
## ❓ 常見問答 (FAQ)

**Q: 神經網路和傳統機器學習演算法有何差異？**
A: 神經網路能自動學習特徵表示，而傳統演算法需要手動特徵工程。神經網路在處理複雜非線性問題時表現更優異，但需要更多資料和計算資源。

**Q: 如何選擇隱藏層數量和神經元數量？**
A: 從簡單架構開始，透過交叉驗證比較不同配置。過少的層/神經元可能欠擬合，過多則可能過擬合。

**Q: ReLU 為什麼是隱藏層的常用激活函數？**
A: ReLU 計算簡單，能有效解決梯度消失問題，且在實務中表現良好。不過在某些情況下可能需要考慮 Leaky ReLU 或 ELU。

**Q: 如何處理過擬合問題？**
A: 使用 Dropout、正則化、早期停止、資料擴增等技術。增加訓練資料量也是有效方法。

**Q: 函數式 API 和序列 API 有何區別？**
A: 序列 API 適用於簡單的層疊結構，函數式 API 支援複雜拓撲如多輸入輸出、跳躍連接等。

---

<a id="hashtags"></a>
## 🏷️ 推薦標籤 (Suggested Hashtags)

#Keras #神經網路 #機器學習 #深度學習 #TensorFlow #人工智慧 #程式設計 #教學 #編程 #開發 #技術分享 #學習筆記 #程式開發者 #軟體工程