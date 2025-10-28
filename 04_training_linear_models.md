<!-- meta-title: Python 機器學習：訓練線性模型完整指南 - 從基礎到進階實戰 -->
<!-- meta-description: 深入探討線性迴歸、梯度下降、正規化等核心概念，包含實戰程式碼與逐行解析，適合機器學習初學者與進階開發者 -->
<!-- meta-keywords: Python, 機器學習, 線性迴歸, 梯度下降, 正規化, Ridge, Lasso, 邏輯迴歸, Softmax -->
<!-- meta-hashtags: #Python #機器學習 #程式設計 #教學 #線性模型 #數據科學 #AI開發 -->

# 🐍 Training Linear Models：機器學習線性模型訓練完整指南

線性模型是機器學習的基石，理解如何訓練線性模型對於掌握更複雜的演算法至關重要。本教學將帶您深入了解各種線性模型訓練技術，從最簡單的線性迴歸到複雜的正規化方法。

## 📝 本文目錄
- [關鍵重點](#key-takeaways)
- [環境設定與資料準備](#setup)
- [線性迴歸 (Linear Regression)](#linear-regression)
  - [正規方程式 (Normal Equation)](#normal-equation)
  - [奇異值分解 (SVD)](#svd)
- [梯度下降 (Gradient Descent)](#gradient-descent)
  - [批次梯度下降 (Batch GD)](#batch-gd)
  - [隨機梯度下降 (Stochastic GD)](#stochastic-gd)
  - [小批次梯度下降 (Mini-batch GD)](#minibatch-gd)
- [多項式迴歸 (Polynomial Regression)](#polynomial-regression)
- [學習曲線 (Learning Curves)](#learning-curves)
- [正規化線性模型](#regularized-models)
  - [Ridge 迴歸](#ridge-regression)
  - [Lasso 迴歸](#lasso-regression)
  - [Elastic Net](#elastic-net)
  - [Early Stopping](#early-stopping)
- [邏輯迴歸 (Logistic Regression)](#logistic-regression)
- [Softmax 迴歸](#softmax-regression)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **線性迴歸**可使用正規方程式或梯度下降來訓練
- **梯度下降**有三種變體：批次、隨機、小批次，各有優缺點
- **多項式迴歸**可以擬合非線性資料，但需注意過擬合
- **正規化技術** (Ridge、Lasso、Elastic Net) 可以防止過擬合
- **邏輯迴歸**用於二元分類，**Softmax 迴歸**用於多類別分類
- **學習曲線**是診斷模型效能的重要工具

## <a id="setup"></a>🔧 環境設定與資料準備

💡 **實際應用情境：** 在開始任何機器學習專案前，我們需要設定適當的環境並載入必要的函式庫。

### 範例 1: 環境設定

```python
import sys
import sklearn
import matplotlib.pyplot as plt
from pathlib import Path

assert sys.version_info >= (3, 7)
from packaging import version
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

IMAGES_PATH = Path() / "images" / "training_linear_models"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1.  `sys`, `sklearn`, `matplotlib.pyplot`, `pathlib.Path`: 匯入必要的函式庫。
2.  `assert sys.version_info >= (3, 7)`: 檢查 Python 版本是否符合要求。
3.  `assert version.parse(sklearn.__version__) >= version.parse("1.0.1")`: 檢查 Scikit-Learn 版本。
4.  `plt.rc(...)`: 設定 Matplotlib 的預設字體大小，使圖表更美觀。
5.  `IMAGES_PATH`: 定義儲存圖片的資料夾路徑。
6.  `save_fig()`: 定義一個函式，用於儲存高解析度的圖表。

**🎯 重點摘要:**

- **核心功能**: 建立一個標準化的環境，確保程式碼在不同機器上都能有一致的表現。
- **最佳使用情境**: 任何機器學習專案的初始設定。

## <a id="linear-regression"></a>線性迴歸 (Linear Regression)

### <a id="normal-equation"></a>正規方程式 (Normal Equation)

💡 **實際應用情境：** 當資料集不大時，正規方程式是計算線性迴歸模型參數最直接的方法。

### 範例 2: 使用正規方程式進行線性迴歸

```python
import numpy as np

np.random.seed(42)  # 設定隨機種子以確保結果可重現
m = 100  # 訓練樣本數量
X = 2 * np.random.rand(m, 1)  # 生成 100 個介於 0 到 2 之間的隨機數作為特徵

# 根據線性方程式 y = 4 + 3x + noise 生成目標值
y = 4 + 3 * X + np.random.randn(m, 1)  # 加入高斯雜訊

# 檢查資料的形狀
print(f'Shape of X: {X.shape}')
print(f'{X[:10]}')

print(f'Shape of y: {y.shape}')
print(f'{y[:10]}')

# 視覺化生成的資料
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
save_fig("generated_data_plot")
plt.show()

# 使用正規方程式計算最佳參數
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)  # 為每個實例添加 x0 = 1
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# 檢視添加偏差項後的資料
print(f'X: \n{X[:10]}')
print(f'X_b: \n{X_b[:10]}')

# 顯示計算出的最佳參數
theta_best  # 由正規方程式計算出的 theta (權重)
```

**✅ 程式碼逐行解析：**

1.  `np.random.seed(42)`: 設定隨機種子以確保結果可重現。
2.  `m = 100`: 定義訓練樣本數量。
3.  `X = 2 * np.random.rand(m, 1)`: 產生 100 個介於 0 和 2 之間的隨機數作為特徵。
4.  `y = 4 + 3 * X + np.random.randn(m, 1)`: 根據線性方程式 `y = 4 + 3x` 生成目標值，並加入高斯雜訊。
5.  `print(f'Shape of X: {X.shape}')`: 顯示特徵矩陣的形狀 (100, 1)。
6.  `print(f'{X[:10]}')`: 顯示前 10 個訓練樣本的特徵值。
7.  `plt.plot(X, y, "b.")`: 繪製散點圖以視覺化訓練資料。
8.  `X_b = add_dummy_feature(X)`: 為每個實例添加 `x0 = 1` 的偏差特徵，將形狀從 (100, 1) 轉換為 (100, 2)。
9.  `theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y`: 使用正規方程式直接計算最佳參數 `theta`。

**🎯 重點摘要:**

- **核心功能**: 直接計算能最小化成本函式的模型參數。
- **潛在問題**: 當特徵數量非常大時，計算逆矩陣會非常慢。
- **最佳使用情境**: 特徵數量不多的小型資料集。

### 範例 2.1: 使用計算出的參數進行預測

```python
# 建立新的資料點進行預測
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new)  # 為新資料添加 x0 = 1
y_predict = X_new_b @ theta_best
y_predict

# 檢查資料形狀
print(f'X_new:\n{X_new}')
print(f'X_new_b:\n{X_new_b}')
print(f'Shape of X_new: {X_new.shape}')
print(f'Shape of X_new_b: {X_new_b.shape}')

# 視覺化預測結果
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.", label="Training data")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.legend(loc="upper left")
save_fig("linear_model_predictions_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `X_new = np.array([[0], [2]])`: 建立兩個新的資料點 (x=0 和 x=2) 用於預測。
2.  `X_new_b = add_dummy_feature(X_new)`: 為新資料添加偏差項，形狀從 (2, 1) 變為 (2, 2)。
3.  `y_predict = X_new_b @ theta_best`: 使用矩陣乘法計算預測值：y = X_b · θ。
4.  `print(f'Shape of X_new: {X_new.shape}')`: 顯示新資料的形狀資訊。
5.  `plt.plot(X_new, y_predict, "r-", label="Predictions")`: 繪製紅色的預測線。
6.  `plt.plot(X, y, "b.", label="Training data")`: 繪製藍色的訓練資料點。

**🎯 重點摘要:**

- **核心功能**: 使用訓練好的參數對新資料進行預測，並視覺化結果。
- **關鍵步驟**: 確保新資料也添加了偏差項 (x0 = 1)。
- **最佳使用情境**: 驗證模型的預測能力並理解線性迴歸的預測過程。

### <a id="svd"></a>奇異值分解 (SVD)

💡 **實際應用情境：** Scikit-Learn 的 `LinearRegression` 類別使用 SVD 方法，它比正規方程式更有效率且數值穩定。

### 範例 3: 使用 Scikit-Learn 的 `LinearRegression`

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
```

**✅ 程式碼逐行解析：**

1.  `lin_reg = LinearRegression()`: 建立一個 `LinearRegression` 模型。
2.  `lin_reg.fit(X, y)`: 使用 SVD 方法訓練模型。
3.  `lin_reg.intercept_, lin_reg.coef_`: 取得模型的截距和係數。

**📊 LinearRegression 參數說明：**

`lin_reg.intercept_` 和 `lin_reg.coef_` 是線性迴歸模型學習到的參數：

#### `lin_reg.intercept_`
- 代表**截距** (*偏差項*) θ₀
- 這是當所有特徵都為零時的預測值
- 對應於迴歸線與 y 軸的交點

#### `lin_reg.coef_`
- 包含每個特徵的**係數** (*權重*)
- 對於我們的單特徵案例，`lin_reg.coef_[0]` 是 θ₁，即直線的斜率
- 顯示特徵增加一個單位時，預測值的變化量

#### 線性方程式
它們共同定義了線性模型：**ŷ = θ₀ + θ₁x₁**

其中：
- `lin_reg.intercept_` = θ₀ 
- `lin_reg.coef_[0]` = θ₁

這些值應該非常接近我們使用正規方程式計算的 `theta_best` 值，因為兩種方法都解決相同的線性迴歸問題，只是使用不同的演算法 (SVD vs. 正規方程式)。

**🎯 重點摘要:**

- **核心功能**: 使用 SVD 分解來求解線性迴歸，計算效率高且數值穩定。
- **最佳使用情境**: 大多數線性迴歸問題的標準選擇。

### 範例 3.1: 使用 LinearRegression 進行預測

```python
# 使用訓練好的模型進行預測
lin_reg.predict(X_new)
```

**✅ 程式碼逐行解析：**

1.  `lin_reg.predict(X_new)`: 使用 Scikit-Learn 的預測方法，內部會自動處理偏差項的添加。

### 範例 3.2: 直接使用 NumPy 的 lstsq 函數

`LinearRegression` 類別基於 `scipy.linalg.lstsq()` 函數 (名稱代表 "least squares")，你也可以直接呼叫它：

```python
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
```

**✅ 程式碼逐行解析：**

1.  `np.linalg.lstsq()`: NumPy 的最小平方法函數。
2.  `rcond=1e-6`: 設定條件數截止值，用於處理奇異矩陣。
3.  返回值包括：最佳參數 `theta_best_svd`、殘差、秩、奇異值。

### 範例 3.3: 使用偽逆矩陣

此函數計算 $\mathbf{X}^+\mathbf{y}$，其中 $\mathbf{X}^{+}$ 是 $\mathbf{X}$ 的_偽逆矩陣_ (特別是 Moore-Penrose 逆矩陣)。你可以使用 `np.linalg.pinv()` 直接計算偽逆矩陣：

```python
np.linalg.pinv(X_b) @ y
```

**✅ 程式碼逐行解析：**

1.  `np.linalg.pinv(X_b)`: 計算 X_b 的 Moore-Penrose 偽逆矩陣。
2.  `@ y`: 與目標值相乘得到最佳參數。

**🎯 重點摘要:**

- **核心功能**: 展示了三種不同但等價的方法來求解線性迴歸。
- **數學原理**: 都基於最小平方法和偽逆矩陣的概念。
- **最佳使用情境**: 了解底層數學原理，在 Scikit-Learn 無法使用時的替代方案。

## <a id="gradient-descent"></a>梯度下降 (Gradient Descent)

梯度下降是一種優化演算法，用於透過迭代地朝著最小值移動來最小化成本函式。它計算成本函式相對於參數的梯度，並在梯度的反方向更新參數。

一般更新規則是：
**θ = θ - η∇J(θ)**

其中：
- **θ** 是模型參數
- **η** (eta) 是學習率
- **∇J(θ)** 是成本函式的梯度

梯度下降有三種主要變體：

1. **批次梯度下降 (Batch GD)**: 使用*整個*訓練集來計算梯度
2. **隨機梯度下降 (Stochastic GD)**: 每次使用*一個隨機實例*
3. **小批次梯度下降 (Mini-batch GD)**: 使用*小批次隨機實例*

### <a id="batch-gd"></a>批次梯度下降 (Batch GD)

💡 **實際應用情境：** 當特徵數量龐大，無法使用正規方程式時，批次梯度下降是一個很好的替代方案。

### 範例 4: 批次梯度下降實現

```python
eta = 0.1  # 學習率
n_epochs = 1000  # 迭代次數
m = len(X_b)  # 訓練樣本數量

np.random.seed(42)
theta = np.random.randn(2, 1)  # 隨機初始化模型參數
print(f'Initial theta:\n{theta}')

gradient_history = []
cost_history = []

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
    gradient_history.append(gradients)
    cost = (1/m) * np.sum((X_b @ theta - y) ** 2)  # 均方誤差
    cost_history.append(cost)
```

**✅ 程式碼逐行解析：**

1.  `eta = 0.1`: 學習率，控制每一步更新的幅度。
2.  `n_epochs = 1000`: 迭代次數，決定訓練多少輪。
3.  `m = len(X_b)`: 訓練集大小，用於梯度正規化。
4.  `theta = np.random.randn(2, 1)`: 隨機初始化模型參數 (θ₀ 和 θ₁)。
5.  `gradients = 2 / m * X_b.T @ (X_b @ theta - y)`: 計算整個訓練集的梯度向量。
6.  `theta = theta - eta * gradients`: 根據梯度更新參數 (梯度下降步驟)。
7.  `gradient_history.append(gradients)`: 記錄每次迭代的梯度值，用於後續分析。
8.  `cost = (1/m) * np.sum((X_b @ theta - y) ** 2)`: 計算均方誤差 (MSE) 成本。
9.  `cost_history.append(cost)`: 記錄成本歷史，用於視覺化收斂過程。

**📊 關鍵變數說明：**

### `theta`
- **用途**: 代表模型的參數 (θ₀ 和 θ₁)
- **初始化**: 隨機初始化的值作為優化的起點
- **更新規則**: 每次迭代透過 `theta = theta - eta * gradients` 更新，使參數朝著成本函式最小值移動

### `gradients`
- **用途**: 包含成本函式 (均方誤差) 對參數的梯度
- **計算**: 公式 `2 / m * X_b.T @ (X_b @ theta - y)` 以向量化方式計算偏導數：
    - `X_b @ theta - y`: 所有訓練實例的預測誤差
    - `X_b.T @ (...)`: 計算梯度向量
- **作用**: 指向成本函式最陡峭上升的方向；移動到相反方向可下降到最小成本

### `cost` 和 `cost_history`
- **`cost`**: 每個 epoch 計算的均方誤差 (MSE)。它衡量預測值 (`X_b @ theta`) 與實際值 (`y`) 之間的平均平方差異。
- **`cost_history`**: 儲存每個 epoch 的 `cost` 的列表。這對於視覺化學習過程並確認成本隨時間遞減非常有用，表明模型正在學習。

### 關鍵變數說明
- **`eta`**: 學習率 (0.1) - 控制參數更新時的步長
- **`n_epochs`**: 迭代次數 (1000) - 決定更新參數的次數
- **`m`**: 訓練集大小 (100) - 用於梯度正規化

### 演算法流程
1. **初始化** 隨機參數
2. **計算** 所有訓練資料的預測誤差
3. **計算** 梯度，顯示成本最陡峭增加的方向
4. **更新** 參數，移動到梯度的相反方向
5. **重複** 直到收斂 (梯度接近零)

迴圈迭代地優化 `theta`，直到收斂到能最小化預測誤差的最佳參數。

**🎯 重點摘要:**

- **核心功能**: 在每次迭代中使用整個訓練集來計算梯度，更新方向穩定。
- **潛在問題**: 對於非常大的資料集，每次迭代都會很慢。
- **最佳使用情境**: 記憶體足以容納整個資料集，且特徵數量龐大的情況。

### 範例 4.1: 視覺化梯度收斂過程

```python
# 繪製梯度收斂圖
print(f"Type of gradient_history: {type(gradient_history)}")
gradient_history = np.array(gradient_history)  # 將列表轉換為 numpy 陣列
print(f"Shape of gradient_history: {gradient_history.shape}")

# 確保資料是 2D: (迭代次數, 參數數量)
y_to_plot = gradient_history.squeeze(-1) if gradient_history.ndim == 3 else gradient_history.reshape(gradient_history.shape[0], -1)
plt.plot(y_to_plot)
plt.xlabel("Iterations")
plt.ylabel("Gradient")
plt.title("Convergence of the Parameters")
plt.legend(["Theta 0", "Theta 1"])
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `gradient_history = np.array(gradient_history)`: 將 Python 列表轉換為 NumPy 陣列以便繪圖。
2.  `print(f"Shape of gradient_history: {gradient_history.shape}")`: 顯示陣列形狀，應該是 (1000, 2, 1)。
3.  `y_to_plot = gradient_history.squeeze(-1) if...`: 條件式重塑資料以適合繪圖。
4.  `gradient_history.squeeze(-1)`: 如果陣列是 3D，移除最後一個維度 (大小為 1)，將形狀從 (1000, 2, 1) 轉換為 (1000, 2)。
5.  `gradient_history.reshape(...)`: 如果不是 3D，作為備用方案將資料重塑為 2D。
6.  `plt.plot(y_to_plot)`: 繪製兩個參數的梯度歷史，每列作為一條線。

**📊 重塑 `gradient_history` 用於繪圖**

這行程式碼 `y_to_plot = gradient_history.squeeze(-1) if gradient_history.ndim == 3 else gradient_history.reshape(gradient_history.shape[0], -1)` 是資料準備步驟，使 `gradient_history` 陣列適合用 Matplotlib 繪圖。

說明如下：

1.  **初始形狀**: 從列表轉換為 NumPy 陣列後，`gradient_history` 的形狀是 `(1000, 2, 1)`。這是一個 3D 陣列，代表 1000 個 epochs、2 個參數 (θ₀ 和 θ₁)，以及大小為 1 的尾隨維度。

2.  **繪圖需求**: Matplotlib 的 `plt.plot()` 函數，當給定 2D 陣列時，會將每一列繪製為單獨的線。當前的 3D 形狀不適合。

3.  **條件式重塑**: 程式碼使用條件表達式來處理：
    *   **`if gradient_history.ndim == 3`**: 首先檢查陣列是否為 3 維。
    *   **`gradient_history.squeeze(-1)`**: 如果是 3D，使用 `squeeze(-1)` 移除最後一個維度，但僅當該維度的大小為 1 時。這將形狀從 `(1000, 2, 1)` 轉換為 `(1000, 2)`。
    *   **`else gradient_history.reshape(...)`**: 如果陣列不是 3D，此備用方案確保資料重塑為 2D 陣列，使程式碼更健壯。

最終的 `y_to_plot` 變數是形狀為 `(1000, 2)` 的 2D 陣列，其中每一列代表一個參數的梯度歷史，準備好繪製為兩條不同的線。

**🎯 重點摘要:**

- **核心功能**: 視覺化梯度如何隨著訓練過程逐漸接近零 (收斂)。
- **關鍵洞察**: 如果梯度在後期仍然很大，可能需要更多迭代或調整學習率。
- **最佳使用情境**: 診斷訓練過程，確保模型正確收斂。

### 範例 4.2: 放大檢視後期收斂

```python
start_iter = 900
plt.plot(np.arange(start_iter, len(y_to_plot)), y_to_plot[start_iter:, 0], 'b-')
plt.plot(np.arange(start_iter, len(y_to_plot)), y_to_plot[start_iter:, 1], 'r-')
plt.xlabel("Iterations")
plt.ylabel("Gradient")
plt.title("Convergence of the Parameters")
plt.legend(["Theta 0", "Theta 1"])
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `start_iter = 900`: 設定起始迭代次數，只顯示最後 100 次迭代。
2.  `y_to_plot[start_iter:, 0]`: 取得 θ₀ 的梯度值，從第 900 次迭代開始。
3.  `y_to_plot[start_iter:, 1]`: 取得 θ₁ 的梯度值。
4.  `'b-'` 和 `'r-'`: 分別使用藍色和紅色實線繪製。

### 範例 4.3: 視覺化成本函數收斂

```python
# 繪製成本函數收斂圖
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Convergence of Gradient Descent")
plt.grid()
plt.show()

# 繪製成本函數收斂圖 (放大檢視後期)
cost_start_iter = 400
plt.plot(np.arange(cost_start_iter, len(cost_history)), cost_history[cost_start_iter:])
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Convergence of Gradient Descent (Zoomed In)")
plt.grid()
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `plt.plot(cost_history)`: 繪製完整的成本歷史。
2.  `cost_start_iter = 400`: 設定起始點，只顯示後 600 次迭代。
3.  `cost_history[cost_start_iter:]`: 切片取得後期的成本值。

**🎯 重點摘要:**

- **核心功能**: 視覺化成本函數隨訓練過程的下降趨勢。
- **關鍵洞察**: 
  - 如果成本持續下降，表示模型正在學習
  - 如果成本停止下降或震盪，可能需要調整學習率
  - 放大後期可以更清楚地看到是否已經收斂
- **最佳使用情境**: 監控訓練過程，調整超參數。

### <a id="stochastic-gd"></a>隨機梯度下降 (Stochastic GD)

💡 **實際應用情境：** 當資料集非常大，無法在記憶體中一次處理時，隨機梯度下降是理想的選擇。

### 範例 5: 隨機梯度下降實現

```python
n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
```

**✅ 程式碼逐行解析：**

1.  `learning_schedule(t)`: 定義一個學習率遞減的排程，有助於模型收斂。
2.  `random_index = np.random.randint(m)`: 在每次迭代中隨機選擇一個樣本。
3.  `gradients = 2 * xi.T @ (xi @ theta - yi)`: 只用一個樣本來計算梯度。
4.  `eta = learning_schedule(epoch * m + i)`: 根據排程更新學習率。

**🎯 重點摘要:**

- **核心功能**: 每次只用一個樣本來更新參數，速度快，適合線上學習。
- **潛在問題**: 更新方向不穩定，成本函式會上下波動。
- **最佳使用情境**: 超大型資料集或需要線上學習的場景。

### <a id="minibatch-gd"></a>小批次梯度下降 (Mini-batch GD)

💡 **實際應用情境：** 小批次梯度下降是批次和隨機梯度下降的折衷方案，在實務中應用最廣。

### 範例 6: 小批次梯度下降實現

```python
from math import ceil

n_epochs = 50
minibatch_size = 20
n_batches_per_epoch = ceil(m / minibatch_size)

np.random.seed(42)
theta = np.random.randn(2, 1)

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for iteration in range(0, n_batches_per_epoch):
        idx = iteration * minibatch_size
        xi = X_b_shuffled[idx : idx + minibatch_size]
        yi = y_shuffled[idx : idx + minibatch_size]
        gradients = 2 / minibatch_size * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * n_batches_per_epoch + iteration)
        theta = theta - eta * gradients
```

**✅ 程式碼逐行解析：**

1.  `minibatch_size`: 設定每個小批次的大小。
2.  `shuffled_indices`: 在每個 epoch 開始時打亂資料順序。
3.  `gradients = 2/minibatch_size * xi.T @ (xi @ theta - yi)`: 使用一個小批次的樣本來計算梯度。

**🎯 重點摘要:**

- **核心功能**: 結合了批次梯度下降的穩定性和隨機梯度下降的效率。
- **最佳使用情境**: 大多數機器學習問題，特別是深度學習。

## <a id="polynomial-regression"></a>多項式迴歸 (Polynomial Regression)

💡 **實際應用情境：** 當資料呈現非線性關係時，可以使用多項式迴歸來擬合曲線。

### 範例 7: 多項式迴歸

```python
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
```

**✅ 程式碼逐行解析：**

1.  `y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)`: 生成一個二次方程式的非線性資料集。
2.  `poly_features = PolynomialFeatures(degree=2, include_bias=False)`: 建立一個 `PolynomialFeatures` 轉換器。
3.  `X_poly = poly_features.fit_transform(X)`: 將 `X` 轉換為包含原始特徵和其平方的 `X_poly`。
4.  `lin_reg.fit(X_poly, y)`: 使用轉換後的多項式特徵來訓練線性迴歸模型。

**🎯 重點摘要:**

- **核心功能**: 讓線性模型能夠擬合非線性資料。
- **潛在問題**: 高次多項式容易導致過擬合。
- **最佳使用情境**: 當資料視覺化中發現明顯的非線性趨勢時。

## <a id="learning-curves"></a>學習曲線 (Learning Curves)

💡 **實際應用情境：** 學習曲線是診斷模型是過擬合還是欠擬合的強大工具。

### 範例 8: 繪製學習曲線

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
```

**✅ 程式碼逐行解析：**

1.  `learning_curve()`: Scikit-Learn 的函式，用於計算不同訓練集大小下的模型效能。
2.  `train_errors`, `valid_errors`: 將得分轉換為均方根誤差 (RMSE)。
3.  `plt.plot()`: 繪製訓練誤差和驗證誤差隨訓練集大小變化的曲線。

**🎯 重點摘要:**

- **核心功能**: 視覺化模型在不同訓練集大小下的效能。
- **解讀**:
    - **欠擬合**: 兩條曲線都很高且彼此接近。
    - **過擬合**: 兩條曲線之間有很大的差距。
- **最佳使用情境**: 評估模型的泛化能力。

## <a id="regularized-models"></a>正規化線性模型

### <a id="ridge-regression"></a>Ridge 迴歸

💡 **實際應用情境：** 當模型過擬合時，Ridge 迴歸可以透過對模型參數的懲罰來降低複雜度。

### 範例 9: Ridge 迴歸

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1.  `ridge_reg = Ridge(alpha=0.1, solver="cholesky")`: 建立一個 Ridge 迴歸模型。
2.  `alpha`: 正規化強度。
3.  `solver`: 求解器。
4.  `ridge_reg.fit(X, y)`: 訓練模型。

**🎯 重點摘要:**

- **核心功能**: 加入 L2 懲罰項，降低過擬合風險。
- **最佳使用情境**: 當懷疑模型過擬合或有多重共線性問題時。

### <a id="lasso-regression"></a>Lasso 迴歸

💡 **實際應用情境：** Lasso 迴歸不僅能防止過擬合，還能自動進行特徵選擇。

### 範例 10: Lasso 迴歸

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
```

**🎯 重點摘要:**

- **核心功能**: 加入 L1 懲罰項，可將不重要的特徵權重降為零。
- **最佳使用情境**: 當你認為某些特徵是多餘的，或想要一個更簡潔的模型時。

### <a id="elastic-net"></a>Elastic Net

💡 **實際應用情境：** Elastic Net 結合了 Ridge 和 Lasso 的優點。

### 範例 11: Elastic Net

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
```

**🎯 重點摘要:**

- **核心功能**: 同時使用 L1 和 L2 懲罰。
- **最佳使用情境**: 當特徵數量大於樣本數，或特徵之間有很強的相關性時。

### <a id="early-stopping"></a>Early Stopping

💡 **實際應用情境：** Early Stopping 是一種簡單而有效的正規化方法。

### 範例 12: Early Stopping

```python
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, shuffle=False)

preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train.ravel())
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
```

**🎯 重點摘要:**

- **核心功能**: 監控驗證集上的效能，在模型開始過擬合之前停止訓練。
- **最佳使用情境**: 任何迭代式的學習演算法。

## <a id="logistic-regression"></a>邏輯迴歸 (Logistic Regression)

💡 **實際應用情境：** 邏輯迴歸是解決二元分類問題最常用和最基礎的演算法。

### 範例 13: 邏輯迴歸

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X = iris.data[["petal width (cm)"]].values
y = (iris.target == 2).astype(int)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)
```

**🎯 重點摘要:**

- **核心功能**: 估計一個樣本屬於某個類別的機率，並進行分類。
- **最佳使用情境**: 二元分類問題，如垃圾郵件檢測。

### 範例 13.1: 使用兩個特徵的邏輯迴歸：機率等高線圖與決策邊界

此範例訓練一個二元邏輯迴歸分類器來區分 Iris virginica，使用兩個特徵並視覺化機率表面和線性決策邊界。

```python
# 準備資料
X = iris.data[["petal length (cm)", "petal width (cm)"]].values  # 兩個特徵的特徵矩陣
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 訓練模型
log_reg = LogisticRegression(C=2, random_state=42)
log_reg.fit(X_train, y_train)

# 建立機率網格用於等高線圖
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]  # 將網格點展開為樣本矩陣
y_proba = log_reg.predict_proba(X_new)
zz = y_proba[:, 1].reshape(x0.shape)

# 計算決策邊界
left_right = np.array([2.9, 7])
boundary = -((log_reg.coef_[0, 0] * left_right + log_reg.intercept_[0])
             / log_reg.coef_[0, 1])

# 視覺化
plt.figure(figsize=(10, 4))
plt.plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], "bs")
plt.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], "g^")
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.27, "Not Iris virginica", color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", color="g", ha="center")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.axis([2.9, 7, 0.8, 2.7])
plt.grid()
save_fig("logistic_regression_contour_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

#### 1) 資料設定與訓練

1. **特徵選擇**: `X = iris.data[["petal length (cm)", "petal width (cm)"]].values`
   - 使用 pandas DataFrame 的雙括號列選擇以確保得到 2D 陣列
   - `.values` 將 DataFrame 轉換為 NumPy 陣列
   - 也可使用 `.to_numpy()` (較新的 pandas 方法)

2. **目標變數**: `y = iris.target_names[iris.target] == 'virginica'`
   - 建立布林值目標 (True 表示 virginica)

3. **資料分割**: `train_test_split(X, y, random_state=42)`
   - 分割訓練集和測試集以確保可重現性

4. **模型**: `LogisticRegression(C=2)`
   - `C` 是正規化強度的倒數 (C 越大，正規化越弱)
   - 在訓練資料上擬合以學習係數和截距

#### 2) 建立機率網格用於等高線圖

5. **使用 `np.meshgrid` 建立密集網格**:
   ```python
   x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                        np.linspace(0.8, 2.7, 200).reshape(-1, 1))
   ```
   - `np.linspace(2.9, 7, 500)` 和 `np.linspace(0.8, 2.7, 200)` 建立 500 個花瓣長度座標和 200 個花瓣寬度座標
   - `.reshape(-1, 1)` 將每個 1D 向量轉換為列向量 (非嚴格必要)
   - `np.meshgrid(...)` 將這些展開為 2D 座標網格
   - `x0` 和 `x1` 的形狀都是 `(200, 500)` = (Ny, Nx)

6. **將網格堆疊為 `X_new` 並計算類別機率**:
   - `X_new = np.c_[x0.ravel(), x1.ravel()]`: 展平網格陣列並堆疊用於預測
   - `y_proba = log_reg.predict_proba(X_new)`: 計算每個網格點的類別機率
   - `zz = y_proba[:, 1].reshape(x0.shape)`: 將機率重塑為網格形狀用於等高線繪圖
   - `zz[i, j]` 是 P(virginica | 網格點 i, j 處的特徵)

#### 3) 決策邊界計算

7. 對於邏輯迴歸，0.5 機率邊界是線性的：
   - 給定權重 w = [w_len, w_wid] 和偏差 b：
   - 邊界：petal_width = -(w_len × petal_length + b) / w_wid
   - `left_right` 設定繪圖的線的 x 範圍

#### 4) 視覺化

- **藍色方塊**: 非 virginica 訓練點
- **綠色三角形**: virginica 訓練點
- **等高線**: 網格上 "virginica" 類別的機率等級
- **黑色虛線**: 決策邊界 (P=0.5)
- **座標軸限制和標籤**: 調整為符合 Iris 特徵範圍
- **圖表儲存**: 透過 `save_fig("logistic_regression_contour_plot")`

#### 如何解讀圖表

- 等高線值越高的區域 (接近綠色) 表示 P(virginica) 越高
- 虛線上方的點被預測為 virginica；下方為非 virginica
- 等高線的間距反映模型信心 (越密集 = 機率變化越陡峭)

**📊 理解 NumPy 的 `ravel()` 方法**

`ravel()` 方法將多維 NumPy 陣列展平為 1D 陣列。在上面的程式碼中，它用於將 2D 網格陣列轉換為列向量以進行處理。

##### 程式碼範例

```python
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))
# x0 和 x1 是形狀為 (200, 500) 的 2D 陣列

X_new = np.c_[x0.ravel(), x1.ravel()]
# ravel() 後，兩者都變成形狀為 (100000,) 的 1D 陣列
# np.c_ 然後將它們堆疊成形狀為 (100000, 2) 的陣列
```

##### 為什麼在這裡使用 `ravel()`？

**目的：** 將 2D 座標網格轉換為可以餵給分類器的個別 (petal_length, petal_width) 點的列表。

**過程：**

1. **網格建立：** `x0` 和 `x1` 是 2D 網格 (200×500)，代表花瓣長度和寬度的所有組合
2. **展平：** `ravel()` 將每個 2D 網格轉換為 100,000 個點的 1D 向量
3. **堆疊：** `np.c_[x0.ravel(), x1.ravel()]` 建立一個 (100,000, 2) 陣列，其中每一行是一個 (長度, 寬度) 對
4. **預測：** 模型為所有 100,000 個點預測類別機率
5. **重塑回去：** `zz = y_proba[:, 1].reshape(x0.shape)` 將預測轉換回 2D 以用於等高線繪圖

##### 視覺範例

```python
# ravel() 之前
x0 = [[2.9, 2.91, 2.92],
      [2.9, 2.91, 2.92]]  # 形狀: (2, 3)

# ravel() 之後
x0.ravel() = [2.9, 2.91, 2.92, 2.9, 2.91, 2.92]  # 形狀: (6,)
```

##### 關鍵差異：`ravel()` vs `flatten()`

- **`ravel()`**: 盡可能返回*視圖* (記憶體效率高，變更會影響原始陣列)
- **`flatten()`**: 總是返回*副本* (安全但使用更多記憶體)

在此程式碼中，優先使用 `ravel()`，因為我們只需要一個臨時的 1D 視圖用於預測，而不是永久副本。

**🎯 重點摘要:**

- **核心功能**: 使用兩個特徵訓練二元分類器，並視覺化決策邊界和機率分布
- **關鍵技術**: 
  - 使用 `np.meshgrid` 建立密集網格以繪製平滑的等高線
  - 使用 `ravel()` 展平陣列用於批量預測
  - 計算線性決策邊界的數學公式
- **最佳使用情境**: 理解邏輯迴歸如何在 2D 特徵空間中劃分類別，診斷模型行為

## <a id="softmax-regression"></a>Softmax 迴歸

💡 **實際應用情境：** 當分類問題有多個類別時，Softmax 迴歸 (也稱為多項式邏輯迴歸) 是邏輯迴歸的自然推廣，用於同時分類多個互斥的類別。

### 範例 14.1: Softmax 迴歸用於多類別分類

此範例展示如何使用 Softmax 迴歸同時分類所有三個 Iris 物種。

```python
# 準備多類別分類資料
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target  # 0=setosa, 1=versicolor, 2=virginica
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 訓練 Softmax 迴歸模型
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

# 預測新樣本的類別
softmax_reg.predict([[5, 2]])  # 輸出: array([2])

# 檢視所有類別的機率
softmax_reg.predict_proba([[5, 2]]).round(2)  # 輸出: array([[0.  , 0.01, 0.99]])
```

**✅ 程式碼逐行解析：**

#### 1) 資料準備

1. **特徵選擇**: `X = iris.data[["petal length (cm)", "petal width (cm)"]].values`
   - 使用兩個特徵 (花瓣長度和寬度) 而非一個，提供更多資訊用於區分三個物種
   - `.values` 轉換 DataFrame 為 NumPy 陣列

2. **目標變數**: `y = iris.target`
   - 使用原始的三類別目標變數 (0, 1, 2) 而非二元分類
   - 0 = setosa, 1 = versicolor, 2 = virginica

3. **資料分割**: `train_test_split(X, y, random_state=42)`
   - 分割訓練集和測試集以確保可重現性

#### 2) 模型配置

4. **LogisticRegression 用於多類別**:
   - `C=30`: 高正規化參數 (正規化強度的倒數)，允許模型更靈活地擬合訓練資料
   - `random_state=42`: 確保結果可重現
   - **預設行為**: 當 `y` 有超過 2 個類別時，Scikit-Learn 的 `LogisticRegression` 自動使用 **multinomial** (Softmax) 迴歸

#### 3) 與二元分類的關鍵差異

**Softmax 函數**:
- 不使用 sigmoid 函數，而是使用 softmax 將類別分數轉換為機率
- 確保所有類別機率的總和為 1
- 每個類別都有自己的一組參數 (每個類別都有 θ₀, θ₁, θ₂)

**決策邊界**:
- 建立同時分離三個類別的決策邊界
- 比二元分類更複雜，因為必須同時區分多個類別

**應用場景**:
- 當需要將實例分類到多個互斥類別之一時，此方法非常理想
- 模型會為每個預測輸出所有三個 Iris 物種的機率估計

#### 4) 預測方法

5. **predict()**: 返回最高機率的類別標籤 (0, 1 或 2)
   - 計算每個類別的分數
   - 使用 softmax 轉換為機率
   - 選擇機率最高的類別 (argmax)

6. **predict_proba()**: 返回所有三個類別的機率估計
   - 機率總和為 1.0
   - 可用於評估模型信心

7. **decision_function()**: 返回 softmax 轉換前的原始類別分數

### 範例 14.2: Softmax 迴歸決策邊界視覺化

此程式碼建立一個全面的視覺化，展示 Softmax 迴歸如何使用花瓣長度和寬度測量值分類三個 Iris 物種。

```python
from matplotlib.colors import ListedColormap

# 建立自訂顏色映射
custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

# 建立網格點覆蓋特徵空間
x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                     np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]

# 在整個特徵空間進行預測
y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)  # Iris versicolor 機率
zz = y_predict.reshape(x0.shape)        # 預測類別

# 繪製決策區域和資料點
plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="center left")
plt.axis([0.5, 7, 0, 3.5])
plt.grid()
save_fig("softmax_regression_contour_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

#### 1) 視覺化設定

1. **自訂顏色映射**:
   - `ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])`: 為三個分類區域建立自訂顏色
   - 淡黃色代表 *Iris setosa*
   - 淡藍色代表 *Iris versicolor*
   - 淡綠色代表 *Iris virginica*

2. **網格建立**:
   - `np.meshgrid(...)`: 建立覆蓋特徵空間的細緻網格
   - 花瓣長度: 0 到 8 cm (500 個點)
   - 花瓣寬度: 0 到 3.5 cm (200 個點)
   - `X_new = np.c_[x0.ravel(), x1.ravel()]`: 將 2D 網格轉換為適合模型預測的格式

#### 2) 整個特徵空間的模型預測

3. **機率和類別預測**:
   - `y_proba = softmax_reg.predict_proba(X_new)`: 計算網格中每個點的類別機率
   - `y_predict = softmax_reg.predict(X_new)`: 確定每個網格點的預測類別
   - `zz1 = y_proba[:, 1].reshape(x0.shape)`: 提取 *Iris versicolor* 機率用於等高線
   - `zz = y_predict.reshape(x0.shape)`: 將預測重塑回網格格式用於視覺化

#### 3) 圖表組成

4. **資料點**:
   - **綠色三角形 (^)**: *Iris virginica* 訓練實例
   - **藍色方塊 (s)**: *Iris versicolor* 訓練實例
   - **黃色圓圈 (o)**: *Iris setosa* 訓練實例

5. **決策區域**:
   - `plt.contourf()`: 用顏色填充區域，顯示預測每個類別的位置
   - 建立不同的彩色區域，代表模型的分類決策

6. **機率等高線**:
   - `plt.contour()` 配合 `plt.clabel()`: 繪製顯示 *Iris versicolor* 機率等級的等高線
   - "hot" 顏色映射建立顯示機率轉變的漸層線
   - 幫助視覺化模型在不同區域的信心程度

#### 4) 關鍵洞察

此視覺化揭示:
- **清楚分離**: *Iris setosa* (黃色區域) 以較小的花瓣測量值容易區分
- **複雜邊界**: *Iris versicolor* 和 *Iris virginica* 之間的邊界更複雜，反映這些物種之間的相似性
- **機率梯度**: 等高線顯示平滑的機率轉變而非尖銳的邊界
- **模型信心**: 接近訓練資料群集的區域顯示更高的信心 (更深的等高線)

**🎯 重點摘要:**

- **核心功能**: Softmax 迴歸將邏輯迴歸擴展到多類別分類，計算每個類別的機率並預測機率最高的類別
- **關鍵技術**:
  - 使用 softmax 函數確保機率總和為 1
  - 每個類別有獨立的參數集
  - 建立複雜的非線性決策邊界分離多個類別
- **最佳使用情境**: 
  - 多類別分類問題 (如手寫數字辨識、文字分類)
  - 需要機率估計以評估信心
  - 類別互斥 (每個實例只屬於一個類別)
- **與二元邏輯迴歸的差異**:
  - 使用 softmax 而非 sigmoid
  - 輸出 K 個機率 (K 個類別) 而非 1 個
  - 自動處理多類別場景

## 💡 總結與最佳實踐

- **從簡單開始**: 始終先嘗試簡單的模型作為基準。
- **資料縮放**: 在使用梯度下降或正規化時，務必對特徵進行縮放。
- **選擇正確的梯度下降**: 小批次 GD 在大多數情況下是最佳選擇。
- **處理非線性**: 嘗試多項式迴歸。
- **防止過擬合**: 使用正規化或 Early Stopping。
- **診斷模型**: 使用學習曲線。

## ❓ 常見問答 (FAQ)

**Q: 正規方程式和梯度下降，我應該選擇哪一個？**
A: 如果特徵數量不多，正規方程式是個好選擇。否則，梯度下降是更佳的選擇。

**Q: 為什麼需要對資料進行縮放？**
A: 梯度下降在特徵尺度相似時收斂得更快。

**Q: Lasso 和 Ridge 有什麼主要區別？**
A: Lasso (L1) 可以用於特徵選擇，Ridge (L2) 不行。

## 🏷️ 推薦標籤 (Suggested Hashtags)

#Python #機器學習 #程式設計 #教學 #線性模型 #數據科學 #AI開發
import sklearn
assert sklearn.__version__ >= "0.20"

# 設定繪圖參數
import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

1. `import sys` 和 `assert sys.version_info >= (3, 5)`: 確保使用 Python 3.5 或更高版本
2. `import sklearn` 和 `assert sklearn.__version__ >= "0.20"`: 確保 Scikit-Learn 版本符合需求
3. `plt.rc('font', size=14)`: 設定matplotlib的預設字型大小為14，提升圖表可讀性
4. `plt.rc('axes', ...)`: 設定座標軸標籤和標題的字型大小
5. `plt.rc('legend', ...)`: 設定圖例字型大小
6. `plt.rc('xtick', ...)` 和 `plt.rc('ytick', ...)`: 設定刻度標籤字型大小為10

**🎯 重點摘要:**

- **核心功能**: 確保開發環境符合最低版本要求，並設定一致的視覺化參數
- **潛在問題**: 版本檢查失敗會導致程式中斷，需要升級相關套件
- **最佳使用情境**: 在專案開始時執行，確保所有協作者使用相同的環境設定

### 範例 2: 生成線性資料

```python
import numpy as np

np.random.seed(42)
m = 100  # number of instances
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
```

**✅ 程式碼逐行解析：**

1. `np.random.seed(42)`: 設定隨機種子為42，確保每次執行產生相同的隨機數據
2. `m = 100`: 定義資料點數量為100個實例
3. `X = 2 * np.random.rand(m, 1)`: 生成100個介於0到2之間的隨機值作為特徵
4. `y = 4 + 3 * X + np.random.randn(m, 1)`: 生成目標值，遵循 y = 4 + 3x + noise 的線性關係

**🎯 重點摘要:**

- **核心功能**: 產生符合線性關係的模擬資料集，用於測試線性模型
- **潛在問題**: 真實世界的資料通常不會這麼完美地符合線性關係
- **最佳使用情境**: 學習和驗證線性模型的基本概念時使用

## <a id="linear-regression"></a>📈 線性迴歸 (Linear Regression)

💡 **實際應用情境：** 線性迴歸是最基本的機器學習演算法，用於預測連續值，如房價預測、銷售預測等。

### <a id="normal-equation"></a>範例 3: 使用正規方程式 (Normal Equation)

```python
X_b = np.c_[np.ones((m, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

**✅ 程式碼逐行解析：**

1. `np.c_[np.ones((m, 1)), X]`: 在特徵矩陣前加入一列全為1的偏差項（bias term），用於表示截距
2. `X_b.T @ X_b`: 計算 X 轉置與 X 的矩陣乘積
3. `np.linalg.inv(...)`: 計算矩陣的反矩陣
4. `@ X_b.T @ y`: 完成正規方程式的計算，得到最佳參數 θ

**🎯 重點摘要:**

- **核心功能**: 直接計算出使成本函數最小化的最佳參數
- **潛在問題**: 當特徵數量很大時，計算反矩陣會非常耗時（O(n³)複雜度）
- **最佳使用情境**: 特徵數量較少（<10000）且記憶體充足的情況

### 範例 4: 使用 Scikit-Learn

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.linear_model import LinearRegression`: 導入 Scikit-Learn 的線性迴歸類別
2. `lin_reg = LinearRegression()`: 建立線性迴歸模型實例
3. `lin_reg.fit(X, y)`: 訓練模型，內部使用 SVD 分解而非正規方程式
4. `print(lin_reg.intercept_, lin_reg.coef_)`: 輸出截距和係數

**🎯 重點摘要:**

- **核心功能**: 使用業界標準函式庫進行線性迴歸，程式碼簡潔且效能優化
- **潛在問題**: 預設使用 SVD，可能不適合超大規模資料集
- **最佳使用情境**: 大多數實際應用場景，特別是需要快速原型開發時

## <a id="gradient-descent"></a>⛰️ 梯度下降 (Gradient Descent)

💡 **實際應用情境：** 梯度下降是訓練各種機器學習模型的核心優化演算法，特別適合大規模資料集。

### <a id="batch-gd"></a>範例 5: 批次梯度下降 (Batch Gradient Descent)

```python
eta = 0.1  # learning rate
n_epochs = 1000
m = len(X_b)

theta = np.random.randn(2, 1)  # random initialization

for epoch in range(n_epochs):
    gradients = 2/m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
```

**✅ 程式碼逐行解析：**

1. `eta = 0.1`: 設定學習率為0.1，控制每次參數更新的步長
2. `n_epochs = 1000`: 設定訓練週期數為1000次
3. `theta = np.random.randn(2, 1)`: 隨機初始化參數
4. `gradients = 2/m * X_b.T @ (X_b @ theta - y)`: 計算成本函數對參數的梯度
5. `theta = theta - eta * gradients`: 沿著梯度的反方向更新參數

**🎯 重點摘要:**

- **核心功能**: 使用全部訓練資料計算梯度，保證收斂到全域最小值（對凸函數）
- **潛在問題**: 每次迭代需要使用全部資料，對大資料集效率較低
- **最佳使用情境**: 資料集大小適中且需要精確收斂時

### <a id="stochastic-gd"></a>範例 6: 隨機梯度下降 (Stochastic Gradient Descent)

```python
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients
```

**✅ 程式碼逐行解析：**

1. `n_epochs = 50`: SGD 通常需要較少的 epoch 數
2. `t0, t1 = 5, 50`: 學習率衰減的超參數
3. `def learning_schedule(t)`: 定義學習率隨時間遞減的函數
4. `random_index = np.random.randint(m)`: 隨機選擇一個訓練樣本
5. `xi = X_b[random_index:random_index+1]`: 取出該樣本的特徵
6. `gradients = 2 * xi.T @ (xi @ theta - yi)`: 僅使用一個樣本計算梯度
7. `eta = learning_schedule(...)`: 根據當前迭代次數調整學習率
8. `theta = theta - eta * gradients`: 更新參數

**🎯 重點摘要:**

- **核心功能**: 每次只用一個樣本更新參數，訓練速度快，適合大資料集
- **潛在問題**: 參數更新有較大的隨機性，可能在最小值附近震盪
- **最佳使用情境**: 大規模資料集或線上學習場景

### 範例 7: 使用 Scikit-Learn 的 SGD

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())
```

**✅ 程式碼逐行解析：**

1. `max_iter=1000`: 設定最大迭代次數為1000
2. `tol=1e-5`: 設定收斂容忍度
3. `penalty=None`: 不使用正規化
4. `eta0=0.01`: 初始學習率為0.01
5. `n_iter_no_change=100`: 如果100次迭代內損失未改善則提前停止
6. `sgd_reg.fit(X, y.ravel())`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 提供現成的 SGD 實作，包含許多優化技巧
- **潛在問題**: 需要仔細調整超參數以獲得最佳效能
- **最佳使用情境**: 需要快速訓練大規模線性模型時

## <a id="polynomial-regression"></a>🔢 多項式迴歸 (Polynomial Regression)

💡 **實際應用情境：** 當資料呈現非線性關係時，多項式迴歸可以透過增加特徵的多項式項來擬合曲線。

### 範例 8: 建立多項式特徵

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
```

**✅ 程式碼逐行解析：**

1. `PolynomialFeatures(degree=2, include_bias=False)`: 建立2次多項式特徵轉換器
2. `X_poly = poly_features.fit_transform(X)`: 將原始特徵轉換為多項式特徵（如 x, x²）
3. `lin_reg = LinearRegression()`: 建立線性迴歸模型
4. `lin_reg.fit(X_poly, y)`: 在多項式特徵上訓練線性模型

**🎯 重點摘要:**

- **核心功能**: 透過特徵工程將線性模型擴展到非線性問題
- **潛在問題**: 高次多項式容易導致過擬合
- **最佳使用情境**: 資料呈現明顯的非線性但又不是特別複雜的模式時

## <a id="learning-curves"></a>📊 學習曲線 (Learning Curves)

💡 **實際應用情境：** 學習曲線可以幫助我們診斷模型是否存在高偏差（欠擬合）或高變異（過擬合）問題。

### 範例 9: 繪製學習曲線

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5, scoring="neg_mean_squared_error")

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
plt.legend()
```

**✅ 程式碼逐行解析：**

1. `learning_curve(LinearRegression(), X, y, ...)`: 計算不同訓練集大小下的模型效能
2. `train_sizes=np.linspace(0.01, 1.0, 40)`: 測試40個不同的訓練集大小
3. `cv=5`: 使用5折交叉驗證
4. `train_errors = -train_scores.mean(axis=1)`: 計算訓練誤差的平均值（轉為正值）
5. `valid_errors = -valid_scores.mean(axis=1)`: 計算驗證誤差的平均值
6. `plt.plot(...)`: 繪製訓練和驗證曲線

**🎯 重點摘要:**

- **核心功能**: 視覺化模型在不同資料量下的效能，診斷偏差-變異問題
- **潛在問題**: 計算學習曲線需要多次訓練模型，耗時較長
- **最佳使用情境**: 模型效能不如預期時，用於診斷問題所在

## <a id="regularized-models"></a>🛡️ 正規化線性模型

💡 **實際應用情境：** 正規化技術透過懲罰過大的參數來防止過擬合，在實際應用中非常重要。

### <a id="ridge-regression"></a>範例 10: Ridge 迴歸

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.linear_model import Ridge`: 導入 Ridge 迴歸類別
2. `alpha=1`: 設定正規化強度，值越大正規化越強
3. `solver="cholesky"`: 使用 Cholesky 分解求解，適合中小型資料集
4. `ridge_reg.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 使用 L2 正規化，懲罰參數的平方和，防止過擬合
- **潛在問題**: alpha 參數需要透過交叉驗證調整
- **最佳使用情境**: 存在多重共線性或特徵數量接近樣本數時

### <a id="lasso-regression"></a>範例 11: Lasso 迴歸

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1, random_state=42)
lasso_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.linear_model import Lasso`: 導入 Lasso 迴歸類別
2. `alpha=0.1`: 設定正規化強度
3. `lasso_reg.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 使用 L1 正規化，會將某些參數縮減為0，實現特徵選擇
- **潛在問題**: 可能不穩定，特別是當特徵高度相關時
- **最佳使用情境**: 需要自動特徵選擇或認為只有少數特徵真正重要時

### <a id="elastic-net"></a>範例 12: Elastic Net

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.linear_model import ElasticNet`: 導入 Elastic Net 類別
2. `alpha=0.1`: 設定正規化強度
3. `l1_ratio=0.5`: 設定 L1 和 L2 正規化的混合比例（0.5表示各半）
4. `elastic_net.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 結合 Ridge 和 Lasso 的優點，平衡兩者的特性
- **潛在問題**: 需要調整兩個超參數（alpha 和 l1_ratio）
- **最佳使用情境**: 當特徵數量遠大於樣本數，或特徵間有強相關性時

### <a id="early-stopping"></a>範例 13: Early Stopping

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

sgd_reg = SGDRegressor(max_iter=1, warm_start=True, penalty=None,
                       learning_rate="constant", eta0=0.0005, random_state=42)

minimum_val_error = float("inf")
best_epoch = None
best_model = None

for epoch in range(1000):
    sgd_reg.fit(X_train, y_train.ravel())
    y_val_predict = sgd_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)
```

**✅ 程式碼逐行解析：**

1. `train_test_split(X, y, random_state=42)`: 分割資料為訓練集和驗證集
2. `max_iter=1, warm_start=True`: 每次只訓練一個 epoch，並保留上次訓練的參數
3. `penalty=None`: 不使用正規化（early stopping 本身就是一種正規化）
4. `for epoch in range(1000)`: 手動控制訓練週期
5. `sgd_reg.fit(X_train, y_train.ravel())`: 訓練一個 epoch
6. `val_error = mean_squared_error(y_val, y_val_predict)`: 計算驗證誤差
7. `if val_error < minimum_val_error`: 如果驗證誤差降低，保存當前模型

**🎯 重點摘要:**

- **核心功能**: 透過監控驗證誤差，在模型開始過擬合前停止訓練
- **潛在問題**: 需要額外的驗證集，減少可用於訓練的資料
- **最佳使用情境**: 訓練深度學習模型或其他容易過擬合的複雜模型時

## <a id="logistic-regression"></a>🎯 邏輯迴歸 (Logistic Regression)

💡 **實際應用情境：** 邏輯迴歸是最常用的二元分類演算法，如垃圾郵件檢測、疾病診斷等。

### 範例 14: 基本邏輯迴歸

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X = iris.data[["petal width (cm)"]].values
y = (iris.target == 2)  # Iris virginica

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `iris = load_iris(as_frame=True)`: 載入鳶尾花資料集
2. `X = iris.data[["petal width (cm)"]].values`: 只使用花瓣寬度作為特徵
3. `y = (iris.target == 2)`: 建立二元標籤（是否為 Iris virginica）
4. `log_reg = LogisticRegression(random_state=42)`: 建立邏輯迴歸模型
5. `log_reg.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 透過 Sigmoid 函數將線性模型的輸出轉換為機率值
- **潛在問題**: 對離群值敏感，需要特徵縮放
- **最佳使用情境**: 二元分類問題，特別是需要機率輸出時

### 範例 15: 多特徵邏輯迴歸

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)

log_reg = LogisticRegression(C=2, random_state=42)
log_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `X = iris.data[["petal length (cm)", "petal width (cm)"]].values`: 使用兩個特徵
2. `C=2`: 設定正規化強度的反向參數（C越大，正規化越弱）
3. `log_reg.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 處理多個特徵的二元分類，支援L1和L2正規化
- **潛在問題**: C 參數需要調整以平衡偏差和變異
- **最佳使用情境**: 需要控制模型複雜度的二元分類問題

## <a id="softmax-regression"></a>🌈 Softmax 迴歸

💡 **實際應用情境：** Softmax 迴歸（也稱為多項式邏輯迴歸）用於多類別分類，如手寫數字辨識、物品分類等。

### 範例 16: Softmax 迴歸

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `X = iris.data[["petal length (cm)", "petal width (cm)"]].values`: 提取兩個特徵
2. `y = iris.target`: 使用三個類別的標籤（0, 1, 2）
3. `softmax_reg = LogisticRegression(C=30, random_state=42)`: 建立模型，Scikit-Learn 自動偵測多類別問題
4. `softmax_reg.fit(X, y)`: 訓練 Softmax 迴歸模型

**🎯 重點摘要:**

- **核心功能**: 將邏輯迴歸擴展到多類別分類，輸出每個類別的機率
- **潛在問題**: 假設類別互斥，不適合多標籤分類問題
- **最佳使用情境**: 多類別分類問題且類別互斥時

### 範例 17: 預測機率

```python
# 預測新樣本的類別機率
sample = [[5, 2]]
probabilities = softmax_reg.predict_proba(sample)
print(f"Class probabilities: {probabilities}")

# 預測類別
prediction = softmax_reg.predict(sample)
print(f"Predicted class: {prediction}")
```

**✅ 程式碼逐行解析：**

1. `sample = [[5, 2]]`: 建立一個測試樣本（花瓣長度5cm，寬度2cm）
2. `softmax_reg.predict_proba(sample)`: 預測各類別的機率
3. `softmax_reg.predict(sample)`: 預測最可能的類別

**🎯 重點摘要:**

- **核心功能**: 提供詳細的機率分布，不只是最終預測
- **潛在問題**: 機率值可能不夠校準，需要額外的校準步驟
- **最佳使用情境**: 需要量化預測不確定性的應用場景

## 💡 總結與最佳實踐

### 演算法選擇指南

| 情境 | 推薦演算法 | 原因 |
|------|-----------|------|
| 小資料集，特徵少 | 正規方程式 | 計算速度快，結果精確 |
| 大資料集 | SGD 或 Mini-batch GD | 訓練速度快，記憶體效率高 |
| 需要特徵選擇 | Lasso | 自動將不重要的特徵係數設為0 |
| 特徵相關性高 | Ridge 或 Elastic Net | 處理多重共線性問題 |
| 二元分類 | 邏輯迴歸 | 簡單高效，提供機率輸出 |
| 多類別分類 | Softmax 迴歸 | 自然的多類別擴展 |

### 最佳實踐建議

1. **資料前處理**
   - 特徵縮放對梯度下降至關重要
   - 檢查並處理離群值
   - 考慮多項式特徵以捕捉非線性關係

2. **超參數調整**
   - 使用交叉驗證選擇最佳的正規化參數
   - 監控學習曲線以診斷偏差-變異問題
   - 考慮使用 Grid Search 或 Random Search

3. **模型評估**
   - 不要只看訓練誤差，驗證集表現更重要
   - 使用適當的評估指標（MSE、RMSE、R²等）
   - 繪製預測值 vs 實際值的散點圖

4. **效能優化**
   - 對大資料集優先考慮 SGD 或 Mini-batch GD
   - 使用 warm_start 參數進行增量學習
   - 考慮使用 Early Stopping 節省訓練時間

## ❓ 常見問答 (FAQ)

**Q1: 何時使用正規方程式，何時使用梯度下降？**

A: 當特徵數量少於10,000且記憶體充足時，正規方程式更快更精確。對於更大的資料集或線上學習場景，梯度下降更合適。

**Q2: Ridge 和 Lasso 有什麼區別？**

A: Ridge 使用 L2 正規化，會縮小所有參數但不會將其設為0。Lasso 使用 L1 正規化，會將一些參數完全設為0，實現特徵選擇。

**Q3: 如何選擇學習率？**

A: 可以從 0.001 開始嘗試，觀察損失函數的變化。如果損失震盪或發散，降低學習率；如果收斂太慢，增加學習率。也可以使用學習率衰減策略。

**Q4: 什麼時候需要多項式特徵？**

A: 當學習曲線顯示高偏差（訓練和驗證誤差都很高）時，可能需要增加模型複雜度，這時可以嘗試多項式特徵。

**Q5: 如何診斷過擬合和欠擬合？**

A: 透過學習曲線：如果訓練誤差低但驗證誤差高，是過擬合；如果兩者都高，是欠擬合。可以分別透過正規化和增加模型複雜度來解決。

## 🏷️ 推薦標籤 (Suggested Hashtags)

#Python #機器學習 #線性迴歸 #梯度下降 #正規化 #邏輯迴歸 #Softmax #數據科學 #AI開發 #程式設計 #教學 #ScikitLearn #深度學習基礎 #模型訓練 #特徵工程
