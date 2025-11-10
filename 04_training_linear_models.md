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
plt.rc('font', family='Microsoft JhengHei')
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('axes', unicode_minus=False)
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
    - `plt.rc('font', family='Microsoft JhengHei')`: 設定字體為微軟正黑體, 適合中文顯示。
    - `plt.rc('axes', unicode_minus=False)`: 確保負號能正確顯示, 避免顯示成方塊。
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
8.  `X_b = add_dummy_feature(X)`: 為每個實例添加 `x0 = 1` 的偏差特徵，將形狀從 (100, 1) 轉換為 (100, 2)，且第一行全為 1，以考慮截距項。
9.  `theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y`: 使用正規方程式直接計算最佳參數 `theta`。
    - $\theta = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
    - `X_b.T @ X_b`: 計算特徵矩陣的轉置與自身的乘積。
    - `np.linalg.inv(...)`: 計算上述矩陣的逆矩陣。
    - `@ X_b.T @ y`: 將逆矩陣與轉置的特徵矩陣及目標值相乘，得到最佳參數。
    - 最終結果 `theta_best` 包含截距項和斜率。

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
- 對於我們的單特徵案例，`lin_reg.coef_[0][0]` 是 θ₁，即直線的斜率
- 顯示特徵增加一個單位時，預測值的變化量

#### 線性方程式
它們共同定義了線性模型：**ŷ = θ₀ + θ₁x₁**

其中：
- `lin_reg.intercept_` = θ₀ 
- `lin_reg.coef_[0][0]` = θ₁

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
- **更新規則**: 每次迭代透過 $\theta = \theta - \eta \nabla J(\theta)$ 更新,使參數朝著成本函式最小值移動
- **計算**: 透過梯度計算和學習率調整
    - `theta - eta * gradients`: 根據計算出的梯度調整參數
    - `gradients` 指向成本函式最陡峭上升的方向，移動到相反方向可下降到最小成本
- **最終結果**: 訓練完成後，`theta` 會收斂到最佳參數

### `gradients`
- **用途**: 包含成本函式 (均方誤差) 對參數的梯度
- **數學公式**: $$\nabla J(\theta) = \frac{2}{m} \mathbf{X}_b^T (\mathbf{X}_b \theta - \mathbf{y})$$
    - 其中 $m$ 是訓練樣本數量
    - $\mathbf{X}_b$ 是包含偏差項的特徵矩陣
    - $\theta$ 是參數向量
    - $\mathbf{y}$ 是目標值向量
- **計算步驟**: `2 / m * X_b.T @ (X_b @ theta - y)` 以向量化方式計算偏導數：
    - `X_b @ theta - y`: 計算所有訓練實例的預測誤差 (預測值 - 實際值)，形狀為 (m, 1)
    - `X_b.T @ (...)`: 將特徵矩陣的轉置與誤差向量相乘，得到每個參數的梯度，形狀為 (2, 1)
    - `2 / m * ...`: 對梯度進行正規化，除以樣本數量 m，係數 2 來自 MSE 成本函式的微分
- **作用**: 梯度就像是指南針，指向成本增加最快的方向。我們往相反方向移動（下坡），就能找到成本最低點

### `cost` 和 `cost_history`
- **`cost`**: 每個 epoch 計算的均方誤差 (MSE)。它衡量預測值 (`X_b @ theta`) 與實際值 (`y`) 之間的平均平方差異。
- **`cost_history`**: 儲存每個 epoch 的 `cost` 的列表。這對於視覺化學習過程並確認成本隨時間遞減非常有用，表明模型正在學習。

### 關鍵參數說明
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
plt.figure(figsize=(10, 6))

# 繪製 theta_0 和 theta_1 的梯度變化
plt.plot(np.arange(start_iter, len(y_to_plot)), y_to_plot[start_iter:, 0], 'b-', label=r"$\theta_0$")
plt.plot(np.arange(start_iter, len(y_to_plot)), y_to_plot[start_iter:, 1], 'r-', label=r"$\theta_1$")

# 在終點添加標記點並標註最終值
final_theta0 = y_to_plot[-1, 0]
final_theta1 = y_to_plot[-1, 1]
final_iter = len(y_to_plot) - 1

plt.plot(final_iter, final_theta0, 'bo', markersize=8, label=fr'$\theta_0$ 終值: {final_theta0:.6e}')
plt.plot(final_iter, final_theta1, 'ro', markersize=8, label=fr'$\theta_1$ 終值: {final_theta1:.6e}')

plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("梯度 (Gradient)")
plt.title("參數收斂過程 (放大) - Zoomed In")
plt.legend(loc='best')
plt.grid()
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `start_iter = 900`: 設定起始迭代次數，只顯示最後 100 次迭代。
2.  `plt.figure(figsize=(10, 6))`: 設定圖表大小。
3.  `plt.plot(..., label=r"$\theta_0$")`: 使用 LaTeX 格式化圖例標籤，使數學符號更美觀。
4.  `final_theta0 = y_to_plot[-1, 0]`: 取得 `theta_0` 的最終梯度值。
5.  `plt.plot(final_iter, ..., 'bo', ...)`: 在梯度曲線的終點繪製一個藍色圓點標記。
6.  `label=fr'$\theta_0$ 終值: {final_theta0:.6e}'`: 在圖例中標註最終的梯度值，使用科學記號格式化。
7.  `plt.legend(loc='best')`: 自動將圖例放置在最佳位置。

### 範例 4.3: 視覺化成本函數收斂

```python
# 繪製成本函數收斂圖
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("成本 (Cost - MSE)")
plt.title("批次梯度下降的成本函數收斂")
plt.grid()
plt.show()

# 繪製成本函數收斂圖 (放大檢視後期)
cost_start_iter = 300
plt.figure(figsize=(10, 6))
plt.plot(np.arange(cost_start_iter, len(cost_history)), cost_history[cost_start_iter:])
plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("成本 (Cost - MSE)")
plt.title("批次梯度下降的成本函數收斂 (放大)")
plt.axis((cost_start_iter, n_epochs, min(cost_history[cost_start_iter:])*0.99, max(cost_history[cost_start_iter:])*1.01))
plt.grid()
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `plt.figure(figsize=(10, 6))`: 設定圖表大小以獲得更好的視覺效果。
2.  `plt.plot(cost_history)`: 繪製完整的成本歷史，顯示成本隨迭代次數下降的趨勢。
3.  `cost_start_iter = 300`: 設定放大檢視的起始點，專注於訓練後期。
4.  `plt.plot(...)`: 繪製從第 300 次迭代開始的成本歷史。
5.  `plt.axis(...)`: 設定座標軸範圍，使放大後的圖表更清晰。

**🎯 重點摘要:**

- **核心功能**: 視覺化成本函數隨訓練過程的下降趨勢。
- **關鍵洞察**: 
  - 如果成本持續下降，表示模型正在學習。
  - 如果成本停止下降或震盪，可能需要調整學習率。
  - 放大後期可以更清楚地看到模型是否已經收斂。
- **最佳使用情境**: 監控訓練過程，診斷收斂問題，並輔助調整學習率等超參數。

### <a id="stochastic-gd"></a>隨機梯度下降 (Stochastic GD; SGD)

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

1.  `n_epochs = 50`: SGD 通常需要較少的 epoch 數
2.  `t0, t1 = 5, 50`: 學習率衰減 (learning rate decay) 的超參數
3.  `def learning_schedule(t)`: 定義學習率隨時間遞減的函數
4.  `random_index = np.random.randint(m)`: 在每次迭代中隨機選擇一個樣本。
5.  `xi = X_b[random_index:random_index+1]`: 取出該樣本的特徵值。
6.  `yi = y[random_index:random_index+1]`: 取出該樣本的目標值。
7.  `gradients = 2 * xi.T @ (xi @ theta - yi)`: 只用一個樣本來計算梯度。
8.  `eta = learning_schedule(epoch * m + i)`: 根據排程更新學習率, 隨著迭代次數增加而減小。
9.  `theta = theta - eta * gradients`: 根據梯度更新參數 (梯度下降步驟)。

**🎯 重點摘要:**

- **核心功能**: 每次只用一個樣本來更新參數，速度快，適合線上學習 (online learning)。
- **潛在問題**: 更新方向不穩定，成本函式會上下波動。
- **最佳使用情境**: 超大型資料集或需要線上學習的場景。

### 範例 5.1: 使用 Scikit-Learn 的 SGD

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())
```

**✅ 程式碼逐行解析：**

1.  `max_iter=1000`: 設定最大迭代次數為1000
2.  `tol=1e-5`: 設定收斂容忍度
3.  `penalty=None`: 不使用正規化
4.  `eta0=0.01`: 初始學習率為0.01
5.  `n_iter_no_change=100`: 如果100次迭代內損失未改善則提前停止
6.  `sgd_reg.fit(X, y.ravel())`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 提供現成的 SGD 實作，包含許多優化技巧
- **最佳使用情境**: 需要快速訓練大規模線性模型時

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
3.  `X_poly = poly_features.fit_transform(X)`:將 `X` 轉換為包含原始特徵和其平方的 `X_poly`。
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
plt.legend()
```

**✅ 程式碼逐行解析：**

1.  `learning_curve()`: Scikit-Learn 的函式，用於計算不同訓練集大小下的模型效能。
2.  `train_sizes=np.linspace(0.01, 1.0, 40)`: 測試40個不同的訓練集大小。
3.  `cv=5`: 使用5折交叉驗證 (5-fold cross-validation)。
4.  `train_errors`, `valid_errors`: 將得分轉換為均方根誤差 (RMSE)。
  -   `-train_scores.mean(axis=1)`: 計算每個訓練集大小的平均訓練誤差, 並取負值轉換為正的 RMSE。
  -   `-valid_scores.mean(axis=1)`: 計算每個訓練集大小的平均驗證誤差, 並取負值轉換為正的 RMSE。
  -   `axis=1`: 指定沿著哪個軸計算平均值，這裡是沿著列 (不同交叉驗證折數)。
5.  `plt.plot()`: 繪製訓練誤差和驗證誤差隨訓練集大小變化的曲線。

**🎯 重點摘要:**

- **核心功能**: 視覺化模型在不同訓練集大小下的效能。
- **解讀**:
    - **欠擬合**: 兩條曲線都很高且彼此接近。
    - **過擬合**: 兩條曲線之間有很大的差距。
- **最佳使用情境**: 評估模型的泛化能力。

## <a id="regularized-models"></a>正規化線性模型

正規化是防止線性模型過擬合的關鍵技術。當模型在訓練資料上表現很好，但在新資料上表現不佳時，就可能發生過擬合。正規化透過在成本函式中添加懲罰項來約束模型參數的大小，迫使模型學習更簡單、更能泛化的模式。

**為什麼需要正規化？**

1. **防止過擬合**: 當模型過於複雜或特徵數量很多時，它可能會「記住」訓練資料的雜訊，而不是學習真正的模式。
2. **處理多重共線性 (Multicollinearity)**: 當特徵之間高度相關時，正規化（特別是 Ridge 或 Elastic Net）可以穩定參數估計，避免模型權重不合理波動。
3. **特徵選擇 (Feature Selection)**: 某些正規化方法（如 Lasso）可以自動識別並忽略不重要的特徵。
4. **提升泛化能力**: 正規化後的模型通常在未見過的資料上表現更好。

**正規化的數學原理**

對於線性迴歸，標準的成本函式是均方誤差（MSE）：

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

正規化透過添加懲罰項 (penalty term) 來修改這個成本函式：

$$J(\theta) = MSE(\theta) + \text{正規化項}$$

這個正規化項會懲罰大的參數值，迫使模型保持簡單。

**三種主要的正規化方法**

1. **Ridge 迴歸 (L2 正規化)**
  - 懲罰項：$\alpha \sum_{i=1}^{n} \theta_i^2$
  - 會縮小所有參數，但不會將其設為零
  - 適合所有特徵都可能有用的情況

2. **Lasso 迴歸 (L1 正規化)**
  - 懲罰項：$\alpha \sum_{i=1}^{n} |\theta_i|$
  - 會將一些參數完全設為零，實現自動特徵選擇
  - 適合懷疑許多特徵是多餘的情況

3. **Elastic Net (L1 + L2)**
  - 結合 Ridge 和 Lasso 的優點
  - 懲罰項：$r \alpha \sum_{i=1}^{n} |\theta_i| + \frac{1-r}{2} \alpha \sum_{i=1}^{n} \theta_i^2$
    - 其中 $r$ 是介於 0 和 1 之間的混合參數, 控制 L1 和 L2 的權重
  - 適合特徵數量遠大於樣本數量的情況
    - 例如基因資料分析或文本分類等高維度資料集

其中 $\alpha$ 是正規化強度參數，控制懲罰的程度。$\alpha$ 越大，正規化效果越強，模型越簡單。

**選擇正規化方法的實用建議**

- 從 Ridge 開始嘗試，它通常是最安全的選擇
- 如果特徵很多且懷疑只有少數有用，使用 Lasso
- 當 Lasso 表現不穩定時，考慮 Elastic Net
- 使用交叉驗證來選擇最佳的 $\alpha$ 值
- 記得在應用正規化之前對特徵進行縮放，因為正規化對特徵的尺度敏感

下面的章節將詳細介紹每種正規化方法的實作和應用。

---

### <a id="ridge-regression"></a>Ridge 迴歸

💡 **實際應用情境：** 當模型過擬合時，Ridge 迴歸可以透過對模型參數的懲罰來降低複雜度。

### 範例 9: Ridge 迴歸

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.linear_model import Ridge`: 導入 Ridge 迴歸類別
2. `alpha=1`: 設定正規化強度，值越大正規化越強
3. `solver="cholesky"`: 使用 Cholesky 分解求解，適合中小型資料集
  - Cholesky 分解是一種數值方法，用於高效解決線性方程組，特別是當矩陣是對稱正定時。它將矩陣分解為一個下三角矩陣及其轉置的乘積，從而簡化計算過程。
4. `ridge_reg.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 加入 L2 懲罰項，降低過擬合風險。
- **最佳使用情境**: 當懷疑模型過擬合或有多重共線性問題時。

---

### <a id="lasso-regression"></a>Lasso 迴歸

💡 **實際應用情境：** Lasso 迴歸不僅能防止過擬合，還能自動進行特徵選擇。

### 範例 10: Lasso 迴歸

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1, random_state=42)
lasso_reg.fit(X, y)
```

**🎯 重點摘要:**

- **核心功能**: 加入 L1 懲罰項，可將不重要的特徵權重降為零。
- **最佳使用情境**: 當你認為某些特徵是多餘的，或想要一個更簡潔的模型時。

---

### <a id="elastic-net"></a>Elastic Net

💡 **實際應用情境：** Elastic Net 結合了 Ridge 和 Lasso 的優點。

### 範例 11: Elastic Net

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.linear_model import ElasticNet`: 導入 Elastic Net 類別
2. `alpha=0.1`: 設定正規化強度，值越大正規化越強
3. `l1_ratio=0.5`: 設定 L1 和 L2 懲罰的比例，0.5 表示兩者權重相等
4. `elastic_net.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 同時使用 L1 和 L2 懲罰。
- **最佳使用情境**: 當特徵數量大於樣本數，或特徵之間有很強的相關性時。

---

### <a id="early-stopping"></a>Early Stopping

💡 **實際應用情境：** Early Stopping 是一種簡單而有效的正規化方法，透過*監控驗證集的效能來決定何時停止訓練*。當驗證誤差開始上升時，表示模型開始過擬合訓練資料，此時應停止訓練並使用到目前為止表現最好的模型。這種方法特別適合用於迭代式學習演算法（如梯度下降），可以在不增加模型複雜度的情況下防止過擬合，同時節省訓練時間。

Early Stopping 的核心思想是：

- **持續監控**：在每個 epoch 後評估驗證集的效能
- **記錄最佳模型**：當驗證誤差創新低時，儲存當前模型的副本
- **及時停止**：當驗證誤差持續一段時間不再改善時，停止訓練
- **使用最佳模型**：最終採用驗證誤差最低時的模型參數

這種方法的優點是不需要調整正規化參數（如 α），但需要額外的驗證集和更多的計算資源來追蹤模型效能。在實務中，Early Stopping 常與其他正規化技術（如 Ridge 或 Lasso）結合使用，以獲得更好的泛化能力。

### 範例 12: Early Stopping

```python
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
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

**✅ 程式碼逐行解析：**

1.  `from copy import deepcopy`: 導入 `deepcopy` 用於複製模型
2.  `train_test_split(...)`: 將資料分割為訓練集和驗證集
  - `shuffle=False`: 保持資料的時間順序，適合時間序列資料
3.  `make_pipeline(...)`: 建立一個包含多項式特徵和標準化的預處理管道
  - `PolynomialFeatures(degree=90, include_bias=False)`: 將特徵擴展為 90 次多項式
  - `StandardScaler()`: 對特徵進行標準化，使其均值為0，標準差為1
4.  `sgd_reg = SGDRegressor(...)`: 初始化一個不帶正規化的隨機梯度下降迴歸模型
  - `penalty=None`: 不使用任何正規化
  - `eta0=0.002`: 設定初始學習率, 較小的學習率有助於穩定訓練過程
5.  `for epoch in range(n_epochs)`: 進行多個訓練迭代; 每次迭代後評估模型效能
6.  `sgd_reg.partial_fit(...)`: 使用部分擬合方法進行訓練
  - `y_train.ravel()`: 將目標變數展平成一維陣列，以符合 Scikit-Learn 的要求
  - `partial_fit()`: 適用於增量學習 (incremental learning or online learning)，允許模型在每個 epoch 後更新參數，而不是重新擬合整個模型
7.  `y_valid_predict = sgd_reg.predict(X_valid_prep)`: 在驗證集上進行預測
8.  `val_error = root_mean_squared_error(...)`: 計算驗證集的均方根誤差
9.  `if val_error < best_valid_rmse`: 如果當前驗證誤差優於最佳誤差，則更新最佳模型
  - `best_valid_rmse = val_error`: 更新最佳驗證誤差
10. `best_model = deepcopy(sgd_reg)`: 使用 `deepcopy` 儲存當前模型的副本作為最佳模型

**🎯 重點摘要:**

- **核心功能**: 監控驗證集上的效能，在模型開始過擬合之前停止訓練。
- **最佳使用情境**: 任何迭代式的學習演算法。

## <a id="logistic-regression"></a>邏輯迴歸 (Logistic Regression)

邏輯迴歸是解決二元分類問題最常用和最基礎的演算法。

它透過邏輯函數（Sigmoid 函數）將線性迴歸的輸出轉換為 0 到 1 之間的機率值，使其適合預測類別歸屬。

常見應用包括：垃圾郵件檢測（是/否）、疾病診斷（陽性/陰性）、客戶流失預測（會/不會）、信用風險評估（違約/正常）等。

邏輯迴歸的優勢在於訓練速度快、解釋性強、可以輸出機率估計，且在許多實際問題中表現穩健。

### 邏輯迴歸的核心原理

邏輯迴歸使用 **Sigmoid 函數**（也稱為 Logistic 函數）將線性組合轉換為機率：

$$\sigma(t) = \frac{1}{1 + e^{-t}}$$

其數值範圍在 0 到 1 之間，適合作為機率值。

對於輸入 $x$，邏輯迴歸模型估計的機率為：

$$\hat{p} = h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

其中：
- $\theta^T x$ 是特徵的線性組合
- $\hat{p}$ 是預測為正類別（y=1）的機率
- 當 $\theta^T x$ 為正且很大時，$\hat{p}$ 接近 1
- 當 $\theta^T x$ 為負且很小時，$\hat{p}$ 接近 0
- 當 $\theta^T x = 0$ 時，$\hat{p} = 0.5$（決策邊界）

**決策規則**：
- 如果 $\hat{p} \geq 0.5$，預測 $\hat{y} = 1$（正類別）
- 如果 $\hat{p} < 0.5$，預測 $\hat{y} = 0$（負類別）

**成本函數（Log Loss）**

邏輯迴歸使用對數損失（log loss），也稱為交叉熵損失 (cross-entropy loss)：

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)})]$$

這個成本函數的特性：
- 當 $y=1$ 且 $\hat{p}$ 接近 1 時，成本接近 0（正確預測）
- 當 $y=1$ 且 $\hat{p}$ 接近 0 時，成本趨近無窮大（嚴重錯誤）
- 當 $y=0$ 且 $\hat{p}$ 接近 0 時，成本接近 0（正確預測）
- 當 $y=0$ 且 $\hat{p}$ 接近 1 時，成本趨近無窮大（嚴重錯誤）

**正規化**

Scikit-Learn 的 `LogisticRegression` 預設使用 L2 正規化，可以透過參數 `C` 控制：
- `C` 是正規化強度的倒數
- `C` 越大，正規化越弱（允許更複雜的模型）
- `C` 越小，正規化越強（模型更簡單）

💡 **實際應用情境：** 邏輯迴歸廣泛應用於各種二元分類問題，如醫療診斷、信用評分、行為預測等，因其解釋性強且計算效率高。

### 範例 13: 邏輯迴歸

此範例展示如何使用邏輯迴歸進行二元分類。我們使用 Iris 資料集，僅選擇一個特徵（花瓣寬度）來預測一朵花是否為 Iris virginica。這是一個簡化的範例，用於說明邏輯迴歸的基本概念。在實務中，使用多個特徵通常能獲得更好的分類效果。

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X = iris.data[["petal width (cm)"]].values
y = (iris.target == 2).astype(int)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import load_iris`: 導入 Iris 資料集載入函數
2. `iris = load_iris(as_frame=True)`: 載入 Iris 資料集並轉換為 pandas DataFrame 格式
3. `X = iris.data[["petal width (cm)"]].values`: 選擇花瓣寬度作為唯一特徵
  - 使用雙括號 `[[...]]` 確保結果是 2D 陣列
  - `.values` 將 DataFrame 轉換為 NumPy 陣列; 也可使用 `.to_numpy()`, 較新的 pandas 方法
4. `y = (iris.target == 2).astype(int)`: 建立二元目標變數
  - `iris.target == 2`: 檢查是否為 virginica（類別 2）
  - `.astype(int)`: 將布林值轉換為整數（0 或 1）
5. `log_reg = LogisticRegression(random_state=42)`: 建立邏輯迴歸模型實例
6. `log_reg.fit(X, y)`: 訓練模型以學習特徵與目標之間的關係

**🎯 重點摘要:**

- **核心功能**: 估計一個樣本屬於某個類別的機率，並進行分類。
- **資料準備**: 展示如何將多類別問題轉換為二元分類問題。
- **模型輸出**: 訓練後的模型可以預測新樣本的類別，並提供屬於正類別的機率估計。
- **最佳使用情境**: 二元分類問題，如垃圾郵件檢測、疾病診斷等。

### 範例 13.1: 使用兩個特徵的邏輯迴歸：機率等高線圖與決策邊界

此範例訓練一個二元邏輯迴歸分類器來區分 Iris virginica，使用兩個特徵 (花瓣長度和花瓣寬度) 並視覺化機率表面和線性決策邊界。

```python
# 準備資料
X = iris.data[["petal length (cm)", "petal width (cm)"]].to_numpy()  # 兩個特徵的特徵矩陣
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

1. **特徵選擇**: `X = iris.data[["petal length (cm)", "petal width (cm)"]].to_numpy()`
   - 使用 pandas DataFrame 的雙括號列選擇以確保得到 2D 陣列
   - `.to_numpy()` 將 DataFrame 轉換為 NumPy 陣列

2. **目標變數**: `y = iris.target_names[iris.target] == 'virginica'`
   - 建立布林值目標 (True 表示 virginica)

3. **資料分割**: `train_test_split(X, y, random_state=42)`
   - 分割訓練集和測試集以確保可重現性
   - 注意：`train_test_split` 輸入X和y是 NumPy 陣列 (numpy.ndarray)
   - 注意：`train_test_split` 輸出也是 NumPy 陣列

4. **模型**: `LogisticRegression(C=2)`
   - `C` 是正規化強度的倒數 (C 越大，正規化越弱)
   - 在訓練資料上擬合以學習係數和截距

#### 2) 建立機率網格用於等高線圖

為了視覺化模型的決策邊界，我們需要在整個特徵空間中對模型的預測進行採樣。這可以透過建立一個密集的點網格，並為每個點預測機率來實現。

##### a. 建立座標網格 (np.meshgrid)

首先，我們使用 `np.meshgrid` 來建立一個覆蓋我們感興趣的特徵範圍的 2D 座標網格，就像在繪圖紙上畫出格線一樣。

   ```python
   # 建立覆蓋特徵空間的密集網格
   x0, x1 = np.meshgrid(
       np.linspace(2.9, 7, 500).reshape(-1, 1),      # 花瓣長度座標 (x 軸)
       np.linspace(0.8, 2.7, 200).reshape(-1, 1),    # 花瓣寬度座標 (y 軸)
   )
   ```

   - **`np.linspace(...)`**: 建立定義網格軸線的 1D 陣列 (500 個 x 座標和 200 個 y 座標)。
   - **`np.meshgrid(...)`**: 將這些軸線擴展為兩個 2D 陣列：
     - `x0` (形狀 `200, 500`)：包含網格中每個點的 x 座標 (花瓣長度)。
     - `x1` (形狀 `200, 500`)：包含網格中每個點的 y 座標 (花瓣寬度)。

##### b. 準備網格用於預測
Scikit-Learn 模型期望輸入是一個樣本列表 (形狀為 `[n_samples, n_features]`)。我們需要將 2D 網格轉換為這種格式。

   ```python
   # 將網格點轉換為樣本矩陣
   X_new = np.c_[x0.ravel(), x1.ravel()]
   ```

   - **`x0.ravel()`**: 將 `x0` 陣列「展平」為一個包含 100,000 個元素的 1D 陣列。
   - **`np.c_[...]`**: 將兩個展平的 1D 陣列並排堆疊，建立一個形狀為 `(100000, 2)` 的陣列。`X_new` 現在是一個包含網格上所有 100,000 個點的列表，每個點都是一個 `(花瓣長度, 花瓣寬度)` 的特徵對。

##### c. 計算機率並重塑
現在我們可以將這個樣本列表餵給模型來預測每個點的機率，然後將結果重塑回原始網格的形狀，以便繪製等高線。

   ```python
   # 計算每個網格點的類別機率
   y_proba = log_reg.predict_proba(X_new)

   # 提取 virginica 的機率並重塑為網格形狀
   zz = y_proba[:, 1].reshape(x0.shape)
   ```

   - `zz` 現在是一個 `(200, 500)` 的陣列，其中 `zz[i, j]` 代表網格點 `(i, j)` 被預測為 "virginica" 的機率。

#### 3) 決策邊界計算

對於邏輯迴歸，0.5 機率邊界是線性的：

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

---

**📊 理解 NumPy 的 `ravel()` 方法**

`ravel()` 方法將多維 NumPy 陣列展平為 1D 陣列。在上面的程式碼中，它用於將 2D 網格陣列轉換為 1D 向量以進行處理。

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
3. **堆疊：** `np.c_[x0.ravel(), x1.ravel()]` 建立一個 (100,000, 2) 陣列，其中每一列是一個 (長度, 寬度) 對
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

## <a id="softmax-regression"></a>🌈 Softmax 迴歸

Softmax 迴歸將二元邏輯迴歸擴展到 *處理 K 個類別（K > 2）* 的情況。它的核心思想是為每個類別學習一組獨立的參數，然後使用 Softmax 函數將原始分數轉換為機率分布。

**關鍵特性：**

1. **多類別擴展**: 對於 K 個類別，模型學習 K 組參數向量（θ₀, θ₁, ..., θₖ₋₁），每組對應一個類別
2. **機率輸出**: Softmax 函數確保所有類別的預測機率總和為 1，符合機率公理
3. **互斥假設**: 適用於每個樣本只能屬於一個類別的情況（如 Iris 花卉分類）
4. **線性決策邊界**: 雖然看起來複雜，但 Softmax 迴歸本質上是線性分類器，在特徵空間中建立線性決策邊界

**數學原理：**

對於輸入 x，類別 k 的分數（logit）計算為：
$$s_k(x) = \theta_k^T x$$

然後使用 Softmax 函數將分數轉換為機率：
$$P(y=k|x) = \frac{e^{s_k(x)}}{\sum_{j=1}^{K} e^{s_j(x)}}$$

模型選擇機率最高的類別作為預測結果：
$$\hat{y} = \arg\max_k P(y=k|x)$$

**與二元邏輯迴歸的關係：**

當 K=2 時，Softmax 迴歸退化為標準的邏輯迴歸。事實上，Scikit-Learn 的 `LogisticRegression` 會根據目標變數的類別數量自動選擇使用二元邏輯迴歸還是 Softmax 迴歸。

### 範例 16: Softmax 迴歸基礎訓練

```python
# 使用兩個特徵提供更多資訊以區分三個物種
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
print(f"特徵矩陣形狀: {X.shape}")

# 使用原始的三類別目標變數 (而非二元分類)
# 0 = setosa, 1 = versicolor, 2 = virginica
y = iris.target
print(f"目標變數形狀: {y.shape}")
print(f"類別分布: setosa={np.sum(y==0)}, versicolor={np.sum(y==1)}, virginica={np.sum(y==2)}")

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 訓練 Softmax 迴歸模型
# 當 y 有超過 2 個類別時，LogisticRegression 自動使用 multinomial (Softmax) 迴歸
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

print(f"模型已訓練完成")
print(f"模型係數形狀: {softmax_reg.coef_.shape}")  # (3, 2) - 每個類別有一組係數
print(f"模型截距形狀: {softmax_reg.intercept_.shape}")  # (3,) - 每個類別有一個截距
```

**✅ 程式碼逐行解析：**

1. **資料準備**:
  - `X = iris.data[["petal length (cm)", "petal width (cm)"]].values`: 提取兩個特徵（花瓣長度和寬度）
  - `y = iris.target`: 使用三個類別的標籤（0=setosa, 1=versicolor, 2=virginica）
  - `train_test_split()`: 分割訓練集和測試集

2. **模型配置**:
  - `C=30`: 高正規化參數（正規化強度的倒數），允許模型更靈活地擬合訓練資料
  - `LogisticRegression` 會自動偵測多類別問題並使用 multinomial (Softmax) 迴歸

3. **模型結構**:
  - `coef_.shape = (3, 2)`: 每個類別（3個）對應一組權重（2個特徵）
  - `intercept_.shape = (3,)`: 每個類別有一個截距項

**🎯 重點摘要:**

- **核心功能**: 將邏輯迴歸擴展到多類別分類，為每個類別學習獨立的參數集
- **自動偵測**: Scikit-Learn 會根據 y 的類別數量，由LogisticRegression自動選擇二元或多元邏輯迴歸
- **最佳使用情境**: 類別互斥的多類別分類問題（每個樣本只能屬於一個類別）

💡 **實際應用情境：** Softmax 迴歸（也稱為多項式邏輯迴歸）是邏輯迴歸的自然推廣，用於同時分類多個互斥的類別，如手寫數字辨識、物品分類、語言偵測等多類別分類問題。

### 範例 17: Softmax 迴歸的三種預測方法

在此範例中，我們展示 Softmax 迴歸模型的三種主要預測方法：`predict()`、`predict_proba()` 和 `decision_function()`。這些方法提供不同層次的預測資訊，適用於不同的應用場景。

a. `predict()`：返回最高機率的類別標籤

b. `predict_proba()`：返回所有類別的機率估計

c. `decision_function()`：返回 Softmax 轉換前的原始分數（logits） 

```python
# 建立測試樣本：花瓣長度 5cm、寬度 2cm
test_sample = np.array([[5, 2]])

# 方法 1: predict() - 返回最高機率的類別標籤
predicted_class = softmax_reg.predict(test_sample)[0]
print(f"預測類別: {predicted_class}")
print(f"對應的物種名稱: {iris.target_names[predicted_class]}")

# 方法 2: predict_proba() - 返回所有三個類別的機率估計
proba = softmax_reg.predict_proba(test_sample).round(2)
print(f"各類別的預測機率: {proba}")
print(f"  - P(setosa)     = {proba[0, 0]:.2f}")
print(f"  - P(versicolor) = {proba[0, 1]:.2f}")
print(f"  - P(virginica)  = {proba[0, 2]:.2f}")
print(f"機率總和: {proba.sum():.2f}")  # 應該等於 1.0

# 方法 3: decision_function() - 返回 softmax 轉換前的原始分數
scores = softmax_reg.decision_function(test_sample)
print(f"原始類別分數: {scores.round(2)}")
```

**✅ 程式碼逐行解析：**

1. **predict() 方法**:
   - 返回最高機率的類別標籤（0, 1 或 2）
   - 適合只需要最終預測結果的情況

2. **predict_proba() 方法**:
   - 返回所有類別的機率估計（總和為 1.0）
   - 透過 Softmax 函數計算：$P(y=k|x) = \frac{e^{s_k(x)}}{\sum_{j=1}^{K} e^{s_j(x)}}$
   - 適合需要量化不確定性或設定自訂決策閾值的情況

3. **decision_function() 方法**:
   - 返回 Softmax 轉換前的原始分數（logits）
      - 也就是每個類別的線性組合結果：$s_k(x) = \theta_k^T x$
   - 分數越高表示模型對該類別越有信心
   - 適合進階分析或自訂 Softmax 溫度參數

**🎯 重點摘要:**

- **三種方法各有用途**: predict() 最簡單，predict_proba() 提供機率資訊，decision_function() 給出原始分數
- **機率解釋**: Softmax 確保所有機率為正數且總和為 1，符合機率公理
- **最佳使用情境**: 根據應用需求選擇合適的預測方法

### 範例 18: Softmax 迴歸決策邊界視覺化

此範例展示如何視覺化 Softmax 迴歸的決策區域和機率分布，幫助理解模型如何在二維特徵空間中分類三個 Iris 物種。

```python
from matplotlib.colors import ListedColormap

# 建立自訂顏色映射: 淡黃色=setosa, 淡藍色=versicolor, 淡綠色=virginica
custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

# 建立覆蓋特徵空間的細緻網格
x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),      # 花瓣長度: 0 到 8 cm (500 個點)
    np.linspace(0, 3.5, 200).reshape(-1, 1),    # 花瓣寬度: 0 到 3.5 cm (200 個點)
)
print(f"網格形狀: x0.shape = {x0.shape}, x1.shape = {x1.shape}")

# 將 2D 網格轉換為適合模型預測的格式
X_new = np.c_[x0.ravel(), x1.ravel()]
print(f"用於預測的網格點數量: {X_new.shape[0]} (200×500 網格)")

# 計算網格中每個點的類別機率和預測
y_proba = softmax_reg.predict_proba(X_new)  # 形狀: (100000, 3)
y_predict = softmax_reg.predict(X_new)

# 提取 Iris versicolor 機率用於等高線 (中間類別，最能顯示漸變)
zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

# 繪製決策區域、等高線和資料點
plt.figure(figsize=(10, 4))

# 繪製所有訓練資料點
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica", markersize=8)
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor", markersize=8)
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa", markersize=8)

# 繪製決策區域 (用顏色填充)
plt.contourf(x0, x1, zz, cmap=custom_cmap, alpha=0.3)

# 繪製 versicolor 機率等高線
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1, fontsize=8)

plt.xlabel("花瓣長度 (cm)")
plt.ylabel("花瓣寬度 (cm)")
plt.legend(loc="center left")
plt.axis([0.5, 7, 0, 3.5])
plt.grid(alpha=0.3)
save_fig("softmax_regression_contour_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

#### 1) 視覺化設定
- `ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])`: 建立自訂顏色映射
  - 淡黃色 (#fafab0) 表示 setosa 區域
  - 淡藍色 (#9898ff) 表示 versicolor 區域  
  - 淡綠色 (#a0faa0) 表示 virginica 區域

#### 2) 網格建立
- `np.meshgrid()`: 建立覆蓋整個特徵空間的密集網格
  - x0: 花瓣長度方向，500 個點從 0 到 8 cm
  - x1: 花瓣寬度方向，200 個點從 0 到 3.5 cm
  - 總共 200×500 = 100,000 個網格點

- `np.c_[x0.ravel(), x1.ravel()]`: 將 2D 網格展平並堆疊為樣本矩陣
  - `ravel()` 將 2D 陣列 (200, 500) 展平為 1D 陣列 (100000,)
  - `np.c_[]` 將兩個 1D 陣列堆疊為 2D 矩陣 (100000, 2)

#### 3) 模型預測
- `predict_proba(X_new)`: 計算每個網格點屬於三個類別的機率
- `predict(X_new)`: 確定每個網格點的預測類別
- `y_proba[:, 1]`: 提取 versicolor 的機率，用於繪製等高線

#### 4) 視覺化元素
- **資料點**: 
  - 黃色圓圈 (yo) 表示 setosa
  - 藍色方塊 (bs) 表示 versicolor
  - 綠色三角形 (g^) 表示 virginica

- **決策區域**: `contourf()` 用顏色填充不同的預測區域
- **機率等高線**: `contour()` 顯示 versicolor 機率的變化，"hot" 顏色映射建立漸層效果

**🎯 重點摘要:**

- **決策邊界**: Softmax 迴歸建立複雜的非線性決策邊界來分離多個類別
- **機率漸變**: 等高線顯示機率如何在特徵空間中平滑轉變
- **類別區分**:
  - Iris setosa（左下角）以較小的花瓣測量值容易區分
  - Iris versicolor 和 virginica 之間的邊界更複雜，有些重疊
  - 接近訓練資料的區域顯示更高的模型信心
- **最佳使用情境**: 理解模型的決策過程，診斷分類困難的區域

---

### Softmax 迴歸數學原理

**Softmax 函數**將原始分數（logits）轉換為機率：

$$P(y=k|x) = \frac{e^{s_k(x)}}{\sum_{j=1}^{K} e^{s_j(x)}}$$

其中：
- $s_k(x) = \theta_k^T x$ 是類別 k 的分數（logit）
- $K$ 是類別總數
- $e^{s_k(x)}$ 確保所有值為正數
- 分母正規化確保所有機率總和為 1

**關鍵特性**:
1. **輸出範圍**: 每個機率在 [0, 1] 之間
2. **正規化**: 所有類別機率總和等於 1
3. **單調性**: 分數越高，機率越大
4. **互斥性**: 適合每個樣本只能屬於一個類別的情況

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

#Python #機器學習 #程式設計 #教學 #線性模型 #數據科學 #AI開發
