# 課程講義：訓練線性模型 (Chapter 04)

線性模型是機器學習的基石，許多複雜模型（神經網路、SVM）的核心優化思想都源於此。本章帶你從**正規方程式**、**梯度下降**，到**正則化技術**，完整理解「模型如何被訓練」的數學本質。理解這些原理，你才能在模型表現不佳時，準確判斷是欠擬合還是過擬合，並選擇正確的解決策略。

---

## 1. 線性迴歸與正規方程式

### 理論背景

線性迴歸假設目標值是特徵的線性組合加上雜訊：

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n = \boldsymbol{\theta}^T \mathbf{x}$$

以向量化形式，預測整個訓練集：

$$\hat{\mathbf{y}} = \mathbf{X} \boldsymbol{\theta}$$

其中 $\mathbf{X}$ 是加了偏差欄（全 1）的特徵矩陣。

最小化 MSE 的解析解（**正規方程式**）：

$$\hat{\boldsymbol{\theta}} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}$$

**計算複雜度**：矩陣求逆的複雜度為 $O(n^{2.4} \sim O(n^3)$，$n$ 是特徵數。特徵數超過數千時改用梯度下降更有效率。

### 核心代碼

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import add_dummy_feature

np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# 方法一：手動正規方程式
X_b = add_dummy_feature(X)  # 新增偏差項 x0 = 1
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"theta_best: {theta_best.ravel()}")  # 接近 [4, 3]

# 方法二：Scikit-Learn（內部使用 SVD，更穩健）
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(f"intercept: {lin_reg.intercept_}, coef: {lin_reg.coef_}")

# 預測
X_new = np.array([[0], [2]])
print(lin_reg.predict(X_new))
```

### ⚡ 補充練習 1

**理論題：** 正規方程式需要計算 $(\mathbf{X}^T\mathbf{X})^{-1}$，若矩陣奇異（不可逆），會發生什麼？什麼情況下會出現奇異矩陣？

**實作題：** 生成一個有 3 個特徵的資料集（其中第 3 個特徵是前兩個的線性組合），嘗試用正規方程式求解，觀察 `np.linalg.inv` 的行為；再改用 `LinearRegression` 觀察是否正常運作（提示：它使用 SVD 的 pseudoinverse）。

---

## 2. 梯度下降三種變體

### 理論背景

梯度下降通過反覆沿梯度**反方向**移動來最小化損失函數：

$$\boldsymbol{\theta}^{(\text{next})} = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta})$$

其中 $\eta$ 是**學習率 (learning rate)**，$\nabla_{\boldsymbol{\theta}} \text{MSE} = \frac{2}{m} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$。

| 變體 | 每步計算量 | 收斂特性 | 適用場景 |
|------|----------|---------|---------|
| **批次梯度下降 (BGD)** | 全部 $m$ 筆 | 穩定，慢 | 小資料集，最終微調 |
| **隨機梯度下降 (SGD)** | 1 筆 | 震盪，快 | 大資料集，線上學習 |
| **小批次梯度下降 (Mini-batch)** | $b$ 筆（$32 \sim 256$）| 均衡 | 最常用（尤其深度學習） |

**學習率排程 (Learning Rate Schedule)**：一開始用大學習率快速逼近，後期用小學習率精確收斂。

### 核心代碼

```python
# 批次梯度下降（手動實作，理解原理）
eta = 0.1         # 學習率
n_epochs = 1000
m_train = len(X_b)

np.random.seed(42)
theta = np.random.randn(2, 1)  # 隨機初始化

for epoch in range(n_epochs):
    gradients = 2 / m_train * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

print(f"BGD theta: {theta.ravel()}")

# Scikit-Learn SGDRegressor
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5,
                       penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())
print(f"SGD intercept: {sgd_reg.intercept_}, coef: {sgd_reg.coef_}")
```

### ⚡ 補充練習 2

**理論題：** 學習率 $\eta$ 設定太大或太小，各會造成什麼問題？若損失函數是「碗形」（凸函數），BGD 保證找到全域最小值嗎？若不是凸函數呢？

**實作題：** 對同一個線性迴歸問題，分別用 $\eta = 0.001$、$0.1$、$0.5$ 訓練 BGD，繪製每個 epoch 的訓練損失曲線，直觀觀察學習率對收斂的影響。

---

## 3. 多項式迴歸與學習曲線診斷

### 理論背景

**多項式迴歸**：將特徵轉換為高次冪後，套用線性迴歸，本質上仍是線性模型（對參數線性）。

$$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_d x^d$$

**學習曲線 (Learning Curves)**：同時繪製訓練誤差和驗證誤差隨訓練樣本數增加的變化。

| 現象 | 診斷 |
|------|------|
| 訓練誤差高 & 驗證誤差高，且兩者接近 | **欠擬合**（模型太簡單） |
| 訓練誤差低 & 驗證誤差高，差距大 | **過擬合**（模型太複雜） |

**偏差-變異數分解**：

$$\text{泛化誤差} = \underbrace{\text{偏差}^2}_{\text{欠擬合}} + \underbrace{\text{變異數}}_{\text{過擬合}} + \text{不可減少的雜訊}$$

### 核心代碼

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# 多項式迴歸
poly_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])

# 學習曲線
train_sizes, train_scores, valid_scores = learning_curve(
    poly_reg, X, y.ravel(),
    train_sizes=np.linspace(0.01, 1.0, 30),
    cv=5, scoring="neg_root_mean_squared_error"
)

plt.plot(train_sizes, -train_scores.mean(axis=1), "r-+", label="Training")
plt.plot(train_sizes, -valid_scores.mean(axis=1), "b-",  label="Validation")
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.legend()
plt.title(f"Learning Curves (Polynomial degree={poly_reg['poly_features'].degree})")
plt.grid()
```

### ⚡ 補充練習 3

**理論題：** 觀察 `degree=1`、`degree=2`、`degree=10` 的學習曲線。哪個 degree 可能欠擬合？哪個可能過擬合？如何從曲線形狀判斷？

**實作題：** 生成一個符合 $y = 0.5x^2 + x + 2 + \text{noise}$ 的資料集（100 樣本），分別用 `degree=1`、`degree=2`、`degree=20` 的多項式迴歸擬合，繪製三個模型在資料上的擬合曲線，觀察過擬合與欠擬合現象。

---

## 4. 正則化線性模型：Ridge、Lasso、ElasticNet

### 理論背景

正則化在損失函數中加入懲罰項，限制模型複雜度：

**Ridge 迴歸（L2 正則化）**：

$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \sum_{i=1}^{n} \theta_i^2$$

**Lasso 迴歸（L1 正則化）**：

$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \sum_{i=1}^{n} |\theta_i|$$

Lasso 的特點：傾向將不重要的特徵係數壓縮為**完全 0**，實現稀疏解（自動特徵選擇）。

**ElasticNet**：L1 + L2 的混合：

$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r \alpha \sum|\theta_i| + \frac{1-r}{2} \alpha \sum\theta_i^2$$

| 方法 | 適用場景 | 稀疏性 |
|------|---------|--------|
| Ridge | 多數特徵都有貢獻 | 無（係數縮小但不為零） |
| Lasso | 特徵多，但只有少數重要 | 有（自動特徵選擇） |
| ElasticNet | 高度相關特徵群組 | 有（介於兩者之間） |

### 核心代碼

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X, y)
print(f"Ridge: {ridge_reg.predict([[1.5]])}")

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
print(f"Lasso: {lasso_reg.predict([[1.5]])}")
print(f"Lasso coef (稀疏): {lasso_reg.coef_}")

en_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
en_reg.fit(X, y)

# Early Stopping（用於 SGD 訓練的正則化）
from sklearn.linear_model import SGDRegressor

sgd_early = SGDRegressor(
    max_iter=1,
    warm_start=True,  # 每次 fit() 從上次停的地方繼續
    penalty=None,
    learning_rate="constant",
    eta0=0.002,
    random_state=42
)
# 手動早停：監控驗證誤差，在最低點停止
```

### ⚡ 補充練習 4

**理論題：** 若 $\alpha = 0$，Ridge 和 Lasso 退化為什麼模型？若 $\alpha \to \infty$，模型的係數會變成什麼？

**實作題：** 在一個有 100 個特徵（但只有 10 個真正有用）的資料集上（用 `make_regression(n_features=100, n_informative=10)`），比較 Ridge 和 Lasso 的係數分佈，驗證 Lasso 確實將不重要的係數歸零。

---

## 5. 邏輯迴歸與 Softmax 迴歸

### 理論背景

**邏輯迴歸 (Logistic Regression)**：用於二元分類，輸出屬於正類的機率：

$$\hat{p} = \sigma(\boldsymbol{\theta}^T \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^T \mathbf{x}}}$$

Sigmoid 函數將任意實數映射到 $(0, 1)$。決策邊界：$\hat{p} \geq 0.5 \Leftrightarrow \boldsymbol{\theta}^T \mathbf{x} \geq 0$。

**對數損失（Log Loss / Binary Cross-Entropy）**：

$$J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{p}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

**Softmax 迴歸**：邏輯迴歸推廣到多類別，對每個類別 $k$ 計算分數：

$$s_k(\mathbf{x}) = \boldsymbol{\theta}_k^T \mathbf{x}$$

$$\hat{p}_k = \frac{e^{s_k}}{\sum_{j=1}^{K} e^{s_j}}$$

### 核心代碼

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = (iris.target == 2).astype(int)  # Iris virginica vs rest

# 二元邏輯迴歸
log_reg = LogisticRegression(C=1.0, random_state=42)
log_reg.fit(X_iris, y_iris)
print(log_reg.predict_proba([[5, 2]]))  # [P(非 virginica), P(virginica)]

# Softmax 迴歸（多類別，solver="lbfgs" 預設支援）
softmax_reg = LogisticRegression(C=30, multi_class="multinomial",
                                 solver="lbfgs", random_state=42)
softmax_reg.fit(iris.data, iris.target)
print(softmax_reg.predict([[5, 3, 1.5, 0.3]]))  # 預測類別
print(softmax_reg.predict_proba([[5, 3, 1.5, 0.3]]))  # 三類別機率
```

### ⚡ 補充練習 5

**理論題：** 邏輯迴歸的損失函數（Log Loss）為何不用 MSE？MSE 在此有什麼問題（提示：凸性）？

**實作題：** 使用 Iris 資料集（三類別），訓練 `LogisticRegression(multi_class="multinomial")`，繪製決策邊界（三類別顏色不同），並計算測試集的 `classification_report`。

---

## 結論

本章建立了線性模型訓練的完整理論基礎：

- **正規方程式**提供解析解，適用於小型資料集
- **梯度下降**用迭代方式最小化損失，是深度學習的基礎
- **多項式迴歸 + 學習曲線**診斷過擬合/欠擬合
- **Ridge/Lasso/ElasticNet**透過正則化控制模型複雜度
- **邏輯迴歸與 Softmax**將線性迴歸擴展到分類問題

下一章（Ch05）將介紹 SVM，它以完全不同的角度（最大化邊界）來解決分類問題。

---

## 課後作業

**作業：正則化對比實驗**

使用 `sklearn.datasets.load_diabetes()`（糖尿病資料集）：

1. 用 5-fold CV 比較無正則化的 `LinearRegression`、`Ridge(alpha=0.1/1/10)`、`Lasso(alpha=0.01/0.1/1.0)` 的 RMSE，找出最佳的模型與 `alpha` 值。
2. 對最佳 `Lasso` 模型，印出所有係數，確認哪些特徵被完全歸零（系數 = 0），這些特徵在醫學上是否合理？
3. 繪製**正規化路徑圖**（x 軸為 `log(alpha)`，y 軸為各特徵係數），觀察 `alpha` 如何影響係數大小。
