<!-- meta-title: 訓練線性模型完整指南：梯度下降、正則化與邏輯回歸實戰 -->
<!-- meta-description: 深入線性回歸、梯度下降三種變體、多項式回歸、Ridge/Lasso/Elastic Net 正則化、Early Stopping、邏輯回歸和 Softmax 迴歸。包含 Scikit-Learn 實戰程式碼與逐行解析。 -->
<!-- meta-keywords: Python, 線性回歸, 梯度下降, 正則化, Ridge, Lasso, 邏輯回歸, Softmax, 機器學習, Scikit-Learn -->
<!-- meta-hashtags: #Python #線性模型 #梯度下降 #機器學習 #ScikitLearn #邏輯回歸 #正則化 #程式設計 #教學 #DataScience -->

# 🐍 訓練線性模型：梯度下降、正則化到邏輯回歸完整實戰

線性模型（Linear Models）是機器學習的基石。看似簡單，卻是理解梯度下降和正則化的最佳起點——這兩個概念貫穿整個深度學習。本教學從最小二乘法出發，帶你深入了解每種梯度下降的差異，以及如何用正則化控制過擬合。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [📐 線性回歸](#linear-regression)
- [⬇️ 梯度下降三種變體](#gradient-descent)
- [🌀 多項式回歸](#polynomial-regression)
- [📉 學習曲線診斷](#learning-curves)
- [🛡️ 正則化線性模型](#regularization)
- [🔀 邏輯回歸與 Softmax](#logistic-softmax)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **正規方程式（Normal Equation）** 一步求解，但計算複雜度為 O(n³)，特徵多時不可行
- **批次/隨機/小批次梯度下降（Batch/SGD/Mini-batch GD）** 各有速度與穩定性的取捨
- **學習率（Learning Rate）** 是最重要的超參數之一，過大發散、過小收斂慢
- **Ridge/Lasso/Elastic Net** 分別對應 L2/L1/混合正則化，各有適用場景
- **邏輯回歸（Logistic Regression）** 輸出機率，適合二元分類；**Softmax** 推廣至多類別

---

## <a id="linear-regression"></a>📐 線性回歸 (Linear Regression)

💡 **實際應用情境：** 預測電費（用電度數→費用）、預測廣告效果（投放金額→銷售額），任何具有線性關係的回歸問題都可以從線性回歸開始。

線性回歸模型：

$$\hat{y} = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n = \theta^T \mathbf{x}$$

訓練目標：最小化均方誤差（MSE）：$\mathcal{L} = \frac{1}{m}\|\mathbf{X}\theta - \mathbf{y}\|^2$

### 範例 1: 正規方程式 vs Sklearn LinearRegression

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# 生成線性資料 y = 4 + 3x + noise
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.ravel() + np.random.randn(100)

# 方法 1：正規方程式（直接求解析解）
X_b = np.c_[np.ones((100, 1)), X]    # 加入偏置項 x0 = 1
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  # θ = (XᵀX)⁻¹ Xᵀy
print(f"正規方程式 θ: {theta_best}")   # 應接近 [4, 3]

# 方法 2：Sklearn（使用 SVD，更穩定）
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(f"Sklearn 截距: {lin_reg.intercept_:.3f}, 係數: {lin_reg.coef_[0]:.3f}")

# 預測
X_new = np.array([[0], [2]])
print(f"預測: {lin_reg.predict(X_new)}")
```

**✅ 程式碼逐行解析：**

1. `np.c_[np.ones((100, 1)), X]`: 加入全 1 的第 0 列作為偏置項（截距）
2. `np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y`: 矩陣形式的正規方程式 $\hat{\theta} = (X^TX)^{-1}X^Ty$
3. Sklearn 內部使用 SVD 分解（奇異值分解），比直接求逆更數值穩定

**🎯 重點摘要:**

- 正規方程式的計算複雜度是 O(n³)（n 為特徵數）：特徵數 < 10,000 時可用，否則改用梯度下降
- `LinearRegression` 無超參數，訓練後應檢查殘差是否呈常態分佈（診斷模型假設是否成立）

---

## <a id="gradient-descent"></a>⬇️ 梯度下降三種變體

💡 **實際應用情境：** 梯度下降是神經網路訓練的核心。理解三種變體有助於你在實際專案中選擇合適的優化器，並調整學習率等關鍵超參數。

### 範例 2: 批次梯度下降 (Batch Gradient Descent)

```python
# 批次梯度下降（每次迭代使用全部訓練資料）
eta = 0.1          # 學習率
n_iterations = 1000
m = len(X_b)       # 訓練樣本數

theta = np.random.randn(2, 1)  # 隨機初始化

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T @ (X_b @ theta - y.reshape(-1, 1))  # MSE 梯度
    theta = theta - eta * gradients  # 參數更新

print(f"批次 GD θ: {theta.ravel()}")  # 應接近 [4, 3]
```

**✅ 程式碼逐行解析：**

1. `2/m * X_b.T @ (X_b @ theta - y)`: MSE 對 θ 的梯度 $\nabla_\theta = \frac{2}{m} X^T(X\theta - y)$
2. `theta = theta - eta * gradients`: 沿負梯度方向移動一步（學習率控制步長）
3. 批次 GD 每步需要計算全部 m 個樣本的梯度，大資料集時計算成本高但收斂穩定

### 範例 3: 隨機梯度下降 (Stochastic GD)

```python
from sklearn.linear_model import SGDRegressor

# 每次只用一個隨機樣本更新參數（快但有噪音）
sgd_reg = SGDRegressor(
    max_iter=1000,
    tol=1e-5,
    penalty=None,      # 無正則化
    eta0=0.01,         # 初始學習率
    random_state=42
)
sgd_reg.fit(X, y.ravel())
print(f"SGD θ: [{sgd_reg.intercept_[0]:.3f}, {sgd_reg.coef_[0]:.3f}]")
```

| 方法 | 每步使用資料量 | 速度 | 穩定性 | 適合場景 |
|------|--------------|------|--------|---------|
| 批次 GD | 全部 m 個 | 慢 | 高 | 小資料集 |
| 隨機 GD (SGD) | 1 個 | 快 | 低 | 大資料集、線上學習 |
| 小批次 GD | b 個 (32~256) | 中 | 中 | 深度學習標準 |

**🎯 重點摘要:**

- 學習率過高：損失函數「跳躍」甚至發散；過低：收斂極慢
- **學習率排程（Learning Rate Schedule）**：先用大學習率快速靠近，再縮小精細調整

---

## <a id="polynomial-regression"></a>🌀 多項式回歸 (Polynomial Regression)

### 範例 4: 用線性模型擬合非線性資料

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)
m = 100
X_poly = 6 * np.random.rand(m, 1) - 3          # X ∈ [-3, 3]
y_poly = 0.5 * X_poly**2 + X_poly + 2 + np.random.randn(m, 1)  # 二次函數 + 雜訊

# PolynomialFeatures：將 X 擴展為 [1, X, X²]
poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),  # 生成 X, X²
    LinearRegression()
)
poly_model.fit(X_poly, y_poly)

# 實際係數應接近 [1, 0.5]（線性項和平方項）
lin_reg_poly = poly_model.named_steps["linearregression"]
print(f"係數: {lin_reg_poly.coef_}")
print(f"截距: {lin_reg_poly.intercept_}")
```

**✅ 程式碼逐行解析：**

1. `PolynomialFeatures(degree=2)`: 將輸入 `[X]` 轉換為 `[X, X²]`（不含偏置項，LinearRegression 自動加）
2. `make_pipeline(...)`: 比 `Pipeline([...])` 更簡潔，自動以小寫類名命名步驟

**🎯 重點摘要:**

- 多項式回歸本質上是線性模型（對 θ 線性），只是對輸入特徵做非線性變換
- degree=300 的多項式可以完美擬合訓練集，但對新資料毫無用處（過擬合）

---

## <a id="learning-curves"></a>📉 學習曲線診斷

### 範例 5: 學習曲線診斷過擬合/欠擬合

```python
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def plot_learning_curve(model, X, y, cv=3):
    """繪製訓練集和驗證集的學習曲線"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    train_rmse = -train_scores.mean(axis=1)
    val_rmse   = -val_scores.mean(axis=1)

    plt.plot(train_sizes, train_rmse, "b-o", label="訓練誤差")
    plt.plot(train_sizes, val_rmse,   "r-o", label="驗證誤差")
    plt.xlabel("訓練樣本數")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.sca(axes[0])
plt.title("線性回歸（欠擬合）")
plot_learning_curve(LinearRegression(), X_poly, y_poly)

plt.sca(axes[1])
plt.title("10 次多項式（過擬合）")
plot_learning_curve(
    make_pipeline(PolynomialFeatures(degree=10), LinearRegression()),
    X_poly, y_poly
)
plt.show()
```

**🎯 重點摘要:**

- **欠擬合（高偏差）**：訓練誤差和驗證誤差都高，且隨資料量增加不顯著改善
- **過擬合（高方差）**：訓練誤差很低，驗證誤差比訓練誤差高很多（間距大）
- 解法：欠擬合→增加特徵或提升模型複雜度；過擬合→增加資料或正則化

---

## <a id="regularization"></a>🛡️ 正則化線性模型

### 範例 6: Ridge、Lasso、Elastic Net 比較

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

np.random.seed(42)
X_reg = 3 * np.random.rand(100, 1)
y_reg = 1 + 0.5 * X_reg.ravel() + np.random.randn(100) / 1.5

# Ridge（L2 正則化）：縮小所有係數，但不會歸零
ridge_reg = Ridge(alpha=1.0)       # alpha 越大，正則化越強
ridge_reg.fit(X_reg, y_reg)

# Lasso（L1 正則化）：係數可被壓縮至 0（自動特徵選擇！）
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_reg, y_reg)

# Elastic Net（L1 + L2 混合）
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50% L1 + 50% L2
elastic_net.fit(X_reg, y_reg)

print(f"Ridge 係數:  {ridge_reg.coef_[0]:.4f}")
print(f"Lasso 係數:  {lasso_reg.coef_[0]:.4f}")
print(f"ElasticNet:  {elastic_net.coef_[0]:.4f}")
```

| 方法 | 正則化項 | 係數特性 | 適用場景 |
|------|---------|---------|---------|
| Ridge | $\alpha \|\theta\|_2^2$ | 縮小但不歸零 | 所有特徵都有用時 |
| Lasso | $\alpha \|\theta\|_1$ | 稀疏（部分歸零） | 特徵選擇、高維資料 |
| Elastic Net | L1 + L2 混合 | 稀疏+穩定 | 相關特徵多時 |

**🎯 重點摘要:**

- `alpha=0` → 無正則化（退化為普通線性回歸）
- Lasso 可用作**特徵選擇**工具：不重要的特徵係數會被壓縮至 0

---

## <a id="logistic-softmax"></a>🔀 邏輯回歸與 Softmax

### 範例 7: 邏輯回歸二元分類

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 使用鳶尾花資料集（只取前兩個特徵便於視覺化）
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = (iris.target == 2)  # 只分辨 Virginica vs 其他（二元）

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_iris, y_iris)

# 輸出機率（不只是預測類別）
print(f"預測機率: {log_reg.predict_proba([[5, 2]])[0]}")  # [P(非Virginica), P(Virginica)]
print(f"決策邊界係數: {log_reg.coef_[0]}")
```

```python
# Softmax 回歸：多類別邏輯回歸
softmax_reg = LogisticRegression(
    multi_class="multinomial",  # Softmax（而非 OvR）
    solver="lbfgs",
    C=30,                       # C = 1/alpha（C 越小正則化越強）
    random_state=42
)
softmax_reg.fit(X_iris, iris.target)  # 預測 0/1/2 三類
print(f"Softmax 準確率: {softmax_reg.score(X_iris, iris.target):.4f}")
```

**✅ 程式碼逐行解析：**

1. 邏輯回歸輸出 Sigmoid 機率：$\hat{p} = \sigma(\theta^T \mathbf{x}) = \frac{1}{1+e^{-\theta^T \mathbf{x}}}$
2. `multi_class="multinomial"`: 使用 Softmax 而非 OvR，輸出所有類別的機率（加總為 1）
3. `C=30`: 弱正則化（高 C = 允許更複雜邊界）

**🎯 重點摘要:**

- 邏輯回歸輸出**機率**（非直接 0/1），適合需要置信度的場景
- Softmax 的損失函數（交叉熵）是 NLP 和深度學習中最常用的損失函數之一

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: 批次 GD 和 SGD 哪個更好？**

A: 取決於資料規模。資料量 < 10 萬可用批次 GD；更大的資料集用 SGD 或小批次 GD。深度學習中幾乎都使用小批次 GD（batch_size=32~256）。

**Q2: Ridge 和 Lasso 如何選擇？**

A: 若你認為**所有特徵都有用**，用 Ridge；若認為**大多數特徵無關**（需要特徵選擇），用 Lasso；若不確定，先試 Elastic Net。

**Q3: 邏輯回歸為什麼叫「回歸」？**

A: 因為它輸出的是**機率**（連續值），本質上在對 log-odds 做線性回歸。最後的「預測類別」只是在機率 > 0.5 時判為正類。

**Q4: 學習率如何選擇？**

A: 常用方法：(1) 搜索網格 `[1e-4, 1e-3, 1e-2, 0.1]`；(2) 學習率衰減策略（指數衰減、余弦退火）；(3) 使用 Keras Tuner 自動搜索。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #線性模型 #梯度下降 #機器學習 #ScikitLearn #邏輯回歸 #正則化 #Ridge #Lasso #程式設計 #教學 #DataScience #MachineLearning #AI
