# 課程講義：支持向量機 (Chapter 05)

支持向量機（SVM）以一個優雅的幾何直覺為核心：**找到使兩類別「邊界（Margin）最寬」的決策邊界**。這種「最大邊界」的設計讓 SVM 對未見資料有優異的泛化能力。本章從線性可分情形出發，引入「軟邊界」處理雜訊，再透過**核技巧 (Kernel Trick)** 將 SVM 擴展到非線性問題——無需顯式計算高維特徵，卻能等效在高維空間分類。

---

## 1. 線性 SVM 分類：硬邊界與軟邊界

### 理論背景

**硬邊界 (Hard Margin)**：要求所有訓練樣本都嚴格在邊界外側。

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y^{(i)} \left(\mathbf{w}^T \mathbf{x}^{(i)} + b\right) \geq 1, \; \forall i$$

邊界寬度（Margin）= $\frac{2}{\|\mathbf{w}\|}$，最大化 Margin 等效於最小化 $\|\mathbf{w}\|^2$。

**問題**：硬邊界對 outlier 極度敏感，且線性不可分時無解。

**軟邊界 (Soft Margin)**：允許部分樣本違反邊界，引入鬆弛變數 $\zeta_i \geq 0$：

$$\min_{\mathbf{w}, b, \boldsymbol{\zeta}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{m} \zeta_i$$

$$\text{s.t.} \quad y^{(i)} \left(\mathbf{w}^T \mathbf{x}^{(i)} + b\right) \geq 1 - \zeta_i, \quad \zeta_i \geq 0$$

- $C$ 小 → 邊界寬，允許更多違規（正則化強，防過擬合）
- $C$ 大 → 邊界窄，嚴格分類（可能過擬合）

**重要：SVM 對特徵尺度非常敏感，務必先做 `StandardScaler`！**

### 核心代碼

```python
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)  # Iris virginica vs rest

# 軟邊界線性 SVM（標準化必不可少）
svm_clf = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1.0, dual=True, random_state=42)
)
svm_clf.fit(X, y)
print(svm_clf.predict([[5.5, 1.7], [5.0, 1.5]]))
print(svm_clf.decision_function([[5.5, 1.7]]))  # 到決策邊界的距離
```

### ⚡ 補充練習 1

**理論題：** 為什麼 SVM 需要特徵縮放？舉一個具體例子說明，若某特徵的值域是 [0, 1000] 而另一個是 [0, 1]，未縮放的 SVM 決策邊界會偏向哪個方向？

**實作題：** 使用 `make_moons(n_samples=200, noise=0.1)` 資料，分別用 `C=0.001`、`C=1`、`C=1000` 訓練 `LinearSVC`，繪製決策邊界，觀察 $C$ 對邊界形狀和支持向量數量的影響。

---

## 2. 核技巧與非線性 SVM

### 理論背景

對線性不可分的資料，可以先映射到高維空間，再用線性 SVM 分類：

$$\phi: \mathbb{R}^n \to \mathbb{R}^d \quad (d \gg n)$$

**核技巧 (Kernel Trick)**：核函數 $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^T \phi(\mathbf{z})$ 直接計算高維內積，無需顯式映射：

| 核函數 | 數學公式 | 適用場景 |
|--------|---------|---------|
| 線性核 | $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$ | 線性可分 |
| 多項式核 | $K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d$ | 曲線邊界 |
| RBF (Gaussian) | $K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right)$ | 通用，最常用 |
| Sigmoid | $K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$ | 類神經網路 |

**RBF 核超參數**：

- $\gamma$ 大 → 每個樣本的「影響範圍」小，模型複雜（可能過擬合）
- $\gamma$ 小 → 影響範圍大，模型平滑（可能欠擬合）
- $C$ 和 $\gamma$ 通常需要搭配 `GridSearchCV` 調整

### 核心代碼

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC

X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)

# 多項式核 SVM
poly_kernel_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)
poly_kernel_clf.fit(X_moons, y_moons)

# RBF 核 SVM（最常用，先試這個）
rbf_kernel_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", gamma=0.1, C=1.0)
)
rbf_kernel_clf.fit(X_moons, y_moons)

# 超參數搜索
from sklearn.model_selection import GridSearchCV
param_grid = {
    "svc__gamma": [0.01, 0.1, 1.0, 10.0],
    "svc__C":     [0.1, 1.0, 10.0, 100.0]
}
svc_grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                        param_grid, cv=5, scoring="accuracy")
svc_grid.fit(X_moons, y_moons)
print(f"Best params: {svc_grid.best_params_}")
print(f"Best score: {svc_grid.best_score_:.4f}")
```

### ⚡ 補充練習 2

**理論題：** 解釋 RBF 核的物理直覺：為何 $\gamma$ 大的 RBF 核決策邊界看起來像是圍繞每個訓練樣本畫圈？

**實作題：** 用 `GridSearchCV` 在 `make_moons` 資料上搜索 `gamma` ∈ [0.01, 0.1, 1, 10] 與 `C` ∈ [0.1, 1, 10, 100] 的最佳組合，繪製 AUC 熱力圖（x 軸 gamma，y 軸 C）。

---

## 3. SVM 迴歸

### 理論背景

SVM 可以翻轉用於迴歸：**最大化「包含最多訓練樣本的管道（$\epsilon$-tube）」**，而非最大化邊界。

**$\varepsilon$-不敏感損失函數 (SVR 的損失)**：

$$\ell_\varepsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \varepsilon)$$

即在 $\varepsilon$ 範圍內的誤差視為 0（容許小誤差），超出才計入損失。

- $\varepsilon$ 大 → 管道寬，允許更多誤差
- $\varepsilon$ 小 → 管道窄，模型更精確但可能 overfit

**超參數** `C` 和 `epsilon` 需要搭配交叉驗證調整。

### 核心代碼

```python
from sklearn.svm import SVR, LinearSVR
import numpy as np

np.random.seed(42)
X_reg = 2 * np.random.rand(50, 1)
y_reg = (4 + 3 * X_reg + np.random.randn(50, 1)).ravel()

# 線性 SVR（速度快）
lin_svr = make_pipeline(
    StandardScaler(),
    LinearSVR(epsilon=0.5, dual=True, random_state=42)
)
lin_svr.fit(X_reg, y_reg)

# RBF 核 SVR（非線性迴歸）
rbf_svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", gamma=0.1, C=1.0, epsilon=0.1)
)
rbf_svr.fit(X_reg, y_reg)

# 評估
from sklearn.metrics import mean_squared_error
print(f"SVR RMSE: {mean_squared_error(y_reg, rbf_svr.predict(X_reg), squared=False):.3f}")
```

### ⚡ 補充練習 3

**理論題：** 比較 SVR 的 $\varepsilon$-不敏感損失與線性迴歸的 MSE 損失：哪個對 outlier 更魯棒？為什麼？

**實作題：** 生成一個含有 5% outlier 的資料集（正常資料符合 $y = 3x + 2$，outlier 的 $y$ 值隨機在 [-20, 20]），比較 `LinearRegression`、`SVR(kernel="linear", epsilon=0.5)` 和 `HuberRegressor` 的擬合效果。

---

## 4. SVM 背後的對偶問題（選讀）

### 理論背景

SVM 的訓練可透過求解**對偶問題 (Dual Problem)** 來完成，這讓核技巧成為可能：

原始問題（Primal）的 Lagrangian 為：

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^m \alpha_i \left[y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) - 1\right]$$

對偶問題（Dual）：

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} \mathbf{x}^{(i)T} \mathbf{x}^{(j)}$$

$$\text{s.t.} \quad \alpha_i \geq 0, \; \sum_{i=1}^m \alpha_i y^{(i)} = 0$$

關鍵觀察：只有支持向量（$\alpha_i > 0$）對預測有貢獻。決策函數：

$$\hat{y} = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i y^{(i)} K(\mathbf{x}^{(i)}, \mathbf{x}) + b\right)$$

### 核心代碼

```python
# 核 SVM 的支持向量查詢
svc = SVC(kernel="rbf", C=1.0, gamma="scale")
svc.fit(X_moons, y_moons)

# 查看支持向量
print(f"支持向量數量: {len(svc.support_vectors_)}")
print(f"支持向量 shape: {svc.support_vectors_.shape}")
print(f"對偶係數 (alpha) shape: {svc.dual_coef_.shape}")
```

### ⚡ 補充練習 4

**理論題：** KKT 條件告訴我們，只有 $\alpha_i > 0$ 的樣本（支持向量）對決策函數有貢獻。當 $C$ 很大時，支持向量的數量通常會增加還是減少？為什麼？

---

## 結論

SVM 的三大核心思想：

1. **最大化邊界**：比最小化訓練誤差更強調泛化能力
2. **軟邊界（超參數 $C$）**：在邊界寬度和訓練誤差之間取捨
3. **核技巧**：以低計算成本實現高維特徵映射，無需顯式擴展特徵

SVM 在**中小型資料集、高維特徵、非線性邊界**的場景下特別強大。對大型資料集，`LinearSVC` 或 `SGDClassifier` 更有效率。

下一章（Ch06）介紹決策樹，一種完全不同的學習方式。

---

## 課後作業

**作業：Wine 資料集 SVM 分類**

使用 `sklearn.datasets.load_wine()` 資料集（3 類，13 個特徵）：

1. 用 `Pipeline` 搭配 `StandardScaler` + `SVC(kernel="rbf")`，用 `GridSearchCV` 找到最佳 `C` 和 `gamma`（各取 5 個值）。
2. 計算最佳模型在測試集的準確率、F1 分數（macro），並繪製混淆矩陣。
3. **思考題**：若將 `SVC` 改為 `LinearSVC`（線性核），準確率是否下降？這說明了 Wine 資料的邊界是線性還是非線性的？
