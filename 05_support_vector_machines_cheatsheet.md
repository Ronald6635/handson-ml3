# Ch05 速查表：Support Vector Machines (SVM)

> **核心主旨**：SVM 最大化類別間距（margin）—— 特徵縮放是成敗關鍵，kernel trick 讓線性模型具備非線性能力。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Hard Margin | 所有樣本必須在正確側，不容錯 | 線性可分且無 outlier |
| Soft Margin (`C`) | 允許部分違規，`C` 越大 margin 越小 | 有雜訊的實務資料 |
| Kernel Trick | 隱式映射到高維空間，不需顯式計算 | 非線性分類 |
| RBF Kernel | $K(x,z) = e^{-\gamma\|x-z\|^2}$，`gamma` 控制局部化 | 最常用的非線性 kernel |
| Polynomial Kernel | $(x \cdot z + r)^d$，`degree` 控制多項式次數 | 低次多項式邊界 |
| SVR (epsilon-insensitive) | 在 $\epsilon$ 管道內的誤差不計入損失 | 迴歸，允許一定誤差容忍 |
| Support Vectors | 距離 margin 最近的訓練樣本，決定超平面 | 理解模型複雜度 |

---

## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `LinearSVC` | `C=1.0`, `max_iter=1000` | 線性 SVM（大資料集快） |
| `SVC` | `kernel="rbf"/"poly"/"linear"`, `C=1.0`, `gamma="scale"` | 非線性 SVM |
| `SVR` | `kernel="rbf"`, `C=1.0`, `epsilon=0.1` | SVM 迴歸 |
| `StandardScaler` | – | **必備**！SVM 對特徵尺度極敏感 |
| `make_pipeline` | – | 快速建立 Scaler + SVM Pipeline |
| `GridSearchCV` | `param_grid={"svc__C": [...], "svc__gamma": [...]}` | 調優 C 與 gamma |

---

## 3. 必備代碼片段

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV

# 線性 SVM（最快，適合大資料）
linear_svm = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=2000, random_state=42))
linear_svm.fit(X_train, y_train)

# RBF kernel SVM（最常用的非線性分類）
svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma=0.05))
svm_clf.fit(X_train, y_train)

# Polynomial kernel
poly_svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)
poly_svm.fit(X_train, y_train)

# SVR 迴歸
svr_reg = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1))
svr_reg.fit(X_train, y_train)

# 超參數調優（C 與 gamma）
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1, "scale"]
}
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# 取得 support vectors 數量
svc = SVC(kernel="rbf", C=5, gamma=0.05)
svc.fit(X_train_scaled, y_train)
print(f"Support vectors: {svc.n_support_}")  # 每個類別的 support vector 數
```

---

## 4. 常見陷阱

- **忘記 Scaling**：SVM 對特徵尺度極度敏感，**永遠要先 StandardScaler**。
- **`LinearSVC` vs `SVC(kernel="linear")`**：前者用最佳化演算法（較快），後者用 kernel trick（較慢但支援更多選項）。
- **`C` 的直覺**：`C` 小 → 大 margin 但更多違規（高 bias，低 variance）；`C` 大 → 小 margin 但少違規（低 bias，高 variance）。
- **`gamma` 的直覺**：`gamma` 大 → 每個樣本影響範圍小（過擬合傾向）；`gamma` 小 → 影響範圍大（欠擬合傾向）。
- **`SVC` 不支援 `predict_proba` 預設**：需加 `probability=True`（訓練時間增加）。

---

## 5. 決策指南

```
選哪種 SVM？
├── 線性可分或特徵數很多（NLP） → LinearSVC（快）
├── 非線性、中小資料集 (< 50k) → SVC(kernel="rbf")
└── 需要迴歸                    → SVR

C 與 gamma 調優範圍建議：
├── C: [0.01, 0.1, 1, 10, 100]（log scale）
└── gamma: [0.001, 0.01, 0.1, 1] 或 "scale"（推薦起點）

SVM vs 其他演算法：
├── 中小資料集、明確間隔 → SVM 可能優於 GBM
├── 大資料集 (> 100k)    → GBM 或深度學習更快
└── 需要可解釋性          → 決策樹或線性模型
```
