# Ch04 速查表：Training Linear Models

> **核心主旨**：線性模型的訓練方法與正則化 —— Ridge/Lasso/ElasticNet 是防止 overfitting 的利器。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Normal Equation | 直接計算 $\hat{\theta} = (X^TX)^{-1}X^Ty$ | 特徵數 < 10k，一步到位 |
| Batch GD | 每次用全部資料計算梯度 | 小資料、凸函數 |
| Stochastic GD (SGD) | 每次用 1 個樣本更新 | 大資料集、online learning |
| Mini-batch GD | 每次用小批次（32-256）更新 | 實務上最常用 |
| Polynomial Regression | 加入 $x^2, x^3, ...$ 特徵擬合非線性 | 非線性資料但想用線性模型 |
| Ridge (L2) | 損失加 $\alpha \sum \theta_i^2$，縮小所有權重 | 所有特徵都有用，需防 overfitting |
| Lasso (L1) | 損失加 $\alpha \sum |\theta_i|$，可將部分權重降為 0 | 特徵選擇（稀疏模型） |
| ElasticNet | L1 + L2 的混合，有 `l1_ratio` 控制比例 | 特徵數 >> 樣本數，或特徵相關時 |
| Early Stopping | 監控 validation loss，停在最低點 | 防止 overfitting 的高效方法 |
| LogisticRegression | 輸出機率，用 sigmoid，做二元分類 | 線性可分的分類問題 |

---

## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `LinearRegression` | – | 最小二乘法迴歸（Normal Equation） |
| `Ridge` | `alpha=1.0` | L2 正則化迴歸 |
| `Lasso` | `alpha=0.1` | L1 正則化迴歸（特徵選擇） |
| `ElasticNet` | `alpha=0.1`, `l1_ratio=0.5` | L1+L2 混合 |
| `SGDRegressor` | `penalty="l2"`, `eta0=0.1`, `max_iter=1000` | SGD 迴歸 |
| `SGDClassifier` | `loss="log_loss"`, `penalty="l2"` | SGD 分類 |
| `PolynomialFeatures` | `degree=2`, `include_bias=False` | 多項式特徵展開 |
| `LogisticRegression` | `C=1.0` (=1/alpha), `max_iter=1000` | 邏輯迴歸 |
| `learning_curve` | – | 繪製學習曲線，診斷 bias/variance |

---

## 3. 必備代碼片段

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve

# 線性迴歸（Normal Equation）
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.intercept_, lin_reg.coef_)

# 多項式迴歸
poly_pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])
poly_pipeline.fit(X_train, y_train)

# Ridge 正則化（L2）
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso 正則化（L1）—— 稀疏模型
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"非零係數數量: {np.sum(lasso.coef_ != 0)}")

# ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

# Early Stopping (SGD)
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty="l2",
                       eta0=0.01, n_iter_no_change=10, random_state=42)
sgd_reg.fit(X_train, y_train.ravel())

# 邏輯迴歸（多類別）
log_reg = LogisticRegression(C=10, max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
print(log_reg.predict_proba(X_test[:3]))  # 輸出機率
```

---

## 4. 常見陷阱

- **Ridge/Lasso 前必須 Scaling**：L1/L2 懲罰對特徵尺度敏感，務必先做 `StandardScaler`。
- **`C` vs `alpha`**：`LogisticRegression`、`SVC` 用 `C`（越大正則化越弱）；`Ridge`、`Lasso` 用 `alpha`（越大正則化越強），兩者互為倒數概念。
- **Lasso 不穩定性**：當特徵相關時，Lasso 可能隨機選其中一個降為 0；ElasticNet 更穩健。
- **`max_iter` 不夠大**：LogisticRegression 預設 `max_iter=100`，複雜問題常需要 `1000+`。

---

## 5. 決策指南

```
選哪種正則化？
├── 所有特徵都有用，只需防止 overfitting → Ridge (L2)
├── 特徵多，需要自動選特徵（稀疏） → Lasso (L1)
└── 特徵相關 or 特徵數 >> 樣本數   → ElasticNet (L1+L2)

選哪種訓練方法？
├── 特徵數 < 10k，資料量 < 1M → LinearRegression (Normal Equation, 快速精確)
├── 特徵數 > 10k 或資料量 > 1M → SGDRegressor (增量學習)
└── 需要 online learning          → SGDRegressor (partial_fit)

正則化強度 alpha：
└── 用 GridSearchCV 搜索 [0.01, 0.1, 1, 10, 100]
```

**學習曲線診斷**：
- Training error 高 → Underfitting（增加模型複雜度或特徵）
- Training/Val error gap 大 → Overfitting（增加正則化、減少特徵、更多資料）
