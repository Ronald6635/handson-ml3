# Ch08 速查表：Dimensionality Reduction

> **核心主旨**：降維加速訓練、去除雜訊、便於視覺化 —— PCA 是首選，t-SNE 專用於視覺化探索。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| PCA | 找最大變異方向，投影到低維空間 | 線性降維，加速訓練，壓縮資料 |
| Explained Variance Ratio | 每個主成分解釋的變異比例 | 決定保留幾個主成分 |
| Randomized PCA | SVD 的隨機近似，速度更快 | 特徵數或樣本數很大時 |
| Incremental PCA | 小批次 PCA，不需全部資料放入記憶體 | 大資料集 / 串流資料 |
| Kernel PCA | 非線性降維，使用 kernel trick | 非線性流形結構的資料 |
| t-SNE | 非線性，保留局部鄰域結構，**僅供視覺化** | 高維資料的 2D/3D 視覺化探索 |
| UMAP | t-SNE 的現代替代，速度更快 | 視覺化 + 偶爾用於特徵萃取 |
| LLE | 局部線性嵌入，保留局部幾何 | 流形學習 |

---

## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `PCA` | `n_components=0.95` (保留 95% 變異) 或整數 | 標準 PCA |
| `IncrementalPCA` | `n_components=154`, `batch_size=1000` | 大資料 PCA |
| `KernelPCA` | `kernel="rbf"`, `gamma=0.04` | 非線性 PCA |
| `TSNE` | `n_components=2`, `perplexity=30`, `random_state=42` | 視覺化（不可反投影） |
| `.explained_variance_ratio_` | – | 各主成分的變異解釋量 |
| `.components_` | – | 主成分向量（shape: n_components × n_features） |
| `.inverse_transform()` | – | 從低維重建（PCA 特有） |

---

## 3. 必備代碼片段

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE

# 標準 PCA（保留 95% 變異）
pca = PCA(n_components=0.95)  # 自動決定維度
X_reduced = pca.fit_transform(X_train)
print(f"原始維度: {X_train.shape[1]}, 降維後: {X_reduced.shape[1]}")
print(f"保留的變異量: {pca.explained_variance_ratio_.sum():.3f}")

# 找出保留 N% 變異所需的維度數
pca_full = PCA()
pca_full.fit(X_train)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumvar >= 0.95) + 1
print(f"保留 95% 變異需要 {n_components_95} 個主成分")

# 繪製 explained variance 曲線
plt.plot(cumvar)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.axhline(y=0.95, color='r', linestyle='--')

# 降維 + 重建（壓縮/去雜訊）
X_recovered = pca.inverse_transform(X_reduced)

# Incremental PCA（大資料集）
inc_pca = IncrementalPCA(n_components=154, batch_size=500)
for X_batch in np.array_split(X_train, 100):
    inc_pca.partial_fit(X_batch)
X_reduced_inc = inc_pca.transform(X_train)

# Kernel PCA（非線性降維）
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04, fit_inverse_transform=True)
X_reduced_kpca = kpca.fit_transform(X_train)

# t-SNE（僅視覺化，不可用於訓練）
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_2d = tsne.fit_transform(X_train[:5000])  # t-SNE 慢，通常只取子集
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train[:5000], cmap="tab10", s=5)
plt.colorbar(); plt.show()

# 在 Pipeline 中使用 PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([
    ("pca", PCA(n_components=0.95)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
```

---

## 4. 常見陷阱

- **PCA 前需要 Scaling**：PCA 基於變異數，特徵尺度不同會讓某些特徵主導主成分，務必先做 `StandardScaler`。
- **t-SNE 不能用於測試集轉換**：t-SNE 沒有 `transform()`，只能 `fit_transform()`，**不能**在 Pipeline 中使用，僅供探索。
- **`n_components=0.95` 是比例**：PCA 中傳入 0~1 的浮點數代表保留的變異比例，傳入整數代表主成分數量。
- **Kernel PCA 選 kernel**：不確定時先試 `rbf`，再視需求試 `sigmoid`、`poly`。

---

## 5. 決策指南

```
選哪種降維方法？
├── 線性降維（加速訓練、壓縮）  → PCA
├── 大資料集 (> 記憶體)          → IncrementalPCA
├── 非線性流形結構                → KernelPCA 或 LLE
├── 視覺化探索（2D/3D）          → t-SNE（< 10k 樣本）或 UMAP（更快）
└── 無監督特徵萃取後做分類       → PCA + 下游分類器

保留多少維度？
├── 一般規則：保留 95% 或 99% 解釋變異
└── 實驗：PCA 後加分類器，用 CV 調整 n_components 
```
