# Ch09 速查表：Unsupervised Learning

> **核心主旨**：無監督學習從資料中發現結構 —— K-Means 快速通用，DBSCAN 抗雜訊，GMM 提供機率估計。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| K-Means | 將樣本分到最近的 K 個群心 | 球形群集、K 已知 |
| K-Means++ 初始化 | 智慧選取初始群心，避免差的局部最小 | 預設使用，幾乎必選 |
| Mini-batch K-Means | 每次只用一小批次更新，速度更快 | 大型資料集 |
| DBSCAN | 用密度定義群集，可發現任意形狀 | 有雜訊、非球形群集 |
| Gaussian Mixture Model | 假設資料由多個高斯分佈混合生成 | 需要機率輸出、橢圓形群集 |
| BIC / AIC | 評估 GMM 模型複雜度，用於選 K | 不知道群集數量時 |
| Silhouette Score | 評估群集質量：[-1, 1]，越高越好 | 評估聚類結果 |
| Semi-supervised Clustering | 用少量標籤引導聚類 | 部分標注的資料集 |

---

## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `KMeans` | `n_clusters=5`, `init="k-means++"`, `n_init=10` | K-Means 聚類 |
| `MiniBatchKMeans` | `n_clusters=5`, `batch_size=1000` | 大資料 K-Means |
| `DBSCAN` | `eps=0.5`, `min_samples=5` | 密度聚類 |
| `AgglomerativeClustering` | `n_clusters=5`, `linkage="ward"` | 層次聚類 |
| `GaussianMixture` | `n_components=5`, `covariance_type="full"` | GMM |
| `BayesianGaussianMixture` | `n_components=10` (自動選 K) | 貝葉斯 GMM（不需指定 K） |
| `silhouette_score` | – | 整體輪廓係數（越高越好） |
| `silhouette_samples` | – | 每個樣本的輪廓係數 |
| `.labels_` | – | 各樣本的群集標籤 |
| `.inertia_` | – | K-Means 群內距離平方和 |
| `.score()` (GMM) | – | 對數似然度（log-likelihood） |

---

## 3. 必備代碼片段

```python
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# K-Means（找最佳 K：Elbow + Silhouette）
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# Elbow 法選 K
plt.plot(k_range, inertias, "bo-")
plt.xlabel("K"); plt.ylabel("Inertia")

# Silhouette 法選 K
plt.plot(k_range, silhouette_scores, "rs-")
plt.xlabel("K"); plt.ylabel("Silhouette Score")

# 使用最佳 K 訓練
km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
y_pred = km.fit_predict(X)
cluster_centers = km.cluster_centers_

# DBSCAN（自動找 outlier）
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)  # -1 代表雜訊點
print(f"群集數: {n_clusters}, 雜訊點: {n_noise}")

# GMM（機率聚類 + 密度估計）
gm = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
gm.fit(X)
proba = gm.predict_proba(X)    # 每個樣本屬於各群集的機率
labels_gm = gm.predict(X)      # 硬性分類

# 用 BIC/AIC 選擇最佳 K
bic_scores = []
for k in k_range:
    gm = GaussianMixture(n_components=k, random_state=42)
    gm.fit(X)
    bic_scores.append(gm.bic(X))
best_k = k_range[np.argmin(bic_scores)]
print(f"BIC 建議 K = {best_k}")

# 異常偵測（GMM 的負對數密度）
densities = gm.score_samples(X)
threshold = np.percentile(densities, 2)  # 最低 2% 為異常
anomaly_mask = densities < threshold
```

---

## 4. 常見陷阱

- **K-Means 前需要 Scaling**：K-Means 基於距離，特徵尺度不同會偏向尺度大的特徵。
- **K-Means 只適合球形群集**：非球形（如半月形、同心圓）需要 DBSCAN 或 GMM。
- **DBSCAN 的 `eps` 參數**：eps 太大 → 全部合成一群；eps 太小 → 全部都是雜訊。建議先用 KNN 距離分佈圖來估計 eps。
- **GMM 的 `covariance_type`**：`"full"` 最靈活但參數最多；`"tied"` 速度快；`"diag"` 無軸向外旋轉；`"spherical"` 最簡單。
- **`BayesianGaussianMixture` 的 K**：設定比真實 K 大即可（例如 10），它會自動將不需要的群集的權重降為 0。

---

## 5. 決策指南

```
選哪種聚類算法？
├── 群集球形、K 已知           → KMeans（快速）
├── 群集形狀任意、有雜訊       → DBSCAN
├── 層次結構（樹狀分類）       → AgglomerativeClustering
├── 需要機率輸出 or 橢圓形群集 → GaussianMixture
└── 不知道 K 是多少            → BayesianGaussianMixture 或 BIC 選 K

異常偵測方法比較：
├── 高維資料                  → IsolationForest（快速）
├── 需要機率解釋              → GaussianMixture（負 log-density）
└── 訓練資料只有正常樣本       → EllipticEnvelope（假設高斯分佈）
```
