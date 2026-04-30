# 課程講義：無監督學習 (Chapter 09)

無監督學習的任務是在**沒有標籤**的資料中發現隱藏結構。本章介紹三大無監督學習技術：**聚類（K-Means、DBSCAN）**、**高斯混合模型（GMM）**，以及它們在半監督學習、影像分割、異常偵測等實際場景的應用。理解這些方法，讓你能從未標記資料中萃取有價值的洞察。

---

## 1. K-Means 聚類

### 理論背景

**K-Means 演算法**：

1. 隨機初始化 $K$ 個群心（Centroid）
2. **分配步驟**：將每個樣本分配到最近的群心
3. **更新步驟**：將每個群心移至分配到它的所有樣本的平均位置
4. 重複步驟 2-3 直到收斂

**目標函數（惰性 Inertia）**：最小化各樣本到所屬群心的距離平方和：

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x}^{(i)} \in C_k} \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|^2$$

**K-Means++ 初始化**：以距離加權的方式選擇初始群心（更遠的點更可能被選中），避免糟糕的隨機初始化，是 sklearn 的預設行為。

**K-Means 的限制**：

- 需要預先指定 $K$
- 假設群集是球形且大小相近
- 對 outlier 和非凸形狀的群集效果差
- 結果可能因初始化不同而異（用 `n_init=10` 多次執行取最佳）

### 核心代碼

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# 生成測試資料
X_blobs, y_blobs = make_blobs(n_samples=500, n_features=2,
                              centers=5, cluster_std=0.8, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans.fit(X_blobs)

print(f"群心:\n{kmeans.cluster_centers_}")
print(f"各樣本標籤 (前 10): {kmeans.labels_[:10]}")
print(f"Inertia: {kmeans.inertia_:.2f}")

# 繪製聚類結果
plt.scatter(X_blobs[:, 0], X_blobs[:, 1],
            c=kmeans.labels_, cmap="tab10", s=10, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.legend()
plt.title("K-Means Clustering")
plt.show()

# Mini-Batch K-Means（大資料集）
mb_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42)
mb_kmeans.fit(X_blobs)
```

### ⚡ 補充練習 1

**理論題：** K-Means 的 Inertia 隨著 $K$ 增大一定會減小（因為群集越多，每個樣本越接近群心）。那麼只看 Inertia 是否能選到最佳的 $K$？Elbow Method 如何利用這個特性？

**實作題：** 對 `make_blobs` 資料，用 $K = 2, 3, 4, 5, 6, 7, 8, 9, 10$ 分別訓練 K-Means，繪製 Inertia 和 Silhouette Score 對 $K$ 的曲線，用 Elbow Method 找到最佳 $K$。

---

## 2. K-Means 的應用：影像分割與半監督學習

### 理論背景

**影像分割（顏色量化）**：

- 將圖片的每個像素視為一個 RGB 資料點
- 用 K-Means 找到 $K$ 個代表顏色（群心）
- 將每個像素替換為其群心的顏色
- 大幅壓縮影像所需的顏色數量

**半監督學習 (Semi-supervised Learning)**：

- 有少量標籤資料 + 大量未標籤資料
- 策略：先用 K-Means 聚類，找出每個群集的代表樣本，手動標記代表樣本，再用標記資訊標記同一群集的其他樣本
- 這種「標籤傳播 (Label Propagation)」策略能用少量標籤達到接近全標籤的效能

**選擇距離閾值**：只傳播給與群心距離在某個閾值內的樣本，避免邊界模糊的樣本干擾。

### 核心代碼

```python
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = load_digits()
X_digits, y_digits = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42
)

# 基準線：只用 50 個標籤樣本
n_labeled = 50
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print(f"只用 {n_labeled} 個標籤的準確率: {log_reg.score(X_test, y_test):.4f}")

# 半監督：K-Means 聚類 → 標籤傳播
k_clusters = 50
kmeans_semi = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
X_digits_dist = kmeans_semi.fit_transform(X_train)  # shape: (n_train, k_clusters)

# 找每個群集中最具代表性的樣本（最接近群心）
representative_idx = np.argmin(X_digits_dist, axis=0)
X_representative = X_train[representative_idx]
y_representative = y_train[representative_idx]

# 傳播標籤：將每個代表樣本的標籤傳播給同群集的所有樣本
y_train_propagated = y_representative[kmeans_semi.labels_]

log_reg_semi = LogisticRegression(max_iter=10000, random_state=42)
log_reg_semi.fit(X_train, y_train_propagated)  # 全部訓練資料但用傳播標籤
print(f"半監督準確率: {log_reg_semi.score(X_test, y_test):.4f}")
```

### ⚡ 補充練習 2

**理論題：** 在半監督學習中，若某個群集的代表樣本被錯誤標記，它的標籤會傳播給整個群集的樣本，造成系統性錯誤。如何設計驗證機制來偵測或緩解這個問題？

**實作題：** 實作帶「距離閾值過濾」的標籤傳播：只對距群心距離在第 20 百分位數以內的樣本傳播標籤（其他樣本不使用），比較有無過濾的半監督準確率。

---

## 3. DBSCAN：密度聚類

### 理論背景

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**：

核心概念：
- **核心點 (Core Point)**：在 $\varepsilon$ 半徑內有至少 `min_samples` 個鄰居
- **邊界點 (Border Point)**：不是核心點，但在某個核心點的 $\varepsilon$ 鄰域內
- **雜訊點 (Noise Point)**：既非核心點也非邊界點，標記為 $-1$

優點（vs K-Means）：

- **不需要預先指定 $K$**
- 能發現**任意形狀**的群集
- 對 outlier 魯棒（自動標記為雜訊）

缺點：

- 對超參數 `eps` 和 `min_samples` 敏感
- 對不同密度的群集效果差
- 時間複雜度 $O(m \log m)$（搭配空間索引）

### 核心代碼

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=42)

# DBSCAN
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X_moons)

print(f"群集標籤: {np.unique(dbscan.labels_)}")  # -1=雜訊, 0=群集0, 1=群集1
print(f"核心點數量: {len(dbscan.core_sample_indices_)}")
print(f"雜訊點數量: {(dbscan.labels_ == -1).sum()}")

# 視覺化
colors = ["blue" if l == 0 else "orange" if l == 1 else "red"
          for l in dbscan.labels_]
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=colors, s=10)
plt.title("DBSCAN on Two Moons (red=noise)")
plt.show()

# DBSCAN 沒有 predict()，新樣本需用 KNN 傳播
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
non_noise = dbscan.labels_ != -1
knn_clf.fit(X_moons[non_noise], dbscan.labels_[non_noise])
print(knn_clf.predict([[0, 0.5], [-0.5, 0]]))
```

### ⚡ 補充練習 3

**理論題：** DBSCAN 的 `eps` 參數應如何設定？有一種經驗方法是繪製「k 距離圖（k-distance graph）」——對每個點計算其第 k 個最近鄰的距離，排序後繪製，在曲線「膝部」選取 `eps`。直觀地說明為什麼這個方法有效。

**實作題：** 用 `make_circles(n_samples=500, noise=0.05)` 生成兩個同心圓資料，分別用 K-Means（$K=2$）和 DBSCAN 聚類，比較結果。解釋為何 K-Means 在此失敗而 DBSCAN 成功。

---

## 4. 高斯混合模型與異常偵測

### 理論背景

**高斯混合模型 (Gaussian Mixture Model, GMM)**：假設資料由 $K$ 個高斯分布混合而成：

$$p(\mathbf{x}) = \sum_{k=1}^{K} \phi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

其中 $\phi_k$ 是第 $k$ 個成分的混合係數（$\sum_k \phi_k = 1$）。

**EM 演算法 (Expectation-Maximization)**：

- **E 步**：計算每個樣本屬於各成分的責任值（Responsibility）
- **M 步**：根據責任值更新 $\phi_k$、$\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$

**選擇 $K$（模型複雜度）**：使用 BIC 或 AIC 平衡擬合程度與模型複雜度：

$$\text{BIC} = k \ln(m) - 2 \ln(\hat{L})$$

$$\text{AIC} = 2k - 2 \ln(\hat{L})$$

$k$ 是模型參數數量，$\hat{L}$ 是最大似然值。**越小越好**。

**異常偵測**：低密度區域的樣本為異常——設定閾值 `score_samples(X) < threshold`。

### 核心代碼

```python
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import load_iris

iris = load_iris()

# 訓練 GMM
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(iris.data)

print(f"均值:\n{gm.means_}")
print(f"協方差 shape: {gm.covariances_.shape}")  # (3, 4, 4)
print(f"混合係數: {gm.weights_}")

# 選擇最佳 K（用 BIC）
bics = [GaussianMixture(n_components=k, n_init=5, random_state=42)
        .fit(iris.data).bic(iris.data)
        for k in range(1, 10)]
best_k = range(1, 10)[np.argmin(bics)]
print(f"BIC 選出的最佳 K: {best_k}")

# 異常偵測（密度低 = 異常）
densities = gm.score_samples(iris.data)  # 對數密度
anomaly_threshold = np.percentile(densities, 4)   # 最低 4% 為異常
anomalies = iris.data[densities < anomaly_threshold]
print(f"偵測到 {len(anomalies)} 個異常點")

# 貝葉斯 GMM（自動決定有效成分數）
bgm = BayesianGaussianMixture(n_components=10,     # 上限設大一點
                               n_init=5, random_state=42)
bgm.fit(iris.data)
print(f"有效成分（權重 > 0.01）: {(bgm.weights_ > 0.01).sum()}")
```

### ⚡ 補充練習 4

**理論題：** GMM 是軟聚類（每個樣本屬於每個群集的機率），而 K-Means 是硬聚類（每個樣本只屬於一個群集）。GMM 的協方差矩陣 $\boldsymbol{\Sigma}_k$ 的形狀參數（`covariance_type`）有 `full`、`tied`、`diag`、`spherical` 四種，它們各假設什麼形狀的群集？

**實作題：** 用 `BayesianGaussianMixture(n_components=10)` 在 Iris 資料集上訓練，觀察 `bgm.weights_`，確認只有 3 個成分的權重顯著大於 0（對應三個 Iris 種類）。再用這個模型生成 200 個新樣本（`bgm.sample(200)`），繪製生成樣本與原始資料的分佈對比。

---

## 結論

無監督學習的工具箱：

- **K-Means**：快速、高效，適合球形、大小相近的群集；應用廣泛（影像分割、半監督學習）
- **DBSCAN**：密度聚類，自動偵測群集數量，適合任意形狀，對 outlier 魯棒
- **GMM**：機率模型，支援軟聚類、生成新樣本和異常偵測；BIC/AIC 選擇 $K$
- **BayesianGMM**：自動決定有效成分數，無需手動調 $K$

下一章（Ch10）開始深度學習之旅，用 Keras 建構神經網路！

---

## 課後作業

**作業：客戶分群分析（聚類應用）**

使用 `sklearn.datasets.fetch_california_housing()` 的地理資訊（`longitude`、`latitude`）：

1. 用 K-Means（$K=8$）對加州各地區的地理位置聚類，代表 8 個「地理區域」。視覺化聚類結果（散點圖，顏色=群集，大小=房價）。

2. 新增「到每個群心的距離」作為 8 個新特徵，加入原有特徵，訓練 `RandomForestRegressor`，與不加這些特徵的基準模型比較 RMSE（5-fold CV）。

3. 用 `GaussianMixture` + `score_samples` 找出加州房價資料中的**地理位置異常點**（與大多數房屋聚集地點不同的地點），在地圖上標注這些點。
