<!-- meta-title: 機器學習無監督學習：聚類、降維與異常檢測完整指南 - Python實作教學 -->
<!-- meta-description: 深入探討無監督學習的核心技術，包括K-Means、DBSCAN、高斯混合模型、PCA等演算法。通過實際案例學習如何應用這些技術進行聚類分析、圖像分割、異常檢測和降維。包含完整的程式碼實作、數學推導和最佳實踐。 -->
<!-- meta-keywords: 機器學習, 無監督學習, 聚類, K-Means, DBSCAN, 高斯混合模型, PCA, 降維, 異常檢測, Python, 教學 -->
<!-- meta-hashtags: #機器學習 #無監督學習 #聚類分析 #KMeans #DBSCAN #高斯混合模型 #PCA #降維 #異常檢測 #Python教學 #程式設計 #編程 #開發 #技術分享 #學習筆記 #程式開發者 #軟體工程 -->

# 🐍 機器學習無監督學習：聚類、降維與異常檢測完整指南

無監督學習是機器學習的重要分支，不依賴標籤資料就能從資料中發現隱藏的模式和結構。本章將深入探討聚類分析、降維技術和異常檢測等核心概念，通過實際案例展示如何應用這些技術解決真實世界的問題。從K-Means到高斯混合模型，從PCA到流形學習，我們將一步步構建完整的無監督學習知識體系。

## 📝 本文目錄
- [🎯 關鍵重點 (Key Takeaways)](#關鍵重點)
- [🐍 設定與準備](#設定與準備)
- [📊 聚類分析 (Clustering)](#聚類分析)
- [🎯 K-Means演算法](#k-means演算法)
- [🔄 K-Means變體](#k-means變體)
- [🖼️ 使用聚類進行圖像分割](#使用聚類進行圖像分割)
- [📈 使用聚類進行半監督學習](#使用聚類進行半監督學習)
- [🔍 DBSCAN](#dbscan)
- [📏 其他聚類演算法](#其他聚類演算法)
- [🌟 高斯混合模型 (Gaussian Mixtures)](#高斯混合模型)
- [🚨 使用高斯混合模型進行異常檢測](#使用高斯混合模型進行異常檢測)
- [📊 選擇聚類數量](#選擇聚類數量)
- [🎲 貝葉斯高斯混合模型](#貝葉斯高斯混合模型)
- [📚 習題解答](#習題解答)
- [💡 總結與最佳實踐](#總結與最佳實踐)
- [❓ 常見問答 (FAQ)](#常見問答)
- [🏷️ 推薦標籤 (Suggested Hashtags)](#推薦標籤)

## 🎯 關鍵重點 (Key Takeaways)
- 無監督學習能夠在沒有標籤的情況下發現資料的隱藏結構
- K-Means是最簡單且最廣泛使用的聚類演算法之一
- DBSCAN能夠發現任意形狀的聚類且對雜訊具有魯棒性
- 高斯混合模型提供了更靈活的聚類和密度估計能力
- PCA是降維最常用的技術，能夠保留資料的主要變異性
- 異常檢測可以用於識別資料中的異常點或新奇點

<a id="設定與準備"></a>
## 🐍 設定與準備

在開始學習無監督學習之前，讓我們先設定好開發環境並匯入必要的套件。

### 範例 1: 環境設定與套件匯入
```python
# 匯入必要的套件
import sys
import numpy as np
import matplotlib.pyplot as plt
from packaging import version
import sklearn

# 設定圖表字體和樣式
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# 建立圖片儲存目錄
from pathlib import Path
IMAGES_PATH = Path() / "images" / "unsupervised_learning"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `import sys, numpy as np, matplotlib.pyplot as plt`: 匯入系統、數值計算和繪圖套件
2. `from packaging import version; import sklearn`: 匯入版本檢查和機器學習套件
3. `plt.rc('font', size=14)`: 設定圖表字體大小為14
4. `IMAGES_PATH.mkdir(parents=True, exist_ok=True)`: 建立圖片儲存目錄，如果不存在則建立
5. `def save_fig(...)`: 定義儲存圖表的函數，支援不同的檔案格式和解析度

**🎯 重點摘要:**

- **核心功能**: 設定開發環境和圖表樣式，提供圖片儲存功能
- **潛在問題**: 需要確保目錄寫入權限和足夠的磁碟空間
- **最佳使用情境**: 在開始任何機器學習專案時進行環境設定

<a id="聚類分析"></a>
## 📊 聚類分析 (Clustering)

聚類分析是將相似的資料點分組在一起的過程。與監督學習不同，聚類不需要預先知道正確的分組標籤。

### 範例 2: 分類 vs 聚類的視覺化比較
```python
# 載入鳶尾花資料集
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
data.target_names

# 繪製分類 vs 聚類比較圖
plt.figure(figsize=(9, 3.5))
plt.subplot(121)
plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris setosa")
plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris versicolor")
plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris virginica")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.grid()
plt.legend()

plt.subplot(122)
plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
plt.xlabel("Petal length")
plt.tick_params(labelleft=False)
plt.gca().set_axisbelow(True)
plt.grid()
save_fig("classification_vs_clustering_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import load_iris`: 匯入鳶尾花資料集
2. `data = load_iris()`: 載入資料集
3. `X = data.data; y = data.target`: 分離特徵和標籤
4. `plt.subplot(121)`: 建立第一個子圖，用於顯示分類結果
5. `plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris setosa")`: 繪製setosa品種的資料點
6. `plt.subplot(122)`: 建立第二個子圖，用於顯示聚類前的資料分佈
7. `plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")`: 以黑色點顯示所有資料點

**🎯 重點摘要:**

- **核心功能**: 視覺化展示監督學習(分類)和無監督學習(聚類)的差異
- **潛在問題**: 圖表可能會因為資料分佈而難以區分聚類
- **最佳使用情境**: 教學場合，用於解釋聚類和分類的根本差異

<a id="k-means演算法"></a>
## 🎯 K-Means演算法

K-Means是最流行且最簡單的聚類演算法之一。它將資料點分成k個聚類，使得每個點到其所屬聚類中心的距離平方和最小化。

### 範例 3: K-Means基本使用
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 產生測試資料
blob_centers = np.array([[ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
                         [-2.8,  2.8], [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std,
                  random_state=7)

# 訓練K-Means模型
k = 5
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.cluster import KMeans`: 匯入K-Means演算法
2. `blob_centers = np.array([...])`: 定義聚類中心位置
3. `blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])`: 定義每個聚類的標準差
4. `X, y = make_blobs(...)`: 產生合成資料，包含2000個樣本
5. `kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)`: 建立K-Means模型，設定5個聚類
6. `y_pred = kmeans.fit_predict(X)`: 訓練模型並預測聚類標籤

**🎯 重點摘要:**

- **核心功能**: 實現基本的K-Means聚類演算法
- **潛在問題**: 需要預先指定聚類數量，可能導致次優解
- **最佳使用情境**: 當資料大致呈球形分佈且聚類數量已知時

### 範例 4: 視覺化聚類結果和決策邊界
```python
# 定義繪圖函數
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title(f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}")
    plt.grid()
    plt.gca().set_axisbelow(True)

# 繪製聚類結果
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
save_fig("voronoi_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `def plot_clusters(X, y=None)`: 定義繪製聚類結果的函數
2. `def plot_centroids(...)`: 定義繪製聚類中心的函數
3. `def plot_decision_boundaries(...)`: 定義繪製決策邊界的函數
4. `mins = X.min(axis=0) - 0.1`: 計算繪圖範圍的最小值
5. `xx, yy = np.meshgrid(...)`: 建立網格點座標
6. `Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])`: 預測網格點的聚類標籤
7. `plt.contourf(Z, extent=..., cmap="Pastel2")`: 繪製填充的等高線圖
8. `plt.contour(Z, extent=..., linewidths=1, colors='k')`: 繪製等高線

**🎯 重點摘要:**

- **核心功能**: 視覺化K-Means的決策邊界，形成Voronoi圖
- **潛在問題**: 高解析度繪圖可能消耗大量計算資源
- **最佳使用情境**: 分析2D資料的聚類結果和邊界

### 範例 5: K-Means演算法的迭代過程
```python
# 建立不同迭代次數的K-Means模型
kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=1,
                      random_state=5)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=2,
                      random_state=5)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=3,
                      random_state=5)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)

# 繪製迭代過程
plt.figure(figsize=(10, 8))

plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", rotation=0)
plt.tick_params(labelbottom=False)
plt.title("Update the centroids (initially randomly)")

plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False,
                         show_ylabels=False)
plt.title("Label the instances")

plt.subplot(323)
plot_decision_boundaries(kmeans_iter1, X, show_centroids=False,
                         show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)

plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False,
                         show_ylabels=False)

plt.subplot(325)
plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
plot_centroids(kmeans_iter3.cluster_centers_)

plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)

save_fig("kmeans_algorithm_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `kmeans_iter1 = KMeans(..., max_iter=1, ...)`: 建立只迭代1次的K-Means模型
2. `kmeans_iter1.fit(X)`: 訓練模型
3. `plt.subplot(321)`: 建立3x2網格的第一個子圖
4. `plot_data(X)`: 繪製原始資料點
5. `plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')`: 以紅色圓圈繪製初始中心
6. 後續子圖展示迭代過程中中心和邊界的變化

**🎯 重點摘要:**

- **核心功能**: 動態展示K-Means演算法的收斂過程
- **潛在問題**: 隨機初始化可能導致不同的最終結果
- **最佳使用情境**: 教學場合，用於解釋K-Means的迭代優化過程

<a id="k-means變體"></a>
## 🔄 K-Means變體

### 範例 6: Mini-Batch K-Means
```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, n_init=3, random_state=42)
minibatch_kmeans.fit(X)
print("Mini-batch K-Means慣性:", minibatch_kmeans.inertia_)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.cluster import MiniBatchKMeans`: 匯入Mini-Batch K-Means
2. `minibatch_kmeans = MiniBatchKMeans(n_clusters=5, n_init=3, random_state=42)`: 建立Mini-Batch模型
3. `minibatch_kmeans.fit(X)`: 使用小批次訓練模型
4. `print("Mini-batch K-Means慣性:", minibatch_kmeans.inertia_)`: 輸出模型的慣性值

**🎯 重點摘要:**

- **核心功能**: 提供更快的K-Means變體，適合大規模資料
- **潛在問題**: 可能產生稍微不準確的結果
- **最佳使用情境**: 處理大型資料集時的聚類分析

<a id="使用聚類進行圖像分割"></a>
## 🖼️ 使用聚類進行圖像分割

### 範例 7: 圖像分割實作
```python
# 下載瓢蟲圖片
import urllib.request
homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
filename = "ladybug.png"
filepath = IMAGES_PATH / filename
if not filepath.is_file():
    print("Downloading", filename)
    url = f"{homl3_root}/images/unsupervised_learning/{filename}"
    urllib.request.urlretrieve(url, filepath)

# 載入和處理圖片
import PIL
image = np.asarray(PIL.Image.open(filepath))
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, n_init=10, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

# 顯示不同聚類數量的分割結果
segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

plt.figure(figsize=(10, 5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(2, 3, 2 + idx)
    plt.imshow(segmented_imgs[idx] / 255)
    plt.title(f"{n_clusters} colors")
    plt.axis('off')

save_fig('image_segmentation_plot', tight_layout=False)
plt.show()
```

**✅ 程式碼逐行解析：**

1. `urllib.request.urlretrieve(url, filepath)`: 下載瓢蟲圖片
2. `image = np.asarray(PIL.Image.open(filepath))`: 將圖片轉換為numpy陣列
3. `X = image.reshape(-1, 3)`: 將圖片重新塑形為像素點集合
4. `kmeans = KMeans(n_clusters=8, n_init=10, random_state=42).fit(X)`: 使用8個顏色聚類
5. `segmented_img = kmeans.cluster_centers_[kmeans.labels_]`: 將每個像素替換為其聚類中心顏色
6. `segmented_img = segmented_img.reshape(image.shape)`: 重新塑形為原始圖片尺寸

**🎯 重點摘要:**

- **核心功能**: 使用K-Means進行圖像顏色量化，實現圖像分割
- **潛在問題**: 聚類數量會影響分割品質和計算時間
- **最佳使用情境**: 圖像壓縮、藝術風格轉換和物體分割

<a id="使用聚類進行半監督學習"></a>
## 📈 使用聚類進行半監督學習

### 範例 8: 聚類輔助的半監督學習
```python
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

# 載入數字資料集
X_digits, y_digits = load_digits(return_X_y=True)
X_train, y_train = X_digits[:1400], y_digits[:1400]
X_test, y_test = X_digits[1400:], y_digits[1400:]

# 使用少量標籤資料訓練
n_labeled = 50
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print("使用50個標籤樣本的準確率:", log_reg.score(X_test, y_test))

# 使用聚類進行標籤傳播
k = 50
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = X_digits_dist.argmin(axis=0)
X_representative_digits = X_train[representative_digit_idx]

# 手動標記代表性數字（在實際應用中這是手動完成的）
y_representative_digits = np.array([
    1, 3, 6, 0, 7, 9, 2, 4, 8, 9,
    5, 4, 7, 1, 2, 6, 1, 2, 5, 1,
    4, 1, 3, 3, 8, 8, 2, 5, 6, 9,
    1, 4, 0, 6, 8, 3, 4, 6, 7, 2,
    4, 1, 0, 7, 5, 1, 9, 9, 3, 7
])

# 使用代表性樣本訓練
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_representative_digits, y_representative_digits)
print("使用聚類代表樣本的準確率:", log_reg.score(X_test, y_test))
```

**✅ 程式碼逐行解析：**

1. `X_digits, y_digits = load_digits(return_X_y=True)`: 載入手寫數字資料集
2. `log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])`: 使用少量標籤資料訓練
3. `X_digits_dist = kmeans.fit_transform(X_train)`: 計算每個樣本到聚類中心的距離
4. `representative_digit_idx = X_digits_dist.argmin(axis=0)`: 找到每個聚類的代表樣本
5. `y_representative_digits = np.array([...])`: 手動標記代表樣本的標籤

**🎯 重點摘要:**

- **核心功能**: 使用聚類技術擴展有限的標籤資料
- **潛在問題**: 需要手動標記代表樣本，可能耗時
- **最佳使用情境**: 標籤資料稀缺但非標籤資料豐富的情況

<a id="dbscan"></a>
## 🔍 DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 是一種基於密度的聚類演算法，能夠發現任意形狀的聚類。

### 範例 9: DBSCAN基本使用
```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)

print("聚類標籤:", dbscan.labels_[:10])
print("核心樣本索引:", dbscan.core_sample_indices_[:10])
print("核心樣本:", dbscan.components_)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.cluster import DBSCAN`: 匯入DBSCAN演算法
2. `X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)`: 產生月亮形狀的資料
3. `dbscan = DBSCAN(eps=0.05, min_samples=5)`: 建立DBSCAN模型
4. `dbscan.fit(X)`: 訓練模型
5. `print("聚類標籤:", dbscan.labels_[:10])`: 顯示前10個樣本的聚類標籤
6. `print("核心樣本索引:", dbscan.core_sample_indices_[:10])`: 顯示核心樣本的索引
7. `print("核心樣本:", dbscan.components_)`: 顯示核心樣本的座標

**🎯 重點摘要:**

- **核心功能**: 基於密度的聚類，能夠處理任意形狀的聚類
- **潛在問題**: 參數eps和min_samples的選擇很重要
- **最佳使用情境**: 聚類形狀不規則或包含雜訊的資料

<a id="其他聚類演算法"></a>
## 📏 其他聚類演算法

### 範例 10: 階層式聚類
```python
from sklearn.cluster import AgglomerativeClustering

X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)
agg = AgglomerativeClustering(linkage="complete").fit(X)

print("子節點:", agg.children_)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.cluster import AgglomerativeClustering`: 匯入階層式聚類
2. `X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)`: 建立一維資料
3. `agg = AgglomerativeClustering(linkage="complete").fit(X)`: 使用完全連結法訓練
4. `print("子節點:", agg.children_)`: 顯示階層樹的結構

**🎯 重點摘要:**

- **核心功能**: 建立資料的階層式分組結構
- **潛在問題**: 計算複雜度較高，不適合大規模資料
- **最佳使用情境**: 需要了解聚類階層結構的小型資料集

<a id="高斯混合模型"></a>
## 🌟 高斯混合模型 (Gaussian Mixtures)

高斯混合模型 (GMM) 假設資料是由多個高斯分佈混合而成的，能夠進行更靈活的聚類和密度估計。

### 範例 11: 高斯混合模型基本使用
```python
from sklearn.mixture import GaussianMixture

# 訓練GMM模型
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

print("權重:", gm.weights_)
print("均值:", gm.means_)
print("協方差矩陣:", gm.covariances_)
print("收斂:", gm.converged_)
print("迭代次數:", gm.n_iter_)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.mixture import GaussianMixture`: 匯入高斯混合模型
2. `gm = GaussianMixture(n_components=3, n_init=10, random_state=42)`: 建立3組分GMM
3. `gm.fit(X)`: 訓練模型
4. `print("權重:", gm.weights_)`: 顯示每個組分的權重
5. `print("均值:", gm.means_)`: 顯示每個組分的均值
6. `print("協方差矩陣:", gm.covariances_)`: 顯示協方差矩陣
7. `print("收斂:", gm.converged_)`: 檢查是否收斂
8. `print("迭代次數:", gm.n_iter_)`: 顯示迭代次數

**🎯 重點摘要:**

- **核心功能**: 提供概率性的聚類和密度估計
- **潛在問題**: 可能陷入局部最優，需要多次初始化
- **最佳使用情境**: 需要軟聚類或密度估計的應用

### 範例 12: 生成新樣本
```python
X_new, y_new = gm.sample(6)
print("新樣本:", X_new)
print("對應組分:", y_new)
```

**✅ 程式碼逐行解析：**

1. `X_new, y_new = gm.sample(6)`: 從學習到的分佈中生成6個新樣本
2. `print("新樣本:", X_new)`: 顯示生成樣本的座標
3. `print("對應組分:", y_new)`: 顯示每個樣本屬於哪個組分

**🎯 重點摘要:**

- **核心功能**: 從學習到的分佈生成新資料
- **潛在問題**: 生成的樣本品質取決於模型擬合程度
- **最佳使用情境**: 資料增強或生成合成資料

<a id="使用高斯混合模型進行異常檢測"></a>
## 🚨 使用高斯混合模型進行異常檢測

### 範例 13: 異常檢測實作
```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 2)
anomalies = X[densities < density_threshold]

print("異常樣本數量:", len(anomalies))
```

**✅ 程式碼逐行解析：**

1. `densities = gm.score_samples(X)`: 計算所有樣本的對數密度分數
2. `density_threshold = np.percentile(densities, 2)`: 設定密度閾值（最低2%）
3. `anomalies = X[densities < density_threshold]`: 識別異常樣本
4. `print("異常樣本數量:", len(anomalies))`: 輸出異常樣本數量

**🎯 重點摘要:**

- **核心功能**: 基於密度的異常檢測
- **潛在問題**: 閾值選擇會影響檢測結果
- **最佳使用情境**: 識別低密度區域的異常點

<a id="選擇聚類數量"></a>
## 📊 選擇聚類數量

### 範例 14: 使用BIC和AIC選擇聚類數量
```python
gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X)
             for k in range(1, 11)]
bics = [model.bic(X) for model in gms_per_k]
aics = [model.aic(X) for model in gms_per_k]

plt.figure(figsize=(8, 3))
plt.plot(range(1, 11), bics, "bo-", label="BIC")
plt.plot(range(1, 11), aics, "go--", label="AIC")
plt.xlabel("$k$")
plt.ylabel("Information Criterion")
plt.axis([1, 9.5, min(aics) - 50, max(aics) + 50])
plt.legend()
plt.grid()
save_fig("aic_bic_vs_k_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `gms_per_k = [...]`: 訓練不同組分數量的高斯混合模型
2. `bics = [model.bic(X) for model in gms_per_k]`: 計算每個模型的BIC分數
3. `aics = [model.aic(X) for model in gms_per_k]`: 計算每個模型的AIC分數
4. `plt.plot(range(1, 11), bics, "bo-", label="BIC")`: 繪製BIC曲線
5. `plt.plot(range(1, 11), aics, "go--", label="AIC")`: 繪製AIC曲線

**🎯 重點摘要:**

- **核心功能**: 使用資訊準則自動選擇最佳聚類數量
- **潛在問題**: BIC和AIC可能選擇不同的k值
- **最佳使用情境**: 當聚類數量未知時的模型選擇

<a id="貝葉斯高斯混合模型"></a>
## 🎲 貝葉斯高斯混合模型

### 範例 15: 貝葉斯高斯混合模型
```python
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
print("權重:", bgm.weights_.round(2))
```

**✅ 程式碼逐行解析：**

1. `from sklearn.mixture import BayesianGaussianMixture`: 匯入貝葉斯高斯混合模型
2. `bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)`: 建立模型
3. `bgm.fit(X)`: 訓練模型
4. `print("權重:", bgm.weights_.round(2))`: 顯示學習到的權重

**🎯 重點摘要:**

- **核心功能**: 自動確定聚類數量，過多組分會有接近零的權重
- **潛在問題**: 計算更複雜，可能需要更多時間
- **最佳使用情境**: 聚類數量未知且希望自動確定的情況

<a id="習題解答"></a>
## 📚 習題解答

### 習題 10: 聚類 Olivetti 人臉資料集

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# 載入資料集
olivetti = fetch_olivetti_faces()
X_train, y_train = olivetti.data[:300], olivetti.target[:300]
X_test, y_test = olivetti.data[300:400], olivetti.target[300:400]

# PCA降維
pca = PCA(0.99)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# K-Means聚類
kmeans = KMeans(n_clusters=40, n_init=10, random_state=42)
X_train_reduced = kmeans.fit_transform(X_train_pca)

# 分類器訓練
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_reduced, y_train)
accuracy = clf.score(kmeans.transform(X_test_pca), y_test)
print(f"聚類後分類準確率: {accuracy:.3f}")
```

**✅ 程式碼逐行解析：**

1. `olivetti = fetch_olivetti_faces()`: 載入Olivetti人臉資料集
2. `pca = PCA(0.99)`: 建立保留99%變異的PCA
3. `X_train_pca = pca.fit_transform(X_train)`: 對訓練資料進行降維
4. `kmeans = KMeans(n_clusters=40, n_init=10, random_state=42)`: 建立40聚類的K-Means
5. `X_train_reduced = kmeans.fit_transform(X_train_pca)`: 計算到聚類中心的距離作為新特徵
6. `clf.fit(X_train_reduced, y_train)`: 訓練隨機森林分類器

**🎯 重點摘要:**

- **核心功能**: 使用聚類作為分類的預處理步驟
- **潛在問題**: 聚類品質會影響最終分類表現
- **最佳使用情境**: 特徵工程和降維結合的分類任務

<a id="總結與最佳實踐"></a>
## 💡 總結與最佳實踐

無監督學習為我們提供了強大的工具來發現資料中的隱藏結構：

1. **選擇合適的演算法**：
   - K-Means：適合球形聚類，快速且可擴展
   - DBSCAN：適合任意形狀聚類，對雜訊魯棒
   - 高斯混合模型：提供概率解釋，適合密度估計

2. **參數調優**：
   - 使用輪廓係數或資訊準則選擇聚類數量
   - 調整DBSCAN的eps和min_samples參數
   - 嘗試不同的協方差類型

3. **應用場景**：
   - 客戶分群和市場細分
   - 圖像和語音處理
   - 異常檢測和欺詐識別
   - 降維和特徵提取

4. **常見陷阱**：
   - 忽略資料預處理的重要性
   - 選擇不適當的距離度量
   - 過度解釋聚類結果

<a id="常見問答"></a>
## ❓ 常見問答 (FAQ)

**Q: K-Means和DBSCAN哪個更好？**
A: 取決於資料特性。K-Means適合球形聚類且速度快；DBSCAN適合任意形狀且能處理雜訊。

**Q: 如何選擇聚類數量？**
A: 使用輪廓分析、肘部法則，或資訊準則(BIC/AIC)來評估不同k值的表現。

**Q: 高斯混合模型和K-Means的差異？**
A: GMM提供軟聚類(概率)而非硬聚類，且能建模橢圓形聚類。

**Q: 聚類可以用於分類嗎？**
A: 可以作為半監督學習的一部分，用聚類結果擴展有限的標籤資料。

<a id="推薦標籤"></a>
## 🏷️ 推薦標籤 (Suggested Hashtags)

#機器學習 #無監督學習 #聚類分析 #KMeans #DBSCAN #高斯混合模型 #PCA #降維 #異常檢測 #Python教學 #程式設計 #編程 #開發 #技術分享 #學習筆記 #程式開發者 #軟體工程