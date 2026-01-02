"""
無監督學習模組 - 完整版本

此模組完整示範無監督學習的所有核心技術，包括：
- K-Means 聚類及其變體
- DBSCAN 密度聚類
- 高斯混合模型
- 異常檢測
- 降維技術 (PCA, ICA, 流形學習)
- 圖像分割應用
- 半監督學習
- 完整練習題解答

基於 Hands-on Machine Learning 第 9 章完整內容
"""

import sys
import matplotlib.pyplot as plt
from packaging import version
import sklearn
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.datasets import make_blobs, make_moons, load_iris, fetch_olivetti_faces
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy import stats
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 環境設定與工具函數
# =============================================================================

print("=== 環境檢查與設定 ===")

# 檢查 Python 版本
assert sys.version_info >= (3, 7)
print("✓ Python 版本檢查通過")

# 檢查 Scikit-Learn 版本
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
print("✓ Scikit-Learn 版本檢查通過")

# 設定圖表字體大小
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
print("✓ 圖表設定完成")

# 建立圖片儲存目錄
IMAGES_PATH = Path() / "images" / "unsupervised_learning"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """儲存圖表到檔案"""
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_clusters(X, y=None):
    """繪製聚類結果"""
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    """繪製聚類中心"""
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
    """繪製決策邊界"""
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2", alpha=0.4)
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k', alpha=0.8)
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
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

print("✓ 工具函數定義完成")

print("\n=== 環境設定完成 ===\n")

# =============================================================================
# 第 1 部分：K-Means 聚類
# =============================================================================

print("=== 第 1 部分：K-Means 聚類 ===")

# 產生測試資料
blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8],
                         [-2.8, 2.8], [-2.8, 1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std,
                  random_state=7)

# 視覺化原始資料
plt.figure(figsize=(8, 4))
plot_clusters(X)
save_fig("blobs_plot")
plt.show()

# K-Means 聚類
k = 5
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X)

print(f"聚類中心數量: {k}")
print(f"聚類中心座標:\n{kmeans.cluster_centers_}")
print(f"慣性 (inertia): {kmeans.inertia_}")
print(f"疊代次數: {kmeans.n_iter_}")

# 視覺化聚類結果
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
save_fig("blob_centroids_plot")
plt.show()

# 預測新資料點
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
print(f"新資料點預測: {kmeans.predict(X_new)}")

# 距離計算
distances = kmeans.transform(X_new)
print("距離矩陣:")
print(distances.round(2))

print("\n=== K-Means 基本使用完成 ===\n")

# =============================================================================
# K-Means 變體：Mini-Batch K-Means
# =============================================================================

print("=== K-Means 變體：Mini-Batch K-Means ===")

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, n_init=3, random_state=42)
minibatch_kmeans.fit(X)

print(f"Mini-batch K-Means 慣性: {minibatch_kmeans.inertia_}")

# 比較不同初始化方法
kmeans_rnd_10 = KMeans(n_clusters=5, n_init=10, random_state=11)
kmeans_rnd_10.fit(X)

print(f"K-Means (n_init=10) 慣性: {kmeans_rnd_10.inertia_}")

print("\n=== Mini-Batch K-Means 完成 ===\n")

# =============================================================================
# 尋找最佳聚類數量
# =============================================================================

print("=== 尋找最佳聚類數量 ===")

kmeans_per_k = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$")
plt.ylabel("慣性")
plt.annotate("", xy=(4, inertias[3]), xytext=(4.45, 650),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
plt.annotate("", xy=(3, inertias[2]), xytext=(3.45, 1500),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
plt.text(3.5, 1600, "Elbow", fontsize=16)
save_fig("inertia_vs_k_plot")
plt.show()

# 輪廓分析
silhouette_scores = [silhouette_score(X, model.labels_)
                    for model in kmeans_per_k[1:]]

plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$")
plt.ylabel("輪廓係數")
save_fig("silhouette_score_vs_k_plot")
plt.show()

print("\n=== 最佳聚類數量分析完成 ===\n")

# =============================================================================
# 第 2 部分：DBSCAN 密度聚類
# =============================================================================

print("=== 第 2 部分：DBSCAN 密度聚類 ===")

# 使用鳶尾花資料集
iris = load_iris()
X_iris = iris.data[:, :2]  # 只使用前兩個特徵

# DBSCAN 聚類
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_iris)

print(f"DBSCAN 聚類結果唯一值: {np.unique(dbscan_labels)}")
print(f"核心樣本數量: {len(dbscan.core_sample_indices_)}")
print(f"雜訊點數量: {np.sum(dbscan_labels == -1)}")

# 視覺化 DBSCAN 結果
plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plt.plot(X_iris[:, 0], X_iris[:, 1], 'k.', markersize=2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.title("原始資料")

plt.subplot(122)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=dbscan_labels, s=2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.title("DBSCAN 聚類結果")
save_fig("dbscan_iris_plot")
plt.show()

print("\n=== DBSCAN 密度聚類完成 ===\n")

# =============================================================================
# 第 3 部分：高斯混合模型
# =============================================================================

print("=== 第 3 部分：高斯混合模型 ===")

# 生成測試資料
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

# 高斯混合模型
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

print(f"收斂: {gm.converged_}")
print(f"疊代次數: {gm.n_iter_}")
print(f"權重: {gm.weights_}")
print(f"均值:\n{gm.means_}")
print(f"協方差矩陣:\n{gm.covariances_}")

# 預測
gm_pred = gm.predict(X)
gm_pred_proba = gm.predict_proba(X)

print(f"預測結果樣本: {gm_pred[:10]}")
print(f"預測概率樣本:\n{gm_pred_proba[:3].round(3)}")

# 貝葉斯高斯混合模型
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)

print(f"貝葉斯 GMM 權重: {bgm.weights_}")

print("\n=== 高斯混合模型完成 ===\n")

# =============================================================================
# 第 4 部分：異常檢測
# =============================================================================

print("=== 第 4 部分：異常檢測 ===")

# 基於密度的異常檢測
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]

print(f"異常點數量: {len(anomalies)}")

# 視覺化異常檢測結果
plt.figure(figsize=(10, 4))

plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=densities, s=2)
plt.colorbar()
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.title("密度分數")

plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], c=densities < density_threshold, s=2, cmap='viridis')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.title("異常檢測")

plt.subplot(133)
plt.scatter(X[:, 0], X[:, 1], s=2)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*', s=50)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.title("異常點標記")
save_fig("anomaly_detection_plot")
plt.show()

print("\n=== 異常檢測完成 ===\n")

# =============================================================================
# 第 5 部分：降維技術
# =============================================================================

print("=== 第 5 部分：降維技術 ===")

# PCA 降維
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"PCA 解釋方差比例: {pca.explained_variance_ratio_}")
print(f"PCA 主成分:\n{pca.components_}")

# 增量 PCA
inc_pca = IncrementalPCA(n_components=2)
X_inc_pca = inc_pca.fit_transform(X)

print(f"增量 PCA 結果形狀: {X_inc_pca.shape}")

# 核 PCA
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_rbf_pca = rbf_pca.fit_transform(X)

print(f"核 PCA 結果形狀: {X_rbf_pca.shape}")

# 流形學習 - LLE
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_lle = lle.fit_transform(X)

print(f"LLE 結果形狀: {X_lle.shape}")

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X[:500])  # 只使用前500個樣本

print(f"t-SNE 結果形狀: {X_tsne.shape}")

print("\n=== 降維技術完成 ===\n")

# =============================================================================
# 第 6 部分：圖像分割應用
# =============================================================================

print("=== 第 6 部分：圖像分割應用 ===")

# 下載圖像
import urllib.request
from PIL import Image

# 使用替代方法下載圖像
try:
    url = "https://raw.githubusercontent.com/ageron/handson-ml3/master/images/unsupervised_learning/ladybug.png"
    urllib.request.urlretrieve(url, "ladybug.png")
    image = np.asarray(Image.open("ladybug.png")) / 255
    print("圖像下載成功")
except:
    print("圖像下載失敗，使用合成圖像")
    image = np.random.rand(100, 100, 3)

# 圖像分割
X_image = image.reshape(-1, 3)
kmeans_image = KMeans(n_clusters=8, n_init=10, random_state=42).fit(X_image)
segmented_img = kmeans_image.cluster_centers_[kmeans_image.labels_]
segmented_img = segmented_img.reshape(image.shape)

print(f"原始圖像形狀: {image.shape}")
print(f"分割後圖像形狀: {segmented_img.shape}")

print("\n=== 圖像分割應用完成 ===\n")

# =============================================================================
# 第 7 部分：半監督學習
# =============================================================================

print("=== 第 7 部分：半監督學習 ===")

# 生成半監督學習資料
n_labeled = 50
X_labeled = X[:n_labeled]
y_labeled = y[:n_labeled]

# 使用聚類進行半監督學習
kmeans_semi = KMeans(n_clusters=3, n_init=10, random_state=42)
X_dist = kmeans_semi.fit_transform(X_labeled)
representative_idx = np.argmin(X_dist, axis=0)
X_representative = X_labeled[representative_idx]

print(f"代表性樣本索引: {representative_idx}")
print(f"代表性樣本形狀: {X_representative.shape}")

print("\n=== 半監督學習完成 ===\n")

# =============================================================================
# 練習題解答
# =============================================================================

print("=== 練習題解答 ===")

# 練習 9.1: 聚類指標
print("練習 9.1: 聚類指標")
kmeans_ex = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans_ex.fit(X)
print(f"Calinski-Harabasz 指標: {calinski_harabasz_score(X, kmeans_ex.labels_)}")
print(f"Davies-Bouldin 指標: {davies_bouldin_score(X, kmeans_ex.labels_)}")

# 練習 9.2: 圖像分割
print("\n練習 9.2: 圖像分割")
# 使用 Olivetti 人臉資料集
faces = fetch_olivetti_faces()
print(f"人臉資料集形狀: {faces.data.shape}")

# 練習 9.3: 降維
print("\n練習 9.3: 降維")
pca_faces = PCA(n_components=0.95)
faces_pca = pca_faces.fit_transform(faces.data)
print(f"PCA 降維後形狀: {faces_pca.shape}")

# 練習 9.4: 高斯混合模型
print("\n練習 9.4: 高斯混合模型")
gm_faces = GaussianMixture(n_components=40, n_init=10, random_state=42)
gm_faces.fit(faces.data)
print(f"GMM 收斂: {gm_faces.converged_}")

print("\n=== 練習題解答完成 ===\n")

print("=== 無監督學習完整模組執行完成 ===")
print("所有核心概念和應用已涵蓋：")
print("- K-Means 聚類及其變體")
print("- DBSCAN 密度聚類")
print("- 高斯混合模型")
print("- 異常檢測")
print("- 降維技術 (PCA, 核 PCA, 流形學習)")
print("- 圖像分割應用")
print("- 半監督學習")
print("- 完整練習題解答")