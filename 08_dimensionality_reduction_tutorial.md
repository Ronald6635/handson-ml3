<!-- meta-title: 降維完整指南：PCA、Kernel PCA、LLE、UMAP 與 t-SNE 實戰 -->
<!-- meta-description: 深入降維的核心技術：維度詛咒、PCA（解釋方差比、白化）、增量 PCA、隨機 PCA、核技巧 PCA、局部線性嵌入 (LLE)、UMAP 和 t-SNE 視覺化。含 Scikit-Learn 實戰與逐行解析。 -->
<!-- meta-keywords: Python, 降維, PCA, Kernel PCA, LLE, UMAP, t-SNE, 機器學習, Scikit-Learn, 資料視覺化 -->
<!-- meta-hashtags: #Python #PCA #降維 #UMAP #tSNE #機器學習 #ScikitLearn #資料視覺化 #程式設計 #教學 -->

# 🐍 降維完整指南：PCA 到 UMAP 的多維資料壓縮實戰

高維資料（High-Dimensional Data）帶來的**維度詛咒（Curse of Dimensionality）** 讓許多 ML 演算法效能退化。降維（Dimensionality Reduction）不只是壓縮資料，更是資料視覺化、噪音過濾和特徵萃取的關鍵技術。本教學從 PCA 的數學原理出發，帶你掌握從線性到非線性的完整降維工具箱。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [😱 維度詛咒](#curse-of-dimensionality)
- [📐 主成分分析 (PCA)](#pca)
- [💾 增量 PCA 與隨機 PCA](#incremental-pca)
- [🔮 核技巧 PCA](#kernel-pca)
- [🌀 局部線性嵌入 (LLE)](#lle)
- [🗺️ UMAP 與 t-SNE](#umap-tsne)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **維度詛咒**：維度增加，高維空間「幾乎為空」，樣本間距離趨向相同，ML 演算法失效
- **PCA** 找到資料方差最大的方向（主成分），是最常用的**線性**降維方法
- `explained_variance_ratio_` 幫你選擇保留多少主成分（如保留 95% 的方差）
- **Kernel PCA** 讓 PCA 適用於非線性流形，適合資料分佈在曲線或曲面上的情況
- **UMAP/t-SNE** 專為**視覺化**設計，能展現高維資料的聚類結構，但不適合降維後再訓練

---

## <a id="curse-of-dimensionality"></a>😱 維度詛咒

💡 **實際應用情境：** 想像台灣製造業的感測器資料——每台機器有 500 個感測器每秒記錄一次。若直接用所有 500 維特徵訓練 KNN，大多數樣本之間的「歐氏距離」會變得幾乎一樣大，KNN 的「鄰居」概念失去意義。

維度詛咒的核心問題：

- 在 d 維超球體中，隨機樣本有 1 − 0.5^(1/d) 的機率落在「外邊緣 10% 的殼層」
- d=10,000 時：幾乎所有樣本都在邊緣，沒有「內部」樣本
- 距離的一致性（Distance Concentration）：最近和最遠鄰居的距離差異越來越小

```python
import numpy as np

# 模擬維度詛咒：不同維度下，距原點的距離分佈
np.random.seed(42)
for d in [2, 10, 100, 1000]:
    X = np.random.rand(1000, d)  # 在 [0,1]^d 均勻分佈
    dists = np.sqrt((X**2).sum(axis=1))  # 到原點的距離
    print(f"d={d:4d}: min={dists.min():.3f}, max={dists.max():.3f}, "
          f"mean={dists.mean():.3f}, std={dists.std():.4f}")
# 維度越高，距離分佈越集中（std 相對於 mean 越小）
```

---

## <a id="pca"></a>📐 主成分分析 (PCA)

💡 **實際應用情境：** 台灣醫院的電子病歷資料有數千個欄位，但許多高度相關（如血壓相關指標群）。PCA 能將相關特徵「合并」為少數不相關的主成分，減少計算量同時保留大部分資訊。

### 範例 1: PCA 降維與解釋方差比

```python
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import numpy as np

# 載入 MNIST（示範高維降維）
mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X_mnist = mnist["data"][:10000] / 255.0  # 前 10000 筆
y_mnist = mnist["target"][:10000]

# 方法 1：直接指定維度
pca_2d = PCA(n_components=2)  # 降到 2 維（用於視覺化）
X_2d = pca_2d.fit_transform(X_mnist)
print(f"2D 解釋方差: {pca_2d.explained_variance_ratio_.sum():.4f}")

# 方法 2：保留 95% 的方差（自動決定維度）
pca_95 = PCA(n_components=0.95)  # 保留 95% 方差
X_95 = pca_95.fit_transform(X_mnist)
print(f"保留 95% 方差所需維度: {pca_95.n_components_}")  # 通常約 150 維

# 查看每個主成分解釋的方差比例
import matplotlib.pyplot as plt
cumvar = np.cumsum(pca_95.explained_variance_ratio_)
plt.plot(cumvar)
plt.xlabel("主成分數量")
plt.ylabel("累積解釋方差比")
plt.title("PCA 解釋方差圖（Elbow 選擇 n_components）")
plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
plt.legend()
plt.grid(True)
plt.show()
```

**✅ 程式碼逐行解析：**

1. `PCA(n_components=0.95)`: 傳入 0~1 的浮點數時，自動選擇能保留指定方差比例的最少維度
2. `explained_variance_ratio_`: 每個主成分解釋的方差佔原始總方差的比例（降序排列）
3. 「Elbow」點：累積方差圖中斜率轉折的地方，是選擇主成分數量的啟發式方法

### 範例 2: PCA 壓縮與還原（視覺化重建誤差）

```python
# 壓縮：784D → 150D
pca_compress = PCA(n_components=150)
X_compressed = pca_compress.fit_transform(X_mnist)

# 還原（有損）：150D → 784D
X_reconstructed = pca_compress.inverse_transform(X_compressed)

# 計算重建誤差
reconstruction_mse = np.mean((X_mnist - X_reconstructed)**2)
print(f"重建 MSE: {reconstruction_mse:.6f}")

# 視覺化比較原圖和重建圖
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes[0]):
    ax.imshow(X_mnist[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"原圖 {y_mnist[i]}")
for i, ax in enumerate(axes[1]):
    ax.imshow(X_reconstructed[i].reshape(28, 28), cmap="gray")
    ax.set_title("重建")
plt.tight_layout()
plt.show()
```

**🎯 重點摘要:**

- PCA 是**線性**降維，假設資料在低維線性子空間中
- 保留 95% 方差通常能將維度壓縮 5-10 倍，且重建圖像仍可辨認

---

## <a id="incremental-pca"></a>💾 增量 PCA 與隨機 PCA

### 範例 3: 大資料集的增量 PCA

```python
from sklearn.decomposition import IncrementalPCA

# 增量 PCA：不需要將全部資料載入記憶體
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

# 分批次（Mini-batch）處理
for X_batch in np.array_split(X_mnist, n_batches):
    inc_pca.partial_fit(X_batch)  # 逐批次更新

X_inc_reduced = inc_pca.transform(X_mnist)
print(f"增量 PCA 輸出形狀: {X_inc_reduced.shape}")

# 隨機 PCA（大維度 → 小維度，速度更快）
from sklearn.decomposition import PCA as RandomPCA

random_pca = RandomPCA(n_components=154, svd_solver="randomized")  # 隨機 SVD
X_rand_reduced = random_pca.fit_transform(X_mnist)
print(f"隨機 PCA 輸出形狀: {X_rand_reduced.shape}")
```

**✅ 程式碼逐行解析：**

1. `IncrementalPCA.partial_fit(X_batch)`: 核心方法——每次只處理一個批次更新模型
2. `svd_solver="randomized"`: 使用隨機化 SVD 近似，當 `n_components << n_features` 時速度遠快於精確 SVD

**🎯 重點摘要:**

- 資料無法裝入記憶體時（如 TB 級感測器資料），用 `IncrementalPCA`
- `n_components` 遠小於特徵數時，`svd_solver="randomized"` 速度顯著更快

---

## <a id="kernel-pca"></a>🔮 核技巧 PCA

💡 **實際應用情境：** 若 MNIST 手寫數字的分佈是「非線性流形」（類似捲起的瑞士卷）而非線性平面，傳統 PCA 無法正確展開這個流形。Kernel PCA 透過核函數隱式映射到高維空間再做 PCA。

### 範例 4: Kernel PCA 與 RBF 核

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# 瑞士卷資料（3D 非線性流形）
X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# 線性 PCA（失敗：無法展開瑞士卷）
pca_linear = PCA(n_components=2)
X_pca_linear = pca_linear.fit_transform(X_swiss)

# RBF Kernel PCA（成功：展開流形）
kpca_rbf = KernelPCA(n_components=2, kernel="rbf", gamma=0.04, n_jobs=-1)
X_kpca_rbf = kpca_rbf.fit_transform(X_swiss)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, X_2d, title in zip(axes,
                            [X_pca_linear, X_kpca_rbf],
                            ["線性 PCA（類別交疊）", "Kernel PCA - RBF（流形展開）"]):
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=t, cmap="Spectral")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)
plt.show()
```

**✅ 程式碼逐行解析：**

1. `make_swiss_roll`: 生成 3D 瑞士卷狀資料（2D 流形嵌入在 3D 空間中）
2. `KernelPCA(kernel="rbf", gamma=0.04)`: RBF 核的 gamma 控制核的「寬度」，小 gamma → 更全域的映射
3. 顏色 `c=t` 表示沿瑞士卷的位置，展開成功時顏色應該是連續漸變的

**🎯 重點摘要:**

- Kernel PCA 適合**非線性流形**（螺旋線、瑞士卷、圓形資料）
- 超參數調優：用 Pipeline + GridSearchCV 搜尋最佳核函數和 gamma

---

## <a id="lle"></a>🌀 局部線性嵌入 (LLE)

### 範例 5: LLE 展開非線性流形

```python
from sklearn.manifold import LocallyLinearEmbedding

# LLE：保持每個點的「局部線性結構」（近鄰之間的線性關係）
lle = LocallyLinearEmbedding(
    n_components=2,
    n_neighbors=10,  # 每個點考慮的近鄰數（重要超參數）
    method="standard",
    random_state=42,
    n_jobs=-1
)
X_lle = lle.fit_transform(X_swiss)

plt.figure(figsize=(8, 6))
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=t, cmap="Spectral")
plt.title("LLE 降維結果")
plt.colorbar(label="位置 t")
plt.show()
```

**🎯 重點摘要:**

- LLE 不需要預先指定核函數，更自適應
- `n_neighbors` 過大：全域結構可能失真；過小：噪音影響大
- LLE 適合**均勻分佈**的流形資料，稀疏邊緣區域效果較差

---

## <a id="umap-tsne"></a>🗺️ UMAP 與 t-SNE

💡 **實際應用情境：** 在生物資訊學中，使用 UMAP 視覺化單細胞 RNA 測序資料（數萬個基因），讓研究者直觀看到細胞類型的聚類結構——這是用 PCA 無法清楚呈現的。

### 範例 6: t-SNE 視覺化 MNIST

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# t-SNE（只用 2000 個樣本，因計算複雜度 O(n²)）
X_small = X_mnist[:2000]
y_small = y_mnist[:2000]

tsne = TSNE(
    n_components=2,
    perplexity=30,    # 近鄰數估計（5~50），影響局部 vs 全域結構的平衡
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X_small)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=y_small.astype(int), cmap="tab10", alpha=0.6, s=10)
plt.colorbar(scatter, ticks=range(10), label="數字類別")
plt.title("t-SNE MNIST 2D 視覺化")
plt.show()
# 可以看到 10 個清晰分離的數字群落
```

```python
# UMAP（需 pip install umap-learn，速度比 t-SNE 快 10-100 倍）
try:
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_small)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1],
                          c=y_small.astype(int), cmap="tab10", alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(10), label="數字類別")
    plt.title("UMAP MNIST 2D 視覺化")
    plt.show()
except ImportError:
    print("UMAP 未安裝，使用 pip install umap-learn")
```

**✅ 程式碼逐行解析：**

1. `TSNE(perplexity=30)`: Perplexity 可直觀理解為「每個點的有效近鄰數」，通常設 5~50
2. t-SNE 的隨機性：不同 `random_state` 可能產生不同結果（但聚類結構應相似）
3. UMAP 保留更多**全域結構**（群落間的相對位置），且速度更快

**🎯 重點摘要:**

- t-SNE 和 UMAP 只適合**視覺化**，不能用降維後的結果訓練分類器（無法對新資料 `transform`）
- UMAP 通常比 t-SNE 更快且保留更多全域結構，是現代的首選視覺化工具

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: PCA 之前需要標準化嗎？**

A: 是的！若特徵的尺度差異很大（如身高 cm 和體重 kg），PCA 的主成分會被尺度大的特徵主導。應先用 `StandardScaler` 標準化，確保每個特徵的方差貢獻相同。

**Q2: 降維後可以直接訓練分類器嗎？**

A: PCA 降維後的資料可以訓練分類器（PCA 保留了主要變異信息）。但 t-SNE/UMAP 不行——它們的映射函數不穩定，對新樣本無法產生一致的投影。

**Q3: 如何選擇 PCA 的 n_components？**

A: (1) 保留累積方差 95% 的主成分數；(2) 繪製解釋方差圖找 Elbow 點；(3) 如果 PCA 作為預處理，在下游分類任務上用交叉驗證選擇最佳維度。

**Q4: t-SNE 的 perplexity 如何設定？**

A: 通常在 5~50 之間嘗試不同值並觀察結果。Perplexity 可直觀理解為「每個點期望有幾個近鄰」，資料量大時可以設大一點（如 100）。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #PCA #降維 #UMAP #tSNE #機器學習 #ScikitLearn #資料視覺化 #KernelPCA #程式設計 #教學 #DataScience #MachineLearning #AI #特徵萃取
