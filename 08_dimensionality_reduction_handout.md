# 課程講義：降維 (Chapter 08)

高維資料無處不在——一張 28×28 的 MNIST 圖片有 784 個維度，基因組資料可能有數萬個特徵。**維度詛咒**使得高維空間的訓練既緩慢又不穩定：資料點之間的距離趨於相等，密度極度稀疏。降維技術通過在低維空間中保留資料的最重要結構，解決計算瓶頸、資料視覺化和去雜訊問題。

---

## 1. 維度詛咒與降維的必要性

### 理論背景

**維度詛咒 (Curse of Dimensionality)**：

- 在 $d$ 維超立方體中，若要讓每條邊長為 $l$ 的超立方體含 1 筆樣本，所需的邊長 $l$ 隨 $d$ 指數級增大
- 1000 維空間中，兩個隨機點的距離幾乎相等（距離集中現象），K-NN 等基於距離的方法失效
- 若需要 10 個維度每維 5 個訓練點來覆蓋空間，100 個特徵需要 $5^{100}$ 筆樣本！

**降維的主要用途**：

| 用途 | 說明 |
|------|------|
| 加速訓練 | 減少特徵數，降低計算量 |
| 資料視覺化 | 投影到 2D/3D，觀察群聚結構 |
| 資料壓縮 | JPEG 使用類似 PCA 的 DCT |
| 去雜訊 | 保留主要方向，過濾雜訊 |

**投影法 vs 流形學習**：

- 投影（PCA）：假設資料位於低維超平面或超空間
- 流形學習（t-SNE, LLE）：假設資料位於低維非線性流形（Swiss Roll 問題）

### 核心代碼

```python
import numpy as np
import matplotlib.pyplot as plt

# 維度詛咒的直觀展示
for d in [1, 2, 3, 10, 100, 1000]:
    n_samples = 1000
    X = np.random.rand(n_samples, d)  # d 維均勻分布
    # 隨機兩點的平均距離隨維度增加
    dist = np.sqrt(((X[0] - X[1:])**2).sum(axis=1)).mean()
    print(f"d={d:5d}: 平均兩點距離 = {dist:.4f}")
```

### ⚡ 補充練習 1

**理論題：** 在 2D 平面上，均勻分布的 1000 個點中，95% 的點距原點的距離在 [0.95, 1.05] 之間的概率是多少（假設資料在單位圓內）？隨著維度增加，這個比例如何變化？

---

## 2. PCA：主成分分析

### 理論背景

**PCA (Principal Component Analysis)**：找出資料**變異數最大的方向**（主成分），投影後保留最多資訊。

**SVD 分解**（PCA 的實際計算方式）：

$$\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

- $\mathbf{V}$ 的列向量即為主成分（Principal Components）
- $\boldsymbol{\Sigma}$ 的對角元素（奇異值）反映各主成分的重要性
- $\mathbf{X} \mathbf{V}_d$（取前 $d$ 個主成分）即為降維後的表示

**解釋變異數比 (Explained Variance Ratio)**：

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^{n} \lambda_i}$$

其中 $\lambda_k$ 是第 $k$ 個主成分的變異數（特徵值）。

**重要假設**：PCA 假設主成分是線性組合；對非線性資料需用核 PCA 或流形學習。

### 核心代碼

```python
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

# 載入 MNIST（784 維 → 降至 2 維用於視覺化）
mnist = fetch_openml("mnist_784", as_frame=False)
X_mnist = mnist.data[:5000]
y_mnist = mnist.target[:5000]

# 保留 95% 的解釋變異數
pca_95 = PCA(n_components=0.95)  # 自動決定維度數
X_reduced = pca_95.fit_transform(X_mnist)
print(f"原始維度: {X_mnist.shape[1]}")              # 784
print(f"降維後維度: {X_reduced.shape[1]}")          # 約 154
print(f"解釋變異數: {pca_95.explained_variance_ratio_.sum():.3f}")

# 解壓縮（重建）並計算重建誤差
X_recovered = pca_95.inverse_transform(X_reduced)
reconstruction_error = np.mean((X_mnist - X_recovered) ** 2)
print(f"重建 MSE: {reconstruction_error:.2f}")

# 視覺化解釋變異數
pca_full = PCA().fit(X_mnist)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.axhline(y=0.95, color="r", linestyle="--", label="95%")
plt.legend()
plt.grid()
plt.title("Explained Variance Ratio vs Number of PCA Components")
plt.show()
```

### ⚡ 補充練習 2

**理論題：** PCA 降維後再用 `inverse_transform` 重建，重建的資料和原始資料不會完全相同——損失的資訊是什麼？這個損失對後續機器學習任務是有益還是有害的？

**實作題：** 對 MNIST 用 `PCA(n_components=2)` 降至 2D，用散點圖繪製（顏色代表數字類別），觀察哪些數字在 2D PCA 空間中容易分離，哪些容易混淆。

---

## 3. 增量 PCA 與隨機化 PCA

### 理論背景

**標準 PCA 的問題**：需要將整個訓練集載入記憶體（SVD 計算）。

**Incremental PCA (IPCA)**：分批次（mini-batch）更新主成分，無需一次載入全部資料：

- 適合大型資料集，記憶體友善
- 計算結果與標準 PCA 近似

**Randomized PCA**：用隨機演算法快速近似前 $d$ 個主成分，比標準 SVD 快得多（當 $d \ll n$ 時）：

```
標準 PCA：O(m × n²) 或 O(n³)
Randomized PCA：O(m × d²) + O(d³)  ← 快很多！
```

### 核心代碼

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

# Incremental PCA（逐批次處理）
n_batches = 10
ipca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_mnist, n_batches):
    ipca.partial_fit(X_batch)      # 逐批更新

X_ipca = ipca.transform(X_mnist)

# Randomized PCA（速度快，適合 n_components 遠小於特徵數時）
pca_random = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_random   = pca_random.fit_transform(X_mnist)

print(f"IPCA vs PCA 差異（均方）: "
      f"{np.mean((np.abs(X_ipca) - np.abs(X_reduced[:, :154]))**2):.6f}")
```

### ⚡ 補充練習 3

**理論題：** 若記憶體限制為 2GB，訓練資料有 1 億筆樣本、每筆 100 個特徵（float32），標準 PCA 能直接計算嗎？如何使用 Incremental PCA 解決？

**實作題：** 比較在 MNIST 上 `PCA(svd_solver="full")`、`PCA(svd_solver="randomized")`、`IncrementalPCA` 的計算時間（`%timeit`），以及降維結果的相似度（用 Frobenius 範數衡量差異）。

---

## 4. 核 PCA 與流形學習

### 理論背景

**核 PCA (kPCA)**：先用核函數映射到高維特徵空間，再在高維空間做 PCA。能處理非線性流形。

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$$

**主要非線性降維方法**：

| 方法 | 核心思想 | 優點 | 缺點 |
|------|---------|------|------|
| **Kernel PCA** | 在核特徵空間做 PCA | 靈活，可調核函數 | 超參數多 |
| **LLE** | 保持局部鄰域關係 | 展開流形（Swiss Roll） | 對雜訊敏感 |
| **t-SNE** | 高維鄰近關係 → 低維t分布 | 視覺化極佳 | 非確定性，不可用於預測 |
| **UMAP** | 拓撲保持 | 快、可擴展、可用於預測 | 需調參 |

**t-SNE** 的核心：讓高維空間中「相近的點」在低維中也相近，「遙遠的點」在低維中也遠（用 KL 散度衡量分布差異）：

$$KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### 核心代碼

```python
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding

# Swiss Roll 資料（典型非線性流形）
from sklearn.datasets import make_swiss_roll
X_swiss, t = make_swiss_roll(n_samples=1000, random_state=42)

# 核 PCA（RBF 核）
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04,
                 fit_inverse_transform=True,  # 允許重建
                 random_state=42)
X_kpca = kpca.fit_transform(X_swiss)

# LLE：最適合展開流形
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_lle = lle.fit_transform(X_swiss)

# t-SNE：視覺化效果最好（但不可預測新樣本）
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_mnist[:2000])

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
            c=y_mnist[:2000].astype(int), cmap="tab10", s=5, alpha=0.7)
plt.colorbar(label="Digit class")
plt.title("t-SNE of MNIST (2000 samples)")
plt.show()
```

### ⚡ 補充練習 4

**理論題：** t-SNE 有一個重要參數 `perplexity`（困惑度），它控制每個點考慮的「有效鄰居數量」。`perplexity` 很小（如 5）和很大（如 100）各會產生什麼視覺效果？

**實作題：** 對 `make_swiss_roll` 資料，分別用標準 `PCA(n_components=2)`、`KernelPCA(kernel="rbf", gamma=0.04)` 和 `LocallyLinearEmbedding` 降至 2D，繪製三個 2D 圖（顏色 = Swiss Roll 的位置 `t`），比較哪種方法能正確「展開」Swiss Roll。

---

## 結論

降維技術的工具箱：

- **PCA**：線性降維的首選，快速、可解釋，支援 `inverse_transform`
- **`n_components=0.95`**：保留 95% 解釋變異數，自動決定維度數
- **Incremental/Randomized PCA**：大資料集的高效替代方案
- **Kernel PCA / LLE / t-SNE**：處理非線性流形，t-SNE 最適合視覺化

下一章（Ch09）進入無監督學習的核心：聚類，學習如何在沒有標籤的情況下發現資料結構。

---

## 課後作業

**作業：PCA 加速分類與視覺化**

1. **加速效果**：在完整 MNIST 上，比較以下兩種管線的訓練時間和測試準確率：
   - 直接 `RandomForestClassifier`
   - `PCA(n_components=0.95)` → `RandomForestClassifier`

2. **視覺化**：對 MNIST 用 `t-SNE(n_components=2, perplexity=30)` 降至 2D，繪製散點圖（10 個數字用 10 種顏色），觀察：哪些數字的群聚最清晰？哪些容易混淆？這與 Ch03 的混淆矩陣分析一致嗎？

3. **壓縮率**：計算 PCA 壓縮後的「位元數」節省（原始 784×8 bits，壓縮後的維度×32 bits），以及重建圖片的視覺品質（用肉眼判斷 `inverse_transform` 的結果）。
