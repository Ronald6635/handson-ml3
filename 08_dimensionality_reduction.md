<!-- meta-title: 第8章 – 降維技術完整指南：主成分分析與其他演算法實戰應用 -->
<!-- meta-description: 本章節將深入探討降維技術的核心概念與實戰應用，包括主成分分析（PCA）、隨機投影、局部線性嵌入（LLE）、t-SNE 等演算法。我們將學習如何使用 Scikit-Learn 實作這些技術，並通過 MNIST 資料集進行實際應用。 -->
<!-- meta-keywords: Python, 機器學習, Scikit-Learn, 降維, PCA, t-SNE, LLE, 資料視覺化, 維度縮減 -->
<!-- meta-hashtags: #Python #機器學習 #降維 #PCA #tSNE #ScikitLearn #資料科學 #技術教學 -->

# 🐍 第8章 – 降維技術完整指南

歡迎來到第8章！本章將探討降維技術，這是機器學習中處理高維資料的重要工具。我們將學習如何使用主成分分析（PCA）和其他演算法來降低資料維度，同時保留最重要的資訊。

## 📝 本文目錄
- [設定](#設定)
- [PCA](#PCA)
- [主成分](#主成分)
- [投影到d維度](#投影到d維度)
- [使用Scikit-Learn](#使用Scikit-Learn)
- [解釋變異比率](#解釋變異比率)
- [選擇正確的維度數量](#選擇正確的維度數量)
- [PCA用於壓縮](#PCA用於壓縮)
- [隨機PCA](#隨機PCA)
- [增量PCA](#增量PCA)
- [隨機投影](#隨機投影)
- [LLE](#LLE)
- [核PCA](#核PCA)
- [練習解答](#練習解答)

## 🎯 關鍵重點 (Key Takeaways)
- **降維動機**: 加速訓練、資料視覺化、壓縮儲存
- **PCA原理**: 找到資料的最大變異方向作為主成分
- **Scikit-Learn實作**: 使用 `PCA`、`RandomizedPCA`、`IncrementalPCA` 等類別
- **其他技術**: 隨機投影、LLE、t-SNE 等流形學習方法
- **實戰應用**: MNIST資料集降維與視覺化

## <a id="設定"></a>🚀 設定

💡 **實際應用情境：** 在開始任何機器學習專案之前，建立適當的開發環境是非常重要的。這包括確認Python版本、安裝必要的函式庫，並設定圖表顯示參數。

### 範例 1: 環境檢查
```python
import sys

assert sys.version_info >= (3, 7)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入sys模組，用於檢查Python版本
2. `第3行`: 斷言Python版本必須大於等於3.7，確保相容性

**🎯 重點摘要:**

- **核心功能**: 檢查Python版本以確保程式碼相容性
- **潛在問題**: 舊版Python可能導致某些功能無法使用
- **最佳使用情境**: 專案初始化階段的環境驗證

### 範例 2: Scikit-Learn版本檢查
```python
from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
```

**✅ 程式碼逐行解析：**

1. `第1行`: 從packaging匯入version，用於版本比較
2. `第2行`: 匯入sklearn函式庫
3. `第4行`: 斷言sklearn版本必須大於等於1.0.1

**🎯 重點摘要:**

- **核心功能**: 確保Scikit-Learn版本符合要求
- **潛在問題**: 舊版函式庫可能缺少新功能或有bug
- **最佳使用情境**: 機器學習專案的依賴檢查

### 範例 3: 圖表設定
```python
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入matplotlib.pyplot作為plt
2. `第3-6行`: 設定字體大小和標籤大小，使圖表更清晰

**🎯 重點摘要:**

- **核心功能**: 統一圖表的外觀和可讀性
- **潛在問題**: 字體大小過大可能導致圖表擁擠
- **最佳使用情境**: 資料視覺化專案的圖表設定

### 範例 4: 儲存圖表函數
```python
from pathlib import Path

IMAGES_PATH = Path() / "images" / "dim_reduction"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 從pathlib匯入Path，用於路徑操作
2. `第3-4行`: 建立圖片儲存目錄
3. `第6-10行`: 定義儲存圖表的函數，支援高解析度輸出

**🎯 重點摘要:**

- **核心功能**: 自動建立目錄並儲存高品質圖表
- **潛在問題**: 目錄權限問題可能導致儲存失敗
- **最佳使用情境**: 書籍或報告中的圖表輸出

## <a id="PCA"></a>🐍 PCA

💡 **實際應用情境：** 主成分分析（Principal Component Analysis）是降維最常用的技術之一。它通過找到資料的最大變異方向來降低維度，同時保留最多的資訊。

### 範例 5: 產生3D資料集
```python
import numpy as np
from scipy.spatial.transform import Rotation

m = 60
X = np.zeros((m, 3))
np.random.seed(42)
angles = (np.random.rand(m) ** 3 + 0.5) * 2 * np.pi
X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) * 0.5
X += 0.28 * np.random.randn(m, 3)
X = Rotation.from_rotvec([np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)
X += [0.2, 0, 0.2]
```

**✅ 程式碼逐行解析：**

1. `第3行`: 設定樣本數量為60
2. `第4行`: 初始化3D資料集
3. `第6行`: 產生不均勻分佈的角度
4. `第7行`: 計算橢圓形座標
5. `第8-10行`: 加入雜訊、旋轉和平移

**🎯 重點摘要:**

- **核心功能**: 產生複雜的3D測試資料集
- **潛在問題**: 隨機種子確保重現性
- **最佳使用情境**: PCA演算法的示範資料

## <a id="主成分"></a>🎯 主成分

💡 **實際應用情境：** 主成分是資料變異最大的方向。通過SVD分解，我們可以找到這些方向並用於降維。

### 範例 6: 計算主成分
```python
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt[0]
c2 = Vt[1]
```

**✅ 程式碼逐行解析：**

1. `第1行`: 將資料中心化（減去均值）
2. `第2行`: 進行SVD分解
3. `第3-4行`: 提取前兩個主成分

**🎯 重點摘要:**

- **核心功能**: 使用SVD找到主成分
- **潛在問題**: 資料必須先中心化
- **最佳使用情境**: 手動實作PCA

## <a id="投影到d維度"></a>📐 投影到d維度

💡 **實際應用情境：** 將高維資料投影到低維空間，保留最重要的特徵。

### 範例 7: 投影到2D
```python
W2 = Vt[:2].T
X2D = X_centered @ W2
```

**✅ 程式碼逐行解析：**

1. `第1行`: 建立投影矩陣
2. `第2行`: 將資料投影到2D

**🎯 重點摘要:**

- **核心功能**: 將3D資料降維到2D
- **潛在問題**: 維度選擇影響資訊保留
- **最佳使用情境**: 資料視覺化

## <a id="使用Scikit-Learn"></a>🔧 使用Scikit-Learn

💡 **實際應用情境：** Scikit-Learn提供了方便的PCA類別，自動處理中心化和投影。

### 範例 8: Scikit-Learn PCA
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入PCA類別
2. `第3-4行`: 建立PCA實例並轉換資料

**🎯 重點摘要:**

- **核心功能**: 簡單易用的PCA實作
- **潛在問題**: 自動中心化處理
- **最佳使用情境**: 大多數降維任務

### 範例 9: 檢查主成分
```python
pca.components_
```

**✅ 程式碼逐行解析：**

1. `第1行`: 存取主成分向量

**🎯 重點摘要:**

- **核心功能**: 查看主成分的方向
- **潛在問題**: 解釋主成分的物理意義
- **最佳使用情境**: 特徵重要性分析

## <a id="解釋變異比率"></a>📊 解釋變異比率

💡 **實際應用情境：** 解釋變異比率告訴我們每個主成分保留了多少原始資料的變異。

### 範例 10: 變異比率
```python
pca.explained_variance_ratio_
```

**✅ 程式碼逐行解析：**

1. `第1行`: 取得各主成分的解釋變異比率

**🎯 重點摘要:**

- **核心功能**: 評估各維度的重要性
- **潛在問題**: 累計比率決定保留維度
- **最佳使用情境**: 選擇最佳降維數量

## <a id="選擇正確的維度數量"></a>🎛️ 選擇正確的維度數量

💡 **實際應用情境：** 選擇合適的維度數量需要在壓縮和資訊保留之間取得平衡。

### 範例 11: MNIST資料集載入
```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False, parser="auto")
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入資料集載入函數
2. `第3行`: 載入MNIST資料集
3. `第4-5行`: 分割訓練和測試集

**🎯 重點摘要:**

- **核心功能**: 載入標準機器學習資料集
- **潛在問題**: 大型資料集需要大量記憶體
- **最佳使用情境**: 分類任務的基準資料

### 範例 12: 自動維度選擇
```python
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
```

**✅ 程式碼逐行解析：**

1. `第1行`: 對所有資料擬合PCA
2. `第2行`: 計算累計解釋變異
3. `第3行`: 找到保留95%變異的最小維度

**🎯 重點摘要:**

- **核心功能**: 自動確定最佳降維數量
- **潛在問題**: 計算成本高
- **最佳使用情境**: 大型高維資料集

## <a id="PCA用於壓縮"></a>🗜️ PCA用於壓縮

💡 **實際應用情境：** PCA可以用於資料壓縮，減少儲存空間同時保留重要資訊。

### 範例 13: 資料壓縮
```python
pca = PCA(0.95)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 建立保留95%變異的PCA
2. `第2行`: 壓縮資料
3. `第3行`: 重建原始資料

**🎯 重點摘要:**

- **核心功能**: 可逆的資料壓縮
- **潛在問題**: 資訊損失不可避免
- **最佳使用情境**: 儲存空間有限的應用

## <a id="隨機PCA"></a>🎲 隨機PCA

💡 **實際應用情境：** 當資料集很大時，隨機PCA提供更快的近似解。

### 範例 14: 隨機PCA
```python
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 使用隨機SVD求解器
2. `第2行`: 快速降維

**🎯 重點摘要:**

- **核心功能**: 近似但更快的PCA
- **潛在問題**: 結果的準確性略低
- **最佳使用情境**: 大型資料集的快速處理

## <a id="增量PCA"></a>📈 增量PCA

💡 **實際應用情境：** 當資料集太大無法放入記憶體時，增量PCA可以分批處理。

### 範例 15: 增量PCA
```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
```

**✅ 程式碼逐行解析：**

1. `第3行`: 建立增量PCA實例
2. `第4-5行`: 分批擬合資料
3. `第7行`: 轉換整個資料集

**🎯 重點摘要:**

- **核心功能**: 記憶體高效的PCA
- **潛在問題**: 需要多次通過資料
- **最佳使用情境**: 超大型資料集

## <a id="隨機投影"></a>🎯 隨機投影

💡 **實際應用情境：** 隨機投影是一種簡單的降維技術，通過隨機矩陣將資料投影到低維空間。

### 範例 16: 高斯隨機投影
```python
from sklearn.random_projection import GaussianRandomProjection

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced = gaussian_rnd_proj.fit_transform(X)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入高斯隨機投影
2. `第3-4行`: 應用隨機投影

**🎯 重點摘要:**

- **核心功能**: 快速近似降維
- **潛在問題**: 理論保證的維度選擇
- **最佳使用情境**: 非常高維資料

## <a id="LLE"></a>🌀 LLE

💡 **實際應用情境：** 局部線性嵌入（Locally Linear Embedding）是一種流形學習技術，適合非線性降維。

### 範例 17: LLE應用
```python
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 產生瑞士捲資料集
2. `第4-5行`: 應用LLE降維

**🎯 重點摘要:**

- **核心功能**: 非線性流形降維
- **潛在問題**: 參數選擇敏感
- **最佳使用情境**: 流形結構資料

## <a id="核PCA"></a>🌰 核PCA

💡 **實際應用情境：** 核PCA使用核技巧處理非線性可分的資料。

### 範例 18: 核PCA
```python
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04, random_state=42)
X_reduced = rbf_pca.fit_transform(X_swiss)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入核PCA
2. `第3-4行`: 使用RBF核進行降維

**🎯 重點摘要:**

- **核心功能**: 非線性PCA變體
- **潛在問題**: 計算成本高
- **最佳使用情境**: 非線性資料結構

## <a id="練習解答"></a>📚 練習解答

💡 **實際應用情境：** 通過實際練習來鞏固降維技術的理解和應用。

### 練習9: MNIST分類比較
```python
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 建立隨機森林分類器
2. `第2行`: 在完整資料上訓練
3. `第3-4行`: 預測和評估

**🎯 重點摘要:**

- **核心功能**: 基準模型訓練
- **潛在問題**: 高維資料訓練慢
- **最佳使用情境**: 比較降維效果

### 練習10: t-SNE視覺化
```python
from sklearn.manifold import TSNE

X_sample, y_sample = X_train[:5000], y_train[:5000]
tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
X_reduced = tsne.fit_transform(X_sample)
```

**✅ 程式碼逐行解析：**

1. `第3行`: 取樣資料子集
2. `第4-5行`: 應用t-SNE降維

**🎯 重點摘要:**

- **核心功能**: 高品質資料視覺化
- **潛在問題**: 計算時間長
- **最佳使用情境**: 探索性資料分析

## 💡 總結與最佳實踐

降維技術是處理高維資料的重要工具：

- **PCA**: 適用於線性降維，快速且有效
- **隨機投影**: 簡單快速，適合非常高維資料
- **流形學習**: LLE、t-SNE等適合非線性結構
- **實戰建議**: 先嘗試PCA，如效果不佳再考慮其他方法

## ❓ 常見問答 (FAQ)

**Q: 降維會不會丟失重要資訊？**
A: 會，但可以通過選擇合適的維度數量來最小化損失。

**Q: 如何選擇降維演算法？**
A: 從PCA開始，如果資料有非線性結構，考慮t-SNE或LLE。

## 🏷️ 推薦標籤 (Suggested Hashtags)

#Python #機器學習 #降維 #PCA #tSNE #ScikitLearn #資料科學 #視覺化