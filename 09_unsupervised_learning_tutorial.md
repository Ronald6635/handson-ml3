<!-- meta-title: 非監督學習完整指南：K-Means、DBSCAN、高斯混合模型與異常偵測 -->
<!-- meta-description: 深入非監督學習的三大工具：K-Means（內聚性、輪廓係數、MiniBatch）、DBSCAN（密度聚類）、高斯混合模型（EM 演算法、BIC/AIC 模型選擇、異常偵測）。含 Scikit-Learn 實戰。 -->
<!-- meta-keywords: Python, K-Means, DBSCAN, 高斯混合模型, 聚類, 異常偵測, 非監督學習, Scikit-Learn, 機器學習 -->
<!-- meta-hashtags: #Python #KMeans #DBSCAN #GMM #非監督學習 #聚類 #異常偵測 #機器學習 #ScikitLearn #教學 -->

# 🐍 非監督學習完整指南：聚類、密度估計與異常偵測

非監督學習（Unsupervised Learning）在沒有標籤的情況下，從原始資料中發現有意義的結構。本教學帶你掌握三大聚類工具——K-Means、DBSCAN 和高斯混合模型（GMM）——並學習如何用 GMM 建構**異常偵測**系統，應用於台灣金融詐欺偵測和製造業品質管控。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🎯 K-Means 聚類](#kmeans)
- [📊 選擇最佳群數 K](#choosing-k)
- [🚀 MiniBatch K-Means](#minibatch)
- [🔵 DBSCAN：密度聚類](#dbscan)
- [🔢 高斯混合模型 (GMM)](#gmm)
- [🚨 異常偵測](#anomaly-detection)
- [🏷️ 半監督學習](#semi-supervised)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **K-Means** 假設群落為球形，對離群值敏感，需要預先指定 K
- **輪廓係數（Silhouette Score）** 和**慣性（Inertia）** 用於選擇最佳 K，但各有局限
- **DBSCAN** 能發現任意形狀的群落，自動識別噪音點，但對密度均勻的資料效果較差
- **GMM（高斯混合模型）** 比 K-Means 更靈活——允許橢圓形群落，輸出軟分配機率
- GMM 的 **BIC/AIC** 可用於模型選擇（群數、協方差結構）

---

## <a id="kmeans"></a>🎯 K-Means 聚類

💡 **實際應用情境：** 電商平台對台灣消費者進行分群——無需標籤，讓演算法從消費頻率、客單價、品類偏好自動發現「輕度用戶/重度用戶/高價值用戶」等族群，再針對每群設計行銷策略。

### 範例 1: KMeans 基礎訓練

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# 生成 5 個球形群落的模擬資料
X_blobs, y_true = make_blobs(
    n_samples=500, centers=5, n_features=2,
    cluster_std=1.0, random_state=42
)

# 訓練 K-Means（指定 K=5）
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300, random_state=42)
kmeans.fit(X_blobs)

# 核心屬性
print(f"群中心:\n{kmeans.cluster_centers_}")
print(f"標籤（前10個）: {kmeans.labels_[:10]}")
print(f"慣性（Inertia）: {kmeans.inertia_:.2f}")  # 越小越緊密

# 預測新樣本所屬群
X_new = np.array([[0, 2], [3, 2], [-3, 3]])
print(f"\n新樣本群標籤: {kmeans.predict(X_new)}")
print(f"到各群中心距離:\n{kmeans.transform(X_new).round(2)}")  # transform 回傳距離矩陣
```

**✅ 程式碼逐行解析：**

1. `KMeans(n_init=10)`: 執行 10 次隨機初始化，取慣性最小的結果（避免局部最優）
2. `kmeans.inertia_`: 所有樣本到其最近群中心的距離平方和，衡量群落緊密程度
3. `kmeans.transform(X_new)`: 回傳 (n_samples, k) 形狀的距離矩陣，可用於特徵工程

**🎯 重點摘要:**

- K-Means 的時間複雜度：O(n × k × iter)，大資料集需用 MiniBatch K-Means
- 重要前提：特徵需要**標準化**，否則大尺度特徵會主導距離計算

---

## <a id="choosing-k"></a>📊 選擇最佳群數 K

### 範例 2: Elbow 方法 + 輪廓分析

```python
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_blobs)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_blobs, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow 法：慣性 vs K
axes[0].plot(k_range, inertias, "bo-")
axes[0].set_xlabel("K（群數）")
axes[0].set_ylabel("慣性（Inertia）")
axes[0].set_title("Elbow 法（找斜率轉折點）")

# 輪廓係數法：
axes[1].plot(k_range, silhouette_scores, "rs-")
axes[1].set_xlabel("K（群數）")
axes[1].set_ylabel("輪廓係數（Silhouette Score）")
axes[1].set_title("輪廓係數（越高越好）")

plt.tight_layout()
plt.show()

print(f"最佳 K（輪廓係數）: {k_range.start + silhouette_scores.index(max(silhouette_scores))}")
```

**✅ 程式碼逐行解析：**

1. **Elbow 法**：繪製慣性 vs K，斜率轉折點（Elbow）即最佳 K（主觀判斷）
2. `silhouette_score(X, labels)`: 輪廓係數 = (b−a)/max(a,b)，範圍 [-1, 1]
   - a = 與同群其他點的平均距離（越小越好）
   - b = 與最近的異群點的平均距離（越大越好）
   - 值越高（接近 1）表示聚類越好

**🎯 重點摘要:**

- 輪廓係數 > 0.5 表示合理聚類；> 0.7 表示優秀聚類
- 兩種方法結合使用，也可考慮業務意義（如「行銷預算分 3 群更實際」）

---

## <a id="minibatch"></a>🚀 MiniBatch K-Means

### 範例 3: MiniBatchKMeans vs KMeans

```python
from sklearn.cluster import MiniBatchKMeans
import time

X_large = np.random.randn(100_000, 2)  # 10 萬個樣本

# 比較訓練時間
for name, model in [
    ("KMeans", KMeans(n_clusters=5, n_init=3, random_state=42)),
    ("MiniBatch", MiniBatchKMeans(n_clusters=5, batch_size=1000, n_init=3, random_state=42)),
]:
    start = time.time()
    model.fit(X_large)
    elapsed = time.time() - start
    score = silhouette_score(X_large, model.labels_, sample_size=5000)
    print(f"{name:12s}: 時間={elapsed:.2f}秒, 輪廓係數≈{score:.4f}")
```

**🎯 重點摘要:**

- MiniBatch K-Means 速度快 3-10 倍，輪廓係數略低（因取近似解）
- 資料量 > 10 萬時，優先使用 `MiniBatchKMeans`

---

## <a id="dbscan"></a>🔵 DBSCAN：密度聚類

💡 **實際應用情境：** 在地圖上識別台北市的「商業熱點」——商家密度高的區域是一個群落（如信義區商業圈），而孤立的商家（如郊區加油站）被視為噪音點，不屬於任何群落。DBSCAN 天生適合這類形狀不規則的地理資料。

### 範例 4: DBSCAN 處理任意形狀群落

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# DBSCAN 參數：eps（鄰域半徑）和 min_samples（核心點最小近鄰數）
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X_moons)

# 標籤：-1 表示噪音點（異常值）
print(f"群標籤: {set(dbscan.labels_)}")       # 如 {0, 1, -1}
print(f"核心點數量: {len(dbscan.core_sample_indices_)}")
print(f"噪音點數量: {(dbscan.labels_ == -1).sum()}")

# 視覺化
plt.figure(figsize=(8, 5))
colors = ["blue" if l == 0 else "orange" if l == 1 else "red" for l in dbscan.labels_]
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=colors, alpha=0.7)
plt.title(f"DBSCAN 結果 (eps={dbscan.eps}, min_samples={dbscan.min_samples})")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `eps=0.2`: 鄰域半徑（需要根據資料尺度調整）——若點 A 和點 B 距離 ≤ eps，則 B 在 A 的鄰域內
2. `min_samples=5`: 「核心點（Core Point）」需在 eps 鄰域內至少有 5 個點
3. `labels_ == -1`: 噪音點的標籤為 -1（不屬於任何群落）

DBSCAN 的三類點：

| 類型 | 條件 | 說明 |
|------|------|------|
| 核心點（Core Point） | eps 鄰域內點數 ≥ min_samples | 群的「骨幹」 |
| 邊界點（Border Point） | 在核心點鄰域內，但自身非核心點 | 群的「邊緣」 |
| 噪音點（Noise Point） | 不在任何核心點鄰域內 | 標籤 = -1 |

**🎯 重點摘要:**

- DBSCAN 的優點：不需指定 K，能識別任意形狀，自動處理噪音
- 缺點：密度不均勻的資料效果差，eps 需要調整
- 調參技巧：先繪製 kNN 距離圖（K = min_samples），找到距離急劇增大的位置設為 eps

---

## <a id="gmm"></a>🔢 高斯混合模型 (GMM)

💡 **實際應用情境：** 在金融市場分析中，股票回報率通常不是單一常態分佈，而是多個市場狀態（牛市、熊市、震盪市）的混合。GMM 能同時建模多個高斯分佈，並輸出每個資料點屬於哪個狀態的機率。

### 範例 5: GaussianMixture 訓練與評估

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# 生成橢圓形群落（K-Means 難以處理）
np.random.seed(42)
X_gmm = np.r_[
    np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100),    # 群 1（橢圓）
    np.random.multivariate_normal([4, 2], [[1.5, -0.3], [-0.3, 0.5]], 100),  # 群 2
    np.random.multivariate_normal([1, 6], [[0.5, 0], [0, 2]], 100),       # 群 3
]

# GMM 訓練（EM 演算法）
gmm = GaussianMixture(
    n_components=3,        # 假設 3 個高斯分佈
    covariance_type="full",  # 每個成分都有獨立的完整協方差矩陣
    n_init=10,             # 10 次隨機初始化
    random_state=42
)
gmm.fit(X_gmm)

# 評估
print(f"收斂次數: {gmm.n_iter_}")
print(f"對數似然: {gmm.score(X_gmm):.4f}")        # 越高越好（越接近 0）

# 軟分配：每個樣本屬於各群的機率
soft_labels = gmm.predict_proba(X_gmm[:3])
print(f"\n前 3 個樣本的軟分配機率:\n{soft_labels.round(3)}")

# 硬分配
hard_labels = gmm.predict(X_gmm)
print(f"\n硬分配標籤（前10個）: {hard_labels[:10]}")
```

**✅ 程式碼逐行解析：**

1. `covariance_type="full"`: 最靈活，每個高斯有獨立協方差矩陣（可以是任意橢圓形）
2. GMM 使用 **EM 演算法（期望最大化）**：E 步驟計算每個點屬於各成分的「責任」，M 步驟更新參數
3. `predict_proba`: 回傳**軟分配（Soft Assignment）**，比 K-Means 的硬分配更豐富

協方差類型選項：

| `covariance_type` | 說明 | 參數量 | 適用場景 |
|-------------------|------|-------|---------|
| `"full"` | 獨立完整協方差 | 最多 | 任意形狀 |
| `"tied"` | 所有群共享協方差 | 少 | 形狀相似的群 |
| `"diag"` | 對角協方差（無相關性） | 中 | 座標軸對齊的橢圓 |
| `"spherical"` | 球形（等同 K-Means） | 最少 | 球形群 |

**🎯 重點摘要:**

- GMM 是 K-Means 的「軟版」，輸出機率而非硬標籤
- 需要資料大致服從高斯分佈，對極度非高斯資料效果有限

---

## <a id="anomaly-detection"></a>🚨 異常偵測

### 範例 6: 用 GMM 的對數密度識別異常值

```python
# 利用 GMM 的密度函數識別異常（密度極低的點 = 異常）
densities = gmm.score_samples(X_gmm)  # 每個樣本的對數密度（log probability）

# 設定閾值：密度低於第 2 百分位數的視為異常
threshold = np.percentile(densities, 2)
anomalies = X_gmm[densities < threshold]

print(f"密度閾值: {threshold:.4f}")
print(f"偵測到的異常樣本數: {len(anomalies)}")

# 視覺化：正常樣本（藍色）和異常樣本（紅色星號）
plt.figure(figsize=(8, 6))
plt.scatter(X_gmm[:, 0], X_gmm[:, 1], c="steelblue", alpha=0.4, s=20, label="正常")
plt.scatter(anomalies[:, 0], anomalies[:, 1],
            c="red", marker="*", s=150, label="異常")
plt.title("GMM 異常偵測結果")
plt.legend()
plt.show()
```

**🎯 重點摘要:**

- **百分位數閾值**：通常設 1~5%，根據業務容忍的誤報率調整
- GMM 異常偵測的優點：**機率解釋性強**（可以量化「有多不正常」）
- 其他常用異常偵測方法：`IsolationForest`（快速，適合高維）、`OneClassSVM`

---

## <a id="semi-supervised"></a>🏷️ 半監督學習應用

### 範例 7: K-Means 輔助標籤傳播

```python
from sklearn.datasets import load_digits

# 手寫數字資料集（少量標籤 + 大量未標籤）
X_digits, y_digits = load_digits(return_X_y=True)
X_train_small = X_digits[:50]   # 只有 50 個標籤
X_train_unlabeled = X_digits[50:1000]
X_test = X_digits[1000:]
y_train_small = y_digits[:50]
y_test = y_digits[1000:]

# Step 1：用 K-Means 在未標籤資料中找 50 個代表性樣本
kmeans_50 = KMeans(n_clusters=50, n_init=10, random_state=42)
kmeans_50.fit(X_train_unlabeled)

# 取每個群中離群中心最近的樣本（最具代表性）
representative_idx = np.argmin(
    kmeans_50.transform(X_train_unlabeled), axis=0
)
X_representative = X_train_unlabeled[representative_idx]
# 手動標記這 50 個代表性樣本（這裡用真實標籤模擬）
y_representative = y_digits[50 + representative_idx]

# Step 2：用標籤傳播——將每個群的代表標籤傳給同群所有樣本
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_representative, y_representative)
print(f"半監督學習準確率: {log_reg.score(X_test, y_test):.4f}")
```

**🎯 重點摘要:**

- 半監督策略：用聚類找「最具代表性的樣本」優先標記，最大化標記效益
- 適合標記成本高的場景（醫療影像、法律文件）

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: K-Means 和 GMM 如何選擇？**

A: 若確定群落是球形（尺度相近）且不需要機率輸出，K-Means 更快速；若群落可能是橢圓形，或需要每個點的歸屬機率（軟分配），用 GMM。

**Q2: DBSCAN 的 eps 如何設定？**

A: 繪製「K-NN 距離圖」：計算每個點到第 k 個近鄰的距離（k=min_samples），排序後繪圖，找到距離急劇增大的「knee 點」作為 eps。

**Q3: 如何評估聚類效果（沒有真實標籤時）？**

A: 常用指標：輪廓係數（`silhouette_score`）、Calinski-Harabasz 指數（`calinski_harabasz_score`）、Davies-Bouldin 指數（`davies_bouldin_score`）。若有真實標籤（但未用於訓練），可用 ARI（調整蘭德指數）。

**Q4: GMM 的 BIC 和 AIC 有什麼差別？**

A: 兩者都是在「模型擬合程度」和「模型複雜度」之間取得平衡。BIC 對複雜模型懲罰更重，傾向選擇更簡單的模型；AIC 懲罰較輕，傾向選擇更好地擬合資料的模型。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #KMeans #DBSCAN #GMM #非監督學習 #聚類 #異常偵測 #機器學習 #ScikitLearn #半監督學習 #程式設計 #教學 #DataScience #MachineLearning
