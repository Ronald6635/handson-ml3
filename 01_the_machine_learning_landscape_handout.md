# 課程講義：機器學習概觀 (Chapter 01)

歡迎來到機器學習的世界！本章是整本書的起點，我們將建立機器學習的完整心智模型：從**為什麼需要機器學習**，到**如何分類各種學習類型**，再到**訓練模型時最常遇到的挑戰**。唯有紮實掌握這些基礎概念，才能在後續章節游刃有餘地選擇適當的演算法與工具。

---

## 1. 機器學習的定義與動機

### 理論背景

傳統程式設計依賴工程師手寫規則（`if-else`），面對複雜問題（垃圾郵件過濾、語音辨識）時規則爆炸且難以維護。機器學習讓電腦從資料中**自動學習規則**。

Arthur Samuel（1959）定義：

> *"Field of study that gives computers the ability to learn without being explicitly programmed."*

Tom Mitchell（1997）更精確的定義：

> *"A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."*

機器學習的典型應用場景：

- 問題過於複雜、規則太多（手寫數字辨識）
- 規則隨時間漂移（詐欺偵測）
- 需要從資料中挖掘隱藏模式（市場分析）
- 處理非結構化資料（影像、文字、語音）

### 核心代碼

本章的經典範例：用 GDP 預測生活滿意度，只需寥寥數行。

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 載入資料（生活滿意度 vs GDP）
data = pd.read_csv("datasets/lifesat/lifesat.csv")
X = data[["GDP per capita (USD)"]].values
y = data[["Life satisfaction"]].values

# 訓練線性模型
model = LinearRegression()
model.fit(X, y)

# 預測塞浦路斯（Cyprus）的生活滿意度
X_new = [[37_655.2]]  # GDP per capita in 2020
print(model.predict(X_new))  # 輸出約 [[6.30]]
```

### ⚡ 補充練習 1

**理論題：** 試說明「基於規則的系統」與「機器學習系統」在垃圾郵件過濾上的差異。若用戶開始使用新詞彙繞過過濾，兩種方法各如何應對？

**實作題：** 將 `LinearRegression` 替換為 `sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)`，重新預測塞浦路斯，並比較兩個模型的輸出差異。思考：為何結果不同？

---

## 2. 機器學習系統的分類

### 理論背景

機器學習系統可沿三個維度分類：

**維度一：是否使用標籤**

| 類型 | 說明 | 代表算法 |
|------|------|---------|
| 監督式學習 (Supervised) | 訓練資料含標籤 $(x_i, y_i)$ | 線性迴歸、SVM、Random Forest |
| 非監督式學習 (Unsupervised) | 訓練資料不含標籤 $\{x_i\}$ | K-Means、PCA、DBSCAN |
| 半監督式學習 (Semi-supervised) | 少量標籤 + 大量無標籤 | 深度信念網路 |
| 強化學習 (Reinforcement) | Agent 透過獎懲學習策略 | DQN、PPO |

**維度二：是否增量學習**

- **批次學習 (Batch Learning)**：用全部資料一次訓練，定期重新訓練。
- **線上學習 (Online Learning)**：資料逐筆或小批次到來時即時更新模型；`learning_rate` 控制適應速度。

**維度三：是否通用化**

- **基於實例 (Instance-based)**：記憶訓練資料，用相似度作預測（如 K-NN）。
- **基於模型 (Model-based)**：學習資料的參數化模型，推論時用方程式計算。

監督式學習的核心目標是學習一個函數 $f$：

$$\hat{y} = f_{\theta}(x)$$

透過最小化損失函數（如 MSE）調整參數 $\theta$：

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{\theta}(x^{(i)}) - y^{(i)}\right)^2$$

### 核心代碼

```python
# 監督式 vs 非監督式 API 對比
from sklearn.linear_model import LinearRegression       # 監督式
from sklearn.cluster import KMeans                       # 非監督式

# 監督式：需要 (X, y)
reg = LinearRegression().fit(X_train, y_train)

# 非監督式：只需要 X（無標籤）
km = KMeans(n_clusters=3, random_state=42).fit(X_train)
labels = km.labels_   # 各樣本的群集編號
```

### ⚡ 補充練習 2

**理論題：** 以下任務分別屬於哪種學習類型？
1. 根據歷史成交資料預測股價
2. 將數百萬篇新聞自動歸類為「財經」、「體育」、「政治」
3. 訓練機器人學習行走

**實作題：** 使用 `sklearn.datasets.make_blobs(n_samples=300, centers=4)` 產生資料，套用 `KMeans(n_clusters=4)` 聚類，繪製聚類結果並標示群心（`cluster_centers_`）。

---

## 3. 主要挑戰：過擬合與欠擬合

### 理論背景

模型訓練的兩大敵人：

**欠擬合 (Underfitting)**：模型過於簡單，連訓練集都無法擬合。

- 原因：特徵不足、模型複雜度不夠
- 對策：增加特徵、換更複雜的模型、降低正則化強度

**過擬合 (Overfitting)**：模型對訓練集記憶太深，無法泛化到新資料。

- 原因：模型複雜度過高、訓練資料太少或含雜訊
- 對策：增加訓練資料、正則化（Ridge/Lasso）、Dropout、減少特徵

可用**偏差-變異數權衡 (Bias-Variance Trade-off)** 描述：

$$\text{泛化誤差} \approx \text{偏差}^2 + \text{變異數} + \text{不可減少的雜訊}$$

**測試集的黃金法則**：

```
只能用訓練集調整模型 → 用驗證集（或 CV）選模型 → 最後只碰一次測試集
```

過多次使用測試集會造成「資訊洩漏」，讓測試集指標過於樂觀。

### 核心代碼

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# 建立具正則化的 Pipeline
ridge_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=10, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))  # alpha 越大正則化越強
])

# 使用交叉驗證評估泛化能力（比單一 val set 更穩健）
scores = cross_val_score(ridge_pipeline, X_train, y_train,
                         cv=5, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-scores)
print(f"CV RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")
```

### ⚡ 補充練習 3

**理論題：** 一個模型在訓練集的準確率為 99%，在測試集為 70%，這是過擬合還是欠擬合？應採取什麼策略改善？

**實作題：** 使用 `sklearn.datasets.make_regression(n_samples=100, noise=20, random_state=42)` 建立資料，分別訓練 `PolynomialFeatures(degree=1)`、`degree=5`、`degree=20` 的模型，比較三者的訓練集 RMSE 與交叉驗證 RMSE，觀察過擬合現象。

---

## 4. 測試與驗證：防止資料洩漏

### 理論背景

**Hold-out 驗證**：

$$\text{訓練集} : \text{驗證集} : \text{測試集} \approx 70\% : 15\% : 15\%$$

**交叉驗證 (k-fold CV)**：將訓練集切成 $k$ 份，輪流用一份作驗證，其餘訓練，最後平均 $k$ 次的結果。計算量是 Hold-out 的 $k$ 倍，但評估更穩健。

**No Free Lunch (NFL) 定理**：不存在對所有問題都最好的模型；必須在資料上實驗驗證，而非憑直覺選擇。

### 核心代碼

```python
from sklearn.model_selection import StratifiedShuffleSplit

# 分層抽樣——確保訓練/測試集有相同的類別比例
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X, y_cat):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

### ⚡ 補充練習 4

**理論題：** 解釋「資料洩漏 (Data Leakage)」的概念，並舉一個在特徵工程中可能發生洩漏的具體案例。

**實作題：** 使用 `StratifiedShuffleSplit` 對 `sklearn.datasets.load_iris()` 進行分層切分，確認訓練集和測試集中各類別的比例與原始資料集一致（用 `pd.value_counts()` 驗證）。

---

## 結論

本章建立了機器學習的整體知識地圖：

- 機器學習用**資料驅動的方式**取代手寫規則
- 學習類型沿**標籤、增量、通用化**三個維度分類
- 訓練的最大挑戰是在**偏差與變異數之間取得平衡**
- 嚴格的**訓練/驗證/測試集分離**是誠實評估的基石

下一章（Ch02）將把這些概念具體化，帶領你完成一個從資料取得到模型部署的完整端到端專案。

---

## 課後作業

**作業一：模型比較實驗**

使用 `lifesat.csv` 資料集，比較以下三種模型在 5-fold 交叉驗證下的 RMSE：

1. `LinearRegression`
2. `KNeighborsRegressor(n_neighbors=3)`
3. `KNeighborsRegressor(n_neighbors=10)`

撰寫結論：哪個模型最佳？為什麼？調整 `n_neighbors` 對結果有何影響？

**作業二：思考題**

假設你正在建立一個信用評分模型，訓練資料中有欄位「是否逾期還款（目前）」。這個欄位能放入特徵嗎？說明你的理由（提示：想想部署時是否能取得此資訊）。
