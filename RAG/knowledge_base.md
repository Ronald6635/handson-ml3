[來源: ch01 | 類型: cheatsheet] # Ch01 速查表：Machine Learning Landscape

> **核心主旨**：ML 的分類體系、基本術語與核心挑戰 —— 選對演算法前先搞清楚問題的類型。

---

---

[來源: ch01 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Supervised Learning | 用有標籤資料訓練，學習 input → output 映射 | 分類、迴歸（房價、垃圾郵件） |
| Unsupervised Learning | 無標籤，自動發現資料結構 | 聚類、降維、異常偵測 |
| Semi-supervised Learning | 少量標籤 + 大量無標籤 | 圖片標注資料稀缺時 |
| Reinforcement Learning | Agent 透過 reward/penalty 學習策略 | 遊戲 AI、機器人控制 |
| Batch Learning | 離線使用全部資料訓練，定期重新訓練 | 資料量固定、不需即時更新 |
| Online Learning | 資料逐步到達，增量更新模型 | 串流資料、記憶體有限時 |
| Instance-based Learning | 直接記憶訓練資料，用相似度預測 | KNN |
| Model-based Learning | 從資料中學習模型參數 | 線性迴歸、決策樹 |
| Overfitting | 模型在訓練集表現好但泛化差 | 模型太複雜 / 資料太少 |
| Underfitting | 模型太簡單，連訓練集都學不好 | 需要更複雜模型或更多特徵 |


---

[來源: ch01 | 類型: cheatsheet] 樹 |
| Overfitting | 模型在訓練集表現好但泛化差 | 模型太複雜 / 資料太少 |
| Underfitting | 模型太簡單，連訓練集都學不好 | 需要更複雜模型或更多特徵 |


---

---

[來源: ch01 | 類型: cheatsheet] ## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `train_test_split()` | `test_size`, `random_state`, `stratify` | 切分資料集 |
| `cross_val_score()` | `cv=5`, `scoring="accuracy"` | K-Fold 交叉驗證 |
| `StandardScaler` | – | 特徵標準化（μ=0, σ=1） |
| `SimpleImputer` | `strategy="median"` | 填補缺失值 |
| `Pipeline` | `steps=[("scaler", ...), ("clf", ...)]` | 串接多個轉換器 |

---

---

[來源: ch01 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

---

[來源: ch01 | 類型: cheatsheet] # 切分資料（分層抽樣）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

---

[來源: ch01 | 類型: cheatsheet] # 建立 Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

---

[來源: ch01 | 類型: cheatsheet] # 交叉驗證
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
print(f"RMSE: {-scores.mean():.2f} ± {scores.std():.2f}")
```

---

---

[來源: ch01 | 類型: cheatsheet] ## 4. 常見陷阱

- **資料洩漏（Data Leakage）**：StandardScaler 必須在 Pipeline 內，不能在 split 前 fit 整個資料集。
- **Snooping Bias**：多次在測試集上評估會讓測試集變成「驗證集」，最終指標過度樂觀。
- **不平衡類別**：`train_test_split` 加 `stratify=y` 確保類別比例一致。
- **測試集僅用一次**：調超參數用驗證集（或 CV），測試集只在最終評估時碰一次。

---

---

[來源: ch01 | 類型: cheatsheet] ## 5. 決策指南

```
問題有標籤嗎？
├── 是 → Supervised
│   ├── 預測連續值 → 迴歸 (LinearRegression, RandomForest)
│   └── 預測類別 → 分類 (LogisticRegression, SVM, GBM)
└── 否 → Unsupervised
    ├── 找群體結構 → 聚類 (KMeans, DBSCAN)
    ├── 降低維度 → PCA, t-SNE
    └── 找異常點 → IsolationForest, EllipticEnvelope
```

**資料量考量**：
- 小資料（< 10k）：SVM、GBM 表現佳
- 大資料（> 100k）：深度學習、線性模型（SGD）更有效率

---

[來源: ch01 | 類型: handout] # 課程講義：機器學習概觀 (Chapter 01)

歡迎來到機器學習的世界！本章是整本書的起點，我們將建立機器學習的完整心智模型：從**為什麼需要機器學習**，到**如何分類各種學習類型**，再到**訓練模型時最常遇到的挑戰**。唯有紮實掌握這些基礎概念，才能在後續章節游刃有餘地選擇適當的演算法與工具。

---

---

[來源: ch01 | 類型: handout] ### 理論背景

傳統程式設計依賴工程師手寫規則（`if-else`），面對複雜問題（垃圾郵件過濾、語音辨識）時規則爆炸且難以維護。機器學習讓電腦從資料中**自動學習規則**。

Arthur Samuel（1959）定義：

> *"Field of study that gives computers the ability to learn without being explicitly programmed."*

Tom Mitchell（1997）更精確的定義：

---

[來源: ch01 | 類型: handout] ves computers the ability to learn without being explicitly programmed."*

Tom Mitchell（1997）更精確的定義：

> *"A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."*

機器學習的典型應用場景：

- 問題過於複雜、規則太多（手寫數字辨識）
- 規則隨時間漂移（詐欺偵測）
- 需要從資料中挖掘隱藏模式（市場分析）
- 處理非結構化資料（影像、文字、語音）

---

[來源: ch01 | 類型: handout] ### 核心代碼

本章的經典範例：用 GDP 預測生活滿意度，只需寥寥數行。

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

---

[來源: ch01 | 類型: handout] # 載入資料（生活滿意度 vs GDP）
data = pd.read_csv("datasets/lifesat/lifesat.csv")
X = data[["GDP per capita (USD)"]].values
y = data[["Life satisfaction"]].values

---

[來源: ch01 | 類型: handout] # 訓練線性模型
model = LinearRegression()
model.fit(X, y)

---

[來源: ch01 | 類型: handout] # 預測塞浦路斯（Cyprus）的生活滿意度
X_new = [[37_655.2]]  # GDP per capita in 2020
print(model.predict(X_new))  # 輸出約 [[6.30]]
```

---

[來源: ch01 | 類型: handout] ### 補充練習 1

**理論題：** 試說明「基於規則的系統」與「機器學習系統」在垃圾郵件過濾上的差異。若用戶開始使用新詞彙繞過過濾，兩種方法各如何應對？

**實作題：** 將 `LinearRegression` 替換為 `sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)`，重新預測塞浦路斯，並比較兩個模型的輸出差異。思考：為何結果不同？

---

---

[來源: ch01 | 類型: handout] ### 理論背景

機器學習系統可沿三個維度分類：

**維度一：是否使用標籤**

| 類型 | 說明 | 代表算法 |
|------|------|---------|
| 監督式學習 (Supervised) | 訓練資料含標籤 $(x_i, y_i)$ | 線性迴歸、SVM、Random Forest |
| 非監督式學習 (Unsupervised) | 訓練資料不含標籤 $\{x_i\}$ | K-Means、PCA、DBSCAN |
| 半監督式學習 (Semi-supervised) | 少量標籤 + 大量無標籤 | 深度信念網路 |
| 強化學習 (Reinforcement) | Agent 透過獎懲學習策略 | DQN、PPO |


**維度二：是否增量學習**

---

[來源: ch01 | 類型: handout] ised) | 少量標籤 + 大量無標籤 | 深度信念網路 |
| 強化學習 (Reinforcement) | Agent 透過獎懲學習策略 | DQN、PPO |


**維度二：是否增量學習**

- **批次學習 (Batch Learning)**：用全部資料一次訓練，定期重新訓練。
- **線上學習 (Online Learning)**：資料逐筆或小批次到來時即時更新模型；`learning_rate` 控制適應速度。

**維度三：是否通用化**

- **基於實例 (Instance-based)**：記憶訓練資料，用相似度作預測（如 K-NN）。
- **基於模型 (Model-based)**：學習資料的參數化模型，推論時用方程式計算。

監督式學習的核心目標是學習一個函數 $f$：

$$\hat{y} = f_{\theta}(x)$$

---

[來源: ch01 | 類型: handout] - **基於模型 (Model-based)**：學習資料的參數化模型，推論時用方程式計算。

監督式學習的核心目標是學習一個函數 $f$：

$$\hat{y} = f_{\theta}(x)$$

透過最小化損失函數（如 MSE）調整參數 $\theta$：

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{\theta}(x^{(i)}) - y^{(i)}\right)^2$$

---

[來源: ch01 | 類型: handout] # 監督式 vs 非監督式 API 對比
from sklearn.linear_model import LinearRegression       # 監督式
from sklearn.cluster import KMeans                       # 非監督式

---

[來源: ch01 | 類型: handout] # 監督式：需要 (X, y)
reg = LinearRegression().fit(X_train, y_train)

---

[來源: ch01 | 類型: handout] # 非監督式：只需要 X（無標籤）
km = KMeans(n_clusters=3, random_state=42).fit(X_train)
labels = km.labels_   # 各樣本的群集編號
```

---

[來源: ch01 | 類型: handout] ### 補充練習 2

**理論題：** 以下任務分別屬於哪種學習類型？
1. 根據歷史成交資料預測股價
2. 將數百萬篇新聞自動歸類為「財經」、「體育」、「政治」
3. 訓練機器人學習行走

**實作題：** 使用 `sklearn.datasets.make_blobs(n_samples=300, centers=4)` 產生資料，套用 `KMeans(n_clusters=4)` 聚類，繪製聚類結果並標示群心（`cluster_centers_`）。

---

---

[來源: ch01 | 類型: handout] ### 理論背景

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

---

[來源: ch01 | 類型: handout] 偏差}^2 + \text{變異數} + \text{不可減少的雜訊}$$

**測試集的黃金法則**：

```
只能用訓練集調整模型 → 用驗證集（或 CV）選模型 → 最後只碰一次測試集
```

過多次使用測試集會造成「資訊洩漏」，讓測試集指標過於樂觀。

---

[來源: ch01 | 類型: handout] ### 核心代碼

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

---

[來源: ch01 | 類型: handout] # 建立具正則化的 Pipeline
ridge_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=10, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))  # alpha 越大正則化越強
])

---

[來源: ch01 | 類型: handout] # 使用交叉驗證評估泛化能力（比單一 val set 更穩健）
scores = cross_val_score(ridge_pipeline, X_train, y_train,
                         cv=5, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-scores)
print(f"CV RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")
```

---

[來源: ch01 | 類型: handout] ### 補充練習 3

**理論題：** 一個模型在訓練集的準確率為 99%，在測試集為 70%，這是過擬合還是欠擬合？應採取什麼策略改善？

**實作題：** 使用 `sklearn.datasets.make_regression(n_samples=100, noise=20, random_state=42)` 建立資料，分別訓練 `PolynomialFeatures(degree=1)`、`degree=5`、`degree=20` 的模型，比較三者的訓練集 RMSE 與交叉驗證 RMSE，觀察過擬合現象。

---

---

[來源: ch01 | 類型: handout] ### 理論背景

**Hold-out 驗證**：

$$\text{訓練集} : \text{驗證集} : \text{測試集} \approx 70\% : 15\% : 15\%$$

**交叉驗證 (k-fold CV)**：將訓練集切成 $k$ 份，輪流用一份作驗證，其餘訓練，最後平均 $k$ 次的結果。計算量是 Hold-out 的 $k$ 倍，但評估更穩健。

**No Free Lunch (NFL) 定理**：不存在對所有問題都最好的模型；必須在資料上實驗驗證，而非憑直覺選擇。

---

[來源: ch01 | 類型: handout] ### 核心代碼

```python
from sklearn.model_selection import StratifiedShuffleSplit

---

[來源: ch01 | 類型: handout] # 分層抽樣——確保訓練/測試集有相同的類別比例
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X, y_cat):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

---

[來源: ch01 | 類型: handout] ### 補充練習 4

**理論題：** 解釋「資料洩漏 (Data Leakage)」的概念，並舉一個在特徵工程中可能發生洩漏的具體案例。

**實作題：** 使用 `StratifiedShuffleSplit` 對 `sklearn.datasets.load_iris()` 進行分層切分，確認訓練集和測試集中各類別的比例與原始資料集一致（用 `pd.value_counts()` 驗證）。

---

---

[來源: ch01 | 類型: handout] ## 結論

本章建立了機器學習的整體知識地圖：

- 機器學習用**資料驅動的方式**取代手寫規則
- 學習類型沿**標籤、增量、通用化**三個維度分類
- 訓練的最大挑戰是在**偏差與變異數之間取得平衡**
- 嚴格的**訓練/驗證/測試集分離**是誠實評估的基石

下一章（Ch02）將把這些概念具體化，帶領你完成一個從資料取得到模型部署的完整端到端專案。

---

---

[來源: ch01 | 類型: handout] ## 課後作業

**作業一：模型比較實驗**

使用 `lifesat.csv` 資料集，比較以下三種模型在 5-fold 交叉驗證下的 RMSE：

1. `LinearRegression`
2. `KNeighborsRegressor(n_neighbors=3)`
3. `KNeighborsRegressor(n_neighbors=10)`

撰寫結論：哪個模型最佳？為什麼？調整 `n_neighbors` 對結果有何影響？

**作業二：思考題**

假設你正在建立一個信用評分模型，訓練資料中有欄位「是否逾期還款（目前）」。這個欄位能放入特徵嗎？說明你的理由（提示：想想部署時是否能取得此資訊）。

---

[來源: ch01 | 類型: tutorial] [標題: 機器學習全景導覽：監督、非監督與強化學習實戰入門 | 描述: 完整介紹機器學習的分類體系、核心術語、訓練流程與常見挑戰。用 Scikit-Learn 實作線性回歸、K-Means 和 Pipeline，適合 ML 初學者到進階開發者。 | 關鍵字: Python, 機器學習, 監督學習, 非監督學習, Scikit-Learn, 資料科學, 教學, 人工智慧]
# 機器學習全景導覽：系統性學習 ML 的第一步

機器學習（Machine Learning）是人工智慧的核心技術，讓電腦從資料中學習，而非靠人工撰寫每一條規則。本教學帶您系統性地了解 ML 的分類體系、核心術語和常見挑戰，並搭配 Scikit-Learn 實作，從第一行程式碼開始建立扎實的 ML 基礎。

---

[來源: ch01 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- 機器學習依**是否有標籤**分為監督、非監督、半監督學習
- 依**訓練時機**分為批次（離線）學習與線上（增量）學習
- 依**學習方法**分為基於實例（相似度比較）與基於模型（參數學習）
- **過擬合（Overfitting）** 是 ML 最常見的挑戰之一，正則化（Regularization）是主要解法
- 使用**交叉驗證（Cross-Validation）** 才能得到可靠的效能評估

---

---

[來源: ch01 | 類型: tutorial] ## 什麼是機器學習？

💡 **實際應用情境：** 垃圾郵件過濾器是最典型的 ML 應用。傳統方式需要人工列出所有關鍵詞規則；而 ML 方式是讓程式從大量標記過的郵件中**自動學習**規則，並隨著新垃圾郵件的出現持續進化。

機器學習（Machine Learning）是讓電腦從**經驗（資料）** 中學習，在**任務（Task）** 上表現越來越好的技術，由 Arthur Samuel（1959）提出。

核心三要素：

- **任務 T**：例如分類郵件是否為垃圾
- **經驗 E**：例如大量已標記的郵件
- **效能衡量 P**：例如分類準確率

---

[來源: ch01 | 類型: tutorial] ### 範例 1: 基礎環境設定

```python
import sys
from packaging import version
import sklearn
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch01 | 類型: tutorial] # 確認 Python 版本 ≥ 3.7
assert sys.version_info >= (3, 7), "Python 版本不符合要求"

---

[來源: ch01 | 類型: tutorial] # 確認 Scikit-Learn ≥ 1.0.1
assert version.parse(sklearn.__version__) >= version.parse("1.0.1"), "Scikit-Learn 版本不符"

---

[來源: ch01 | 類型: tutorial] # 設定 Matplotlib 預設樣式
plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
print(f"Python {sys.version_info.major}.{sys.version_info.minor}, Sklearn {sklearn.__version__}")
```

**✅ 程式碼逐行解析：**

1. `assert sys.version_info >= (3, 7)`: 元組比較，確認 Python 3.7+
2. `version.parse(...)`: 將版本字串轉為可比較物件，避免字串比較出錯（如 `"1.9" > "1.10"` 錯誤結果）
3. `plt.rc(...)`: 全域設定繪圖參數，確保圖表風格一致

**🎯 重點摘要:**

- **核心功能**: 建立可重現的開發環境
- **潛在問題**: 版本不符可能導致 API 差異錯誤
- **最佳使用情境**: 每個 ML 專案或 Notebook 的第一個 Cell

---

---

[來源: ch01 | 類型: tutorial] ## 監督式學習 (Supervised Learning)

💡 **實際應用情境：** 台灣的信用評分模型即為典型監督學習——銀行有歷史客戶的還款記錄（標籤），用這些資料訓練模型預測新客戶的違約風險。

監督式學習使用帶有**標籤（Label）** 的訓練資料，學習從輸入 X 到輸出 y 的映射。

| 任務類型 | 輸出 y | 典型演算法 | 範例 |
|----------|--------|-----------|------|
| 分類（Classification） | 離散類別 | SVM、Random Forest | 垃圾郵件偵測 |
| 迴歸（Regression） | 連續數值 | Linear Regression、SVR | 房價預測 |

---

[來源: ch01 | 類型: tutorial] ### 範例 2: 線性回歸基礎示範

```python
import numpy as np
from sklearn.linear_model import LinearRegression

---

[來源: ch01 | 類型: tutorial] # 生成模擬資料：房屋面積（坪）→ 房價（萬元）
np.random.seed(42)
X = np.random.rand(100, 1) * 50 + 10   # 面積：10~60 坪
y = 35 * X.ravel() + np.random.randn(100) * 100  # 每坪 35 萬 + 雜訊

---

[來源: ch01 | 類型: tutorial] # 訓練線性回歸模型
model = LinearRegression()
model.fit(X, y)

---

[來源: ch01 | 類型: tutorial] # 預測新樣本
X_new = np.array([[30], [45]])  # 預測 30 坪和 45 坪的房價
y_pred = model.predict(X_new)

print(f"模型係數（每坪單價）: {model.coef_[0]:.1f} 萬元/坪")
print(f"模型截距: {model.intercept_:.1f} 萬元")
print(f"30 坪預測房價: {y_pred[0]:.0f} 萬元")
print(f"45 坪預測房價: {y_pred[1]:.0f} 萬元")
```

**✅ 程式碼逐行解析：**

1. `np.random.rand(100, 1) * 50 + 10`: 生成 100 個 [10, 60] 範圍內的面積值，形狀 `(100, 1)` 符合 Scikit-Learn 輸入格式
2. `LinearRegression().fit(X, y)`: 用正規方程式 $\hat{\theta} = (X^T X)^{-1} X^T y$ 求解最優係數
3. `model.coef_[0]`: 學到的斜率（每坪單價估計值）
4. `model.predict(X_new)`: 對新樣本進行預測

**🎯 重點摘要:**

---

[來源: ch01 | 類型: tutorial] } X^T y$ 求解最優係數
3. `model.coef_[0]`: 學到的斜率（每坪單價估計值）
4. `model.predict(X_new)`: 對新樣本進行預測

**🎯 重點摘要:**

- **核心功能**: 從歷史資料學習 X→y 的線性映射
- **潛在問題**: 若 X 與 y 非線性關係，線性回歸表現不佳
- **最佳使用情境**: 特徵與目標有線性關係且資料量適中

---

---

[來源: ch01 | 類型: tutorial] ## 非監督式學習 (Unsupervised Learning)

💡 **實際應用情境：** 電商平台對消費者分群（Customer Segmentation）——沒有「正確答案」，讓演算法從消費行為資料中自動找出有意義的族群，再針對不同族群設計行銷策略。

非監督學習**沒有標籤**，演算法從原始資料中自行發現結構和模式。

主要技術：

- **聚類（Clustering）**：K-Means、DBSCAN
- **降維（Dimensionality Reduction）**：PCA、t-SNE、UMAP
- **密度估計（Density Estimation）**：GMM
- **異常偵測（Anomaly Detection）**：Isolation Forest

---

[來源: ch01 | 類型: tutorial] ### 範例 3: K-Means 客群分析

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

---

[來源: ch01 | 類型: tutorial] # 模擬消費者資料：購買頻率（次/月）× 平均消費金額（百元）
X_customers = np.r_[
    np.random.randn(100, 2) * 0.5 + [2, 5],   # 低頻低消費族群
    np.random.randn(100, 2) * 0.5 + [8, 15],  # 高頻高消費族群
    np.random.randn(100, 2) * 0.5 + [5, 8],   # 中頻中消費族群
]

---

[來源: ch01 | 類型: tutorial] # K-Means 分成 3 群
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_customers)

print(f"各群中心: {kmeans.cluster_centers_.round(2)}")
print(f"慣性（Inertia）: {kmeans.inertia_:.2f}")  # 越小越緊密
```

**✅ 程式碼逐行解析：**

1. `np.r_[...]`: 垂直串接多個陣列，建立混合群集的模擬資料
2. `KMeans(n_clusters=3, n_init=10)`: 指定分 3 群，執行 10 次隨機初始化取最佳結果（避免局部最優）
3. `kmeans.inertia_`: 所有樣本到其群中心距離的平方和，評估群集品質

**🎯 重點摘要:**

- **核心功能**: 在無標籤情況下發現資料中的自然分組
- **潛在問題**: K-Means 對異常值敏感，群數 K 需提前指定
- **最佳使用情境**: 客群分析、圖片色彩量化、文件分群

---

---

[來源: ch01 | 類型: tutorial] ### 半監督學習 (Semi-supervised Learning)

💡 **實際應用情境：** 醫學影像標注成本高昂（需要醫師判讀），但收集大量未標注影像相對容易。半監督學習利用**少量標注資料 + 大量未標注資料**，在資料稀缺時仍能訓練出優秀模型。

- 先用非監督學習（如聚類）發現資料結構
- 再用少量標籤「解釋」每個群的含義
- 典型應用：Google Photos 的人臉辨識、語音辨識

---

[來源: ch01 | 類型: tutorial] ### 強化學習 (Reinforcement Learning)

💡 **實際應用情境：** AlphaGo、ChatGPT 的 RLHF 訓練都使用強化學習。Agent 在環境中採取動作、獲得獎勵，透過無數次嘗試和錯誤找到最優策略。

核心概念：**Agent**（智能體）→ **Action**（動作）→ **Environment**（環境）→ **Reward**（獎勵）→ **Policy**（策略）

---

---

[來源: ch01 | 類型: tutorial] ### 批次學習 (Batch Learning)

- 使用全部可用資料**離線訓練一次**
- 若有新資料需**重新訓練整個模型**
- 適合：資料量固定、模型無需頻繁更新

---

[來源: ch01 | 類型: tutorial] ### 線上學習 (Online Learning)

```python
from sklearn.linear_model import SGDRegressor

---

[來源: ch01 | 類型: tutorial] # SGD 迴歸支援增量學習（partial_fit）
model_online = SGDRegressor(random_state=42, max_iter=1)

---

[來源: ch01 | 類型: tutorial] # 模擬串流資料：每次收到一批新資料
for batch_start in range(0, len(X), 10):
    X_batch = X[batch_start:batch_start+10]
    y_batch = y[batch_start:batch_start+10]
    model_online.partial_fit(X_batch, y_batch)  # 增量更新，不需重新訓練

print(f"線上學習最終係數: {model_online.coef_[0]:.1f}")
```

**✅ 程式碼逐行解析：**

1. `SGDRegressor(max_iter=1)`: 每次只訓練 1 個 epoch，適合增量更新
2. `partial_fit(X_batch, y_batch)`: 核心方法——用新批次資料更新模型，不破壞已學到的知識

**🎯 重點摘要:**

- **核心功能**: 讓模型在新資料到達時即時更新
- **潛在問題**: 若新資料品質差（「資料毒化」攻擊），模型效能可能迅速退化
- **最佳使用情境**: 串流資料（股價、感測器）、記憶體有限的設備

---

---

[來源: ch01 | 類型: tutorial] ### 過擬合 (Overfitting) vs 欠擬合 (Underfitting)

| 問題 | 症狀 | 解法 |
|------|------|------|
| 過擬合（模型太複雜） | 訓練集準確率高，測試集低 | 正則化、更多資料、Dropout |
| 欠擬合（模型太簡單） | 訓練集和測試集準確率都低 | 更複雜模型、更多特徵 |

---

[來源: ch01 | 類型: tutorial] ### 範例 4: 學習曲線診斷過擬合/欠擬合

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge

---

[來源: ch01 | 類型: tutorial] # 高次多項式 → 過擬合
overfitting_model = make_pipeline(
    PolynomialFeatures(degree=20),  # 20 次多項式，幾乎記住訓練資料
    LinearRegression()
)

---

[來源: ch01 | 類型: tutorial] # 加入 Ridge 正則化 → 緩解過擬合
regularized_model = make_pipeline(
    PolynomialFeatures(degree=20),
    Ridge(alpha=10)  # alpha 越大，正則化越強
)

---

[來源: ch01 | 類型: tutorial] # 解法：增加更多訓練資料或降低模型複雜度
```

**🎯 重點摘要:**

- **核心功能**: 識別並解決過擬合與欠擬合問題
- **重要原則**: 在驗證集上評估，而非訓練集
- **最佳使用情境**: 任何 ML 模型訓練後的必要診斷步驟

---

---

[來源: ch01 | 類型: tutorial] ### 範例 5: 交叉驗證最佳實踐

```python
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

---

[來源: ch01 | 類型: tutorial] # 切分資料（先保留測試集，不碰它！）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

---

[來源: ch01 | 類型: tutorial] # 建立包含預處理的 Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # 特徵標準化（μ=0, σ=1）
    ("model", Ridge(alpha=1.0))
])

---

[來源: ch01 | 類型: tutorial] # 5-Fold 交叉驗證（在訓練集上）
cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"  # 使用 RMSE
)
print(f"CV RMSE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

---

[來源: ch01 | 類型: tutorial] # 最終評估（只在最後用測試集一次）
pipeline.fit(X_train, y_train)
test_rmse = pipeline.score(X_test, y_test)
print(f"測試集 R²: {test_rmse:.3f}")
```

**✅ 程式碼逐行解析：**

1. `train_test_split(..., random_state=42)`: 固定隨機種子確保可重現性，`test_size=0.2` 保留 20% 為測試集
2. `Pipeline([...])`: 將預處理和模型串接，防止資料洩漏（Data Leakage）
3. `cross_val_score(cv=5)`: 將訓練集分為 5 份輪流作為驗證集，得到更穩健的效能估計
4. `scoring="neg_root_mean_squared_error"`: Scikit-Learn 遵循「越高越好」慣例，故 RMSE 前加負號

**🎯 重點摘要:**

- **核心功能**: 使用不偏的方式評估模型泛化能力
- **重要原則**: 測試集只能在最終評估時使用**一次**
- **最佳使用情境**: 所有有監督學習模型的標準評估流程

---

---

[來源: ch01 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: 監督學習與非監督學習如何選擇？**

A: 關鍵問題是「是否有標籤資料」。若業務問題有明確的輸入-輸出對應（如信用評分），用監督學習。若只有原始資料且目標是「發現結構」（如客群分析），用非監督學習。

**Q2: 為什麼要用 Pipeline 而非手動呼叫 `fit_transform`？**

A: Pipeline 防止**資料洩漏（Data Leakage）**。若手動在訓練集上 `fit_transform` 後再分割，測試集的統計資訊（如均值、標準差）已污染了 Scaler，導致驗證結果過於樂觀。

**Q3: 過擬合和高偏差/高方差的關係是什麼？**

A: 過擬合對應**高方差（High Variance）**——模型對訓練資料的微小變化過度敏感；欠擬合對應**高偏差（High Bias）**——模型假設太簡單，無法捕捉真實規律。這兩者的取捨稱為**偏差-方差權衡（Bias-Variance Tradeoff）**。

**Q4: 什麼時候用 `cross_val_score` vs 保留驗證集？**

A: 資料量充足時兩者皆可；資料量少時，交叉驗證能充分利用每個樣本，是更好的選擇。注意：無論如何，測試集（最終評估集）應與這個過程完全隔離。

---

---

[來源: ch01 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #機器學習 #監督學習 #非監督學習 #Scikit學習 #資料科學 #AI入門 #程式設計 #深度學習 #技術分享 #MachineLearning #DataScience #學習筆記 #軟體工程

---

[來源: ch02] [標題: 🐍 端到端機器學習專案實戰：從數據到模型的完整指南 | 描述: 本教學將引導您完成一個完整的端到端機器學習專案。我們將從獲取真實世界的房價數據開始，一路進行數據探索、預處理、模型訓練、微調，最終部署一個能夠預測房價的模型。這是一個專為初學者到中級開發者設計的實戰演練。 | 關鍵字: Python, 機器學習, Scikit-Learn, Pandas, 數據科學, 端到端專案, 房價預測, 數據分析, 模型訓練]
# 端到端機器學習專案實戰

歡迎來到您的第二個機器學習專案！本章節將引導您完成一個完整的端到端（End-to-End）專案，模擬真實世界中數據科學家的工作流程。我們將使用加州房價數據集，從數據獲取、探索、準備，到模型選擇、訓練、微調，最終評估我們的模型。

---

[來源: ch02] ## 關鍵重點 (Key Takeaways)
- **完整流程**: 學習一個典型機器學習專案從頭到尾的完整步驟。
- **數據處理**: 掌握使用 Pandas 進行數據清洗、處理缺失值、特徵縮放和轉換的關鍵技術。
- **模型訓練**: 了解如何使用 Scikit-Learn 訓練多種回歸模型，如線性回歸、決策樹和隨機森林。
- **模型評估**: 學習使用交叉驗證（Cross-Validation）來更穩健地評估模型性能。
- **超參數調優**: 探索網格搜索（Grid Search）和隨機搜索（Randomized Search）等方法來找到模型的最佳超參數。

---

[來源: ch02] ## 專案設定與環境準備
💡 **實際應用情境：** 在任何專案開始之前，確保您的開發環境已經準備就緒是至關重要的第一步。這包括安裝必要的 Python 版本、函式庫，並設定好專案的工作目錄。一個穩定且一致的環境可以避免許多不必要的問題。

---

[來源: ch02] # Python ≥3.7 is required
import sys
assert sys.version_info >= (3, 7)

---

[來源: ch02] # Scikit-Learn ≥1.0.1 is required
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

---

[來源: ch02] # To plot pretty figures
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

---

[來源: ch02] ze=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**
1.  `第 2-3 行`: 檢查並斷言（assert）當前的 Python 版本是否大於等於 3.7。如果版本不符，程式將會中斷並報錯，這確保了程式碼在兼容的環境中運行。
2.  `第 6-7 行`: 同樣地，檢查 Scikit-Learn 函式庫的版本是否大於等於 1.0.1。這是為了確保我們能使用到特定版本之後的新功能或修復。
3.  `第 10-16 行`: 設定 Matplotlib 繪圖的預設樣式。`%matplotlib inline` 是一個 Jupyter Notebook 的魔法指令，它讓繪圖結果直接顯示在儲存格下方。後續的 `plt.rc` 指令則是為了讓圖表有更好的一致性與可讀性，統一設定了字體大小、座標軸標籤大小等。

---

[來源: ch02] e` 是一個 Jupyter Notebook 的魔法指令，它讓繪圖結果直接顯示在儲存格下方。後續的 `plt.rc` 指令則是為了讓圖表有更好的一致性與可讀性，統一設定了字體大小、座標軸標籤大小等。

**🎯 重點摘要:**
- **核心功能**: 建立一個穩定且版本一致的開發環境，並設定好繪圖的基礎樣式。
- **潛在問題**: 如果環境中的函式庫版本不符，可能會導致程式碼無法執行或出現預期外的錯誤。
- **最佳使用情境**: 在任何 Python 專案或 Notebook 的開頭進行此類設定，以確保可重現性（reproducibility）。

---

[來源: ch02] ## 獲取數據
💡 **實際應用情境：** 機器學習專案的第一步通常是獲取數據。數據可以來自資料庫、API、或是像本範例一樣的壓縮檔。我們需要編寫函式來自動化下載和解壓縮的過程，這樣不僅方便自己，也方便他人重現我們的專案。

---

[來源: ch02] ### 範例 2: 下載並載入數據
```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    # 設定數據檔案的路徑
    tarball_path = Path("datasets/housing.tgz")

# 檢查檔案是否已存在，避免重複下載
    if not tarball_path.is_file():
        # 建立資料夾結構，parents=True 允許建立多層目錄
        Path("datasets").mkdir(parents=True, exist_ok=True)

---

[來源: ch02] # 建立資料夾結構，parents=True 允許建立多層目錄
        Path("datasets").mkdir(parents=True, exist_ok=True)

# 指定數據來源 URL
        url = "https://github.com/ageron/handson-ml3/raw/main/datasets/housing.tgz"

# 下載檔案到本地
        urllib.request.urlretrieve(url, tarball_path)

# 解壓縮 tar.gz 檔案
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")

---

[來源: ch02] rfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")

# 讀取解壓縮後的 CSV 檔案並回傳 DataFrame
    return pd.read_csv(Path("datasets/housing/housing.csv"))

---

[來源: ch02] # 執行函式獲取數據
housing = load_housing_data()
```

**✅ 程式碼逐行解析：**
1.  `第 6 行`: 定義一個名為 `load_housing_data` 的函式，將獲取數據的邏輯封裝起來。
2.  `第 7-13 行`: 檢查 `datasets/housing.tgz` 這個壓縮檔是否存在。如果不存在，就建立 `datasets` 資料夾，然後從指定的 URL 下載檔案，並使用 `tarfile` 函式庫將其解壓縮。
3.  `第 14 行`: 使用 Pandas 的 `read_csv` 函式讀取解壓縮後的 `housing.csv` 檔案，並將其作為一個 DataFrame 回傳。
4.  `第 16 行`: 呼叫 `load_housing_data()` 函式，將回傳的 DataFrame 存儲在 `housing` 變數中。

---

[來源: ch02] 檔案，並將其作為一個 DataFrame 回傳。
4.  `第 16 行`: 呼叫 `load_housing_data()` 函式，將回傳的 DataFrame 存儲在 `housing` 變數中。

**🎯 重點摘要:**
- **核心功能**: 自動化數據獲取流程，包括下載、解壓縮和載入。
- **潛在問題**: 如果網路連線失敗或 URL 失效，下載過程會失敗。函式庫 `pathlib` 提供了現代化的檔案路徑操作方式，比傳統的 `os.path` 更直觀。
- **最佳使用情境**: 當您需要分發您的專案，並希望其他人能夠輕鬆地獲取所需數據時，這種自動化腳本非常有用。
- **故障排除**: 如果下載失敗，請檢查網路連線或嘗試手動下載檔案。相關的數據探索技巧請參考[範例 3](#explore-data)。

---

[來源: ch02] ### 範例 3: 數據初步探索

```python
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
```

---

[來源: ch02] ython
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: `housing.head()` 顯示 DataFrame 的前五行，讓我們可以快速一覽數據的樣貌和欄位。
2.  `第 2 行`: `housing.info()` 提供 DataFrame 的簡潔摘要，包括每個欄位的名稱、非空值的數量以及數據類型（Dtype）。這對於快速發現缺失值和了解數據結構至關重要。
3.  `第 3 行`: `housing["ocean_proximity"].value_counts()` 針對 `ocean_proximity` 這個類別欄位，計算每個類別的出現次數。
4.  `第 4 行`: `housing.describe()` 針對數值型欄位，產生描述性統計數據，如計數、平均值、標準差、最小值、最大值以及百分位數。

---

[來源: ch02] ` 這個類別欄位，計算每個類別的出現次數。
4.  `第 4 行`: `housing.describe()` 針對數值型欄位，產生描述性統計數據，如計數、平均值、標準差、最小值、最大值以及百分位數。

**🎯 重點摘要:**
- **核心功能**: 快速了解數據集的整體情況，包括大小、欄位、數據類型、缺失值和基本統計特性。
- **潛在問題**: `info()` 顯示的非空值數量可以幫助我們快速識別哪些欄位有缺失數據。`describe()` 只對數值欄位有效。
- **最佳使用情境**: 在數據載入後，立即進行這些初步探索，是數據分析的標準起手式。

---

[來源: ch02] ### 範例 4: 繪製數據分佈直方圖

```python
housing.hist(bins=50, figsize=(12, 8))
plt.show()
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: `housing.hist()` 是 Pandas DataFrame 內建的繪圖方法，可以為所有數值型欄位繪製直方圖（histogram）。
2.  `bins=50`: 指定直方圖的條數為 50。越多的條數可以讓我們更細緻地觀察數據分佈，但太多也可能引入雜訊。
3.  `figsize=(12, 8)`: 設定整個圖表的大小為 12x8 英吋。
4.  `plt.show()`: 顯示圖表。

---

[來源: ch02] 。越多的條數可以讓我們更細緻地觀察數據分佈，但太多也可能引入雜訊。
3.  `figsize=(12, 8)`: 設定整個圖表的大小為 12x8 英吋。
4.  `plt.show()`: 顯示圖表。

**🎯 重點摘要:**
- **核心功能**: 可視化各個數值特徵的數據分佈。
- **潛在問題**: 從直方圖中，我們可以觀察到特徵的尺度（scale）差異很大，例如 `median_house_value` 的值遠大於 `housing_median_age`。我們也看到某些特徵有著「長尾」分佈，或是被限制在某個最大/最小值（如 `median_house_value`）。
- **最佳使用情境**: 在數據探索階段，使用直方圖來了解特徵的分佈、範圍、偏態（skewness）以及是否存在異常值。

---

[來源: ch02] ## 探索與可視化數據
💡 **實際應用情境：** 在深入模型訓練之前，花時間探索數據是非常有價值的。透過地理位置繪圖，我們可以將經緯度數據與房價等其他屬性結合，從而發現數據中的地理模式或群聚現象。

---

[來源: ch02] ### 範例 5: 地理位置散點圖

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()
```

---

[來源: ch02] , cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: 使用 DataFrame 的 `plot` 方法，設定 `kind="scatter"` 來繪製散點圖。`x` 和 `y` 軸分別對應經度和緯度。
2.  `s=housing["population"] / 100`: 設定每個點的大小（size）與該地區的人口數成正比。除以 100 是為了縮放點的大小，使其在圖上更合適。
3.  `c="median_house_value"`: 設定每個點的顏色（color）對應於房價中位數。
4.  `cmap="jet"`: 使用名為 "jet" 的顏色映射（colormap），從藍色（低房價）到紅色（高房價）進行漸變。
5.  `colorbar=True`: 顯示一個顏色條，標示顏色與數值的對應關係。

---

[來源: ch02] ="jet"`: 使用名為 "jet" 的顏色映射（colormap），從藍色（低房價）到紅色（高房價）進行漸變。
5.  `colorbar=True`: 顯示一個顏色條，標示顏色與數值的對應關係。

**🎯 重點摘要:**
- **核心功能**: 將地理數據與其他特徵（人口、房價）結合，進行多維度可視化。
- **潛在問題**: 這張圖清楚地顯示了房價與地理位置（特別是沿海地區）以及人口密度有很強的關聯性。
- **最佳使用情境**: 當數據集包含地理資訊（如經緯度）時，繪製此類圖表有助於發現空間上的關聯性。

---

[來源: ch02] ## 特徵工程與分層抽樣
💡 **實際應用情境：** 在建立測試集時，如果我們使用純隨機抽樣，可能會因為偶然性導致測試集無法代表整體數據的分佈，特別是在重要特徵上。這稱為抽樣偏差（sampling bias）。為了解決這個問題，我們使用**分層抽樣（Stratified Sampling）**。我們根據一個重要的特徵（如此處的收入中位數）將數據分成幾個階層（strata），然後從每個階層中抽取等比例的樣本來組成測試集，以確保測試集能反映整體的特徵分佈。

---

[來源: ch02] ### 範例 6: 建立收入類別以進行分層抽樣
為了進行分層抽樣，我們首先需要將連續的收入中位數特徵轉換為類別特徵。

```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```

---

[來源: ch02] bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: 我們使用 `pd.cut` 函式將 `median_income` 這個連續型特徵切分成數個區間（bins）。
2.  `bins=[...]`: 這個列表定義了區間的邊界。例如，`0.` 到 `1.5` 是一個區間。`np.inf` 代表正無窮大，確保所有高於 6.0 的值都被包含在最後一個區間。
3.  `labels=[1, 2, 3, 4, 5]`: 為每個區間指定一個標籤。所有收入在 0 到 1.5 之間的地區都會被標記為 1，依此類推。
4.  結果會被存儲在一個新的 `income_cat` 欄位中。

---

[來源: ch02] ls=[1, 2, 3, 4, 5]`: 為每個區間指定一個標籤。所有收入在 0 到 1.5 之間的地區都會被標記為 1，依此類推。
4.  結果會被存儲在一個新的 `income_cat` 欄位中。

**🎯 重點摘要:**
- **核心功能**: 將連續特徵離散化（discretize），為分層抽樣做準備。
- **潛在問題**: 區間的選擇是主觀的，需要對數據有一定了解。不合理的切分可能會影響抽樣的代表性。
- **最佳使用情境**: 當某個連續特徵對目標變數有顯著影響，且希望在訓練集和測試集中保持其分佈一致時。

---

[來源: ch02] ### 範例 7: 使用 Scikit-Learn 進行分層抽樣
現在我們可以使用這個新的類別來進行分層抽樣。

```python
from sklearn.model_selection import train_test_split

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
```

---

[來源: ch02] = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
```

**✅ 程式碼逐行解析：**
1.  `第 3 行`: 我們使用 Scikit-Learn 的 `train_test_split` 函式。
2.  `stratify=housing["income_cat"]`: 這是分層抽樣的關鍵。它告訴函式在切分數據時，要確保訓練集和測試集中的 `income_cat` 欄位的類別比例與原始數據集中的比例相同。
3.  `test_size=0.2`: 指定測試集應佔總數據的 20%。
4.  `random_state=42`: 確保每次執行的切分結果都一樣，以利於重現。

---

[來源: ch02] 類別比例與原始數據集中的比例相同。
3.  `test_size=0.2`: 指定測試集應佔總數據的 20%。
4.  `random_state=42`: 確保每次執行的切分結果都一樣，以利於重現。

**🎯 重點摘要:**
- **核心功能**: 根據指定特徵（`income_cat`）的分佈比例來切分數據，確保測試集的代表性。
- **最佳使用情境**: 在任何監督式學習任務中，建立訓練集和測試集時都推薦使用分層抽樣，特別是當數據集不大或類別不平衡時。

---

[來源: ch02] ### 範例 8: 比較不同抽樣方法的誤差
讓我們來看看分層抽樣與純隨機抽樣相比，在收入類別分佈上的表現。

---

[來源: ch02] ```python
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)
(compare_props * 100).round(2)
```

---

[來源: ch02] compare_props["Overall %"] - 1)
(compare_props * 100).round(2)
```

**✅ 程式碼逐行解析：**
1.  `第 1-2 行`: 定義一個輔助函式來計算每個收入類別的比例。
2.  `第 4 行`: 產生一個純隨機抽樣的測試集作為對比。
3.  `第 6-14 行`: 建立一個 DataFrame 來比較三種分佈：整體數據、分層抽樣測試集、隨機抽樣測試集。並計算分層抽樣和隨機抽樣相對於整體數據的誤差百分比。
4.  `第 15 行`: 顯示結果，乘以 100 並四捨五入到小數點後兩位，方便閱讀。

---

[來源: ch02] 來比較三種分佈：整體數據、分層抽樣測試集、隨機抽樣測試集。並計算分層抽樣和隨機抽樣相對於整體數據的誤差百分比。
4.  `第 15 行`: 顯示結果，乘以 100 並四捨五入到小數點後兩位，方便閱讀。

**🎯 重點摘要:**
- **核心功能**: 量化比較不同抽樣方法所產生的抽樣偏差。
- **結論**: 從結果中可以看出，分層抽樣（Stratified）的誤差遠小於純隨機抽樣（Random），證明了其在維持數據分佈代表性上的優越性。

---

[來源: ch02] ### 範例 9: 清理數據
在完成抽樣後，我們應該移除臨時創建的 `income_cat` 欄位，以免它干擾後續的模型訓練。

```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: 遍歷訓練集和測試集。
2.  `第 2 行`: 從每個數據集中移除 `income_cat` 欄位。`axis=1` 表示我們要移除的是欄位，`inplace=True` 表示直接在原始 DataFrame 上進行修改。

---

[來源: ch02] 2.  `第 2 行`: 從每個數據集中移除 `income_cat` 欄位。`axis=1` 表示我們要移除的是欄位，`inplace=True` 表示直接在原始 DataFrame 上進行修改。

**🎯 重點摘要:**
- **核心功能**: 清理輔助特徵，避免數據洩漏。
- **最佳實踐**: 用於輔助數據處理（如分層抽樣）的臨時特徵，在完成其任務後應立即移除。

---

[來源: ch02] ## 數據前處理
💡 **實際應用情境：** 真實世界的數據很少是完美的。它們通常包含缺失值、異常值，或者其格式不適合直接用於機器學習模型。數據前處理是整個專案中至關重要且通常最耗時的階段。Scikit-Learn 的 `Pipeline` 和 `ColumnTransformer` 是處理這個階段的強大工具。

---

[來源: ch02] ### 範例 10: 處理缺失值
```python
from sklearn.impute import SimpleImputer

---

[來源: ch02] # 建立填補器，策略設為中位數（對異常值較不敏感）
imputer = SimpleImputer(strategy="median")

---

[來源: ch02] # 只選擇數值型欄位進行處理
housing_num = housing.select_dtypes(include=[np.number])

---

[來源: ch02] # 學習每個欄位的中位數（fit 階段）
imputer.fit(housing_num)

---

[來源: ch02] # 使用學習到的中位數填補缺失值（transform 階段）
X = imputer.transform(housing_num)
```

---

[來源: ch02] **✅ 程式碼逐行解析：**
1.  `第 2 行`: 從 Scikit-Learn 導入 `SimpleImputer`，這是一個用於處理缺失值的轉換器。
2.  `第 4 行`: 建立一個 `SimpleImputer` 的實例，並設定 `strategy="median"`，表示我們將用每個欄位的中位數來填補該欄位的缺失值。
3.  `第 5 行`: `select_dtypes` 方法只選擇數值型的欄位。
4.  `第 6 行`: `imputer.fit(housing_num)` 計算 `housing_num` 中每個欄位的中位數，並將結果存儲在 `imputer` 的 `statistics_` 屬性中。
5.  `第 7 行`: `imputer.transform(housing_num)` 使用計算好的中位數來填補缺失值，並回傳一個包含轉換後數據的 NumPy 陣列。

---

[來源: ch02] tistics_` 屬性中。
5.  `第 7 行`: `imputer.transform(housing_num)` 使用計算好的中位數來填補缺失值，並回傳一個包含轉換後數據的 NumPy 陣列。

**🎯 重點摘要:**
- **核心功能**: 使用特徵的中位數來系統性地填補數據集中的缺失值。
- **潛在問題**: 選擇 "median" 策略對處理有異常值的數據比較穩健。其他策略還有 "mean"（平均值）和 "most_frequent"（眾數）。
- **最佳使用情境**: 當數值特徵存在缺失值時，`SimpleImputer` 提供了一個簡單而有效的解決方案。

---

[來源: ch02] ### 範例 11: 處理文本與類別屬性
```python
from sklearn.preprocessing import OneHotEncoder

---

[來源: ch02] # 建立獨熱編碼器，sparse_output=False 確保輸出為密集陣列
cat_encoder = OneHotEncoder(sparse_output=False)

---

[來源: ch02] # 注意: 需要用雙層中括號 [[]] 來保持 DataFrame 格式
housing_cat_1hot = cat_encoder.fit_transform(housing[["ocean_proximity"]])
```

---

[來源: ch02] [[]] 來保持 DataFrame 格式
housing_cat_1hot = cat_encoder.fit_transform(housing[["ocean_proximity"]])
```

**✅ 程式碼逐行解析：**
1.  `第 2 行`: 導入 `OneHotEncoder`，它用於將類別特徵轉換為數值格式。
2.  `第 4 行`: 建立 `OneHotEncoder` 的實例。`sparse_output=False` 確保輸出是一個密集的 NumPy 陣列，而不是稀疏矩陣。
3.  `第 5 行`: `fit_transform` 方法會先學習類別（`fit`），然後將類別轉換為獨熱編碼（`transform`）。例如，如果 `ocean_proximity` 有 5 個類別，那麼每個地區都會被轉換成一個長度為 5 的向量，其中只有一個元素是 1，其餘都是 0。

---

[來源: ch02] 後將類別轉換為獨熱編碼（`transform`）。例如，如果 `ocean_proximity` 有 5 個類別，那麼每個地區都會被轉換成一個長度為 5 的向量，其中只有一個元素是 1，其餘都是 0。

**🎯 重點摘要:**
- **核心功能**: 將無法直接計算的文本類別轉換為機器學習模型可以處理的數值格式。
- **潛在問題**: 獨熱編碼會產生很多新的欄位，如果一個類別特徵有非常多的類別，可能會導致維度災難。
- **最佳使用情境**: 當類別特徵的類別數量不多時，獨熱編碼是處理標稱型（Nominal）類別數據的標準方法。

---

[來源: ch02] ### 範例 12: 建立數據轉換 Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

---

[來源: ch02] # 建立數值特徵的處理流水線
num_pipeline = Pipeline([
    # 步驟1: 用中位數填補缺失值
    ('imputer', SimpleImputer(strategy="median")),
    # 步驟2: 標準化特徵（平均值=0，標準差=1）
    ('std_scaler', StandardScaler()),
])

---

[來源: ch02] # 定義欄位類型列表
num_attribs = list(housing_num)  # 數值欄位
cat_attribs = ["ocean_proximity"]  # 類別欄位

---

[來源: ch02] # 建立完整的欄位轉換器，可對不同類型欄位應用不同處理
full_pipeline = ColumnTransformer([
    # 對數值欄位應用數值處理流水線
    ("num", num_pipeline, num_attribs),
    # 對類別欄位應用獨熱編碼
    ("cat", OneHotEncoder(sparse_output=False), cat_attribs),
])

---

[來源: ch02] # 一次性應用所有轉換步驟，得到處理完成的數據
housing_prepared = full_pipeline.fit_transform(housing)
```

---

[來源: ch02] **✅ 程式碼逐行解析：**
1.  `第 5-8 行`: 建立一個名為 `num_pipeline` 的 `Pipeline`。它將一系列轉換步驟串聯起來：首先用中位數填補缺失值（`imputer`），然後進行特徵縮放（`std_scaler`）。
2.  `第 13-16 行`: 建立一個 `ColumnTransformer`。這個強大的工具可以讓我們對不同的欄位應用不同的轉換流程。這裡，我們對所有數值屬性（`num_attribs`）應用 `num_pipeline`，並對類別屬性（`cat_attribs`）應用 `OneHotEncoder`。
3.  `第 18 行`: `full_pipeline.fit_transform(housing)` 對整個 `housing` DataFrame 應用我們定義好的所有前處理步驟，回傳一個準備好用於模型訓練的 NumPy 陣列。

---

[來源: ch02] ll_pipeline.fit_transform(housing)` 對整個 `housing` DataFrame 應用我們定義好的所有前處理步驟，回傳一個準備好用於模型訓練的 NumPy 陣列。

**🎯 重點摘要:**
- **核心功能**: 將所有數據前處理步驟封裝成一個單一的轉換器，極大地簡化了工作流程。
- **潛在問題**: 確保 `num_attribs` 和 `cat_attribs` 的欄位列表是正確且沒有重疊的。
- **最佳使用情境**: 在任何需要對不同欄位進行不同處理的複雜數據準備場景中，`Pipeline` 和 `ColumnTransformer` 都是最佳實踐。
- **故障排除**: 如果遇到欄位類型錯誤，請檢查數據類型。相關的模型訓練步驟請參考[選擇與訓練模型](#select-train-model)。

---

[來源: ch02] ## 選擇與訓練模型
💡 **實際應用情境：** 在數據準備好之後，下一步就是選擇、訓練和評估模型。通常我們會從一些簡單的模型開始，建立一個基準性能，然後再嘗試更複雜的模型。

---

[來源: ch02] ### 範例 13: 訓練線性回歸模型

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing["median_house_value"])
```

**✅ 程式碼逐行解析：**
1.  `第 2 行`: 導入 `LinearRegression` 模型。
2.  `第 4 行`: 建立 `LinearRegression` 模型的實例。
3.  `第 5 行`: `fit` 方法使用準備好的數據 `housing_prepared`（特徵）和 `housing["median_house_value"]`（目標值）來訓練模型。

---

[來源: ch02] 實例。
3.  `第 5 行`: `fit` 方法使用準備好的數據 `housing_prepared`（特徵）和 `housing["median_house_value"]`（目標值）來訓練模型。

**🎯 重點摘要:**
- **核心功能**: 訓練一個基本的線性回歸模型。
- **最佳使用情境**: 作為建立性能基準的第一個模型。線性回歸速度快、易於解釋，但可能因為對數據的假設過於簡單而導致欠擬合（underfitting）。

---

[來源: ch02] ### 範例 14: 使用交叉驗證評估模型
```python
from sklearn.model_selection import cross_val_score

---

[來源: ch02] # 使用 K 折交叉驗證評估線性回歸模型
scores = cross_val_score(lin_reg, housing_prepared, housing["median_house_value"],
                         scoring="neg_mean_squared_error",  # 負的均方誤差
                         cv=10)  # 10 折交叉驗證

---

[來源: ch02] # 轉換為 RMSE（均方根誤差）- 注意要先取負號再開根號
lin_rmse_scores = np.sqrt(-scores)
```

**✅ 程式碼逐行解析：**
1.  `第 2 行`: 導入 `cross_val_score`，這是 Scikit-Learn 中用於交叉驗證的工具。
2.  `第 4-5 行`: `cross_val_score` 會自動將數據集分成 `cv=10` 折。在每一折中，它會用 9 折的數據訓練模型，並在剩下的 1 折上進行評估。`scoring="neg_mean_squared_error"` 指定了評估指標。
3.  `第 6 行`: Scikit-Learn 的評分函式通常回傳的是效用（utility，越大越好），而不是成本（cost，越小越好），所以它計算的是負的 MSE。我們需要取負號再開根號，才能得到 RMSE。

---

[來源: ch02] 6 行`: Scikit-Learn 的評分函式通常回傳的是效用（utility，越大越好），而不是成本（cost，越小越好），所以它計算的是負的 MSE。我們需要取負號再開根號，才能得到 RMSE。

**🎯 重點摘要:**
- **核心功能**: 使用 K-折交叉驗證來獲得比單純的訓練/測試集劃分更穩健的模型性能評估。
- **潛在問題**: 交叉驗證的計算成本是單次訓練的 K 倍。
- **最佳使用情境**: 在模型評估和比較階段，交叉驗證是標準做法，它可以有效避免因為某次特定的訓練/測試集劃分而導致的偶然性結果。
- **故障排除**: 如果計算時間過長，可以減少 `cv` 的值或使用較小的數據樣本。接下來的超參數調優請參考[模型微調](#fine-tune-model)。

---

[來源: ch02] ## 模型微調
💡 **實際應用情境：** 大多數機器學習模型都帶有超參數（Hyperparameters），這些是在訓練前設定的參數，例如決策樹的最大深度。找到一組好的超參數組合可以顯著提升模型性能。網格搜索和隨機搜索是兩種常用的自動化超參數調優方法。

---

[來源: ch02] ```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing["median_house_value"])
```

---

[來源: ch02] return_train_score=True)

grid_search.fit(housing_prepared, housing["median_house_value"])
```

**✅ 程式碼逐行解析：**
1.  `第 5-8 行`: 定義一個 `param_grid`，這是一個字典列表，其中包含了我們想要測試的超參數名稱和值的組合。`GridSearchCV` 將會遍歷所有可能的組合。
2.  `第 12-15 行`: 建立 `GridSearchCV` 的實例。它接收一個模型（`forest_reg`）、超參數網格（`param_grid`），並使用 `cv=5` 的交叉驗證來評估每種組合的性能。
3.  `第 17 行`: `fit` 方法會執行搜索。這是一個計算密集型的過程，因為它需要訓練 `(3*4 + 2*3) * 5 = 90` 個模型。

---

[來源: ch02] `cv=5` 的交叉驗證來評估每種組合的性能。
3.  `第 17 行`: `fit` 方法會執行搜索。這是一個計算密集型的過程，因為它需要訓練 `(3*4 + 2*3) * 5 = 90` 個模型。

**🎯 重點摘要:**
- **核心功能**: 自動化地搜索最佳的超參數組合。
- **潛在問題**: 當超參數空間很大時，網格搜索的計算成本會非常高。在這種情況下，`RandomizedSearchCV`（隨機搜索）通常是更好的選擇。
- **最佳使用情境**: 當您想要徹底地探索一個較小的超參數空間時，網格搜索非常有用。

---

[來源: ch02] ## 評估最終模型
💡 **實際應用情境：** 在經過多輪的模型選擇和微調後，我們選出了最終的模型。現在，我們需要在「從未見過」的測試集上對其進行最後一次評估，以估計它在真實世界中的泛化性能。

---

[來源: ch02] ### 範例 16: 在測試集上評估最終模型
```python

---

[來源: ch02] # 獲取網格搜索找到的最佳模型
final_model = grid_search.best_estimator_

---

[來源: ch02] # 從測試集分離特徵和目標變數
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

---

[來源: ch02] # 【關鍵】只使用 transform，不用 fit_transform

---

[來源: ch02] # 必須使用訓練時學習的轉換參數，避免數據洩漏
X_test_prepared = full_pipeline.transform(X_test)

---

[來源: ch02] # 使用最終模型進行預測
final_predictions = final_model.predict(X_test_prepared)

---

[來源: ch02] # 計算測試集上的最終性能指標
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```

---

[來源: ch02] 最終性能指標
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: 從 `grid_search` 物件中獲取性能最好的模型。
2.  `第 3-4 行`: 從測試集中分離出特徵（`X_test`）和標籤（`y_test`）。
3.  `第 6 行`: **非常重要**：這裡我們只使用 `transform` 方法，而不是 `fit_transform`。我們必須使用從訓練集中學習到的轉換規則（如中位數、縮放參數）來處理測試集，以避免數據洩漏（data leakage）。
4.  `第 8 行`: 使用最終模型對處理過的測試集進行預測。
5.  `第 10-11 行`: 計算並打印出最終的均方根誤差（RMSE）。

---

[來源: ch02] 處理測試集，以避免數據洩漏（data leakage）。
4.  `第 8 行`: 使用最終模型對處理過的測試集進行預測。
5.  `第 10-11 行`: 計算並打印出最終的均方根誤差（RMSE）。

**🎯 重點摘要:**
- **核心功能**: 在獨立的測試集上評估最終模型的性能，得到其泛化誤差的估計。
- **潛在問題**: 絕對不能在測試集上進行任何的 `fit` 操作（包括前處理和模型訓練），否則評估結果將會過於樂觀且不可信。
- **最佳使用情境**: 這是機器學習專案的最後一步，用於報告模型的最終性能。

---

[來源: ch02] ## 總結與最佳實踐
這個端到端的專案向我們展示了一個完整的機器學習工作流程。我們從定義問題開始，獲取數據，進行探索性分析，然後投入大量精力進行數據前處理。我們學習了如何使用 Scikit-Learn 的 `Pipeline` 和 `ColumnTransformer` 來建立可重用的數據準備流程。接著，我們訓練了多個模型，使用交叉驗證進行評估，並透過網格搜索來微調超參數。最後，我們在測試集上評估了最終模型的性能。

**最佳實踐：**
- **自動化**: 盡可能將數據獲取和轉換的過程編寫成函式或 `Pipeline`，以方便重現。
- **數據探索**: 不要急著訓練模型，花足夠的時間去理解你的數據。
- **交叉驗證**: 使用交叉驗證來獲得更可靠的模型性能評估。
- **避免數據洩漏**: 嚴格區分訓練集和測試集，絕不在測試集上進行任何形式的擬合（fitting）。

---

[來源: ch02] ## 常見問答 (FAQ)
**Q: 為什麼需要將數據分成訓練集和測試集？**
A: 這是為了評估模型的泛化能力。模型在訓練集上表現好並不意味著它在從未見過的新數據上也會表現好。測試集模擬了這種新數據，讓我們可以估計模型在真實世界中的性能。如果我們在所有數據上訓練模型，我們就無法知道它是否只是「記住」了訓練數據（過擬合），而不是學到了通用的模式。

---

[來源: ch02] 從未見過的新數據上也會表現好。測試集模擬了這種新數據，讓我們可以估計模型在真實世界中的性能。如果我們在所有數據上訓練模型，我們就無法知道它是否只是「記住」了訓練數據（過擬合），而不是學到了通用的模式。

**Q: `fit`, `transform`, `fit_transform` 有什麼區別？**
A: - `fit()`: 轉換器學習數據的參數（例如，`SimpleImputer` 學習中位數，`StandardScaler` 學習平均值和標準差）。它只計算，不應用。
   - `transform()`: 使用**已經學習到**的參數來轉換數據。
   - `fit_transform()`: 先執行 `fit()`，然後執行 `transform()`。這是一個方便的捷徑，但只能對訓練數據使用。對於測試數據，必須只使用 `transform()`，以確保使用從訓練數據中學到的相同參數。

---

[來源: ch02] : 先執行 `fit()`，然後執行 `transform()`。這是一個方便的捷徑，但只能對訓練數據使用。對於測試數據，必須只使用 `transform()`，以確保使用從訓練數據中學到的相同參數。

**Q: 網格搜索（Grid Search）和隨機搜索（Randomized Search）哪個更好？**
A: 這取決於情況。如果超參數的可能組合不多，網格搜索可以窮盡所有可能，找到理論上的最優組合。但如果超參數空間很大，網格搜索的計算成本會急劇增加。在這種情況下，隨機搜索通常更高效，它在指定的範圍內隨機抽樣組合，通常能在更短的時間內找到一個非常好（即使不一定是絕對最優）的解。

---

[來源: ch02] 到理論上的最優組合。但如果超參數空間很大，網格搜索的計算成本會急劇增加。在這種情況下，隨機搜索通常更高效，它在指定的範圍內隨機抽樣組合，通常能在更短的時間內找到一個非常好（即使不一定是絕對最優）的解。

**Q: 如果我的電腦記憶體不足，無法處理大型數據集怎麼辦？**
A: 您可以考慮以下幾種策略：
   - 使用數據抽樣，先在較小的數據子集上驗證您的方法
   - 採用增量學習算法（如 `SGDRegressor`）
   - 使用更高效的數據格式（如 Parquet）
   - 考慮雲端運算平台的資源

---

[來源: ch02] 使用數據抽樣，先在較小的數據子集上驗證您的方法
   - 採用增量學習算法（如 `SGDRegressor`）
   - 使用更高效的數據格式（如 Parquet）
   - 考慮雲端運算平台的資源

**Q: 為什麼我的模型在訓練集上表現很好，但在測試集上表現很差？**
A: 這是典型的過擬合現象。可能的解決方案包括：
   - 增加更多的訓練數據
   - 使用正則化技術
   - 減少模型複雜度
   - 使用交叉驗證進行更穩健的模型選擇
   - 檢查數據前處理是否存在數據洩漏

---

[來源: ch02] 過擬合現象。可能的解決方案包括：
   - 增加更多的訓練數據
   - 使用正則化技術
   - 減少模型複雜度
   - 使用交叉驗證進行更穩健的模型選擇
   - 檢查數據前處理是否存在數據洩漏

**Q: 如何選擇最適合的評估指標？**
A: 選擇評估指標應該基於您的業務目標：
   - RMSE：當大誤差比小誤差更不可接受時使用
   - MAE：當所有誤差同等重要時使用
   - R²：當您想了解模型解釋了多少變異時使用
   - 對於房價預測，RMSE 通常是合適的選擇，因為大的預測誤差（如預測錯誤 10 萬美元）比小誤差更嚴重

---

[來源: ch02] ## 推薦標籤 (Suggested Hashtags)
#Python #機器學習 #數據科學 #ScikitLearn #Pandas #專案實戰 #房價預測 #技術教學 #程式設計 #數據分析 #模型訓練 #特徵工程

---

[來源: ch02 | 類型: cheatsheet] # Ch02 速查表：End-to-End Machine Learning Project

> **核心主旨**：完整 ML 專案流程 —— 從資料獲取、探索、前處理到模型選擇與超參數調優。

---

---

[來源: ch02 | 類型: cheatsheet] ## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Stratified Sampling | 依目標變數分層切分，保持分佈一致 | 類別不均衡時的 train/test split |
| ColumnTransformer | 對不同欄位套用不同前處理器 | 混合數值 + 類別特徵 |
| Pipeline | 將前處理 + 模型串成單一物件 | 避免 data leakage，便於部署 |
| Cross-Validation | K 折驗證，更穩健的效能估計 | 資料量不多時替代單一 val set |
| GridSearchCV | 枚舉所有超參數組合，找最佳 | 超參數空間小時 |
| RandomizedSearchCV | 隨機取樣超參數空間 | 超參數空間大時，節省時間 |
| Feature Importance | `model.feature_importances_` 排序特徵貢獻 | 特徵選擇、模型解釋 |

---

---

[來源: ch02 | 類型: cheatsheet] | sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `StratifiedShuffleSplit` | `n_splits=1`, `test_size=0.2` | 分層切分 |
| `SimpleImputer` | `strategy="median"/"most_frequent"` | 填補缺失值 |
| `StandardScaler` | – | 數值特徵標準化 |
| `OneHotEncoder` | `sparse_output=False`, `handle_unknown="ignore"` | 類別特徵編碼 |
| `ColumnTransformer` | `transformers=[("num", ..., num_cols), ("cat", ..., cat_cols)]` | 混合前處理 |
| `Pipeline` | `steps=[("prep", ct), ("model", rf)]` | 串接轉換 + 模型 |
| `cross_val_score` | `cv=10`, `scoring="neg_root_mean_squared_error"` | 交叉驗證評估 |
| `GridSearchCV` | `param_grid`, `cv=5`, `refit=True` | 網格搜索 |
| `RandomizedSearchCV` | `param_distributions`, `n_iter=100` | 隨機搜索 |
| `RandomForestRegressor` | `n_estimators=100`, `max_features="sqrt"` | 隨機森林迴歸 |


---

[來源: ch02 | 類型: cheatsheet] iter=100` | 隨機搜索 |
| `RandomForestRegressor` | `n_estimators=100`, `max_features="sqrt"` | 隨機森林迴歸 |


---

---

[來源: ch02 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

---

[來源: ch02 | 類型: cheatsheet] # 分層切分
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(df, df["income_cat"]):
    train_set = df.iloc[train_idx]
    test_set = df.iloc[test_idx]

---

[來源: ch02 | 類型: cheatsheet] # 定義 Pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

---

[來源: ch02 | 類型: cheatsheet] # 完整模型 Pipeline
full_model = Pipeline([
    ("preprocessing", full_pipeline),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

---

[來源: ch02 | 類型: cheatsheet] # 交叉驗證
scores = cross_val_score(full_model, X_train, y_train,
                         cv=10, scoring="neg_root_mean_squared_error")
print(f"RMSE: {-scores.mean():.0f} ± {scores.std():.0f}")

---

[來源: ch02 | 類型: cheatsheet] # 隨機搜索超參數
param_distributions = {
    "model__n_estimators": [100, 200, 300],
    "model__max_features": [4, 6, 8, "sqrt"]
}
rnd_search = RandomizedSearchCV(full_model, param_distributions, n_iter=20,
                                cv=5, scoring="neg_root_mean_squared_error",
                                random_state=42)
rnd_search.fit(X_train, y_train)
print(rnd_search.best_params_)
```

---

---

[來源: ch02 | 類型: cheatsheet] ## 4. 常見陷阱

- **Pipeline 外做 Scaling**：會造成 data leakage，scaling 的 `fit()` 必須只看訓練資料。
- **直接用 `train_test_split` 不做分層**：若目標分佈不均，小型測試集的評估會有偏差。
- **GridSearch 看 `refit=True`**：預設會在全部訓練資料上 refit 最佳模型，直接呼叫 `.predict()` 即可。
- **特徵工程做在 Pipeline 外**：若手動加特徵（例如 `rooms_per_household`），新資料也要同樣處理，最好包進 `FunctionTransformer` 或自訂 Transformer。

---

---

[來源: ch02 | 類型: cheatsheet] ## 5. 決策指南

```
選 GridSearchCV vs RandomizedSearchCV?
├── 超參數組合 < 100 → GridSearchCV
└── 超參數組合 > 100 → RandomizedSearchCV（相同時間預算下通常更好）

選 Pipeline vs 手動前處理?
└── 永遠選 Pipeline（防止 data leakage + 部署一致性）
```

**完整流程摘要**：
1. 探索資料（EDA）：`df.info()`, `df.describe()`, `df.hist()`, `df.corr()`
2. 切分：`StratifiedShuffleSplit`
3. 前處理：`ColumnTransformer` + `Pipeline`
4. 基準線（baseline）：簡單模型先建立 RMSE 基準
5. 選模型：Cross-validation 比較多個候選
6. 調超參數：`RandomizedSearchCV` → `GridSearchCV` 精調
7. 最終評估：在 **test set 只跑一次**

---

[來源: ch02 | 類型: handout] # 課程講義：端到端機器學習專案實戰 (Chapter 02)

恭喜你進入第一個完整的機器學習專案！本章以**加州房價預測**為主軸，帶你走過一位資料科學家的完整工作流程：從取得資料、探索、清洗、特徵工程、模型訓練，到超參數調整與最終評估。這套流程是工業界的通用範本，每一步都有不可省略的原因。

---

---

[來源: ch02 | 類型: handout] ### 理論背景

開始動手前，務必回答三個問題：

1. **目標**：預測房價中位數（連續值 → 迴歸問題）
2. **評估指標**：RMSE（Root Mean Squared Error）對大誤差懲罰更重；若 outlier 多可改用 MAE
3. **下游影響**：預測結果用於何處？允許的誤差範圍是多少？

$$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2}$$

**分層抽樣 (Stratified Sampling)**：若收入中位數（`median_income`）對房價影響最大，應按收入分層後再切分訓練/測試集，確保兩者分佈一致，避免抽樣偏差（Sampling Bias）。

---

[來源: ch02 | 類型: handout] ### 核心代碼

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

housing = pd.read_csv("datasets/housing/housing.csv")

---

[來源: ch02 | 類型: handout] # 建立收入類別，用於分層
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

---

[來源: ch02 | 類型: handout] # 分層抽樣
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(housing, housing["income_cat"]):
    train_set = housing.loc[train_idx]
    test_set  = housing.loc[test_idx]

---

[來源: ch02 | 類型: handout] # 移除輔助欄位
for s in (train_set, test_set):
    s.drop("income_cat", axis=1, inplace=True)
```

---

[來源: ch02 | 類型: handout] ### 補充練習 1

**理論題：** 為什麼要用分層抽樣而非隨機抽樣？若資料量很大（例如 100 萬筆），分層抽樣還有必要嗎？

**實作題：** 比較 `StratifiedShuffleSplit` 與 `train_test_split(random_state=42)` 在各收入類別比例上的差異，使用 `value_counts(normalize=True)` 量化。

---

---

[來源: ch02 | 類型: handout] ### 理論背景

EDA 的目的是發現：資料分佈、異常值、特徵間相關性，以及需要什麼前處理。

**相關係數矩陣 (Correlation Matrix)**：

$$r_{XY} = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2} \cdot \sqrt{\sum (Y_i - \bar{Y})^2}}$$

$r$ 範圍在 $[-1, 1]$：$|r|$ 越大相關越強；正值為正相關，負值為負相關。注意：相關係數只衡量**線性**相關。

**特徵工程的黃金法則**：先 EDA，再動手創造特徵。

---

[來源: ch02 | 類型: handout] # 繪製地理分布（圓的大小=人口，顏色=房價）
housing.plot(kind="scatter", x="longitude", y="latitude",
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             alpha=0.4, figsize=(10, 7))

---

[來源: ch02 | 類型: handout] # 相關係數矩陣
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

---

[來源: ch02 | 類型: handout] # 創造衍生特徵（通常比原始特徵更有用）
housing["rooms_per_house"]  = housing["total_rooms"]  / housing["households"]
housing["bedrooms_ratio"]   = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"]  / housing["households"]
```

---

[來源: ch02 | 類型: handout] ### 補充練習 2

**理論題：** 散點圖顯示 `median_income` 與 `median_house_value` 高度相關，但存在水平條紋（房價被限制在 500,000 美元）。這對模型有何影響？如何處理？

**實作題：** 使用 `scatter_matrix` 或 `seaborn.pairplot` 視覺化 `median_income`、`housing_median_age`、`median_house_value`、`rooms_per_house` 四個特徵的兩兩相關圖，觀察哪些特徵組合最值得關注。

---

---

[來源: ch02 | 類型: handout] ### 理論背景

**為何用 Pipeline？**

- 確保訓練集的統計量（均值、標準差）不洩漏到測試集
- 前處理步驟自動化，部署時一致性有保障
- 支援 `GridSearchCV` 交叉驗證時的正確資料切分

常用前處理步驟：

| 問題 | 解法 |
|------|------|
| 缺失值 | `SimpleImputer(strategy="median")` |
| 類別特徵 | `OrdinalEncoder` 或 `OneHotEncoder` |
| 數值特徵縮放 | `StandardScaler`（零均值、單位變異數） |
| 混合型特徵 | `ColumnTransformer` 分別處理 |

**StandardScaler 公式：**

$$z = \frac{x - \mu}{\sigma}$$

---

[來源: ch02 | 類型: handout] ### 核心代碼

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

---

[來源: ch02 | 類型: handout] sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_features = ["longitude", "latitude", "housing_median_age",
                "total_rooms", "total_bedrooms", "population",
                "households", "median_income",
                "rooms_per_house", "bedrooms_ratio", "people_per_house"]
cat_features = ["ocean_proximity"]

---

[來源: ch02 | 類型: handout] # 數值特徵 Pipeline：填補缺失值 → 縮放
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

---

[來源: ch02 | 類型: handout] # 完整前處理管線
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])
```

---

[來源: ch02 | 類型: handout] ### 補充練習 3

**理論題：** 為什麼 `SimpleImputer` 的 `strategy="median"` 比 `strategy="mean"` 更適合有偏態分佈的特徵？

**實作題：** 手動建立一個 `FunctionTransformer`，將 `total_rooms` 除以 `households` 得到 `rooms_per_house`，並整合進 `ColumnTransformer`。確認前處理後的資料形狀正確。

---

---

[來源: ch02 | 類型: handout] ### 理論背景

模型選擇策略：先選幾種不同類型的候選模型（線性、樹狀、集成），用交叉驗證快速篩選，再對最佳候選深入調參。

**超參數搜索**：

| 方法 | 說明 | 適用場景 |
|------|------|---------|
| `GridSearchCV` | 窮舉所有組合 | 超參數少（< 5）|
| `RandomizedSearchCV` | 隨機取樣指定次數 | 超參數多 |
| `HalvingRandomSearchCV` | 進化式篩選，速度最快 | 資源有限時 |

最終模型評估：計算測試集 RMSE，並用 95% 信賴區間量化估計的不確定性：

$$\text{95\% CI} \approx \bar{e} \pm 1.96 \cdot \frac{\sigma_e}{\sqrt{m}}$$

---

[來源: ch02 | 類型: handout] ### 核心代碼

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

---

[來源: ch02 | 類型: handout] # 完整的端到端 Pipeline
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", RandomForestRegressor(random_state=42))
])

---

[來源: ch02 | 類型: handout] # 超參數搜索
param_distributions = {
    "model__n_estimators":  randint(100, 500),
    "model__max_features":  randint(2, 20),
    "model__max_leaf_nodes": randint(10, 200),
}
rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions,
    n_iter=50,           # 嘗試 50 組
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42
)
rnd_search.fit(X_train, y_train)

---

[來源: ch02 | 類型: handout] # 最終測試集評估
final_model = rnd_search.best_estimator_
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()
final_rmse = mean_squared_error(y_test, final_model.predict(X_test), squared=False)
print(f"測試集 RMSE: {final_rmse:.0f}")
```

---

[來源: ch02 | 類型: handout] ### 補充練習 4

**理論題：** `RandomizedSearchCV` 與 `GridSearchCV` 的主要差異是什麼？在哪些情況下 Randomized 更有優勢？

**實作題：** 使用 `feature_importances_` 提取 `RandomForestRegressor` 的特徵重要性，繪製條形圖，指出重要性最高的前 5 個特徵是否符合你的直覺。

---

---

[來源: ch02 | 類型: handout] ## 結論

本章展示了一個完整的機器學習工作流程，其關鍵里程碑：

1. **問題定義** → 確定任務類型與評估指標
2. **資料取得與分層切分** → 防止測試集偏差
3. **EDA + 特徵工程** → 發現有用訊號
4. **Pipeline 封裝前處理** → 防止資料洩漏
5. **交叉驗證模型選擇 + 超參數搜索** → 找到最佳模型
6. **測試集最終評估** → 誠實估計泛化能力

下一章（Ch03）將深入分類問題的評估指標，你會發現準確率（Accuracy）有時是具有誤導性的指標。

---

---

[來源: ch02 | 類型: handout] ## 課後作業

**作業：California Housing 完整專案**

在本章的 California Housing 資料集上，完成以下擴展：

1. 在 `preprocessing` Pipeline 中加入一個 `FunctionTransformer`，新增 `rooms_per_house`、`bedrooms_ratio`、`people_per_house` 三個衍生特徵。
2. 用 `RandomizedSearchCV` 比較 `RandomForestRegressor` 與 `GradientBoostingRegressor` 的交叉驗證 RMSE，哪個更好？
3. 計算最佳模型在測試集上的 95% 信賴區間（提示：使用 `scipy.stats.t.interval`）。

---

[來源: ch02 | 類型: tutorial] [標題: 端到端機器學習專案實戰：加州房價預測完整指南 | 描述: 跟著完整流程走一遍 ML 專案：資料下載、EDA、特徵工程、Pipeline 建構、交叉驗證、GridSearch 調參，到最終模型評估。使用 Scikit-Learn 實戰加州房價資料集。 | 關鍵字: Python, 機器學習, Scikit-Learn, 端到端, 資料科學, 房價預測, Pipeline, GridSearch, 交叉驗證]
# 端到端機器學習專案實戰：加州房價預測

一個真實的 ML 專案需要哪些步驟？本教學帶您走完**完整的工作流程**——從原始資料到可部署的模型，以加州房價資料集為例，示範資料科學家的真實日常工作。

---

[來源: ch02 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **Pipeline** 是防止資料洩漏（Data Leakage）的關鍵工具
- **分層抽樣（Stratified Sampling）** 確保測試集代表性
- **交叉驗證（Cross-Validation）** 提供更穩健的效能估計
- **GridSearchCV / RandomizedSearchCV** 系統性搜尋最佳超參數
- 最終評估應只用**測試集一次**，避免過度擬合評估指標

---

---

[來源: ch02 | 類型: tutorial] ## 環境設定與資料下載

💡 **實際應用情境：** 在正式啟動任何 ML 專案前，建立可重現的環境和自動化資料下載流程，能讓團隊成員快速接手繼續工作。

---

[來源: ch02 | 類型: tutorial] ```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data() -> pd.DataFrame:
    """下載並載入加州房價資料集"""
    tarball_path = Path("datasets/housing.tgz")

    if not tarball_path.is_file():
        # 建立目錄結構（等同 mkdir -p）
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/handson-ml3/raw/main/datasets/housing.tgz"
        # 從 URL 下載到本地
        urllib.request.urlretrieve(url, tarball_path)

    # 解壓縮 .tgz 檔案
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")

    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
print(f"資料集大小: {housing.shape}")  # (20640, 10)
print(housing.head())
```

---

[來源: ch02 | 類型: tutorial] sing = load_housing_data()
print(f"資料集大小: {housing.shape}")  # (20640, 10)
print(housing.head())
```

**✅ 程式碼逐行解析：**

1. `Path("datasets/housing.tgz").is_file()`: 用 `pathlib.Path` 取代 `os.path`，語法更清晰
2. `tarball_path.mkdir(parents=True, exist_ok=True)`: `parents=True` 允許建立多層目錄；`exist_ok=True` 不會因目錄已存在而報錯
3. `urllib.request.urlretrieve(url, tarball_path)`: 標準庫下載，無需額外安裝
4. `tarfile.open(tarball_path)`: Context Manager 確保檔案正確關閉

**🎯 重點摘要:**

- **核心功能**: 自動化資料獲取，確保可重現性
- **潛在問題**: 下載失敗時無重試機制，生產環境建議加入 `try-except`
- **最佳使用情境**: 任何需要從遠端獲取資料的 ML 專案

---

---

[來源: ch02 | 類型: tutorial] ## 探索性資料分析 (EDA)

💡 **實際應用情境：** 在建模前花時間理解資料，能避免後續的許多陷阱。例如：若收入欄位被人為限制在 [0.5, 15]，直接用它訓練可能導致模型無法預測極端值。

---

[來源: ch02 | 類型: tutorial] # 基本統計資訊
print(housing.info())        # 資料類型、缺失值概況
print(housing.describe())    # 數值型欄位的統計量

---

[來源: ch02 | 類型: tutorial] # 檢查缺失值
missing = housing.isnull().sum()
print(f"\n缺失值統計:\n{missing[missing > 0]}")

---

[來源: ch02 | 類型: tutorial] # 分類變數的值分佈
print(f"\nocean_proximity 分佈:\n{housing['ocean_proximity'].value_counts()}")

---

[來源: ch02 | 類型: tutorial] # 繪製數值型欄位的直方圖
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(12, 8))
plt.tight_layout()
plt.savefig("images/housing_hist.png", dpi=100)
plt.show()
```

**✅ 程式碼逐行解析：**

1. `housing.info()`: 顯示每欄的資料類型和非空值數量，快速識別缺失值
2. `housing.describe()`: 提供 count/mean/std/min/25%/50%/75%/max，識別異常值
3. `missing[missing > 0]`: 布林索引篩選只顯示有缺失值的欄位

**🎯 重點摘要:**

- **core feature**: 在訓練前理解資料分佈和品質
- **潛在問題**: `median_house_value` 的值被截斷在 500,000，可能需要移除這些樣本
- **最佳使用情境**: 每個新資料集必須先做 EDA

---

[來源: ch02 | 類型: tutorial] ### 範例 3: 分層抽樣建立訓練/測試集

```python
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

---

[來源: ch02 | 類型: tutorial] # 確保測試集的收入分佈代表整體
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

---

[來源: ch02 | 類型: tutorial] # 分層抽樣：每個 income_cat 按比例取 20% 進入測試集
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(housing, housing["income_cat"]):
    strat_train_set = housing.iloc[train_idx].drop("income_cat", axis=1)
    strat_test_set  = housing.iloc[test_idx].drop("income_cat", axis=1)

print(f"訓練集大小: {len(strat_train_set)}")   # ~16512
print(f"測試集大小:  {len(strat_test_set)}")    # ~4128
```

**✅ 程式碼逐行解析：**

---

[來源: ch02 | 類型: tutorial] trat_train_set)}")   # ~16512
print(f"測試集大小:  {len(strat_test_set)}")    # ~4128
```

**✅ 程式碼逐行解析：**

1. `pd.cut(...)`: 將連續的收入欄位離散化為 5 個等級（用於分層抽樣的基準）
2. `StratifiedShuffleSplit(n_splits=1)`: 確保每個收入等級在訓練集和測試集中的比例相同
3. `.drop("income_cat", axis=1)`: 分層用的臨時欄位，不需要進入模型

**🎯 重點摘要:**

- **核心功能**: 避免隨機抽樣造成的樣本代表性偏差（Sampling Bias）
- **重要原則**: 若資料有明顯的分佈不均（如標籤類別不平衡），必須分層抽樣

---

---

[來源: ch02 | 類型: tutorial] ```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureAdder(BaseEstimator, TransformerMixin):
    """新增組合特徵：每戶房間數、每戶臥室比、每人房間數"""

    def fit(self, X, y=None):
        return self  # 無需訓練

    def transform(self, X):
        # 透過欄位索引存取（確保在 Pipeline 中正確運作）
        rooms_per_household    = X[:, 3] / X[:, 6]  # total_rooms / households
        bedrooms_ratio         = X[:, 4] / X[:, 3]  # total_bedrooms / total_rooms
        population_per_household = X[:, 5] / X[:, 6]  # population / households
        return np.c_[X, rooms_per_household, bedrooms_ratio, population_per_household]
```

---

[來源: ch02 | 類型: tutorial] ouseholds
        return np.c_[X, rooms_per_household, bedrooms_ratio, population_per_household]
```

**✅ 程式碼逐行解析：**

1. 繼承 `BaseEstimator, TransformerMixin`: 自動取得 `get_params()` 和 `fit_transform()` 方法
2. `fit(...)` 回傳 `self`: 此轉換器無需學習任何統計量，直接傳回自身
3. `np.c_[...]`: 按列（column）串接原始特徵和新特徵

**🎯 重點摘要:**

- **核心功能**: 從原有特徵組合出更具預測力的新特徵
- **設計原則**: 繼承 sklearn 基礎類別，確保與 Pipeline 無縫整合
- **最佳使用情境**: 需要將領域知識（Domain Knowledge）編碼為特徵時

---

---

[來源: ch02 | 類型: tutorial] ### 範例 5: 完整資料前處理 Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

---

[來源: ch02 | 類型: tutorial] # 數值型特徵處理流程
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # 填補缺失值（用中位數）
    ("attribs_adder", FeatureAdder()),               # 新增組合特徵
    ("std_scaler", StandardScaler()),                # 標準化（μ=0, σ=1）
])

---

[來源: ch02 | 類型: tutorial] # 識別數值型和類別型欄位
housing_num = strat_train_set.drop("median_house_value", axis=1)
num_attribs = list(housing_num.select_dtypes(include=[np.number]))
cat_attribs = ["ocean_proximity"]

---

[來源: ch02 | 類型: tutorial] # ColumnTransformer：對不同欄位套用不同前處理
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),                   # 數值欄位
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),  # 類別欄位
])

---

[來源: ch02 | 類型: tutorial] # 準備訓練資料
X_train = full_pipeline.fit_transform(
    strat_train_set.drop("median_house_value", axis=1)
)
y_train = strat_train_set["median_house_value"].copy()
print(f"訓練特徵形狀: {X_train.shape}")
```

**✅ 程式碼逐行解析：**

1. `SimpleImputer(strategy="median")`: 用每欄的**中位數**填補缺失值（比平均數更抗離群值）
2. `ColumnTransformer([...])`: 對不同欄位套用不同的前處理策略，最後自動合併
3. `OneHotEncoder(handle_unknown="ignore")`: 對測試集中出現但訓練集未見的類別，以全零向量處理

**🎯 重點摘要:**

- **核心功能**: 自動化、可重現的完整資料前處理
- **重要原則**: 用 `fit_transform` 訓練集，用 `transform` 測試集（避免洩漏）

---

---

[來源: ch02 | 類型: tutorial] ```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

models = {
    "線性回歸": LinearRegression(),
    "決策樹": DecisionTreeRegressor(random_state=42),
    "隨機森林": RandomForestRegressor(n_estimators=100, random_state=42),
}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )
    rmse_mean = -scores.mean()
    rmse_std  = scores.std()
    print(f"{name}: RMSE = {rmse_mean/1000:.1f}K ± {rmse_std/1000:.1f}K")
```

---

[來源: ch02 | 類型: tutorial] se_std  = scores.std()
    print(f"{name}: RMSE = {rmse_mean/1000:.1f}K ± {rmse_std/1000:.1f}K")
```

**🎯 重點摘要:**

- **核心功能**: 用統一標準比較多種候選模型
- **選擇依據**: 在交叉驗證 RMSE 最低的模型上進行超參數調優

---

---

[來源: ch02 | 類型: tutorial] ### 範例 7: GridSearchCV 系統搜尋

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

---

[來源: ch02 | 類型: tutorial] # 定義超參數搜尋空間
param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    forest_reg, param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    return_train_score=True,  # 同時返回訓練集分數
    n_jobs=-1                 # 使用所有 CPU 核心
)
grid_search.fit(X_train, y_train)

print(f"最佳參數: {grid_search.best_params_}")
print(f"最佳 RMSE: {-grid_search.best_score_/1000:.1f}K")

---

[來源: ch02 | 類型: tutorial] # 特徵重要性分析
feature_importances = grid_search.best_estimator_.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
print("\n最重要的 5 個特徵:")
feature_names = num_attribs + list(full_pipeline.named_transformers_["cat"].get_feature_names_out())
for i in sorted_idx[:5]:
    print(f"  {feature_names[i]}: {feature_importances[i]:.4f}")
```

**✅ 程式碼逐行解析：**

1. `param_grid` 是個 list of dict：Scikit-Learn 會對每個 dict 中的超參數做笛卡爾積
2. `n_jobs=-1`: 使用全部 CPU 核心平行計算（大幅加速）
3. `return_train_score=True`: 同時記錄訓練集分數，用於診斷過擬合

**🎯 重點摘要:**

---

[來源: ch02 | 類型: tutorial] 積
2. `n_jobs=-1`: 使用全部 CPU 核心平行計算（大幅加速）
3. `return_train_score=True`: 同時記錄訓練集分數，用於診斷過擬合

**🎯 重點摘要:**

- **核心功能**: 系統性找到最佳超參數組合
- **注意**: GridSearch 複雜度是 O(參數組合數 × CV折數)，參數空間大時改用 `RandomizedSearchCV`

---

---

[來源: ch02 | 類型: tutorial] ### 範例 8: 測試集最終評估

```python
from sklearn.metrics import mean_squared_error

---

[來源: ch02 | 類型: tutorial] # 取出最佳模型
final_model = grid_search.best_estimator_

---

[來源: ch02 | 類型: tutorial] # 前處理測試集（只 transform，不 fit！）
X_test = full_pipeline.transform(strat_test_set.drop("median_house_value", axis=1))
y_test = strat_test_set["median_house_value"].copy()

---

[來源: ch02 | 類型: tutorial] # 最終預測
y_pred = final_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"測試集最終 RMSE: {final_rmse/1000:.1f}K 美元")

---

[來源: ch02 | 類型: tutorial] # 95% 置信區間
from scipy import stats
confidence = 0.95
squared_errors = (y_pred - y_test) ** 2
interval = np.sqrt(stats.t.interval(
    confidence,
    len(squared_errors) - 1,
    loc=squared_errors.mean(),
    scale=stats.sem(squared_errors)
))
print(f"95% CI: [{np.sqrt(interval[0])/1000:.1f}K, {np.sqrt(interval[1])/1000:.1f}K]")
```

**✅ 程式碼逐行解析：**

1. `full_pipeline.transform(...)`: 測試集只能 `transform`，不能 `fit`（否則會造成測試集洩漏）
2. `stats.t.interval(confidence, ...)`: 計算 t 分佈的置信區間，量化預測的不確定性

**🎯 重點摘要:**

- **黃金法則**: 測試集在整個開發過程中應嚴格封存，只在最後使用一次
- **置信區間**: 量化最終評估的統計不確定性，有助於向利益相關者溝通

---

---

[來源: ch02 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: 為什麼一定要用 Pipeline？**

A: Pipeline 防止**資料洩漏（Data Leakage）**。若手動先 `fit_transform` 整個資料集再分割，則測試集的統計資訊（如 StandardScaler 的均值/標準差）已污染了轉換，導致評估結果虛高。Pipeline 確保 `scaler.fit` 只在訓練集上執行。

**Q2: GridSearch 和 RandomizedSearch 怎麼選？**

A: 超參數組合少（< 100）時用 GridSearch；組合多時用 RandomizedSearch（`n_iter` 參數控制嘗試次數）。RandomizedSearch 的優點是對連續型超參數效果更好，且計算成本可控。

**Q3: 特徵重要性分析有什麼用？**

A: 它能告訴你哪些特徵對模型預測最重要，幫助你：(1) 移除不重要的特徵降低維度；(2) 聚焦於最有價值的資料收集方向；(3) 向業務部門解釋模型決策依據。

**Q4: RMSE 和 MAE 有什麼不同？**

A: RMSE（均方根誤差）對大誤差懲罰更重（平方放大了大誤差的影響）；MAE（平均絕對誤差）對所有誤差一視同仁。若異常值/大誤差在業務上代價很高，用 RMSE；若想要更具解釋性的指標，用 MAE。

---

---

[來源: ch02 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #機器學習 #ScikitLearn #端到端 #資料科學 #房價預測 #Pipeline #GridSearch #特徵工程 #交叉驗證 #程式設計 #教學 #DataScience #MachineLearning

---

[來源: ch03] [標題: 🐍 第三章：分類 (Classification) - Hands-On Machine Learning 實作筆記 | 描述: 本章深入探討了機器學習中的分類問題，從 MNIST 資料集入手，涵蓋了二元分類器、多類別分類、多標籤分類以及多輸出分類的理論與實作。我們將學習如何評估模型效能，包括準確率、混淆矩陣、精確率、召回率、F1 分數、ROC 曲線與 AUC。 | 關鍵字: Python, 機器學習, 分類, MNIST, Scikit-Learn, SGD, SVM, 隨機森林, 效能評估, ROC曲線, Hands-On Machine Learning]
# 第三章：分類 (Classification)

本章節將帶您深入了解機器學習中的「分類」任務。我們將使用经典的 MNIST 手寫數字資料集，從頭開始建立並評估一個分類模型。您將學習到如何訓練二元分類器、多類別分類器，並探索多標籤與多輸出分類等進階主題。

---

[來源: ch03] ## 關鍵重點 (Key Takeaways)

- **分類器種類**: 了解二元、多類別、多標籤與多輸出分類的區別與應用場景。
- **效能指標**: 學習準確率的陷阱，並掌握精確率、召回率、F1 分數、ROC 曲線與 AUC 等關鍵評估指標。
- **模型選擇**: 透過 `SGDClassifier`、`RandomForestClassifier` 與 `SVC` 等不同模型，了解其在分類任務中的優劣。
- **資料前處理**: 學習如何使用 `StandardScaler` 對資料進行特徵縮放，以提升模型效能。
- **錯誤分析**: 透過混淆矩陣視覺化模型的錯誤，找出改進方向。

---

---

[來源: ch03] ## 設定

在開始之前，我們需要確認 Python 環境與必要的套件版本。

---

[來源: ch03] ### 範例 1: 確認 Python 版本

```python

---

[來源: ch03] # 斷言 Python 版本是否大於等於 3.7
assert sys.version_info >= (3, 7)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 載入 Python 的 `sys` 模組，它提供了對 Python 解譯器本身的存取。
2. `assert sys.version_info >= (3, 7)`: 使用 `assert` 語句檢查當前的 Python 版本是否為 3.7 或更高。`sys.version_info` 會回傳一個包含主版號、次版號等的元組。如果條件不成立，程式將會中斷並拋出 `AssertionError`。

**🎯 重點摘要:**

---

[來源: ch03] 版本是否為 3.7 或更高。`sys.version_info` 會回傳一個包含主版號、次版號等的元組。如果條件不成立，程式將會中斷並拋出 `AssertionError`。

**🎯 重點摘要:**

- **核心功能**: 確保執行環境符合最低 Python 版本要求，避免因版本不相容導致的錯誤。
- **最佳使用情境**: 在專案或腳本的開頭進行環境檢查，確保程式在預期的環境中運行。

---

[來源: ch03] ### 範例 2: 確認 Scikit-Learn 版本

```python

---

[來源: ch03] # 從 packaging 套件中載入 version
from packaging import version

---

[來源: ch03] # 載入 scikit-learn
import sklearn

---

[來源: ch03] # 斷言 scikit-learn 版本是否大於等於 1.0.1
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
```

**✅ 程式碼逐行解析：**

---

[來源: ch03] 是否大於等於 1.0.1
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
```

**✅ 程式碼逐行解析：**

1. `from packaging import version`: 從 `packaging` 套件中匯入 `version` 模組，它提供了一個強大的版本號解析與比較工具。
2. `import sklearn`: 載入 `scikit-learn` 函式庫。
3. `assert version.parse(sklearn.__version__) >= version.parse("1.0.1")`: 使用 `version.parse()` 將字串形式的版本號（如 `"1.0.1"`）轉換為可比較的物件，然後斷言當前安裝的 `scikit-learn` 版本是否符合最低要求。

---

[來源: ch03] "1.0.1")`: 使用 `version.parse()` 將字串形式的版本號（如 `"1.0.1"`）轉換為可比較的物件，然後斷言當前安裝的 `scikit-learn` 版本是否符合最低要求。

**🎯 重點摘要:**

- **核心功能**: 確保 `scikit-learn` 函式庫的版本足夠新，以支援本筆記中使用的所有功能。
- **潛在問題**: 如果未使用 `version.parse()`，直接比較字串版本號（如 `"1.2"` > `"1.10"`）可能會得到錯誤的結果。

---

[來源: ch03] ### 範例 3: 設定 Matplotlib 圖表樣式

```python

---

[來源: ch03] # 載入 matplotlib.pyplot
import matplotlib.pyplot as plt

---

[來源: ch03] # 設定全域字體大小
plt.rc('font', size=14)

---

[來源: ch03] # 設定座標軸標籤與標題大小
plt.rc('axes', labelsize=14, titlesize=14)

---

[來源: ch03] # 設定圖例字體大小
plt.rc('legend', fontsize=14)

---

[來源: ch03] # 設定 x, y 軸刻度標籤大小
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

1. `import matplotlib.pyplot as plt`: 載入 `matplotlib` 的 `pyplot` 模組，這是繪製圖表的主要介面。
2. `plt.rc(...)`: `rc` 是 "runtime configuration" 的縮寫，用於設定 `matplotlib` の全域參數，讓圖表風格保持一致。這裡我們統一設定了字體、座標軸、圖例等元素的預設大小。

**🎯 重點摘要:**

- **核心功能**: 統一圖表視覺風格，提升可讀性。
- **最佳使用情境**: 在筆記本或專案的開頭設定，確保所有後續生成的圖表都遵循相同的樣式規範。

---

[來源: ch03] ### 範例 4: 建立圖片儲存路徑與輔助函式

```python

---

[來源: ch03] # 載入 pathlib 中的 Path
from pathlib import Path

---

[來源: ch03] # 定義圖片儲存路徑
IMAGES_PATH = Path() / "images" / "classification"

---

[來源: ch03] # 建立路徑 (如果不存在)
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

---

[來源: ch03] # 定義儲存圖片的函式
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

---

[來源: ch03] plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `from pathlib import Path`: 載入 `pathlib` 模組的 `Path` 物件，它提供了物件導向的檔案系統路徑操作。
2. `IMAGES_PATH = Path() / "images" / "classification"`: 使用 `/` 運算子串接路徑，建立一個指向 `images/classification` 資料夾的 `Path` 物件。
3. `IMAGES_PATH.mkdir(parents=True, exist_ok=True)`: 建立資料夾。`parents=True` 表示如果上層資料夾不存在，也會一併建立；`exist_ok=

---

[來源: ch03] IMAGES_PATH.mkdir(parents=True, exist_ok=True)`: 建立資料夾。`parents=True` 表示如果上層資料夾不存在，也會一併建立；`exist_ok=True` 表示如果資料夾已存在，則不會拋出錯誤。
4. `def save_fig(...)`: 定義一個輔助函式，用於儲存 `matplotlib` 圖表。
5. `plt.tight_layout()`: 自動調整子圖參數，使之緊密排列，避免標籤重疊。
6. `plt.savefig(...)`: 將當前圖表儲存到指定路徑，並設定格式與解析度。

**🎯 重點摘要:**

- **核心功能**: 自動化圖片的儲存流程，並統一管理圖片路徑。
- **最佳使用情境**: 在需要重複儲存多張圖表的筆記本中，定義此類輔助函式可以讓程式碼更簡潔。

---

---

[來源: ch03] ## MNIST 資料集

💡 **實際應用情境：**
MNIST 資料集是機器學習领域的 "Hello, World!"。它包含了 70,000 張手寫數字的灰階圖片（0 到 9），每張圖片大小為 28x28 像素。這個資料集常用於評估與比較各種分類演算法的效能。

---

[來源: ch03] ### 範例 5: 載入 MNIST 資料集

```python

---

[來源: ch03] # 從 sklearn.datasets 載入 fetch_openml
from sklearn.datasets import fetch_openml

---

[來源: ch03] # 從 OpenML 抓取 mnist_784 資料集
mnist = fetch_openml('mnist_784', as_frame=False)
```

**✅ 程式碼逐行解析：**

---

[來源: ch03] 1. `from sklearn.datasets import fetch_openml`: `fetch_openml` 是一個方便的函式，可以直接從 [OpenML.org](https://www.openml.org/) 網站下載公開資料集。
2. `mnist = fetch_openml('mnist_784', as_frame=False)`: 下載名為 `mnist_784` 的資料集。`as_frame=False` 參數表示我們希望資料以 NumPy 陣列的形式回傳，而不是 Pandas DataFrame。回傳的 `mnist` 物件是一個類似字典的 `Bunch` 物件，包含了資料、標籤與描述資訊。

**🎯 重點摘要:**

---

[來源: ch03] 資料以 NumPy 陣列的形式回傳，而不是 Pandas DataFrame。回傳的 `mnist` 物件是一個類似字典的 `Bunch` 物件，包含了資料、標籤與描述資訊。

**🎯 重點摘要:**

- **核心功能**: 方便地從網路獲取標準資料集，無需手動下載與解壓縮。
- **潛在問題**: `fetch_openml` 需要網路連線。第一次下載後，Scikit-Learn 會將資料快取在本地，後續載入會變快。

---

[來源: ch03] # 將資料與標籤分離
X, y = mnist.data, mnist.target

---

[來源: ch03] # 顯示特徵資料
X
```

![image](images/classification/some_digit_plot.png)

**✅ 程式碼逐行解析：**

1. `X, y = mnist.data, mnist.target`: `mnist.data` 包含了所有的圖片資料（特徵），`mnist.target` 則是對應的標籤（0-9 的數字）。我們將它們分別指派給 `X` 和 `y`。
2. `X`: `X` 是一個 NumPy 陣列，形狀為 `(70000, 784)`。每一行代表一張圖片，784 個特徵對應 28x28 像素的灰階值。

**🎯 重點摘要:**

- **核心功能**: 將資料集的特徵（`X`）與目標（`y`）分離，這是機器學習工作流程的標準步驟。
- **資料結構**: `X` 是二維陣列（樣本數 x 特徵數），`y` 是一維陣列（樣本數）。

---

[來源: ch03] ### 範例 7: 視覺化單一數字

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()
```

![image](images/classification/some_digit_plot.png)

**✅ 程式碼逐行解析：**

---

[來源: ch03] igit(some_digit)
plt.show()
```

![image](images/classification/some_digit_plot.png)

**✅ 程式碼逐行解析：**

1. `def plot_digit(image_data)`: 定義一個函式，用於將一維的像素陣列視覺化。
2. `image = image_data.reshape(28, 28)`:將 784 個元素的一維陣列重塑為 28x28 的二維陣列，以符合圖片的原始尺寸。
3. `plt.imshow(image, cmap="binary")`: 使用 `imshow` 函式顯示圖片。`cmap="binary"` 表示使用黑白色彩映射，像素值低的點顯示為黑色，高的點顯示為白色。
4. `plt.axis("off")`: 關閉座標軸，讓圖片更清晰。
5. `some_digit = X[0]`: 選取資料集中的第一張圖片。
6. `plot_digit(some_digit)`: 呼叫函式繪製該圖片。

---

[來源: ch03] "off")`: 關閉座標軸，讓圖片更清晰。
5. `some_digit = X[0]`: 選取資料集中的第一張圖片。
6. `plot_digit(some_digit)`: 呼叫函式繪製該圖片。

**🎯 重點摘要:**

- **核心功能**: 將扁平化的特徵向量還原成二維圖像並顯示，方便我們直觀地理解資料內容。
- **最佳使用情境**: 在進行資料分析或模型除錯時，視覺化樣本有助於檢查資料是否正確載入或模型預測是否合理。

---

[來源: ch03] ### 範例 8: 分割訓練集與測試集

```python

---

[來源: ch03] # 分割資料為訓練集與測試集
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

**✅ 程式碼逐行解析：**

1. `X_train, X_test = X[:60000], X[60000:]`: MNIST 資料集已經預先打亂，且通常前 60,000 筆作為訓練集，後 10,000 筆作為測試集。這裡我們使用 Python 的切片語法來進行分割。
2. `y_train, y_test = y[:60000], y[60000:]`: 同樣地，對標籤 `y` 進行相同的分割。

**🎯 重點摘要:**

---

[來源: ch03] 使用 Python 的切片語法來進行分割。
2. `y_train, y_test = y[:60000], y[60000:]`: 同樣地，對標籤 `y` 進行相同的分割。

**🎯 重點摘要:**

- **核心功能**: 將資料集劃分為訓練集和測試集，是所有監督式學習任務的標準流程。模型在訓練集上學習，在測試集上評估其泛化能力。
- **潛在問題**: 確保訓練集與測試集沒有重疊，否則會導致評估結果過於樂觀，無法反映模型在未知資料上的真實表現。

---

---

[來源: ch03] ## 訓練一個二元分類器

💡 **實際應用情境：**
二元分類器用於解決「是」或「否」的問題。例如，判斷一封郵件是否為垃圾郵件、一張圖片是否為貓、或一個交易是否為詐騙。在這裡，我們將簡化問題：只判斷一個數字**是否為 `5`**。

---

[來源: ch03] # 建立二元分類的目標向量
y_train_5 = (y_train == '5')  # 如果是 '5'，則為 True，否則為 False
y_test_5 = (y_test == '5')
```

**✅ 程式碼逐行解析：**

1. `y_train_5 = (y_train == '5')`: 這行程式碼會對 `y_train` 中的每個元素進行比較。如果元素等於字串 `'5'`，則新陣列 `y_train_5` 在該位置的值為 `True`，否則為 `False`。
2. `y_test_5 = (y_test == '5')`: 同樣地，為測試集建立二元標籤。

**🎯 重點摘要:**

---

[來源: ch03] y_train_5` 在該位置的值為 `True`，否則為 `False`。
2. `y_test_5 = (y_test == '5')`: 同樣地，為測試集建立二元標籤。

**🎯 重點摘要:**

- **核心功能**: 將一個多類別分類問題（0-9）轉換為一個二元分類問題（是 5 vs 不是 5）。
- **資料型態**: `y_train_5` 和 `y_test_5` 是布林值（Boolean）陣列，這在 Scikit-Learn 中可以被直接當作 `0` 和 `1` 來處理。

---

[來源: ch03] ### 範例 10: 訓練 SGD 分類器

```python

---

[來源: ch03] # 從 sklearn.linear_model 載入 SGDClassifier
from sklearn.linear_model import SGDClassifier

---

[來源: ch03] # 建立 SGDClassifier 實例
sgd_clf = SGDClassifier(random_state=42)

---

[來源: ch03] # 使用訓練資料擬合模型
sgd_clf.fit(X_train, y_train_5)
```

**✅ 程式碼逐行解析：**

---

[來源: ch03] 1. `from sklearn.linear_model import SGDClassifier`: `SGDClassifier` 是一個使用隨機梯度下降（Stochastic Gradient Descent）演算法來訓練的線性分類器。
2. `sgd_clf = SGDClassifier(random_state=42)`: 建立一個 `SGDClassifier` 的實例。`random_state=42` 確保了每次執行程式碼時，SGD 的隨機過程都是一樣的，使得結果可以重現。
3. `sgd_clf.fit(X_train, y_train_5)`: 使用訓練集的特徵 `X_train` 和二元目標 `y_train_5` 來訓練模型。模型會學習如何從 784 個像素值中辨識出數字 `5`。

**🎯 重點摘要:**

---

[來源: ch03] n, y_train_5)`: 使用訓練集的特徵 `X_train` 和二元目標 `y_train_5` 來訓練模型。模型會學習如何從 784 個像素值中辨識出數字 `5`。

**🎯 重點摘要:**

- **核心功能**: `SGDClassifier` 是一個高效的分類器，特別適合處理像 MNIST 這樣的大型資料集。
- **最佳使用情境**: 當資料量很大，無法一次性載入記憶體時，SGD 是一個很好的選擇，因為它可以進行增量學習（online learning）。

---

[來源: ch03] # 預測 `some_digit` 是否為 5
sgd_clf.predict([some_digit])
```

**✅ 程式碼逐行解析：**

1. `sgd_clf.predict([some_digit])`: 使用訓練好的 `sgd_clf` 模型來預測 `some_digit`（我們之前選取的第一張圖片）。`predict` 方法會回傳 `True` 或 `False`。因為 `some_digit` 確實是數字 `5`，所以模型正確地預測為 `True`。

**🎯 重點摘要:**

- **核心功能**: `.predict()` 方法是使用訓練好的模型進行預測的標準介面。
- **輸入格式**: 注意 `[some_digit]` 使用了方括號，因為 Scikit-Learn 的 `predict` 方法期望收到一個二維陣列（即使只有一個樣本）。

---

---

[來源: ch03] ## 效能衡量

💡 **實際應用情境：**
模型訓練完後，我們如何知道它做得好不好？單純看「準確率」有時會誤導人，尤其是在資料不平衡的情況下。本節將介紹更全面的評估方法。

---

[來源: ch03] ### 範例 12: 使用交叉驗證衡量準確率

```python

---

[來源: ch03] # 從 sklearn.model_selection 載入 cross_val_score
from sklearn.model_selection import cross_val_score

---

[來源: ch03] # 使用 3-fold 交叉驗證計算準確率
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

**✅ 程式碼逐行解析：**

---

[來源: ch03] 交叉驗證計算準確率
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

**✅ 程式碼逐行解析：**

1. `from sklearn.model_selection import cross_val_score`: `cross_val_score` 是一個方便的函式，可以自動完成交叉驗證的整個流程。
2. `cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")`:
   - `sgd_clf`: 要評估的模型。
   - `X_train`, `y_train_5`: 訓練資料。
   - `cv=3`: `cv` 代表 "cross-validation"，這裡設定為 3，表示要進行 3-fold

---

[來源: ch03] 的模型。
   - `X_train`, `y_train_5`: 訓練資料。
   - `cv=3`: `cv` 代表 "cross-validation"，這裡設定為 3，表示要進行 3-fold 交叉驗證。資料會被分成 3 折，輪流將其中 1 折作為驗證集，另外 2 折作為訓練集，重複 3 次。
   - `scoring="accuracy"`: 指定評估指標為「準確率」。
   - **回傳值**: 一個包含 3 次驗證準確率的陣列。

**🎯 重點摘要:**

- **核心功能**: 交叉驗證提供比單次訓練/驗證分割更穩健的模型效能評估。它可以減少因偶然的資料分割所帶來的評估偏差。
- **潛在問題**: 交叉驗證的計算成本較高，因為模型需要被訓練 `cv` 次。

---

[來源: ch03] ### 範例 13: 混淆矩陣 (Confusion Matrix)

```python

---

[來源: ch03] # 從 sklearn.model_selection 載入 cross_val_predict
from sklearn.model_selection import cross_val_predict

---

[來源: ch03] # 從 sklearn.metrics 載入 confusion_matrix
from sklearn.metrics import confusion_matrix

---

[來源: ch03] # 取得交叉驗證的預測結果
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

---

[來源: ch03] # 計算混淆矩陣
cm = confusion_matrix(y_train_5, y_train_pred)
cm
```

**✅ 程式碼逐行解析：**

---

[來源: ch03] 1. `from sklearn.model_selection import cross_val_predict`: 與 `cross_val_score` 不同，`cross_val_predict` 回傳的是在交叉驗證過程中，對每個樣本的「預測值」，而不是效能分數。
2. `y_train_pred = cross_val_predict(...)`: 這裡的 `y_train_pred` 是一個與 `y_train_5` 大小相同的陣列，其中每個預測都是由一個「沒有看過」該樣本的模型所做出的。
3. `cm = confusion_matrix(y_train_5, y_train_pred)`: `confusion_matrix` 函式接收真實標籤 `y_train_5` 和預測標籤 `y_train_pred`，並回傳一個 2x2 的矩陣。

**🎯 重點摘要:**

---

[來源: ch03] ain_pred)`: `confusion_matrix` 函式接收真實標籤 `y_train_5` 和預測標籤 `y_train_pred`，並回傳一個 2x2 的矩陣。

**🎯 重點摘要:**

- **核心功能**: 混淆矩陣提供了比單一準確率更詳細的效能資訊。它顯示了模型在各類別上的預測分佈情況。
- **矩陣解讀**:
  - 左上 (TN): 正確預測為「非 5」。
  - 右上 (FP): 錯誤預測為「是 5」（偽陽性）。
  - 左下 (FN): 錯誤預測為「非 5」（偽陰性）。
  - 右下 (TP): 正確預測為「是 5」。
- **最佳使用情境**: 混淆矩陣是計算精確率、召回率等指標的基礎，對於理解模型的錯誤類型至關重要。

---

[來源: ch03] ### 範例 14: 精確率與召回率 (Precision and Recall)

```python

---

[來源: ch03] # 從 sklearn.metrics 載入 precision_score, recall_score
from sklearn.metrics import precision_score, recall_score

---

[來源: ch03] # 計算精確率
precision_score(y_train_5, y_train_pred)

---

[來源: ch03] # 計算召回率
recall_score(y_train_5, y_train_pred)
```

**✅ 程式碼逐行解析：**

1. `precision_score(y_train_5, y_train_pred)`: 計算精確率，公式為 `TP / (TP + FP)`。在所有被模型預測為「是 5」的樣本中，有多少是真的 `5`？
2. `recall_score(y_train_5, y_train_pred)`: 計算召回率，公式為 `TP / (TP + FN)`。在所有真正的 `5` 中，有多少被模型成功找出來了？

**🎯 重點摘要:**

---

[來源: ch03] _score(y_train_5, y_train_pred)`: 計算召回率，公式為 `TP / (TP + FN)`。在所有真正的 `5` 中，有多少被模型成功找出來了？

**🎯 重點摘要:**

- **核心功能**: 精確率和召回率是衡量分類器效能的兩個核心指標，特別是在类别不平衡的資料集上。
- **權衡 (Trade-off)**: 通常情況下，提高精確率會導致召回率下降，反之亦然。這被稱為「精確率/召回率權衡」。
- **最佳使用情境**:
  - **高精確率**: 當「偽陽性」的代價很高時（例如：將正常郵件誤判為垃圾郵件）。
  - **高召回率**: 當「偽陰性」的代價很高時（例如：未能檢測出癌症病患）。

---

[來源: ch03] # 從 sklearn.metrics 載入 f1_score
from sklearn.metrics import f1_score

---

[來源: ch03] # 計算 F1 分數
f1_score(y_train_5, y_train_pred)
```

**✅ 程式碼逐行解析：**

1. `f1_score(y_train_5, y_train_pred)`: F1 分數是精確率和召回率的「調和平均數」。公式為 `2 * (precision * recall) / (precision + recall)`。

**🎯 重點摘要:**

- **核心功能**: F1 分數提供了一個結合精確率與召回率的單一指標。當你希望兩者都有不錯的表現時，F1 分數是一個很好的綜合評估指標。
- **特性**: F1 分數對於較低的值更為敏感。只有當精確率和召回率都很高時，F1 分數才會高。

---

[來源: ch03] ### 範例 16: 精確率/召回率權衡與決策邊界

```python

---

[來源: ch03] # 取得單一樣本的決策分數
y_scores = sgd_clf.decision_function([some_digit])
y_scores

---

[來源: ch03] # 設定不同的閾值
threshold = 0
y_some_digit_pred = (y_scores > threshold) # > True

threshold = 3000
y_some_digit_pred = (y_scores > threshold) # > False
```

**✅ 程式碼逐行解析：**

---

[來源: ch03] # > True

threshold = 3000
y_some_digit_pred = (y_scores > threshold) # > False
```

**✅ 程式碼逐行解析：**

1. `y_scores = sgd_clf.decision_function([some_digit])`: `decision_function` 回傳每個樣本的「決策分數」。對於線性分類器，這代表樣本到決策邊界的帶符號距離。分數越高，模型越確信樣本屬於正類。
2. `y_some_digit_pred = (y_scores > threshold)`: `predict` 方法的內部機制其實就是比較決策分數和一個閾值（預設為 0）。透過手動調整 `threshold`，我們可以改變模型的決策邊界。

**🎯 重點摘要:**

---

[來源: ch03] threshold)`: `predict` 方法的內部機制其實就是比較決策分數和一個閾值（預設為 0）。透過手動調整 `threshold`，我們可以改變模型的決策邊界。

**🎯 重點摘要:**

- **核心功能**: 透過調整決策閾值，我們可以在精確率和召回率之間进行權衡。
- **閾值影響**:
  - **提高閾值**: 模型變得更「嚴格」，只在非常有把握時才預測為正類。這會提高精確率，但可能漏掉一些正樣本，從而降低召回率。
  - **降低閾值**: 模型變得更「寬鬆」，更容易將樣本預測為正類。這會提高召回率，但可能引入更多偽陽性，從而降低精確率。

---

[來源: ch03] ### 範例 17: 繪製精確率-召回率曲線 (Precision-Recall Curve)

```python

---

[來源: ch03] # 取得所有訓練樣本的決策分數
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

---

[來源: ch03] # 從 sklearn.metrics 載入 precision_recall_curve
from sklearn.metrics import precision_recall_curve

---

[來源: ch03] # 計算不同閾值下的精確率與召回率
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

---

[來源: ch03] # 繪製曲線
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend()
plt.grid()
plt.xlabel("Threshold")
plt.show()
```

![image](images/classification/precision_recall_vs_threshold_plot.png)

**✅ 程式碼逐行解析：**

---

[來源: ch03] t.show()
```

![image](images/classification/precision_recall_vs_threshold_plot.png)

**✅ 程式碼逐行解析：**

1. `y_scores = cross_val_predict(..., method="decision_function")`: 這次我們使用 `cross_val_predict` 來取得每個樣本的決策分數。
2. `precisions, recalls, thresholds = precision_recall_curve(...)`: 這個函式會計算在所有可能的閾值下，對應的精確率和召回率。
3. `plt.plot(...)`:我們將精確率和召回率作為閾值的函式繪製出來。注意 `precisions` 和 `recalls` 的長度比 `thresholds` 多一，所以我們使用 `[:-1]` 來切片。

---

[來源: ch03] lot(...)`:我們將精確率和召回率作為閾值的函式繪製出來。注意 `precisions` 和 `recalls` 的長度比 `thresholds` 多一，所以我們使用 `[:-1]` 來切片。

**🎯 重點摘要:**

- **核心功能**: PR 曲線視覺化了精確率和召回率之間的權衡。
- **曲線解讀**: 隨著閾值的升高，精確率曲線（藍色虛線）趨於上升，而召回率曲線（綠色實線）趟於下降。你可以根據業務需求，在這條曲線上選擇一個最優的平衡點（閾值）。

---

[來源: ch03] ### 範例 18: ROC 曲線 (Receiver Operating Characteristic Curve)

```python

---

[來源: ch03] # 從 sklearn.metrics 載入 roc_curve
from sklearn.metrics import roc_curve

---

[來源: ch03] # 計算 FPR, TPR
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

---

[來源: ch03] # 繪製 ROC 曲線
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.legend()
plt.show()
```

![image](images/classification/roc_curve_plot.png)

**✅ 程式碼逐行解析：**

---

[來源: ch03] id()
plt.legend()
plt.show()
```

![image](images/classification/roc_curve_plot.png)

**✅ 程式碼逐行解析：**

1. `from sklearn.metrics import roc_curve`: 載入計算 ROC 曲線所需的函式。
2. `fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)`: `roc_curve` 計算在不同閾值下的「偽陽性率 (FPR)」和「真陽性率 (TPR)」。
   - **TPR (召回率)**: `TP / (TP + FN)`
   - **FPR**: `FP / (FP + TN)`，即在所有負樣本中，被錯誤預測為正樣本的比例。
3. `plt.plot(fpr, tpr, ...)`: 繪製 TPR vs FPR 的曲線。
4. `plt.plot([0, 1], [0, 1], 'k:')`: 繪製一條對角虛線，它代表一個完全隨機的分類器。

---

[來源: ch03] lot(fpr, tpr, ...)`: 繪製 TPR vs FPR 的曲線。
4. `plt.plot([0, 1], [0, 1], 'k:')`: 繪製一條對角虛線，它代表一個完全隨機的分類器。

**🎯 重點摘要:**

- **核心功能**: ROC 曲線是另一個評估分類器效能的重要工具。它展示了 TPR 和 FPR 之間的權衡。
- **曲線解讀**: 一個好的分類器，其 ROC 曲線會盡量往左上角靠近（TPR 高，FPR 低）。曲線離對角線越遠，模型效能越好。
- **AUC (Area Under the Curve)**: ROC 曲線下的面積（AUC）是一個綜合性的效能指標。AUC 為 1.0 代表完美分類器，0.5 代表隨機分類器。

---

[來源: ch03] ### 範例 19: 比較不同模型的 ROC 曲線

```python

---

[來源: ch03] # 訓練一個隨機森林分類器
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

---

[來源: ch03] # 取得隨機森林的分數 (使用正類的機率)
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

---

[來源: ch03] # 繪製兩條 ROC 曲線
plt.plot(fpr, tpr, "b:", label="SGD")
plt.plot(fpr_forest, tpr_forest, label="Random Forest")
plt.legend(loc="lower right")
plt.show()
```

![image](images/classification/pr_curve_comparison_plot.png)

**✅ 程式碼逐行解析：**

---

[來源: ch03] right")
plt.show()
```

![image](images/classification/pr_curve_comparison_plot.png)

**✅ 程式碼逐行解析：**

1. `y_probas_forest = cross_val_predict(..., method="predict_proba")`: 對於 `RandomForestClassifier`，我們使用 `predict_proba` 方法來取得每個樣本屬於各類別的「機率」。
2. `y_scores_forest = y_probas_forest[:, 1]`: 我們選取屬於正類（'是 5'）的機率作為決策分數。
3. `fpr_forest, tpr_forest, ... = roc_curve(...)`: 計算隨機森林模型的 ROC 曲線數據。
4. `plt.plot(...)`: 在同一張圖上繪製 SGD 和隨機森林的 ROC 曲線。

---

[來源: ch03] r_forest, ... = roc_curve(...)`: 計算隨機森林模型的 ROC 曲線數據。
4. `plt.plot(...)`: 在同一張圖上繪製 SGD 和隨機森林的 ROC 曲線。

**🎯 重點摘要:**

- **核心功能**: 透過比較不同模型的 ROC 曲線與 AUC 分數，我們可以客觀地判斷哪個模型在該分類任務上表現更好。
- **結論**: 從圖中可以看出，隨機森林的 ROC 曲線更靠近左上角，且其 AUC 分數（可透過 `roc_auc_score` 計算）也更高，表明它是一個比 SGD 更好的分類器。

---

---

[來源: ch03] ## 多類別分類

💡 **實際應用情境：**
當分類任務的目標超過兩個類別時，就稱為多類別分類。例如，將手寫數字辨識為 0 到 9 中的任何一個，或者將新聞文章分類到體育、政治、娛樂等多個頻道。

Scikit-Learn 會自動偵測到你傳入的 `y` 包含了多個類別，並自動採用相應的策略。

---

[來源: ch03] ### 範例 20: 訓練多類別 SVM 分類器

```python

---

[來源: ch03] # 從 sklearn.svm 載入 SVC
from sklearn.svm import SVC

---

[來源: ch03] # 建立 SVC 實例
svm_clf = SVC(random_state=42)

---

[來源: ch03] # 為了速度，這裡只使用前 2000 個樣本
svm_clf.fit(X_train[:2000], y_train[:2000])
```

**✅ 程式碼逐行解析：**

1. `from sklearn.svm import SVC`: `SVC` (Support Vector Classifier) 是一個強大的分類模型，常用於複雜但中小型規模的資料集。
2. `svm_clf.fit(X_train[:2000], y_train[:2000])`: 我們直接使用包含 0-9 標籤的 `y_train` 來訓練模型。Scikit-Learn 的 `SVC` 在底層會自動採用 **OvO (One-vs-One)** 策略。

**🎯 重點摘要:**

---

[來源: ch03] 我們直接使用包含 0-9 標籤的 `y_train` 來訓練模型。Scikit-Learn 的 `SVC` 在底層會自動採用 **OvO (One-vs-One)** 策略。

**🎯 重點摘要:**

- **核心功能**: Scikit-Learn 的分類器大多支援多類別分類，無需手動轉換。
- **OvO vs OvR**:
  - **OvO (一對一)**: 為每一對類別訓練一個二元分類器（例如，為 '0' vs '1', '0' vs '2', ..., '8' vs '9' 都訓練一個）。對於 N 個類別，需要訓練 `N * (N-1) / 2` 個分類器。`SVC` 預設使用此策略。
  - **OvR (一對多)**: 為每個類別訓練一個二元分類器，用以區分該類別與所有其他類別（例如，'0' vs '非 0', '1' vs '非 1', ...）。對於 N 個類別，需要訓練 N 個分類器。`SGDClassifier` 和 `LogisticRegression` 預設使用此策略。

---

[來源: ch03] ### 範例 21: 檢視多類別決策分數

```python

---

[來源: ch03] # 取得單一樣本的決策分數
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)

---

[來源: ch03] # 找出分數最高的類別索引
class_id = some_digit_scores.argmax()

---

[來源: ch03] # 根據索引找到對應的類別名稱
svm_clf.classes_[class_id]
```

**✅ 程式碼逐行解析：**

1. `some_digit_scores = svm_clf.decision_function([some_digit])`: 對於多類別分類，`decision_function` 會回傳每個類別的分數。
2. `class_id = some_digit_scores.argmax()`: `argmax()` 函式會回傳分數最高（最可能）的類別的「索引」。
3. `svm_clf.classes_[class_id]`: `svm_clf.classes_` 屬性儲存了模型學習到的所有類別標籤的列表。我們使用 `class_id` 索引來取得最終的預測類別。

**🎯 重點摘要:**

---

[來源: ch03] sses_[class_id]`: `svm_clf.classes_` 屬性儲存了模型學習到的所有類別標籤的列表。我們使用 `class_id` 索引來取得最終的預測類別。

**🎯 重點摘要:**

- **核心功能**: 多類別分類的預測過程，是基於比較所有類別的決策分數，並選擇分數最高者作為最終結果。

---

---

[來源: ch03] ## 錯誤分析

💡 **實際应用情境：**
模型效能達到瓶頸時，我們需要深入分析它「錯在哪裡」，才能找到改進的方向。錯誤分析可以幫助我們識別模型的弱點，例如，模型是否經常混淆某幾個特定的類別？

---

[來源: ch03] ### 範例 22: 繪製多類別混淆矩陣

```python

---

[來源: ch03] # 從 sklearn.metrics 載入 ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay

---

[來源: ch03] # 取得交叉驗證的預測結果
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

---

[來源: ch03] # 繪製混淆矩陣
plt.rc('font', size=9)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()
```

![image](images/classification/confusion_matrix_plot_1.png)

**✅ 程式碼逐行解析：**

---

[來源: ch03] in_pred)
plt.show()
```

![image](images/classification/confusion_matrix_plot_1.png)

**✅ 程式碼逐行解析：**

1. `y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)`: 首先，我們需要一組「乾淨」的預測。這裡使用 `cross_val_predict` 確保每個樣本的預測都是由未見過該樣本的模型做出的。注意，這裡使用了經過特徵縮放的 `X_train_scaled`，這對 `SGDClassifier` 的效能至關重要。
2. `ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)`: 這是 Scikit-Learn 1.0 之後新增的便捷函式，可以直接從真實標籤和預測標籤繪製出帶有標籤和顏色條的混淆矩陣圖。

---

[來源: ch03] _predictions(y_train, y_train_pred)`: 這是 Scikit-Learn 1.0 之後新增的便捷函式，可以直接從真實標籤和預測標籤繪製出帶有標籤和顏色條的混淆矩陣圖。

**🎯 重點摘要:**

- **核心功能**: 視覺化多類別分類的混淆矩陣。
- **矩陣解讀**:
  - **對角線**: 對角線上的數字代表「正確預測」的樣本數。數字越亮（或越大），表示該類別的預測效果越好。
  - **非對角線**: 非對角線上的數字代表「錯誤預測」。例如，第 3 行第 5 列的數字表示有多少個真實的 `3` 被錯誤地預測成了 `5`。
- **分析**: 從圖中可以看出，數字 `5` 的對角線格子看起來比其他數字暗一些，表示模型在辨識 `5` 方面的準確率較低。此外，許多數字被錯誤地分類為 `8`。

---

[來源: ch03] ### 範例 23: 正規化混淆矩陣以分析錯誤率

```python

---

[來源: ch03] # 繪製按真實標籤正規化的混淆矩陣
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
plt.show()
```

![image](images/classification/confusion_matrix_plot_2.png)

**✅ 程式碼逐行解析：**

1. `normalize="true"`: 這個參數會將混淆矩陣的每一行（代表每個「真實類別」）除以該行的總和。這樣，矩陣中的每個值就變成了「比例」或「錯誤率」。
2. `values_format=".0%"`: 這個參數用於格式化矩陣中的數字，使其以百分比形式顯示。

---

[來源: ch03] 行（代表每個「真實類別」）除以該行的總和。這樣，矩陣中的每個值就變成了「比例」或「錯誤率」。
2. `values_format=".0%"`: 這個參數用於格式化矩陣中的數字，使其以百分比形式顯示。

**🎯 重點摘要:**

- **核心功能**: 透過正規化，我們可以更清楚地看到每個類別的「錯誤分佈」。
- **分析**:
  - **行分析**: 查看第 8 行，我們可以看到只有 85% 的 `8` 被正確分類，而有 10% 的 `8` 被錯誤地預測為 `1`, `3`, `5` 等。
  - **列分析**: 查看第 8 列，我們可以看到許多其他數字（特別是 `1`, `2`, `3`, `5`）都被錯誤地預測成了 `8`。
  - **對稱性**: 矩陣在 `3` 和 `5` 之間看起來也有些對稱，表示 `3` 和 `5` 經常被互相混淆。

---

[來源: ch03] ### 範例 24: 檢視個別錯誤範例

```python

---

[來源: ch03] # 找出真實為 3 但預測為 5 的圖片
X_35 = X_train[(y_train == '3') & (y_train_pred == '5')]

---

[來源: ch03] # 找出真實為 5 但預測為 3 的圖片
X_53 = X_train[(y_train == '5') & (y_train_pred == '3')]

---

[來源: ch03] # (此處省略繪圖程式碼，詳見 notebook)
```

![image](images/classification/error_analysis_digits_plot.png)

**✅ 程式碼逐行解析：**

1. `X_35 = X_train[(y_train == '3') & (y_train_pred == '5')]`: 使用布林索引來篩選出所有「真實標籤是 '3' 但模型預測是 '5'」的圖片。
2. `X_53 = X_train[(y_train == '5') & (y_train_pred == '3')]`: 使用布林索引來篩選出所有「真實標籤是 '5' 但模型預測是 '3'」的圖片。

**🎯 重點摘要:**

---

[來源: ch03] rain[(y_train == '5') & (y_train_pred == '3')]`: 使用布林索引來篩選出所有「真實標籤是 '5' 但模型預測是 '3'」的圖片。

**🎯 重點摘要:**

- **核心功能**: 深入到實際的錯誤樣本中，直觀地理解模型為什麼會犯錯。
- **分析**: 透過觀察這些被混淆的圖片，我們發現許多 `3` 和 `5` 的寫法確實非常相似，甚至人眼也很難區分。這表明這個問題本身具有一定的模糊性。對於線性模型 `SGDClassifier` 來說，這些細微的差異很難捕捉。
- **改進方向**: 針對這些特定的混淆對，我們可以嘗試收集更多難以區分的樣本來增強訓練集，或者使用更能捕捉複雜特徵的模型（如 CNN）。

---

---

[來源: ch03] ## 多標籤分類

💡 **實際應用情境：**
在某些情況下，我們希望一個樣本可以同時擁有多個標籤。例如，在一張包含多個人物的照片中，我們可能希望模型能辨識出所有出現的人物（例如，同時標記出 "愛麗絲"、"鮑勃" 和 "查理"）。這就是多標籤分類。

---

[來源: ch03] ### 範例 25: 建立多標籤目標

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

---

[來源: ch03] # 標籤 1: 數字是否大於等於 7
y_train_large = (y_train.astype('int8') >= 7)

---

[來源: ch03] # 標籤 2: 數字是否為奇數
y_train_odd = (y_train.astype('int8') % 2 == 1)

---

[來源: ch03] # 將兩個標籤合併成一個多標籤陣列
y_multilabel = np.c_[y_train_large, y_train_odd]

---

[來源: ch03] # 訓練一個 KNN 分類器
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

**✅ 程式碼逐行解析：**

---

[來源: ch03] 訓練一個 KNN 分類器
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

**✅ 程式碼逐行解析：**

1. `y_train_large = (y_train.astype('int8') >= 7)`: 建立第一個標籤，判斷數字是否 >= 7。
2. `y_train_odd = (y_train.astype('int8') % 2 == 1)`: 建立第二個標籤，判斷數字是否為奇數。
3. `y_multilabel = np.c_[y_train_large, y_train_odd]`: `np.c_` 是 NumPy 中一個方便的工具，可以將兩個一維陣列按列合併成一個二維陣列。現在，`y_multilabel` 的每一行都包含了兩個布林標籤。
4. `knn_clf.fit(X_train, y_multilabel)`: `KNeighborsClassifier` 本身就支援多標籤分類，所以我們可以像平常一樣直接訓練它。

---

[來源: ch03] 兩個布林標籤。
4. `knn_clf.fit(X_train, y_multilabel)`: `KNeighborsClassifier` 本身就支援多標籤分類，所以我們可以像平常一樣直接訓練它。

**🎯 重點摘要:**

- **核心功能**: 建立一個多標籤的目標 `y`，其中每個樣本可以對應多個 `True`/`False` 標籤。
- **模型支援**: 并非所有 Scikit-Learn 分類器都原生支援多標籤分類，但 `KNeighborsClassifier` 和 `RandomForestClassifier` 等模型可以。

---

[來源: ch03] ### 範例 26: 進行多標籤預測與評估

```python

---

[來源: ch03] # 預測 `some_digit` (數字 5) 的多個標籤
knn_clf.predict([some_digit])

---

[來源: ch03] # > array([[False,  True]])  (不大於等於 7, 是奇數)

---

[來源: ch03] # 評估多標籤分類的 F1 分數
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
```

**✅ 程式碼逐行解析：**

---

[來源: ch03] n, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
```

**✅ 程式碼逐行解析：**

1. `knn_clf.predict([some_digit])`: 預測結果是一個包含兩個布林值的陣列。對於數字 `5`，模型正確地預測出它「不是大數字」(`False`) 且「是奇數」(`True`)。
2. `f1_score(..., average="macro")`: 在評估多標籤分類時，我們需要指定 `average` 參數。
   - `"macro"`: 分別計算每個標籤的 F1 分數，然後取其「未加權」的平均值。所有標籤同等重要。
   - `"weighted"`: 同樣計算每個標籤的 F1 分數，但會根據每個標籤的樣本數（support）進行加權平均。

---

[來源: ch03] 算每個標籤的 F1 分數，然後取其「未加權」的平均值。所有標籤同等重要。
   - `"weighted"`: 同樣計算每個標籤的 F1 分數，但會根據每個標籤的樣本數（support）進行加權平均。

**🎯 重點摘要:**

- **核心功能**: 多標籤分類的評估需要考慮所有標籤的綜合表現。
- **`average` 參數的選擇**: 如果你認為所有標籤同等重要，使用 `"macro"`；如果某些標籤的樣本數遠多於其他標籤，且你希望給樣本數多的標籤更大權重，則使用 `"weighted"`。

---

---

[來源: ch03] ## 多輸出分類

💡 **實際應用情境：**
多輸出分類（Multioutput Classification）是多標籤分類的一般化形式，其中每個標籤可以是多類別的（而不僅僅是 `True`/`False`）。一個典型的例子是圖像去噪：模型的輸入是一張帶有噪點的圖片，輸出是「乾淨」的圖片。在這個場景下，模型的每個輸出（像素點）都可以有多個值（0-255 的灰階值）。

---

[來源: ch03] ### 範例 27: 建立多輸出資料集

```python

---

[來源: ch03] # 為訓練集和測試集添加噪點
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

---

[來源: ch03] # 目標是還原成原始的、乾淨的圖片
y_train_mod = X_train
y_test_mod = X_test
```

![image](images/classification/noisy_digit_example_plot.png)

**✅ 程式碼逐行解析：**

1. `noise = np.random.randint(0, 100, ...)`: 生成一個與 `X_train` 形狀相同的隨機整數矩陣，值域在 0 到 100 之間。
2. `X_train_mod = X_train + noise`: 將噪點加到原始圖片上，生成帶噪點的輸入特徵。
3. `y_train_mod = X_train`: 設定目標 `y` 為「原始的、乾淨的」圖片。

**🎯 重點摘要:**

---

[來源: ch03] ain + noise`: 將噪點加到原始圖片上，生成帶噪點的輸入特徵。
3. `y_train_mod = X_train`: 設定目標 `y` 為「原始的、乾淨的」圖片。

**🎯 重點摘要:**

- **核心功能**: 建立一個多輸出任務。輸入是 784 維的帶噪點像素，輸出是 784 維的乾淨像素。每個輸出維度（像素）都是一個多類別問題（0-255）。

---

[來源: ch03] # 訓練 KNN 分類器來去噪
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)

---

[來源: ch03] # 預測一張帶噪點圖片的乾淨版本
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
```

![image](images/classification/cleaned_digit_example_plot.png)

**✅ 程式碼逐行解析：**

1. `knn_clf.fit(X_train_mod, y_train_mod)`: 訓練 KNN 模型。模型學習從帶噪點的圖片 `X_train_mod` 映射到乾淨的圖片 `y_train_mod`。
2. `clean_digit = knn_clf.predict([X_test_mod[0]])`: 使用訓練好的模型對一張帶噪點的測試圖片進行預測。
3. `plot_digit(clean_digit)`: 繪製出去噪後的圖片。

---

[來源: ch03] knn_clf.predict([X_test_mod[0]])`: 使用訓練好的模型對一張帶噪點的測試圖片進行預測。
3. `plot_digit(clean_digit)`: 繪製出去噪後的圖片。

**🎯 重點摘要:**

- **核心功能**: 多輸出分類器可以處理複雜的結構化輸出，例如圖像。
- **結論**: 儘管這是一個非常困難的任務，但 KNN 分類器仍然成功地移除了大部分噪點，並還原了數字的基本形狀。

---

---

[來源: ch03] ### 1. MNIST 97% 準確率分類器

嘗試建立一個在 MNIST 測試集上達到 97% 準確率的分類器。一個提示是 `KNeighborsClassifier` 在這個任務上表現得相當不錯。你可能需要使用網格搜尋（`GridSearchCV`）來找到最佳的超參數組合。

---

[來源: ch03] 試集上達到 97% 準確率的分類器。一個提示是 `KNeighborsClassifier` 在這個任務上表現得相當不錯。你可能需要使用網格搜尋（`GridSearchCV`）來找到最佳的超參數組合。

```python
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

```python

---

[來源: ch03] # 最佳超參數
grid_search.best_params_

---

[來源: ch03] # 最佳分數
grid_search.best_score_

---

[來源: ch03] # 使用最佳估計器在測試集上評估
from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)
accuracy_score(y_test, y_pred)
```

透過網格搜尋，我們可以找到 `weights='distance'` 和 `n_neighbors=4` 的組合，在交叉驗證中達到了約 97.1% 的準確率，並在最終的測試集上達到了 97.14% 的準確率。

---

[來源: ch03] ### 2. 資料增強 (Data Augmentation)

編寫一個函式，可以將 MNIST 圖像向任何方向（上、下、左、右）移動一個像素。然後，對於訓練集中的每張圖片，創建四個移動過的副本（每個方向一個），並將它們添加到訓練集中。最後，在這個擴展的訓練集上訓練你的最佳模型，並在測試集上評估它。你應該會觀察到模型效能有顯著提升。

```python
from scipy.ndimage import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])
```

---

[來源: ch03] d_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])
```

```python

---

[來源: ch03] # 創建增強後的資料集
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

---

[來源: ch03] X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

```

```

python

---

[來源: ch03] # 在增強後的資料集上重新訓練
knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(X_train_augmented, y_train_augmented)

y_pred = knn_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```

透過資料增強，我們將訓練集的大小擴大了五倍。在這個擴展的資料集上重新訓練 `KNeighborsClassifier`，測試集上的準確率從 97.14% 提升到了 97.56%。這看似微小的提升，實際上讓模型的錯誤率降低了約 17%，效果顯著。

---

[來源: ch03] ### 3. 鐵達尼號資料集挑戰

挑戰：處理鐵達尼號資料集。目標是預測一個乘客是否能夠存活（`Survived`）。

**步驟：**

---

[來源: ch03] 1. **載入資料**: 使用 Pandas 載入 `train.csv` 和 `test.csv`。
2. **探索資料**:
    - `head()`: 查看前幾行。
    - `info()`: 檢查缺失值和資料類型。`Age`, `Cabin`, `Embarked` 有缺失值。`Cabin` 缺失嚴重，可能需要放棄。
    - `describe()`: 查看數值屬性的統計摘要。
    - `value_counts()`: 查看類別屬性的分佈，如 `Sex`, `Pclass`, `Embarked`。
3. **建立前處理管線**:
    - **數值管線**: 處理 `Age`, `SibSp`, `Parch`, `Fare`。使用 `SimpleImputer` 填充 `Age` 的缺失值（使用中位數），然後使用 `StandardScaler` 進行特徵縮

---

[來源: ch03]  `Age`, `SibSp`, `Parch`, `Fare`。使用 `SimpleImputer` 填充 `Age` 的缺失值（使用中位數），然後使用 `StandardScaler` 進行特徵縮放。
    - **類別管線**: 處理 `Pclass`, `Sex`, `Embarked`。使用 `SimpleImputer` 填充 `Embarked` 的缺失值（使用最頻繁值），然後使用 `OneHotEncoder` 進行獨熱編碼。
    - **合併管線**: 使用 `ColumnTransformer` 將數值和類別管線合併。

```python

---

[來源: ch03] # (此處省略了詳細的資料載入與前處理管線程式碼，詳見 notebook)

---

[來源: ch03] # 訓練一個隨機森林分類器
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train_prepared, y_train)

---

[來源: ch03] # 評估模型
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_clf, X_train_prepared, y_train, cv=10)
forest_scores.mean()

---

[來源: ch03] # > 約 81.2% 的準確率
```

我們建立了一個完整的前處理管線，並使用 `RandomForestClassifier` 進行訓練。透過 10-fold 交叉驗證，模型的平均準確率約為 81.2%。

---

[來源: ch03] ### 4. 垃圾郵件分類器

挑戰：建立一個垃圾郵件分類器。

**步驟：**

1. **獲取資料**: 下載並解壓縮 [SpamAssassin 公開資料集](https://spamassassin.apache.org/old/publiccorpus/)。
2. **探索與前處理**: 探索電子郵件的結構，提取有用的特徵（如字詞頻率、主題長度等）。
3. **建立管線**: 建立一個完整的前處理管線，包括文字清理、特徵提取和縮放。
4. **訓練模型**: 使用多種不同的分類器進行訓練和比較。
5. **評估**: 使用交叉驗證評估模型的效能。

透過以上步驟，我們可以建立一個有效的垃圾郵件分類器，並深入了解文字分類的挑戰與技巧。

---

---

[來源: ch03] ## 總結與最佳實踐

本章我們深入探討了機器學習中的分類任務，從基礎的二元分類到進階的多標籤、多輸出分類。以下是關鍵的總結和最佳實踐建議：

---

[來源: ch03] ### 關鍵學習重點

1. **選擇適當的評估指標**:
   - **準確率**並非萬能，在不平衡資料集上可能會誤導
   - **精確率 vs 召回率**：根據業務需求選擇重點
   - **F1 分數**：當你需要平衡精確率和召回率時
   - **ROC 曲線與 AUC**：適合比較不同模型的整體表現

2. **模型選擇指南**:
   - **SGDClassifier**：適合大型資料集，訓練速度快
   - **SVM (SVC)**：適合中小型資料集，支援非線性分類
   - **RandomForestClassifier**：通常表現良好，較少需要調參
   - **KNeighborsClassifier**：簡單有效，適合作為基準模型

---

[來源: ch03] 料集，支援非線性分類
   - **RandomForestClassifier**：通常表現良好，較少需要調參
   - **KNeighborsClassifier**：簡單有效，適合作為基準模型

3. **資料前處理的重要性**:
   - **特徵縮放**：對於距離導向的演算法（如 SVM、KNN）特別重要
   - **資料增強**：可以有效提升模型的泛化能力
   - **交叉驗證**：確保模型評估的可信度

---

[來源: ch03] ### 實務開發建議

1. **從簡單開始**：
   - 先建立一個簡單的基準模型
   - 逐步增加複雜度，觀察效能提升

2. **重視錯誤分析**：
   - 使用混淆矩陣識別模型的弱點
   - 分析錯誤案例，指導特徵工程

3. **適當的模型選擇**：
   - 考慮資料大小、維度和問題複雜度
   - 平衡模型複雜度與可解釋性

4. **持續監控與改進**：
   - 定期檢查模型在新資料上的表現
   - 根據業務回饋調整評估指標

---

[來源: ch03] ### 常見陷阱與注意事項

1. **過度擬合**：使用交叉驗證和適當的正則化
2. **資料洩漏**：確保測試集完全獨立
3. **評估偏差**：在不平衡資料上選擇合適的評估指標
4. **特徵工程**：領域知識比演算法選擇更重要

---

---

[來源: ch03] ### Q1: 何時應該使用精確率，何時使用召回率？

**A**: 這取決於「偽陽性」和「偽陰性」的代價：
- **高精確率優先**：當偽陽性代價很高時（例如：垃圾郵件過濾，將重要郵件誤判為垃圾會很糟糕）
- **高召回率優先**：當偽陰性代價很高時（例如：癌症檢測，漏診比誤診更危險）
- **F1 分數**：當兩者同等重要時

---

[來源: ch03] ### Q2: SGD 與 SVM 有何區別？何時選擇哪一個？

**A**: 
- **SGD（隨機梯度下降）**：
  - 是一種**最佳化演算法**，可以訓練多種模型（包括線性 SVM）
  - 適合**大型資料集**，記憶體效率高
  - 支援**線上學習**（incremental learning）

- **SVM（支援向量機）**：
  - 是一種**特定的機器學習演算法**
  - 使用核技巧支援**非線性分類**
  - 適合**中小型資料集**，在高維空間表現優異

**選擇建議**：
- 資料量大（>100K 樣本）→ SGDClassifier
- 資料量中等且需要非線性邊界 → SVC with RBF kernel
- 需要機率輸出 → LogisticRegression 或 RandomForestClassifier

---

[來源: ch03] ### Q3: 如何處理不平衡的資料集？

**A**: 多種策略可以組合使用：

1. **評估指標調整**：
   - 使用精確率、召回率、F1 分數而非準確率
   - 關注 ROC-AUC 或 Precision-Recall AUC

2. **資料層面**：
   - **過度採樣**：SMOTE、ADASYN
   - **欠採樣**：隨機欠採樣、Tomek links
   - **資料增強**：針對少數類生成更多樣本

3. **演算法層面**：
   - 調整 `class_weight='balanced'` 參數
   - 使用成本敏感學習
   - 集成方法（如 BalancedRandomForestClassifier）

---

[來源: ch03] ### Q4: 交叉驗證的 fold 數應該設多少？

**A**: 常見的選擇：
- **k=5**：計算效率與評估品質的良好平衡，適合大多數情況
- **k=10**：更穩健的評估，但計算成本較高
- **留一交叉驗證 (LOOCV)**：資料集很小時（<1000 樣本）
- **分層交叉驗證**：不平衡資料集的首選

**經驗法則**：資料越少，fold 數可以越高；資料越多，fold 數可以適當減少。

---

[來源: ch03] ### Q5: 如何選擇合適的決策閾值？

**A**: 根據業務需求調整：

1. **分析 PR 曲線和 ROC 曲線**：找到最佳的精確率/召回率平衡點
2. **使用 GridSearchCV**：將閾值當作超參數進行搜尋
3. **成本分析**：計算不同閾值下的業務成本
4. **驗證集調參**：在驗證集上測試不同閾值的效果

```python

---

[來源: ch03] # 範例：找到 90% 精確率對應的閾值
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
idx_90_precision = np.argmax(precisions >= 0.90)
threshold_90_precision = thresholds[idx_90_precision]
```

---

[來源: ch03] ### Q6: 多類別分類中，OvO 和 OvR 哪個更好？

**A**: 各有優缺點：

**One-vs-One (OvO)**：
- ✅ 每個分類器只需要處理兩個類別的資料
- ✅ 對不平衡資料更穩健
- ❌ 需要訓練 N(N-1)/2 個分類器
- 🎯 適合：SVM 等訓練時間與樣本數呈超線性關係的演算法

**One-vs-Rest (OvR)**：
- ✅ 只需要訓練 N 個分類器
- ✅ 訓練和預測都比較快
- ❌ 對不平衡資料較敏感
- 🎯 適合：線性模型或大型資料集

**Scikit-Learn 的預設選擇通常是最佳的**，但你可以手動指定：
```python
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

---

[來源: ch03] # 強制使用 OvO
ovo_clf = OneVsOneClassifier(SGDClassifier())

---

[來源: ch03] # 強制使用 OvR  
ovr_clf = OneVsRestClassifier(SGDClassifier())
```

---

---

[來源: ch03] ### 主要標籤

`#Python` `#程式設計` `#教學` `#MachineLearning` `#Classification` `#ScikitLearn` `#DataScience`

---

[來源: ch03] ### 技術標籤

`#SGD` `#SVM` `#RandomForest` `#ConfusionMatrix` `#Precision` `#Recall` `#ROC` `#AUC` `#F1Score`

---

[來源: ch03] ### 學習標籤

`#編程` `#開發` `#技術分享` `#學習筆記` `#程式開發者` `#軟體工程` `#AI` `#人工智慧`

---

[來源: ch03] ### 社群標籤

`#ThreadsTech` `#開發者社群` `#台灣開發者` `#程式學習` `#ML入門` `#實戰教學`

---

[來源: ch03] ### 專題標籤

`#MNIST` `#HandsOnML` `#分類演算法` `#模型評估` `#特徵工程` `#資料科學實戰`

---

[來源: ch03 | 類型: cheatsheet] # Ch03 速查表：Classification

> **核心主旨**：分類任務的評估指標 —— 準確率只是起點，Precision/Recall/ROC-AUC 才是關鍵。

---

---

[來源: ch03 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Binary Classification | 二元分類，輸出 0 或 1 | 垃圾郵件偵測、詐欺偵測 |
| Confusion Matrix | TP/FP/TN/FN 的完整統計 | 模型錯誤分析 |
| Precision | $\frac{TP}{TP+FP}$：預測為正中有多少真的是正 | 假陽性代價高時（如垃圾郵件） |
| Recall (Sensitivity) | $\frac{TP}{TP+FN}$：真正的正有多少被找出來 | 假陰性代價高時（如癌症篩檢） |
| F1 Score | Precision 與 Recall 的調和平均 | 兩者都重要時 |
| ROC-AUC | 不同閾值下的 TPR vs FPR 曲線面積 | 比較分類器整體能力 |
| PR Curve | Precision-Recall 曲線 | 類別極度不平衡時優先看 |
| OvR / OvO | 多類別策略：One-vs-Rest / One-vs-One | 將二元分類器用於多類別 |


---

[來源: ch03 | 類型: cheatsheet] | Precision-Recall 曲線 | 類別極度不平衡時優先看 |
| OvR / OvO | 多類別策略：One-vs-Rest / One-vs-One | 將二元分類器用於多類別 |


---

---

[來源: ch03 | 類型: cheatsheet] | sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `SGDClassifier` | `loss="hinge"/"log_loss"`, `random_state=42` | 線性分類器（支援 SGD） |
| `cross_val_predict` | `cv=3`, `method="decision_function"` | 取每個樣本的 OOF 預測 |
| `confusion_matrix` | – | 輸出 TP/FP/TN/FN 矩陣 |
| `precision_score` | `average="binary"/"macro"/"weighted"` | 精確率 |
| `recall_score` | `average="binary"` | 召回率 |
| `f1_score` | – | F1 Score |
| `roc_auc_score` | – | ROC AUC |
| `precision_recall_curve` | – | PR 曲線資料點 |
| `roc_curve` | – | ROC 曲線資料點 |
| `classification_report` | – | 一次印出所有指標 |
| `RandomForestClassifier` | `n_estimators=100` | 隨機森林分類 |


---

[來源: ch03 | 類型: cheatsheet] `classification_report` | – | 一次印出所有指標 |
| `RandomForestClassifier` | `n_estimators=100` | 隨機森林分類 |


---

---

[來源: ch03 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt

---

[來源: ch03 | 類型: cheatsheet] # 訓練分類器
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)  # y_train_5: True/False

---

[來源: ch03 | 類型: cheatsheet] # 交叉驗證取 OOF 預測
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5,
                              cv=3, method="decision_function")

---

[來源: ch03 | 類型: cheatsheet] # 混淆矩陣與各項指標
y_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_pred)
print(f"Precision: {precision_score(y_train_5, y_pred):.3f}")
print(f"Recall:    {recall_score(y_train_5, y_pred):.3f}")
print(f"F1:        {f1_score(y_train_5, y_pred):.3f}")
print(f"ROC AUC:   {roc_auc_score(y_train_5, y_scores):.3f}")
print(classification_report(y_train_5, y_pred))

---

[來源: ch03 | 類型: cheatsheet] # PR 曲線
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plt.plot(recalls, precisions)
plt.xlabel("Recall"); plt.ylabel("Precision")

---

[來源: ch03 | 類型: cheatsheet] # 調整決策閾值
threshold_90_precision = thresholds[precisions >= 0.90][0]
y_pred_90 = (y_scores >= threshold_90_precision)
```

---

---

[來源: ch03 | 類型: cheatsheet] ## 4. 常見陷阱

- **準確率的陷阱**：極度不平衡資料（如 99% 負例），全預測負例也有 99% 準確率，但完全沒用。
- **`cross_val_score` vs `cross_val_predict`**：前者給摘要分數，後者給每個樣本的 OOF 預測（才能畫 PR/ROC 曲線）。
- **多類別指標 `average` 參數**：`"macro"` 對每個類別平均處理，`"weighted"` 依樣本數加權，二元分類用 `"binary"`。
- **`roc_auc_score` 需要機率/分數**：不能直接傳 0/1 預測，要傳 `decision_function()` 或 `predict_proba()` 的輸出。

---

---

[來源: ch03 | 類型: cheatsheet] ## 5. 決策指南

```
評估指標的選擇：
├── 假陽性代價高（如垃圾郵件誤判） → 最大化 Precision
├── 假陰性代價高（如癌症漏診）     → 最大化 Recall
├── 兩者平衡                        → F1 Score
└── 整體比較多個模型                → ROC-AUC

類別不平衡時：
├── 優先看 PR Curve（而非 ROC Curve）
├── 使用 class_weight="balanced"
└── 考慮 SMOTE 過採樣（需 imblearn 套件）
```

**精確率 vs 召回率的取捨**：
- 一個分類器不能同時最大化 precision 和 recall
- 提高閾值 → precision ↑, recall ↓
- 降低閾值 → precision ↓, recall ↑

---

[來源: ch03 | 類型: handout] # 課程講義：分類 (Chapter 03)

本章聚焦於機器學習中的**分類問題**，以 MNIST 手寫數字資料集為實驗場域。你將發現：看似直觀的「準確率」在面對不平衡資料時完全失效，而精確率、召回率、F1 分數、ROC 曲線才是評估分類器的真正工具。掌握這些指標，是工業界機器學習工程師的必備技能。

---

---

[來源: ch03 | 類型: handout] ### 理論背景

二元分類（Binary Classification）：將樣本判斷為正類（Positive）或負類（Negative）。

**混淆矩陣 (Confusion Matrix)**：

$$\begin{pmatrix} TN & FP \\ FN & TP \end{pmatrix}$$

- **TP (True Positive)**：實際為正，預測為正（正確）
- **TN (True Negative)**：實際為負，預測為負（正確）
- **FP (False Positive)**：實際為負，預測為正（型一誤差，False Alarm）
- **FN (False Negative)**：實際為正，預測為負（型二誤差，Miss）

**準確率的陷阱**：若正類只佔 1%（不平衡資料），一個永遠預測負類的分類器準確率高達 99%，但完全無用！

---

[來源: ch03 | 類型: handout] *FN (False Negative)**：實際為正，預測為負（型二誤差，Miss）

**準確率的陷阱**：若正類只佔 1%（不平衡資料），一個永遠預測負類的分類器準確率高達 99%，但完全無用！

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

---

[來源: ch03 | 類型: handout] ### 核心代碼

```python
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

---

[來源: ch03 | 類型: handout] # 載入 MNIST
mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target

---

[來源: ch03 | 類型: handout] # 分割（MNIST 前 60000 為訓練集）
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

---

[來源: ch03 | 類型: handout] # 建立二元分類器（判斷是否為數字 5）
y_train_5 = (y_train == "5")
y_test_5  = (y_test  == "5")

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

---

[來源: ch03 | 類型: handout] # 用交叉驗證取得預測（不偷看測試集）
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)
ConfusionMatrixDisplay(cm).plot()
```

---

[來源: ch03 | 類型: handout] ### 補充練習 1

**理論題：** 醫療診斷中（陽性=有病），FP 和 FN 哪個代價更高？對應到垃圾郵件過濾（陽性=垃圾郵件），答案有何不同？

**實作題：** 建立一個「笨分類器」：永遠預測 `False`（非 5）。計算它的準確率，並與 `SGDClassifier` 比較，解釋為何準確率在此不是好指標。

---

---

[來源: ch03 | 類型: handout] ### 理論背景

從混淆矩陣衍生的更有意義的指標：

$$\text{精確率 (Precision)} = \frac{TP}{TP + FP}$$

$$\text{召回率 (Recall) / 靈敏度 (Sensitivity)} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

**精確率-召回率的權衡**：

- 降低決策閾值 → 召回率↑、精確率↓（找到更多正例，但誤報增加）
- 提高決策閾值 → 精確率↑、召回率↓（更保守，但可能漏掉正例）

選擇哪個指標取決於業務目標：

---

[來源: ch03 | 類型: handout] *精確率-召回率的權衡**：

- 降低決策閾值 → 召回率↑、精確率↓（找到更多正例，但誤報增加）
- 提高決策閾值 → 精確率↑、召回率↓（更保守，但可能漏掉正例）

選擇哪個指標取決於業務目標：

- 影片內容審查（避免誤殺好內容）→ 偏重精確率
- 癌症篩查（避免漏掉病患）→ 偏重召回率

---

[來源: ch03 | 類型: handout] ### 核心代碼

```python
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              precision_recall_curve, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt

---

[來源: ch03 | 類型: handout] # 基本指標
print(f"Precision: {precision_score(y_train_5, y_train_pred):.3f}")
print(f"Recall:    {recall_score(y_train_5, y_train_pred):.3f}")
print(f"F1 Score:  {f1_score(y_train_5, y_train_pred):.3f}")

---

[來源: ch03 | 類型: handout] # 取得決策分數（非二元預測）
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5,
                             cv=3, method="decision_function")

---

[來源: ch03 | 類型: handout] # 精確率-召回率曲線
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1],    "g-",  label="Recall")
plt.xlabel("Threshold")
plt.legend()
plt.title("Precision-Recall vs Threshold")
plt.grid()

---

[來源: ch03 | 類型: handout] # 設定目標精確率，找出對應閾值
idx_90_precision = (precisions >= 0.90).argmax()
threshold_90_precision = thresholds[idx_90_precision]
y_pred_90 = (y_scores >= threshold_90_precision)
print(f"Precision @90: {precision_score(y_train_5, y_pred_90):.3f}")
print(f"Recall  @90:   {recall_score(y_train_5, y_pred_90):.3f}")
```

---

[來源: ch03 | 類型: handout] ### 補充練習 2

**理論題：** 若模型的精確率為 0.8、召回率為 0.6，計算 $F_1$ 分數。若業務目標是「召回率至少 95%，在此前提下精確率越高越好」，應如何設定閾值？

**實作題：** 繪製 Precision-Recall 曲線（以召回率為 x 軸，精確率為 y 軸），並計算曲線下面積（Average Precision）。找出使 $F_1$ 最大化的閾值。

---

---

[來源: ch03 | 類型: handout] ### 理論背景

**ROC 曲線 (Receiver Operating Characteristic)**：

- x 軸：偽陽性率（FPR）$= \frac{FP}{TN + FP}$，即 1 - 特異度
- y 軸：真陽性率（TPR）$= \frac{TP}{TP + FN}$，即召回率

**AUC (Area Under Curve)**：

- AUC = 1.0 → 完美分類器
- AUC = 0.5 → 隨機猜測（對角線）
- AUC 越大越好

**ROC vs Precision-Recall 曲線選擇**：

- 正負類別**嚴重不平衡**時，PR 曲線更具鑑別力（ROC 曲線可能過於樂觀）
- 兩類別比例均衡時，ROC 曲線更常用

---

[來源: ch03 | 類型: handout] ### 核心代碼

```python
from sklearn.metrics import roc_curve, roc_auc_score

---

[來源: ch03 | 類型: handout] # ROC 曲線
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, "b-",  linewidth=2, label=f"SGD AUC={roc_auc_score(y_train_5, y_scores):.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR = Recall)")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

---

[來源: ch03 | 類型: handout] # 與 RandomForestClassifier 比較
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(rf_clf, X_train, y_train_5,
                                   cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]  # 正類的機率
print(f"RF ROC AUC: {roc_auc_score(y_train_5, y_scores_forest):.4f}")
```

---

[來源: ch03 | 類型: handout] ### 補充練習 3

**理論題：** 在同一張圖上繪製 SGDClassifier 與 RandomForestClassifier 的 ROC 曲線，從圖形判斷哪個分類器更好。AUC 的物理意義是什麼（從機率角度解釋）？

**實作題：** 使用 `classification_report` 輸出完整的分類報告（Precision、Recall、F1 for each class），並解讀 `macro avg` 與 `weighted avg` 的差異。

---

---

[來源: ch03 | 類型: handout] ### 理論背景

**多類別分類 (Multi-class)**：每個樣本屬於一個類別（如 MNIST 的 0-9）。

- **OvR (One-vs-Rest)**：訓練 10 個二元分類器，取分數最高者
- **OvO (One-vs-One)**：訓練 $C(n,2)$ 個分類器（每對類別一個），適合 SVM 等對樣本數敏感的算法

**多標籤分類 (Multi-label)**：每個樣本可同時屬於多個類別（如圖片標籤：{人臉, 微笑, 帽子}）。

**多輸出分類 (Multi-output)**：每個輸出可以是多類別（如影像去噪：每個像素的灰階值）。

---

[來源: ch03 | 類型: handout] ### 核心代碼

```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

---

[來源: ch03 | 類型: handout] # SVC 預設使用 OvO 策略
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])  # 用子集加速
print(svm_clf.predict([X_test[0]]))          # 預測數字類別

---

[來源: ch03 | 類型: handout] # 多標籤分類（判斷是否為大數字 & 是否為奇數）
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train.astype(int) >= 7)
y_train_odd   = (y_train.astype(int) % 2 == 1)
y_multilabel  = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([X_test[0]]))  # [[True, False]] → 7 是大數字但不是奇數（非7的情況）
```

---

[來源: ch03 | 類型: handout] ### 補充練習 4

**理論題：** 對 10 個類別的多類別問題，OvR 需要訓練幾個分類器？OvO 需要幾個？當訓練速度很慢時，哪個策略比較好？

**實作題：** 對完整 MNIST（10 類）訓練 `SGDClassifier`（預設使用 OvR），計算測試集的混淆矩陣，繪製熱力圖（`plt.matshow`），找出最常被混淆的數字對。

---

---

[來源: ch03 | 類型: handout] ## 結論

本章揭示了分類問題評估的完整工具箱：

- **混淆矩陣**是一切指標的起點
- **精確率、召回率、F1** 比準確率更能反映不平衡資料的模型效能
- **ROC / AUC** 是模型比較的通用基準
- **閾值調整**允許根據業務目標在精確率與召回率之間取捨

下一章（Ch04）深入線性模型的數學原理，理解梯度下降如何最小化損失函數。

---

---

[來源: ch03 | 類型: handout] ## 課後作業

**作業：MNIST 錯誤分析與改進**

1. 訓練 `SGDClassifier` 在完整 MNIST（10 類），計算測試集 `f1_score(average="macro")`。
2. 繪製**正規化後的混淆矩陣**（每列除以該類別的真實樣本數），用顏色深淺顯示，找出模型最常搞混的兩個數字。
3. 針對最容易混淆的那對數字（例如 4 和 9），查看被分類錯誤的圖片，思考：能否透過特徵工程（如計算影像的某種統計量）幫助區分？

---

[來源: ch03 | 類型: tutorial] [標題: Python 分類實戰：從 MNIST 到 ROC 曲線的完整機器學習教學 | 描述: 使用 MNIST 資料集完整實作分類任務：SGDClassifier、混淆矩陣、精確率/召回率、ROC 曲線、多類別分類、多標籤分類。包含逐行程式碼解析，適合 ML 學習者。 | 關鍵字: Python, 機器學習, 分類, MNIST, Scikit-Learn, 混淆矩陣, ROC曲線, AUC, SGD, 精確率, 召回率]
# Python 分類實戰：MNIST 到 ROC 曲線完整指南

分類（Classification）是監督學習中最常見的任務類型。本教學以 **MNIST 手寫數字辨識**為主線，帶你從零開始建構二元分類器，深入了解如何評估模型效能，並延伸至多類別、多標籤、多輸出分類。

---

[來源: ch03 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **準確率（Accuracy）** 不是萬能指標——在不平衡資料集上可能嚴重誤導
- **混淆矩陣（Confusion Matrix）** 提供比準確率更細緻的效能視圖
- **精確率/召回率（Precision/Recall）** 是評估不平衡分類問題的核心指標
- **ROC-AUC** 評估分類器在所有閾值下的整體表現
- `cross_val_predict` + `confusion_matrix` 是標準的模型診斷流程

---

---

[來源: ch03 | 類型: tutorial] ## MNIST 資料集

💡 **實際應用情境：** MNIST 被稱為 ML 的「Hello World」——70,000 張 28×28 灰階手寫數字圖片，是評估分類演算法的標準基準資料集。

---

[來源: ch03 | 類型: tutorial] ### 範例 1: 載入並探索 MNIST

```python
from sklearn.datasets import fetch_openml
import numpy as np

---

[來源: ch03 | 類型: tutorial] # 載入 MNIST（首次需要聯網下載，會快取到本地）
mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X, y = mnist["data"], mnist["target"]

print(f"特徵矩陣形狀: {X.shape}")  # (70000, 784) — 70k 張 28×28 圖片
print(f"標籤形狀: {y.shape}")       # (70000,) — 字串類別 '0'~'9'

---

[來源: ch03 | 類型: tutorial] # MNIST 已按慣例分割：前 60k 為訓練，後 10k 為測試
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

---

[來源: ch03 | 類型: tutorial] # 標準化：將像素值從 [0, 255] 縮放到 [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0
```

**✅ 程式碼逐行解析：**

1. `fetch_openml("mnist_784", as_frame=False)`: 從 OpenML 下載資料集，`as_frame=False` 回傳 numpy 陣列
2. `X.shape = (70000, 784)`: 784 = 28 × 28，每張圖片被展平為一維向量
3. `y` 是字串類型（`'0'`, `'1'`, ...），需要在訓練前轉換

**🎯 重點摘要:**

- 像素值標準化（÷255）能加速梯度下降收斂
- MNIST 的固定分割使不同論文的結果可直接比較

---

---

[來源: ch03 | 類型: tutorial] ## 訓練二元分類器 (Binary Classifier)

💡 **實際應用情境：** 二元分類是最基礎的分類形式。以 MNIST 為例，「是否為數字 5」就是一個典型的二元問題，等同於工業檢測中的「良品/不良品」判斷。

---

[來源: ch03 | 類型: tutorial] ### 範例 2: SGDClassifier 辨識數字 5

```python
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

---

[來源: ch03 | 類型: tutorial] # 建立二元標籤：是 5 → True，否則 → False
y_train_5 = (y_train == "5")
y_test_5  = (y_test  == "5")

---

[來源: ch03 | 類型: tutorial] # SGDClassifier（隨機梯度下降，適合大型資料集）
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train_5)

---

[來源: ch03 | 類型: tutorial] # 預測單張圖片
some_digit = X_test[0]
print(f"預測: {'是5' if sgd_clf.predict([some_digit])[0] else '不是5'}")

---

[來源: ch03 | 類型: tutorial] # 交叉驗證準確率
scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(f"3-Fold 準確率: {scores.mean():.4f}")  # 約 97%

---

[來源: ch03 | 類型: tutorial] # 危險！「全預測為非5」的笨模型也有 ~90% 準確率
```

**✅ 程式碼逐行解析：**

1. `y_train == "5"`: 向量化比較，建立布林陣列（10% 為 True）
2. `SGDClassifier(random_state=42)`: SGD 每次看一個樣本更新梯度，速度快，適合大型資料集
3. 97% 準確率看起來很好，但「全說不是5」也有90%——說明準確率在不平衡資料時會誤導

**🎯 重點摘要:**

- 準確率在類別不平衡（Imbalanced Classes）時毫無意義
- 需要更好的評估指標：混淆矩陣、精確率、召回率

---

---

[來源: ch03 | 類型: tutorial] ### 範例 3: 混淆矩陣 (Confusion Matrix)

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

---

[來源: ch03 | 類型: tutorial] # 用交叉驗證取得每個樣本的「乾淨」預測（每個樣本只在驗證集時被預測一次）
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

---

[來源: ch03 | 類型: tutorial] # 混淆矩陣
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

---

[來源: ch03 | 類型: tutorial] # 例如：[[53892, 687],   ← 非5：53892 正確，687 誤報為5

---

[來源: ch03 | 類型: tutorial] #        [1891, 3530]]  ← 是5：1891 漏報，3530 正確

---

[來源: ch03 | 類型: tutorial] # 視覺化
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["非5", "是5"])
disp.plot(cmap="Blues")
plt.title("SGDClassifier 混淆矩陣")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `cross_val_predict(cv=3)`: 用 3-Fold CV 取得訓練集每個樣本的「跨折預測」，比直接在訓練集預測更能反映真實效能
2. 混淆矩陣的四個格：**TP**（真陽性）、**TN**（真陰性）、**FP**（假陽性/誤報）、**FN**（假陰性/漏報）

**🎯 重點摘要:**

- **FP（假警報）** 和 **FN（漏報）** 的業務代價往往不同（如醫療診斷漏報比假警報更嚴重）
- 混淆矩陣讓你看清模型的「錯誤模式」

---

---

[來源: ch03 | 類型: tutorial] ### 範例 4: 精確率、召回率、F1

```python
from sklearn.metrics import precision_score, recall_score, f1_score

---

[來源: ch03 | 類型: tutorial] # 精確率（Precision）= TP / (TP + FP)：預測為正時，有多少真的是正的
precision = precision_score(y_train_5, y_train_pred)
print(f"精確率: {precision:.4f}")  # 約 0.84 — 預測為5的中，84%真的是5

---

[來源: ch03 | 類型: tutorial] # 召回率（Recall）= TP / (TP + FN)：真正是正的，被找出多少
recall = recall_score(y_train_5, y_train_pred)
print(f"召回率: {recall:.4f}")  # 約 0.65 — 真正是5的，只找出65%

---

[來源: ch03 | 類型: tutorial] # F1 分數：精確率和召回率的調和平均數
f1 = f1_score(y_train_5, y_train_pred)
print(f"F1 分數: {f1:.4f}")  # 約 0.73

---

[來源: ch03 | 類型: tutorial] # 精確率/召回率曲線
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

---

[來源: ch03 | 類型: tutorial] # 找到精確率 ≥ 90% 的閾值
idx_90_precision = (precisions >= 0.90).argmax()
print(f"90% 精確率對應閾值: {thresholds[idx_90_precision]:.2f}")
print(f"對應召回率: {recalls[idx_90_precision]:.4f}")
```

**✅ 程式碼逐行解析：**

1. `precision_score`: 衡量「說是5有多可靠」——假警報少則精確率高
2. `recall_score`: 衡量「把5找完整有多好」——漏報少則召回率高
3. F1 = 2 × (P × R) / (P + R): 兩者的調和平均，當 P 和 R 都高時 F1 才高
4. `decision_function`: 回傳原始決策分數（而非二元預測），用於 PR 曲線計算

**🎯 重點摘要:**

- **精確率 vs 召回率的取捨**：提高閾值 → 精確率↑召回率↓；降低閾值 → 精確率↓召回率↑
- 業務決策決定如何設定閾值：垃圾郵件過濾（希望精確率高）vs 癌症篩查（希望召回率高）

---

---

[來源: ch03 | 類型: tutorial] ### 範例 5: ROC 曲線比較兩個分類器

```python
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

---

[來源: ch03 | 類型: tutorial] # SGD 的 ROC
fpr_sgd, tpr_sgd, _ = roc_curve(y_train_5, y_scores)
auc_sgd = roc_auc_score(y_train_5, y_scores)

---

[來源: ch03 | 類型: tutorial] # 隨機森林（用 predict_proba 取機率分數）
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)
fpr_forest, tpr_forest, _ = roc_curve(y_train_5, y_probas_forest[:, 1])
auc_forest = roc_auc_score(y_train_5, y_probas_forest[:, 1])

---

[來源: ch03 | 類型: tutorial] # 繪製比較圖
plt.figure(figsize=(8, 6))
plt.plot(fpr_sgd,    tpr_sgd,    "b-", linewidth=2, label=f"SGD (AUC={auc_sgd:.3f})")
plt.plot(fpr_forest, tpr_forest, "r-", linewidth=2, label=f"Random Forest (AUC={auc_forest:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="隨機分類器 (AUC=0.5)")
plt.xlabel("False Positive Rate (偽陽率)")
plt.ylabel("True Positive Rate (真陽率 / 召回率)")
plt.title("ROC 曲線比較")
plt.legend()
plt.grid(True)
plt.show()
print(f"SGD AUC: {auc_sgd:.3f},  Random Forest AUC: {auc_forest:.3f}")
```

**✅ 程式碼逐行解析：**

---

[來源: ch03 | 類型: tutorial] lt.show()
print(f"SGD AUC: {auc_sgd:.3f},  Random Forest AUC: {auc_forest:.3f}")
```

**✅ 程式碼逐行解析：**

1. `roc_curve(y_true, y_scores)`: 在所有可能閾值下計算 FPR 和 TPR 組成 ROC 曲線
2. `roc_auc_score`: AUC 為 ROC 曲線下面積，範圍 [0, 1]，越高越好，0.5 = 隨機猜測
3. `predict_proba[:, 1]`: 取正類（是5）的機率作為分數

**🎯 重點摘要:**

- **何時用 PR 曲線**：正負樣本極不平衡時（ROC 曲線可能過於樂觀）
- **何時用 ROC**：正負樣本相對平衡時，AUC 是最常用的單一評估指標

---

---

[來源: ch03 | 類型: tutorial] ## 多類別分類 (Multiclass Classification)

---

[來源: ch03 | 類型: tutorial] ### 範例 6: 辨識 0~9 所有數字

```python
from sklearn.svm import SVC
from sklearn.multiclass import OvRClassifier

---

[來源: ch03 | 類型: tutorial] # 10 類 → C(10,2) = 45 個二元分類器
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])  # 先用小資料測試

print(f"預測結果: {svm_clf.predict([X_test[0]])}")

---

[來源: ch03 | 類型: tutorial] # 強制使用 OvR（One-vs-Rest）策略：10 個二元分類器
ovr_clf = OvRClassifier(SVC())
ovr_clf.fit(X_train[:1000], y_train[:1000])
print(f"OvR 分類器數量: {len(ovr_clf.estimators_)}")  # 10

---

[來源: ch03 | 類型: tutorial] # SGDClassifier 天生支援多類別（直接輸出 10 類）
sgd_multi = SGDClassifier(random_state=42, max_iter=1000)
sgd_multi.fit(X_train, y_train)
scores_multi = cross_val_score(sgd_multi, X_train, y_train, cv=3, scoring="accuracy")
print(f"SGD 多類別準確率: {scores_multi.mean():.4f}")
```

**✅ 程式碼逐行解析：**

1. **OvO（One-vs-One）**：每對類別訓練一個分類器，適合訓練較慢但對大資料效能好的分類器（如 SVM）
2. **OvR（One-vs-Rest）**：每個類別 vs 其他，10 個分類器，更常用
3. `SGDClassifier` 天生多類別：內部使用 OvR 策略

**🎯 重點摘要:**

- Sklearn 的大多數分類器直接支援多類別，無需手動設定
- SVM 預設 OvO，SGD/LogReg 預設 OvR

---

---

[來源: ch03 | 類型: tutorial] ### 範例 7: 混淆矩陣熱力圖診斷錯誤

```python
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch03 | 類型: tutorial] # 取得 10 類分類的交叉驗證預測
y_train_pred_multi = cross_val_predict(sgd_multi, X_train, y_train, cv=3)

---

[來源: ch03 | 類型: tutorial] # 混淆矩陣正規化（每行除以該行總數）
cm = confusion_matrix(y_train, y_train_pred_multi)
cm_normalized = cm / cm.sum(axis=1, keepdims=True)  # 每行除以真實數量
np.fill_diagonal(cm_normalized, 0)  # 對角線設為0，聚焦於錯誤

---

[來源: ch03 | 類型: tutorial] # 視覺化
plt.figure(figsize=(8, 6))
plt.matshow(cm_normalized, cmap="hot", fignum=1)
plt.colorbar()
plt.title("正規化混淆矩陣（對角線設為0）")
plt.xlabel("預測類別")
plt.ylabel("真實類別")
plt.show()

---

[來源: ch03 | 類型: tutorial] # 亮格 = 常見錯誤，例如 3 和 5 互相混淆
```

**🎯 重點摘要:**

- 正規化混淆矩陣揭示「哪些類別最難分辨」（3↔5、4↔9、7↔1 最容易混淆）
- 根據錯誤模式決定改進方向：擴充容易混淆的訓練樣本、或加入更具區分力的特徵

---

---

[來源: ch03 | 類型: tutorial] ### 範例 8: 多標籤分類

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import numpy as np

---

[來源: ch03 | 類型: tutorial] # 建立多標籤目標：[是否 ≥ 7, 是否為奇數]
y_train_large = (y_train.astype(int) >= 7)  # 7, 8, 9
y_train_odd   = (y_train.astype(int) % 2 == 1)  # 1, 3, 5, 7, 9
y_multilabel  = np.c_[y_train_large, y_train_odd]

---

[來源: ch03 | 類型: tutorial] # KNN 支援多標籤
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train[:1000], y_multilabel[:1000])  # 用小資料示範

---

[來源: ch03 | 類型: tutorial] # 預測：回傳兩個布林值
pred = knn_clf.predict([X_test[0]])
print(f"預測 [≥7, 奇數]: {pred}")  # 例如 [[False, False]] 表示 <7 且 偶數

---

[來源: ch03 | 類型: tutorial] # y_pred_multi = knn_clf.predict(X_train[:1000])

---

[來源: ch03 | 類型: tutorial] # print(f"多標籤 F1: {f1_score(y_multilabel[:1000], y_pred_multi, average='macro'):.3f}")
```

**✅ 程式碼逐行解析：**

1. `y_train_large = (y_train.astype(int) >= 7)`: 建立布林陣列（True/False）
2. `np.c_[...]`: 水平串接，建立 (n, 2) 形狀的多標籤矩陣
3. KNN 的多標籤預測：每個樣本輸出一個布林向量

**🎯 重點摘要:**

- 多標籤分類：每個樣本可同時屬於多個類別（如圖片同時含有貓和狗）
- 多輸出分類：輸出的每個標籤可以有多個可能值（更一般化的多標籤）

---

---

[來源: ch03 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: 準確率高是否代表模型好？**

A: 不一定。以 MNIST 的 5-detector 為例：若資料中只有 10% 是 5，「全部預測非5」能達到 90% 準確率，卻毫無用處。不平衡資料集應使用 F1 分數、AUC 等指標。

**Q2: 精確率和召回率哪個更重要？**

A: 取決於業務場景。癌症篩查優先召回率（漏掉一個病患代價高）；垃圾郵件過濾優先精確率（把正常郵件誤判為垃圾代價高）。可用 `fbeta_score(beta)` 調整側重比例。

**Q3: `cross_val_predict` 和 `cross_val_score` 的區別？**

A: `cross_val_score` 回傳每折的分數；`cross_val_predict` 回傳每個樣本的「乾淨」預測值（每個樣本只在作為驗證集時被預測），適合用來計算混淆矩陣。

**Q4: OvO 和 OvR 哪個更好？**

A: 一般 OvR 更常用（訓練和預測更快）。OvO 的優點是每個分類器只需要看兩個類別的資料，對某些二元分類器（如 SVM）的核計算更有效率。

---

---

[來源: ch03 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #機器學習 #分類 #MNIST #ScikitLearn #混淆矩陣 #ROC曲線 #AUC #精確率 #召回率 #程式設計 #教學 #DataScience #MachineLearning #AI

---

[來源: ch04] [標題: Python 機器學習：訓練線性模型完整指南 - 從基礎到進階實戰 | 描述: 深入探討線性迴歸、梯度下降、正規化等核心概念，包含實戰程式碼與逐行解析，適合機器學習初學者與進階開發者 | 關鍵字: Python, 機器學習, 線性迴歸, 梯度下降, 正規化, Ridge, Lasso, 邏輯迴歸, Softmax]
# Training Linear Models：機器學習線性模型訓練完整指南

線性模型是機器學習的基石，理解如何訓練線性模型對於掌握更複雜的演算法至關重要。本教學將帶您深入了解各種線性模型訓練技術，從最簡單的線性迴歸到複雜的正規化方法。

---

[來源: ch04] ## 關鍵重點 (Key Takeaways)

- **線性迴歸**可使用正規方程式或梯度下降來訓練
- **梯度下降**有三種變體：批次、隨機、小批次，各有優缺點
- **多項式迴歸**可以擬合非線性資料，但需注意過擬合
- **正規化技術** (Ridge、Lasso、Elastic Net) 可以防止過擬合
- **邏輯迴歸**用於二元分類，**Softmax 迴歸**用於多類別分類
- **學習曲線**是診斷模型效能的重要工具

---

[來源: ch04] ## 環境設定與資料準備

💡 **實際應用情境：** 在開始任何機器學習專案前，我們需要設定適當的環境並載入必要的函式庫。

---

[來源: ch04] ```python
import sys
import sklearn
import matplotlib.pyplot as plt
from pathlib import Path

assert sys.version_info >= (3, 7)
from packaging import version
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

plt.rc('font', size=14)
plt.rc('font', family='Microsoft JhengHei')
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('axes', unicode_minus=False)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

IMAGES_PATH = Path() / "images" / "training_linear_models"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

---

[來源: ch04] t_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1.  `sys`, `sklearn`, `matplotlib.pyplot`, `pathlib.Path`: 匯入必要的函式庫。
2.  `assert sys.version_info >= (3, 7)`: 檢查 Python 版本是否符合要求。
3.  `assert version.parse(sklearn.__version__) >= version.parse("1.0.1")`: 檢查 Scikit-Learn 版本。
4.  `plt.rc(...)`: 設定 Matplotlib 的預設字體大小，使圖表更美觀。
    - `plt.rc('font', fami

---

[來源: ch04] ")`: 檢查 Scikit-Learn 版本。
4.  `plt.rc(...)`: 設定 Matplotlib 的預設字體大小，使圖表更美觀。
    - `plt.rc('font', family='Microsoft JhengHei')`: 設定字體為微軟正黑體, 適合中文顯示。
    - `plt.rc('axes', unicode_minus=False)`: 確保負號能正確顯示, 避免顯示成方塊。
5.  `IMAGES_PATH`: 定義儲存圖片的資料夾路徑。
6.  `save_fig()`: 定義一個函式，用於儲存高解析度的圖表。

**🎯 重點摘要:**

- **核心功能**: 建立一個標準化的環境，確保程式碼在不同機器上都能有一致的表現。
- **最佳使用情境**: 任何機器學習專案的初始設定。

---

[來源: ch04] ### 正規方程式 (Normal Equation)

💡 **實際應用情境：** 當資料集不大時，正規方程式是計算線性迴歸模型參數最直接的方法。

---

[來源: ch04] ### 範例 2: 使用正規方程式進行線性迴歸

```python
import numpy as np

np.random.seed(42)  # 設定隨機種子以確保結果可重現
m = 100  # 訓練樣本數量
X = 2 * np.random.rand(m, 1)  # 生成 100 個介於 0 到 2 之間的隨機數作為特徵

---

[來源: ch04] # 根據線性方程式 y = 4 + 3x + noise 生成目標值
y = 4 + 3 * X + np.random.randn(m, 1)  # 加入高斯雜訊

---

[來源: ch04] # 檢查資料的形狀
print(f'Shape of X: {X.shape}')
print(f'{X[:10]}')

print(f'Shape of y: {y.shape}')
print(f'{y[:10]}')

---

[來源: ch04] # 視覺化生成的資料
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
save_fig("generated_data_plot")
plt.show()

---

[來源: ch04] # 使用正規方程式計算最佳參數
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)  # 為每個實例添加 x0 = 1
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

---

[來源: ch04] # 檢視添加偏差項後的資料
print(f'X: \n{X[:10]}')
print(f'X_b: \n{X_b[:10]}')

---

[來源: ch04] # 顯示計算出的最佳參數
theta_best  # 由正規方程式計算出的 theta (權重)
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] 1.  `np.random.seed(42)`: 設定隨機種子以確保結果可重現。
2.  `m = 100`: 定義訓練樣本數量。
3.  `X = 2 * np.random.rand(m, 1)`: 產生 100 個介於 0 和 2 之間的隨機數作為特徵。
4.  `y = 4 + 3 * X + np.random.randn(m, 1)`: 根據線性方程式 `y = 4 + 3x` 生成目標值，並加入高斯雜訊。
5.  `print(f'Shape of X: {X.shape}')`: 顯示特徵矩陣的形狀 (100, 1)。
6.  `print(f'{X[:10]}')`: 顯示前 10 個訓練樣本的特徵值。
7.  `plt.plot(X, y, "b.")`: 繪製散點圖以視覺化訓練資料。
8.  `X_b = add_dummy_feature(X)`: 為每個實例添加

---

[來源: ch04] 前 10 個訓練樣本的特徵值。
7.  `plt.plot(X, y, "b.")`: 繪製散點圖以視覺化訓練資料。
8.  `X_b = add_dummy_feature(X)`: 為每個實例添加 `x0 = 1` 的偏差特徵，將形狀從 (100, 1) 轉換為 (100, 2)，且第一行全為 1，以考慮截距項。
9.  `theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y`: 使用正規方程式直接計算最佳參數 `theta`。
    - $\theta = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
    - `X_b.T @ X_b`: 計算特徵矩陣的轉置與自身的乘積。
    - `np.linalg.inv(...)`: 計算上述矩陣的逆矩陣。
    - 

---

[來源: ch04] }^T \mathbf{y}$
    - `X_b.T @ X_b`: 計算特徵矩陣的轉置與自身的乘積。
    - `np.linalg.inv(...)`: 計算上述矩陣的逆矩陣。
    - `@ X_b.T @ y`: 將逆矩陣與轉置的特徵矩陣及目標值相乘，得到最佳參數。
    - 最終結果 `theta_best` 包含截距項和斜率。

**🎯 重點摘要:**

- **核心功能**: 直接計算能最小化成本函式的模型參數。
- **潛在問題**: 當特徵數量非常大時，計算逆矩陣會非常慢。
- **最佳使用情境**: 特徵數量不多的小型資料集。

---

[來源: ch04] ### 範例 2.1: 使用計算出的參數進行預測

```python

---

[來源: ch04] # 建立新的資料點進行預測
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new)  # 為新資料添加 x0 = 1
y_predict = X_new_b @ theta_best
y_predict

---

[來源: ch04] # 檢查資料形狀
print(f'X_new:\n{X_new}')
print(f'X_new_b:\n{X_new_b}')
print(f'Shape of X_new: {X_new.shape}')
print(f'Shape of X_new_b: {X_new_b.shape}')

---

[來源: ch04] # 視覺化預測結果
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.", label="Training data")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.legend(loc="upper left")
save_fig("linear_model_predictions_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] lt.legend(loc="upper left")
save_fig("linear_model_predictions_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `X_new = np.array([[0], [2]])`: 建立兩個新的資料點 (x=0 和 x=2) 用於預測。
2.  `X_new_b = add_dummy_feature(X_new)`: 為新資料添加偏差項，形狀從 (2, 1) 變為 (2, 2)。
3.  `y_predict = X_new_b @ theta_best`: 使用矩陣乘法計算預測值：y = X_b · θ。
4.  `print(f'Shape of X_new: {X_new.shape}')`: 顯示新資料的形狀資訊。
5.  `plt.plot(X_new, y_predict, "r-", label="Predictions")`: 繪製紅色的預測線。
6.  `plt.plot(X, y, "b.", label="Training data")`: 繪製藍色的訓練資料點。

---

[來源: ch04] r-", label="Predictions")`: 繪製紅色的預測線。
6.  `plt.plot(X, y, "b.", label="Training data")`: 繪製藍色的訓練資料點。

**🎯 重點摘要:**

- **核心功能**: 使用訓練好的參數對新資料進行預測，並視覺化結果。
- **關鍵步驟**: 確保新資料也添加了偏差項 (x0 = 1)。
- **最佳使用情境**: 驗證模型的預測能力並理解線性迴歸的預測過程。

---

[來源: ch04] ### 奇異值分解 (SVD)

💡 **實際應用情境：** Scikit-Learn 的 `LinearRegression` 類別使用 SVD 方法，它比正規方程式更有效率且數值穩定。

---

[來源: ch04] ### 範例 3: 使用 Scikit-Learn 的 `LinearRegression`

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
```

**✅ 程式碼逐行解析：**

1.  `lin_reg = LinearRegression()`: 建立一個 `LinearRegression` 模型。
2.  `lin_reg.fit(X, y)`: 使用 SVD 方法訓練模型。
3.  `lin_reg.intercept_, lin_reg.coef_`: 取得模型的截距和係數。

**📊 LinearRegression 參數說明：**

---

[來源: ch04] `: 使用 SVD 方法訓練模型。
3.  `lin_reg.intercept_, lin_reg.coef_`: 取得模型的截距和係數。

**📊 LinearRegression 參數說明：**

`lin_reg.intercept_` 和 `lin_reg.coef_` 是線性迴歸模型學習到的參數：

---

[來源: ch04] #### `lin_reg.intercept_`
- 代表**截距** (*偏差項*) θ₀
- 這是當所有特徵都為零時的預測值
- 對應於迴歸線與 y 軸的交點

---

[來源: ch04] #### `lin_reg.coef_`
- 包含每個特徵的**係數** (*權重*)
- 對於我們的單特徵案例，`lin_reg.coef_[0][0]` 是 θ₁，即直線的斜率
- 顯示特徵增加一個單位時，預測值的變化量

---

[來源: ch04] #### 線性方程式
它們共同定義了線性模型：**ŷ = θ₀ + θ₁x₁**

其中：
- `lin_reg.intercept_` = θ₀ 
- `lin_reg.coef_[0][0]` = θ₁

這些值應該非常接近我們使用正規方程式計算的 `theta_best` 值，因為兩種方法都解決相同的線性迴歸問題，只是使用不同的演算法 (SVD vs. 正規方程式)。

**🎯 重點摘要:**

- **核心功能**: 使用 SVD 分解來求解線性迴歸，計算效率高且數值穩定。
- **最佳使用情境**: 大多數線性迴歸問題的標準選擇。

---

[來源: ch04] ### 範例 3.1: 使用 LinearRegression 進行預測

```python

---

[來源: ch04] # 使用訓練好的模型進行預測
lin_reg.predict(X_new)
```

**✅ 程式碼逐行解析：**

1.  `lin_reg.predict(X_new)`: 使用 Scikit-Learn 的預測方法，內部會自動處理偏差項的添加。

---

[來源: ch04] ### 範例 3.2: 直接使用 NumPy 的 lstsq 函數

`LinearRegression` 類別基於 `scipy.linalg.lstsq()` 函數 (名稱代表 "least squares")，你也可以直接呼叫它：

```python
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
```

**✅ 程式碼逐行解析：**

1.  `np.linalg.lstsq()`: NumPy 的最小平方法函數。
2.  `rcond=1e-6`: 設定條件數截止值，用於處理奇異矩陣。
3.  返回值包括：最佳參數 `theta_best_svd`、殘差、秩、奇異值。

---

[來源: ch04] ### 範例 3.3: 使用偽逆矩陣

此函數計算 $\mathbf{X}^+\mathbf{y}$，其中 $\mathbf{X}^{+}$ 是 $\mathbf{X}$ 的_偽逆矩陣_ (特別是 Moore-Penrose 逆矩陣)。你可以使用 `np.linalg.pinv()` 直接計算偽逆矩陣：

```python
np.linalg.pinv(X_b) @ y
```

**✅ 程式碼逐行解析：**

1.  `np.linalg.pinv(X_b)`: 計算 X_b 的 Moore-Penrose 偽逆矩陣。
2.  `@ y`: 與目標值相乘得到最佳參數。

**🎯 重點摘要:**

---

[來源: ch04] 解析：**

1.  `np.linalg.pinv(X_b)`: 計算 X_b 的 Moore-Penrose 偽逆矩陣。
2.  `@ y`: 與目標值相乘得到最佳參數。

**🎯 重點摘要:**

- **核心功能**: 展示了三種不同但等價的方法來求解線性迴歸。
- **數學原理**: 都基於最小平方法和偽逆矩陣的概念。
- **最佳使用情境**: 了解底層數學原理，在 Scikit-Learn 無法使用時的替代方案。

---

[來源: ch04] ## 梯度下降 (Gradient Descent)

梯度下降是一種優化演算法，用於透過迭代地朝著最小值移動來最小化成本函式。它計算成本函式相對於參數的梯度，並在梯度的反方向更新參數。

一般更新規則是：
**θ = θ - η∇J(θ)**

其中：
- **θ** 是模型參數
- **η** (eta) 是學習率
- **∇J(θ)** 是成本函式的梯度

梯度下降有三種主要變體：

1. **批次梯度下降 (Batch GD)**: 使用*整個*訓練集來計算梯度
2. **隨機梯度下降 (Stochastic GD)**: 每次使用*一個隨機實例*
3. **小批次梯度下降 (Mini-batch GD)**: 使用*小批次隨機實例*

---

[來源: ch04] ### 批次梯度下降 (Batch GD)

💡 **實際應用情境：** 當特徵數量龐大，無法使用正規方程式時，批次梯度下降是一個很好的替代方案。

---

[來源: ch04] ```python
eta = 0.1  # 學習率
n_epochs = 1000  # 迭代次數
m = len(X_b)  # 訓練樣本數量

np.random.seed(42)
theta = np.random.randn(2, 1)  # 隨機初始化模型參數
print(f'Initial theta:\n{theta}')

gradient_history = []
cost_history = []

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
    gradient_history.append(gradients)
    cost = (1/m) * np.sum((X_b @ theta - y) ** 2)  # 均方誤差
    cost_history.append(cost)
```

---

[來源: ch04] adients)
    cost = (1/m) * np.sum((X_b @ theta - y) ** 2)  # 均方誤差
    cost_history.append(cost)
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] t = (1/m) * np.sum((X_b @ theta - y) ** 2)  # 均方誤差
    cost_history.append(cost)
```

**✅ 程式碼逐行解析：**

1.  `eta = 0.1`: 學習率，控制每一步更新的幅度。
2.  `n_epochs = 1000`: 迭代次數，決定訓練多少輪。
3.  `m = len(X_b)`: 訓練集大小，用於梯度正規化。
4.  `theta = np.random.randn(2, 1)`: 隨機初始化模型參數 (θ₀ 和 θ₁)。
5.  `gradients = 2 / m * X_b.T @ (X_b @ theta - y)`: 計算整個訓練集的梯度向量。
6.  `theta = theta - eta * gradients`: 根據梯度更新參數 (梯度下降步驟)。
7.  `gradi

---

[來源: ch04] _b @ theta - y)`: 計算整個訓練集的梯度向量。
6.  `theta = theta - eta * gradients`: 根據梯度更新參數 (梯度下降步驟)。
7.  `gradient_history.append(gradients)`: 記錄每次迭代的梯度值，用於後續分析。
8.  `cost = (1/m) * np.sum((X_b @ theta - y) ** 2)`: 計算均方誤差 (MSE) 成本。
9.  `cost_history.append(cost)`: 記錄成本歷史，用於視覺化收斂過程。

**📊 關鍵變數說明：**

---

[來源: ch04] ### `theta`
- **用途**: 代表模型的參數 (θ₀ 和 θ₁)
- **初始化**: 隨機初始化的值作為優化的起點
- **更新規則**: 每次迭代透過 $\theta = \theta - \eta \nabla J(\theta)$ 更新,使參數朝著成本函式最小值移動
- **計算**: 透過梯度計算和學習率調整
    - `theta - eta * gradients`: 根據計算出的梯度調整參數
    - `gradients` 指向成本函式最陡峭上升的方向，移動到相反方向可下降到最小成本
- **最終結果**: 訓練完成後，`theta` 會收斂到最佳參數

---

[來源: ch04] ### `gradients`
- **用途**: 包含成本函式 (均方誤差) 對參數的梯度
- **數學公式**: $$\nabla J(\theta) = \frac{2}{m} \mathbf{X}_b^T (\mathbf{X}_b \theta - \mathbf{y})$$
    - 其中 $m$ 是訓練樣本數量
    - $\mathbf{X}_b$ 是包含偏差項的特徵矩陣
    - $\theta$ 是參數向量
    - $\mathbf{y}$ 是目標值向量
- **計算步驟**: `2 / m * X_b.T @ (X_b @ theta - y)` 以向量化方式計算偏導數：
    - `X_b @ theta - y`: 計算所有訓練實例的預測誤差 (預測值 - 實際值)，形狀為 (m, 1)
    - `X_b.T @ (...)`: 將特徵矩陣的轉置與誤差向量相乘，得到每個參數的梯度，形狀為 (2, 1)
    - `2 / m * ...`: 對梯度進行正規化，除以樣本數量 m，係數 2 來自 MSE 成本函式的微分
- **作用**: 梯度就像是指南針，指向成本增加最快的方向。我們往相反方向移動（下坡），就能找到成本最低點

---

[來源: ch04] ### `cost` 和 `cost_history`
- **`cost`**: 每個 epoch 計算的均方誤差 (MSE)。它衡量預測值 (`X_b @ theta`) 與實際值 (`y`) 之間的平均平方差異。
- **`cost_history`**: 儲存每個 epoch 的 `cost` 的列表。這對於視覺化學習過程並確認成本隨時間遞減非常有用，表明模型正在學習。

---

[來源: ch04] ### 關鍵參數說明
- **`eta`**: 學習率 (0.1) - 控制參數更新時的步長
- **`n_epochs`**: 迭代次數 (1000) - 決定更新參數的次數
- **`m`**: 訓練集大小 (100) - 用於梯度正規化

---

[來源: ch04] ### 演算法流程
1. **初始化** 隨機參數
2. **計算** 所有訓練資料的預測誤差
3. **計算** 梯度，顯示成本最陡峭增加的方向
4. **更新** 參數，移動到梯度的相反方向
5. **重複** 直到收斂 (梯度接近零)

迴圈迭代地優化 `theta`，直到收斂到能最小化預測誤差的最佳參數。

**🎯 重點摘要:**

- **核心功能**: 在每次迭代中使用整個訓練集來計算梯度，更新方向穩定。
- **潛在問題**: 對於非常大的資料集，每次迭代都會很慢。
- **最佳使用情境**: 記憶體足以容納整個資料集，且特徵數量龐大的情況。

---

[來源: ch04] ### 範例 4.1: 視覺化梯度收斂過程

```python

---

[來源: ch04] # 繪製梯度收斂圖
print(f"Type of gradient_history: {type(gradient_history)}")
gradient_history = np.array(gradient_history)  # 將列表轉換為 numpy 陣列
print(f"Shape of gradient_history: {gradient_history.shape}")

---

[來源: ch04] # 確保資料是 2D: (迭代次數, 參數數量)
y_to_plot = gradient_history.squeeze(-1) if gradient_history.ndim == 3 else gradient_history.reshape(gradient_history.shape[0], -1)
plt.plot(y_to_plot)
plt.xlabel("Iterations")
plt.ylabel("Gradient")
plt.title("Convergence of the Parameters")
plt.legend(["Theta 0", "Theta 1"])
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] e("Convergence of the Parameters")
plt.legend(["Theta 0", "Theta 1"])
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `gradient_history = np.array(gradient_history)`: 將 Python 列表轉換為 NumPy 陣列以便繪圖。
2.  `print(f"Shape of gradient_history: {gradient_history.shape}")`: 顯示陣列形狀，應該是 (1000, 2, 1)。
3.  `y_to_plot = gradient_history.squeeze(-1) if...`: 條件式重塑資料以適合繪圖。
4.  `gradient_history.squeeze(-1)`: 如果陣列是 3D，移除最後一個維度 

---

[來源: ch04] ent_history.squeeze(-1) if...`: 條件式重塑資料以適合繪圖。
4.  `gradient_history.squeeze(-1)`: 如果陣列是 3D，移除最後一個維度 (大小為 1)，將形狀從 (1000, 2, 1) 轉換為 (1000, 2)。
5.  `gradient_history.reshape(...)`: 如果不是 3D，作為備用方案將資料重塑為 2D。
6.  `plt.plot(y_to_plot)`: 繪製兩個參數的梯度歷史，每列作為一條線。

**📊 重塑 `gradient_history` 用於繪圖**

---

[來源: ch04] D，作為備用方案將資料重塑為 2D。
6.  `plt.plot(y_to_plot)`: 繪製兩個參數的梯度歷史，每列作為一條線。

**📊 重塑 `gradient_history` 用於繪圖**

這行程式碼 `y_to_plot = gradient_history.squeeze(-1) if gradient_history.ndim == 3 else gradient_history.reshape(gradient_history.shape[0], -1)` 是資料準備步驟，使 `gradient_history` 陣列適合用 Matplotlib 繪圖。

說明如下：

---

[來源: ch04] ory.reshape(gradient_history.shape[0], -1)` 是資料準備步驟，使 `gradient_history` 陣列適合用 Matplotlib 繪圖。

說明如下：

1.  **初始形狀**: 從列表轉換為 NumPy 陣列後，`gradient_history` 的形狀是 `(1000, 2, 1)`。這是一個 3D 陣列，代表 1000 個 epochs、2 個參數 (θ₀ 和 θ₁)，以及大小為 1 的尾隨維度。

2.  **繪圖需求**: Matplotlib 的 `plt.plot()` 函數，當給定 2D 陣列時，會將每一列繪製為單獨的線。當前的 3D 形狀不適合。

---

[來源: ch04] θ₁)，以及大小為 1 的尾隨維度。

2.  **繪圖需求**: Matplotlib 的 `plt.plot()` 函數，當給定 2D 陣列時，會將每一列繪製為單獨的線。當前的 3D 形狀不適合。

3.  **條件式重塑**: 程式碼使用條件表達式來處理：
    *   **`if gradient_history.ndim == 3`**: 首先檢查陣列是否為 3 維。
    *   **`gradient_history.squeeze(-1)`**: 如果是 3D，使用 `squeeze(-1)` 移除最後一個維度，但僅當該維度的大小為 1 時。這將形狀從 `(1000, 2, 1)` 轉換為 `(1000, 2)`。
    *   **`else gradient_history.reshape(...)`**: 如果陣列不是 3D，此備用方案確保資料重塑為 2D 陣列，使程式碼更健壯。

---

[來源: ch04] `(1000, 2)`。
    *   **`else gradient_history.reshape(...)`**: 如果陣列不是 3D，此備用方案確保資料重塑為 2D 陣列，使程式碼更健壯。

最終的 `y_to_plot` 變數是形狀為 `(1000, 2)` 的 2D 陣列，其中每一列代表一個參數的梯度歷史，準備好繪製為兩條不同的線。

**🎯 重點摘要:**

- **核心功能**: 視覺化梯度如何隨著訓練過程逐漸接近零 (收斂)。
- **關鍵洞察**: 如果梯度在後期仍然很大，可能需要更多迭代或調整學習率。
- **最佳使用情境**: 診斷訓練過程，確保模型正確收斂。

---

[來源: ch04] ### 範例 4.2: 放大檢視後期收斂

```python
start_iter = 900
plt.figure(figsize=(10, 6))

---

[來源: ch04] # 繪製 theta_0 和 theta_1 的梯度變化
plt.plot(np.arange(start_iter, len(y_to_plot)), y_to_plot[start_iter:, 0], 'b-', label=r"$\theta_0$")
plt.plot(np.arange(start_iter, len(y_to_plot)), y_to_plot[start_iter:, 1], 'r-', label=r"$\theta_1$")

---

[來源: ch04] # 在終點添加標記點並標註最終值
final_theta0 = y_to_plot[-1, 0]
final_theta1 = y_to_plot[-1, 1]
final_iter = len(y_to_plot) - 1

plt.plot(final_iter, final_theta0, 'bo', markersize=8, label=fr'$\theta_0$ 終值: {final_theta0:.6e}')
plt.plot(final_iter, final_theta1, 'ro', markersize=8, label=fr'$\theta_1$ 終值: {final_theta1:.6e}')

---

[來源: ch04] plt.plot(final_iter, final_theta1, 'ro', markersize=8, label=fr'$\theta_1$ 終值: {final_theta1:.6e}')

plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("梯度 (Gradient)")
plt.title("參數收斂過程 (放大) - Zoomed In")
plt.legend(loc='best')
plt.grid()
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] lt.title("參數收斂過程 (放大) - Zoomed In")
plt.legend(loc='best')
plt.grid()
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `start_iter = 900`: 設定起始迭代次數，只顯示最後 100 次迭代。
2.  `plt.figure(figsize=(10, 6))`: 設定圖表大小。
3.  `plt.plot(..., label=r"$\theta_0$")`: 使用 LaTeX 格式化圖例標籤，使數學符號更美觀。
4.  `final_theta0 = y_to_plot[-1, 0]`: 取得 `theta_0` 的最終梯度值。
5.  `plt.plot(final_iter, ..., 'bo', ...)`: 在梯度曲線的終點繪製一個藍色圓點標記。
6.  `label=fr'

---

[來源: ch04]  取得 `theta_0` 的最終梯度值。
5.  `plt.plot(final_iter, ..., 'bo', ...)`: 在梯度曲線的終點繪製一個藍色圓點標記。
6.  `label=fr'$\theta_0$ 終值: {final_theta0:.6e}'`: 在圖例中標註最終的梯度值，使用科學記號格式化。
7.  `plt.legend(loc='best')`: 自動將圖例放置在最佳位置。

---

[來源: ch04] ### 範例 4.3: 視覺化成本函數收斂

```python

---

[來源: ch04] # 繪製成本函數收斂圖
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("成本 (Cost - MSE)")
plt.title("批次梯度下降的成本函數收斂")
plt.grid()
plt.show()

---

[來源: ch04] # 繪製成本函數收斂圖 (放大檢視後期)
cost_start_iter = 300
plt.figure(figsize=(10, 6))
plt.plot(np.arange(cost_start_iter, len(cost_history)), cost_history[cost_start_iter:])
plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("成本 (Cost - MSE)")
plt.title("批次梯度下降的成本函數收斂 (放大)")
plt.axis((cost_start_iter, n_epochs, min(cost_history[cost_start_iter:])*0.99, max(cost_history[cost_start_iter:])*1.01))
plt.grid()
plt.show()
```

---

[來源: ch04] istory[cost_start_iter:])*0.99, max(cost_history[cost_start_iter:])*1.01))
plt.grid()
plt.show()
```

**✅ 程式碼逐行解析：**

1.  `plt.figure(figsize=(10, 6))`: 設定圖表大小以獲得更好的視覺效果。
2.  `plt.plot(cost_history)`: 繪製完整的成本歷史，顯示成本隨迭代次數下降的趨勢。
3.  `cost_start_iter = 300`: 設定放大檢視的起始點，專注於訓練後期。
4.  `plt.plot(...)`: 繪製從第 300 次迭代開始的成本歷史。
5.  `plt.axis(...)`: 設定座標軸範圍，使放大後的圖表更清晰。

**🎯 重點摘要:**

---

[來源: ch04] 練後期。
4.  `plt.plot(...)`: 繪製從第 300 次迭代開始的成本歷史。
5.  `plt.axis(...)`: 設定座標軸範圍，使放大後的圖表更清晰。

**🎯 重點摘要:**

- **核心功能**: 視覺化成本函數隨訓練過程的下降趨勢。
- **關鍵洞察**: 
  - 如果成本持續下降，表示模型正在學習。
  - 如果成本停止下降或震盪，可能需要調整學習率。
  - 放大後期可以更清楚地看到模型是否已經收斂。
- **最佳使用情境**: 監控訓練過程，診斷收斂問題，並輔助調整學習率等超參數。

---

[來源: ch04] ### 隨機梯度下降 (Stochastic GD; SGD)

💡 **實際應用情境：** 當資料集非常大，無法在記憶體中一次處理時，隨機梯度下降是理想的選擇。

---

[來源: ch04] ```python
n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
```

---

[來源: ch04] eta - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
```

**✅ 程式碼逐行解析：**

1.  `n_epochs = 50`: SGD 通常需要較少的 epoch 數
2.  `t0, t1 = 5, 50`: 學習率衰減 (learning rate decay) 的超參數
3.  `def learning_schedule(t)`: 定義學習率隨時間遞減的函數
4.  `random_index = np.random.randint(m)`: 在每次迭代中隨機選擇一個樣本。
5.  `xi = X_b[random_index:random_index+1]`: 取出該樣本的特徵值。
6.  `yi = y[random_index:random_index+1]`: 

---

[來源: ch04] 
5.  `xi = X_b[random_index:random_index+1]`: 取出該樣本的特徵值。
6.  `yi = y[random_index:random_index+1]`: 取出該樣本的目標值。
7.  `gradients = 2 * xi.T @ (xi @ theta - yi)`: 只用一個樣本來計算梯度。
8.  `eta = learning_schedule(epoch * m + i)`: 根據排程更新學習率, 隨著迭代次數增加而減小。
9.  `theta = theta - eta * gradients`: 根據梯度更新參數 (梯度下降步驟)。

**🎯 重點摘要:**

---

[來源: ch04] i)`: 根據排程更新學習率, 隨著迭代次數增加而減小。
9.  `theta = theta - eta * gradients`: 根據梯度更新參數 (梯度下降步驟)。

**🎯 重點摘要:**

- **核心功能**: 每次只用一個樣本來更新參數，速度快，適合線上學習 (online learning)。
- **潛在問題**: 更新方向不穩定，成本函式會上下波動。
- **最佳使用情境**: 超大型資料集或需要線上學習的場景。

---

[來源: ch04] ### 範例 5.1: 使用 Scikit-Learn 的 SGD

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())
```

**✅ 程式碼逐行解析：**

1.  `max_iter=1000`: 設定最大迭代次數為1000
2.  `tol=1e-5`: 設定收斂容忍度
3.  `penalty=None`: 不使用正規化
4.  `eta0=0.01`: 初始學習率為0.01
5.  `n_iter_no_change=100`: 如果100次迭代內損失未改善則提前停止
6.  `sgd_reg.fit(X, y.ravel())`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 提供現成的 SGD 實作，包含許多優化技巧
- **最佳使用情境**: 需要快速訓練大規模線性模型時

---

[來源: ch04] ### 小批次梯度下降 (Mini-batch GD)

💡 **實際應用情境：** 小批次梯度下降是批次和隨機梯度下降的折衷方案，在實務中應用最廣。

---

[來源: ch04] ```python
from math import ceil

n_epochs = 50
minibatch_size = 20
n_batches_per_epoch = ceil(m / minibatch_size)

np.random.seed(42)
theta = np.random.randn(2, 1)

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for iteration in range(0, n_batches_per_epoch):
        idx = iteration * minibatch_size
        xi = X_b_shuffled[idx : idx + minibatch_size]
        yi = y_shuffled[idx : idx + minibatch_size]
        gradients = 2 / minibatch_size * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * n_batches_per_epoch + iteration)
        theta = theta - eta * gradients
```

---

[來源: ch04] arning_schedule(epoch * n_batches_per_epoch + iteration)
        theta = theta - eta * gradients
```

**✅ 程式碼逐行解析：**

1.  `minibatch_size`: 設定每個小批次的大小。
2.  `shuffled_indices`: 在每個 epoch 開始時打亂資料順序。
3.  `gradients = 2/minibatch_size * xi.T @ (xi @ theta - yi)`: 使用一個小批次的樣本來計算梯度。

**🎯 重點摘要:**

- **核心功能**: 結合了批次梯度下降的穩定性和隨機梯度下降的效率。
- **最佳使用情境**: 大多數機器學習問題，特別是深度學習。

---

[來源: ch04] ## 多項式迴歸 (Polynomial Regression)

💡 **實際應用情境：** 當資料呈現非線性關係時，可以使用多項式迴歸來擬合曲線。

---

[來源: ch04] ### 範例 7: 多項式迴歸

```python
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] y_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
```

**✅ 程式碼逐行解析：**

1.  `y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)`: 生成一個二次方程式的非線性資料集。
2.  `poly_features = PolynomialFeatures(degree=2, include_bias=False)`: 建立一個 `PolynomialFeatures` 轉換器。
3.  `X_poly = poly_features.fit_transform(X)`:將 `X` 轉換為包含原始特徵和其平方的 `X_poly`。
4.  `lin_reg.fit(X_poly, y)`: 使用轉換後的多項式特徵來訓練線性迴歸模型。

---

[來源: ch04] fit_transform(X)`:將 `X` 轉換為包含原始特徵和其平方的 `X_poly`。
4.  `lin_reg.fit(X_poly, y)`: 使用轉換後的多項式特徵來訓練線性迴歸模型。

**🎯 重點摘要:**

- **核心功能**: 讓線性模型能夠擬合非線性資料。
- **潛在問題**: 高次多項式容易導致過擬合。
- **最佳使用情境**: 當資料視覺化中發現明顯的非線性趨勢時。

---

[來源: ch04] ## 學習曲線 (Learning Curves)

💡 **實際應用情境：** 學習曲線是診斷模型是過擬合還是欠擬合的強大工具。

---

[來源: ch04] ```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
plt.legend()
```

---

[來源: ch04] abel="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
plt.legend()
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] t.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
plt.legend()
```

**✅ 程式碼逐行解析：**

1.  `learning_curve()`: Scikit-Learn 的函式，用於計算不同訓練集大小下的模型效能。
2.  `train_sizes=np.linspace(0.01, 1.0, 40)`: 測試40個不同的訓練集大小。
3.  `cv=5`: 使用5折交叉驗證 (5-fold cross-validation)。
4.  `train_errors`, `valid_errors`: 將得分轉換為均方根誤差 (RMSE)。
  -   `-train_scores.mean(axis=1)`: 計算每個訓練集大小的平均訓練誤差, 並取負值轉換為正的 RMSE。
  -

---

[來源: ch04] rors`: 將得分轉換為均方根誤差 (RMSE)。
  -   `-train_scores.mean(axis=1)`: 計算每個訓練集大小的平均訓練誤差, 並取負值轉換為正的 RMSE。
  -   `-valid_scores.mean(axis=1)`: 計算每個訓練集大小的平均驗證誤差, 並取負值轉換為正的 RMSE。
  -   `axis=1`: 指定沿著哪個軸計算平均值，這裡是沿著列 (不同交叉驗證折數)。
5.  `plt.plot()`: 繪製訓練誤差和驗證誤差隨訓練集大小變化的曲線。

**🎯 重點摘要:**

- **核心功能**: 視覺化模型在不同訓練集大小下的效能。
- **解讀**:
    - **欠擬合**: 兩條曲線都很高且彼此接近。
    - **過擬合**: 兩條曲線之間有很大的差距。
- **最佳使用情境**: 評估模型的泛化能力。

---

[來源: ch04] ## 正規化線性模型

正規化是防止線性模型過擬合的關鍵技術。當模型在訓練資料上表現很好，但在新資料上表現不佳時，就可能發生過擬合。正規化透過在成本函式中添加懲罰項來約束模型參數的大小，迫使模型學習更簡單、更能泛化的模式。

**為什麼需要正規化？**

1. **防止過擬合**: 當模型過於複雜或特徵數量很多時，它可能會「記住」訓練資料的雜訊，而不是學習真正的模式。
2. **處理多重共線性 (Multicollinearity)**: 當特徵之間高度相關時，正規化（特別是 Ridge 或 Elastic Net）可以穩定參數估計，避免模型權重不合理波動。
3. **特徵選擇 (Feature Selection)**: 某些正規化方法（如 Lasso）可以自動識別並忽略不重要的特徵。
4. **提升泛化能力**: 正規化後的模型通常在未見過的資料上表現更好。

**正規化的數學原理**

---

[來源: ch04] e Selection)**: 某些正規化方法（如 Lasso）可以自動識別並忽略不重要的特徵。
4. **提升泛化能力**: 正規化後的模型通常在未見過的資料上表現更好。

**正規化的數學原理**

對於線性迴歸，標準的成本函式是均方誤差（MSE）：

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

正規化透過添加懲罰項 (penalty term) 來修改這個成本函式：

$$J(\theta) = MSE(\theta) + \text{正規化項}$$

這個正規化項會懲罰大的參數值，迫使模型保持簡單。

**三種主要的正規化方法**

---

[來源: ch04] erm) 來修改這個成本函式：

$$J(\theta) = MSE(\theta) + \text{正規化項}$$

這個正規化項會懲罰大的參數值，迫使模型保持簡單。

**三種主要的正規化方法**

1. **Ridge 迴歸 (L2 正規化)**
  - 懲罰項：$\alpha \sum_{i=1}^{n} \theta_i^2$
  - 會縮小所有參數，但不會將其設為零
  - 適合所有特徵都可能有用的情況

2. **Lasso 迴歸 (L1 正規化)**
  - 懲罰項：$\alpha \sum_{i=1}^{n}

 |\theta_i|$


- 會將一些參數完全設為零，實現自動特徵選擇
  - 適合懷疑許多特徵是多餘的情況

---

[來源: ch04] (L1 正規化)**
  - 懲罰項：$\alpha \sum_{i=1}^{n}

 |\theta_i|$


- 會將一些參數完全設為零，實現自動特徵選擇
  - 適合懷疑許多特徵是多餘的情況

3. **Elastic Net (L1 + L2)**
  - 結合 Ridge 和 Lasso 的優點
  - 懲罰項：$r \alpha \sum_{i=1}^{n}

 |\theta_i| + \frac{1-r}{2} \alpha \sum_{i=1}^{n} \theta_i^2$


- 其中 $r$ 是介於 0 和 1 之間的混合參數, 控制 L1 和 L2 的權重
  - 適合特徵數量遠大於樣本數量的情況
    - 例如基因資料分析或文本分類等高維度資料集

其中 $\alpha$ 是正規化強度參數，控制懲罰的程度。$\alpha$ 越大，正規化效果越強，模型越簡單。

---

[來源: ch04] 適合特徵數量遠大於樣本數量的情況
    - 例如基因資料分析或文本分類等高維度資料集

其中 $\alpha$ 是正規化強度參數，控制懲罰的程度。$\alpha$ 越大，正規化效果越強，模型越簡單。

**選擇正規化方法的實用建議**

- 從 Ridge 開始嘗試，它通常是最安全的選擇
- 如果特徵很多且懷疑只有少數有用，使用 Lasso
- 當 Lasso 表現不穩定時，考慮 Elastic Net
- 使用交叉驗證來選擇最佳的 $\alpha$ 值
- 記得在應用正規化之前對特徵進行縮放，因為正規化對特徵的尺度敏感

下面的章節將詳細介紹每種正規化方法的實作和應用。

---

---

[來源: ch04] ### Ridge 迴歸

💡 **實際應用情境：** 當模型過擬合時，Ridge 迴歸可以透過對模型參數的懲罰來降低複雜度。

---

[來源: ch04] ### 範例 9: Ridge 迴歸

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] dge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.linear_model import Ridge`: 導入 Ridge 迴歸類別
2. `alpha=1`: 設定正規化強度，值越大正規化越強
3. `solver="cholesky"`: 使用 Cholesky 分解求解，適合中小型資料集
  - Cholesky 分解是一種數值方法，用於高效解決線性方程組，特別是當矩陣是對稱正定時。它將矩陣分解為一個下三角矩陣及其轉置的乘積，從而簡化計算過程。
4. `ridge_reg.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

---

[來源: ch04] 用於高效解決線性方程組，特別是當矩陣是對稱正定時。它將矩陣分解為一個下三角矩陣及其轉置的乘積，從而簡化計算過程。
4. `ridge_reg.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 加入 L2 懲罰項，降低過擬合風險。
- **最佳使用情境**: 當懷疑模型過擬合或有多重共線性問題時。

---

---

[來源: ch04] ### Lasso 迴歸

💡 **實際應用情境：** Lasso 迴歸不僅能防止過擬合，還能自動進行特徵選擇。

---

[來源: ch04] ### 範例 10: Lasso 迴歸

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1, random_state=42)
lasso_reg.fit(X, y)
```

**🎯 重點摘要:**

- **核心功能**: 加入 L1 懲罰項，可將不重要的特徵權重降為零。
- **最佳使用情境**: 當你認為某些特徵是多餘的，或想要一個更簡潔的模型時。

---

---

[來源: ch04] ### Elastic Net

💡 **實際應用情境：** Elastic Net 結合了 Ridge 和 Lasso 的優點。

---

[來源: ch04] ### 範例 11: Elastic Net

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.linear_model import ElasticNet`: 導入 Elastic Net 類別
2. `alpha=0.1`: 設定正規化強度，值越大正規化越強
3. `l1_ratio=0.5`: 設定 L1 和 L2 懲罰的比例，0.5 表示兩者權重相等
4. `elastic_net.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

---

[來源: ch04] 正規化越強
3. `l1_ratio=0.5`: 設定 L1 和 L2 懲罰的比例，0.5 表示兩者權重相等
4. `elastic_net.fit(X, y)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 同時使用 L1 和 L2 懲罰。
- **最佳使用情境**: 當特徵數量大於樣本數，或特徵之間有很強的相關性時。

---

---

[來源: ch04] ### Early Stopping

💡 **實際應用情境：** Early Stopping 是一種簡單而有效的正規化方法，透過*監控驗證集的效能來決定何時停止訓練*。當驗證誤差開始上升時，表示模型開始過擬合訓練資料，此時應停止訓練並使用到目前為止表現最好的模型。這種方法特別適合用於迭代式學習演算法（如梯度下降），可以在不增加模型複雜度的情況下防止過擬合，同時節省訓練時間。

Early Stopping 的核心思想是：

- **持續監控**：在每個 epoch 後評估驗證集的效能
- **記錄最佳模型**：當驗證誤差創新低時，儲存當前模型的副本
- **及時停止**：當驗證誤差持續一段時間不再改善時，停止訓練
- **使用最佳模型**：最終採用驗證誤差最低時的模型參數

---

[來源: ch04] 集的效能
- **記錄最佳模型**：當驗證誤差創新低時，儲存當前模型的副本
- **及時停止**：當驗證誤差持續一段時間不再改善時，停止訓練
- **使用最佳模型**：最終採用驗證誤差最低時的模型參數

這種方法的優點是不需要調整正規化參數（如 α），但需要額外的驗證集和更多的計算資源來追蹤模型效能。在實務中，Early Stopping 常與其他正規化技術（如 Ridge 或 Lasso）結合使用，以獲得更好的泛化能力。

---

[來源: ch04] ```python
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, shuffle=False)

preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train.ravel())
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
```

---

[來源: ch04] or < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
```

**✅ 程式碼逐行解析：**

1.  `from copy import deepcopy`: 導入 `deepcopy` 用於複製模型
2.  `train_test_split(...)`: 將資料分割為訓練集和驗證集
  - `shuffle=False`: 保持資料的時間順序，適合時間序列資料
3.  `make_pipeline(...)`: 建立一個包含多項式特徵和標準化的預處理管道
  - `PolynomialFeatures(degree=90, include_bias=False)`: 將特徵擴展為 90 次多項式
  - `StandardScaler()`: 對特徵進行標準化，使其均值為0，標

---

[來源: ch04] alFeatures(degree=90, include_bias=False)`: 將特徵擴展為 90 次多項式
  - `StandardScaler()`: 對特徵進行標準化，使其均值為0，標準差為1
4.  `sgd_reg = SGDRegressor(...)`: 初始化一個不帶正規化的隨機梯度下降迴歸模型
  - `penalty=None`: 不使用任何正規化
  - `eta0=0.002`: 設定初始學習率, 較小的學習率有助於穩定訓練過程
5.  `for epoch in range(n_epochs)`: 進行多個訓練迭代; 每次迭代後評估模型效能
6.  `sgd_reg.partial_fit(...)`: 使用部分擬合方法進行訓練
  - `y_train.ravel()`: 將目標變數展平成一維陣列，以符合 Scikit-Learn 的要求
  - `p

---

[來源: ch04] _reg.partial_fit(...)`: 使用部分擬合方法進行訓練
  - `y_train.ravel()`: 將目標變數展平成一維陣列，以符合 Scikit-Learn 的要求
  - `partial_fit()`: 適用於增量學習 (incremental learning or online learning)，允許模型在每個 epoch 後更新參數，而不是重新擬合整個模型
7.  `y_valid_predict = sgd_reg.predict(X_valid_prep)`: 在驗證集上進行預測
8.  `val_error = root_mean_squared_error(...)`: 計算驗證集的均方根誤差
9.  `if val_error < best_valid_rmse`: 如果當前驗證誤差優於最佳誤差，則更新最佳模型
  - `best_valid_r

---

[來源: ch04] or(...)`: 計算驗證集的均方根誤差
9.  `if val_error < best_valid_rmse`: 如果當前驗證誤差優於最佳誤差，則更新最佳模型
  - `best_valid_rmse = val_error`: 更新最佳驗證誤差
10. `best_model = deepcopy(sgd_reg)`: 使用 `deepcopy` 儲存當前模型的副本作為最佳模型

**🎯 重點摘要:**

- **核心功能**: 監控驗證集上的效能，在模型開始過擬合之前停止訓練。
- **最佳使用情境**: 任何迭代式的學習演算法。

---

[來源: ch04] ## 邏輯迴歸 (Logistic Regression)

邏輯迴歸是解決二元分類問題最常用和最基礎的演算法。

它透過邏輯函數（Sigmoid 函數）將線性迴歸的輸出轉換為 0 到 1 之間的機率值，使其適合預測類別歸屬。

常見應用包括：垃圾郵件檢測（是/否）、疾病診斷（陽性/陰性）、客戶流失預測（會/不會）、信用風險評估（違約/正常）等。

邏輯迴歸的優勢在於訓練速度快、解釋性強、可以輸出機率估計，且在許多實際問題中表現穩健。

---

[來源: ch04] ### 邏輯迴歸的核心原理

邏輯迴歸使用 **Sigmoid 函數**（也稱為 Logistic 函數）將線性組合轉換為機率：

$$\sigma(t) = \frac{1}{1 + e^{-t}}$$

其數值範圍在 0 到 1 之間，適合作為機率值。

對於輸入 $x$，邏輯迴歸模型估計的機率為：

$$\hat{p} = h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

---

[來源: ch04] 輸入 $x$，邏輯迴歸模型估計的機率為：

$$\hat{p} = h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

其中：
- $\theta^T x$ 是特徵的線性組合
- $\hat{p}$ 是預測為正類別（y=1）的機率
- 當 $\theta^T x$ 為正且很大時，$\hat{p}$ 接近 1
- 當 $\theta^T x$ 為負且很小時，$\hat{p}$ 接近 0
- 當 $\theta^T x = 0$ 時，$\hat{p} = 0.5$（決策邊界）

**決策規則**：
- 如果 $\hat{p} \geq 0.5$，預測 $\hat{y} = 1$（正類別）
- 如果 $\hat{p} < 0.5$，預測 $\hat{y} = 0$（負類別）

**成本函數（Log Loss）**

---

[來源: ch04] at{p} \geq 0.5$，預測 $\hat{y} = 1$（正類別）
- 如果 $\hat{p} < 0.5$，預測 $\hat{y} = 0$（負類別）

**成本函數（Log Loss）**

邏輯迴歸使用對數損失（log loss），也稱為交叉熵損失 (cross-entropy loss)：

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)})]$$

---

[來源: ch04] -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)})]$$

這個成本函數的特性：
- 當 $y=1$ 且 $\hat{p}$ 接近 1 時，成本接近 0（正確預測）
- 當 $y=1$ 且 $\hat{p}$ 接近 0 時，成本趨近無窮大（嚴重錯誤）
- 當 $y=0$ 且 $\hat{p}$ 接近 0 時，成本接近 0（正確預測）
- 當 $y=0$ 且 $\hat{p}$ 接近 1 時，成本趨近無窮大（嚴重錯誤）

**正規化**

---

[來源: ch04] （嚴重錯誤）
- 當 $y=0$ 且 $\hat{p}$ 接近 0 時，成本接近 0（正確預測）
- 當 $y=0$ 且 $\hat{p}$ 接近 1 時，成本趨近無窮大（嚴重錯誤）

**正規化**

Scikit-Learn 的 `LogisticRegression` 預設使用 L2 正規化，可以透過參數 `C` 控制：
- `C` 是正規化強度的倒數
- `C` 越大，正規化越弱（允許更複雜的模型）
- `C` 越小，正規化越強（模型更簡單）

💡 **實際應用情境：** 邏輯迴歸廣泛應用於各種二元分類問題，如醫療診斷、信用評分、行為預測等，因其解釋性強且計算效率高。

---

[來源: ch04] ### 範例 13: 邏輯迴歸

此範例展示如何使用邏輯迴歸進行二元分類。我們使用 Iris 資料集，僅選擇一個特徵（花瓣寬度）來預測一朵花是否為 Iris virginica。這是一個簡化的範例，用於說明邏輯迴歸的基本概念。在實務中，使用多個特徵通常能獲得更好的分類效果。

---

[來源: ch04] 我們使用 Iris 資料集，僅選擇一個特徵（花瓣寬度）來預測一朵花是否為 Iris virginica。這是一個簡化的範例，用於說明邏輯迴歸的基本概念。在實務中，使用多個特徵通常能獲得更好的分類效果。

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X = iris.data[["petal width (cm)"]].values
y = (iris.target == 2).astype(int)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] isticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import load_iris`: 導入 Iris 資料集載入函數
2. `iris = load_iris(as_frame=True)`: 載入 Iris 資料集並轉換為 pandas DataFrame 格式
3. `X = iris.data[["petal width (cm)"]].values`: 選擇花瓣寬度作為唯一特徵
  - 使用雙括號 `[[...]]` 確保結果是 2D 陣列
  - `.values` 將 DataFrame 轉換為 NumPy 陣列; 也可使用 `.to_numpy()`, 較新的 panda

---

[來源: ch04]  - 使用雙括號 `[[...]]` 確保結果是 2D 陣列
  - `.values` 將 DataFrame 轉換為 NumPy 陣列; 也可使用 `.to_numpy()`, 較新的 pandas 方法
4. `y = (iris.target == 2).astype(int)`: 建立二元目標變數
  - `iris.target == 2`: 檢查是否為 virginica（類別 2）
  - `.astype(int)`: 將布林值轉換為整數（0 或 1）
5. `log_reg = LogisticRegression(random_state=42)`: 建立邏輯迴歸模型實例
6. `log_reg.fit(X, y)`: 訓練模型以學習特徵與目標之間的關係

**🎯 重點摘要:**

---

[來源: ch04] isticRegression(random_state=42)`: 建立邏輯迴歸模型實例
6. `log_reg.fit(X, y)`: 訓練模型以學習特徵與目標之間的關係

**🎯 重點摘要:**

- **核心功能**: 估計一個樣本屬於某個類別的機率，並進行分類。
- **資料準備**: 展示如何將多類別問題轉換為二元分類問題。
- **模型輸出**: 訓練後的模型可以預測新樣本的類別，並提供屬於正類別的機率估計。
- **最佳使用情境**: 二元分類問題，如垃圾郵件檢測、疾病診斷等。

---

[來源: ch04] ### 範例 13.1: 使用兩個特徵的邏輯迴歸：機率等高線圖與決策邊界

此範例訓練一個二元邏輯迴歸分類器來區分 Iris virginica，使用兩個特徵 (花瓣長度和花瓣寬度) 並視覺化機率表面和線性決策邊界。

```python

---

[來源: ch04] # 準備資料
X = iris.data[["petal length (cm)", "petal width (cm)"]].to_numpy()  # 兩個特徵的特徵矩陣
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

---

[來源: ch04] # 訓練模型
log_reg = LogisticRegression(C=2, random_state=42)
log_reg.fit(X_train, y_train)

---

[來源: ch04] # 建立機率網格用於等高線圖
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]  # 將網格點展開為樣本矩陣
y_proba = log_reg.predict_proba(X_new)
zz = y_proba[:, 1].reshape(x0.shape)

---

[來源: ch04] # 計算決策邊界
left_right = np.array([2.9, 7])
boundary = -((log_reg.coef_[0, 0] * left_right + log_reg.intercept_[0])
             / log_reg.coef_[0, 1])

---

[來源: ch04] # 視覺化
plt.figure(figsize=(10, 4))
plt.plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], "bs")
plt.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], "g^")
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.27, "Not Iris virginica", color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", color="g", ha="center")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.axis([2.9, 7, 0.8, 2.7])
plt.grid()
save_fig("logistic_regression_contour_plot")
plt.show()
```

---

[來源: ch04] plt.axis([2.9, 7, 0.8, 2.7])
plt.grid()
save_fig("logistic_regression_contour_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] #### 1) 資料設定與訓練

1. **特徵選擇**: `X = iris.data[["petal length (cm)", "petal width (cm)"]].to_numpy()`
   - 使用 pandas DataFrame 的雙括號列選擇以確保得到 2D 陣列
   - `.to_numpy()` 將 DataFrame 轉換為 NumPy 陣列

2. **目標變數**: `y = iris.target_names[iris.target] == 'virginica'`
   - 建立布林值目標 (True 表示 virginica)

---

[來源: ch04] 列

2. **目標變數**: `y = iris.target_names[iris.target] == 'virginica'`
   - 建立布林值目標 (True 表示 virginica)

3. **資料分割**: `train_test_split(X, y, random_state=42)`
   - 分割訓練集和測試集以確保可重現性
   - 注意：`train_test_split` 輸入X和y是 NumPy 陣列 (numpy.ndarray)
   - 注意：`train_test_split` 輸出也是 NumPy 陣列

4. **模型**: `LogisticRegression(C=2)`
   - `C` 是正規化強度的倒數 (C 越大，正規化越弱)
   - 在訓練資料上擬合以學習係數和截距

---

[來源: ch04] #### 2) 建立機率網格用於等高線圖

為了視覺化模型的決策邊界，我們需要在整個特徵空間中對模型的預測進行採樣。這可以透過建立一個密集的點網格，並為每個點預測機率來實現。

---

[來源: ch04] ##### a. 建立座標網格 (np.meshgrid)

首先，我們使用 `np.meshgrid` 來建立一個覆蓋我們感興趣的特徵範圍的 2D 座標網格，就像在繪圖紙上畫出格線一樣。

```python
   # 建立覆蓋特徵空間的密集網格
   x0, x1 = np.meshgrid(
       np.linspace(2.9, 7, 500).reshape(-1, 1),      # 花瓣長度座標 (x 軸)
       np.linspace(0.8, 2.7, 200).reshape(-1, 1),    # 花瓣寬度座標 (y 軸)
   )
   ```

---

[來源: ch04] # 花瓣長度座標 (x 軸)
       np.linspace(0.8, 2.7, 200).reshape(-1, 1),    # 花瓣寬度座標 (y 軸)
   )
   ```

- **`np.linspace(...)`**: 建立定義網格軸線的 1D 陣列 (500 個 x 座標和 200 個 y 座標)。
   - **`np.meshgrid(...)`**: 將這些軸線擴展為兩個 2D 陣列：
     - `x0` (形狀 `200, 500`)：包含網格中每個點的 x 座標 (花瓣長度)。
     - `x1` (形狀 `200, 500`)：包含網格中每個點的 y 座標 (花瓣寬度)。

---

[來源: ch04] ##### b. 準備網格用於預測
Scikit-Learn 模型期望輸入是一個樣本列表 (形狀為 `[n_samples, n_features]`)。我們需要將 2D 網格轉換為這種格式。

   ```python
   # 將網格點轉換為樣本矩陣
   X_new = np.c_[x0.ravel(), x1.ravel()]
   ```

   - **`x0.ravel()`**: 將 `x0` 陣列「展平」為一個包含 100,000 個元素的 1D 陣列。
   - **`np.c_[...]`**: 將兩個展平的 1D 陣列並排堆疊，建立一個形狀為 `(100000, 2)` 的陣列。`X_new` 現在是一個包含網格上所有 100,000 個點的列表，每個點都是一個 `(花瓣長度, 花瓣寬度)` 的特徵對。

---

[來源: ch04] ##### c. 計算機率並重塑
現在我們可以將這個樣本列表餵給模型來預測每個點的機率，然後將結果重塑回原始網格的形狀，以便繪製等高線。

   ```python
   # 計算每個網格點的類別機率
   y_proba = log_reg.predict_proba(X_new)

   # 提取 virginica 的機率並重塑為網格形狀
   zz = y_proba[:, 1].reshape(x0.shape)
   ```

   - `zz` 現在是一個 `(200, 500)` 的陣列，其中 `zz[i, j]` 代表網格點 `(i, j)` 被預測為 "virginica" 的機率。

---

[來源: ch04] #### 3) 決策邊界計算

對於邏輯迴歸，0.5 機率邊界是線性的：

- 給定權重 w = [w_len, w_wid] 和偏差 b：
- 邊界：petal_width = -(w_len × petal_length + b) / w_wid
- `left_right` 設定繪圖的線的 x 範圍

---

[來源: ch04] #### 4) 視覺化

- **藍色方塊**: 非 virginica 訓練點
- **綠色三角形**: virginica 訓練點
- **等高線**: 網格上 "virginica" 類別的機率等級
- **黑色虛線**: 決策邊界 (P=0.5)
- **座標軸限制和標籤**: 調整為符合 Iris 特徵範圍
- **圖表儲存**: 透過 `save_fig("logistic_regression_contour_plot")`

---

[來源: ch04] #### 如何解讀圖表

- 等高線值越高的區域 (接近綠色) 表示 P(virginica) 越高
- 虛線上方的點被預測為 virginica；下方為非 virginica
- 等高線的間距反映模型信心 (越密集 = 機率變化越陡峭)

---

**📊 理解 NumPy 的 `ravel()` 方法**

`ravel()` 方法將多維 NumPy 陣列展平為 1D 陣列。在上面的程式碼中，它用於將 2D 網格陣列轉換為 1D 向量以進行處理。

---

[來源: ch04] ##### 程式碼範例

```python
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))

---

[來源: ch04] # x0 和 x1 是形狀為 (200, 500) 的 2D 陣列

X_new = np.c_[x0.ravel(), x1.ravel()]

---

[來源: ch04] # ravel() 後，兩者都變成形狀為 (100000,) 的 1D 陣列

---

[來源: ch04] # np.c_ 然後將它們堆疊成形狀為 (100000, 2) 的陣列
```

---

[來源: ch04] ##### 為什麼在這裡使用 `ravel()`？

**目的：** 將 2D 座標網格轉換為可以餵給分類器的個別 (petal_length, petal_width) 點的列表。

**過程：**

1. **網格建立：** `x0` 和 `x1` 是 2D 網格 (200×500)，代表花瓣長度和寬度的所有組合
2. **展平：** `ravel()` 將每個 2D 網格轉換為 100,000 個點的 1D 向量
3. **堆疊：** `np.c_[x0.ravel(), x1.ravel()]` 建立一個 (100,000, 2) 陣列，其中每一列是一個 (長度, 寬度) 對
4. **預測：** 模型為所有 100,000 個點預測類別機率
5. **重塑回去：** `zz = y_proba[:, 1].reshape(x0.shape)` 將預測轉換回 2D 以用於等高線繪圖

---

[來源: ch04] # ravel() 之前
x0 = [[2.9, 2.91, 2.92],
      [2.9, 2.91, 2.92]]  # 形狀: (2, 3)

---

[來源: ch04] # ravel() 之後
x0.ravel() = [2.9, 2.91, 2.92, 2.9, 2.91, 2.92]  # 形狀: (6,)
```

---

[來源: ch04] ##### 關鍵差異：`ravel()` vs `flatten()`

- **`ravel()`**: 盡可能返回*視圖* (記憶體效率高，變更會影響原始陣列)
- **`flatten()`**: 總是返回*副本* (安全但使用更多記憶體)

在此程式碼中，優先使用 `ravel()`，因為我們只需要一個臨時的 1D 視圖用於預測，而不是永久副本。

**🎯 重點摘要:**

- **核心功能**: 使用兩個特徵訓練二元分類器，並視覺化決策邊界和機率分布
- **關鍵技術**: 
  - 使用 `np.meshgrid` 建立密集網格以繪製平滑的等高線
  - 使用 `ravel()` 展平陣列用於批量預測
  - 計算線性決策邊界的數學公式
- **最佳使用情境**: 理解邏輯迴歸如何在 2D 特徵空間中劃分類別，診斷模型行為

---

[來源: ch04] ## Softmax 迴歸

Softmax 迴歸將二元邏輯迴歸擴展到 *處理 K 個類別（K > 2）* 的情況。它的核心思想是為每個類別學習一組獨立的參數，然後使用 Softmax 函數將原始分數轉換為機率分布。

**關鍵特性：**

1. **多類別擴展**: 對於 K 個類別，模型學習 K 組參數向量（θ₀, θ₁, ..., θₖ₋₁），每組對應一個類別
2. **機率輸出**: Softmax 函數確保所有類別的預測機率總和為 1，符合機率公理
3. **互斥假設**: 適用於每個樣本只能屬於一個類別的情況（如 Iris 花卉分類）
4. **線性決策邊界**: 雖然看起來複雜，但 Softmax 迴歸本質上是線性分類器，在特徵空間中建立線性決策邊界

**數學原理：**

---

[來源: ch04] 於每個樣本只能屬於一個類別的情況（如 Iris 花卉分類）
4. **線性決策邊界**: 雖然看起來複雜，但 Softmax 迴歸本質上是線性分類器，在特徵空間中建立線性決策邊界

**數學原理：**

對於輸入 x，類別 k 的分數（logit）計算為：
$$s_k(x) = \theta_k^T x$$

然後使用 Softmax 函數將分數轉換為機率：
$$P(y=k|x) = \frac{e^{s_k(x)}}{\sum_{j=1}^{K} e^{s_j(x)}}$$

模型選擇機率最高的類別作為預測結果：
$$\hat{y} = \arg\max_k P(y=k|x)$$

**與二元邏輯迴歸的關係：**

---

[來源: ch04] {\sum_{j=1}^{K} e^{s_j(x)}}$$

模型選擇機率最高的類別作為預測結果：
$$\hat{y} = \arg\max_k P(y=k|x)$$

**與二元邏輯迴歸的關係：**

當 K=2 時，Softmax 迴歸退化為標準的邏輯迴歸。事實上，Scikit-Learn 的 `LogisticRegression` 會根據目標變數的類別數量自動選擇使用二元邏輯迴歸還是 Softmax 迴歸。

---

[來源: ch04] ### 範例 16: Softmax 迴歸基礎訓練

```python

---

[來源: ch04] # 使用兩個特徵提供更多資訊以區分三個物種
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
print(f"特徵矩陣形狀: {X.shape}")

---

[來源: ch04] # 0 = setosa, 1 = versicolor, 2 = virginica
y = iris.target
print(f"目標變數形狀: {y.shape}")
print(f"類別分布: setosa={np.sum(y==0)}, versicolor={np.sum(y==1)}, virginica={np.sum(y==2)}")

---

[來源: ch04] # 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

---

[來源: ch04] # 當 y 有超過 2 個類別時，LogisticRegression 自動使用 multinomial (Softmax) 迴歸
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

print(f"模型已訓練完成")
print(f"模型係數形狀: {softmax_reg.coef_.shape}")  # (3, 2) - 每個類別有一組係數
print(f"模型截距形狀: {softmax_reg.intercept_.shape}")  # (3,) - 每個類別有一個截距
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] - 每個類別有一組係數
print(f"模型截距形狀: {softmax_reg.intercept_.shape}")  # (3,) - 每個類別有一個截距
```

**✅ 程式碼逐行解析：**

1. **資料準備**:
  - `X = iris.data[["petal length (cm)", "petal width (cm)"]].values`: 提取兩個特徵（花瓣長度和寬度）
  - `y = iris.target`: 使用三個類別的標籤（0=setosa, 1=versicolor, 2=virginica）
  - `train_test_split()`: 分割訓練集和測試集

---

[來源: ch04] y = iris.target`: 使用三個類別的標籤（0=setosa, 1=versicolor, 2=virginica）
  - `train_test_split()`: 分割訓練集和測試集

2. **模型配置**:
  - `C=30`: 高正規化參數（正規化強度的倒數），允許模型更靈活地擬合訓練資料
  - `LogisticRegression` 會自動偵測多類別問題並使用 multinomial (Softmax) 迴歸

3. **模型結構**:
  - `coef_.shape = (3, 2)`: 每個類別（3個）對應一組權重（2個特徵）
  - `intercept_.shape = (3,)`: 每個類別有一個截距項

**🎯 重點摘要:**

---

[來源: ch04] `coef_.shape = (3, 2)`: 每個類別（3個）對應一組權重（2個特徵）
  - `intercept_.shape = (3,)`: 每個類別有一個截距項

**🎯 重點摘要:**

- **核心功能**: 將邏輯迴歸擴展到多類別分類，為每個類別學習獨立的參數集
- **自動偵測**: Scikit-Learn 會根據 y 的類別數量，由LogisticRegression自動選擇二元或多元邏輯迴歸
- **最佳使用情境**: 類別互斥的多類別分類問題（每個樣本只能屬於一個類別）

💡 **實際應用情境：** Softmax 迴歸（也稱為多項式邏輯迴歸）是邏輯迴歸的自然推廣，用於同時分類多個互斥的類別，如手寫數字辨識、物品分類、語言偵測等多類別分類問題。

---

[來源: ch04] ### 範例 17: Softmax 迴歸的三種預測方法

在此範例中，我們展示 Softmax 迴歸模型的三種主要預測方法：`predict()`、`predict_proba()` 和 `decision_function()`。這些方法提供不同層次的預測資訊，適用於不同的應用場景。

a. `predict()`：返回最高機率的類別標籤

b. `predict_proba()`：返回所有類別的機率估計

c. `decision_function()`：返回 Softmax 轉換前的原始分數（logits） 

```python

---

[來源: ch04] # 建立測試樣本：花瓣長度 5cm、寬度 2cm
test_sample = np.array([[5, 2]])

---

[來源: ch04] # 方法 1: predict() - 返回最高機率的類別標籤
predicted_class = softmax_reg.predict(test_sample)[0]
print(f"預測類別: {predicted_class}")
print(f"對應的物種名稱: {iris.target_names[predicted_class]}")

---

[來源: ch04] # 方法 2: predict_proba() - 返回所有三個類別的機率估計
proba = softmax_reg.predict_proba(test_sample).round(2)
print(f"各類別的預測機率: {proba}")
print(f"  - P(setosa)     = {proba[0, 0]:.2f}")
print(f"  - P(versicolor) = {proba[0, 1]:.2f}")
print(f"  - P(virginica)  = {proba[0, 2]:.2f}")
print(f"機率總和: {proba.sum():.2f}")  # 應該等於 1.0

---

[來源: ch04] # 方法 3: decision_function() - 返回 softmax 轉換前的原始分數
scores = softmax_reg.decision_function(test_sample)
print(f"原始類別分數: {scores.round(2)}")
```

**✅ 程式碼逐行解析：**

1. **predict() 方法**:
   - 返回最高機率的類別標籤（0, 1 或 2）
   - 適合只需要最終預測結果的情況

2. **predict_proba() 方法**:
   - 返回所有類別的機率估計（總和為 1.0）
   - 透過 Softmax 函數計算：$P(y=k|x) = \frac{e^{s_k(x)}}{\sum_{j=1}^{K} e^{s_j(x)}}$
   - 適合需要量化不確定性或設定自訂決策閾值的情況

---

[來源: ch04] 過 Softmax 函數計算：$P(y=k|x) = \frac{e^{s_k(x)}}{\sum_{j=1}^{K} e^{s_j(x)}}$
   - 適合需要量化不確定性或設定自訂決策閾值的情況

3. **decision_function() 方法**:
   - 返回 Softmax 轉換前的原始分數（logits）
      - 也就是每個類別的線性組合結果：$s_k(x) = \theta_k^T x$
   - 分數越高表示模型對該類別越有信心
   - 適合進階分析或自訂 Softmax 溫度參數

**🎯 重點摘要:**

---

[來源: ch04] 就是每個類別的線性組合結果：$s_k(x) = \theta_k^T x$
   - 分數越高表示模型對該類別越有信心
   - 適合進階分析或自訂 Softmax 溫度參數

**🎯 重點摘要:**

- **三種方法各有用途**: predict() 最簡單，predict_proba() 提供機率資訊，decision_function() 給出原始分數
- **機率解釋**: Softmax 確保所有機率為正數且總和為 1，符合機率公理
- **最佳使用情境**: 根據應用需求選擇合適的預測方法

---

[來源: ch04] ### 範例 18: Softmax 迴歸決策邊界視覺化

此範例展示如何視覺化 Softmax 迴歸的決策區域和機率分布，幫助理解模型如何在二維特徵空間中分類三個 Iris 物種。

```python
from matplotlib.colors import ListedColormap

---

[來源: ch04] # 建立自訂顏色映射: 淡黃色=setosa, 淡藍色=versicolor, 淡綠色=virginica
custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

---

[來源: ch04] # 建立覆蓋特徵空間的細緻網格
x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),      # 花瓣長度: 0 到 8 cm (500 個點)
    np.linspace(0, 3.5, 200).reshape(-1, 1),    # 花瓣寬度: 0 到 3.5 cm (200 個點)
)
print(f"網格形狀: x0.shape = {x0.shape}, x1.shape = {x1.shape}")

---

[來源: ch04] # 將 2D 網格轉換為適合模型預測的格式
X_new = np.c_[x0.ravel(), x1.ravel()]
print(f"用於預測的網格點數量: {X_new.shape[0]} (200×500 網格)")

---

[來源: ch04] # 計算網格中每個點的類別機率和預測
y_proba = softmax_reg.predict_proba(X_new)  # 形狀: (100000, 3)
y_predict = softmax_reg.predict(X_new)

---

[來源: ch04] # 提取 Iris versicolor 機率用於等高線 (中間類別，最能顯示漸變)
zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

---

[來源: ch04] # 繪製決策區域、等高線和資料點
plt.figure(figsize=(10, 4))

---

[來源: ch04] # 繪製所有訓練資料點
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica", markersize=8)
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor", markersize=8)
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa", markersize=8)

---

[來源: ch04] # 繪製決策區域 (用顏色填充)
plt.contourf(x0, x1, zz, cmap=custom_cmap, alpha=0.3)

---

[來源: ch04] # 繪製 versicolor 機率等高線
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1, fontsize=8)

plt.xlabel("花瓣長度 (cm)")
plt.ylabel("花瓣寬度 (cm)")
plt.legend(loc="center left")
plt.axis([0.5, 7, 0, 3.5])
plt.grid(alpha=0.3)
save_fig("softmax_regression_contour_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch04] #### 1) 視覺化設定
- `ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])`: 建立自訂顏色映射
  - 淡黃色 (#fafab0) 表示 setosa 區域
  - 淡藍色 (#9898ff) 表示 versicolor 區域  
  - 淡綠色 (#a0faa0) 表示 virginica 區域

---

[來源: ch04] #### 2) 網格建立
- `np.meshgrid()`: 建立覆蓋整個特徵空間的密集網格
  - x0: 花瓣長度方向，500 個點從 0 到 8 cm
  - x1: 花瓣寬度方向，200 個點從 0 到 3.5 cm
  - 總共 200×500 = 100,000 個網格點

- `np.c_[x0.ravel(), x1.ravel()]`: 將 2D 網格展平並堆疊為樣本矩陣
  - `ravel()` 將 2D 陣列 (200, 500) 展平為 1D 陣列 (100000,)
  - `np.c_[]` 將兩個 1D 陣列堆疊為 2D 矩陣 (100000, 2)

---

[來源: ch04] #### 3) 模型預測
- `predict_proba(X_new)`: 計算每個網格點屬於三個類別的機率
- `predict(X_new)`: 確定每個網格點的預測類別
- `y_proba[:, 1]`: 提取 versicolor 的機率，用於繪製等高線

---

[來源: ch04] #### 4) 視覺化元素
- **資料點**: 
  - 黃色圓圈 (yo) 表示 setosa
  - 藍色方塊 (bs) 表示 versicolor
  - 綠色三角形 (g^) 表示 virginica

- **決策區域**: `contourf()` 用顏色填充不同的預測區域
- **機率等高線**: `contour()` 顯示 versicolor 機率的變化，"hot" 顏色映射建立漸層效果

**🎯 重點摘要:**

---

[來源: ch04] `contourf()` 用顏色填充不同的預測區域
- **機率等高線**: `contour()` 顯示 versicolor 機率的變化，"hot" 顏色映射建立漸層效果

**🎯 重點摘要:**

- **決策邊界**: Softmax 迴歸建立複雜的非線性決策邊界來分離多個類別
- **機率漸變**: 等高線顯示機率如何在特徵空間中平滑轉變
- **類別區分**:
  - Iris setosa（左下角）以較小的花瓣測量值容易區分
  - Iris versicolor 和 virginica 之間的邊界更複雜，有些重疊
  - 接近訓練資料的區域顯示更高的模型信心
- **最佳使用情境**: 理解模型的決策過程，診斷分類困難的區域

---

---

[來源: ch04] ### Softmax 迴歸數學原理

**Softmax 函數**將原始分數（logits）轉換為機率：

$$P(y=k|x) = \frac{e^{s_k(x)}}{\sum_{j=1}^{K} e^{s_j(x)}}$$

其中：
- $s_k(x) = \theta_k^T x$ 是類別 k 的分數（logit）
- $K$ 是類別總數
- $e^{s_k(x)}$ 確保所有值為正數
- 分母正規化確保所有機率總和為 1

**關鍵特性**:
1. **輸出範圍**: 每個機率在 [0, 1] 之間
2. **正規化**: 所有類別機率總和等於 1
3. **單調性**: 分數越高，機率越大
4. **互斥性**: 適合每個樣本只能屬於一個類別的情況

---

[來源: ch04] ### 演算法選擇指南

| 情境 | 推薦演算法 | 原因 |
|------|-----------|------|
| 小資料集，特徵少 | 正規方程式 | 計算速度快，結果精確 |
| 大資料集 | SGD 或 Mini-batch GD | 訓練速度快，記憶體效率高 |
| 需要特徵選擇 | Lasso | 自動將不重要的特徵係數設為0 |
| 特徵相關性高 | Ridge 或 Elastic Net | 處理多重共線性問題 |
| 二元分類 | 邏輯迴歸 | 簡單高效，提供機率輸出 |
| 多類別分類 | Softmax 迴歸 | 自然的多類別擴展 |

---

[來源: ch04] ### 最佳實踐建議

1. **資料前處理**
   - 特徵縮放對梯度下降至關重要
   - 檢查並處理離群值
   - 考慮多項式特徵以捕捉非線性關係

2. **超參數調整**
   - 使用交叉驗證選擇最佳的正規化參數
   - 監控學習曲線以診斷偏差-變異問題
   - 考慮使用 Grid Search 或 Random Search

3. **模型評估**
   - 不要只看訓練誤差，驗證集表現更重要
   - 使用適當的評估指標（MSE、RMSE、R²等）
   - 繪製預測值 vs 實際值的散點圖

4. **效能優化**
   - 對大資料集優先考慮 SGD 或 Mini-batch GD
   - 使用 warm_start 參數進行增量學習
   - 考慮使用 Early Stopping 節省訓練時間

---

[來源: ch04] ## 常見問答 (FAQ)

**Q1: 何時使用正規方程式，何時使用梯度下降？**

A: 當特徵數量少於10,000且記憶體充足時，正規方程式更快更精確。對於更大的資料集或線上學習場景，梯度下降更合適。

**Q2: Ridge 和 Lasso 有什麼區別？**

A: Ridge 使用 L2 正規化，會縮小所有參數但不會將其設為0。Lasso 使用 L1 正規化，會將一些參數完全設為0，實現特徵選擇。

**Q3: 如何選擇學習率？**

A: 可以從 0.001 開始嘗試，觀察損失函數的變化。如果損失震盪或發散，降低學習率；如果收斂太慢，增加學習率。也可以使用學習率衰減策略。

**Q4: 什麼時候需要多項式特徵？**

A: 當學習曲線顯示高偏差（訓練和驗證誤差都很高）時，可能需要增加模型複雜度，這時可以嘗試多項式特徵。

**Q5: 如何診斷過擬合和欠擬合？**

---

[來源: ch04] 策略。

**Q4: 什麼時候需要多項式特徵？**

A: 當學習曲線顯示高偏差（訓練和驗證誤差都很高）時，可能需要增加模型複雜度，這時可以嘗試多項式特徵。

**Q5: 如何診斷過擬合和欠擬合？**

A: 透過學習曲線：如果訓練誤差低但驗證誤差高，是過擬合；如果兩者都高，是欠擬合。可以分別透過正規化和增加模型複雜度來解決。

---

[來源: ch04] ## 推薦標籤 (Suggested Hashtags)

#Python #機器學習 #程式設計 #教學 #線性模型 #數據科學 #AI開發

---

[來源: ch04 | 類型: cheatsheet] # Ch04 速查表：Training Linear Models

> **核心主旨**：線性模型的訓練方法與正則化 —— Ridge/Lasso/ElasticNet 是防止 overfitting 的利器。

---

---

[來源: ch04 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Normal Equation | 直接計算 $\hat{\theta} = (X^TX)^{-1}X^Ty$ | 特徵數 < 10k，一步到位 |
| Batch GD | 每次用全部資料計算梯度 | 小資料、凸函數 |
| Stochastic GD (SGD) | 每次用 1 個樣本更新 | 大資料集、online learning |
| Mini-batch GD | 每次用小批次（32-256）更新 | 實務上最常用 |
| Polynomial Regression | 加入 $x^2, x^3, ...$ 特徵擬合非線性 | 非線性資料但想用線性模型 |
| Ridge (L2) | 損失加 $\alpha \sum \theta_i^2$，縮小所有權重 | 所有特徵都有用，需防 overfitting |
| Lasso (L1) | 損失加 $\alpha \sum |\theta_i|$，可將部分權重降為 0 | 特徵選擇（稀疏模型） |
| ElasticNet | L1 + L2 的混合，有 `l1_ratio` 控制比例 | 特徵數 >> 樣本數，或特徵相關時 |
| Early Stopping | 監控 validation loss，停在最低點 | 防止 overfitting 的高效方法 |
| LogisticRegression | 輸出機率，用 sigmoid，做二元分類 | 線性可分的分類問題 |


---

[來源: ch04 | 類型: cheatsheet] ation loss，停在最低點 | 防止 overfitting 的高效方法 |
| LogisticRegression | 輸出機率，用 sigmoid，做二元分類 | 線性可分的分類問題 |


---

---

[來源: ch04 | 類型: cheatsheet] | sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `LinearRegression` | – | 最小二乘法迴歸（Normal Equation） |
| `Ridge` | `alpha=1.0` | L2 正則化迴歸 |
| `Lasso` | `alpha=0.1` | L1 正則化迴歸（特徵選擇） |
| `ElasticNet` | `alpha=0.1`, `l1_ratio=0.5` | L1+L2 混合 |
| `SGDRegressor` | `penalty="l2"`, `eta0=0.1`, `max_iter=1000` | SGD 迴歸 |
| `SGDClassifier` | `loss="log_loss"`, `penalty="l2"` | SGD 分類 |
| `PolynomialFeatures` | `degree=2`, `include_bias=False` | 多項式特徵展開 |
| `LogisticRegression` | `C=1.0` (=1/alpha), `max_iter=1000` | 邏輯迴歸 |
| `learning_curve` | – | 繪製學習曲線，診斷 bias/variance |


---

[來源: ch04 | 類型: cheatsheet] ` | `C=1.0` (=1/alpha), `max_iter=1000` | 邏輯迴歸 |
| `learning_curve` | – | 繪製學習曲線，診斷 bias/variance |


---

---

[來源: ch04 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve

---

[來源: ch04 | 類型: cheatsheet] # 線性迴歸（Normal Equation）
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.intercept_, lin_reg.coef_)

---

[來源: ch04 | 類型: cheatsheet] # 多項式迴歸
poly_pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])
poly_pipeline.fit(X_train, y_train)

---

[來源: ch04 | 類型: cheatsheet] # Ridge 正則化（L2）
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

---

[來源: ch04 | 類型: cheatsheet] # Lasso 正則化（L1）—— 稀疏模型
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"非零係數數量: {np.sum(lasso.coef_ != 0)}")

---

[來源: ch04 | 類型: cheatsheet] # ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

---

[來源: ch04 | 類型: cheatsheet] # Early Stopping (SGD)
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty="l2",
                       eta0=0.01, n_iter_no_change=10, random_state=42)
sgd_reg.fit(X_train, y_train.ravel())

---

[來源: ch04 | 類型: cheatsheet] # 邏輯迴歸（多類別）
log_reg = LogisticRegression(C=10, max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
print(log_reg.predict_proba(X_test[:3]))  # 輸出機率
```

---

---

[來源: ch04 | 類型: cheatsheet] ## 4. 常見陷阱

- **Ridge/Lasso 前必須 Scaling**：L1/L2 懲罰對特徵尺度敏感，務必先做 `StandardScaler`。
- **`C` vs `alpha`**：`LogisticRegression`、`SVC` 用 `C`（越大正則化越弱）；`Ridge`、`Lasso` 用 `alpha`（越大正則化越強），兩者互為倒數概念。
- **Lasso 不穩定性**：當特徵相關時，Lasso 可能隨機選其中一個降為 0；ElasticNet 更穩健。
- **`max_iter` 不夠大**：LogisticRegression 預設 `max_iter=100`，複雜問題常需要 `1000+`。

---

---

[來源: ch04 | 類型: cheatsheet] ## 5. 決策指南

```
選哪種正則化？
├── 所有特徵都有用，只需防止 overfitting → Ridge (L2)
├── 特徵多，需要自動選特徵（稀疏） → Lasso (L1)
└── 特徵相關 or 特徵數 >> 樣本數   → ElasticNet (L1+L2)

選哪種訓練方法？
├── 特徵數 < 10k，資料量 < 1M → LinearRegression (Normal Equation, 快速精確)
├── 特徵數 > 10k 或資料量 > 1M → SGDRegressor (增量學習)
└── 需要 online learning          → SGDRegressor (partial_fit)

正則化強度 alpha：
└── 用 GridSearchCV 搜索 [0.01, 0.1, 1, 10, 100]
```

---

[來源: ch04 | 類型: cheatsheet] → SGDRegressor (partial_fit)

正則化強度 alpha：
└── 用 GridSearchCV 搜索 [0.01, 0.1, 1, 10, 100]
```

**學習曲線診斷**：
- Training error 高 → Underfitting（增加模型複雜度或特徵）
- Training/Val error gap 大 → Overfitting（增加正則化、減少特徵、更多資料）

---

[來源: ch04 | 類型: handout] # 課程講義：訓練線性模型 (Chapter 04)

線性模型是機器學習的基石，許多複雜模型（神經網路、SVM）的核心優化思想都源於此。本章帶你從**正規方程式**、**梯度下降**，到**正則化技術**，完整理解「模型如何被訓練」的數學本質。理解這些原理，你才能在模型表現不佳時，準確判斷是欠擬合還是過擬合，並選擇正確的解決策略。

---

---

[來源: ch04 | 類型: handout] ### 理論背景

線性迴歸假設目標值是特徵的線性組合加上雜訊：

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n = \boldsymbol{\theta}^T \mathbf{x}$$

以向量化形式，預測整個訓練集：

$$\hat{\mathbf{y}} = \mathbf{X} \boldsymbol{\theta}$$

其中 $\mathbf{X}$ 是加了偏差欄（全 1）的特徵矩陣。

最小化 MSE 的解析解（**正規方程式**）：

$$\hat{\boldsymbol{\theta}} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}$$

---

[來源: ch04 | 類型: handout] ：

$$\hat{\boldsymbol{\theta}} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}$$

**計算複雜度**：矩陣求逆的複雜度為 $O(n^{2.4} \sim O(n^3)$，$n$ 是特徵數。特徵數超過數千時改用梯度下降更有效率。

---

[來源: ch04 | 類型: handout] ### 核心代碼

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import add_dummy_feature

np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

---

[來源: ch04 | 類型: handout] # 方法一：手動正規方程式
X_b = add_dummy_feature(X)  # 新增偏差項 x0 = 1
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"theta_best: {theta_best.ravel()}")  # 接近 [4, 3]

---

[來源: ch04 | 類型: handout] # 方法二：Scikit-Learn（內部使用 SVD，更穩健）
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(f"intercept: {lin_reg.intercept_}, coef: {lin_reg.coef_}")

---

[來源: ch04 | 類型: handout] # 預測
X_new = np.array([[0], [2]])
print(lin_reg.predict(X_new))
```

---

[來源: ch04 | 類型: handout] ### 補充練習 1

**理論題：** 正規方程式需要計算 $(\mathbf{X}^T\mathbf{X})^{-1}$，若矩陣奇異（不可逆），會發生什麼？什麼情況下會出現奇異矩陣？

**實作題：** 生成一個有 3 個特徵的資料集（其中第 3 個特徵是前兩個的線性組合），嘗試用正規方程式求解，觀察 `np.linalg.inv` 的行為；再改用 `LinearRegression` 觀察是否正常運作（提示：它使用 SVD 的 pseudoinverse）。

---

---

[來源: ch04 | 類型: handout] ### 理論背景

梯度下降通過反覆沿梯度**反方向**移動來最小化損失函數：

$$\boldsymbol{\theta}^{(\text{next})} = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta})$$

其中 $\eta$ 是**學習率 (learning rate)**，$\nabla_{\boldsymbol{\theta}} \text{MSE} = \frac{2}{m} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$。

---

[來源: ch04 | 類型: handout] symbol{\theta}} \text{MSE} = \frac{2}{m} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$。

| 變體 | 每步計算量 | 收斂特性 | 適用場景 |
|------|----------|---------|---------|
| **批次梯度下降 (BGD)** | 全部 $m$ 筆 | 穩定，慢 | 小資料集，最終微調 |
| **隨機梯度下降 (SGD)** | 1 筆 | 震盪，快 | 大資料集，線上學習 |
| **小批次梯度下降 (Mini-batch)** | $b$ 筆（$32 \sim 256$）| 均衡 | 最常用（尤其深度學習） |


**學習率排程 (Learning Rate Schedule)**：一開始用大學習率快速逼近，後期用小學習率精確收斂。

---

[來源: ch04 | 類型: handout] # 批次梯度下降（手動實作，理解原理）
eta = 0.1         # 學習率
n_epochs = 1000
m_train = len(X_b)

np.random.seed(42)
theta = np.random.randn(2, 1)  # 隨機初始化

for epoch in range(n_epochs):
    gradients = 2 / m_train * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

print(f"BGD theta: {theta.ravel()}")

---

[來源: ch04 | 類型: handout] # Scikit-Learn SGDRegressor
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5,
                       penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())
print(f"SGD intercept: {sgd_reg.intercept_}, coef: {sgd_reg.coef_}")
```

---

[來源: ch04 | 類型: handout] ### 補充練習 2

**理論題：** 學習率 $\eta$ 設定太大或太小，各會造成什麼問題？若損失函數是「碗形」（凸函數），BGD 保證找到全域最小值嗎？若不是凸函數呢？

**實作題：** 對同一個線性迴歸問題，分別用 $\eta = 0.001$、$0.1$、$0.5$ 訓練 BGD，繪製每個 epoch 的訓練損失曲線，直觀觀察學習率對收斂的影響。

---

---

[來源: ch04 | 類型: handout] ### 理論背景

**多項式迴歸**：將特徵轉換為高次冪後，套用線性迴歸，本質上仍是線性模型（對參數線性）。

$$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_d x^d$$

**學習曲線 (Learning Curves)**：同時繪製訓練誤差和驗證誤差隨訓練樣本數增加的變化。

| 現象 | 診斷 |
|------|------|
| 訓練誤差高 & 驗證誤差高，且兩者接近 | **欠擬合**（模型太簡單） |
| 訓練誤差低 & 驗證誤差高，差距大 | **過擬合**（模型太複雜） |


**偏差-變異數分解**：

---

[來源: ch04 | 類型: handout] ---|
| 訓練誤差高 & 驗證誤差高，且兩者接近 | **欠擬合**（模型太簡單） |
| 訓練誤差低 & 驗證誤差高，差距大 | **過擬合**（模型太複雜） |


**偏差-變異數分解**：

$$\text{泛化誤差} = \underbrace{\text{偏差}^2}_{\text{欠擬合}} + \underbrace{\text{變異數}}_{\text{過擬合}} + \text{不可減少的雜訊}$$

---

[來源: ch04 | 類型: handout] ### 核心代碼

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

---

[來源: ch04 | 類型: handout] # 多項式迴歸
poly_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])

---

[來源: ch04 | 類型: handout] # 學習曲線
train_sizes, train_scores, valid_scores = learning_curve(
    poly_reg, X, y.ravel(),
    train_sizes=np.linspace(0.01, 1.0, 30),
    cv=5, scoring="neg_root_mean_squared_error"
)

---

[來源: ch04 | 類型: handout] vel(),
    train_sizes=np.linspace(0.01, 1.0, 30),
    cv=5, scoring="neg_root_mean_squared_error"
)

plt.plot(train_sizes, -train_scores.mean(axis=1), "r-+", label="Training")
plt.plot(train_sizes, -valid_scores.mean(axis=1), "b-",  label="Validation")
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.legend()
plt.title(f"Learning Curves (Polynomial degree={poly_reg['poly_features'].degree})")
plt.grid()
```

---

[來源: ch04 | 類型: handout] ### 補充練習 3

**理論題：** 觀察 `degree=1`、`degree=2`、`degree=10` 的學習曲線。哪個 degree 可能欠擬合？哪個可能過擬合？如何從曲線形狀判斷？

**實作題：** 生成一個符合 $y = 0.5x^2 + x + 2 + \text{noise}$ 的資料集（100 樣本），分別用 `degree=1`、`degree=2`、`degree=20` 的多項式迴歸擬合，繪製三個模型在資料上的擬合曲線，觀察過擬合與欠擬合現象。

---

---

[來源: ch04 | 類型: handout] ## 4. 正則化線性模型：Ridge、Lasso、ElasticNet

---

[來源: ch04 | 類型: handout] ### 理論背景

正則化在損失函數中加入懲罰項，限制模型複雜度：

**Ridge 迴歸（L2 正則化）**：

$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \sum_{i=1}^{n} \theta_i^2$$

**Lasso 迴歸（L1 正則化）**：

$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \sum_{i=1}^{n}

 |\theta_i|$$


Lasso 的特點：傾向將不重要的特徵係數壓縮為**完全 0**，實現稀疏解（自動特徵選擇）。

**ElasticNet**：L1 + L2 的混合：

---

[來源: ch04 | 類型: handout] 1}^{n}

 |\theta_i|$$


Lasso 的特點：傾向將不重要的特徵係數壓縮為**完全 0**，實現稀疏解（自動特徵選擇）。

**ElasticNet**：L1 + L2 的混合：

$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r \alpha \sum

|\theta_i| + \frac{1-r}{2} \alpha \sum\theta_i^2$$


| 方法 | 適用場景 | 稀疏性 |
|------|---------|--------|
| Ridge | 多數特徵都有貢獻 | 無（係數縮小但不為零） |
| Lasso | 特徵多，但只有少數重要 | 有（自動特徵選擇） |
| ElasticNet | 高度相關特徵群組 | 有（介於兩者之間） |

---

[來源: ch04 | 類型: handout] ### 核心代碼

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X, y)
print(f"Ridge: {ridge_reg.predict([[1.5]])}")

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
print(f"Lasso: {lasso_reg.predict([[1.5]])}")
print(f"Lasso coef (稀疏): {lasso_reg.coef_}")

en_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
en_reg.fit(X, y)

---

[來源: ch04 | 類型: handout] # Early Stopping（用於 SGD 訓練的正則化）
from sklearn.linear_model import SGDRegressor

sgd_early = SGDRegressor(
    max_iter=1,
    warm_start=True,  # 每次 fit() 從上次停的地方繼續
    penalty=None,
    learning_rate="constant",
    eta0=0.002,
    random_state=42
)

---

[來源: ch04 | 類型: handout] ### 補充練習 4

**理論題：** 若 $\alpha = 0$，Ridge 和 Lasso 退化為什麼模型？若 $\alpha \to \infty$，模型的係數會變成什麼？

**實作題：** 在一個有 100 個特徵（但只有 10 個真正有用）的資料集上（用 `make_regression(n_features=100, n_informative=10)`），比較 Ridge 和 Lasso 的係數分佈，驗證 Lasso 確實將不重要的係數歸零。

---

---

[來源: ch04 | 類型: handout] ### 理論背景

**邏輯迴歸 (Logistic Regression)**：用於二元分類，輸出屬於正類的機率：

$$\hat{p} = \sigma(\boldsymbol{\theta}^T \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^T \mathbf{x}}}$$

Sigmoid 函數將任意實數映射到 $(0, 1)$。決策邊界：$\hat{p} \geq 0.5 \Leftrightarrow \boldsymbol{\theta}^T \mathbf{x} \geq 0$。

**對數損失（Log Loss / Binary Cross-Entropy）**：

---

[來源: ch04 | 類型: handout] Leftrightarrow \boldsymbol{\theta}^T \mathbf{x} \geq 0$。

**對數損失（Log Loss / Binary Cross-Entropy）**：

$$J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{p}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

**Softmax 迴歸**：邏輯迴歸推廣到多類別，對每個類別 $k$ 計算分數：

$$s_k(\mathbf{x}) = \boldsymbol{\theta}_k^T \mathbf{x}$$

$$\hat{p}_k = \frac{e^{s_k}}{\sum_{j=1}^{K} e^{s_j}}$$

---

[來源: ch04 | 類型: handout] ### 核心代碼

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = (iris.target == 2).astype(int)  # Iris virginica vs rest

---

[來源: ch04 | 類型: handout] # 二元邏輯迴歸
log_reg = LogisticRegression(C=1.0, random_state=42)
log_reg.fit(X_iris, y_iris)
print(log_reg.predict_proba([[5, 2]]))  # [P(非 virginica), P(virginica)]

---

[來源: ch04 | 類型: handout] # Softmax 迴歸（多類別，solver="lbfgs" 預設支援）
softmax_reg = LogisticRegression(C=30, multi_class="multinomial",
                                 solver="lbfgs", random_state=42)
softmax_reg.fit(iris.data, iris.target)
print(softmax_reg.predict([[5, 3, 1.5, 0.3]]))  # 預測類別
print(softmax_reg.predict_proba([[5, 3, 1.5, 0.3]]))  # 三類別機率
```

---

[來源: ch04 | 類型: handout] ### 補充練習 5

**理論題：** 邏輯迴歸的損失函數（Log Loss）為何不用 MSE？MSE 在此有什麼問題（提示：凸性）？

**實作題：** 使用 Iris 資料集（三類別），訓練 `LogisticRegression(multi_class="multinomial")`，繪製決策邊界（三類別顏色不同），並計算測試集的 `classification_report`。

---

---

[來源: ch04 | 類型: handout] ## 結論

本章建立了線性模型訓練的完整理論基礎：

- **正規方程式**提供解析解，適用於小型資料集
- **梯度下降**用迭代方式最小化損失，是深度學習的基礎
- **多項式迴歸 + 學習曲線**診斷過擬合/欠擬合
- **Ridge/Lasso/ElasticNet**透過正則化控制模型複雜度
- **邏輯迴歸與 Softmax**將線性迴歸擴展到分類問題

下一章（Ch05）將介紹 SVM，它以完全不同的角度（最大化邊界）來解決分類問題。

---

---

[來源: ch04 | 類型: handout] ## 課後作業

**作業：正則化對比實驗**

使用 `sklearn.datasets.load_diabetes()`（糖尿病資料集）：

1. 用 5-fold CV 比較無正則化的 `LinearRegression`、`Ridge(alpha=0.1/1/10)`、`Lasso(alpha=0.01/0.1/1.0)` 的 RMSE，找出最佳的模型與 `alpha` 值。
2. 對最佳 `Lasso` 模型，印出所有係數，確認哪些特徵被完全歸零（系數 = 0），這些特徵在醫學上是否合理？
3. 繪製**正規化路徑圖**（x 軸為 `log(alpha)`，y 軸為各特徵係數），觀察 `alpha` 如何影響係數大小。

---

[來源: ch04 | 類型: tutorial] [標題: 訓練線性模型完整指南：梯度下降、正則化與邏輯回歸實戰 | 描述: 深入線性回歸、梯度下降三種變體、多項式回歸、Ridge/Lasso/Elastic Net 正則化、Early Stopping、邏輯回歸和 Softmax 迴歸。包含 Scikit-Learn 實戰程式碼與逐行解析。 | 關鍵字: Python, 線性回歸, 梯度下降, 正則化, Ridge, Lasso, 邏輯回歸, Softmax, 機器學習, Scikit-Learn]
# 訓練線性模型：梯度下降、正則化到邏輯回歸完整實戰

線性模型（Linear Models）是機器學習的基石。看似簡單，卻是理解梯度下降和正則化的最佳起點——這兩個概念貫穿整個深度學習。本教學從最小二乘法出發，帶你深入了解每種梯度下降的差異，以及如何用正則化控制過擬合。

---

[來源: ch04 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **正規方程式（Normal Equation）** 一步求解，但計算複雜度為 O(n³)，特徵多時不可行
- **批次/隨機/小批次梯度下降（Batch/SGD/Mini-batch GD）** 各有速度與穩定性的取捨
- **學習率（Learning Rate）** 是最重要的超參數之一，過大發散、過小收斂慢
- **Ridge/Lasso/Elastic Net** 分別對應 L2/L1/混合正則化，各有適用場景
- **邏輯回歸（Logistic Regression）** 輸出機率，適合二元分類；**Softmax** 推廣至多類別

---

---

[來源: ch04 | 類型: tutorial] ## 線性回歸 (Linear Regression)

💡 **實際應用情境：** 預測電費（用電度數→費用）、預測廣告效果（投放金額→銷售額），任何具有線性關係的回歸問題都可以從線性回歸開始。

線性回歸模型：

$$\hat{y} = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n = \theta^T \mathbf{x}$$

訓練目標：最小化均方誤差（MSE）：$\mathcal{L} = \frac{1}{m}\|\mathbf{X}\theta - \mathbf{y}\|^2$

---

[來源: ch04 | 類型: tutorial] ### 範例 1: 正規方程式 vs Sklearn LinearRegression

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)

---

[來源: ch04 | 類型: tutorial] # 生成線性資料 y = 4 + 3x + noise
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.ravel() + np.random.randn(100)

---

[來源: ch04 | 類型: tutorial] # 方法 1：正規方程式（直接求解析解）
X_b = np.c_[np.ones((100, 1)), X]    # 加入偏置項 x0 = 1
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  # θ = (XᵀX)⁻¹ Xᵀy
print(f"正規方程式 θ: {theta_best}")   # 應接近 [4, 3]

---

[來源: ch04 | 類型: tutorial] # 方法 2：Sklearn（使用 SVD，更穩定）
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(f"Sklearn 截距: {lin_reg.intercept_:.3f}, 係數: {lin_reg.coef_[0]:.3f}")

---

[來源: ch04 | 類型: tutorial] # 預測
X_new = np.array([[0], [2]])
print(f"預測: {lin_reg.predict(X_new)}")
```

**✅ 程式碼逐行解析：**

1. `np.c_[np.ones((100, 1)), X]`: 加入全 1 的第 0 列作為偏置項（截距）
2. `np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y`: 矩陣形式的正規方程式 $\hat{\theta} = (X^TX)^{-1}X^Ty$
3. Sklearn 內部使用 SVD 分解（奇異值分解），比直接求逆更數值穩定

**🎯 重點摘要:**

- 正規方程式的計算複雜度是 O(n³)（n 為特徵數）：特徵數 < 10,000 時可用，否則改用梯度下降
- `LinearRegression` 無超參數，訓練後應檢查殘差是否呈常態分佈（診斷模型假設是否成立）

---

---

[來源: ch04 | 類型: tutorial] ## ⬇️ 梯度下降三種變體

💡 **實際應用情境：** 梯度下降是神經網路訓練的核心。理解三種變體有助於你在實際專案中選擇合適的優化器，並調整學習率等關鍵超參數。

---

[來源: ch04 | 類型: tutorial] ### 範例 2: 批次梯度下降 (Batch Gradient Descent)

```python

---

[來源: ch04 | 類型: tutorial] # 批次梯度下降（每次迭代使用全部訓練資料）
eta = 0.1          # 學習率
n_iterations = 1000
m = len(X_b)       # 訓練樣本數

theta = np.random.randn(2, 1)  # 隨機初始化

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T @ (X_b @ theta - y.reshape(-1, 1))  # MSE 梯度
    theta = theta - eta * gradients  # 參數更新

print(f"批次 GD θ: {theta.ravel()}")  # 應接近 [4, 3]
```

**✅ 程式碼逐行解析：**

1. `2/m * X_b.T @ (X_b @ theta - y)`: MSE 對 θ 的梯度 $\nabla_\theta = \frac{2}{m} X^T(X\theta - y)$
2. `theta = theta - eta * gradients`: 沿負梯度方向移動一步（學習率控制步長）
3. 批次 GD 每步需要計算全部 m 個樣本的梯度，大資料集時計算成本高但收斂穩定

---

[來源: ch04 | 類型: tutorial] ### 範例 3: 隨機梯度下降 (Stochastic GD)

```python
from sklearn.linear_model import SGDRegressor

---

[來源: ch04 | 類型: tutorial] # 每次只用一個隨機樣本更新參數（快但有噪音）
sgd_reg = SGDRegressor(
    max_iter=1000,
    tol=1e-5,
    penalty=None,      # 無正則化
    eta0=0.01,         # 初始學習率
    random_state=42
)
sgd_reg.fit(X, y.ravel())
print(f"SGD θ: [{sgd_reg.intercept_[0]:.3f}, {sgd_reg.coef_[0]:.3f}]")
```

| 方法 | 每步使用資料量 | 速度 | 穩定性 | 適合場景 |
|------|--------------|------|--------|---------|
| 批次 GD | 全部 m 個 | 慢 | 高 | 小資料集 |
| 隨機 GD (SGD) | 1 個 | 快 | 低 | 大資料集、線上學習 |
| 小批次 GD | b 個 (32~256) | 中 | 中 | 深度學習標準 |

**🎯 重點摘要:**

- 學習率過高：損失函數「跳躍」甚至發散；過低：收斂極慢
- **學習率排程（Learning Rate Schedule）**：先用大學習率快速靠近，再縮小精細調整

---

---

[來源: ch04 | 類型: tutorial] ## 多項式回歸 (Polynomial Regression)

---

[來源: ch04 | 類型: tutorial] ### 範例 4: 用線性模型擬合非線性資料

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)
m = 100
X_poly = 6 * np.random.rand(m, 1) - 3          # X ∈ [-3, 3]
y_poly = 0.5 * X_poly**2 + X_poly + 2 + np.random.randn(m, 1)  # 二次函數 + 雜訊

---

[來源: ch04 | 類型: tutorial] # PolynomialFeatures：將 X 擴展為 [1, X, X²]
poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),  # 生成 X, X²
    LinearRegression()
)
poly_model.fit(X_poly, y_poly)

---

[來源: ch04 | 類型: tutorial] # 實際係數應接近 [1, 0.5]（線性項和平方項）
lin_reg_poly = poly_model.named_steps["linearregression"]
print(f"係數: {lin_reg_poly.coef_}")
print(f"截距: {lin_reg_poly.intercept_}")
```

**✅ 程式碼逐行解析：**

1. `PolynomialFeatures(degree=2)`: 將輸入 `[X]` 轉換為 `[X, X²]`（不含偏置項，LinearRegression 自動加）
2. `make_pipeline(...)`: 比 `Pipeline([...])` 更簡潔，自動以小寫類名命名步驟

**🎯 重點摘要:**

- 多項式回歸本質上是線性模型（對 θ 線性），只是對輸入特徵做非線性變換
- degree=300 的多項式可以完美擬合訓練集，但對新資料毫無用處（過擬合）

---

---

[來源: ch04 | 類型: tutorial] ```python
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def plot_learning_curve(model, X, y, cv=3):
    """繪製訓練集和驗證集的學習曲線"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    train_rmse = -train_scores.mean(axis=1)
    val_rmse   = -val_scores.mean(axis=1)

    plt.plot(train_sizes, train_rmse, "b-o", label="訓練誤差")
    plt.plot(train_sizes, val_rmse,   "r-o", label="驗證誤差")
    plt.xlabel("訓練樣本數")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.sca(axes[0])
plt.title("線性回歸（欠擬合）")
plot_learning_curve(LinearRegression(), X_poly, y_poly)

plt.sca(axes[1])
plt.title("10 次多項式（過擬合）")
plot_learning_curve(
    make_pipeline(PolynomialFeatures(degree=10), LinearRegression()),
    X_poly, y_poly
)
plt.show()
```

---

[來源: ch04 | 類型: tutorial] ake_pipeline(PolynomialFeatures(degree=10), LinearRegression()),
    X_poly, y_poly
)
plt.show()
```

**🎯 重點摘要:**

- **欠擬合（高偏差）**：訓練誤差和驗證誤差都高，且隨資料量增加不顯著改善
- **過擬合（高方差）**：訓練誤差很低，驗證誤差比訓練誤差高很多（間距大）
- 解法：欠擬合→增加特徵或提升模型複雜度；過擬合→增加資料或正則化

---

---

[來源: ch04 | 類型: tutorial] ### 範例 6: Ridge、Lasso、Elastic Net 比較

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

np.random.seed(42)
X_reg = 3 * np.random.rand(100, 1)
y_reg = 1 + 0.5 * X_reg.ravel() + np.random.randn(100) / 1.5

---

[來源: ch04 | 類型: tutorial] # Ridge（L2 正則化）：縮小所有係數，但不會歸零
ridge_reg = Ridge(alpha=1.0)       # alpha 越大，正則化越強
ridge_reg.fit(X_reg, y_reg)

---

[來源: ch04 | 類型: tutorial] # Lasso（L1 正則化）：係數可被壓縮至 0（自動特徵選擇！）
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_reg, y_reg)

---

[來源: ch04 | 類型: tutorial] # Elastic Net（L1 + L2 混合）
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50% L1 + 50% L2
elastic_net.fit(X_reg, y_reg)

print(f"Ridge 係數:  {ridge_reg.coef_[0]:.4f}")
print(f"Lasso 係數:  {lasso_reg.coef_[0]:.4f}")
print(f"ElasticNet:  {elastic_net.coef_[0]:.4f}")
```

| 方法 | 正則化項 | 係數特性 | 適用場景 |
|------|---------|---------|---------|
| Ridge | $\alpha \|\theta\|_2^2$ | 縮小但不歸零 | 所有特徵都有用時 |
| Lasso | $\alpha \|\theta\|_1$ | 稀疏（部分歸零） | 特徵選擇、高維資料 |
| Elastic Net | L1 + L2 混合 | 稀疏+穩定 | 相關特徵多時 |

**🎯 重點摘要:**

- `alpha=0` → 無正則化（退化為普通線性回歸）
- Lasso 可用作**特徵選擇**工具：不重要的特徵係數會被壓縮至 0

---

---

[來源: ch04 | 類型: tutorial] ### 範例 7: 邏輯回歸二元分類

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

---

[來源: ch04 | 類型: tutorial] # 使用鳶尾花資料集（只取前兩個特徵便於視覺化）
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = (iris.target == 2)  # 只分辨 Virginica vs 其他（二元）

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_iris, y_iris)

---

[來源: ch04 | 類型: tutorial] # 輸出機率（不只是預測類別）
print(f"預測機率: {log_reg.predict_proba([[5, 2]])[0]}")  # [P(非Virginica), P(Virginica)]
print(f"決策邊界係數: {log_reg.coef_[0]}")
```

```python

---

[來源: ch04 | 類型: tutorial] # Softmax 回歸：多類別邏輯回歸
softmax_reg = LogisticRegression(
    multi_class="multinomial",  # Softmax（而非 OvR）
    solver="lbfgs",
    C=30,                       # C = 1/alpha（C 越小正則化越強）
    random_state=42
)
softmax_reg.fit(X_iris, iris.target)  # 預測 0/1/2 三類
print(f"Softmax 準確率: {softmax_reg.score(X_iris, iris.target):.4f}")
```

**✅ 程式碼逐行解析：**

1. 邏輯回歸輸出 Sigmoid 機率：$\hat{p} = \sigma(\theta^T \mathbf{x}) = \frac{1}{1+e^{-\theta^T \mathbf{x}}}$
2. `multi_class="multinomial"`: 使用 Softmax 而非 OvR，輸出所有類別的機率（加總為 1）
3. `C=30`: 弱正則化（高 C = 允許更複雜邊界）

**🎯 重點摘要:**

---

[來源: ch04 | 類型: tutorial] class="multinomial"`: 使用 Softmax 而非 OvR，輸出所有類別的機率（加總為 1）
3. `C=30`: 弱正則化（高 C = 允許更複雜邊界）

**🎯 重點摘要:**

- 邏輯回歸輸出**機率**（非直接 0/1），適合需要置信度的場景
- Softmax 的損失函數（交叉熵）是 NLP 和深度學習中最常用的損失函數之一

---

---

[來源: ch04 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: 批次 GD 和 SGD 哪個更好？**

A: 取決於資料規模。資料量 < 10 萬可用批次 GD；更大的資料集用 SGD 或小批次 GD。深度學習中幾乎都使用小批次 GD（batch_size=32~256）。

**Q2: Ridge 和 Lasso 如何選擇？**

A: 若你認為**所有特徵都有用**，用 Ridge；若認為**大多數特徵無關**（需要特徵選擇），用 Lasso；若不確定，先試 Elastic Net。

**Q3: 邏輯回歸為什麼叫「回歸」？**

A: 因為它輸出的是**機率**（連續值），本質上在對 log-odds 做線性回歸。最後的「預測類別」只是在機率 > 0.5 時判為正類。

**Q4: 學習率如何選擇？**

A: 常用方法：(1) 搜索網格 `[1e-4, 1e-3, 1e-2, 0.1]`；(2) 學習率衰減策略（指數衰減、余弦退火）；(3) 使用 Keras Tuner 自動搜索。

---

---

[來源: ch04 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #線性模型 #梯度下降 #機器學習 #ScikitLearn #邏輯回歸 #正則化 #Ridge #Lasso #程式設計 #教學 #DataScience #MachineLearning #AI

---

[來源: ch05] [標題: 支持向量機實戰指南 | 從線性分類到核技巧與迴歸 | 描述: 以 Scikit-Learn 完成 SVM 線性分類、非線性核技巧、迴歸與高維調參，涵蓋 Iris、make_moons、Wine 與 California Housing 實務案例。 | 關鍵字: Python, SVM, 支持向量機, Scikit-Learn, 分類, 迴歸, 核技巧, RBF, 線性分類, 教學]
# 支持向量機實戰：從線性分類到核技巧與迴歸

SVM（Support Vector Machine）以「最大化間隔」為核心，能在中小型資料集上取得高準確度。本文以台灣常見的風控、製造良品檢測與房價預估情境為例，示範如何透過 Scikit-Learn 打造可量產的 SVM 模型，並同時提供操作程式碼、逐行解析與實務建議。

---

[來源: ch05] # 安裝檢查與共用設定 (Traditional Chinese 註解)
import sys
from pathlib import Path
from packaging import version
import sklearn
import matplotlib.pyplot as plt

assert sys.version_info >= (3, 8), "請升級至 Python 3.8 以上"
assert version.parse(sklearn.__version__) >= version.parse("1.2.0"), "需要較新的 Scikit-Learn"

IMAGES_PATH = Path("images") / "svm"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

---

[來源: ch05] 的 Scikit-Learn"

IMAGES_PATH = Path("images") / "svm"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

plt.rc("font", size=13)
plt.rc("axes", labelsize=13, titlesize=13)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
```

---

[來源: ch05] ("axes", labelsize=13, titlesize=13)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
```

**✅ 程式碼逐行解析：**
1. 第 1-6 行：匯入標準函式庫與 Scikit-Learn，用 `assert` 避免舊版環境造成 API 不相容。
2. 第 8-9 行：確保筆記型電腦或 CI 環境具備需求版本，提前避免訓練階段錯誤。
3. 第 11-12 行：建立專屬圖像資料夾，與團隊共用輸出格式。
4. 第 14-15 行：設定 Matplotlib 字型大小，報表輸出更易閱讀。

---

[來源: ch05] CI 環境具備需求版本，提前避免訓練階段錯誤。
3. 第 11-12 行：建立專屬圖像資料夾，與團隊共用輸出格式。
4. 第 14-15 行：設定 Matplotlib 字型大小，報表輸出更易閱讀。

**🎯 重點摘要：**
- **核心功能**：提供最小可行設定與版本鎖定。
- **潛在問題**：忽略版本檢查將導致 `dual="auto"` 等新參數行為改變。
- **最佳使用情境**：專案初始化、CI 腳本或技術分享前置作業。

<h2 id="linear-svm">🧭 線性 SVM 分類策略</h2>

💡 **實際應用情境：** 以花瓣尺寸判斷 Iris 品種，或用信用卡交易金額/時段區分正常與異常。

---

[來源: ch05] ### 範例 1：`LinearSVC` + `Pipeline` 建立穩健基準線
```python

---

[來源: ch05] # 線性 SVM：以 Pipeline 確保流程可重現
import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)  # True 代表 Iris virginica

---

[來源: ch05] a[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)  # True 代表 Iris virginica

svm_clf = make_pipeline(
    StandardScaler(),  # 先做特徵縮放，避免單位差異
    LinearSVC(C=1.0, dual=True, random_state=42)
)
svm_clf.fit(X, y)

new_samples = [[5.5, 1.7], [5.0, 1.5]]
print("預測結果:", svm_clf.predict(new_samples))
print("決策分數:", svm_clf.decision_function(new_samples))
```

---

[來源: ch05] nt("預測結果:", svm_clf.predict(new_samples))
print("決策分數:", svm_clf.decision_function(new_samples))
```

**✅ 程式碼逐行解析：**
1. 第 1-6 行：載入資料集、建構標準化與線性分類器的組合。
2. 第 8-10 行：選取最具區別力的兩個特徵並建立二元標籤。
3. 第 12-16 行：`Pipeline` 保障縮放流程與模型一同部署，`C` 控制錯誤懲罰。
4. 第 18-21 行：示範推論，同步輸出類別與決策距離，利於風控閾值設定。

---

[來源: ch05] 特徵並建立二元標籤。
3. 第 12-16 行：`Pipeline` 保障縮放流程與模型一同部署，`C` 控制錯誤懲罰。
4. 第 18-21 行：示範推論，同步輸出類別與決策距離，利於風控閾值設定。

**🎯 重點摘要：**
- **核心功能**：以最少特徵建立高可讀性分類器。
- **潛在問題**：若 `dual=True` 時特徵遠大於樣本數需調整設定；未縮放會導致決策邊界偏移。
- **最佳使用情境**：特徵維度少、分類邊界接近線性的稽核、醫療初篩。

---

[來源: ch05] ### 範例 2：調整 `C` 理解軟邊界
```python

---

[來源: ch05] # 對照不同 C 值的邊界寬度
import matplotlib.pyplot as plt
C_grid = [0.5, 100]
models = []
for C in C_grid:
    model = make_pipeline(StandardScaler(), LinearSVC(C=C, dual=True, random_state=42))
    model.fit(X, y)
    models.append(model)
    print(f"C={C}，支持向量數量估計：", np.mean(np.abs(model.decision_function(X)) < 1))
```

---

[來源: ch05] els.append(model)
    print(f"C={C}，支持向量數量估計：", np.mean(np.abs(model.decision_function(X)) < 1))
```

**✅ 程式碼逐行解析：**
1. 第 1-4 行：建立不同 `C` 值以觀察正則化強度。
2. 第 5-7 行：維持相同資料流程，只調整懲罰係數確保公平比較。
3. 第 8-9 行：利用距離 < 1 近似支持向量比例，理解邊界緊迫程度。

**🎯 重點摘要：**
- **核心功能**：展示 `C` 對容錯的影響。
- **潛在問題**：`C` 過大導致過擬合且收斂較慢；過小則出現欠擬合。
- **最佳使用情境**：需要在靈敏與穩健之間快速試錯的金融風險模型。

<h2 id="kernel">🔄 核技巧與非線性分類</h2>

---

[來源: ch05] 題**：`C` 過大導致過擬合且收斂較慢；過小則出現欠擬合。
- **最佳使用情境**：需要在靈敏與穩健之間快速試錯的金融風險模型。

<h2 id="kernel">🔄 核技巧與非線性分類</h2>

💡 **實際應用情境：** 製造瑕疵圖像或 IoT 感測資料多呈環狀、曲線型可分，此時需要 Kernel Trick。

---

[來源: ch05] ### 範例 3：多項式特徵 + 線性分類器
```python

---

[來源: ch05] # 多項式特徵把曲線拉直
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
poly_linear_clf = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    LinearSVC(C=10, dual=True, random_state=42, max_iter=15_000)
)
poly_linear_clf.fit(X_moons, y_moons)
```

---

[來源: ch05] arSVC(C=10, dual=True, random_state=42, max_iter=15_000)
)
poly_linear_clf.fit(X_moons, y_moons)
```

**✅ 程式碼逐行解析：**
1. 第 1-4 行：生成「兩輪月亮」資料，模擬有明顯曲線邊界的感測場景。
2. 第 6-9 行：`PolynomialFeatures` 直接建構高維特徵，再透過線性 SVM 切分。
3. 第 10-11 行：`max_iter` 適度提高避免收斂警告。

**🎯 重點摘要：**
- **核心功能**：用特徵工程方式達成非線性分類。
- **潛在問題**：高次多項式易造成維度爆炸與共線性。
- **最佳使用情境**：特徵數少但需要處理弧形決策邏輯的品質檢查儀表。

---

[來源: ch05] # 以 kernel="poly" 精準控制高次互動關係
from sklearn.svm import SVC
poly_kernel_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)
poly_kernel_clf.fit(X_moons, y_moons)
```

**✅ 程式碼逐行解析：**
1. 第 1-5 行：`coef0` 控制高次與低次項比重，`degree=3` 足以擬合月亮資料。
2. 第 6 行：訓練後即可直接用於推論或繪製決策邊界。

---

[來源: ch05] ```

**✅ 程式碼逐行解析：**
1. 第 1-5 行：`coef0` 控制高次與低次項比重，`degree=3` 足以擬合月亮資料。
2. 第 6 行：訓練後即可直接用於推論或繪製決策邊界。

**🎯 重點摘要：**
- **核心功能**：核技巧免去手動創造特徵。
- **潛在問題**：`degree` 過高計算成本大且容易過擬合。
- **最佳使用情境**：邏輯複雜、樣本 < 1 萬筆的舊系統升級案。

<h2 id="rbf">📡 RBF 超參數調優實戰</h2>

💡 **實際應用情境：** 在布建詐欺偵測服務時，RBF 核常能兼顧準確率與彈性。

---

[來源: ch05] ### 範例 5：探索 `gamma` 與 `C`
```python

---

[來源: ch05] # 建立小型網格觀察 RBF 行為
import itertools
rbf_candidates = list(itertools.product([0.1, 1, 5], [0.1, 1, 10]))
for gamma, C in rbf_candidates:
    rbf_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=gamma, C=C))
    rbf_clf.fit(X_moons, y_moons)
    score = rbf_clf.score(X_moons, y_moons)
    print(f"gamma={gamma:<4} C={C:<4} -> 訓練準確率 {score:.3f}")
```

---

[來源: ch05] e = rbf_clf.score(X_moons, y_moons)
    print(f"gamma={gamma:<4} C={C:<4} -> 訓練準確率 {score:.3f}")
```

**✅ 程式碼逐行解析：**
1. 第 1-3 行：列舉多種 `gamma` 與 `C`，掌握感受野與懲罰的交互作用。
2. 第 4-6 行：每組參數皆建立獨立 `Pipeline`，確保縮放與模型同步。
3. 第 7-8 行：即時計算訓練準確率，做為後續交叉驗證的參考。

**🎯 重點摘要：**
- **核心功能**：快速探索超參數影響。
- **潛在問題**：只觀察訓練集會高估效能，務必再以 `GridSearchCV` 驗證。
- **最佳使用情境**：專案前期需要直覺理解 `gamma` 的工程團隊。

<h2 id="regression">💡 SVM 迴歸應用</h2>

---

[來源: ch05] ，務必再以 `GridSearchCV` 驗證。
- **最佳使用情境**：專案前期需要直覺理解 `gamma` 的工程團隊。

<h2 id="regression">💡 SVM 迴歸應用</h2>

💡 **實際應用情境：** 半導體晶圓厚度、能源即時報價等連續數值，容忍小幅誤差但需偵測異常尖峰。

---

[來源: ch05] ### 範例 6：`LinearSVR` 的 ε-不敏感區
```python

---

[來源: ch05] # 線性 SVR：偏好線性趨勢且希望忽略噪音
from sklearn.svm import LinearSVR
np.random.seed(42)
X_reg = 2 * np.random.rand(60, 1)
y_reg = 4 + 3 * X_reg[:, 0] + np.random.randn(60)

lin_svr = make_pipeline(
    StandardScaler(),
    LinearSVR(epsilon=0.5, dual=True, random_state=42)
)
lin_svr.fit(X_reg, y_reg)
print("訓練 RMSE:", root_mean_squared_error(y_reg, lin_svr.predict(X_reg)))
```

---

[來源: ch05] _svr.fit(X_reg, y_reg)
print("訓練 RMSE:", root_mean_squared_error(y_reg, lin_svr.predict(X_reg)))
```

**✅ 程式碼逐行解析：**
1. 第 1-4 行：建立線性資料並加入雜訊模擬感測器漂移。
2. 第 6-9 行：`epsilon=0.5` 代表 ±0.5 內的誤差不計入懲罰。
3. 第 10-11 行：訓練並立即計算 RMSE，快速檢查模型是否過度受噪音影響。

**🎯 重點摘要：**
- **核心功能**：提供對離群較不敏感的迴歸器。
- **潛在問題**：`epsilon` 過小會造成支持向量極多，模型緩慢。
- **最佳使用情境**：生產線趨勢監控、需求量近似線性成長的報表。

---

[來源: ch05] ### 範例 7：多項式核 SVR 擬合曲線
```python

---

[來源: ch05] # 非線性價格曲線或排程需求
from sklearn.svm import SVR
np.random.seed(42)
X_curve = 2 * np.random.rand(80, 1) - 1
y_curve = 0.2 + 0.1 * X_curve[:, 0] + 0.5 * X_curve[:, 0] ** 2 + np.random.randn(80) / 10

poly_svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
)
poly_svr.fit(X_curve, y_curve)
```

---

[來源: ch05] Scaler(),
    SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
)
poly_svr.fit(X_curve, y_curve)
```

**✅ 程式碼逐行解析：**
1. 第 1-4 行：模擬二次函數趨勢並加入白噪音。
2. 第 6-9 行：`degree=2` 恰好對應曲線階次，`C=0.01` 保持強正則化。
3. 第 10 行：訓練完成後即可用於預測或視覺化。

**🎯 重點摘要：**
- **核心功能**：以少量資料擬合平滑曲線。
- **潛在問題**：`C` 過小可能欠擬合，需針對資料噪音調整。
- **最佳使用情境**：頻寬需求、用電負載等存在平滑趨勢的序列。

<h2 id="wine">🧪 多分類與調參技巧：Wine 案例</h2>

---

[來源: ch05] 問題**：`C` 過小可能欠擬合，需針對資料噪音調整。
- **最佳使用情境**：頻寬需求、用電負載等存在平滑趨勢的序列。

<h2 id="wine">🧪 多分類與調參技巧：Wine 案例</h2>

💡 **實際應用情境：** 食品化驗或化工原料分類，需要在 3 類以上中分辨供應來源。

---

[來源: ch05] ### 範例 8：Randomized Search + `LinearSVC`
```python

---

[來源: ch05] # Wine 資料：示範多分類 OvR 與調參流程
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import loguniform

wine = load_wine(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

---

[來源: ch05] st, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

base_pipeline = make_pipeline(
    StandardScaler(),
    LinearSVC(dual=True, random_state=42, max_iter=20_000)
)

---

[來源: ch05] = make_pipeline(
    StandardScaler(),
    LinearSVC(dual=True, random_state=42, max_iter=20_000)
)

param_distrib = {"linearsvc__C": loguniform(1e-2, 1e2)}
rnd_search = RandomizedSearchCV(base_pipeline, param_distrib, n_iter=20, cv=5, random_state=42)
rnd_search.fit(X_train, y_train)
print("最佳參數:", rnd_search.best_params_)
print("測試準確率:", rnd_search.score(X_test, y_test))
```

---

[來源: ch05] train)
print("最佳參數:", rnd_search.best_params_)
print("測試準確率:", rnd_search.score(X_test, y_test))
```

**✅ 程式碼逐行解析：**
1. 第 1-6 行：載入 Wine 特徵並拆分訓練/測試集。
2. 第 8-12 行：`Pipeline` 自動完成縮放與線性多分類（OvR 內建）。
3. 第 14-17 行：使用對數均勻分布搜尋 `C`，平衡探索範圍。
4. 第 18-20 行：輸出最佳參數與測試集分數，供後續部署參考。

**🎯 重點摘要：**
- **核心功能**：示範如何在多分類下調整 `C`。
- **潛在問題**：若特徵未縮放會導致 `LinearSVC` 無法收斂。
- **最佳使用情境**：化工、食品或製藥領域的品管分類。

---

[來源: ch05] ：**
- **核心功能**：示範如何在多分類下調整 `C`。
- **潛在問題**：若特徵未縮放會導致 `LinearSVC` 無法收斂。
- **最佳使用情境**：化工、食品或製藥領域的品管分類。

<h2 id="housing">🏙️ California Housing 迴歸挑戰</h2>

💡 **實際應用情境：** 預估台灣類似房價資料時，可參考加州公開數據建模策略。

---

[來源: ch05] ### 範例 9：RBF SVR + 隨機搜尋
```python

---

[來源: ch05] # 大型資料需子抽樣加速調參
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

---

[來源: ch05] train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

svr_pipeline = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
param_distrib = {
    "svr__gamma": loguniform(1e-4, 1e-1),
    "svr__C": loguniform(1, 1e3)
}
rnd_search_reg = RandomizedSearchCV(
    svr_pipeline, param_distrib, n_iter=40, cv=3, random_state=42
)
rnd_search_reg.fit(X_train[:3000], y_train[:3000])  # 以 3k 筆加速搜尋

---

[來源: ch05] n_iter=40, cv=3, random_state=42
)
rnd_search_reg.fit(X_train[:3000], y_train[:3000])  # 以 3k 筆加速搜尋

best_model = rnd_search_reg.best_estimator_
y_pred = best_model.predict(X_test)
print("測試 RMSE (單位:10萬美金):", root_mean_squared_error(y_test, y_pred))
```

---

[來源: ch05] best_model.predict(X_test)
print("測試 RMSE (單位:10萬美金):", root_mean_squared_error(y_test, y_pred))
```

**✅ 程式碼逐行解析：**
1. 第 1-6 行：載入加州房價並拆分資料，目標值為 10 萬美元為單位。
2. 第 8-13 行：建立 RBF SVR `Pipeline` 並設定對數搜尋空間。
3. 第 14-18 行：僅抽樣 3000 筆加速超參數搜尋，再以全資料重新訓練。
4. 第 20-21 行：評估 RMSE 以千美元為單位，便於與實際預算對照。

---

[來源: ch05] peline` 並設定對數搜尋空間。
3. 第 14-18 行：僅抽樣 3000 筆加速超參數搜尋，再以全資料重新訓練。
4. 第 20-21 行：評估 RMSE 以千美元為單位，便於與實際預算對照。

**🎯 重點摘要：**
- **核心功能**：示範大資料下的調參策略。
- **潛在問題**：SVR 在 2 萬筆以上資料會相當緩慢，可考慮 `LinearSVR` 或樹模型。
- **最佳使用情境**：需要更平滑預測且樣本仍可抽樣的房價或能源需求。

<h2 id="faq">❓ 常見問答 (FAQ)</h2>

---

[來源: ch05] 以上資料會相當緩慢，可考慮 `LinearSVR` 或樹模型。
- **最佳使用情境**：需要更平滑預測且樣本仍可抽樣的房價或能源需求。

<h2 id="faq">❓ 常見問答 (FAQ)</h2>

1. **何時選擇 `LinearSVC`、`SVC` 或 `SGDClassifier`？**  
   - 大量特徵/樣本：`LinearSVC` 或 `SGDClassifier`。  
   - 需要 kernel：`SVC`。  
   - 線上學習：`SGDClassifier` 支援 partial fit。
2. **如何取得概率？**  
   `SVC(probability=True)` 會在訓練結束後以交叉驗證擬合 Platt scaling，成本較高但可得到 `predict_proba()`。
3. **為什麼模型不收斂？**  
   通常是未縮放、`C` 過

---

[來源: ch05] y=True)` 會在訓練結束後以交叉驗證擬合 Platt scaling，成本較高但可得到 `predict_proba()`。
3. **為什麼模型不收斂？**  
   通常是未縮放、`C` 過大或 `max_iter` 過低。先確認 `StandardScaler` 已放入 `Pipeline`。
4. **`gamma` 調整方向？**  
   欠擬合 → 增加 `gamma`。過擬合 → 降低 `gamma` 或 `C`。
5. **SVM 迴歸 `epsilon` 如何設定？**  
   以商業可接受誤差為基準，例如價格可容忍 ±5 萬，就設定 `epsilon=0.5` (單位 10 萬)。

<h2 id="best-practices">💼 總結與最佳實踐</h2>

---

[來源: ch05] *  
   以商業可接受誤差為基準，例如價格可容忍 ±5 萬，就設定 `epsilon=0.5` (單位 10 萬)。

<h2 id="best-practices">💼 總結與最佳實踐</h2>

- **流程化**：所有 SVM 範例均包於 `Pipeline`，避免資料洩漏並便於部署。
- **縮放優先**：無論線性或核 SVM，都先做標準化；對稀疏向量可改用 `MaxAbsScaler`。
- **交叉驗證**：小型資料建議 `StratifiedKFold`，大資料則使用 `ShuffleSplit` 加速。
- **解釋性**：線性 SVM 可檢視 `coef_` 排序，協助法遵或商務單位理解判斷依據。
- **效能取捨**：當資料量 > 5 萬筆時，優先評估線性方法或樹模型，以避免核 SVM 訓練時間過長。

---

[來源: ch05] **解釋性**：線性 SVM 可檢視 `coef_` 排序，協助法遵或商務單位理解判斷依據。
- **效能取捨**：當資料量 > 5 萬筆時，優先評估線性方法或樹模型，以避免核 SVM 訓練時間過長。

<h2 id="hashtags">🏷️ 推薦標籤 (Suggested Hashtags)</h2>

#Python #SVM #支持向量機 #機器學習 #ScikitLearn #資料科學 #程式設計 #技術教學 #ThreadsTech #開發者社群

---

[來源: ch05 | 類型: cheatsheet] # Ch05 速查表：Support Vector Machines (SVM)

> **核心主旨**：SVM 最大化類別間距（margin）—— 特徵縮放是成敗關鍵，kernel trick 讓線性模型具備非線性能力。

---

---

[來源: ch05 | 類型: cheatsheet] ## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Hard Margin | 所有樣本必須在正確側，不容錯 | 線性可分且無 outlier |
| Soft Margin (`C`) | 允許部分違規，`C` 越大 margin 越小 | 有雜訊的實務資料 |
| Kernel Trick | 隱式映射到高維空間，不需顯式計算 | 非線性分類 |
| RBF Kernel | $K(x,z) = e^{-\gamma\|x-z\|^2}$，`gamma` 控制局部化 | 最常用的非線性 kernel |
| Polynomial Kernel | $(x \cdot z + r)^d$，`degree` 控制多項式次數 | 低次多項式邊界 |
| SVR (epsilon-insensitive) | 在 $\epsilon$ 管道內的誤差不計入損失 | 迴歸，允許一定誤差容忍 |
| Support Vectors | 距離 margin 最近的訓練樣本，決定超平面 | 理解模型複雜度 |


---

[來源: ch05 | 類型: cheatsheet] | 在 $\epsilon$ 管道內的誤差不計入損失 | 迴歸，允許一定誤差容忍 |
| Support Vectors | 距離 margin 最近的訓練樣本，決定超平面 | 理解模型複雜度 |


---

---

[來源: ch05 | 類型: cheatsheet] ## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `LinearSVC` | `C=1.0`, `max_iter=1000` | 線性 SVM（大資料集快） |
| `SVC` | `kernel="rbf"/"poly"/"linear"`, `C=1.0`, `gamma="scale"` | 非線性 SVM |
| `SVR` | `kernel="rbf"`, `C=1.0`, `epsilon=0.1` | SVM 迴歸 |
| `StandardScaler` | – | **必備**！SVM 對特徵尺度極敏感 |
| `make_pipeline` | – | 快速建立 Scaler + SVM Pipeline |
| `GridSearchCV` | `param_grid={"svc__C": [...], "svc__gamma": [...]}` | 調優 C 與 gamma |

---

---

[來源: ch05 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV

---

[來源: ch05 | 類型: cheatsheet] # 線性 SVM（最快，適合大資料）
linear_svm = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=2000, random_state=42))
linear_svm.fit(X_train, y_train)

---

[來源: ch05 | 類型: cheatsheet] # RBF kernel SVM（最常用的非線性分類）
svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma=0.05))
svm_clf.fit(X_train, y_train)

---

[來源: ch05 | 類型: cheatsheet] # Polynomial kernel
poly_svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)
poly_svm.fit(X_train, y_train)

---

[來源: ch05 | 類型: cheatsheet] # SVR 迴歸
svr_reg = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1))
svr_reg.fit(X_train, y_train)

---

[來源: ch05 | 類型: cheatsheet] # 超參數調優（C 與 gamma）
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1, "scale"]
}
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

---

[來源: ch05 | 類型: cheatsheet] # 取得 support vectors 數量
svc = SVC(kernel="rbf", C=5, gamma=0.05)
svc.fit(X_train_scaled, y_train)
print(f"Support vectors: {svc.n_support_}")  # 每個類別的 support vector 數
```

---

---

[來源: ch05 | 類型: cheatsheet] ## 4. 常見陷阱

- **忘記 Scaling**：SVM 對特徵尺度極度敏感，**永遠要先 StandardScaler**。
- **`LinearSVC` vs `SVC(kernel="linear")`**：前者用最佳化演算法（較快），後者用 kernel trick（較慢但支援更多選項）。
- **`C` 的直覺**：`C` 小 → 大 margin 但更多違規（高 bias，低 variance）；`C` 大 → 小 margin 但少違規（低 bias，高 variance）。
- **`gamma` 的直覺**：`gamma` 大 → 每個樣本影響範圍小（過擬合傾向）；`gamma` 小 → 影響範圍大（欠擬合傾向）。
- **`SVC` 不支援 `predict_proba` 預設**：需加 `probability=True`（訓練時間增加）。

---

---

[來源: ch05 | 類型: cheatsheet] ## 5. 決策指南

```
選哪種 SVM？
├── 線性可分或特徵數很多（NLP） → LinearSVC（快）
├── 非線性、中小資料集 (< 50k) → SVC(kernel="rbf")
└── 需要迴歸                    → SVR

C 與 gamma 調優範圍建議：
├── C: [0.01, 0.1, 1, 10, 100]（log scale）
└── gamma: [0.001, 0.01, 0.1, 1] 或 "scale"（推薦起點）

SVM vs 其他演算法：
├── 中小資料集、明確間隔 → SVM 可能優於 GBM
├── 大資料集 (> 100k)    → GBM 或深度學習更快
└── 需要可解釋性          → 決策樹或線性模型
```

---

[來源: ch05 | 類型: handout] # 課程講義：支持向量機 (Chapter 05)

支持向量機（SVM）以一個優雅的幾何直覺為核心：**找到使兩類別「邊界（Margin）最寬」的決策邊界**。這種「最大邊界」的設計讓 SVM 對未見資料有優異的泛化能力。本章從線性可分情形出發，引入「軟邊界」處理雜訊，再透過**核技巧 (Kernel Trick)** 將 SVM 擴展到非線性問題——無需顯式計算高維特徵，卻能等效在高維空間分類。

---

---

[來源: ch05 | 類型: handout] ### 理論背景

**硬邊界 (Hard Margin)**：要求所有訓練樣本都嚴格在邊界外側。

$$\min_{\mathbf{w}, b} \frac{1}{2} \

|\mathbf{w}\|^2 \quad \text{s.t.} \quad y^{(i)} \left(\mathbf{w}^T \mathbf{x}^{(i)} + b\right) \geq 1, \; \forall i$$


邊界寬度（Margin）= $\frac{2}{\

|\mathbf{w}\|}$，最大化 Margin 等效於最小化 $\|\mathbf{w}\|^2$。


**問題**：硬邊界對 outlier 極度敏感，且線性不可分時無解。

**軟邊界 (Soft Margin)**：允許部分樣本違反邊界，引入鬆弛變數 $\zeta_i \geq 0$：

---

[來源: ch05 | 類型: handout] 2$。


**問題**：硬邊界對 outlier 極度敏感，且線性不可分時無解。

**軟邊界 (Soft Margin)**：允許部分樣本違反邊界，引入鬆弛變數 $\zeta_i \geq 0$：

$$\min_{\mathbf{w}, b, \boldsymbol{\zeta}} \frac{1}{2} \

|\mathbf{w}\|^2 + C \sum_{i=1}^{m} \zeta_i$$


$$\text{s.t.} \quad y^{(i)} \left(\mathbf{w}^T \mathbf{x}^{(i)} + b\right) \geq 1 - \zeta_i, \quad \zeta_i \geq 0$$

- $C$ 小 → 邊界寬，允許更多違規（正則化強，防過擬合）
- $C$ 大 → 邊界窄，嚴格分類（可能過擬合）

---

[來源: ch05 | 類型: handout] \geq 1 - \zeta_i, \quad \zeta_i \geq 0$$

- $C$ 小 → 邊界寬，允許更多違規（正則化強，防過擬合）
- $C$ 大 → 邊界窄，嚴格分類（可能過擬合）

**重要：SVM 對特徵尺度非常敏感，務必先做 `StandardScaler`！**

---

[來源: ch05 | 類型: handout] ### 核心代碼

```python
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)  # Iris virginica vs rest

---

[來源: ch05 | 類型: handout] # 軟邊界線性 SVM（標準化必不可少）
svm_clf = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1.0, dual=True, random_state=42)
)
svm_clf.fit(X, y)
print(svm_clf.predict([[5.5, 1.7], [5.0, 1.5]]))
print(svm_clf.decision_function([[5.5, 1.7]]))  # 到決策邊界的距離
```

---

[來源: ch05 | 類型: handout] ### 補充練習 1

**理論題：** 為什麼 SVM 需要特徵縮放？舉一個具體例子說明，若某特徵的值域是 [0, 1000] 而另一個是 [0, 1]，未縮放的 SVM 決策邊界會偏向哪個方向？

**實作題：** 使用 `make_moons(n_samples=200, noise=0.1)` 資料，分別用 `C=0.001`、`C=1`、`C=1000` 訓練 `LinearSVC`，繪製決策邊界，觀察 $C$ 對邊界形狀和支持向量數量的影響。

---

---

[來源: ch05 | 類型: handout] ### 理論背景

對線性不可分的資料，可以先映射到高維空間，再用線性 SVM 分類：

$$\phi: \mathbb{R}^n \to \mathbb{R}^d \quad (d \gg n)$$

**核技巧 (Kernel Trick)**：核函數 $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^T \phi(\mathbf{z})$ 直接計算高維內積，無需顯式映射：

---

[來源: ch05 | 類型: handout] rnel Trick)**：核函數 $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^T \phi(\mathbf{z})$ 直接計算高維內積，無需顯式映射：

| 核函數 | 數學公式 | 適用場景 |
|--------|---------|---------|
| 線性核 | $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$ | 線性可分 |
| 多項式核 | $K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d$ | 曲線邊界 |
| RBF (Gaussian) | $K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right)$ | 通用，最常用 |
| Sigmoid | $K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$ | 類神經網路 |


---

[來源: ch05 | 類型: handout] 最常用 |
| Sigmoid | $K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$ | 類神經網路 |


**RBF 核超參數**：

- $\gamma$ 大 → 每個樣本的「影響範圍」小，模型複雜（可能過擬合）
- $\gamma$ 小 → 影響範圍大，模型平滑（可能欠擬合）
- $C$ 和 $\gamma$ 通常需要搭配 `GridSearchCV` 調整

---

[來源: ch05 | 類型: handout] ### 核心代碼

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC

X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)

---

[來源: ch05 | 類型: handout] # 多項式核 SVM
poly_kernel_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)
poly_kernel_clf.fit(X_moons, y_moons)

---

[來源: ch05 | 類型: handout] # RBF 核 SVM（最常用，先試這個）
rbf_kernel_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", gamma=0.1, C=1.0)
)
rbf_kernel_clf.fit(X_moons, y_moons)

---

[來源: ch05 | 類型: handout] # 超參數搜索
from sklearn.model_selection import GridSearchCV
param_grid = {
    "svc__gamma": [0.01, 0.1, 1.0, 10.0],
    "svc__C":     [0.1, 1.0, 10.0, 100.0]
}
svc_grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                        param_grid, cv=5, scoring="accuracy")
svc_grid.fit(X_moons, y_moons)
print(f"Best params: {svc_grid.best_params_}")
print(f"Best score: {svc_grid.best_score_:.4f}")
```

---

[來源: ch05 | 類型: handout] ### 補充練習 2

**理論題：** 解釋 RBF 核的物理直覺：為何 $\gamma$ 大的 RBF 核決策邊界看起來像是圍繞每個訓練樣本畫圈？

**實作題：** 用 `GridSearchCV` 在 `make_moons` 資料上搜索 `gamma` ∈ [0.01, 0.1, 1, 10] 與 `C` ∈ [0.1, 1, 10, 100] 的最佳組合，繪製 AUC 熱力圖（x 軸 gamma，y 軸 C）。

---

---

[來源: ch05 | 類型: handout] ### 理論背景

SVM 可以翻轉用於迴歸：**最大化「包含最多訓練樣本的管道（$\epsilon$-tube）」**，而非最大化邊界。

**$\varepsilon$-不敏感損失函數 (SVR 的損失)**：

$$\ell_\varepsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \varepsilon)$$

即在 $\varepsilon$ 範圍內的誤差視為 0（容許小誤差），超出才計入損失。

- $\varepsilon$ 大 → 管道寬，允許更多誤差
- $\varepsilon$ 小 → 管道窄，模型更精確但可能 overfit

**超參數** `C` 和 `epsilon` 需要搭配交叉驗證調整。

---

[來源: ch05 | 類型: handout] ### 核心代碼

```python
from sklearn.svm import SVR, LinearSVR
import numpy as np

np.random.seed(42)
X_reg = 2 * np.random.rand(50, 1)
y_reg = (4 + 3 * X_reg + np.random.randn(50, 1)).ravel()

---

[來源: ch05 | 類型: handout] # 線性 SVR（速度快）
lin_svr = make_pipeline(
    StandardScaler(),
    LinearSVR(epsilon=0.5, dual=True, random_state=42)
)
lin_svr.fit(X_reg, y_reg)

---

[來源: ch05 | 類型: handout] # RBF 核 SVR（非線性迴歸）
rbf_svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", gamma=0.1, C=1.0, epsilon=0.1)
)
rbf_svr.fit(X_reg, y_reg)

---

[來源: ch05 | 類型: handout] # 評估
from sklearn.metrics import mean_squared_error
print(f"SVR RMSE: {mean_squared_error(y_reg, rbf_svr.predict(X_reg), squared=False):.3f}")
```

---

[來源: ch05 | 類型: handout] ### 補充練習 3

**理論題：** 比較 SVR 的 $\varepsilon$-不敏感損失與線性迴歸的 MSE 損失：哪個對 outlier 更魯棒？為什麼？

**實作題：** 生成一個含有 5% outlier 的資料集（正常資料符合 $y = 3x + 2$，outlier 的 $y$ 值隨機在 [-20, 20]），比較 `LinearRegression`、`SVR(kernel="linear", epsilon=0.5)` 和 `HuberRegressor` 的擬合效果。

---

---

[來源: ch05 | 類型: handout] ### 理論背景

SVM 的訓練可透過求解**對偶問題 (Dual Problem)** 來完成，這讓核技巧成為可能：

原始問題（Primal）的 Lagrangian 為：

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\

|\mathbf{w}\|^2 - \sum_{i=1}^m \alpha_i \left[y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) - 1\right]$$


對偶問題（Dual）：

---

[來源: ch05 | 類型: handout] |^2 - \sum_{i=1}^m \alpha_i \left[y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) - 1\right]$$


對偶問題（Dual）：

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} \mathbf{x}^{(i)T} \mathbf{x}^{(j)}$$

$$\text{s.t.} \quad \alpha_i \geq 0, \; \sum_{i=1}^m \alpha_i y^{(i)} = 0$$

關鍵觀察：只有支持向量（$\alpha_i > 0$）對預測有貢獻。決策函數：

---

[來源: ch05 | 類型: handout] uad \alpha_i \geq 0, \; \sum_{i=1}^m \alpha_i y^{(i)} = 0$$

關鍵觀察：只有支持向量（$\alpha_i > 0$）對預測有貢獻。決策函數：

$$\hat{y} = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i y^{(i)} K(\mathbf{x}^{(i)}, \mathbf{x}) + b\right)$$

---

[來源: ch05 | 類型: handout] # 核 SVM 的支持向量查詢
svc = SVC(kernel="rbf", C=1.0, gamma="scale")
svc.fit(X_moons, y_moons)

---

[來源: ch05 | 類型: handout] # 查看支持向量
print(f"支持向量數量: {len(svc.support_vectors_)}")
print(f"支持向量 shape: {svc.support_vectors_.shape}")
print(f"對偶係數 (alpha) shape: {svc.dual_coef_.shape}")
```

---

[來源: ch05 | 類型: handout] ### 補充練習 4

**理論題：** KKT 條件告訴我們，只有 $\alpha_i > 0$ 的樣本（支持向量）對決策函數有貢獻。當 $C$ 很大時，支持向量的數量通常會增加還是減少？為什麼？

---

---

[來源: ch05 | 類型: handout] ## 結論

SVM 的三大核心思想：

1. **最大化邊界**：比最小化訓練誤差更強調泛化能力
2. **軟邊界（超參數 $C$）**：在邊界寬度和訓練誤差之間取捨
3. **核技巧**：以低計算成本實現高維特徵映射，無需顯式擴展特徵

SVM 在**中小型資料集、高維特徵、非線性邊界**的場景下特別強大。對大型資料集，`LinearSVC` 或 `SGDClassifier` 更有效率。

下一章（Ch06）介紹決策樹，一種完全不同的學習方式。

---

---

[來源: ch05 | 類型: handout] ## 課後作業

**作業：Wine 資料集 SVM 分類**

使用 `sklearn.datasets.load_wine()` 資料集（3 類，13 個特徵）：

1. 用 `Pipeline` 搭配 `StandardScaler` + `SVC(kernel="rbf")`，用 `GridSearchCV` 找到最佳 `C` 和 `gamma`（各取 5 個值）。
2. 計算最佳模型在測試集的準確率、F1 分數（macro），並繪製混淆矩陣。
3. **思考題**：若將 `SVC` 改為 `LinearSVC`（線性核），準確率是否下降？這說明了 Wine 資料的邊界是線性還是非線性的？

---

[來源: ch05 | 類型: tutorial] [標題: 支持向量機完整指南：從線性分類到核技巧與 SVR 實戰 | 描述: 深入理解 SVM 的最大間隔分類、軟間隔 C 參數、核技巧（RBF/多項式）、SVM 回歸 (SVR)。含 Scikit-Learn 實戰程式碼、逐行解析，以台灣製造業與金融情境為例。 | 關鍵字: Python, SVM, 支持向量機, 核技巧, RBF, Scikit-Learn, SVR, 分類, 迴歸, 機器學習]
# 支持向量機完整指南：從最大間隔到核技巧

支持向量機（Support Vector Machine，SVM）以「最大化分類間隔」為核心，在中小型資料集上效能出色，是二元分類的強力工具。本教學帶你從線性 SVM 開始，逐步深入核技巧（Kernel Trick），讓線性模型具備處理非線性問題的能力。

---

[來源: ch05 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **特徵縮放**是 SVM 成敗關鍵——必須在 Pipeline 中使用 `StandardScaler`
- **C 參數**控制軟間隔寬度：C 小→更寬間隔（更正則化）；C 大→更緊間隔（更擬合）
- **核技巧（Kernel Trick）** 讓線性 SVM 隱式在高維空間分類，計算效率高
- **RBF 核的 gamma**：高 gamma → 決策邊界複雜（過擬合風險）；低 gamma → 平滑邊界
- 大資料集（> 10 萬）改用 `LinearSVC` 或 `SGDClassifier`，訓練速度更快

---

---

[來源: ch05 | 類型: tutorial] ## 線性 SVM 分類

💡 **實際應用情境：** 在 PCB 板良品/不良品檢測中，若兩類的特徵（如缺陷面積、灰度均值）線性可分，SVM 能找到「最大間隔」的決策邊界，對邊界附近的樣本分類更穩健。

SVM 的核心思想：找到**最寬的分類間隔（Margin）**。決策函數：

$$\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$$

**支持向量（Support Vectors）**：位於間隔邊界上的訓練樣本，決定決策邊界的位置。

---

[來源: ch05 | 類型: tutorial] ### 範例 1: Iris 資料集線性 SVM

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

---

[來源: ch05 | 類型: tutorial] # 載入 Iris 資料（只取後兩類：Versicolor vs Virginica）
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values[50:]  # 後 100 筆
y = (iris.target[50:] == 2).astype(int)  # 1 = Virginica, 0 = Versicolor

---

[來源: ch05 | 類型: tutorial] # SVM 對特徵尺度敏感，必須標準化！
svm_clf = Pipeline([
    ("scaler", StandardScaler()),   # 特徵縮放（必要！）
    ("linear_svc", LinearSVC(C=1.0, max_iter=2000, random_state=42))
])
svm_clf.fit(X, y)
print(f"預測結果: {svm_clf.predict([[5.5, 1.7], [4.0, 1.0]])}")
```

**✅ 程式碼逐行解析：**

1. `StandardScaler()`: SVM 對特徵尺度極敏感——若特徵 1 的範圍是 [0, 1000]，特徵 2 是 [0, 1]，SVM 會嚴重偏向特徵 1
2. `LinearSVC(C=1.0)`: C=1 是預設軟間隔強度，增大 C 讓 SVM 更努力分對訓練樣本（但可能過擬合）
3. `max_iter=2000`: LinearSVC 使用迭代優化，預設可能不夠收斂，適當增加

**🎯 重點摘要:**

- **Pipeline 的重要性**: Scaler 必須在 Pipeline 內，確保測試集只 `transform`
- **C 參數直覺**: 想像 C 是違規停車的罰款——C 高，每個誤分類代價高，SVM 努力避免（間隔變窄）

---

---

[來源: ch05 | 類型: tutorial] ## 軟間隔 SVM (Soft Margin SVM)

💡 **實際應用情境：** 真實資料幾乎不可能完美線性可分（雜訊、異常值很常見）。軟間隔允許部分樣本落在間隔內或甚至越過邊界，透過 C 參數控制這種「容忍度」。

硬間隔 vs 軟間隔：

- **硬間隔（Hard Margin）**：要求所有樣本都在正確側，對離群值不容忍
- **軟間隔（Soft Margin）**：允許一些「鬆弛（Slack）」，更泛化

---

[來源: ch05 | 類型: tutorial] ### 範例 2: C 參數對決策邊界的影響

```python
from sklearn.svm import SVC
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, C in zip(axes, [0.01, 100]):
    svm_c = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", C=C))  # 線性核 + 不同 C
    ])
    svm_c.fit(X, y)
    ax.set_title(f"C = {C}")
    # （此處省略決策邊界繪製代碼）

plt.show()

---

[來源: ch05 | 類型: tutorial] # C=0.01: 更寬間隔，允許更多誤分類（高偏差/欠擬合）

---

[來源: ch05 | 類型: tutorial] # C=100:  更窄間隔，努力分對所有點（高方差/過擬合）
```

**🎯 重點摘要:**

- **C 小（強正則化）**：更寬間隔，更多樣本可能在間隔內 → 更泛化
- **C 大（弱正則化）**：更窄間隔，努力分對訓練集 → 可能過擬合
- 用 `GridSearchCV` 或 `RandomizedSearchCV` 搜尋最佳 C

---

---

[來源: ch05 | 類型: tutorial] ## 核技巧與非線性分類

💡 **實際應用情境：** 如圓形邊界的問題（如用半徑和角度分類衛星軌道），在原始空間線性不可分，但映射到高維空間後可能線性可分。核技巧讓我們**不需要顯式計算高維映射**。

核函數類型：

| 核函數 | 特點 | 超參數 |
|--------|------|--------|
| 線性核（linear） | 等同 LinearSVC | 無 |
| 多項式核（poly） | 處理多項式邊界 | degree, coef0 |
| RBF（高斯）核 | 最通用，處理任意邊界 | gamma |
| Sigmoid 核 | 類似神經網路 | gamma, coef0 |

---

[來源: ch05 | 類型: tutorial] ### 範例 3: RBF 核處理非線性問題

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

---

[來源: ch05 | 類型: tutorial] # 月牙形資料（線性不可分）
X_moons, y_moons = make_moons(n_samples=100, noise=0.15, random_state=42)

---

[來源: ch05 | 類型: tutorial] # RBF 核 SVM
rbf_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", gamma=5, C=0.001)  # gamma 大 → 決策邊界更複雜
)
rbf_kernel_svm_clf.fit(X_moons, y_moons)
print(f"RBF SVM 訓練準確率: {rbf_kernel_svm_clf.score(X_moons, y_moons):.4f}")
```

**✅ 程式碼逐行解析：**

1. `make_moons(noise=0.15)`: 生成帶雜訊的月牙形資料，是測試非線性分類器的標準資料集
2. `SVC(kernel="rbf", gamma=5)`: RBF 核的 gamma 控制「影響半徑」——gamma 越大，每個支持向量的影響範圍越小，邊界越複雜
3. `C=0.001`: 搭配大 gamma，使用小 C 平衡複雜度

**🎯 重點摘要:**

---

[來源: ch05 | 類型: tutorial] RBF 核的 gamma 控制「影響半徑」——gamma 越大，每個支持向量的影響範圍越小，邊界越複雜
3. `C=0.001`: 搭配大 gamma，使用小 C 平衡複雜度

**🎯 重點摘要:**

- **核技巧（Kernel Trick）** 計算技巧：$K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \phi(\mathbf{x}^{(i)})^T \phi(\mathbf{x}^{(j)})$，不需顯式計算映射
- RBF 核是預設首選，對大多數問題效果良好

---

---

[來源: ch05 | 類型: tutorial] ### 範例 4: GridSearchCV 搜尋最佳 C 和 gamma

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {"svc__kernel": ["rbf"],
     "svc__gamma": [0.01, 0.1, 1, 10, 100],
     "svc__C":     [0.001, 0.01, 0.1, 1, 10, 100]}
]

svm_pipeline = make_pipeline(StandardScaler(), SVC(random_state=42))
grid_search = GridSearchCV(
    svm_pipeline, param_grid,
    cv=5, scoring="accuracy",
    n_jobs=-1, verbose=1
)
grid_search.fit(X_moons, y_moons)

print(f"最佳參數: {grid_search.best_params_}")
print(f"最佳準確率: {grid_search.best_score_:.4f}")
```

**🎯 重點摘要:**

---

[來源: ch05 | 類型: tutorial] f"最佳參數: {grid_search.best_params_}")
print(f"最佳準確率: {grid_search.best_score_:.4f}")
```

**🎯 重點摘要:**

- gamma 和 C 通常在對數尺度上搜尋（`np.logspace(-3, 3, 7)` 或 `[0.001, 0.01, ..., 1000]`）
- 若計算資源有限，改用 `RandomizedSearchCV` 加快搜尋

---

---

[來源: ch05 | 類型: tutorial] ## SVM 回歸 (SVR)

💡 **實際應用情境：** SVR 用於預測連續值，如廠房能耗預測。它建立一個寬度為 $2\varepsilon$ 的「容忍區」，在區內的預測誤差不懲罰。

---

[來源: ch05 | 類型: tutorial] ### 範例 5: SVR 房價預測

```python
from sklearn.svm import SVR
import numpy as np

np.random.seed(42)
X_svr = 2 * np.random.rand(50, 1)
y_svr = (4 + 3 * X_svr + np.random.randn(50, 1)).ravel()

---

[來源: ch05 | 類型: tutorial] # SVR：在 epsilon 容忍區內的誤差不懲罰
svm_poly_reg = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)  # epsilon 容忍帶
)
svm_poly_reg.fit(X_svr, y_svr)

---

[來源: ch05 | 類型: tutorial] # 評估
from sklearn.metrics import mean_squared_error
y_pred_svr = svm_poly_reg.predict(X_svr)
rmse = np.sqrt(mean_squared_error(y_svr, y_pred_svr))
print(f"SVR 訓練集 RMSE: {rmse:.3f}")
```

**✅ 程式碼逐行解析：**

1. `SVR(epsilon=0.1)`: 誤差在 0.1 以內不計入損失（容忍帶寬度 = 2 × epsilon）
2. `C=100`: 對超出容忍帶的誤差施加高懲罰（等同分類中的 C）

**🎯 重點摘要:**

- SVR = 找到一條讓**最多點落在容忍帶內**的回歸線
- `epsilon` 越大，容忍帶越寬，支持向量越少，模型越平滑

---

---

[來源: ch05 | 類型: tutorial] ## LinearSVC vs SVC vs SGDClassifier

```python

---

[來源: ch05 | 類型: tutorial] # 三種 SVM 實作的比較
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier

---

[來源: ch05 | 類型: tutorial] # LinearSVC：用優化演算法直接求解，O(m × n)，大資料集快
linear_svc = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC(C=1, max_iter=2000))])

---

[來源: ch05 | 類型: tutorial] # SVC(kernel="linear")：用核演算法，O(m² × n) 到 O(m³ × n)，小資料集準確
svc_linear = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear", C=1))])

---

[來源: ch05 | 類型: tutorial] # SGDClassifier(loss="hinge")：等效線性 SVM，支援線上學習
sgd_hinge = Pipeline([("scaler", StandardScaler()),
                      ("sgd", SGDClassifier(loss="hinge", alpha=1/(len(X)*1.0)))])
```

| 方法 | 複雜度 | 特點 | 適用場景 |
|------|--------|------|---------|
| `LinearSVC` | O(m × n) | 快速線性 SVM | 大型資料集、線性問題 |
| `SVC` | O(m² ~ m³ × n) | 支援各種核 | 小型非線性問題 |
| `SGDClassifier(loss="hinge")` | O(m) per epoch | 支援線上學習 | 超大資料集、串流資料 |

**🎯 重點摘要:**

- 預設先試 `LinearSVC`（速度快）
- 若線性不夠，改用 `SVC(kernel="rbf")`
- 資料量 > 100 萬，用 `SGDClassifier`

---

---

[來源: ch05 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: 為什麼 SVM 對特徵縮放那麼敏感？**

A: SVM 最大化分類間隔，間隔的計算依賴各特徵的「距離」。若特徵 A 的範圍是 [0, 1000] 而特徵 B 是 [0, 1]，SVM 會認為特徵 A 更重要，導致偏頗的決策邊界。

**Q2: 核技巧為什麼能提高計算效率？**

A: 若要將 n 維特徵映射到 p 維空間，顯式計算需要 O(p) 空間。但核函數 $K(a, b) = \phi(a)^T \phi(b)$ 只需計算一個純量（不需要顯式計算 $\phi$），對無限維映射（如 RBF 的隱式映射）特別有效。

**Q3: C 和 gamma 同時調，方向有何規律？**

A: (1) 先用小 gamma（平滑邊界），觀察 C 的影響；(2) 找到合適的 C 後，再調整 gamma；(3) 或直接用 `GridSearchCV` 二維搜尋。高 C + 高 gamma 通常嚴重過擬合。

**Q4: SVR 和 Ridge 回歸有什麼差別？**

A: Ridge 懲罰所有誤差；SVR 對 epsilon 容忍帶內的誤差不懲罰（只懲罰大誤差），這使 SVR 對異常值更健壯。

---

---

[來源: ch05 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #SVM #支持向量機 #核技巧 #機器學習 #ScikitLearn #SVR #RBF #GridSearch #程式設計 #教學 #DataScience #MachineLearning #AI #特徵工程

---

[來源: ch06] [標題: 決策樹在機器學習中的應用：完整 Python 教學 | 描述: 學習如何使用 Scikit-Learn 建構決策樹模型，包括訓練、可視化、預測和正則化技巧。包含實用範例和逐步解析，適合初學者到進階開發者。 | 關鍵字: Python, 機器學習, 決策樹, Scikit-Learn, 教學, 程式設計, 資料科學]
# 決策樹在機器學習中的應用：完整 Python 教學

學習決策樹（Decision Trees）是機器學習中的基礎演算法之一。本教學將帶您從基礎概念開始，逐步深入到實際應用，使用 Scikit-Learn 建構決策樹模型，並探討其在分類和回歸問題中的應用。

---

[來源: ch06] ## 關鍵重點 (Key Takeaways)

- 決策樹是一種強大的監督學習演算法，能處理分類和回歸問題
- Scikit-Learn 的 `DecisionTreeClassifier` 和 `DecisionTreeRegressor` 提供簡單易用的 API
- 正則化技巧如限制樹深度和葉節點數量有助於避免過擬合
- 決策樹對軸向旋轉敏感，可使用 PCA 預處理改善
- 決策樹具有高變異性，容易過擬合訓練資料

---

[來源: ch06] ## 設定與環境準備

在開始之前，我們需要確保 Python 環境正確設定，並匯入必要的程式庫。

---

[來源: ch06] ### 範例 1: 環境檢查與設定

```python
import sys

---

[來源: ch06] # 檢查 Python 版本是否為 3.7 或以上
assert sys.version_info >= (3, 7)

---

[來源: ch06] # 匯入並檢查 Scikit-Learn 版本
from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

---

[來源: ch06] # 設定 Matplotlib 字體大小
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

---

[來源: ch06] # 建立圖片儲存目錄
from pathlib import Path

IMAGES_PATH = Path() / "images" / "decision_trees"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

---

[來源: ch06] # 定義圖片儲存函數
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

---

[來源: ch06] plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 匯入 sys 模組，用於檢查 Python 版本
2. `assert sys.version_info >= (3, 7)`: 確保 Python 版本至少為 3.7
3. `from packaging import version`: 匯入版本比較工具
4. `assert version.parse(sklearn.__version__) >= version.parse("1.0.1")`: 檢查 Scikit-Learn 版本
5. `plt.rc(...)`: 設定 Matplotlib 的預設字體和大小
6. `IMAGES_PATH.mkdir(...)`: 建立圖片儲存目錄
7. `def save_fig(...)`: 定義儲存圖片的輔助函數

---

[來源: ch06] )`: 設定 Matplotlib 的預設字體和大小
6. `IMAGES_PATH.mkdir(...)`: 建立圖片儲存目錄
7. `def save_fig(...)`: 定義儲存圖片的輔助函數

**🎯 重點摘要:**

- **核心功能**: 環境設定和圖片儲存工具
- **潛在問題**: 版本檢查失敗會引發 AssertionError
- **最佳使用情境**: 在機器學習專案開始時進行環境驗證

---

[來源: ch06] ## 訓練與視覺化決策樹

讓我們使用鳶尾花（Iris）資料集來訓練第一個決策樹模型。

---

[來源: ch06] ### 範例 2: 載入資料並訓練決策樹

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

---

[來源: ch06] # 載入鳶尾花資料集
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

---

[來源: ch06] # 建立並訓練決策樹分類器
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)
```

**✅ 程式碼逐行解析：**

---

[來源: ch06] ecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import load_iris`: 匯入鳶尾花資料集
2. `iris = load_iris(as_frame=True)`: 載入資料並轉為 DataFrame 格式
3. `X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values`: 選取花瓣長度和寬度作為特徵
4. `y_iris = iris.target`: 取得目標標籤; 分別為三種鳶尾花類別: 0 (Setosa), 1 (Versicolor), 2 (Virginica)
5. `tre

---

[來源: ch06] 為特徵
4. `y_iris = iris.target`: 取得目標標籤; 分別為三種鳶尾花類別: 0 (Setosa), 1 (Versicolor), 2 (Virginica)
5. `tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)`: 建立決策樹分類器，限制深度為 2
6. `tree_clf.fit(X_iris, y_iris)`: 在訓練資料上擬合模型

**🎯 重點摘要:**

- **核心功能**: 使用鳶尾花資料集訓練決策樹
- **潛在問題**: 資料預處理和特徵選擇很重要
- **最佳使用情境**: 分類問題的基礎模型訓練

---

[來源: ch06] ### 範例 3: 匯出決策樹圖形

```python
from sklearn.tree import export_graphviz
from graphviz import Source

---

[來源: ch06] # 匯出決策樹為 dot 格式
export_graphviz(
        tree_clf,
        out_file=str(IMAGES_PATH / "iris_tree.dot"),
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

---

[來源: ch06] # 顯示決策樹圖形
Source.from_file(str(IMAGES_PATH / "iris_tree.dot"))
```

**✅ 程式碼逐行解析：**

1. `from sklearn.tree import export_graphviz`: 匯入圖形匯出函數
2. `export_graphviz(...)`: 將決策樹匯出為 GraphViz dot 檔案
3. `out_file=...`: 指定輸出檔案路徑
4. `feature_names=...`: 設定特徵名稱
5. `class_names=...`: 設定類別名稱
6. `rounded=True, filled=True`: 設定視覺化樣式
7. `Source.from_file(...)`: 從 dot 檔案讀取並顯示決策樹圖形

**🎯 重點摘要:**

---

[來源: ch06] 6. `rounded=True, filled=True`: 設定視覺化樣式
7. `Source.from_file(...)`: 從 dot 檔案讀取並顯示決策樹圖形

**🎯 重點摘要:**

- **核心功能**: 將決策樹視覺化為圖形檔案
- **潛在問題**: 需要透過 GraphViz 才能轉換為圖片
- **最佳使用情境**: 模型解釋和教學用途

---

[來源: ch06] ## 進行預測

訓練好模型後，我們可以使用它進行預測。

假設我們有一個新樣本，其花瓣長度為 5 公分，寬度為 1.5 公分。

---

[來源: ch06] # 使用訓練好的模型進行預測
predictions = tree_clf.predict([[5, 1.5]])
print("預測結果:", predictions)
```

**✅ 程式碼逐行解析：**

1. `tree_clf.predict([[5, 1.5]])`: 使用模型預測新樣本
2. `print(...)`: 輸出預測結果

**🎯 重點摘要:**

- **核心功能**: 使用決策樹進行單一樣本預測
- **潛在問題**: 輸入資料格式必須與訓練時一致
- **最佳使用情境**: 即時預測應用

---

[來源: ch06] ## 估計類別機率

決策樹不僅能進行預測，還能估計每個類別的機率。

---

[來源: ch06] # 估計類別機率
probabilities = tree_clf.predict_proba([[5, 1.5]])
print("類別機率:", probabilities.round(3))
```

**✅ 程式碼逐行解析：**

1. `tree_clf.predict_proba(...)`: 計算每個類別的預測機率
2. `.round(3)`: 將機率四捨五入到小數點後三位
3. `print(...)`: 輸出機率分佈

**🎯 重點摘要:**

- **核心功能**: 獲取決策樹的機率預測
- **潛在問題**: 機率估計依賴於樹的結構
- **最佳使用情境**: 需要不確定性估計的應用

---

[來源: ch06] ## 正則化超參數

為了避免過擬合 (Overfitting)，我們需要調整決策樹的正則化參數。

---

[來源: ch06] ### 範例 6: 比較不同正則化設定的效果

在此例，我們將使用月亮形資料集來比較無限制和有限制的決策樹模型。

藉由`min_samples_leaf` 參數，我們可以限制每個葉節點 (leaf node) 的最小樣本數量。

```python
from sklearn.datasets import make_moons

---

[來源: ch06] # 生成月亮形資料集
X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

---

[來源: ch06] # 建立兩個決策樹：一個無限制，一個有限制
tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)

---

[來源: ch06] # 訓練模型
tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)
```

**✅ 程式碼逐行解析：**

---

[來源: ch06] 1. `from sklearn.datasets import make_moons`: 匯入月亮形資料集生成器
2. `X_moons, y_moons = make_moons(...)`: 生成非線性可分的資料
3. `tree_clf1 = DecisionTreeClassifier(random_state=42)`: 無限制的決策樹
4. `tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, ...)`: 限制最小葉節點數量的決策樹
5. `tree_clf1.fit(...)`: 訓練無限制模型
6. `tree_clf2.fit(...)`: 訓練正則化模型

**🎯 重點摘要:**

---

[來源: ch06] ...)`: 限制最小葉節點數量的決策樹
5. `tree_clf1.fit(...)`: 訓練無限制模型
6. `tree_clf2.fit(...)`: 訓練正則化模型

**🎯 重點摘要:**

- **核心功能**: 比較過擬合和正則化模型的差異
- **潛在問題**: 無限制模型容易過擬合複雜資料
- **最佳使用情境**: 調整模型複雜度以平衡偏差和變異

---

[來源: ch06] ### 範例 7: 決策樹回歸

```python
from sklearn.tree import DecisionTreeRegressor

---

[來源: ch06] # 生成二次函數資料
np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5
y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)

---

[來源: ch06] # 訓練決策樹回歸器
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)
```

**✅ 程式碼逐行解析：**

---

[來源: ch06] DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.tree import DecisionTreeRegressor`: 匯入決策樹回歸器
2. `np.random.seed(42)`: 設定隨機種子確保重現性
3. `X_quad = np.random.rand(200, 1) - 0.5`: 生成輸入特徵
4. `y_quad = X_quad ** 2 + 0.025 * np.random.randn(...)`: 生成二次函數目標值加雜訊
5. `tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)`: 建立回歸樹
6. `tree_reg.fit(...)`: 訓練回歸模型

---

[來源: ch06] ree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)`: 建立回歸樹
6. `tree_reg.fit(...)`: 訓練回歸模型

**🎯 重點摘要:**

- **核心功能**: 使用決策樹進行回歸預測
- **潛在問題**: 回歸樹對離群值敏感
- **最佳使用情境**: 非線性回歸問題

---

[來源: ch06] # 生成方形資料並旋轉
np.random.seed(6)
X_square = np.random.rand(100, 2) - 0.5
y_square = (X_square[:, 0] > 0).astype(np.int64)

---

[來源: ch06] # 旋轉資料
angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
X_rotated_square = X_square.dot(rotation_matrix)

---

[來源: ch06] # 訓練兩個模型
tree_clf_square = DecisionTreeClassifier(random_state=42)
tree_clf_square.fit(X_square, y_square)
tree_clf_rotated_square = DecisionTreeClassifier(random_state=42)
tree_clf_rotated_square.fit(X_rotated_square, y_square)
```

**✅ 程式碼逐行解析：**

---

[來源: ch06] ssifier(random_state=42)
tree_clf_rotated_square.fit(X_rotated_square, y_square)
```

**✅ 程式碼逐行解析：**

1. `X_square = np.random.rand(100, 2) - 0.5`: 生成方形分佈的資料
2. `y_square = (X_square[:, 0] > 0).astype(np.int64)`: 基於 x 軸建立二元標籤
3. `rotation_matrix = ...`: 定義旋轉矩陣
4. `X_rotated_square = X_square.dot(rotation_matrix)`: 旋轉資料
5. `tree_clf_square.fit(...)`: 在原始資料上訓練
6. `tree_clf_rotated_square.fit(...)`: 在旋轉資料上訓練

---

[來源: ch06] trix)`: 旋轉資料
5. `tree_clf_square.fit(...)`: 在原始資料上訓練
6. `tree_clf_rotated_square.fit(...)`: 在旋轉資料上訓練

**🎯 重點摘要:**

- **核心功能**: 展示決策樹對軸向的敏感性
- **潛在問題**: 資料旋轉會導致完全不同的決策邊界
- **最佳使用情境**: 理解模型對特徵尺度化的需求

---

[來源: ch06] ## 決策樹的高變異性

決策樹具有高變異性，即使在相同資料上也可能產生不同模型。

---

[來源: ch06] # 使用不同隨機種子訓練模型
tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked.fit(X_iris, y_iris)
```

**✅ 程式碼逐行解析：**

1. `tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)`: 使用不同隨機種子`random_state=40`建立決策樹
2. `tree_clf_tweaked.fit(X_iris, y_iris)`: 在相同資料上訓練

**🎯 重點摘要:**

---

[來源: ch06] 0)`: 使用不同隨機種子`random_state=40`建立決策樹
2. `tree_clf_tweaked.fit(X_iris, y_iris)`: 在相同資料上訓練

**🎯 重點摘要:**

- **核心功能**: 展示決策樹的隨機性
- **潛在問題**: CART 演算法的隨機性導致模型變異
- **最佳使用情境**: 理解為何需要集成方法如隨機森林 (Random Forest) 來降低變異性

---

[來源: ch06] ## 額外內容：存取樹狀結構

決策樹不僅提供預測功能，還允許直接存取其內部結構，這對於模型解釋、除錯和進階分析非常有用。Scikit-Learn 的決策樹物件包含一個 `tree_` 屬性，這是一個低階物件，儲存了樹的完整結構，包括節點資訊、分割條件和葉節點的統計資料。

透過存取樹狀結構，您可以：

- 檢查每個節點的分割特徵和閾值
- 獲取葉節點的類別分佈或回歸值
- 計算樹的複雜度指標
- 實作自訂的樹遍歷演算法

這對於理解模型決策過程、進行特徵重要性分析，或開發自訂的可視化工具特別有幫助。請注意，直接操作內部結構需要對 CART 演算法有一定了解，並且在不同版本的 Scikit-Learn 中可能有所差異。

---

[來源: ch06] ### 範例 10: 存取樹狀結構

```python
tree = tree_clf.tree_
print("節點數量:", tree.node_count)
print("最大深度:", tree.max_depth)
print("葉節點數量:", tree.n_leaves)
```

**✅ 程式碼逐行解析：**

1. `tree = tree_clf.tree_`: 取得樹的內部結構
2. `tree.node_count`: 總節點數
3. `tree.max_depth`: 最大深度
4. `tree.n_leaves`: 葉節點數

**🎯 重點摘要:**

- **核心功能**: 檢查決策樹的結構屬性
- **潛在問題**: 直接存取內部結構需要了解實作細節
- **最佳使用情境**: 模型分析和除錯

---

[來源: ch06] # 存取節點資訊
print("節點 0 的分割特徵索引:", tree.feature[0])
print("節點 0 的分割閾值:", tree.threshold[0])
print("節點 0 的雜質:", tree.impurity[0])

---

[來源: ch06] # 對於葉節點，檢查樣本數量和值
leaf_mask = tree.children_left == tree.children_right  # 葉節點的左右子節點相同
print("葉節點樣本數量:", tree.n_node_samples[leaf_mask])
if hasattr(tree, 'value'):  # 分類樹
    print("葉節點類別分佈:", tree.value[leaf_mask])
```

**✅ 程式碼逐行解析:**

---

[來源: ch06] )
if hasattr(tree, 'value'):  # 分類樹
    print("葉節點類別分佈:", tree.value[leaf_mask])
```

**✅ 程式碼逐行解析:**

1. `tree.feature[0]`: 根節點的分割特徵索引
2. `tree.threshold[0]`: 根節點的分割閾值
3. `tree.impurity[0]`: 根節點的雜質度量 (如 Gini 或熵)
4. `leaf_mask = ...`: 識別葉節點
5. `tree.n_node_samples[leaf_mask]`: 葉節點的樣本數量
6. `tree.value[leaf_mask]`: 葉節點的類別計數 (分類) 或平均值 (回歸)

**🎯 重點摘要:**

---

[來源: ch06] _node_samples[leaf_mask]`: 葉節點的樣本數量
6. `tree.value[leaf_mask]`: 葉節點的類別計數 (分類) 或平均值 (回歸)

**🎯 重點摘要:**

- **核心功能**: 深入檢查樹的每個節點
- **潛在問題**: 索引操作需要小心，避免超出範圍
- **最佳使用情境**: 自訂模型解釋和特徵分析

---

[來源: ch06] ## 總結與最佳實踐

決策樹是機器學習中的重要工具，具有直觀、可解釋的優點。本教學涵蓋了從基礎訓練到進階應用的完整流程。

**最佳實踐：**

- 使用交叉驗證 (Cross-Validation) 選擇超參數
- 考慮使用 PCA 處理軸向敏感性
- 對於高變異性問題，考慮使用隨機森林
- 正確處理分類和回歸問題的不同需求

---

[來源: ch06] ## 常見問答 (FAQ)

**Q: 決策樹是否需要特徵標準化？**

A: 不需要，決策樹對特徵尺度不敏感。

**Q: 如何避免決策樹過擬合？**

A: 限制樹深度、葉節點數量，或使用最小樣本分割數。

**Q: 決策樹適合什麼類型的資料？**

A: 適合處理分類和回歸問題，尤其在需要模型解釋時。

---

[來源: ch06] ## 推薦標籤

推薦標籤：#Python #機器學習 #決策樹 #ScikitLearn #程式教學 #資料科學 #人工智慧 #演算法

---

[來源: ch06 | 類型: cheatsheet] # Ch06 速查表：Decision Trees

> **核心主旨**：決策樹直觀可解釋，但容易 overfitting —— 正則化（限制深度）是必要的。

---

---

[來源: ch06 | 類型: cheatsheet] ## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Gini Impurity | 節點不純度，越低越好，預設分割準則 | 分類問題（預設） |
| Entropy | 資訊熵，與 Gini 效果相近但略慢 | 分類問題（另一選項） |
| Regularization | 限制樹的成長防止 overfitting | 小/中型資料集 |
| `max_depth` | 限制樹的最大深度 | 最常用的正則化參數 |
| `min_samples_leaf` | 葉節點最少樣本數 | 防止過細的分割 |
| Feature Importance | 每個特徵在所有節點的 impurity 減少量加總 | 特徵選擇、模型解釋 |
| `predict_proba` | 輸出每個類別的機率 | 需要機率輸出時 |

---

---

[來源: ch06 | 類型: cheatsheet] | sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `DecisionTreeClassifier` | `max_depth=5`, `min_samples_leaf=10`, `criterion="gini"` | 分類樹 |
| `DecisionTreeRegressor` | `max_depth=5`, `min_samples_split=20` | 迴歸樹 |
| `export_graphviz` | `feature_names=`, `class_names=`, `filled=True` | 輸出 .dot 圖形檔 |
| `.feature_importances_` | – | 特徵重要性陣列 |
| `.predict_proba()` | – | 輸出類別機率 |
| `plot_tree` | `filled=True`, `feature_names=`, `max_depth=3` | 直接在 matplotlib 上繪製 |


---

[來源: ch06 | 類型: cheatsheet] – | 輸出類別機率 |
| `plot_tree` | `filled=True`, `feature_names=`, `max_depth=3` | 直接在 matplotlib 上繪製 |


---

---

[來源: ch06 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_graphviz
import matplotlib.pyplot as plt
import numpy as np

---

[來源: ch06 | 類型: cheatsheet] # 訓練分類樹（加正則化）
tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_clf.fit(X_train, y_train)

---

[來源: ch06 | 類型: cheatsheet] # 視覺化（最直觀的方法）
plt.figure(figsize=(12, 6))
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names, rounded=True)
plt.show()

---

[來源: ch06 | 類型: cheatsheet] # 匯出 graphviz（需要 graphviz 套件）
export_graphviz(tree_clf, out_file="tree.dot",
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                filled=True, rounded=True)

---

[來源: ch06 | 類型: cheatsheet] # 預測機率
iris = load_iris(as_frame=True)
X, y = iris.data[["petal length (cm)", "petal width (cm)"]], iris.target
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)
print(tree_clf.predict_proba([[5, 1.5]]))  # 各類別機率
print(tree_clf.predict([[5, 1.5]]))         # 類別預測

---

[來源: ch06 | 類型: cheatsheet] # 特徵重要性
importances = tree_clf.feature_importances_
for name, imp in zip(iris.feature_names, importances):
    print(f"{name}: {imp:.3f}")

---

[來源: ch06 | 類型: cheatsheet] # 迴歸樹
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train, y_train)

---

[來源: ch06 | 類型: cheatsheet] # 存取樹結構（進階）
n_nodes = tree_clf.tree_.node_count
print(f"節點數: {n_nodes}")
```

---

---

[來源: ch06 | 類型: cheatsheet] ## 4. 常見陷阱

- **不需要 Scaling**：決策樹基於排序（分割點），特徵尺度不影響結果，**不需要 StandardScaler**。
- **不穩定性（High Variance）**：資料略微改動可能產生完全不同的樹結構，這是決策樹的天然缺陷 —— 改用 Random Forest 更穩定。
- **對旋轉敏感**：決策樹只做軸向切割，若類別邊界是斜線，需要 PCA 預處理或改用其他模型。
- **過深的樹 = 過擬合**：通常 `max_depth=4~8` 是合理起點，用 CV 調整。

---

---

[來源: ch06 | 類型: cheatsheet] ## 5. 決策指南

```
決策樹的最佳使用場景：
├── 需要可解釋性（可畫圖給非技術人員看）
├── 混合數值 + 類別特徵（不需前處理）
└── 作為 Ensemble 的基底學習器（Random Forest, GBM）

正則化參數推薦搜索範圍：
├── max_depth: [2, 3, 4, 5, 8, None]
├── min_samples_leaf: [1, 5, 10, 20, 50]
└── min_samples_split: [2, 5, 10]

單一決策樹 vs Random Forest：
└── 若不需要單棵樹的可解釋性，Random Forest 幾乎永遠更好
```

---

[來源: ch06 | 類型: handout] # 課程講義：決策樹 (Chapter 06)

決策樹是機器學習中最「可解釋」的模型之一——它的決策過程就像一系列的「是/否」問題，任何人都能讀懂。本章帶你理解決策樹的訓練演算法（CART）、Gini 不純度、以及為何決策樹對訓練集的「邊界形狀」非常敏感。這些知識是理解第七章集成學習（尤其是隨機森林）的必要基礎。

---

---

[來源: ch06 | 類型: handout] ### 理論背景

**CART (Classification and Regression Trees)** 演算法：

在每個節點，CART 窮舉所有特徵 $j$ 和所有閾值 $t_k$，找到能最小化加權不純度的分裂點：

$$J(j, t_k) = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}$$

**Gini 不純度**：

$$G_i = 1 - \sum_{k=1}^{K} p_{i,k}^2$$

其中 $p_{i,k}$ 是節點 $i$ 中類別 $k$ 的樣本比例。$G_i = 0$ 表示節點完全純淨（只含一種類別）。

**信息增益（Entropy 基準）**：

---

[來源: ch06 | 類型: handout] } p_{i,k}^2$$

其中 $p_{i,k}$ 是節點 $i$ 中類別 $k$ 的樣本比例。$G_i = 0$ 表示節點完全純淨（只含一種類別）。

**信息增益（Entropy 基準）**：

$$H_i = -\sum_{k=1, p_{i,k} \neq 0}^{K} p_{i,k} \log_2(p_{i,k})$$

Gini 計算較快，Entropy 有時能產生更平衡的樹；實際差異通常不大，Gini 是預設值。

**時間複雜度**：訓練 $O(n \cdot m \log m)$，預測 $O(\log m)$（$m$ 為訓練樣本數，$n$ 為特徵數）。

---

[來源: ch06 | 類型: handout] ### 核心代碼

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

---

[來源: ch06 | 類型: handout] # 訓練決策樹（限制深度防止過擬合）
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

---

[來源: ch06 | 類型: handout] # 視覺化樹結構
plt.figure(figsize=(12, 5))
plot_tree(tree_clf,
          feature_names=["petal length", "petal width"],
          class_names=iris.target_names,
          filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree (max_depth=2)")
plt.tight_layout()
plt.show()
```

---

[來源: ch06 | 類型: handout] ### 補充練習 1

**理論題：** 手動計算根節點（所有 150 筆 Iris 資料，三類各 50 筆）的 Gini 不純度。再計算根節點按「花瓣長度 ≤ 2.45 cm」分裂後，左子節點的 Gini 不純度（左節點包含 50 筆 setosa）。

**實作題：** 訓練 `max_depth=3` 的決策樹，用 `plot_tree` 視覺化，找出在哪個節點 Gini 不純度最低，對應的分裂條件是什麼？

---

---

[來源: ch06 | 類型: handout] ### 理論背景

每個節點顯示的資訊：

```
petal length (cm) <= 2.45
gini = 0.667
samples = 150
value = [50, 50, 50]
class = setosa
```

- **分裂條件**：用於路由樣本
- **gini**：此節點的不純度
- **samples**：此節點的訓練樣本數
- **value**：各類別的樣本數
- **class**：多數類（葉節點的預測結果）

**預測機率 `predict_proba()`**：葉節點中各類別的比例即為預測機率：

$$\hat{p}_k = \frac{\text{此葉節點中類別 k 的樣本數}}{\text{此葉節點的總樣本數}}$$

---

[來源: ch06 | 類型: handout] predict_proba()`**：葉節點中各類別的比例即為預測機率：

$$\hat{p}_k = \frac{\text{此葉節點中類別 k 的樣本數}}{\text{此葉節點的總樣本數}}$$

**白盒 vs 黑盒**：決策樹是「白盒」模型——可以追蹤任何預測的決策路徑，清楚說明「為何」做出此預測，對高風險應用（醫療、法律）非常重要。

---

[來源: ch06 | 類型: handout] # 預測機率和類別
X_new = [[5, 1.5]]
print(tree_clf.predict_proba(X_new))  # [[0., 0.907, 0.093]]
print(tree_clf.predict(X_new))        # [1] → Iris versicolor

---

[來源: ch06 | 類型: handout] # 查看決策路徑（某樣本走哪條路）
from sklearn.tree import _tree

---

[來源: ch06 | 類型: handout] def get_decision_path(clf, X_sample):
    node_indicator = clf.decision_path(X_sample)
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    for node_id in node_indicator.indices:
        if feature[node_id] != _tree.TREE_UNDEFINED:
            print(f"  節點 {node_id}: X[{feature[node_id]}] "
                  f"{'<=' if X_sample[0, feature[node_id]] <= threshold[node_id] else '>'

---

[來源: ch06 | 類型: handout] de_id]}] "
                  f"{'<=' if X_sample[0, feature[node_id]] <= threshold[node_id] else '>'} "
                  f"{threshold[node_id]:.2f}")

import numpy as np
get_decision_path(tree_clf, np.array([[5, 1.5]]))
```

---

[來源: ch06 | 類型: handout] ### 補充練習 2

**理論題：** 決策樹預測機率時，葉節點中某類別只有 1 筆樣本，但被預測為 100% 機率，這合理嗎？Laplace Smoothing 如何解決這個問題？

**實作題：** 用 `DecisionTreeClassifier(max_depth=4)` 繪製決策邊界（在 petal length × petal width 的 2D 空間）。注意邊界是水平和垂直線組成的「直角」形狀——這正是決策樹偏好軸對齊分裂的特點。

---

---

[來源: ch06 | 類型: handout] ### 理論背景

未加限制的決策樹會生長到每個葉節點只含一個樣本（訓練準確率 100%，嚴重過擬合）。

**主要正則化超參數**：

| 超參數 | 說明 | 增大的效果 |
|--------|------|-----------|
| `max_depth` | 樹的最大深度 | 降低複雜度（防過擬合） |
| `min_samples_split` | 節點分裂所需最少樣本數 | 降低複雜度 |
| `min_samples_leaf` | 葉節點最少樣本數 | 降低複雜度 |
| `max_features` | 每次分裂考慮的最多特徵數 | 降低複雜度（增加隨機性） |
| `max_leaf_nodes` | 最多葉節點數 | 降低複雜度 |


---

[來源: ch06 | 類型: handout] 本數 | 降低複雜度 |
| `max_features` | 每次分裂考慮的最多特徵數 | 降低複雜度（增加隨機性） |
| `max_leaf_nodes` | 最多葉節點數 | 降低複雜度 |


**剪枝 (Pruning)**：先訓練完整的樹，再移除對測試集效能貢獻不大的節點（後剪枝）。`ccp_alpha` 參數控制最小代價複雜度剪枝。

---

[來源: ch06 | 類型: handout] ### 核心代碼

```python
from sklearn.model_selection import train_test_split, GridSearchCV

---

[來源: ch06 | 類型: handout] # 各種深度對比
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris,
                                                     test_size=0.3, random_state=42)
for depth in [None, 2, 3, 5]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    print(f"max_depth={str(depth):4s}  "
          f"train acc={clf.score(X_train, y_train):.3f}  "
          f"test acc={clf.score(X_test, y_test):.3f}")

---

[來源: ch06 | 類型: handout] # 最小代價複雜度剪枝
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # 去掉最後一個（退化的樹）
clfs = [DecisionTreeClassifier(ccp_alpha=a, random_state=42).fit(X_train, y_train)
        for a in ccp_alphas]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
best_alpha = ccp_alphas[test_scores.index(max(test_scores))]
print(f"最佳 ccp_alpha: {best_alpha:.6f}")
```

---

[來源: ch06 | 類型: handout] ### 補充練習 3

**理論題：** `min_samples_leaf=5` 意味著什麼？為什麼設定葉節點的最小樣本數比限制深度更「均勻」地正則化樹？

**實作題：** 在 `make_moons(n_samples=500, noise=0.3)` 資料集上，用 `GridSearchCV` 搜索 `max_depth` ∈ [3, 5, 7, 10] 與 `min_samples_leaf` ∈ [1, 3, 5, 10] 的最佳組合。繪製最佳模型的決策邊界。

---

---

[來源: ch06 | 類型: handout] ### 理論背景

**迴歸樹 (Regression Tree)**：葉節點的預測值是該節點訓練樣本的**平均值**。

CART 迴歸的分裂準則：最小化 MSE（而非 Gini 不純度）：

$$J(j, t_k) = \frac{m_{\text{left}}}{m} \text{MSE}_{\text{left}} + \frac{m_{\text{right}}}{m} \text{MSE}_{\text{right}}$$

**決策樹的根本弱點**：

1. **對旋轉敏感**：決策邊界只能軸對齊；若資料旋轉 45°，需要更深的樹才能近似斜線邊界。
2. **高變異數 (High Variance)**：訓練資料稍有變動，整棵樹可能截然不同。
3. **貪婪演算法**：每次只選當前最好的分裂，不保證全域最優。

---

[來源: ch06 | 類型: handout] 5°，需要更深的樹才能近似斜線邊界。
2. **高變異數 (High Variance)**：訓練資料稍有變動，整棵樹可能截然不同。
3. **貪婪演算法**：每次只選當前最好的分裂，不保證全域最優。

這些弱點正是為什麼**集成方法（隨機森林、Gradient Boosting）**能大幅超越單棵決策樹。

---

[來源: ch06 | 類型: handout] ### 核心代碼

```python
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X_reg = np.sort(6 * np.random.rand(200, 1) - 3, axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.randn(200) * 0.2

tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_reg, y_reg)

---

[來源: ch06 | 類型: handout] # 視覺化預測曲線（決策樹迴歸是分段常數函數）
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
plt.scatter(X_reg, y_reg, s=10, alpha=0.5)
plt.plot(X_plot, tree_reg.predict(X_plot), "r-", linewidth=2)
plt.title("Decision Tree Regression (max_depth=3)")
plt.show()
```

---

[來源: ch06 | 類型: handout] ### 補充練習 4

**理論題：** 決策樹迴歸預測的是分段常數函數（每個葉節點的輸出是一個固定值）。這意味著決策樹**無法外推（Extrapolate）**——超出訓練資料範圍的預測會怎樣？

**實作題：** 用 `max_depth=None`（未剪枝）和 `max_depth=3` 分別訓練迴歸樹，比較兩者在訓練集和測試集的 RMSE，並繪製預測曲線，直觀觀察過擬合現象。

---

---

[來源: ch06 | 類型: handout] ## 結論

決策樹以直觀的樹狀結構實現分類與迴歸：

- **CART 演算法**透過貪婪地最小化 Gini 不純度（或 MSE）分裂節點
- **葉節點**給出預測類別（多數決）或預測值（平均）
- **正則化超參數**（`max_depth`、`min_samples_leaf`、`ccp_alpha`）是防止過擬合的關鍵
- 決策樹的**高變異數弱點**促成了第七章的集成學習

下一章（Ch07）展示如何將許多棵弱決策樹組合成強大的集成模型。

---

---

[來源: ch06 | 類型: handout] ## 課後作業

**作業：月亮資料集完整分析**

1. 使用 `make_moons(n_samples=10000, noise=0.4)` 產生資料集，切分為訓練集（8000）和測試集（2000）。
2. 用 `GridSearchCV` 找到最佳的 `DecisionTreeClassifier`（搜索 `max_leaf_nodes` ∈ [10, 50, 100, 500]、`min_samples_split` ∈ [2, 5, 10]）。
3. 在測試集評估最佳模型的準確率。
4. **進階**：訓練 1000 個決策樹，每個樹用 100 個隨機抽取的訓練樣本（帶回放），每棵樹對測試集預測，最終用多數投票決定類別。這就是隨機森林的原型！比較這個手工隨機森林與 `RandomForestClassifier` 的準確率。

---

[來源: ch06 | 類型: tutorial] [標題: 決策樹完整指南：CART 演算法、Gini 不純度與剪枝策略實戰 | 描述: 深入理解決策樹的 CART 演算法、Gini 不純度、熵、資訊增益，學習如何用 Scikit-Learn 訓練、視覺化、正規化決策樹，包含分類和回歸完整實戰。 | 關鍵字: Python, 決策樹, CART, Gini, 資訊增益, Scikit-Learn, 機器學習, 正規化, 過擬合]
# 決策樹完整指南：從 CART 到剪枝策略

決策樹（Decision Trees）是機器學習中最直觀的演算法——它的決策過程就像人類的問答遊戲。決策樹不僅能單獨使用，更是隨機森林（Random Forest）和梯度提升（Gradient Boosting）等集成方法的核心組件。本教學帶你深入理解決策樹的建構原理，並學習如何避免它天生的過擬合傾向。

---

[來源: ch06 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- 決策樹使用 **CART 演算法**（分類與回歸樹），每次分裂選擇最能降低不純度的特徵和閾值
- **Gini 不純度**（預設）和**熵（Entropy）** 都是衡量節點不純度的指標，通常結果相似
- 決策樹本身有**高方差（High Variance）** 傾向——對訓練集的微小變化非常敏感
- 透過 `max_depth`、`min_samples_leaf` 等參數**預剪枝**是控制過擬合的主要方式
- 決策樹的決策邊界**平行於座標軸**，對旋轉資料敏感（可用 PCA 前處理改善）

---

---

[來源: ch06 | 類型: tutorial] ## 訓練與視覺化決策樹

💡 **實際應用情境：** 台灣銀行的信用評估系統常使用決策樹，因為樹的結構可以直接向客戶解釋「為何拒絕貸款」（例如：年收入 < 60 萬 且 無房產 → 拒絕）。這種可解釋性在金融監管合規中尤其重要。

---

[來源: ch06 | 類型: tutorial] ### 範例 1: 訓練鳶尾花分類樹

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import matplotlib.pyplot as plt

---

[來源: ch06 | 類型: tutorial] # 載入資料
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

---

[來源: ch06 | 類型: tutorial] # 訓練決策樹（限制深度防止過擬合）
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)
print(f"訓練準確率: {tree_clf.score(X_iris, y_iris):.4f}")

---

[來源: ch06 | 類型: tutorial] # 視覺化：Matplotlib 版（不需要 Graphviz）
plt.figure(figsize=(12, 6))
plot_tree(
    tree_clf,
    feature_names=["petal length", "petal width"],
    class_names=iris.target_names,
    filled=True,        # 以顏色填充（多數類決定顏色深淺）
    rounded=True,       # 圓角方框
    impurity=True,      # 顯示 Gini 不純度
    proportion=False    # 顯示樣本數（而非比例）
)
plt.title("Iris 決策樹（max_depth=2）")
plt.savefig("iris_decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch06 | 類型: tutorial] )
plt.savefig("iris_decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `DecisionTreeClassifier(max_depth=2)`: 限制樹的最大深度為 2，避免過擬合
2. `plot_tree(filled=True)`: 以顏色深淺表示節點的主要類別和純度（越深表示越純）
3. 每個節點顯示：分裂條件、Gini 不純度、樣本數、各類別分佈

**🎯 重點摘要:**

- `export_graphviz()` 可生成 `.dot` 文件，用 Graphviz 渲染更美觀的圖表
- `plot_tree()` 無需額外安裝，適合快速視覺化

---

---

[來源: ch06 | 類型: tutorial] # 預測類別機率（而非只預測類別）
sample = [[5.0, 1.5]]  # 花瓣長 5cm，花瓣寬 1.5cm

proba = tree_clf.predict_proba(sample)
print("各類別機率:")
for cls, p in zip(iris.target_names, proba[0]):
    print(f"  {cls}: {p:.2%}")

---

[來源: ch06 | 類型: tutorial] # 預測類別（機率最高的那個）
pred_class = tree_clf.predict(sample)
print(f"預測類別: {iris.target_names[pred_class[0]]}")

---

[來源: ch06 | 類型: tutorial] # 查看預測路徑（Decision Path）
decision_path = tree_clf.decision_path([sample])
print(f"經過的節點: {decision_path.indices}")
```

**✅ 程式碼逐行解析：**

1. `predict_proba(sample)`: 回傳每個類別的機率，對應節點中各類別的樣本比例
2. 決策樹的機率估計：到達葉節點時，機率 = 該葉節點中該類別的樣本數 ÷ 葉節點總樣本數
3. `decision_path()`: 回傳樣本通過哪些節點（用於模型解釋）

**🎯 重點摘要:**

- 決策樹的機率估計**不夠精細**（同一葉節點所有樣本得到相同機率）
- 若需要平滑機率，考慮使用隨機森林（對多棵樹的機率平均）

---

---

[來源: ch06 | 類型: tutorial] ### 理論說明：CART 分裂準則

**Gini 不純度（預設）**：

$$G_i = 1 - \sum_{k=1}^{K} p_{i,k}^2$$

- $p_{i,k}$：節點 $i$ 中類別 $k$ 的比例
- 純節點（只有一種類別）：$G_i = 0$
- 最不純（所有類別均等分佈）：$G_i = 1 - 1/K$

**熵（Entropy）**：

$$H_i = -\sum_{k=1}^{K} p_{i,k} \log_2(p_{i,k})$$

---

[來源: ch06 | 類型: tutorial] ### 範例 3: Gini vs Entropy 的比較

```python

---

[來源: ch06 | 類型: tutorial] # Gini vs Entropy 的決策樹比較
tree_gini = DecisionTreeClassifier(criterion="gini",    max_depth=3, random_state=42)
tree_entr = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)

from sklearn.model_selection import cross_val_score

for name, clf in [("Gini", tree_gini), ("Entropy", tree_entr)]:
    scores = cross_val_score(clf, X_iris, y_iris, cv=5, scoring="accuracy")
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

---

[來源: ch06 | 類型: tutorial] # 實際上兩者效果通常差不多（Gini 稍快，Entropy 稍準）
```

**🎯 重點摘要:**

- Gini 計算比熵快（不需要 log），通常是首選
- 兩種準則在大多數情況下產生相似的樹，差異不大
- **訊息增益（Information Gain）** = 父節點熵 − 加權子節點熵，即 CART 最大化的目標

---

---

[來源: ch06 | 類型: tutorial] ## 正規化：控制樹的複雜度

💡 **實際應用情境：** 未限制的決策樹會完美記憶訓練資料（訓練準確率 100%），但在新資料上效果很差。正規化參數就像「剪枝」，讓樹更簡潔、更泛化。

---

[來源: ch06 | 類型: tutorial] ### 範例 4: 正規化超參數的影響

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score

X_moons, y_moons = make_moons(n_samples=200, noise=0.25, random_state=42)

---

[來源: ch06 | 類型: tutorial] # 各種正規化設定
configs = {
    "無限制": DecisionTreeClassifier(random_state=42),
    "max_depth=3": DecisionTreeClassifier(max_depth=3, random_state=42),
    "min_samples_leaf=5": DecisionTreeClassifier(min_samples_leaf=5, random_state=42),
    "min_samples_split=10": DecisionTreeClassifier(min_samples_split=10, random_state=42),
    "max_leaf_nodes=10": DecisionTreeClassifier(max_leaf_nodes=10, random_state=42),
}

---

[來源: ch06 | 類型: tutorial] om_state=42),
    "max_leaf_nodes=10": DecisionTreeClassifier(max_leaf_nodes=10, random_state=42),
}

print("正規化參數比較（5-Fold CV 準確率）:")
for name, clf in configs.items():
    scores = cross_val_score(clf, X_moons, y_moons, cv=5, scoring="accuracy")
    print(f"  {name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

主要正規化參數：

---

[來源: ch06 | 類型: tutorial] oring="accuracy")
    print(f"  {name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

主要正規化參數：

| 參數 | 說明 | 過大時（欠擬合） | 過小時（過擬合） |
|------|------|----------------|----------------|
| `max_depth` | 最大樹深 | 模型太簡單 | 無限深度 |
| `min_samples_leaf` | 葉節點最小樣本數 | 葉節點太大 | 每個樣本一個葉 |
| `min_samples_split` | 分裂所需最小樣本數 | 提前停止分裂 | 只有 2 個就分裂 |
| `max_leaf_nodes` | 最大葉節點數 | 樹很小 | 無限制 |


**🎯 重點摘要:**

- `max_depth` 是最常用的正規化參數，通常從 `3-8` 開始嘗試
- `min_samples_leaf=1%~5% × 訓練集大小` 是一個常用的啟發式設定

---

---

[來源: ch06 | 類型: tutorial] ### 範例 5: DecisionTreeRegressor

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

np.random.seed(42)
X_reg = np.sort(5 * np.random.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.randn(80) * 0.1

---

[來源: ch06 | 類型: tutorial] # 兩種深度的回歸樹
tree_reg_2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg_5 = DecisionTreeRegressor(max_depth=5, random_state=42)

tree_reg_2.fit(X_reg, y_reg)
tree_reg_5.fit(X_reg, y_reg)

print(f"depth=2 訓練 MSE: {((tree_reg_2.predict(X_reg) - y_reg)**2).mean():.4f}")
print(f"depth=5 訓練 MSE: {((tree_reg_5.predict(X_reg) - y_reg)**2).mean():.4f}")

---

[來源: ch06 | 類型: tutorial] # depth=5 的訓練誤差遠低，但可能過擬合
```

**✅ 程式碼逐行解析：**

1. 決策樹回歸的節點分裂準則：最小化加權 MSE（均方誤差）
2. 葉節點的預測值 = 落入該葉節點的所有訓練樣本的**均值**
3. `max_depth=5` 產生的階梯形預測曲線過度擬合了訓練資料的噪音

**🎯 重點摘要:**

- 決策樹回歸預測**分段常數函數**（階梯形），不能外推
- 回歸樹和分類樹除了分裂準則不同（MSE vs Gini），結構完全相同

---

---

[來源: ch06 | 類型: tutorial] ### 範例 6: 決策邊界平行軸的限制

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

np.random.seed(6)
X_square = np.random.rand(100, 2) - 0.5
y_square = ((X_square[:, 0] > 0) ^ (X_square[:, 1] > 0)).astype(int)

---

[來源: ch06 | 類型: tutorial] # 旋轉 45 度後，決策樹需要更多層才能分類
angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle),  np.cos(angle)]])
X_rotated = X_square @ rotation_matrix

---

[來源: ch06 | 類型: tutorial] # 對比：旋轉前後的決策樹深度需求
tree_original = DecisionTreeClassifier(random_state=42)
tree_rotated  = DecisionTreeClassifier(random_state=42)

tree_original.fit(X_square, y_square)
tree_rotated.fit(X_rotated, y_square)

print(f"原始資料 - 樹深度: {tree_original.get_depth()}, 葉節點: {tree_original.get_n_leaves()}")
print(f"旋轉資料 - 樹深度: {tree_rotated.get_depth()}, 葉節點: {tree_rotated.get_n_leaves()}")

---

[來源: ch06 | 類型: tutorial] # 旋轉後需要更多分裂才能完成相同的分類任務
```

**🎯 重點摘要:**

- 決策樹的邊界**只能平行於特徵軸**，對角線邊界需要很多分裂
- 解法：用 PCA 預處理（旋轉特徵），或改用隨機森林（多棵樹的平均消除這個限制）

---

---

[來源: ch06 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: 決策樹需要特徵縮放嗎？**

A: 不需要！決策樹使用閾值分裂（`feature > threshold`），不依賴特徵之間的距離或尺度。這是決策樹相較於 SVM、KNN 的一大優點。

**Q2: Gini 和 Entropy 哪個更好？**

A: 通常差別不大。Gini 計算稍快（不需要 log），是預設值。若想精確測試，用交叉驗證比較兩者。

**Q3: 為什麼決策樹是「高方差」模型？**

A: 決策樹對訓練資料的微小變化非常敏感——移除幾個樣本或加入少量雜訊，可能完全改變樹的結構。這就是為什麼隨機森林（多棵決策樹的集成）通常比單棵決策樹效果好得多。

**Q4: `max_features` 參數有什麼用？**

A: 在隨機森林中使用，每次分裂只從隨機選出的部分特徵中搜索最佳分裂點。這引入了隨機性，讓每棵樹更獨立，集成效果更好。對單棵決策樹通常不需要設定。

---

---

[來源: ch06 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #決策樹 #CART #機器學習 #ScikitLearn #Gini #資訊增益 #隨機森林 #程式設計 #教學 #DataScience #MachineLearning #AI #特徵重要性

---

[來源: ch07] [標題: 第7章 – 集成學習與隨機森林 | 描述: 本章探討集成學習技術，結合多個機器學習模型以提升預測效能與穩健性。我們將涵蓋投票分類器、裝袋與粘貼、隨機森林、提升方法如AdaBoost與梯度提升，以及堆疊。這些方法利用群眾智慧來減少過擬合並提升準確度。 | 關鍵字: 機器學習, 集成學習, 隨機森林, 投票分類器, 裝袋, 提升, 堆疊, Python, Scikit-Learn]
# 第7章 – 集成學習與隨機森林 (Ensemble Learning and Random Forests)

本章探討集成學習 (Ensemble Learning) 技術，結合多個機器學習模型以提升預測效能與穩健性。我們將涵蓋投票分類器、裝袋與粘貼、隨機森林、提升方法如AdaBoost與梯度提升，以及堆疊。這些方法利用群眾智慧來減少過擬合並提升準確度。

---

[來源: ch07] ## 關鍵重點 (Key Takeaways)

- 集成學習結合多個模型以提升效能與穩健性。
- 投票分類器使用硬投票或軟投票來聚合預測。
- 裝袋與粘貼通過隨機子集訓練減少過擬合。
- 隨機森林是裝袋決策樹的擴展，具有特徵重要性。
- 提升方法如AdaBoost與梯度提升依次修正錯誤。
- 堆疊使用元學習器組合多個模型的預測。

---

---

[來源: ch07] ## 設定

此專案需要Python 3.7或以上：

```python
import sys

assert sys.version_info >= (3, 7)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 匯入sys模組以檢查Python版本。
2. `assert sys.version_info >= (3, 7)`: 確保Python版本至少為3.7，否則引發AssertionError。

**🎯 重點摘要:**

- **核心功能**: 版本檢查確保相容性。
- **潛在問題**: 舊版Python可能導致相容性問題。
- **最佳使用情境**: 在腳本開始處進行環境驗證。

它也需要Scikit-Learn ≥ 1.0.1：

---

[來源: ch07] **: 版本檢查確保相容性。
- **潛在問題**: 舊版Python可能導致相容性問題。
- **最佳使用情境**: 在腳本開始處進行環境驗證。

它也需要Scikit-Learn ≥ 1.0.1：

```python
from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] ort sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
```

**✅ 程式碼逐行解析：**

1. `from packaging import version`: 匯入version模組以比較版本。
2. `import sklearn`: 匯入sklearn模組。
3. `assert version.parse(sklearn.__version__) >= version.parse("1.0.1")`: 確保sklearn版本至少為1.0.1。

**🎯 重點摘要:**

- **核心功能**: 檢查sklearn版本。
- **潛在問題**: 舊版可能缺少功能。
- **最佳使用情境**: 確保依賴項版本正確。

讓我們定義預設字體大小以使圖表更美觀：

---

[來源: ch07] 點摘要:**

- **核心功能**: 檢查sklearn版本。
- **潛在問題**: 舊版可能缺少功能。
- **最佳使用情境**: 確保依賴項版本正確。

讓我們定義預設字體大小以使圖表更美觀：

```python
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] egend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

1. `import matplotlib.pyplot as plt`: 匯入matplotlib.pyplot。
2. `plt.rc('font', size=14)`: 設定預設字體大小為14。
3. `plt.rc('axes', labelsize=14, titlesize=14)`: 設定軸標籤和標題字體大小。
4. `plt.rc('legend', fontsize=14)`: 設定圖例字體大小。
5. `plt.rc('xtick', labelsize=10)`: 設定x軸刻度字體大小。
6. `plt.rc('ytick', labelsize=10)`: 設定y軸刻度字體大小。

---

[來源: ch07] 體大小。
5. `plt.rc('xtick', labelsize=10)`: 設定x軸刻度字體大小。
6. `plt.rc('ytick', labelsize=10)`: 設定y軸刻度字體大小。

**🎯 重點摘要:**

- **核心功能**: 統一圖表字體大小。
- **潛在問題**: 可能影響圖表可讀性。
- **最佳使用情境**: 製作出版品質圖表。

並建立`images/ensembles`資料夾（如果不存在），並定義`save_fig()`函數在本筆記本中使用以將圖表以高解析度儲存：

---

[來源: ch07] 表可讀性。
- **最佳使用情境**: 製作出版品質圖表。

並建立`images/ensembles`資料夾（如果不存在），並定義`save_fig()`函數在本筆記本中使用以將圖表以高解析度儲存：

```python
from pathlib import Path

IMAGES_PATH = Path() / "images" / "ensembles"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

---

[來源: ch07] t_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `from pathlib import Path`: 匯入Path類別。
2. `IMAGES_PATH = Path() / "images" / "ensembles"`: 定義圖片儲存路徑。
3. `IMAGES_PATH.mkdir(parents=True, exist_ok=True)`: 建立目錄如果不存在。
4. `def save_fig(...)`: 定義儲存圖表函數。
5. `path = IMAGES_PATH / f"{fig_id}.{fig_extension}"`: 建構檔案路徑。
6. `if tight_layout: plt.tight_layout()`: 調整佈局。
7. `plt.savefig(...)`: 儲存圖表。

---

[來源: ch07] g_extension}"`: 建構檔案路徑。
6. `if tight_layout: plt.tight_layout()`: 調整佈局。
7. `plt.savefig(...)`: 儲存圖表。

**🎯 重點摘要:**

- **核心功能**: 自動建立目錄並儲存圖表。
- **潛在問題**: 檔案覆蓋。
- **最佳使用情境**: 批次儲存圖表。

---

---

[來源: ch07] ## 投票分類器 (Voting Classifier)

投票分類器是集成方法，結合多個個別分類器的預測以提升整體準確度和穩健性。通過聚合不同模型（如邏輯回歸、隨機森林和支援向量機）的輸出，它們利用「群眾智慧」來減少過擬合並提升未見資料的效能。本節示範如何在Scikit-Learn中實作硬投票和軟投票。

為了說明大數法則，我們可以模擬多個序列的偏斜硬幣投擲，並繪製每個序列的運行頭部比例。

```python

---

[來源: ch07] # extra code – this cell generates and saves Figure 7–3

import matplotlib.pyplot as plt
import numpy as np

heads_proba = 0.51
np.random.seed(42)
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
cumulative_heads = coin_tosses.cumsum(axis=0)
cumulative_heads_ratio = cumulative_heads / np.arange(1, 10001).reshape(-1, 1)

---

[來源: ch07] tosses.cumsum(axis=0)
cumulative_heads_ratio = cumulative_heads / np.arange(1, 10001).reshape(-1, 1)

plt.figure(figsize=(8, 3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
plt.grid()
save_fig("law_of_large_numbers_plot")
plt.show()

---

[來源: ch07] right")
plt.axis([0, 10000, 0.42, 0.58])
plt.grid()
save_fig("law_of_large_numbers_plot")
plt.show()

```

**✅ 程式碼逐行解析：**

1. `heads_proba = 0.51`: 設定頭部機率為0.51。
2. `np.random.seed(42)`: 設定隨機種子。
3. `coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)`: 生成投擲結果。
4. `cumulative_heads = coin_tosses.cumsum(axis=0)`: 計算累積頭部數。
5. `cumulative_heads_ratio = cumulative_heads / np.arange(1, 10001).reshape(-1, 1)`: 計算運行比例。
6. `plt.figure(figsize=(8, 3.5))`: 建立圖表。
7. `plt.plot(cumulative_heads_ratio)`: 繪製運行比例。
8. `plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")`: 繪製51%參考線。
9. `plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")`: 繪製50%參考線。
10. `plt.xlabel("Number of coin tosses")`: 設定x軸標籤。
11. `plt.ylabel("Heads ratio")`: 設定y軸標籤。
12. `plt.legend(loc="lower right")`: 設定圖例。
13. `plt.axis([0, 10000, 0.42, 0.58])`: 設定軸範圍。
14. `plt.grid()`: 顯示網格。
15. `save_fig("law_of_large_numbers_plot")`: 儲存圖表。
16. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 模擬大數法則。
- **潛在問題**: 隨機性可能導致變異。
- **最佳使用情境**: 示範統計收斂。

讓我們建立一個投票分類器：

```

---

[來源: ch07] 。

**🎯 重點摘要:**

- **核心功能**: 模擬大數法則。
- **潛在問題**: 隨機性可能導致變異。
- **最佳使用情境**: 示範統計收斂。

讓我們建立一個投票分類器：

```

python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

---

[來源: ch07] LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

---

[來源: ch07] se=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)

---

[來源: ch07] er(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)

```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import make_moons`: 匯入資料集生成器。
2. `from sklearn.ensemble import RandomForestClassifier, VotingClassifier`: 匯入分類器。
3. `from sklearn.linear_model import LogisticRegression`: 匯入邏輯回歸。
4. `from sklearn.model_selection import train_test_split`: 匯入資料分割。
5. `from sklearn.svm import SVC`: 匯入支援向量機。
6. `X, y = make_moons(n_samples=500, noise=0.30, random_state=42)`: 生成資料。
7. `X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)`: 分割資料。
8. `voting_clf = VotingClassifier(...)`: 建立投票分類器。
9. `voting_clf.fit(X_train, y_train)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 建立並訓練投票分類器。
- **潛在問題**: 模型可能過擬合。
- **最佳使用情境**: 結合多個分類器。

```

---

[來源: ch07] ain)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 建立並訓練投票分類器。
- **潛在問題**: 模型可能過擬合。
- **最佳使用情境**: 結合多個分類器。

```

python
print(voting_clf.estimators_) # list of all base classifiers
print(voting_clf.named_estimators_) # dictionary of all base classifiers with their names
print(voting_clf.named_estimators_.items()) # items view of the dictionary

---

[來源: ch07] ssifiers with their names
print(voting_clf.named_estimators_.items()) # items view of the dictionary

```

**✅ 程式碼逐行解析：**

1. `print(voting_clf.estimators_)`: 列印基礎分類器列表。
2. `print(voting_clf.named_estimators_)`: 列印具名分類器字典。
3. `print(voting_clf.named_estimators_.items())`: 列印字典項目。

**🎯 重點摘要:**

- **核心功能**: 檢查分類器結構。
- **潛在問題**: 無。
- **最佳使用情境**: 除錯與檢查。

```

python

---

[來源: ch07] # Evaluate each base classifier
for name, clf in voting_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test, y_test))

```

**✅ 程式碼逐行解析：**

1. `for name, clf in voting_clf.named_estimators_.items()`: 遍歷具名分類器。
2. `print(name, "=", clf.score(X_test, y_test))`: 列印每個分類器的準確度。

**🎯 重點摘要:**

- **核心功能**: 評估個別分類器效能。
- **潛在問題**: 測試資料洩漏。
- **最佳使用情境**: 比較模型效能。

```

---

[來源: ch07] )`: 列印每個分類器的準確度。

**🎯 重點摘要:**

- **核心功能**: 評估個別分類器效能。
- **潛在問題**: 測試資料洩漏。
- **最佳使用情境**: 比較模型效能。

```

python
X_test[:1] # the first instance in the test set

```

**✅ 程式碼逐行解析：**

1. `X_test[:1]`: 取得測試集第一個樣本。

**🎯 重點摘要:**

- **核心功能**: 檢查樣本資料。
- **潛在問題**: 無。
- **最佳使用情境**: 資料檢查。

```

python
voting_clf.predict(X_test[:1]) # the ensemble's prediction for that instance

---

[來源: ch07] 境**: 資料檢查。

```

python
voting_clf.predict(X_test[:1]) # the ensemble's prediction for that instance

```

**✅ 程式碼逐行解析：**

1. `voting_clf.predict(X_test[:1])`: 預測第一個樣本。

**🎯 重點摘要:**

- **核心功能**: 集成預測。
- **潛在問題**: 單一樣本預測。
- **最佳使用情境**: 測試預測。

使用個別分類器對測試集進行預測：

```

python
[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]

---

[來源: ch07] **: 測試預測。

使用個別分類器對測試集進行預測：

```

python
[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]

```

**✅ 程式碼逐行解析：**

1. `[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]`: 每個基礎分類器的預測。

**🎯 重點摘要:**

- **核心功能**: 比較個別預測。
- **潛在問題**: 無。
- **最佳使用情境**: 理解投票機制。

```

python
voting_clf.score(X_test, y_test)

---

[來源: ch07] **核心功能**: 比較個別預測。
- **潛在問題**: 無。
- **最佳使用情境**: 理解投票機制。

```

python
voting_clf.score(X_test, y_test)

```

**✅ 程式碼逐行解析：**

1. `voting_clf.score(X_test, y_test)`: 計算集成準確度。

**🎯 重點摘要:**

- **核心功能**: 評估集成效能。
- **潛在問題**: 過擬合。
- **最佳使用情境**: 模型評估。

現在讓我們使用軟投票：

```

---

[來源: ch07] `: 計算集成準確度。

**🎯 重點摘要:**

- **核心功能**: 評估集成效能。
- **潛在問題**: 過擬合。
- **最佳使用情境**: 模型評估。

現在讓我們使用軟投票：

```

python
voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)

---

[來源: ch07] timators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)

```

**✅ 程式碼逐行解析：**

1. `voting_clf.voting = "soft"`: 設定為軟投票。
2. `voting_clf.named_estimators["svc"].probability = True`: 啟用SVM機率預測。
3. `voting_clf.fit(X_train, y_train)`: 重新訓練。
4. `voting_clf.score(X_test, y_test)`: 評估準確度。

**🎯 重點摘要:**

- **核心功能**: 實作軟投票。
- **潛在問題**: SVM機率計算成本高。
- **最佳使用情境**: 需要機率預測時。

```

---

[來源: ch07] _test)`: 評估準確度。

**🎯 重點摘要:**

- **核心功能**: 實作軟投票。
- **潛在問題**: SVM機率計算成本高。
- **最佳使用情境**: 需要機率預測時。

```

python
voting_clf
```

**✅ 程式碼逐行解析：**

1. `voting_clf`: 顯示分類器物件。

**🎯 重點摘要:**

- **核心功能**: 檢查分類器配置。
- **潛在問題**: 無。
- **最佳使用情境**: 除錯。

---

---

[來源: ch07] ## 裝袋與粘貼 (Bagging and Pasting)

裝袋(Bootstrap Aggregating,簡稱Bagging)與粘貼(Pasting)是集成學習技術，通過在訓練資料的隨機子集上訓練相同基礎估計器的多個實例來提升模型穩定性和準確性。裝袋使用替換抽樣(replacement sampling)，允許實例重複，而粘貼使用不替換抽樣。這隨機性減少過擬合和變異，使集成比個別模型更穩健。在Scikit-Learn中，這些方法通過`BaggingClassifier`和`BaggingRegressor`類實作。

---

[來源: ch07] ### Scikit-Learn中的裝袋與粘貼

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import BaggingClassifier`: 匯入裝袋分類器。
2. `from sklearn.tree import DecisionTreeClassifier`: 匯入決策樹。
3. `bag_clf = BaggingClassifier(...)`: 建立裝袋分類器。
4. `bag_clf.fit(X_train, y_train)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 實作裝袋集成。
- **潛在問題**: 計算成本高。
- **最佳使用情境**: 減少過擬合。

```python

---

[來源: ch07] # extra code – this cell generates and saves Figure 7–5

def plot_decision_boundary(clf, X, y, alpha=1.0):
    """Plot the decision boundary of a classifier.

This function visualizes the decision boundary of a given classifier by
    creating a meshgrid over the feature space, predicting class labels for
    each point, and plotting filled contours and scatter points for the data.

---

[來源: ch07] cting class labels for
    each point, and plotting filled contours and scatter points for the data.

Args:
        clf: A trained classifier with a predict method.
        X: array-like of shape (n_samples, 2), feature matrix.
        y: array-like of shape (n_samples,), target labels.
        alpha: float, optional, default=1.0
            Transparency level for the contour plots.
    """
    ax

---

[來源: ch07] a: float, optional, default=1.0
            Transparency level for the contour plots.
    """
    axes=[-1.5, 2.4, -1, 1.5]
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

---

[來源: ch07] ], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

plt.contourf(x1, x2, y_pred, alpha=0.3 * alpha, cmap='Wistia')
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8 * alpha)
    colors = ["#78785c", "#c47b27"]
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[i

---

[來源: ch07] dx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

---

[來源: ch07] _2$", rotation=0)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf, X_train, y_train)
plt.title("Decision Tree")
plt.sca(axes[1])
plot_decision_boundary(bag_clf, X_train, y_train)
plt.title("Decision Trees with Bagging")
plt.ylabel("")
save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()
```

---

[來源: ch07] with Bagging")
plt.ylabel("")
save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] lt.ylabel("")
save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `def plot_decision_boundary(...)`: 定義繪圖函數。
2. `axes=[-1.5, 2.4, -1, 1.5]`: 設定軸範圍。
3. `x1, x2 = np.meshgrid(...)`: 建立網格。
4. `X_new = np.c_[x1.ravel(), x2.ravel()]`: 建立預測點。
5. `y_pred = clf.predict(X_new).reshape(x1.shape)`: 預測。
6. `plt.contourf(...)`: 繪製填充輪廓。
7. `plt.contour(...)`: 繪製輪廓線。
8. `fo

---

[來源: ch07] (X_new).reshape(x1.shape)`: 預測。
6. `plt.contourf(...)`: 繪製填充輪廓。
7. `plt.contour(...)`: 繪製輪廓線。
8. `for idx in (0, 1): plt.plot(...)`: 繪製資料點。
9. `plt.axis(axes)`: 設定軸。
10. `plt.xlabel(r"$x_1$")`: 設定x軸標籤。
11. `plt.ylabel(r"$x_2$", rotation=0)`: 設定y軸標籤。
12. `tree_clf = DecisionTreeClassifier(random_state=42)`: 建立單一決策樹。
13. `tree_clf.fit(X_train, y_train)`: 訓練決策樹。
14. `fig, axes = plt.subplots(ncols=2,

---

[來源: ch07] e=42)`: 建立單一決策樹。
13. `tree_clf.fit(X_train, y_train)`: 訓練決策樹。
14. `fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)`: 建立子圖。
15. `plt.sca(axes[0])`: 選擇第一個子圖。
16. `plot_decision_boundary(tree_clf, X_train, y_train)`: 繪製決策樹邊界。
17. `plt.title("Decision Tree")`: 設定標題。
18. `plt.sca(axes[1])`: 選擇第二個子圖。
19. `plot_decision_boundary(bag_clf, X_train, y_train)`: 繪製裝袋邊界。
20. `plt.title("Decisio

---

[來源: ch07] `: 選擇第二個子圖。
19. `plot_decision_boundary(bag_clf, X_train, y_train)`: 繪製裝袋邊界。
20. `plt.title("Decision Trees with Bagging")`: 設定標題。
21. `plt.ylabel("")`: 清除y軸標籤。
22. `save_fig("decision_tree_without_and_with_bagging_plot")`: 儲存圖表。
23. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 可視化決策邊界比較。
- **潛在問題**: 計算密集。
- **最佳使用情境**: 比較模型差異。

---

[來源: ch07] ### 袋外評估

袋外（Out-of-Bag）評估是裝袋集成中的技術，用於在沒有單獨驗證集的情況下評估模型效能。通過利用每個引導樣本中留出的實例，它提供對泛化誤差的無偏估計。

```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            oob_score=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] ue, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_
```

**✅ 程式碼逐行解析：**

1. `bag_clf = BaggingClassifier(..., oob_score=True, ...)`: 啟用OOB評估。
2. `bag_clf.fit(X_train, y_train)`: 訓練分類器。
3. `bag_clf.oob_score_`: 取得OOB分數。

**🎯 重點摘要:**

- **核心功能**: 無需額外驗證集的評估。
- **潛在問題**: 僅適用於裝袋。
- **最佳使用情境**: 資源有限時。

---

[來源: ch07] oob_score_`: 取得OOB分數。

**🎯 重點摘要:**

- **核心功能**: 無需額外驗證集的評估。
- **潛在問題**: 僅適用於裝袋。
- **最佳使用情境**: 資源有限時。

```python
bag_clf.oob_decision_function_[:3]  # probas for the first 3 instances
```

**✅ 程式碼逐行解析：**

1. `bag_clf.oob_decision_function_[:3]`: 取得前3個樣本的OOB決策函數。

**🎯 重點摘要:**

- **核心功能**: OOB機率估計。
- **潛在問題**: 無。
- **最佳使用情境**: 檢查預測信心。

然後，我們可以評估OOB準確度：

---

[來源: ch07] 本的OOB決策函數。

**🎯 重點摘要:**

- **核心功能**: OOB機率估計。
- **潛在問題**: 無。
- **最佳使用情境**: 檢查預測信心。

然後，我們可以評估OOB準確度：

```python
from sklearn.metrics import accuracy_score

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.metrics import accuracy_score`: 匯入準確度函數。
2. `y_pred = bag_clf.predict(X_test)`: 預測測試集。
3. `accuracy_score(y_test, y_pred)`: 計算準確度。

---

[來源: ch07] : 匯入準確度函數。
2. `y_pred = bag_clf.predict(X_test)`: 預測測試集。
3. `accuracy_score(y_test, y_pred)`: 計算準確度。

**🎯 重點摘要:**

- **核心功能**: 評估測試準確度。
- **潛在問題**: 過擬合。
- **最佳使用情境**: 最終模型評估。

---

---

[來源: ch07] ## 隨機森林 (Random Forests)

隨機森林是集成學習技術，通過建構多個決策樹並組合其預測來建立更穩健和準確的模型。通過在特徵選擇和引導樣本中引入隨機性，它們減少過擬合並提升泛化優於單一決策樹。本節示範如何在Scikit-Learn中實作和使用隨機森林。

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

---

[來源: ch07] n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import RandomForestClassifier`: 匯入隨機森林。
2. `rnd_clf = RandomForestClassifier(...)`: 建立分類器。
3. `rnd_clf.fit(X_train, y_train)`: 訓練分類器。
4. `y_pred_rf = rnd_clf.predict(X_test)`: 預測測試集。

**🎯 重點摘要:**

- **核心功能**: 實作隨機森林。
- **潛在問題**: 計算成本。
- **最佳使用情境**: 高維資料分類。

---

[來源: ch07] f.predict(X_test)`: 預測測試集。

**🎯 重點摘要:**

- **核心功能**: 實作隨機森林。
- **潛在問題**: 計算成本。
- **最佳使用情境**: 高維資料分類。

隨機森林等同於裝袋決策樹：

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] es="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)
```

**✅ 程式碼逐行解析：**

1. `bag_clf = BaggingClassifier(...)`: 建立等效裝袋分類器。
2. `DecisionTreeClassifier(max_features="sqrt", ...)`: 限制特徵數量。

**🎯 重點摘要:**

- **核心功能**: 模擬隨機森林行為。
- **潛在問題**: 手動配置。
- **最佳使用情境**: 理解隨機森林機制。

```python

---

[來源: ch07] # extra code – verifies that the predictions are identical
bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)
np.all(y_pred_bag == y_pred_rf)  # same predictions
```

**✅ 程式碼逐行解析：**

1. `bag_clf.fit(X_train, y_train)`: 訓練裝袋分類器。
2. `y_pred_bag = bag_clf.predict(X_test)`: 預測。
3. `np.all(y_pred_bag == y_pred_rf)`: 檢查預測是否相同。

**🎯 重點摘要:**

---

[來源: ch07] red_bag = bag_clf.predict(X_test)`: 預測。
3. `np.all(y_pred_bag == y_pred_rf)`: 檢查預測是否相同。

**🎯 重點摘要:**

- **核心功能**: 驗證等效性。
- **潛在問題**: 無。
- **最佳使用情境**: 測試配置。

---

[來源: ch07] ### 特徵重要性

機器學習模型中的特徵重要性，特別是在集成方法如隨機森林中，量化每個特徵對模型預測的貢獻程度。通過分析特徵如何在決策樹中減少雜質（如基尼雜質），我們可以識別最有影響力的變數，有助於特徵選擇、模型解釋和資料理解。本節示範如何使用Scikit-Learn計算和可視化特徵重要性。

---

[來源: ch07] 模型預測的貢獻程度。通過分析特徵如何在決策樹中減少雜質（如基尼雜質），我們可以識別最有影響力的變數，有助於特徵選擇、模型解釋和資料理解。本節示範如何使用Scikit-Learn計算和可視化特徵重要性。

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(f'importance: {round(score, 2)}, feature: {name}')
```

---

[來源: ch07] e_importances_, iris.data.columns):
    print(f'importance: {round(score, 2)}, feature: {name}')
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] iris.data.columns):
    print(f'importance: {round(score, 2)}, feature: {name}')
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import load_iris`: 匯入鳶尾花資料集。
2. `iris = load_iris(as_frame=True)`: 載入資料。
3. `rnd_clf = RandomForestClassifier(...)`: 建立隨機森林。
4. `rnd_clf.fit(iris.data, iris.target)`: 訓練分類器。
5. `for score, name in zip(...)`: 遍歷特徵重要性。
6. `print(f'importance: {round(score, 2)}, feature: {name}')`: 列印重要性。

---

[來源: ch07] e, name in zip(...)`: 遍歷特徵重要性。
6. `print(f'importance: {round(score, 2)}, feature: {name}')`: 列印重要性。

**🎯 重點摘要:**

- **核心功能**: 計算特徵重要性。
- **潛在問題**: 相對而非絕對重要性。
- **最佳使用情境**: 特徵選擇。

隨機森林中的特徵重要性通過測量每個特徵在森林的所有樹中減少雜質的程度來計算。**導致更大雜質減少的特徵被認為更重要。**這提供有價值的見解，了解哪些特徵對模型的預測最有影響，並有助於特徵選擇和理解資料集。

接下來，讓我們可視化特徵重要性

此單元在MNIST影像上訓練隨機森林（平坦化為784特徵），計算特徵重要性，並將其作為28×28熱圖可視化：

- 目的  
    - 顯示隨機森林認為哪些像素（特徵）最重要進行分類。

---

[來源: ch07] 特徵重要性

此單元在MNIST影像上訓練隨機森林（平坦化為784特徵），計算特徵重要性，並將其作為28×28熱圖可視化：

- 目的  
    - 顯示隨機森林認為哪些像素（特徵）最重要進行分類。

- 程式碼所做的事  
    - 在MNIST資料上訓練RandomForestClassifier（100棵樹）。  
    - 讀取`rnd_clf.feature_importances_`並重新塑形為28×28以匹配影像佈局。  
        - 每個像素的重要性反映它在所有樹中減少雜質的貢獻程度。
    - 使用`plt.imshow(..., cmap="hot")`顯示重要性熱圖。更亮的像素=更重要。  
    - 新增色條與標籤「Not important」→「Very important」並使用`save_fig()`儲存高解析度影像。

---

[來源: ch07] ap="hot")`顯示重要性熱圖。更亮的像素=更重要。  
    - 新增色條與標籤「Not important」→「Very important」並使用`save_fig()`儲存高解析度影像。

- 解釋  
    - 更高強度的區域表示對模型決策貢獻最大的像素（例如，數字的筆畫和邊緣）。這有助於理解模型學到了什麼以及它在區分數字時關注的位置。

```python

---

[來源: ch07] # extra code – this cell generates and saves Figure 7–6

from sklearn.datasets import fetch_openml

X_mnist, y_mnist = fetch_openml('mnist_784', return_X_y=True, as_frame=False,
                                parser='auto')

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(X_mnist, y_mnist)

---

[來源: ch07] )

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(X_mnist, y_mnist)

heatmap_image = rnd_clf.feature_importances_.reshape(28, 28)
plt.imshow(heatmap_image, cmap="hot")
cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(),
                           rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'], fontsize=14)
plt.axis("off")
save_fig("mnist_feature_importance_plot")
plt.show()
```

---

[來源: ch07] y important'], fontsize=14)
plt.axis("off")
save_fig("mnist_feature_importance_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] ontsize=14)
plt.axis("off")
save_fig("mnist_feature_importance_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import fetch_openml`: 匯入MNIST資料集。
2. `X_mnist, y_mnist = fetch_openml(...)`: 載入資料。
3. `rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)`: 建立隨機森林。
4. `rnd_clf.fit(X_mnist, y_mnist)`: 訓練分類器。
5. `heatmap_image = rnd_clf.feature_importances_.reshape(28, 28)

---

[來源: ch07] _clf.fit(X_mnist, y_mnist)`: 訓練分類器。
5. `heatmap_image = rnd_clf.feature_importances_.reshape(28, 28)`: 重塑重要性為影像。
6. `plt.imshow(heatmap_image, cmap="hot")`: 顯示熱圖。
7. `cbar = plt.colorbar(...)`: 新增色條。
8. `cbar.ax.set_yticklabels(['Not important', 'Very important'], fontsize=14)`: 設定色條標籤。
9. `plt.axis("off")`: 隱藏軸。
10. `save_fig("mnist_feature_importance_plot")`: 儲存圖表。
11. `plt.show()`: 顯示圖表。

---

[來源: ch07] lt.axis("off")`: 隱藏軸。
10. `save_fig("mnist_feature_importance_plot")`: 儲存圖表。
11. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 可視化像素重要性。
- **潛在問題**: 高維資料。
- **最佳使用情境**: 影像特徵分析。

---

---

[來源: ch07] ## 提升 (Boosting)

提升是集成學習技術，依次訓練弱學習器，其中每個後續模型關注前一個的錯誤。這方法通常導致強預測效能，因為它結合簡單模型成為強集成。常見提升演算法包括AdaBoost和梯度提升。

---

[來源: ch07] ### AdaBoost

**AdaBoost（Adaptive Boosting）**是集成學習技術，結合多個弱分類器以建立強分類器。它通過依次訓練弱學習器運作，其中每個學習器*更關注前一個誤分類的實例*。最終預測通過結合所有弱學習器的加權輸出進行，導致提升準確度和穩健性。

```python

---

[來源: ch07] # extra code – this cell generates and saves Figure 7–8

m = len(X_train)

---

[來源: ch07] fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
for subplot, learning_rate in ((0, 1), (1, 0.5)):
    sample_weights = np.ones(m) / m # initialize sample weights
    plt.sca(axes[subplot])
    for i in range(5): # 5 boosting rounds
        # train weak classifier with current sample weights
        svm_clf = SVC(C=0.2, gamma=0.6, random_state=42)
        svm_clf.fit(X_train, y_trai

---

[來源: ch07] weights
        svm_clf = SVC(C=0.2, gamma=0.6, random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
        y_pred = svm_clf.predict(X_train)

---

[來源: ch07] lf.fit(X_train, y_train, sample_weight=sample_weights * m)
        y_pred = svm_clf.predict(X_train)

# compute weighted error rate
        error_weights = sample_weights[y_pred != y_train].sum()
        r = error_weights / sample_weights.sum()  # equation 7-1
        # compute classifier weight
        alpha = learning_rate * np.log((1 - r) / r)  # equation 7-2
        # update sample weights
   

---

[來源: ch07]      alpha = learning_rate * np.log((1 - r) / r)  # equation 7-2
        # update sample weights
        sample_weights[y_pred != y_train] *= np.exp(alpha)  # equation 7-3
        sample_weights /= sample_weights.sum()  # normalization step

---

[來源: ch07] = np.exp(alpha)  # equation 7-3
        sample_weights /= sample_weights.sum()  # normalization step

plot_decision_boundary(svm_clf, X_train, y_train, alpha=0.4)
        plt.title(f"learning_rate = {learning_rate}")
    if subplot == 0:
        plt.text(-0.75, -0.95, "1", fontsize=16)
        plt.text(-1.05, -0.95, "2", fontsize=16)
        plt.text(1.0, -0.95, "3", fontsize=16)
        plt.text(

---

[來源: ch07] ext(-1.05, -0.95, "2", fontsize=16)
        plt.text(1.0, -0.95, "3", fontsize=16)
        plt.text(-1.45, -0.5, "4", fontsize=16)
        plt.text(1.36,  -0.95, "5", fontsize=16)
    else:
        plt.ylabel("")

save_fig("boosting_plot")
plt.show()

---

[來源: ch07] 36,  -0.95, "5", fontsize=16)
    else:
        plt.ylabel("")

save_fig("boosting_plot")
plt.show()

```

**✅ 程式碼逐行解析：**

1. `m = len(X_train)`: 取得訓練樣本數。
2. `fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)`: 建立子圖。
3. `for subplot, learning_rate in ((0, 1), (1, 0.5))`: 遍歷學習率。
4. `sample_weights = np.ones(m) / m`: 初始化樣本權重。
5. `plt.sca(axes[subplot])`: 選擇子圖。
6. `for i in range(5)`: 5輪提升。
7. `svm_clf = SVC(...)`: 建立SVM分類器。
8. `svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)`: 訓練SVM。
9. `y_pred = svm_clf.predict(X_train)`: 預測訓練集。
10. `error_weights = sample_weights[y_pred != y_train].sum()`: 計算加權錯誤。
11. `r = error_weights / sample_weights.sum()`: 計算錯誤率。
12. `alpha = learning_rate * np.log((1 - r) / r)`: 計算分類器權重。
13. `sample_weights[y_pred != y_train] *= np.exp(alpha)`: 更新樣本權重。
14. `sample_weights /= sample_weights.sum()`: 正規化權重。
15. `plot_decision_boundary(svm_clf, X_train, y_train, alpha=0.4)`: 繪製決策邊界。
16. `plt.title(f"learning_rate = {learning_rate}")`: 設定標題。
17. `if subplot == 0: plt.text(...)`: 新增文字標籤。
18. `else: plt.ylabel("")`: 清除y軸標籤。
19. `save_fig("boosting_plot")`: 儲存圖表。
20. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 手動實作AdaBoost。
- **潛在問題**: 複雜且容易出錯。
- **最佳使用情境**: 理解AdaBoost機制。

```

---

[來源: ch07] 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 手動實作AdaBoost。
- **潛在問題**: 複雜且容易出錯。
- **最佳使用情境**: 理解AdaBoost機制。

```

python
plt.figure()
plt.plot(sample_weights)
plt.title("Sample Weights")
plt.show()

---

[來源: ch07] daBoost機制。

```

python
plt.figure()
plt.plot(sample_weights)
plt.title("Sample Weights")
plt.show()

```

**✅ 程式碼逐行解析：**

1. `plt.figure()`: 建立新圖表。
2. `plt.plot(sample_weights)`: 繪製樣本權重。
3. `plt.title("Sample Weights")`: 設定標題。
4. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 可視化權重變化。
- **潛在問題**: 無。
- **最佳使用情境**: 檢查提升過程。

```

python
from sklearn.ensemble import AdaBoostClassifier

---

[來源: ch07] 化。
- **潛在問題**: 無。
- **最佳使用情境**: 檢查提升過程。

```

python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30,
    learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

---

[來源: ch07] max_depth=1), n_estimators=30,
    learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import AdaBoostClassifier`: 匯入AdaBoost。
2. `ada_clf = AdaBoostClassifier(...)`: 建立AdaBoost分類器。
3. `ada_clf.fit(X_train, y_train)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 使用Scikit-Learn實作AdaBoost。
- **潛在問題**: 過擬合風險。
- **最佳使用情境**: 簡單分類任務。

```

python

---

[來源: ch07] # extra code – in case you're curious to see what the decision boundary

---

[來源: ch07] #             looks like for the AdaBoost classifier
plot_decision_boundary(ada_clf, X_train, y_train)

```

**✅ 程式碼逐行解析：**

1. `plot_decision_boundary(ada_clf, X_train, y_train)`: 繪製AdaBoost決策邊界。

**🎯 重點摘要:**

- **核心功能**: 可視化AdaBoost邊界。
- **潛在問題**: 無。
- **最佳使用情境**: 比較模型。

同樣地，我們可以使用SVM作為AdaBoost的基礎估計器：

```

---

[來源: ch07] - **核心功能**: 可視化AdaBoost邊界。
- **潛在問題**: 無。
- **最佳使用情境**: 比較模型。

同樣地，我們可以使用SVM作為AdaBoost的基礎估計器：

```

python
svm_ada_clf = AdaBoostClassifier(
    SVC(probability=True), n_estimators=30,
    learning_rate=0.5, random_state=42)
svm_ada_clf.fit(X_train, y_train)
plot_decision_boundary(svm_ada_clf, X_train, y_train)
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] _clf.fit(X_train, y_train)
plot_decision_boundary(svm_ada_clf, X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `svm_ada_clf = AdaBoostClassifier(SVC(probability=True), ...)`: 使用SVM作為基礎估計器。
2. `svm_ada_clf.fit(X_train, y_train)`: 訓練分類器。
3. `plot_decision_boundary(svm_ada_clf, X_train, y_train)`: 繪製決策邊界。

**🎯 重點摘要:**

- **核心功能**: SVM與AdaBoost結合。
- **潛在問題**: 計算成本高。
- **最佳使用情境**: 非線性分類。

---

[來源: ch07] y_train)`: 繪製決策邊界。

**🎯 重點摘要:**

- **核心功能**: SVM與AdaBoost結合。
- **潛在問題**: 計算成本高。
- **最佳使用情境**: 非線性分類。

使用SVM作為AdaBoost中的基礎估計器時，重要的是設定`probability=True`以啟用機率估計，這在提升過程中使用。

---

[來源: ch07] ### 梯度提升 (Gradient Boosting)

梯度提升是集成學習技術，依次建構模型，其中*每個新模型修正前一個集成的錯誤（殘差）*。通過迭代擬合弱學習器到殘差，它最小化可微損失函數，通常導致高度準確的預測。本節示範手動實作和使用Scikit-Learn的GradientBoostingRegressor。

讓我們建立一個簡單的二次資料集並擬合`DecisionTreeRegressor`：

---

[來源: ch07] 準確的預測。本節示範手動實作和使用Scikit-Learn的GradientBoostingRegressor。

讓我們建立一個簡單的二次資料集並擬合`DecisionTreeRegressor`：

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5 # 100 instances in [-0.5, 0.5]
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)  # y = 3x² + Gaussian noise

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
```

---

[來源: ch07] ssian noise

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] e_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `import numpy as np`: 匯入NumPy。
2. `from sklearn.tree import DecisionTreeRegressor`: 匯入決策樹回歸。
3. `np.random.seed(42)`: 設定隨機種子。
4. `X = np.random.rand(100, 1) - 0.5`: 生成特徵。
5. `y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)`: 生成目標。
6. `tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)`: 建立回歸器。
7. `tree_reg1.fit(X, y)`: 訓練回歸器。

---

[來源: ch07] reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)`: 建立回歸器。
7. `tree_reg1.fit(X, y)`: 訓練回歸器。

**🎯 重點摘要:**

- **核心功能**: 建立二次資料集並訓練第一棵樹。
- **潛在問題**: 資料簡單。
- **最佳使用情境**: 示範梯度提升。

現在讓我們訓練另一個決策樹回歸器在前一個預測器的殘差錯誤上：

```python
y2 = y - tree_reg1.predict(X) # residuals
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] _reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)
```

**✅ 程式碼逐行解析：**

1. `y2 = y - tree_reg1.predict(X)`: 計算殘差。
2. `tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)`: 建立第二棵樹。
3. `tree_reg2.fit(X, y2)`: 訓練第二棵樹。

**🎯 重點摘要:**

- **核心功能**: 訓練第二棵樹於殘差上。
- **潛在問題**: 無。
- **最佳使用情境**: 梯度提升步驟。

---

[來源: ch07] reg2.fit(X, y2)`: 訓練第二棵樹。

**🎯 重點摘要:**

- **核心功能**: 訓練第二棵樹於殘差上。
- **潛在問題**: 無。
- **最佳使用情境**: 梯度提升步驟。

```python
y3 = y2 - tree_reg2.predict(X) # residuals
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] _reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)
```

**✅ 程式碼逐行解析：**

1. `y3 = y2 - tree_reg2.predict(X)`: 計算新殘差。
2. `tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)`: 建立第三棵樹。
3. `tree_reg3.fit(X, y3)`: 訓練第三棵樹。

**🎯 重點摘要:**

- **核心功能**: 繼續梯度提升過程。
- **潛在問題**: 無。
- **最佳使用情境**: 多輪提升。

---

[來源: ch07] ree_reg3.fit(X, y3)`: 訓練第三棵樹。

**🎯 重點摘要:**

- **核心功能**: 繼續梯度提升過程。
- **潛在問題**: 無。
- **最佳使用情境**: 多輪提升。

```python
X_new = np.array([[-0.4], [0.], [0.5]])
sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

**✅ 程式碼逐行解析：**

1. `X_new = np.array([[-0.4], [0.], [0.5]])`: 建立新樣本。
2. `sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))`: 總和所有樹的預測。

**🎯 重點摘要:**

---

[來源: ch07] 2. `sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))`: 總和所有樹的預測。

**🎯 重點摘要:**

- **核心功能**: 集成預測。
- **潛在問題**: 無。
- **最佳使用情境**: 測試集成。

```python
y_new = 3 * X_new ** 2
```

**✅ 程式碼逐行解析：**

1. `y_new = 3 * X_new ** 2`: 計算真實值。

**🎯 重點摘要:**

- **核心功能**: 計算真實二次函數值。
- **潛在問題**: 無。
- **最佳使用情境**: 比較預測與真實值。

```python

---

[來源: ch07] # extra code – this cell generates and saves Figure 7–9

---

[來源: ch07] def plot_predictions(regressors, X, y, axes, style,
                     label=None, data_style="b.", data_label=None):
    """Plot predictions of an ensemble of regressors along with the data.
    Args:
        regressors: list of regressors in the ensemble
        X: array-like of shape (n_samples, n_features), feature matrix
        y: array-like of shape (n_samples,), target values
        axe

---

[來源: ch07] , n_features), feature matrix
        y: array-like of shape (n_samples,), target values
        axes: list of four floats, [x_min, x_max, y_min, y_max] for the plot
        style: string, style for the prediction line
        label: string, label for the prediction line
        data_style: string, style for the data points
        data_label: string, label for the data points
    """
    x1 = np.

---

[來源: ch07] style for the data points
        data_label: string, label for the data points
    """
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1))
                 for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upp

---

[來源: ch07] x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center")
    plt.axis(axes)

plt.figure(figsize=(11, 11))

plt.subplot(3, 2, 1)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], style="g-",
                 label="$h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$  ", rotation=0)
plt.title("Residuals and tree predictions")

---

[來源: ch07] ta_label="Training set")
plt.ylabel("$y$  ", rotation=0)
plt.title("Residuals and tree predictions")

plt.subplot(3, 2, 2)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                 label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.title("Ensemble predictions")

---

[來源: ch07] label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.title("Ensemble predictions")

plt.subplot(3, 2, 3)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                 label="$h_2(x_1)$", data_style="k+",
                 data_label="Residuals: $y - h_1(x_1)$")
plt.ylabel("$y$  ", rotation=0)

---

[來源: ch07] style="k+",
                 data_label="Residuals: $y - h_1(x_1)$")
plt.ylabel("$y$  ", rotation=0)

plt.subplot(3, 2, 4)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.2, 0.8],
                  style="r-", label="$h(x_1) = h_1(x_1) + h_2(x_1)$")

---

[來源: ch07] , axes=[-0.5, 0.5, -0.2, 0.8],
                  style="r-", label="$h(x_1) = h_1(x_1) + h_2(x_1)$")

plt.subplot(3, 2, 5)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                 label="$h_3(x_1)$", data_style="k+",
                 data_label="Residuals: $y - h_1(x_1) - h_2(x_1)$")
plt.xlabel("$x_1$")
plt.ylabel("$y$  ", rotation=0)

---

[來源: ch07] ta_label="Residuals: $y - h_1(x_1) - h_2(x_1)$")
plt.xlabel("$x_1$")
plt.ylabel("$y$  ", rotation=0)

plt.subplot(3, 2, 6)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y,
                 axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                 label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$")

save_fig("gradient_boosting_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] + h_3(x_1)$")
plt.xlabel("$x_1$")

save_fig("gradient_boosting_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `def plot_predictions(...)`: 定義繪圖函數。
2. `plt.figure(figsize=(11, 11))`: 建立大圖表。
3. `plt.subplot(3, 2, 1)`: 選擇第一個子圖。
4. `plot_predictions([tree_reg1], X, y, ...)`: 繪製第一棵樹。
5. `plt.ylabel("$y$  ", rotation=0)`: 設定y軸標籤。
6. `plt.title("Residuals and tree predictions")`: 設定標題。
7. `plt.subplot(3, 2, 2

---

[來源: ch07] otation=0)`: 設定y軸標籤。
6. `plt.title("Residuals and tree predictions")`: 設定標題。
7. `plt.subplot(3, 2, 2)`: 選擇第二個子圖。
8. `plot_predictions([tree_reg1], X, y, ...)`: 繪製集成預測。
9. `plt.title("Ensemble predictions")`: 設定標題。
10. `plt.subplot(3, 2, 3)`: 選擇第三個子圖。
11. `plot_predictions([tree_reg2], X, y2, ...)`: 繪製第二棵樹於殘差。
12. `plt.subplot(3, 2, 4)`: 選擇第四個子圖。
13. `plot_predictions([tree_reg1, tree_reg2], X, y, 

---

[來源: ch07] 繪製第二棵樹於殘差。
12. `plt.subplot(3, 2, 4)`: 選擇第四個子圖。
13. `plot_predictions([tree_reg1, tree_reg2], X, y, ...)`: 繪製前兩棵樹集成。
14. `plt.subplot(3, 2, 5)`: 選擇第五個子圖。
15. `plot_predictions([tree_reg3], X, y3, ...)`: 繪製第三棵樹於殘差。
16. `plt.xlabel("$x_1$")`: 設定x軸標籤。
17. `plt.subplot(3, 2, 6)`: 選擇第六個子圖。
18. `plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, ...)`: 繪製完整集成。
19. `plt.xlabel("$x_1$")`: 設定x軸標籤。
2

---

[來源: ch07] ictions([tree_reg1, tree_reg2, tree_reg3], X, y, ...)`: 繪製完整集成。
19. `plt.xlabel("$x_1$")`: 設定x軸標籤。
20. `save_fig("gradient_boosting_plot")`: 儲存圖表。
21. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 可視化梯度提升過程。
- **潛在問題**: 圖表複雜。
- **最佳使用情境**: 教育目的。

---

[來源: ch07] #### 梯度提升中的殘差錯誤解釋

在梯度提升中，每個新模型被訓練來預測*前一個集成錯誤的殘差（差異）*。這個迭代過程通過專注於先前模型的錯誤來最小化整體錯誤。

---

[來源: ch07] ##### 範例中的關鍵步驟：
- **資料集建立**：生成二次資料集與`y = 3x² + 高斯雜訊`以模擬非線性關係。
- **第一棵樹（`tree_reg1`）**：擬合原始資料`(X, y)`，擷取初始近似。
- **殘差計算**：`y2 = y - tree_reg1.predict(X)`計算第一棵樹的錯誤（殘差）。
- **第二棵樹（`tree_reg2`）**：訓練於`(X, y2)`以預測這些殘差，提升擬合。
- **第三棵樹（`tree_reg3`）**：進一步精煉通過計算`y3 = y2 - tree_reg2.predict(X)`並擬合於`(X, y3)`。
- **集成預測**：最終預測總和所有樹：`tree_reg1.predict(X_new) + tree_reg2.predict(X_new) + tree_reg3.predict(X_new)`，近似真實二次函數。

---

[來源: ch07] ##### 圖7–9的解釋：
圖表使用3×2網格可視化梯度提升過程：
- **頂列**：顯示個別樹於殘差的預測（左：`tree_reg1`於原始資料；右：前兩棵樹集成）。
- **中列**：`tree_reg2`於`y2`殘差（左），以及前三棵樹集成（右）。
- **底列**：`tree_reg3`於`y3`殘差（左），以及最終集成預測（右）。
- **觀察**：集成逐步減少錯誤，收斂至真實二次曲線。這示範提升如何從弱學習器建構強回歸器通過迭代修正殘差。

現在讓我們嘗試梯度提升回歸器：

---

[來源: ch07] eg3`於`y3`殘差（左），以及最終集成預測（右）。
- **觀察**：集成逐步減少錯誤，收斂至真實二次曲線。這示範提升如何從弱學習器建構強回歸器通過迭代修正殘差。

現在讓我們嘗試梯度提升回歸器：

```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                 learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import GradientBoostingRegressor`: 匯入梯度提升回歸。
2. `gbrt = GradientBoostingRegressor(...)`: 建立回歸器。
3. `gbrt.fit(X, y)`: 訓練回歸器。

**🎯 重點摘要:**

- **核心功能**: 使用Scikit-Learn實作梯度提升。
- **潛在問題**: 學習率高可能過擬合。
- **最佳使用情境**: 回歸任務。

---

[來源: ch07] `: 訓練回歸器。

**🎯 重點摘要:**

- **核心功能**: 使用Scikit-Learn實作梯度提升。
- **潛在問題**: 學習率高可能過擬合。
- **最佳使用情境**: 回歸任務。

梯度提升通過依次新增預測器來建構集成，每個新預測器修正當前集成的錯誤。這個程式碼訓練GradientBoostingRegressor與3棵樹，每棵最大深度2，以及學習率1.0。演算法擬合每個新樹於當前集成的殘差錯誤，逐步改善整體預測。

然後，以下程式碼片段示範另一個Gradient Boosting Regressor模型的實例化和訓練以供比較，具有不同的超參數：

---

[來源: ch07] ##### 程式碼分解
- **實例化**：建立名為`gbrt_best`的實例，具有調校的超參數以提升效能。
- **訓練**：`gbrt_best.fit(X, y)`在特徵矩陣`X`和目標向量`y`上訓練模型。這個方法迭代建構集成，擬合每個樹於先前預測的殘差（錯誤）。

---

[來源: ch07] ##### 關鍵參數
- `max_depth=2`：限制每個決策樹深度為2，防止過擬合通過保持樹淺且訓練快（儘管這可能低估複雜資料）。
- `learning_rate=0.05`：控制每個新樹對集成的貢獻；較低率需要更多樹以收斂但通常導致更好泛化。
- `n_estimators=500`：設定提升階段（樹）數量為500，高於先前範例，允許模型學習更複雜模式但增加計算時間。
- `n_iter_no_change=10`：啟用早期停止；如果驗證分數在10個連續迭代中沒有改善，訓練提前停止以節省時間並防止過擬合。
- `random_state=42`：通過植入隨機數生成器確保可重複結果。

---

[來源: ch07] ##### 運作方式
- 內部使用指定損失函數（預設平方誤差）計算梯度並更新模型。
- 每個新樹訓練於當前集成的殘差，逐步改善預測。

---

[來源: ch07] ##### 陷阱
- `n_iter_no_change`需要驗證集，自動分割10%訓練資料（透過`validation_fraction=0.1`在類別預設）除非覆蓋，這可能影響小資料集效能。
- 確保資料預處理（例如，處理缺失值或縮放特徵）事先進行，因為`X`或`y`格式不當（例如，非數值型別）可能引發內部驗證錯誤。
- 對於較大資料集（n_samples >= 10,000），考慮更快的`HistGradientBoostingRegressor`變體在類別文件提及。

---

[來源: ch07] 當（例如，非數值型別）可能引發內部驗證錯誤。
- 對於較大資料集（n_samples >= 10,000），考慮更快的`HistGradientBoostingRegressor`變體在類別文件提及。

```python
gbrt_best = GradientBoostingRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=500,
    n_iter_no_change=10, random_state=42)
gbrt_best.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `gbrt_best = GradientBoostingRegressor(...)`: 建立調校回歸器。
2. `gbrt_best.fit(X, y)`: 訓練回歸器。

**🎯 重點摘要:**

---

[來源: ch07] `gbrt_best = GradientBoostingRegressor(...)`: 建立調校回歸器。
2. `gbrt_best.fit(X, y)`: 訓練回歸器。

**🎯 重點摘要:**

- **核心功能**: 調校梯度提升參數。
- **潛在問題**: 早期停止可能過早停止。
- **最佳使用情境**: 平衡偏差與變異。

```python
gbrt_best.n_estimators_
```

**✅ 程式碼逐行解析：**

1. `gbrt_best.n_estimators_`: 取得實際使用的估計器數量。

**🎯 重點摘要:**

- **核心功能**: 檢查早期停止效果。
- **潛在問題**: 無。
- **最佳使用情境**: 評估訓練過程。

```python

---

[來源: ch07] # extra code – this cell generates and saves Figure 7–10

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

---

[來源: ch07] ell generates and saves Figure 7–10

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

plt.sca(axes[0])
plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], style="r-",
                 label="Ensemble predictions")
plt.title(f"learning_rate={gbrt.learning_rate}, "
          f"n_estimators={gbrt.n_estimators_}")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)

---

[來源: ch07] "
          f"n_estimators={gbrt.n_estimators_}")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)

plt.sca(axes[1])
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8], style="r-")
plt.title(f"learning_rate={gbrt_best.learning_rate}, "
          f"n_estimators={gbrt_best.n_estimators_}")
plt.xlabel("$x_1$")

save_fig("gbrt_learning_rate_plot")
plt.show()

---

[來源: ch07] tors={gbrt_best.n_estimators_}")
plt.xlabel("$x_1$")

save_fig("gbrt_learning_rate_plot")
plt.show()

```

**✅ 程式碼逐行解析：**

1. `fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)`: 建立子圖。
2. `plt.sca(axes[0])`: 選擇第一個子圖。
3. `plot_predictions([gbrt], X, y, ...)`: 繪製第一個模型。
4. `plt.title(f"learning_rate={gbrt.learning_rate}, n_estimators={gbrt.n_estimators_}")`: 設定標題。
5. `plt.xlabel("$x_1$")`: 設定x軸標籤。
6. `plt.ylabel("$y$", rotation=0)`: 設定y軸標籤。
7. `plt.sca(axes[1])`: 選擇第二個子圖。
8. `plot_predictions([gbrt_best], X, y, ...)`: 繪製第二個模型。
9. `plt.title(f"learning_rate={gbrt_best.learning_rate}, n_estimators={gbrt_best.n_estimators_}")`: 設定標題。
10. `plt.xlabel("$x_1$")`: 設定x軸標籤。
11. `save_fig("gbrt_learning_rate_plot")`: 儲存圖表。
12. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 比較不同學習率的影響。
- **潛在問題**: 無。
- **最佳使用情境**: 超參數調校。

此節示範如何準備加州住房資料集以進行機器學習實驗。程式碼下載資料集如果需要，載入到pandas DataFrame，分割成訓練和測試集，並分離特徵從目標變數（`median_house_value`）。這個資料準備步驟對於使用集成方法評估如裝袋、提升和堆疊等至關重要，以可重複方式在真實世界回歸問題上進行。

```

---

[來源: ch07] 分割成訓練和測試集，並分離特徵從目標變數（`median_house_value`）。這個資料準備步驟對於使用集成方法評估如裝袋、提升和堆疊等至關重要，以可重複方式在真實世界回歸問題上進行。

```

python

---

[來源: ch07] # extra code – at least not in this chapter, it's presented in chapter 2

import pandas as pd
from sklearn.model_selection import train_test_split
import tarfile
import urllib.request

---

[來源: ch07] ndas as pd
from sklearn.model_selection import train_test_split
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        

---

[來源: ch07] com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

---

[來源: ch07] tasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing_labels = train_set["median_house_value"]
housing = train_set.drop("median_house_value", axis=1)

---

[來源: ch07] sing_labels = train_set["median_house_value"]
housing = train_set.drop("median_house_value", axis=1)

```

**✅ 程式碼逐行解析：**

1. `import pandas as pd`: 匯入pandas。
2. `from sklearn.model_selection import train_test_split`: 匯入資料分割。
3. `import tarfile`: 匯入tarfile。
4. `import urllib.request`: 匯入urllib。
5. `def load_housing_data()`: 定義資料載入函數。
6. `tarball_path = Path("datasets/housing.tgz")`: 定義壓縮檔案路徑。
7. `if not tarball_path.is_file()`: 檢查檔案是否存在。
8. `Path("datasets").mkdir(parents=True, exist_ok=True)`: 建立目錄。
9. `urllib.request.urlretrieve(url, tarball_path)`: 下載檔案。
10. `with tarfile.open(tarball_path) as housing_tarball`: 開啟壓縮檔案。
11. `housing_tarball.extractall(path="datasets")`: 解壓縮。
12. `return pd.read_csv(Path("datasets/housing/housing.csv"))`: 載入CSV。
13. `housing = load_housing_data()`: 載入資料。
14. `train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)`: 分割資料。
15. `housing_labels = train_set["median_house_value"]`: 提取標籤。
16. `housing = train_set.drop("median_house_value", axis=1)`: 移除標籤列。

**🎯 重點摘要:**

- **核心功能**: 載入並準備住房資料集。
- **潛在問題**: 網路下載失敗。
- **最佳使用情境**: 資料準備。

```

---

[來源: ch07] axis=1)`: 移除標籤列。

**🎯 重點摘要:**

- **核心功能**: 載入並準備住房資料集。
- **潛在問題**: 網路下載失敗。
- **最佳使用情境**: 資料準備。

```

python
housing.head()

```

**✅ 程式碼逐行解析：**

1. `housing.head()`: 顯示資料前幾行。

**🎯 重點摘要:**

- **核心功能**: 檢查資料結構。
- **潛在問題**: 無。
- **最佳使用情境**: 資料探索。

```

---

[來源: ch07] `housing.head()`: 顯示資料前幾行。

**🎯 重點摘要:**

- **核心功能**: 檢查資料結構。
- **潛在問題**: 無。
- **最佳使用情境**: 資料探索。

```

python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

---

[來源: ch07] learn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough"),
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)
hgb_reg.fit(housing, housing_labels)

---

[來源: ch07] tBoostingRegressor(categorical_features=[0], random_state=42)
)
hgb_reg.fit(housing, housing_labels)

```

**✅ 程式碼逐行解析：**

1. `from sklearn.pipeline import make_pipeline`: 匯入pipeline。
2. `from sklearn.compose import make_column_transformer`: 匯入column transformer。
3. `from sklearn.ensemble import HistGradientBoostingRegressor`: 匯入直方圖梯度提升。
4. `from sklearn.preprocessing import OrdinalEncoder`: 匯入序數編碼器。
5. `hgb_reg = make_pipeline(...)`: 建立pipeline。
6. `hgb_reg.fit(housing, housing_labels)`: 訓練模型。

**🎯 重點摘要:**

- **核心功能**: 建立並訓練直方圖梯度提升回歸器。
- **潛在問題**: 分類特徵處理。
- **最佳使用情境**: 處理混合資料型別。

```

---

[來源: ch07] `: 訓練模型。

**🎯 重點摘要:**

- **核心功能**: 建立並訓練直方圖梯度提升回歸器。
- **潛在問題**: 分類特徵處理。
- **最佳使用情境**: 處理混合資料型別。

```

python

---

[來源: ch07] # extra code – evaluate the RMSE stats for the hgb_reg model

from sklearn.model_selection import cross_val_score

hgb_rmses = -cross_val_score(hgb_reg, housing, housing_labels,
                             scoring="neg_root_mean_squared_error", cv=10)
pd.Series(hgb_rmses).describe()
```

**✅ 程式碼逐行解析：**

---

[來源: ch07] scoring="neg_root_mean_squared_error", cv=10)
pd.Series(hgb_rmses).describe()
```

**✅ 程式碼逐行解析：**

1. `from sklearn.model_selection import cross_val_score`: 匯入交叉驗證。
2. `hgb_rmses = -cross_val_score(...)`: 計算RMSE。
3. `pd.Series(hgb_rmses).describe()`: 描述統計。

**🎯 重點摘要:**

- **核心功能**: 評估模型效能。
- **潛在問題**: 負分數處理。
- **最佳使用情境**: 模型驗證。

---

[來源: ch07] ### 直方圖基礎梯度提升（HistGradientBoostingRegressor）

`HistGradientBoostingRegressor`是梯度提升的高效實作，適用於大型資料集。與傳統梯度提升不同，它將連續特徵離散化為區間（直方圖），顯著加速訓練並減少記憶體使用。這個方法特別適用於具有數萬或更多樣本的資料集。

**關鍵特點：**
- 快速訓練和預測，即使在大型資料集上。
- 原生支援數值和分類特徵。
- 自動處理缺失值。
- 通常以最小調校達成最先進效能。

在Scikit-Learn中，`HistGradientBoostingRegressor`可用作`GradientBoostingRegressor`的即插即用替代品，提供更好可擴充性和效能以進行真實世界機器學習任務。

---

---

[來源: ch07] ## 堆疊 (Stacking)

**堆疊（Stacked Generalization）**是集成學習技術，結合多個模型通過訓練元學習器（混成器）於其預測上。這方法可以提升效能，因為它利用不同模型的優勢並通常優於簡單集成方法如投票或裝袋。本節示範如何在Scikit-Learn中實作堆疊。

---

[來源: ch07] **是集成學習技術，結合多個模型通過訓練元學習器（混成器）於其預測上。這方法可以提升效能，因為它利用不同模型的優勢並通常優於簡單集成方法如投票或裝袋。本節示範如何在Scikit-Learn中實作堆疊。

```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43), # meta-classifier
    cv=5  # number of cross-validation folds
)
stacking_clf.fit(X_train, y_train)
```

---

[來源: ch07] eta-classifier
    cv=5  # number of cross-validation folds
)
stacking_clf.fit(X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import StackingClassifier`: 匯入堆疊分類器。
2. `stacking_clf = StackingClassifier(...)`: 建立堆疊分類器。
3. `stacking_clf.fit(X_train, y_train)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 實作堆疊集成。
- **潛在問題**: 計算成本高。
- **最佳使用情境**: 結合多樣模型。

---

[來源: ch07] _train, y_train)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 實作堆疊集成。
- **潛在問題**: 計算成本高。
- **最佳使用情境**: 結合多樣模型。

```python
stacking_clf.score(X_test, y_test)
```

**✅ 程式碼逐行解析：**

1. `stacking_clf.score(X_test, y_test)`: 評估堆疊分類器。

**🎯 重點摘要:**

- **核心功能**: 測試準確度。
- **潛在問題**: 過擬合。
- **最佳使用情境**: 最終評估。

---

[來源: ch07] ore(X_test, y_test)`: 評估堆疊分類器。

**🎯 重點摘要:**

- **核心功能**: 測試準確度。
- **潛在問題**: 過擬合。
- **最佳使用情境**: 最終評估。

```python
for name, clf in stacking_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test, y_test))
```

**✅ 程式碼逐行解析：**

1. `for name, clf in stacking_clf.named_estimators_.items()`: 遍歷基礎估計器。
2. `print(name, "=", clf.score(X_test, y_test))`: 列印各估計器準確度。

**🎯 重點摘要:**

---

[來源: ch07] mators_.items()`: 遍歷基礎估計器。
2. `print(name, "=", clf.score(X_test, y_test))`: 列印各估計器準確度。

**🎯 重點摘要:**

- **核心功能**: 比較個別效能。
- **潛在問題**: 無。
- **最佳使用情境**: 分析貢獻。

```python
stacking_clf.final_estimator_.n_features_in_ # number of features used by the final estimator
```

**✅ 程式碼逐行解析：**

1. `stacking_clf.final_estimator_.n_features_in_`: 取得最終估計器特徵數。

**🎯 重點摘要:**

---

[來源: ch07] tor
```

**✅ 程式碼逐行解析：**

1. `stacking_clf.final_estimator_.n_features_in_`: 取得最終估計器特徵數。

**🎯 重點摘要:**

- **核心功能**: 檢查元學習器輸入。
- **潛在問題**: 無。
- **最佳使用情境**: 除錯。

---

[來源: ch07] ## 總結與最佳實踐

集成學習是機器學習中的強大技術，通過結合多個模型來提升預測效能與穩健性。本章涵蓋了投票分類器、裝袋與粘貼、隨機森林、提升方法和堆疊等關鍵技術。每種方法都有其優勢，適用於不同情境。

---

[來源: ch07] ## 常見問答 (FAQ)

**Q: 何時使用裝袋而不是提升？**  
A: 裝袋適合減少變異和過擬合，提升適合處理偏差和逐步改善模型。

**Q: 隨機森林的特徵重要性可靠嗎？**  
A: 是的，但它是相對的，適用於特徵選擇而非絕對重要性。

**Q: 如何處理集成中的過擬合？**  
A: 使用交叉驗證、早期停止和適當的正規化參數。

---

[來源: ch07] ## 推薦標籤 (Suggested Hashtags)

#Python #機器學習 #集成學習 #隨機森林 #裝袋 #提升 #堆疊 #ScikitLearn #資料科學 #人工智慧

---

[來源: ch07 | 類型: cheatsheet] # Ch07 速查表：Ensemble Learning & Random Forests

> **核心主旨**：集成多個弱學習器往往勝過單一強模型 —— Random Forest 是最佳預設選擇，GBM 通常更強但需謹慎調參。

---

---

[來源: ch07 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Voting Classifier | 聚合多個不同分類器的預測 | 模型多樣性高時（soft voting 較佳） |
| Bagging | 對資料有放回取樣，訓練多個同型分類器 | Random Forest 的基礎 |
| Pasting | 不放回取樣（樣本不重複） | 比 Bagging 多樣性略差 |
| Random Forest | Bagging + 每次分割只隨機選部分特徵 | 表格資料的首選基準線 |
| Extra-Trees | 極度隨機切割點（不找最佳），訓練更快 | 需要超快速訓練時 |
| AdaBoost | 序列式訓練，每次調高錯誤樣本的權重 | 弱學習器提升 |
| Gradient Boosting | 序列式擬合前一棵樹的殘差 | 表格資料競賽常勝軍（搭配 XGBoost） |
| Stacking | 用 meta-learner 學習如何組合各模型預測 | 競賽最後衝分階段 |
| Out-of-Bag (OOB) | Bagging 中未被取樣的樣本可做驗證 | 免費的驗證集（不需獨立切分） |


---

[來源: ch07 | 類型: cheatsheet] meta-learner 學習如何組合各模型預測 | 競賽最後衝分階段 |
| Out-of-Bag (OOB) | Bagging 中未被取樣的樣本可做驗證 | 免費的驗證集（不需獨立切分） |


---

---

[來源: ch07 | 類型: cheatsheet] | sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `VotingClassifier` | `estimators=[(...)], voting="soft"` | 軟投票集成 |
| `BaggingClassifier` | `estimator`, `n_estimators=100`, `max_samples=0.8` | Bagging 集成 |
| `RandomForestClassifier` | `n_estimators=100`, `max_features="sqrt"`, `oob_score=True` | 隨機森林分類 |
| `RandomForestRegressor` | `n_estimators=200`, `max_leaf_nodes=16` | 隨機森林迴歸 |
| `ExtraTreesClassifier` | `n_estimators=100` | 極端隨機樹（速度快） |
| `AdaBoostClassifier` | `n_estimators=100`, `learning_rate=0.5` | AdaBoost |
| `GradientBoostingClassifier` | `n_estimators=100`, `learning_rate=0.1`, `subsample=0.8` | GBM 分類 |
| `GradientBoostingRegressor` | `max_depth=2`, `n_estimators=300` | GBM 迴歸 |
| `StackingClassifier` | `estimators=[...], final_estimator=...` | 堆疊集成 |
| `.feature_importances_` | – | 隨機森林特徵重要性 |
| `.oob_score_` | – | OOB 評估分數 |


---

[來源: ch07 | 類型: cheatsheet] stimator=...` | 堆疊集成 |
| `.feature_importances_` | – | 隨機森林特徵重要性 |
| `.oob_score_` | – | OOB 評估分數 |


---

---

[來源: ch07 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
from sklearn.ensemble import (VotingClassifier, BaggingClassifier,
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, GradientBoostingClassifier,
    GradientBoostingRegressor, StackingClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

---

[來源: ch07 | 類型: cheatsheet] # 軟投票集成（不同類型模型）
voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(probability=True, random_state=42))
    ],
    voting="soft"  # 用 predict_proba 平均，通常優於 hard voting
)
voting_clf.fit(X_train, y_train)

---

[來源: ch07 | 類型: cheatsheet] # Random Forest（最實用的基準線）
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",   # 每次分割只看 sqrt(n_features) 個特徵
    oob_score=True,        # 免費的 OOB 驗證
    n_jobs=-1,             # 並行使用所有 CPU
    random_state=42
)
rf_clf.fit(X_train, y_train)
print(f"OOB score: {rf_clf.oob_score_:.4f}")

---

[來源: ch07 | 類型: cheatsheet] # 特徵重要性（排序）
importances = rf_clf.feature_importances_
feature_names = X_train.columns.tolist()
sorted_idx = np.argsort(importances)[::-1]
for i in sorted_idx[:10]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

---

[來源: ch07 | 類型: cheatsheet] # Gradient Boosting（調參關鍵：少棵樹 + 低學習率）
gbm_clf = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,  # 學習率越低需要越多樹，但泛化越好
    max_depth=3,         # 淺樹（2-5）是 GBM 的慣例
    subsample=0.8,       # 隨機取 80% 樣本，防 overfitting
    random_state=42
)
gbm_clf.fit(X_train, y_train)

---

[來源: ch07 | 類型: cheatsheet] # Stacking（競賽提分利器）
stacking_clf = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gbm", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5  # 使用 5-fold CV 產生 meta-features
)
stacking_clf.fit(X_train, y_train)
```

---

---

[來源: ch07 | 類型: cheatsheet] ## 4. 常見陷阱

- **Voting 需要 `probability=True`**：`SVC` 預設不輸出機率，soft voting 時必須加 `probability=True`（訓練慢）。
- **GBM learning_rate 與 n_estimators 互相制約**：降低 `learning_rate` 必須同時增加 `n_estimators`；常用配合：`lr=0.05, n_estimators=500`。
- **Random Forest 不需要 Scaling**：基於決策樹，特徵尺度無關。
- **AdaBoost 對 outlier 敏感**：outlier 會被反覆增加權重，影響後續模型。
- **`n_jobs=-1` 別忘加**：RF 和 GBM 原生支援並行，加了訓練速度倍增。

---

---

[來源: ch07 | 類型: cheatsheet] ## 5. 決策指南

```
選哪種集成方法？
├── 快速建立基準線            → RandomForestClassifier (n_estimators=100)
├── 追求最高準確度（表格資料）→ GradientBoosting 或 XGBoost/LightGBM
├── 多種不同類型模型要融合    → VotingClassifier (soft)
└── 競賽最後衝分              → StackingClassifier

GBM 調參優先順序：
1. max_depth (通常 2-5)
2. n_estimators + learning_rate (成反比調整)
3. subsample (0.6-0.9)
4. min_samples_leaf (防止過細分割)
```

---

[來源: ch07 | 類型: handout] # 課程講義：集成學習與隨機森林 (Chapter 07)

「三個臭皮匠，勝過一個諸葛亮。」集成學習（Ensemble Learning）正是這個哲學的機器學習體現：將**許多個弱模型組合**成一個強大的預測器。本章介紹四大集成技術：投票、Bagging、Boosting 和 Stacking，以及它們的理論基礎——為何一群「差不多但互不相同」的模型，組合後可以大幅超越其中任何一個。

---

---

[來源: ch07 | 類型: handout] ### 理論背景

**弱學習器的集體智慧**：設每個分類器的準確率為 51%，且它們的錯誤**相互獨立**。1000 個這樣的分類器多數投票的準確率是多少？

$$P(\text{多數正確}) = \sum_{k=501}^{1000} \binom{1000}{k} (0.51)^k (0.49)^{1000-k} \approx 75\%$$

即使每個分類器只比隨機略好一點，大量集成後效果顯著提升！

**硬投票 (Hard Voting)**：每個分類器投一票，取多數類別。

**軟投票 (Soft Voting)**：取所有分類器**預測機率的平均值**，選機率最高的類別。通常優於硬投票（因為高確信度的預測有更大的發言權）。

$$\hat{y} = \arg\max_k \frac{1}{N} \sum_{i=1}^{N} \hat{p}_{i,k}$$

---

[來源: ch07 | 類型: handout] 率最高的類別。通常優於硬投票（因為高確信度的預測有更大的發言權）。

$$\hat{y} = \arg\max_k \frac{1}{N} \sum_{i=1}^{N} \hat{p}_{i,k}$$

**關鍵前提**：分類器的錯誤需要**多樣化（多樣性）**——犯的錯誤盡量不同。若所有分類器犯同樣的錯誤，集成無效！

---

[來源: ch07 | 類型: handout] ### 核心代碼

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

---

[來源: ch07 | 類型: handout] er, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

---

[來源: ch07 | 類型: handout] # 三種不同類型的分類器（多樣性的來源）
log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
svm_clf = SVC(probability=True, random_state=42)  # 軟投票需要 probability=True

---

[來源: ch07 | 類型: handout] # 軟投票集成
voting_clf = VotingClassifier(
    estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
    voting="soft"
)
voting_clf.fit(X_train, y_train)

---

[來源: ch07 | 類型: handout] # 比較各個分類器和集成的準確率
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    print(f"{clf.__class__.__name__:25s}: {clf.score(X_test, y_test):.4f}")
```

---

[來源: ch07 | 類型: handout] ### 補充練習 1

**理論題：** 軟投票通常優於硬投票，但有一個例外情況：若某個分類器的機率輸出未校正（systematically overconfident），會對軟投票造成什麼影響？可以用什麼方法校正機率輸出？

**實作題：** 分別建立「3 個完全相同的 `LogisticRegression`」和「`LogisticRegression` + `RandomForest` + `SVC`」的硬投票集成，比較兩者的測試集準確率，驗證「多樣性」對集成效果的重要性。

---

---

[來源: ch07 | 類型: handout] ### 理論背景

**Bagging (Bootstrap Aggregating)**：用**有放回抽樣 (Bootstrap)** 從訓練集中抽取子集，訓練同一類型的多個分類器。

**Pasting**：用**無放回抽樣**。

Bagging 的優點：
- 每個基學習器看到的資料不同 → 多樣性
- Bootstrap 抽樣讓每個樣本有約 $1 - (1-1/m)^m \approx 63.2\%$ 的機率被選中；剩下 ~36.8% 的樣本作為天然的**袋外樣本 (OOB)**，可用於驗證而無需額外驗證集

**OOB 評估 (Out-of-Bag Evaluation)**：每個訓練樣本對沒有用到它的那些基學習器進行預測，聚合後得到 OOB 分數，是對泛化誤差的合理估計。

---

[來源: ch07 | 類型: handout] ### 核心代碼

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

---

[來源: ch07 | 類型: handout] # Bagging：500 棵決策樹，各用 100 個樣本（有放回）
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,      # True=Bagging, False=Pasting
    oob_score=True,      # 啟用 OOB 評估（不需要額外驗證集）
    n_jobs=-1,           # 使用所有 CPU 核心
    random_state=42
)
bag_clf.fit(X_train, y_train)

---

[來源: ch07 | 類型: handout] 不需要額外驗證集）
    n_jobs=-1,           # 使用所有 CPU 核心
    random_state=42
)
bag_clf.fit(X_train, y_train)

print(f"OOB 評估準確率: {bag_clf.oob_score_:.4f}")     # 接近測試集準確率
print(f"測試集準確率:   {bag_clf.score(X_test, y_test):.4f}")
print(f"OOB 決策機率（前 3 筆）:\n{bag_clf.oob_decision_function_[:3]}")
```

---

[來源: ch07 | 類型: handout] ### 補充練習 2

**理論題：** 解釋 OOB 評估為何能作為測試集的替代品。若 `n_estimators` 很小（比如 10），OOB 評估還可靠嗎？

**實作題：** 比較 Bagging（`bootstrap=True`）和 Pasting（`bootstrap=False`）在 `make_moons` 上的 OOB 分數和測試集準確率，哪種方法通常表現更好？為什麼？

---

---

[來源: ch07 | 類型: handout] ### 理論背景

**隨機森林 (Random Forest)**：Bagging 的特化版，在每次分裂時，從**隨機抽取的 $\sqrt{n}$ 個特徵**中選最佳分裂點。這增加了樹之間的多樣性，通常優於純 Bagging。

**極端隨機樹 (Extra-Trees)**：進一步隨機化——不只隨機選特徵，連閾值也隨機選（而非最佳）。訓練更快，偏差略高但變異數更低。

**特徵重要性**：每個特徵在所有樹中造成的平均不純度下降量：

$$\text{Feature Importance}_j = \frac{\sum_{\text{nodes using } j} w_i \cdot \Delta G_i}{\text{sum of all node importances}}$$

---

[來源: ch07 | 類型: handout] ### 核心代碼

```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,   # 每棵樹限制葉節點數（輕量樹）
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(iris.data, iris.target)

---

[來源: ch07 | 類型: handout] # 特徵重要性
feature_importances = rf_clf.feature_importances_
for name, score in zip(iris.feature_names, feature_importances):
    print(f"{name:25s}: {score:.4f}")

---

[來源: ch07 | 類型: handout] # 視覺化
plt.bar(iris.feature_names, feature_importances)
plt.title("Feature Importances (Random Forest)")
plt.ylabel("Mean Decrease in Impurity")
plt.show()

---

[來源: ch07 | 類型: handout] # Extra-Trees（速度更快）
et_clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)
et_clf.fit(iris.data, iris.target)
```

---

[來源: ch07 | 類型: handout] ### 補充練習 3

**理論題：** 隨機森林在每次分裂時只考慮 $\sqrt{n}$ 個特徵，這帶來了偏差-變異數的什麼取捨？為何減少考慮的特徵數量反而能提升泛化效果？

**實作題：** 在 MNIST 數據集（10000 筆子集）上訓練 `RandomForestClassifier`，計算 `feature_importances_`，將重要性值 reshape 為 28×28 並顯示為熱力圖，觀察哪些像素對手寫數字分類最重要。

---

---

[來源: ch07 | 類型: handout] ## 4. Boosting：AdaBoost 與 Gradient Boosting

---

[來源: ch07 | 類型: handout] ### 理論背景

**Boosting**：循序訓練分類器，每個分類器**修正前一個的錯誤**。

**AdaBoost**：提高被前一個分類器誤分類的樣本的權重：

$$\hat{y}(\mathbf{x}) = \text{sign}\left(\sum_{j=1}^{N} \alpha_j h_j(\mathbf{x})\right)$$

其中 $\alpha_j = \eta \log\frac{1 - r_j}{r_j}$（$r_j$ 是加權誤差率，誤差越低 $\alpha_j$ 越大）。

**Gradient Boosting**：每個新樹擬合**前一個集成的殘差（偽殘差）**：

$$\hat{y}^{(i)}_{\text{new}} = \hat{y}^{(i)}_{\text{old}} + \eta h_t(\mathbf{x}^{(i)})$$

---

[來源: ch07 | 類型: handout] 殘差（偽殘差）**：

$$\hat{y}^{(i)}_{\text{new}} = \hat{y}^{(i)}_{\text{old}} + \eta h_t(\mathbf{x}^{(i)})$$

其中 $h_t$ 是擬合殘差 $y^{(i)} - \hat{y}^{(i)}_{\text{old}}$ 的決策樹，$\eta$ 是學習率（縮水因子）。

**過擬合防治**：
- `n_estimators` + `learning_rate`：小學習率需要更多樹（通常效果更好）
- `subsample < 1.0`：每棵樹用隨機子集 → **Stochastic GBM**
- 早停（`n_iter_no_change`）

---

[來源: ch07 | 類型: handout] ### 核心代碼

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor

---

[來源: ch07 | 類型: handout] # AdaBoost（使用深度 1 的決策樹作為基學習器）
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),  # 決策樹樁（stump）
    n_estimators=30,
    learning_rate=0.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)
print(f"AdaBoost 測試準確率: {ada_clf.score(X_test, y_test):.4f}")

---

[來源: ch07 | 類型: handout] # Gradient Boosting（迴歸）
from sklearn.datasets import make_regression
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)

---

[來源: ch07 | 類型: handout] ke_regression
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)

gbm_reg = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.8,         # Stochastic GBM
    n_iter_no_change=20,   # 早停：驗證誤差 20 次未改善就停止
    random_state=42
)
gbm_reg.fit(X_reg, y_reg)
print(f"GBM n_estimators (早停): {gbm_reg.n_estimators_}")
```

---

[來源: ch07 | 類型: handout] ### 補充練習 4

**理論題：** Gradient Boosting 的 `learning_rate` 和 `n_estimators` 之間的關係是什麼？為什麼「小學習率 + 多棵樹」通常優於「大學習率 + 少棵樹」？

**實作題：** 在 `make_regression` 的資料上，分別訓練 `n_estimators=10/50/200/500`（固定 `learning_rate=0.1`）的 GBM，繪製訓練和驗證損失隨 `n_estimators` 的變化曲線，找到最佳的樹的數量。

---

---

[來源: ch07 | 類型: handout] ### 理論背景

**Stacking (Stacked Generalization)**：用一個**元學習器 (Meta-learner)** 來組合基學習器的預測：

1. 訓練集切為 $k$ 個 fold（類似 K-fold CV）
2. 每個基學習器在 $k-1$ 個 fold 上訓練，對剩餘的 1 個 fold 做 OOF（Out-of-Fold）預測
3. 將所有 OOF 預測拼接，作為元學習器的訓練集
4. 元學習器學習「如何組合基學習器的輸出」

Stacking 比固定權重的投票更靈活，可以學習到「在什麼情況下信任哪個模型」。

---

[來源: ch07 | 類型: handout] ```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ("lr",  LogisticRegression(random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ],
    final_estimator=LogisticRegression(),  # 元學習器
    cv=5,                                   # OOF 使用 5-fold
    passthrough=False                       # 元學習器只看基學習器的輸出
)
stacking_clf.fit(X_train, y_train)
print(f"Stacking 測試準確率: {stacking_clf.score(X_test, y_test):.4f}")
```

---

[來源: ch07 | 類型: handout] ### 補充練習 5

**理論題：** 為什麼 Stacking 的基學習器必須使用 OOF 預測（而非訓練集預測）來訓練元學習器？若用訓練集預測會發生什麼問題？

**實作題：** 在 `make_moons` 上比較 VotingClassifier（軟投票）、BaggingClassifier、RandomForestClassifier、GradientBoostingClassifier 和 StackingClassifier 的測試集準確率，製成比較表。

---

---

[來源: ch07 | 類型: handout] ## 結論

集成學習的核心思想：

- **多樣性** + **數量** = 更好的泛化能力
- **Bagging**（隨機森林）：並行訓練，高變異數降低
- **Boosting**（AdaBoost、GBM、XGBoost）：循序訓練，高偏差降低
- **Stacking**：最靈活，元學習器自動調整組合權重

下一章（Ch08）轉向無監督學習的前置工作：降維，解決高維資料的「維度詛咒」。

---

---

[來源: ch07 | 類型: handout] ## 課後作業

**作業：集成方法全面比較**

在 MNIST 資料集（取前 5000 筆）上：

1. 分別訓練以下模型，記錄準確率和訓練時間：
   - 單棵 `DecisionTreeClassifier`
   - `RandomForestClassifier(n_estimators=100)`
   - `GradientBoostingClassifier(n_estimators=100, max_depth=3)`
   - `StackingClassifier`（基學習器：DecisionTree + KNN + LogisticRegression）

2. 分析：哪種方法「性價比」最高（準確率提升 / 訓練時間）？

---

[來源: ch07 | 類型: handout] `StackingClassifier`（基學習器：DecisionTree + KNN + LogisticRegression）

2. 分析：哪種方法「性價比」最高（準確率提升 / 訓練時間）？

3. 對最佳 `RandomForestClassifier`，取出 `feature_importances_`，找出最重要的前 10 個像素位置，討論這些像素的分布是否合理。

---

[來源: ch07 | 類型: tutorial] [標題: 集成學習與隨機森林完整指南：投票、Bagging、Boosting 到 Stacking | 描述: 深入集成學習的核心概念：投票分類器、Bagging/Pasting、Out-of-Bag 評估、隨機森林、Extra-Trees、特徵重要性、AdaBoost、Gradient Boosting、XGBoost、Stacking。含完整 Scikit-Learn 實戰。 | 關鍵字: Python, 集成學習, 隨機森林, Boosting, AdaBoost, XGBoost, Bagging, Stacking, Scikit-Learn, 機器學習]
# 集成學習完整指南：從投票到 Stacking 的群體智慧

集成學習（Ensemble Learning）的核心思想：**眾多弱模型的智慧，勝過單一強模型**。就像台股分析師的委員會比單一分析師更準確——只要每個成員犯的錯不同，集體投票就能超越個人。本教學帶你掌握從簡單投票到複雜 Stacking 的所有主流集成方法。

---

[來源: ch07 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **Bagging**（有放回取樣）和 **Pasting**（無放回）建立多個模型並行訓練，再投票
- **隨機森林**是 Bagging + 特徵隨機化，預設幾乎不需要調參，是最佳「入手」演算法
- **Boosting（提升）** 串行訓練，每個模型專注糾正前一個的錯誤（AdaBoost / Gradient Boosting）
- **XGBoost** 是實務上最強的 Gradient Boosting 實作，擁有正則化和早停止機制
- **Stacking** 用「元學習器（Meta Learner）」學習如何結合各基礎模型的預測

---

---

[來源: ch07 | 類型: tutorial] ## 投票分類器

💡 **實際應用情境：** 醫療診斷的第二意見制度——不同科別醫師（SVM、Random Forest、Logistic Regression）各自給出診斷，最終以多數意見決定。只要醫師的失誤不相關，集體判斷比個人更可靠。

---

[來源: ch07 | 類型: tutorial] ### 範例 1: 硬投票 vs 軟投票

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

---

[來源: ch07 | 類型: tutorial] # 三個多樣化的基礎分類器（多樣性是關鍵！）
log_clf   = LogisticRegression(random_state=42)
rnd_clf   = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf   = SVC(probability=True, random_state=42)  # probability=True 用於軟投票

---

[來源: ch07 | 類型: tutorial] # 硬投票（Hard Voting）：多數決
hard_voting_clf = VotingClassifier(
    estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
    voting="hard"
)

---

[來源: ch07 | 類型: tutorial] # 軟投票（Soft Voting）：加權平均機率（通常更好）
soft_voting_clf = VotingClassifier(
    estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
    voting="soft"
)

for clf in [log_clf, rnd_clf, svm_clf, hard_voting_clf, soft_voting_clf]:
    clf.fit(X_train, y_train)
    print(f"{clf.__class__.__name__:30s}: {clf.score(X_test, y_test):.4f}")
```

**✅ 程式碼逐行解析：**

1. `VotingClassifier(voting="hard")`: 多數決——統計每個分類器的預測類別，取最多票的
2. `VotingClassifier(voting="soft")`: 平均每個分類器的類別機率，取機率最高的（需要 `probability=True`）
3. 基礎分類器多樣化（LR + RF + SVM）是關鍵——相關性低的模型集成效果更好

**🎯 重點摘要:**

---

[來源: ch07 | 類型: tutorial] 平均每個分類器的類別機率，取機率最高的（需要 `probability=True`）
3. 基礎分類器多樣化（LR + RF + SVM）是關鍵——相關性低的模型集成效果更好

**🎯 重點摘要:**

- **硬投票**簡單但丟失了機率信息；**軟投票**保留機率，通常效果更好
- 集成效果的上限：基礎模型的多樣性和每個模型的個別準確率

---

---

[來源: ch07 | 類型: tutorial] ### 範例 2: BaggingClassifier 與 OOB 評估

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

---

[來源: ch07 | 類型: tutorial] # Bagging：有放回取樣（bootstrap=True）
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,       # 500 棵決策樹
    max_samples=100,        # 每棵樹取 100 個樣本訓練
    bootstrap=True,         # 有放回取樣（Bagging）；False=Pasting
    oob_score=True,         # 啟用 Out-of-Bag 評估
    n_jobs=-1,              # 平行訓練
    random_state=42
)
bag_clf.fit(X_train, y_train)

print(f"OOB 準確率:  {bag_clf.oob_score_:.4f}")    # 用未被取樣的樣本評估（免費的驗證集！）
print(f"測試集準確率: {bag_clf.score(X_test, y_test):.4f}")
```

**✅ 程式碼逐行解析：**

---

[來源: ch07 | 類型: tutorial] )    # 用未被取樣的樣本評估（免費的驗證集！）
print(f"測試集準確率: {bag_clf.score(X_test, y_test):.4f}")
```

**✅ 程式碼逐行解析：**

1. `bootstrap=True`: 有放回取樣，平均每個樣本被選中的機率 ≈ 63.2%（剩下 36.8% 成為 OOB 樣本）
2. `oob_score=True`: 利用「從未訓練過該樣本」的估計器評估那個樣本，相當於**免費**的交叉驗證
3. `n_jobs=-1`: 使用所有 CPU 核心平行訓練 500 棵樹

**🎯 重點摘要:**

- OOB（Out-of-Bag）評估是 Bagging 獨有的免費驗證機制，與 5-Fold CV 效果相當
- Bagging（有放回）通常比 Pasting（無放回）效果更好，因為取樣多樣性更高

---

---

[來源: ch07 | 類型: tutorial] ## 隨機森林

💡 **實際應用情境：** 台灣金融監理機關使用隨機森林預測金融機構的違規風險——因為它的特徵重要性分析可以告訴監理人員「哪些指標最能預測違規」，而且對異常值不敏感。

---

[來源: ch07 | 類型: tutorial] ### 範例 3: RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

---

[來源: ch07 | 類型: tutorial] # 隨機森林 = Bagging 決策樹 + 特徵隨機化
rnd_clf = RandomForestClassifier(
    n_estimators=500,      # 500 棵樹
    max_leaf_nodes=16,     # 限制葉節點數（降低複雜度）
    max_features="sqrt",   # 每次分裂只考慮 √n 個特徵（引入隨機性）
    n_jobs=-1,
    random_state=42
)
rnd_clf.fit(X_train, y_train)
print(f"隨機森林準確率: {rnd_clf.score(X_test, y_test):.4f}")

---

[來源: ch07 | 類型: tutorial] # Extra-Trees（極端隨機樹）：更隨機，更快
extra_clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)
extra_clf.fit(X_train, y_train)
print(f"Extra-Trees 準確率: {extra_clf.score(X_test, y_test):.4f}")
```

隨機森林 vs Extra-Trees 對比：

| 特性 | 隨機森林 | Extra-Trees |
|------|---------|------------|
| 分裂點選擇 | 最佳閾值 | **隨機**閾值 |
| 訓練速度 | 較慢 | **更快** |
| 偏差 | 較低 | 稍高 |
| 方差 | 較高 | **更低** |

**🎯 重點摘要:**

- 隨機森林幾乎不需要調參（預設就很好），是「開箱即用」的強力分類器
- 特徵數量多時，Extra-Trees 的訓練速度優勢更明顯

---

---

[來源: ch07 | 類型: tutorial] ### 範例 4: 分析 MNIST 的像素重要性

```python
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch07 | 類型: tutorial] # 載入 MNIST（用小型子集示範）
mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X_mnist, y_mnist = mnist["data"][:10000] / 255.0, mnist["target"][:10000]

---

[來源: ch07 | 類型: tutorial] # 訓練隨機森林
rnd_clf_mnist = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rnd_clf_mnist.fit(X_mnist, y_mnist)

---

[來源: ch07 | 類型: tutorial] # 特徵重要性即為每個像素對分類的貢獻
importances = rnd_clf_mnist.feature_importances_
importance_image = importances.reshape(28, 28)

plt.figure(figsize=(6, 6))
plt.imshow(importance_image, cmap="hot")
plt.colorbar()
plt.title("MNIST 隨機森林像素重要性熱力圖")
plt.axis("off")
plt.show()

---

[來源: ch07 | 類型: tutorial] # 圖中較亮的像素（中央）對分類更重要，邊緣（黑色）幾乎不重要
```

**✅ 程式碼逐行解析：**

1. `feature_importances_`: Gini 重要性——每個特徵在所有樹中所有分裂降低不純度的加權平均
2. `.reshape(28, 28)`: 將 784 維特徵重塑為 28×28 圖像，便於視覺化

**🎯 重點摘要:**

- 特徵重要性可用於**特徵選擇**（移除重要性低的特徵）和**模型解釋**
- 注意：高相關性特徵的重要性會被稀釋（彼此「搶分」）

---

---

[來源: ch07 | 類型: tutorial] ## Boosting：AdaBoost 與 Gradient Boosting

💡 **實際應用情境：** 銀行反欺詐系統用 Gradient Boosting 逐步學習「欺詐交易的模式」——每個新的決策樹專注糾正前一棵樹犯錯的交易，最終形成對欺詐行為的精細識別能力。

---

[來源: ch07 | 類型: tutorial] ### 範例 5: AdaBoostClassifier

```python
from sklearn.ensemble import AdaBoostClassifier

---

[來源: ch07 | 類型: tutorial] # AdaBoost：每輪調高被誤分類樣本的權重
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),  # 「樁」弱分類器
    n_estimators=200,
    algorithm="SAMME",    # 多類別 AdaBoost
    learning_rate=0.5,    # 每個估計器的貢獻程度
    random_state=42
)
ada_clf.fit(X_train, y_train)
print(f"AdaBoost 準確率: {ada_clf.score(X_test, y_test):.4f}")
```

---

[來源: ch07 | 類型: tutorial] ### 範例 6: GradientBoostingRegressor

```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

np.random.seed(42)
X_reg = np.random.rand(100, 1) - 0.5
y_reg = 3 * X_reg.ravel()**2 + 0.05 * np.random.randn(100)

---

[來源: ch07 | 類型: tutorial] # Gradient Boosting：每棵樹擬合前一棵樹的「殘差（Residuals）」
gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=120,
    learning_rate=0.1,    # 縮減（Shrinkage）：學習率越小需要越多樹
    subsample=0.25,       # 隨機取 25% 樣本訓練（Stochastic GB）
    random_state=42
)
gbrt.fit(X_reg, y_reg)

---

[來源: ch07 | 類型: tutorial] # 早停止：找到最佳樹數
errors = [((y_reg - gbrt.staged_predict(X_reg)).__next__())**2 for _ in range(1)]

---

[來源: ch07 | 類型: tutorial] # 使用 staged_predict 追蹤訓練誤差
```

**🎯 重點摘要:**

- AdaBoost 調整**樣本權重**；Gradient Boosting 擬合**殘差（偽殘差）**
- 兩者的 `learning_rate` 越小，需要越多的 `n_estimators`（可用早停止自動決定）

---

---

[來源: ch07 | 類型: tutorial] ## XGBoost 與 HistGradientBoosting

---

[來源: ch07 | 類型: tutorial] ### 範例 7: XGBoost 與 HistGradientBoostingClassifier

```python

---

[來源: ch07 | 類型: tutorial] # HistGradientBoosting（sklearn 內建，速度接近 XGBoost）
from sklearn.ensemble import HistGradientBoostingClassifier

hgb_clf = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.05,
    max_depth=4,
    early_stopping=True,    # 自動早停止
    validation_fraction=0.1,
    random_state=42
)
hgb_clf.fit(X_train, y_train)
print(f"HGB 準確率: {hgb_clf.score(X_test, y_test):.4f}")
print(f"最佳迭代次數: {hgb_clf.n_iter_}")

---

[來源: ch07 | 類型: tutorial] # XGBoost（需 pip install xgboost）
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,  # 每棵樹隨機取 80% 特徵
        eval_metric="logloss",
        random_state=42,
        early_stopping_rounds=10,
    )
    # early_stopping 需要提供驗證集
    xgb_clf.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False)
    print(f"XGBoost 準確率: {xgb_clf.score(X_test, y_test):.4f}")
except ImportError:
    print("XGBoost 未安裝，使用 pip install xgboost")
```

---

[來源: ch07 | 類型: tutorial] score(X_test, y_test):.4f}")
except ImportError:
    print("XGBoost 未安裝，使用 pip install xgboost")
```

**✅ 程式碼逐行解析：**

1. `HistGradientBoostingClassifier`: sklearn 內建的高效 GBDT，支援缺失值，不需要填補
2. `early_stopping=True`: 當驗證集效能不再改善時自動停止，防止過擬合
3. `colsample_bytree=0.8`: XGBoost 每棵樹隨機取 80% 特徵，類似隨機森林的特徵隨機化

**🎯 重點摘要:**

- XGBoost 是 Kaggle 競賽的常勝將軍，優點：速度快、正則化（L1/L2）、缺失值處理
- HistGradientBoosting 無需安裝額外套件，是 sklearn 的首選 Boosting 實作

---

---

[來源: ch07 | 類型: tutorial] ### 範例 8: StackingClassifier

```python
from sklearn.ensemble import StackingClassifier

---

[來源: ch07 | 類型: tutorial] # 定義基礎學習器（Layer 1）
base_estimators = [
    ("lr",  LogisticRegression(random_state=42)),
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
    ("svm", SVC(probability=True, random_state=42)),
]

---

[來源: ch07 | 類型: tutorial] # 元學習器（Meta Learner / Layer 2）：學習如何組合基礎模型的輸出
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),  # 元學習器
    cv=5,              # 用 5-Fold CV 生成基礎模型的「乾淨」預測
    passthrough=False  # 不將原始特徵傳給元學習器（只用基礎模型的預測）
)
stacking_clf.fit(X_train, y_train)
print(f"Stacking 準確率: {stacking_clf.score(X_test, y_test):.4f}")
```

**✅ 程式碼逐行解析：**

1. `cv=5`: 用交叉驗證生成基礎模型的「乾淨」預測（防止洩漏），這些預測作為元學習器的輸入
2. `passthrough=False`: 元學習器只接收基礎模型的預測（若 `True` 還接收原始特徵）
3. 訓練流程：Layer 1 模型 → 生成 meta-features → 元學習器學習如何整合

**🎯 重點摘要:**

---

[來源: ch07 | 類型: tutorial] e`: 元學習器只接收基礎模型的預測（若 `True` 還接收原始特徵）
3. 訓練流程：Layer 1 模型 → 生成 meta-features → 元學習器學習如何整合

**🎯 重點摘要:**

- Stacking 理論上最強，但需要更多計算資源和更仔細的設計
- 避免元學習器過於複雜（常用簡單的 LogReg 或 Ridge 作為元學習器）

---

---

[來源: ch07 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: 隨機森林 vs Gradient Boosting 如何選擇？**

A: 隨機森林是並行訓練（快速，調參少），適合快速建立 baseline；Gradient Boosting 是串行訓練（更準確，需要調參），適合追求最高準確率。通常先用 RF 作 baseline，再用 XGBoost 挑戰。

**Q2: Boosting 的 learning_rate 和 n_estimators 如何平衡？**

A: 這是「縮減（Shrinkage）」的核心思想：`learning_rate` 越小（如 0.01），需要越多的 `n_estimators`（如 1000），但泛化效果通常更好。搭配早停止（Early Stopping）使用。

**Q3: 為什麼集成方法效果更好？**

A: 數學依據：若每個分類器的錯誤是**相互獨立**的，集成模型的錯誤率呈指數下降。如 500 個準確率 0.75 的獨立分類器的集成，準確率 > 0.97（大數法則）。

**Q4: Stacking 一定比其他集成方法好嗎？**

A: 不一定。Stacking 的優勢是「讓元學習器自動學習如何結合各模型」，但也帶來過擬合風險和計算成本。實際上 XGBoost 的效果往往就已經很好了。

---

---

[來源: ch07 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #集成學習 #隨機森林 #XGBoost #Boosting #機器學習 #ScikitLearn #Stacking #AdaBoost #程式設計 #教學 #DataScience #MachineLearning #特徵重要性

---

[來源: ch08] [標題: 第8章 – 降維技術完整指南：主成分分析與其他演算法實戰應用 | 機器學習全書 | 描述: 本章節深入解析降維技術，包括主成分分析（PCA）、隨機投影、局部線性嵌入（LLE）、t-SNE 等，並以MNIST資料集實戰。適用於資料科學、AI、Threads社群分享。 | 關鍵字: Python, 機器學習, Scikit-Learn, 降維, PCA, t-SNE, LLE, 資料視覺化, 維度縮減, Threads, AI, 資料科學, 教學, 深度學習]
# 第8章 – 降維技術完整指南

> 📢 **Threads專屬精華**：本章內容特別優化，適合Threads社群分享與討論，歡迎標註 #Threads教學 #AI #資料科學！

歡迎來到第8章！本章將探討降維技術，這是機器學習中處理高維資料的重要工具。我們將學習如何使用主成分分析（PCA）和其他演算法來降低資料維度，同時保留最重要的資訊。

---

[來源: ch08] ## 關鍵重點 (Key Takeaways)
> - **降維動機**: 加速訓練、資料視覺化、壓縮儲存
> - **PCA原理**: 找到資料的最大變異方向作為主成分
> - **Scikit-Learn實作**: 使用 `PCA`、`RandomizedPCA`、`IncrementalPCA` 等類別
> - **其他技術**: 隨機投影、LLE、t-SNE 等流形學習方法
> - **實戰應用**: MNIST資料集降維與視覺化

---

[來源: ch08] ## 設定


💡 **實際應用情境：** 在開始任何機器學習專案之前，建立適當的開發環境是非常重要的。這包括確認Python版本、安裝必要的函式庫，並設定圖表顯示參數。

---

[來源: ch08] ### 範例 1: 環境檢查

```python
import sys

assert sys.version_info >= (3, 7)
```


**✅ 程式碼逐行解析：**
1. `第1行`: 匯入sys模組，用於檢查Python版本
2. `第3行`: 斷言Python版本必須大於等於3.7，確保相容性


**🎯 重點摘要:**
> - **核心功能**: 檢查Python版本以確保程式碼相容性
> - **潛在問題**: 舊版Python可能導致某些功能無法使用
> - **最佳使用情境**: 專案初始化階段的環境驗證

---

[來源: ch08] ### 範例 2: Scikit-Learn版本檢查

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
> - **核心功能**: 確保Scikit-Learn版本符合要求
> - **潛在問題**: 舊版函式庫可能缺少新功能或有bug
> - **最佳使用情境**: 機器學習專案的依賴檢查

---

[來源: ch08] ### 範例 3: 圖表設定

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
> - **核心功能**: 統一圖表的外觀和可讀性
> - **潛在問題**: 字體大小過大可能導致圖表擁擠
> - **最佳使用情境**: 資料視覺化專案的圖表設定

---

[來源: ch08] ### 範例 4: 儲存圖表函數

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

---

[來源: ch08] t_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 從pathlib匯入Path，用於路徑操作
2. `第3-4行`: 建立圖片儲存目錄
3. `第6-10行`: 定義儲存圖表的函數，支援高解析度輸出

**🎯 重點摘要:**
> - **核心功能**: 自動建立目錄並儲存高品質圖表
> - **潛在問題**: 目錄權限問題可能導致儲存失敗
> - **最佳使用情境**: 書籍或報告中的圖表輸出

---

[來源: ch08] ## PCA

💡 **實際應用情境：** 主成分分析（Principal Component Analysis）是降維最常用的技術之一。它通過找到資料的最大變異方向來降低維度，同時保留最多的資訊。

---

[來源: ch08] ### 範例 5: 產生3D資料集

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

---

[來源: ch08] on.from_rotvec([np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)
X += [0.2, 0, 0.2]
```

**✅ 程式碼逐行解析：**

1. `第3行`: 設定樣本數量為60
2. `第4行`: 初始化3D資料集
3. `第6行`: 產生不均勻分佈的角度
4. `第7行`: 計算橢圓形座標
5. `第8-10行`: 加入雜訊、旋轉和平移

**🎯 重點摘要:**
> - **核心功能**: 產生複雜的3D測試資料集
> - **潛在問題**: 隨機種子確保重現性
> - **最佳使用情境**: PCA演算法的示範資料

---

[來源: ch08] ## 主成分

💡 **實際應用情境：** 主成分是資料變異最大的方向。通過SVD分解，我們可以找到這些方向並用於降維。

---

[來源: ch08] ### 範例 6: 計算主成分
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
> - **核心功能**: 使用SVD找到主成分
> - **潛在問題**: 資料必須先中心化
> - **最佳使用情境**: 手動實作PCA

---

[來源: ch08] ## 投影到d維度

💡 **實際應用情境：** 將高維資料投影到低維空間，保留最重要的特徵。

---

[來源: ch08] ### 範例 7: 投影到2D
```python
W2 = Vt[:2].T
X2D = X_centered @ W2
```

**✅ 程式碼逐行解析：**

1. `第1行`: 建立投影矩陣
2. `第2行`: 將資料投影到2D


**🎯 重點摘要:**
> - **核心功能**: 將3D資料降維到2D
> - **潛在問題**: 維度選擇影響資訊保留
> - **最佳使用情境**: 資料視覺化

---

[來源: ch08] ## 使用Scikit-Learn

💡 **實際應用情境：** Scikit-Learn提供了方便的PCA類別，自動處理中心化和投影。

---

[來源: ch08] ### 範例 8: Scikit-Learn PCA
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入PCA類別
2. `第3-4行`: 建立PCA實例並轉換資料


**🎯 重點摘要:**
> - **核心功能**: 簡單易用的PCA實作
> - **潛在問題**: 自動中心化處理
> - **最佳使用情境**: 大多數降維任務

---

[來源: ch08] ### 範例 9: 檢查主成分
```python
pca.components_
```

**✅ 程式碼逐行解析：**

1. `第1行`: 存取主成分向量


**🎯 重點摘要:**
> - **核心功能**: 查看主成分的方向
> - **潛在問題**: 解釋主成分的物理意義
> - **最佳使用情境**: 特徵重要性分析

---

[來源: ch08] ## 解釋變異比率

💡 **實際應用情境：** 解釋變異比率告訴我們每個主成分保留了多少原始資料的變異。

---

[來源: ch08] ### 範例 10: 變異比率
```python
pca.explained_variance_ratio_
```

**✅ 程式碼逐行解析：**

1. `第1行`: 取得各主成分的解釋變異比率


**🎯 重點摘要:**
> - **核心功能**: 評估各維度的重要性
> - **潛在問題**: 累計比率決定保留維度
> - **最佳使用情境**: 選擇最佳降維數量

---

[來源: ch08] ## 選擇正確的維度數量

💡 **實際應用情境：** 選擇合適的維度數量需要在壓縮和資訊保留之間取得平衡。

---

[來源: ch08] ### 範例 11: MNIST資料集載入

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

---

[來源: ch08] 00:]
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入資料集載入函數
2. `第3行`: 載入MNIST資料集
3. `第4-5行`: 分割訓練和測試集

**🎯 重點摘要:**

- **核心功能**: 載入標準機器學習資料集
- **潛在問題**: 大型資料集需要大量記憶體
- **最佳使用情境**: 分類任務的基準資料

---

[來源: ch08] ### 範例 12: 自動維度選擇
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
> - **核心功能**: 自動確定最佳降維數量
> - **潛在問題**: 計算成本高
> - **最佳使用情境**: 大型高維資料集

---

[來源: ch08] ## PCA用於壓縮

💡 **實際應用情境：** PCA可以用於資料壓縮，減少儲存空間同時保留重要資訊。

---

[來源: ch08] ### 範例 13: 資料壓縮
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
> - **核心功能**: 可逆的資料壓縮
> - **潛在問題**: 資訊損失不可避免
> - **最佳使用情境**: 儲存空間有限的應用

---

[來源: ch08] ## 隨機PCA

💡 **實際應用情境：** 當資料集很大時，隨機PCA提供更快的近似解。

---

[來源: ch08] ### 範例 14: 隨機PCA
```python
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 使用隨機SVD求解器
2. `第2行`: 快速降維


**🎯 重點摘要:**
> - **核心功能**: 近似但更快的PCA
> - **潛在問題**: 結果的準確性略低
> - **最佳使用情境**: 大型資料集的快速處理

---

[來源: ch08] ## 增量PCA

💡 **實際應用情境：** 當資料集太大無法放入記憶體時，增量PCA可以分批處理。

---

[來源: ch08] ### 範例 15: 增量PCA

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

---

[來源: ch08] pca.transform(X_train)
```

**✅ 程式碼逐行解析：**

1. `第3行`: 建立增量PCA實例
2. `第4-5行`: 分批擬合資料
3. `第7行`: 轉換整個資料集

**🎯 重點摘要:**
> - **核心功能**: 記憶體高效的PCA
> - **潛在問題**: 需要多次通過資料
> - **最佳使用情境**: 超大型資料集

---

[來源: ch08] ## 隨機投影

💡 **實際應用情境：** 隨機投影是一種簡單的降維技術，通過隨機矩陣將資料投影到低維空間。

---

[來源: ch08] ### 範例 16: 高斯隨機投影
```python
from sklearn.random_projection import GaussianRandomProjection

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced = gaussian_rnd_proj.fit_transform(X)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入高斯隨機投影
2. `第3-4行`: 應用隨機投影


**🎯 重點摘要:**
> - **核心功能**: 快速近似降維
> - **潛在問題**: 理論保證的維度選擇
> - **最佳使用情境**: 非常高維資料

---

[來源: ch08] ## LLE

💡 **實際應用情境：** 局部線性嵌入（Locally Linear Embedding）是一種流形學習技術，適合非線性降維。

---

[來源: ch08] ### 範例 17: LLE應用

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

---

[來源: ch08] X_unrolled = lle.fit_transform(X_swiss)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 產生瑞士捲資料集
2. `第4-5行`: 應用LLE降維

**🎯 重點摘要:**
> - **核心功能**: 非線性流形降維
> - **潛在問題**: 參數選擇敏感
> - **最佳使用情境**: 流形結構資料

---

[來源: ch08] ## 核PCA

💡 **實際應用情境：** 核PCA使用核技巧處理非線性可分的資料。

---

[來源: ch08] ### 範例 18: 核PCA
```python
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04, random_state=42)
X_reduced = rbf_pca.fit_transform(X_swiss)
```

**✅ 程式碼逐行解析：**

1. `第1行`: 匯入核PCA
2. `第3-4行`: 使用RBF核進行降維


**🎯 重點摘要:**
> - **核心功能**: 非線性PCA變體
> - **潛在問題**: 計算成本高
> - **最佳使用情境**: 非線性資料結構

---

[來源: ch08] ## 練習解答

💡 **實際應用情境：** 通過實際練習來鞏固降維技術的理解和應用。

---

[來源: ch08] ### 練習9: MNIST分類比較
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
> - **核心功能**: 基準模型訓練
> - **潛在問題**: 高維資料訓練慢
> - **最佳使用情境**: 比較降維效果

---

[來源: ch08] ### 練習10: t-SNE視覺化
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
> - **核心功能**: 高品質資料視覺化
> - **潛在問題**: 計算時間長
> - **最佳使用情境**: 探索性資料分析

---

[來源: ch08] ## 總結與最佳實踐

降維技術是處理高維資料的重要工具：

- **PCA**: 適用於線性降維，快速且有效
- **隨機投影**: 簡單快速，適合非常高維資料
- **流形學習**: LLE、t-SNE等適合非線性結構
- **實戰建議**: 先嘗試PCA，如效果不佳再考慮其他方法

---

[來源: ch08] ## 常見問答 (FAQ)

**Q: 降維會不會丟失重要資訊？**
A: 會，但可以通過選擇合適的維度數量來最小化損失。

**Q: 如何選擇降維演算法？**
A: 從PCA開始，如果資料有非線性結構，考慮t-SNE或LLE。

---

[來源: ch08] ## 推薦標籤 (Suggested Hashtags)

#Python #機器學習 #降維 #PCA #tSNE #ScikitLearn #資料科學 #視覺化 #Threads教學 #AI #深度學習

---

[來源: ch08 | 類型: cheatsheet] # Ch08 速查表：Dimensionality Reduction

> **核心主旨**：降維加速訓練、去除雜訊、便於視覺化 —— PCA 是首選，t-SNE 專用於視覺化探索。

---

---

[來源: ch08 | 類型: cheatsheet] ## 1. 核心概念一覽

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

---

[來源: ch08 | 類型: cheatsheet] | sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `PCA` | `n_components=0.95` (保留 95% 變異) 或整數 | 標準 PCA |
| `IncrementalPCA` | `n_components=154`, `batch_size=1000` | 大資料 PCA |
| `KernelPCA` | `kernel="rbf"`, `gamma=0.04` | 非線性 PCA |
| `TSNE` | `n_components=2`, `perplexity=30`, `random_state=42` | 視覺化（不可反投影） |
| `.explained_variance_ratio_` | – | 各主成分的變異解釋量 |
| `.components_` | – | 主成分向量（shape: n_components × n_features） |
| `.inverse_transform()` | – | 從低維重建（PCA 特有） |


---

[來源: ch08 | 類型: cheatsheet] ts_` | – | 主成分向量（shape: n_components × n_features） |
| `.inverse_transform()` | – | 從低維重建（PCA 特有） |


---

---

[來源: ch08 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE

---

[來源: ch08 | 類型: cheatsheet] # 標準 PCA（保留 95% 變異）
pca = PCA(n_components=0.95)  # 自動決定維度
X_reduced = pca.fit_transform(X_train)
print(f"原始維度: {X_train.shape[1]}, 降維後: {X_reduced.shape[1]}")
print(f"保留的變異量: {pca.explained_variance_ratio_.sum():.3f}")

---

[來源: ch08 | 類型: cheatsheet] # 找出保留 N% 變異所需的維度數
pca_full = PCA()
pca_full.fit(X_train)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumvar >= 0.95) + 1
print(f"保留 95% 變異需要 {n_components_95} 個主成分")

---

[來源: ch08 | 類型: cheatsheet] # 繪製 explained variance 曲線
plt.plot(cumvar)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.axhline(y=0.95, color='r', linestyle='--')

---

[來源: ch08 | 類型: cheatsheet] # 降維 + 重建（壓縮/去雜訊）
X_recovered = pca.inverse_transform(X_reduced)

---

[來源: ch08 | 類型: cheatsheet] # Incremental PCA（大資料集）
inc_pca = IncrementalPCA(n_components=154, batch_size=500)
for X_batch in np.array_split(X_train, 100):
    inc_pca.partial_fit(X_batch)
X_reduced_inc = inc_pca.transform(X_train)

---

[來源: ch08 | 類型: cheatsheet] # Kernel PCA（非線性降維）
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04, fit_inverse_transform=True)
X_reduced_kpca = kpca.fit_transform(X_train)

---

[來源: ch08 | 類型: cheatsheet] # t-SNE（僅視覺化，不可用於訓練）
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_2d = tsne.fit_transform(X_train[:5000])  # t-SNE 慢，通常只取子集
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train[:5000], cmap="tab10", s=5)
plt.colorbar(); plt.show()

---

[來源: ch08 | 類型: cheatsheet] # 在 Pipeline 中使用 PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([
    ("pca", PCA(n_components=0.95)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
```

---

---

[來源: ch08 | 類型: cheatsheet] ## 4. 常見陷阱

- **PCA 前需要 Scaling**：PCA 基於變異數，特徵尺度不同會讓某些特徵主導主成分，務必先做 `StandardScaler`。
- **t-SNE 不能用於測試集轉換**：t-SNE 沒有 `transform()`，只能 `fit_transform()`，**不能**在 Pipeline 中使用，僅供探索。
- **`n_components=0.95` 是比例**：PCA 中傳入 0~1 的浮點數代表保留的變異比例，傳入整數代表主成分數量。
- **Kernel PCA 選 kernel**：不確定時先試 `rbf`，再視需求試 `sigmoid`、`poly`。

---

---

[來源: ch08 | 類型: cheatsheet] ## 5. 決策指南

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

---

[來源: ch08 | 類型: handout] # 課程講義：降維 (Chapter 08)

高維資料無處不在——一張 28×28 的 MNIST 圖片有 784 個維度，基因組資料可能有數萬個特徵。**維度詛咒**使得高維空間的訓練既緩慢又不穩定：資料點之間的距離趨於相等，密度極度稀疏。降維技術通過在低維空間中保留資料的最重要結構，解決計算瓶頸、資料視覺化和去雜訊問題。

---

---

[來源: ch08 | 類型: handout] ### 理論背景

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

---

[來源: ch08 | 類型: handout] | 資料視覺化 | 投影到 2D/3D，觀察群聚結構 |
| 資料壓縮 | JPEG 使用類似 PCA 的 DCT |
| 去雜訊 | 保留主要方向，過濾雜訊 |


**投影法 vs 流形學習**：

- 投影（PCA）：假設資料位於低維超平面或超空間
- 流形學習（t-SNE, LLE）：假設資料位於低維非線性流形（Swiss Roll 問題）

---

[來源: ch08 | 類型: handout] ### 核心代碼

```python
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch08 | 類型: handout] # 維度詛咒的直觀展示
for d in [1, 2, 3, 10, 100, 1000]:
    n_samples = 1000
    X = np.random.rand(n_samples, d)  # d 維均勻分布
    # 隨機兩點的平均距離隨維度增加
    dist = np.sqrt(((X[0] - X[1:])**2).sum(axis=1)).mean()
    print(f"d={d:5d}: 平均兩點距離 = {dist:.4f}")
```

---

[來源: ch08 | 類型: handout] ### 補充練習 1

**理論題：** 在 2D 平面上，均勻分布的 1000 個點中，95% 的點距原點的距離在 [0.95, 1.05] 之間的概率是多少（假設資料在單位圓內）？隨著維度增加，這個比例如何變化？

---

---

[來源: ch08 | 類型: handout] ### 理論背景

**PCA (Principal Component Analysis)**：找出資料**變異數最大的方向**（主成分），投影後保留最多資訊。

**SVD 分解**（PCA 的實際計算方式）：

$$\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

- $\mathbf{V}$ 的列向量即為主成分（Principal Components）
- $\boldsymbol{\Sigma}$ 的對角元素（奇異值）反映各主成分的重要性
- $\mathbf{X} \mathbf{V}_d$（取前 $d$ 個主成分）即為降維後的表示

**解釋變異數比 (Explained Variance Ratio)**：

---

[來源: ch08 | 類型: handout] ）反映各主成分的重要性
- $\mathbf{X} \mathbf{V}_d$（取前 $d$ 個主成分）即為降維後的表示

**解釋變異數比 (Explained Variance Ratio)**：

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^{n} \lambda_i}$$

其中 $\lambda_k$ 是第 $k$ 個主成分的變異數（特徵值）。

**重要假設**：PCA 假設主成分是線性組合；對非線性資料需用核 PCA 或流形學習。

---

[來源: ch08 | 類型: handout] ### 核心代碼

```python
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

---

[來源: ch08 | 類型: handout] # 載入 MNIST（784 維 → 降至 2 維用於視覺化）
mnist = fetch_openml("mnist_784", as_frame=False)
X_mnist = mnist.data[:5000]
y_mnist = mnist.target[:5000]

---

[來源: ch08 | 類型: handout] # 保留 95% 的解釋變異數
pca_95 = PCA(n_components=0.95)  # 自動決定維度數
X_reduced = pca_95.fit_transform(X_mnist)
print(f"原始維度: {X_mnist.shape[1]}")              # 784
print(f"降維後維度: {X_reduced.shape[1]}")          # 約 154
print(f"解釋變異數: {pca_95.explained_variance_ratio_.sum():.3f}")

---

[來源: ch08 | 類型: handout] # 解壓縮（重建）並計算重建誤差
X_recovered = pca_95.inverse_transform(X_reduced)
reconstruction_error = np.mean((X_mnist - X_recovered) ** 2)
print(f"重建 MSE: {reconstruction_error:.2f}")

---

[來源: ch08 | 類型: handout] # 視覺化解釋變異數
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

---

[來源: ch08 | 類型: handout] ### 補充練習 2

**理論題：** PCA 降維後再用 `inverse_transform` 重建，重建的資料和原始資料不會完全相同——損失的資訊是什麼？這個損失對後續機器學習任務是有益還是有害的？

**實作題：** 對 MNIST 用 `PCA(n_components=2)` 降至 2D，用散點圖繪製（顏色代表數字類別），觀察哪些數字在 2D PCA 空間中容易分離，哪些容易混淆。

---

---

[來源: ch08 | 類型: handout] ### 理論背景

**標準 PCA 的問題**：需要將整個訓練集載入記憶體（SVD 計算）。

**Incremental PCA (IPCA)**：分批次（mini-batch）更新主成分，無需一次載入全部資料：

- 適合大型資料集，記憶體友善
- 計算結果與標準 PCA 近似

**Randomized PCA**：用隨機演算法快速近似前 $d$ 個主成分，比標準 SVD 快得多（當 $d \ll n$ 時）：

```
標準 PCA：O(m × n²) 或 O(n³)
Randomized PCA：O(m × d²) + O(d³)  ← 快很多！
```

---

[來源: ch08 | 類型: handout] ### 核心代碼

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

---

[來源: ch08 | 類型: handout] # Incremental PCA（逐批次處理）
n_batches = 10
ipca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_mnist, n_batches):
    ipca.partial_fit(X_batch)      # 逐批更新

X_ipca = ipca.transform(X_mnist)

---

[來源: ch08 | 類型: handout] # Randomized PCA（速度快，適合 n_components 遠小於特徵數時）
pca_random = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_random   = pca_random.fit_transform(X_mnist)

print(f"IPCA vs PCA 差異（均方）: "
      f"{np.mean((np.abs(X_ipca) - np.abs(X_reduced[:, :154]))**2):.6f}")
```

---

[來源: ch08 | 類型: handout] ### 補充練習 3

**理論題：** 若記憶體限制為 2GB，訓練資料有 1 億筆樣本、每筆 100 個特徵（float32），標準 PCA 能直接計算嗎？如何使用 Incremental PCA 解決？

**實作題：** 比較在 MNIST 上 `PCA(svd_solver="full")`、`PCA(svd_solver="randomized")`、`IncrementalPCA` 的計算時間（`%timeit`），以及降維結果的相似度（用 Frobenius 範數衡量差異）。

---

---

[來源: ch08 | 類型: handout] ### 理論背景

**核 PCA (kPCA)**：先用核函數映射到高維特徵空間，再在高維空間做 PCA。能處理非線性流形。

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$$

**主要非線性降維方法**：

---

[來源: ch08 | 類型: handout] 理非線性流形。

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$$

**主要非線性降維方法**：

| 方法 | 核心思想 | 優點 | 缺點 |
|------|---------|------|------|
| **Kernel PCA** | 在核特徵空間做 PCA | 靈活，可調核函數 | 超參數多 |
| **LLE** | 保持局部鄰域關係 | 展開流形（Swiss Roll） | 對雜訊敏感 |
| **t-SNE** | 高維鄰近關係 → 低維t分布 | 視覺化極佳 | 非確定性，不可用於預測 |
| **UMAP** | 拓撲保持 | 快、可擴展、可用於預測 | 需調參 |


---

[來源: ch08 | 類型: handout] 雜訊敏感 |
| **t-SNE** | 高維鄰近關係 → 低維t分布 | 視覺化極佳 | 非確定性，不可用於預測 |
| **UMAP** | 拓撲保持 | 快、可擴展、可用於預測 | 需調參 |


**t-SNE** 的核心：讓高維空間中「相近的點」在低維中也相近，「遙遠的點」在低維中也遠（用 KL 散度衡量分布差異）：

$$KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

---

[來源: ch08 | 類型: handout] ### 核心代碼

```python
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding

---

[來源: ch08 | 類型: handout] # Swiss Roll 資料（典型非線性流形）
from sklearn.datasets import make_swiss_roll
X_swiss, t = make_swiss_roll(n_samples=1000, random_state=42)

---

[來源: ch08 | 類型: handout] # 核 PCA（RBF 核）
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04,
                 fit_inverse_transform=True,  # 允許重建
                 random_state=42)
X_kpca = kpca.fit_transform(X_swiss)

---

[來源: ch08 | 類型: handout] # LLE：最適合展開流形
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_lle = lle.fit_transform(X_swiss)

---

[來源: ch08 | 類型: handout] # t-SNE：視覺化效果最好（但不可預測新樣本）
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_mnist[:2000])

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
            c=y_mnist[:2000].astype(int), cmap="tab10", s=5, alpha=0.7)
plt.colorbar(label="Digit class")
plt.title("t-SNE of MNIST (2000 samples)")
plt.show()
```

---

[來源: ch08 | 類型: handout] ### 補充練習 4

**理論題：** t-SNE 有一個重要參數 `perplexity`（困惑度），它控制每個點考慮的「有效鄰居數量」。`perplexity` 很小（如 5）和很大（如 100）各會產生什麼視覺效果？

**實作題：** 對 `make_swiss_roll` 資料，分別用標準 `PCA(n_components=2)`、`KernelPCA(kernel="rbf", gamma=0.04)` 和 `LocallyLinearEmbedding` 降至 2D，繪製三個 2D 圖（顏色 = Swiss Roll 的位置 `t`），比較哪種方法能正確「展開」Swiss Roll。

---

---

[來源: ch08 | 類型: handout] ## 結論

降維技術的工具箱：

- **PCA**：線性降維的首選，快速、可解釋，支援 `inverse_transform`
- **`n_components=0.95`**：保留 95% 解釋變異數，自動決定維度數
- **Incremental/Randomized PCA**：大資料集的高效替代方案
- **Kernel PCA / LLE / t-SNE**：處理非線性流形，t-SNE 最適合視覺化

下一章（Ch09）進入無監督學習的核心：聚類，學習如何在沒有標籤的情況下發現資料結構。

---

---

[來源: ch08 | 類型: handout] ## 課後作業

**作業：PCA 加速分類與視覺化**

1. **加速效果**：在完整 MNIST 上，比較以下兩種管線的訓練時間和測試準確率：
   - 直接 `RandomForestClassifier`
   - `PCA(n_components=0.95)` → `RandomForestClassifier`

2. **視覺化**：對 MNIST 用 `t-SNE(n_components=2, perplexity=30)` 降至 2D，繪製散點圖（10 個數字用 10 種顏色），觀察：哪些數字的群聚最清晰？哪些容易混淆？這與 Ch03 的混淆矩陣分析一致嗎？

3. **壓縮率**：計算 PCA 壓縮後的「位元數」節省（原始 784×8 bits，壓縮後的維度×32 bits），以及重建圖片的視覺品質（用肉眼判斷 `inverse_transform` 的結果）。

---

[來源: ch08 | 類型: tutorial] [標題: 降維完整指南：PCA、Kernel PCA、LLE、UMAP 與 t-SNE 實戰 | 描述: 深入降維的核心技術：維度詛咒、PCA（解釋方差比、白化）、增量 PCA、隨機 PCA、核技巧 PCA、局部線性嵌入 (LLE)、UMAP 和 t-SNE 視覺化。含 Scikit-Learn 實戰與逐行解析。 | 關鍵字: Python, 降維, PCA, Kernel PCA, LLE, UMAP, t-SNE, 機器學習, Scikit-Learn, 資料視覺化]
# 降維完整指南：PCA 到 UMAP 的多維資料壓縮實戰

高維資料（High-Dimensional Data）帶來的**維度詛咒（Curse of Dimensionality）** 讓許多 ML 演算法效能退化。降維（Dimensionality Reduction）不只是壓縮資料，更是資料視覺化、噪音過濾和特徵萃取的關鍵技術。本教學從 PCA 的數學原理出發，帶你掌握從線性到非線性的完整降維工具箱。

---

[來源: ch08 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **維度詛咒**：維度增加，高維空間「幾乎為空」，樣本間距離趨向相同，ML 演算法失效
- **PCA** 找到資料方差最大的方向（主成分），是最常用的**線性**降維方法
- `explained_variance_ratio_` 幫你選擇保留多少主成分（如保留 95% 的方差）
- **Kernel PCA** 讓 PCA 適用於非線性流形，適合資料分佈在曲線或曲面上的情況
- **UMAP/t-SNE** 專為**視覺化**設計，能展現高維資料的聚類結構，但不適合降維後再訓練

---

---

[來源: ch08 | 類型: tutorial] ## 維度詛咒

💡 **實際應用情境：** 想像台灣製造業的感測器資料——每台機器有 500 個感測器每秒記錄一次。若直接用所有 500 維特徵訓練 KNN，大多數樣本之間的「歐氏距離」會變得幾乎一樣大，KNN 的「鄰居」概念失去意義。

維度詛咒的核心問題：

- 在 d 維超球體中，隨機樣本有 1 − 0.5^(1/d) 的機率落在「外邊緣 10% 的殼層」
- d=10,000 時：幾乎所有樣本都在邊緣，沒有「內部」樣本
- 距離的一致性（Distance Concentration）：最近和最遠鄰居的距離差異越來越小

```python
import numpy as np

---

[來源: ch08 | 類型: tutorial] # 模擬維度詛咒：不同維度下，距原點的距離分佈
np.random.seed(42)
for d in [2, 10, 100, 1000]:
    X = np.random.rand(1000, d)  # 在 [0,1]^d 均勻分佈
    dists = np.sqrt((X**2).sum(axis=1))  # 到原點的距離
    print(f"d={d:4d}: min={dists.min():.3f}, max={dists.max():.3f}, "
          f"mean={dists.mean():.3f}, std={dists.std():.4f}")

---

[來源: ch08 | 類型: tutorial] # 維度越高，距離分佈越集中（std 相對於 mean 越小）
```

---

---

[來源: ch08 | 類型: tutorial] ## 主成分分析 (PCA)

💡 **實際應用情境：** 台灣醫院的電子病歷資料有數千個欄位，但許多高度相關（如血壓相關指標群）。PCA 能將相關特徵「合并」為少數不相關的主成分，減少計算量同時保留大部分資訊。

---

[來源: ch08 | 類型: tutorial] ### 範例 1: PCA 降維與解釋方差比

```python
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import numpy as np

---

[來源: ch08 | 類型: tutorial] # 載入 MNIST（示範高維降維）
mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X_mnist = mnist["data"][:10000] / 255.0  # 前 10000 筆
y_mnist = mnist["target"][:10000]

---

[來源: ch08 | 類型: tutorial] # 方法 1：直接指定維度
pca_2d = PCA(n_components=2)  # 降到 2 維（用於視覺化）
X_2d = pca_2d.fit_transform(X_mnist)
print(f"2D 解釋方差: {pca_2d.explained_variance_ratio_.sum():.4f}")

---

[來源: ch08 | 類型: tutorial] # 方法 2：保留 95% 的方差（自動決定維度）
pca_95 = PCA(n_components=0.95)  # 保留 95% 方差
X_95 = pca_95.fit_transform(X_mnist)
print(f"保留 95% 方差所需維度: {pca_95.n_components_}")  # 通常約 150 維

---

[來源: ch08 | 類型: tutorial] # 查看每個主成分解釋的方差比例
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

---

[來源: ch08 | 類型: tutorial] ### 範例 2: PCA 壓縮與還原（視覺化重建誤差）

```python

---

[來源: ch08 | 類型: tutorial] # 壓縮：784D → 150D
pca_compress = PCA(n_components=150)
X_compressed = pca_compress.fit_transform(X_mnist)

---

[來源: ch08 | 類型: tutorial] # 還原（有損）：150D → 784D
X_reconstructed = pca_compress.inverse_transform(X_compressed)

---

[來源: ch08 | 類型: tutorial] # 計算重建誤差
reconstruction_mse = np.mean((X_mnist - X_reconstructed)**2)
print(f"重建 MSE: {reconstruction_mse:.6f}")

---

[來源: ch08 | 類型: tutorial] # 視覺化比較原圖和重建圖
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

---

[來源: ch08 | 類型: tutorial] ### 範例 3: 大資料集的增量 PCA

```python
from sklearn.decomposition import IncrementalPCA

---

[來源: ch08 | 類型: tutorial] # 增量 PCA：不需要將全部資料載入記憶體
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

---

[來源: ch08 | 類型: tutorial] # 分批次（Mini-batch）處理
for X_batch in np.array_split(X_mnist, n_batches):
    inc_pca.partial_fit(X_batch)  # 逐批次更新

X_inc_reduced = inc_pca.transform(X_mnist)
print(f"增量 PCA 輸出形狀: {X_inc_reduced.shape}")

---

[來源: ch08 | 類型: tutorial] # 隨機 PCA（大維度 → 小維度，速度更快）
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

---

[來源: ch08 | 類型: tutorial] ## 核技巧 PCA

💡 **實際應用情境：** 若 MNIST 手寫數字的分佈是「非線性流形」（類似捲起的瑞士卷）而非線性平面，傳統 PCA 無法正確展開這個流形。Kernel PCA 透過核函數隱式映射到高維空間再做 PCA。

---

[來源: ch08 | 類型: tutorial] ### 範例 4: Kernel PCA 與 RBF 核

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

---

[來源: ch08 | 類型: tutorial] # 瑞士卷資料（3D 非線性流形）
X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

---

[來源: ch08 | 類型: tutorial] # 線性 PCA（失敗：無法展開瑞士卷）
pca_linear = PCA(n_components=2)
X_pca_linear = pca_linear.fit_transform(X_swiss)

---

[來源: ch08 | 類型: tutorial] # RBF Kernel PCA（成功：展開流形）
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

---

[來源: ch08 | 類型: tutorial] ="Spectral")
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

---

[來源: ch08 | 類型: tutorial] ### 範例 5: LLE 展開非線性流形

```python
from sklearn.manifold import LocallyLinearEmbedding

---

[來源: ch08 | 類型: tutorial] # LLE：保持每個點的「局部線性結構」（近鄰之間的線性關係）
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

---

[來源: ch08 | 類型: tutorial] ## UMAP 與 t-SNE

💡 **實際應用情境：** 在生物資訊學中，使用 UMAP 視覺化單細胞 RNA 測序資料（數萬個基因），讓研究者直觀看到細胞類型的聚類結構——這是用 PCA 無法清楚呈現的。

---

[來源: ch08 | 類型: tutorial] ### 範例 6: t-SNE 視覺化 MNIST

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

---

[來源: ch08 | 類型: tutorial] # t-SNE（只用 2000 個樣本，因計算複雜度 O(n²)）
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

---

[來源: ch08 | 類型: tutorial] # 可以看到 10 個清晰分離的數字群落
```

```python

---

[來源: ch08 | 類型: tutorial] # UMAP（需 pip install umap-learn，速度比 t-SNE 快 10-100 倍）
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

---

[來源: ch08 | 類型: tutorial] plt.show()
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

---

[來源: ch08 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: PCA 之前需要標準化嗎？**

A: 是的！若特徵的尺度差異很大（如身高 cm 和體重 kg），PCA 的主成分會被尺度大的特徵主導。應先用 `StandardScaler` 標準化，確保每個特徵的方差貢獻相同。

**Q2: 降維後可以直接訓練分類器嗎？**

A: PCA 降維後的資料可以訓練分類器（PCA 保留了主要變異信息）。但 t-SNE/UMAP 不行——它們的映射函數不穩定，對新樣本無法產生一致的投影。

**Q3: 如何選擇 PCA 的 n_components？**

A: (1) 保留累積方差 95% 的主成分數；(2) 繪製解釋方差圖找 Elbow 點；(3) 如果 PCA 作為預處理，在下游分類任務上用交叉驗證選擇最佳維度。

**Q4: t-SNE 的 perplexity 如何設定？**

A: 通常在 5~50 之間嘗試不同值並觀察結果。Perplexity 可直觀理解為「每個點期望有幾個近鄰」，資料量大時可以設大一點（如 100）。

---

---

[來源: ch08 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #PCA #降維 #UMAP #tSNE #機器學習 #ScikitLearn #資料視覺化 #KernelPCA #程式設計 #教學 #DataScience #MachineLearning #AI #特徵萃取

---

[來源: ch09] [標題: 機器學習無監督學習：聚類、降維與異常檢測完整指南 - Python實作教學 | 描述: 深入探討無監督學習的核心技術，包括K-Means、DBSCAN、高斯混合模型、PCA等演算法。通過實際案例學習如何應用這些技術進行聚類分析、圖像分割、異常檢測和降維。包含完整的程式碼實作、數學推導和最佳實踐。 | 關鍵字: 機器學習, 無監督學習, 聚類, K-Means, DBSCAN, 高斯混合模型, PCA, 降維, 異常檢測, Python, 教學]
# 機器學習無監督學習：聚類、降維與異常檢測完整指南

無監督學習是機器學習的重要分支，不依賴標籤資料就能從資料中發現隱藏的模式和結構。本章將深入探討聚類分析、降維技術和異常檢測等核心概念，通過實際案例展示如何應用這些技術解決真實世界的問題。從K-Means到高斯混合模型，從PCA到流形學習，我們將一步步構建完整的無監督學習知識體系。

---

[來源: ch09] ## 關鍵重點 (Key Takeaways)
- 無監督學習能夠在沒有標籤的情況下發現資料的隱藏結構
- K-Means是最簡單且最廣泛使用的聚類演算法之一
- DBSCAN能夠發現任意形狀的聚類且對雜訊具有魯棒性
- 高斯混合模型提供了更靈活的聚類和密度估計能力
- PCA是降維最常用的技術，能夠保留資料的主要變異性
- 異常檢測可以用於識別資料中的異常點或新奇點

---

[來源: ch09] ## 設定與準備

在開始學習無監督學習之前，讓我們先設定好開發環境並匯入必要的套件。

---

[來源: ch09] # 匯入必要的套件
import sys
import numpy as np
import matplotlib.pyplot as plt
from packaging import version
import sklearn

---

[來源: ch09] # 設定圖表字體和樣式
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

---

[來源: ch09] # 建立圖片儲存目錄
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

---

[來源: ch09] plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `import sys, numpy as np, matplotlib.pyplot as plt`: 匯入系統、數值計算和繪圖套件
2. `from packaging import version; import sklearn`: 匯入版本檢查和機器學習套件
3. `plt.rc('font', size=14)`: 設定圖表字體大小為14
4. `IMAGES_PATH.mkdir(parents=True, exist_ok=True)`: 建立圖片儲存目錄，如果不存在則建立
5. `def save_fig(...)`: 定義儲存圖表的函數，支援不同的檔案格式和解析度

---

[來源: ch09] dir(parents=True, exist_ok=True)`: 建立圖片儲存目錄，如果不存在則建立
5. `def save_fig(...)`: 定義儲存圖表的函數，支援不同的檔案格式和解析度

**🎯 重點摘要:**

- **核心功能**: 設定開發環境和圖表樣式，提供圖片儲存功能
- **潛在問題**: 需要確保目錄寫入權限和足夠的磁碟空間
- **最佳使用情境**: 在開始任何機器學習專案時進行環境設定

---

[來源: ch09] ## 聚類分析 (Clustering)

聚類分析是將相似的資料點分組在一起的過程。與監督學習不同，聚類不需要預先知道正確的分組標籤。

---

[來源: ch09] ### 範例 2: 分類 vs 聚類的視覺化比較
```python

---

[來源: ch09] # 載入鳶尾花資料集
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
data.target_names

---

[來源: ch09] # 繪製分類 vs 聚類比較圖
plt.figure(figsize=(9, 3.5))
plt.subplot(121)
plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris setosa")
plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris versicolor")
plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris virginica")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.grid()
plt.legend()

---

[來源: ch09] label="Iris virginica")
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

---

[來源: ch09] sbelow(True)
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

---

[來源: ch09] bplot(122)`: 建立第二個子圖，用於顯示聚類前的資料分佈
7. `plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")`: 以黑色點顯示所有資料點

**🎯 重點摘要:**

- **核心功能**: 視覺化展示監督學習(分類)和無監督學習(聚類)的差異
- **潛在問題**: 圖表可能會因為資料分佈而難以區分聚類
- **最佳使用情境**: 教學場合，用於解釋聚類和分類的根本差異

---

[來源: ch09] ## K-Means演算法

K-Means是最流行且最簡單的聚類演算法之一。它將資料點分成k個聚類，使得每個點到其所屬聚類中心的距離平方和最小化。

---

[來源: ch09] ### 範例 3: K-Means基本使用
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

---

[來源: ch09] # 產生測試資料
blob_centers = np.array([[ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
                         [-2.8,  2.8], [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std,
                  random_state=7)

---

[來源: ch09] # 訓練K-Means模型
k = 5
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X)
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] KMeans(n_clusters=k, n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.cluster import KMeans`: 匯入K-Means演算法
2. `blob_centers = np.array([...])`: 定義聚類中心位置
3. `blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])`: 定義每個聚類的標準差
4. `X, y = make_blobs(...)`: 產生合成資料，包含2000個樣本
5. `kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)`: 建立K-Means模型，設定5個聚類
6. `y_pred = kmeans.fit_predict(X)`: 訓練模型並預測聚類標籤

---

[來源: ch09] k, n_init=10, random_state=42)`: 建立K-Means模型，設定5個聚類
6. `y_pred = kmeans.fit_predict(X)`: 訓練模型並預測聚類標籤

**🎯 重點摘要:**

- **核心功能**: 實現基本的K-Means聚類演算法
- **潛在問題**: 需要預先指定聚類數量，可能導致次優解
- **最佳使用情境**: 當資料大致呈球形分佈且聚類數量已知時

---

[來源: ch09] ### 範例 4: 視覺化聚類結果和決策邊界
```python

---

[來源: ch09] # 定義繪圖函數
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)

---

[來源: ch09] plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_colo

---

[來源: ch09] , centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

---

[來源: ch09] marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linsp

---

[來源: ch09]    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

---

[來源: ch09] ], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

---

[來源: ch09] lors='k')
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

---

[來源: ch09] # 繪製聚類結果
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
save_fig("voronoi_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] =(8, 4))
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
7. 

---

[來源: ch09] np.meshgrid(...)`: 建立網格點座標
6. `Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])`: 預測網格點的聚類標籤
7. `plt.contourf(Z, extent=..., cmap="Pastel2")`: 繪製填充的等高線圖
8. `plt.contour(Z, extent=..., linewidths=1, colors='k')`: 繪製等高線

**🎯 重點摘要:**

- **核心功能**: 視覺化K-Means的決策邊界，形成Voronoi圖
- **潛在問題**: 高解析度繪圖可能消耗大量計算資源
- **最佳使用情境**: 分析2D資料的聚類結果和邊界

---

[來源: ch09] ### 範例 5: K-Means演算法的迭代過程
```python

---

[來源: ch09] # 建立不同迭代次數的K-Means模型
kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=1,
                      random_state=5)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=2,
                      random_state=5)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=3,
                      random_state=5)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)

---

[來源: ch09] # 繪製迭代過程
plt.figure(figsize=(10, 8))

plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", rotation=0)
plt.tick_params(labelbottom=False)
plt.title("Update the centroids (initially randomly)")

---

[來源: ch09] otation=0)
plt.tick_params(labelbottom=False)
plt.title("Update the centroids (initially randomly)")

plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False,
                         show_ylabels=False)
plt.title("Label the instances")

---

[來源: ch09] X, show_xlabels=False,
                         show_ylabels=False)
plt.title("Label the instances")

plt.subplot(323)
plot_decision_boundaries(kmeans_iter1, X, show_centroids=False,
                         show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)

plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False,
                         show_ylabels=False)

---

[來源: ch09] ecision_boundaries(kmeans_iter2, X, show_xlabels=False,
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

---

[來源: ch09] eans_iter3, X, show_ylabels=False)

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

---

[來源: ch09] uster_centers_, circle_color='r', cross_color='w')`: 以紅色圓圈繪製初始中心
6. 後續子圖展示迭代過程中中心和邊界的變化

**🎯 重點摘要:**

- **核心功能**: 動態展示K-Means演算法的收斂過程
- **潛在問題**: 隨機初始化可能導致不同的最終結果
- **最佳使用情境**: 教學場合，用於解釋K-Means的迭代優化過程

---

[來源: ch09] ### 範例 6: Mini-Batch K-Means

```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, n_init=3, random_state=42)
minibatch_kmeans.fit(X)
print("Mini-batch K-Means慣性:", minibatch_kmeans.inertia_)
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] inibatch_kmeans.fit(X)
print("Mini-batch K-Means慣性:", minibatch_kmeans.inertia_)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.cluster import MiniBatchKMeans`: 匯入Mini-Batch K-Means
2. `minibatch_kmeans = MiniBatchKMeans(n_clusters=5, n_init=3, random_state=42)`: 建立Mini-Batch模型
3. `minibatch_kmeans.fit(X)`: 使用小批次訓練模型
4. `print("Mini-batch K-Means慣性:", minibatch_kmeans.inertia_)`: 輸出模型的慣性值

**🎯 重點摘要:**

---

[來源: ch09] X)`: 使用小批次訓練模型
4. `print("Mini-batch K-Means慣性:", minibatch_kmeans.inertia_)`: 輸出模型的慣性值

**🎯 重點摘要:**

- **核心功能**: 提供更快的K-Means變體，適合大規模資料
- **潛在問題**: 可能產生稍微不準確的結果
- **最佳使用情境**: 處理大型資料集時的聚類分析

---

[來源: ch09] # 下載瓢蟲圖片
import urllib.request
homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
filename = "ladybug.png"
filepath = IMAGES_PATH / filename
if not filepath.is_file():
    print("Downloading", filename)
    url = f"{homl3_root}/images/unsupervised_learning/{filename}"
    urllib.request.urlretrieve(url, filepath)

---

[來源: ch09] # 載入和處理圖片
import PIL
image = np.asarray(PIL.Image.open(filepath))
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, n_init=10, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

---

[來源: ch09] # 顯示不同聚類數量的分割結果
segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

plt.figure(figsize=(10, 5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

---

[來源: ch09] _img.reshape(image.shape))

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

---

[來源: ch09] plt.imshow(segmented_imgs[idx] / 255)
    plt.title(f"{n_clusters} colors")
    plt.axis('off')

save_fig('image_segmentation_plot', tight_layout=False)
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] .axis('off')

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

---

[來源: ch09] eans.labels_]`: 將每個像素替換為其聚類中心顏色
6. `segmented_img = segmented_img.reshape(image.shape)`: 重新塑形為原始圖片尺寸

**🎯 重點摘要:**

- **核心功能**: 使用K-Means進行圖像顏色量化，實現圖像分割
- **潛在問題**: 聚類數量會影響分割品質和計算時間
- **最佳使用情境**: 圖像壓縮、藝術風格轉換和物體分割

---

[來源: ch09] ### 範例 8: 聚類輔助的半監督學習
```python
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

---

[來源: ch09] # 載入數字資料集
X_digits, y_digits = load_digits(return_X_y=True)
X_train, y_train = X_digits[:1400], y_digits[:1400]
X_test, y_test = X_digits[1400:], y_digits[1400:]

---

[來源: ch09] # 使用少量標籤資料訓練
n_labeled = 50
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print("使用50個標籤樣本的準確率:", log_reg.score(X_test, y_test))

---

[來源: ch09] # 使用聚類進行標籤傳播
k = 50
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = X_digits_dist.argmin(axis=0)
X_representative_digits = X_train[representative_digit_idx]

---

[來源: ch09] # 手動標記代表性數字（在實際應用中這是手動完成的）
y_representative_digits = np.array([
    1, 3, 6, 0, 7, 9, 2, 4, 8, 9,
    5, 4, 7, 1, 2, 6, 1, 2, 5, 1,
    4, 1, 3, 3, 8, 8, 2, 5, 6, 9,
    1, 4, 0, 6, 8, 3, 4, 6, 7, 2,
    4, 1, 0, 7, 5, 1, 9, 9, 3, 7
])

---

[來源: ch09] # 使用代表性樣本訓練
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_representative_digits, y_representative_digits)
print("使用聚類代表樣本的準確率:", log_reg.score(X_test, y_test))
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] , y_representative_digits)
print("使用聚類代表樣本的準確率:", log_reg.score(X_test, y_test))
```

**✅ 程式碼逐行解析：**

1. `X_digits, y_digits = load_digits(return_X_y=True)`: 載入手寫數字資料集
2. `log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])`: 使用少量標籤資料訓練
3. `X_digits_dist = kmeans.fit_transform(X_train)`: 計算每個樣本到聚類中心的距離
4. `representative_digit_idx = X_digits_dist.argmin(axis=0)`: 找到每個聚類的代表樣本
5. `y_representative_digits = np.array([...])`: 手動標記代表樣本的標籤

---

[來源: ch09] digits_dist.argmin(axis=0)`: 找到每個聚類的代表樣本
5. `y_representative_digits = np.array([...])`: 手動標記代表樣本的標籤

**🎯 重點摘要:**

- **核心功能**: 使用聚類技術擴展有限的標籤資料
- **潛在問題**: 需要手動標記代表樣本，可能耗時
- **最佳使用情境**: 標籤資料稀缺但非標籤資料豐富的情況

---

[來源: ch09] ## DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 是一種基於密度的聚類演算法，能夠發現任意形狀的聚類。

---

[來源: ch09] ### 範例 9: DBSCAN基本使用

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

---

[來源: ch09] ("核心樣本索引:", dbscan.core_sample_indices_[:10])
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

---

[來源: ch09] :", dbscan.core_sample_indices_[:10])`: 顯示核心樣本的索引
7. `print("核心樣本:", dbscan.components_)`: 顯示核心樣本的座標

**🎯 重點摘要:**

- **核心功能**: 基於密度的聚類，能夠處理任意形狀的聚類
- **潛在問題**: 參數eps和min_samples的選擇很重要
- **最佳使用情境**: 聚類形狀不規則或包含雜訊的資料

---

[來源: ch09] ### 範例 10: 階層式聚類

```python
from sklearn.cluster import AgglomerativeClustering

X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)
agg = AgglomerativeClustering(linkage="complete").fit(X)

print("子節點:", agg.children_)
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] AgglomerativeClustering(linkage="complete").fit(X)

print("子節點:", agg.children_)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.cluster import AgglomerativeClustering`: 匯入階層式聚類
2. `X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)`: 建立一維資料
3. `agg = AgglomerativeClustering(linkage="complete").fit(X)`: 使用完全連結法訓練
4. `print("子節點:", agg.children_)`: 顯示階層樹的結構

**🎯 重點摘要:**

---

[來源: ch09] ring(linkage="complete").fit(X)`: 使用完全連結法訓練
4. `print("子節點:", agg.children_)`: 顯示階層樹的結構

**🎯 重點摘要:**

- **核心功能**: 建立資料的階層式分組結構
- **潛在問題**: 計算複雜度較高，不適合大規模資料
- **最佳使用情境**: 需要了解聚類階層結構的小型資料集

---

[來源: ch09] ## 高斯混合模型 (Gaussian Mixtures)

高斯混合模型 (GMM) 假設資料是由多個高斯分佈混合而成的，能夠進行更靈活的聚類和密度估計。

---

[來源: ch09] ### 範例 11: 高斯混合模型基本使用
```python
from sklearn.mixture import GaussianMixture

---

[來源: ch09] # 訓練GMM模型
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

print("權重:", gm.weights_)
print("均值:", gm.means_)
print("協方差矩陣:", gm.covariances_)
print("收斂:", gm.converged_)
print("迭代次數:", gm.n_iter_)
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] 協方差矩陣:", gm.covariances_)
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

---

[來源: ch09] riances_)`: 顯示協方差矩陣
7. `print("收斂:", gm.converged_)`: 檢查是否收斂
8. `print("迭代次數:", gm.n_iter_)`: 顯示迭代次數

**🎯 重點摘要:**

- **核心功能**: 提供概率性的聚類和密度估計
- **潛在問題**: 可能陷入局部最優，需要多次初始化
- **最佳使用情境**: 需要軟聚類或密度估計的應用

---

[來源: ch09] ### 範例 12: 生成新樣本
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

---

[來源: ch09] ### 範例 13: 異常檢測實作

```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 2)
anomalies = X[densities < density_threshold]

print("異常樣本數量:", len(anomalies))
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] )
anomalies = X[densities < density_threshold]

print("異常樣本數量:", len(anomalies))
```

**✅ 程式碼逐行解析：**

1. `densities = gm.score_samples(X)`: 計算所有樣本的對數密度分數
2. `density_threshold = np.percentile(densities, 2)`: 設定密度閾值（最低2%）
3. `anomalies = X[densities < density_threshold]`: 識別異常樣本
4. `print("異常樣本數量:", len(anomalies))`: 輸出異常樣本數量

**🎯 重點摘要:**

---

[來源: ch09] densities < density_threshold]`: 識別異常樣本
4. `print("異常樣本數量:", len(anomalies))`: 輸出異常樣本數量

**🎯 重點摘要:**

- **核心功能**: 基於密度的異常檢測
- **潛在問題**: 閾值選擇會影響檢測結果
- **最佳使用情境**: 識別低密度區域的異常點

---

[來源: ch09] ```python
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

---

[來源: ch09] in(aics) - 50, max(aics) + 50])
plt.legend()
plt.grid()
save_fig("aic_bic_vs_k_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] ax(aics) + 50])
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

---

[來源: ch09] bics, "bo-", label="BIC")`: 繪製BIC曲線
5. `plt.plot(range(1, 11), aics, "go--", label="AIC")`: 繪製AIC曲線

**🎯 重點摘要:**

- **核心功能**: 使用資訊準則自動選擇最佳聚類數量
- **潛在問題**: BIC和AIC可能選擇不同的k值
- **最佳使用情境**: 當聚類數量未知時的模型選擇

---

[來源: ch09] ### 範例 15: 貝葉斯高斯混合模型

```python
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
print("權重:", bgm.weights_.round(2))
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] s=10, n_init=10, random_state=42)
bgm.fit(X)
print("權重:", bgm.weights_.round(2))
```

**✅ 程式碼逐行解析：**

1. `from sklearn.mixture import BayesianGaussianMixture`: 匯入貝葉斯高斯混合模型
2. `bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)`: 建立模型
3. `bgm.fit(X)`: 訓練模型
4. `print("權重:", bgm.weights_.round(2))`: 顯示學習到的權重

**🎯 重點摘要:**

---

[來源: ch09] ate=42)`: 建立模型
3. `bgm.fit(X)`: 訓練模型
4. `print("權重:", bgm.weights_.round(2))`: 顯示學習到的權重

**🎯 重點摘要:**

- **核心功能**: 自動確定聚類數量，過多組分會有接近零的權重
- **潛在問題**: 計算更複雜，可能需要更多時間
- **最佳使用情境**: 聚類數量未知且希望自動確定的情況

---

[來源: ch09] ### 習題 10: 聚類 Olivetti 人臉資料集

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

---

[來源: ch09] # 載入資料集
olivetti = fetch_olivetti_faces()
X_train, y_train = olivetti.data[:300], olivetti.target[:300]
X_test, y_test = olivetti.data[300:400], olivetti.target[300:400]

---

[來源: ch09] # PCA降維
pca = PCA(0.99)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

---

[來源: ch09] # K-Means聚類
kmeans = KMeans(n_clusters=40, n_init=10, random_state=42)
X_train_reduced = kmeans.fit_transform(X_train_pca)

---

[來源: ch09] # 分類器訓練
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_reduced, y_train)
accuracy = clf.score(kmeans.transform(X_test_pca), y_test)
print(f"聚類後分類準確率: {accuracy:.3f}")
```

**✅ 程式碼逐行解析：**

---

[來源: ch09] f.score(kmeans.transform(X_test_pca), y_test)
print(f"聚類後分類準確率: {accuracy:.3f}")
```

**✅ 程式碼逐行解析：**

1. `olivetti = fetch_olivetti_faces()`: 載入Olivetti人臉資料集
2. `pca = PCA(0.99)`: 建立保留99%變異的PCA
3. `X_train_pca = pca.fit_transform(X_train)`: 對訓練資料進行降維
4. `kmeans = KMeans(n_clusters=40, n_init=10, random_state=42)`: 建立40聚類的K-Means
5. `X_train_reduced = kmeans.fit_transform(X_train_pca)`: 計算到聚類中心的距離作為新特徵
6. `clf.fit(X_train_reduced, y_train)`: 訓練隨機森林分類器

---

[來源: ch09] means.fit_transform(X_train_pca)`: 計算到聚類中心的距離作為新特徵
6. `clf.fit(X_train_reduced, y_train)`: 訓練隨機森林分類器

**🎯 重點摘要:**

- **核心功能**: 使用聚類作為分類的預處理步驟
- **潛在問題**: 聚類品質會影響最終分類表現
- **最佳使用情境**: 特徵工程和降維結合的分類任務

---

[來源: ch09] ## 總結與最佳實踐

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

---

[來源: ch09] ## 常見問答 (FAQ)

**Q: K-Means和DBSCAN哪個更好？**
A: 取決於資料特性。K-Means適合球形聚類且速度快；DBSCAN適合任意形狀且能處理雜訊。

**Q: 如何選擇聚類數量？**
A: 使用輪廓分析、肘部法則，或資訊準則(BIC/AIC)來評估不同k值的表現。

**Q: 高斯混合模型和K-Means的差異？**
A: GMM提供軟聚類(概率)而非硬聚類，且能建模橢圓形聚類。

**Q: 聚類可以用於分類嗎？**
A: 可以作為半監督學習的一部分，用聚類結果擴展有限的標籤資料。

---

[來源: ch09] ## 推薦標籤 (Suggested Hashtags)

#機器學習 #無監督學習 #聚類分析 #KMeans #DBSCAN #高斯混合模型 #PCA #降維 #異常檢測 #Python教學 #程式設計 #編程 #開發 #技術分享 #學習筆記 #程式開發者 #軟體工程

---

[來源: ch09 | 類型: cheatsheet] # Ch09 速查表：Unsupervised Learning

> **核心主旨**：無監督學習從資料中發現結構 —— K-Means 快速通用，DBSCAN 抗雜訊，GMM 提供機率估計。

---

---

[來源: ch09 | 類型: cheatsheet] ## 1. 核心概念一覽

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

---

[來源: ch09 | 類型: cheatsheet] | sklearn Class / Function | 重點參數 | 用途 |
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

[來源: ch09 | 類型: cheatsheet] | 各樣本的群集標籤 |
| `.inertia_` | – | K-Means 群內距離平方和 |
| `.score()` (GMM) | – | 對數似然度（log-likelihood） |


---

---

[來源: ch09 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch09 | 類型: cheatsheet] # K-Means（找最佳 K：Elbow + Silhouette）
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

---

[來源: ch09 | 類型: cheatsheet] # Elbow 法選 K
plt.plot(k_range, inertias, "bo-")
plt.xlabel("K"); plt.ylabel("Inertia")

---

[來源: ch09 | 類型: cheatsheet] # Silhouette 法選 K
plt.plot(k_range, silhouette_scores, "rs-")
plt.xlabel("K"); plt.ylabel("Silhouette Score")

---

[來源: ch09 | 類型: cheatsheet] # 使用最佳 K 訓練
km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
y_pred = km.fit_predict(X)
cluster_centers = km.cluster_centers_

---

[來源: ch09 | 類型: cheatsheet] # DBSCAN（自動找 outlier）
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)  # -1 代表雜訊點
print(f"群集數: {n_clusters}, 雜訊點: {n_noise}")

---

[來源: ch09 | 類型: cheatsheet] # GMM（機率聚類 + 密度估計）
gm = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
gm.fit(X)
proba = gm.predict_proba(X)    # 每個樣本屬於各群集的機率
labels_gm = gm.predict(X)      # 硬性分類

---

[來源: ch09 | 類型: cheatsheet] # 用 BIC/AIC 選擇最佳 K
bic_scores = []
for k in k_range:
    gm = GaussianMixture(n_components=k, random_state=42)
    gm.fit(X)
    bic_scores.append(gm.bic(X))
best_k = k_range[np.argmin(bic_scores)]
print(f"BIC 建議 K = {best_k}")

---

[來源: ch09 | 類型: cheatsheet] # 異常偵測（GMM 的負對數密度）
densities = gm.score_samples(X)
threshold = np.percentile(densities, 2)  # 最低 2% 為異常
anomaly_mask = densities < threshold
```

---

---

[來源: ch09 | 類型: cheatsheet] ## 4. 常見陷阱

- **K-Means 前需要 Scaling**：K-Means 基於距離，特徵尺度不同會偏向尺度大的特徵。
- **K-Means 只適合球形群集**：非球形（如半月形、同心圓）需要 DBSCAN 或 GMM。
- **DBSCAN 的 `eps` 參數**：eps 太大 → 全部合成一群；eps 太小 → 全部都是雜訊。建議先用 KNN 距離分佈圖來估計 eps。
- **GMM 的 `covariance_type`**：`"full"` 最靈活但參數最多；`"tied"` 速度快；`"diag"` 無軸向外旋轉；`"spherical"` 最簡單。
- **`BayesianGaussianMixture` 的 K**：設定比真實 K 大即可（例如 10），它會自動將不需要的群集的權重降為 0。

---

---

[來源: ch09 | 類型: cheatsheet] ## 5. 決策指南

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

---

[來源: ch09 | 類型: handout] # 課程講義：無監督學習 (Chapter 09)

無監督學習的任務是在**沒有標籤**的資料中發現隱藏結構。本章介紹三大無監督學習技術：**聚類（K-Means、DBSCAN）**、**高斯混合模型（GMM）**，以及它們在半監督學習、影像分割、異常偵測等實際場景的應用。理解這些方法，讓你能從未標記資料中萃取有價值的洞察。

---

---

[來源: ch09 | 類型: handout] ### 理論背景

**K-Means 演算法**：

1. 隨機初始化 $K$ 個群心（Centroid）
2. **分配步驟**：將每個樣本分配到最近的群心
3. **更新步驟**：將每個群心移至分配到它的所有樣本的平均位置
4. 重複步驟 2-3 直到收斂

**目標函數（惰性 Inertia）**：最小化各樣本到所屬群心的距離平方和：

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x}^{(i)} \in C_k} \

|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|^2$$


**K-Means++ 初始化**：以距離加權的方式選擇初始群心（更遠的點更可能被選中），避免糟糕的隨機初始化，是 sklearn 的預設行為。

**K-Means 的限制**：

---

[來源: ch09 | 類型: handout] k\|^2$$


**K-Means++ 初始化**：以距離加權的方式選擇初始群心（更遠的點更可能被選中），避免糟糕的隨機初始化，是 sklearn 的預設行為。

**K-Means 的限制**：

- 需要預先指定 $K$
- 假設群集是球形且大小相近
- 對 outlier 和非凸形狀的群集效果差
- 結果可能因初始化不同而異（用 `n_init=10` 多次執行取最佳）

---

[來源: ch09 | 類型: handout] ### 核心代碼

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch09 | 類型: handout] # 生成測試資料
X_blobs, y_blobs = make_blobs(n_samples=500, n_features=2,
                              centers=5, cluster_std=0.8, random_state=42)

---

[來源: ch09 | 類型: handout] # K-Means
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans.fit(X_blobs)

print(f"群心:\n{kmeans.cluster_centers_}")
print(f"各樣本標籤 (前 10): {kmeans.labels_[:10]}")
print(f"Inertia: {kmeans.inertia_:.2f}")

---

[來源: ch09 | 類型: handout] # 繪製聚類結果
plt.scatter(X_blobs[:, 0], X_blobs[:, 1],
            c=kmeans.labels_, cmap="tab10", s=10, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.legend()
plt.title("K-Means Clustering")
plt.show()

---

[來源: ch09 | 類型: handout] # Mini-Batch K-Means（大資料集）
mb_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42)
mb_kmeans.fit(X_blobs)
```

---

[來源: ch09 | 類型: handout] ### 補充練習 1

**理論題：** K-Means 的 Inertia 隨著 $K$ 增大一定會減小（因為群集越多，每個樣本越接近群心）。那麼只看 Inertia 是否能選到最佳的 $K$？Elbow Method 如何利用這個特性？

**實作題：** 對 `make_blobs` 資料，用 $K = 2, 3, 4, 5, 6, 7, 8, 9, 10$ 分別訓練 K-Means，繪製 Inertia 和 Silhouette Score 對 $K$ 的曲線，用 Elbow Method 找到最佳 $K$。

---

---

[來源: ch09 | 類型: handout] ### 理論背景

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

---

[來源: ch09 | 類型: handout] ### 核心代碼

```python
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = load_digits()
X_digits, y_digits = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42
)

---

[來源: ch09 | 類型: handout] # 基準線：只用 50 個標籤樣本
n_labeled = 50
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print(f"只用 {n_labeled} 個標籤的準確率: {log_reg.score(X_test, y_test):.4f}")

---

[來源: ch09 | 類型: handout] # 半監督：K-Means 聚類 → 標籤傳播
k_clusters = 50
kmeans_semi = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
X_digits_dist = kmeans_semi.fit_transform(X_train)  # shape: (n_train, k_clusters)

---

[來源: ch09 | 類型: handout] # 找每個群集中最具代表性的樣本（最接近群心）
representative_idx = np.argmin(X_digits_dist, axis=0)
X_representative = X_train[representative_idx]
y_representative = y_train[representative_idx]

---

[來源: ch09 | 類型: handout] # 傳播標籤：將每個代表樣本的標籤傳播給同群集的所有樣本
y_train_propagated = y_representative[kmeans_semi.labels_]

log_reg_semi = LogisticRegression(max_iter=10000, random_state=42)
log_reg_semi.fit(X_train, y_train_propagated)  # 全部訓練資料但用傳播標籤
print(f"半監督準確率: {log_reg_semi.score(X_test, y_test):.4f}")
```

---

[來源: ch09 | 類型: handout] ### 補充練習 2

**理論題：** 在半監督學習中，若某個群集的代表樣本被錯誤標記，它的標籤會傳播給整個群集的樣本，造成系統性錯誤。如何設計驗證機制來偵測或緩解這個問題？

**實作題：** 實作帶「距離閾值過濾」的標籤傳播：只對距群心距離在第 20 百分位數以內的樣本傳播標籤（其他樣本不使用），比較有無過濾的半監督準確率。

---

---

[來源: ch09 | 類型: handout] ### 理論背景

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

---

[來源: ch09 | 類型: handout] 非核心點也非邊界點，標記為 $-1$

優點（vs K-Means）：

- **不需要預先指定 $K$**
- 能發現**任意形狀**的群集
- 對 outlier 魯棒（自動標記為雜訊）

缺點：

- 對超參數 `eps` 和 `min_samples` 敏感
- 對不同密度的群集效果差
- 時間複雜度 $O(m \log m)$（搭配空間索引）

---

[來源: ch09 | 類型: handout] ### 核心代碼

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=42)

---

[來源: ch09 | 類型: handout] # DBSCAN
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X_moons)

print(f"群集標籤: {np.unique(dbscan.labels_)}")  # -1=雜訊, 0=群集0, 1=群集1
print(f"核心點數量: {len(dbscan.core_sample_indices_)}")
print(f"雜訊點數量: {(dbscan.labels_ == -1).sum()}")

---

[來源: ch09 | 類型: handout] # 視覺化
colors = ["blue" if l == 0 else "orange" if l == 1 else "red"
          for l in dbscan.labels_]
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=colors, s=10)
plt.title("DBSCAN on Two Moons (red=noise)")
plt.show()

---

[來源: ch09 | 類型: handout] # DBSCAN 沒有 predict()，新樣本需用 KNN 傳播
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
non_noise = dbscan.labels_ != -1
knn_clf.fit(X_moons[non_noise], dbscan.labels_[non_noise])
print(knn_clf.predict([[0, 0.5], [-0.5, 0]]))
```

---

[來源: ch09 | 類型: handout] ### 補充練習 3

**理論題：** DBSCAN 的 `eps` 參數應如何設定？有一種經驗方法是繪製「k 距離圖（k-distance graph）」——對每個點計算其第 k 個最近鄰的距離，排序後繪製，在曲線「膝部」選取 `eps`。直觀地說明為什麼這個方法有效。

**實作題：** 用 `make_circles(n_samples=500, noise=0.05)` 生成兩個同心圓資料，分別用 K-Means（$K=2$）和 DBSCAN 聚類，比較結果。解釋為何 K-Means 在此失敗而 DBSCAN 成功。

---

---

[來源: ch09 | 類型: handout] ### 理論背景

**高斯混合模型 (Gaussian Mixture Model, GMM)**：假設資料由 $K$ 個高斯分布混合而成：

$$p(\mathbf{x}) = \sum_{k=1}^{K} \phi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

其中 $\phi_k$ 是第 $k$ 個成分的混合係數（$\sum_k \phi_k = 1$）。

**EM 演算法 (Expectation-Maximization)**：

- **E 步**：計算每個樣本屬於各成分的責任值（Responsibility）
- **M 步**：根據責任值更新 $\phi_k$、$\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$

---

[來源: ch09 | 類型: handout] 個樣本屬於各成分的責任值（Responsibility）
- **M 步**：根據責任值更新 $\phi_k$、$\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$

**選擇 $K$（模型複雜度）**：使用 BIC 或 AIC 平衡擬合程度與模型複雜度：

$$\text{BIC} = k \ln(m) - 2 \ln(\hat{L})$$

$$\text{AIC} = 2k - 2 \ln(\hat{L})$$

$k$ 是模型參數數量，$\hat{L}$ 是最大似然值。**越小越好**。

**異常偵測**：低密度區域的樣本為異常——設定閾值 `score_samples(X) < threshold`。

---

[來源: ch09 | 類型: handout] ### 核心代碼

```python
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import load_iris

iris = load_iris()

---

[來源: ch09 | 類型: handout] # 訓練 GMM
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(iris.data)

print(f"均值:\n{gm.means_}")
print(f"協方差 shape: {gm.covariances_.shape}")  # (3, 4, 4)
print(f"混合係數: {gm.weights_}")

---

[來源: ch09 | 類型: handout] # 選擇最佳 K（用 BIC）
bics = [GaussianMixture(n_components=k, n_init=5, random_state=42)
        .fit(iris.data).bic(iris.data)
        for k in range(1, 10)]
best_k = range(1, 10)[np.argmin(bics)]
print(f"BIC 選出的最佳 K: {best_k}")

---

[來源: ch09 | 類型: handout] # 異常偵測（密度低 = 異常）
densities = gm.score_samples(iris.data)  # 對數密度
anomaly_threshold = np.percentile(densities, 4)   # 最低 4% 為異常
anomalies = iris.data[densities < anomaly_threshold]
print(f"偵測到 {len(anomalies)} 個異常點")

---

[來源: ch09 | 類型: handout] # 貝葉斯 GMM（自動決定有效成分數）
bgm = BayesianGaussianMixture(n_components=10,     # 上限設大一點
                               n_init=5, random_state=42)
bgm.fit(iris.data)
print(f"有效成分（權重 > 0.01）: {(bgm.weights_ > 0.01).sum()}")
```

---

[來源: ch09 | 類型: handout] ### 補充練習 4

**理論題：** GMM 是軟聚類（每個樣本屬於每個群集的機率），而 K-Means 是硬聚類（每個樣本只屬於一個群集）。GMM 的協方差矩陣 $\boldsymbol{\Sigma}_k$ 的形狀參數（`covariance_type`）有 `full`、`tied`、`diag`、`spherical` 四種，它們各假設什麼形狀的群集？

**實作題：** 用 `BayesianGaussianMixture(n_components=10)` 在 Iris 資料集上訓練，觀察 `bgm.weights_`，確認只有 3 個成分的權重顯著大於 0（對應三個 Iris 種類）。再用這個模型生成 200 個新樣本（`bgm.sample(200)`），繪製生成樣本與原始資料的分佈對比。

---

---

[來源: ch09 | 類型: handout] ## 結論

無監督學習的工具箱：

- **K-Means**：快速、高效，適合球形、大小相近的群集；應用廣泛（影像分割、半監督學習）
- **DBSCAN**：密度聚類，自動偵測群集數量，適合任意形狀，對 outlier 魯棒
- **GMM**：機率模型，支援軟聚類、生成新樣本和異常偵測；BIC/AIC 選擇 $K$
- **BayesianGMM**：自動決定有效成分數，無需手動調 $K$

下一章（Ch10）開始深度學習之旅，用 Keras 建構神經網路！

---

---

[來源: ch09 | 類型: handout] ## 課後作業

**作業：客戶分群分析（聚類應用）**

使用 `sklearn.datasets.fetch_california_housing()` 的地理資訊（`longitude`、`latitude`）：

1. 用 K-Means（$K=8$）對加州各地區的地理位置聚類，代表 8 個「地理區域」。視覺化聚類結果（散點圖，顏色=群集，大小=房價）。

2. 新增「到每個群心的距離」作為 8 個新特徵，加入原有特徵，訓練 `RandomForestRegressor`，與不加這些特徵的基準模型比較 RMSE（5-fold CV）。

3. 用 `GaussianMixture` + `score_samples` 找出加州房價資料中的**地理位置異常點**（與大多數房屋聚集地點不同的地點），在地圖上標注這些點。

---

[來源: ch09 | 類型: tutorial] [標題: 非監督學習完整指南：K-Means、DBSCAN、高斯混合模型與異常偵測 | 描述: 深入非監督學習的三大工具：K-Means（內聚性、輪廓係數、MiniBatch）、DBSCAN（密度聚類）、高斯混合模型（EM 演算法、BIC/AIC 模型選擇、異常偵測）。含 Scikit-Learn 實戰。 | 關鍵字: Python, K-Means, DBSCAN, 高斯混合模型, 聚類, 異常偵測, 非監督學習, Scikit-Learn, 機器學習]
# 非監督學習完整指南：聚類、密度估計與異常偵測

非監督學習（Unsupervised Learning）在沒有標籤的情況下，從原始資料中發現有意義的結構。本教學帶你掌握三大聚類工具——K-Means、DBSCAN 和高斯混合模型（GMM）——並學習如何用 GMM 建構**異常偵測**系統，應用於台灣金融詐欺偵測和製造業品質管控。

---

[來源: ch09 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **K-Means** 假設群落為球形，對離群值敏感，需要預先指定 K
- **輪廓係數（Silhouette Score）** 和**慣性（Inertia）** 用於選擇最佳 K，但各有局限
- **DBSCAN** 能發現任意形狀的群落，自動識別噪音點，但對密度均勻的資料效果較差
- **GMM（高斯混合模型）** 比 K-Means 更靈活——允許橢圓形群落，輸出軟分配機率
- GMM 的 **BIC/AIC** 可用於模型選擇（群數、協方差結構）

---

---

[來源: ch09 | 類型: tutorial] ## K-Means 聚類

💡 **實際應用情境：** 電商平台對台灣消費者進行分群——無需標籤，讓演算法從消費頻率、客單價、品類偏好自動發現「輕度用戶/重度用戶/高價值用戶」等族群，再針對每群設計行銷策略。

---

[來源: ch09 | 類型: tutorial] ### 範例 1: KMeans 基礎訓練

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch09 | 類型: tutorial] # 生成 5 個球形群落的模擬資料
X_blobs, y_true = make_blobs(
    n_samples=500, centers=5, n_features=2,
    cluster_std=1.0, random_state=42
)

---

[來源: ch09 | 類型: tutorial] # 訓練 K-Means（指定 K=5）
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300, random_state=42)
kmeans.fit(X_blobs)

---

[來源: ch09 | 類型: tutorial] # 核心屬性
print(f"群中心:\n{kmeans.cluster_centers_}")
print(f"標籤（前10個）: {kmeans.labels_[:10]}")
print(f"慣性（Inertia）: {kmeans.inertia_:.2f}")  # 越小越緊密

---

[來源: ch09 | 類型: tutorial] # 預測新樣本所屬群
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

---

[來源: ch09 | 類型: tutorial] ### 範例 2: Elbow 方法 + 輪廓分析

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

---

[來源: ch09 | 類型: tutorial] # Elbow 法：慣性 vs K
axes[0].plot(k_range, inertias, "bo-")
axes[0].set_xlabel("K（群數）")
axes[0].set_ylabel("慣性（Inertia）")
axes[0].set_title("Elbow 法（找斜率轉折點）")

---

[來源: ch09 | 類型: tutorial] # 輪廓係數法：
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

---

[來源: ch09 | 類型: tutorial] ### 範例 3: MiniBatchKMeans vs KMeans

```python
from sklearn.cluster import MiniBatchKMeans
import time

X_large = np.random.randn(100_000, 2)  # 10 萬個樣本

---

[來源: ch09 | 類型: tutorial] # 比較訓練時間
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

---

[來源: ch09 | 類型: tutorial] ## DBSCAN：密度聚類

💡 **實際應用情境：** 在地圖上識別台北市的「商業熱點」——商家密度高的區域是一個群落（如信義區商業圈），而孤立的商家（如郊區加油站）被視為噪音點，不屬於任何群落。DBSCAN 天生適合這類形狀不規則的地理資料。

---

[來源: ch09 | 類型: tutorial] ### 範例 4: DBSCAN 處理任意形狀群落

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

---

[來源: ch09 | 類型: tutorial] # DBSCAN 參數：eps（鄰域半徑）和 min_samples（核心點最小近鄰數）
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X_moons)

---

[來源: ch09 | 類型: tutorial] # 標籤：-1 表示噪音點（異常值）
print(f"群標籤: {set(dbscan.labels_)}")       # 如 {0, 1, -1}
print(f"核心點數量: {len(dbscan.core_sample_indices_)}")
print(f"噪音點數量: {(dbscan.labels_ == -1).sum()}")

---

[來源: ch09 | 類型: tutorial] # 視覺化
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

---

[來源: ch09 | 類型: tutorial] amples=5`: 「核心點（Core Point）」需在 eps 鄰域內至少有 5 個點
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

---

[來源: ch09 | 類型: tutorial] ## 高斯混合模型 (GMM)

💡 **實際應用情境：** 在金融市場分析中，股票回報率通常不是單一常態分佈，而是多個市場狀態（牛市、熊市、震盪市）的混合。GMM 能同時建模多個高斯分佈，並輸出每個資料點屬於哪個狀態的機率。

---

[來源: ch09 | 類型: tutorial] ### 範例 5: GaussianMixture 訓練與評估

```python
from sklearn.mixture import GaussianMixture
import numpy as np

---

[來源: ch09 | 類型: tutorial] # 生成橢圓形群落（K-Means 難以處理）
np.random.seed(42)
X_gmm = np.r_[
    np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100),    # 群 1（橢圓）
    np.random.multivariate_normal([4, 2], [[1.5, -0.3], [-0.3, 0.5]], 100),  # 群 2
    np.random.multivariate_normal([1, 6], [[0.5, 0], [0, 2]], 100),       # 群 3
]

---

[來源: ch09 | 類型: tutorial] # GMM 訓練（EM 演算法）
gmm = GaussianMixture(
    n_components=3,        # 假設 3 個高斯分佈
    covariance_type="full",  # 每個成分都有獨立的完整協方差矩陣
    n_init=10,             # 10 次隨機初始化
    random_state=42
)
gmm.fit(X_gmm)

---

[來源: ch09 | 類型: tutorial] # 評估
print(f"收斂次數: {gmm.n_iter_}")
print(f"對數似然: {gmm.score(X_gmm):.4f}")        # 越高越好（越接近 0）

---

[來源: ch09 | 類型: tutorial] # 軟分配：每個樣本屬於各群的機率
soft_labels = gmm.predict_proba(X_gmm[:3])
print(f"\n前 3 個樣本的軟分配機率:\n{soft_labels.round(3)}")

---

[來源: ch09 | 類型: tutorial] # 硬分配
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

---

[來源: ch09 | 類型: tutorial] `"diag"` | 對角協方差（無相關性） | 中 | 座標軸對齊的橢圓 |
| `"spherical"` | 球形（等同 K-Means） | 最少 | 球形群 |


**🎯 重點摘要:**

- GMM 是 K-Means 的「軟版」，輸出機率而非硬標籤
- 需要資料大致服從高斯分佈，對極度非高斯資料效果有限

---

---

[來源: ch09 | 類型: tutorial] ### 範例 6: 用 GMM 的對數密度識別異常值

```python

---

[來源: ch09 | 類型: tutorial] # 利用 GMM 的密度函數識別異常（密度極低的點 = 異常）
densities = gmm.score_samples(X_gmm)  # 每個樣本的對數密度（log probability）

---

[來源: ch09 | 類型: tutorial] # 設定閾值：密度低於第 2 百分位數的視為異常
threshold = np.percentile(densities, 2)
anomalies = X_gmm[densities < threshold]

print(f"密度閾值: {threshold:.4f}")
print(f"偵測到的異常樣本數: {len(anomalies)}")

---

[來源: ch09 | 類型: tutorial] # 視覺化：正常樣本（藍色）和異常樣本（紅色星號）
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

---

[來源: ch09 | 類型: tutorial] ### 範例 7: K-Means 輔助標籤傳播

```python
from sklearn.datasets import load_digits

---

[來源: ch09 | 類型: tutorial] # 手寫數字資料集（少量標籤 + 大量未標籤）
X_digits, y_digits = load_digits(return_X_y=True)
X_train_small = X_digits[:50]   # 只有 50 個標籤
X_train_unlabeled = X_digits[50:1000]
X_test = X_digits[1000:]
y_train_small = y_digits[:50]
y_test = y_digits[1000:]

---

[來源: ch09 | 類型: tutorial] # Step 1：用 K-Means 在未標籤資料中找 50 個代表性樣本
kmeans_50 = KMeans(n_clusters=50, n_init=10, random_state=42)
kmeans_50.fit(X_train_unlabeled)

---

[來源: ch09 | 類型: tutorial] # 取每個群中離群中心最近的樣本（最具代表性）
representative_idx = np.argmin(
    kmeans_50.transform(X_train_unlabeled), axis=0
)
X_representative = X_train_unlabeled[representative_idx]

---

[來源: ch09 | 類型: tutorial] # 手動標記這 50 個代表性樣本（這裡用真實標籤模擬）
y_representative = y_digits[50 + representative_idx]

---

[來源: ch09 | 類型: tutorial] # Step 2：用標籤傳播——將每個群的代表標籤傳給同群所有樣本
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_representative, y_representative)
print(f"半監督學習準確率: {log_reg.score(X_test, y_test):.4f}")
```

**🎯 重點摘要:**

- 半監督策略：用聚類找「最具代表性的樣本」優先標記，最大化標記效益
- 適合標記成本高的場景（醫療影像、法律文件）

---

---

[來源: ch09 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: K-Means 和 GMM 如何選擇？**

A: 若確定群落是球形（尺度相近）且不需要機率輸出，K-Means 更快速；若群落可能是橢圓形，或需要每個點的歸屬機率（軟分配），用 GMM。

**Q2: DBSCAN 的 eps 如何設定？**

A: 繪製「K-NN 距離圖」：計算每個點到第 k 個近鄰的距離（k=min_samples），排序後繪圖，找到距離急劇增大的「knee 點」作為 eps。

**Q3: 如何評估聚類效果（沒有真實標籤時）？**

A: 常用指標：輪廓係數（`silhouette_score`）、Calinski-Harabasz 指數（`calinski_harabasz_score`）、Davies-Bouldin 指數（`davies_bouldin_score`）。若有真實標籤（但未用於訓練），可用 ARI（調整蘭德指數）。

**Q4: GMM 的 BIC 和 AIC 有什麼差別？**

A: 兩者都是在「模型擬合程度」和「模型複雜度」之間取得平衡。BIC 對複雜模型懲罰更重，傾向選擇更簡單的模型；AIC 懲罰較輕，傾向選擇更好地擬合資料的模型。

---

---

[來源: ch09 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #KMeans #DBSCAN #GMM #非監督學習 #聚類 #異常偵測 #機器學習 #ScikitLearn #半監督學習 #程式設計 #教學 #DataScience #MachineLearning

---

[來源: ch10] [標題: 使用 Keras 建構神經網路：從基礎到進階實作 | 描述: 完整教學：學習使用 Keras 建構神經網路，從生物神經元到人工神經網路，包含分類、回歸、複雜模型設計、超參數調優等實戰技巧。 | 關鍵字: Keras, 神經網路, 機器學習, TensorFlow, 深度學習, 人工智慧, 程式設計, 教學]
# 使用 Keras 建構神經網路：從基礎到進階實作

神經網路是現代人工智慧的核心技術，而 Keras 作為 TensorFlow 的高階 API，讓建構和訓練神經網路變得簡單直觀。本教學將帶您從生物神經元的概念出發，逐步深入人工神經網路的實作，包含分類、回歸、複雜模型設計、超參數調優等完整流程。

---

[來源: ch10] ## 關鍵重點 (Key Takeaways)
- 神經網路由互連的神經元組成，能學習複雜的非線性關係
- Keras 提供三種建構模型的 API：序列、函數式和子類化
- 活化函數決定神經元的輸出行為，ReLU 是隱藏層的常用選擇
- 過擬合是神經網路常見問題，可透過正則化、Dropout 等技術解決
- 超參數調優對於模型效能至關重要，可使用 Keras Tuner 等工具

---

---

[來源: ch10] ## 環境設定與準備
💡 **實際應用情境：** 在開始建構神經網路之前，需要確保開發環境已正確設定，包括必要的程式庫版本檢查。

---

[來源: ch10] ### 範例 1: 環境檢查與設定
```python
import sys
import matplotlib.pyplot as plt

---

[來源: ch10] # 檢查 Python 版本
assert sys.version_info >= (3, 7)

---

[來源: ch10] # 設定圖表字體大小
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] egend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 匯入 sys 模組，用於檢查 Python 版本
2. `assert sys.version_info >= (3, 7)`: 確保 Python 版本至少為 3.7
3. `plt.rc('font', size=14)`: 設定 Matplotlib 字體大小為 14
4. `plt.rc('axes', labelsize=14, titlesize=14)`: 設定軸標籤和標題字體大小
5. `plt.rc('legend', fontsize=14)`: 設定圖例字體大小
6. `plt.rc('xtick', labelsize=10)`: 設定 x 軸刻度字體大小
7. `plt.rc('ytick', labelsize=10)`: 設定 y 軸刻度字體大小

---

[來源: ch10] 大小
6. `plt.rc('xtick', labelsize=10)`: 設定 x 軸刻度字體大小
7. `plt.rc('ytick', labelsize=10)`: 設定 y 軸刻度字體大小

**🎯 重點摘要:**

- **核心功能**: 確保開發環境符合最低要求，並設定一致的視覺化樣式
- **潛在問題**: 版本不相容可能導致程式錯誤
- **最佳使用情境**: 在每個機器學習專案開始時執行環境檢查

---

---

[來源: ch10] ### 感知器 (The Perceptron)
💡 **實際應用情境：** 感知器是最簡單的人工神經元，能解決線性可分的二元分類問題，如鳶尾花分類。

---

[來源: ch10] ### 範例 2: 使用 Scikit-Learn 實作感知器
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

---

[來源: ch10] # 載入鳶尾花資料集
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)  # 鳶尾花 setosa

---

[來源: ch10] # 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

---

[來源: ch10] # 建立並訓練感知器
per_clf = Perceptron(random_state=42)
per_clf.fit(X_train, y_train)

---

[來源: ch10] # 進行預測
X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] 1. `from sklearn.datasets import load_iris`: 匯入鳶尾花資料集
2. `iris = load_iris(as_frame=True)`: 載入資料集並轉為 DataFrame 格式
3. `X = iris.data[["petal length (cm)", "petal width (cm)"]].values`: 選取花瓣長度和寬度作為特徵
4. `y = (iris.target == 0)`: 將目標轉為二元分類 (是否為 setosa)
5. `per_clf = Perceptron(random_state=42)`: 建立感知器模型
6. `per_clf.fit(X_train, y_train)`: 訓練模型
7. `y_pred = per_clf.predict(X_new)`: 對新資料進行預測

---

[來源: ch10] 2)`: 建立感知器模型
6. `per_clf.fit(X_train, y_train)`: 訓練模型
7. `y_pred = per_clf.predict(X_new)`: 對新資料進行預測

**🎯 重點摘要:**

- **核心功能**: 實作簡單的線性分類器
- **潛在問題**: 只能處理線性可分資料，無法學習複雜模式
- **最佳使用情境**: 快速原型設計或作為基準模型

---

---

[來源: ch10] ### 活化函數 (Activation Functions)
💡 **實際應用情境：** 活化函數決定神經元的輸出，選擇合適的活化函數對於模型效能至關重要。

---

[來源: ch10] ```python
import numpy as np
from scipy.special import expit as sigmoid

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

max_z = 4.5
z = np.linspace(-max_z, max_z, 200)

plt.figure(figsize=(11, 3.1))

plt.subplot(121)
plt.plot([-max_z, 0], [0, 0], "r-", linewidth=2, label="Heaviside")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.plot([0, 0], [0, 1], "r-", linewidth=0.5)
plt.plot([0, max_z], [1, 1], "r-", linewidth=2)
plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, np.tanh(z), "b-", linewidth=1, label="Tanh")
plt.grid(True)
plt.title("活化函數")
plt.axis([-max_z, max_z, -1.65, 2.4])
plt.legend(loc="lower right", fontsize=13)

plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Heaviside")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=1, label="Tanh")
plt.grid(True)
plt.title("導數")
plt.axis([-max_z, max_z, -0.2, 1.2])

plt.show()
```

---

[來源: ch10] 1, label="Tanh")
plt.grid(True)
plt.title("導數")
plt.axis([-max_z, max_z, -0.2, 1.2])

plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] plt.grid(True)
plt.title("導數")
plt.axis([-max_z, max_z, -0.2, 1.2])

plt.show()
```

**✅ 程式碼逐行解析：**

1. `def relu(z): return np.maximum(0, z)`: 定義 ReLU 活化函數
2. `def derivative(f, z, eps=0.000001)`: 定義數值微分函數
3. `z = np.linspace(-max_z, max_z, 200)`: 產生輸入值範圍
4. `plt.subplot(121)`: 建立第一個子圖
5. `plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")`: 繪製 ReLU 函數
6. `plt.subplot(122)`: 建立第二個子圖
7. `plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")`: 繪製 Sigmoid 導數

---

[來源: ch10] 建立第二個子圖
7. `plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")`: 繪製 Sigmoid 導數

**🎯 重點摘要:**

- **核心功能**: 視覺化不同活化函數及其導數
- **潛在問題**: 某些函數可能導致梯度消失或爆炸
- **最佳使用情境**: 選擇適合任務的活化函數

---

---

[來源: ch10] ## 回歸多層感知器 (Regression MLPs)
💡 **實際應用情境：** 回歸 MLP 用於預測連續值，如房價預測。

---

[來源: ch10] ### 範例 4: 使用 Scikit-Learn 建構回歸 MLP
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

---

[來源: ch10] # 載入加州房價資料集
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

---

[來源: ch10] # 建立 MLP 回歸模型
mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)

---

[來源: ch10] # 計算 RMSE
rmse = root_mean_squared_error(y_valid, y_pred)
print(f"驗證集 RMSE: {rmse:.4f}")
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] E
rmse = root_mean_squared_error(y_valid, y_pred)
print(f"驗證集 RMSE: {rmse:.4f}")
```

**✅ 程式碼逐行解析：**

1. `housing = fetch_california_housing()`: 載入加州房價資料集
2. `X_train, X_valid, y_train, y_valid = train_test_split(...)`: 分割訓練和驗證資料
3. `mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)`: 建立三層隱藏層的 MLP
4. `pipeline = make_pipeline(StandardScaler(), mlp_reg)`: 建立包含標準化和 MLP 的管道
5. 

---

[來源: ch10] te=42)`: 建立三層隱藏層的 MLP
4. `pipeline = make_pipeline(StandardScaler(), mlp_reg)`: 建立包含標準化和 MLP 的管道
5. `pipeline.fit(X_train, y_train)`: 訓練模型
6. `y_pred = pipeline.predict(X_valid)`: 進行預測
7. `rmse = root_mean_squared_error(y_valid, y_pred)`: 計算均方根誤差

**🎯 重點摘要:**

- **核心功能**: 使用 MLP 進行回歸預測
- **潛在問題**: 需要適當的資料預處理和超參數調優
- **最佳使用情境**: 處理複雜的非線性回歸問題

---

---

[來源: ch10] ## 分類多層感知器 (Classification MLPs)
💡 **實際應用情境：** 分類 MLP 用於將輸入分類到不同類別，如影像分類。

---

[來源: ch10] ### 範例 5: 使用 Scikit-Learn 建構分類 MLP
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

---

[來源: ch10] # 載入鳶尾花資料集
iris = load_iris()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    iris.data, iris.target, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42)

---

[來源: ch10] # 建立 MLP 分類器
mlp_clf = MLPClassifier(hidden_layer_sizes=[5], max_iter=10_000, random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_clf)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_valid, y_valid)
print(f"驗證集準確率: {accuracy:.4f}")
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] in)
accuracy = pipeline.score(X_valid, y_valid)
print(f"驗證集準確率: {accuracy:.4f}")
```

**✅ 程式碼逐行解析：**

1. `iris = load_iris()`: 載入鳶尾花資料集
2. `X_train, X_valid, y_train, y_valid = train_test_split(...)`: 分割資料
3. `mlp_clf = MLPClassifier(hidden_layer_sizes=[5], max_iter=10_000, random_state=42)`: 建立單隱藏層 MLP
4. `pipeline = make_pipeline(StandardScaler(), mlp_clf)`: 建立管道
5. `pipeline.fit(X_train, y_train)`: 訓練模型
6. `accuracy = pipeline.score(X_valid, y_valid)`: 計算準確率

---

[來源: ch10] 管道
5. `pipeline.fit(X_train, y_train)`: 訓練模型
6. `accuracy = pipeline.score(X_valid, y_valid)`: 計算準確率

**🎯 重點摘要:**

- **核心功能**: 多類別分類任務
- **潛在問題**: 過擬合風險，需要適當的正則化
- **最佳使用情境**: 中小型分類資料集

---

[來源: ch10] ### 使用序列 API 建構影像分類器
💡 **實際應用情境：** Fashion MNIST 是常見的影像分類基準資料集，用於測試分類演算法。

---

[來源: ch10] ### 範例 6: 載入和預處理 Fashion MNIST 資料集
```python
import tensorflow as tf

---

[來源: ch10] # 載入 Fashion MNIST 資料集
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

---

[來源: ch10] # 正規化像素值
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

---

[來源: ch10] # 類別名稱
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] ress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

**✅ 程式碼逐行解析：**

1. `fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()`: 載入資料集
2. `X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]`: 分割訓練資料
3. `X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]`: 分割驗證資料
4. `X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.`: 正規化到 0-1 範圍
5. `class_names = [...]`: 定義類別名稱

---

[來源: ch10] _test = X_train / 255., X_valid / 255., X_test / 255.`: 正規化到 0-1 範圍
5. `class_names = [...]`: 定義類別名稱

**🎯 重點摘要:**

- **核心功能**: 準備影像分類資料集
- **潛在問題**: 忘記正規化可能導致訓練不穩定
- **最佳使用情境**: 任何影像分類任務的起點

---

---

[來源: ch10] ### 範例 7: 建構和編譯序列模型
```python
tf.random.set_seed(42)

---

[來源: ch10] # 建構模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch10] # 編譯模型
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

---

[來源: ch10] # 顯示模型摘要
model.summary()
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] 1. `tf.random.set_seed(42)`: 設定隨機種子確保重現性
2. `tf.keras.layers.Flatten(input_shape=[28, 28])`: 將 28x28 影像展平為 784 維向量
3. `tf.keras.layers.Dense(300, activation="relu")`: 第一隱藏層，300 個神經元，ReLU 活化
4. `tf.keras.layers.Dense(100, activation="relu")`: 第二隱藏層，100 個神經元
5. `tf.keras.layers.Dense(10, activation="softmax")`: 輸出層，10 個類別，softmax 活化
6. `model.compile(...)`: 編譯模型，指定損失函數、優化器和指標
7. `model.summary()`: 顯

---

[來源: ch10] oftmax")`: 輸出層，10 個類別，softmax 活化
6. `model.compile(...)`: 編譯模型，指定損失函數、優化器和指標
7. `model.summary()`: 顯示模型結構和參數數量

**🎯 重點摘要:**

- **核心功能**: 建構標準的密集神經網路
- **潛在問題**: 層數和神經元數量需根據任務調整
- **最佳使用情境**: 大多數分類和回歸任務

---

---

[來源: ch10] # 訓練模型
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

---

[來源: ch10] # 評估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"測試準確率: {test_accuracy:.4f}")
```

**✅ 程式碼逐行解析：**

1. `history = model.fit(...)`: 訓練模型 30 個 epoch，使用驗證資料
2. `test_loss, test_accuracy = model.evaluate(X_test, y_test)`: 在測試集上評估
3. `print(f"測試準確率: {test_accuracy:.4f}")`: 輸出測試準確率

**🎯 重點摘要:**

- **核心功能**: 訓練和評估神經網路模型
- **潛在問題**: 過擬合，需監控驗證指標
- **最佳使用情境**: 模型訓練和效能評估

---

---

[來源: ch10] ### 使用序列 API 建構回歸模型
💡 **實際應用情境：** 使用神經網路進行回歸預測，如加州房價預測。

---

[來源: ch10] # 載入資料
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

---

[來源: ch10] # 建構模型
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)  # 回歸任務無活化函數
])

---

[來源: ch10] lu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)  # 回歸任務無活化函數
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)

---

[來源: ch10] # 訓練模型
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

---

[來源: ch10] # 進行預測
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
```

**✅ 程式碼逐行解析：**

1. `norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])`: 建立歸一化層
2. `tf.keras.layers.Dense(1)`: 回歸輸出層無活化函數
3. `optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)`: 使用 Adam 優化器
4. `model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])`: 編譯回歸模型
5. `norm_layer.adapt(X_train)`: 適應訓練資料進行歸一化
6. `y_pred = model.predict(X_new)`: 對新資料進行預測

---

[來源: ch10] ])`: 編譯回歸模型
5. `norm_layer.adapt(X_train)`: 適應訓練資料進行歸一化
6. `y_pred = model.predict(X_new)`: 對新資料進行預測

**🎯 重點摘要:**

- **核心功能**: 神經網路回歸預測
- **潛在問題**: 需注意輸出範圍和損失函數選擇
- **最佳使用情境**: 複雜非線性回歸問題

---

---

[來源: ch10] ### 使用函數式 API 建構複雜模型
💡 **實際應用情境：** Wide & Deep 模型結合寬路徑和深路徑，能同時學習記憶和泛化。

---

[來源: ch10] ### 範例 10: Wide & Deep 模型
```python
tf.keras.backend.clear_session()
tf.random.set_seed(42)

---

[來源: ch10] # 定義各層
normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

---

[來源: ch10] # 建構模型
input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])

---

[來源: ch10] # 編譯和訓練
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
normalization_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] ain, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
```

**✅ 程式碼逐行解析：**

1. `input_ = tf.keras.layers.Input(shape=X_train.shape[1:])`: 定義輸入層
2. `normalized = normalization_layer(input_)`: 正規化輸入
3. `hidden1 = hidden_layer1(normalized)`: 第一隱藏層
4. `concat = concat_layer([normalized, hidden2])`: 串接寬路徑和深路徑
5. `model = tf.keras.Model(inputs=[input_], outputs=[output])`: 建立函數式模型
6. `normalization_layer.adapt(X_train)`: 適應正規化參數

---

[來源: ch10] .Model(inputs=[input_], outputs=[output])`: 建立函數式模型
6. `normalization_layer.adapt(X_train)`: 適應正規化參數

**🎯 重點摘要:**

- **核心功能**: 建構多輸入多輸出複雜模型
- **潛在問題**: 拓撲複雜度增加維護難度
- **最佳使用情境**: 需要特殊架構的進階模型

---

---

[來源: ch10] ### 使用子類化 API 建構動態模型
💡 **實際應用情境：** 子類化允許完全自訂模型行為，如動態架構或自訂訓練邏輯。

---

[來源: ch10] ### 範例 11: 自訂 WideAndDeepModel 類別
```python
class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

---

[來源: ch10] self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_w

---

[來源: ch10] m_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output

---

[來源: ch10] # 建立和訓練模型
tf.random.set_seed(42)
model = WideAndDeepModel(30, activation="relu", name="my_cool_model")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=optimizer,
              metrics=["RootMeanSquaredError", "RootMeanSquaredError"])
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] timizer,
              metrics=["RootMeanSquaredError", "RootMeanSquaredError"])
```

**✅ 程式碼逐行解析：**

1. `class WideAndDeepModel(tf.keras.Model)`: 繼承 tf.keras.Model
2. `def __init__(self, units=30, activation="relu", **kwargs)`: 初始化方法
3. `def call(self, inputs)`: 前向傳播邏輯
4. `input_wide, input_deep = inputs`: 解包輸入
5. `concat = tf.keras.layers.concatenate([norm_wide, hidden2])`: 串接特徵
6. `return output, aux_output`: 返回主輸出和輔助輸出

---

[來源: ch10] tf.keras.layers.concatenate([norm_wide, hidden2])`: 串接特徵
6. `return output, aux_output`: 返回主輸出和輔助輸出

**🎯 重點摘要:**

- **核心功能**: 完全自訂模型架構和行為
- **潛在問題**: 複雜度高，易出錯
- **最佳使用情境**: 需要高度自訂邏輯的研究性模型

---

---

[來源: ch10] ### 模型的儲存與載入
💡 **實際應用情境：** 儲存訓練好的模型以便後續使用或部署。

---

[來源: ch10] ### 範例 12: 儲存和載入 Keras 模型
```python

---

[來源: ch10] # 儲存模型
model.save("my_model.keras")

---

[來源: ch10] # 載入模型
loaded_model = tf.keras.models.load_model("my_model.keras")

---

[來源: ch10] # 儲存權重
model.save_weights("my_weights.weights.h5")

---

[來源: ch10] # 載入權重
model.load_weights("my_weights.weights.h5")
```

**✅ 程式碼逐行解析：**

1. `model.save("my_model.keras")`: 以 Keras 格式儲存完整模型
2. `loaded_model = tf.keras.models.load_model("my_model.keras")`: 載入模型
3. `model.save_weights("my_weights.weights.h5")`: 僅儲存權重
4. `model.load_weights("my_weights.weights.h5")`: 載入權重

**🎯 重點摘要:**

- **核心功能**: 模型持久化
- **潛在問題**: 版本相容性問題
- **最佳使用情境**: 模型部署和重用

---

---

[來源: ch10] ### 回呼函數 (Callbacks)
💡 **實際應用情境：** 回呼函數允許在訓練過程中執行自訂邏輯，如早期停止或檢查點儲存。

---

[來源: ch10] # 檢查點回呼
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints.weights.h5",
                                                   save_weights_only=True)

---

[來源: ch10] # 早期停止回呼
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

---

[來源: ch10] # 自訂回呼
class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")

---

[來源: ch10] # 在訓練中使用
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=100,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[checkpoint_cb, early_stopping_cb, PrintValTrainRatioCallback()])
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] callbacks=[checkpoint_cb, early_stopping_cb, PrintValTrainRatioCallback()])
```

**✅ 程式碼逐行解析：**

1. `checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(...)`: 定期儲存最佳權重
2. `early_stopping_cb = tf.keras.callbacks.EarlyStopping(...)`: 早期停止訓練
3. `class PrintValTrainRatioCallback(tf.keras.callbacks.Callback)`: 自訂回呼類別
4. `def on_epoch_end(self, epoch, logs)`: 每個 epoch 結束時執行的方法
5. `callbacks=[...]` : 在 fit 方法中指定回呼列表

---

[來源: ch10] 別
4. `def on_epoch_end(self, epoch, logs)`: 每個 epoch 結束時執行的方法
5. `callbacks=[...]` : 在 fit 方法中指定回呼列表

**🎯 重點摘要:**
回呼函數的使用能有效管理訓練過程，以下為常見回呼函數的重點摘要：
- **ModelCheckpoint**: 定期儲存模型權重，避免訓練中斷導致的損失
- **EarlyStopping**: 監控驗證指標，當指標不再改善時自動停止訓練
- **自訂回呼**: 可根據需求自訂回呼函數，如紀錄特定指標、動態調整學習率等

---

---

[來源: ch10] ### 範例 13b: 指數學習率搜尋 (Learning-rate finder) 回呼

---

[來源: ch10] ```python
import tensorflow as tf
class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor=1.005):
        super().__init__()
        self.factor = factor
        self.lrs = []
        self.losses = []

    def on_train_begin(self, logs=None):
        self.lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.lrs.append(self.lr)
        self.losses.append(logs.get("loss"))
        self.lr *= self.factor
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
```

---

[來源: ch10] self.lr *= self.factor
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
```

**✅ 程式碼逐行解析：**
1. `class ExponentialLearningRate(...)`: 定義自訂回呼類別用於逐批提高學習率  
2. `self.factor = factor`: 每個 batch 增長因子  
3. `on_train_begin`: 取得並儲存優化器初始學習率  
4. `on_batch_end`: 每 batch 記錄目前 lr 與 loss，並將 lr 乘上因子後更新優化器

---

[來源: ch10] 因子  
3. `on_train_begin`: 取得並儲存優化器初始學習率  
4. `on_batch_end`: 每 batch 記錄目前 lr 與 loss，並將 lr 乘上因子後更新優化器

**🎯 重點摘要:**
- 快速探索合適學習率範圍，找出 loss 開始急劇上升前的最佳 lr。  
- 使用方法：在短訓練中加入此回呼，訓練結束後繪製 lrs vs losses。

範例使用與繪圖：
```python

---

[來源: ch10] # 範例：lr finder 使用方式
lr_cb = ExponentialLearningRate(factor=1.01)
model.compile(... )  # 如前面所示
history = model.fit(X_train, y_train, epochs=1, callbacks=[lr_cb], batch_size=128)
import matplotlib.pyplot as plt
plt.semilogx(lr_cb.lrs, lr_cb.losses)
plt.xlabel("learning rate")
plt.ylabel("loss")
plt.show()
```

---

---

[來源: ch10] ### 使用 TensorBoard 視覺化
💡 **實際應用情境：** TensorBoard 提供豐富的視覺化工具來監控訓練過程和模型效能。

---

[來源: ch10] ### 範例 14: 設定 TensorBoard 回呼
```python
from pathlib import Path
from time import strftime

def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir()

---

[來源: ch10] # 建立 TensorBoard 回呼
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir,
                                                profile_batch=(100, 200))

---

[來源: ch10] # 在訓練中使用
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])

---

[來源: ch10] # 啟動 TensorBoard
%load_ext tensorboard
%tensorboard --logdir=./my_logs
```

**✅ 程式碼逐行解析：**

1. `def get_run_logdir(root_logdir="my_logs")`: 建立唯一的執行日誌目錄
2. `run_logdir = get_run_logdir()`: 取得當前執行的日誌目錄
3. `tensorboard_cb = tf.keras.callbacks.TensorBoard(...)`: 建立 TensorBoard 回呼
4. `%load_ext tensorboard`: 載入 TensorBoard Jupyter 擴充
5. `%tensorboard --logdir=./my_logs`: 啟動 TensorBoard 伺服器

---

[來源: ch10] ext tensorboard`: 載入 TensorBoard Jupyter 擴充
5. `%tensorboard --logdir=./my_logs`: 啟動 TensorBoard 伺服器

**🎯 重點摘要:**
TensorBoard 回呼的使用重點如下：
- **自動紀錄**: 搭配 Keras 回呼自動紀錄訓練過程中的指標、損失等資訊
- **視覺化**: 提供豐富的視覺化工具，如曲線圖、直方圖、影像等
- **性能分析**: 可視化每層的權重分佈、激活值等，協助除錯和優化模型

---

---

[來源: ch10] ### 範例 14b: 使用 tf.summary 在自訂訓練回圈或回呼中記錄指標
```python
import tensorflow as tf
from pathlib import Path
from time import strftime

def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir()
file_writer = tf.summary.create_file_writer(str(run_logdir))

---

[來源: ch10] # 在自訂訓練迴圈或回呼內使用
with file_writer.as_default():
    tf.summary.scalar("train/loss", 0.1234, step=1)
    tf.summary.scalar("val/accuracy", 0.8765, step=1)
```

**✅ 程式碼逐行解析：**
1. `get_run_logdir(...)`: 建立唯一執行日誌路徑  
2. `tf.summary.create_file_writer(...)`: 建立 summary 寫入器（TensorBoard 可讀取）  
3. `tf.summary.scalar(...)`: 在指定 step 寫入 scalar 指標

---

[來源: ch10] e_writer(...)`: 建立 summary 寫入器（TensorBoard 可讀取）  
3. `tf.summary.scalar(...)`: 在指定 step 寫入 scalar 指標

**🎯 重點摘要:**
- tf.summary 可與自訂訓練迴圈或回呼無縫整合，提供更細緻的監控（例如 batch 級別指標）。

---

---

[來源: ch10] ## 微調神經網路超參數
💡 **實際應用情境：** 超參數調優對於提升模型效能至關重要，Keras Tuner 提供自動化工具。

---

[來源: ch10] ### 範例 15: 使用 Keras Tuner 進行超參數搜尋
```python
import keras_tuner as kt

---

[來源: ch10] def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.o

---

[來源: ch10] hoice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

---

[來源: ch10] e=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model

---

[來源: ch10] # 執行隨機搜尋
random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)
random_search_tuner.search(X_train, y_train, epochs=10,
                           validation_data=(X_valid, y_valid))

---

[來源: ch10] # 取得最佳模型
top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]
```

**✅ 程式碼逐行解析：**

1. `def build_model(hp)`: 定義模型建構函數，接受超參數物件
2. `n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)`: 定義隱藏層數量搜尋空間
3. `n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)`: 定義神經元數量搜尋空間
4. `learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"

---

[來源: ch10] 元數量搜尋空間
4. `learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")`: 定義學習率搜尋空間
5. `optimizer = hp.Choice("optimizer", values=["sgd", "adam"])`: 定義優化器選擇
6. `random_search_tuner = kt.RandomSearch(...)`: 建立隨機搜尋調優器
7. `random_search_tuner.search(...)`: 執行超參數搜尋
8. `top3_models = random_search_tuner.get_best_models(num_models=3)`: 取得前三名模型

---

[來源: ch10] .search(...)`: 執行超參數搜尋
8. `top3_models = random_search_tuner.get_best_models(num_models=3)`: 取得前三名模型

**🎯 重點摘要:**
Keras Tuner 超參數調優的重點摘要如下：
- **自動化搜尋**: 可自動搜尋最佳超參數組合，節省時間與計算資源
- **多種搜尋策略**: 提供隨機搜尋、貝葉斯優化等多種策略
- **易於整合**: 與 Keras 模型無縫整合，使用簡單

---

---

[來源: ch10] ### 範例 15b: 使用 HyperModel 與多種搜尋策略 (RandomSearch / Hyperband / Bayesian)
```python
import keras_tuner as kt
import tensorflow as tf

class MyClassificationHyperModel(kt.HyperModel):
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

---

[來源: ch10] , input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.input_shape))
        for i in range(hp.Int("n_layers", 1, 4, default=2)):
            model.add(tf.keras.layers.Dense(
                units=hp.Int(f"units_{i}", 32, 256, step=32,

---

[來源: ch10]        model.add(tf.keras.layers.Dense(
                units=hp.Int(f"units_{i}", 32, 256, step=32, default=64),
                activation="relu"))
        model.add(tf.keras.layers.Dense(self.n_classes, activation="softmax"))
        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss="

---

[來源: ch10] ult=1e-3)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

hm = MyClassificationHyperModel(input_shape=(28,28), n_classes=10)

---

[來源: ch10] # Random Search
rnd = kt.RandomSearch(hm, objective="val_accuracy", max_trials=5,
                     directory="kt_dir", project_name="rnd")
rnd.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
best_rnd = rnd.get_best_models(num_models=1)[0]

---

[來源: ch10] # Hyperband
hb = kt.Hyperband(hm, objective="val_accuracy", max_epochs=20,
                  directory="kt_dir", project_name="hyperband")
hb.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
best_hb = hb.get_best_models(num_models=1)[0]

---

[來源: ch10] # Bayesian Optimization (若安裝)
try:
    bo = kt.BayesianOptimization(hm, objective="val_accuracy", max_trials=10,
                                 directory="kt_dir", project_name="bayes")
    bo.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
    best_bo = bo.get_best_models(num_models=1)[0]
except Exception:
    best_bo = None
```

**✅ 程式碼逐行解析：**

---

[來源: ch10] st_bo = bo.get_best_models(num_models=1)[0]
except Exception:
    best_bo = None
```

**✅ 程式碼逐行解析：**

1. `class MyClassificationHyperModel(...)`: 利用 HyperModel 封裝模型建構邏輯與搜尋空間  
2. `hp.Int / hp.Float`: 定義離散與連續超參數空間  
3. `kt.RandomSearch / kt.Hyperband / kt.BayesianOptimization`: 不同搜尋策略，根據資源與需求選擇  
4. `get_best_models(...)`: 取回最佳訓練模型（可進一步微調或評估）

---

[來源: ch10] erband / kt.BayesianOptimization`: 不同搜尋策略，根據資源與需求選擇  
4. `get_best_models(...)`: 取回最佳訓練模型（可進一步微調或評估）

**🎯 重點摘要:**
- HyperModel 能把複雜搜尋空間與建模邏輯封裝起來，便於重複實驗。  
- 選擇搜尋器時需考量計算資源：Hyperband 對早停友好，Bayesian 在樣本效率上較佳。  
- 儲存與對比不同搜尋結果（project_name）有助於實驗管理。

---

---

[來源: ch10] ## 總結與最佳實踐

神經網路是強大的機器學習工具，能學習複雜的模式和關係。本教學涵蓋了從基礎概念到進階實作的完整流程。

**最佳實踐：**
- 從簡單模型開始，逐步增加複雜度
- 始終使用驗證集監控過擬合
- 正確預處理資料，包括正規化和特徵工程
- 使用適當的激活函數和優化器
- 利用回呼函數進行訓練管理
- 定期儲存模型檢查點
- 使用 TensorBoard 監控訓練過程
- 進行系統性的超參數調優

---

---

[來源: ch10] ## 常見問答 (FAQ)

**Q: 神經網路和傳統機器學習演算法有何差異？**
A: 神經網路能自動學習特徵表示，而傳統演算法需要手動特徵工程。神經網路在處理複雜非線性問題時表現更優異，但需要更多資料和計算資源。

**Q: 如何選擇隱藏層數量和神經元數量？**
A: 從簡單架構開始，透過交叉驗證比較不同配置。過少的層/神經元可能欠擬合，過多則可能過擬合。

**Q: ReLU 為什麼是隱藏層的常用激活函數？**
A: ReLU 計算簡單，能有效解決梯度消失問題，且在實務中表現良好。不過在某些情況下可能需要考慮 Leaky ReLU 或 ELU。

**Q: 如何處理過擬合問題？**
A: 使用 Dropout、正則化、早期停止、資料擴增等技術。增加訓練資料量也是有效方法。

---

[來源: ch10] 良好。不過在某些情況下可能需要考慮 Leaky ReLU 或 ELU。

**Q: 如何處理過擬合問題？**
A: 使用 Dropout、正則化、早期停止、資料擴增等技術。增加訓練資料量也是有效方法。

**Q: 函數式 API 和序列 API 有何區別？**
A: 序列 API 適用於簡單的層疊結構，函數式 API 支援複雜拓撲如多輸入輸出、跳躍連接等。

---

---

[來源: ch10] ## 推薦標籤 (Suggested Hashtags)

#Keras #神經網路 #機器學習 #深度學習 #TensorFlow #人工智慧 #程式設計 #教學 #編程 #開發 #技術分享 #學習筆記 #程式開發者 #軟體工程

---

[來源: ch10 | 類型: cheatsheet] # Ch10 速查表：Neural Networks with Keras

> **核心主旨**：Keras 的三種建模 API + 訓練工作流程 —— `model.compile` / `model.fit` / Callbacks 是每次建模的核心。

---

---

[來源: ch10 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Sequential API | 層的線性堆疊 | 標準 MLP/CNN，最常用 |
| Functional API | 可定義多輸入/多輸出/分支架構 | 複雜網路拓撲 |
| Subclassing API | 繼承 `tf.keras.Model`，完全自訂 | 研究用非標準架構 |
| `model.compile()` | 設定 optimizer、loss、metrics | 訓練前必做 |
| `model.fit()` | 執行訓練，傳回 history | 支援 validation_split/data |
| `EarlyStopping` | 監控 val_loss，提前停止防 overfit | 幾乎必用 |
| `ModelCheckpoint` | 自動儲存最佳模型 | 長時間訓練保險 |
| `TensorBoard` | 可視化訓練過程 | 除錯和比較實驗 |
| `keras_tuner` | 自動超參數搜索 | 調網路結構和 lr |


---

[來源: ch10 | 類型: cheatsheet] 自動儲存最佳模型 | 長時間訓練保險 |
| `TensorBoard` | 可視化訓練過程 | 除錯和比較實驗 |
| `keras_tuner` | 自動超參數搜索 | 調網路結構和 lr |


---

---

[來源: ch10 | 類型: cheatsheet] | Keras Class / Function | 重點參數 | 用途 |
|------------------------|---------|------|
| `tf.keras.Sequential` | `layers=[...]` | 序列模型 |
| `tf.keras.layers.Dense` | `units=128`, `activation="relu"` | 全連接層 |
| `tf.keras.layers.Dropout` | `rate=0.2` | 正則化 |
| `tf.keras.layers.BatchNormalization` | – | 批次歸一化 |
| `model.compile()` | `optimizer`, `loss`, `metrics` | 設定訓練配置 |
| `model.fit()` | `epochs`, `batch_size`, `validation_split`, `callbacks` | 執行訓練 |
| `model.evaluate()` | – | 評估測試集 |
| `model.predict()` | – | 取得預測 |
| `model.save("model.keras")` | – | 儲存完整模型 |
| `tf.keras.models.load_model()` | – | 載入模型 |
| `EarlyStopping` | `patience=10`, `restore_best_weights=True` | 提前停止 |
| `ModelCheckpoint` | `save_best_only=True` | 儲存最佳 checkpoint |
| `TensorBoard` | `log_dir="logs/"` | TensorBoard 日誌 |


---

[來源: ch10 | 類型: cheatsheet] | `save_best_only=True` | 儲存最佳 checkpoint |
| `TensorBoard` | `log_dir="logs/"` | TensorBoard 日誌 |


---

---

[來源: ch10 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import tensorflow as tf

---

[來源: ch10 | 類型: cheatsheet] # Sequential API（最常用）
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[28, 28]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

---

[來源: ch10 | 類型: cheatsheet] ers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir="logs/")
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks
)

---

[來源: ch10 | 類型: cheatsheet] # Functional API（多輸入/多輸出）
input_ = tf.keras.layers.Input(shape=[28, 28])
hidden1 = tf.keras.layers.Dense(300, activation="relu")(input_)
hidden2 = tf.keras.layers.Dense(100, activation="relu")(hidden1)
output = tf.keras.layers.Dense(10, activation="softmax")(hidden2)
model_func = tf.keras.Model(inputs=input_, outputs=output)

---

[來源: ch10 | 類型: cheatsheet] # Subclassing API（研究用途）
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(300, activation="relu")
        self.dense2 = tf.keras.layers.Dense(100, activation="relu")
        self.out = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

---

[來源: ch10 | 類型: cheatsheet] # 儲存與載入
model.save("my_model.keras")
loaded_model = tf.keras.models.load_model("my_model.keras")
```

---

---

[來源: ch10 | 類型: cheatsheet] ## 4. 常見陷阱

- **`loss="sparse_categorical_crossentropy"` vs `"categorical_crossentropy"`**：標籤是整數 → `sparse`；標籤是 one-hot → 不加 `sparse`。
- **`validation_split` 是從資料尾端切**：若資料有時間順序必須手動 shuffle 或用 `validation_data`。
- **`EarlyStopping(restore_best_weights=True)`**：不加這個選項，停止時的模型不是最佳的。
- **`BatchNormalization` 在 `training=True/False` 的行為不同**：`model.fit()` 自動處理，但在自訂訓練迴圈中要手動傳。
- **`model.save()` 的格式**：推薦 `.keras` 格式（新版），舊版 SavedModel 格式用 `model.save("dir/")`。

---

---

[來源: ch10 | 類型: cheatsheet] ```
選哪種 Keras API？
├── 標準架構（MLP/CNN/RNN）   → Sequential
├── 多輸入/多輸出/殘差連接    → Functional API
└── 非標準前向傳播邏輯        → Subclassing API

Loss Function 選擇：
├── 二元分類      → binary_crossentropy
├── 多類別分類    → sparse_categorical_crossentropy（整數標籤）
├── 連續值迴歸    → mse 或 mae（有 outlier 用 mae）
└── 多標籤分類    → binary_crossentropy（每個輸出獨立 sigmoid）

Optimizer 建議：
├── 通用起點   → Adam(learning_rate=1e-3)
├── Fine-tuning → SGD(learning_rate=1e-4, momentum=0.9)
└── 研究       → AdamW 或 SGD with cosine annealing
```

---

[來源: ch10 | 類型: handout] # 課程講義：人工神經網路簡介與 Keras 實作 (Chapter 10)

---

[來源: ch10 | 類型: handout] ## 導論

各位同學大家好！歡迎來到本章課程，我們將深入探討人工神經網路 (Artificial Neural Networks, ANNs) 的世界。本章旨在為大家介紹 ANNs 的基本概念、它們如何從生物神經元中汲取靈感，並透過業界標準的 Keras 函式庫，實際建構、訓練與評估用於分類和迴歸任務的類神經網路模型。我們將涵蓋感知器、激活函數、多層感知器 (MLP) 的實作，以及模型儲存、回呼函數和 TensorBoard 視覺化等進階主題，最後探討超參數調優的策略。

透過本章的學習，您將能夠理解神經網路的運作原理，並具備使用 Keras 解決實際機器學習問題的能力。

---

[來源: ch10 | 類型: handout] ### 為什麼重要？

人工神經網路的靈感源於人腦中的生物神經元，理解這些生物基礎有助於我們掌握 ANNs 的核心建構模塊。本節將探討感知器和激活函數等基本概念，為建構更複雜的神經網路模型奠定基礎。

---

[來源: ch10 | 類型: handout] ### 如何運作？

人工神經網路的核心組件是人工神經元，它們模擬生物神經元的資訊處理方式。

* **感知器 (The Perceptron)**
    感知器是最簡單的人工神經元形式，它接收多個二元輸入 $x_i$，為每個輸入分配權重 $w_i$，然後將加權輸入求和，加上一個偏差項 $b$，最後通過一個激活函數（通常是步階函數）產生二元輸出 ($y=0$ 或 $1$)。

---

[來源: ch10 | 類型: handout] 神經元形式，它接收多個二元輸入 $x_i$，為每個輸入分配權重 $w_i$，然後將加權輸入求和，加上一個偏差項 $b$，最後通過一個激活函數（通常是步階函數）產生二元輸出 ($y=0$ 或 $1$)。

數學上，對於輸入 $x_1, x_2, \dots, x_n$ 和權重 $w_1, w_2, \dots, w_n$，輸出 $y$ 為：
    $$
    y = \begin{cases} 
    1 & \text{if } \sum_{i=1}^n w_i x_i + b > 0 \\
    0 & \text{otherwise}
    \end{cases}
    $$
    感知器可以透過訓練來學習線性可分離的模式，但無法處理非線性問題，例如 XOR 問題。

---

[來源: ch10 | 類型: handout] 0 \\
    0 & \text{otherwise}
    \end{cases}
    $$
    感知器可以透過訓練來學習線性可分離的模式，但無法處理非線性問題，例如 XOR 問題。

* **激活函數 (Activation Functions)**
    激活函數引入了非線性，它將層的預激活值（加權和加上偏差）轉換為其輸出。這種非線性使得深度網路能夠學習複雜的非線性模式。常見的激活函數包括：

---

[來源: ch10 | 類型: handout] 數 (Activation Functions)**
    激活函數引入了非線性，它將層的預激活值（加權和加上偏差）轉換為其輸出。這種非線性使得深度網路能夠學習複雜的非線性模式。常見的激活函數包括：

* **步階 (Heaviside) 函數**: 根據輸入是否超過閾值輸出 0 或 1，用於早期的感知器。
  * **Sigmoid 函數**: 將輸入映射到 (0, 1) 區間，適用於二元分類，但可能存在梯度飽和問題。
  * **雙曲正切 (tanh) 函數**: 將輸入映射到 (-1, 1) 區間，零中心化，但同樣可能飽和。
  * **整流線性單元 (Rectified Linear Unit, ReLU)**: 輸出 $\max(0, x)$。有助於緩解梯度消失問題，但可能導致「死亡」神經元。
  * **Leaky ReLU / ELU**: ReLU 的變體，在單元不活躍時

---

[來源: ch10 | 類型: handout]  Unit, ReLU)**: 輸出 $\max(0, x)$。有助於緩解梯度消失問題，但可能導致「死亡」神經元。
  * **Leaky ReLU / ELU**: ReLU 的變體，在單元不活躍時允許一個小的非零梯度，有助於防止死亡神經元。
  * **Softmax 函數**: 用於多類分類的輸出層，將 logits 轉換為機率分佈。
  * **線性激活 (Linear activation)**: 用於迴歸輸出層，直接傳遞輸入。

每個激活函數都有不同的特性，會影響梯度的流動和學習動態。

---

[來源: ch10 | 類型: handout] ### 補充練習 1：

* **理論題**: 比較傳統感知器與羅吉斯迴歸分類器在收斂性、機率估計能力以及處理非線性問題上的差異。在何種情況下，兩者可以被視為等價？
* **實作題**: 參考 `10_neural_nets_with_keras.ipynb` 中感知器的範例程式碼。嘗試將感知器中的激活函數替換為 Sigmoid 函數，並說明這對模型的輸出和決策邊界可能產生什麼影響（無需實際運行，只需概念性說明）。

---

[來源: ch10 | 類型: handout] ### 為什麼重要？

Keras 提供了一個高階且易於使用的 API，可以快速建構、訓練和評估神經網路。本節將展示如何利用 Keras 實作用於影像分類和迴歸的多層感知器，並介紹 Sequential API、Functional API 和 Subclassing API。

---

[來源: ch10 | 類型: handout] ### 如何運作？

Keras 是一個強大的深度學習函式庫，支援多種神經網路架構的建立。

* **建立影像分類器 (Building an Image Classifier Using the Sequential API)**
    我們將使用 Fashion MNIST 資料集來展示影像分類的端到端工作流程。

---

[來源: ch10 | 類型: handout] Building an Image Classifier Using the Sequential API)**
    我們將使用 Fashion MNIST 資料集來展示影像分類的端到端工作流程。

1. **資料準備**: 載入 Fashion MNIST 資料集，並將訓練集劃分為訓練集和驗證集。將像素強度縮放到 0-1 範圍以進行正規化。
    2. **模型建構 (Sequential API)**: 使用 `tf.keras.Sequential` 建立模型，層層堆疊：
        * `tf.keras.layers.InputLayer`: 定義輸入形狀 (例如 28x28 影像)。
        * `tf.keras.layers.Flatten()`: 將 2D 輸入轉換為 1D 向量。
        * `tf.keras.layers.Dense(uni

---

[來源: ch10 | 類型: handout] 8 影像)。
        * `tf.keras.layers.Flatten()`: 將 2D 輸入轉換為 1D 向量。
        * `tf.keras.layers.Dense(units, activation="relu")`: 全連接層，使用 ReLU 激活。
        * `tf.keras.layers.Dense(10, activation="softmax")`: 輸出層，10 個類別使用 Softmax 激活函數。
    3. **模型編譯**: 使用 `model.compile()` 配置學習過程，指定損失函數 (例如 `sparse_categorical_crossentropy`)、優化器 (例如 `"sgd"`) 和評估指標 (例如 `"accuracy"`)。
    4. **模型訓練與評估**: 使用 `model.fit()` 

---

[來源: ch10 | 類型: handout] rical_crossentropy`)、優化器 (例如 `"sgd"`) 和評估指標 (例如 `"accuracy"`)。
    4. **模型訓練與評估**: 使用 `model.fit()` 訓練模型，並使用 `model.evaluate()` 評估模型在測試集上的性能。

---

[來源: ch10 | 類型: handout] 指標 (例如 `"accuracy"`)。
    4. **模型訓練與評估**: 使用 `model.fit()` 訓練模型，並使用 `model.evaluate()` 評估模型在測試集上的性能。

* **建立迴歸多層感知器 (Building a Regression MLP Using the Sequential API)**
    針對迴歸任務 (例如預測房價)，我們同樣可以使用 Sequential API 建立 MLP。
    1. **資料準備**: 載入 California Housing 資料集，並進行訓練集、驗證集和測試集的劃分。使用 `tf.keras.layers.Normalization` 或 `StandardScaler` 對特徵進行標準化。
    2. **模型建構**: 定義幾個 `Dense` 隱藏層 (使用 ReLU 激活) 和一個單一

---

[來源: ch10 | 類型: handout] ers.Normalization` 或 `StandardScaler` 對特徵進行標準化。
    2. **模型建構**: 定義幾個 `Dense` 隱藏層 (使用 ReLU 激活) 和一個單一線性輸出神經元。
    3. **模型編譯**: 使用迴歸損失函數 (例如 `"mse"`) 和優化器 (例如 `tf.keras.optimizers.Adam`)，並監控 RMSE。
    4. **模型訓練與評估**: 訓練模型並評估其在測試集上的性能。

---

[來源: ch10 | 類型: handout] 函數 (例如 `"mse"`) 和優化器 (例如 `tf.keras.optimizers.Adam`)，並監控 RMSE。
    4. **模型訓練與評估**: 訓練模型並評估其在測試集上的性能。

* **使用 Functional API 建立複雜模型 (Building Complex Models Using the Functional API)**
    對於具有複雜拓撲或多輸入/多輸出的模型，Keras Functional API 提供更大的彈性。我們將建立一個 "Wide & Deep" 迴歸模型：
  * **多輸入**: 定義多個 `tf.keras.layers.Input` 層，每個輸入處理不同的特徵子集。
  * **分支路徑**: 建立「寬路徑」(直接連接輸入到輸出) 和「深路徑」(包含多個隱藏層)。
  * **合併**: 使用 `tf.keras.l

---

[來源: ch10 | 類型: handout] Input` 層，每個輸入處理不同的特徵子集。
  * **分支路徑**: 建立「寬路徑」(直接連接輸入到輸出) 和「深路徑」(包含多個隱藏層)。
  * **合併**: 使用 `tf.keras.layers.Concatenate()` 合併不同路徑的輸出。
  * **多輸出**: 為主輸出和輔助輸出定義不同的輸出層，輔助輸出可用於正規化。
  * **模型建構**: 使用 `tf.keras.Model(inputs=[...], outputs=[...])` 建立模型。

---

[來源: ch10 | 類型: handout] : 為主輸出和輔助輸出定義不同的輸出層，輔助輸出可用於正規化。
  * **模型建構**: 使用 `tf.keras.Model(inputs=[...], outputs=[...])` 建立模型。

* **使用 Subclassing API 建立動態模型 (Using the Subclassing API to Build Dynamic Models)**
    當模型架構需要動態行為或高度客製化時，可以透過繼承 `tf.keras.Model` 並實作 `call()` 方法來建立自定義模型。
  * **`__init__` 方法**: 初始化模型中使用的所有層。
  * **`call()` 方法**: 定義模型的前向傳播邏輯，處理輸入並產生輸出。

---

[來源: ch10 | 類型: handout] ### 補充練習 2：

* **理論題**: 解釋 `model.compile()` 方法中 `loss`、`optimizer` 和 `metrics` 這三個參數的作用，以及在分類和迴歸任務中，如何根據資料類型和任務目標選擇合適的選項。
* **實作題**: 參考 Functional API 建立 Wide & Deep 模型的範例。設計並實作一個新的 Functional API 模型，該模型具有兩個輸入，其中一個輸入經過一個隱藏層，然後與另一個原始輸入連接，最終產生一個輸出。

---

[來源: ch10 | 類型: handout] ### 為什麼重要？

在訓練完神經網路模型後，我們需要將其儲存起來以便日後重新載入進行預測，或繼續訓練。模型儲存確保了模型的可重用性和持久性。

---

[來源: ch10 | 類型: handout] ### 如何運作？

Keras 提供多種儲存和恢復模型的方法。

* **儲存與載入整個模型 (Using the .keras format)**
    Keras 建議使用 `.keras` 格式儲存整個模型，它包含模型的架構、權重和優化器狀態。
  * **儲存**: `model.save("my_model.keras")`
  * **載入**: `loaded_model = tf.keras.models.load_model("my_model.keras", custom_objects={"WideAndDeepModel": WideAndDeepModel})` (如果模型包含自定義層或模型，需提供 `custom_objects`)

---

[來源: ch10 | 類型: handout] keras", custom_objects={"WideAndDeepModel": WideAndDeepModel})` (如果模型包含自定義層或模型，需提供 `custom_objects`)

* **儲存與載入模型權重 (Saving and loading model weights)**
    如果只需要儲存模型的權重，可以使用 HDF5 格式的 `.weights.h5` 擴展名。
  * **儲存**: `model.save_weights("my_weights.weights.h5")`
  * **載入**: `model.load_weights("my_weights.weights.h5")` (需先建立模型架構)

---

[來源: ch10 | 類型: handout] ### 補充練習 3：

* **理論題**: 說明儲存整個 Keras 模型 (`.keras` 格式) 與只儲存模型權重 (`.weights.h5` 格式) 之間的主要差異。在何種情境下，你會選擇只儲存權重而非整個模型？
* **實作題**: 假設你已經訓練好一個 Keras Sequential 模型。請撰寫程式碼片段，展示如何將這個模型的權重儲存到檔案中，然後建立一個新的相同架構的模型，並從檔案中載入之前儲存的權重。

---

[來源: ch10 | 類型: handout] ### 為什麼重要？

回呼函數 (Callbacks) 是 Keras 提供的一種強大機制，可以在訓練過程中的不同階段執行自定義操作，例如在每個 epoch 結束時。它們對於自動化訓練流程、監控模型性能和防止過度擬合至關重要。

---

[來源: ch10 | 類型: handout] ### 如何運作？

回呼函數透過繼承 `tf.keras.callbacks.Callback` 類別並覆寫其方法來實作。

* **模型檢查點 (Model Checkpointing)**
    `tf.keras.callbacks.ModelCheckpoint` 回呼函數允許您在訓練期間定期儲存模型的權重或整個模型。
  * `filepath`: 儲存路徑 (例如 `"my_checkpoints.weights.h5"`)。
  * `save_weights_only=True`: 只儲存權重。
  * `save_best_only=True`: 只儲存驗證集上性能最好的模型。
  * `monitor`: 監控的指標 (例如 `"val_loss"`)。

---

[來源: ch10 | 類型: handout] nly=True`: 只儲存權重。
  * `save_best_only=True`: 只儲存驗證集上性能最好的模型。
  * `monitor`: 監控的指標 (例如 `"val_loss"`)。

* **提早停止 (Early Stopping)**
    `tf.keras.callbacks.EarlyStopping` 回呼函數用於在模型性能不再提升時停止訓練，從而防止過度擬合。
  * `patience`: 在多少個 epoch 內性能沒有改善就停止訓練。
  * `restore_best_weights=True`: 訓練結束時恢復到性能最好的 epoch 的權重。

---

[來源: ch10 | 類型: handout] 。
  * `patience`: 在多少個 epoch 內性能沒有改善就停止訓練。
  * `restore_best_weights=True`: 訓練結束時恢復到性能最好的 epoch 的權重。

* **自定義回呼函數 (Custom Callbacks)**
    您可以透過繼承 `tf.keras.callbacks.Callback` 來自定義回呼函數，例如在每個 epoch 結束時印出訓練/驗證損失比率。
  * 覆寫 `on_epoch_end(self, epoch, logs)` 等方法來實作自定義邏輯。

---

[來源: ch10 | 類型: handout] ### 補充練習 4：

* **理論題**: 解釋 `tf.keras.callbacks.EarlyStopping` 中 `patience` 和 `restore_best_weights` 這兩個參數的具體作用，以及它們如何協同工作來防止模型過度擬合。
* **實作題**: 撰寫一個自定義 Keras 回呼函數 `LearningRateLogger`，它應在每個訓練批次 (batch) 結束時記錄並印出當前學習率。然後將此回呼函數添加到一個簡單模型的訓練過程中。

---

[來源: ch10 | 類型: handout] ### 為什麼重要？

TensorBoard 是 TensorFlow 和 Keras 內建的強大視覺化工具。它允許開發者監控和視覺化模型訓練的各個方面，例如損失和準確度曲線、權重和偏差的直方圖以及模型圖。透過將 TensorBoard 整合到訓練流程中，可以深入了解模型的學習過程並識別潛在問題。

---

[來源: ch10 | 類型: handout] ### 如何運作？

TensorBoard 透過在訓練期間記錄事件檔案來運作，這些檔案包含訓練指標、圖形定義和其他視覺化資料。

* **TensorBoard 回呼函數 (TensorBoard Callback)**
    `tf.keras.callbacks.TensorBoard` 回呼函數用於在訓練期間自動生成 TensorBoard 日誌。
  * `log_dir`: 指定日誌檔案的儲存目錄。
  * `profile_batch`: 啟用指定批次範圍的性能分析。

---

[來源: ch10 | 類型: handout] ard` 回呼函數用於在訓練期間自動生成 TensorBoard 日誌。
  * `log_dir`: 指定日誌檔案的儲存目錄。
  * `profile_batch`: 啟用指定批次範圍的性能分析。

* **啟動 TensorBoard 伺服器 (Launching TensorBoard Server)**
    在訓練結束後，您可以使用命令或 Jupyter 筆記本中的 `%tensorboard` 魔術指令來啟動 TensorBoard 伺服器，並在瀏覽器中查看視覺化結果。
  * `%tensorboard --logdir=./my_logs`

---

[來源: ch10 | 類型: handout] ### 補充練習 5：

* **理論題**: 除了追蹤損失和準確度曲線外，請列舉 TensorBoard 提供的至少三種其他有用的視覺化功能，並簡要說明它們能幫助我們分析模型的哪些方面。
* **實作題**: 修改 TensorBoard 回呼函數的 `profile_batch` 參數，使其對模型訓練的前 50 個批次進行性能分析。說明你預期透過這個設定能夠觀察到什麼樣的性能數據。

---

[來源: ch10 | 類型: handout] ### 為什麼重要？

超參數調優是優化神經網路性能的關鍵步驟。超參數是控制模型訓練過程和架構的設定，例如學習率、批次大小、層數、每層神經元數量、激活函數和正規化技術。適當地調整這些超參數可以顯著影響模型的學習能力和泛化能力。

---

[來源: ch10 | 類型: handout] ### 如何運作？

Keras Tuner 是一個函式庫，專門用於自動化超參數調優過程。

---

[來源: ch10 | 類型: handout] * **Keras Tuner 設定與搜尋策略 (Keras Tuner setup and search strategy)**
  * **`build_model(hp)` 函數**: 定義一個模型建構函數，它接受一個 `kt.HyperParameters` 物件 (`hp`)，並根據 `hp` 中定義的超參數 (例如 `n_hidden`, `n_neurons`, `learning_rate`, `optimizer`) 來建構 Keras 模型。
  * **`kt.RandomSearch`**: 一種隨機搜索策略，在給定的超參數空間中隨機採樣組合。
    * `objective`: 優化目標 (例如 `"val_accuracy"`)。
    * `max_trials`: 最大嘗試次數。
    * `overwrite=True`: 覆蓋之前的搜索結果。

---

[來源: ch10 | 類型: handout] jective`: 優化目標 (例如 `"val_accuracy"`)。
    * `max_trials`: 最大嘗試次數。
    * `overwrite=True`: 覆蓋之前的搜索結果。

* **Hyperband 演算法 (Hyperband)**
    Hyperband 是一種資源感知型的超參數優化算法，它結合了隨機搜索和自適應早期停止。它以較小的預算評估許多配置，並逐漸將更多資源分配給最有潛力的配置，從而避免在較差的候選方案上浪費計算資源。
  * `hypermodel`: 您的自定義 `HyperModel` (用於建構模型並處理預處理)。
  * `max_epochs`: 任何候選模型訓練的最大 epoch 數。
  * `factor`: 每次連續減半的縮減因子。

---

[來源: ch10 | 類型: handout] 您的自定義 `HyperModel` (用於建構模型並處理預處理)。
  * `max_epochs`: 任何候選模型訓練的最大 epoch 數。
  * `factor`: 每次連續減半的縮減因子。

* **貝氏優化 (Bayesian Optimization)**
    貝氏優化器使用機率模型來引導超參數搜索。它根據過去的試驗建立目標函數的替代模型 (例如驗證準確度)，並使用該模型選擇接下來要評估的有潛力的超參數配置。這種方法平衡了對超參數空間新區域的探索與對已知良好區域的利用。
  * `alpha`: 控制高斯過程的先驗/噪聲項。
  * `beta`: 控制採集函數中探索與利用的權衡。

---

[來源: ch10 | 類型: handout] 估的有潛力的超參數配置。這種方法平衡了對超參數空間新區域的探索與對已知良好區域的利用。
  * `alpha`: 控制高斯過程的先驗/噪聲項。
  * `beta`: 控制採集函數中探索與利用的權衡。

* **超參數調優的實用技巧 (Practical tips for hyperparameter tuning)**
  * **逐步優化**: 從粗略搜索 (例如 `RandomSearch`) 開始，然後使用 `BayesianOptimization` 精煉有潛力的區域。
  * **增加資源**: 逐漸增加 `max_trials` 或 `epochs`。
  * **監控**: 使用 `TensorBoard` 監控訓練過程以檢測過度擬合。
  * **回呼函數**: 在搜索過程中應用 `EarlyStopping` 和 `ModelCheckpoint`。
  * **重現性**: 設定隨機種子和目錄以確保實驗的可重現性。

---

[來源: ch10 | 類型: handout] ### 補充練習 6：

* **理論題**: 比較 Keras Tuner 中 `RandomSearch`、`Hyperband` 和 `BayesianOptimization` 這三種超參數調優方法的優缺點，並說明它們各自最適合在什麼樣的場景下使用。
* **實作題**: 選擇一個你認為最適合 MNIST 分類任務的 Keras Tuner 策略 (例如 `Hyperband` 或 `BayesianOptimization`)，設定其參數，並重新執行 MNIST 資料集的超參數調優過程。完成後，印出最佳的超參數組合。

---

[來源: ch10 | 類型: handout] ## 課後作業

請完成 `10_neural_nets_with_keras.ipynb` 筆記本中的第 10 題練習：

* **練習**：在 MNIST 資料集上訓練一個深度 MLP（您可以使用 `tf.keras.datasets.mnist.load_data()` 載入資料集）。嘗試透過手動調整超參數來達到超過 98% 的準確度。嘗試使用本章介紹的方法（即通過指數級增長學習率、繪製損失圖並找到損失飆升的點）來搜尋最佳學習率。接下來，嘗試使用 Keras Tuner 並搭配所有功能——儲存檢查點、使用提早停止以及使用 TensorBoard 繪製學習曲線。

請務必將您的程式碼、訓練結果和對學習率、超參數調優策略選擇的說明，以及您從 TensorBoard 視覺化中獲得的洞見，一併提交。

---

[來源: ch10 | 類型: tutorial] [標題: 使用 Keras 學習人工神經網路：完整教學指南 | 描述: 從生物神經元到深度學習的完整 Keras 神經網路教學，包含實作範例、逐行解析與最佳實踐。 | 關鍵字: Keras, 神經網路, 人工智慧, 機器學習, 深度學習, 教學, Python]
# 使用 Keras 介紹人工神經網路

這份教學基於《Hands-On Machine Learning》第 3 版第 10 章，完整介紹人工神經網路的基本概念與 Keras 實作。從生物神經元開始，逐步學習感知器、多層感知器（MLP）、激活函數，以及使用 Keras 的三種 API 建構神經網路模型。

---

[來源: ch10 | 類型: tutorial] ## 關鍵重點
- 人工神經網路模仿生物神經系統，透過層層神經元處理資訊
- Keras 提供三種建構模型的 API：Sequential、Functional、Subclassing
- 激活函數決定神經元的輸出，非線性函數讓網路能學習複雜模式
- 反向傳播演算法透過梯度下降優化網路權重
- 正規化技巧如 Dropout 能防止過擬合

---

[來源: ch10 | 類型: tutorial] ## 從生物神經元到人工神經網路

生物神經元是神經系統的基本單位，接收來自其他神經元的訊號，當累積訊號超過閾值時會發射訊號。人工神經網路模仿這個概念，建構出能學習複雜模式的計算模型。

---

[來源: ch10 | 類型: tutorial] ### 範例 1: 檢查 Python 與套件版本

```python
import sys

---

[來源: ch10 | 類型: tutorial] # 確保 Python 版本至少 3.7
assert sys.version_info >= (3, 7)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 匯入 sys 模組，用於檢查系統資訊
2. `assert sys.version_info >= (3, 7)`: 斷言 Python 版本必須大於等於 3.7，否則拋出錯誤

**🎯 重點摘要:**

- **核心功能**: 版本檢查確保程式相容性
- **潛在問題**: 舊版 Python 可能缺少某些功能
- **最佳使用情境**: 在程式開始時進行環境檢查

---

[來源: ch10 | 類型: tutorial] ### 範例 2: 檢查 Scikit-Learn 版本

```python
import sklearn

---

[來源: ch10 | 類型: tutorial] # 檢查 Scikit-Learn 版本
print(sklearn.__version__)
```

**✅ 程式碼逐行解析：**

1. `import sklearn`: 匯入 scikit-learn 機器學習套件
2. `print(sklearn.__version__)`: 印出套件版本號

**🎯 重點摘要:**

- **核心功能**: 確認機器學習套件版本
- **潛在問題**: 不同版本 API 可能有差異
- **最佳使用情境**: 除錯時確認套件版本

---

[來源: ch10 | 類型: tutorial] ## 感知器

感知器是最簡單的人工神經網路，由 Frank Rosenblatt 在 1957 年提出。它是一個單層神經網路，能學習線性可分的模式。

---

[來源: ch10 | 類型: tutorial] ### 範例 3: 使用 Scikit-Learn 實作感知器

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

---

[來源: ch10 | 類型: tutorial] # 載入鳶尾花資料集
iris = load_iris()
X = iris.data[:, (2, 3)]  # 花瓣長度和寬度
y = (iris.target == 0).astype(int)  # 是否為 Setosa

---

[來源: ch10 | 類型: tutorial] # 建立感知器模型
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

---

[來源: ch10 | 類型: tutorial] # 預測
y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)
```

**✅ 程式碼逐行解析：**

1. `import numpy as np`: 匯入 NumPy 用於數值運算
2. `from sklearn.datasets import load_iris`: 載入鳶尾花資料集
3. `from sklearn.linear_model import Perceptron`: 匯入感知器類別
4. `iris = load_iris()`: 載入資料集
5. `X = iris.data[:, (2, 3)]`: 選取花瓣特徵
6. `y = (iris.target == 0).astype(int)`: 建立二元分類標籤
7. `per_clf = Perceptron(random_state=42)`: 建立感知器模型
8. `per_clf.fit(X, y)`: 訓練模型
9. `y_pred = per_clf.predict([[2, 0.5]])`: 預測新樣本
10. `print(y_pred)`: 印出預測結果

**🎯 重點摘要:**

- **核心功能**: 實作單層神經網路進行二元分類
- **潛在問題**: 只能學習線性可分問題
- **最佳使用情境**: 簡單分類任務的基準模型

---

[來源: ch10 | 類型: tutorial] ## 多層感知器與反向傳播

多層感知器（MLP）由多層神經元組成，能學習非線性模式。反向傳播演算法透過計算梯度來更新權重。

---

[來源: ch10 | 類型: tutorial] ### 範例 4: 實作多層感知器

```python
from sklearn.neural_network import MLPClassifier

---

[來源: ch10 | 類型: tutorial] # 建立 MLP 分類器
mlp_clf = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', 
                       solver='adam', random_state=42, max_iter=1000)
mlp_clf.fit(X, y)

---

[來源: ch10 | 類型: tutorial] # 預測
y_pred = mlp_clf.predict([[2, 0.5]])
print(y_pred)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.neural_network import MLPClassifier`: 匯入多層感知器
2. `mlp_clf = MLPClassifier(...)`: 設定隱藏層大小、激活函數等參數
3. `mlp_clf.fit(X, y)`: 訓練模型
4. `y_pred = mlp_clf.predict([[2, 0.5]])`: 預測
5. `print(y_pred)`: 輸出結果

**🎯 重點摘要:**

- **核心功能**: 使用多層網路學習非線性模式
- **潛在問題**: 訓練時間較長，可能過擬合
- **最佳使用情境**: 非線性分類與迴歸任務

---

[來源: ch10 | 類型: tutorial] ## 使用 Keras 建構神經網路

Keras 是高階神經網路 API，提供三種建構模型的方式：Sequential、Functional、Subclassing。

---

[來源: ch10 | 類型: tutorial] ### 範例 5: 使用 Sequential API 建構模型

```python
import tensorflow as tf
from tensorflow import keras

---

[來源: ch10 | 類型: tutorial] # 建立 Sequential 模型
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch10 | 類型: tutorial] # 編譯模型
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
```

**✅ 程式碼逐行解析：**

1. `import tensorflow as tf`: 匯入 TensorFlow
2. `from tensorflow import keras`: 匯入 Keras
3. `model = keras.models.Sequential([...])`: 建立序列模型
4. `keras.layers.Flatten(input_shape=[28, 28])`: 攤平輸入層
5. `keras.layers.Dense(300, activation="relu")`: 全連接層
6. `model.compile(...)`: 設定損失函數、優化器、評估指標

**🎯 重點摘要:**

- **核心功能**: 簡單直覺的模型建構方式
- **潛在問題**: 無法處理複雜架構如多輸入輸出
- **最佳使用情境**: 標準前饋網路

---

[來源: ch10 | 類型: tutorial] ### 範例 6: 使用 Functional API

```python

---

[來源: ch10 | 類型: tutorial] # 建立 Functional 模型
input_ = keras.layers.Input(shape=[28, 28])
flatten = keras.layers.Flatten(input_shape=[28, 28])(input_)
hidden1 = keras.layers.Dense(300, activation="relu")(flatten)
hidden2 = keras.layers.Dense(100, activation="relu")(hidden1)
output = keras.layers.Dense(10, activation="softmax")(hidden2)
model = keras.models.Model(inputs=[input_], outputs=[output])
```

**✅ 程式碼逐行解析：**

---

[來源: ch10 | 類型: tutorial] softmax")(hidden2)
model = keras.models.Model(inputs=[input_], outputs=[output])
```

**✅ 程式碼逐行解析：**

1. `input_ = keras.layers.Input(shape=[28, 28])`: 定義輸入
2. `flatten = keras.layers.Flatten(...)(input_)`: 連接層
3. `hidden1 = keras.layers.Dense(...)(flatten)`: 第一隱藏層
4. `model = keras.models.Model(...)`: 建立模型物件

**🎯 重點摘要:**

- **核心功能**: 靈活的模型架構設計
- **潛在問題**: 語法較複雜
- **最佳使用情境**: 複雜網路如多分支或殘差網路

---

[來源: ch10 | 類型: tutorial] ### 範例 7: 使用 Subclassing API

```python
class MyModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(300, activation="relu")
        self.dense2 = keras.layers.Dense(100, activation="relu")
        self.dense3 = keras.layers.Dense(10, activation="softmax")
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

model = MyModel()
```

**✅ 程式碼逐行解析：**

---

[來源: ch10 | 類型: tutorial] 1(x)
        x = self.dense2(x)
        return self.dense3(x)

model = MyModel()
```

**✅ 程式碼逐行解析：**

1. `class MyModel(keras.models.Model)`: 繼承 Model 類別
2. `def __init__(self)`: 初始化層
3. `def call(self, inputs)`: 定義前向傳播
4. `model = MyModel()`: 實例化模型

**🎯 重點摘要:**

- **核心功能**: 最大靈活性，自訂邏輯
- **潛在問題**: 較難除錯，程式碼較長
- **最佳使用情境**: 研究與自訂網路

---

[來源: ch10 | 類型: tutorial] ## 模型儲存與載入

訓練好的模型可以儲存為 HDF5 格式或 SavedModel 格式。

---

[來源: ch10 | 類型: tutorial] # 儲存模型
model.save("my_model.h5")

---

[來源: ch10 | 類型: tutorial] # 載入模型
loaded_model = keras.models.load_model("my_model.h5")
```

**✅ 程式碼逐行解析：**

1. `model.save("my_model.h5")`: 儲存為 HDF5 格式
2. `loaded_model = keras.models.load_model("my_model.h5")`: 載入模型

**🎯 重點摘要:**

- **核心功能**: 模型持久化
- **潛在問題**: HDF5 格式即將棄用
- **最佳使用情境**: 模型部署與重用

---

[來源: ch10 | 類型: tutorial] ## 訓練神經網路

使用回呼函數（callbacks）可以監控訓練過程並實作早期停止等技巧。

---

[來源: ch10 | 類型: tutorial] # 定義回呼
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_model.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

---

[來源: ch10 | 類型: tutorial] # 訓練模型
history = model.fit(X_train, y_train, epochs=100,
                   validation_data=(X_valid, y_valid),
                   callbacks=[checkpoint_cb, early_stopping_cb])
```

**✅ 程式碼逐行解析：**

1. `checkpoint_cb = keras.callbacks.ModelCheckpoint(...)`: 儲存最佳模型
2. `early_stopping_cb = keras.callbacks.EarlyStopping(...)`: 早期停止
3. `model.fit(...)`: 訓練模型並使用回呼

**🎯 重點摘要:**

- **核心功能**: 自動化訓練過程管理
- **潛在問題**: 需要適當設定 patience 參數
- **最佳使用情境**: 防止過擬合與最佳化訓練

---

[來源: ch10 | 類型: tutorial] ## 視覺化與除錯

TensorBoard 提供強大的視覺化功能來監控訓練過程。

---

[來源: ch10 | 類型: tutorial] ### 範例 10: 使用 TensorBoard

```python
import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

---

[來源: ch10 | 類型: tutorial] # 訓練時使用 TensorBoard 回呼
history = model.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid),
                   callbacks=[tensorboard_cb])
```

**✅ 程式碼逐行解析：**

1. `root_logdir = os.path.join(os.curdir, "my_logs")`: 設定日誌目錄
2. `def get_run_logdir()`: 產生唯一執行目錄
3. `tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)`: 建立 TensorBoard 回呼
4. `model.fit(..., callbacks=[tensorboard_cb])`: 訓練時記錄日誌

**🎯 重點摘要:**

- **核心功能**: 即時視覺化訓練指標
- **潛在問題**: 日誌檔案可能佔用大量空間
- **最佳使用情境**: 模型開發與除錯

---

[來源: ch10 | 類型: tutorial] ## 超參數調校

使用 Scikit-Learn 的 RandomizedSearchCV 進行超參數搜尋。

---

[來源: ch10 | 類型: tutorial] ```python
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[28, 28]):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
```

---

[來源: ch10 | 類型: tutorial] ta=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
```

**✅ 程式碼逐行解析：**

1. `def build_model(...)`: 定義模型建構函數
2. `keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)`: 包裝為 Scikit-Learn 分類器
3. `param_distribs = {...}`: 定義參數分佈
4. `rnd_search_cv = RandomizedSearchCV(...)`: 建立隨機搜尋物件
5. `rnd_search_cv.fit(...)`: 執行超參數調校

**🎯 重點摘要:**

- **核心功能**: 自動化超參數最佳化
- **潛在問題**: 計算成本高，需要大量時間
- **最佳使用情境**: 模型效能最佳化

---

[來源: ch10 | 類型: tutorial] ## 常見問答

**Q: Sequential API 與 Functional API 的差異？**
A: Sequential API 適合簡單的線性堆疊架構，而 Functional API 能處理複雜的網路結構如多輸入輸出或共享層。

**Q: 如何避免過擬合？**
A: 使用 Dropout、早期停止、正規化技巧，或增加訓練資料。

**Q: 激活函數選擇的考量？**
A: ReLU 適合隱藏層避免梯度消失，Softmax 適用於多分類輸出層。

---

[來源: ch10 | 類型: tutorial] ## 推薦標籤

#Keras #神經網路 #人工智慧 #機器學習 #深度學習 #Python教學 #程式設計 #TensorFlow #AI #資料科學

---

[來源: ch11] [標題: 訓練深度神經網路：解決梯度消失與爆炸問題的實戰指南 | 描述: 深入探討訓練深度神經網路的關鍵技術，包括初始化方法、激活函數、批次正規化、優化器與學習率調度。透過實戰範例學習如何建構穩定、高效能的深度學習模型。 | 關鍵字: Python, 深度學習, 神經網路, 梯度消失, 批次正規化, Adam優化器, 學習率調度, 程式設計, 教學]
# 訓練深度神經網路：解決梯度消失(Vanishing Gradient)與爆炸(Exploding Gradient)問題的實戰指南

---

[來源: ch11] 在深度學習(Deep Learning)領域，訓練深度神經網路(Deep Neural Network, DNN)往往面臨**梯度消失(Vanishing Gradient)**或**爆炸(Exploding Gradient)** 的挑戰。
本教學將帶您探索有效的解決方案，包括**Xavier與He初始化(Initialization)**、**ReLU及其變體活化函數(Activation Function)**、
**批次歸一化(Batch Normalization)**、**梯度裁剪(Gradient Clipping)**、**遷移學習(Transfer Learning)**，以及**Adam等先進優化器(Optimizer)**。
這些技術有助於在複雜資料集上建構穩定且高效能的模型。

---

[來源: ch11] ## 關鍵重點 (Key Takeaways)

- 理解梯度消失(Vanishing Gradient)與爆炸(Exploding Gradient)問題，並學習初始化(Initialization)與活化函數(Activation Function)的解決方案
- 掌握批次歸一化(Batch Normalization)與梯度裁剪(Gradient Clipping)技術，提升訓練穩定性
- 熟悉各種優化器(Optimizer)與學習率調度(Learning Rate Scheduling)策略，加速收斂
- 應用正規化(Regularization)技術避免過擬合(Overfitting)，建構泛化能力強的模型

---

---

[來源: ch11] ## 梯度消失與爆炸問題 (Vanishing/Exploding Gradients) {#vanishing-exploding-gradients}

💡 **實際應用情境(Application Scenario)：** 在訓練深度神經網路(Deep Neural Network, DNN)時，特別是處理複雜的圖像分類任務時，
梯度(Gradient)可能會隨著層數增加而變得極小（消失, vanishing）或極大（爆炸, exploding），導致訓練緩慢或不穩定。
這在自然語言處理(Natural Language Processing, NLP)或電腦視覺(Computer Vision, CV)任務中特別常見。
如果梯度消失，底層連接權重將幾乎不更新，導致訓練無法收斂至良好解；若梯度爆炸，權重會更新過大，導致演算法發散。

---

[來源: ch11] - 傳遞與連乘效應：在反向傳播過程中，梯度會因為層層相乘而產生連乘效應。
當我們計算損失函數對第 l 層輸入的梯度時，根據鏈式法則 (chain rule)，會得到所有後續層的Jacobian矩陣（實務上常以權重矩陣的導數近似）連乘的結果。
數學上可表示為：
$$\nabla_{x_l}L = \left(\prod_{k=l+1}^{L} J_k\right) \nabla_{x_L}L$$
其中 $J_k$ 是第 k 層的 Jacobian Matrix，$\nabla_{x_L}L$ 是損失函數 $L$ 對輸出層輸入 $x_L$ 的梯度。
重點在於，如果這些矩陣的特徵值或奇異值普遍小於 1，梯度在反向傳播時會指數級衰減，這就是「梯度消失」問題；
反之，若特徵值大於 1，梯度則會指數級增長，造成「梯度爆炸」。
這兩種現象都會讓深層網路的訓練變得困難，因為底層的參數無法有效學習。

---

[來源: ch11] 遍小於 1，梯度在反向傳播時會指數級衰減，這就是「梯度消失」問題；
反之，若特徵值大於 1，梯度則會指數級增長，造成「梯度爆炸」。
這兩種現象都會讓深層網路的訓練變得困難，因為底層的參數無法有效學習。

- 常見觸發因子：
  - **飽和活化函數(Saturating Activation Function, 如 sigmoid、tanh)**：在極端輸入下導數接近 0，造成梯度被壓扁。
  - **不當的權重初始化(Poor Weight Initialization)**：使得層的縮放因子偏離 1，長深度網路中連乘效應被放大。
    - **「縮放因子」(Scaling Factor)** 在深度學習中通常指的是每一層在前向或反向傳播時，對信號（如活化值或梯度）造成的放大或縮小比例。
      這個因子會影響信號在多層網路中傳遞時是否穩定。
      如果每層的縮放因子偏離 1

---

[來源: ch11] ** 在深度學習中通常指的是每一層在前向或反向傳播時，對信號（如活化值或梯度）造成的放大或縮小比例。
      這個因子會影響信號在多層網路中傳遞時是否穩定。
      如果每層的縮放因子偏離 1，信號就會在多層連乘後指數級衰減（導致梯度消失）或增長（導致梯度爆炸）。
      因此，設計合適的初始化方法和活化函數，讓每層的縮放因子接近 1，是避免梯度問題、確保深層網路能有效學習的關鍵。
  - **深度或長序列結構(Very Deep Nets / RNN)**：多次相乘會把微小縮放累積成巨大的衰減或增長。
  - **非線性與偏移的累積**：每層輸入分佈偏移會改變導數的分布，進一步影響梯度流。

---

[來源: ch11] 深度或長序列結構(Very Deep Nets / RNN)**：多次相乘會把微小縮放累積成巨大的衰減或增長。
  - **非線性與偏移的累積**：每層輸入分佈偏移會改變導數的分布，進一步影響梯度流。

- 直觀示例：若每層平均縮放因子為 0.9，深度為 100，則梯度約為 $0.9^{100}\approx 2.7\times10^{-5}$，幾乎消失；若為 1.1，則 $1.1^{100}\approx 13{,}780$，會爆炸。

---

[來源: ch11] 0.9，深度為 100，則梯度約為 $0.9^{100}\approx 2.7\times10^{-5}$，幾乎消失；若為 1.1，則 $1.1^{100}\approx 13{,}780$，會爆炸。

- 快速對策（為何前文方法有效）：
  - 使用*非飽和或部分非飽和活化*(Non-saturating or Partially Non-saturating Activation，如ReLU／LeakyReLU)以維持導數大小。
  - 採用*合適初始化*(Proper Initialization，如Xavier／He)使每層輸出與梯度的變異數接近恆定。
  - *批次歸一化*(Batch Normalization)或*層歸一化*(Layer Normalization)穩定每層輸入分佈，減少導數偏移。
  - *殘差連接*(Residual Connection, ResNet)

---

[來源: ch11] h Normalization)或*層歸一化*(Layer Normalization)穩定每層輸入分佈，減少導數偏移。
  - *殘差連接*(Residual Connection, ResNet)與*跳躍連接*(Skip Connection)提供直接梯度通路，緩解連乘效應。
  - *梯度裁剪*(Gradient Clipping)可控制爆炸情況。
  - 調整*學習率*(Learning Rate)與*優化器*(Optimizer)也能幫助穩定訓練。

---

[來源: ch11] ### 範例 1: 繪製Sigmoid激活函數的飽和問題

```python

---

[來源: ch11] # 匯入必要的函式庫
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch11] # 定義sigmoid函數
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

---

[來源: ch11] # 建立z值的範圍
z = np.linspace(-5, 5, 200)

---

[來源: ch11] # 繪製圖表
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [1, 1], 'k--')
plt.plot([0, 0], [-0.2, 1.2], 'k-')
plt.plot([-5, 5], [-3/4, 7/4], 'g--')
plt.plot(z, sigmoid(z), "b-", linewidth=2,
         label=r"$\sigma(z) = \dfrac{1}{1+e^{-z}}$")

---

[來源: ch11] # 設定圖表屬性並顯示
plt.axis([-5, 5, -0.2, 1.2])
plt.xlabel("$z$")
plt.legend(loc="upper left", fontsize=16)
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] .2, 1.2])
plt.xlabel("$z$")
plt.legend(loc="upper left", fontsize=16)
plt.show()
```

**✅ 程式碼逐行解析：**

1. `import numpy as np`: 匯入NumPy函式庫，用於數值計算
2. `import matplotlib.pyplot as plt`: 匯入Matplotlib繪圖函式庫
3. `def sigmoid(z):`: 定義sigmoid激活函數
4. `return 1 / (1 + np.exp(-z))`: 計算sigmoid函數的值
5. `z = np.linspace(-5, 5, 200)`: 建立從-5到5的200個等間距點
6. `plt.plot([-5, 5], [0, 0], 'k-')`: 繪製x軸
7. `plt.plot([-5, 5],

---

[來源: ch11] pace(-5, 5, 200)`: 建立從-5到5的200個等間距點
6. `plt.plot([-5, 5], [0, 0], 'k-')`: 繪製x軸
7. `plt.plot([-5, 5], [1, 1], 'k--')`: 繪製y=1的虛線
8. `plt.plot([0, 0], [-0.2, 1.2], 'k-')`: 繪製y軸
9. `plt.plot([-5, 5], [-3/4, 7/4], 'g--')`: 繪製線性區域的參考線
10. `plt.plot(z, sigmoid(z), "b-", linewidth=2, label=...)`: 繪製sigmoid曲線
11. `plt.axis([-5, 5, -0.2, 1.2])`: 設定軸範圍
12. `plt.xlabel("$z$")`: 設定x軸標籤
13. `plt.legend(...)`: 顯

---

[來源: ch11] 線
11. `plt.axis([-5, 5, -0.2, 1.2])`: 設定軸範圍
12. `plt.xlabel("$z$")`: 設定x軸標籤
13. `plt.legend(...)`: 顯示圖例
14. `plt.show()`: 顯示圖表

**🎯 重點摘要:**

- **核心功能**: 視覺化sigmoid激活函數的飽和行為，展示梯度消失問題
- **潛在問題**: 在極端z值處梯度接近零，導致深度網路訓練困難
- **最佳使用情境**: 用於理解為何需要更好的激活函數和初始化方法

---

---

[來源: ch11] ## Xavier與He初始化 (Xavier/He Initialization) {#xavier-he-initialization}

💡 **實際應用情境(Application Scenario)：** 在建構深度卷積神經網路(Convolutional Neural Network, CNN)處理圖像分類任務時，正確的權重初始化(Weight Initialization)對於避免梯度問題至關重要。

---

[來源: ch11] 在建構深度卷積神經網路(Convolutional Neural Network, CNN)處理圖像分類任務時，正確的權重初始化(Weight Initialization)對於避免梯度問題至關重要。

為了使信號在深層網路中雙向流動（預測時向前，梯度回傳時向後），需要確保各層輸出與輸入的變異數(Variance)相等。
- **Xavier (Glorot) 初始化(Xavier Initialization)：** 主要用於 Sigmoid、tanh 或 Softmax 激活函數。它根據層的輸入與輸出連接數（fan-in 與 fan-out）來隨機初始化權重。
- **He 初始化(He Initialization)：** 專為 **ReLU 及其變體**（如 Leaky ReLU、ELU、GELU 等）設計的策略。這能顯著緩解訓練初期的梯度不穩定問題。

---

[來源: ch11] ### 為何合適的初始化能避免梯度消失 / 爆炸

直觀與數學要點：

---

[來源: ch11] - 前向傳播（方差保持）：對一層線性近似 y = W x，若 x 與 W 的元素獨立且均值為 0，則 Var(y) ≈ fan_in * Var(w) * Var(x)。為了避免輸出方差在多層間指數放大或衰減，通常選擇 Var(w) ≈ 1 / fan_in（或以 fan_in/fan_out 的平均值），使每層輸出方差與輸入方差大致相等。
- 激活函數的影響：ReLU 類激活會使約一半輸出為零，會縮減輸出方差；因此 He 初始化採用 Var(w) = 2 / fan_in 來補償。對稱且近似線性的激活（如 tanh）則常用 Xavier/Glorot（Var(w) ≈ 2 / (fan_in + fan_out)）。
- 反向傳播（梯度穩定性）：若每層都維持激活與梯度的方差近似恆定，反向傳播時雅可比矩陣的連乘不會使梯度的標準差指數衰減或增長，從而避免梯度消失或爆炸。
- 直觀結論：良好的

---

[來源: ch11]  + fan_out)）。
- 反向傳播（梯度穩定性）：若每層都維持激活與梯度的方差近似恆定，反向傳播時雅可比矩陣的連乘不會使梯度的標準差指數衰減或增長，從而避免梯度消失或爆炸。
- 直觀結論：良好的初始化使得網路在深度方向上的縮放因子接近 1，換言之將各層的奇異值分佈約束在不會導致指數放大或衰減的範圍內。

實務建議（簡短）：
- Xavier (Glorot)：Var(w) ≈ 2 / (fan_in + fan_out)（適合 tanh/sigmoid 類）
- He（for ReLU）：Var(w) ≈ 2 / fan_in（適合 ReLU 及其變體）

上述原則可讓每層的輸入／梯度方差大致保持恆定，從而有效緩解梯度消失與爆炸，並與批次正規化、殘差連接等方法協同提升訓練穩定性。

---

[來源: ch11] ### 範例 2: 使用He初始化建立Dense層

```python

---

[來源: ch11] # 匯入TensorFlow Keras層
import tensorflow as tf

---

[來源: ch11] # 使用He正態初始化建立Dense層，適用於ReLU激活函數
dense = tf.keras.layers.Dense(50, activation="relu",
                              kernel_initializer="he_normal")
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] activation="relu",
                              kernel_initializer="he_normal")
```

**✅ 程式碼逐行解析：**

1. `import tensorflow as tf`: 匯入TensorFlow函式庫
2. `dense = tf.keras.layers.Dense(50, activation="relu", kernel_initializer="he_normal")`: 建立具有50個神經元、ReLU活化函數和He常態初始化的Dense層
    - `activation="relu"`：指定使用ReLU活化函數
    - `kernel_initializer="he_normal"`：指定使用He常態初始化方法來初始化權重
        - `he_normal` 是 He 初

---

[來源: ch11] 指定使用ReLU活化函數
    - `kernel_initializer="he_normal"`：指定使用He常態初始化方法來初始化權重
        - `he_normal` 是 He 初始化的一種實現，會根據 fan_in 計算適當的標準差來生成常態分佈 (Normal distribution) 的權重；其標準差 $ \sigma $ 計算公式為：
            $$\sigma = \sqrt{\frac{2}{\text{fan\_in}}}$$

**🎯 重點摘要:**

- **核心功能**: 使用He初始化來適應ReLU活化函數，維持活化和梯度的穩定變異數
- **潛在問題**: Xavier初始化更適合tanh或sigmoid，He初始化最適合ReLU及其變體
- **最佳使用情境**: 深度網路中使用ReLU活化函數時

---

[來源: ch11] ### 範例 3: 自訂He初始化變體

```python

---

[來源: ch11] # 使用VarianceScaling初始化器自訂He初始化
he_avg_init = tf.keras.initializers.VarianceScaling(scale=2., mode="fan_avg",
                                                    distribution="uniform")
dense = tf.keras.layers.Dense(50, activation="sigmoid",
                              kernel_initializer=he_avg_init)
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] ivation="sigmoid",
                              kernel_initializer=he_avg_init)
```

**✅ 程式碼逐行解析：**

1. `he_avg_init = tf.keras.initializers.VarianceScaling(scale=2., mode="fan_avg", distribution="uniform")`: 建立自訂的He初始化變體，使用統一分佈和平均扇入扇出模式
    - `scale=2.`：指定縮放因子為2，適合ReLU類活化函數
    - `mode="fan_avg"`：使用fan_in和fan_out的平均值來計算縮放，適合某些活化函數
    - `distribution="uniform"`：使用均勻分佈來生成權重
2. `dense = tf.keras.layers.Dense(50, activation="sigmoid", kernel_initializer=he_avg_init)`: 建立Dense層使用此自訂初始化

---

[來源: ch11] = tf.keras.layers.Dense(50, activation="sigmoid", kernel_initializer=he_avg_init)`: 建立Dense層使用此自訂初始化

**🎯 重點摘要:**

- **潛在問題**: 需要根據活化函數調整scale參數
- **最佳使用情境**: 需要精細控制初始化行為時

---

---

[來源: ch11] ## 非飽和激活函數 (Nonsaturating Activation Functions) {#nonsaturating-activation-functions}

💡 **實際應用情境(Application Scenario)：** 在訓練深層網路（例如語音辨識或大型卷積網路）時，非飽和激活函數(Nonsaturating Activation Function，如 ReLU、LeakyReLU、ELU、SELU)通常比 sigmoid/tanh 更穩定。主要理由包括：

---

[來源: ch11] 路）時，非飽和激活函數(Nonsaturating Activation Function，如 ReLU、LeakyReLU、ELU、SELU)通常比 sigmoid/tanh 更穩定。主要理由包括：

- 更好的梯度流：在非飽和區域導數不會趨近於零，能顯著減少梯度消失並改善深層梯度傳遞。
- 加快收斂並提升表徵能力：ReLU 的正區域近似線性，能加速學習；負區域帶來稀疏激活，有助正規化。
- 與初始化相容：例如 ReLU 搭配 He 初始化可維持輸出與梯度的方差穩定，降低訓練不穩定性。
- 計算效率高：實作簡單（max），比含指數或雙曲函數的激活快。
- 可緩解「死亡神經元」問題：LeakyReLU、ELU 等變體對負輸入保留微小梯度，避免單一神經元長時間不更新。

---

[來源: ch11] 降低訓練不穩定性。
- 計算效率高：實作簡單（max），比含指數或雙曲函數的激活快。
- 可緩解「死亡神經元」問題：LeakyReLU、ELU 等變體對負輸入保留微小梯度，避免單一神經元長時間不更新。

總結：非飽和激活函數在深層網路中通常能提供更穩定的梯度流、更快的收斂與較好的實務表現；必要時可選擇變體以平衡 ReLU 的潛在缺點。

---

[來源: ch11] ### 範例 4: Leaky ReLU激活函數

```python

---

[來源: ch11] # 定義Leaky ReLU函數
def leaky_relu(z, alpha):
    return np.maximum(alpha * z, z)

---

[來源: ch11] # 繪製Leaky ReLU曲線
z = np.linspace(-5, 5, 200)
plt.plot(z, leaky_relu(z, 0.1), "b-", linewidth=2,
         label=r"$LeakyReLU(z) = max(\alpha z, z)$")
plt.axis([-5, 5, -1, 3.7])
plt.xlabel("$z$")
plt.legend()
plt.show()
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] ha z, z)$")
plt.axis([-5, 5, -1, 3.7])
plt.xlabel("$z$")
plt.legend()
plt.show()
```

**✅ 程式碼逐行解析：**

1. `def leaky_relu(z, alpha):`: 定義Leaky ReLU函數
2. `return np.maximum(alpha * z, z)`: 計算Leaky ReLU值
3. `z = np.linspace(-5, 5, 200)`: 建立z值範圍
4. `plt.plot(z, leaky_relu(z, 0.1), ...)`: 繪製Leaky ReLU曲線
5. `plt.axis([-5, 5, -1, 3.7])`: 設定軸範圍
6. `plt.xlabel("$z$")`: 設定x軸標籤
7. `plt.legend()`: 顯示圖例
8. `plt.show()`: 顯示圖表

---

[來源: ch11] 5, 5, -1, 3.7])`: 設定軸範圍
6. `plt.xlabel("$z$")`: 設定x軸標籤
7. `plt.legend()`: 顯示圖例
8. `plt.show()`: 顯示圖表

**🎯 重點摘要:**

- **核心功能**: 允許負輸入有小梯度，防止神經元死亡
- **潛在問題**: alpha參數需要調整
- **最佳使用情境**: ReLU可能導致神經元死亡的網路

---

[來源: ch11] ### 範例 5: 使用TensorFlow的LeakyReLU

```python

---

[來源: ch11] # 使用TensorFlow的LeakyReLU層
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
dense = tf.keras.layers.Dense(50, activation=leaky_relu,
                              kernel_initializer="he_normal")
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] vation=leaky_relu,
                              kernel_initializer="he_normal")
```

**✅ 程式碼逐行解析：**

1. `leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)`: 建立LeakyReLU層，alpha=0.2
2. `dense = tf.keras.layers.Dense(50, activation=leaky_relu, kernel_initializer="he_normal")`: 建立Dense層使用LeakyReLU活化

**🎯 重點摘要:**

- **核心功能**: 將LeakyReLU作為單獨層使用，便於控制
- **潛在問題**: 增加網路深度
- **最佳使用情境**: 需要精細控制活化函數時

---

[來源: ch11] ### 範例 6: Swish激活函數

Swish激活函數定義為 swish(x) = x * sigmoid(x)，它是一種平滑、非單調的激活函數，能提供更好的梯度流和性能。

```python

---

[來源: ch11] # 定義Swish函數
def swish(z):
    return z * (1 / (1 + np.exp(-z)))

---

[來源: ch11] # 繪製Swish曲線
z = np.linspace(-5, 5, 200)
plt.plot(z, swish(z), "b-", linewidth=2,
         label=r"$Swish(z) = z \cdot \sigma(z)$")
plt.axis([-5, 5, -2.5, 5])
plt.xlabel("$z$")
plt.legend()
plt.show()
```

**✅ 程式碼逐行解析:**

---

[來源: ch11] sigma(z)$")
plt.axis([-5, 5, -2.5, 5])
plt.xlabel("$z$")
plt.legend()
plt.show()
```

**✅ 程式碼逐行解析:**

1. `def swish(z):`: 定義Swish函數
2. `return z * (1 / (1 + np.exp(-z)))`: 計算Swish值
3. `z = np.linspace(-5, 5, 200)`: 建立z值範圍
4. `plt.plot(z, swish(z), ...)`: 繪製Swish曲線
5. `plt.axis([-5, 5, -2.5, 5])`: 設定軸範圍
6. `plt.xlabel("$z$")`: 設定x軸標籤
7. `plt.legend()`: 顯示圖例
8. `plt.show()`: 顯示圖表

**🎯 重點摘要:**

---

[來源: ch11] ])`: 設定軸範圍
6. `plt.xlabel("$z$")`: 設定x軸標籤
7. `plt.legend()`: 顯示圖例
8. `plt.show()`: 顯示圖表

**🎯 重點摘要:**

- **核心功能**: 平滑激活，適合深層網路
- **潛在問題**: 計算稍複雜於ReLU
- **最佳使用情境**: 深層網路中需要更好性能時

---

---

[來源: ch11] ## 批次正規化 (Batch Normalization) {#batch-normalization}

💡 **實際應用情境：** 在訓練大型卷積網路處理醫療影像分類時，批次正規化可以加速收斂並提升穩定性。

簡短定義：批次正規化（Batch Normalization, BN）會對一個 **mini-batch 中每個神經元的輸入進行零中心化（zero-centering）與正規化（normalizing）**，並透過學習兩個新參數（縮放與平移）來決定最佳的平均值與標準差，從而讓每層的輸入分佈更穩定。

---

[來源: ch11] tch 中每個神經元的輸入進行零中心化（zero-centering）與正規化（normalizing）**，並透過學習兩個新參數（縮放與平移）來決定最佳的平均值與標準差，從而讓每層的輸入分佈更穩定。

直觀效果與好處：
- **加速訓練：** 減輕了對權重初始化的敏感性，並允許使用更大的學習率。
- **減少過擬合：** BN 具有某種程度的正規化效果，可減少對其他正規化技術（如 Dropout）的需求。
- 穩定每層輸入分佈，減少所謂的「internal covariate shift」（雖然名稱有爭議，但實務上確實穩定了訓練過程）。
- 幫助梯度流動，對深層網路特別有用。

---

[來源: ch11] Dropout）的需求。
- 穩定每層輸入分佈，減少所謂的「internal covariate shift」（雖然名稱有爭議，但實務上確實穩定了訓練過程）。
- 幫助梯度流動，對深層網路特別有用。

訓練 vs 推論（關鍵差異）：
- 訓練時：BN 使用 mini-batch 的均值與方差來標準化，並同時更新「running mean/variance」。
- 推論時：BN 使用累積的 running mean/variance（moving averages），而非當前樣本的統計量——因此訓練與推論行為不同，這一點在微調或小 batch 情境下需要特別留意。

---

[來源: ch11] 論時：BN 使用累積的 running mean/variance（moving averages），而非當前樣本的統計量——因此訓練與推論行為不同，這一點在微調或小 batch 情境下需要特別留意。

實作要點與最佳實踐：
- 典型擺放：在**非線性之前對線性輸出做 BN**（例如：Dense/Conv2D -> **BatchNormalization** -> Activation）。
- 若使用 BN，通常可將上一層的 bias 關閉（`use_bias=False`），因為 BN 有自己的平移參數 beta。
- 小 batch（例如 batch size < 8）會削弱 BN 的效果；可改用 LayerNorm、GroupNorm 或 InstanceNorm 作為替代。
- 調整 momentum（running average 的慣性）與 epsilon（數值穩定項）可改善累積統計量的品質，特別是在遷移學習或非平穩資料上。
- 與 Dropout 的互動：BN 已有正則化效果，但在某些架構中兩者結合仍有益；順序上通常**先 BN 再 Dropout**（若同時使用）。

---

[來源: ch11] 善累積統計量的品質，特別是在遷移學習或非平穩資料上。
- 與 Dropout 的互動：BN 已有正則化效果，但在某些架構中兩者結合仍有益；順序上通常**先 BN 再 Dropout**（若同時使用）。

遷移學習與微調的注意事項：
- 當凍結大部分層只微調少數層時，建議也凍結 `BatchNormalization（layer.trainable = False）`，以避免 running statistics 在小資料上被扭曲。
- 若要在新資料上重新計算 running stats，可在保留權重的情況下以較小學習率、較大 batch 重訓幾個 epoch，或使用 model.fit(..., `callbacks=[tf.keras.callbacks.BatchNormalizationMomentum(... )]`) 調整策略。

---

[來源: ch11] 幾個 epoch，或使用 model.fit(..., `callbacks=[tf.keras.callbacks.BatchNormalizationMomentum(... )]`) 調整策略。

限制與替代方案：
- 對非常小的 batch 或序列模型（RNN）效果有限；改用 LayerNorm、GroupNorm（對 batch size 不敏感）或 Weight Standardization + GroupNorm（在 CNN 中常見）。

範例 6: 在 Sequential 模型中加入批次正規化（推薦寫法）

```python

---

[來源: ch11] # 推薦做法：在線性層後、Activation 前使用 BatchNormalization，並關閉 bias
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, use_bias=False,
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),

---

[來源: ch11] izer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),

tf.keras.layers.Dense(100, use_bias=False,
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),

tf.keras.layers.Dense(10, activation="softmax")
])
```

簡短說明：上述寫法能確保 BN 正確作用於線性輸出，減少不必要的 bias 並提升數值穩定性。

---

[來源: ch11] keras.layers.Dense(10, activation="softmax")
])
```

簡短說明：上述寫法能確保 BN 正確作用於線性輸出，減少不必要的 bias 並提升數值穩定性。

進階提示（快速 checklist）：
- batch size 太小？考慮改用 LayerNorm/GroupNorm。
- 需要更快收斂？嘗試提高 learning rate 並使用 BN（同時監控訓練穩定性）。
- 微調時觀察 running mean/var；必要時凍結 BN 或重估其 momentum。

**🎯 重點摘要:**

---

[來源: ch11] ？嘗試提高 learning rate 並使用 BN（同時監控訓練穩定性）。
- 微調時觀察 running mean/var；必要時凍結 BN 或重估其 momentum。

**🎯 重點摘要:**

- **核心功能**: 穩定並標準化每層輸入分佈，改善訓練穩定性與收斂速度
- **實作要點**: 將 BN 放在 Activation 之前、關閉前一層 bias；小 batch 時選用替代正規化
- **最佳使用情境**: 深層 CNN 或其他需要穩定梯度流的架構；在遷移學習時注意 running statistics

---

---

[來源: ch11] ## 梯度裁剪 (Gradient Clipping) {#gradient-clipping}

💡 **實際應用情境：** 在訓練循環神經網路處理序列資料時，梯度裁剪可以防止梯度爆炸問題。

---

[來源: ch11] ### 範例 7: 使用梯度裁剪的SGD優化器

```python

---

[來源: ch11] # 使用clipvalue進行梯度裁剪
optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)
```

**✅ 程式碼逐行解析：**

1. `optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)`: 建立SGD優化器，梯度裁剪值為1.0
    - `clipvalue=1.0`：表示如果梯度的絕對值超過1.0，則將其裁剪到1.0，從而防止梯度爆炸。
2. `model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)`: 編譯模型使用此優化器

---

[來源: ch11] 而防止梯度爆炸。
2. `model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)`: 編譯模型使用此優化器

**🎯 重點摘要:**

- **核心功能**: 限制梯度最大值，防止爆炸
- **潛在問題**: 可能影響收斂速度
- **最佳使用情境**: RNN或非常深的網路

---

---

[來源: ch11] ## 重用預訓練層 (Reusing Pretrained Layers) {#reusing-pretrained-layers}

💡 **實際應用情境：** 在處理相似任務時，重用預訓練模型可以節省訓練時間並提升效能。

透過重用現有模型的預訓練層 (Pretrained Layers) 來進行遷移學習 (Transfer Learning)，以節省時間並提升相似任務的效能。它
涵蓋載入模型、凍結層以保留已學特徵，以及針對新目標進行微調。

---

[來源: ch11] ### 遷移學習策略 (Transfer Learning Strategies)

- **凍結層 (Freezing Layers)：** 訓練新任務時，先凍結預訓練層的權重，**只訓練新加入的層**。這能保留低層的特徵提取能力，避免過擬合。
- **微調 (Fine-Tuning)：** 訓練一段時間後，*解凍部分預訓練層並以小學習率繼續訓練*，讓模型適應新資料。
- **選擇性重用：** 根據*任務相似度*決定重用層數；相似任務重用更多層，不相似任務則重用較少或僅用於特徵提取。

---

[來源: ch11] ### 範例 8: 載入並重用預訓練模型

```python

---

[來源: ch11] # 載入預訓練模型
model_A = tf.keras.models.load_model("my_model_A.keras")

---

[來源: ch11] # 建立新模型重用前幾層
model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] _A.layers[:-1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))
```

**✅ 程式碼逐行解析：**

1. `model_A = tf.keras.models.load_model("my_model_A.keras")`: 載入預訓練模型
2. `model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])`: 重用除了輸出層外的所有層
    - `model_A.layers[:-1]`：表示取出預訓練模型的所有層，除了最後一層（通常是輸出層），以便在新模型中重用這些層的權重。
3. `model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))`: 加入新的輸出層

---

[來源: ch11] 輸出層），以便在新模型中重用這些層的權重。
3. `model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))`: 加入新的輸出層
    - `tf.keras.layers.Dense(1, activation="sigmoid")`：為新模型添加一個全連接層，輸出為1個單位，使用sigmoid激活函數，適合二分類 (binary classification) 任務。

**🎯 重點摘要:**

- **核心功能**: 遷移學習，重用已學會的特徵
- **潛在問題**: 需要凍結層以避免破壞預訓練權重
- **最佳使用情境**: 相似任務的遷移學習

---

[來源: ch11] ### 如何進行凍結層 (How to Freeze Layers)

在遷移學習中，凍結層 (Freezing Layers) 是指將預訓練模型的某些層設為不可訓練，以保留其已學會的特徵，避免在訓練新任務時破壞這些權重。這樣可以加速訓練並防止過擬合，尤其當新資料集較小時。

---

[來源: ch11] #### 步驟與實作要點

1. **載入預訓練模型**：先載入現有的模型。
2. **選擇要凍結的層**：通常凍結前幾層（低層特徵提取），解凍後幾層以適應新任務。
3. **設定 `trainable` 屬性**：將選定層的 `trainable` 設為 `False`。
4. **編譯並訓練**：編譯模型後訓練，只更新非凍結層的權重。
5. **微調 (Fine-Tuning)**：訓練一段時間後，可解凍部分層並以小學習率繼續訓練。

---

[來源: ch11] #### 範例：凍結預訓練模型的前幾層

```python

---

[來源: ch11] # 載入預訓練模型
model_A = tf.keras.models.load_model("my_model_A.keras")

---

[來源: ch11] # 建立新模型，重用所有層
model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))

---

[來源: ch11] # 凍結前幾層（例如前5層），保留低層特徵
for layer in model_B_on_A.layers[:5]:  # 假設前5層為特徵提取層
    layer.trainable = False

---

[來源: ch11] # 編譯模型（只訓練非凍結層）
model_B_on_A.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

---

[來源: ch11] # 訓練模型
model_B_on_A.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] 模型
model_B_on_A.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

**✅ 程式碼逐行解析：**

1. `model_A = tf.keras.models.load_model("my_model_A.keras")`: 載入預訓練模型。
2. `model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])`: 重用除了輸出層外的所有層。
3. `model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))`: 添加新輸出層。
4. `for layer in model_B_on_A.layers[:5]: layer.trainable = False`: 凍結前

---

[來源: ch11] vation="sigmoid"))`: 添加新輸出層。
4. `for layer in model_B_on_A.layers[:5]: layer.trainable = False`: 凍結前5層，防止其權重更新。
    - `layer.trainable = False`：將該層設為不可訓練，這樣在訓練過程中其權重不會被更新。
5. `model_B_on_A.compile(...)`: 編譯模型，只訓練可訓練層。
6. `model_B_on_A.fit(...)`: 訓練模型，非凍結層會更新權重。

**🎯 重點摘要：**

- **核心功能**: 保留預訓練特徵，加速新任務訓練。
- **潛在問題**: 凍結太多層可能導致新任務適應不足；凍結太少可能破壞預訓練知識。
- **最佳使用情境**: 任務相似度高時凍結更多層；訓練後可微調解凍層以提升效能。

---

---

[來源: ch11] ## 更快的優化器 (Faster Optimizers) {#faster-optimizers}

💡 **實際應用情境：** 比起標準梯度下降，更快的優化器能大幅縮短模型達到最佳解的時間。

---

[來源: ch11] - **Momentum (動量)：** 模擬物理動量，讓權重更新具有「慣性」，幫助跳出局部最小值並加速通過平緩區域。
- **AdaGrad：** 根據梯度的陡峭程度自動縮放學習率，給予平緩維度較大的步長。
- **RMSProp：** 修正了 AdaGrad 停止過早的問題，僅累積最近迭代的梯度。
- **Adam (自適應動量估計)：** 結合了動量與 RMSProp 的優點，通常只需微調學習率即可表現優異。
- **Nadam：** 為帶有 Nesterov 技巧的 Adam，收斂速度通常比 Adam 更快。
- **AdamW：** 專門針對 Adam 結合權重衰減（weight decay）不佳的問題進行修正，能提供更好的泛化能力。

---

[來源: ch11] ### 範例 9: 使用Adam優化器

```python

---

[來源: ch11] # 使用Adam優化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                     beta_2=0.999)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] egorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
```

**✅ 程式碼逐行解析：**

1. `optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)`: 建立Adam優化器
    - `learning_rate=0.001`：設定學習率
    - `beta_1=0.9`：一階矩估計的衰減率 (first-moment decay rate)
        - `beta_1` 控制了動量的衰減率，通常設為0.9表示過去梯度的影響會逐漸減弱，但仍保留足夠的歷史信息來加速收斂。
    - `beta_2=0.999`：二階矩估計的衰減率 (second-m

---

[來源: ch11]  `beta_1` 控制了動量的衰減率，通常設為0.9表示過去梯度的影響會逐漸減弱，但仍保留足夠的歷史信息來加速收斂。
    - `beta_2=0.999`：二階矩估計的衰減率 (second-moment decay rate)
        - `beta_2` 控制了梯度平方的衰減率，通常設為0.999表示過去梯度平方的影響會非常緩慢地減弱，這有助於穩定學習率的調整。
2. `model.compile(...)`: 編譯模型使用Adam優化器

**🎯 重點摘要:**

- **核心功能**: 自適應學習率，適合大多數任務
- **潛在問題**: 有時泛化不如SGD+momentum
- **最佳使用情境**: 快速原型設計

---

[來源: ch11] ### 範例 10: 使用Nadam優化器

```python

---

[來源: ch11] # 使用Nadam優化器
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9,
                                      beta_2=0.999)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
```

**✅ 程式碼逐行解析：**

1. `optimizer = tf.keras.optimizers.Nadam(...)`: 建立Nadam優化器
2. `model.compile(...)`: 編譯模型使用Nadam優化器

**🎯 重點摘要:**

---

[來源: ch11] r = tf.keras.optimizers.Nadam(...)`: 建立Nadam優化器
2. `model.compile(...)`: 編譯模型使用Nadam優化器

**🎯 重點摘要:**

- **核心功能**: Nadam通常收斂速度比Adam更快
    - Nadam 在 Adam 的基礎上引入了 Nesterov 加速梯度（NAG）的技巧，這使得它在某些情況下能夠更快地收斂，特別是在訓練深層神經網路時。
- **潛在問題**: 參數調整較為敏感
- **最佳使用情境**: 需要更快收斂的任務

---

---

[來源: ch11] ## 學習率調度 (Learning Rate Scheduling) {#learning-rate-scheduling}

---

[來源: ch11] ### 學習率調度策略 (Learning Rate Scheduling Strategies)

學習率調度 (Learning Rate Scheduling) 是指在訓練過程中*動態調整學習率* (Learning Rate) 的技術，以提升模型收斂速度、避免過早收斂到局部最小值，並改善最終效能。
當學習率過高時，可能導致訓練不穩定或跳過最佳解；過低則收斂緩慢。調度策略通常根據訓練步數 (steps) 或 epoch 來調整學習率。

常見策略包括：

---

[來源: ch11] 、避免過早收斂到局部最小值，並改善最終效能。
當學習率過高時，可能導致訓練不穩定或跳過最佳解；過低則收斂緩慢。調度策略通常根據訓練步數 (steps) 或 epoch 來調整學習率。

常見策略包括：

- **指數衰減 (Exponential Decay)**: 學習率以指數方式衰減，公式為 \( lr = lr_0 \times decay\_rate^{\frac{step}{decay\_steps}} \)，適合長時間訓練以逐步穩定權重更新。
- **階梯衰減 (Step Decay)**: 在特定里程碑 (如每隔幾個 epoch) 將學習率乘以一個衰減因子 (e.g., 0.1)，提供階段性調整。
- **餘弦衰減 (Cosine Decay)**: 學習率隨餘弦函數變化，從初始值平滑衰減到最小值，公式為 \( lr = lr_{min} + 0.5 \times (lr_0

---

[來源: ch11] ，提供階段性調整。
- **餘弦衰減 (Cosine Decay)**: 學習率隨餘弦函數變化，從初始值平滑衰減到最小值，公式為 \( lr = lr_{min} + 0.5 \times (lr_0 - lr_{min}) \times (1 + \cos(\frac{step \times \pi}{total\_steps})) \)，常用於現代優化器如 Adam。
- **線性衰減 (Linear Decay)**: 學習率線性從初始值降至最小值，簡單且直觀。
- **暖啟動 (Warmup)**: 訓練初期以小學習率開始，逐步增加至目標值，避免早期梯度爆炸。

實務建議：

---

[來源: ch11] 衰減 (Linear Decay)**: 學習率線性從初始值降至最小值，簡單且直觀。
- **暖啟動 (Warmup)**: 訓練初期以小學習率開始，逐步增加至目標值，避免早期梯度爆炸。

實務建議：

- 與 Adam 等自適應優化器結合使用時，調度能進一步提升效能。
- 監控驗證損失來決定衰減時機；過早衰減可能導致欠擬合 (underfitting)。
- 在 TensorFlow 中，可使用 `tf.keras.optimizers.schedules` 系列類別實作。

---

[來源: ch11] ### 範例 10: 指數衰減學習率調度

```python

---

[來源: ch11] # 使用指數衰減調度器
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=20_000,
    decay_rate=0.1,
    staircase=False
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] staircase=False
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

**✅ 程式碼逐行解析：**

1. `lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(...)`: 建立指數衰減調度器
    - `initial_learning_rate=0.01`：設定初始學習率
    - `decay_steps=20_000`：每20,000步衰減一次
    - `decay_rate=0.1`：每次衰減為原來的10%
    - `staircase=False`：使用平滑衰減而非階梯式衰減
2. `optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)`: 使用調度器的SGD優化器

---

[來源: ch11] lse`：使用平滑衰減而非階梯式衰減
2. `optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)`: 使用調度器的SGD優化器

**🎯 重點摘要:**

- **核心功能**: 學習率隨訓練進度指數衰減
- **潛在問題**: 需要調整衰減參數
- **最佳使用情境**: 需要學習率逐漸減少的訓練

---

[來源: ch11] ### 範例 10: 指數衰減學習率調度

```python

---

[來源: ch11] # 使用指數衰減調度器
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=20_000,
    decay_rate=0.1,
    staircase=False
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] staircase=False
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

**✅ 程式碼逐行解析：**

1. `lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(...)`: 建立指數衰減調度器
    - `initial_learning_rate=0.01`：設定初始學習率
    - `decay_steps=20_000`：每20,000步衰減一次
    - `decay_rate=0.1`：每次衰減為原來的10%
    - `staircase=False`：使用平滑衰減而非階梯式衰減
2. `optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)`: 使用調度器的SGD優化器
    - 這樣在訓練過程中，學習率會根據步數自動調整，從而幫助模型更快地收斂並避免過早停滯在局部最小值。

---

[來源: ch11] .SGD(learning_rate=lr_schedule)`: 使用調度器的SGD優化器
    - 這樣在訓練過程中，學習率會根據步數自動調整，從而幫助模型更快地收斂並避免過早停滯在局部最小值。

**🎯 重點摘要:**

- **核心功能**: 學習率隨訓練進度指數衰減
- **潛在問題**: 需要調整衰減參數
- **最佳使用情境**: 需要學習率逐漸減少的訓練

---

---

[來源: ch11] ## 透過正規化避免過擬合 (Avoiding Overfitting - Regularization) {#avoiding-overfitting-regularization}

💡 **實際應用情境：** 在處理小資料集時，正規化技術可以防止模型過度擬合訓練資料。

簡短說明：正規化是一組用來降低模型複雜度或在訓練中引入不確定性的技術，目的是提升模型在未見資料上的泛化能力。當訓練誤差顯著低於驗證誤差（或訓練/驗證損失出現明顯分歧）時，通常表示模型開始過擬合。

主要方法（快速參考）：

---

[來源: ch11] 組用來降低模型複雜度或在訓練中引入不確定性的技術，目的是提升模型在未見資料上的泛化能力。當訓練誤差顯著低於驗證誤差（或訓練/驗證損失出現明顯分歧）時，通常表示模型開始過擬合。

主要方法（快速參考）：

- **Dropout：** 在每次訓練迭代中，隨機將部分神經元暫時「丟棄」（設為 0），強迫神經元獨立學習有用的特徵，減少彼此間的過度依賴。
- **Max-Norm：** 對每個神經元的傳入權重實施約束，使其範數不超過預定閾值。這有助於緩解梯度消失或爆炸問題。
- **早停法 (Early Stopping)：** 監控驗證集的效能，當效能不再提升時立即停止訓練，避免模型對訓練數據過度擬合。
- **權重懲罰（L1 / L2）**：抑制權重放大，常用於線性與深度模型的基礎正則化。
- **資料增強（data augmentation）**：透過合成或變換擴充資料集，直接改善泛化。
- **正規化層（BatchNorm / LayerNorm / GroupNorm）**：穩定中間表示並帶來某種正則化效果。

---

[來源: ch11] augmentation）**：透過合成或變換擴充資料集，直接改善泛化。
- **正規化層（BatchNorm / LayerNorm / GroupNorm）**：穩定中間表示並帶來某種正則化效果。

實務建議（要點）：

- 先以簡單指標判定：檢查 train vs val 的學習曲線與泛化差距。
- 優先使用資料增強與適度的 L2；Dropout 在大網路或高度過擬合時很有效。
- 小 batch 時偏好 LayerNorm/GroupNorm；微調預訓練模型時可凍結 BatchNorm 的 running stats。
- 以驗證集為準，逐步調整正規化強度（避免過度抑制導致欠擬合）。

---

[來源: ch11] ### 範例 11: 使用Dropout正規化

```python

---

[來源: ch11] # 在模型中加入Dropout層
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dropout(rate=0.2),  # 輸入Dropout
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),  # 隱藏層Dropout
    tf.keras.layers.Dense(10, activation="softmax")
])
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] (rate=0.2),  # 隱藏層Dropout
    tf.keras.layers.Dense(10, activation="softmax")
])
```

**✅ 程式碼逐行解析：**

1. `tf.keras.layers.Dropout(rate=0.2)`: 建立Dropout層，丟棄率20%
    - `rate=0.2`：表示在訓練過程中，每次迭代將隨機丟棄20%的神經元，這有助於減少過擬合。
2. 將Dropout層插入網路中，在激活函數前或後

**🎯 重點摘要:**

- **核心功能**: 隨機丟棄神經元，防止共適應
    - Dropout 透過在訓練過程中隨機丟棄神經元，迫使模型學習更健壯的特徵表示，從而減少對特定神經元的過度依賴，提升泛化能力。
- **潛在問題**: 訓練時間增加，推論時無效
- **最佳使用情境**: 網路容量過大時

---

[來源: ch11] 過在訓練過程中隨機丟棄神經元，迫使模型學習更健壯的特徵表示，從而減少對特定神經元的過度依賴，提升泛化能力。
- **潛在問題**: 訓練時間增加，推論時無效
- **最佳使用情境**: 網路容量過大時

---

---

[來源: ch11] ## CIFAR10 上的 DNN 訓練實戰 (CIFAR10 DNN Implementation) {#cifar10-dnn-implementation}

💡 **實際應用情境：** 在完成理論學習後，建議進行以下實驗以鞏固知識。

---

[來源: ch11] ### 網路結構建議
建立一個具有 20 個隱藏層（每層 100 個神經元）的深層網路。

---

[來源: ch11] ### 配置建議
- **初始化：** 採用 **He 初始化**
- **激活函數：** 使用 **Swish 激活函數**（適用於深層網路）
- **優化器：** 使用 **Nadam 優化器**
- **正規化：** 應用 **早停法**

---

[來源: ch11] ### 進階比較
嘗試添加 **Batch Normalization** 並與使用 **SELU** 的自正規化網路進行比較，觀察收斂速度與最終準確度的差異。

---

[來源: ch11] ### 範例 12: CIFAR10 深層網路訓練

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

---

[來源: ch11] # 載入CIFAR10資料集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

---

[來源: ch11] # 建立深層網路
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))

for _ in range(20):
    model.add(tf.keras.layers.Dense(100, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Activation('swish'))  # 使用Swish激活

model.add(tf.keras.layers.Dense(10, activation='softmax'))

---

[來源: ch11] # 編譯模型
optimizer = tf.keras.optimizers.Nadam()
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

---

[來源: ch11] # 訓練模型（使用早停法）
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, validation_split=0.2, 
          callbacks=[early_stopping])
```

**✅ 程式碼逐行解析：**

---

[來源: ch11] _train, epochs=100, validation_split=0.2, 
          callbacks=[early_stopping])
```

**✅ 程式碼逐行解析：**

1. `import tensorflow as tf`: 匯入TensorFlow
2. `from tensorflow.keras.datasets import cifar10`: 匯入CIFAR10資料集
3. `(X_train, y_train), (X_test, y_test) = cifar10.load_data()`: 載入資料
4. `X_train = X_train.astype('float32') / 255.0`: 正規化輸入
5. `model = tf.keras.Sequential()`: 建立順序模型
6. `model.add(tf.kera

---

[來源: ch11] n.astype('float32') / 255.0`: 正規化輸入
5. `model = tf.keras.Sequential()`: 建立順序模型
6. `model.add(tf.keras.layers.Flatten(...))`: 展平輸入
7. `for _ in range(20):`: 加入20個隱藏層
8. `model.add(tf.keras.layers.Dense(100, kernel_initializer='he_normal'))`: Dense層使用He初始化
9. `model.add(tf.keras.layers.Activation('swish'))`: Swish激活
10. `model.add(tf.keras.layers.Dense(10, activation='softmax'))`: 輸出層
11. `optimizer

---

[來源: ch11] sh'))`: Swish激活
10. `model.add(tf.keras.layers.Dense(10, activation='softmax'))`: 輸出層
11. `optimizer = tf.keras.optimizers.Nadam()`: Nadam優化器
12. `model.compile(...)`: 編譯模型
13. `early_stopping = tf.keras.callbacks.EarlyStopping(...)`: 早停回呼
14. `model.fit(...)`: 訓練模型

**🎯 重點摘要:**

- **核心功能**: 實戰深層網路訓練於CIFAR10
- **潛在問題**: 深層網路可能過擬合，需要調整正規化
- **最佳使用情境**: 學習DNN訓練技術

---

[來源: ch11] ## 總結與最佳實踐

訓練深度神經網路需要仔細處理梯度問題、初始化、正規化和優化。本教學涵蓋了從基礎到進階的技術，包括適當的激活函數選擇（如Swish）、批次正規化應用、各種優化器（Adam、Nadam等）選用以及學習率調度。實務上，建議從Adam或Nadam開始原型設計，然後使用SGD配合動量進行最終訓練，並應用適當的正規化技術如早停法。透過CIFAR10實戰範例，可以鞏固這些知識。

---

[來源: ch11] ## 常見問答 (FAQ)

**Q: 為什麼需要特殊的初始化方法？**  
A: 標準初始化可能導致梯度消失或爆炸，特殊的初始化如He和Xavier可以維持激活和梯度的穩定方差。

**Q: ReLU為什麼比sigmoid更好？**  
A: ReLU不會在正輸入處飽和，提供更快的收斂和更好的梯度流。

**Q: 批次正規化什麼時候使用？**  
A: 當訓練不穩定或收斂緩慢時，特別適用於深度網路。

**Q: Adam和SGD+momentum哪個更好？**  
A: Adam適合快速原型，SGD+momentum通常提供更好的最終泛化。

**Q: Nadam和Adam有什麼差別？**  
A: Nadam結合了Nesterov動量，通常收斂速度比Adam更快。

**Q: Swish激活函數適合什麼情況？**  
A: Swish適用於深層網路，提供平滑的梯度流和更好的性能。

---

[來源: ch11] ## 推薦標籤 (Suggested Hashtags)

#Python #深度學習 #神經網路 #機器學習 #程式設計 #教學 #編程 #開發 #技術分享 #學習筆記 #AI #人工智慧

---

[來源: ch11 | 類型: cheatsheet] # Ch11 速查表：Training Deep Neural Networks

> **核心主旨**：深層網路訓練的六大工具箱 —— 正確的 initialization + BN + optimizer 組合是穩定訓練的基礎。

---

---

[來源: ch11 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| He Initialization | 針對 ReLU 設計，保持各層訊號變異數 | 搭配 ReLU / Leaky ReLU / SELU |
| Glorot (Xavier) Init | 針對 sigmoid/tanh 設計 | 搭配 sigmoid / tanh |
| Batch Normalization | 每批次正規化每層輸出，解決 internal covariate shift | 幾乎所有深層網路 |
| Dropout | 訓練時隨機丟棄神經元，防止 overfitting | 全連接層的正則化 |
| Adam | 自適應學習率，結合 RMSProp + momentum | 通用最佳起點 |
| SELU | 自我正規化活化函數，需搭配 LeCun 初始化 | 全連接網路的 ReLU 替代 |
| Gradient Clipping | 限制梯度的最大範數，防止梯度爆炸 | RNN / 深層網路 |
| Learning Rate Schedule | 訓練過程中動態調整 lr | 精調收斂性 |
| Transfer Learning | 複用預訓練模型的底層，只訓練頂層 | 資料量少的任務 |


---

[來源: ch11 | 類型: cheatsheet] Learning Rate Schedule | 訓練過程中動態調整 lr | 精調收斂性 |
| Transfer Learning | 複用預訓練模型的底層，只訓練頂層 | 資料量少的任務 |


---

---

[來源: ch11 | 類型: cheatsheet] | Keras Class / Function | 重點參數 | 用途 |
|------------------------|---------|------|
| `kernel_initializer="he_normal"` | – | He 初始化（用於 ReLU） |
| `kernel_initializer="glorot_uniform"` | – | Glorot 初始化（預設） |
| `tf.keras.layers.BatchNormalization` | `momentum=0.99`, `epsilon=0.001` | 批次歸一化 |
| `tf.keras.layers.Dropout` | `rate=0.2` | 標準 Dropout |
| `tf.keras.layers.AlphaDropout` | `rate=0.1` | SELU 專用 Dropout |
| `tf.keras.optimizers.Adam` | `learning_rate=1e-3`, `clipnorm=1.0` | Adam 優化器 |
| `tf.keras.optimizers.SGD` | `learning_rate=0.01`, `momentum=0.9`, `nesterov=True` | SGD + Nesterov momentum |
| `tf.keras.callbacks.ReduceLROnPlateau` | `factor=0.5`, `patience=5` | 自動降低學習率 |
| `tf.keras.callbacks.LearningRateScheduler` | `schedule=lambda epoch, lr: lr * 0.95` | 自訂 LR 排程 |


---

[來源: ch11 | 類型: cheatsheet] | `tf.keras.callbacks.LearningRateScheduler` | `schedule=lambda epoch, lr: lr * 0.95` | 自訂 LR 排程 |


---

---

[來源: ch11 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import tensorflow as tf

---

[來源: ch11 | 類型: cheatsheet] # 推薦的深層網路配方（BN + He init + ReLU）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    *[
        layer
        for _ in range(20)
        for layer in [
            tf.keras.layers.Dense(100, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ]
    ],
    tf.keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch11 | 類型: cheatsheet] # 如果 BN 加在激活之後（更常見的寫法）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch11 | 類型: cheatsheet] # Dropout（用於全連接層）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch11 | 類型: cheatsheet] # Adam + Gradient Clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

---

[來源: ch11 | 類型: cheatsheet] # Learning Rate Schedule（指數衰減）
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

---

[來源: ch11 | 類型: cheatsheet] # 1cycle schedule 的近似寫法
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

---

[來源: ch11 | 類型: cheatsheet] # Transfer Learning：凍結底層，只訓練頂層
base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(10, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

---

[來源: ch11 | 類型: cheatsheet] nputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

---

[來源: ch11 | 類型: cheatsheet] # Fine-tuning：解凍部分層
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

---

---

[來源: ch11 | 類型: cheatsheet] ## 4. 常見陷阱

- **BN 的位置**：BN 可放在激活函數前（原論文）或後（實務更常見），兩種都有人用，專案內保持一致即可。
- **Dropout 在推論時關閉**：`model.predict()` 自動關閉 Dropout；在自訂迴圈中要傳 `training=False`。
- **轉移學習重新 compile**：解凍層後 **必須重新 compile**，否則 optimizer 的動量狀態可能造成問題。
- **BN + Dropout 同時用**：效果不一定加乘，有時互相干擾，建議分別實驗。
- **He init 必須搭配 ReLU 類**：若用 sigmoid/tanh，改用 `glorot_uniform`（預設值）。

---

---

[來源: ch11 | 類型: cheatsheet] ## 5. 決策指南

```
活化函數選擇（由強到弱）：
├── 一般深層網路  → SELU（若全連接且標準化輸入）
├── CNN / 通用    → ELU → Leaky ReLU → ReLU
└── 輸出層        → softmax（多類）/ sigmoid（二元）/ linear（迴歸）

初始化配對規則：
├── ReLU / Leaky ReLU / ELU / SELU → He normal
└── Sigmoid / Tanh                 → Glorot uniform（預設）

正則化策略：
├── 一般過擬合     → Dropout (0.1~0.5)
├── 深層 CNN       → Dropout + BatchNormalization
├── 非常小的資料集 → L2 regularization + Dropout
└── 大資料集       → BatchNormalization 通常足夠
```

---

[來源: ch11 | 類型: handout] # 課程講義：深度神經網路訓練優化與技術 (Chapter 11)

**學習目標：** 本章節旨在幫助電機系同學掌握克服「梯度消失與爆炸」的核心技術，並學會如何透過進階優化器與正則化手段，穩定且高效地訓練深層神經網路。

---

你好，各位同學。我是你們這學期「深度學習實務」課程的教授。

今天我們要探討的是深度神經網路（DNN）訓練中最核心的挑戰：**如何讓深層模型穩定且快速地收斂**。在建構深層網路時，我們經常會遇到梯度消失（Vanishing Gradients）或梯度爆炸（Exploding Gradients）的問題，這會讓模型難以訓練。我們將從權重初始化、活化函數 (Activation Function)、歸一化、遷移學習、優化器以及正則化這六大面向出發，打造你的「深度學習工具箱」。

---

---

[來源: ch11 | 類型: handout] ## 1. 權重初始化與梯度問題 (Initialization & Gradient Problems)

當訊號在深層網路中反向傳播時，梯度可能會趨近於 0（梯度消失）或是無窮大（梯度爆炸）

* **關鍵概念：**
  * **He 初始化 (He Initialization)**：專門為 **ReLU** 及其變體（Leaky ReLU, ELU, SELU）設計。
  * **Xavier (Glorot) 初始化**：適用於 **Sigmoid** 或 **Tanh** 活化函數。
  * **核心目標**：在每一層的輸入與輸出之間保持信號的變異數（Variance）一致。

---

[來源: ch11 | 類型: handout] avier (Glorot) 初始化**：適用於 **Sigmoid** 或 **Tanh** 活化函數。
  * **核心目標**：在每一層的輸入與輸出之間保持信號的變異數（Variance）一致。

* **⚡ 補充練習 1：**
  1. 請說明為什麼在深層網路中將所有權重初始化為 0 會導致模型無法學習？（提示：對稱性打破，Symmetry Breaking）。
  2. 撰寫程式碼比較：在一個 20 層的 Dense 網路中，分別使用 `random_normal`（預設變異數）與 `he_normal` 初始化，觀察第一層梯度的數值分佈差異。

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fan_in = 20

---

[來源: ch11 | 類型: handout] # He Normal Initialization
init_he = tf.keras.initializers.he_normal()
w = init_he(shape=(fan_in, 50))
print(w.numpy().std(), np.sqrt(2.0 / fan_in))

---

[來源: ch11 | 類型: handout] # Random Normal Initialization
init_random = tf.keras.initializers.random_normal()
w_random = init_random(shape=(fan_in, 50))
print(w_random.numpy().std())

plt.hist(w.numpy().flatten(), bins=50, alpha=0.5, label='He Normal')
plt.hist(w_random.numpy().flatten(), bins=50, alpha=0.5, label='Random Normal (常態分佈)')
plt.legend()
plt.title('Weight Distribution Comparison')
plt.show()

```

```

---

[來源: ch11 | 類型: handout] Random Normal (常態分佈)')
plt.legend()
plt.title('Weight Distribution Comparison')
plt.show()

```

```

python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch11 | 類型: handout] .show()

```

```

python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def make_model(init):
    inp = tf.keras.layers.Input(shape=(100,))
    x = inp
    for _ in range(20):
        x = tf.keras.layers.Dense(50,
                                  activation="relu",
                                  kernel_initializer=init)(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inp, out)

---

[來源: ch11 | 類型: handout] # fake batch
x_batch = tf.random.normal((32, 100))
y_batch = tf.random.normal((32, 1))

---

[來源: ch11 | 類型: handout] plt.figure()
for name, init in [("random_normal", "random_normal"),
                   ("he_normal", "he_normal")]:
    model = make_model(init)
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        loss = tf.reduce_mean(tf.square(preds - y_batch))
    # 第一個 Dense 層的權重梯度
    grads = tape.gradient(loss, [model.layers[1].kernel, model.layers[-1].kernel])
    vals 

---

[來源: ch11 | 類型: handout]  層的權重梯度
    grads = tape.gradient(loss, [model.layers[1].kernel, model.layers[-1].kernel])
    vals = grads[0].numpy().flatten()
    plt.hist(vals, bins=50, alpha=0.5, label=name)
    print(f"{name} - grad std: {vals.std():.4f}")
    print(f"{name} - grad mean: {vals.mean():.4f}")
    print(f"{name} - grad max: {vals.max():.4f}")
    print(f"{name} - grad min: {vals.min():.4f}")
    print("-" * 20

---

[來源: ch11 | 類型: handout]  - grad max: {vals.max():.4f}")
    print(f"{name} - grad min: {vals.min():.4f}")
    print("-" * 20)

---

[來源: ch11 | 類型: handout] - grad max: {vals.max():.4f}")
    print(f"{name} - grad min: {vals.min():.4f}")
    print("-" * 20)

grad_last = grads[1]
    print(f"{name} - last layer grad std: {grad_last.numpy().flatten().std():.4f}")
    print(f"{name} - last layer grad mean: {grad_last.numpy().flatten().mean():.4f}")
    print(f"{name} - last layer grad max: {grad_last.numpy().flatten().max():.4f}")
    print(f"{name} - last layer grad min: {grad_last.numpy().flatten().min():.4f}")
    print("-" * 30)

---

[來源: ch11 | 類型: handout] print(f"{name} - last layer grad min: {grad_last.numpy().flatten().min():.4f}")
    print("-" * 30)

plt.legend()
plt.title("Comparison of Gradients for Different Initializations")
plt.show()
```

---

在深層神經網路中，**若將所有權重初始化為 $0$，模型將無法學習**，其根本原因在於「對稱性無法被打破（Symmetry Breaking）」。

---

[來源: ch11 | 類型: handout] ons")
plt.show()
```

---

在深層神經網路中，**若將所有權重初始化為 $0$，模型將無法學習**，其根本原因在於「對稱性無法被打破（Symmetry Breaking）」。

假設某一層的輸出為 $h_i$，則
$$
h_i = \sigma\left(\sum_j w_{ij} x_j + b_i\right)
$$
若所有權重 $w_{ij} = 0$，且偏置 $b_i$ 也相同，則所有神經元的輸出 $h_i$ 都完全一樣。

---

[來源: ch11 | 類型: handout] sigma\left(\sum_j w_{ij} x_j + b_i\right)
$$
若所有權重 $w_{ij} = 0$，且偏置 $b_i$ 也相同，則所有神經元的輸出 $h_i$ 都完全一樣。

在反向傳播時，權重的梯度為
$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial h_i} \cdot x_j
$$
由於所有 $h_i$ 都一樣，$\frac{\partial \mathcal{L}}{\partial h_i}$ 也會一樣，導致所有 $w_{ij}$ 的梯度完全相同。

---

[來源: ch11 | 類型: handout] \cdot x_j
$$
由於所有 $h_i$ 都一樣，$\frac{\partial \mathcal{L}}{\partial h_i}$ 也會一樣，導致所有 $w_{ij}$ 的梯度完全相同。

因此，權重更新步驟
$$
w_{ij} \leftarrow w_{ij} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}}
$$
會讓每個權重依然保持一致，無法產生差異。這種「對稱性」會讓每一層的所有神經元永遠學到相同的特徵，等同於網路只有一個有效神經元，**模型表現力大幅受限**。

**結論：**  
為了讓每個神經元能學習不同特徵，必須用隨機初始化（如 He、Glorot 等），讓每個權重一開始就有微小差異，才能有效打破對稱性，讓網路具備學習能力。

---

---

[來源: ch11 | 類型: handout] ## 2. 非飽和活化函數 (Nonsaturating Activation Functions)

Sigmoid 在輸入值極大或極小時，梯度幾乎為 0（飽和區），這會導致訓練停滯。

* **關鍵概念：**
  * **ReLU**：計算最快，但可能有「Dead ReLU」問題（神經元輸出永遠為 0）。
  * **Leaky ReLU**：透過負數端的一個小斜率（如 0.01）解決 Dead ReLU 問題。
  * **SELU / GELU / Swish**：平滑且具有自我歸一化（Self-Normalizing）特性，在 Transformer 或深層視覺模型中表現卓越。

---

[來源: ch11 | 類型: handout] Dead ReLU 問題。
  * **SELU / GELU / Swish**：平滑且具有自我歸一化（Self-Normalizing）特性，在 Transformer 或深層視覺模型中表現卓越。

* **⚡ 補充練習 2：**
  1. 什麼是「自我歸一化（Self-Normalization）」？在使用 SELU 活化函數時，需要滿足哪些條件（例如輸入特徵需標準化、初始化方法等）？
  2. 實作練習：定義一個 `LeakyReLU` 活化函數，將斜率 $\alpha$ 設為 0.2，並套用至一個隱藏層中。

```python
import tensorflow as tf
leaky_relu_layer = tf.keras.layers.LeakyReLU(alpha=0.2)

---

[來源: ch11 | 類型: handout] # 方法 1：作為獨立層插入
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, input_shape=(100,)),
  leaky_relu_layer,
  tf.keras.layers.Dense(10, activation='softmax')
])

---

[來源: ch11 | 類型: handout] # 方法 2：直接在 Dense 層指定 activation 參數
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation=leaky_relu_layer, input_shape=(100,)),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

---

---

[來源: ch11 | 類型: handout] ## 3. 批次歸一化 (Batch Normalization, BN)

BN 是訓練深層模型最常用的技術之一，它在每一層活化函數前後對資料進行標準化。

* **關鍵概念：**
  * **目的**：減輕內部協方差偏移（Internal Covariate Shift），允許使用更高的學習率（Learning Rate）。
  * **運算**：使用小批次的均值與變異數進行縮放，並引入可學習的偏移量（beta）與縮放因子（gamma）。
  * **缺點**：會增加每輪訓練的時間成本，且在推論（Inference）時需要處理統計量的差異。

---

[來源: ch11 | 類型: handout] *：使用小批次的均值與變異數進行縮放，並引入可學習的偏移量（beta）與縮放因子（gamma）。
  * **缺點**：會增加每輪訓練的時間成本，且在推論（Inference）時需要處理統計量的差異。

* **⚡ 補充練習 3：**
  1. 批次歸一化（BN）通常放在活化函數「之前」還是「之後」？請查閱最新文獻並簡述不同學派的看法。
  2. 在 Keras 中實作一個含有 10 個 Hidden Layers 的模型，在有 BN 與沒有 BN 的情況下，觀察模型對學習率（Learning Rate）的耐受度差異。

```python
import tensorflow as tf
import matplotlib.pyplot as plt

---

[來源: ch11 | 類型: handout] # Prepare dataset
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
X_train, X_valid, X_test = X_train / 255, X_valid / 255, X_test / 255

---

[來源: ch11 | 類型: handout] # Standardize the dataset
pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

---

[來源: ch11 | 類型: handout] # Build model with and without Batch Normalization
def build_model(use_bn=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[28,28])) # 以 MNIST 或 Fashion MNIST 為例
    model.add(tf.keras.layers.Flatten())
    for _ in range(10):
        model.add(tf.keras.layers.Dense(256, kernel_initializer="he_normal"))
        if use_bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU()) # 或者 tf.keras.layers.Activation('relu')
    model.add(tf.keras.layers.Dense(10, activation='softmax')) # 輸出層依任務而定
    return model

---

[來源: ch11 | 類型: handout] # visualize the learning curve of both models to compare their training speed and performance with and without batch normalization for the various learning rates
epochs = 30
for lr in [0.1, 0.01, 0.001, 0.0001]:
    # rebuild the models on each loop iteration to ensure independence
    model_no_bn = build_model(use_bn=False)
    model_with_bn = build_model(use_bn=True)

---

[來源: ch11 | 類型: handout] ndependence
    model_no_bn = build_model(use_bn=False)
    model_with_bn = build_model(use_bn=True)

model_no_bn.compile(loss="sparse_categorical_crossentropy",
                        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                        metrics=["accuracy"])
    model_with_bn.compile(loss="sparse_categorical_crossentropy",
                          optimizer=tf.keras.opti

---

[來源: ch11 | 類型: handout] bn.compile(loss="sparse_categorical_crossentropy",
                          optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                    metrics=["accuracy"])
    print(f"\n--- Training with Learning Rate: {lr} ---") # Added for clarity
    print("Training model_no_bn...")
    history_no_bn = model_no_bn.fit(X_train_scaled, y_train, epochs=epochs,
                                    v

---

[來源: ch11 | 類型: handout] o_bn = model_no_bn.fit(X_train_scaled, y_train, epochs=epochs,
                                    validation_data=(X_valid_scaled, y_valid),
                                    verbose=0) # Set verbose to 0 to avoid printing all epoch logs
    print("Training model_with_bn...")
    history_with_bn = model_with_bn.fit(X_train_scaled, y_train, epochs=epochs,
                                        

---

[來源: ch11 | 類型: handout] = model_with_bn.fit(X_train_scaled, y_train, epochs=epochs,
                                        validation_data=(X_valid_scaled, y_valid),
                                        verbose=0) # Set verbose to 0 to avoid printing all epoch logs

---

[來源: ch11 | 類型: handout] verbose=0) # Set verbose to 0 to avoid printing all epoch logs

print("\nEvaluation results:")
    loss_no_bn, acc_no_bn = model_no_bn.evaluate(X_test_scaled, y_test, verbose=0)
    loss_with_bn, acc_with_bn = model_with_bn.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Model No BN (LR={lr}) - Test Loss: {loss_no_bn:.4f}, Test Accuracy: {acc_no_bn:.4f}")
    print(f"Model With BN (LR={lr})

---

[來源: ch11 | 類型: handout]  - Test Loss: {loss_no_bn:.4f}, Test Accuracy: {acc_no_bn:.4f}")
    print(f"Model With BN (LR={lr}) - Test Loss: {loss_with_bn:.4f}, Test Accuracy: {acc_with_bn:.4f}")

---

[來源: ch11 | 類型: handout] print(f"Model With BN (LR={lr}) - Test Loss: {loss_with_bn:.4f}, Test Accuracy: {acc_with_bn:.4f}")

# plot the learning curves for both models here
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # use each model's history to draw the curves
    plt.plot(history_no_bn.history['accuracy'], '--' ,label='No BN - Train Acc')
    plt.plot(history_no_bn.history['val_accuracy'], '--', label

---

[來源: ch11 | 類型: handout] '], '--' ,label='No BN - Train Acc')
    plt.plot(history_no_bn.history['val_accuracy'], '--', label='No BN - Val Acc')
    plt.plot(history_with_bn.history['accuracy'], label='BN - Train Acc')
    plt.plot(history_with_bn.history['val_accuracy'], label='BN - Val Acc')
    plt.title(f'Learning Curves (LR={lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
``

---

[來源: ch11 | 類型: handout] s (LR={lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
```

---

---

[來源: ch11 | 類型: handout] ## 4. 遷移學習 (Transfer Learning)

在電機工程中，「重用」是提高效率的關鍵。遷移學習讓我們可以站在巨人的肩膀上。

* **關鍵概念：**
  * **凍結（Freezing）**：在初期訓練時，固定預訓練層的權重，只訓練頂層的分類器。
  * **微調（Fine-tuning）**：在模型收斂後，以極低學習率解凍部分底層進行細微調整。

* **⚡ 補充練習 4：**
  1. 為什麼在進行微調（Fine-tuning）時，建議使用較小的學習率（如 $10^{-5}$）？
  2. 練習：載入 `VGG16` 的權重，凍結所有卷積層，僅更換最後的 Dense Layer 來進行 CIFAR-10 的圖像分類，並觀察準確率。

```python
import tensorflow as tf

---

[來源: ch11 | 類型: handout] # 載入 CIFAR-10 資料集
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0

---

[來源: ch11 | 類型: handout] # 1. 載入預訓練模型，不包含原本的 1000 類輸出層 (include_top=False)
base_model = tf.keras.applications.VGG16(weights='imagenet', 
                                          include_top=False, 
                                          input_shape=(32, 32, 3))

---

[來源: ch11 | 類型: handout] # 2. 凍結卷積層，防止權重在訓練初期被破壞
base_model.trainable = False

---

[來源: ch11 | 類型: handout] # 3. 建立新模型，串接自定義的 Dense Layer
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # CIFAR-10 有 10 個類別
])

---

[來源: ch11 | 類型: handout] # 4. 編譯模型並觀察準確率
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

---

[來源: ch11 | 類型: handout] # 5. 訓練模型
history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_valid, y_valid))

---

[來源: ch11 | 類型: handout] # 6. 評估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

---

---

[來源: ch11 | 類型: handout] ## 5. 進階優化器 (Advanced Optimizers)

除了隨機梯度下降（SGD），我們有更多工具來加速在複雜地形（Loss Landscape）中的搜索。

* **關鍵概念：**
  * **Momentum (動量)**：模擬物理慣性，加速越過局部最小值與鞍點。
  * **Adam / AdamW**：自動調整每個參數的學習率。**AdamW** 修復了 Adam 在權重衰減（Weight Decay）上的缺陷，是目前最推薦的優化器。
  * **Nadam**：Adam 加上 Nesterov 加速動量。

---

[來源: ch11 | 類型: handout] 每個參數的學習率。**AdamW** 修復了 Adam 在權重衰減（Weight Decay）上的缺陷，是目前最推薦的優化器。
  * **Nadam**：Adam 加上 Nesterov 加速動量。

* **⚡ 補充練習 5：**
  1. 簡述 Adam 與 SGD + Momentum 的權衡（Trade-off）：哪一個在初期收斂快？哪一個通常能達到更好的最終泛化（Generalization）？
  2. 實作比較：使用同一個模型分別配置 `Adam(learning_rate=0.001)` 與 `SGD(learning_rate=0.01, momentum=0.9)` 訓練 20 個 Epochs，繪製 Loss 曲線圖。

---

[來源: ch11 | 類型: handout] 配置 `Adam(learning_rate=0.001)` 與 `SGD(learning_rate=0.01, momentum=0.9)` 訓練 20 個 Epochs，繪製 Loss 曲線圖。

```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

---

[來源: ch11 | 類型: handout] # Prepare dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0

---

[來源: ch11 | 類型: handout] # Build a simple model
def build_model(use_bn=False):  
  model = keras.Sequential()

model.add(keras.layers.Input(shape=(28, 28)))
  model.add(keras.layers.Flatten())

for _ in range(15):
    model.add(keras.layers.Dense(300, kernel_initializer="he_normal"))
    if use_bn:
      model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

---

[來源: ch11 | 類型: handout] if use_bn:
      model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
  if use_bn:
    model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.ReLU())

model.add(keras.layers.Dense(10, activation="softmax"))

return model

---

[來源: ch11 | 類型: handout] # Train with Adam
model_adam = build_model()
model_adam.compile(loss="sparse_categorical_crossentropy",
                   optimizer=keras.optimizers.Adam(learning_rate=0.001),
                   metrics=["accuracy"])
history_adam = model_adam.fit(X_train, y_train, epochs=20,
                              validation_data=(X_valid, y_valid),
                              verbose=0)

---

[來源: ch11 | 類型: handout] # Train with SGD + Momentum
model_sgd = build_model()
model_sgd.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  metrics=["accuracy"])
history_sgd = model_sgd.fit(X_train, y_train, epochs=20,
                            validation_data=(X_valid, y_valid),
                            verbose=0)

---

[來源: ch11 | 類型: handout] # Plot Loss Curves and accuracy Curves for both optimizers
plt.figure(figsize=(12, 5))
plt.title("Training Curves: Adam vs SGD + Momentum")
plt.plot(history_adam.history["loss"], ".-",label="Adam Loss")
plt.plot(history_sgd.history["loss"], "-",label="SGD + Momentum Loss")
plt.plot(history_adam.history["accuracy"], ".-",label="Adam Accuracy")
plt.plot(history_sgd.history["accuracy"], "-",label="SGD + Momentum Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.show()

---

[來源: ch11 | 類型: handout] # Use BN to stabilize training and allow higher learning rates

---

[來源: ch11 | 類型: handout] # Train with Adam
model_adam = build_model(use_bn=True)
model_adam.compile(loss="sparse_categorical_crossentropy",
                   optimizer=keras.optimizers.Adam(learning_rate=0.01),
                   metrics=["accuracy"])
history_adam = model_adam.fit(X_train, y_train, epochs=20,
                              validation_data=(X_valid, y_valid),
                              verbose=0)

---

[來源: ch11 | 類型: handout] # Train with SGD + Momentum
model_sgd = build_model(use_bn=True)
model_sgd.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
                  metrics=["accuracy"])
history_sgd = model_sgd.fit(X_train, y_train, epochs=20,
                            validation_data=(X_valid, y_valid),
                            verbose=0)

---

[來源: ch11 | 類型: handout] # Plot Loss Curves and accuracy Curves for both optimizers
plt.figure(figsize=(12, 5))
plt.title("Training Curves with Batch Normalization: Adam vs SGD + Momentum")
plt.plot(history_adam.history["loss"], ".-",label="Adam Loss")
plt.plot(history_sgd.history["loss"], "-",label="SGD + Momentum Loss")
plt.plot(history_adam.history["accuracy"], ".-",label="Adam Accuracy")
plt.plot(history_sgd.history["accuracy"], "-",label="SGD + Momentum Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.show()

---

[來源: ch11 | 類型: handout] # Self-Normalizing Networks (SNNs) with SELU activation and AlphaDropout
def build_model_selu():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())

---

[來源: ch11 | 類型: handout] Sequential()
    model.add(keras.layers.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())

# 使用迴圈建立 15 層深層隱藏層
    for _ in range(15):
        # 關鍵：SELU 必須搭配 lecun_normal 初始化才能實現自我歸一化
        model.add(keras.layers.Dense(300, 
                                     activation="selu", 
                                     kernel_initializer="lecun_normal"))

---

[來源: ch11 | 類型: handout] activation="selu", 
                                     kernel_initializer="lecun_normal"))

# 倒數第二層隱藏層
    model.add(keras.layers.Dense(100, 
                                 activation="selu", 
                                 kernel_initializer="lecun_normal"))

# 輸出層
    model.add(keras.layers.Dense(10, activation="softmax"))

return model

---

[來源: ch11 | 類型: handout] er="lecun_normal"))

# 輸出層
    model.add(keras.layers.Dense(10, activation="softmax"))

return model

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

---

[來源: ch11 | 類型: handout] id_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

model_selu = build_model_selu()
model_selu.compile(loss="sparse_categorical_crossentropy",
                   optimizer=keras.optimizers.Nadam(learning_rate=0.001),
                   metrics=["accuracy"])
history_selu = model_selu.fit(X_train_scaled, y_train, epochs=20,
                              validation_data=(X_valid_scaled, y_valid),
                              verbose=0)

---

[來源: ch11 | 類型: handout] # Plot Loss Curves and accuracy Curves for all optimizers
plt.figure(figsize=(12, 5))
plt.title("Training Curves: Adam vs SGD + Momentum vs SELU")
plt.plot(history_adam.history["loss"], ".-",label="Adam Loss")
plt.plot(history_sgd.history["loss"], "-",label="SGD + Momentum Loss")
plt.plot(history_adam.history["accuracy"], ".-",label="Adam Accuracy")
plt.plot(history_sgd.history["accuracy"], "-",label="SGD + Momentum Accuracy")
plt.plot(history_selu.history["loss"], "--",label="SELU Loss")
plt.plot(history_selu.history["accuracy"], "--",label="SELU Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.show()
```

---

[來源: ch11 | 類型: handout] abel="SELU Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.show()
```

---

---

[來源: ch11 | 類型: handout] ## 6. 學習率排程與正則化 (LR Scheduling & Regularization)

控制學習率的變化以及防止模型「過度擬合」訓練資料。

* **關鍵概念：**
  * **ReduceLROnPlateau**：當驗證集準確率 (accuracy of the validation set) 不再提升時，自動降低學習率。
  * **1Cycle Scheduling**：先快速提高學習率再緩慢降低，能顯著縮短訓練時間（Super-convergence）。
  * **Dropout / MC Dropout**：訓練時隨機關閉神經元。MC Dropout 則是在預測時也開啟 Dropout 來估算模型的不確定性。

---

[來源: ch11 | 類型: handout] Super-convergence）。
  * **Dropout / MC Dropout**：訓練時隨機關閉神經元。MC Dropout 則是在預測時也開啟 Dropout 來估算模型的不確定性。

* **⚡ 補充練習 6：**
  1. 解釋為什麼「權重衰減（L2 Regularization）」可以防止模型權重過大？（從 Loss Function 的懲罰項角度說明）。
  2. 實作練習：定義一個 `MCDropout` 類別，並對測試集進行 50 次預測，計算其預測結果的標準差，視覺化模型在哪些影像上最「猶豫」。

---

[來源: ch11 | 類型: handout] ### 權重衰減（L2 Regularization）與損失函數的關係
在機器學習中，**權重衰減（Weight Decay / L2 Regularization）** 是防止模型過擬合（Overfitting）最常用的技術之一。從損失函數（Loss Function）的角度來看，其運作邏輯如下：

---

[來源: ch11 | 類型: handout] #### 1. 損失函數的構成
在加入 L2 正則化後，模型原本的損失函數 $J(\theta)$ 會被修改為：

$$ J(\mathbf{w}) = \text{MSE}(\mathbf{w}) + \alpha \frac{1}{2} \sum_{i=1}^n w_i^2 $$

這裡的 $\text{MSE}(\mathbf{w})$ 是原始的均方誤差（代表模型對資料的擬合程度），而 $\alpha \frac{1}{2} \sum w_i^2$ 就是**懲罰項（Penalty Term）**。$\alpha$ 是一個超參數，用來控制正則化的強度。

---

[來源: ch11 | 類型: handout] #### 2. 懲罰項的作用
當模型進行訓練（即最小化 $J(\mathbf{w})$）時，最佳化演算法（如梯度下降）現在必須同時兼顧兩個目標：
*   **縮小預測誤差**：讓 $\text{MSE}(\mathbf{w})$ 越小越好。
*   **縮小權重數值**：讓權重的平方和 $\sum w_i^2$ 越小越好。

如果某個權重 $w_i$ 變得非常大，懲罰項會迅速增加（因為是平方關係），進而導致總損失 $J(\mathbf{w})$ 大幅上升。為了降低總損失，最佳化過程會強迫權重往較小的值移動。

---

[來源: ch11 | 類型: handout] #### 3. 梯度下降與權重更新
從數學更新公式來看，權重 $w$ 的梯度更新為：

$$ w_{next} = w - \eta \left( \frac{\partial \text{MSE}}{\partial w} + \alpha w \right) $$

整理後可得：
$$ w_{next} = (1 - \eta \alpha) w - \eta \frac{\partial \text{MSE}}{\partial w} $$

其中 $\eta$ 是學習率。由於 $(1 - \eta \alpha)$ 是一個略小於 1 的數值，這意味著在每一次更新權重之前，系統都會先將權重**等比例地縮小（Decay）**。只有當資料產生的梯度 $\frac{\partial \text{MSE}}{\partial w}$ 足夠強大到能抵銷這個縮小效果時，權重才會保持較大的數值。

---

[來源: ch11 | 類型: handout] #### 4. 結論
權重衰減透過在損失函數中引入「對權重大小的成本」，確保模型不會為了完美擬合每一個訓練樣本（包含雜訊）而產生極端的權重值。這使得模型函數變得更平滑，提高了對未見資料的泛化能力。

---

[來源: ch11 | 類型: handout] ### 實作 `MCDropout` 類別並視覺化不確定性
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch11 | 類型: handout] # Prepare dataset
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0

---

[來源: ch11 | 類型: handout] :], y_train_full[-5000:]
X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

---

[來源: ch11 | 類型: handout] # 1. 定義 MC Dropout 類別
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        # 強制在推論階段也保持 Dropout 開啟
        return super().call(inputs, training=True)

---

[來源: ch11 | 類型: handout] # 2. 假設我們有一個模型並套用此層
model_mc = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(28, 28)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
  MCDropout(rate=0.2),
  tf.keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch11 | 類型: handout] # 3. 編譯並訓練模型
model_mc.compile(loss="sparse_categorical_crossentropy",
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 metrics=["accuracy"])
model_mc.fit(X_train_scaled, y_train, epochs=20,
             validation_data=(X_valid_scaled, y_valid),
              verbose=1)

---

[來源: ch11 | 類型: handout] # 4. 進行 100 次預測並計算標準差
y_probas = np.stack([model_mc.predict(X_test_scaled) for _ in range(100)])
y_std = y_probas.std(axis=0) # 形狀為 (10000, 10) -> 10000 個樣本各個類別的標準差

---

[來源: ch11 | 類型: handout] # 找出預測類別中標準差最高的樣本，即為最「猶豫」的影像
uncertainty = y_std.max(axis=1) # 每個樣本的最大標準差
most_uncertain_idx = np.argsort(uncertainty)[-5:] # 取出最不確定的前 5 個樣本

---

[來源: ch11 | 類型: handout] # 定義類別名稱（以 Fashion MNIST 為例）
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

---

[來源: ch11 | 類型: handout] # 計算平均預測機率
y_probas_mean = y_probas.mean(axis=0)

---

[來源: ch11 | 類型: handout] # 建立畫布：每一列顯示一個樣本（左圖：影像，右圖：機率分佈）
n_samples = len(most_uncertain_idx)
fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))

---

[來源: ch11 | 類型: handout] # 從最不確定的樣本開始顯示 (逆序)
for i, idx in enumerate(most_uncertain_idx[::-1]):
    # --- 左圖：原始影像 ---
    axes[i, 0].imshow(X_test[idx], cmap="binary")
    axes[i, 0].set_title(f"Actual: {class_names[y_test[idx]]}")
    axes[i, 0].axis('off')

---

[來源: ch11 | 類型: handout] "binary")
    axes[i, 0].set_title(f"Actual: {class_names[y_test[idx]]}")
    axes[i, 0].axis('off')

# --- 右圖：預測機率分佈與標準差 ---
    # 使用 yerr=y_std[idx] 來繪製誤差棒，代表 MC Dropout 的不確定性
    axes[i, 1].bar(class_names, y_probas_mean[idx], 
                   yerr=y_std[idx], capsize=5, color="skyblue", edgecolor="black")
    axes[i, 1].set_ylim(0, 1.1) # 預留空間給誤差棒
    axes[i, 1].set_title(f"MC Dropout Proba

---

[來源: ch11 | 類型: handout] olor="black")
    axes[i, 1].set_ylim(0, 1.1) # 預留空間給誤差棒
    axes[i, 1].set_title(f"MC Dropout Probabilities (Max Std: {uncertainty[idx]:.4f})")
    plt.setp(axes[i, 1].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()
```

---

---

[來源: ch11 | 類型: handout] ## 結語

掌握這些技術，代表你已經具備了處理「深度」模型的能力。下週我們將進入第 14 章：卷積神經網路（CNN），並結合今天所學的 BN、Dropout 與 He 初始化。

**課後作業：** 請完成筆記本（Notebook）最後的 CIFAR-10 綜合練習，目標是在不使用卷積層的情況下，透過本章優化技術讓準確率達到 50% 以上。

---

[來源: ch11 | 類型: tutorial] [標題: 訓練深度神經網路完整指南：梯度消失、BatchNorm、遷移學習與優化器 | 描述: 深入深度神經網路訓練技術：梯度消失/爆炸問題、Xavier/He 初始化、Batch Normalization、梯度裁剪、遷移學習、Adam/Nadam 優化器、學習率排程、Dropout 與 Max-Norm 正則化。 | 關鍵字: Python, 深度學習, Keras, TensorFlow, Batch Normalization, 遷移學習, Adam, Dropout, 梯度消失, 神經網路]
# 訓練深度神經網路：從梯度消失到 Adam 優化器

為什麼深度神經網路難以訓練？**梯度消失（Vanishing Gradient）** 讓早期層幾乎停止學習；**梯度爆炸（Exploding Gradient）** 讓參數更新失控。本教學帶你掌握現代深度學習中所有關鍵的訓練技術，讓你的神經網路真正學得快、學得好。

---

[來源: ch11 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **He 初始化** + **ReLU/ELU 激活函數** 是現代深度網路的標準起點
- **Batch Normalization（批次標準化）** 大幅加速訓練，允許更大學習率
- **Adam/Nadam** 是最常用的優化器，通常無需大幅調整超參數
- **遷移學習（Transfer Learning）** 在小資料集上效果顯著，是實務首選策略
- **Dropout** 是最有效的正則化之一；**MC Dropout** 可獲得不確定性估計

---

---

[來源: ch11 | 類型: tutorial] ## 梯度消失/爆炸問題

💡 **實際應用情境：** 在 2012 年以前，深度網路幾乎無法訓練。當 Sigmoid 的梯度只有 ~0.25，通過 10 層後梯度縮小到 0.25^10 ≈ 0.000001——早期層幾乎不學習。理解這個問題是掌握所有後續技術的基礎。

梯度消失的根本原因：Sigmoid 函數的飽和區梯度趨近於 0，反向傳播時梯度逐層相乘後指數衰減。

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

---

[來源: ch11 | 類型: tutorial] # 示範：Sigmoid 的梯度問題
x = tf.Variable(np.linspace(-5, 5, 1000), dtype=tf.float32)

with tf.GradientTape() as tape:
    y_sigmoid = tf.sigmoid(x)
    y_relu = tf.nn.relu(x)

grad_sigmoid = tape.gradient(y_sigmoid, x)  # 最大梯度約 0.25
print(f"Sigmoid 最大梯度: {grad_sigmoid.numpy().max():.4f}")  # ≈ 0.25
```

解決方案演進：

| 技術 | 解決問題 | 出現年份 |
|------|---------|---------|
| ReLU 激活 | 梯度消失 | 2010 |
| Xavier/He 初始化 | 梯度消失/爆炸 | 2010/2015 |
| Batch Normalization | 梯度問題 + 加速訓練 | 2015 |
| 梯度裁剪 | 梯度爆炸（RNN） | 常用 |

---

---

[來源: ch11 | 類型: tutorial] ### 範例 1: 各種初始化方法

```python
from tensorflow import keras

---

[來源: ch11 | 類型: tutorial] # 方差 = 2 / (fan_in + fan_out)
layer_xavier = keras.layers.Dense(
    300,
    activation="sigmoid",
    kernel_initializer="glorot_uniform"  # Xavier 均勻分佈（預設）
)

---

[來源: ch11 | 類型: tutorial] # 方差 = 2 / fan_in
layer_he = keras.layers.Dense(
    300,
    activation="relu",
    kernel_initializer="he_normal"  # He 常態分佈
)

---

[來源: ch11 | 類型: tutorial] # LeCun 初始化（適合 SELU）
layer_lecun = keras.layers.Dense(
    300,
    activation="selu",
    kernel_initializer="lecun_normal"
)

---

[來源: ch11 | 類型: tutorial] # 完整網路：He 初始化 + ReLU（現代標準）
model_he = keras.Sequential([
    keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dense(10,  activation="softmax")
])
model_he.summary()
```

**✅ 程式碼逐行解析：**

1. `kernel_initializer="glorot_uniform"`: 從均勻分佈 $U[-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})}]$ 初始化
2. `kernel_initializer="he_normal"`: 從常態分佈 $N(0, \sqrt{2/n_{in}})$ 初始化，為 ReLU 設計
3. 激活函數和初始化的配對規則：Sigmoid/tanh → Xavier；ReLU/Leaky ReLU/ELU → He；SELU → LeCun

**🎯 重點摘要:**

---

[來源: ch11 | 類型: tutorial] ，為 ReLU 設計
3. 激活函數和初始化的配對規則：Sigmoid/tanh → Xavier；ReLU/Leaky ReLU/ELU → He；SELU → LeCun

**🎯 重點摘要:**

- 不匹配的初始化可能導致梯度在第一個 epoch 就消失或爆炸
- 使用 ReLU + He 初始化是目前最常用的安全起點

---

---

[來源: ch11 | 類型: tutorial] ## Batch Normalization

💡 **實際應用情境：** 批次正規化大幅降低了「內部協變量偏移（Internal Covariate Shift）」問題——每層輸入的分佈不斷改變，後面的層需要不斷適應。BN 讓每一層的輸入分佈穩定在 N(0,1)。

---

[來源: ch11 | 類型: tutorial] ### 範例 2: 使用 Batch Normalization

```python

---

[來源: ch11 | 類型: tutorial] # 方法 1：BN 在激活函數之前（原始論文）
model_bn_before = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),   # ← BN 在激活函數之前
    keras.layers.Dense(300, use_bias=False),  # BN 有 bias，Dense 不需要
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(100, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch11 | 類型: tutorial] # 方法 2：BN 在激活函數之後（更常見的實務做法）
model_bn_after = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),   # ← BN 在激活函數之後
    keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

model_bn_after.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model_bn_after.summary()
```

**✅ 程式碼逐行解析：**

---

[來源: ch11 | 類型: tutorial] e_categorical_crossentropy",
    metrics=["accuracy"]
)
model_bn_after.summary()
```

**✅ 程式碼逐行解析：**

1. `BatchNormalization()`: 在訓練時計算 batch 的均值/方差正規化；推論時使用訓練期間的移動平均
2. `use_bias=False`: BN 層有自己的偏置參數（beta），所以 Dense 層的 bias 是多餘的
3. BN 有 4 個可學習參數：gamma（縮放）、beta（平移）和用於推論的移動平均/方差

**🎯 重點摘要:**

- BN 的主要優點：允許更大的學習率（加速訓練）、提供一定的正則化效果
- 批次太小（< 16）時 BN 效果不好，可改用 Layer Normalization

---

---

[來源: ch11 | 類型: tutorial] ### 範例 3: 梯度裁剪（RNN 的關鍵技術）

```python

---

[來源: ch11 | 類型: tutorial] # 梯度裁剪：防止梯度爆炸（RNN/LSTM 中特別重要）
optimizer_clipped = keras.optimizers.Adam(
    learning_rate=1e-3,
    clipnorm=1.0    # 梯度的 L2 範數裁剪至 1.0
    # 或 clipvalue=0.5  # 每個梯度值裁剪至 [-0.5, 0.5]
)

model_clip = keras.Sequential([
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10,  activation="softmax")
])
model_clip.compile(
    optimizer=optimizer_clipped,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

**🎯 重點摘要:**

---

[來源: ch11 | 類型: tutorial] zer_clipped,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

**🎯 重點摘要:**

- `clipnorm=1.0`: 若梯度向量的 L2 範數 > 1，等比例縮放整個梯度向量
- `clipvalue=0.5`: 直接裁剪每個梯度值到 [-0.5, 0.5]（會改變梯度方向）
- RNN/LSTM 訓練時梯度裁剪幾乎是標配

---

---

[來源: ch11 | 類型: tutorial] ## 遷移學習

💡 **實際應用情境：** 台灣醫療新創的皮膚癌偵測系統——沒有足夠的標記資料從頭訓練 CNN，但可以用 ImageNet 預訓練的 ResNet50 作為特徵提取器，只訓練最後幾層。

---

[來源: ch11 | 類型: tutorial] ### 範例 4: 凍結層遷移學習

```python
import tensorflow as tf
from tensorflow import keras

---

[來源: ch11 | 類型: tutorial] # 載入預訓練模型（不包含頂部分類層）
base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,          # 不含最後的全連接層
    input_shape=(224, 224, 3)
)

---

[來源: ch11 | 類型: tutorial] # 凍結預訓練層（不訓練）
base_model.trainable = False

---

[來源: ch11 | 類型: tutorial] # 添加自訂分類頭
model_tl = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),         # 防止過擬合
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax")  # 10 類自訂任務
])

model_tl.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

---

[來源: ch11 | 類型: tutorial] # history_1 = model_tl.fit(train_dataset, epochs=10, validation_data=val_dataset)

---

[來源: ch11 | 類型: tutorial] # 第二階段（Fine-tuning：解凍部分底層）
base_model.trainable = True

---

[來源: ch11 | 類型: tutorial] # 凍結前 100 層，只 fine-tune 後面的層
for layer in base_model.layers[:100]:
    layer.trainable = False

model_tl.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # 使用極小的學習率！
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

---

[來源: ch11 | 類型: tutorial] # history_2 = model_tl.fit(train_dataset, epochs=10, validation_data=val_dataset)
print(f"可訓練參數: {sum(p.numpy().size for p in model_tl.trainable_variables):,}")
```

**✅ 程式碼逐行解析：**

1. `weights="imagenet"`: 載入在 ImageNet 上預訓練的權重（包含豐富的視覺特徵知識）
2. `base_model.trainable = False`: 凍結所有層，只訓練新增的分類頭（快速收斂）
3. Fine-tuning 階段使用極小學習率（1e-5）：避免破壞預訓練的有價值權重

**🎯 重點摘要:**

- 第一階段（凍結基礎層）：幾個 epoch 即可得到不錯的結果
- 第二階段（Fine-tuning）：使用更小的學習率，逐步調整底層特徵
- 資料越少，凍結的層數應越多（避免過擬合）

---

---

[來源: ch11 | 類型: tutorial] ### 範例 5: Adam、Nadam、AdaGrad 配置

```python

---

[來源: ch11 | 類型: tutorial] # Adam（Adaptive Moment Estimation）：最常用

---

[來源: ch11 | 類型: tutorial] # 結合 Momentum（第一矩）+ RMSProp（第二矩）
optimizer_adam = keras.optimizers.Adam(
    learning_rate=1e-3,
    beta_1=0.9,    # 第一矩（動量）的衰減率
    beta_2=0.999,  # 第二矩（RMSProp）的衰減率
    epsilon=1e-7
)

---

[來源: ch11 | 類型: tutorial] # Nadam = Nesterov + Adam（略優於 Adam）
optimizer_nadam = keras.optimizers.Nadam(learning_rate=1e-3)

---

[來源: ch11 | 類型: tutorial] # AdaGrad（適合稀疏資料，如 NLP）
optimizer_adagrad = keras.optimizers.Adagrad(learning_rate=0.01)

---

[來源: ch11 | 類型: tutorial] # SGD + Nesterov Momentum（收斂質量通常最好，需要更多調參）
optimizer_sgd = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True  # Nesterov 加速梯度
)
```

| 優化器 | 特點 | 建議場景 |
|--------|------|---------|
| SGD + Momentum | 收斂質量好，但調參難 | 有充足資源調參時 |
| AdaGrad | 學習率自適應，但可能過早衰減 | NLP 稀疏梯度 |
| RMSProp | 解決 AdaGrad 衰減問題 | RNN |
| Adam | 自適應 + 動量，開箱即用 | **大多數情況的預設選擇** |
| Nadam | Adam + Nesterov | 需要略好效果時 |

---

---

[來源: ch11 | 類型: tutorial] ### 範例 6: 指數衰減與 1Cycle 排程

```python

---

[來源: ch11 | 類型: tutorial] # 指數衰減（每 100 個 step 衰減 10%）
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,         # 每 100 步衰減一次
    decay_rate=0.9,          # 衰減率
    staircase=True           # 階梯式衰減（True）vs 連續衰減（False）
)

---

[來源: ch11 | 類型: tutorial] # 分段常數衰減
lr_piecewise = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[50000, 100000],  # 在第 50k 和 100k 步衰減
    values=[1e-3, 1e-4, 1e-5]   # 對應的學習率
)

---

[來源: ch11 | 類型: tutorial] # 使用排程
optimizer_with_schedule = keras.optimizers.Adam(learning_rate=lr_schedule)
```

**🎯 重點摘要:**

- 好的學習率策略：熱身期（線性增加）→ 高學習率期 → 衰減期
- 簡單有效：`ExponentialDecay` 或 `ReduceLROnPlateau`（根據驗證損失自動調整）

---

---

[來源: ch11 | 類型: tutorial] ### 範例 7: Dropout 與 MC Dropout

```python

---

[來源: ch11 | 類型: tutorial] # 標準 Dropout
model_dropout = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dropout(rate=0.3),  # 訓練時隨機丟棄 30% 的神經元
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch11 | 類型: tutorial] # MC Dropout（Monte Carlo Dropout）：推論時也啟用 Dropout

---

[來源: ch11 | 類型: tutorial] # 多次預測取平均 → 得到不確定性估計
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)  # 推論時也保持 Dropout

model_mc = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    MCDropout(rate=0.3),
    keras.layers.Dense(100, activation="relu"),
    MCDropout(rate=0.3),
    keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch11 | 類型: tutorial] # MC Dropout 的不確定性估計（多次預測取均值和方差）

---

[來源: ch11 | 類型: tutorial] # y_probas = np.stack([model_mc.predict(X_test[:1]) for _ in range(100)])

---

[來源: ch11 | 類型: tutorial] # y_mean = y_probas.mean(axis=0)

---

[來源: ch11 | 類型: tutorial] # y_std  = y_probas.std(axis=0)

---

[來源: ch11 | 類型: tutorial] # L1/L2 正則化（對核權重施加懲罰）
model_reg = keras.Sequential([
    keras.layers.Dense(
        300, activation="relu",
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)  # Elastic Net
    ),
    keras.layers.Dense(
        100, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4)
    ),
    keras.layers.Dense(10, activation="softmax")
])
```

**🎯 重點摘要:**

- Dropout 是最有效的正則化之一，通常 rate=0.2~0.5
- MC Dropout：只需修改 `training=True`，即可獲得**不確定性估計**（對醫療 AI 特別有價值）
- Max-Norm 正則化：`kernel_constraint=keras.constraints.max_norm(3.0)`

---

---

[來源: ch11 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: Batch Normalization 放在激活函數之前還是之後？**

A: 原始論文（2015）建議放在激活函數之前；但實務上很多人放在激活函數之後，效果差異不大。若用 ReLU，放在之前可能導致 BN 後的激活值都為正，減少表達能力；放在之後通常更安全。

**Q2: Adam 和 SGD 哪個更好？**

A: 通常 Adam 收斂更快且更穩定（開箱即用），但 SGD + Nesterov + 仔細調參的最終效能可能更好。實務上：先用 Adam 快速得到 baseline，若需最優效能再切換 SGD 精調。

**Q3: 遷移學習適用於哪些場景？**

A: (1) 訓練資料量少（< 10,000）；(2) 新任務與預訓練任務有相似的底層特徵（如都是視覺任務）；(3) 計算資源有限，需要快速收斂。

**Q4: Dropout 的 rate 如何設定？**

A: 通常 0.2~0.5。小型網路用 0.2；大型網路或容易過擬合時用 0.5。輸入層通常用較小的 rate（0.1~0.2），隱藏層用較大的 rate。

---

---

[來源: ch11 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #深度學習 #Keras #TensorFlow #BatchNorm #遷移學習 #Adam #Dropout #梯度消失 #神經網路 #程式設計 #教學 #DataScience #MachineLearning

---

[來源: ch12] [標題: TensorFlow 自訂模型與訓練：從基礎到進階實作 | 描述: 深入探討 TensorFlow 自訂模型、損失函數、層、指標與訓練迴圈的完整指南，包含實務範例與最佳實踐。 | 關鍵字: TensorFlow, 自訂模型, 訓練迴圈, 損失函數, 機器學習, Python, 深度學習]
# TensorFlow 自訂模型與訓練：從基礎到進階實作

這份教學基於《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》第三版第12章，深入探討如何在 **TensorFlow** 中建立**自訂模型 (Custom Models)**、**損失函數 (Loss Functions)**、**層 (Layers)**、**指標 (Metrics)** 以及**訓練迴圈 (Training Loops)**。無論你是初學者還是進階使用者，這份指南將帶你從基礎概念逐步建構複雜的機器學習應用。

---

[來源: ch12] ## 關鍵重點 (Key Takeaways)

- 了解 TensorFlow **張量 (Tensor)** 操作與 NumPy 的差異。
- 實作自訂損失函數與指標，提升模型彈性。
- 建立自訂層與模型架構，打造獨特網路。
- 使用 **GradientTape (梯度帶)** 進行**自動微分 (Autodiff)**，精確控制梯度計算。
- 設計自訂訓練迴圈，掌握訓練過程的每一個細節。

---

---

[來源: ch12] ## 像 NumPy 一樣使用 TensorFlow {#tensorflow-numpy}

💡 **實際應用情境：** **TensorFlow (TF)** 的核心是**張量操作 (Tensor Operations)**，類似 NumPy 但支援 GPU 運算與**自動微分 (Autodiff)**。

---

[來源: ch12] #### 張量

```python
t = tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
t
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 使用 `tf.constant` 建立一個 $2 \times 3$ 的常數張量矩陣，包含浮點數。
    - 注意：`tf.constant` 建立的張量是**不可變 (immutable)** 的，無法直接修改其值。
    - `dtype` 依輸入值自動推斷：浮點數會選 `tf.float32`，整數會選 `tf.int32`。例如 `tf.constant([[1, 2, 3], [4, 5, 6]])` 的預設 dtype 是 `tf.int32`；若要硬性指定，請使用 `dtype=tf.float32`（或其他支援類型）。

**🎯 重點摘要:**

---

[來源: ch12] [[1, 2, 3], [4, 5, 6]])` 的預設 dtype 是 `tf.int32`；若要硬性指定，請使用 `dtype=tf.float32`（或其他支援類型）。

**🎯 重點摘要:**

-   **核心功能**: 建立不可變的張量。
-   **潛在問題**: 張量不可變，若需修改值，應使用 `tf.Variable`。
-   **最佳使用情境**: 定義模型權重或輸入資料。

---

[來源: ch12] #### 索引
```python
t[:, 1:]
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 使用 NumPy 風格的索引方式 `[:, 1:]` 取得張量的子集，表示所有行、從第二列到最後一列。

---

[來源: ch12] #### 操作
```python
t + 10
tf.square(t)
t @ tf.transpose(t)
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 對張量 `t` 中的每個元素執行加 10 的操作。
2.  `第 2 行`: 使用 `tf.square` 計算張量 `t` 中每個元素的平方。
3.  `第 3 行`: 使用 `@` 符號執行矩陣乘法，將 `t` 與其**轉置 (transpose)** `tf.transpose(t)` 相乘。

---

[來源: ch12] ### 張量與 NumPy 互轉
```python
import numpy as np

a = np.array([2., 4., 5.])
tf.constant(a)
t.numpy()
np.array(t)
tf.square(a)
np.square(t)
```

**✅ 程式碼逐行解析：**

1.  `第 3 行`: 將 NumPy 陣列 `a` 轉換為 TensorFlow 張量。
2.  `第 4 行`: 將 TensorFlow 張量 `t` 轉換為 NumPy 陣列。
3.  `第 5-6 行`: 展示 TensorFlow 操作可以直接應用於 NumPy 陣列 (`tf.square(a)`)，以及 NumPy 操作可以直接應用於 TensorFlow 張量 (`np.square(t)`)，顯示兩者良好的互通性。

---

[來源: ch12] ### 型別轉換
> 這一節示範當兩個張量的 dtype 不一致時，TensorFlow 會產生錯誤，並介紹如何使用 `tf.cast()` 做型別統一。

```python
try:
    tf.constant(2.0) + tf.constant(40)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
```

**✅ 程式碼逐行解析：**

1.  `第 2 行`: 嘗試將 `float32` 類型的 `tf.constant(2.0)` 與 `int32` 類型的 `tf.constant(40)` 相加。
2.  `第 3-4 行`: 預期會捕獲 `tf.errors.InvalidArgumentError`，因為 TensorFlow 不會自動進行不同資料型別的隱式轉換。

**解決方法：**

---

[來源: ch12] 加。
2.  `第 3-4 行`: 預期會捕獲 `tf.errors.InvalidArgumentError`，因為 TensorFlow 不會自動進行不同資料型別的隱式轉換。

**解決方法：**

-   `tf.constant(2.0)` 的 dtype 為 `tf.float32`。
-   `tf.constant(40)` 的 dtype 為 `tf.int32`。
-   這兩者直接相加會報錯，因為 TensorFlow 要求運算張量 dtype 一致。

**調整方法：**

---

[來源: ch12] -   `tf.constant(40)` 的 dtype 為 `tf.int32`。
-   這兩者直接相加會報錯，因為 TensorFlow 要求運算張量 dtype 一致。

**調整方法：**

```python
try:
    tf.constant(2.0) + tf.constant(40)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
    val_float = tf.cast(tf.constant(40), dtype=tf.float32)
    print("After casting to float32:", tf.constant(2.0) + val_float)
```

**🎯 重點摘要:**

---

[來源: ch12] pe=tf.float32)
    print("After casting to float32:", tf.constant(2.0) + val_float)
```

**🎯 重點摘要:**

-   TensorFlow 張量運算需要相容的 dtype。
-   使用 `tf.cast()` 做顯式型別轉換可以避免錯誤。
-   這是處理不同型別張量時的推薦模式。

---

[來源: ch12] ### 變數

這一節說明 `tf.Variable` 如何建立可變張量，並示範如何用 `assign` 和 `scatter_nd_update` 等 API 逐元素更新內容。

```python
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
v
v.assign(2 * v)
v[0, 1].assign(42)
v[:, 2].assign([0., 1.])
v.scatter_nd_update(
    indices=[[0, 0], [1, 2]], updates=[100., 200.])
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] ., 1.])
v.scatter_nd_update(
    indices=[[0, 0], [1, 2]], updates=[100., 200.])
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 使用 `tf.Variable` 建立一個可變張量 `v`，常用於模型權重。
2.  `第 2 行`: 顯示變數 `v` 的當前值。
3.  `第 3 行`: 使用 `assign` 方法將 `v` 的所有元素更新為其兩倍的值。
4.  `第 4 行`: 將 `v` 中索引為 `[0, 1]` 的元素（第一列第二行）更新為 `42`。
5.  `第 5 行`: 將 `v` 中所有列、第三行的元素更新為 `[0., 1.]`。
6.  `第 6-7 行`: 使用 `scatter_nd_update` 根據提供的 `indices` 和 `updates` 進行稀疏更新，將 `[0, 0]` 更新為 `100.`，`[1, 2]` 更新為 `200.`。

---

[來源: ch12] ### 字串

這一節展示 TensorFlow 如何處理字串資料，包括位元組字串、Unicode 字串、編碼與解碼操作。

```python
tf.constant(b"hello world")
tf.constant("café")
u = tf.constant([ord(c) for c in "café"])
b = tf.strings.unicode_encode(u, "UTF-8")
tf.strings.length(b, unit="UTF8_CHAR")
tf.strings.unicode_decode(b, "UTF-8")
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] 8")
tf.strings.length(b, unit="UTF8_CHAR")
tf.strings.unicode_decode(b, "UTF-8")
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 建立一個位元組字串張量 `b"hello world"`。
2.  `第 2 行`: 建立一個 Unicode 字串張量 `"café"`。
3.  `第 3 行`: 將字串 `"café"` 轉換為 Unicode 編碼的整數張量 `u`。
    - `ord(c)` 會返回字元 `c` 的 Unicode 代碼點（例如 `ord('c')` 返回 `99`）。
    - 因此 `u` 會包含 `[99, 97, 102, 233]`，分別對應於 `c`, `a`, `f`, `é` 的 Unicode 代碼點。
4.  `第 4 行`: 使用 `tf.st

---

[來源: ch12]     - 因此 `u` 會包含 `[99, 97, 102, 233]`，分別對應於 `c`, `a`, `f`, `é` 的 Unicode 代碼點。
4.  `第 4 行`: 使用 `tf.strings.unicode_encode` 將 `u` 編碼為 UTF-8 位元組字串張量 `b`。
    - `b` 的值將是 `b'caf\xc3\xa9'`，其中 `\xc3\xa9` 是 `é` 的 UTF-8 編碼。
5.  `第 5 行`: 使用 `tf.strings.length` 計算 `b` 中每個字串的 UTF-8 字元長度。
6.  `第 6 行`: 使用 `tf.strings.unicode_decode` 將 `b` 解碼回 Unicode 字串張量。
    - 解碼後的張量將包含 `[99, 97, 102, 233]` 等 Unicode 代碼點；若要再轉

---

[來源: ch12] ings.unicode_decode` 將 `b` 解碼回 Unicode 字串張量。
    - 解碼後的張量將包含 `[99, 97, 102, 233]` 等 Unicode 代碼點；若要再轉回 `'café'` 字串，可搭配 `tf.strings.unicode_encode` 或其他字串處理函式進一步轉換。

---

---

[來源: ch12] ## 自訂損失函數 (Custom Loss Functions) {#custom-loss}

💡 **實際應用情境：** **Huber 損失函數 (Huber Loss Function)** 對於處理**離群值 (Outliers)** 比**均方誤差 (Mean Squared Error, MSE)** 更穩定。

---

[來源: ch12] **Huber 損失函數 (Huber Loss Function)** 對於處理**離群值 (Outliers)** 比**均方誤差 (Mean Squared Error, MSE)** 更穩定。

```python
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] .abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
```

**✅ 程式碼逐行解析：**

1.  `第 2 行`: 計算真實值 `y_true` 與預測值 `y_pred` 之間的誤差 `error`。
2.  `第 3 行`: 判斷誤差的絕對值是否小於 `1`，儲存為布林張量 `is_small_error`。
3.  `第 4 行`: 如果誤差小，計算平方損失 ($error^2 / 2$)。
4.  `第 5 行`: 如果誤差大，計算線性損失 ($

|error| - 0.5$)。


5.  `第 6 行`: 使用 `tf.where` 根據 `is_small_error` 的條件，選擇性地返回 `squared_loss` 或 `linear_loss`。

---

[來源: ch12] - 0.5$)。


5.  `第 6 行`: 使用 `tf.where` 根據 `is_small_error` 的條件，選擇性地返回 `squared_loss` 或 `linear_loss`。

**🎯 重點摘要:**

-   **核心功能**: 實作 Huber 損失，能減少離群值對訓練的影響。
-   **潛在問題**: 閾值 (threshold) 的選擇會影響損失函數的行為。
-   **最佳使用情境**: 對離群值敏感的迴歸任務。

```python

---

[來源: ch12] # visualize the Huber loss function
y_pred = tf.linspace(-3, 3, 100)
y_true = tf.constant(0., dtype=y_pred.dtype)
import matplotlib.pyplot as plt
plt.plot(y_pred, huber_fn(y_true, y_pred), label="Huber Loss (delta=1)", linewidth=2)
plt.axvline(x=1, color='r', linestyle='--', alpha=0.5, label="Threshold (delta)")
plt.axvline(x=-1, color='r', linestyle='--', alpha=0.5)
plt.title("Huber Loss Visualization")
plt.xlabel("Error (y_true - y_pred)")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()
```

---

[來源: ch12] plt.xlabel("Error (y_true - y_pred)")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()
```

---

---

[來源: ch12] ## 儲存與載入自訂物件的模型 (Saving/Loading Models with Custom Objects) {#saving-loading}

💡 **實際應用情境：** 儲存包含**自訂元件 (Custom Components)** 的模型時，需要特別處理，以便在載入時 TensorFlow 能夠識別這些自訂物件。

```python
model.save("my_model_with_a_custom_loss.keras")
model = tf.keras.models.load_model("my_model_with_a_custom_loss.keras",
                                   custom_objects={"huber_fn": huber_fn})
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] eras",
                                   custom_objects={"huber_fn": huber_fn})
```

**✅ 程式碼逐行解析：**

1.  `第 1 行`: 使用 `model.save` 將訓練好的模型儲存為 Keras 格式 (`.keras` 副檔名)。
2.  `第 2-3 行`: 使用 `tf.keras.models.load_model` 載入模型，並透過 `custom_objects` 參數明確告知 TensorFlow 如何解析自訂的 `huber_fn` 損失函數。

**🎯 重點摘要:**

---

[來源: ch12] models.load_model` 載入模型，並透過 `custom_objects` 參數明確告知 TensorFlow 如何解析自訂的 `huber_fn` 損失函數。

**🎯 重點摘要:**

-   **核心功能**：保存自訂元件時需明確告知載入器對應的函數/類別。
-   **潛在問題**：未提供 `custom_objects` 會導致 `ValueError: Unknown loss function` 等錯誤。
-   **最佳使用情境**：部署時將模型搬到不同環境（例如雲端服務），必須確保所有自訂程式碼可用。

---

---

[來源: ch12] ## 其他自訂函數 (Other Custom Functions) {#other-custom-functions}

💡 **實際應用情境：** 除了損失函數，你也可以自訂**活化函數 (Activation Functions)**、**權重初始化器 (Weight Initializers)**、**正規化器 (Regularizers)** 和**約束條件 (Constraints)**。

---

[來源: ch12] ation Functions)**、**權重初始化器 (Weight Initializers)**、**正規化器 (Regularizers)** 和**約束條件 (Constraints)**。

```python
def my_softplus(z):
    return tf.math.log(1.0 + tf.exp(z))

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)
```

---

[來源: ch12] my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] hts(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)
```

**✅ 程式碼逐行解析：**

1.  `第 1-2 行`: 定義一個自訂的 **Softplus 活化函數 (Softplus Activation Function)**，計算 $log(1 + e^z)$。
2.  `第 4-6 行`: 定義一個自訂的 **Glorot 權重初始化器 (Glorot Weight Initializer)**，根據輸入和輸出單元數計算標準差，然後生成常態分佈的隨機值。
3.  `第 8-9 行`: 定義一個自訂的 **L1 正規化器 (L1 Regularizer)**，計算權重絕對值的總和並乘以一個因子。
4.  `第 11-12 行`: 定義一個自訂的 **正權重約束 (Positive Weight Constraint)**，將所有負權重設置為零。

---

[來源: ch12] er)**，計算權重絕對值的總和並乘以一個因子。
4.  `第 11-12 行`: 定義一個自訂的 **正權重約束 (Positive Weight Constraint)**，將所有負權重設置為零。

---

---

[來源: ch12] ## 自訂指標 (Custom Metrics) {#custom-metrics}

💡 **實際應用情境：** 自訂指標用於追蹤訓練過程中的特定效能測量。
自訂指標（Custom Metrics）讓你能夠在訓練過程中追蹤特定的效能指標，除了內建的準確率、損失等，也可以根據任務需求設計專屬的評估方式。例如在回歸問題中，除了均方誤差（MSE），你可能會希望追蹤 Huber 損失或其他自訂指標。自訂指標通常繼承自 `tf.keras.metrics.Metric`，並實作 `update_state()`、`result()` 及 `get_config()` 方法，確保在訓練、驗證與模型儲存時都能正確運作。

---

[來源: ch12]  `tf.keras.metrics.Metric`，並實作 `update_state()`、`result()` 及 `get_config()` 方法，確保在訓練、驗證與模型儲存時都能正確運作。

```python
class HuberMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_metrics = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(sample_metrics))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```

---

[來源: ch12] base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] super().get_config()
        return {**base_config, "threshold": self.threshold}
```

**✅ 程式碼逐行解析：**

1.  `第 1-10 行` (`__init__`): 初始化指標，設定 `threshold`，並用 `add_weight` 建立 `total`（累計損失）與 `count`（樣本計數）兩個狀態變數，確保跨批次累加。
2.  `第 12-15 行` (`update_state`): 每次 `update_state()` 被呼叫（batch 內），計算該 batch 的 Huber 損失：`total += batch_loss_sum`，`count += batch_size`。
    - 注意：此實作值用 `tf.shape(y_true)[0]` 取得樣本數（而不是 `

---

[來源: ch12] 損失：`total += batch_loss_sum`，`count += batch_size`。
    - 注意：此實作值用 `tf.shape(y_true)[0]` 取得樣本數（而不是 `tf.size(y_true)` 元素數），避免在多維輸出時產生元素平均誤差。
3.  `第 17-18 行` (`result`): 回傳 `tf.math.divide_no_nan(self.total, self.count)`（平均 Huber 損失），避免 `count=0` 分母錯誤。
4.  `第 20-22 行` (`get_config`): 實作序列化邏輯，確保 `threshold` 儲存在 model.save()/load_model 的 custom_objects 設定裡。

**🎯 重點摘要:**

---

[來源: ch12] `get_config`): 實作序列化邏輯，確保 `threshold` 儲存在 model.save()/load_model 的 custom_objects 設定裡。

**🎯 重點摘要:**

-   **核心功能**：可在訓練過程中保持指標狀態（累積損失、樣本計數）。
-   **注意**：如果手動訓練迴圈，需在每個 epoch 結束呼叫 `metric.reset_state()`；`model.fit()` 內建會自動處理。
-   **最佳使用情境**：對 epoch 平均值/整體訓練趨勢進行跟蹤（如 PR 曲線、累計誤差）。

---

---

[來源: ch12] ## 自訂層 (Custom Layers) {#custom-layers}

💡 **實際應用情境：** 自訂層允許實作特殊的網路架構元件。

---

[來源: ch12] ```python
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="he_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation)}
```

---

[來源: ch12] its": self.units,
                "activation": tf.keras.activations.serialize(self.activation)}
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] ,
                "activation": tf.keras.activations.serialize(self.activation)}
```

**✅ 程式碼逐行解析：**

1.  `第 1-5 行` (`__init__`): 接收超參數（如 `units`）並將 `activation` 字串轉換為對應的活化函數。同時呼叫父類別 `super().__init__(**kwargs)` 以處理標準 Keras 參數（如 `name` 或 `trainable`）。
2.  `第 7-12 行` (`build`): 建立層的權重 (`kernel` 與 `bias`)。這是在層第一次被使用時調用的，我們可以在此時取得輸入形狀 (`batch_input_shape[-1]`) 來動態決定權重矩陣的大小。
3.  `第 14-15 行` (`call`): 定義前向傳播邏輯，執行矩陣乘法並加上偏差項，最後應用活化函數。
4.  `第 17-19 行` (`get_config`): 儲存超參數。這對於模型的序列化至關重要，確保載入模型時能正確重建該自訂層。

---

[來源: ch12] 義前向傳播邏輯，執行矩陣乘法並加上偏差項，最後應用活化函數。
4.  `第 17-19 行` (`get_config`): 儲存超參數。這對於模型的序列化至關重要，確保載入模型時能正確重建該自訂層。

**🎯 重點摘要:**

-   **核心功能**：自訂層可以實現複雜功能（例如自注意力、自訂正則化／約束）。
-   **潛在問題**：若未實作 `get_config()`，模型儲存/載入會失敗。
-   **最佳使用情境**：需要複用自訂動作、或希望透過 `model.save()` 向他人分享層時。

---

---

[來源: ch12] ## 自訂模型 (Custom Models) {#custom-models}

💡 **實際應用情境：** 自訂模型提供對網路架構的完全控制。

---

[來源: ch12] ```python
class ResidualRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden1 = tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "output_dim": self.output_dim}
```

---

[來源: ch12] base_config = super().get_config()
        return {**base_config, "output_dim": self.output_dim}
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] per().get_config()
        return {**base_config, "output_dim": self.output_dim}
```

**✅ 程式碼逐行解析：**

1.  `第 1-10 行` (`__init__`): 初始化模型元件，包括設定輸出維度 `output_dim`，定義一個標準的 `Dense` 隱藏層 `hidden1`，以及兩個 `ResidualBlock` (假設已定義的自訂層)。
2.  `第 12-17 行` (`call`): 定義模型的前向傳播邏輯。輸入 `inputs` 首先通過 `hidden1` 層，然後依序通過多個 `ResidualBlock`。這個範例中，`block1` 被重複使用了 4 次 (1 + 3)，接著通過 `block2`，最後輸出到 `out` 層。
3.  `第 19-21 行` (`get_config`): 實作序列化，確保 `output_dim` 超參數在模型儲存與載入時能夠被正確地保存與重建。

---

[來源: ch12] lock2`，最後輸出到 `out` 層。
3.  `第 19-21 行` (`get_config`): 實作序列化，確保 `output_dim` 超參數在模型儲存與載入時能夠被正確地保存與重建。

**🎯 重點摘要:**

-   **核心功能**：自訂模型讓你自由定義資料流與計算邏輯（例如多分支、條件分支、共用權重）。
-   **潛在問題**：若在 `call()` 內建立變數，會導致重複建立或 tracing 失敗。
-   **最佳使用情境**：當 `Sequential` / functional API 不夠時（例如多任務模型、連續記憶模型）。

---

---

[來源: ch12] ## 基於模型內部的損失 (Losses Based on Model Internals) {#losses-internal}

💡 **實際應用情境：** 將重建損失加入主要損失以改善模型泛化。

---

[來源: ch12] ```python
class ReconstructingRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
                       for _ in range(5)]
        self.out = tf.keras.layers.Dense(output_dim)

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = tf.keras.layers.Dense(n_inputs)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)
```

---

[來源: ch12] re(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] n - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)
```

**✅ 程式碼逐行解析：**

1.  `第 1-8 行` (`__init__`): 初始化模型結構，建立一系列隱藏層並設定輸出層 `out`，為後續 `build` 與 `call` 做準備。
2.  `第 10-12 行` (`build`): 根據輸入形狀動態建立 `reconstruct` 層（輸入維度為 `n_inputs`），這裡在第一次呼叫模型時被執行以確保權重形狀正確。
3.  `第 14-20 行` (`call`): 前向傳播中計算重建輸出 `reconstruction`，接著計算重建損失 `recon_loss`，使用 `add_loss()` 將加權重建損失併入模型總損失，最後回傳主輸出 `out(Z)`。

---

[來源: ch12] 前向傳播中計算重建輸出 `reconstruction`，接著計算重建損失 `recon_loss`，使用 `add_loss()` 將加權重建損失併入模型總損失，最後回傳主輸出 `out(Z)`。

**🎯 重點摘要:**

-   **核心功能**：透過 `add_loss()` 將自訂損失併入模型總損失，讓 `model.compile()` / `model.fit()` 一併優化。
-   **潛在問題**：若在 `call()` 中多次調用 `add_loss()` 可能導致損失重複計算。
-   **最佳使用情境**：多任務訓練、或在網路內部加入自監督損失（如重建、對比學習）。

---

---

[來源: ch12] ## 使用 Autodiff 計算梯度 (Autodiff) {#autodiff}

💡 **實際應用情境：** `tf.GradientTape` 允許精確控制梯度計算。

```python
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] radientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
```

**✅ 程式碼逐行解析：**

1. `第 1-2 行` (`f`, `w1`, `w2`): 定義函數 `f(w1, w2)`，並建立兩個可訓練的變數 `w1` 與 `w2`（使用 `tf.Variable`）。
2. `第 4-6 行` (`GradientTape` 上下文與梯度計算): 使用 `with tf.GradientTape() as tape:` 來追蹤變數上的運算，於上下文內計算標量輸出 `z = f(w1, w2)`，然後呼叫 `tape.gradient(z, [w1, w2])` 取得 `z` 對 `w1` 與 `w2` 的梯度（偏導數）。

**🎯 重點摘要:**

---

[來源: ch12] 計算標量輸出 `z = f(w1, w2)`，然後呼叫 `tape.gradient(z, [w1, w2])` 取得 `z` 對 `w1` 與 `w2` 的梯度（偏導數）。

**🎯 重點摘要:**

-   **核心功能**：`GradientTape` 允許你手動控制梯度計算範圍與更新時機。
-   **潛在問題**：若不妥善管理 `tape` 的生命週期（例如不釋放 persistent tape），會造成記憶體爆增。
-   **最佳使用情境**：自訂優化流程、混合優化器、多任務優化、進階梯度裁剪。

---

---

[來源: ch12] ## 自訂訓練迴圈 (Custom Training Loops) {#custom-training}

💡 **實際應用情境：** 自訂訓練迴圈提供對訓練過程的完全控制。

本節示範如何從頭實作自訂訓練迴圈，包含批次抽樣、手動前向/反向傳播、整合模型內部損失（如 `add_loss()`）、以及更新與重置度量。透過這些範例，你將能在多優化器、對抗訓練或進階梯度處理等非標準情境下精細掌控訓練流程。

---

[來源: ch12] 訓練迴圈，包含批次抽樣、手動前向/反向傳播、整合模型內部損失（如 `add_loss()`）、以及更新與重置度量。透過這些範例，你將能在多優化器、對抗訓練或進階梯度處理等非標準情境下精細掌控訓練流程。

```python
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

def print_status_bar(step, total, loss, metrics=None):
    metrics = " - ".join([f"{m.name}: {m.result():.4f}"
                          for m in [loss] + (metrics or [])])
    end = "" if step < total else "\n"
    print(f"\r{step}/{total} - " + metrics, end=end)

n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.MeanAbsoluteError()]

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train_scaled, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)

        print_status_bar(step, n_steps, mean_loss, metrics)

    for metric in [mean_loss] + metrics:
        metric.reset_state()
```

---

[來源: ch12] teps, mean_loss, metrics)

    for metric in [mean_loss] + metrics:
        metric.reset_state()
```

**✅ 程式碼逐行解析：**

---

[來源: ch12] metrics)

    for metric in [mean_loss] + metrics:
        metric.reset_state()
```

**✅ 程式碼逐行解析：**

1. `第 1-3 行` (`random_batch`): 隨機抽樣一個小批次，回傳 `(X_batch, y_batch)`，用於在訓練迴圈中取樣訓練資料。
2. `第 5-9 行` (`print_status_bar`): 列印訓練進度與目前指標（例如 loss、accuracy），以便即時監控每個 step 的狀態。
3. `第 11-17 行` (訓練參數與度量初始化): 設定 `n_epochs`、`batch_size`、`optimizer`、`loss_fn`，並建立要追蹤的度量（例如 `mean_loss`、`metrics` 列表）。
4. `第 19-35 行` (

---

[來源: ch12] _epochs`、`batch_size`、`optimizer`、`loss_fn`，並建立要追蹤的度量（例如 `mean_loss`、`metrics` 列表）。
4. `第 19-35 行` (訓練迴圈主體): 主要訓練流程：從 `random_batch` 取得批次、使用 `tf.GradientTape()` 計算 forward 與總損失（`main_loss` + `model.losses`）、計算並套用梯度、更新並記錄度量，最後在每個 epoch 結束時重置指標狀態。

**🎯 重點摘要:**

---

[來源: ch12] )` 計算 forward 與總損失（`main_loss` + `model.losses`）、計算並套用梯度、更新並記錄度量，最後在每個 epoch 結束時重置指標狀態。

**🎯 重點摘要:**

-   **核心功能**：自訂訓練迴圈可讓你整合非標準訓練邏輯（例如對抗訓練、教師學生、分段優化）。
-   **潛在問題**：必須手動照顧所有指標和狀態重置，容易忘記 `metric.reset_state()`。
-   **最佳使用情境**：需要實作多優化器、梯度累積、或精細的批次內追蹤。

---

---

[來源: ch12] ## TensorFlow 函數 (TF Functions) {#tf-functions}

💡 **實際應用情境：** `@tf.function` 可將 Python 函數轉換為高效可重用的計算圖 (Computation Graph)，對於追求效能或部署至 TensorFlow Serving 等環境時非常重要。

```python
@tf.function
def cube(x):
    return x ** 3

---

[來源: ch12] # 第一次呼叫會觸發 tracing (追蹤)，之後會重用相同計算圖
tf_cube = cube
print(tf_cube(tf.constant(2.0)))
```

**✅ 程式碼逐行解析：**

1. `第 1 行`: 使用 `@tf.function` 將 `cube` 函數轉換為 TF 函數，這會在第一次執行時建立計算圖。
2. `第 3 行`: 定義函數的實際運算內容。
3. `第 7 行`: 呼叫 `tf_cube`（即 `cube`），觸發 tracing 並執行圖。

**🎯 重點摘要:**

---

[來源: ch12] 次執行時建立計算圖。
2. `第 3 行`: 定義函數的實際運算內容。
3. `第 7 行`: 呼叫 `tf_cube`（即 `cube`），觸發 tracing 並執行圖。

**🎯 重點摘要:**

-   **核心功能**：將 Python 代碼轉換為高效計算圖，改善執行效能。
-   **潛在問題**：若在 TF 函數內動態建立變數或使用 Python 控制流，可能導致頻繁 tracing 或錯誤。
-   **最佳使用情境**：模型推論、批次處理、需要跨平台執行的情境。

---

[來源: ch12] ### Tracing 行為觀察

```python
@tf.function
def traced_add(x, y):
    print("Tracing...")
    return x + y

---

[來源: ch12] # 第一次執行會看到 Tracing 訊息
traced_add(tf.constant(1), tf.constant(2))

---

[來源: ch12] # 第二次執行不會再 tracing（使用同樣的 input signature）
traced_add(tf.constant(3), tf.constant(4))
```

---

[來源: ch12] ### `run_eagerly=True`（禁用計算圖）

若你需要逐步偵錯或必須執行動態 Python 控制流程，可在 `model.compile()` 中啟用：

```python
model.compile(optimizer="nadam", loss="mse", run_eagerly=True)
```

> ⚠️ 注意：`run_eagerly=True` 會降低效能，但可在開發/偵錯階段提升可讀性。

---

---

[來源: ch12] ## 與 tf.keras 一起使用 TF 函數 (Using TF Functions with tf.keras) {#tf-keras}

💡 **實際應用情境：** TensorFlow/Keras 預設會將自訂 layer、loss、metric 等轉成 TF 函數，以優化訓練與推論效能。

**🎯 重點摘要:**

---

[來源: ch12] f-keras}

💡 **實際應用情境：** TensorFlow/Keras 預設會將自訂 layer、loss、metric 等轉成 TF 函數，以優化訓練與推論效能。

**🎯 重點摘要:**

-   Keras 會自動將大多數 `call()` 轉成 TF 函數，因此請遵守 TF 函數規則：避免在 `call()` 裡建立新變數、避免使用 Python side-effect（例如列表/字典的 append、print、random 等），盡量只使用 TensorFlow 運算。
-   若你需要混用任意 Python 代碼（例如第三方套件、條件邏輯、除錯輸出），可以使用 `tf.py_function()`（但會降低效能且降低模型可移植性），或在訓練/推論時啟用 `run_eagerly=True` 以強制以 eager 模式執行。

---

[來源: ch12] 件、條件邏輯、除錯輸出），可以使用 `tf.py_function()`（但會降低效能且降低模型可移植性），或在訓練/推論時啟用 `run_eagerly=True` 以強制以 eager 模式執行。

```python
    def numpy_op(x):
        import numpy as np
        return np.log(x)

    @tf.function
    def wrapped(x):
        # tf.py_function 會將 numpy_op 包裝為 TensorFlow op
        y = tf.py_function(func=numpy_op, inp=[x], Tout=tf.float32)
        return y + 1.0

    print(wrapped(tf.constant([1.0, 2.0, 3.0])))
    ```

---

[來源: ch12] ], Tout=tf.float32)
        return y + 1.0

    print(wrapped(tf.constant([1.0, 2.0, 3.0])))
    ```

-   在 Keras 3 之後，`dynamic` 參數已不再存在；如果你仍需「動態模型」行為，可透過 `run_eagerly=True` 或在模型/層級別使用 `tf.keras.layers.Layer`/`tf.keras.Model` 時自行控制 Python 執行流程（但這會關閉圖優化並影響效能）。
-   若在 `@tf.function` 內要 log 訊息，請使用 `tf.print()` 以確保在 graph/已編譯模式下也能正確輸出。

---

---

[來源: ch12] ## 習題解答 (Exercises) {#exercises}

---

[來源: ch12] ### 12. 實作 Layer Normalization 層

---

[來源: ch12] ```python
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, eps=0.001, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, batch_input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=batch_input_shape[-1:],
            initializer="ones")
        self.beta = self.add_weight(
            name="beta", shape=batch_input_shape[-1:],
            initializer="zeros")

    def call(self, X):
        mean, variance = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.alpha * (X - mean) / (tf.sqrt(variance + self.eps)) + self.beta

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "eps": self.eps}
```

---

[來源: ch12] ### 13. 使用自訂訓練迴圈訓練 Fashion MNIST

---

[來源: ch12] ```python
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test.astype(np.float32) / 255.

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = tf.keras.losses.sparse_categorical_crossentropy
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        status = OrderedDict()
        mean_loss(loss)
        status["loss"] = mean_loss.result().numpy()
        for metric in metrics:
            metric(y_batch, y_pred)
            status[metric.name] = metric.result().numpy()
        
    y_pred = model(X_valid)
    status["val_loss"] = np.mean(loss_fn(y_valid, y_pred))
    status["val_accuracy"] = np.mean(tf.keras.metrics.sparse_categorical_accuracy(
        tf.constant(y_valid, dtype=np.float32), y_pred))
    
    for metric in [mean_loss] + metrics:
        metric.reset_state()
```

---

[來源: ch12] np.float32), y_pred))
    
    for metric in [mean_loss] + metrics:
        metric.reset_state()
```

---

---

[來源: ch12] ## 總結與最佳實踐

TensorFlow 的自訂功能提供了極大的彈性，讓你可以針對特定問題打造專屬架構與訓練流程。記住以下原則：

1. **從簡單開始**：先使用內建元件，必要時再抽換自訂邏輯。
2. **測試 thoroughly**：自訂元件應有對應單元測試與數值驗證。
3. **效能考量**：`@tf.function` 可提高效能，但請優先在穩定後啟用。
4. **可重用性**：讓自訂層/模型可序列化 (`get_config`) 會大幅提升跨專案使用性。
5. **版本管理**：訓練環境、TensorFlow 版本與依賴需要同步，避免儲存後載入失敗。

---

---

[來源: ch12] ## 常見問答 (FAQ)

**Q: 自訂層與內建層的差異？**
A: 內建層是經過優化、測試與廣泛支援的元件，適合大多數情境。自訂層則提供你對計算流程與變數管理的完全控制，適合需要特殊行為或自訂梯度的案例。

**Q: 何時使用自訂訓練迴圈 (Custom Training Loop)？**
A: 當你需要異常流程（如多優化器、策略梯度、對抗訓練、分段 loss）或要在每個 step 中插入複雜邏輯（如動態梯度裁剪、梯度累積）時，自訂訓練迴圈是最靈活的方式。

---

[來源: ch12] aining Loop)？**
A: 當你需要異常流程（如多優化器、策略梯度、對抗訓練、分段 loss）或要在每個 step 中插入複雜邏輯（如動態梯度裁剪、梯度累積）時，自訂訓練迴圈是最靈活的方式。

**Q: 為什麼我的 `tf.function` 一直 retrace（重複 tracing）？**
A: 常見原因包含輸入 shape 或 dtype 變化、在函數內建立新變量、或使用 Python list/dict 每次傳入不同結構。解法：使用 `input_signature` 固定形狀，避免在函數內建立變量，或改用 `tf.constant`/`tf.Tensor`。

---

[來源: ch12] 或使用 Python list/dict 每次傳入不同結構。解法：使用 `input_signature` 固定形狀，避免在函數內建立變量，或改用 `tf.constant`/`tf.Tensor`。

**Q: 儲存自訂模型時出現 `Unknown loss function`？**
A: 這表示載入時未提供對應的 `custom_objects` 或自訂類別未能正確序列化。確保你的自訂 Loss/Metric/Layer 有 `get_config()`，並在 `load_model(..., custom_objects={...})` 中註冊。

---

[來源: ch12] 訂類別未能正確序列化。確保你的自訂 Loss/Metric/Layer 有 `get_config()`，並在 `load_model(..., custom_objects={...})` 中註冊。

**Q: 怎麼在 tf.keras 中 debug TF 函數？**
A: 可使用 `run_eagerly=True`（會降低效能），或用 `tf.print()` 取代 `print()`，並搭配 `tf.autograph.to_code()` 查看轉換後的圖。

---

---

[來源: ch12] ## 推薦標籤 (Suggested Hashtags)

#TensorFlow #自訂模型 #機器學習 #Python #深度學習 #訓練迴圈 #神經網路 #程式設計

---

[來源: ch12 | 類型: cheatsheet] # Ch12 速查表：Custom Models & Training with TensorFlow

> **核心主旨**：打開 Keras 的黑盒子 —— `GradientTape` 自訂訓練迴圈、自訂 Layer/Loss/Metric 讓你掌控每一步。

---

---

[來源: ch12 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| `tf.Tensor` | 不可變的多維陣列（類 numpy），支援 GPU / 自動微分 | TF 的基本計算單元 |
| `tf.Variable` | 可變的張量，用於儲存模型權重 | 自訂層的參數 |
| `tf.GradientTape` | 記錄前向計算，用於計算梯度（自動微分） | 自訂訓練迴圈 |
| Custom Loss Function | 接受 `(y_true, y_pred)` 並返回純量張量的函數 | Huber Loss、Focal Loss 等 |
| Custom Metric | 繼承 `tf.keras.metrics.Metric`，支援跨批次累積 | Precision@K 等複雜指標 |
| Custom Layer | 繼承 `tf.keras.layers.Layer`，定義 `build()` + `call()` | 殘差塊、自注意力層 |
| Custom Model | 繼承 `tf.keras.Model`，自訂 `call()` 的資料流 | 非標準架構 |
| `@tf.function` | 將 Python 函數編譯為 TF Graph，大幅加速 | 高頻呼叫的函數 |


---

[來源: ch12 | 類型: cheatsheet] keras.Model`，自訂 `call()` 的資料流 | 非標準架構 |
| `@tf.function` | 將 Python 函數編譯為 TF Graph，大幅加速 | 高頻呼叫的函數 |


---

---

[來源: ch12 | 類型: cheatsheet] | TF / Keras API | 重點說明 | 用途 |
|----------------|---------|------|
| `tf.constant(...)` | 建立不可變張量 | 建立資料 |
| `tf.Variable(...)` | 建立可變張量（權重） | 自訂層的權重 |
| `tf.cast(x, tf.float32)` | 類型轉換 | 型別不符時必用 |
| `tf.GradientTape()` | 前向計算錄影帶 | 計算梯度 |
| `tape.gradient(loss, variables)` | 計算梯度 | 取得 dLoss/dW |
| `optimizer.apply_gradients(zip(grads, vars))` | 更新權重 | 手動梯度下降 |
| `tf.keras.losses.Huber()` | Huber loss（內建） | 對 outlier 魯棒的迴歸 |
| `tf.keras.layers.Layer` | 自訂層的基類 | 定義 `build()` + `call()` |
| `tf.keras.Model` | 自訂模型的基類 | 定義複雜資料流 |
| `@tf.function` | Graph 模式加速裝飾器 | 加速純 TF 函數 |


---

[來源: ch12 | 類型: cheatsheet] `call()` |
| `tf.keras.Model` | 自訂模型的基類 | 定義複雜資料流 |
| `@tf.function` | Graph 模式加速裝飾器 | 加速純 TF 函數 |


---

---

[來源: ch12 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import tensorflow as tf

---

[來源: ch12 | 類型: cheatsheet] # 基本張量操作
t = tf.constant([[1., 2.], [3., 4.]])
v = tf.Variable([[1., 2.], [3., 4.]])
v.assign(v + 1)      # 原地更新
v.assign_add([[1, 1], [1, 1]])

---

[來源: ch12 | 類型: cheatsheet] # 自訂 Loss Function
def huber_fn(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < delta
    squared_loss = tf.square(error) / 2
    linear_loss = delta * tf.abs(error) - delta**2 / 2
    return tf.where(is_small_error, squared_loss, linear_loss)

model.compile(loss=huber_fn, optimizer="adam")

---

[來源: ch12 | 類型: cheatsheet] # 自訂 Metric（跨批次累積）
class HuberMetric(tf.keras.metrics.Metric):
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.huber_fn = huber_fn
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

---

[來源: ch12 | 類型: cheatsheet] total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

def update_state(self, y_true, y_pred, sample_weight=None):
        sample_metrics = self.huber_fn(y_true, y_pred, self.delta)
        self.total.assign_add(tf.reduce_sum(sample_metrics))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

def result(self):
        return self.total / self.count

---

[來源: ch12 | 類型: cheatsheet] # 自訂 Layer
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

---

[來源: ch12 | 類型: cheatsheet] (**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

def build(self, input_shape):  # 在知道輸入形狀後才建立權重
        self.W = self.add_weight(name="W",
                                  shape=(input_shape[-1], self.units),
                                  initializer="glorot_normal")
        self.b = self.add_weight(name="b", shape=(self.units,),
                                  initializer="zeros")
        super().build(input_shape)

---

[來源: ch12 | 類型: cheatsheet] f.units,),
                                  initializer="zeros")
        super().build(input_shape)

def call(self, X):
        return self.activation(X @ self.W + self.b)

---

[來源: ch12 | 類型: cheatsheet] # GradientTape 自訂訓練迴圈
@tf.function  # 編譯為 Graph，加速執行
def train_step(X_batch, y_batch, model, optimizer, loss_fn, metric):
    with tf.GradientTape() as tape:
        y_pred = model(X_batch, training=True)
        main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
        loss = tf.add_n([main_loss] + model.losses)  # 含正則化損失
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metric.update_state(y_batch, y_pred)
    return loss

---

[來源: ch12 | 類型: cheatsheet] # 完整自訂訓練迴圈
n_epochs = 5
batch_size = 32
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metric = tf.keras.metrics.MeanAbsoluteError()

---

[來源: ch12 | 類型: cheatsheet] ptimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metric = tf.keras.metrics.MeanAbsoluteError()

for epoch in range(n_epochs):
    for step in range(len(X_train) // batch_size):
        X_batch = X_train[step*batch_size:(step+1)*batch_size]
        y_batch = y_train[step*batch_size:(step+1)*batch_size]
        loss = train_step(X_batch, y_batch, model, optimizer, loss_fn, metric)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, MAE: {metric.result():.4f}")
    metric.reset_state()
```

---

---

[來源: ch12 | 類型: cheatsheet] ## 4. 常見陷阱

- **`tf.Variable` 需要在 `tape` 的 watch 範圍內**：`tape` 預設只錄 `tf.Variable`，非 variable 張量需手動 `tape.watch(tensor)`。
- **`tf.function` 的 tracing**：第一次呼叫會追蹤（建立 Graph），後續呼叫用快取。傳入不同形狀的張量會觸發重新追蹤，影響速度。
- **型別轉換**：TF 不做隱式型別轉換，`float32` 和 `float64` 混合會報錯，用 `tf.cast()` 手動轉換。
- **`model.losses`**：Layer 用 `self.add_loss()` 新增的正則化損失，在自訂迴圈中要手動加進總損失。

---

---

[來源: ch12 | 類型: cheatsheet] ## 5. 決策指南

```
何時用自訂訓練迴圈？
├── 需要不同層不同 lr      → 是（GradientTape 分開計算）
├── GAN、MAML 等複雜訓練   → 是
├── 標準訓練流程           → 否，用 model.fit()（更快更穩）

自訂 Layer 的結構：
├── __init__: 儲存超參數（units, activation...）
├── build: 建立權重（self.add_weight()），延遲到知道 input_shape
└── call: 前向傳播邏輯，接受 training=False 參數
```

---

[來源: ch12 | 類型: handout] # 課程講義：自定義模型與 TensorFlow 訓練 (Chapter 12)

大家早安。在之前的課程中，我們主要使用 Keras 的高階 API 來建構模型。然而，當你面對尖端的學術研究或需要極致的效能優化時，標準的層與損失函數往往不夠用。今天，我們要「打開黑盒子」，學習如何利用 TensorFlow 的低階運算來建構完全自定義的深度學習架構。

---

---

[來源: ch12 | 類型: handout] ## 1. TensorFlow 基礎：張量與運算 (Tensors & Ops)

TensorFlow 的核心數據結構是**張量 (Tensors)**。它們與 NumPy 的 `ndarray` 非常相似，但具備兩個關鍵優勢：支援 GPU 加速且對自動微分友好。

- **不可變性 (Immutability)**：與 NumPy 陣列不同，普通張量是不可變的。若需儲存模型權重等可變狀態，必須使用 `tf.Variable`。
- **類型嚴格性**：TensorFlow 不會自動執行隱式類型轉換（例如 `float32` 與 `int32` 相加會報錯）。請務必使用 `tf.cast()` 手動轉換。

---

[來源: ch12 | 類型: handout] Variable`。
- **類型嚴格性**：TensorFlow 不會自動執行隱式類型轉換（例如 `float32` 與 `int32` 相加會報錯）。請務必使用 `tf.cast()` 手動轉換。

* **⚡ 補充練習 1：**
    1. **理論題**：為什麼 TensorFlow 將張量設計為不可變的？這對計算圖（Computation Graph）的優化有什麼幫助？
    2. **實作題**：建立一個隨機的 $3 \times 3$ 張量 $A$，計算 $A^T A$，並將結果轉換為 `tf.float64` 類型。

---

---

[來源: ch12 | 類型: handout] ## 2. 自定義損失函數與指標 (Custom Loss & Metrics)

當標準的 MSE 或 Cross-Entropy 無法捕捉問題的特性時（例如需要對離群值具備魯棒性的 Huber Loss），我們需要自定義函數。

- **自定義損失**：通常寫成一個接收 `(y_true, y_pred)` 並返回張量的函數即可。
- **串流指標 (Streaming Metrics)**：與損失函數不同，指標（如 Precision）需要在整個訓練週期中累積狀態。此時應繼承 `tf.keras.metrics.Metric` 並實作 `update_state()` 與 `result()`。

* **⚡ 補充練習 2：**

---

[來源: ch12 | 類型: handout] n）需要在整個訓練週期中累積狀態。此時應繼承 `tf.keras.metrics.Metric` 並實作 `update_state()` 與 `result()`。

* **⚡ 補充練習 2：**

1. **理論題**：在實作 Huber Loss 時，為什麼建議計算 $\sqrt{\text{variance} + \epsilon}$ 而非僅僅是 $\sqrt{\text{variance}}$？
    2. **實作題**：請實作 `log_cosh_loss(y_true, y_pred)`，公式為 $L = \sum \log(\cosh(y_{pred} - y_{true}))$。

---

---

[來源: ch12 | 類型: handout] ## 3. 自定義層與模型 (Custom Layers & Models)

這是建構如殘差連接（Residual Connections）或自注意力機制（Self-Attention）等複雜架構的基礎。

- **Layer Subclassing**：繼承 `tf.keras.layers.Layer`。
  - `__init__`：儲存超參數。
  - `build()`：定義權重（這是在得知輸入形狀後延遲初始化的最佳時機）。
  - `call()`：定義前向傳播邏輯。
- **Model Subclassing**：繼承 `tf.keras.Model`。通常用於定義模型整體的拓撲結構，這讓你可以自由控制 `call()` 內部的複雜數據流。

* **⚡ 補充練習 3：**

---

[來源: ch12 | 類型: handout] odel Subclassing**：繼承 `tf.keras.Model`。通常用於定義模型整體的拓撲結構，這讓你可以自由控制 `call()` 內部的複雜數據流。

* **⚡ 補充練習 3：**

1. **理論題**：請說明 `tf.keras.layers.Layer` 與 `tf.keras.Model` 的主要差異與各自的使用場景。
    2. **實作題**：建立一個自定義層 `MySoftmax`，其實作標準的 Softmax 運算：$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$。

---

---

[來源: ch12 | 類型: handout] ## 4. 自動微分與訓練迴圈 (Autodiff & Custom Loops)

`tf.GradientTape` 是 TensorFlow 實現自動微分的神經中樞。

- **Gradient Tape**：在 `with tf.GradientTape() as tape:` 區塊內執行的運算會被記錄在「磁帶」上，隨後可用 `tape.gradient(target, sources)` 計算梯度。
- **自定義訓練迴圈**：這讓你可以手動控制每一個訓練步驟。雖然比 `model.fit()` 複雜且易出錯，但它允許你在訓練過程中加入非標準邏輯（例如為不同層設定不同的學習率，或手動裁剪梯度）。

* **⚡ 補充練習 4：**

---

[來源: ch12 | 類型: handout] 你可以手動控制每一個訓練步驟。雖然比 `model.fit()` 複雜且易出錯，但它允許你在訓練過程中加入非標準邏輯（例如為不同層設定不同的學習率，或手動裁剪梯度）。

* **⚡ 補充練習 4：**

1. **理論題**：為什麼在計算高階導數（如 Hessian 矩陣）時，我們需要巢狀（Nested）的 `GradientTape`？
    2. **實作題**：撰寫一段虛擬碼（Pseudocode），展示如何結合 `GradientTape` 與 `optimizer.apply_gradients()` 來更新一個簡單模型的權重。

---

---

[來源: ch12 | 類型: handout] ## 5. TensorFlow 函數與圖形 (TF Functions & Graphs)

透過 `@tf.function` 裝飾器，TensorFlow 能將 Python 函數轉換為高效的、可移植的**計算圖**。

- **AutoGraph**：自動將 Python 控制流（如 `if`, `for`, `while`）轉換為對應的 TensorFlow 運算節點。
- **效能規則**：避免在 `@tf.function` 內定義變數，且儘量使用張量運算而非純 Python 運算，以避免頻繁的「追蹤（Tracing）」導致效能下降。

* **⚡ 補充練習 5：**

---

[來源: ch12 | 類型: handout] **效能規則**：避免在 `@tf.function` 內定義變數，且儘量使用張量運算而非純 Python 運算，以避免頻繁的「追蹤（Tracing）」導致效能下降。

* **⚡ 補充練習 5：**

1. **理論題**：什麼是「多形性（Polymorphism）」在 TensorFlow 函數中的意義？傳入不同形狀的張量會觸發什麼行為？
    2. **實作題**：撰寫一個使用 `@tf.function` 的函數，並使用 `tf.autograph.to_code()` 觀察其生成的圖形代碼。

---

---

[來源: ch12 | 類型: handout] ## 結論

掌握了自定義組件與底層 API 後，你將不再受限於框架預設的功能。這不僅是開發新演算法的必經之路，也是深入理解深度學習運作原理的最佳途徑。

---

---

[來源: ch12 | 類型: handout] ## 課後作業 (Assignment)

**題目：實作 Layer Normalization 層**
請參考課本練習 12，自行實作一個 `LayerNormalization` 層。該層需包含：

1. 可訓練的權重 $\alpha$ (Scale) 與 $\beta$ (Offset)。
2. 在 `call()` 中計算每個樣本特徵的平均值 $\mu$ 與標準差 $\sigma$。
3. 最終輸出公式為：$Y = \alpha \otimes \frac{X - \mu}{\sigma + \epsilon} + \beta$。
**驗證**：請將你的實作結果與 `tf.keras.layers.LayerNormalization` 進行對比，確保在相同輸入下的誤差極小。

---

[來源: ch12 | 類型: tutorial] [標題: TensorFlow 自訂模型與訓練迴圈完整指南：GradientTape、自訂層與損失函數 | 描述: 深入 TensorFlow 的自訂開發：自訂損失函數、自訂層（含可訓練參數）、自訂訓練迴圈（GradientTape）、自訂評估指標，以及如何儲存和載入含自訂物件的模型。 | 關鍵字: Python, TensorFlow, Keras, GradientTape, 自訂層, 自訂損失, 自訂訓練迴圈, 自動微分, 深度學習]
# TensorFlow 自訂模型與訓練迴圈：GradientTape 完全指南

Keras 的高階 API 讓建模變得容易，但當你需要**非標準損失函數**、**特殊層結構**或**精細控制訓練過程**時，就需要深入 TensorFlow 的底層。本教學帶你掌握 TensorFlow 的自訂開發能力，讓你突破 Keras 預設 API 的限制。

---

[來源: ch12 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **自訂損失函數**：繼承 `keras.losses.Loss` 或直接傳入函數，`sample_weight` 自動支援
- **自訂層**：繼承 `keras.layers.Layer`，在 `build()` 中建立可學習參數，在 `call()` 中定義前向傳播
- **GradientTape** 是 TF 的自動微分引擎，讓你完全控制梯度計算和參數更新
- **`@tf.function`** 將 Python 函數編譯成 TF 計算圖，加速推論（通常 2-5 倍）
- 自訂物件的模型儲存：必須提供 `get_config()` 方法才能正確反序列化

---

---

[來源: ch12 | 類型: tutorial] ## TensorFlow 張量基礎

💡 **實際應用情境：** 了解 TF 張量操作是進行自訂開發的前提，特別是在撰寫自訂損失函數時需要用 TF 操作（而非 NumPy）確保支援自動微分。

---

[來源: ch12 | 類型: tutorial] ### 範例 1: 張量建立與操作

```python
import tensorflow as tf
import numpy as np

---

[來源: ch12 | 類型: tutorial] # 建立張量
t_const  = tf.constant([[1., 2., 3.],
                         [4., 5., 6.]])   # 不可變張量

t_var    = tf.Variable([[1., 2., 3.],
                         [4., 5., 6.]])   # 可變張量（可學習的參數）

print(f"形狀: {t_const.shape}, 資料類型: {t_const.dtype}")

---

[來源: ch12 | 類型: tutorial] # 基本操作（類似 NumPy，但在 GPU 上執行）
t_sq = tf.square(t_const)              # 元素平方
t_matmul = t_const @ tf.transpose(t_const)  # 矩陣乘法
t_concat = tf.concat([t_const, t_const], axis=0)  # 沿行方向串接

---

[來源: ch12 | 類型: tutorial] # 與 NumPy 互轉
np_array = t_const.numpy()             # Tensor → NumPy
tf_from_np = tf.constant(np_array)     # NumPy → Tensor

---

[來源: ch12 | 類型: tutorial] # 稀疏張量（NLP 中常用）
sparse = tf.SparseTensor(
    indices=[[0, 0], [1, 2]],
    values=[1., 2.],
    dense_shape=[3, 4]
)
print(tf.sparse.to_dense(sparse))
```

**✅ 程式碼逐行解析：**

1. `tf.constant`: 不可變，一旦建立就不能更改值（適合固定資料）
2. `tf.Variable`: 可變，訓練時 GradientTape 會追蹤其梯度（適合模型參數）
3. `t_const.numpy()`: 將張量轉回 NumPy 陣列（需要在 Eager Mode 下）

**🎯 重點摘要:**

- TF 的操作語法與 NumPy 非常相似，但在 GPU/TPU 上執行
- 自訂函數中必須使用 TF 操作（如 `tf.reduce_mean`）而非 NumPy，確保自動微分有效

---

---

[來源: ch12 | 類型: tutorial] ## 自訂損失函數

💡 **實際應用情境：** 預測房價時，低估（預測值低於實際）比高估代價更高（銀行損失更大）。自訂非對稱損失函數讓你將業務邏輯編碼到訓練過程中。

---

[來源: ch12 | 類型: tutorial] ### 範例 2: 自訂 Huber 損失函數

```python

---

[來源: ch12 | 類型: tutorial] # 方法 1：簡單函數（適合無超參數的損失）
def huber_loss(y_true, y_pred, threshold=1.0):
    """Huber 損失：小誤差用 MSE，大誤差用 MAE（對離群值更健壯）"""
    error = y_true - y_pred
    is_small_error = tf.abs(error) < threshold
    squared_loss = tf.square(error) / 2
    linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
    return tf.where(is_small_error, squared_loss, linear_loss)

---

[來源: ch12 | 類型: tutorial] # 方法 2：繼承 Loss 類別（支援超參數、序列化）
class HuberLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

---

[來源: ch12 | 類型: tutorial] bs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

def get_config(self):
        """支援序列化（儲存/載入模型時必需）"""
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

---

[來源: ch12 | 類型: tutorial] # 使用自訂損失
model_huber = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(1)
])
model_huber.compile(
    optimizer="adam",
    loss=HuberLoss(threshold=1.5)  # 自訂損失函數
)
```

**✅ 程式碼逐行解析：**

1. `tf.where(condition, x, y)`: 元素級條件選擇（類似 `np.where`）
2. 繼承 `keras.losses.Loss` 的優點：自動支援 `sample_weight`、`reduction` 策略
3. `get_config()`: 返回超參數字典，讓模型可以正確儲存和載入

---

---

[來源: ch12 | 類型: tutorial] ### 範例 3: 含可訓練參數的自訂層

```python
class ResidualBlock(tf.keras.layers.Layer):
    """殘差塊（ResNet 的核心組件）"""

def __init__(self, n_layers: int, n_neurons: int, **kwargs):
        super().__init__(**kwargs)
        # 在 __init__ 建立子層（但不建立權重，因為不知道輸入維度）
        self.hidden = [
            tf.keras.layers.Dense(n_neurons, activation="relu",
                                   kernel_initializer="he_normal")
            for _ in range(n_layers)
        ]

---

[來源: ch12 | 類型: tutorial] kernel_initializer="he_normal")
            for _ in range(n_layers)
        ]

def call(self, inputs):
        """定義前向傳播（建議在此用 @tf.function 加速）"""
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z  # 殘差連接（shortcut）

def get_config(self):
        config = super().get_config()
        config.update({"n_layers": len(self.hidden),
                        "n_neurons": self.hidden[0].units})
        return config

class NormalizationLayer(tf.keras.layers.Layer):
    """自訂標準化層（示範 build 的用法）"""

---

[來源: ch12 | 類型: tutorial] return config

class NormalizationLayer(tf.keras.layers.Layer):
    """自訂標準化層（示範 build 的用法）"""

def build(self, batch_input_shape):
        """在第一次呼叫時根據輸入形狀建立權重"""
        n_inputs = batch_input_shape[-1]
        self.scale = self.add_weight(
            name="scale",
            shape=[n_inputs],
            initializer="ones"     # 初始化為全 1
        )
        self.offset = self.add_weight(
            name="offset",
            shape=[n_inputs],
            initializer="zeros"    # 初始化為全 0
        )
        super().build(batch_input_shape)

---

[來源: ch12 | 類型: tutorial] ts],
            initializer="zeros"    # 初始化為全 0
        )
        super().build(batch_input_shape)

def call(self, X):
        mean, variance = tf.nn.moments(X, axes=[0])
        return (X - mean) / (tf.sqrt(variance) + 1e-8) * self.scale + self.offset

---

[來源: ch12 | 類型: tutorial] # 使用自訂層
model_custom = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    NormalizationLayer(),
    ResidualBlock(n_layers=2, n_neurons=64),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

**✅ 程式碼逐行解析：**

1. `build(batch_input_shape)`: 在第一次 `call()` 時自動被呼叫，此時才知道輸入維度
2. `self.add_weight(...)`: 建立可學習的張量（會被 GradientTape 追蹤）
3. `inputs + Z`: 殘差連接——讓梯度繞過中間層直接傳回（解決深網路的梯度消失）

**🎯 重點摘要:**

- `build()` vs `__init__()` 的分工：`__init__` 建立子層（不知道維度）；`build` 建立需要輸入維度的參數
- 自訂層繼承後可以完全整合到 `model.fit()` / `model.save()` 流程中

---

---

[來源: ch12 | 類型: tutorial] ### 範例 4: 自訂 Streaming Metric

```python
class RMSEMetric(tf.keras.metrics.Metric):
    """自訂 RMSE 評估指標（支援 streaming 累積計算）"""

def __init__(self, name="RMSE", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_sq_error = self.add_weight(name="sum_sq_error", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

---

[來源: ch12 | 類型: tutorial] zer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

def update_state(self, y_true, y_pred, sample_weight=None):
        """每個 batch 後更新累積統計量"""
        sq_error = tf.reduce_sum(tf.square(y_pred - y_true))
        self.sum_sq_error.assign_add(sq_error)
        self.total_samples.assign_add(tf.cast(tf.size(y_true), tf.float32))

def result(self):
        """計算最終指標值"""
        return tf.sqrt(self.sum_sq_error / self.total_samples)

---

[來源: ch12 | 類型: tutorial] f result(self):
        """計算最終指標值"""
        return tf.sqrt(self.sum_sq_error / self.total_samples)

def reset_state(self):
        """每個 epoch 後重置"""
        self.sum_sq_error.assign(0.)
        self.total_samples.assign(0.)

---

[來源: ch12 | 類型: tutorial] # 使用自訂指標
model_metric = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])
model_metric.compile(optimizer="adam", loss="mse", metrics=[RMSEMetric()])
```

---

---

[來源: ch12 | 類型: tutorial] ### 範例 5: 完整的自訂訓練迴圈

```python
import tensorflow as tf
import numpy as np

---

[來源: ch12 | 類型: tutorial] # 準備資料
X_train = np.random.randn(1000, 8).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(1000).batch(32)

---

[來源: ch12 | 類型: tutorial] # 模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch12 | 類型: tutorial] # 優化器和損失
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function  # 編譯為計算圖（加速）
def train_step(X_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred = model(X_batch, training=True)  # 前向傳播
        loss   = loss_fn(y_batch, y_pred)        # 計算損失

---

[來源: ch12 | 類型: tutorial] pred = model(X_batch, training=True)  # 前向傳播
        loss   = loss_fn(y_batch, y_pred)        # 計算損失

# 計算梯度
    gradients = tape.gradient(loss, model.trainable_variables)
    # 應用梯度更新模型參數
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy_metric.update_state(y_batch, y_pred)
    return loss

---

[來源: ch12 | 類型: tutorial] # 訓練迴圈
for epoch in range(5):
    epoch_losses = []
    accuracy_metric.reset_state()

for X_batch, y_batch in dataset:
        batch_loss = train_step(X_batch, y_batch)
        epoch_losses.append(batch_loss.numpy())

epoch_acc = accuracy_metric.result().numpy()
    print(f"Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}, acc={epoch_acc:.4f}")
```

**✅ 程式碼逐行解析：**

---

[來源: ch12 | 類型: tutorial] print(f"Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}, acc={epoch_acc:.4f}")
```

**✅ 程式碼逐行解析：**

1. `with tf.GradientTape() as tape`: 進入 tape 上下文後，所有對 `trainable_variables` 的操作都被記錄
2. `tape.gradient(loss, model.trainable_variables)`: 對 `trainable_variables` 中每個變數計算損失的偏導數
3. `optimizer.apply_gradients(zip(gradients, variables))`: 根據梯度和優化器策略更新參數
4. `@tf.function`: 第一次呼叫時追蹤計算圖，後續呼叫直接執行圖（速度快 2-5 倍）

**🎯 重點摘要:**

- 自訂訓練迴圈給你**完全的控制**：自訂梯度（如梯度裁剪）、多個優化器、多個損失
- `model(X, training=True)` vs `model(X, training=False)`: BN 和 Dropout 在訓練/推論模式行為不同

---

---

[來源: ch12 | 類型: tutorial] ### 範例 6: GradientTape 高階用法

```python

---

[來源: ch12 | 類型: tutorial] # 計算二階梯度（梯度的梯度）
x = tf.Variable(3.0)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        y = x**3          # y = x³
    dy_dx = inner_tape.gradient(y, x)       # dy/dx = 3x² = 27
d2y_dx2 = outer_tape.gradient(dy_dx, x)    # d²y/dx² = 6x = 18

print(f"dy/dx   = {dy_dx.numpy():.1f}")    # 27
print(f"d²y/dx² = {d2y_dx2.numpy():.1f}") # 18

---

[來源: ch12 | 類型: tutorial] # 自訂梯度（如梯度裁剪）
@tf.custom_gradient
def clip_gradients(y):
    def backward(dy):
        return tf.clip_by_value(dy, -1.0, 1.0)  # 裁剪梯度
    return y, backward
```

---

---

[來源: ch12 | 類型: tutorial] # 方法 1：SavedModel 格式（推薦，保存計算圖）
model.save("my_model")  # 儲存為 SavedModel 目錄格式

---

[來源: ch12 | 類型: tutorial] # 方法 2：.keras 格式（新格式，需要自訂物件可序列化）

---

[來源: ch12 | 類型: tutorial] # 確保自訂損失/層有 get_config() 方法
model.save("my_model.keras")

---

[來源: ch12 | 類型: tutorial] # 載入含自訂物件的模型
loaded_model = tf.keras.models.load_model(
    "my_model.keras",
    custom_objects={
        "HuberLoss": HuberLoss,
        "ResidualBlock": ResidualBlock
    }
)

---

[來源: ch12 | 類型: tutorial] # 方法 3：只儲存權重（最輕量）
model.save_weights("my_weights.weights.h5")

---

[來源: ch12 | 類型: tutorial] # 需要重新建立模型結構才能載入
model.load_weights("my_weights.weights.h5")
```

**🎯 重點摘要:**

- `.keras` 格式是 Keras 3 推薦的格式，跨後端（TF/JAX/PyTorch）
- 自訂物件**必須**實作 `get_config()` 才能被 `load_model` 正確重建

---

---

[來源: ch12 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: 為什麼 GradientTape 只追蹤 `tf.Variable`？**

A: GradientTape 預設只追蹤 `tf.Variable`（可學習的參數）。若要計算對 `tf.Tensor` 的梯度，需要顯式呼叫 `tape.watch(tensor)`。

**Q2: `@tf.function` 有什麼限制？**

A: Python 的動態行為（如 `print`、Python 迴圈、列表操作）在 `@tf.function` 中可能不如預期——需要改用 TF 操作（`tf.print`、`tf.while_loop`）。初期不確定時，先不加 `@tf.function` 確保邏輯正確。

**Q3: 自訂訓練迴圈什麼時候比 `model.fit()` 更好？**

A: 需要以下功能時：(1) 多個損失函數分別優化不同部分（如 GAN）；(2) 動態調整訓練策略；(3) 自訂梯度累積；(4) 教學用途（理解訓練細節）。

**Q4: `build()` 和 `__init__()` 的差別？**

A: `__init__` 在實例化時呼叫，但此時不知道輸入維度（無法建立需要維度的權重）；`build()` 在第一次 `call()` 時自動觸發，此時輸入形狀已知，適合建立需要輸入維度的 `add_weight`。

---

[來源: ch12 | 類型: tutorial] __` 在實例化時呼叫，但此時不知道輸入維度（無法建立需要維度的權重）；`build()` 在第一次 `call()` 時自動觸發，此時輸入形狀已知，適合建立需要輸入維度的 `add_weight`。

---

---

[來源: ch12 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #TensorFlow #Keras #GradientTape #自訂層 #自訂訓練 #深度學習 #自動微分 #自訂損失 #程式設計 #教學 #DataScience #MachineLearning #AI

---

[來源: ch13] [標題: TensorFlow 資料載入與預處理教學：tf.data、TFRecord、Keras 前處理層 | 描述: 本文介紹 Chapter 13 的 TensorFlow 資料載入與預處理實務，從 tf.data 管線、CSV/TFRecord、Protobuf，到 Keras 前處理層與 TFDS 全面解析。 | 關鍵字: TensorFlow, tf.data, TFRecord, Keras, 資料預處理, Protobuf, tfds, Python, 教學]
# Chapter 13：TensorFlow 資料載入與預處理實戰

本章節範例來自 13_loading_and_preprocessing_data.ipynb，我們將用 TensorFlow 的 `tf.data` API 建立高效資料管線，講解從 CSV、TFRecord、Protobuf 到 Keras 前處理層的實作流程，並補充實務觀察與優化建議。

---

[來源: ch13] ## 本文目錄
- 📦 [tf.data API 基礎](#tf-data-basics)
- 🔀 [資料串接與轉換](#data-transform)
- 🎲 [資料洗牌與交錯讀取](#shuffle-interleave)
- 🧾 [CSV 讀取與前處理](#csv-preprocess)
- 🔧 [Keras 與 Dataset 串接](#keras-with-dataset)
- 📦 [TFRecord 格式與 Protobuf](#tfrecord-protobuf)
- 🧩 [Keras 前處理層](#keras-preprocessing)
- 🌐 [TensorFlow Datasets (TFDS)](#tfds)
- ✅ [章節重點與實務建議](#summary)
- ❓ [常見問答](#faq)

---

---

[來源: ch13] ## 關鍵重點 (Key Takeaways)
- `tf.data` 是建立可重用、高效資料管線的核心 API。
- `TFRecord` 與 `Protobuf` 適合序列化大量資料、跨平台部署與分散式訓練。
- Keras 前處理層（如 `Normalization`、`TextVectorization`）可將資料預處理內建到模型中。
- 針對大型資料集，讀取、解析、批次化與預取是性能優化的四大要素。

---

---

[來源: ch13] ##  tf.data API 基礎

`tf.data` 提供一個統一 API 來處理資料集，從單一張量到結構化嵌套資料都能無縫支援。

```python
import tensorflow as tf

X = tf.range(10)  # any data tensor
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset
```

✅ 程式碼逐行解析:
1. `tf.range(10)` 建立一個從 0 到 9 的整數張量。
2. `tf.data.Dataset.from_tensor_slices(X)` 將張量切片成一個資料集，資料集每次產生一個元素。

---

[來源: ch13] tf.range(10)` 建立一個從 0 到 9 的整數張量。
2. `tf.data.Dataset.from_tensor_slices(X)` 將張量切片成一個資料集，資料集每次產生一個元素。

🎯 重點摘要:
- `from_tensor_slices()` 適合記憶體中資料。
- 若資料量太大，應改用檔案資料來源或 `TFRecordDataset`。
- 這是 `tf.data` 管線的最簡單入口。

```python
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `for item in dataset:` 逐項遍歷 `tf.data.Dataset`。
2. `print(item)` 印出每個元素。

---

[來源: ch13] int(item)
```

✅ 程式碼逐行解析:
1. `for item in dataset:` 逐項遍歷 `tf.data.Dataset`。
2. `print(item)` 印出每個元素。

🎯 重點摘要:
- 直接迭代資料集可以檢查內容。
- 注意：在真實訓練時應避免逐元素 Python 迭代，應使用 `batch()` 批次化。

---

[來源: ch13] ##  資料串接與轉換

`tf.data` 的強大之處在於可以串接多個轉換步驟，形成可重用的管線。

```python
dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `Dataset.from_tensor_slices(tf.range(10))` 建立壹個 0-9 的資料集。
2. `.repeat(3)` 讓資料集重複三次。
3. `.batch(7)` 將資料切成每批 7 個元素。
4. `for item in dataset:` 印出每批資料。

---

[來源: ch13] 0-9 的資料集。
2. `.repeat(3)` 讓資料集重複三次。
3. `.batch(7)` 將資料切成每批 7 個元素。
4. `for item in dataset:` 印出每批資料。

🎯 重點摘要:
- `repeat()` 可用於訓練多個 epoch。
- `batch()` 是資料管線中最重要的性能優化步驟。
    - 批次化後的資料可以更有效率地送入 GPU/TPU。
    - 如果最後一個批次太小（例如只有 2 個樣本），有時會導致 Batch Normalization 層出現統計不穩定的問題。這時我們會用 `drop_remainder=True` 來丟棄最後一個不完整的批次。
- 若在交錯讀取前使用 `shuffle()`，可得到更好的隨機化效果。

---

---

[來源: ch13] on 層出現統計不穩定的問題。這時我們會用 `drop_remainder=True` 來丟棄最後一個不完整的批次。
- 若在交錯讀取前使用 `shuffle()`，可得到更好的隨機化效果。

---

接下來，我們來看看 tf.data 最強大的 **轉換（Transformations）** 功能：

```python
dataset = dataset.map(lambda x: x * 2)  # x is a batch
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `.map(lambda x: x * 2)` 對每個批次進行元素級轉換。
2. `x` 表示整個批次資料，而不是單一元素。

---

[來源: ch13] t:
    print(item)
```

✅ 程式碼逐行解析:
1. `.map(lambda x: x * 2)` 對每個批次進行元素級轉換。
2. `x` 表示整個批次資料，而不是單一元素。

🎯 重點摘要:
- `map()` 可在批次級別或元素級別執行轉換。
- 當資料已 `batch()` 後，`map()` 代表的是批次操作。
    - 效能優化: 在 `map()` 中使用 `num_parallel_calls=tf.data.AUTOTUNE` 可自動調整並行度，提升轉換效率。

---

[來源: ch13] ##  資料洗牌與交錯讀取

資料洗牌與交錯讀取可改善資料隨機性與輸入效率。

```python
dataset = tf.data.Dataset.range(10).repeat(2)
dataset = dataset.shuffle(buffer_size=4, seed=42).batch(7)
for item in dataset:
    print(item)
```

---

[來源: ch13] dataset = dataset.shuffle(buffer_size=4, seed=42).batch(7)
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `Dataset.range(10).repeat(2)` 建立 0-9 的資料並重複兩次。
2. `.shuffle(buffer_size=4, seed=42)` 以 4 個緩衝區大小洗牌。
    - 資料集會維持一個最多 4 個元素的緩衝區，從中隨機抽出元素並補入新的元素，這樣可以在串流讀取資料時提供一定程度的隨機性。
    - `buffer_size` 決定洗牌品質與記憶體使用，越大越隨機但越佔記憶體。
3. `.batch(7)` 批次大小 7。
4. `for item in dataset:` 印出每個批次。

---

[來源: ch13] - `buffer_size` 決定洗牌品質與記憶體使用，越大越隨機但越佔記憶體。
3. `.batch(7)` 批次大小 7。
4. `for item in dataset:` 印出每個批次。

🎯 重點摘要:
- `buffer_size` 影響洗牌品質與記憶體使用。
- 若需高品質洗牌，緩衝區應至少等於資料集大小。
- `seed` 保證重現性。

---

[來源: ch13] in dataset:` 印出每個批次。

🎯 重點摘要:
- `buffer_size` 影響洗牌品質與記憶體使用。
- 若需高品質洗牌，緩衝區應至少等於資料集大小。
- `seed` 保證重現性。

> 工業級的解決方案：多層次打亂 (Multi-stage Shuffling) 🌪️
在處理幾百 GB 甚至 TB 等級的資料時，我們不能只依賴 shuffle()。實務上的標準作法是：
第一步：打亂檔案清單。將資料分散存成多個檔案（Shards），先打亂這些檔案的讀取順序。
第二步：交錯讀取 (interleave)。同時從多個檔案中讀取資料。這樣可以在不增加太多記憶體使用的情況下，達到更好的隨機性。
第三步：使用適度的 buffer_size。在記憶體允許的範圍內（例如 10,000）進行最後的隨機採樣。

---

[來源: ch13] ##  CSV 讀取與前處理

本章講解如何從 California housing dataset 分割資料並寫入 CSV，再利用 `tf.data` 讀取。

---

[來源: ch13] ```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
```

---

[來源: ch13] , X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
```

✅ 程式碼逐行解析:
1. `fetch_california_housing()` 下載 California housing 資料。
2. `train_test_split(..., random_state=42)` 先切成訓練與測試。
3. 再次 `train_test_split` 取得驗證集。

🎯 重點摘要:
- 先分割資料再做標準化，可避免資料外洩。
- `reshape(-1, 1)` 將標籤調整為二維矩陣，符合 Keras 輸入格式。

---

[來源: ch13] train_test_split` 取得驗證集。

🎯 重點摘要:
- 先分割資料再做標準化，可避免資料外洩。
- `reshape(-1, 1)` 將標籤調整為二維矩陣，符合 Keras 輸入格式。

```python
def save_to_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = Path() / "datasets" / "housing"
    housing_dir.mkdir(parents=True, exist_ok=True)
    filename_format = "my_{}_{:02d}.csv"

    filepaths = []
    m = len(data)
    chunks = np.array_split(np.arange(m), n_parts)
    for file_idx, row_indices in enumerate(chunks):
        part_csv = housing_dir / filename_format.format(name_prefix, file_idx)
        filepaths.append(str(part_csv))
        with open(part_csv, "w") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([str(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths

HEADER = "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,MedianHouseValue"
train_filepaths = save_to_csv_files(
    np.c_[X_train, y_train], "train", header=HEADER)
valid_filepaths = save_to_csv_files(
    np.c_[X_valid, y_valid], "valid", header=HEADER)
test_filepaths = save_to_csv_files(
    np.c_[X_test, y_test], "test", header=HEADER)

```

---

[來源: ch13] r=HEADER)
test_filepaths = save_to_csv_files(
    np.c_[X_test, y_test], "test", header=HEADER)

```

✅ 程式碼逐行解析:
1. `Path() / "datasets" / "housing"` 定義輸出資料夾。
2. `mkdir(..., exist_ok=True)` 確保資料夾存在。
3. `np.array_split(np.arange(m), n_parts)` 將索引切成多份。
4. 用 `with open(...)` 寫入 CSV。
5. 如果有 `header`，先寫入欄位名稱。
6. `",".join([str(col) for col in data[row_idx]])` 將每行資料格式化成 CSV。

---

[來源: ch13] ` 寫入 CSV。
5. 如果有 `header`，先寫入欄位名稱。
6. `",".join([str(col) for col in data[row_idx]])` 將每行資料格式化成 CSV。

🎯 重點摘要:
- 多文件存儲可提高 I/O 並行讀取效能。
- 將資料拆成多個 CSV 有助於在雲端或分散式環境讀取。

```python
dataset = tf.data.Dataset.list_files(filepaths, seed=42)
dataset = dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers)
```

---

[來源: ch13] ave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers)
```

✅ 程式碼逐行解析:
1. `list_files(filepaths, seed=42)` 取得檔案路徑資料集並洗牌。
2. `interleave(...)` 並行讀取多個檔案。
    - `lambda filepath: tf.data.TextLineDataset(filepath).skip(1)` 定義每個檔案的讀取方式，使用 `TextLineDataset` 讀取文本行，並用 `.skip(1)` 跳過 CSV 標頭。
    - `cycle_length=n_readers` 定義同時讀取的檔案數量。

---

[來源: ch13] 讀取方式，使用 `TextLineDataset` 讀取文本行，並用 `.skip(1)` 跳過 CSV 標頭。
    - `cycle_length=n_readers` 定義同時讀取的檔案數量。

🎯 重點摘要:
- `interleave()` 可以同時讀多個檔案，提高磁碟吞吐量。
- 搭配 `skip(1)` 可避免重複標頭被當成資料。

---

---

[來源: ch13] ##  Keras 與 Dataset 串接

建立完整資料管線後，可直接餵給 Keras 模型訓練。

---

[來源: ch13] ```python
def csv_reader_dataset(filepaths, n_readers=5, n_read_threads=None,
                       n_parse_threads=5, shuffle_buffer_size=10_000, seed=42,
                       batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=seed)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size).prefetch(1)
```

---

[來源: ch13] dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size).prefetch(1)
```

✅ 程式碼逐行解析:
1. `list_files` 建立檔案列表。
2. `interleave(..., num_parallel_calls=n_read_threads)` 以並行方式讀取文件。
3. `map(preprocess, num_parallel_calls=n_parse_threads)` 進行並行解析與前處理。
4. `shuffle()` 增加資料隨機性。
5. `batch()` 批次化。
6. `prefetch(1)` 提前讀取下一批，減少 CPU/GPU 等待。

---

[來源: ch13] eads)` 進行並行解析與前處理。
4. `shuffle()` 增加資料隨機性。
5. `batch()` 批次化。
6. `prefetch(1)` 提前讀取下一批，減少 CPU/GPU 等待。

🎯 重點摘要:
- `prefetch()` 是訓練效率的關鍵。
- `.map()` 與 `.interleave()` 都可以使用 `num_parallel_calls` 加速。
- `shuffle_buffer_size` 要夠大才能得到更均勻的隨機性。

> 注意：定義 `preprocess()` 函數來解析 CSV 行並轉換成模型輸入格式是必要的，這裡實作 `preprocess()` 如下:

```python

---

[來源: ch13] # Determine the number of features for record_defaults
n_features = X_train.shape[1]
n_columns = n_features + 1 # Total columns: features + 1 label
record_defaults = [tf.constant(0.0, dtype=tf.float32)] * n_columns

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

def preprocess(line):
    """Parses a CSV line and normalizes features.

---

[來源: ch13] X_std = X_train.std(axis=0)

def preprocess(line):
    """Parses a CSV line and normalizes features.

Args:
        line: A string tensor representing a single CSV line.

---

[來源: ch13] V line and normalizes features.

Args:
        line: A string tensor representing a single CSV line.

Returns:
        A tuple of (normalized_features, label) where normalized_features
        are scaled using mean and standard deviation, and label is a scalar tensor.
    """
    # Use tf.io.decode_csv for robust parsing
    columns = tf.io.decode_csv(line, record_defaults=record_defaults)
    fea

---

[來源: ch13] csv for robust parsing
    columns = tf.io.decode_csv(line, record_defaults=record_defaults)
    features = tf.stack(columns[:-1])  # All but the last column are features
    label = columns[-1]                # The last column is the label
    return (features - X_mean) / X_std, label

```

接著，我們就可以直接把 `tf.data.Dataset` 物件傳給 Keras 的 `fit()` 方法：

```

---

[來源: ch13] (features - X_mean) / X_std, label

```

接著，我們就可以直接把 `tf.data.Dataset` 物件傳給 Keras 的 `fit()` 方法：

```

python
train_set = csv_reader_dataset(train_filepaths)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

---

[來源: ch13] paths)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train.shape[1:]),
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(1),
])
model.compile(loss="mse", optimizer="sgd")
model.fit(train_set, validation_data=valid_set, epochs=5)
```

---

[來源: ch13] l.compile(loss="mse", optimizer="sgd")
model.fit(train_set, validation_data=valid_set, epochs=5)
```

✅ 程式碼逐行解析:
1. `csv_reader_dataset(...)` 分別建立訓練、驗證、測試資料集。
2. `Sequential([...])` 定義簡單全連接模型。
3. `compile(loss="mse", optimizer="sgd")` 設定損失與優化器。
4. `fit(train_set, validation_data=valid_set, epochs=5)` 執行訓練。

---

[來源: ch13] ss="mse", optimizer="sgd")` 設定損失與優化器。
4. `fit(train_set, validation_data=valid_set, epochs=5)` 執行訓練。

🎯 重點摘要:
- 將 `tf.data.Dataset` 作為 `fit()` 輸入是最佳實務。
- 這樣模型可以同時使用資料預處理與批次化。
- 若資料集包含不定長度元素，`batch()` 之後可能需要 `padded_batch()`。

---

---

[來源: ch13] ##  TFRecord 格式與 Protobuf (Protocol Buffers)

TFRecord 讓你把資料序列化為二進位檔案，適合大規模訓練及跨平台部署。

```python
with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")
```

✅ 程式碼逐行解析:
1. `TFRecordWriter("my_data.tfrecord")` 建立 TFRecord 寫入器。
2. `f.write(...)` 逐條寫入二進位紀錄。

---

[來源: ch13] ``

✅ 程式碼逐行解析:
1. `TFRecordWriter("my_data.tfrecord")` 建立 TFRecord 寫入器。
2. `f.write(...)` 逐條寫入二進位紀錄。

🎯 重點摘要:
- TFRecord 本質上是 bytes 的串列。
- 內容可以是任意二進位資料，但常見於序列化 Protobuf。

```python
filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `TFRecordDataset(filepaths)` 建立讀取器。
2. 逐條列印每個二進位紀錄。

---

[來源: ch13] m in dataset:
    print(item)
```

✅ 程式碼逐行解析:
1. `TFRecordDataset(filepaths)` 建立讀取器。
2. 逐條列印每個二進位紀錄。

🎯 重點摘要:
- TFRecord 播放與 CSV 讀取方式類似，但更適合二進位序列化資料。
- 可以搭配 `num_parallel_reads` 讀多個 TFRecord 檔案。

---

[來源: ch13] ### Protobuf 基本介紹

TensorFlow 常用 `tf.train.Example` 來表示結構化樣本，這種格式實際上是 Protobuf 序列化的二進位資料。透過 `tf.train.Feature` 與 `tf.train.Example` 建立樣本後，可寫入 TFRecord；在讀取時，使用 `tf.io.parse_single_example()` 將二進位資料解析回張量。

---

[來源: ch13] #### 實作範例：寫入 TFRecord 與 Protobuf

```python
import tensorflow as tf

---

[來源: ch13] # 建立對應的 Feature helper (Protobuf Helper Functions)
def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

---

[來源: ch13] y()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """ Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

---

[來源: ch13] from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """ Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

---

[來源: ch13] ol / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(num_rooms, median_income, median_house_value):
    """ Creates a tf.train.Example message ready to be written to a file."""
    feature = {
        "num_rooms": _float_feature(num_rooms),
        "median_income": _float_feature(median_income),
        "median_house_value": _f

---

[來源: ch13] num_rooms),
        "median_income": _float_feature(median_income),
        "median_house_value": _float_feature(median_house_value),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

record = serialize_example(6.0, 3.1, 265000.0)
with tf.io.TFRecordWriter("my_data.tfrecord") as writer:
    writer.write(record)
```

---

[來源: ch13] 3.1, 265000.0)
with tf.io.TFRecordWriter("my_data.tfrecord") as writer:
    writer.write(record)
```

✅ 程式碼逐行解析:
1. 定義 `_bytes_feature`, `_float_feature`, `_int64_feature` 等輔助函數，將 Python 值轉換為 `tf.train.Feature` 物件，以適應不同資料型別。
2. `serialize_example()` 函式接收原始資料，將其包裝成 `tf.train.Example` 訊息，並序列化為二進位字串。
3. 建立一個 `tf.io.TFRecordWriter` 物件，用於寫入 TFRecord 檔案。
4. 調用 `writer.write(record)` 將序列化後的 `tf.train.Example` 寫入到 `my_data.tfrecord` 檔案中。

---

[來源: ch13] #### 實作範例：讀取 TFRecord 與 Protobuf

當我們需要讀取這些資料時，必須先定義一個 `feature_description` 字典，說明每個欄位的資料型別與形狀，然後使用 `tf.io.parse_single_example()` 來解析 TFRecord 中的二進位資料。

---

[來源: ch13] 一個 `feature_description` 字典，說明每個欄位的資料型別與形狀，然後使用 `tf.io.parse_single_example()` 來解析 TFRecord 中的二進位資料。

```python
feature_description = {
    "num_rooms": tf.io.FixedLenFeature([], tf.float32),
    "median_income": tf.io.FixedLenFeature([], tf.float32),
    "median_house_value": tf.io.FixedLenFeature([], tf.float32),
}


def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


dataset = tf.data.TFRecordDataset(["my_data.tfrecord"])
parsed_dataset = dataset.map(parse_example)
for parsed_record in parsed_dataset.take(1):
    print(parsed_record)
```

---

[來源: ch13] dataset.map(parse_example)
for parsed_record in parsed_dataset.take(1):
    print(parsed_record)
```

✅ 程式碼逐行解析:
1. `feature_description` 定義每個欄位的資料型別與形狀。
2. `parse_single_example()` 解析 TFRecord bytes 成字典張量。
3. `TFRecordDataset` 搭配 `.map(parse_example)` 讀取與反序列化資料。

🎯 重點摘要:
- `tf.train.Example` 與 `TFRecord` 讓資料序列化成單一二進位檔案，適合大規模訓練與分散式讀取。
- 讀取時需先定義 `feature_description`，才能把 Protobuf bytes 轉回可訓練的張量。

---

[來源: ch13] TFRecord` 讓資料序列化成單一二進位檔案，適合大規模訓練與分散式讀取。
- 讀取時需先定義 `feature_description`，才能把 Protobuf bytes 轉回可訓練的張量。

> 實務觀察：在實際應用中，使用 TFRecord 與 Protobuf 可以大幅提升資料讀取效率，尤其是在分散式訓練環境中。建議在資料預處理階段就將資料轉換為 TFRecord 格式，並利用 Protobuf 定義清晰的資料結構，以確保訓練過程中的資料一致性與可重用性。
> 以照片為例，直接將圖片序列化為 TFRecord 中的 bytes 欄位，並在讀取時解析回圖片張量，可以避免在訓練過程中頻繁的磁碟 I/O 操作，提升整體訓練效率。

---

[來源: ch13] 過程中的資料一致性與可重用性。
> 以照片為例，直接將圖片序列化為 TFRecord 中的 bytes 欄位，並在讀取時解析回圖片張量，可以避免在訓練過程中頻繁的磁碟 I/O 操作，提升整體訓練效率。

本節將示範如何從磁碟讀取圖片的原始位元組，將其與對應的整數標籤打包成 `tf.train.Example` 訊息，然後序列化並寫入 TFRecord 檔案。

---

[來源: ch13] 的磁碟 I/O 操作，提升整體訓練效率。

本節將示範如何從磁碟讀取圖片的原始位元組，將其與對應的整數標籤打包成 `tf.train.Example` 訊息，然後序列化並寫入 TFRecord 檔案。

```python
def create_image_example(image_bytes, label):
    """Creates a tf.train.Example message ready to be written to a file."""
    # 1. Map your data to the helper functions
    feature = {
        'image_raw': _bytes_feature(image_bytes), # Storing the compressed JPEG bytes
        'label': _int64_feature(label),           # Storing the class ID
    }

---

[來源: ch13] compressed JPEG bytes
        'label': _int64_feature(label),           # Storing the class ID
    }

# 2. Wrap it in a Features message, then an Example message
    return tf.train.Example(features=tf.train.Features(feature=feature))

---

[來源: ch13] # Example Writing Loop
tfrecord_filename = 'dataset_shard_0001.tfrec'

---

[來源: ch13] # tf.io.TFRecordWriter is the object that actually writes the file to disk
with tf.io.TFRecordWriter(tfrecord_filename) as writer:

# Assuming 'image_paths' is a list of file paths and 'labels' are their classes
    for image_path, label in zip(image_paths, labels):

---

[來源: ch13] of file paths and 'labels' are their classes
    for image_path, label in zip(image_paths, labels):

# PRODUCTION RULE: Read raw bytes! Do not use tf.image.decode_jpeg here.
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

# Create the Protobuf
        tf_example = create_image_example(image_bytes, label)

---

[來源: ch13] ytes = f.read()

# Create the Protobuf
        tf_example = create_image_example(image_bytes, label)

# Serialize to binary string and write to file
        writer.write(tf_example.SerializeToString())

print(f"Successfully wrote {tfrecord_filename}")

```

```

python

---

[來源: ch13] # 1. Create a dictionary describing the features to parse
feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

---

[來源: ch13] mage_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def parse_and_decode(serialized_example):
    """Parses a single tf.train.Example and decodes the image."""
    # Parse the binary string back into a dictionary
    parsed = tf.io.parse_single_example(serialized_example, feature_description)

---

[來源: ch13] k into a dictionary
    parsed = tf.io.parse_single_example(serialized_example, feature_description)

# NOW we decode the JPEG bytes into a mathematical tensor
    image = tf.io.decode_jpeg(parsed['image_raw'], channels=3)

# Resize and normalize for the neural network
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0

label = parsed['label']
    return image, label

---

[來源: ch13] # 2. Build the Production Pipeline

---

[來源: ch13] # Use tf.data.TFRecordDataset to read the binary files
dataset = tf.data.TFRecordDataset([tfrecord_filename])

---

[來源: ch13] # Apply the parsing function in parallel using AUTOTUNE
dataset = dataset.map(parse_and_decode, num_parallel_calls=tf.data.AUTOTUNE)

---

[來源: ch13] # Standard pipeline operations: shuffle, batch, and crucially, prefetch!
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

---

[來源: ch13] # The dataset is now ready to be passed directly to model.fit()

---

[來源: ch13] # model.fit(dataset, epochs=10)
```

> The High-Performance TFRecord Pipeline (batch->map->prefetch) is the gold standard for training efficiency. By reading raw bytes, we minimize disk I/O overhead during training. The parsing and decoding steps are done in parallel, and prefetching ensures that the GPU always has data ready to process, maximizing throughput.

```python

---

[來源: ch13] # Parses a BATCH of 32 examples at once!
def parse_batch(serialized_batch):
    # Notice we use parse_example instead of parse_single_example
    parsed = tf.io.parse_example(serialized_batch, features)
    
    # Mathematical augmentations applied to all 32 images instantly
    images = parsed['image_raw'] / 255.0 
    return images, parsed['label']

dataset = dataset.shuffle(1000)

---

[來源: ch13] # We batch the raw string bytes!
dataset = dataset.batch(32)

---

[來源: ch13] # Now we map the batch
dataset = dataset.map(parse_batch, num_parallel_calls=tf.data.AUTOTUNE)
```

---

[來源: ch13] #### 終極混合方法：真實世界的資料管線優化策略

在實際生產環境中，處理影像資料集時常常會遇到一個常見的障礙：**影像尺寸不一致**。由於壓縮的 JPEG 影像解析度各異，`dataset.batch()` 操作會因元素形狀不一致而崩潰。這意味著，在將影像批次化之前，我們必須將它們全部調整為統一的尺寸。

為了解決這個問題並最大化訓練吞吐量，業界通常會採用一套經過實戰驗證的「終極資料管線」策略，將 `map` 函數巧妙地拆分成兩個階段。這套管線的步驟及其背後的設計理念如下：

---

[來源: ch13] 它們全部調整為統一的尺寸。

為了解決這個問題並最大化訓練吞吐量，業界通常會採用一套經過實戰驗證的「終極資料管線」策略，將 `map` 函數巧妙地拆分成兩個階段。這套管線的步驟及其背後的設計理念如下：

1.  **打亂 (Shuffle)：混洗壓縮後的字串**
    *   **目的**：在讀取資料的早期階段，盡快對資料進行初步的隨機混洗。此時，資料通常以輕量的、壓縮的字串形式存在（例如，TFRecord 中的二進位資料或檔案路徑）。對這些「小巧」的元素進行打亂，可以消耗較少的記憶體資源，並為後續的訓練提供良好的隨機性。
    *   **實作**：使用 `tf.data.Dataset.shuffle(buffer_size)` 進行操作，確保資料在進入解碼階段前已被充分打亂。

---

[來源: ch13] 源，並為後續的訓練提供良好的隨機性。
    *   **實作**：使用 `tf.data.Dataset.shuffle(buffer_size)` 進行操作，確保資料在進入解碼階段前已被充分打亂。

2.  **映射 (Map)：解碼與調整大小 (decode_and_resize)**
    *   **目的**：將壓縮的影像字串解碼為原始像素資料，並將其調整為統一的形狀（例如，224x224 像素）。這是批次化操作的先決條件，因為只有所有影像都具有相同尺寸，才能正確地形成批次。這個階段通常是計算密集型的，因此會利用多個平行執行緒來加速處理。
    *   **實作**：使用 `tf.data.Dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)`，讓 TensorFlow 自動調整平行處理的效率。

---

[來源: ch13] .data.Dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)`，讓 TensorFlow 自動調整平行處理的效率。

3.  **批次化 (Batch)：將統一尺寸的影像分組**
    *   **目的**：一旦所有影像都被解碼並調整為相同尺寸，就可以將它們分組為固定大小的批次（例如，每批 32 張影像）。批次化對於高效利用 GPU/TPU 至關重要，因為它們擅長處理大型的矩陣運算。
    *   **實作**：呼叫 `dataset.batch(batch_size)`。

---

[來源: ch13] 例如，每批 32 張影像）。批次化對於高效利用 GPU/TPU 至關重要，因為它們擅長處理大型的矩陣運算。
    *   **實作**：呼叫 `dataset.batch(batch_size)`。

4.  **映射 (Map)：向量化資料增強 (vectorized_augmentations)**
    *   **目的**：在影像被批次化之後，對整個批次的影像同時應用各種數學密集型的資料增強操作。這些操作，例如調整亮度、對比度或進行正規化，可以在 GPU 上以高度並行的方式執行，大幅提升效率。將增強操作放在批次化之後，可以最大程度地利用硬體的向量化處理能力。
    *   **實作**：使用 `dataset.map(vectorized_augmentations, num_parallel_calls=tf.data.AUTOTUNE)`。請確保 `vectorized_augmentations` 函數能夠接受並處理整個影像批次。

---

[來源: ch13] augmentations, num_parallel_calls=tf.data.AUTOTUNE)`。請確保 `vectorized_augmentations` 函數能夠接受並處理整個影像批次。

5.  **預取 (Prefetch)：為 GPU 準備資料**
    *   **目的**：這是資料管線的最終優化步驟。`prefetch()` 允許資料處理（CPU 密集型）與模型訓練（GPU 密集型）並行執行。當 GPU 正在處理當前批次的資料時，CPU 會在後台預先載入並處理下一批資料，從而減少 GPU 的等待時間，最大化整體訓練吞吐量。
    *   **實作**：呼叫 `dataset.prefetch(buffer_size=tf.data.AUTOTUNE)`。

---

[來源: ch13] 下一批資料，從而減少 GPU 的等待時間，最大化整體訓練吞吐量。
    *   **實作**：呼叫 `dataset.prefetch(buffer_size=tf.data.AUTOTUNE)`。

透過理解並實踐這些操作的執行順序及其背後的邏輯，您將能夠像一位專業的 MLOps 工程師一樣，設計並優化高效能的資料管線，從而實現最大化的模型訓練吞吐量！

---

---

[來源: ch13] ##  Keras 前處理層 (Preprocessing Layers)

Keras 前處理層可以把資料規範化、離散化或文字向量化註入模型架構中，確保訓練與推論的一致性。

---

[來源: ch13] ```python
tf.random.set_seed(42)  # extra code – ensures reproducibility
norm_layer = tf.keras.layers.Normalization()
model = tf.keras.models.Sequential([
    norm_layer,
    tf.keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))
norm_layer.adapt(X_train)  # computes the mean and variance of every feature
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5)
```

---

[來源: ch13] iance of every feature
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5)
```

✅ 程式碼逐行解析:
1. `tf.random.set_seed(42)` 設定隨機種子，確保重現性。
2. `Normalization()` 建立輸入規範化層。
3. `Sequential([...])` 將規範化層放在模型最前端。
    - `norm_layer` 會在訓練過程中自動將輸入資料標準化，確保訓練與推論的一致性。
4. `norm_layer.adapt(X_train)` 計算訓練資料的平均值與標準差。
5. `fit(...)` 將標準化層與模型一起訓練。

---

[來源: ch13] 程中自動將輸入資料標準化，確保訓練與推論的一致性。
4. `norm_layer.adapt(X_train)` 計算訓練資料的平均值與標準差。
5. `fit(...)` 將標準化層與模型一起訓練。

🎯 重點摘要:
- 將前處理層放在模型內能避免訓練/推論不一致。
- `adapt()` 只需呼叫一次，之後相同層可重複使用。

```python
age = tf.constant([[10.], [93.], [57.], [18.], [37.], [5.]])
discretize_layer = tf.keras.layers.Discretization(bin_boundaries=[18., 50.])
age_categories = discretize_layer(age)
age_categories
```

---

[來源: ch13] .Discretization(bin_boundaries=[18., 50.])
age_categories = discretize_layer(age)
age_categories
```

✅ 程式碼逐行解析:
1. 建立年齡張量。
2. `Discretization(...)` 設定分箱邊界。
    - 年齡小於 18 歲的會被分到第一類（0），介於 18-50 歲的會被分到第二類（1），大於 50 歲的會被分到第三類（2）。
3. 呼叫前處理層得到類別索引。

🎯 重點摘要:
- `Discretization` 適合連續數值特徵離散化。
- 若要建立排行榜、年齡段等特徵工程，這是理想工具。

---

[來源: ch13] ### 將類別索引轉換為 one-hot 編碼

類別索引通常以整數形式表示，但神經網路傾向於 one-hot 編碼。`CategoryEncoding` 層可將類別索引轉換為稀疏向量表示，使模型能更有效地處理類別特徵。

```python
onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3)
onehot_layer(age_categories)
```

✅ 程式碼逐行解析:
1. `CategoryEncoding(num_tokens=3)` 建立 one-hot/多熱編碼層。
2. `onehot_layer(age_categories)` 轉換類別索引為稀疏表示。

---

[來源: ch13] 1. `CategoryEncoding(num_tokens=3)` 建立 one-hot/多熱編碼層。
2. `onehot_layer(age_categories)` 轉換類別索引為稀疏表示。

🎯 重點摘要:
- `CategoryEncoding` 可用於 `one_hot`、`multi_hot`、`count` 等模式。
- 需要先經過 `StringLookup` 或離散化層將類別轉成索引。

---

[來源: ch13] 摘要:
- `CategoryEncoding` 可用於 `one_hot`、`multi_hot`、`count` 等模式。
- 需要先經過 `StringLookup` 或離散化層將類別轉成索引。

```python
cities = ["Auckland", "Paris", "Paris", "San Francisco"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(cities)
print(str_lookup_layer.get_vocabulary())
str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])
```

---

[來源: ch13] _layer.get_vocabulary())
str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])
```

✅ 程式碼逐行解析:
1. `StringLookup()` 建立文字索引層。
2. `adapt(cities)` 根據資料建立詞彙表。
3. `get_vocabulary()` 顯示詞彙表內容，此函數回傳實際分配的詞彙表順序。
    - `[UNK]` 是 OOV token，索引為 0。
    - `Paris` 索引為 1，`San Francisco` 索引為 2，`Auckland` 索引為 3。
4. 呼叫該層轉換字串為整數索引。
    - `["Paris"]` 轉為 1，`["Auckland"]` 轉為 3，`["Montreal"]` 轉為 0（OOV）。

---

[來源: ch13] kland` 索引為 3。
4. 呼叫該層轉換字串為整數索引。
    - `["Paris"]` 轉為 1，`["Auckland"]` 轉為 3，`["Montreal"]` 轉為 0（OOV）。

🎯 重點摘要:
- `StringLookup` 支援 `num_oov_indices` 來處理未登錄文字。
- 若要直接輸入文字特徵到模型，先做 `StringLookup` 是最佳做法。

---

---

[來源: ch13] ##  TensorFlow Datasets (TFDS)

TFDS 提供標準資料集載入器，避免手動處理資料集檔案。

```python
import tensorflow_datasets as tfds

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]
```

✅ 程式碼逐行解析:
1. `tfds.load(name="mnist")` 下載並回傳資料集字典。
2. `datasets["train"]` 與 `datasets["test"]` 取得訓練與測試集。

---

[來源: ch13] 逐行解析:
1. `tfds.load(name="mnist")` 下載並回傳資料集字典。
2. `datasets["train"]` 與 `datasets["test"]` 取得訓練與測試集。

🎯 重點摘要:
- TFDS 內建大量資料集，節省資料準備時間。
- 可直接取得 `as_supervised=True` 的 `(features, label)` 格式資料。

---

[來源: ch13] 取得訓練與測試集。

🎯 重點摘要:
- TFDS 內建大量資料集，節省資料準備時間。
- 可直接取得 `as_supervised=True` 的 `(features, label)` 格式資料。

```python
train_set, valid_set, test_set = tfds.load(
    name="mnist",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True
)
train_set = train_set.shuffle(10_000, seed=42).batch(32).prefetch(1)
```

---

[來源: ch13] ],
    as_supervised=True
)
train_set = train_set.shuffle(10_000, seed=42).batch(32).prefetch(1)
```

✅ 程式碼逐行解析:
1. `split=["train[:90%]", "train[90%:]", "test"]` 將資料集切成訓練、驗證、測試。
    - `train[:90%]` 取前 90% 作為訓練集，`train[90%:]` 取後 10% 作為驗證集。
    - `test` 直接使用測試集。
2. `.shuffle(10_000, seed=42)` 進行洗牌。
3. `.batch(32).prefetch(1)` 批次化並提前預取。

🎯 重點摘要:
- TFDS 的分割語法可精準控制資料比例。
- 與 `tf.data` 搭配使用可取得高效輸入管線。

---

[來源: ch13] 。
3. `.batch(32).prefetch(1)` 批次化並提前預取。

🎯 重點摘要:
- TFDS 的分割語法可精準控制資料比例。
- 與 `tf.data` 搭配使用可取得高效輸入管線。

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification.cats_vs_dogs import CatsVsDogs, _NAME_RE
import io, zipfile

---

[來源: ch13] tensorflow_datasets.image_classification.cats_vs_dogs import CatsVsDogs, _NAME_RE
import io, zipfile

def fixed_generate_examples(self, archive):
    num_skipped = 0
    for fname, fobj in archive:
        norm_fname = fname.replace("\\", "/")
        res = _NAME_RE.match(norm_fname)
        if not res:
            continue
        label = res.group(1).lower()
        if tf.compat.as_bytes("JFIF")

---

[來源: ch13] res:
            continue
        label = res.group(1).lower()
        if tf.compat.as_bytes("JFIF") not in fobj.peek(10):
            num_skipped += 1
            continue
        img_data = fobj.read()
        img_tensor = tf.image.decode_image(img_data)
        img_recoded = tf.io.encode_jpeg(img_tensor)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as new_zip:
       

---

[來源: ch13] _tensor)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as new_zip:
            new_zip.writestr(norm_fname, img_recoded.numpy())
        new_fobj = zipfile.ZipFile(buffer).open(norm_fname)
        yield norm_fname, {"image": new_fobj, "image/filename": norm_fname, "label": label}

CatsVsDogs._generate_examples = fixed_generate_examples

---

[來源: ch13] # 1. Load the dataset (Automatically fetches/builds TFRecords locally)

---

[來源: ch13] # as_supervised=True returns a clean (features, label) tuple instead of a dictionary

---

[來源: ch13] # with_info=True gives us metadata (like the exact number of examples for shuffling)
(ds_train, ds_test), ds_info = tfds.load(
    'cats_vs_dogs', 
    split=['train[:80%]', 'train[80%:]'], # Native slice support
    as_supervised=True, 
    with_info=True
)

---

[來源: ch13] # 2. Define a batch-aware mapping function
def decode_and_resize(image, label):
    # Remember: Keep complex image processing on the CPU to prevent Training-Serving Skew later!
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

---

[來源: ch13] # 3. Build the highly-optimized asynchronous pipeline
BATCH_SIZE = 32

---

[來源: ch13] ds_train = (ds_train
    .cache() # Cache to RAM or disk if the dataset fits
    .shuffle(buffer_size=ds_info.splits['train'].num_examples) # Full random shuffle
    .map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE) # Multi-threaded processing
    .batch(BATCH_SIZE) # Group for vectorized GPU math
    .prefetch(tf.data.AUTOTUNE) # Overlap CPU and GPU work
)
```

---

[來源: ch13] ZE) # Group for vectorized GPU math
    .prefetch(tf.data.AUTOTUNE) # Overlap CPU and GPU work
)
```

✅ 程式碼逐行解析:
1. `tfds.load(...)` 下載並分割 `cats_vs_dogs`
    - `split=['train[:80%]', 'train[80%:]']` 直接在載入階段切分訓練與測試集。
    - `as_supervised=True` 以 `(features, label)` 格式回傳資料。
    - `with_info=True` 同時回傳資料集的元資訊，包含樣本數量等。
2. 定義 `decode_and_resize` 函數，將影像調整為 224x224 並標準化。
3. 建立資料管線：
    - `.cache()` 將資料集緩

---

[來源: ch13] 資料集的元資訊，包含樣本數量等。
2. 定義 `decode_and_resize` 函數，將影像調整為 224x224 並標準化。
3. 建立資料管線：
    - `.cache()` 將資料集緩存在記憶體或磁碟中（如果資料集足夠小）。
    - `.shuffle(buffer_size=ds_info.splits['train'].num_examples)` 進行完全隨機洗牌。
    - `.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)` 多線程處理影像解碼與調整大小。
    - `.batch(BATCH_SIZE)` 批次化資料。
    - `.prefetch(tf.data.AUTOTUNE)` 讓 CPU 與 GPU 工作重疊，提升效率。

---

---

[來源: ch13] ##  章節重點與實務建議

- `tf.data` 管線是建立可重用資料處理流程的基礎，建議先從 `list_files()`、`interleave()`、`map()`、`batch()`、`prefetch()` 熟悉起。
- 若資料集無法全部放進記憶體，優先使用檔案格式（CSV、TFRecord）搭配 `TextLineDataset` 或 `TFRecordDataset`。
- Keras 前處理層可提高模型可移植性，尤其在部署時可避免訓練與推論資料不一致。
- `TFRecord + Protobuf` 是大規模訓練與分散式資料處理的首選，但需要額外的序列化與解析邏輯。

---

---

[來源: ch13] ##  常見問答

**Q: 為什麼要用 `prefetch()`？**  
A: `prefetch()` 可以讓資料讀取與模型訓練併行，避免 GPU/TPU 等待資料，提升整體吞吐量。

**Q: CSV 與 TFRecord 哪個比較好？**  
A: 若只是小型資料集，CSV 方便觀察與除錯；若是大規模資料、二進位資料或分散式訓練，TFRecord 與 Protobuf 更穩定。

**Q: `StringLookup` 可以直接當第一層嗎？**  
A: 在某些舊版 Keras 有相容性問題，建議加 `InputLayer(input_shape=[], dtype=tf.string)` 作為第一層做為最佳實務。

---

[來源: ch13] 可以直接當第一層嗎？**  
A: 在某些舊版 Keras 有相容性問題，建議加 `InputLayer(input_shape=[], dtype=tf.string)` 作為第一層做為最佳實務。

**Q: 我該在哪裡做標準化？**  
A: 若希望模型可直接部署，最好把標準化層內置到模型中；若需要更高效訓練，可以在 `tf.data` 管線中先做好數值標準化。

---

[來源: ch13] ## 建議標籤
#Python #TensorFlow #資料預處理 #機器學習 #教學 #tfdata #TFRecord #Keras #開發者分享

---

[來源: ch13 | 類型: cheatsheet] # Ch13 速查表：Loading & Preprocessing Data with TensorFlow

> **核心主旨**：`tf.data` 讓資料管線成為訓練瓶頸的解決方案 —— `map().cache().shuffle().batch().prefetch()` 是黃金公式。

---

---

[來源: ch13 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| `tf.data.Dataset` | TF 的資料管線 API，支援惰性求值 | 所有 TF 訓練流程 |
| `from_tensor_slices()` | 從記憶體中的陣列/dict 建立 dataset | 資料可放入記憶體 |
| `from_generator()` | 從 Python generator 建立 dataset | 自訂複雜資料來源 |
| `map()` | 對每個元素套用轉換函數 | 圖片解碼、特徵工程 |
| `cache()` | 快取 dataset 到記憶體（第一個 epoch 後） | 避免重複 I/O |
| `shuffle()` | 隨機打亂（重要！用足夠大的 buffer） | 訓練集必用 |
| `batch()` | 組合成批次 | 每次訓練一批 |
| `prefetch()` | 預取下一批，GPU 訓練時重疊 I/O | 幾乎永遠需要 |
| TFRecord | TF 專用二進位格式，讀取極快 | 大型資料集 |
| Keras Preprocessing Layers | 將前處理嵌入模型，部署更簡單 | 標準化、Token化等 |


---

[來源: ch13 | 類型: cheatsheet] TFRecord | TF 專用二進位格式，讀取極快 | 大型資料集 |
| Keras Preprocessing Layers | 將前處理嵌入模型，部署更簡單 | 標準化、Token化等 |


---

---

[來源: ch13 | 類型: cheatsheet] | TF API | 重點參數 | 用途 |
|--------|---------|------|
| `tf.data.Dataset.from_tensor_slices()` | `tensors` | 從記憶體建立 dataset |
| `dataset.map()` | `map_func`, `num_parallel_calls=tf.data.AUTOTUNE` | 並行轉換 |
| `dataset.cache()` | `filename=""` (空字串=記憶體快取) | 快取 |
| `dataset.shuffle()` | `buffer_size=1000`, `seed=42` | 打亂 |
| `dataset.batch()` | `batch_size=32`, `drop_remainder=False` | 批次化 |
| `dataset.prefetch()` | `buffer_size=tf.data.AUTOTUNE` | 預取 |
| `dataset.repeat()` | `count=None` (無限重複) | 多 epoch 時 |
| `tf.io.read_file()` | – | 讀取原始檔案 |
| `tf.io.decode_jpeg()` | – | 解碼 JPEG 圖片 |
| `tf.image.resize()` | `size=[224, 224]` | 調整圖片大小 |
| `tf.keras.layers.Normalization` | `axis=-1` | 特徵標準化層 |
| `tf.keras.layers.TextVectorization` | `max_tokens=`, `output_sequence_length=` | 文字 Token 化 |


---

[來源: ch13 | 類型: cheatsheet] 層 |
| `tf.keras.layers.TextVectorization` | `max_tokens=`, `output_sequence_length=` | 文字 Token 化 |


---

---

[來源: ch13 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import tensorflow as tf

---

[來源: ch13 | 類型: cheatsheet] # 黃金公式：訓練集管線
def create_train_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X), seed=42)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

---

[來源: ch13 | 類型: cheatsheet] # 驗證集管線（不 shuffle）
def create_val_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

---

[來源: ch13 | 類型: cheatsheet] # 圖片資料管線範例
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.xception.preprocess_input(image)
    return image, label

---

[來源: ch13 | 類型: cheatsheet] # TFRecord 寫入
def serialize_example(feature_dict):
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()

---

[來源: ch13 | 類型: cheatsheet] ain.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()

with tf.io.TFRecordWriter("data.tfrecord") as writer:
    for X, y in zip(X_train, y_train):
        feature = {
            "X": tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
            "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
        }
        writer.write(serialize_example(feature))

---

[來源: ch13 | 類型: cheatsheet] # TFRecord 讀取
feature_description = {
    "X": tf.io.FixedLenFeature([n_features], tf.float32),
    "y": tf.io.FixedLenFeature([], tf.int64)
}

def parse_example(serialized):
    example = tf.io.parse_single_example(serialized, feature_description)
    return example["X"], example["y"]

dataset = tf.data.TFRecordDataset("data.tfrecord").map(parse_example)

---

[來源: ch13 | 類型: cheatsheet] # Keras 前處理層（嵌入模型，部署一致性）
normalization_layer = tf.keras.layers.Normalization(axis=-1)
normalization_layer.adapt(X_train)  # 計算 mean 和 std

model = tf.keras.Sequential([
    normalization_layer,          # 前處理包在模型裡
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch13 | 類型: cheatsheet] # tfds 載入公開資料集
import tensorflow_datasets as tfds
(train_ds, valid_ds), ds_info = tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)
n_classes = ds_info.features["label"].num_classes
```

---

---

[來源: ch13 | 類型: cheatsheet] ## 4. 常見陷阱

- **`shuffle` buffer_size 要夠大**：`buffer_size=1000` 只能保證 1000 個樣本的隨機性；理想情況用整個資料集大小，但記憶體有限時用 `min(len(dataset), 10000)`。
- **`cache()` 的位置**：必須放在 `map()` 之後（快取處理後的資料），`shuffle()` 之前（每 epoch 重新打亂）。
- **`prefetch()` 要放最後**：讓 CPU 預取下一批的同時 GPU 正在訓練當前批。
- **`num_parallel_calls=tf.data.AUTOTUNE`**：讓 TF 自動決定並行程度，幾乎每次 `map()` 都應加。
- **前處理層 `adapt()` 只能在訓練集上執行**：和 sklearn 的 `StandardScaler.fit()` 一樣，不能看驗證/測試集。

---

---

[來源: ch13 | 類型: cheatsheet] ## 5. 決策指南

```
資料大小 vs 策略：
├── 全部放入記憶體 (< ~10GB)
│   → from_tensor_slices + cache() 在 shuffle 前
├── 中型資料（SSD 上的檔案）
│   → 讀取路徑清單 + map(load_file) + cache() + shuffle
└── 超大資料（雲端 / 分散式）
    → TFRecord 格式 + TFRecordDataset + 分片

管線效能調優 checklist：
1. map() 加 num_parallel_calls=AUTOTUNE
2. cache() 避免重複 I/O
3. shuffle() 的 buffer_size 夠大
4. prefetch(AUTOTUNE) 放最後
5. 圖片解碼考慮用 tf.io.decode_image（更通用）
```

---

[來源: ch13 | 類型: handout] # 課程講義：TensorFlow 資料載入與預處理 (Chapter 13)

本章節聚焦於 TensorFlow 資料管線的建構與優化，說明如何使用 `tf.data` 讀取、轉換、批次化與預取資料，並延伸到 TFRecord、Protobuf 以及 Keras 前處理層的實務應用。

---

[來源: ch13 | 類型: handout] ## 1. 引言

在深度學習系統中，資料輸入管線往往比模型本身更容易成為效能瓶頸。本講義以第13章教材為基礎，解析 TensorFlow 的資料載入策略，讓你能夠建立可擴充、可重用且具備訓練效率的資料處理流程。

---

[來源: ch13 | 類型: handout] ### 2.1 tf.data API 的基礎與資料流

**為什麼重要**：`tf.data` 是 TensorFlow 建立輸入管線的核心 API，它能將資料讀取、轉換與批次化串成單一管線，避免在訓練期間反覆進行 Python 層級資料處理。

**如何實作**：

```python
import tensorflow as tf

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
```

- `from_tensor_slices()` 將張量切片成單一元素資料集。
- 這種作法適合資料可放入記憶體的情境。
- 若資料太大，應改用檔案來源（如 CSV、TFRecord、Image）。

---

[來源: ch13 | 類型: handout] - `from_tensor_slices()` 將張量切片成單一元素資料集。
- 這種作法適合資料可放入記憶體的情境。
- 若資料太大，應改用檔案來源（如 CSV、TFRecord、Image）。

**⚡ 補充練習 2.1：**
- 理論：比較 `from_tensor_slices()` 與 `from_tensors()` 的行為差異，並說明在何種情境下選用各自 API。

```python
dataset2 = tf.data.Dataset.from_tensors(X)
```

```python
import tensorflow as tf

---

[來源: ch13 | 類型: handout] # 假設這是你的特徵和標籤
features = tf.constant([[1, 2], [3, 4], [5, 6]]) # 3個樣本，每個樣本有2個特徵
labels = tf.constant([0, 1, 0]) # 3個對應的標籤

dataset_slices = tf.data.Dataset.from_tensor_slices((features, labels))

---

[來源: ch13 | 類型: handout] # 檢視資料集的元素規格
dataset_slices.element_spec

print("from_tensor_slices() 的輸出：")
for element in dataset_slices:
    print(element)

---

[來源: ch13 | 類型: handout] # 假設這是一個完整的張量，你希望它作為資料集的唯一元素
full_tensor = tf.constant([[1, 2], [3, 4], [5, 6]])

dataset_tensors = tf.data.Dataset.from_tensors(full_tensor)

print("\nfrom_tensors() 的輸出：")
for element in dataset_tensors:
    print(element)
```

- 實作：建立一個 `tf.data.Dataset`，然後使用 `batch(3)` 與 `repeat(2)` 觀察輸出形態。

```python
import tensorflow as tf

---

[來源: ch13 | 類型: handout] # 包含從 0 到 9 的 10 個元素
initial_data = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(initial_data)

print("--- 原始資料集 (10 個元素) ---")
for i, element in enumerate(dataset):
    print(f"元素 {i}: {element.numpy()}")

---

[來源: ch13 | 類型: handout] # 2. 應用 repeat(2) 後再應用 batch(3)

---

[來源: ch13 | 類型: handout] # 整個資料集重複 2 次 (邏輯上變成 20 個元素)，然後每 3 個元素組成一個批次
processed_dataset = dataset.repeat(2).batch(3)

print("\n--- 經過 repeat(2).batch(3) 處理後的資料集 ---")
print("資料集的每個元素將是一個批次 (Tensor)")
print("資料集總共有 20 個邏輯元素，會被分成 7 個批次 (6 個批次大小為 3，1 個批次大小為 2)")

---

[來源: ch13 | 類型: handout] 集 ---")
print("資料集的每個元素將是一個批次 (Tensor)")
print("資料集總共有 20 個邏輯元素，會被分成 7 個批次 (6 個批次大小為 3，1 個批次大小為 2)")

for i, batch_element in enumerate(processed_dataset):
    print(f"批次 {i+1}:")
    print(f"  內容: {batch_element.numpy()}")
    print(f"  形狀: {batch_element.shape}") # 觀察每個批次張量的形狀
    print(f"  批次大小: {batch_element.shape[0]}") # 觀察批次大小
```

---

[來源: ch13 | 類型: handout] ### 2.2 轉換與交錯讀取：map、filter、interleave

**為什麼重要**：資料前處理通常需要多次轉換，若每次都在 Python 中完成，會嚴重拖慢訓練速度。`tf.data` 提供的 `map()`、`filter()` 與 `interleave()` 可將轉換移到 TensorFlow 執行，並行化處理。

**如何實作**：

```python
dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))
dataset = dataset.repeat(3).batch(7)
dataset = dataset.map(lambda x: x * 2)
```

---

[來源: ch13 | 類型: handout] slices(tf.range(10))
dataset = dataset.repeat(3).batch(7)
dataset = dataset.map(lambda x: x * 2)
```

- `repeat(3)` 代表資料集重複3次。
    - 這會讓資料集的邏輯元素從 10 個變成 30 個。
    - 形成的資料集會是 [0, 1, 2, ..., 9, 0, 1, 2, ..., 9, 0, 1, 2, ..., 9]。
- `batch(7)` 可減少模型呼叫次數，提高 GPU 利用率。
    - 這會將資料集分成批次，每個批次包含 7 個元素。
    - 由於資料集有 30 個元素，會形成 5 個批次（4 個批次大小為 7，1 個批次大小為 2）。
- `map()` 會對每個批次或元素執行轉換，與是否先 `batch()` 密切相關。
    - 若在 `batch()` 前使用 `map()`，則轉換會對每個元素執行。
    - 若在 `batch()` 後使用 `map()`，則轉換會對每個批次執行，這通常更有效率。

---

[來源: ch13 | 類型: handout] 切相關。
    - 若在 `batch()` 前使用 `map()`，則轉換會對每個元素執行。
    - 若在 `batch()` 後使用 `map()`，則轉換會對每個批次執行，這通常更有效率。

```python
filepaths = 'C:\\Users\\yuanh\\Box\\tech\\Python\\side_projects\\twstock\\engine\\robot\\reports\\*.csv'
dataset = tf.data.Dataset.list_files(filepaths, seed=42) # 讀取多個 CSV 檔案

for fp in dataset:
    print(fp)

---

[來源: ch13 | 類型: handout] # 使用 interleave 同時讀取多個檔案，並跳過 CSV 標頭
dataset = dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=5)
```

- `interleave()` 可以同時讀取多個檔案，適合多檔案資料集。
- `skip(1)` 用於跳過 CSV 標頭。

---

[來源: ch13 | 類型: handout] th).skip(1),
    cycle_length=5)
```

- `interleave()` 可以同時讀取多個檔案，適合多檔案資料集。
- `skip(1)` 用於跳過 CSV 標頭。

**⚡ 補充練習 2.2：**
- 理論：說明為何在多檔案讀取時，`interleave()` 通常比 `flat_map()` 更適合。
    - `interleave()` 會在多個檔案之間交錯讀取，能更快地提供資料給模型，減少 I/O 等待時間。
    - `flat_map()` 會將每個檔案的資料完全讀取後才開始讀取下一個檔案，可能導致資料供應不均，增加訓練等待時間。
- 實作：建立一個多檔案 `TextLineDataset`，並使用 `interleave()` 同時讀取多個檔案。

---

[來源: ch13 | 類型: handout] ### 2.3 CSV、TFRecord 與資料預處理管線

**為什麼重要**：當資料量增長時，單一 CSV 或 NumPy 檔案將無法支援高效訓練。TFRecord 可序列化資料，並搭配 `tf.data` 形成高效管線。

**如何實作**：

---

[來源: ch13 | 類型: handout] 預處理管線

**為什麼重要**：當資料量增長時，單一 CSV 或 NumPy 檔案將無法支援高效訓練。TFRecord 可序列化資料，並搭配 `tf.data` 形成高效管線。

**如何實作**：

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
```

---

[來源: ch13 | 類型: handout] ll, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
```

```python
import numpy as np
from pathlib import Path

def save_to_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = Path() / "datasets" / "housing"
    housing_dir.mkdir(parents=True, exist_ok=True)
    chunks = np.array_split(data, n_parts)
    for file_idx, chunk in enumerate(chunks):
        part_csv = housing_dir / f"my_{name_prefix}_{file_idx:02d}.csv"
        np.savetxt(part_csv, chunk, delimiter=",", header=header, comments="")

save_to_csv_files(X_train_full, "train_X", header="MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude")
save_to_csv_files(y_train_full, "train_y", header="MedHouseVal")
```

---

[來源: ch13 | 類型: handout] n,AveOccup,Latitude,Longitude")
save_to_csv_files(y_train_full, "train_y", header="MedHouseVal")
```

- 將資料拆成多個 CSV 檔案可提升 I/O 吞吐量。
- 若資料位於遠端儲存或分散式環境，分割檔案更具可擴充性。

---

[來源: ch13 | 類型: handout] ll, "train_y", header="MedHouseVal")
```

- 將資料拆成多個 CSV 檔案可提升 I/O 吞吐量。
- 若資料位於遠端儲存或分散式環境，分割檔案更具可擴充性。

```python
def parse_features(line):
    fields = tf.strings.split(line, ",")
    return tf.strings.to_number(fields, out_type=tf.float32)

def parse_labels(line):
    return tf.strings.to_number(line, out_type=tf.float32)

def csv_reader_dataset(filepaths, parser, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=42)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=5)
    dataset = dataset.map(parser, num_parallel_calls=5)
    dataset = dataset.shuffle(10_000, seed=42)
    return dataset.batch(batch_size).prefetch(1)

housing_dir = Path() / "datasets" / "housing"
train_set = csv_reader_dataset(str(housing_dir / "my_train_X_*.csv"), parse_features)
target_set = csv_reader_dataset(str(housing_dir / "my_train_y_*.csv"), parse_labels)
dataset = tf.data.Dataset.zip((train_set, target_set))

```

---

[來源: ch13 | 類型: handout] dir / "my_train_y_*.csv"), parse_labels)
dataset = tf.data.Dataset.zip((train_set, target_set))

```

- `prefetch(1)` 讓資料載入與模型訓練交錯執行。
- `shuffle(10_000)` 提高隨機化效果，減少過擬合風險。

**⚡ 補充練習 2.3：**
- 理論：說明為何 `prefetch()` 能改善 GPU 利用率，以及何時不應該過多預取。
- 實作：加入 `cache()` 於 `tf.data` 管線中，觀察同一資料集不同快取位置的效能改變。

---

[來源: ch13 | 類型: handout] 理論：說明為何 `prefetch()` 能改善 GPU 利用率，以及何時不應該過多預取。
- 實作：加入 `cache()` 於 `tf.data` 管線中，觀察同一資料集不同快取位置的效能改變。

```python
def csv_reader_dataset_cache(filepaths, parser, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=42)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=5)
    dataset = dataset.cache() # 將資料集快取

---

[來源: ch13 | 類型: handout] ta.TextLineDataset(filepath).skip(1),
        cycle_length=5)
    dataset = dataset.cache() # 將資料集快取在記憶體中
    dataset = dataset.map(parser, num_parallel_calls=5)
    dataset = dataset.shuffle(10_000, seed=42)
    return dataset.batch(batch_size).prefetch(1)

---

[來源: ch13 | 類型: handout] # housing_dir = Path() / "datasets" / "housing"
train_set = csv_reader_dataset_cache(str(housing_dir / "my_train_X_*.csv"), parse_features)
target_set = csv_reader_dataset_cache(str(housing_dir / "my_train_y_*.csv"), parse_labels)
dataset = tf.data.Dataset.zip((train_set, target_set))
```

---

[來源: ch13 | 類型: handout] ### 2.4 Protobuf 與 TFRecord 的實作原理

**為什麼重要**：`tf.train.Example` 透過 Protobuf 描述資料欄位，適合儲存圖片、文字、數值等多樣資料；TFRecord 是序列化此資料的高效格式。

**如何實作**：

---

[來源: ch13 | 類型: handout] 
**為什麼重要**：`tf.train.Example` 透過 Protobuf 描述資料欄位，適合儲存圖片、文字、數值等多樣資料；TFRecord 是序列化此資料的高效格式。

**如何實作**：

```python
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

person_example = Example(
    features=Features(feature={
        "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
        "id": Feature(int64_list=Int64List(value=[123]))
    }))
person_example
person_example.features.feature["name"].bytes_list.value[0] # b"Alice"
person_example.features.feature["id"].int64_list.value[0] # 123
```

---

[來源: ch13 | 類型: handout] ].bytes_list.value[0] # b"Alice"
person_example.features.feature["id"].int64_list.value[0] # 123
```

- `BytesList`、`FloatList`、`Int64List` 分別對應不同資料型態。
- `Feature` 以 `oneof` 方式封裝實際欄位值。
- `Example` 是序列化後的單筆資料格式。

```python
record_path = str(Path() / "datasets" / "my_data.tfrecord")
with tf.io.TFRecordWriter(record_path) as f:
    f.write(person_example.SerializeToString())
```

---

[來源: ch13 | 類型: handout] d")
with tf.io.TFRecordWriter(record_path) as f:
    f.write(person_example.SerializeToString())
```

- `SerializeToString()` 將 Example 轉成二進位紀錄。
- `TFRecordWriter` 提供一個簡潔寫入介面。

**⚡ 補充練習 2.4：**
- 理論：比較 `tf.io.parse_single_example()` 與 `tf.io.parse_example()` 的差異，說明何時使用 batch 解析更有效率。
- 實作：將 Fashion MNIST 圖片序列化為 TFRecord，並撰寫解析函數還原成原始影像。

```python

---

[來源: ch13 | 類型: handout] # 讀取 TFRecord
raw_dataset = tf.data.TFRecordDataset([record_path])

---

[來源: ch13 | 類型: handout] # 定義解析規格
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string),
    "id": tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(serialized_example):
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    parsed["name"] = tf.cast(parsed["name"], tf.string)
    return parsed

parsed_dataset = raw_dataset.map(parse_example)

---

[來源: ch13 | 類型: handout] # 檢視解析結果
for item in parsed_dataset:
    print("name:", item["name"].numpy().decode("utf-8"))
    print("id:  ", item["id"].numpy())
```

```python

---

[來源: ch13 | 類型: handout] # 解析 Fashion MNIST TFRecord 的範例
import tensorflow as tf

---

[來源: ch13 | 類型: handout] # 1) 讀取 Fashion MNIST
(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()

---

[來源: ch13 | 類型: handout] # 2) helper：將欄位轉成 tf.train.Feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

---

[來源: ch13 | 類型: handout] def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_label_to_example(image, label):
    image_bytes = tf.io.encode_png(image[..., tf.newaxis]).numpy()
    feature = {
        "image": _bytes_feature(image_bytes),
        "label": _int64_feature(int(label)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

---

[來源: ch13 | 類型: handout] # 3) 寫入 TFRecord
record_path = str(Path() / "datasets" / "fashion_mnist.tfrecord")
with tf.io.TFRecordWriter(record_path) as writer:
    for image, label in zip(train_images, train_labels):
        example = image_label_to_example(image, label)
        writer.write(example.SerializeToString())

---

[來源: ch13 | 類型: handout] # 4) 讀取與解析 TFRecord
feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}

---

[來源: ch13 | 類型: handout] "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(serialized_example):
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_png(parsed["image"], channels=1)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(parsed["label"], tf.int32)
    return image, label

---

[來源: ch13 | 類型: handout] st(image, tf.float32) / 255.0
    label = tf.cast(parsed["label"], tf.int32)
    return image, label

dataset = tf.data.TFRecordDataset([record_path])
dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

---

[來源: ch13 | 類型: handout] # 5) 檢視解析後的資料
for images, labels in dataset.take(1):
    print("影像批次形狀:", images.shape) # (32, 28, 28, 1)
    print("標籤批次形狀:", labels.shape) # (32,)
    print("第一張影像的像素值範圍:", tf.reduce_min(images[0]), "到", tf.reduce_max(images[0]))
    print("第一張影像的標籤:", labels[0].numpy())

```

```

python
import tensorflow as tf
from pathlib import Path

---

[來源: ch13 | 類型: handout] t("第一張影像的標籤:", labels[0].numpy())

```

```

python
import tensorflow as tf
from pathlib import Path

(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
record_dir = Path() / "datasets" / "fashion_mnist_tfrecords"
record_dir.mkdir(exist_ok=True)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

---

[來源: ch13 | 類型: handout] def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

---

[來源: ch13 | 類型: handout] def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_label_to_example(image, label):
    image_bytes = tf.io.encode_png(image[..., None]).numpy()
    return tf.train.Example(
        features=tf.train.Features(feature={
            "image": _bytes_feature(image_bytes),
            "label": _int64_feature(int(label)),
        })
    )

---

[來源: ch13 | 類型: handout] age": _bytes_feature(image_bytes),
            "label": _int64_feature(int(label)),
        })
    )

def write_fashion_mnist_shards(images, labels, record_dir, num_shards=10):
    n = len(images)
    shard_size = (n + num_shards - 1) // num_shards
    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n)
        shard_path = record_dir /

---

[來源: ch13 | 類型: handout] t = shard_id * shard_size
        end = min(start + shard_size, n)
        shard_path = record_dir / f"fashion_mnist_{shard_id:02d}.tfrecord"
        with tf.io.TFRecordWriter(str(shard_path)) as writer:
            for image, label in zip(images[start:end], labels[start:end]):
                writer.write(image_label_to_example(image, label).SerializeToString())

---

[來源: ch13 | 類型: handout] [start:end]):
                writer.write(image_label_to_example(image, label).SerializeToString())

write_fashion_mnist_shards(train_images, train_labels, record_dir, num_shards=10)

---

[來源: ch13 | 類型: handout] ializeToString())

write_fashion_mnist_shards(train_images, train_labels, record_dir, num_shards=10)

files = tf.io.gfile.glob(str(record_dir / "fashion_mnist_*.tfrecord"))
raw_dataset = tf.data.TFRecordDataset(files)
dataset = raw_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

---

[來源: ch13 | 類型: handout] # plot the first batch of images
import matplotlib.pyplot as plt

for images, labels in dataset.take(1):
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, :, :, 0], cmap='gray')
        ax.set_title(labels[i].numpy())
        ax.axis('off')
    plt.show()
```

---

[來源: ch13 | 類型: handout] ### 2.5 Keras 前處理層的整合與實務

**為什麼重要**：Keras 前處理層可以將輸入標準化、離散化、文字向量化等步驟內建到模型中，使訓練與部署一致。

**如何實作**：
以下是將 `Normalization` 層整合到 Keras 模型中的範例：

---

[來源: ch13 | 類型: handout] *：Keras 前處理層可以將輸入標準化、離散化、文字向量化等步驟內建到模型中，使訓練與部署一致。

**如何實作**：
以下是將 `Normalization` 層整合到 Keras 模型中的範例：

```python
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train_full)
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Input(shape=X_train_full.shape[1:]),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])
```

---

[來源: ch13 | 類型: handout] ull.shape[1:]),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])
```

- `Normalization` 會自動計算特徵平均值與變異數。
- `adapt()` 只需對訓練資料執行一次。
- 模型內建前處理層可確保訓練與推論使用相同的轉換邏輯。

這是將 `TextVectorization` 層整合到文本分類模型中的範例：

```python
text_vec_layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")
text_vec_layer.adapt(train_data)
```

---

[來源: ch13 | 類型: handout] layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")
text_vec_layer.adapt(train_data)
```

- `TextVectorization` 可直接將文字轉成向量表示。
- `output_mode="tf_idf"` 適合文本分類與檢索任務。
    - 若要使用 embedding，則 `output_mode="int"` 並搭配 `Embedding` 層。

---

[來源: ch13 | 類型: handout] - `output_mode="tf_idf"` 適合文本分類與檢索任務。
    - 若要使用 embedding，則 `output_mode="int"` 並搭配 `Embedding` 層。

**⚡ 補充練習 2.5：**
- 理論：說明 `StringLookup` 與 `Embedding` 的角色差異，並討論何時採用 embedding 取代 one-hot。
- 實作：用 `TextVectorization` 建立 IMDB 影評分類模型，並比較 `tf_idf` 與 `int` + `Embedding` 兩種前處理效果。

---

[來源: ch13 | 類型: handout] ## 3. 結論

本章重點在於建立一個從資料讀取到模型輸入的完整資料管線。掌握 `tf.data` 的組合方式、TFRecord/Protobuf 的序列化設計，以及 Keras 前處理層的整合，能讓你在實際專案中同時提升訓練效能與部署穩定性。下一章將深入影像資料與更複雜的前處理策略，進一步拓展資料工程能力。

---

[來源: ch13 | 類型: handout] ## 4. 課後作業

1. **實作題**：使用 Fashion MNIST，將資料分割成多個 TFRecord 檔案，編寫 `tf.data` 管線讀取並訓練簡單分類模型。比較直接使用 `tf.data.Dataset.from_tensor_slices()` 與 TFRecord 兩種方式的訓練速度。
2. **思考題**：若資料集包含文字、數值與類別欄位，請設計一個 `tf.data` + Keras 前處理層的完整管線，說明每個步驟為何要放在資料管線中、每個步驟為何要放入模型中。

---

[來源: ch13 | 類型: tutorial] [標題: TensorFlow 資料載入與預處理完整指南：tf.data API 與 TFRecord | 描述: 掌握 TensorFlow 的資料管線：tf.data API（from_tensor_slices、map、batch、prefetch）、TFRecord 格式、Protocol Buffers、Keras 預處理層（Normalization、TextVectorization）與圖像增強。 | 關鍵字: Python, TensorFlow, tf.data, TFRecord, 資料管線, Keras, 預處理, TextVectorization, 圖像增強, 機器學習]
# TensorFlow 資料載入與預處理：tf.data API 完整指南

在深度學習中，**資料管線（Data Pipeline）** 往往是訓練速度的瓶頸。GPU 等待資料的時間比計算時間更長——這就是 tf.data API 誕生的原因。本教學帶你建構高效的 TF 資料管線，從原始資料到模型輸入的每一步都能充分利用多核心 CPU 和磁碟 I/O。

---

[來源: ch13 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **`tf.data.Dataset`** 是 TF 的惰性評估資料管線，支援平行預取和多執行緒映射
- **`prefetch(tf.data.AUTOTUNE)`** 讓 CPU 預處理和 GPU 訓練**同步進行**，消除等待時間
- **TFRecord** 是 TF 原生的二進位格式，對大型資料集（圖像、音訊）讀取效率最高
- **Keras 預處理層**（`Normalization`、`TextVectorization`）可以整合到模型中，在推論時自動應用
- **圖像增強**（隨機翻轉、裁切、亮度調整）應只在訓練時使用（`training=True`）

---

---

[來源: ch13 | 類型: tutorial] ## tf.data API 基礎

💡 **實際應用情境：** 台灣電商平台的推薦系統每天需要處理數 TB 的使用者行為日誌。用 tf.data 的平行映射和預取，可以讓 GPU 的利用率從 20% 提升到 90% 以上——訓練速度翻倍。

---

[來源: ch13 | 類型: tutorial] ### 範例 1: 建立和操作 Dataset

```python
import tensorflow as tf
import numpy as np

---

[來源: ch13 | 類型: tutorial] # 方法 1：從記憶體資料建立（小型資料集）
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randint(0, 10, 1000)

---

[來源: ch13 | 類型: tutorial] # from_tensor_slices：沿第一個軸切片（每個樣本一個元素）
dataset = tf.data.Dataset.from_tensor_slices((X, y))
print(f"資料集大小: {len(dataset)}")     # 1000
print(f"元素規格: {dataset.element_spec}")  # 顯示形狀和 dtype

---

[來源: ch13 | 類型: tutorial] # 查看前 3 個元素
for x_sample, y_sample in dataset.take(3):
    print(f"x shape: {x_sample.shape}, y: {y_sample.numpy()}")

---

[來源: ch13 | 類型: tutorial] # 方法 2：從生成器建立（適合無法全部載入記憶體的資料）
def data_generator():
    for i in range(1000):
        yield np.random.randn(10), np.random.randint(10)

gen_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(10,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

---

[來源: ch13 | 類型: tutorial] # text_dataset = tf.data.TextLineDataset(["train.txt"])
```

**✅ 程式碼逐行解析：**

1. `from_tensor_slices((X, y))`: 建立 (X, y) 對的資料集，每次迭代回傳一對樣本
2. `dataset.element_spec`: 顯示每個元素的 `TensorSpec`（形狀 + dtype），有助於除錯
3. `from_generator`: 惰性評估——只在需要時呼叫生成器，適合大型資料集

**🎯 重點摘要:**

- `Dataset` 物件是**惰性的**——只有在迭代時才真正執行資料讀取和轉換
- `take(n)` 只取前 n 個元素，非常適合除錯時快速驗證管線

---

---

[來源: ch13 | 類型: tutorial] ### 範例 2: 標準訓練資料管線

```python
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE  # 讓 TF 自動選擇最佳平行度

def preprocess(x, y):
    """資料預處理函數（在 CPU 上平行執行）"""
    # 特徵標準化
    x = (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-8)
    return x, y

---

[來源: ch13 | 類型: tutorial] # 標準訓練資料管線
train_dataset = (
    dataset
    .shuffle(buffer_size=1000)    # 隨機打亂（buffer_size 越大越隨機）
    .map(preprocess,               # 對每個元素應用預處理函數
         num_parallel_calls=AUTOTUNE)  # 多執行緒平行映射
    .batch(BATCH_SIZE)             # 組合成 batch
    .prefetch(AUTOTUNE)            # 預取下一個 batch（消除 CPU-GPU 等待）
)

---

[來源: ch13 | 類型: tutorial] # 驗證/測試集管線（不需要 shuffle）
val_dataset = (
    dataset
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
    # 可加 .cache() 緩存到記憶體（如果資料集夠小）
)

print(f"每個 batch 的形狀: {next(iter(train_dataset))[0].shape}")
```

**✅ 程式碼逐行解析：**

1. `shuffle(buffer_size=1000)`: 維持一個大小為 1000 的緩衝區，從中隨機取樣；buffer_size 越大越接近真正隨機但記憶體用量也越大
2. `map(fn, num_parallel_calls=AUTOTUNE)`: 多執行緒平行應用 fn，AUTOTUNE 讓 TF 自動決定最佳執行緒數
3. `prefetch(AUTOTUNE)`: 在 GPU 執行當前 batch 的同時，預先準備下一個 batch——這是最重要的效能優化！

**🎯 重點摘要:**

---

[來源: ch13 | 類型: tutorial] TF 自動決定最佳執行緒數
3. `prefetch(AUTOTUNE)`: 在 GPU 執行當前 batch 的同時，預先準備下一個 batch——這是最重要的效能優化！

**🎯 重點摘要:**

- **必加** `prefetch(AUTOTUNE)` — 這是零成本的最大效能提升
- 管線順序很重要：`shuffle → map → batch → prefetch`（而非 `batch` 後再 `shuffle`）

---

---

[來源: ch13 | 類型: tutorial] ## TFRecord 格式

💡 **實際應用情境：** 醫療影像資料集（胸部 X 光，每張幾 MB）直接從 JPEG 讀取效率低下。轉換為 TFRecord 後，I/O 速度可提升 3-5 倍，因為順序讀取大型二進制文件遠快於隨機讀取許多小文件。

---

[來源: ch13 | 類型: tutorial] ### 範例 3: 寫入 TFRecord 文件

```python
import os

---

[來源: ch13 | 類型: tutorial] # 特徵序列化輔助函數
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

---

[來源: ch13 | 類型: tutorial] def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(x: np.ndarray, label: int) -> bytes:
    """將一個樣本序列化為 TFRecord 格式"""
    feature = {
        "features": _bytes_feature(x.tobytes()),   # 將陣列轉為 bytes
        "label":    _int64_feature(label),
        "n_features": _int64_feature(len(x))        # 存儲形狀資訊
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()  # 序列化為 bytes

---

[來源: ch13 | 類型: tutorial] # 寫入 TFRecord
output_path = "my_data.tfrecord"
with tf.io.TFRecordWriter(output_path) as writer:
    for i in range(len(X)):
        serialized = serialize_example(X[i], y[i])
        writer.write(serialized)
print(f"TFRecord 已儲存: {os.path.getsize(output_path)/1024:.1f} KB")
```

---

[來源: ch13 | 類型: tutorial] ### 範例 4: 讀取 TFRecord 文件

```python

---

[來源: ch13 | 類型: tutorial] # 描述特徵格式（用於反序列化）
feature_description = {
    "features":   tf.io.FixedLenFeature([], tf.string),
    "label":      tf.io.FixedLenFeature([], tf.int64),
    "n_features": tf.io.FixedLenFeature([], tf.int64),
}

def parse_tfrecord(serialized_example):
    """將 bytes 反序列化回張量"""
    parsed = tf.io.parse_single_example(serialized_example, feature_description)

    # 將 bytes 還原為 float32 陣列
    n = parsed["n_features"]
    x = tf.io.decode_raw(parsed["features"], tf.float32)
    x = tf.reshape(x, [n])
    label = parsed["label"]
    return x, label

---

[來源: ch13 | 類型: tutorial] # 建立 TFRecord 資料集
tfrecord_dataset = (
    tf.data.TFRecordDataset([output_path])
    .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    .shuffle(500)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

---

[來源: ch13 | 類型: tutorial] # 驗證
for x_batch, y_batch in tfrecord_dataset.take(1):
    print(f"從 TFRecord 讀取: x shape={x_batch.shape}, y shape={y_batch.shape}")
```

**✅ 程式碼逐行解析：**

1. `tf.train.Feature(bytes_list=...)`: TFRecord 支援 bytes/float/int64 三種特徵類型
2. `tf.train.Example(features=...)`: 一個 Example 包含多個命名特徵，相當於一行資料
3. `tf.io.parse_single_example`: 根據 `feature_description` 反序列化，類型不符會報錯

---

---

[來源: ch13 | 類型: tutorial] ### 範例 5: Normalization 層（嵌入模型的標準化）

```python
import numpy as np

X_housing = np.random.randn(1000, 8).astype(np.float32)
y_housing = np.random.randn(1000, 1).astype(np.float32)

---

[來源: ch13 | 類型: tutorial] # Normalization 層：計算並儲存訓練集的均值和方差
normalizer = tf.keras.layers.Normalization(input_shape=[8])
normalizer.adapt(X_housing)  # 在訓練資料上計算統計量（類似 fit）

print(f"均值: {normalizer.mean.numpy().round(3)}")
print(f"方差: {normalizer.variance.numpy().round(3)}")

---

[來源: ch13 | 類型: tutorial] # 整合進模型（推論時自動標準化）
model_with_norm = tf.keras.Sequential([
    normalizer,                               # 第一層：標準化
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)                  # 回歸輸出
])

model_with_norm.compile(optimizer="adam", loss="mse")

---

[來源: ch13 | 類型: tutorial] # model_with_norm.fit(X_housing, y_housing, epochs=5)
```

**🎯 重點摘要:**

- 將預處理嵌入模型的**核心優點**：部署時無需單獨執行預處理——用戶只需傳入原始資料
- `adapt()` 類似 `fit()`，需要在訓練資料上呼叫（不能在測試集上）

---

---

[來源: ch13 | 類型: tutorial] ### 範例 6: TextVectorization 層

```python

---

[來源: ch13 | 類型: tutorial] # 示範文字資料
texts = [
    "台灣的半導體產業很強大",
    "機器學習改變了科技產業",
    "深度學習需要大量資料",
    "GPU 是深度學習的核心硬體",
]

---

[來源: ch13 | 類型: tutorial] # TextVectorization：字符/詞語 → 整數索引
text_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=50,            # 詞彙表大小
    output_mode="int",        # 輸出整數索引（也可以是 "tf-idf" 或 "binary"）
    output_sequence_length=10  # 固定序列長度（截斷或填充）
)

---

[來源: ch13 | 類型: tutorial] # 訓練詞彙表
text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(4))
print(f"詞彙表大小: {text_vectorizer.vocabulary_size()}")
print(f"前 10 個詞: {text_vectorizer.get_vocabulary()[:10]}")

---

[來源: ch13 | 類型: tutorial] # 向量化
vectorized = text_vectorizer(["機器學習改變了科技產業"])
print(f"向量化結果: {vectorized.numpy()}")

---

[來源: ch13 | 類型: tutorial] # 嵌入到模型中
vocab_size = text_vectorizer.vocabulary_size()
text_model = tf.keras.Sequential([
    text_vectorizer,
    tf.keras.layers.Embedding(vocab_size, 16, mask_zero=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
text_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

**🎯 重點摘要:**

- `output_mode="int"` 適合 RNN/Transformer；`"tf-idf"` 或 `"binary"` 適合傳統 ML 模型
- `mask_zero=True`：讓模型知道哪些位置是填充（padding）

---

---

[來源: ch13 | 類型: tutorial] ### 範例 7: 圖像增強管線

```python
from tensorflow import keras

---

[來源: ch13 | 類型: tutorial] # Keras 內建圖像增強層（只在訓練時啟用）
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),  # 隨機水平/垂直翻轉
    keras.layers.RandomRotation(0.1),    # 隨機旋轉 ±10%（以 2π 為單位）
    keras.layers.RandomZoom(0.1),        # 隨機縮放 ±10%
    keras.layers.RandomBrightness(0.1),  # 隨機調整亮度
    keras.layers.RandomContrast(0.1),    # 隨機調整對比度
    keras.layers.Rescaling(1./255)       # 像素值正規化到 [0, 1]
], name="data_augmentation")

---

[來源: ch13 | 類型: tutorial] # 整合圖像增強到模型（增強只在 training=True 時生效）
base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs, training=True)  # 訓練時增強，推論時跳過
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

model_img = keras.Model(inputs=inputs, outputs=outputs)
model_img.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
```

---

---

[來源: ch13 | 類型: tutorial] ### 範例 8: 測量管線效能

```python
import time

def measure_dataset_performance(dataset, n_batches=100):
    """測量資料管線每秒能處理多少批次"""
    start = time.time()
    for i, _ in enumerate(dataset.take(n_batches)):
        pass
    elapsed = time.time() - start
    return n_batches / elapsed

---

[來源: ch13 | 類型: tutorial] # 無優化 vs 有優化的管線比較
raw_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
optimized_dataset = (
    tf.data.Dataset.from_tensor_slices((X, y))
    .shuffle(1000)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .cache()         # 緩存到記憶體（第一個 epoch 後免費！）
    .prefetch(AUTOTUNE)
)

---

[來源: ch13 | 類型: tutorial] # .map(preprocess).cache().batch().prefetch()

---

[來源: ch13 | 類型: tutorial] # 這樣 cache 儲存的是預處理後的資料，避免重複計算
```

**🎯 重點摘要:**

- **效能優化黃金法則**：`shuffle → map → cache → batch → prefetch`
- `cache()` 在第一個 epoch 後將資料緩存，後續 epoch 的 I/O 幾乎降為零
- 使用 `tf.data.experimental.AutoShardPolicy` 在多 GPU 環境中自動分片

---

---

[來源: ch13 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: `shuffle()` 的 `buffer_size` 如何設定？**

A: 理想情況下設為資料集大小（完全隨機），但記憶體有限時可以設為 5× batch_size 或更大。關鍵原則：至少設為 batch_size 的幾倍，否則不同 batch 間可能高度相關。

**Q2: 應該在 `batch` 之前還是之後做 `map`？**

A: 通常在 `batch` 之前 `map`（對每個樣本單獨處理），因為更容易寫；但某些增強操作（如 `MixUp`）需要在 batch 後執行。`map` 之後加 `cache()` 效率最高。

**Q3: TFRecord vs TF Datasets（tfds）的選擇？**

A: TFRecord 適合自訂格式的大型資料集（完全控制）；`tensorflow_datasets`（tfds）提供 300+ 標準資料集，一行代碼即可載入。快速實驗用 `tfds`，生產系統用 TFRecord。

**Q4: 為什麼要將預處理層嵌入模型而非在管線中處理？**

A: 嵌入模型的預處理在**部署時自動執行**——只需傳入原始資料，模型內部處理所有預處理。如果在管線中處理，部署時需要額外的預處理代碼，增加出錯風險。

---

---

[來源: ch13 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #TensorFlow #tfdata #TFRecord #Keras #資料管線 #預處理 #TextVectorization #深度學習 #圖像增強 #程式設計 #教學 #DataScience #MachineLearning

---

[來源: ch14 | 類型: cheatsheet] # Ch14 速查表：Deep Computer Vision with CNNs

> **核心主旨**：CNN 的三大核心：卷積層、殘差連接、轉移學習 —— `Xception(include_top=False)` + 微調是視覺任務的最快起點。

---

---

[來源: ch14 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Conv2D | 用 kernel 掃描影像，提取局部特徵（邊緣、紋理） | 所有影像任務的基礎層 |
| Padding `"same"` | 補零使輸出尺寸 = $\lceil H/s \rceil$ | 保持特徵圖大小 |
| Padding `"valid"` | 不補零，輸出 = $\lfloor (H-k)/s \rfloor + 1$ | 精確控制輸出大小 |
| MaxPool2D | 取窗口最大值，下採樣 × 2 | 降低計算量、擴大感受野 |
| GlobalAvgPool2D | 每個通道取平均，輸出 `[channels]` | 分類頭前的特徵壓縮 |
| ResidualUnit | Skip Connection：$y = F(x) + x$，緩解梯度消失 | 深層網路（20層+） |
| Transfer Learning | 複用 ImageNet 預訓練特徵，凍結底層只訓練頂層 | 小資料集視覺任務 |
| Fine-tuning | 解凍部分底層，以低 lr 繼續訓練 | 轉移學習第二階段 |


---

[來源: ch14 | 類型: cheatsheet] Learning | 複用 ImageNet 預訓練特徵，凍結底層只訓練頂層 | 小資料集視覺任務 |
| Fine-tuning | 解凍部分底層，以低 lr 繼續訓練 | 轉移學習第二階段 |


---

---

[來源: ch14 | 類型: cheatsheet] | Keras API | 重點參數 | 用途 |
|-----------|---------|------|
| `tf.keras.layers.Conv2D` | `filters=64`, `kernel_size=3`, `padding="same"`, `activation="relu"` | 卷積層 |
| `tf.keras.layers.MaxPool2D` | `pool_size=2`, `strides=2` | 最大池化（預設減半） |
| `tf.keras.layers.GlobalAvgPool2D` | – | 全域平均池化 |
| `tf.keras.layers.SpatialDropout2D` | `rate=0.2` | 2D Dropout（整通道遮蔽） |
| `tf.keras.applications.Xception` | `weights="imagenet"`, `include_top=False` | 預訓練基底模型 |
| `tf.keras.applications.ResNet50` | 同上 | ResNet 基底模型 |
| `tf.keras.applications.xception.preprocess_input` | – | Xception 專屬前處理（縮放到 [-1,1]） |
| `tf.keras.applications.resnet.preprocess_input` | – | ResNet 專屬前處理 |
| `tf.keras.layers.RandomFlip` | `mode="horizontal"` | 資料增強：隨機翻轉 |
| `tf.keras.layers.RandomRotation` | `factor=0.05` | 資料增強：隨機旋轉 |


---

[來源: ch14 | 類型: cheatsheet] `mode="horizontal"` | 資料增強：隨機翻轉 |
| `tf.keras.layers.RandomRotation` | `factor=0.05` | 資料增強：隨機旋轉 |


---

---

[來源: ch14 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import tensorflow as tf
from functools import partial

---

[來源: ch14 | 類型: cheatsheet] # 基礎 CNN（Fashion MNIST）
DefaultConv2D = partial(tf.keras.layers.Conv2D,
    kernel_size=3, padding="same", activation="relu",
    kernel_initializer="he_normal")

---

[來源: ch14 | 類型: cheatsheet] rs.Conv2D,
    kernel_size=3, padding="same", activation="relu",
    kernel_initializer="he_normal")

model = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activati

---

[來源: ch14 | 類型: cheatsheet]  tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])

---

[來源: ch14 | 類型: cheatsheet] # 殘差單元（ResNet 核心元件）
class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        Conv = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=strides,
                       padding="same", kernel_initializer="he_normal", use_bias=False)
        self.main_layers = [
            Conv(filters), tf.keras.layers.BatchNormalization(), self.activation,
            Conv(filters), tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:  # 尺寸不同時，skip 也需要卷積對齊
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides=strides, padding="same",
                                        kernel_initializer="he_normal", use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]

---

[來源: ch14 | 類型: cheatsheet] zer="he_normal", use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]

def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

---

[來源: ch14 | 類型: cheatsheet] # 轉移學習（標準 4 步驟）
import tensorflow_datasets as tfds

---

[來源: ch14 | 類型: cheatsheet] # 步驟 1：載入並前處理資料
(train_ds, valid_ds), ds_info = tfds.load("tf_flowers",
    split=["train[:80%]", "train[80%:]"], as_supervised=True, with_info=True)
n_classes = ds_info.features["label"].num_classes

---

[來源: ch14 | 類型: cheatsheet] train[80%:]"], as_supervised=True, with_info=True)
n_classes = ds_info.features["label"].num_classes

preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(224, 224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=42),
    tf.keras.layers.RandomRotation(0.05, seed=42)
])

---

[來源: ch14 | 類型: cheatsheet] keras.layers.RandomFlip("horizontal", seed=42),
    tf.keras.layers.RandomRotation(0.05, seed=42)
])

train_set = train_ds.map(lambda x, y: (preprocess(x), y)).batch(32).prefetch(tf.data.AUTOTUNE)
valid_set = valid_ds.map(lambda x, y: (preprocess(x), y)).batch(32).prefetch(tf.data.AUTOTUNE)

---

[來源: ch14 | 類型: cheatsheet] # 步驟 2：建立模型
base_model = tf.keras.applications.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

---

[來源: ch14 | 類型: cheatsheet] # 步驟 3：凍結底層，訓練新頂層
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_top = model.fit(train_set, validation_data=valid_set, epochs=5)

---

[來源: ch14 | 類型: cheatsheet] # 步驟 4：Fine-tuning（解凍部分層，降低 lr）
for layer in base_model.layers[100:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.SGD(1e-4, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_ft = model.fit(train_set, validation_data=valid_set, epochs=5)
```

---

---

[來源: ch14 | 類型: cheatsheet] ## 4. 常見陷阱

- **`include_top=False` 移除了什麼**：移除了 `GlobalAvgPool + Dense(1000, softmax)`，保留所有卷積特徵提取層。
- **Fine-tuning 前必須重新 compile**：解凍層後 optimizer 狀態需要重置，且學習率必須降低（通常降低 10 倍）。
- **輸入尺寸要符合預訓練模型**：Xception / ResNet50 需要 224×224，使用 `Resizing` 層統一。
- **`preprocess_input` 不能省**：每個預訓練模型有自己的輸入值域，Xception 是 [-1,1]，ResNet 是 channel-wise 減均值，不正確的前處理會讓模型表現很差。
- **資料增強只在訓練時**：`model.fit()` 自動處理（`training=True`），`model.predict()` 不增強。

---

---

[來源: ch14 | 類型: cheatsheet] ## 5. 決策指南

```
輸出形狀公式：
├── valid padding: floor((H - k) / s) + 1
└── same padding:  ceil(H / s)

何時用預訓練模型？
└── 幾乎永遠（除非訓練資料 >> 100k 且與 ImageNet 差異極大）

底層凍結 vs 微調：
├── 資料 < 2k 且與 ImageNet 相似     → 只訓練頂層
├── 資料 2k-20k 且與 ImageNet 相似   → 凍結底部 80%，微調頂部 20%
├── 資料 > 20k 或與 ImageNet 差異大  → 微調更多層甚至全部
└── 微調學習率 = 初始訓練學習率 / 10

預訓練模型選擇（由快到慢）：
MobileNetV2 → EfficientNetB0 → ResNet50 → Xception → EfficientNetB7
```

---

[來源: ch14 | 類型: handout] # 課程講義：深度電腦視覺與卷積神經網路 (Chapter 14)

---

[來源: ch14 | 類型: handout] ## 導論

本章聚焦於深度電腦視覺的核心技術：**卷積神經網路（Convolutional Neural
Networks, CNNs）**。在機器學習體系中，CNN 是處理影像資料的基礎架構，
能夠從原始像素中自動學習局部特徵、空間結構與層次表示。

你將理解卷積 (Convolution) 運算的數學原理、池化 (Pooling) 的設計哲學、ResNet 殘差單元 (Residual Unit) 的動機，
以及如何將預訓練模型應用於轉移學習（Transfer Learning）與微調。

本章內容是後續目標檢測、語義分割與視覺 Transformer 的重要基礎。

---

---

[來源: ch14 | 類型: handout] ## 1. 卷積層 (Convolutional Layer, Conv)：從影像到特徵圖

---

[來源: ch14 | 類型: handout] ### 1.1 為什麼重要？

卷積層是 CNN 的基本構建模組。它透過**局部感受野（local receptive field）**
與**共享權重（weight sharing）**，將影像中的邊緣、紋理和形狀轉換為特徵圖
（feature maps），大幅降低參數數量並強化平移不變性。

相較於全連接層，卷積層在影像任務中有兩大優勢：

- **稀疏連接**：每個神經元只連接輸入的一個局部區域，減少參數量。
- **特徵可重用**：同一濾波器在影像各位置共享權重，能偵測同一特徵。

---

[來源: ch14 | 類型: handout] ### 1.2 輸出形狀公式

在卷積神經網路中，**卷積核（kernel）** 在輸入特徵圖上滑動，產生輸出特徵圖。這個過程會受到 **補零（padding）** 策略與 **步幅（stride）** 的影響，進而決定輸出尺寸。

首先，讓我們先明確公式中的符號：
*   $H$: 輸入特徵圖的高度（由於卷積運算通常在寬度 $W$ 方向也應用相同的邏輯，因此這些公式也適用於寬度）。
*   $k$: 卷積核的尺寸。
*   $s$: 卷積運算的步幅，即卷積核每次移動的像素數。

---

[來源: ch14 | 類型: handout] ### valid padding（不補零）

當我們採用 **valid padding** 時，表示在輸入特徵圖的邊界不進行任何補零操作。卷積核只會在其能夠*完全覆蓋輸入區域的位置*進行計算。這意味著卷積核在輸入特徵圖的邊緣部分無法進行運算，因此輸出特徵圖的尺寸通常會比輸入特徵圖小。其輸出高度（或寬度）的計算公式為：

$$
\text{output} = \left\lfloor \frac{H - k}{s} \right\rfloor + 1
$$

---

[來源: ch14 | 類型: handout] 尺寸通常會比輸入特徵圖小。其輸出高度（或寬度）的計算公式為：

$$
\text{output} = \left\lfloor \frac{H - k}{s} \right\rfloor + 1
$$

這個公式可以這樣理解：$H - k$ 代表在不考慮步幅的情況下，卷積核可以從最左上角移動到最右下角的「有效滑動範圍」。將此範圍除以步幅 $s$，再使用**地板函數 (floor function)** $ \lfloor \cdot \rfloor $ 向下取整，得到卷積核可以滑動的「步數」。最後加上 $1$，是因為起始位置也算一個輸出。這種方式能確保每個輸出像素都完全由原始輸入數據計算而來，不會引入補零帶來的額外資訊。

---

[來源: ch14 | 類型: handout] ### same padding（補零以維持尺寸）

相對地，**same padding** 的目標是透過在輸入特徵圖的周圍增加零值像素（補零），使得輸出特徵圖的空間尺寸能夠盡可能地與輸入特徵圖保持「相同」或按比例縮小。當步幅 $s=1$ 時，輸出尺寸將與輸入尺寸完全相同；當步幅 $s>1$ 時，輸出尺寸會約為輸入尺寸除以步幅。

對於最常見的步幅 $s=1$ 且卷積核尺寸 $k$ 為奇數的情況，在輸入特徵圖的每個空間維度（例如高度或寬度）上，每一側（頂/底或左/右）的補零量 $P$ 通常計算為：

$$
P = \frac{k-1}{2}
$$

這表示在每個空間維度上，總共會補上 $k-1$ 個零。例如，當卷積核尺寸 $k=3$ 時，每一側會補上 $(3-1)/2 = 1$ 個零。當卷積核尺寸 $k=7$ 時，每一側會補上 $(7-1)/2 = 3$ 個零。

---

[來源: ch14 | 類型: handout] 維度上，總共會補上 $k-1$ 個零。例如，當卷積核尺寸 $k=3$ 時，每一側會補上 $(3-1)/2 = 1$ 個零。當卷積核尺寸 $k=7$ 時，每一側會補上 $(7-1)/2 = 3$ 個零。

如果卷積核尺寸 $k$ 為偶數，或者步幅 $s>1$，補零的數量可能會不對稱（例如，某一側比另一側多一個像素），以精確達成目標輸出尺寸。然而，這些細節通常由深度學習框架自動處理，你只需要指定 `padding="same"` 即可。

其輸出高度（或寬度）的計算公式為：

$$
\text{output} = \left\lceil \frac{H}{s} \right\rceil
$$

---

[來源: ch14 | 類型: handout] `padding="same"` 即可。

其輸出高度（或寬度）的計算公式為：

$$
\text{output} = \left\lceil \frac{H}{s} \right\rceil
$$

這個公式相對簡潔，它直接將輸入高度 $H$ 除以步幅 $s$，並使用**天花板函數 (ceiling function)** $ \lceil \cdot \rceil $ 向上取整。這確保了即使 $H$ 無法被 $s$ 整除，也能夠透過足夠的補零來覆蓋整個輸入區域，使卷積核能夠完成所有的運算。在深度學習模型中，`same padding` 常用於保持特徵圖的空間尺寸，以便於構建更深層次的網路，避免過快地丟失邊緣資訊或使特徵圖尺寸歸零。

---

[來源: ch14 | 類型: handout] ### 1.3 核心代碼

這段程式碼示範如何在 TensorFlow/Keras 中使用 `Conv2D` 層，並比較 `valid` 與 `same` padding 以及步幅對輸出特徵圖大小的影響。

```python
import tensorflow as tf
from sklearn.datasets import load_sample_images

images = load_sample_images()["images"]
images = tf.keras.layers.CenterCrop(height=70, width=120)(images) # Shape (2, 70, 120, 3)
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)

---

[來源: ch14 | 類型: handout] # valid padding（預設）
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
fmaps = conv_layer(images)
print(f'Shape (valid padding): {fmaps.shape[1:3].as_list()}') # 輸出特徵圖的空間尺寸

---

[來源: ch14 | 類型: handout] # same padding
conv_layer_same = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="same")
fmaps_same = conv_layer_same(images)
print(f'Shape (same padding): {fmaps_same.shape[1:3].as_list()}')

---

[來源: ch14 | 類型: handout] # same + stride=2（下採樣）
conv_layer_stride = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="same", strides=2)
fmaps_stride = conv_layer_stride(images)
print(f'Shape (same padding, stride=2): {fmaps_stride.shape[1:3].as_list()}')
```

---

[來源: ch14 | 類型: handout] ### 1.4 手動濾波器示範

```python
import numpy as np

filters = np.zeros([7, 7, 3, 2])
filters[:, 3, :, 0] = 1   # 第 1 個濾波器：偵測垂直線
filters[3, :, :, 1] = 1   # 第 2 個濾波器：偵測水平線
biases = tf.zeros([2])
fmaps = tf.nn.conv2d(
    images, filters, strides=1, padding="SAME") + biases
print(f'Shape (manual filters): {fmaps.shape[1:3].as_list()}')
```

---

[來源: ch14 | 類型: handout] ### 補充練習 1

- **理論題**：比較 `padding="same"` 與 `padding="valid"` 的輸出形狀與
  資訊保留差異。為何深度 CNN 中常用 `same`？
- **實作題**：使用 7×7 卷積層，分別列印 `strides=1` 和 `strides=2`
  的輸出大小，說明改變步幅對計算量與感受野(receptive field)的影響。

---

---

[來源: ch14 | 類型: handout] ## 2. 池化層 (Pooling Layer)：下採樣(Downsampling)與特徵聚合(Feature Aggregation)

---

[來源: ch14 | 類型: handout] ### 2.1 為什麼重要？

池化層用於減少空間維度、擴大感受野並提取不變特徵，
降低計算成本同時保留最顯著的局部特徵。

---

[來源: ch14 | 類型: handout] ### 2.2 如何運作？

- **最大池化（Max Pooling）**：在池化窗口中選取最大值，保留最強激活。
- **深度池化（Depth-wise Pooling）**：沿通道方向進行最大化，壓縮通道維度。
- **全域平均池化（Global Average Pooling）**：對每個通道的空間維度(高度和寬度)取平均，
  將特徵圖壓縮為單一標量值，常接於分類層前。換句話說，它把輸入特徵圖從`[height, width, channels]` 壓縮成 `[channels]`。
    - 在 CNN 中，全域平均池化常用於卷積層與**分類層**之間，因為它能夠：
        - 去除空間資訊後保留每個通道的整體響應強度
        - 減少參數數量，避免 Flatten() 後過度擴展
        - 提升模型對平移的魯棒性，適合作為全連接層前的最後一步

---

[來源: ch14 | 類型: handout] ### 2.3 核心代碼

```python
print(f'Input shape: {images.shape[1:3].as_list()}')

---

[來源: ch14 | 類型: handout] # 最大池化
max_pool = tf.keras.layers.MaxPool2D(pool_size=2)
output = max_pool(images)
print(f'Shape (max pooling): {output.shape[1:3].as_list()}')

---

[來源: ch14 | 類型: handout] # 全域平均池化
print(f'Input shape: {images.shape.as_list()}')
global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
global_avg_pool(images)
print(f'Shape (global average pooling): {global_avg_pool(images).shape.as_list()}')

---

[來源: ch14 | 類型: handout] # 等價 Lambda 寫法
global_avg_pool_lam = tf.keras.layers.Lambda(
    lambda X: tf.reduce_mean(X, axis=[1, 2]))
global_avg_pool_lam(images)
print(f'Shape (global average pooling, Lambda): {global_avg_pool_lam(images).shape.as_list()}')
```

---

[來源: ch14 | 類型: handout] ```python
class DepthPool(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        shape = tf.shape(inputs)
        groups = shape[-1] // self.pool_size
        new_shape = tf.concat(
            [shape[:-1], [groups, self.pool_size]], axis=0)
        return tf.reduce_max(
            tf.reshape(inputs, new_shape), axis=-1)
```

---

[來源: ch14 | 類型: handout] ### 補充練習 2

- **理論題**：為何全域平均池化常用於 CNN 的最後一層？它相比
  `Flatten()` + `Dense()` 有哪些優缺點？
- **實作題**：使用 `DepthPool(pool_size=3)` 對隨機特徵圖進行運算，
  並與 `tf.nn.max_pool` 的結果比較，確認是否一致。

---

---

[來源: ch14 | 類型: handout] ## 3. CNN 架構：LeNet-5 與 ResNet 殘差單元

---

[來源: ch14 | 類型: handout] ### 3.1 為什麼重要？

CNN 架構決定模型的表示能力與訓練穩定性。本節從經典的 LeNet-5 出發，
進而介紹 Fashion MNIST 範例 CNN，再到 ResNet 殘差單元（Residual Unit），
反映了視覺模型從淺到深的演進。

---

[來源: ch14 | 類型: handout] ### 3.2 LeNet-5 架構

LeNet-5 是最早成功應用於手寫數字辨識的 CNN，架構如下：

---

[來源: ch14 | 類型: handout] | 層   | 類型       | 特徵圖 | 大小       | 核尺寸  | 步幅 | 活化函數 |
|------|-----------|-------|-----------|--------|------|---------|
| In   | 輸入       | 1     | 32×32     | –      | –    | –       |
| C1   | 卷積       | 6     | 28×28     | 5×5    | 1    | tanh    |
| S2   | 平均池化   | 6     | 14×14     | 2×2    | 2    | tanh    |
| C3   | 卷積       | 16    | 10×10     | 5×5    | 1    | tanh    |
| S4   | 平均池化   | 16    | 5×5       | 2×2    | 2    | tanh    |
| C5   | 卷積       | 120   | 1×1       | 5×5    | 1    | tanh    |
| F6   | 全連接     | –     | 84        | –      | –    | tanh    |
| Out  | 全連接     | –     | 10        | –      | –    | RBF     |

---

[來源: ch14 | 類型: handout] ```python
from functools import partial

DefaultConv2D = partial(
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer="he_normal")

model = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7,
                  input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

model.summary()
```

---

[來源: ch14 | 類型: handout] ### 3.4 ResNet 殘差單元與殘差連接

**殘差連接（skip connection）** 讓輸入直接加到主幹輸出，
緩解深度網路的退化（degradation）問題，使梯度更容易傳播。

---

[來源: ch14 | 類型: handout] ```python
import tensorflow as tf
from functools import partial
DefaultConv2D = partial(
    tf.keras.layers.Conv2D,
    kernel_size=3, strides=1,
    padding="same",
    kernel_initializer="he_normal",
    use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", skip_connection=True, **kwargs):
        super().__init__(**kwargs)
        self.skip_connection = skip_connection # 新增開關參數
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        
        # 實作捷徑路徑
        if self.skip_connection:
            skip_Z = inputs
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)
        else:
            skip_Z = tf.zeros_like(Z) # 若關閉則設為全零張量

        return self.activation(Z + skip_Z)
```

---

[來源: ch14 | 類型: handout] ```python
model = tf.keras.Sequential([
    DefaultConv2D(64, kernel_size=7, strides=2,
                  input_shape=[224, 224, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                              padding="same"),
])
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides, skip_connection=True))
    prev_filters = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

---

[來源: ch14 | 類型: handout] ### 補充練習 3

- **理論題**：為何殘差連接能讓更深的網路比純粹堆疊卷積層更容易訓練？
  請說明殘差單元的梯度流動機制。
- **實作題**：在 `ResidualUnit.call()` 中移除捷徑路徑（令
  `skip_Z = 0`），比較訓練 Fashion MNIST 前後的驗證準確度差異。

---

[來源: ch14 | 類型: handout] ### 深度解析：殘差連接的優勢

1. **參數重構（Re-parameterization）**：
   傳統網路旨在學習一個潛在映射 $H(x)$。而殘差單元將學習目標重構為 $y = F(x, \{W_i\}) + x$。若將最終映射定義為 $H(x) = F(x) + x$，則學習「殘差」$F(x) = H(x) - x$ 在優化上更具優勢。當底層特徵已足夠時，$F(x)$ 僅需逼近零，讓單元退化為**恒等映射（Identity Mapping）**，有效解決了深度網路中的「退化（Degradation）」問題。

---

[來源: ch14 | 類型: handout] x$ 在優化上更具優勢。當底層特徵已足夠時，$F(x)$ 僅需逼近零，讓單元退化為**恒等映射（Identity Mapping）**，有效解決了深度網路中的「退化（Degradation）」問題。

2. **梯度傳播的資訊高速公路（Grandient Highway）**：
   根據連鎖律，損失函數 $L$ 對輸入 $x$ 的梯度可表示為：
   $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left( \frac{\partial F}{\partial x} + 1 \right)$$
   這項公式揭示了關鍵機制：梯度項中的「$+ 1$」確保了即使主

---

[來源: ch14 | 類型: handout] artial y} \cdot \left( \frac{\partial F}{\partial x} + 1 \right)$$
   這項公式揭示了關鍵機制：梯度項中的「$+ 1$」確保了即使主幹路徑 $\frac{\partial F}{\partial x}$ 的梯度極小或消失，訊號仍能經由跳接路徑（Skip Connection）毫無阻礙地流向淺層。這種**梯度加法（Gradient Addition）** 性質打破了深度網路梯度連乘導致的指數級衰減，讓訓練數百層的網路成為可能。

```python
import tensorflow as tf

---

[來源: ch14 | 類型: handout] # load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train[:-5000], X_train[-5000:]
y_train, y_valid = y_train[:-5000], y_train[-5000:]
X_train = X_train[..., tf.newaxis] / 255.0
X_valid = X_valid[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).prefetch(tf.data.AUTOTUNE)
valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(64).prefetch(tf.data.AUTOTUNE)
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.AUTOTUNE)

---

[來源: ch14 | 類型: handout] # preprocess = tf.keras.layers.Resizing(height=224, width=224)

---

[來源: ch14 | 類型: handout] # build ResNet-34 with and without skip connections
def build_resnet(skip_connection=True):
    model = tf.keras.Sequential([
        # preprocess,
        # DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 1]),
        DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[28, 28, 1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
    ])
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides, skip_connection=skip_connection))
        prev_filters = filters

---

[來源: ch14 | 類型: handout] idualUnit(filters, strides=strides, skip_connection=skip_connection))
        prev_filters = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

---

[來源: ch14 | 類型: handout] # 建立實驗模型
model_with_skip = build_resnet(skip_connection=True)
model_without_skip = build_resnet(skip_connection=False)

---

[來源: ch14 | 類型: handout] # 編譯與訓練模型
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model_with_skip.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
model_without_skip.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
history_with_skip = model_with_skip.fit(train_set, validation_data=valid_set, epochs=10)
history_without_skip = model_without_skip.fit(train_set, validation_data=valid_set, epochs=10)

---

[來源: ch14 | 類型: handout] s=10)
history_without_skip = model_without_skip.fit(train_set, validation_data=valid_set, epochs=10)

import matplotlib.pyplot as plt
plt.plot(history_with_skip.history['val_accuracy'], label='With Skip Connection')
plt.plot(history_without_skip.history['val_accuracy'], label='Without Skip Connection')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
```

---

[來源: ch14 | 類型: handout] ## 4. 預訓練模型 (Pre-trained Models) 與轉移學習 (Transfer Learning)

---

[來源: ch14 | 類型: handout] ### 4.1 為什麼重要？

轉移學習（Transfer Learning）可將大型資料集（如 ImageNet）
上預訓練的特徵提取器，*應用到新的小資料集*，節省訓練時間並提升效能。
在資料量有限的視覺任務中，這是最常採用的實務策略。
在`tf.keras.applications` 模組中，提供了多種預訓練模型（如 VGG、ResNet、Xception 等），這些模型在 ImageNet 上訓練，能夠提取通用的視覺特徵，適用於各種下游任務。
> ImageNet 是一個包含超過 1400 萬張標註圖像的資料集，涵蓋了 1000 個類別。預訓練模型在 ImageNet 上學習到的特徵（如邊緣、紋理、形狀等）具有高度的泛化 (generalization) 能力，能夠在其他視覺任務中有效地提取有用資訊，即使這些任務的資料集規模較小。

---

[來源: ch14 | 類型: handout] ### 4.2 轉移學習流程

1. **載入基底模型**：`include_top=False` 移除原始分類器。
2. **加入自訂輸出層**：`GlobalAveragePooling2D()` + `Dense(n_classes)`。
3. **凍結基底層**：`layer.trainable = False`，僅訓練頂部新層。
4. **微調（Fine-tuning）**：頂層收斂後，解凍部分基底層並以低學習率繼續訓練。

---

[來源: ch14 | 類型: handout] ### 4.3 核心代碼

```python
n_classes = 10  # 有 10 個類別的 Fashion MNIST 資料集

---

[來源: ch14 | 類型: handout] # 1. 載入 Xception 基底模型，排除頂部原有的分類層 (GlobalAvgPool + Dense)
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=False)

---

[來源: ch14 | 類型: handout] # 2. 加入自訂分類頭部：全域平均池化將 3D 特徵圖壓縮為 1D 向量
avg = tf.keras.layers.GlobalAveragePooling2D()(
    base_model.output)
output = tf.keras.layers.Dense(
    n_classes, activation="softmax")(avg)

---

[來源: ch14 | 類型: handout] # 3. 建立最終模型：定義輸入與自訂輸出
model = tf.keras.Model(
    inputs=base_model.input, outputs=output)

---

[來源: ch14 | 類型: handout] # 4. 凍結基底層：防止預訓練的權重在初始訓練階段被破壞
for layer in base_model.layers:
    layer.trainable = False

---

[來源: ch14 | 類型: handout] # 5. 編譯模型：使用帶動量的 SGD，並設定適合分類的損失函數
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.1, momentum=0.9)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

---

[來源: ch14 | 類型: handout] # 6. 訓練新層：通常只需少數 epoch 即可使頂部分類器收斂
history = model.fit(train_set, validation_data=valid_set, epochs=3)
```

檢視基底模型與頂部分類器的結構，確認 `include_top=False` 的效果：

```python
xception_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=True)

---

[來源: ch14 | 類型: handout] # 列印 Xception 模型的前 10 層名稱與輸出形狀，了解基底模型結構
for n in range(10):
    print(f"Layer {n}: its name is {xception_model.layers[n].name} with output {xception_model.layers[n].output}")

---

[來源: ch14 | 類型: handout] # 列印 Xception 模型的後 10 層名稱與輸出形狀，確認頂部分類器結構
for n in range(-1, -11, -1):
    print(f"Layer {n}: its name is {xception_model.layers[n].name} with output {xception_model.layers[n].output}")

---

[來源: ch14 | 類型: handout] # 同樣的列印基底模型（不包含頂部分類器）的前後 10 層，確認 `include_top=False` 的效果
for n in range(10):
    print(f"Layer {n}: its name is {base_model.layers[n].name} with output {base_model.layers[n].output}")

---

[來源: ch14 | 類型: handout] # 列印基底模型的後 10 層，確認頂部分類器已被移除
for n in range(-1, -11, -1):
    print(f"Layer {n}: its name is {base_model.layers[n].name} with output {base_model.layers[n].output}")

```

>Xception 模型結構圖：

![Xception 模型結構圖](xception_model_structure.png)

>精準地列印每層的名稱與類型，確認基底模型與頂部分類器的結構差異：

```

python
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.__class__.__name__)

---

[來源: ch14 | 類型: handout] hon
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.__class__.__name__)

for i, layer in enumerate(xception_model.layers):
    print(i, layer.name, layer.__class__.__name__)
```

---

[來源: ch14 | 類型: handout] ### 4.4 資料前處理與資料增強

由於預訓練模型對輸入影像有特定的格式規範（如尺寸與數值範圍），我們將介紹如何使用 Keras Preprocessing Layers 來建立高效的影像處理流水線，包括：
- **尺寸調整 (Resizing)**：將影像統一縮放至模型預期的維度。
- **預處理函數 (Preprocessing Function)**：執行模型專屬的數值標準化（例如將像素縮放至 [−1,1]）。
- **資料增強 (Data Augmentation)**：透過隨機翻轉、旋轉等技巧人工擴張訓練集，以強化模型的泛化能力並防止過度擬合（overfitting）。

這對於處理像 Fashion MNIST 這種小尺寸、單通道的資料集尤其關鍵，因為我們必須先將其轉換為模型可識別的格式，才能有效發揮轉移學習的威力。

---

[來源: ch14 | 類型: handout] 化能力並防止過度擬合（overfitting）。

這對於處理像 Fashion MNIST 這種小尺寸、單通道的資料集尤其關鍵，因為我們必須先將其轉換為模型可識別的格式，才能有效發揮轉移學習的威力。

```python
batch_size = 32
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(
        height=224, width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(
        tf.keras.applications.xception.preprocess_input)
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.05, seed=42),
    tf.keras.layers.RandomContrast(factor=0.2, seed=42)
])
```

---

[來源: ch14 | 類型: handout] ### 4.5 微調範例

解凍 (unfreeze) 模型的部分層，並以較低的學習率繼續訓練，以微調預訓練權重適應新任務。

---

[來源: ch14 | 類型: handout] ```python
print(f"模型的總層數: {len(model.layers)}")

for layer in model.layers[56:]:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set,
                    epochs=10)
```

```python

---

[來源: ch14 | 類型: handout] # 列印模型的每一層名稱、類型與是否可訓練，確認微調設定
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({layer.__class__.__name__}) - Trainable: {layer.trainable}")
```

---

[來源: ch14 | 類型: handout] ### 4.6 分類與定位（多輸出模型）

實際應用中，我們不僅需要知道物體的類別，還需要精確標定其在影像中的位置，這就是「影像定位」（Image Localization）。
透過多輸出模型，我們可以同時訓練網路進行兩個任務：一個負責預測類別（分類），另一個負責預測物體的 **邊界框 (Bounding Box)** 座標（定位）。
這不僅節省了重新計算特徵的時間，更為隨後更複雜的 **「物件偵測」（Object Detection）** 奠定了基礎。

```python
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=False)

---

[來源: ch14 | 類型: handout] hon
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=False)

avg = tf.keras.layers.GlobalAveragePooling2D()(
    base_model.output)

---

[來源: ch14 | 類型: handout] # 分類輸出：10 類別的 softmax 預測
class_output = tf.keras.layers.Dense(
    n_classes, activation="softmax")(avg)

---

[來源: ch14 | 類型: handout] # 定位輸出：4 個座標值的線性預測（x_min, y_min, x_max, y_max）
loc_output = tf.keras.layers.Dense(4)(avg)

---

[來源: ch14 | 類型: handout] # 建立多輸出模型：同時輸出分類與定位結果
model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[class_output, loc_output])

---

[來源: ch14 | 類型: handout] 分類與定位結果
model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[class_output, loc_output])

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9)
model.compile(
    loss=["sparse_categorical_crossentropy", "mse"], # 分別為分類與定位的損失函數
    loss_weights=[0.8, 0.2], # 給予分類損失較高的權重，因為它通常更重要且更難學習
    optimizer=optimizer,
    metrics=["accuracy", "mse"]) # 分別評估分類準確度與定位誤差
```

---

[來源: ch14 | 類型: handout] ### 補充練習 4

- **理論題**：解釋為何轉移學習階段通常先凍結基底模型再訓練新頂層。
  微調時為何改用更低的學習率？
- **實作題**：使用
  `tf.keras.applications.ResNet50(weights="imagenet", include_top=False)`
  作為基底模型，以 `tf_flowers` 資料集（224×224）建立分類器並訓練。

```python
import tensorflow as tf
import tensorflow_datasets as tfds

---

[來源: ch14 | 類型: handout] # 載入 tf_flowers 資料集
(train_ds, valid_ds), ds_info = tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True)

---

[來源: ch14 | 類型: handout] # 定義資料前處理與增強流水線
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(
        tf.keras.applications.resnet.preprocess_input)
])
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.05, seed=42),
    tf.keras.layers.RandomContrast(factor=0.2, seed=42)
])

---

[來源: ch14 | 類型: handout] # 建立轉移學習模型
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(ds_info.features["label"].num_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

---

[來源: ch14 | 類型: handout] # 凍結基底模型
for layer in base_model.layers:
    layer.trainable = False

---

[來源: ch14 | 類型: handout] # 編譯模型
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

---

[來源: ch14 | 類型: handout] # 訓練資料集：先 map (調整尺寸與增強)，再 batch
training_data = train_ds.map(
    lambda x, y: (
        tf.squeeze(data_augmentation(preprocess(tf.expand_dims(x, 0)), training=True), 0), 
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
).batch(32).prefetch(tf.data.AUTOTUNE)

---

[來源: ch14 | 類型: handout] # 驗證資料集：同樣先 map，再 batch
validation_data = valid_ds.map(
    lambda x, y: (
        tf.squeeze(preprocess(tf.expand_dims(x, 0)), 0), 
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
).batch(32).prefetch(tf.data.AUTOTUNE)

---

[來源: ch14 | 類型: handout] # 執行訓練
history = model.fit(
    training_data,
    validation_data=validation_data,
    epochs=5)
```

---

---

[來源: ch14 | 類型: handout] ## 結論

本章提供了電腦視覺的核心工具：**卷積層、池化層、LeNet-5 與 ResNet
殘差架構，以及轉移學習技術**。你應該能理解 CNN 為何在影像任務中
比全連接網路更有效，並能將預訓練特徵提取器應用於新的分類任務。

接下來的章節可進一步延伸至**目標檢測、語義分割與生成對抗網路**，
也可探索更高階的視覺 Transformer（ViT）與弱監督學習方法。

---

---

[來源: ch14 | 類型: handout] ## 課後作業

1. **實作題**：依據第 3.3 節的 Fashion MNIST CNN 範例，
   建立自己的卷積神經網路，並在 `X_train`／`X_valid`／`X_test`
   分割後完成訓練。請記錄：
   - 使用 `Conv2D` 與 `MaxPool2D` 的層結構
   - 每次訓練後的驗證準確度
   - 有無 `Dropout` 的比較結果

```python
import tensorflow as tf

---

[來源: ch14 | 類型: handout] # Define parameters
BATCH_SIZE = 64
DROPOUT_RATE = 0.2
EPOCHS = 30

---

[來源: ch14 | 類型: handout] # load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train[:-5000], X_train[-5000:]
y_train, y_valid = y_train[:-5000], y_train[-5000:]
X_train = X_train[..., tf.newaxis] / 255.0
X_valid = X_valid[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

---

[來源: ch14 | 類型: handout] # 建立 CNN 模型
model_with_spatial_dropout = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1/255.0, input_shape=[28, 28, 1]),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.SpatialDropout2D(DROPOUT_RATE),  # 在卷積層後使用 SpatialDropout2D
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.SpatialDropout2D(DROPOUT_RATE),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
])
model_without_dropout = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1/255.0, input_shape=[28, 28, 1]),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

---

[來源: ch14 | 類型: handout] # 編譯模型
model_with_spatial_dropout.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"])

model_without_dropout.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"])

---

[來源: ch14 | 類型: handout] # 訓練模型
print("訓練帶有 SpatialDropout2D 的模型...")
history_with_spatial_dropout = model_with_spatial_dropout.fit(train_set, validation_data=valid_set, epochs=EPOCHS)
print("\n訓練不帶 Dropout 的模型...")
history_without_dropout = model_without_dropout.fit(train_set, validation_data=valid_set, epochs=EPOCHS)

---

[來源: ch14 | 類型: handout] ory_without_dropout = model_without_dropout.fit(train_set, validation_data=valid_set, epochs=EPOCHS)

import matplotlib.pyplot as plt
plt.plot(history_with_spatial_dropout.history['accuracy'], 'r--', label='With SpatialDropout2D (training)')
plt.plot(history_without_dropout.history['accuracy'], 'b--', label='Without Dropout (training)')
plt.plot(history_with_spatial_dropout.history['val_accuracy']

---

[來源: ch14 | 類型: handout] ', label='Without Dropout (training)')
plt.plot(history_with_spatial_dropout.history['val_accuracy'], 'r', label='With SpatialDropout2D (validation)')
plt.plot(history_without_dropout.history['val_accuracy'], 'b', label='Without Dropout (validation)')
plt.title('Training and Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

---

[來源: ch14 | 類型: handout] Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

```

2. **轉移學習題**：使用
   `tf.keras.applications.Xception(include_top=False)`
   對 `tf_flowers` 資料集進行轉移學習。請完成以下步驟：
   - 影像尺寸調整為 224×224
   - 使用 `xception.preprocess_input` 處理輸入
   - 先凍結基底模型訓練新頂層，再解凍部分基底層進行微調
   - 比較微調前後的驗證準確度變化

```

python
import tensorflow as tf
import tensorflow_datasets as tfds

---

[來源: ch14 | 類型: handout] # 0. Define parameters
BATCH_SIZE = 32
UNFREEZE_FROM_LAYER = 100
EPOCHS_TOP = 5
EPOCHS_FINE_TUNE = 5
SPLIT_RATIO = 0.8

---

[來源: ch14 | 類型: handout] # 1. 載入 tf_flowers 資料集
(train_ds, valid_ds), ds_info = tfds.load(
    "tf_flowers",
    split=[f"train[:{int(SPLIT_RATIO*100)}%]", f"train[{int(SPLIT_RATIO*100)}%:]"],
    as_supervised=True,
    with_info=True)

n_classes = ds_info.features["label"].num_classes

---

[來源: ch14 | 類型: handout] # 2. 定義前處理流水線 (尺寸調整與 Xception 專屬預處理)
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])

def preprocess_fn(image, label):
    return preprocess(image), label

---

[來源: ch14 | 類型: handout] .xception.preprocess_input)
])

def preprocess_fn(image, label):
    return preprocess(image), label

train_set = train_ds.map(preprocess_fn).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_set = valid_ds.map(preprocess_fn).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

---

[來源: ch14 | 類型: handout] # 3. 建立轉移學習模型
base_model = tf.keras.applications.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

---

[來源: ch14 | 類型: handout] # 4. 第一階段：凍結基底模型權重，僅訓練新加入的頂層
for layer in base_model.layers:
    layer.trainable = False

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("--- 階段 1：訓練新頂層 ---")
history_top = model.fit(train_set, validation_data=valid_set, epochs=EPOCHS_TOP)

---

[來源: ch14 | 類型: handout] # 5. 第二階段：解凍部分基底層進行微調 (Fine-tuning)

---

[來源: ch14 | 類型: handout] # 解凍最後端的部分層 (例如從第 100 層開始)
for layer in base_model.layers[UNFREEZE_FROM_LAYER:]:
    layer.trainable = True

---

[來源: ch14 | 類型: handout] # 微調時必須調低學習率，避免破壞已訓練好的特徵
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("\n--- 階段 2：微調基底模型 ---")
history_fine_tune = model.fit(train_set, validation_data=valid_set, epochs=EPOCHS_FINE_TUNE)

---

[來源: ch14 | 類型: handout] # 6. 比較準確度
val_acc_before = history_top.history["val_accuracy"][-1]
val_acc_after = history_fine_tune.history["val_accuracy"][-1]
print(f"\n微調前驗證準確度: {val_acc_before:.4f}")
print(f"微調後驗證準確度: {val_acc_after:.4f}")
```

---

[來源: ch14 | 類型: tutorial] [標題: 深度電腦視覺與 CNN 實作教學｜TensorFlow Keras | 描述: 從卷積層、池化層、ResNet 殘差單元到轉移學習，解析第 14 章 CNN 核心實作。 | 關鍵字: CNN, 卷積神經網路, 深度學習, TensorFlow, Keras, 轉移學習, ResNet]
# 深度電腦視覺與卷積神經網路 (CNN) 實作教學

這份教學基於《Hands-On Machine Learning》第 3 版第 14 章，帶你從卷積層
與池化層的基本運作開始，進而學習 CNN 架構、ResNet 殘差單元，
並應用預訓練模型進行轉移學習 (Transfer Learning) 和微調。

---

[來源: ch14 | 類型: tutorial] ## 本文目錄

- 🎯 [關鍵重點](#key-takeaways)
- 🧠 [卷積層 (Convolutional Layer) 與特徵圖](#conv-layer)
- 📉 [池化層 (Pooling Layer) 與全域平均池化](#pooling-layer)
- 👗 [Fashion MNIST CNN 範例](#fashion-mnist-cnn)
- 🔗 [ResNet 殘差單元與深度架構](#resnet)
- 🚀 [預訓練模型與轉移學習 (Transfer Learning)](#transfer-learning)
- 🧩 [進階微調與多輸出分類](#fine-tuning)
- ❓ [常見問答](#faq)
- 🏷️ [推薦標籤](#hashtags)

---

[來源: ch14 | 類型: tutorial] ##  關鍵重點

- 卷積層 (Convolutional Layer) 能從影像中提取局部特徵，並透過共享
  權重大幅降低參數量。
- `padding="same"` 與步幅 (stride) 影響特徵圖 (feature map) 的空間大小。
- 池化層 (Pooling Layer) 可進行下採樣、擴大感受野、並減少計算成本。
- ResNet 的殘差連接 (skip connection) 讓深層網路更容易訓練。
- 轉移學習 (Transfer Learning) 可將預訓練特徵快速應用到新任務。

---

[來源: ch14 | 類型: tutorial] ##  卷積層 (Convolutional Layer) 與特徵圖

卷積層是 CNN 的核心。它透過濾波器 (filter) 在影像上滑動，對局部區域做
加權求和，產生特徵圖 (feature map)。這個過程可以理解為「局部感受野
(receptive field)」加上「權重共享 (weight sharing)」的組合。

---

[來源: ch14 | 類型: tutorial] ### 1. 讀取範例影像與標準化

```python
from sklearn.datasets import load_sample_images
import tensorflow as tf

images = load_sample_images()["images"]
images = tf.keras.layers.CenterCrop(height=70, width=120)(images)
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)
```

✅ 程式碼逐行解析

1. `load_sample_images()`: 載入 Scikit-Learn 的內建範例影像資料集。
2. `import tensorflow as tf`: 匯入 TensorFlow，用於建構 CNN 層。
3. `["images"]`: 取得內建的兩張示例影像。
4. `CenterCrop(...)`: 將影像裁剪為 70×120，標準化後續卷積示範。
5. `Rescaling(scale=1 / 255)`: 將像素值縮放到 [0, 1]，提高訓練穩定性。

🎯 重點摘要

- 先行資料預處理可保持模型輸入一致性。
- 將像素歸一化有助於梯度穩定。
- 中心裁剪是視覺實作中常見的尺寸標準化方式。

---

[來源: ch14 | 類型: tutorial] ### 2. 卷積層基本實作

```python
tf.random.set_seed(42)
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
fmaps = conv_layer(images)
```

✅ 程式碼逐行解析

1. `tf.random.set_seed(42)`: 設定隨機種子，使範例可重現。
2. `Conv2D(filters=32, kernel_size=7)`: 建立 32 個 7×7 的卷積核 (kernel)。
3. `fmaps = conv_layer(images)`: 將卷積層套用到影像，得到輸出特徵圖。

🎯 重點摘要

- 32 個濾波器會產生 32 張特徵圖。
- 卷積核大小影響可偵測的局部結構。
- 預設 `padding="valid"` 會讓輸出尺寸縮小。

---

[來源: ch14 | 類型: tutorial] ### 3. Padding 與 Stride 對輸出大小的影響

```python
conv_layer_same = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="same")
fmaps_same = conv_layer_same(images)

conv_layer_stride = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="same", strides=2)
fmaps_stride = conv_layer_stride(images)
```

---

[來源: ch14 | 類型: tutorial]   filters=32, kernel_size=7, padding="same", strides=2)
fmaps_stride = conv_layer_stride(images)
```

```python
import numpy as np

def conv_output_size(input_size, kernel_size, strides=1, padding="valid"):
    if padding == "valid":
        z = input_size - kernel_size + strides
        output_size = z // strides
        num_ignored = z % strides
        return output_size, num_ignored
    else:
        output_size = (input_size - 1) // strides + 1
        num_padded = (output_size - 1) * strides + kernel_size - input_size
        return output_size, num_padded

conv_output_size(
    np.array([70, 120]), kernel_size=7, strides=2, padding="same")
```

---

[來源: ch14 | 類型: tutorial] num_padded

conv_output_size(
    np.array([70, 120]), kernel_size=7, strides=2, padding="same")
```

✅ 程式碼逐行解析

1. `padding="same"`: 在輸入邊緣補零，輸出與輸入空間尺寸相同（步幅為 1）。
2. `strides=2`: 濾波器每次移動 2 個像素，輸出尺寸變為原來的一半。
3. `conv_output_size(...)`: 自訂函數計算卷積後輸出大小。
4. `num_ignored` / `num_padded`: 分別表示被捨去或補零的列/欄數。

🎯 重點摘要

- `same` padding 適合多層卷積網路，能保留空間大小。
- stride 越大，輸出越小，計算成本越低。
- 了解輸出尺寸公式有助於設計合理的 CNN 結構。

---

[來源: ch14 | 類型: tutorial] ### 4. 手動濾波器示範：垂直與水平線檢測

```python
filters = np.zeros([7, 7, 3, 2])
filters[:, 3, :, 0] = 1
filters[3, :, :, 1] = 1
biases = tf.zeros([2])
fmaps = tf.nn.conv2d(
    images, filters, strides=1, padding="SAME") + biases
```

✅ 程式碼逐行解析

1. `np.zeros([7, 7, 3, 2])`: 建立兩個 7×7 濾波器，輸入通道為 3。
2. `filters[:, 3, :, 0] = 1`: 第一個濾波器偵測垂直線。
3. `filters[3, :, :, 1] = 1`: 第二個濾波器偵測水平線。
4. `tf.nn.conv2d(...)`: 使用低階卷積運算直接計算輸出。
5. `+ biases`: 加上零偏差，保持輸出純粹反映濾波器響應。

🎯 重點摘要

- CNN 內部濾波器可以自動學習邊緣與線條特徵。
- 手動設計濾波器有助於理解 CNN 的特徵提取機制。
- `padding="SAME"` 顯示填充對邊緣響應的影響。

---

[來源: ch14 | 類型: tutorial] ##  池化層 (Pooling Layer) 與全域平均池化

池化層負責下採樣與特徵聚合，是 CNN 中常見的空間壓縮方法。

---

[來源: ch14 | 類型: tutorial] ### 1. 最大池化 (Max Pooling)

```python
max_pool = tf.keras.layers.MaxPool2D(pool_size=2)
output = max_pool(images)
```

✅ 程式碼逐行解析

1. `MaxPool2D(pool_size=2)`: 設定 2×2 的池化窗口。
2. `output = max_pool(images)`: 對每個 2×2 區塊取最大值，完成下採樣。

🎯 重點摘要

- 最大池化保留最強激活值，適合保留邊緣與紋理特徵。
- 池化可減少參數與運算量，並提升平移不變性 (translation invariance)。

---

[來源: ch14 | 類型: tutorial] ### 2. 深度池化 (Depth-wise Pooling)

```python
class DepthPool(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        shape = tf.shape(inputs)
        groups = shape[-1] // self.pool_size
        new_shape = tf.concat(
            [shape[:-1], [groups, self.pool_size]], axis=0)
        return tf.reduce_max(tf.reshape(inputs, new_shape), axis=-1)
```

✅ 程式碼逐行解析

---

[來源: ch14 | 類型: tutorial] _size]], axis=0)
        return tf.reduce_max(tf.reshape(inputs, new_shape), axis=-1)
```

✅ 程式碼逐行解析

1. `class DepthPool(...)`: 自訂 Keras Layer，沿通道維度池化。
2. `shape = tf.shape(inputs)`: 取得輸入張量的動態形狀。
3. `groups = shape[-1] // self.pool_size`: 池化後的通道分組數。
4. `tf.reshape(...)`: 重塑張量，使通道可分成多個子群組。
5. `tf.reduce_max(..., axis=-1)`: 在每個通道群組上取最大值。

🎯 重點摘要

- 深度池化提供另一種壓縮通道信息的方法。
- 自訂層展示 TensorFlow 函式庫的彈性。
- 適合在深度特徵抽取後進行通道壓縮。

---

[來源: ch14 | 類型: tutorial] ### 3. 全域平均池化 (Global Average Pooling)

```python
global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
global_avg_pool(images)
```

✅ 程式碼逐行解析

1. `GlobalAvgPool2D()`: 建立全域平均池化層。
2. `global_avg_pool(images)`: 將每個通道的高度與寬度平均為單一值。

🎯 重點摘要

- 全域平均池化可將每個特徵圖壓縮為一個標量，適用於分類器之前。
- 它能顯著降低參數數量並減少過擬合風險。
- 與 `Flatten()` 相比，更適合多尺度輸入。

---

[來源: ch14 | 類型: tutorial] ##  Fashion MNIST CNN 範例

這個範例展示如何使用 CNN 進行 Fashion MNIST 影像分類，包含常見的層堆疊、
Dropout 與 Dense 分類器。

---

[來源: ch14 | 類型: tutorial] ```python
from functools import partial

default_conv = partial(tf.keras.layers.Conv2D,
                       kernel_size=3,
                       padding="same",
                       activation="relu",
                       kernel_initializer="he_normal")

model = tf.keras.Sequential([
    default_conv(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPool2D(),
    default_conv(filters=128),
    default_conv(filters=128),
    tf.keras.layers.MaxPool2D(),
    default_conv(filters=256),
    default_conv(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation="softmax")
])
```

---

[來源: ch14 | 類型: tutorial] ,
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation="softmax")
])
```

✅ 程式碼逐行解析

1. `partial(...)`: 使用 `functools.partial` 簡化 `Conv2D` 建構。
2. `default_conv(filters=64, kernel_size=7, input_shape=[28, 28, 1])`:
   第一層卷積，輸入為 28×28 灰階影像。
3. `MaxPool2D()`: 每次下採樣 2×2。
4. 重複 128 和 256 通道的卷積堆疊：增加特徵抽取能力。
5. `Flatten()`: 將特徵圖攤平成向量，送入密集層 (Dense layer)。
6. `Dense(128, ...)` / `Dense(64, ...)`: 隱藏層。
7. `Dropout(0.5)`: 隨機失活 50%，避免過擬合 (overfitting)。
8. `Dense(10, activation="softmax")`: 輸出 10 類的機率分佈。

🎯 重點摘要

---

[來源: ch14 | 類型: tutorial] out(0.5)`: 隨機失活 50%，避免過擬合 (overfitting)。
8. `Dense(10, activation="softmax")`: 輸出 10 類的機率分佈。

🎯 重點摘要

- 這是典型的影像分類 CNN 架構：卷積 + 池化 + 全連接層。
- Dropout 有助於穩定訓練並提升泛化能力。
- `he_normal` 初始化適合 ReLU 活化函數 (activation function)。

---

[來源: ch14 | 類型: tutorial] ##  ResNet 殘差單元 (Residual Unit) 與深度架構

ResNet 的殘差單元是解決深度網路退化 (degradation) 問題的關鍵。它透過
捷徑連接 (skip connection) 讓輸入直接疊加到後續層，改善梯度傳播。

---

[來源: ch14 | 類型: tutorial] nit) 與深度架構

ResNet 的殘差單元是解決深度網路退化 (degradation) 問題的關鍵。它透過
捷徑連接 (skip connection) 讓輸入直接疊加到後續層，改善梯度傳播。

```python
DefaultConv2D = partial(tf.keras.layers.Conv2D,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        kernel_initializer="he_normal",
                        use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
```

---

[來源: ch14 | 類型: tutorial] self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
```

✅ 程式碼逐行解析

1. `use_bias=False`: 使用批次歸一化 (Batch Normalization) 時可省略偏差項。
2. `self.main_layers`: 主路徑包含兩次卷積與批次歸一化。
3. `self.skip_layers`: 當 `strides > 1` 時，捷徑路徑也做下採樣。
4. `Z = layer(Z)`: 主路徑逐層計算卷積輸出。
5. `skip_Z = layer(skip_Z)`: 捷徑路徑保留輸入資訊。
6. `return self.activation(Z + skip_Z)`: 主路徑與捷徑相加後再套用激活函數。

🎯 重點摘要

- 殘差連接讓深層網路更容易收斂，也能保留低層特徵。
- `strides > 1` 時，捷徑路徑必須做空間縮放以匹配主路徑輸出。
- 這種設計是 ResNet 成功的核心原因。

---

[來源: ch14 | 類型: tutorial] ```python
model = tf.keras.Sequential([
    DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
])
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

---

[來源: ch14 | 類型: tutorial] model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

🎯 重點摘要

- ResNet-34 使用多個相同濾波器數量的殘差區塊。
- `GlobalAvgPool2D()` 將每個通道壓縮為單一統計量，適合分類器輸入。
- 這樣的架構適合深層影像分類問題。

---

[來源: ch14 | 類型: tutorial] ##  預訓練模型與轉移學習 (Transfer Learning)

轉移學習可讓你重用大型資料集（如 ImageNet）上訓練好的特徵，
快速部署到新任務。

```python
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False
```

✅ 程式碼逐行解析

---

[來源: ch14 | 類型: tutorial] l.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False
```

✅ 程式碼逐行解析

1. `xception.Xception(...)`: 載入 Xception 模型，不包含頂層分類器。
2. `include_top=False`: 保留特徵提取器，丟棄原始 ImageNet 分類層。
3. `GlobalAveragePooling2D()`: 將空間特徵聚合成向量。
4. `Dense(n_classes, activation="softmax")`: 新的分類器輸出層。
5. `tf.keras.Model(...)`: 建立完整模型。
6. `layer.trainable = False`: 凍結基底模型，僅訓練新加入的分類層。

🎯 重點摘要

- 凍結基底模型可避免一次訓練過多參數，適合資料量較少的任務。
- 新頂層會學習對應新任務的類別。
- 這是實務上常見的快速部署策略。

---

[來源: ch14 | 類型: tutorial] ### 資料前處理與增強 (Data Augmentation)

```python
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224,
                             crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(
        tf.keras.applications.xception.preprocess_input)
])

train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
```

🎯 重點摘要

- 預訓練模型通常要求固定輸入尺寸，例如 224×224。
- `preprocess_input` 會將像素值轉換到模型預期的區間，例如 [-1, 1]。
- `shuffle()`、`batch()` 和 `prefetch()` 可提升訓練效率。

---

[來源: ch14 | 類型: tutorial] ##  進階微調 (Fine-Tuning) 與多輸出分類

當新加入的分類層收斂後，可對基底模型最後幾層進行微調 (fine-tuning)，
讓整體模型更適應新資料集。

```python
for layer in base_model.layers[56:]:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=10)
```

🎯 重點摘要

- 微調時應使用較低學習率，以避免破壞預訓練權重。
- 只解凍部分基底層可以平衡訓練穩定性與適應性。
- 這樣的流程通常先「凍結訓練」、「再解凍微調」。

---

[來源: ch14 | 類型: tutorial] ### 範例：分類與定位的多輸出模型

```python
class_output = tf.keras.layers.Dense(
    n_classes, activation="softmax")(avg)
loc_output = tf.keras.layers.Dense(4)(avg)
model = tf.keras.Model(inputs=base_model.input,
                       outputs=[class_output, loc_output])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
              loss_weights=[0.8, 0.2],
              optimizer=optimizer, metrics=["accuracy", "mse"])
```

🎯 重點摘要

---

[來源: ch14 | 類型: tutorial] loss_weights=[0.8, 0.2],
              optimizer=optimizer, metrics=["accuracy", "mse"])
```

🎯 重點摘要

- 多輸出模型同時預測類別與邊界框 (bounding box)，適用於定位任務。
- `loss_weights` 可調整不同目標的影響力。
- 這類架構是目標檢測 (object detection) 任務的基礎概念。

---

[來源: ch14 | 類型: tutorial] ### 問：為什麼要用 `padding="same"` 而不是 `padding="valid"`？

`padding="same"` 可以保持輸入與輸出空間尺寸一致，讓卷積網路更容易疊加，
特別適合深層 CNN。`padding="valid"` 則會在每層逐漸縮小空間尺寸，
適合需要快速下採樣的架構。

---

[來源: ch14 | 類型: tutorial] ### 問：為何轉移學習時要先凍結基底模型？

因為預訓練模型已經學到通用特徵，先凍結基底模型可避免新任務資料量不足時
破壞這些特徵。這樣可以只訓練新頂層，降低過擬合風險。

---

[來源: ch14 | 類型: tutorial] ### 問：ResNet 的殘差連接解決了什麼問題？

它解決了深層網路的梯度退化 (gradient degradation) 與訓練困難問題，
讓深層網路可以像淺層網路一樣傳遞訊號。殘差連接提供了捷徑路徑，
讓梯度更容易回傳到前層。

---

[來源: ch14 | 類型: tutorial] ##  推薦標籤

標籤：#CNN #深度學習 #TensorFlow #Keras #Python教學 #機器學習 #電腦視覺

---

[來源: ch15 | 類型: cheatsheet] # Ch15 速查表：Processing Sequences with RNNs & CNNs

> **核心主旨**：時間序列 + 序列模型 —— LSTM/GRU 處理長期依賴，Conv1D 提取局部模式，WaveNet 串接兩者。

---

---

[來源: ch15 | 類型: cheatsheet] ## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| SimpleRNN | 最基本的 RNN，容易梯度消失 | 短序列實驗 |
| LSTM | 有遺忘門/輸入門/輸出門，記憶長期依賴 | 長序列（文字、時間序列） |
| GRU | LSTM 的精簡版（合併遺忘+輸入門），速度更快 | 資源有限時的 LSTM 替代 |
| Deep RNN | 堆疊多層 RNN，`return_sequences=True` | 提升序列表示能力 |
| Bidirectional RNN | 正向 + 反向 RNN 並行，雙向看序列 | NLP、序列分類 |
| Conv1D | 在時間維度做 1D 卷積，提取局部模式 | 時間序列特徵提取 |
| WaveNet | 膨脹因果卷積（dilation），接受野指數增長 | 音訊生成、長序列建模 |
| ARIMA | 統計時間序列模型（差分+自迴歸+移動平均） | 週期性單變數時間序列基準 |

---

---

[來源: ch15 | 類型: cheatsheet] | Keras API | 重點參數 | 用途 |
|-----------|---------|------|
| `tf.keras.layers.SimpleRNN` | `units=20`, `return_sequences=True` | 基本 RNN（不推薦用於實務） |
| `tf.keras.layers.LSTM` | `units=20`, `return_sequences=True/False` | 長短期記憶 |
| `tf.keras.layers.GRU` | `units=20`, `return_sequences=True/False` | 閘控循環單元 |
| `tf.keras.layers.Bidirectional` | `layer=LSTM(...)` | 雙向 RNN 包裝器 |
| `tf.keras.layers.Conv1D` | `filters=32`, `kernel_size=5`, `padding="causal"` | 1D 因果卷積 |
| `tf.keras.layers.LayerNormalization` | – | 層歸一化（RNN 常用，代替 BN） |
| `tf.keras.layers.TimeDistributed` | `layer=Dense(...)` | 對每個時間步獨立應用層 |
| `padding="causal"` | – | 確保只看過去資料，不看未來 |
| `dilation_rate=2` | – | 膨脹卷積，擴大感受野 |


---

[來源: ch15 | 類型: cheatsheet] ` | 對每個時間步獨立應用層 |
| `padding="causal"` | – | 確保只看過去資料，不看未來 |
| `dilation_rate=2` | – | 膨脹卷積，擴大感受野 |


---

---

[來源: ch15 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import tensorflow as tf
import numpy as np

---

[來源: ch15 | 類型: cheatsheet] # 時間序列資料準備（seq2seq 格式）
def to_windows(series, seq_length):
    """將序列切成 (input, target) 滑動視窗"""
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(seq_length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(seq_length + 1))
    dataset = dataset.map(lambda w: (w[:-1], w[-1]))  # 最後一步為 target
    return dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

---

[來源: ch15 | 類型: cheatsheet] # Simple RNN（基礎）
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])

---

[來源: ch15 | 類型: cheatsheet] # Deep LSTM（實務推薦）
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=[None, n_features]),
    tf.keras.layers.LSTM(64),          # 最後一層不需要 return_sequences
    tf.keras.layers.Dense(1)           # 迴歸輸出
])

---

[來源: ch15 | 類型: cheatsheet] # GRU（比 LSTM 輕量）
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True, input_shape=[None, n_features]),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(n_outputs)
])

---

[來源: ch15 | 類型: cheatsheet] # 雙向 LSTM（NLP 任務）
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

---

[來源: ch15 | 類型: cheatsheet] # Conv1D 時間序列（因果填充）
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                            padding="causal", activation="relu",
                            input_shape=[None, n_features]),
    tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="causal", activation="relu"),
    tf.keras.layers.Dense(1)
])

---

[來源: ch15 | 類型: cheatsheet] # WaveNet 風格（膨脹因果卷積）
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[None, n_features]))
for dilation_rate in [1, 2, 4, 8, 16, 32, 64, 128]:
    model.add(tf.keras.layers.Conv1D(
        filters=32, kernel_size=2,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu"))
model.add(tf.keras.layers.Conv1D(filters=n_outputs, kernel_size=1))

---

[來源: ch15 | 類型: cheatsheet] # 訓練時間序列模型
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,
                                restore_best_weights=True)])
```

---

---

[來源: ch15 | 類型: cheatsheet] ## 4. 常見陷阱

- **`return_sequences=True` vs `False`**：堆疊多層 RNN 時，除最後一層外全部要 `return_sequences=True`；若需 seq2seq 輸出，最後一層也要 True。
- **因果卷積 `padding="causal"`**：時間序列預測時，**絕對不能**看未來資料，必須用 `"causal"` 而非 `"same"`。
- **LSTM 比 SimpleRNN 慢 3-5 倍**：輕量任務或 CPU 環境可考慮 GRU（速度接近 LSTM，效果略差一點）。
- **LayerNormalization vs BatchNormalization**：RNN 中使用 LayerNorm（對每個樣本的所有特徵歸一化），不用 BatchNorm（對每個批次的樣本歸一化，序列長度不固定時問題很多）。

---

---

[來源: ch15 | 類型: cheatsheet] ## 5. 決策指南

```
時間序列任務選模型：
├── 統計基準線         → ARIMA/SARIMA（statsmodels）
├── 短期局部模式       → Conv1D（快速）
├── 長期依賴           → LSTM 或 GRU
├── 超長序列（音訊）   → WaveNet（膨脹卷積）
└── 混合               → Conv1D → LSTM（先提取局部，再建模全局）

LSTM vs GRU：
├── 精度優先           → LSTM
├── 速度/記憶體優先    → GRU
└── 實務差異不大       → 兩者都試，選驗證集更好的

多步預測策略：
├── 遞推（1步→多步）→ 誤差累積
├── 直接（一次預測多步）→ 更穩定（推薦）
└── seq2seq Encoder-Decoder → 複雜但靈活
```

---

[來源: ch15 | 類型: handout] # 課程講義：使用 RNN 與 CNN 處理序列 (Chapter 15)

時間序列、文字、音訊——這些資料的共同特點是**順序很重要**。本章介紹循環神經網路（RNN），一種內建「記憶」的架構：它在處理序列時，隱藏狀態會將過去的資訊傳遞給未來的時間步驟。我們以芝加哥公共交通乘客量預測為主線，從 ARIMA 基準線出發，逐步引入 SimpleRNN、LSTM，最後以 WaveNet 的膨脹卷積收尾。

---

---

[來源: ch15 | 類型: handout] ### 理論背景

在建立神經網路模型之前，必須先了解資料的統計特性，並建立**傳統基準線**——若神經網路無法超越 ARIMA，那就不值得用神經網路。

**時間序列的關鍵概念**：

- **趨勢 (Trend)**：長期的上升或下降方向
- **季節性 (Seasonality)**：固定週期的規律（如每週、每年）
- **差分 (Differencing)**：$\nabla y_t = y_t - y_{t-k}$ 移除趨勢/季節性，使序列平穩

**ARIMA (Autoregressive Integrated Moving Average)**：

$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t$$

---

[來源: ch15 | 類型: handout] _t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t$$

SARIMA = ARIMA + 季節性項目，適合有週期性的時間序列。

**序列到監督式學習**：給定過去 $n$ 步，預測未來 1 步（或多步）。

---

[來源: ch15 | 類型: handout] ### 核心代碼

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

---

[來源: ch15 | 類型: handout] # 載入芝加哥每日乘客量
df = pd.read_csv("datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv",
                 parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)

rail = df["rail"] / 1e6  # 縮放至百萬人次

---

[來源: ch15 | 類型: handout] # 7 天差分消除季節性（一週週期）
diff_7 = rail.diff(7).dropna()

---

[來源: ch15 | 類型: handout] # ARIMA 基準線（每日重新訓練）
from statsmodels.tsa.arima.model import ARIMA

---

[來源: ch15 | 類型: handout] # 訓練 ARIMA 並預測下一步
model = ARIMA(rail.iloc[:-1], order=(1, 0, 0))  # 簡化版 AR(1)
model = model.fit()
y_pred = model.forecast()  # 預測下一天

---

[來源: ch15 | 類型: handout] # 轉換為監督式學習資料集
def to_supervised(series, seq_length=56):
    """將時間序列轉換為滑動視窗資料集"""
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series.iloc[i:i+seq_length].values)
        y.append(series.iloc[i+seq_length])
    return np.array(X)[..., np.newaxis], np.array(y)

---

[來源: ch15 | 類型: handout] es)
        y.append(series.iloc[i+seq_length])
    return np.array(X)[..., np.newaxis], np.array(y)

seq_length = 56  # 8 週歷史
X_train_ts, y_train_ts = to_supervised(rail[:int(len(rail)*0.7)], seq_length)
print(f"X_train shape: {X_train_ts.shape}")  # (n_samples, 56, 1)
```

---

[來源: ch15 | 類型: handout] ### 補充練習 1

**理論題：** 為什麼要用 7 天差分而非 1 天差分？如何用「自相關函數（ACF）」圖判斷一個時間序列的季節性週期？

**實作題：** 計算芝加哥 Rail 乘客量的 7 天和 365 天自相關，繪製 ACF 圖（用 `statsmodels.graphics.tsaplots.plot_acf`），從圖中識別出週週期和年週期。

---

---

[來源: ch15 | 類型: handout] ### 理論背景

**RNN 的基本公式**：

$$\mathbf{h}_t = \tanh\left(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}\right)$$

隱藏狀態 $\mathbf{h}_t$ 既是目前時間步的輸出，也傳遞給下一個時間步作為記憶。

**梯度消失 (Vanishing Gradient)**：

反向傳播時梯度需要沿時間步「回傳」，每步乘以 $\mathbf{W}_h^T$。若 $\|\mathbf{W}_h\| < 1$，梯度指數級萎縮 → RNN 難以學習長期依賴關係。

**梯度爆炸**的解法：梯度裁剪 (`clipnorm`, `clipvalue`)

**梯度消失**的解法：LSTM、GRU（門控機制）

---

[來源: ch15 | 類型: handout] ### 核心代碼

```python
import tensorflow as tf

tf.random.set_seed(42)

---

[來源: ch15 | 類型: handout] # SimpleRNN 序列預測
model_simple = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])

---

[來源: ch15 | 類型: handout] ces=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])

model_simple.compile(optimizer="adam", loss="huber")
history = model_simple.fit(
    X_train_ts, y_train_ts,
    epochs=20,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

---

[來源: ch15 | 類型: handout] # 深層 RNN（堆疊 SimpleRNN）
model_deep = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20, return_sequences=True),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])
```

---

[來源: ch15 | 類型: handout] ### 補充練習 2

**理論題：** `return_sequences=True` 與 `return_sequences=False`（預設）的差異是什麼？當你需要堆疊多個 RNN 層時，除了最後一層，所有中間層都需要設定什麼？

**實作題：** 比較 `SimpleRNN(20)` 和 `SimpleRNN(20, return_sequences=True) → Dense(1)` 兩種結構在時間序列預測上的 MAE，哪一種對序列預測任務更合適？

---

---

[來源: ch15 | 類型: handout] ### 理論背景

**LSTM (Long Short-Term Memory)** 引入**細胞狀態 (Cell State)** $\mathbf{c}_t$，解決梯度消失：

**遺忘閘 (Forget Gate)**：決定丟棄多少舊記憶：

$$\mathbf{f}_t = \sigma\left(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\right)$$

**輸入閘 (Input Gate)**：決定加入多少新資訊：

$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

---

[來源: ch15 | 類型: handout] ：決定加入多少新資訊：

$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**細胞狀態更新**：

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**輸出閘 (Output Gate)**：

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

---

[來源: ch15 | 類型: handout] e{\mathbf{c}}_t$$

**輸出閘 (Output Gate)**：

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

**GRU**：LSTM 的簡化版（合併遺忘閘和輸入閘），參數更少，通常速度更快，效果相近。

---

[來源: ch15 | 類型: handout] ### 核心代碼

```python
tf.random.set_seed(42)

---

[來源: ch15 | 類型: handout] # LSTM 模型（深層）
model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dense(1)
])
model_lstm.compile(optimizer="adam", loss="huber")

---

[來源: ch15 | 類型: handout] # GRU 模型（更快，效果相近）
model_gru = tf.keras.Sequential([
    tf.keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.GRU(20),
    tf.keras.layers.Dense(1)
])
model_gru.compile(optimizer="adam", loss="huber")

---

[來源: ch15 | 類型: handout] # 多變數時間序列（bus + rail + day_type）

---

[來源: ch15 | 類型: handout] # 輸入形狀：(batch_size, seq_length, n_features)
model_multivar = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),  # 5 個特徵
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1)
])
```

---

[來源: ch15 | 類型: handout] ### 補充練習 3

**理論題：** LSTM 的細胞狀態 $\mathbf{c}_t$ 如何解決梯度消失問題？從反向傳播的角度，說明為何梯度可以「穿越時間」而不會大幅縮減。

**實作題：** 在芝加哥乘客量資料上，比較 `SimpleRNN`、`LSTM`、`GRU` 三種架構的測試集 MAE 和訓練時間（各 20 epochs），製成比較表。

---

---

[來源: ch15 | 類型: handout] ### 理論背景

**多步預測策略**：

1. **迭代預測**：每次只預測 1 步，將預測值加入輸入，再預測下一步（誤差會累積）
2. **直接多步輸出**：最後一層改為 `Dense(n_steps)`，一次輸出多步（不累積誤差）
3. **Seq2Seq**：`TimeDistributed(Dense(n_steps))` 對每個時間步都輸出，訓練更高效

**`TimeDistributed` 層**：將同一個 `Dense` 層應用到序列的每個時間步：

```python

---

[來源: ch15 | 類型: handout] # 等效寫法
TimeDistributed(Dense(14))  ≡  Dense(14)  # 對 LSTM 的 return_sequences=True 輸出
```

---

[來源: ch15 | 類型: handout] ### 核心代碼

```python
tf.random.set_seed(42)

---

[來源: ch15 | 類型: handout] # 直接多步輸出（預測未來 14 天）
model_ahead = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(14)  # 一次輸出 14 步
])
model_ahead.compile(optimizer="adam", loss="mae")

---

[來源: ch15 | 類型: handout] # Seq2Seq（TimeDistributed，每步都預測）
model_seq2seq = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),
    tf.keras.layers.LSTM(16, return_sequences=True),
    # 等效：tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(14))
    tf.keras.layers.Dense(14)  # Keras Dense 自動對序列每步應用
])
```

---

[來源: ch15 | 類型: handout] ### 補充練習 4

**理論題：** 迭代預測（步驟1）的誤差如何「雪球效應」累積？在預測 14 天後，直接多步輸出（策略2）的誤差是否一定優於迭代預測？各有什麼情境下更合適？

**實作題：** 實作迭代預測：訓練一個只預測 1 步的 LSTM，然後循環呼叫 14 次，每次將前一次的預測加入輸入序列。比較迭代預測與直接 `Dense(14)` 輸出在 14 天預測上的 MAE。

---

---

[來源: ch15 | 類型: handout] ### 理論背景

**WaveNet** 使用 **膨脹因果卷積 (Dilated Causal Convolutions)** 處理長序列：

- **因果 (Causal)**：只看過去的輸入，不洩漏未來資訊（`padding="causal"`）
- **膨脹 (Dilated)**：卷積核的「感受野」以指數速度增長

膨脹率倍增：1, 2, 4, 8, 16, ...

- Dilation=1：感受野 = 2 步
- Dilation=2：感受野 = 3 步（跳隔取樣）
- Dilation=4：感受野 = 5 步
- 10 層（膨脹率 1→512）：感受野 = **1023 步**（$2^{10} - 1$）

優點：不受梯度消失影響、平行計算比 RNN 快、長期依賴性強。

---

[來源: ch15 | 類型: handout] ### 核心代碼

```python
tf.random.set_seed(42)

---

[來源: ch15 | 類型: handout] # WaveNet 架構
wavenet_model = tf.keras.Sequential()
wavenet_model.add(tf.keras.layers.Input(shape=[None, 5]))

---

[來源: ch15 | 類型: handout] # 堆疊膨脹因果卷積層（膨脹率：1, 2, 4, 8）
for dilation_rate in (1, 2, 4, 8):
    wavenet_model.add(
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=2,
            padding="causal",      # 因果填充（只看過去）
            activation="relu",
            dilation_rate=dilation_rate  # 膨脹率
        )
    )

---

[來源: ch15 | 類型: handout] （只看過去）
            activation="relu",
            dilation_rate=dilation_rate  # 膨脹率
        )
    )

wavenet_model.add(tf.keras.layers.Dense(14))
wavenet_model.compile(optimizer="adam", loss="mae")
print(wavenet_model.summary())
```

---

[來源: ch15 | 類型: handout] ### 補充練習 5

**理論題：** WaveNet 使用 `kernel_size=2`、`dilation_rate=8` 的卷積，這個卷積核能看到多少步之前的資訊（感受野是多少）？計算 4 層（dilation 1, 2, 4, 8）堆疊後的總感受野。

**實作題：** 比較 LSTM 模型和 WaveNet 模型在芝加哥乘客量資料上的訓練速度（每 epoch 時間）和最終 MAE，哪種在此資料集上更有優勢？

---

---

[來源: ch15 | 類型: handout] ## 結論

序列模型的演進：

- **SimpleRNN**：概念清晰，但梯度消失使其難以學習長期依賴
- **LSTM/GRU**：門控機制解決梯度消失，是最常用的序列模型
- **多步預測**：直接輸出多步通常優於迭代預測
- **WaveNet**：膨脹因果卷積，長感受野、平行計算，適合長序列

下一章（Ch16）將 RNN 應用於 NLP：文字生成、情感分析、機器翻譯和 Attention 機制。

---

---

[來源: ch15 | 類型: handout] ## 課後作業

**作業：時間序列預測競賽**

使用芝加哥乘客量資料集（或自選時間序列）：

1. 實作 ARIMA 基準線和 LSTM 模型，比較兩者在測試集的 MAE（以百萬人次為單位）。

2. 建立**多變數 LSTM**，將 `bus`、`rail`、以及 `day_type` 的 One-Hot 編碼同時作為輸入特徵，是否能提升預測準確率？

3. 嘗試 WaveNet（4 層膨脹卷積），比較其與 LSTM 的訓練速度和最終預測效果，製成對比表（模型 / MAE / 訓練時間 / 參數量）。

---

[來源: ch15 | 類型: tutorial] [標題: RNN 與 CNN 序列處理完整指南：LSTM、GRU、WaveNet 與時間序列預測 | 描述: 深入序列建模：SimpleRNN、LSTM、GRU 的工作原理與實作、序列到序列預測、TimeDistributed 層、WaveNet 膨脹因果捲積，以及時間序列預測的最佳實踐。 | 關鍵字: Python, RNN, LSTM, GRU, TensorFlow, Keras, 時間序列, WaveNet, 序列建模, 深度學習]
# RNN 與 CNN 序列處理：從 SimpleRNN 到 WaveNet

語音識別、股票預測、機器翻譯——這些都是**序列問題（Sequence Problem）**。傳統前饋網路無法處理序列資料，因為它無法利用**時間上下文（Temporal Context）**。本教學帶你從最基礎的 RNN 出發，逐步掌握 LSTM、GRU，最終理解 WaveNet 的膨脹因果捲積架構。

---

[來源: ch15 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **SimpleRNN** 因梯度消失問題，無法捕捉長期依賴（>10~20 步）
- **LSTM** 的三個門（遺忘門、輸入門、輸出門）讓它能記憶/遺忘長期資訊
- **GRU** 是 LSTM 的簡化版，少一個門，速度更快，效能相近
- **`return_sequences=True`**：回傳每個時間步的輸出（給下一層 RNN 或序列輸出）
- **WaveNet** 的膨脹因果捲積可以平行訓練（不像 RNN 必須序列計算），感受野指數增長

---

---

[來源: ch15 | 類型: tutorial] ## SimpleRNN 基礎

💡 **實際應用情境：** 用感測器數據預測機器故障——感測器每秒回傳一個讀數，我們需要利用過去 24 小時的資料來預測接下來是否會故障。這是一個序列分類問題。

---

[來源: ch15 | 類型: tutorial] ### 範例 1: 建立 SimpleRNN 模型

```python
import tensorflow as tf
import numpy as np

---

[來源: ch15 | 類型: tutorial] # 產生簡單的序列資料（用於示範）
def generate_time_series(batch_size, n_steps):
    """生成包含兩個不同頻率正弦波的時間序列"""
    freq1, freq2, offsets1, offsets2 = (
        np.random.rand(4, batch_size, 1)
    )
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)  # [batch, steps, 1]

---

[來源: ch15 | 類型: tutorial] tch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)  # [batch, steps, 1]

np.random.seed(42)
n_steps = 50
X_train = generate_time_series(7000, n_steps + 1)
X_valid = generate_time_series(2000, n_steps + 1)
X_test  = generate_time_series(500,  n_steps + 1)

---

[來源: ch15 | 類型: tutorial] # 輸入：前 50 步；輸出：第 51 步（預測下一個值）
y_train, y_valid, y_test = X_train[:, -1], X_valid[:, -1], X_test[:, -1]
X_train, X_valid, X_test = X_train[:, :-1], X_valid[:, :-1], X_test[:, :-1]

print(f"X_train shape: {X_train.shape}")  # (7000, 50, 1)
print(f"y_train shape: {y_train.shape}")  # (7000, 1)

---

[來源: ch15 | 類型: tutorial] # 最簡單的基準：直接用最後一個值（天真預測）
y_naive = X_valid[:, -1]
baseline_mse = np.mean(tf.keras.losses.mean_squared_error(y_valid, y_naive))
print(f"基準 MSE（天真預測）: {baseline_mse:.4f}")
```

**✅ 程式碼逐行解析：**

1. `[..., np.newaxis]`: 在最後添加維度，讓形狀從 `[batch, steps]` 變為 `[batch, steps, 1]`（RNN 需要特徵維度）
2. `X_train[:, -1]`: 取每個序列的最後一步作為目標值（預測未來一步）
3. 基準測試：用最後已知值直接預測——若你的 RNN 連這個都贏不了，說明模型有問題

---

---

[來源: ch15 | 類型: tutorial] ## LSTM 長短期記憶

💡 **實際應用情境：** 翻譯「我 **非常** 喜歡這部電影」時，LSTM 的遺忘門會根據「非常」修改對情感的記憶強度，而 SimpleRNN 在到達「電影」時早已「忘記」了「非常」的影響。

---

[來源: ch15 | 類型: tutorial] ### 範例 2: LSTM 時間序列預測

```python

---

[來源: ch15 | 類型: tutorial] # 深層 LSTM 模型
model_lstm = tf.keras.Sequential([
    # 輸入形狀：(時間步數, 特徵數) = (50, 1)
    tf.keras.layers.LSTM(
        64,
        return_sequences=True,   # 回傳每個時間步的隱藏狀態（給下一層用）
        input_shape=[None, 1]    # None 表示可接受任意長度的序列
    ),
    tf.keras.layers.LSTM(32),    # 最後一層：只回傳最後時間步
    tf.keras.layers.Dense(1)     # 輸出一個值（預測下一個時間步）
])

model_lstm.compile(optimizer="adam", loss="mse")
model_lstm.summary()

---

[來源: ch15 | 類型: tutorial] # 訓練（使用早停回調）
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True  # 恢復最佳權重
    )
]

history = model_lstm.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=callbacks,
    verbose=0  # 靜默訓練（不輸出每個 epoch）
)

---

[來源: ch15 | 類型: tutorial] # 評估
lstm_mse = model_lstm.evaluate(X_valid, y_valid, verbose=0)
print(f"LSTM 驗證 MSE: {lstm_mse:.4f}")
print(f"基準 MSE: {baseline_mse:.4f}")
print(f"改善幅度: {(baseline_mse - lstm_mse) / baseline_mse * 100:.1f}%")
```

**LSTM 的核心結構（理解門控機制）：**

$$\text{遺忘門}: f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$\text{輸入門}: i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\text{輸出門}: o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$\text{記憶單元}: C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**✅ 程式碼逐行解析：**

---

[來源: ch15 | 類型: tutorial] t{記憶單元}: C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**✅ 程式碼逐行解析：**

1. `return_sequences=True`: LSTM 回傳每個時間步的輸出 `(batch, steps, units)`，讓下一層 LSTM 能看到完整序列
2. `input_shape=[None, 1]`: `None` 允許可變長度的序列（適應不同長度的輸入）
3. `EarlyStopping(restore_best_weights=True)`: 驗證損失不再改善後停止，並恢復最佳時期的權重

**🎯 重點摘要:**

- LSTM 的 `units` 參數是隱藏狀態的維度（不是時間步數）
- 堆疊多層 LSTM 時，中間層需要 `return_sequences=True`，最後一層預設 `return_sequences=False`

---

---

[來源: ch15 | 類型: tutorial] ### 範例 3: GRU 模型（更快的 LSTM 替代品）

```python

---

[來源: ch15 | 類型: tutorial] # GRU 只有兩個門（更新門 + 重置門）vs LSTM 的三個門
model_gru = tf.keras.Sequential([
    tf.keras.layers.GRU(
        64,
        return_sequences=True,
        input_shape=[None, 1]
    ),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(1)
])

model_gru.compile(optimizer="adam", loss="mse")

---

[來源: ch15 | 類型: tutorial] # GRU 參數更少，通常訓練更快
lstm_params = model_lstm.count_params()
gru_params  = model_gru.count_params()
print(f"LSTM 參數量: {lstm_params:,}")
print(f"GRU 參數量: {gru_params:,}")
print(f"GRU 比 LSTM 少 {(lstm_params - gru_params)/lstm_params*100:.1f}% 參數")
```

**🎯 重點摘要:**

- GRU 參數約為 LSTM 的 75%，速度更快，長序列上效能相近
- 實務上：先嘗試 LSTM，若太慢或過擬合嚴重則改用 GRU

---

---

[來源: ch15 | 類型: tutorial] ### 範例 4: 雙向 LSTM（適合非實時任務）

```python

---

[來源: ch15 | 類型: tutorial] # 注意：不適合實時預測（需要未來資訊），但適合情感分析等任務
model_bidirectional = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        input_shape=[None, 1]
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32)
    ),
    tf.keras.layers.Dense(1)
])

---

[來源: ch15 | 類型: tutorial] # 雙向 LSTM 的輸出維度是 2 × units（前向 + 反向）
model_bidirectional.summary()
```

---

---

[來源: ch15 | 類型: tutorial] ## 序列到序列預測

💡 **實際應用情境：** 氣象站的台灣天氣預測——輸入過去 50 天的氣溫，**一次輸出**未來 10 天的預測（而非每次只輸出一個值）。

---

[來源: ch15 | 類型: tutorial] # 目標：給定前 50 步，同時預測接下來 10 步
n_steps = 50
n_future = 10

---

[來源: ch15 | 類型: tutorial] # 重新準備資料
X_train_seq = generate_time_series(7000, n_steps + n_future)
y_train_seq = X_train_seq[:, n_steps:, 0]  # 未來 10 步作為目標
X_train_seq = X_train_seq[:, :n_steps]      # 前 50 步作為輸入

---

[來源: ch15 | 類型: tutorial] # 方法 1：Vector 輸出（最簡單）
model_vector = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(n_future)  # 直接輸出 10 個值
])

---

[來源: ch15 | 類型: tutorial] # 方法 2：Sequence 輸出（使用 TimeDistributed）
model_seq2seq = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_future)
    )
    # TimeDistributed 對每個時間步都應用 Dense，輸出 (batch, steps, n_future)
])

model_vector.compile(optimizer="adam", loss="mse")
model_vector.summary()
```

**✅ 程式碼逐行解析：**

1. `Dense(n_future)`: 最終 Dense 層直接輸出未來 10 個值——最簡單的多步預測方法
2. `TimeDistributed(Dense(n_future))`: 對 LSTM 每個時間步的輸出都應用 Dense，生成序列輸出

---

---

[來源: ch15 | 類型: tutorial] ## WaveNet 膨脹因果捲積

💡 **實際應用情境：** Google 的語音合成系統（WaveNet）可生成逼真的語音。它的核心是**膨脹因果捲積（Dilated Causal Convolution）**——感受野呈指數增長，且可以完全平行訓練（不像 RNN 必須序列計算）。

---

[來源: ch15 | 類型: tutorial] ### 範例 6: WaveNet 風格架構

```python

---

[來源: ch15 | 類型: tutorial] # WaveNet 的關鍵：膨脹因果捲積（Dilated Causal Convolution）

---

[來源: ch15 | 類型: tutorial] # 膨脹率 1, 2, 4, 8 ... → 感受野從 1 指數增長

def build_wavenet(n_steps: int, n_features: int = 1,
                  dilation_rates=(1, 2, 4, 8, 16, 32)) -> tf.keras.Model:
    """建立 WaveNet 風格的膨脹因果捲積模型"""
    inputs = tf.keras.Input(shape=(n_steps, n_features))
    x = inputs

for dilation_rate in dilation_rates:
        x = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=2,
            dilation_rate=dilation_rate,   # 膨脹率：跳步取樣
            padding="causal",              # 因果填充：只看過去（不看未來）
            activation="relu"
        )(x)

---

[來源: ch15 | 類型: tutorial] padding="causal",              # 因果填充：只看過去（不看未來）
            activation="relu"
        )(x)

outputs = tf.keras.layers.Conv1D(filters=1, kernel_size=1)(x)
    # 取最後一個時間步的輸出
    outputs = tf.keras.layers.Lambda(lambda x: x[:, -1:])(outputs)
    outputs = tf.keras.layers.Flatten()(outputs)

return tf.keras.Model(inputs=inputs, outputs=outputs)

model_wavenet = build_wavenet(n_steps=50)
model_wavenet.compile(optimizer="adam", loss="mse")
model_wavenet.summary()

---

[來源: ch15 | 類型: tutorial] # dilation=1:  感受野 = 2 (看前1個時間步)

---

[來源: ch15 | 類型: tutorial] # dilation=2:  感受野 = 4 (看前2個時間步)

---

[來源: ch15 | 類型: tutorial] # dilation=4:  感受野 = 8 (看前4個時間步)

---

[來源: ch15 | 類型: tutorial] # 到 dilation=32: 感受野 = 64 步！
print("WaveNet 感受野大小:")
receptive_field = 1
for d in [1, 2, 4, 8, 16, 32]:
    receptive_field += d
    print(f"  dilation={d:2d}: 累積感受野 = {receptive_field + 1}")
```

**✅ 程式碼逐行解析：**

1. `dilation_rate`: 捲積核元素之間的間距——dilation=4 時，kernel 看的是位置 t 和 t-4（跳過中間）
2. `padding="causal"`: 確保位置 t 的輸出只依賴 t 之前的輸入（不洩露未來資訊）
3. 感受野以 $2^L$ 增長（L 為層數），通過指數增長的感受野捕捉長期依賴

**🎯 重點摘要:**

- WaveNet vs RNN：RNN 必須序列計算（慢）；WaveNet 可完全平行（快）
- 膨脹捲積適合需要大感受野的序列任務（語音、音樂生成）

---

---

[來源: ch15 | 類型: tutorial] # 完整的時間序列預測流程
import matplotlib.pyplot as plt

def plot_series(time, series, label=None, y_range=None):
    plt.plot(time, series, label=label)
    if y_range:
        plt.ylim(*y_range)

---

[來源: ch15 | 類型: tutorial] # 生成測試序列
test_series = generate_time_series(1, n_steps + 10)
X_test_single = test_series[:, :n_steps]
y_test_single = test_series[:, n_steps:, 0]

---

[來源: ch15 | 類型: tutorial] # y_pred = model_lstm.predict(X_test_single)

---

[來源: ch15 | 類型: tutorial] # 多步預測：逐步預測並將預測值作為下一步的輸入
def multi_step_predict(model, X_input: np.ndarray, n_future: int) -> np.ndarray:
    """自迴歸多步預測"""
    current_input = X_input.copy()  # shape: (1, n_steps, 1)
    predictions = []
    for _ in range(n_future):
        pred = model.predict(current_input, verbose=0)  # (1, 1)
        predictions.append(pred[0, 0])
        # 滾動視窗：移除最舊的時間步，加入新預測值
        current_input = np.roll(current_input, shift=-1, axis=1)
        current_input[0, -1, 0] = pred[0, 0]
    return np.array(predictions)

print("多步預測完成！可視化預測結果。")
```

---

---

[來源: ch15 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: LSTM 和 GRU 如何選擇？**

A: GRU 計算更快（參數少），在許多任務上效能相近；LSTM 在極長序列（>100 步）上通常略好。建議先用 GRU 快速實驗，若效能不足再試 LSTM。

**Q2: `return_sequences=True/False` 怎麼選？**

A: 中間 RNN 層用 `return_sequences=True`（傳遞完整序列給下一層）；最後一層根據任務選擇：序列到序列用 `True`，序列到值用 `False`（預設）。

**Q3: 時間序列資料如何正確分割訓練/測試集？**

A: 必須按**時間順序**分割（不能隨機！）：前 70% 訓練，中間 15% 驗證，最後 15% 測試。隨機分割會造成資料洩露（訓練集中包含測試集的未來資訊）。

**Q4: RNN 訓練很慢怎麼辦？**

A: (1) 改用 GRU（比 LSTM 快 ~25%）；(2) 使用 `tf.keras.layers.CuDNNLSTM`（需要 GPU，但有限制）；(3) 考慮 WaveNet（可平行訓練）或 Transformer（現代替代品）。

---

---

[來源: ch15 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #RNN #LSTM #GRU #TensorFlow #Keras #時間序列 #WaveNet #序列建模 #深度學習 #程式設計 #教學 #DataScience #MachineLearning

---

[來源: ch16 | 類型: cheatsheet] # Ch16 速查表：NLP with RNNs & Attention

> **核心主旨**：從字元 RNN 到 Transformer —— Attention 機制是現代 NLP 的核心；實務上優先考慮 Hugging Face 預訓練模型。

---

---

[來源: ch16 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| `TextVectorization` | 將文字轉為 token 索引 | NLP 管線的第一層 |
| `Embedding` | 將 token 索引映射為密集向量 | 所有 NLP 任務 |
| Stateful RNN | 跨批次保留隱藏狀態，適合長文字串流 | 字元級語言模型 |
| Encoder-Decoder (seq2seq) | 編碼輸入序列 → 解碼輸出序列 | 機器翻譯、摘要 |
| Attention Mechanism | Decoder 每步「查詢」Encoder 所有輸出 | seq2seq 改進 |
| Multi-Head Attention | 多組 Q/K/V，並行學習多種注意力模式 | Transformer 核心 |
| Positional Encoding | 為 Transformer 注入位置資訊（無序列歸納偏置） | Transformer 必備 |
| Transformer | MHA + Feed-Forward + Residual + LayerNorm | 現代 NLP 標準架構 |
| BERT / GPT | 預訓練 Transformer；BERT 雙向，GPT 單向 | 幾乎所有 NLP 任務 |


---

[來源: ch16 | 類型: cheatsheet] Residual + LayerNorm | 現代 NLP 標準架構 |
| BERT / GPT | 預訓練 Transformer；BERT 雙向，GPT 單向 | 幾乎所有 NLP 任務 |


---

---

[來源: ch16 | 類型: cheatsheet] | Keras / HuggingFace API | 重點參數 | 用途 |
|-------------------------|---------|------|
| `tf.keras.layers.TextVectorization` | `max_tokens=`, `output_mode="int"`, `output_sequence_length=` | 文字前處理 |
| `tf.keras.layers.Embedding` | `input_dim=vocab_size`, `output_dim=16`, `mask_zero=True` | 嵌入層 |
| `tf.keras.layers.Masking` | `mask_value=0.0` | 遮蔽填充 token |
| `tf.keras.layers.Bidirectional` | `layer=LSTM(...)` | 雙向 RNN |
| `tf.keras.layers.MultiHeadAttention` | `num_heads=8`, `key_dim=64` | Multi-Head Attention |
| `tf.keras.layers.LayerNormalization` | – | Transformer 中的歸一化 |
| `transformers.pipeline()` | `task`, `model=` | HF 快速推論 |
| `transformers.AutoTokenizer` | `from_pretrained("bert-base-uncased")` | 載入 tokenizer |
| `transformers.TFAutoModelForSequenceClassification` | `from_pretrained(...)` | 載入預訓練模型 |


---

[來源: ch16 | 類型: cheatsheet] enizer |
| `transformers.TFAutoModelForSequenceClassification` | `from_pretrained(...)` | 載入預訓練模型 |


---

---

[來源: ch16 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import tensorflow as tf

---

[來源: ch16 | 類型: cheatsheet] # 文字前處理管線
vocab_size = 1000
max_length = 200

text_vec_layer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size, output_sequence_length=max_length)
text_vec_layer.adapt(train_texts)  # 建立詞彙表

---

[來源: ch16 | 類型: cheatsheet] # 情感分析（Embedding + LSTM）
model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Embedding(input_dim=vocab_size + 2, output_dim=16, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

---

[來源: ch16 | 類型: cheatsheet] # Transformer Block（自訂）
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

---

[來源: ch16 | 類型: cheatsheet] self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

def call(self, x, training=False):
        attn_output = self.mha(x, x, training=training)  # Self-attention
        x = self.ln1(x + self.drop1(attn_output, training=training))
        ffn_output = self.ffn(x)
        return self.ln2(x + self.drop2(ffn_output, training=training))

---

[來源: ch16 | 類型: cheatsheet] # Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        P = tf.Variable(tf.zeros((1, max_len, d_model)), trainable=False)
        positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angles = positions / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        P = tf.concat([tf.sin(angles[:, 0::2]), tf.cos(angles[:, 1::2])], axis=-1)
        self.P = tf.expand_dims(P, 0)

---

[來源: ch16 | 類型: cheatsheet] t([tf.sin(angles[:, 0::2]), tf.cos(angles[:, 1::2])], axis=-1)
        self.P = tf.expand_dims(P, 0)

def call(self, x):
        return x + self.P[:, :tf.shape(x)[1]]

---

[來源: ch16 | 類型: cheatsheet] # HuggingFace Pipeline（最快的 NLP 起點）
from transformers import pipeline

---

[來源: ch16 | 類型: cheatsheet] # 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love TensorFlow!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]

---

[來源: ch16 | 類型: cheatsheet] # 文字生成
generator = pipeline("text-generation", model="gpt2")
generated = generator("The future of AI is", max_length=50, num_return_sequences=1)

---

[來源: ch16 | 類型: cheatsheet] # Fine-tune BERT（TensorFlow）
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = TFAutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)

---

[來源: ch16 | 類型: cheatsheet] _bert = TFAutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)

inputs = tokenizer(train_texts, truncation=True, padding=True,
                   return_tensors="tf", max_length=128)
dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), train_labels)).batch(16)

model_bert.compile(optimizer=tf.keras.optimizers.Adam(2e-5),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_bert.fit(dataset, epochs=3)
```

---

---

[來源: ch16 | 類型: cheatsheet] ## 4. 常見陷阱

- **`mask_zero=True` 要在 Embedding 層設定**：告訴後續層忽略填充 token，不然 LSTM/Attention 會把 padding 也計算進去。
- **Stateful RNN 要手動 reset state**：`model.reset_states()` 要在每個文件/序列開始時呼叫，批次大小必須固定。
- **Transformer 不包含歸納偏置（位置）**：`Positional Encoding` 是必要的，否則模型不知道 token 的順序。
- **`TextVectorization.adapt()` 只能看訓練集**：測試集的罕見詞會映射到 `[UNK]` token，這是正常行為。

---

---

[來源: ch16 | 類型: cheatsheet] ```
NLP 任務選模型：
├── 文字分類/情感分析       → 用 HuggingFace BERT fine-tune（首選）
│                            → 或 TextVec + Embedding + Bidirectional LSTM
├── 文字生成                → HuggingFace GPT-2/GPT-3.5（API）
├── 機器翻譯/摘要            → HuggingFace seq2seq 模型（T5/mBART）
├── 命名實體識別（NER）      → HuggingFace pipeline("ner")
└── 從頭訓練（罕見）         → Transformer + 大量資料 + 大 GPU

Attention 類型：
├── Self-Attention (Encoder)    → 理解句子上下文（BERT）
├── Causal Self-Attention (Decoder) → 生成任務（GPT）
└── Cross-Attention             → seq2seq（翻譯、摘要）
```

---

[來源: ch16 | 類型: handout] # 課程講義：使用 RNN 與 Attention 的 NLP (Chapter 16)

自然語言是人類最複雜的資訊載體，而本章帶你用深度學習征服它。從**字元級語言模型**生成莎士比亞風格文字，到情感分析、神經機器翻譯，再到**Attention 機制**——現代 GPT、BERT 等大語言模型的核心。理解本章，你將掌握現代 NLP 的技術基礎。

---

---

[來源: ch16 | 類型: handout] ### 理論背景

**字元級語言模型**：學習在給定前 $t$ 個字元後，預測第 $t+1$ 個字元的條件機率：

$$P(\text{char}_{t+1} \mid \text{char}_1, \ldots, \text{char}_t)$$

**TextVectorization**：將原始文字轉換為整數序列（字元ID或詞彙ID）。

**Embedding 層**：將離散的字元ID映射到連續的稠密向量：

$$\text{embedding}(i) = \mathbf{E}[i, :] \in \mathbb{R}^d$$

Embedding 層的優點：相似的字元/詞彙在嵌入空間中位置相近，且向量可學習。

**文字採樣策略（Temperature）**：

---

[來源: ch16 | 類型: handout] hbf{E}[i, :] \in \mathbb{R}^d$$

Embedding 層的優點：相似的字元/詞彙在嵌入空間中位置相近，且向量可學習。

**文字採樣策略（Temperature）**：

$$P(\text{char}_k \propto) = \frac{\exp(\log p_k / T)}{\sum_j \exp(\log p_j / T)}$$

- $T = 1$：正常機率採樣
- $T < 1$：更保守（偏好高機率字元）
- $T > 1$：更多樣、創意（但可能亂碼）

---

[來源: ch16 | 類型: handout] ### 核心代碼

```python
import tensorflow as tf
import numpy as np

---

[來源: ch16 | 類型: handout] # 下載莎士比亞文本
shakespeare_url = "https://homl.info/shakespeare"
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

---

[來源: ch16 | 類型: handout] # 字元級向量化
text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                    standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]
encoded -= 2  # 移除 <PAD> 和 <UNK>
n_tokens = text_vec_layer.vocabulary_size() - 2  # 約 39 個字元

---

[來源: ch16 | 類型: handout] # 建立資料集（滑動視窗）
def to_dataset(sequence, length, shuffle=False, seed=42, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(10_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda w: (w[:, :-1], w[:, 1:])).prefetch(1)

---

[來源: ch16 | 類型: handout] d=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda w: (w[:, :-1], w[:, 1:])).prefetch(1)

train_set = to_dataset(encoded[:1_000_000], length=100, shuffle=True)

---

[來源: ch16 | 類型: handout] # Char-RNN 模型
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens + 2, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens + 2, activation="softmax")
])
model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy")

---

[來源: ch16 | 類型: handout] # 文字生成（溫度採樣）
def next_char(text, temperature=1):
    y_proba = model.predict([text_vec_layer([text])])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]  # +2 for offset

---

[來源: ch16 | 類型: handout] ogits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]  # +2 for offset

def generate_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text
```

---

[來源: ch16 | 類型: handout] ### 補充練習 1

**理論題：** 語言模型的「困惑度 (Perplexity)」定義為：

$$\text{Perplexity} = 2^{H} = 2^{-\frac{1}{N}\sum_t \log_2 P(w_t \mid w_{<t})}$$

困惑度越低代表模型越好。若字元集大小為 39，一個「隨機猜測」的模型困惑度是多少？

**實作題：** 分別用 `temperature=0.5`、`1.0`、`2.0` 生成 200 個字元的莎士比亞風格文字，比較三種溫度下文字的多樣性和可讀性。

---

---

[來源: ch16 | 類型: handout] ### 理論背景

**Stateful RNN**：跨批次保留隱藏狀態（上一個批次的最終狀態傳給下一個批次的初始狀態）。適合非常長的序列（如音訊、書籍文本），讓 RNN 能學習跨批次的長期模式。

設定要點：
1. `stateful=True` 建立帶狀態的 RNN 層
2. 每個 epoch 結束後需要 `model.reset_states()`
3. 批次必須**有序且不打亂**（每個批次繼接上一個）

**情感分析 (Sentiment Analysis)**：文字分類問題，常用於電影評論、社交媒體情緒分析。

**Masking**：序列長度不一時，需要填充（Padding）到相同長度。`Masking` 層告訴模型哪些位置是填充，不應計入損失計算。

---

[來源: ch16 | 類型: handout] # 情感分析：IMDB 資料集
import tensorflow_datasets as tfds

raw_train_ds = tfds.load("imdb_reviews", split="train", as_supervised=True)
raw_test_ds  = tfds.load("imdb_reviews", split="test",  as_supervised=True)

---

[來源: ch16 | 類型: handout] # 詞彙級向量化
vocab_size = 1000
max_length  = 600

text_vec = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_sequence_length=max_length
)
text_vec.adapt(raw_train_ds.map(lambda reviews, labels: reviews))

---

[來源: ch16 | 類型: handout] # 帶 Masking 的情感分析模型
tf.random.set_seed(42)
sentiment_model = tf.keras.Sequential([
    text_vec,
    tf.keras.layers.Embedding(input_dim=vocab_size + 2, output_dim=16,
                               mask_zero=True),  # mask_zero=True 自動產生 mask
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
sentiment_model.compile(optimizer="nadam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"])
```

---

[來源: ch16 | 類型: handout] ### 補充練習 2

**理論題：** `Embedding(mask_zero=True)` 會自動生成 mask，讓後續的 LSTM 忽略填充位置。若不使用 Masking，LSTM 將填充位置的 `0` 視為真實輸入，這對模型有何影響？

**實作題：** 在 IMDB 資料集上比較三種模型的驗證集準確率：(1) `LSTM(32)`；(2) `Bidirectional(LSTM(32))`；(3) `LSTM(32)` + `LSTM(16)`（堆疊）。

---

---

[來源: ch16 | 類型: handout] ### 理論背景

**Encoder-Decoder 架構**（序列到序列）：

- **Encoder**：讀取輸入序列（源語言），將整個序列壓縮為一個固定長度的**上下文向量 (Context Vector)** $\mathbf{c}$（最後時間步的隱藏狀態）
- **Decoder**：以 $\mathbf{c}$ 為初始狀態，逐步生成目標序列（目標語言）

**訓練策略（Teacher Forcing）**：
- 訓練時：Decoder 的輸入是**真實的**目標詞彙（即使前一步預測錯誤）
- 推理時：Decoder 的輸入是**自己前一步的預測**
- 使用特殊標記：`startofseq` 作為解碼起始符，`endofseq` 作為結束符

---

[來源: ch16 | 類型: handout] **真實的**目標詞彙（即使前一步預測錯誤）
- 推理時：Decoder 的輸入是**自己前一步的預測**
- 使用特殊標記：`startofseq` 作為解碼起始符，`endofseq` 作為結束符

**限制**：固定長度的 Context Vector 是瓶頸——輸入序列越長，資訊越容易丟失。這正是 **Attention 機制**誕生的動機。

---

[來源: ch16 | 類型: handout] # Encoder-Decoder NMT 架構
tf.random.set_seed(42)

---

[來源: ch16 | 類型: handout] # 準備雙語語料對 (英語-西班牙語)
import urllib.request, os
url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"

---

[來源: ch16 | 類型: handout] # （假設資料已下載）

encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

---

[來源: ch16 | 類型: handout] # Encoder
enc_text_vec = tf.keras.layers.TextVectorization(max_tokens=1000,
                                                  output_sequence_length=50)
enc_embed = tf.keras.layers.Embedding(input_dim=1002, output_dim=128, mask_zero=True)
enc_lstm = tf.keras.layers.LSTM(512, return_state=True)

---

[來源: ch16 | 類型: handout] ut_dim=1002, output_dim=128, mask_zero=True)
enc_lstm = tf.keras.layers.LSTM(512, return_state=True)

enc_encoded = enc_text_vec(encoder_inputs)
enc_emb = enc_embed(enc_encoded)
enc_out, enc_h, enc_c = enc_lstm(enc_emb)

---

[來源: ch16 | 類型: handout] # enc_h, enc_c 是 Context Vector（傳給 Decoder 的初始狀態）

---

[來源: ch16 | 類型: handout] # Decoder（訓練時用 Teacher Forcing）
dec_text_vec = tf.keras.layers.TextVectorization(max_tokens=1000,
                                                  output_sequence_length=50)
dec_embed = tf.keras.layers.Embedding(input_dim=1002, output_dim=128, mask_zero=True)
dec_lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
dec_dense = tf.keras.layers.Dense(1000, activation="softmax")

---

[來源: ch16 | 類型: handout] urn_sequences=True, return_state=True)
dec_dense = tf.keras.layers.Dense(1000, activation="softmax")

dec_encoded = dec_text_vec(decoder_inputs)
dec_emb = dec_embed(dec_encoded)
dec_out, _, _ = dec_lstm(dec_emb, initial_state=[enc_h, enc_c])  # 注入 Context
dec_proba = dec_dense(dec_out)

---

[來源: ch16 | 類型: handout] _, _ = dec_lstm(dec_emb, initial_state=[enc_h, enc_c])  # 注入 Context
dec_proba = dec_dense(dec_out)

model_nmt = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
                           outputs=dec_proba)
```

---

[來源: ch16 | 類型: handout] ### 補充練習 3

**理論題：** 在 Encoder-Decoder 架構中，「Context Vector 瓶頸」問題是什麼？隨著輸入序列長度從 10 個詞增加到 50 個詞，翻譯品質如何變化？Attention 機制如何解決此問題？

**實作題：** 實作貪婪解碼（Greedy Decoding）函式：給定一個英文句子，循環呼叫 Decoder，每步取機率最高的詞彙，直到輸出 `endofseq` 或達到最大長度。測試 10 個英文句子的翻譯結果。

---

---

[來源: ch16 | 類型: handout] ## 4. Attention 機制與 Transformer

---

[來源: ch16 | 類型: handout] ### 理論背景

**Attention 機制**：讓 Decoder 在每個時間步能夠「關注」Encoder 輸出序列的不同部分，而不只依賴固定的 Context Vector。

$$\text{attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

- **Query (Q)**：當前 Decoder 隱藏狀態
- **Key (K)**：所有 Encoder 輸出
- **Value (V)**：所有 Encoder 輸出
- $\sqrt{d_k}$：縮放因子，避免內積過大（Scaled Dot-Product Attention）

---

[來源: ch16 | 類型: handout] 所有 Encoder 輸出
- **Value (V)**：所有 Encoder 輸出
- $\sqrt{d_k}$：縮放因子，避免內積過大（Scaled Dot-Product Attention）

**Multi-Head Attention**：在多個子空間平行執行 Attention，再拼接：

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O$$

**Positional Encoding**：Transformer 無序列順序感知，需注入位置資訊：

---

[來源: ch16 | 類型: handout] head}_1, \ldots, \text{head}_h) \mathbf{W}^O$$

**Positional Encoding**：Transformer 無序列順序感知，需注入位置資訊：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

---

[來源: ch16 | 類型: handout] ### 核心代碼

```python
tf.random.set_seed(42)

---

[來源: ch16 | 類型: handout] # 使用 Keras 的 MultiHeadAttention 層
encoder_inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int64)
decoder_inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int64)

embed_size = 128

---

[來源: ch16 | 類型: handout] # Positional Encoding（自訂層）
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, embed_size, **kwargs):
        super().__init__(**kwargs)
        p, i = np.meshgrid(np.arange(max_length),
                           2 * np.arange(embed_size // 2))
        angles = p / 10_000 ** (i / embed_size)
        sin_cos = np.empty((1, max_length, embed_size))
        sin_cos[0, :, ::2]  = np.sin(angles).T
        sin_cos[0, :, 1::2] = np.cos(angles).T
        self.positional_encoding = tf.constant(sin_cos.astype(np.float32))

---

[來源: ch16 | 類型: handout] 1::2] = np.cos(angles).T
        self.positional_encoding = tf.constant(sin_cos.astype(np.float32))

def call(self, X):
        return X + self.positional_encoding[:, :tf.shape(X)[1], :]

---

[來源: ch16 | 類型: handout] # Transformer Encoder Block
enc_emb = tf.keras.layers.Embedding(input_dim=1002, output_dim=embed_size)(encoder_inputs)
enc_pos = PositionalEncoding(max_length=512, embed_size=embed_size)(enc_emb)

---

[來源: ch16 | 類型: handout] d_size)(encoder_inputs)
enc_pos = PositionalEncoding(max_length=512, embed_size=embed_size)(enc_emb)

mha = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=embed_size // 8)
enc_out = mha(enc_pos, enc_pos)  # Self-Attention
enc_out = tf.keras.layers.LayerNormalization()(enc_out + enc_pos)
enc_out = tf.keras.layers.Dense(256, activation="relu")(enc_out)
enc_out = tf.keras.layers.Dense(embed_size)(enc_out)
enc_out = tf.keras.layers.LayerNormalization()(enc_out + enc_pos)
```

---

[來源: ch16 | 類型: handout] ### 補充練習 4

**理論題：** Scaled Dot-Product Attention 為何需要除以 $\sqrt{d_k}$？若不縮放，當 $d_k$ 很大時，Softmax 的梯度會有什麼問題？

**實作題：** 用 `tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)` 建立一個簡單的 Transformer Encoder（2 層），在 IMDB 情感分析任務上訓練，比較其與 LSTM 的收斂速度和最終準確率。

---

---

[來源: ch16 | 類型: handout] ## 結論

NLP 的深度學習演進：

- **Char-RNN**：文字生成的入門，理解語言模型訓練
- **Embedding + LSTM**：情感分析的標準做法
- **Encoder-Decoder**：序列到序列（機器翻譯）的基礎架構
- **Attention + Transformer**：現代 LLM（GPT、BERT）的基石

下一章（Ch17）轉向生成模型：自動編碼器、VAE 和 GAN。

---

---

[來源: ch16 | 類型: handout] ## 課後作業

**作業：建立一個多語言情感分析器**

1. 使用 `tensorflow_datasets.load("imdb_reviews")` 訓練一個帶 Masking 的 LSTM 情感分析模型（Embedding + LSTM + Dense），在驗證集達到至少 85% 準確率。

2. 比較以下三種輸入表示方式的效果：
   - `TextVectorization(max_tokens=1000)`（詞彙索引）
   - `TextVectorization(max_tokens=10000)`（更大詞彙表）
   - Bidirectional LSTM

3. 用你的模型預測 5 條自己寫的英文影評，判斷其情感正負面。

---

[來源: ch16 | 類型: tutorial] [標題: NLP 與注意力機制完整指南：情感分析、機器翻譯、多頭注意力 | 描述: 深入 NLP 深度學習：字符級 RNN 文字生成、溫度採樣、Stateful RNN、詞嵌入情感分析、Encoder-Decoder 機器翻譯、Bahdanau 注意力機制，以及 Keras 多頭注意力層。 | 關鍵字: Python, NLP, RNN, LSTM, 注意力機制, 機器翻譯, 情感分析, Transformer, Keras, TensorFlow]
# NLP 與注意力機制：從文字生成到 Transformer 基礎

從 GPT 到 BERT，現代 NLP 的核心都是**注意力機制（Attention Mechanism）**。本教學帶你走過 NLP 深度學習的完整路徑：字符級 RNN 文字生成、詞嵌入情感分析、Encoder-Decoder 翻譯系統，最終理解多頭注意力——Transformer 的核心組件。

---

[來源: ch16 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **字符級 RNN** 學習字符序列的統計規律，可生成新文字（但不理解語義）
- **溫度（Temperature）** 控制生成的創意度：低溫 → 保守（重複）；高溫 → 創意（隨機）
- **詞嵌入（Word Embedding）** 將詞語映射到密集向量空間，語義相近的詞距離相近
- **注意力機制** 讓解碼器在生成每個詞時，能「關注」源序列中最相關的部分
- **`MultiHeadAttention`** 是 Keras 內建的多頭注意力層，是 Transformer 架構的核心

---

---

[來源: ch16 | 類型: tutorial] ## 字符級 RNN 文字生成

💡 **實際應用情境：** 用莎士比亞文集訓練字符級 RNN，讓它自動生成莎士比亞風格的文字。這雖然是玩具範例，但背後的原理（學習序列的統計分佈）正是語言模型的基礎。

---

[來源: ch16 | 類型: tutorial] ### 範例 1: 資料準備與字符編碼

```python
import tensorflow as tf
import numpy as np

---

[來源: ch16 | 類型: tutorial] # 示範文字（實際使用時替換為莎士比亞文集）
shakespeare_text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them."""

---

[來源: ch16 | 類型: tutorial] # 字符集和編碼
tokenizer = tf.keras.layers.TextVectorization(
    split="character",          # 字符級分詞
    standardize="lower",        # 統一轉為小寫
)
tokenizer.adapt([shakespeare_text])

---

[來源: ch16 | 類型: tutorial] # 字符到索引的映射
char_vocab = tokenizer.get_vocabulary()
n_tokens = len(char_vocab)
print(f"字符數量（詞彙表大小）: {n_tokens}")  # 約 40~50 個不同字符

---

[來源: ch16 | 類型: tutorial] # 編碼整個文本
encoded = tokenizer([shakespeare_text])[0]  # 1D 整數陣列

---

[來源: ch16 | 類型: tutorial] # 建立訓練資料：輸入序列 → 目標（下一個字符）
def create_sequences(encoded_text: tf.Tensor, seq_length: int,
                     step: int = 1):
    """從編碼文字建立訓練序列"""
    dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
    dataset = dataset.window(seq_length + 1, shift=step, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(seq_length + 1))
    dataset = dataset.map(lambda seq: (seq[:-1], seq[1:]))  # x, y（偏移一步）
    return dataset

---

[來源: ch16 | 類型: tutorial] th + 1))
    dataset = dataset.map(lambda seq: (seq[:-1], seq[1:]))  # x, y（偏移一步）
    return dataset

seq_length = 100
train_dataset = (
    create_sequences(encoded, seq_length)
    .shuffle(10000)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)
print(f"資料集準備完成，序列長度: {seq_length}")
```

**✅ 程式碼逐行解析：**

1. `split="character"`: 在字符級別分詞（而非詞語級別），每個字母/標點是一個 token
2. `window(seq_length + 1, shift=1)`: 滑動視窗——每次移動 1 步，建立重疊的訓練序列
3. `(seq[:-1], seq[1:])`: x 是前 n 個字符，y 是後 n 個字符（預測下一個字符）

---

[來源: ch16 | 類型: tutorial] ### 範例 2: 字符級 RNN 模型

```python

---

[來源: ch16 | 類型: tutorial] # 字符級語言模型
char_model = tf.keras.Sequential([
    # Embedding 層：整數索引 → 密集向量（不需要 One-hot）
    tf.keras.layers.Embedding(
        input_dim=n_tokens,
        output_dim=16,            # 每個字符嵌入到 16 維向量
    ),
    tf.keras.layers.GRU(
        128,
        return_sequences=True,    # 需要預測每個時間步的下一個字符
    ),
    tf.keras.layers.Dense(n_tokens, activation="softmax")  # 輸出各字符的機率
])

char_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # y 是整數（字符索引）
    metrics=["accuracy"]
)
char_model.summary()
```

---

---

[來源: ch16 | 類型: tutorial] ## 溫度採樣

💡 **實際應用情境：** 寫詩用高溫（更有創意）；填寫法律文件用低溫（保守準確）。溫度是控制 AI 創意度的旋鈕。

---

[來源: ch16 | 類型: tutorial] ### 範例 3: 生成文字（溫度採樣）

```python
def generate_text(model, tokenizer: tf.keras.layers.TextVectorization,
                  seed_text: str, n_chars: int = 200,
                  temperature: float = 1.0) -> str:
    """使用訓練好的模型生成文字

temperature: < 1 更保守，> 1 更隨機/創意
    """
    char_vocab = tokenizer.get_vocabulary()
    index_to_char = {i: c for i, c in enumerate(char_vocab)}

result = seed_text
    for _ in range(n_chars):
        # 編碼當前文字（取最後 seq_length 個字符）
        encoded = tokenizer([result[-seq_length:]])  # (1, seq_length)

---

[來源: ch16 | 類型: tutorial] # 編碼當前文字（取最後 seq_length 個字符）
        encoded = tokenizer([result[-seq_length:]])  # (1, seq_length)

# 獲取下一個字符的機率分佈
        logits = model(encoded)[:, -1, :]  # 最後時間步的輸出 (1, n_tokens)

# 溫度縮放（在 softmax 之前）
        scaled_logits = logits / temperature
        probs = tf.nn.softmax(scaled_logits).numpy()[0]

# 根據機率採樣（而非貪婪取最大值）
        next_char_id = np.random.choice(len(char_vocab), p=probs)
        result += index_to_char[next_char_id]

return result

---

[來源: ch16 | 類型: tutorial] # text_conservative = generate_text(char_model, tokenizer, "To be", temperature=0.3)

---

[來源: ch16 | 類型: tutorial] # text_balanced     = generate_text(char_model, tokenizer, "To be", temperature=1.0)

---

[來源: ch16 | 類型: tutorial] # text_creative     = generate_text(char_model, tokenizer, "To be", temperature=2.0)
print("溫度效果：低溫(0.3) = 保守重複；中溫(1.0) = 平衡；高溫(2.0) = 隨機創意")
```

**✅ 程式碼逐行解析：**

1. `logits / temperature`: 溫度 < 1 使分佈更尖銳（集中在高機率字符）；> 1 使分佈更平坦（更均勻）
2. `np.random.choice(len(vocab), p=probs)`: 按機率採樣（非貪婪）——保留一定的隨機性

**🎯 重點摘要:**

- 貪婪解碼（每次選最高機率）會導致文字單調重複
- `temperature=1.0` 是標準採樣；`0.5~0.8` 在創意和準確性間平衡

---

---

[來源: ch16 | 類型: tutorial] ### 範例 4: LSTM 情感分類（使用預訓練嵌入）

```python
import tensorflow_datasets as tfds  # 需要安裝：pip install tensorflow-datasets

---

[來源: ch16 | 類型: tutorial] # raw_train_data, raw_test_data = tfds.load(

---

[來源: ch16 | 類型: tutorial] #    "imdb_reviews", split=["train", "test"], as_supervised=True

---

[來源: ch16 | 類型: tutorial] # 文字向量化
MAX_TOKENS = 10000  # 詞彙表大小
MAX_LEN = 200       # 最大序列長度

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_sequence_length=MAX_LEN
)

---

[來源: ch16 | 類型: tutorial] # 情感分析模型（使用可學習的詞嵌入）
sentiment_model = tf.keras.Sequential([
    vectorizer,
    tf.keras.layers.Embedding(
        MAX_TOKENS + 2,   # +2 for padding [0] and OOV [1]
        64,               # 嵌入維度
        mask_zero=True    # 告訴模型哪些是 padding
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32)
    ),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")  # 二元分類
])

---

[來源: ch16 | 類型: tutorial] u"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")  # 二元分類
])

sentiment_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
sentiment_model.summary()
```

**🎯 重點摘要:**

- `Embedding` 層把整數索引映射到密集向量（可學習），比 One-hot 效率高很多
- `mask_zero=True`：將 0（padding）標記為遮罩，後續層忽略這些位置
- 雙向 LSTM 同時考慮前後文，適合需要理解完整句子的任務（但不適合實時任務）

---

---

[來源: ch16 | 類型: tutorial] ### 範例 5: 簡易 Encoder-Decoder 架構

```python

---

[來源: ch16 | 類型: tutorial] # Encoder-Decoder (seq2seq) 架構：用於翻譯、摘要等任務

---

[來源: ch16 | 類型: tutorial] # Encoder：壓縮源語言序列為 context vector

---

[來源: ch16 | 類型: tutorial] # Decoder：根據 context vector 生成目標語言序列

encoder_vocab_size = 5000   # 源語言詞彙表
decoder_vocab_size = 5000   # 目標語言詞彙表
embed_dim = 64
latent_dim = 256

---

[來源: ch16 | 類型: tutorial] # ── Encoder ──
encoder_inputs = tf.keras.Input(shape=(None,), name="encoder_input")
encoder_embed   = tf.keras.layers.Embedding(encoder_vocab_size, embed_dim,
                                             mask_zero=True)(encoder_inputs)
_, state_h, state_c = tf.keras.layers.LSTM(
    latent_dim,
    return_state=True  # 同時回傳隱藏狀態 h 和記憶體狀態 c
)(encoder_embed)
encoder_states  = [state_h, state_c]  # 這是 context vector

---

[來源: ch16 | 類型: tutorial] # ── Decoder（訓練時）──
decoder_inputs = tf.keras.Input(shape=(None,), name="decoder_input")
decoder_embed   = tf.keras.layers.Embedding(decoder_vocab_size, embed_dim,
                                             mask_zero=True)(decoder_inputs)
decoder_lstm    = tf.keras.layers.LSTM(
    latent_dim,
    return_sequences=True,  # 輸出每個時間步
    return_state=True
)
decoder_outputs, _, _ = decoder_lstm(
    decoder_embed,
    initial_state=encoder_states  # 用 encoder 的最終狀態初始化
)
decoder_dense   = tf.keras.layers.Dense(decoder_vocab_size, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

---

[來源: ch16 | 類型: tutorial] # 訓練模型（Teacher Forcing：訓練時輸入正確的前一個詞）
training_model = tf.keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)
training_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
training_model.summary()
```

**✅ 程式碼逐行解析：**

1. `return_state=True`: LSTM 同時回傳 `(output, state_h, state_c)`——encoder 只需要 state（丟棄 output）
2. `initial_state=encoder_states`: 用 encoder 的最終隱藏狀態初始化 decoder（這就是「context vector」）
3. Teacher Forcing：訓練時 decoder 的輸入是正確的目標序列（而非上一步的預測），加速收斂

---

---

[來源: ch16 | 類型: tutorial] ## 注意力機制

💡 **實際應用情境：** 翻譯「我愛台灣的夜市文化」時，解碼器在生成「night markets」時應該更關注「夜市」；生成「culture」時應該關注「文化」——這就是注意力機制的直觀含義。

---

[來源: ch16 | 類型: tutorial] ```python
class BahdanauAttention(tf.keras.layers.Layer):
    """Bahdanau（加法）注意力機制"""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)  # 處理 encoder 輸出
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)  # 處理 decoder 狀態
        self.V  = tf.keras.layers.Dense(1)                       # 計算注意力分數

    def call(self, decoder_state, encoder_outputs):
        # decoder_state: (batch, latent_dim)
        # encoder_outputs: (batch, seq_len, latent_dim)

        # 擴展維度以便廣播
        decoder_state_expanded = tf.expand_dims(decoder_state, 1)
        # (batch, 1, latent_dim)

        # 計算注意力分數（每個 encoder 時間步的相關性）
        score = self.V(
            tf.nn.tanh(
                self.W1(encoder_outputs) + self.W2(decoder_state_expanded)
            )
        )  # (batch, seq_len, 1)

        # Softmax 得到注意力權重（加總為 1）
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len, 1)

        # 加權求和 encoder 輸出（context vector）
        context_vector = tf.reduce_sum(
            attention_weights * encoder_outputs, axis=1
        )  # (batch, latent_dim)

        return context_vector, attention_weights

print("Bahdanau 注意力機制定義完成！")
```

---

[來源: ch16 | 類型: tutorial] tch, latent_dim)

        return context_vector, attention_weights

print("Bahdanau 注意力機制定義完成！")
```

---

---

[來源: ch16 | 類型: tutorial] ### 範例 7: Keras MultiHeadAttention 層

```python

---

[來源: ch16 | 類型: tutorial] # Keras 內建 Multi-Head Attention（Transformer 的核心）
batch_size = 2
seq_len = 10
embed_dim = 64

---

[來源: ch16 | 類型: tutorial] # 建立 MultiHeadAttention 層
mha = tf.keras.layers.MultiHeadAttention(
    num_heads=8,              # 注意力頭的數量
    key_dim=embed_dim // 8,   # 每個頭的 key 維度 = embed_dim / num_heads
)

---

[來源: ch16 | 類型: tutorial] # 自注意力（Self-Attention）：query, key, value 都來自同一個序列
x = tf.random.uniform((batch_size, seq_len, embed_dim))
output, attention_scores = mha(
    query=x,
    value=x,
    key=x,
    return_attention_scores=True  # 回傳注意力權重（可視化用）
)
print(f"輸出形狀: {output.shape}")           # (2, 10, 64)
print(f"注意力分數形狀: {attention_scores.shape}")  # (2, 8, 10, 10)

---

[來源: ch16 | 類型: tutorial] # Transformer Encoder Block
def transformer_encoder_block(inputs, d_model: int, num_heads: int,
                                dff: int, dropout_rate: float = 0.1):
    """標準 Transformer Encoder 塊"""
    # Multi-Head Self-Attention
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(inputs, inputs)

# 殘差連接 + Layer Normalization
    x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

---

[來源: ch16 | 類型: tutorial] Layer Normalization
    x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

# 前饋神經網路（FFN）
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"),
        tf.keras.layers.Dense(d_model)
    ])
    ffn_output = ffn(x1)

# 殘差連接 + Layer Normalization
    x2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x1 + ffn_output)
    return x2

---

[來源: ch16 | 類型: tutorial] # 示範 Transformer Encoder
encoder_input = tf.keras.Input(shape=(seq_len, embed_dim))
encoder_output = transformer_encoder_block(encoder_input, d_model=64,
                                            num_heads=8, dff=256)
encoder_model = tf.keras.Model(encoder_input, encoder_output)
encoder_model.summary()
```

**✅ 程式碼逐行解析：**

1. `num_heads=8`: 8 個注意力頭各學習不同面向的依賴關係（語法、語義、位置等）
2. `query/key/value`: 自注意力中三者相同（來自同一個序列）；跨注意力中 query 來自 decoder，key/value 來自 encoder
3. `LayerNormalization` + 殘差連接：Transformer 的標準組件，解決梯度消失

**🎯 重點摘要:**

---

[來源: ch16 | 類型: tutorial] 來自 decoder，key/value 來自 encoder
3. `LayerNormalization` + 殘差連接：Transformer 的標準組件，解決梯度消失

**🎯 重點摘要:**

- 多頭注意力的複雜度是 $O(n^2 d)$（n 是序列長度）——長序列時計算成本高
- BERT/GPT 等大型語言模型都基於 Transformer 架構，而 `MultiHeadAttention` 就是它的核心

---

---

[來源: ch16 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: 為什麼要用溫度採樣而非直接取最高機率？**

A: 貪婪解碼（取最高機率）會導致輸出單調重複（如「the the the」）。採樣保留多樣性，而溫度控制多樣性的程度。對話系統用中溫；摘要/翻譯用低溫。

**Q2: Encoder-Decoder 訓練時的 Teacher Forcing 是什麼？**

A: 訓練時，decoder 的輸入是真正的目標序列（而非上一步的預測），這讓訓練更穩定更快。推論時，decoder 只能使用自己的預測——兩種模式的不一致稱為「曝光偏差（Exposure Bias）」。

**Q3: Self-Attention vs Cross-Attention 的區別？**

A: Self-Attention：query/key/value 都來自同一序列，捕捉序列內部的依賴（BERT 的核心）。Cross-Attention：query 來自 decoder，key/value 來自 encoder，用於翻譯等 Seq2Seq 任務。

**Q4: 現在還有必要學 RNN/LSTM 嗎？**

---

[來源: ch16 | 類型: tutorial] Cross-Attention：query 來自 decoder，key/value 來自 encoder，用於翻譯等 Seq2Seq 任務。

**Q4: 現在還有必要學 RNN/LSTM 嗎？**

A: Transformer 在大多數 NLP 任務上已超越 RNN，但 RNN 仍有優勢：(1) 輸入序列非常長時記憶體效率更好；(2) 實時流式處理（Transformer 需要完整序列）；(3) 理解 Transformer 的基礎知識。

---

---

[來源: ch16 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #NLP #LSTM #注意力機制 #機器翻譯 #情感分析 #Transformer #MultiHeadAttention #Keras #深度學習 #程式設計 #教學 #DataScience #MachineLearning

---

[來源: ch17 | 類型: cheatsheet] # Ch17 速查表：Autoencoders, GANs & Diffusion Models

> **核心主旨**：生成模型家族 —— Autoencoder 學習壓縮表示，VAE 採樣生成，GAN 對抗生成，Diffusion 逐步去噪生成。

---

---

[來源: ch17 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Autoencoder | Encoder 壓縮 → Bottleneck → Decoder 重建，無監督學習表示 | 特徵萃取、去噪、異常偵測 |
| Denoising AE | 輸入加入雜訊，訓練重建乾淨影像，學習魯棒特徵 | 資料去噪、魯棒表示學習 |
| Sparse AE | Bottleneck 加 L1 正則，強迫稀疏啟動 | 更具解釋性的潛在表示 |
| Variational AE (VAE) | Encoder 輸出 μ 和 σ，用重參數化技巧取樣 z | 生成新樣本、插值 |
| KL Divergence Loss | VAE 的正則化項，讓潛在空間接近標準常態 | VAE 訓練必要損失 |
| GAN | Generator 造假 vs Discriminator 判真偽，對抗訓練 | 高品質影像生成 |
| DCGAN | GAN + 轉置卷積，用於影像生成 | 影像合成的基礎 GAN |
| Diffusion Model | 正向加噪 → 反向去噪，學習逐步還原資料 | 最先進的影像生成（DALL-E, SD） |


---

[來源: ch17 | 類型: cheatsheet] GAN + 轉置卷積，用於影像生成 | 影像合成的基礎 GAN |
| Diffusion Model | 正向加噪 → 反向去噪，學習逐步還原資料 | 最先進的影像生成（DALL-E, SD） |


---

---

[來源: ch17 | 類型: cheatsheet] | Keras API | 重點參數 | 用途 |
|-----------|---------|------|
| `tf.keras.layers.Dense` | `units=coding_size`, `activation="relu"` | AE 的 Encoder/Decoder 層 |
| `tf.keras.layers.Conv2DTranspose` | `filters=32`, `kernel_size=3`, `strides=2`, `padding="same"` | 卷積 AE 的 Decoder 上採樣 |
| `tf.keras.layers.UpSampling2D` | `size=(2, 2)` | 上採樣（不含可訓練參數） |
| `tf.keras.regularizers.l1(activity)` | `l1=1e-3` | 稀疏自編碼器的稀疏正則 |
| `model.add_loss()` | – | VAE 中添加 KL 散度損失 |
| `tf.keras.losses.binary_crossentropy` | – | AE/GAN 重建損失 |


---

[來源: ch17 | 類型: cheatsheet] del.add_loss()` | – | VAE 中添加 KL 散度損失 |
| `tf.keras.losses.binary_crossentropy` | – | AE/GAN 重建損失 |


---

---

[來源: ch17 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import tensorflow as tf
import numpy as np

---

[來源: ch17 | 類型: cheatsheet] # 基本 Stacked Autoencoder
coding_dim = 30

---

[來源: ch17 | 類型: cheatsheet] encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(coding_dim, activation="selu")
])
decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="selu", input_shape=[coding_dim]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])
autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(loss="binary_crossentrop

---

[來源: ch17 | 類型: cheatsheet] )
autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(loss="binary_crossentropy", optimizer="nadam")
autoencoder.fit(X_train, X_train, epochs=20, validation_data=(X_val, X_val))

---

[來源: ch17 | 類型: cheatsheet] # Denoising Autoencoder（加入高斯雜訊層）
denoising_ae = tf.keras.Sequential([
    tf.keras.layers.GaussianNoise(0.2),  # 只在訓練時有效
    encoder,
    decoder
])
denoising_ae.compile(loss="binary_crossentropy", optimizer="nadam")
denoising_ae.fit(X_train, X_train, epochs=20, validation_data=(X_val, X_val))

---

[來源: ch17 | 類型: cheatsheet] # Variational Autoencoder (VAE)
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean

---

[來源: ch17 | 類型: cheatsheet] # Encoder
codings_size = 10
inputs = tf.keras.layers.Input(shape=[28, 28])
Z = tf.keras.layers.Flatten()(inputs)
Z = tf.keras.layers.Dense(150, activation="selu")(Z)
codings_mean = tf.keras.layers.Dense(codings_size)(Z)
codings_log_var = tf.keras.layers.Dense(codings_size)(Z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(inputs=inputs, outputs=[codings_mean, codings_log_var, codings])

---

[來源: ch17 | 類型: cheatsheet] # Decoder
decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
x = tf.keras.layers.Dense(150, activation="selu")(decoder_inputs)
x = tf.keras.layers.Dense(28 * 28, activation="sigmoid")(x)
outputs = tf.keras.layers.Reshape([28, 28])(x)
variational_decoder = tf.keras.Model(inputs=decoder_inputs, outputs=outputs)

---

[來源: ch17 | 類型: cheatsheet] # VAE Model
class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

---

[來源: ch17 | 類型: cheatsheet] s):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

def call(self, inputs):
        codings_mean, codings_log_var, codings = self.encoder(inputs)
        reconstructions = self.decoder(codings)
        # KL 散度損失：讓 latent space 接近 N(0,1)
        kl_loss = -0.5 * tf.reduce_sum(
            1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
            axis=-1)
        self.add_loss(tf.reduce_mean(kl_loss) / (28 * 28))
        return reconstructions

---

[來源: ch17 | 類型: cheatsheet] axis=-1)
        self.add_loss(tf.reduce_mean(kl_loss) / (28 * 28))
        return reconstructions

vae = VariationalAutoEncoder(variational_encoder, variational_decoder)
vae.compile(loss="binary_crossentropy", optimizer="nadam")
vae.fit(X_train, X_train, epochs=20, validation_data=(X_val, X_val))

---

[來源: ch17 | 類型: cheatsheet] # 從 VAE 採樣生成新圖片
codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings).numpy()

---

[來源: ch17 | 類型: cheatsheet] # GAN（基本框架）
codings_size = 100
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    tf.keras.layers.Dense(150, activation="selu"),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

---

[來源: ch17 | 類型: cheatsheet] ,
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(150, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

---

[來源: ch17 | 類型: cheatsheet] e(1, activation="sigmoid")
])
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

gan = tf.keras.Sequential([generator, discriminator])
discriminator.trainable = False  # GAN 訓練時凍結 Discriminator
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

---

[來源: ch17 | 類型: cheatsheet] # GAN 訓練步驟
def train_gan(X_train, gan, generator, discriminator, batch_size=32, codings_size=100):
    half_batch = batch_size // 2
    # Phase 1: 訓練 Discriminator
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    X_real = X_train[idx]
    noise = tf.random.normal(shape=[half_batch, codings_size])
    X_fake = generator(noise)
    X_combined = tf.concat([X_real, X_fake], axis=0)
    y_combined = tf.concat([tf.ones((half_batch, 1)), tf.zeros((half_batch, 1))], axis=0)
    discriminator.trainable = True
    discriminator.train_on_batch(X_combined, y_combined)

---

[來源: ch17 | 類型: cheatsheet] axis=0)
    discriminator.trainable = True
    discriminator.train_on_batch(X_combined, y_combined)

# Phase 2: 訓練 Generator（通過 GAN，凍結 Discriminator）
    noise = tf.random.normal(shape=[batch_size, codings_size])
    y_gen = tf.ones((batch_size, 1))  # 希望 Discriminator 判為真
    discriminator.trainable = False
    gan.train_on_batch(noise, y_gen)
```

---

---

[來源: ch17 | 類型: cheatsheet] ## 4. 常見陷阱

- **GAN Mode Collapse**：Generator 只生成少數幾種輸出（多樣性消失）。對策：使用 minibatch discrimination 或 Wasserstein GAN。
- **GAN 訓練不穩定**：Discriminator 太強 → Generator 梯度消失；Discriminator 太弱 → Generator 學不到東西。平衡兩者是 GAN 的藝術。
- **VAE 生成圖片模糊**：重建損失（binary_crossentropy）和 KL 損失的權重比例影響品質；KL 太強 → 潛在空間太緊，重建差。
- **`GaussianNoise` 只在訓練時有效**：推論時自動關閉，不需手動處理。

---

---

[來源: ch17 | 類型: cheatsheet] ## 5. 決策指南

```
選哪種生成模型？
├── 學習資料的壓縮表示（特徵萃取） → Autoencoder
├── 資料去噪                         → Denoising AE
├── 異常偵測                          → AE（重建誤差作為異常分數）
├── 從連續潛在空間採樣生成            → VAE（生成效果較 AE 好）
├── 高品質影像/影片生成               → GAN（或 Diffusion 現代方法）
└── 最先進的影像生成（2023+）         → Diffusion Models (Stable Diffusion)

VAE Loss = Reconstruction Loss + β × KL Divergence
├── β 小 → 重建清晰但潛在空間雜亂
└── β 大 → 潛在空間規則但圖片模糊
```

---

[來源: ch17 | 類型: handout] # 課程講義：自動編碼器、GAN 與擴散模型 (Chapter 17)

生成模型（Generative Models）的目標是學習資料的分佈，讓模型能夠**生成從未見過的新樣本**。本章介紹三大生成模型技術：**自動編碼器（AE）**用於降維與去噪、**變分自動編碼器（VAE）**用於連續潛空間採樣、**生成對抗網路（GAN）**用於超高品質圖片生成，最後簡介擴散模型（Diffusion Models）的基本原理。

---

---

[來源: ch17 | 類型: handout] ### 理論背景

**自動編碼器 (Autoencoder)**：一種無監督學習的神經網路，以**重建輸入**為目標，學習資料的壓縮表示。

$$\text{Encoder}: \mathbf{x} \mapsto \mathbf{z} = f(\mathbf{x})$$
$$\text{Decoder}: \mathbf{z} \mapsto \hat{\mathbf{x}} = g(\mathbf{z})$$

訓練目標：最小化重建誤差：

$$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} \

|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\|^2$$


---

[來源: ch17 | 類型: handout] 誤差：

$$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} \

|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\|^2$$


**潛空間 (Latent Space) / 瓶頸 (Bottleneck)**：中間層維度 $d \ll$ 輸入維度 $n$，迫使模型學習最重要的特徵表示。

**主要應用**：

| 應用 | 說明 |
|------|------|
| 非線性降維 | 比 PCA 更強的表示能力 |
| 去噪 | 去噪自動編碼器學習移除雜訊 |
| 異常偵測 | 正常樣本重建誤差小，異常樣本大 |
| 預訓練 | 為分類器提供良好的初始特徵 |

---

[來源: ch17 | 類型: handout] ### 核心代碼

```python
import tensorflow as tf
import numpy as np

---

[來源: ch17 | 類型: handout] # 堆疊自動編碼器（Stacked Autoencoder）
tf.random.set_seed(42)

---

[來源: ch17 | 類型: handout] # Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full = X_train_full.astype("float32") / 255.

---

[來源: ch17 | 類型: handout] # 建立 Encoder + Decoder
stacked_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),   # 瓶頸層（30 維）
])

---

[來源: ch17 | 類型: handout] rs.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),   # 瓶頸層（30 維）
])

stacked_decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=[30]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),  # 還原為 784 維
    tf.keras.layers.Reshape([28, 28])
])

---

[來源: ch17 | 類型: handout] s.layers.Dense(28 * 28, activation="sigmoid"),  # 還原為 784 維
    tf.keras.layers.Reshape([28, 28])
])

stacked_ae = tf.keras.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy", optimizer="nadam")
stacked_ae.fit(X_train_full, X_train_full,  # 輸入 = 標籤 = 原圖
               epochs=20, validation_split=0.1)

---

[來源: ch17 | 類型: handout] # 重建效果
X_reconstructed = stacked_ae.predict(X_test[:5])
```

---

[來源: ch17 | 類型: handout] ### 補充練習 1

**理論題：** 自動編碼器與 PCA 的關係：線性自動編碼器（沒有活化函數、一個瓶頸層）與 PCA 等價嗎？若使用非線性活化函數，自動編碼器能學習到什麼 PCA 學不到的表示？

**實作題：** 訓練一個在 Fashion MNIST 上的堆疊自動編碼器，將瓶頸層維度分別設為 2、30、100，比較重建圖片的視覺品質，並繪製 2D 潛空間的散點圖（顏色=類別），觀察哪些類別在潛空間中聚集。

---

---

[來源: ch17 | 類型: handout] ### 理論背景

**去噪自動編碼器 (Denoising Autoencoder)**：

訓練時加入雜訊，要求輸出乾淨的重建：

$$\mathcal{L}_{\text{denoise}} = \

|\mathbf{x} - g(f(\tilde{\mathbf{x}}))\|^2$$


其中 $\tilde{\mathbf{x}} = \mathbf{x} + \varepsilon$（高斯雜訊）或隨機置零某些輸入（Dropout 雜訊）。

好處：迫使編碼器學習**魯棒的特徵表示**，而非簡單地記憶輸入。

**稀疏自動編碼器 (Sparse Autoencoder)**：

對瓶頸層的激活值加入稀疏懲罰（L1 正則化），迫使大多數神經元「靜默」：

---

[來源: ch17 | 類型: handout] 編碼器學習**魯棒的特徵表示**，而非簡單地記憶輸入。

**稀疏自動編碼器 (Sparse Autoencoder)**：

對瓶頸層的激活值加入稀疏懲罰（L1 正則化），迫使大多數神經元「靜默」：

$$\mathcal{L}_{\text{sparse}} = \text{Reconstruction Loss} + \alpha \sum_j

 |a_j|$$


用 `activity_regularizer=tf.keras.regularizers.L1(1e-4)` 實現。

---

[來源: ch17 | 類型: handout] # 去噪自動編碼器（使用 GaussianNoise 層）
tf.random.set_seed(42)

denoising_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.GaussianNoise(stddev=0.2),  # 訓練時加入雜訊，推理時自動關閉
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),
])

---

[來源: ch17 | 類型: handout] tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),
])

denoising_ae = tf.keras.Sequential([
    denoising_encoder,
    stacked_decoder  # 與前面共用 decoder（不含雜訊層）
])
denoising_ae.compile(loss="binary_crossentropy", optimizer="nadam")
denoising_ae.fit(X_train_full, X_train_full, epochs=10)

---

[來源: ch17 | 類型: handout] # 稀疏自動編碼器
sparse_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu",
                          activity_regularizer=tf.keras.regularizers.L1(1e-4))
])
```

---

[來源: ch17 | 類型: handout] ### 補充練習 2

**理論題：** 稀疏自動編碼器的潛空間向量通常大多數值接近 0，只有少數維度被激活。這與人腦神經元的「稀疏啟動」假說有何相似之處？這種稀疏性對下游分類任務有何好處？

**實作題：** 用去噪自動編碼器對 Fashion MNIST 測試圖片加入高斯雜訊（`stddev=0.5`），輸入給訓練好的模型，展示去噪前後的圖片對比。

---

---

[來源: ch17 | 類型: handout] ### 理論背景

**VAE (Variational Autoencoder)**：將潛空間從**點**（確定性向量）改為**分布**（高斯分布的均值和變異數）。

Encoder 輸出 $\boldsymbol{\mu}$ 和 $\log \boldsymbol{\sigma}^2$（對數變異數）。

**重參數化技巧 (Reparameterization Trick)**：

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

---

[來源: ch17 | 類型: handout] \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

這樣隨機採樣步驟的梯度可以流過 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$（不能對隨機採樣求梯度，但可以對參數求梯度）。

**VAE 損失函數**（ELBO 的負值）：

---

[來源: ch17 | 類型: handout] 步驟的梯度可以流過 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$（不能對隨機採樣求梯度，但可以對參數求梯度）。

**VAE 損失函數**（ELBO 的負值）：

$$\mathcal{L}_{\text{VAE}} = \underbrace{\text{Reconstruction Loss}}_{\text{重建品質}} + \underbrace{KL\left(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \| \mathcal{N}(\mathbf{0}, \mathbf{I})\right)}_{\text{正則化潛空間}}$$

KL 散度（封閉形式）：

---

[來源: ch17 | 類型: handout] boldsymbol{\sigma}^2) \| \mathcal{N}(\mathbf{0}, \mathbf{I})\right)}_{\text{正則化潛空間}}$$

KL 散度（封閉形式）：

$$KL = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**VAE 的關鍵特性**：潛空間平滑且連續，可以在潛空間中插值或採樣，生成新圖片。

---

[來源: ch17 | 類型: handout] # VAE 編碼器
tf.random.set_seed(42)

codings_size = 10

---

[來源: ch17 | 類型: handout] inputs = tf.keras.layers.Input(shape=[28, 28])
Z = tf.keras.layers.Flatten()(inputs)
Z = tf.keras.layers.Dense(150, activation="relu")(Z)
Z = tf.keras.layers.Dense(100, activation="relu")(Z)
codings_mean    = tf.keras.layers.Dense(codings_size)(Z)       # μ
codings_log_var = tf.keras.layers.Dense(codings_size)(Z)       # log(σ²)
codings         = Sampling()([codings_mean, codings_log_var])  # z = 

---

[來源: ch17 | 類型: handout] dings_size)(Z)       # log(σ²)
codings         = Sampling()([codings_mean, codings_log_var])  # z = μ + σε

---

[來源: ch17 | 類型: handout] # 自訂 Sampling 層（重參數化技巧）
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return mean + tf.exp(log_var / 2) * tf.random.normal(tf.shape(mean))

---

[來源: ch17 | 類型: handout] # VAE 損失（含 KL 散度）
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

---

[來源: ch17 | 類型: handout] , decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

def call(self, X):
        codings_mean, codings_log_var, codings = self.encoder(X)
        reconstructions = self.decoder(codings)
        # KL 散度損失
        kl_loss = -0.5 * tf.reduce_sum(
            1 + codings_log_var - tf.square(codings_mean) - tf.exp(codings_log_var),
            axis=-1
        )
        self.add_loss(tf.reduce_mean(kl_loss) / (28 * 28))
        return reconstructions

---

[來源: ch17 | 類型: handout] # 生成新圖片（從標準常態分布採樣）
codings_random = tf.random.normal([12, codings_size])
generated_images = vae_decoder.predict(codings_random)
```

---

[來源: ch17 | 類型: handout] ### 補充練習 3

**理論題：** VAE 的 KL 散度項有什麼正則化效果？若 KL 散度的係數設為 0（只保留重建損失），VAE 退化成什麼？若係數設得太大，潛空間的特性會有什麼問題（「後驗崩潰」）？

**實作題：** 在 Fashion MNIST 的 2D VAE（`codings_size=2`）訓練完成後，繪製潛空間的 2D 網格：對 $z_1, z_2 \in [-3, 3]$ 均勻採樣，解碼每個點，組成 $10 \times 10$ 的圖片矩陣，觀察潛空間的語意插值效果。

---

---

[來源: ch17 | 類型: handout] ### 理論背景

**GAN (Generative Adversarial Network)**：Generator 和 Discriminator 的雙人零和博弈。

- **Generator $G$**：從潛空間隨機雜訊 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 生成假圖片
- **Discriminator $D$**：判斷圖片是真實（1）還是假的（0）

訓練目標：

$$\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z} [\log(1 - D(G(\mathbf{z})))]$$

**交替訓練流程**：

---

[來源: ch17 | 類型: handout] [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z} [\log(1 - D(G(\mathbf{z})))]$$

**交替訓練流程**：

1. **訓練 Discriminator**：固定 Generator，用真假資料更新 D（最大化分辨能力）
2. **訓練 Generator**：固定 Discriminator，更新 G（欺騙 D）

**GAN 的訓練難點**：

- **模式崩潰 (Mode Collapse)**：Generator 只生成少數幾種樣本
- **訓練不穩定**：D 太強或太弱都會使訓練崩潰
- **解法**：Wasserstein GAN（WGAN）、漸進式訓練（ProGAN）、歸一化技術

---

[來源: ch17 | 類型: handout] ### 核心代碼

```python
tf.random.set_seed(42)

codings_size = 30

---

[來源: ch17 | 類型: handout] # Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=[codings_size]),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

---

[來源: ch17 | 類型: handout] # Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # 真/假
])
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

---

[來源: ch17 | 類型: handout] # GAN（訓練 Generator 時，Discriminator 凍結）
discriminator.trainable = False
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

---

[來源: ch17 | 類型: handout] # 交替訓練
def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            # 訓練 Discriminator
            noise = tf.random.normal([batch_size, codings_size])
            fake_images = generator(noise, training=True)
            X_combined = tf.concat([X_batch, fake_images], axis=0)
            y_combined  = tf.constant([[1.]] * batch_size + [[0.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_combined, y_combined)
            # 訓練 Generator
            noise = tf.random.normal([batch_size, codings_size])
            y_mislead = tf.ones([batch_size, 1])  # 標記為「真實」來欺騙 D
            discriminator.trainable = False
            gan.train_on_batch(noise, y_mislead)
```

---

[來源: ch17 | 類型: handout] ### 補充練習 4

**理論題：** GAN 的「均衡點」在哪裡？理想情況下，訓練結束時 Discriminator 的輸出是什麼（對真實和假圖片各為多少）？這個均衡點是穩定的嗎？

**實作題：** 訓練 Fashion MNIST GAN，每 10 個 epoch 保存 Generator 生成的 10 張圖片，製成動畫或拼圖，觀察生成品質隨訓練進行的改善（注意模式崩潰的跡象）。

---

---

[來源: ch17 | 類型: handout] ## 結論

生成模型的三大典範：

- **AE（自動編碼器）**：確定性壓縮，適合去噪和降維；不適合生成多樣樣本
- **VAE**：機率性潛空間，平滑可插值；生成圖片略模糊但多樣
- **GAN**：對抗訓練，生成品質最高（可達照片級）；訓練不穩定
- **Diffusion**：現代 SOTA，透過迭代去噪生成高品質圖片（Stable Diffusion、DALL-E 2）

下一章（Ch18）轉向強化學習，讓 Agent 在環境中學習最優策略。

---

---

[來源: ch17 | 類型: handout] ## 課後作業

**作業：潛空間操作實驗**

在 Fashion MNIST 上訓練 VAE（`codings_size=10`）：

1. 視覺化**潛空間插值**：找到一件T恤（類別0）和一件連衣裙（類別3）在潛空間的均值 $\boldsymbol{\mu}$，在兩點之間線性插值（10個均勻步驟），解碼每個插值點，觀察圖片的「變形」過程。

2. 實作**潛空間算術**：計算某個類別的平均潛空間向量（如所有「鞋子」的均值），將另一個樣本的潛空間向量加上這個方向，觀察解碼後的圖片是否呈現鞋子的特徵。

3. **對比 AE vs VAE 的潛空間**：訓練相同架構但潛空間為點（非分布）的 AE，比較 AE 和 VAE 的潛空間在 2D（`codings_size=2`）下的散點圖分布，解釋為何 VAE 的潛空間更「均勻」。

---

[來源: ch17 | 類型: tutorial] [標題: 自動編碼器、GAN 與擴散模型完整指南：生成式 AI 原理與實作 | 描述: 深入生成式 AI：堆疊自動編碼器、去噪自動編碼器、稀疏自動編碼器、變分自動編碼器（VAE 與重參數化技巧）、生成對抗網路（GAN 交替訓練）與現代擴散模型的原理。 | 關鍵字: Python, 自動編碼器, VAE, GAN, 生成式AI, 擴散模型, Keras, TensorFlow, 深度學習, 降維]
# 自動編碼器、GAN 與擴散模型：生成式 AI 從零開始

Midjourney、Stable Diffusion、DALL-E——這些令人驚嘆的圖像生成 AI 背後都依賴本章的技術。從最基礎的**自動編碼器（Autoencoder）**到**變分自動編碼器（VAE）**、**生成對抗網路（GAN）**，最後認識現代的**擴散模型（Diffusion Model）**——帶你從原理到實作，理解生成式 AI 的本質。

---

[來源: ch17 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **自動編碼器**強制網路學習資料的**壓縮表示（潛在空間）**，捕捉最重要的特徵
- **VAE** 的核心是**重參數化技巧（Reparameterization Trick）**：讓採樣操作可微分，使梯度能反向傳播
- **GAN** 的訓練是博弈（Generator vs Discriminator）：困難之處在於維持二者的平衡
- **模式崩潰（Mode Collapse）** 是 GAN 的主要問題：Generator 只生成幾種模式，忽略資料的多樣性
- 現代**擴散模型**逐步加噪然後反向去噪，效果優於 GAN，且訓練更穩定

---

---

[來源: ch17 | 類型: tutorial] ## 堆疊自動編碼器

💡 **實際應用情境：** 工廠感測器資料的異常檢測——用正常資料訓練自動編碼器後，異常資料的重建誤差會明顯偏高（因為編碼器沒見過這類模式），以此識別設備故障。

---

[來源: ch17 | 類型: tutorial] ### 範例 1: 基礎堆疊自動編碼器

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

---

[來源: ch17 | 類型: tutorial] # 載入 Fashion-MNIST（示範用）
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype(np.float32) / 255.0  # 正規化到 [0, 1]
X_test  = X_test.astype(np.float32) / 255.0
X_train_flat = X_train.reshape(-1, 28 * 28)   # 展平
X_test_flat  = X_test.reshape(-1, 28 * 28)

---

[來源: ch17 | 類型: tutorial] # Encoder：784 → 256 → 128 → 32（潛在空間）

---

[來源: ch17 | 類型: tutorial] # Decoder：32 → 128 → 256 → 784（重建）

---

[來源: ch17 | 類型: tutorial] # 方法 1：Sequential API（簡潔）
autoencoder = keras.Sequential([
    # Encoder
    keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32),       # 潛在空間（線性激活）
    # Decoder
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid")  # 輸出在 [0,1]
])

---

[來源: ch17 | 類型: tutorial] # 輸入 = 輸出（重建任務）
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

---

[來源: ch17 | 類型: tutorial] # 訓練
history = autoencoder.fit(
    X_train_flat, X_train_flat,    # X 和 y 都是 X（自監督）
    epochs=20,
    batch_size=256,
    validation_data=(X_test_flat, X_test_flat),
    verbose=0
)
print(f"驗證重建損失: {min(history.history['val_loss']):.4f}")
```

---

[來源: ch17 | 類型: tutorial] ### 範例 2: 分離 Encoder 和 Decoder（方法 2：Functional API）

```python

---

[來源: ch17 | 類型: tutorial] # 使用 Functional API 分別建立 Encoder 和 Decoder
encoder_input = keras.Input(shape=(784,), name="encoder_input")
z_encoded = keras.layers.Dense(256, activation="relu")(encoder_input)
z_encoded = keras.layers.Dense(128, activation="relu")(z_encoded)
z_encoded = keras.layers.Dense(32, name="latent_space")(z_encoded)
encoder = keras.Model(encoder_input, z_encoded, name="encoder")

---

[來源: ch17 | 類型: tutorial] (32, name="latent_space")(z_encoded)
encoder = keras.Model(encoder_input, z_encoded, name="encoder")

decoder_input = keras.Input(shape=(32,), name="decoder_input")
x_decoded = keras.layers.Dense(128, activation="relu")(decoder_input)
x_decoded = keras.layers.Dense(256, activation="relu")(x_decoded)
x_decoded = keras.layers.Dense(784, activation="sigmoid")(x_decoded)
decoder = keras.Model(decoder_input, x_decoded, name="decoder")

---

[來源: ch17 | 類型: tutorial] # 組合成完整自動編碼器
ae_input = keras.Input(shape=(784,))
ae_output = decoder(encoder(ae_input))
autoencoder_full = keras.Model(ae_input, ae_output, name="autoencoder")
autoencoder_full.compile(optimizer="adam", loss="binary_crossentropy")

---

[來源: ch17 | 類型: tutorial] # 異常檢測
def detect_anomalies(model, X: np.ndarray,
                     threshold_percentile: float = 95) -> np.ndarray:
    """基於重建誤差的異常檢測"""
    reconstructed = model.predict(X, verbose=0)
    recon_errors = np.mean(np.square(X - reconstructed), axis=1)  # MSE per sample
    threshold = np.percentile(recon_errors, threshold_percentile)
    return recon_errors > threshold  # True = 異常

print("自動編碼器異常檢測框架建立完成！")
```

**✅ 程式碼逐行解析：**

1. 損失函數 `binary_crossentropy`: 輸入像素被視為 Bernoulli 分佈，重建誤差用交叉熵而非 MSE（更適合 [0,1] 輸出）
2. Encoder 和 Decoder 分開定義：可以單獨使用 Encoder 進行降維，或單獨使用 Decoder 從潛在向量生成圖像

---

---

[來源: ch17 | 類型: tutorial] ### 範例 3: 用於圖像的卷積自動編碼器

```python

---

[來源: ch17 | 類型: tutorial] # 卷積自動編碼器（圖像資料）
conv_autoencoder = keras.Sequential([
    # Encoder（卷積下採樣）
    keras.layers.Reshape((28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2), padding="same"),  # 28×28 → 14×14
    keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2), padding="same"),  # 14×14 → 7×7

---

[來源: ch17 | 類型: tutorial] ation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2), padding="same"),  # 14×14 → 7×7

# Decoder（轉置卷積上採樣）
    keras.layers.Conv2DTranspose(16, (3, 3), activation="relu", padding="same"),
    keras.layers.UpSampling2D((2, 2)),                  # 7×7 → 14×14
    keras.layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same"),
    keras.layers.UpSampling2D((2, 2)),                  # 14×14 → 28×28
    keras.layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same"),
    keras.layers.Reshape((28 * 28,))
], name="conv_autoencoder")

---

[來源: ch17 | 類型: tutorial] ivation="sigmoid", padding="same"),
    keras.layers.Reshape((28 * 28,))
], name="conv_autoencoder")

conv_autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
```

---

---

[來源: ch17 | 類型: tutorial] # 去噪自動編碼器：輸入帶噪聲的圖像，輸出乾淨的圖像
def add_gaussian_noise(X: np.ndarray, noise_factor: float = 0.3) -> np.ndarray:
    """添加高斯噪聲"""
    noisy = X + noise_factor * np.random.randn(*X.shape)
    return np.clip(noisy, 0., 1.)  # 確保值在 [0, 1]

X_train_noisy = add_gaussian_noise(X_train_flat)
X_test_noisy  = add_gaussian_noise(X_test_flat)

---

[來源: ch17 | 類型: tutorial] # 訓練：輸入帶噪圖像，目標是乾淨圖像
denoising_ae = keras.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid")
])
denoising_ae.compile(optimizer="adam", loss="binary_crossentropy")

---

[來源: ch17 | 類型: tutorial] # denoising_ae.fit(X_train_noisy, X_train_flat, epochs=10, ...)
print("去噪自動編碼器可用於圖像去噪和缺失資料填補！")
```

**🎯 重點摘要:**

- 去噪 AE 比普通 AE 學到更**健壯的特徵**——必須學會忽略噪聲，只關注本質結構
- 應用：圖像修復、缺失值填補（在 ID 欄位添加噪聲後重建）

---

---

[來源: ch17 | 類型: tutorial] ## 變分自動編碼器 (VAE)

💡 **實際應用情境：** 普通自動編碼器的潛在空間不連續——在兩個點之間插值可能產生無意義的結果。VAE 強制潛在空間呈高斯分佈，插值生成有意義的新圖像（如：在「運動鞋」和「靴子」之間生成過渡鞋款）。

---

[來源: ch17 | 類型: tutorial] ### 範例 5: VAE（含重參數化技巧）

```python
class Sampling(keras.layers.Layer):
    """重參數化技巧：z = μ + ε * σ（ε ~ N(0,1)）

    將採樣轉化為可微分操作：隨機性來自 ε，而非 z
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]

        # 從標準常態分佈採樣噪聲
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))

        # 重參數化：z = μ + σ * ε = μ + exp(log_var/2) * ε
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

---

[來源: ch17 | 類型: tutorial] # ── VAE Encoder ──
latent_dim = 16
encoder_input = keras.Input(shape=(784,), name="vae_encoder_input")
x = keras.layers.Dense(256, activation="relu")(encoder_input)
x = keras.layers.Dense(128, activation="relu")(x)
z_mean    = keras.layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
z         = Sampling()([z_mean, z_log_var])  # 重參數化採樣

vae_encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="vae_encoder")

---

[來源: ch17 | 類型: tutorial] # ── VAE Decoder ──
decoder_input = keras.Input(shape=(latent_dim,))
x = keras.layers.Dense(128, activation="relu")(decoder_input)
x = keras.layers.Dense(256, activation="relu")(x)
vae_decoder_output = keras.layers.Dense(784, activation="sigmoid")(x)
vae_decoder = keras.Model(decoder_input, vae_decoder_output, name="vae_decoder")

---

[來源: ch17 | 類型: tutorial] # ── 自訂 VAE 訓練邏輯（覆寫 train_step）──
class VAE(keras.Model):
    """變分自動編碼器（Variable Autoencoder）"""

def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

---

[來源: ch17 | 類型: tutorial] :
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

# 重建損失
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            ) * 784

# KL 散度損失（讓潛在空間接近標準正態分佈）
            # KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )

# 總損失 = 重建損失 + KL 散度損失
            total_loss = reconstruction_loss + kl_loss

---

[來源: ch17 | 類型: tutorial] og_var)
            )

# 總損失 = 重建損失 + KL 散度損失
            total_loss = reconstruction_loss + kl_loss

gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

return {"total_loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss}

vae = VAE(vae_encoder, vae_decoder)
vae.compile(optimizer=keras.optimizers.Adam(1e-3))

---

[來源: ch17 | 類型: tutorial] # vae.fit(X_train_flat, epochs=30, batch_size=128)
print("VAE 定義完成！潛在空間維度:", latent_dim)
```

**✅ 程式碼逐行解析：**

1. `Sampling` 層的重參數化技巧：$z = \mu + \sigma \cdot \varepsilon$，$\varepsilon \sim N(0,I)$——隨機性來自 ε（固定），梯度通過 μ 和 σ 反向傳播
2. KL 散度損失：懲罰潛在分佈偏離標準正態分佈，確保潛在空間是連續的（可插值）
3. `reconstruction_loss + kl_loss`：重建損失讓 VAE 學會重建；KL 損失讓潛在空間有意義

**🎯 重點摘要:**

- VAE 的潛在空間是**連續的**：從 N(0, I) 採樣 z，通過 decoder 生成新圖像
- 調整 KL 損失的權重（β-VAE）：更大的 KL 係數 → 更解耦的潛在因子，但重建質量下降

---

---

[來源: ch17 | 類型: tutorial] ### 範例 6: GAN 交替訓練迴圈

```python

---

[來源: ch17 | 類型: tutorial] # GAN = Generator（生成器）+ Discriminator（判別器）的博弈

---

[來源: ch17 | 類型: tutorial] # Generator：雜訊 → 假圖像（試圖騙過 Discriminator）

---

[來源: ch17 | 類型: tutorial] # Discriminator：圖像 → 真/假分類（試圖分辨真假）

latent_dim_gan = 32

---

[來源: ch17 | 類型: tutorial] # ── Generator ──
generator = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(latent_dim_gan,)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid"),
    keras.layers.Reshape((28, 28))
], name="generator")

---

[來源: ch17 | 類型: tutorial] # ── Discriminator ──
discriminator = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.4),      # 正則化（防止 Discriminator 過擬合）
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1, activation="sigmoid")  # 1=真, 0=假
], name="discriminator")

class GAN(keras.Model):
    """生成對抗網路（交替訓練）"""

---

[來源: ch17 | 類型: tutorial] tion="sigmoid")  # 1=真, 0=假
], name="discriminator")

class GAN(keras.Model):
    """生成對抗網路（交替訓練）"""

def __init__(self, discriminator, generator, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator     = generator
        self.latent_dim    = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

---

[來源: ch17 | 類型: tutorial] = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

# ── 訓練 Discriminator ──
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator(noise, training=False)

---

[來源: ch17 | 類型: tutorial] mal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator(noise, training=False)

# 合併真實和假圖像
        combined_images = tf.concat([fake_images, real_images], axis=0)
        labels = tf.concat([
            tf.zeros((batch_size, 1)),  # 假圖像 → 0
            tf.ones((batch_size, 1))    # 真圖像 → 1
        ], axis=0)
        # 添加標籤噪聲（提升穩定性）
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

---

[來源: ch17 | 類型: tutorial] ], axis=0)
        # 添加標籤噪聲（提升穩定性）
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )

---

[來源: ch17 | 類型: tutorial] izer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )

# ── 訓練 Generator（讓 Discriminator 認為假圖像是真的）──
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))  # Generator 目標：讓 D 輸出 1（真）

---

[來源: ch17 | 類型: tutorial] , self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))  # Generator 目標：讓 D 輸出 1（真）

with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            predictions = self.discriminator(fake_images, training=False)
            g_loss = self.loss_fn(misleading_labels, predictions)
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

---

[來源: ch17 | 類型: tutorial] ptimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {"d_loss": self.d_loss_tracker.result(),
                "g_loss": self.g_loss_tracker.result()}

---

[來源: ch17 | 類型: tutorial] urn {"d_loss": self.d_loss_tracker.result(),
                "g_loss": self.g_loss_tracker.result()}

gan = GAN(discriminator, generator, latent_dim=latent_dim_gan)
gan.compile(
    d_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),  # β₁=0.5 是 GAN 的常見設定
    g_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy()
)
print("GAN 建立完成！訓練時注意 d_loss ≈ g_loss 才代表平衡。")
```

**🎯 重點摘要:**

- GAN 訓練訣竅：如果 d_loss → 0，Generator 已被 Discriminator 完全壓制（模式崩潰前兆）
- 標籤噪聲（Label Smoothing）：將真實標籤從 1 改為 0.9~1，防止 Discriminator 過度自信

---

---

[來源: ch17 | 類型: tutorial] ### 範例 7: 擴散過程概念（前向加噪）

```python

---

[來源: ch17 | 類型: tutorial] # 前向過程：原始圖像 x₀ → 逐步添加噪聲 → 純雜訊 xₜ

---

[來源: ch17 | 類型: tutorial] # 反向過程：訓練一個 U-Net 預測每步的噪聲，逆向還原圖像

def forward_diffusion(x0: np.ndarray, t: int, T: int = 1000) -> tuple:
    """前向擴散過程：在時間步 t 添加噪聲

    x_t = sqrt(α̅_t) * x₀ + sqrt(1 - α̅_t) * ε
    """
    # 線性噪聲排程（β 從 0.0001 到 0.02）
    betas  = np.linspace(0.0001, 0.02, T)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)  # 累積乘積

    alpha_bar_t = alpha_bars[t]
    noise = np.random.randn(*x0.shape).astype(np.float32)

    # 加噪公式（可以直接從 x₀ 跳到任意時間步 t）
    x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise

---

[來源: ch17 | 類型: tutorial] # 示範不同時間步的加噪效果
sample_image = X_train_flat[0:1]
for t_step in [0, 100, 500, 999]:
    x_t, noise = forward_diffusion(sample_image, t_step)
    signal_ratio = np.var(np.sqrt(1-1e-4) * sample_image) / np.var(x_t)
    print(f"t={t_step:4d}: 訊噪比 ≈ {signal_ratio:.4f}")

print("\n擴散模型的訓練：預測每個時間步添加的噪聲 ε（MSE 損失）")
print("推論：從純雜訊開始，反覆去噪 T 步，生成新圖像")
```

**🎯 重點摘要:**

- 擴散模型 vs GAN：擴散模型訓練更穩定（無博弈問題），但推論速度慢（需要反覆去噪 T 步）
- DDPM（Denoising Diffusion Probabilistic Models）是 Stable Diffusion 的基礎架構
- 現代加速方法（DDIM、DPM-Solver）將推論步數從 1000 降至 20~50

---

---

[來源: ch17 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: VAE 和普通自動編碼器有什麼差別？**

A: 普通 AE 的潛在空間是任意的（不連續、不規則）；VAE 強制潛在空間服從高斯分佈（通過 KL 損失），使潛在空間連續且可插值。VAE 可以從潛在空間採樣**生成新圖像**，普通 AE 不能。

**Q2: GAN 的模式崩潰（Mode Collapse）如何緩解？**

A: (1) 使用 Wasserstein GAN（WGAN）改變損失函數；(2) Mini-Batch Discrimination（讓 Discriminator 看到一個 batch 的多個樣本）；(3) 漸進式增長 GAN（Progressive Growing GAN）；(4) 添加標籤噪聲。

**Q3: 什麼情況下用 VAE vs GAN？**

A: VAE 適合需要**插值/探索潛在空間**的場景（生成過渡圖像），且訓練穩定；GAN 生成的圖像視覺質量通常更高（更清晰），但訓練不穩定。現代最佳方案：擴散模型（兩者的優點）。

**Q4: 自動編碼器適合什麼任務？**

A: (1) **降維/視覺化**（比 PCA 更強）；(2) **異常檢測**（重建誤差高 = 異常）；(3) **去噪**；(4) **特徵學習**（作為預訓練的 encoder）。

---

---

[來源: ch17 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #自動編碼器 #VAE #GAN #生成式AI #擴散模型 #Keras #TensorFlow #深度學習 #重參數化技巧 #程式設計 #教學 #DataScience #MachineLearning

---

[來源: ch18 | 類型: cheatsheet] # Ch18 速查表：Reinforcement Learning

> **核心主旨**：RL 的核心循環：Agent 觀察狀態 → 選擇動作 → 環境返回獎勵 → 更新策略。DQN 是深度 RL 的基礎。

---

---

[來源: ch18 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Environment (`gym`) | 提供狀態、接收動作、返回獎勵的模擬環境 | 所有 RL 任務 |
| Policy (π) | 從狀態到動作的映射（函數或查找表） | RL 的核心 |
| Reward (r) | 每步的立即回報，RL 的唯一訓練訊號 | 獎勵設計至關重要 |
| Discounted Return (G) | $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$，折扣累積回報 | 考量長期回報 |
| Policy Gradient (REINFORCE) | 直接優化 E[G]，梯度 ∝ log_prob × G | 連續動作空間 |
| Q-Value Q(s,a) | 在狀態 s 採取動作 a 後的折扣期望回報 | Q-Learning 基礎 |
| DQN | 用神經網路近似 Q 函數 + experience replay + target network | 離散動作空間的深度 RL |
| Epsilon-Greedy | 以 ε 機率探索隨機動作，否則選最佳動作 | 探索 vs 利用平衡 |
| Experience Replay | 用 replay buffer 打亂相關性，提升樣本效率 | DQN 的關鍵技巧 |
| Target Network | 定期複製的凍結 Q-network，穩定訓練目標 | DQN 訓練穩定性 |


---

[來源: ch18 | 類型: cheatsheet] replay buffer 打亂相關性，提升樣本效率 | DQN 的關鍵技巧 |
| Target Network | 定期複製的凍結 Q-network，穩定訓練目標 | DQN 訓練穩定性 |


---

---

[來源: ch18 | 類型: cheatsheet] ## 2. 關鍵 API 速查

| API | 重點參數 | 用途 |
|-----|---------|------|
| `gym.make("CartPole-v1")` | 環境名稱 | 建立 Gym 環境 |
| `env.reset(seed=42)` | – | 重置環境，回傳初始觀測 |
| `env.step(action)` | action | 執行動作，回傳 (obs, reward, terminated, truncated, info) |
| `env.action_space` | – | 動作空間描述 |
| `env.observation_space` | – | 觀測空間描述 |
| `env.render()` | – | 渲染畫面（human/rgb_array mode） |
| `collections.deque(maxlen=N)` | – | 固定大小的 replay buffer |

---

---

[來源: ch18 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import gymnasium as gym
import tensorflow as tf
import numpy as np
from collections import deque

---

[來源: ch18 | 類型: cheatsheet] # 建立環境與探索
env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)
print(f"觀測空間: {env.observation_space}")  # Box(4,), 連續狀態
print(f"動作空間: {env.action_space}")       # Discrete(2), 左/右

---

[來源: ch18 | 類型: cheatsheet] # 基本環境互動迴圈
total_reward = 0
obs, _ = env.reset()
for step in range(200):
    action = env.action_space.sample()  # 隨機策略
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
print(f"Total reward: {total_reward}")

---

[來源: ch18 | 類型: cheatsheet] # Policy Gradient (REINFORCE) 網路
n_inputs = env.observation_space.shape[0]  # 4 for CartPole
n_outputs = env.action_space.n              # 2 for CartPole

policy_net = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu", input_shape=[n_inputs]),
    tf.keras.layers.Dense(n_outputs, activation="softmax")
])

---

[來源: ch18 | 類型: cheatsheet] ation="relu", input_shape=[n_inputs]),
    tf.keras.layers.Dense(n_outputs, activation="softmax")
])

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        probas = model(obs[tf.newaxis])  # 前向傳播
        action = tf.random.categorical(tf.math.log(probas), 1)[0, 0]
        loss = tf.reduce_mean(loss_fn(tf.expand_dims(action, 0),
                                      probas))  # log_prob
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, 

---

[來源: ch18 | 類型: cheatsheet]    probas))  # log_prob
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, terminated, truncated, _ = env.step(int(action))
    return obs, reward, terminated or truncated, grads

def discount_rewards(rewards, discount_factor=0.95):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

---

[來源: ch18 | 類型: cheatsheet] # DQN（完整範例）
class DQN:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=2000)
        self.gamma = 0.95       # 折扣率
        self.epsilon = 1.0      # 探索率（初始）
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model(state_size, action_size)
        self.target_model = self._build_model(state_size, action_size)
        self.update_target_network()

---

[來源: ch18 | 類型: cheatsheet] self.target_model = self._build_model(state_size, action_size)
        self.update_target_network()

def _build_model(self, state_size, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation="relu", input_shape=[state_size]),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(action_size, activation="linear")  # Q-values 不用 softmax
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss="mse")
        return model

---

[來源: ch18 | 類型: cheatsheet] tf.keras.optimizers.Adam(self.learning_rate),
                      loss="mse")
        return model

def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

---

[來源: ch18 | 類型: cheatsheet] ard, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

def act(self, state):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()  # Epsilon-greedy 探索
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

---

[來源: ch18 | 類型: cheatsheet] q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

def replay(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = np.array(self.replay_buffer)[
            np.random.choice(len(self.replay_buffer), batch_size, replace=False)]
        states, actions, rewards, next_states, dones = (
            np.stack(batch[:, i]) for i in range(5))
        next_q_values = self.target_model.predict(next_states,

---

[來源: ch18 | 類型: cheatsheet] stack(batch[:, i]) for i in range(5))
        next_q_values = self.target_model.predict(next_states, verbose=0)
        target_q = rewards + (1 - dones.astype(float)) * self.gamma * next_q_values.max(axis=1)
        q_values = self.model.predict(states, verbose=0)
        q_values[np.arange(batch_size), actions.astype(int)] = target_q
        self.model.train_on_batch(states, q_values)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

---

[來源: ch18 | 類型: cheatsheet] # 訓練迴圈
dqn = DQN(state_size=4, action_size=2)
for episode in range(1000):
    obs, _ = env.reset()
    for step in range(200):
        action = dqn.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        dqn.remember(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            break
    dqn.replay(batch_size=32)
    if episode % 10 == 0:
        dqn.update_target_network()  # 定期更新目標網路
```

---

---

[來源: ch18 | 類型: cheatsheet] ## 4. 常見陷阱

- **Gymnasium vs Gym**：新版本 `gymnasium` 的 `env.step()` 回傳 5 個值（多了 `truncated`），舊版 `gym` 只回傳 4 個，注意版本。
- **Target Network 更新頻率**：太頻繁（每步）→ 不穩定；太少（萬步一次）→ 學習緩慢。通常每 100-1000 步更新一次。
- **Epsilon 衰減**：`epsilon` 過快衰減 → 過早停止探索；太慢 → 效率低下。通常訓練的前 50-80% 期間維持探索。
- **獎勵設計（Reward Shaping）**：原始環境的 reward 可能稀疏（只有最後才有），可以手動加中間獎勵，但要小心引入偏差。

---

---

[來源: ch18 | 類型: cheatsheet] ```
RL 算法選擇：
├── 離散動作（遊戲、選擇題）      → DQN / Double DQN / Dueling DQN
├── 連續動作（機器人控制）        → PPO / SAC / TD3
├── 環境模型已知                  → 動態規劃（Value Iteration）
└── 需要可解釋的策略              → Policy Gradient (REINFORCE)

DQN 進階改進：
├── Double DQN     → 用 online 網路選動作，target 網路評估，減少高估 Q 值
├── Dueling DQN    → 分開估計 V(s) 和 A(s,a)，學習更穩定
└── Prioritized ER → 優先回放 TD-error 大的樣本

常用 Gym 環境：
├── CartPole-v1        → 倒立擺（入門首選）
├── MountainCar-v0     → 稀疏獎勵的挑戰性任務
├── LunarLander-v2     → 連續動作空間入門
└── Atari Breakout     → 圖像輸入的進階任務（需 CNN）
```

---

[來源: ch18 | 類型: handout] # 課程講義：強化學習 (Chapter 18)

強化學習（Reinforcement Learning，RL）是機器學習的第三大典範：Agent 不依賴標籤資料，而是透過**在環境中行動、接收獎勵**來學習最優策略。從 OpenAI Gym 的遊戲環境，到 DeepMind AlphaGo、OpenAI ChatGPT 的 RLHF——強化學習正在重塑人工智慧的邊界。本章從 MDP 的數學基礎出發，帶你實作策略梯度、Q 學習和深度 Q 網路（DQN）。

---

---

[來源: ch18 | 類型: handout] ### 理論背景

**馬可夫決策過程 (MDP, Markov Decision Process)**：

- **狀態 (State) $s$**：Agent 對環境的觀測
- **動作 (Action) $a$**：Agent 可以執行的操作
- **獎勵 (Reward) $r$**：執行動作後環境給的即時回饋
- **策略 (Policy) $\pi(a|s)$**：在狀態 $s$ 下選擇動作 $a$ 的機率

**折扣累積獎勵 (Return)**：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

其中折扣因子 $\gamma \in [0, 1)$：

---

[來源: ch18 | 類型: handout] 1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

其中折扣因子 $\gamma \in [0, 1)$：

- $\gamma = 0$：只在乎即時獎勵（短視）
- $\gamma \to 1$：長遠考慮（有遠見）

**`gymnasium` 環境介面**（OpenAI Gym 的繼承者）：

```
env.reset()  → state
env.step(action) → (next_state, reward, terminated, truncated, info)
env.close()
```

---

[來源: ch18 | 類型: handout] ### 核心代碼

```python
import gymnasium as gym
import numpy as np

---

[來源: ch18 | 類型: handout] # 建立 CartPole 環境（平衡桿問題）
env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)
print(f"初始狀態: {obs}")   # [位置, 速度, 角度, 角速度]
print(f"動作空間: {env.action_space}")    # Discrete(2): 向左(0) / 向右(1)
print(f"狀態空間: {env.observation_space}")  # Box([-4.8, -inf, -0.41, -inf], ...)

---

[來源: ch18 | 類型: handout] # 隨機策略基準線
total_rewards = 0
obs, info = env.reset(seed=42)
for step in range(1000):
    action = env.action_space.sample()  # 隨機選動作
    obs, reward, terminated, truncated, info = env.step(action)
    total_rewards += reward
    if terminated or truncated:
        obs, info = env.reset()
        break

print(f"隨機策略總獎勵: {total_rewards}")
env.close()
```

---

[來源: ch18 | 類型: handout] ### 補充練習 1

**理論題：** 在 CartPole 中，折扣因子 $\gamma = 0.99$ 和 $\gamma = 0.9$ 對 Agent 的行為有何影響？哪個設定會讓 Agent 更傾向於避免長期崩潰而非最大化即時獎勵？

**實作題：** 用 `gymnasium` 的 `MountainCar-v0` 環境跑 10 個隨機策略 episode，記錄每個 episode 的總獎勵，計算平均和標準差。觀察隨機策略是否能偶爾解決這個問題（達到山頂）。

---

---

[來源: ch18 | 類型: handout] ### 理論背景

**策略梯度定理**：直接優化策略 $\pi_\theta$ 以最大化期望回報：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [G_0]$$

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[\sum_t G_t \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

**REINFORCE 演算法**（Monte Carlo 策略梯度）：

---

[來源: ch18 | 類型: handout] t[\sum_t G_t \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

**REINFORCE 演算法**（Monte Carlo 策略梯度）：

1. 使用當前策略 $\pi_\theta$ 跑完整個 episode，收集軌跡 $(s_0, a_0, r_0, s_1, \ldots)$
2. 計算每個時間步的折扣回報 $G_t$
3. 梯度更新：$\theta \leftarrow \theta + \eta G_t \nabla_\theta \log \pi_\theta(a_t | s_t)$

**Keras 的實現方式**：自定義損失函數 $\mathcal{L} = -\log \pi_\theta(a_t | s_t) \cdot G_t$（梯度上升等效於最小化負值）。

---

[來源: ch18 | 類型: handout] s_t)$

**Keras 的實現方式**：自定義損失函數 $\mathcal{L} = -\log \pi_\theta(a_t | s_t) \cdot G_t$（梯度上升等效於最小化負值）。

**方差問題**：REINFORCE 的梯度方差很高（每次完整 episode 的 $G_t$ 差異大）。解法：減去基線（Baseline）如 $G_t - b$。

---

[來源: ch18 | 類型: handout] ### 核心代碼

```python
import tensorflow as tf

tf.random.set_seed(42)

---

[來源: ch18 | 類型: handout] # 策略網路（給定狀態，輸出各動作的機率）
policy_net = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu", input_shape=[4]),  # CartPole: 4 個狀態
    tf.keras.layers.Dense(2, activation="softmax")  # 2 個動作
])

---

[來源: ch18 | 類型: handout] input_shape=[4]),  # CartPole: 4 個狀態
    tf.keras.layers.Dense(2, activation="softmax")  # 2 個動作
])

def play_one_step(env, obs, model, loss_fn):
    """執行一步，記錄梯度"""
    with tf.GradientTape() as tape:
        proba = model(obs[np.newaxis], training=True)
        # 從機率分布採樣動作
        action = tf.random.categorical(tf.math.log(proba), num_samples=1)[0, 0]
        loss = loss_fn(tf.constant([[1, 0]] 

---

[來源: ch18 | 類型: handout] om.categorical(tf.math.log(proba), num_samples=1)[0, 0]
        loss = loss_fn(tf.constant([[1, 0]] if action == 0 else [[0, 1]]), proba)
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, terminated, truncated, info = env.step(int(action))
    return obs, reward, terminated or truncated, grads

---

[來源: ch18 | 類型: handout] ated, truncated, info = env.step(int(action))
    return obs, reward, terminated or truncated, grads

def discount_rewards(rewards, discount_rate=0.95):
    """計算折扣回報"""
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

---

[來源: ch18 | 類型: handout] - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

def normalize_rewards(all_discounted_rewards):
    """正規化回報（降低方差）"""
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std  = flat_rewards.std() + 1e-8
    return [(dr - reward_mean) / reward_std for dr in all_discounted_rewards]
```

---

[來源: ch18 | 類型: handout] ### 補充練習 2

**理論題：** REINFORCE 演算法在稀疏獎勵環境（如 MountainCar，只有到達山頂才有正獎勵）中表現極差，為什麼？Actor-Critic 方法如何緩解這個問題？

**實作題：** 在 CartPole 上訓練 REINFORCE（50 個 episodes/批次，discount_rate=0.95），繪製每 10 個批次的平均總獎勵曲線，觀察學習進程。策略何時開始穩定地讓桿子平衡超過 100 步？

---

---

[來源: ch18 | 類型: handout] ### 理論背景

**Q 函數 (Action-Value Function)**：在狀態 $s$ 下執行動作 $a$，然後遵循策略 $\pi$ 的期望回報：

$$Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

**Bellman 最優方程**（Q 學習的核心）：

$$Q^*(s, a) = \mathbb{E} \left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

**Q 學習（表格型）**：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

---

[來源: ch18 | 類型: handout] 型）**：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

括號內是 **TD 誤差 (Temporal Difference Error)**：目標值與估計值的差。

**$\varepsilon$-greedy 策略**：以 $\varepsilon$ 的機率隨機探索，以 $1-\varepsilon$ 的機率利用（選最大 Q 值的動作）。$\varepsilon$ 隨訓練逐漸衰減。

---

[來源: ch18 | 類型: handout] # 表格型 Q 學習（MountainCar 離散化示例）
class QTable:
    def __init__(self, n_states, n_actions):
        self.Q = np.zeros((n_states, n_actions))

def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.Q.shape[1])  # 探索
        return self.Q[state].argmax()  # 利用

---

[來源: ch18 | 類型: handout] return np.random.randint(self.Q.shape[1])  # 探索
        return self.Q[state].argmax()  # 利用

def update(self, state, action, reward, next_state, gamma=0.99, alpha=0.1):
        td_error = (reward + gamma * self.Q[next_state].max()
                    - self.Q[state, action])
        self.Q[state, action] += alpha * td_error

---

[來源: ch18 | 類型: handout] # 訓練循環
epsilon_start, epsilon_end = 1.0, 0.05
n_episodes = 1000

---

[來源: ch18 | 類型: handout] for episode in range(n_episodes):
    epsilon = max(epsilon_end, epsilon_start - episode / (n_episodes * 0.8))
    obs, _ = env.reset()
    done = False
    while not done:
        state = discretize(obs)  # 連續狀態→離散
        action = q_table.get_action(state, epsilon)
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(obs)
        q_table.update(state, 

---

[來源: ch18 | 類型: handout]  truncated, _ = env.step(action)
        next_state = discretize(obs)
        q_table.update(state, action, reward, next_state)
        done = terminated or truncated
```

---

[來源: ch18 | 類型: handout] ### 補充練習 3

**理論題：** Q 學習是「off-policy」演算法（可以從其他策略收集的資料中學習），而 REINFORCE 是「on-policy」演算法（只能從當前策略收集的資料中學習）。解釋這個區別，以及為何 off-policy 演算法可以使用「經驗回放緩衝區」？

**實作題：** 手動實作 Bellman 方程的值迭代（Value Iteration）在一個小型網格世界（如 5×5，有陷阱和獎勵格）上，計算每個格子的最優 Q 值，並推導最優策略（用箭頭圖表示）。

---

---

[來源: ch18 | 類型: handout] ### 理論背景

**DQN (Deep Q-Network)**：用神經網路逼近 Q 函數：

$$Q_\theta(s, a) \approx Q^*(s, a)$$

**DQN 的兩個關鍵創新**（解決訓練不穩定問題）：

**1. 經驗回放 (Experience Replay)**：

將過去的轉移 $(s, a, r, s')$ 存入緩衝區（Replay Buffer），訓練時隨機取樣 mini-batch。

- 打破序列相關性（相鄰步驟的資料高度相關，違反 SGD 假設）
- 同一個轉移可以多次用於訓練（提高資料效率）

**2. 目標網路 (Target Network)**：

使用**凍結**的舊網路參數 $\theta^-$ 計算 TD 目標：

$$y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')$$

---

[來源: ch18 | 類型: handout] **：

使用**凍結**的舊網路參數 $\theta^-$ 計算 TD 目標：

$$y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')$$

- 穩定訓練目標（避免「移動靶」問題）
- 每 $N$ 步才更新一次目標網路的參數

**Double DQN**：分離「選動作」（用主網路）和「評估 Q 值」（用目標網路），減少 Q 值高估。

---

[來源: ch18 | 類型: handout] ### 核心代碼

```python
from collections import deque
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

---

[來源: ch18 | 類型: handout] # 建立 DQN 網路（CartPole: 4 個輸入, 2 個動作）
def build_dqn(n_inputs, n_outputs):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=[n_inputs]),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_outputs)  # 輸出每個動作的 Q 值（無活化函數）
    ])

---

[來源: ch18 | 類型: handout] .Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_outputs)  # 輸出每個動作的 Q 值（無活化函數）
    ])

model        = build_dqn(4, 2)
target_model = build_dqn(4, 2)
target_model.set_weights(model.get_weights())  # 初始化相同

optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
loss_fn   = tf.keras.losses.MeanSquaredError()

---

[來源: ch18 | 類型: handout] # 經驗回放緩衝區
replay_buffer = deque(maxlen=2000)

def training_step(batch_size=64, gamma=0.99):
    # 從緩衝區隨機取樣
    batch = [replay_buffer[np.random.randint(len(replay_buffer))]
             for _ in range(batch_size)]
    states, actions, rewards, next_states, dones = zip(*batch)

---

[來源: ch18 | 類型: handout] for _ in range(batch_size)]
    states, actions, rewards, next_states, dones = zip(*batch)

states = np.array(states, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    dones   = np.array(dones,   dtype=np.float32)

---

[來源: ch18 | 類型: handout] rewards = np.array(rewards, dtype=np.float32)
    dones   = np.array(dones,   dtype=np.float32)

# 計算 TD 目標（用目標網路）
    next_Q_values = target_model.predict(next_states, verbose=0)
    max_next_Q = np.max(next_Q_values, axis=1)
    target_Q = rewards + (1 - dones) * gamma * max_next_Q

---

[來源: ch18 | 類型: handout] max_next_Q = np.max(next_Q_values, axis=1)
    target_Q = rewards + (1 - dones) * gamma * max_next_Q

with tf.GradientTape() as tape:
        Q_values = model(states, training=True)
        action_mask = tf.one_hot(actions, depth=2)
        Q_action = tf.reduce_sum(Q_values * action_mask, axis=1)
        loss = loss_fn(target_Q, Q_action)

---

[來源: ch18 | 類型: handout] Q_action = tf.reduce_sum(Q_values * action_mask, axis=1)
        loss = loss_fn(target_Q, Q_action)

grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

---

[來源: ch18 | 類型: handout] # 每 50 步同步目標網路
TARGET_UPDATE_FREQ = 50
```

---

[來源: ch18 | 類型: handout] ### 補充練習 4

**理論題：** 「移動靶（Moving Target）」問題是什麼？若用同一個網路計算 TD 目標和 Q 值估計，為何訓練會不穩定（類比：你一邊移動，一邊試圖追著自己的影子跑）？

**實作題：** 在 CartPole-v1 上訓練 DQN（回放緩衝區 2000、batch_size=64、gamma=0.99、目標網路每 50 步更新），繪製每 10 個 episode 的平均總獎勵，觀察 Agent 何時學會穩定平衡（總獎勵 > 400）。

---

---

[來源: ch18 | 類型: handout] ## 結論

強化學習的核心工具箱：

- **MDP + Gymnasium**：環境交互的標準介面
- **REINFORCE**：策略梯度方法的起點；高方差，適合連續動作空間
- **Q 學習 + Bellman 方程**：值函數方法；表格型適合小型離散空間
- **DQN**：結合深度學習和 Q 學習；經驗回放 + 目標網路是訓練穩定的關鍵
- **Double DQN**：緩解 Q 值高估問題

現代 RL 的最新進展（本書未覆蓋但值得了解）：PPO、SAC、AlphaZero、RLHF（ChatGPT 的訓練方法）。

下一章（Ch19）聚焦於大規模生產部署：如何將模型從 Jupyter Notebook 推進到真正服務用戶的生產系統。

---

---

[來源: ch18 | 類型: handout] ## 課後作業

**作業：DQN 訓練與分析**

在 `CartPole-v1` 環境上：

1. 訓練完整 DQN（含經驗回放和目標網路），直到平均總獎勵超過 450（CartPole 的最大獎勵為 500）。記錄訓練所需的 episode 數。

2. **消融實驗**：分別移除「經驗回放」和「目標網路」之一，觀察訓練穩定性的變化。製成比較圖（x 軸：episode，y 軸：總獎勵的移動平均）。

3. **思考題**：CartPole 的最優策略是什麼（用自然語言描述：在哪種情況下應該向左/向右推）？觀察訓練好的 DQN Agent 的行為，它的策略是否符合你的直覺？

---

[來源: ch18 | 類型: tutorial] [標題: 強化學習完整指南：MDP、Q-Learning、DQN 與 Double DQN 實作 | 描述: 深入強化學習：馬可夫決策過程（MDP）、OpenAI Gymnasium 環境、REINFORCE 策略梯度、Q-Learning 與 Bellman 方程、DQN（經驗回放、目標網路）、Double DQN，以及如何訓練 Atari 遊戲 AI。 | 關鍵字: Python, 強化學習, MDP, Q-Learning, DQN, 深度強化學習, Gymnasium, OpenAI, TensorFlow, 策略梯度]
# 強化學習：從馬可夫決策過程到 DQN 打 Atari 遊戲

AlphaGo 下棋、Tesla 自動駕駛、資料中心冷卻系統優化——**強化學習（Reinforcement Learning, RL）** 是讓 AI 自主學習最優策略的技術。本教學從最基礎的 MDP 概念出發，帶你一步步實作 Q-Learning、DQN，直到能訓練出可以玩 Atari 遊戲的 AI。

---

[來源: ch18 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **MDP** 的五元組：(狀態 S, 動作 A, 獎勵 R, 轉移 T, 折扣因子 γ)
- **Q-Learning** 是離線策略方法：學習最優 Q 值，不需要遵循當前策略
- **DQN** 用神經網路替代 Q-Table，兩個關鍵技巧：**經驗回放（Experience Replay）** 和 **目標網路（Target Network）**
- **ε-greedy** 策略平衡探索（Exploration）和利用（Exploitation）
- **Double DQN** 解決 Q-Learning 的過度估計問題，通常比標準 DQN 表現更好

---

---

[來源: ch18 | 類型: tutorial] ## 強化學習基本概念

💡 **實際應用情境：** 想像訓練一隻機器狗學習行走——沒有人告訴它如何走路（無監督），但每次成功往前走一步就得到正獎勵，摔倒就得到負獎勵。通過無數次嘗試，它學會了最優的行走策略。

---

[來源: ch18 | 類型: tutorial] ### 強化學習的要素

| 要素 | 描述 | 例子（CartPole） |
|------|------|----------------|
| 環境（Environment） | 智能體（Agent）互動的世界 | CartPole 物理模擬器 |
| 狀態（State, s） | 環境的當前描述 | 小車位置、桿子角度 |
| 動作（Action, a） | 智能體可執行的操作 | 向左推、向右推 |
| 獎勵（Reward, r） | 執行動作後的即時反饋 | 桿子直立 +1；倒下 0 |
| 策略（Policy, π） | 從狀態到動作的映射 | π(s) → a |
| 折扣因子（γ） | 未來獎勵的重要性（0~1） | γ=0.99 重視長期 |

**累積折扣獎勵（Return）：**

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

---

---

[來源: ch18 | 類型: tutorial] ## Gymnasium 環境

💡 **實際應用情境：** Gymnasium（原 OpenAI Gym）提供標準化的 RL 測試環境，從簡單的 CartPole 到複雜的 Atari 遊戲，讓我們可以快速測試算法。

---

[來源: ch18 | 類型: tutorial] ### 範例 1: 基本環境操作

```python
import gymnasium as gym
import numpy as np

---

[來源: ch18 | 類型: tutorial] # 建立 CartPole 環境（最常用的 RL 入門環境）
env = gym.make("CartPole-v1", render_mode=None)

print(f"狀態空間: {env.observation_space}")      # Box(4,) - 4 個連續數值
print(f"動作空間: {env.action_space}")            # Discrete(2) - 0=左, 1=右
print(f"動作數量: {env.action_space.n}")          # 2

---

[來源: ch18 | 類型: tutorial] # 與環境互動的基本流程
state, info = env.reset(seed=42)  # 重置環境，返回初始狀態
total_reward = 0

for step in range(200):
    # 隨機策略（基準）
    action = env.action_space.sample()  # 隨機選擇動作

    # 執行動作，獲取下一狀態、獎勵、終止信號
    next_state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode 結束，共 {step+1} 步，總獎勵: {total_reward:.1f}")
        break

    state = next_state

env.close()

---

[來源: ch18 | 類型: tutorial] # 狀態含義：[小車位置, 小車速度, 桿子角度, 桿子角速度]
print(f"\nCartPole 狀態示例: {state}")
print("目標：讓桿子保持直立盡量久（最多 500 步）")
```

**✅ 程式碼逐行解析：**

1. `env.reset()`: 重置環境到初始狀態，每個 Episode 開始時呼叫
2. `env.step(action)`: 執行動作，返回 (next_state, reward, terminated, truncated, info)
3. `terminated`: 達到終止條件（桿子倒了）；`truncated`: 達到最大步數限制

---

---

[來源: ch18 | 類型: tutorial] ### 範例 2: REINFORCE 算法

```python
import tensorflow as tf
from tensorflow import keras

---

[來源: ch18 | 類型: tutorial] # 策略網路（Policy Network）：輸入狀態 → 輸出每個動作的機率
def build_policy_network(n_inputs: int, n_outputs: int) -> keras.Model:
    return keras.Sequential([
        keras.layers.Dense(32, activation="relu", input_shape=(n_inputs,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(n_outputs, activation="softmax")  # 動作機率分佈
    ])

n_inputs  = env.observation_space.shape[0]  # 4（CartPole 狀態維度）
n_outputs = env.action_space.n              # 2（動作數）
policy_net = build_policy_network(n_inputs, n_outputs)

---

[來源: ch18 | 類型: tutorial] ts = env.action_space.n              # 2（動作數）
policy_net = build_policy_network(n_inputs, n_outputs)

def discount_rewards(rewards: list, gamma: float = 0.99) -> np.ndarray:
    """計算折扣累積獎勵（從後往前計算）"""
    discounted = np.zeros(len(rewards), dtype=np.float32)
    discounted[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        discounted[t] = rewards[t] + gamma * discounted[t + 1]
    # 標準化（減均值除標準差）→ 降低方差，穩定訓練
    mean, std = discounted.mean(), discounted.std()
    return (discounted - mean) / (std + 1e-8)

---

[來源: ch18 | 類型: tutorial] 訓練
    mean, std = discounted.mean(), discounted.std()
    return (discounted - mean) / (std + 1e-8)

def reinforce_train_step(policy_net: keras.Model,
                          optimizer: keras.optimizers.Optimizer,
                          episode_states: list,
                          episode_actions: list,
                          episode_rewards: list) -> float:
    """REINFORCE 算法的一步更新"""
    discounted = discount_rewards(episode_rewards)

with tf.GradientTape() as tape:
        logits = policy_net(np.array(episode_states), training=True)
        action_probs = tf.nn.softmax(logits)

---

[來源: ch18 | 類型: tutorial] s = policy_net(np.array(episode_states), training=True)
        action_probs = tf.nn.softmax(logits)

# 選擇實際執行動作的 log 機率
        action_masks = tf.one_hot(episode_actions, n_outputs)
        log_probs = tf.reduce_sum(
            tf.math.log(action_probs + 1e-8) * action_masks, axis=1
        )

# REINFORCE 損失：-E[log π(a|s) * G_t]（負號因為要最大化）
        loss = -tf.reduce_mean(log_probs * discounted)

gradients = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_net.trainable_variables))
    return loss.numpy()

---

[來源: ch18 | 類型: tutorial] # 訓練迴圈
optimizer_reinforce = keras.optimizers.Adam(1e-3)
env_train = gym.make("CartPole-v1")

best_reward = 0
for episode in range(200):
    states, actions, rewards = [], [], []
    state, _ = env_train.reset()

while True:
        probs = policy_net(state[np.newaxis]).numpy()[0]
        action = np.random.choice(n_outputs, p=probs)  # 按機率採樣

states.append(state)
        actions.append(action)
        next_state, reward, terminated, truncated, _ = env_train.step(action)
        rewards.append(reward)
        state = next_state

if terminated or truncated:
            break

---

[來源: ch18 | 類型: tutorial] rewards.append(reward)
        state = next_state

if terminated or truncated:
            break

loss = reinforce_train_step(policy_net, optimizer_reinforce,
                                  states, actions, rewards)
    total = sum(rewards)
    best_reward = max(best_reward, total)
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}: total_reward={total:.0f}, "
              f"best={best_reward:.0f}, loss={loss:.4f}")

env_train.close()
```

**🎯 重點摘要:**

- REINFORCE 是**蒙地卡羅策略梯度**：整個 episode 結束後才更新
- 獎勵標準化（減均值除標準差）大幅降低梯度估計的方差，加速收斂

---

---

[來源: ch18 | 類型: tutorial] ## Q-Learning 與 Bellman 方程

💡 **實際應用情境：** Q-Learning 學習「在狀態 s 執行動作 a 的長期價值 Q(s,a)」，而不是直接學習策略。

$$Q^*(s, a) = r + \gamma \cdot \max_{a'} Q^*(s', a') \quad \text{（Bellman 方程）}$$

---

[來源: ch18 | 類型: tutorial] ### 範例 3: 表格 Q-Learning（離散環境）

```python

---

[來源: ch18 | 類型: tutorial] # FrozenLake：4×4 網格，目標是從 S 走到 G，避開冰洞 H
env_lake = gym.make("FrozenLake-v1", is_slippery=False)

n_states  = env_lake.observation_space.n   # 16 個格子
n_actions = env_lake.action_space.n        # 4 個方向

---

[來源: ch18 | 類型: tutorial] # Q-Table：行=狀態，列=動作，值=Q(s,a)
Q_table = np.zeros((n_states, n_actions))

---

[來源: ch18 | 類型: tutorial] # Q-Learning 超參數
alpha   = 0.1    # 學習率
gamma   = 0.99   # 折扣因子
epsilon = 1.0    # 初始探索率
epsilon_min = 0.01
epsilon_decay = 0.995

---

[來源: ch18 | 類型: tutorial] # 訓練
for episode in range(5000):
    state, _ = env_lake.reset()
    done = False

while not done:
        # ε-greedy 策略（平衡探索與利用）
        if np.random.random() < epsilon:
            action = env_lake.action_space.sample()  # 探索：隨機動作
        else:
            action = np.argmax(Q_table[state])        # 利用：取最大 Q 值

next_state, reward, terminated, truncated, _ = env_lake.step(action)
        done = terminated or truncated

---

[來源: ch18 | 類型: tutorial] ate, reward, terminated, truncated, _ = env_lake.step(action)
        done = terminated or truncated

# Q-Learning 更新（Bellman 方程的增量形式）
        td_target = reward + gamma * np.max(Q_table[next_state]) * (not done)
        td_error  = td_target - Q_table[state, action]
        Q_table[state, action] += alpha * td_error

state = next_state

# 衰減探索率
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

---

[來源: ch18 | 類型: tutorial] lpha * td_error

state = next_state

# 衰減探索率
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

env_lake.close()
print(f"Q-Table 訓練完成，探索率衰減至: {epsilon:.4f}")
print(f"學到的最優策略（每個格子的最佳動作）:")
action_names = ["←", "↓", "→", "↑"]
policy = [action_names[np.argmax(Q_table[s])] for s in range(n_states)]
print(np.array(policy).reshape(4, 4))
```

**✅ 程式碼逐行解析：**

1. `td_target = r + γ * max Q(s')`: Bellman 方程——當前獎勵 + 折扣後的未來最大價值
2. `Q[s,a] += α * (target - Q[s,a])`: 增量更新，alpha 控制更新速度（學習率）
3. ε-greedy 的 epsilon 隨訓練遞減：早期多探索，後期多利用

---

---

[來源: ch18 | 類型: tutorial] ## 深度 Q 網路（DQN）

💡 **實際應用情境：** CartPole 只有 4 個狀態變數，Q-Table 可行。但 Atari 遊戲的狀態是 84×84 像素圖像——Q-Table 的大小會是天文數字。DQN 用神經網路替代 Q-Table，解決高維連續狀態空間的問題。

---

[來源: ch18 | 類型: tutorial] ### 範例 4: DQN 的核心組件

```python
import collections
import random

---

[來源: ch18 | 類型: tutorial] # ── 1. DQN 網路結構 ──
def build_dqn(n_inputs: int, n_outputs: int) -> keras.Model:
    """Q 網路：輸入狀態 → 輸出每個動作的 Q 值"""
    return keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(n_inputs,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_outputs)  # 線性輸出（Q 值可為負）
    ])

---

[來源: ch18 | 類型: tutorial] # ── 2. 經驗回放緩衝區（Experience Replay Buffer）──
class ReplayBuffer:
    """環形緩衝區，儲存過去的 (s, a, r, s', done) 轉換"""

def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

---

[來源: ch18 | 類型: tutorial] on, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

def sample(self, batch_size: int) -> tuple:
        """隨機採樣一個 mini-batch"""
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (np.array(states, dtype=np.float32),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

---

[來源: ch18 | 類型: tutorial] np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

def __len__(self):
        return len(self.buffer)

---

[來源: ch18 | 類型: tutorial] # ── 3. DQN 訓練邏輯 ──
class DQNAgent:
    """DQN 智能體（含目標網路和經驗回放）"""

---

[來源: ch18 | 類型: tutorial] def __init__(self, n_states: int, n_actions: int,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 100):
        self.n_actions = n_actions
        self.gamma  = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = e

---

[來源: ch18 | 類型: tutorial]   self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0

# 主 Q 網路（頻繁更新）
        self.q_network = build_dqn(n_states, n_actions)
        # 目標網路（緩慢更新，提供穩定的訓練目標）
        self.target_network = build_dqn(n_states, n_actions)
        self.target_network.set_weights(self.q_network.get_weights())

self.optimizer = keras.optimizers.Adam(lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

---

[來源: ch18 | 類型: tutorial] elf.optimizer = keras.optimizers.Adam(lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

def select_action(self, state: np.ndarray) -> int:
        """ε-greedy 動作選擇"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.q_network(state[np.newaxis], training=False)[0]
        return int(np.argmax(q_values))

def train_step(self) -> float | None:
        """從 Replay Buffer 採樣並更新 Q 網路"""
        if len(self.replay_buffer) < self.batch_size:
            return None

---

[來源: ch18 | 類型: tutorial] y Buffer 採樣並更新 Q 網路"""
        if len(self.replay_buffer) < self.batch_size:
            return None

states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

with tf.GradientTape() as tape:
            # 當前 Q 值
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, self.n_actions)
            q_selected = tf.reduce_sum(q_values * action_masks, axis=1)

---

[來源: ch18 | 類型: tutorial] hot(actions, self.n_actions)
            q_selected = tf.reduce_sum(q_values * action_masks, axis=1)

# 目標 Q 值（使用目標網路，提供穩定目標）
            next_q = tf.reduce_max(
                self.target_network(next_states, training=False), axis=1
            )
            td_target = rewards + self.gamma * next_q * (1 - dones)

# Huber 損失（比 MSE 對離群值更健壯）
            loss = keras.losses.huber(tf.stop_gradient(td_target), q_selected)

---

[來源: ch18 | 類型: tutorial] ber 損失（比 MSE 對離群值更健壯）
            loss = keras.losses.huber(tf.stop_gradient(td_target), q_selected)

gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

# 定期更新目標網路
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())

# 衰減探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss)

---

[來源: ch18 | 類型: tutorial] # ── 4. 訓練迴圈 ──
env_dqn = gym.make("CartPole-v1")
agent = DQNAgent(
    n_states=env_dqn.observation_space.shape[0],
    n_actions=env_dqn.action_space.n
)

episode_rewards = []
for episode in range(300):
    state, _ = env_dqn.reset()
    total_reward = 0

while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env_dqn.step(action)
        done = terminated or truncated

agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.train_step()

state = next_state
        total_reward += reward

if done:
            break

---

[來源: ch18 | 類型: tutorial] agent.train_step()

state = next_state
        total_reward += reward

if done:
            break

episode_rewards.append(total_reward)
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode+1}: avg_reward={avg_reward:.1f}, "
              f"epsilon={agent.epsilon:.3f}")

env_dqn.close()
```

**✅ 程式碼逐行解析：**

1. **經驗回放** (`ReplayBuffer`): 打亂時間相關性——若直接用連續 batch，相鄰樣本高度相關，導致訓練不穩定
2. **目標網路** (`target_network`): 每隔 N 步從主網路複製權重，在此期間提供固定目標——避免「追逐移動目標」的發散問題
3. `tf.stop_gradient(td_target)`: 計算損失時，target 不應產生梯度（只更新 q_network）

**🎯 重點摘要:**

---

[來源: ch18 | 類型: tutorial] 標——避免「追逐移動目標」的發散問題
3. `tf.stop_gradient(td_target)`: 計算損失時，target 不應產生梯度（只更新 q_network）

**🎯 重點摘要:**

- DQN 的兩個關鍵技巧缺一不可：沒有 Replay Buffer → 訓練不穩定；沒有目標網路 → 訓練發散

---

---

[來源: ch18 | 類型: tutorial] ### 範例 5: Double DQN 的差異（只需修改一行）

```python

---

[來源: ch18 | 類型: tutorial] # 標準 DQN：目標 Q 值 = r + γ * max Q_target(s')

---

[來源: ch18 | 類型: tutorial] # 問題：用 max 選擇和評估同一個動作 → 高估 Q 值

---

[來源: ch18 | 類型: tutorial] #  選擇：用主網路選擇最佳動作 a* = argmax Q_main(s')

---

[來源: ch18 | 類型: tutorial] #  評估：用目標網路評估該動作 Q_target(s', a*)

---

[來源: ch18 | 類型: tutorial] # 分離選擇和評估 → 減少過度估計

class DoubleDQNAgent(DQNAgent):
    """Double DQN：分離動作選擇和 Q 值評估"""

def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, self.n_actions)
            q_selected = tf.reduce_sum(q_values * action_masks, axis=1)

---

[來源: ch18 | 類型: tutorial] hot(actions, self.n_actions)
            q_selected = tf.reduce_sum(q_values * action_masks, axis=1)

# ── Double DQN 的關鍵修改 ──
            # 1. 主網路選擇最佳動作
            next_q_main = self.q_network(next_states, training=False)
            best_actions = tf.argmax(next_q_main, axis=1)  # 主網路選動作

# 2. 目標網路評估該動作的 Q 值
            next_q_target = self.target_network(next_states, training=False)
            best_action_mask = tf.one_hot(best_actions, self.n_actions)
            next_q = tf.reduce_sum(next_q_target * best_action_mask, axis=1)

---

[來源: ch18 | 類型: tutorial] ctions, self.n_actions)
            next_q = tf.reduce_sum(next_q_target * best_action_mask, axis=1)

td_target = rewards + self.gamma * next_q * (1 - dones)
            loss = keras.losses.huber(tf.stop_gradient(td_target), q_selected)

gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

---

[來源: ch18 | 類型: tutorial] .optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss)

print("Double DQN 定義完成！與標準 DQN 的唯一差異：目標 Q 值的計算方式。")
```

**🎯 重點摘要:**

- Double DQN 通常比標準 DQN 穩定，且最終效能更好
- 其他 DQN 改進：Dueling DQN（分離狀態值和優勢函數）、Prioritized Experience Replay（高 TD-Error 樣本更頻繁採樣）

---

---

[來源: ch18 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: ε-greedy 的 epsilon 如何設定？**

A: 通常從 1.0（完全隨機探索）開始，指數衰減到 0.01~0.1（以利用為主）。衰減速度取決於任務複雜度——太快則沒探索完就開始利用；太慢則浪費時間在隨機探索上。

**Q2: 為什麼需要 Replay Buffer？**

A: 連續互動的樣本高度相關（每一步的狀態和下一步非常相似），直接訓練相當於只用一個樣本——梯度高方差。Replay Buffer 儲存過去的經驗並隨機採樣，打破時間相關性，提供獨立同分佈的 mini-batch。

**Q3: 目標網路更新頻率如何選擇？**

A: 通常每 100~1000 步更新一次。更新太頻繁（接近 1）→ 等於沒有目標網路；更新太慢 → 目標過於陳舊，學習緩慢。另一種方式：使用軟更新（`θ_target = τ*θ_main + (1-τ)*θ_target`，τ≈0.01）。

**Q4: DQN 適用的場景？離散 vs 連續動作空間？**

A: DQN 只適用於**離散動作空間**（因為需要對所有動作計算 max Q）。連續動作空間需用 DDPG（Deep Deterministic Policy Gradient）或 SAC（Soft Actor-Critic）。

---

---

[來源: ch18 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #強化學習 #MDP #QLearning #DQN #DoubleDQN #DeepRL #Gymnasium #TensorFlow #策略梯度 #AI #程式設計 #教學 #MachineLearning

---

[來源: ch19 | 類型: cheatsheet] # Ch19 速查表：Training & Deploying at Scale

> **核心主旨**：規模化訓練與生產部署 —— `MirroredStrategy` 多 GPU，TF Serving 提供推論 API，TFLite 縮小模型上邊緣裝置。

---

---

[來源: ch19 | 類型: cheatsheet] | 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| SavedModel | TF 的標準部署格式，包含計算圖與權重 | 所有部署場景 |
| TensorFlow Serving | 為 SavedModel 提供 REST / gRPC API | 伺服器端推論 |
| TFLite | 輕量模型格式，支援量化，用於行動/嵌入式裝置 | IoT、行動端 |
| TensorFlow.js | 在瀏覽器或 Node.js 運行模型 | 前端 ML 應用 |
| MirroredStrategy | 單機多 GPU：每個 GPU 複製一份模型，梯度 all-reduce | 單機多 GPU 訓練 |
| MultiWorkerMirroredStrategy | 多台機器多 GPU 分散式訓練 | 大規模分散式訓練 |
| Vertex AI | Google Cloud 的 ML 平台：訓練、超參數調整、部署 | 雲端 ML 工作流程 |
| Keras Tuner | 自動超參數搜索（Random, Bayesian, Hyperband） | 超參數優化 |
| Quantization | 將 float32 權重轉為 int8，大幅縮小模型 | TFLite 部署前優化 |


---

[來源: ch19 | 類型: cheatsheet] Random, Bayesian, Hyperband） | 超參數優化 |
| Quantization | 將 float32 權重轉為 int8，大幅縮小模型 | TFLite 部署前優化 |


---

---

[來源: ch19 | 類型: cheatsheet] | TF / Cloud API | 重點參數 | 用途 |
|----------------|---------|------|
| `model.save("model.keras")` | – | 儲存 Keras 格式 |
| `model.save("my_model")` | – | 儲存 SavedModel 格式（目錄） |
| `tf.keras.models.load_model("model.keras")` | – | 載入模型 |
| `tf.distribute.MirroredStrategy()` | – | 單機多 GPU 策略 |
| `tf.distribute.MultiWorkerMirroredStrategy()` | – | 多機多 GPU 策略 |
| `tf.lite.TFLiteConverter.from_keras_model(model)` | – | 轉換為 TFLite |
| `converter.optimizations = [tf.lite.Optimize.DEFAULT]` | – | 啟用 quantization |
| `converter.convert()` | – | 執行轉換 |
| `keras_tuner.RandomSearch` | `hypermodel=`, `objective=`, `max_trials=` | 隨機超參數搜索 |
| `keras_tuner.Hyperband` | `factor=3`, `max_epochs=` | Hyperband 算法（更快） |
| `keras_tuner.BayesianOptimization` | `max_trials=20` | 貝葉斯優化 |


---

[來源: ch19 | 類型: cheatsheet] `max_epochs=` | Hyperband 算法（更快） |
| `keras_tuner.BayesianOptimization` | `max_trials=20` | 貝葉斯優化 |


---

---

[來源: ch19 | 類型: cheatsheet] ## 3. 必備代碼片段

```python
import tensorflow as tf

---

[來源: ch19 | 類型: cheatsheet] # -------- 模型儲存與載入 --------
model.save("my_model.keras")  # 推薦格式
loaded = tf.keras.models.load_model("my_model.keras")

---

[來源: ch19 | 類型: cheatsheet] # 只儲存權重（快速 checkpoint）
model.save_weights("my_weights.weights.h5")
model.load_weights("my_weights.weights.h5")

---

[來源: ch19 | 類型: cheatsheet] # -------- 多 GPU 訓練（MirroredStrategy）--------
strategy = tf.distribute.MirroredStrategy()
print(f"使用 GPU 數量: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu", input_shape=[28*28]),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

---

[來源: ch19 | 類型: cheatsheet] # batch_size 應乘以 GPU 數量
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
model.fit(X_train, y_train, batch_size=GLOBAL_BATCH_SIZE, epochs=10)

---

[來源: ch19 | 類型: cheatsheet] # -------- TFLite 轉換（行動端部署）--------

---

[來源: ch19 | 類型: cheatsheet] # 標準轉換
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

---

[來源: ch19 | 類型: cheatsheet] # 動態範圍量化（降低模型大小 ~4x，略微降低精度）
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite = converter.convert()

---

[來源: ch19 | 類型: cheatsheet] # TFLite 推論
interpreter = tf.lite.Interpreter(model_content=quantized_tflite)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], X_test[:1])
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

---

[來源: ch19 | 類型: cheatsheet] # -------- Keras Tuner 超參數搜索 --------
import keras_tuner as kt

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=1, max_value=8)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256, step=16)
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])

---

[來源: ch19 | 類型: cheatsheet] 1e-4, max_value=1e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])

model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

---

[來源: ch19 | 類型: cheatsheet] # Hyperband（自適應資源分配，最常用）
tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="my_tuner",
    project_name="fashion_mnist"
)
tuner.search(X_train, y_train, epochs=10, validation_split=0.1)
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)

---

[來源: ch19 | 類型: cheatsheet] # -------- TF Serving（Docker 方式）--------

---

[來源: ch19 | 類型: cheatsheet] # 1. 儲存 SavedModel
tf.saved_model.save(model, "my_model/1")  # 版本號 = 1

---

[來源: ch19 | 類型: cheatsheet] # 2. 啟動 TF Serving（在 terminal 執行）

---

[來源: ch19 | 類型: cheatsheet] # docker run -p 8501:8501 -v "$(pwd)/my_model:/models/my_model" \

---

[來源: ch19 | 類型: cheatsheet] #  -e MODEL_NAME=my_model tensorflow/serving

---

[來源: ch19 | 類型: cheatsheet] # 3. 發送 REST 請求
import requests, json
import numpy as np

X_sample = X_test[:3].tolist()
response = requests.post(
    "http://localhost:8501/v1/models/my_model:predict",
    data=json.dumps({"instances": X_sample})
)
predictions = response.json()["predictions"]
```

---

---

[來源: ch19 | 類型: cheatsheet] ## 4. 常見陷阱

- **`MirroredStrategy` 必須在 `strategy.scope()` 內建立模型**：scope 外建立的模型不會被分散，白費工夫。
- **batch_size 要隨 GPU 數量等比放大**：每個 GPU 分到 `batch_size / n_gpu` 個樣本，等效 batch size 不變，學習率也無需調整。
- **TFLite 量化後要驗證精度**：Dynamic Range Quantization 通常精度損失 < 1%，但某些任務影響較大，建議用測試集驗證。
- **SavedModel 版本目錄**：TF Serving 期望路徑格式為 `model_name/版本號/saved_model.pb`，版本號必須是整數目錄名稱。
- **Keras Tuner 的 `directory`**：每次搜索結果存在這裡，同 `project_name` 的搜索會繼續上次的進度。

---

---

[來源: ch19 | 類型: cheatsheet] ```
部署目標 vs 策略：
├── 伺服器端（REST API）    → SavedModel + TF Serving 或 FastAPI
├── 行動端 / 邊緣裝置       → TFLite（+ 量化壓縮）
├── 瀏覽器（JavaScript）    → TensorFlow.js
└── Google Cloud            → Vertex AI Prediction

訓練規模 vs 分散策略：
├── 單機單 GPU              → 直接 model.fit()
├── 單機多 GPU（最常見）    → MirroredStrategy
├── 多機多 GPU（大型訓練）  → MultiWorkerMirroredStrategy
└── 超大模型（模型並行）    → tf.distribute.TPUStrategy（Cloud TPU）

超參數搜索算法比較：
├── 小資源 (< 30 trials)    → RandomSearch（簡單）
├── 中等資源                → Hyperband（最有效率）
└── 大量資源 (> 50 trials)  → BayesianOptimization（樣本效率最高）
```

---

[來源: ch19 | 類型: handout] # 課程講義：大規模訓練與部署 (Chapter 19)

模型訓練完畢只是起點，真正的挑戰在於**把模型送到用戶手中**。本章涵蓋模型的完整生命週期：從儲存格式、API 服務、行動/邊緣部署、瀏覽器端執行，到多 GPU 分散式訓練和超參數自動調整。這些技術是工業界 MLOps 工程師的日常必備。

---

---

[來源: ch19 | 類型: handout] ### 理論背景

Keras 支援兩種主要儲存格式：

| 格式 | 副檔名 | 說明 | 建議使用場景 |
|------|--------|------|------------|
| **Keras v3** | `.keras` | 新格式（Keras 3+），跨後端 | 一般情況首選 |
| **SavedModel** | 目錄 | TensorFlow 原生格式，包含計算圖 | 部署到 TF Serving、TFLite、TF.js |
| HDF5（舊版） | `.h5` | Keras 舊格式 | 向下相容 |


**SavedModel 的內容**：

---

[來源: ch19 | 類型: handout] ，包含計算圖 | 部署到 TF Serving、TFLite、TF.js |
| HDF5（舊版） | `.h5` | Keras 舊格式 | 向下相容 |


**SavedModel 的內容**：

```
saved_model/
├── saved_model.pb          # 計算圖定義
├── variables/              # 訓練好的權重
│   ├── variables.index
│   └── variables.data-00000-of-00001
└── assets/                 # 輔助資料（如詞彙表）
```

---

[來源: ch19 | 類型: handout] ─ variables.index
│   └── variables.data-00000-of-00001
└── assets/                 # 輔助資料（如詞彙表）
```

**自訂物件的儲存**：若模型含自訂層、損失函數，需用 `@tf.keras.utils.register_keras_serializable()` 裝飾器，或在載入時傳入 `custom_objects` 字典。

---

[來源: ch19 | 類型: handout] ### 核心代碼

```python
import tensorflow as tf
import numpy as np

---

[來源: ch19 | 類型: handout] # 訓練一個範例模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", input_shape=[8]),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="sgd", loss="mse")

---

[來源: ch19 | 類型: handout] # model.fit(X_train, y_train, ...)

---

[來源: ch19 | 類型: handout] # 儲存為 Keras 格式
model.save("my_model.keras")
model_loaded = tf.keras.models.load_model("my_model.keras")

---

[來源: ch19 | 類型: handout] # 儲存為 SavedModel 格式（用於部署）
model.export("my_saved_model")  # Keras 3 方式

---

[來源: ch19 | 類型: handout] # 或：tf.saved_model.save(model, "my_saved_model")

---

[來源: ch19 | 類型: handout] # 載入 SavedModel
infer = tf.saved_model.load("my_saved_model")

---

[來源: ch19 | 類型: handout] # 只儲存/載入權重（常用於遷移學習）
model.save_weights("my_weights.weights.h5")
model.load_weights("my_weights.weights.h5")

---

[來源: ch19 | 類型: handout] # Callbacks 自動儲存最佳模型
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    save_best_only=True,
    monitor="val_loss"
)
```

---

[來源: ch19 | 類型: handout] ### 補充練習 1

**理論題：** `.keras` 格式和 SavedModel 格式各有什麼優缺點？若模型需要部署到 TF Serving 的 Docker 容器，應使用哪種格式？若需要讓前端工程師在瀏覽器中執行，又應使用哪種？

**實作題：** 訓練一個含有自訂層的模型（如自訂 `Standardization` 層），分別儲存為 `.keras` 和 `SavedModel` 格式，重新載入後確認預測結果與原始模型完全一致。

---

---

[來源: ch19 | 類型: handout] ### 理論背景

**TF Serving**：TensorFlow 官方提供的生產級模型服務器，支援：

- **REST API**：標準 HTTP 端點，使用 JSON（易於整合）
- **gRPC API**：二進制協議，速度更快（適合低延遲場景）
- **版本管理**：同時服務多個模型版本，支援灰度發布
- **動態批次 (Dynamic Batching)**：自動將多個請求合併為一個 batch，提升 GPU 利用率

**部署流程**：

1. 將 SavedModel 存至標準目錄結構（`model_name/version_number/`）
2. 啟動 TF Serving Docker 容器
3. 送出 HTTP POST 請求

```bash

---

[來源: ch19 | 類型: handout] # 啟動 TF Serving
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/models,target=/models/my_model \
  -e MODEL_NAME=my_model \
  -t tensorflow/serving
```

**雲端部署**：

- **Vertex AI**（Google Cloud）：托管服務，自動擴縮容
- **SageMaker**（AWS）：Amazon 的機器學習平台
- **Azure ML**：微軟的 MLOps 平台

---

[來源: ch19 | 類型: handout] ### 核心代碼

```python
import requests
import json
import numpy as np

---

[來源: ch19 | 類型: handout] # 儲存模型到 TF Serving 期待的目錄結構
import os
model_version = 1
model_name = "my_california_housing_model"
model_path = os.path.join(model_name, str(model_version))
model.export(model_path)

---

[來源: ch19 | 類型: handout] # 呼叫 TF Serving REST API
SERVER_URL = "http://localhost:8501/v1/models/my_california_housing_model:predict"

X_new = np.random.rand(3, 8).tolist()  # 3 筆樣本，8 個特徵
request_data = json.dumps({"instances": X_new})

---

[來源: ch19 | 類型: handout] X_new = np.random.rand(3, 8).tolist()  # 3 筆樣本，8 個特徵
request_data = json.dumps({"instances": X_new})

response = requests.post(SERVER_URL,
                         data=request_data,
                         headers={"content-type": "application/json"})
if response.status_code == 200:
    predictions = json.loads(response.text)["predictions"]
    print(f"預測結果: {predictions}")
else:
    print(f"錯誤: {response.status_code}, {response.text}")
```

---

[來源: ch19 | 類型: handout] ### 補充練習 2

**理論題：** TF Serving 的「動態批次」如何在服務延遲和吞吐量之間取捨？若設定 `max_batch_delay_millis=5`，意味著服務最多等待 5ms 才將請求合批，這適合哪些應用場景？哪些場景不適合（如自動駕駛實時決策）？

**實作題：** 設計一個完整的部署流程文件（不需要實際部署）：(1) 模型訓練後儲存為 SavedModel；(2) 建立 Docker Compose 配置啟動 TF Serving；(3) 建立 Python 客戶端函式，呼叫 REST API 並解析返回結果。

---

---

[來源: ch19 | 類型: handout] ### 理論背景

**TFLite (TensorFlow Lite)**：針對行動裝置（iOS、Android）和嵌入式設備（Raspberry Pi）優化的推論框架。

**模型量化 (Quantization)**：將模型的浮點數（float32）參數轉換為低精度（int8 或 float16）：

| 量化類型 | 精度 | 模型大小 | 速度 | 準確率影響 |
|--------|------|---------|------|-----------|
| Dynamic Range | int8（權重）| 縮小 4x | 快 2-3x | 輕微 |
| Full Integer | int8（全部）| 縮小 4x | 最快 | 輕微-中等 |
| Float16 | float16 | 縮小 2x | GPU 加速 | 極輕微 |


---

[來源: ch19 | 類型: handout] | 輕微 |
| Full Integer | int8（全部）| 縮小 4x | 最快 | 輕微-中等 |
| Float16 | float16 | 縮小 2x | GPU 加速 | 極輕微 |


**量化感知訓練 (QAT)**：在訓練時模擬量化誤差，讓模型適應低精度，減少準確率下降。

**TF.js**：讓模型在瀏覽器的 JavaScript 環境中運行，支援 WebGL 加速（GPU）。

---

[來源: ch19 | 類型: handout] ### 核心代碼

```python
import tensorflow as tf

---

[來源: ch19 | 類型: handout] # 將 SavedModel 轉換為 TFLite（含量化）
converter = tf.lite.TFLiteConverter.from_saved_model("my_saved_model")

---

[來源: ch19 | 類型: handout] # 動態範圍量化（最簡單，推薦先試）
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

---

[來源: ch19 | 類型: handout] # 儲存
with open("my_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"模型大小: {len(tflite_model) / 1024:.1f} KB")

---

[來源: ch19 | 類型: handout] # 用 TFLite Interpreter 執行推論
interpreter = tf.lite.Interpreter(model_path="my_model.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

---

[來源: ch19 | 類型: handout] input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

X_test_sample = np.array([X_test[0]], dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], X_test_sample)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]["index"])
print(f"TFLite 預測: {prediction}")
```

---

[來源: ch19 | 類型: handout] ### 補充練習 3

**理論題：** 量化將 float32 的參數轉換為 int8，數值精度從 7 位有效數字降至約 2-3 位。為什麼神經網路對這種大幅精度損失有如此強的魯棒性（準確率下降通常不超過 1%）？

**實作題：** 將一個在 MNIST 上訓練的模型（準確率 > 98%）分別轉換為原始 TFLite（無量化）、動態範圍量化、int8 全量化，比較三者的：(1) 模型檔案大小；(2) 推論時間（1000 次預測的平均）；(3) 測試集準確率。

---

---

[來源: ch19 | 類型: handout] ### 理論背景

**為何需要分散式訓練？**

- 模型太大，單個 GPU 記憶體不夠
- 訓練資料龐大，單個 GPU 速度太慢
- 縮短訓練時間（從數天縮短到數小時）

**Keras 分散式訓練策略 (`tf.distribute.Strategy`)**：

| 策略 | 適用場景 | 說明 |
|------|---------|------|
| `MirroredStrategy` | 單機多 GPU | 每個 GPU 保存完整模型副本，同步更新 |
| `MultiWorkerMirroredStrategy` | 多機多 GPU | 跨機器同步分散式訓練 |
| `TPUStrategy` | Google TPU | TPU Pod 上的分散式訓練 |
| `ParameterServerStrategy` | 超大規模 | 非同步訓練，適合異質硬體 |


---

[來源: ch19 | 類型: handout] `TPUStrategy` | Google TPU | TPU Pod 上的分散式訓練 |
| `ParameterServerStrategy` | 超大規模 | 非同步訓練，適合異質硬體 |


**`MirroredStrategy` 的同步機制**：

1. 每個 GPU 計算自己的梯度
2. **AllReduce** 操作：彙整所有 GPU 的梯度（求平均）
3. 所有 GPU 用相同的梯度更新各自的模型副本（保持同步）

**有效批次大小**：$\text{Batch Size}_{\text{effective}} = \text{Batch Size}_{\text{per GPU}} \times \text{GPU 數量}$

通常需要相應調高學習率（Linear Scaling Rule：學習率 × GPU 數量）。

---

[來源: ch19 | 類型: handout] ### 核心代碼

```python
import tensorflow as tf

---

[來源: ch19 | 類型: handout] # 單機多 GPU 訓練
strategy = tf.distribute.MirroredStrategy()
print(f"使用 {strategy.num_replicas_in_sync} 個 GPU")

---

[來源: ch19 | 類型: handout] # 在 Strategy 的 scope 內建立模型
with strategy.scope():
    model_dist = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu", input_shape=[8]),
        tf.keras.layers.Dense(1)
    ])
    model_dist.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3 * strategy.num_replicas_in_sync),
        loss="mse",
        metrics=["mae"]
    )

---

[來源: ch19 | 類型: handout] # 訓練時和一般模型完全相同（Strategy 自動處理分散式）
batch_size_per_gpu = 32
total_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync

---

[來源: ch19 | 類型: handout] # model_dist.fit(X_train, y_train, batch_size=total_batch_size, epochs=10)

---

[來源: ch19 | 類型: handout] # 多機多 Worker
import json, os
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["worker1:12345", "worker2:12345"]
    },
    "task": {"type": "worker", "index": 0}  # 每台機器設不同 index
})
strategy_multi = tf.distribute.MultiWorkerMirroredStrategy()
```

---

[來源: ch19 | 類型: handout] ### 補充練習 4

**理論題：** 在 `MirroredStrategy` 中，若有 4 個 GPU，每個 GPU 使用 batch_size=32，有效批次大小是多少？為什麼需要相應提高學習率？這個「Linear Scaling Rule」在批次大小超大時為何開始失效？

**實作題：** 用 `tf.distribute.MirroredStrategy` 重新訓練 MNIST CNN 模型（即使只有 1 個 GPU，`MirroredStrategy` 仍可測試 API），確認在 `strategy.scope()` 內外建立模型的行為差異，測量訓練時間是否有提升。

---

---

[來源: ch19 | 類型: handout] ### 理論背景

**手動調參的問題**：超參數空間巨大，人工試誤耗時且不系統。

**主要超參數搜索策略**：

| 策略 | 說明 | 優點 | 缺點 |
|------|------|------|------|
| `RandomSearch` | 隨機取樣 | 簡單、不會困於局部最優 | 效率低 |
| `Hyperband` | 早停式 Random Search | 快速剔除差的配置 | 需要可以早停的任務 |
| `BayesianOptimization` | 用代理模型引導搜索 | 樣本效率高 | 初期需要預熱 |
| `GridSearch` | 窮舉網格 | 全面 | 指數級計算量 |


**Keras Tuner 工作流程**：

---

[來源: ch19 | 類型: handout] zation` | 用代理模型引導搜索 | 樣本效率高 | 初期需要預熱 |
| `GridSearch` | 窮舉網格 | 全面 | 指數級計算量 |


**Keras Tuner 工作流程**：

1. 定義超參數空間（`hp.Int`, `hp.Float`, `hp.Choice`）
2. 建立包含超參數的模型構建函式
3. 執行搜索（自動訓練多個模型配置）
4. 取出最佳超參數，重新訓練最終模型

---

[來源: ch19 | 類型: handout] ### 核心代碼

```python
import keras_tuner as kt

---

[來源: ch19 | 類型: handout] # 定義帶超參數的模型構建函式
def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=1, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256, step=16)
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer_type = hp.Choice("optimizer", ["sgd", "adam"])

---

[來源: ch19 | 類型: handout] e=1e-4, max_value=1e-2, sampling="log")
    optimizer_type = hp.Choice("optimizer", ["sgd", "adam"])

model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[8]))
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

---

[來源: ch19 | 類型: handout] del.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.get(optimizer_type)
    optimizer.learning_rate = learning_rate
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model

---

[來源: ch19 | 類型: handout] # Hyperband 搜索（速度最快）
tuner = kt.Hyperband(
    build_model,
    objective="val_mae",
    max_epochs=30,
    factor=3,
    hyperband_iterations=2,
    overwrite=True,
    directory="my_kt_results",
    project_name="california_housing_kt"
)

---

[來源: ch19 | 類型: handout] # 執行搜索
tuner.search(X_train, y_train,
             validation_data=(X_valid, y_valid),
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

---

[來源: ch19 | 類型: handout] # 取出最佳超參數
best_params = tuner.get_best_hyperparameters()[0]
print(f"最佳 n_hidden: {best_params['n_hidden']}")
print(f"最佳 n_neurons: {best_params['n_neurons']}")
print(f"最佳 lr: {best_params['lr']:.4f}")

---

[來源: ch19 | 類型: handout] # 重新訓練最終模型
best_model = tuner.hypermodel.build(best_params)

---

[來源: ch19 | 類型: handout] # best_model.fit(X_train_full, y_train_full, epochs=100)
```

---

[來源: ch19 | 類型: handout] ### 補充練習 5

**理論題：** Hyperband 演算法的「早停」策略為何能大幅減少計算量？它的核心思想是什麼（類比體育賽事的「淘汰制」和「循環賽」）？

**實作題：** 用 `keras_tuner.BayesianOptimization` 在 California Housing 資料集上搜索最佳的神經網路架構（層數 1-5，每層神經元數 16-256，學習率 1e-4 到 1e-2），運行 20 次試驗，繪製每次試驗的驗證 MAE，觀察 Bayesian 搜索是否逐漸收斂到更好的區域。

---

---

[來源: ch19 | 類型: handout] ## 結論

工業級機器學習部署的完整工具鏈：

- **模型儲存**：`.keras`（通用）或 SavedModel（部署）
- **TF Serving**：REST/gRPC API，動態批次，多版本管理
- **TFLite**：邊緣設備，量化壓縮（模型縮小 4x，速度提升 2-3x）
- **TF.js**：瀏覽器端推論，零安裝用戶體驗
- **分散式訓練**：`MirroredStrategy`（單機多 GPU）→ `MultiWorkerMirroredStrategy`（多機）
- **Keras Tuner**：Hyperband / Bayesian 自動超參數搜索

恭喜你完成本書！從 Ch01 的 Hello World 到 Ch19 的生產部署，你已建立了完整的機器學習知識體系。下一步：選擇一個真實問題，端到端地實踐！

---

---

[來源: ch19 | 類型: handout] ## 課後作業

**作業：端到端生產部署模擬**

1. 在 California Housing 資料集上，用 `keras_tuner.Hyperband` 找到最佳神經網路架構，重新訓練最終模型，確認測試集 MAE < 0.4（以 $100k 為單位，即 MAE < $40,000）。

2. 將模型儲存為 SavedModel 格式，用 `tf.lite.TFLiteConverter` 轉換為 int8 量化的 TFLite 模型，比較轉換前後的模型大小和 MAE。

3. 撰寫一個模擬 TF Serving 的 Python 函式 `serve_prediction(X)`，接受 numpy array 輸入，載入 SavedModel 並返回預測結果（不需要真的啟動 Docker，用 `tf.saved_model.load` 模擬）。

---

[來源: ch19 | 類型: tutorial] [標題: TensorFlow 大規模訓練與部署完整指南：TF Serving、TFLite、分散式訓練 | 描述: 深入 TensorFlow 生產環境：SavedModel/Keras 格式儲存、TF Serving REST/gRPC 部署、TFLite 量化、TF.js 瀏覽器推論、MirroredStrategy 多 GPU 訓練、MultiWorkerMirroredStrategy 多機訓練、Keras Tuner 超參數搜索。 | 關鍵字: Python, TensorFlow, TF Serving, TFLite, 分散式訓練, MirroredStrategy, Keras Tuner, 模型部署, 量化, TF.js]
# TensorFlow 大規模訓練與部署：從實驗到生產環境

「模型在 notebook 裡跑得很好，怎麼部署？」這是每個 ML 工程師都會面對的問題。本教學帶你從模型儲存格式、TF Serving API 部署、TFLite 行動端優化，到多 GPU 分散式訓練——完整覆蓋將 TF 模型推向生產的所有關鍵技術。

---

[來源: ch19 | 類型: tutorial] ## 關鍵重點 (Key Takeaways)

- **SavedModel** 格式是 TF 生態系統的標準格式：保存計算圖 + 權重，跨語言可用
- **TF Serving** 用 Docker 一行部署 REST/gRPC API，生產效能接近原生 TF
- **TFLite 量化（INT8）** 可以將模型大小減小 4 倍、推論速度提升 2-3 倍，適合行動裝置
- **`MirroredStrategy`** 是單機多 GPU 的標準方案，程式碼改動極小（幾乎只需 2 行）
- **Keras Tuner** 的 `RandomSearch`/`BayesianOptimization` 自動搜索最優超參數

---

---

[來源: ch19 | 類型: tutorial] ## 模型儲存格式

💡 **實際應用情境：** 訓練好的模型需要在不同環境中使用：Python 推論服務、Android App、瀏覽器——每種環境需要不同的格式。

---

[來源: ch19 | 類型: tutorial] ### 範例 1: 各種儲存格式比較

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

---

[來源: ch19 | 類型: tutorial] # 建立示範模型
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(8,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

---

[來源: ch19 | 類型: tutorial] # 儲存為目錄（包含計算圖 + 權重 + 簽名）
model.save("my_model_saved")         # 目錄格式

---

[來源: ch19 | 類型: tutorial] # └── saved_model.pb               ← 計算圖

---

[來源: ch19 | 類型: tutorial] # └── variables/                   ← 權重

---

[來源: ch19 | 類型: tutorial] # └── assets/                      ← 額外資源

---

[來源: ch19 | 類型: tutorial] # 格式 2：.keras 格式（Keras 3 推薦）
model.save("my_model.keras")         # 單一 ZIP 文件

---

[來源: ch19 | 類型: tutorial] # 格式 3：只儲存架構（不含權重）
model_json = model.to_json()

---

[來源: ch19 | 類型: tutorial] # model_from_json = keras.models.model_from_json(model_json)

---

[來源: ch19 | 類型: tutorial] # 格式 4：只儲存權重
model.save_weights("my_weights.weights.h5")

---

[來源: ch19 | 類型: tutorial] # 格式 5：HDF5（舊格式，仍廣泛使用）
model.save("my_model.h5")            # 需要 h5py

---

[來源: ch19 | 類型: tutorial] # 載入模型
loaded_model = keras.models.load_model("my_model.keras")

---

[來源: ch19 | 類型: tutorial] # 驗證載入後的模型等效
X_test = np.random.randn(100, 8).astype(np.float32)
y_original = model.predict(X_test, verbose=0)
y_loaded   = loaded_model.predict(X_test, verbose=0)
print(f"最大預測差異: {np.max(np.abs(y_original - y_loaded)):.2e}")  # ≈ 0
```

**✅ 程式碼逐行解析：**

1. `SavedModel` 格式的優點：儲存完整計算圖，可以用 TF Serving/TFLite 轉換，跨語言（Python/Java/C++）
2. `.keras` 格式：Keras 3 的新格式，更緊湊，支援自訂物件序列化
3. 只儲存權重：最輕量，但需要事先知道模型架構才能載入

---

---

[來源: ch19 | 類型: tutorial] ### 範例 2: 儲存 Serving 格式 + API 客戶端

```python

---

[來源: ch19 | 類型: tutorial] # 儲存為 TF Serving 可識別的 versioned 目錄結構

---

[來源: ch19 | 類型: tutorial] # /models/my_model/2/  ← 版本 2

import os

model_version = 1
serving_path = f"./tf_serving_models/my_model/{model_version}"
model.save(serving_path)
print(f"TF Serving 模型已儲存至: {serving_path}")

---

[來源: ch19 | 類型: tutorial] # 查看模型簽名（API 規格）
loaded_tf = tf.saved_model.load(serving_path)
print("可用簽名:", list(loaded_tf.signatures.keys()))

---

[來源: ch19 | 類型: tutorial] # 通常是 ['serving_default']

infer = loaded_tf.signatures["serving_default"]
print("輸入規格:", infer.structured_input_signature)
print("輸出規格:", infer.structured_outputs)

---

[來源: ch19 | 類型: tutorial] # ── 部署 TF Serving（用 Docker）──

---

[來源: ch19 | 類型: tutorial] #  -v "$(pwd)/tf_serving_models:/models" \

---

[來源: ch19 | 類型: tutorial] # ── REST API 客戶端 ──
import json

def predict_via_rest_api(data: np.ndarray, server_url: str = "http://localhost:8501") -> np.ndarray:
    """通過 TF Serving REST API 進行推論"""
    import urllib.request

payload = json.dumps({
        "instances": data.tolist()
    })

request = urllib.request.Request(
        f"{server_url}/v1/models/my_model:predict",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

---

[來源: ch19 | 類型: tutorial] =payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())
            return np.array(result["predictions"])
    except Exception as e:
        print(f"API 呼叫失敗（服務未啟動）: {e}")
        return None

---

[來源: ch19 | 類型: tutorial] # predictions = predict_via_rest_api(X_test[:5])
print("\nTF Serving REST API URL: POST /v1/models/{model_name}:predict")
print("TF Serving gRPC Port: 8500（比 REST 更快）")
```

**🎯 重點摘要:**

- TF Serving 支援**版本管理**：可以同時維護多個版本，並進行 A/B 測試
- REST API（8501 port）適合一般應用；gRPC（8500 port）延遲更低，適合高性能場景

---

---

[來源: ch19 | 類型: tutorial] ## TFLite 行動端部署

💡 **實際應用情境：** 台灣某醫療 App 需要在使用者手機上離線執行皮膚分析——TFLite INT8 量化後，模型從 50MB 縮小到 12MB，推論速度提升 2.5 倍，同時準確率僅下降 0.3%。

---

[來源: ch19 | 類型: tutorial] ### 範例 3: TFLite 轉換與量化

```python

---

[來源: ch19 | 類型: tutorial] # ── TFLite 轉換（不量化）──
converter = tf.lite.TFLiteConverter.from_saved_model(serving_path)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"TFLite 模型大小: {len(tflite_model) / 1024:.1f} KB")

---

[來源: ch19 | 類型: tutorial] # ── TFLite 動態範圍量化（INT8，模型大小 ~4x 縮小）──
converter_quant = tf.lite.TFLiteConverter.from_saved_model(serving_path)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]  # 啟用量化
tflite_quant_model = converter_quant.convert()

with open("model_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)
print(f"量化後大小: {len(tflite_quant_model) / 1024:.1f} KB")

---

[來源: ch19 | 類型: tutorial] # ── 全整數量化（需要校準資料）──
def representative_dataset():
    """提供少量代表性資料用於量化校準"""
    cal_data = np.random.randn(100, 8).astype(np.float32)
    for sample in cal_data:
        yield [sample[np.newaxis]]

converter_int8 = tf.lite.TFLiteConverter.from_saved_model(serving_path)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset

---

[來源: ch19 | 類型: tutorial] # 強制輸入/輸出也是 INT8（更快的嵌入式推論）
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type  = tf.int8   # 注意：需要應用端轉換資料
converter_int8.inference_output_type = tf.int8
tflite_int8_model = converter_int8.convert()
print(f"INT8 量化後大小: {len(tflite_int8_model) / 1024:.1f} KB")

---

[來源: ch19 | 類型: tutorial] # ── TFLite 推論（Python 模擬行動端）──
def run_tflite_inference(tflite_model_bytes: bytes,
                          input_data: np.ndarray) -> np.ndarray:
    """在 Python 中模擬 TFLite 推論"""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

---

[來源: ch19 | 類型: tutorial] put_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

results = []
    for i in range(len(input_data)):
        interpreter.set_tensor(
            input_details[0]['index'],
            input_data[i:i+1]
        )
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        results.append(output[0])

return np.array(results)

tflite_preds = run_tflite_inference(tflite_model, X_test[:10])
print(f"TFLite 推論完成，輸出形狀: {tflite_preds.shape}")
```

**✅ 程式碼逐行解析：**

---

[來源: ch19 | 類型: tutorial] ence(tflite_model, X_test[:10])
print(f"TFLite 推論完成，輸出形狀: {tflite_preds.shape}")
```

**✅ 程式碼逐行解析：**

1. `Optimize.DEFAULT`: 動態範圍量化——將 float32 權重量化為 int8，運行時再反量化（零代價優化）
2. `representative_dataset`: 全整數量化需要 100~200 個代表性樣本來校準量化閾值
3. `tf.lite.Interpreter.invoke()`: 執行推論（需要手動迴圈，每次只能推論一個樣本或 batch）

---

---

[來源: ch19 | 類型: tutorial] ### 範例 4: 轉換為 TF.js 格式

```python

---

[來源: ch19 | 類型: tutorial] # tensorflowjs_converter --input_format=keras my_model.keras tfjs_model/

---

[來源: ch19 | 類型: tutorial] # 或在 Python 中轉換
import subprocess

---

[來源: ch19 | 類型: tutorial] #    "tensorflowjs_converter",

---

[來源: ch19 | 類型: tutorial] #    "--input_format=tf_saved_model",

---

[來源: ch19 | 類型: tutorial] #    "--output_format=tfjs_graph_model",

---

[來源: ch19 | 類型: tutorial] # ── JavaScript 推論代碼（在 HTML/Node.js 中使用）──
tfjs_inference_code = """
// 載入模型
const model = await tf.loadGraphModel('/tfjs_model/model.json');

// 準備輸入
const inputData = tf.tensor2d([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);

// 推論
const prediction = model.predict(inputData);
const result = await prediction.data();
console.log('預測結果:', result);
"""

print("TF.js 模型可以在瀏覽器中零後端執行推論（用戶資料不離開裝置）！")
print("\nJavaScript 推論代碼:")
print(tfjs_inference_code)
```

---

---

[來源: ch19 | 類型: tutorial] ## 多 GPU 分散式訓練

💡 **實際應用情境：** 大型圖像分類模型在單 GPU 上需要 72 小時；用 4 GPU + `MirroredStrategy` 縮短到 ~18 小時，且程式碼改動極小。

---

[來源: ch19 | 類型: tutorial] ### 範例 5: MirroredStrategy 多 GPU 訓練

```python

---

[來源: ch19 | 類型: tutorial] # ── 單機多 GPU（MirroredStrategy）──
strategy = tf.distribute.MirroredStrategy()

---

[來源: ch19 | 類型: tutorial] # 自動偵測可用的 GPU 並在所有 GPU 間同步梯度

print(f"可用 GPU 數量: {strategy.num_replicas_in_sync}")

with strategy.scope():
    # 在 strategy.scope() 內建立的所有變數會自動跨 GPU 複製
    distributed_model = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(8,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    distributed_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

---

[來源: ch19 | 類型: tutorial] # 分散式訓練時，建議調整 batch_size = 原始 batch_size × GPU 數

---

[來源: ch19 | 類型: tutorial] # 例如單 GPU 用 32，4 個 GPU 則用 32 × 4 = 128
n_gpus = max(1, strategy.num_replicas_in_sync)
batch_size = 32 * n_gpus
print(f"分散式 batch_size: {batch_size}")

---

[來源: ch19 | 類型: tutorial] # ── 多機多 GPU（MultiWorkerMirroredStrategy）──

---

[來源: ch19 | 類型: tutorial] # 環境變數 TF_CONFIG 需要在每台機器上設置
tf_config_example = {
    "cluster": {
        "worker": ["worker0.example.com:2222",
                   "worker1.example.com:2222"]
    },
    "task": {"type": "worker", "index": 0}  # 當前機器的角色和索引
}

---

[來源: ch19 | 類型: tutorial] # os.environ["TF_CONFIG"] = json.dumps(tf_config_example)

---

[來源: ch19 | 類型: tutorial] # multi_worker_strategy = tf.distribute.MultiWorkerMirroredStrategy()
print("MultiWorkerMirroredStrategy 在多台機器間使用 AllReduce 同步梯度")

---

[來源: ch19 | 類型: tutorial] # ── TPU 訓練（Google Colab 免費 TPU）──

---

[來源: ch19 | 類型: tutorial] # resolver = tf.distribute.cluster_resolver.TPUClusterResolver()

---

[來源: ch19 | 類型: tutorial] # tf.config.experimental_connect_to_cluster(resolver)

---

[來源: ch19 | 類型: tutorial] # tf.tpu.experimental.initialize_tpu_system(resolver)

---

[來源: ch19 | 類型: tutorial] # tpu_strategy = tf.distribute.TPUStrategy(resolver)
print("TPU 訓練：在 Colab 中免費使用，速度是 GPU 的 5-10 倍！")
```

**✅ 程式碼逐行解析：**

1. `with strategy.scope()`: 作用域內建立的層/變數自動「鏡射」到所有 GPU
2. MirroredStrategy 使用 **AllReduce** 算法（NCCL）在 GPU 間高效同步梯度
3. `num_replicas_in_sync`: 返回實際可用的 GPU 數，用來計算分散式 batch size

---

---

[來源: ch19 | 類型: tutorial] # 安裝：pip install keras-tuner
import keras_tuner as kt

def build_model_for_tuning(hp: kt.HyperParameters) -> keras.Model:
    """定義超參數搜索空間和模型建構邏輯"""
    model = keras.Sequential()

# 搜索隱藏層數量（1~3 層）
    for i in range(hp.Int("n_hidden_layers", min_value=1, max_value=3)):
        # 搜索每層神經元數量（32~512，步長 32）
        units = hp.Int(f"units_{i}", min_value=32, max_value=512, step=32)
        # 搜索激活函數
        activation = hp.Choice(f"activation_{i}", values=["relu", "elu", "selu"])

---

[來源: ch19 | 類型: tutorial] )
        # 搜索激活函數
        activation = hp.Choice(f"activation_{i}", values=["relu", "elu", "selu"])

model.add(keras.layers.Dense(
            units, activation=activation,
            kernel_initializer="he_normal"
        ))
        # 搜索是否加 Dropout
        if hp.Boolean(f"dropout_{i}"):
            model.add(keras.layers.Dropout(
                rate=hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.5)
            ))

model.add(keras.layers.Dense(1, activation="sigmoid"))

---

[來源: ch19 | 類型: tutorial] min_value=0.1, max_value=0.5)
            ))

model.add(keras.layers.Dense(1, activation="sigmoid"))

# 搜索學習率
    lr = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2,
                   sampling="log")  # 對數尺度採樣（因為 LR 在對數空間更均勻）

model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

---

[來源: ch19 | 類型: tutorial] # 建立搜索器
tuner = kt.BayesianOptimization(
    build_model_for_tuning,
    objective="val_accuracy",       # 最大化驗證準確率
    max_trials=20,                  # 嘗試 20 種超參數組合
    directory="kt_search",
    project_name="binary_classifier"
)

---

[來源: ch19 | 類型: tutorial] # 準備資料
X_data = np.random.randn(1000, 8).astype(np.float32)
y_data = (np.random.randn(1000) > 0).astype(np.float32)

X_tr, X_val = X_data[:800], X_data[800:]
y_tr, y_val = y_data[:800], y_data[800:]

---

[來源: ch19 | 類型: tutorial] # 執行搜索
print("開始超參數搜索（20 個 trial）...")
tuner.search(
    X_tr, y_tr,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[keras.callbacks.EarlyStopping(patience=3)],
    verbose=0
)

---

[來源: ch19 | 類型: tutorial] # 取得最佳超參數
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\n最佳超參數:")
print(f"  隱藏層數: {best_hps.get('n_hidden_layers')}")
print(f"  學習率: {best_hps.get('learning_rate'):.5f}")

---

[來源: ch19 | 類型: tutorial] # 用最佳超參數建立最終模型
best_model = tuner.hypermodel.build(best_hps)

---

[來源: ch19 | 類型: tutorial] # best_model.fit(X_tr, y_tr, epochs=50, validation_data=(X_val, y_val))
```

**✅ 程式碼逐行解析：**

1. `hp.Int/Float/Choice/Boolean`: 定義不同類型的超參數搜索空間
2. `sampling="log"`: 在對數尺度上均勻採樣學習率（因為 1e-4 和 1e-3 的差距比 1e-2 和 2e-2 更重要）
3. `BayesianOptimization`: 根據之前 trial 的結果預測哪些超參數組合更有前途（比 RandomSearch 效率高）

**🎯 重點摘要:**

- Keras Tuner 的其他搜索算法：`RandomSearch`（快速入門）、`Hyperband`（效率最高，基於早停的漏斗篩選）
- 實務上：先用 `RandomSearch` 快速了解超參數重要性，再用 `BayesianOptimization` 精細搜索

---

---

[來源: ch19 | 類型: tutorial] ## 常見問答 (FAQ)

**Q1: SavedModel 和 .keras 格式如何選擇？**

A: 需要跨語言部署（Java/C++/TF Serving）→ SavedModel；純 Keras/Python 工作流 → `.keras`（含自訂物件時更方便）；歷史相容性 → `.h5`。TF 2.x 環境建議優先用 SavedModel。

**Q2: TFLite 量化後準確率損失多少？**

A: 動態範圍量化通常損失 < 0.5%；全整數量化（INT8）通常損失 0.5~2%。若精度下降過多，嘗試量化感知訓練（Quantization-Aware Training）——訓練時模擬量化效果，最終準確率可接近浮點模型。

**Q3: MirroredStrategy 需要修改哪些代碼？**

A: 幾乎只需兩行：(1) `strategy = tf.distribute.MirroredStrategy()`；(2) `with strategy.scope():` 包裹模型建立和編譯。`fit()` 不需修改，建議同時調整 `batch_size`（乘以 GPU 數量）。

**Q4: Keras Tuner 搜索時間太長怎麼辦？**

---

[來源: ch19 | 類型: tutorial] egy.scope():` 包裹模型建立和編譯。`fit()` 不需修改，建議同時調整 `batch_size`（乘以 GPU 數量）。

**Q4: Keras Tuner 搜索時間太長怎麼辦？**

A: 使用 Hyperband：它先用少量 epoch 訓練所有 trial，淘汰差的，逐漸增加存活 trial 的 epoch——通常比 RandomSearch 快 5-10 倍。也可以縮小搜索空間或減少 `max_trials`。

---

---

[來源: ch19 | 類型: tutorial] ## 推薦標籤 (Suggested Hashtags)

\#Python #TensorFlow #TFServing #TFLite #分散式訓練 #MirroredStrategy #KerasTuner #模型部署 #量化 #TFjs #深度學習 #程式設計 #教學 #MachineLearning

---

[來源: tools_numpy] [標題: 🐍 NumPy 工具教學：從陣列建立到線性代數 | 描述: 學習 NumPy 的完整指南，從基本陣列操作到進階線性代數功能。包含實用範例、逐行解析和最佳實踐。 | 關鍵字: NumPy, Python, 陣列, 線性代數, 科學計算, 教學]
# NumPy 完整教學指南：陣列建立、操作與線性代數實戰技巧

NumPy 是 Python 科學計算領域的核心函式庫，專為高效能 N 維陣列運算設計。

其底層以 C 語言實作，提供極快的數值處理速度，並支援線性代數、傅立葉變換、隨機數生成等進階功能。

NumPy 不僅是資料科學、機器學習、人工智慧等領域的基礎工具，也是許多 Python 標準函式庫和第三方套件的效能關鍵。透過 Python 封裝，使用者能輕鬆進行大規模數據分析與科學運算，提升開發效率與程式效能。

以下就讓我們深入探索 NumPy 的強大功能，從基礎的陣列建立到複雜的線性代數運算，並透過實際範例與逐行解析，幫助你掌握這個不可或缺的工具。

---

[來源: tools_numpy] ## 關鍵重點

- NumPy以高效能的 N 維陣列 (N-dimensional arrays) 為核心，提供廣泛的數學和科學計算功能
- 掌握陣列建立 (array creation)、索引 (indexing)、運算 (operations) 和重塑 (reshaping) 是使用 NumPy 的基礎
- 廣播機制 (broadcasting mechanism) 允許不同形狀陣列間的靈活運算
- 向量化操作 (vectorized operations) 比迴圈 (loops) 更有效率，能充分利用 NumPy 的優化
- 線性代數 (linear algebra) 模組提供矩陣運算 (matrix operations)、特徵值 (eigenvalues) 等進階功能

---

---

[來源: tools_numpy] ## 建立陣列

💡 **實際應用情境：** 在資料科學 (data science) 和機器學習 (machine learning) 中，陣列(array) 是處理數值資料的基本單位，從簡單的向量(vector) 到複雜的張量(tensor)，都依賴 NumPy 的陣列建立功能。

---

[來源: tools_numpy] ### 基本概念

NumPy 中的每個維度稱為軸（axis）。軸的數量稱為秩（rank），軸長度的清單稱為形狀（shape），總元素數稱為大小（size）。

如果一個陣列只有一個軸，則稱為一維陣列（向量）；如果有兩個軸，則稱為二維陣列（矩陣）。

假設有一個形狀為 (3, 5) 的二維陣列，它有 2 個軸（秩為 2），第一個軸長度為 3，第二個軸長度為 5，總共有 15 個元素。

```python
import numpy as np

---

[來源: tools_numpy] # 建立一個包含 6 個 0 的陣列
a = np.zeros(6)
print("a =", a)

---

[來源: tools_numpy] # 建立 3x5 矩陣
b = np.zeros((3, 5))
print("b =", b)
print("b.shape =", b.shape)
print("b.ndim =", b.ndim)
print("b.size =", b.size)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] print("b.shape =", b.shape)
print("b.ndim =", b.ndim)
print("b.size =", b.size)
```

**✅ 程式碼逐行解析：**

1. `import numpy as np`：匯入 NumPy 模組並簡稱為 np，這是標準慣例。
2. `a = np.zeros(6)`：使用 zeros 函數建立一個包含 6 個 0 的一維陣列。
3. `print("a =", a)`：輸出陣列 a 的內容。
4. `b = np.zeros((3, 5))`：建立一個形狀為 (3, 5) 的二維陣列（3 行 5 列），所有元素為 0。
5. `print("b =", b)`：輸出陣列 b 的內容。
6. `print("b.shape =", b.shape)`：輸出陣列 b 的形狀，即 (3, 5)。
7. `print("b.ndim =", b.ndim)`：輸出陣列 b 的維度數，即 2。
8. `print("b.size =", b.size)`：輸出陣列 b 的總元素數，即 15。

---

[來源: tools_numpy] )。
7. `print("b.ndim =", b.ndim)`：輸出陣列 b 的維度數，即 2。
8. `print("b.size =", b.size)`：輸出陣列 b 的總元素數，即 15。

**🎯 重點摘要：**

- **核心功能**：zeros 函數用於建立指定形狀的全零陣列，是初始化陣列的常用方法。
- **潛在問題**：忘記指定形狀的括號會導致錯誤，如誤用 np.zeros(3,4) 而非正確的 np.zeros((3,4))。
- **最佳使用情境**：需要初始化陣列但尚未有具體數值時，或在演算法中需要重置陣列為零。

---

[來源: tools_numpy] ### 其他建立函數

💡 實際應用情境： 在陣列初始化中，除了全零陣列，還需全一陣列、指定值陣列或未初始化陣列，以適應計數、常數或效能關鍵場景。

此節介紹 NumPy 的 ones、full 和 empty 函數，分別用於建立全 1、指定值和未初始化陣列，實現高效率的初始化。

```python

---

[來源: tools_numpy] # 全 1 陣列
ones = np.ones((3, 8))
print("ones =", ones)

---

[來源: tools_numpy] # 指定值的陣列
full = np.full((3, 3), np.pi)
print("full =", full)

---

[來源: tools_numpy] # 未初始化的陣列
empty = np.empty((2, 3))
print("empty =", empty)
```

**✅ 程式碼逐行解析：**

1. `ones = np.ones((3, 8))`：建立一個形狀為 (3, 8) 的全 1 陣列。
2. `print("ones =", ones)`：輸出全 1 陣列。
3. `full = np.full((3, 3), np.pi)`：建立一個形狀為 (3, 3) 的陣列，所有元素為 π（圓周率）。
4. `print("full =", full)`：輸出指定值陣列。
5. `empty = np.empty((2, 3))`：建立一個形狀為 (2, 3) 的未初始化陣列，其內容不可預測。
6. `print("empty =", empty)`：輸出未初始化陣列。

**🎯 重點摘要：**

---

[來源: tools_numpy] p.empty((2, 3))`：建立一個形狀為 (2, 3) 的未初始化陣列，其內容不可預測。
6. `print("empty =", empty)`：輸出未初始化陣列。

**🎯 重點摘要：**

- **核心功能**：ones、full 和 empty 分別用於建立全 1、指定值和未初始化陣列。
- **潛在問題**：empty 陣列的內容不可預測，可能包含垃圾值，只在需要立即覆寫時使用。
- **最佳使用情境**：ones 用於計數初始化，full 用於常數陣列，empty 用於效能關鍵的初始化。

---

[來源: tools_numpy] ### 從序列建立陣列

💡 **實際應用情境**：
在資料處理和數值計算中，經常需要從現有序列（如特定範圍數值、等間距點或 Python 列表）建立陣列，這是初始化資料結構的基礎步驟。

此節介紹 NumPy 提供的高效函數：

  * `arange`：用於生成整數或浮點數序列（類似 Python 的 range）。
  * `linspace`：用於建立固定數量的等間距點（Linear Space）。
  * `array`：將 Python 列表（List）轉換為多維陣列。

這些方法支援自訂步長和資料類型，適合各種數值範圍的應用。

```python

---

[來源: tools_numpy] # 整數：[1, 2, 3, 4, 5]
arange_int = np.arange(1, 6)
print("arange_int =", arange_int)

---

[來源: tools_numpy] # 浮點數：[1., 2., 3., 4., 5.]
arange_float = np.arange(1.0, 6.0)
print("arange_float =", arange_float)

---

[來源: tools_numpy] # 指定步長 0.5：[1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5]
arange_step = np.arange(1, 6, 0.5)
print("arange_step =", arange_step)

---

[來源: tools_numpy] # 2. 比較 arange 和 linspace 的差異（使用相同範圍）

---

[來源: tools_numpy] # 試圖用 arange 切分出 6 個點 (步長 = 總長度 / 5)

---

[來源: tools_numpy] # 注意：由於浮點數誤差，arange 有時可能無法準確包含或排除終點
arange_like_linspace = np.arange(0, 10/3, (10/3)/5)
print("arange_like_linspace =", arange_like_linspace)

---

[來源: tools_numpy] # 建立 6 個點，自動計算步長，且精確包含終點
linspace = np.linspace(0, 10/3, 6)
print("linspace =", linspace)

---

[來源: tools_numpy] # 4. 從 Python 列表 (List) 轉換
array_from_list = np.array([[1, 2, 3, 4], [6, 7, 8, 9]])
print("array_from_list =\n", array_from_list)
print("array_from_list.shape =", array_from_list.shape)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] t =\n", array_from_list)
print("array_from_list.shape =", array_from_list.shape)
```

**✅ 程式碼逐行解析：**

1.  `np.arange(1, 6)`：建立從 1 到 5 的整數陣列（**不包含 6**，即左閉右開區間）。
2.  `np.arange(1.0, 6.0)`：建立從 1.0 到 5.0 的浮點數陣列，預設步長為 1。
3.  `np.arange(1, 6, 0.5)`：建立從 1 開始，步長為 0.5 的陣列，直到小於 6 為止。
4.  `np.arange(0, 10/3, (10/3)/5)`：嘗試模擬等間距，但在浮點數運算下，`arange` 對於「終點是否包含」的判定較不穩定，不建議這樣做。
5.  `np.linspace(0, 10/3, 6)`：在 0 到 10

---

[來源: tools_numpy] 10/3)/5)`：嘗試模擬等間距，但在浮點數運算下，`arange` 對於「終點是否包含」的判定較不穩定，不建議這樣做。
5.  `np.linspace(0, 10/3, 6)`：在 0 到 10/3 之間建立 **6 個**等間距點（預設**包含終點**）。這是在浮點數範圍取樣的最佳方式。
6.  `np.array(...)`：將 Python 的巢狀列表 `[[...], [...]]` 轉換為 NumPy 的二維陣列。
7.  `.shape`：屬性顯示陣列為 `(2, 4)`，代表 2 列 4 行。

**🎯 重點摘要：**

---

[來源: tools_numpy] ython 的巢狀列表 `[[...], [...]]` 轉換為 NumPy 的二維陣列。
7.  `.shape`：屬性顯示陣列為 `(2, 4)`，代表 2 列 4 行。

**🎯 重點摘要：**

* **核心區別**：
      * `arange` 關注的是 **「步長 (Step)」**（例如：每隔 0.5 走一步）。
      * `linspace` 關注的是 **「點數 (Count)」**（例如：我要這段路中間有 10 個點）。
  * **潛在陷阱**：使用 `arange` 處理浮點數時，因為二進位浮點數精度的關係，可能會導致結果陣列的長度不如預期（多一個或少一個點）。
  * **最佳實踐**：
      * **整數序列** ➝ 使用 `arange`。
      * **浮點數範圍 / 固定取樣點數** ➝ 優先使用 `linspace`。

---

[來源: tools_numpy] ### 隨機陣列

💡實際應用情境： 在模擬資料、初始化神經網路權重或進行隨機取樣時，需要產生隨機數陣列，以引入變異性和不確定性。

此節介紹 NumPy 的 `random.rand` 和 `random.randn` 函數，分別用於建立**均勻分佈**（範圍 [0, 1)）和**常態分佈**（均值 0，標準差 1）的隨機浮點數陣列。

```python

---

[來源: tools_numpy] # 均勻分佈隨機數
rand = np.random.rand(3, 5)
print("rand =", rand)

---

[來源: tools_numpy] # 常態分佈隨機數
randn = np.random.randn(3, 5)
print("randn =", randn)
```

**✅ 程式碼逐行解析：**

1. `rand = np.random.rand(3, 5)`：建立形狀為 (3, 5) 的隨機浮點數陣列，範圍在 [0, 1) 間的均勻分佈 (Uniform Distribution)。
2. `print("rand =", rand)`：輸出均勻分佈隨機陣列。
3. `randn = np.random.randn(3, 5)`：建立形狀為 (3, 5) 的隨機浮點數陣列，均值 0 標準差 1 的常態分佈 (Normal Distribution)。
4. `print("randn =", randn)`：輸出常態分佈隨機陣列。

**🎯 重點摘要：**

---

[來源: tools_numpy] 機浮點數陣列，均值 0 標準差 1 的常態分佈 (Normal Distribution)。
4. `print("randn =", randn)`：輸出常態分佈隨機陣列。

**🎯 重點摘要：**

- **核心功能**：rand 產生均勻分佈，randn 產生常態分佈的隨機數。
- **潛在問題**：隨機數每次執行結果不同，測試時需設定種子確保重現性。
- **最佳使用情境**：模擬資料、初始化權重、隨機取樣等需要隨機性的應用。

---

[來源: tools_numpy] ### 使用函數建立

💡實際應用情境： 當陣列中的元素值取決於其「位置索引（Index）」時（例如建立網格座標、棋盤格紋或距離矩陣），使用 fromfunction 可以避免撰寫低效的巢狀迴圈。

我們使用 NumPy 的 fromfunction 函數，它會將每個座標點的索引值傳入指定的函數中進行計算。

```python

---

[來源: tools_numpy] # 定義計算規則
def rule_function(i, j):
    # i 代表第 0 軸 (row) 的索引
    # j 代表第 1 軸 (col) 的索引
    return j + 10 * i

---

[來源: tools_numpy] # dtype=int 確保傳入函數的座標是整數 (預設為 float)
result = np.fromfunction(rule_function, (3, 2), dtype=int)

print("fromfunction result shape:", result.shape)
print("result =\n", result)
```

**✅ 程式碼逐行解析：**

1. `def rule_function(i, j):`  
    定義一個函數，參數數量需與陣列維度相符（此例為 2 維）。

2. `return j + 10 * i`  
    計算邏輯：十位數代表列索引，個位數代表行索引。

---

[來源: tools_numpy] on(i, j):`  
    定義一個函數，參數數量需與陣列維度相符（此例為 2 維）。

2. `return j + 10 * i`  
    計算邏輯：十位數代表列索引，個位數代表行索引。

3. `result = np.fromfunction(rule_function, (3, 2), dtype=int)`  
    建立形狀為 (3, 2) 的陣列。NumPy 會自動生成座標網格：  
    - `i` 為 `[[0, 0], [1, 1], [2, 2]]`  
    - `j` 為 `[[0, 1], [0, 1], [0, 1]]`  
    並將座標傳入函數計算。

4. `print(...)`  
    輸出結果，例如 (1, 0) 的位置值為 10，(2, 1) 的位置值為 21。

**🎯 重點摘要：**

---

[來源: tools_numpy] 1]]`  
    並將座標傳入函數計算。

4. `print(...)`  
    輸出結果，例如 (1, 0) 的位置值為 10，(2, 1) 的位置值為 21。

**🎯 重點摘要：**

- **核心功能**：利用座標（索引）生成陣列內容，屬於向量化運算，效能遠高於 Python 迴圈。
- **參數對應**：函數的第 1 個參數對應 Axis 0（列），第 2 個參數對應 Axis 1（行），依序類推，順序一致。
- **注意事項**：fromfunction 預設傳入座標為浮點數，若需整數運算請加上 `dtype=int`。
- **最佳使用情境**：建立幾何變換矩陣、特殊數列圖案（如 Pascal 三角形）、初始化物理場等。

---

---

[來源: tools_numpy] ## 陣列資料 (Array Data)

💡 **實際應用情境：** 在處理大規模資料集時，精確掌握 NumPy 陣列的資料類型（Data Types）和記憶體佈局（Memory Layout）是提升效能和避免錯誤的關鍵。資料類型決定了數值範圍、精確度和運算效率，例如使用 `int32` 而非 `int64` 可節省記憶體並加速計算；記憶體佈局則影響資料存取速度，如連續佈局（Contiguous Layout）能充分利用 CPU 快取，減少快取遺漏（Cache Miss）。在機器學習中，這有助於優化模型訓練速度和資源使用；在影像處理中，選擇合適類型（如 `uint8`）可避免溢位並提升處理效率。若忽略這些細節，可能導致記憶體浪費、運算緩慢或資料損壞，因此建議在陣列建立時明確指定 `dtype`，並使用工具如 `nbytes` 屬性監控記憶體使用量。

---

[來源: tools_numpy] ### 資料類型

```python
c = np.arange(1, 6)
print("c.dtype =", c.dtype, "c =", c)

c_float = np.arange(1.0, 6.0)
print("c_float.dtype =", c_float.dtype, "c_float =", c_float)

c_complex = np.arange(1, 6, dtype=np.complex64)
print("c_complex.dtype =", c_complex.dtype, "c_complex =", c_complex)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] complex64)
print("c_complex.dtype =", c_complex.dtype, "c_complex =", c_complex)
```

**✅ 程式碼逐行解析：**

1. `c = np.arange(1, 6)`：建立從 1 到 5 的整數陣列，預設為 int32 或 int64。
2. `print("c.dtype =", c.dtype, "c =", c)`：輸出陣列的資料類型和內容。
3. `c_float = np.arange(1.0, 6.0)`：建立浮點數陣列，自動推斷為 float64。
4. `print("c_float.dtype =", c_float.dtype, "c_float =", c_float)`：輸出浮點數陣列的類型和內容。
5. `c_complex = np.arange(1, 6, dtype=np.

---

[來源: tools_numpy] e =", c_float.dtype, "c_float =", c_float)`：輸出浮點數陣列的類型和內容。
5. `c_complex = np.arange(1, 6, dtype=np.complex64)`：明確指定複數類型建立陣列。
6. `print("c_complex.dtype =", c_complex.dtype, "c_complex =", c_complex)`：輸出複數陣列的類型和內容。

**🎯 重點摘要：**

- **核心功能**：NumPy 陣列有統一資料類型，可顯式指定 dtype。
- **潛在問題**：不同類型混合運算會自動提升，可能導致記憶體使用增加。
- **最佳使用情境**：根據資料範圍選擇適當類型，如 uint8 用於影像，float32 用於一般計算。

---

[來源: tools_numpy] ### 記憶體資訊

💡 實際應用情境： 在處理大規模資料時，了解陣列的記憶體佈局和大小對於效能調優和資源管理至關重要。

在這裡，我們將介紹 NumPy 陣列的 itemsize 和 data 屬性，用於檢查元素大小和存取原始位元組資料。

---

[來源: tools_numpy]  在處理大規模資料時，了解陣列的記憶體佈局和大小對於效能調優和資源管理至關重要。

在這裡，我們將介紹 NumPy 陣列的 itemsize 和 data 屬性，用於檢查元素大小和存取原始位元組資料。

```python
d_cpx64 = np.arange(1, 6, dtype=np.complex64)
print(f"The itemsize of d_cpx64 elements = {d_cpx64.itemsize}")

d_int32 = np.array([[1, 2], [-3000, 4000]], dtype=np.int32)
print(f"Raw byte representation of d_int32 array (first 20 bytes): {d_int32.data[:20].tobytes()}")
```

---

[來源: tools_numpy] int(f"Raw byte representation of d_int32 array (first 20 bytes): {d_int32.data[:20].tobytes()}")
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] epresentation of d_int32 array (first 20 bytes): {d_int32.data[:20].tobytes()}")
```

**✅ 程式碼逐行解析：**

1. `d_cpx64 = np.arange(1, 6, dtype=np.complex64)`：建立複數陣列。
2. `print(f"The itemsize of d_cpx64 elements = {d_cpx64.itemsize}")`：輸出每個元素的位元組大小（複數 64 為 8 位元組）。
3. `d_int32 = np.array([[1, 2], [-3000, 4000]], dtype=np.int32)`：建立 int32 二維陣列。
4. `print(f"Raw byte representation of d_int32 array (first 2

---

[來源: tools_numpy] 000]], dtype=np.int32)`：建立 int32 二維陣列。
4. `print(f"Raw byte representation of d_int32 array (first 20 bytes): {d_int32.data[:20].tobytes()}")`：輸出陣列資料緩衝區的前 20 位元組，以位元組字串格式顯示。
    - `data` 屬性提供對底層位元組的直接存取，`tobytes()` 方法將其轉換為位元組字串。
    - 注意直接操作 `data` 可能導致資料損壞，應謹慎使用。

**🎯 重點摘要：**

- **核心功能**：itemsize 顯示元素大小，data 提供原始位元組存取。
- **潛在問題**：直接操作 data 緩衝區可能導致資料損壞，應謹慎使用。
- **最佳使用情境**：檢查記憶體使用量或與低階程式設計介面互動時。

---

[來源: tools_numpy] e 顯示元素大小，data 提供原始位元組存取。
- **潛在問題**：直接操作 data 緩衝區可能導致資料損壞，應謹慎使用。
- **最佳使用情境**：檢查記憶體使用量或與低階程式設計介面互動時。

---

---

[來源: tools_numpy] ## 重塑陣列 (Reshaping Arrays)

💡 實際應用情境： 陣列重塑（reshape）是資料預處理和張量操作的常用技巧，能將一維資料轉換為多維結構。例如，將原始數據重塑為影像矩陣，方便後續分析或模型訓練。這在資料整理、特徵工程及深度學習中非常實用。

---

[來源: tools_numpy] ### 就地重塑 (使用 shape 屬性)

當需要直接改變陣列的形狀而不複製資料時，可利用就地重塑，快速將一維資料轉換為多維結構，方便後續分析或模型訓練。

```python
a24 = np.arange(24)
print("a24 =", a24)
print("a24.shape =", a24.shape)

a24.shape = (6, 4)
print("after reshape a24 =", a24)
print("a24.shape =", a24.shape)

a24.shape = (2, 3, 4)
print("3D a24 =", a24)
print("a24.shape =", a24.shape)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] e)

a24.shape = (2, 3, 4)
print("3D a24 =", a24)
print("a24.shape =", a24.shape)
```

**✅ 程式碼逐行解析：**

1. `a24 = np.arange(24)`：建立包含 0 到 23 的陣列。
2. `print("a24 =", a24)`：輸出原始一維陣列。
3. `print("a24.shape =", a24.shape)`：輸出原始形狀 (24,)。
4. `a24.shape = (6, 4)`：將形狀修改為 (6, 4)，元素數必須相同。
5. `print("after reshape a24 =", a24)`：輸出重塑後的陣列。
6. `print("a24.shape =", a24.shape)`：輸出新形狀。
7. `a24.shape = (2, 3, 4)`：進一步重塑為三維。
8. `print("3D a24 =", a24)`：輸出三維陣列。
9. `print("a24.shape =", a24.shape)`：輸出三維形狀。

---

[來源: tools_numpy] 2, 3, 4)`：進一步重塑為三維。
8. `print("3D a24 =", a24)`：輸出三維陣列。
9. `print("a24.shape =", a24.shape)`：輸出三維形狀。

**🎯 重點摘要：**

- **核心功能**：直接修改 shape 屬性進行就地重塑，元素數必須保持不變。
- **潛在問題**：形狀乘積不等於元素數會引發錯誤。
- **最佳使用情境**：需要改變陣列維度結構但保持資料不變的情況。

---

[來源: tools_numpy] ### 使用 reshape函數

💡實際應用情境： 在資料分析和機器學習中，經常需要將一維資料重塑為多維結構以符合模型輸入需求。使用 reshape 函數可以方便地創建新的視圖，適應不同的形狀要求。

```python
a12 = np.arange(12)
a12_rs = a12.reshape(3, 4)
print("a12_rs =", a12_rs)

a12_rs[1, 2] = 999
print(f'a12_rs after modification =\n{a12_rs}')
print(f'a12 after a12_rs modification =\n{a12}')
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] fter modification =\n{a12_rs}')
print(f'a12 after a12_rs modification =\n{a12}')
```

**✅ 程式碼逐行解析：**

1. `a12_rs = a12.reshape(3, 4)`：建立 a12 的重塑視圖，形狀為 (3, 4)。
2. `print("a12_rs =", a12_rs)`：輸出重塑後的陣列。
3. `a12_rs[1, 2] = 999`：修改 a12_rs 的元素。
4. `print(f'a12_rs after modification =\n{a12_rs}')`：顯示修改後的 a12_rs。
5. `print(f'a12 after a12_rs modification =\n{a12}')`：顯示原始陣列 a12 也被修改，因為共享資料。

**🎯 重點摘要：**

---

[來源: tools_numpy] 的 a12_rs。
5. `print(f'a12 after a12_rs modification =\n{a12}')`：顯示原始陣列 a12 也被修改，因為共享資料。

**🎯 重點摘要：**

- **核心功能**：reshape 返回新視圖，*與原陣列共享資料*。
- **潛在問題**：修改任一陣列都會影響另一個，造成意外副作用。
- **最佳使用情境**：需要不同形狀視圖但不想複製資料時。

---

[來源: tools_numpy] ### 陣列展平：使用 ravel

💡實際應用情境： 在將多維陣列輸入到只能接受一維資料的演算法（如 Scikit-Learn 的某些模型）或進行繪圖時，我們需要將陣列「拉平」。ravel 是最有效率的方法，因為它通常不會複製資料。

```python

---

[來源: tools_numpy] # 1. 建立一個多維陣列 md_array
md_array = np.array([[1, 2, 3], [4, 5, 6]])
print("Original md_array:\n", md_array)

---

[來源: tools_numpy] # 2. 使用 ravel 展平
flat = md_array.ravel()
print("flat =", flat)

---

[來源: tools_numpy] # 3. 驗證：修改展平後的視圖，是否會影響原陣列？
flat[0] = 888 
print("\nAfter modifying flat:")
print("md_array (Original) has changed:\n", md_array)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] "\nAfter modifying flat:")
print("md_array (Original) has changed:\n", md_array)
```

**✅ 程式碼逐行解析：**

1. `md_array = np.array(...)`：初始化一個 (2, 3) 的二維陣列。
2. `flat = md_array.ravel()`：將多維陣列展平為一維。注意：這裡返回的是「視圖 (View)」，記憶體位置仍指向原本的資料。
3. `flat[0] = 888`：修改 flat 的第一個元素。
4. `print("md_array (Original) has changed:\n", md_array)`：你會發現原陣列 md_array 的 (0, 0) 位置也變成了 888。這證明了兩者共享記憶體。

**🎯 重點摘要：**

---

[來源: tools_numpy] Original) has changed:\n", md_array)`：你會發現原陣列 md_array 的 (0, 0) 位置也變成了 888。這證明了兩者共享記憶體。

**🎯 重點摘要：**

- **核心功能**：`ravel` 可將多維陣列展平成一維序列。
- **關鍵特性**：通常返回「視圖 (View)」而非複製，速度快且節省記憶體。
- **潛在風險**：因為與原陣列共享資料，修改展平後的結果會同步影響原始陣列。若需獨立副本，請使用 `flatten()`。
- **最佳使用情境**：適合唯讀存取所有元素，或需同步修改原資料時。

---

---

[來源: tools_numpy] ## 算術運算 (Arithmetic Operations)

💡**實際應用情境：** 在資料科學中，我們很少使用迴圈來逐個計算數據。NumPy 的算術運算皆為 **元素級 (Element-wise)**，這意味著運算是並行作用於陣列中對應位置的元素。這在影像處理（如調整亮度 img + 10）或神經網路權重更新中無處不在。

```python
a = np.array([26, 22, 30, 11])
b = np.array([4, 3, 2, 1])

---

[來源: tools_numpy] # 基礎四則運算
print("a + b =", a + b)  # [30 25 32 12]
print("a - b =", a - b)  # [22 19 28 10]
print("a * b =", a * b)  # [104 66 60 11] (注意：這是元素對應相乘)
print("a / b =", a / b)  # [6.5 7.333 15. 11.] (結果為浮點數)

---

[來源: tools_numpy] # 進階運算
print("a // b =", a // b) # [6 7 15 11] (整數除法/地板除法)
print("a % b =", a % b)   # [2 1 0 0] (取餘數)
print("a ** b =", a ** b) # [456976 10648 900 11] (指數運算)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] b)   # [2 1 0 0] (取餘數)
print("a ** b =", a ** b) # [456976 10648 900 11] (指數運算)
```

**✅ 程式碼逐行解析：**

1. a 與 b：建立兩個形狀相同的陣列。NumPy 要求參與運算的陣列形狀必須相同，或符合「廣播 (Broadcasting)」規則。
2. a * b：特別注意，這是將對應位置的數字相乘（例如 26*4），並非線性代數中的矩陣乘法。
3. a / b：即使輸入是整數，普通除法 / 的結果預設也會是浮點數（float）。
4. a // b：若只需要整數部分的商，使用 //。
5. a ** b：對應元素的指數運算（例如 $26^4$），這在 Python 中非常方便。

**🎯 重點摘要：**

---

[來源: tools_numpy] 浮點數（float）。
4. a // b：若只需要整數部分的商，使用 //。
5. a ** b：對應元素的指數運算（例如 $26^4$），這在 Python 中非常方便。

**🎯 重點摘要：**

- **核心機制**：所有標準運算子（`+`, `-`, `*`, `/`）在 NumPy 中預設為元素級（Element-wise）運算，即對應位置的元素逐一計算。

- **常見陷阱**：
    - `*` 是元素乘法，不是矩陣乘法。
    - 若需矩陣乘法（Matrix Multiplication），請使用 `@` 運算子（如 `A @ B`）或 `np.dot(A, B)`。
- **最佳使用情境**：適用於批次數據的統一數學變換，如單位換算、影像濾鏡、訊號增益等。

---

---

[來源: tools_numpy] ## 廣播 (Broadcasting)

💡 **實際應用情境：** 廣播機制允許 NumPy 在不同形狀的陣列之間進行算術運算，而無需手動複製資料。這在機器學習中極為常見，例如：將一個偏置向量 (Bias) 加到整個資料批次 (Batch) 的每一行，或進行影像的通道正規化。

廣播的優勢在於能自動擴展較小的陣列，使其形狀與較大的陣列相容，從而實現批次運算、特徵標準化、資料中心化等操作。例如，對每個樣本加上同一個偏置向量，或將一維陣列加到多維矩陣的每一行或每一列。這不僅簡化程式碼，也提升運算效率，避免不必要的記憶體複製。

常見應用包括：

- 對所有樣本批次加上偏置或標準化參數
- 對影像資料進行通道級正規化（如每個 RGB 通道減去平均值）
- 批次資料的特徵縮放與中心化
- 將純量或一維陣列自動擴展到多維陣列進行運算

---

[來源: tools_numpy] 見應用包括：

- 對所有樣本批次加上偏置或標準化參數
- 對影像資料進行通道級正規化（如每個 RGB 通道減去平均值）
- 批次資料的特徵縮放與中心化
- 將純量或一維陣列自動擴展到多維陣列進行運算

廣播機制是 NumPy 向量化運算的基礎，建議在資料處理和科學計算中充分利用，能顯著提升程式效能與可讀性。

---

[來源: tools_numpy] ### 廣播規則

廣播規則（Broadcasting Rules）是 NumPy 允許不同形狀的陣列進行算術運算的核心機制。其運作方式如下：

1. **維度對齊**：從右往左比較兩個陣列的每個軸（dimension）。
2. **相容條件**：每個軸的長度必須「相等」或其中之一為 1。
3. **自動擴展**：若某一軸長度為 1，NumPy 會自動將其複製（虛擬擴展）至另一陣列的長度。
4. **最終形狀**：運算結果的形狀為兩陣列在每個軸上的最大值。

**範例：**

---

[來源: tools_numpy] 之一為 1。
3. **自動擴展**：若某一軸長度為 1，NumPy 會自動將其複製（虛擬擴展）至另一陣列的長度。
4. **最終形狀**：運算結果的形狀為兩陣列在每個軸上的最大值。

**範例：**

- (2, 3) + (3,) → (2, 3)（第二個陣列自動擴展為 (2, 3)）  
- (4, 1, 6) + (3, 6) → (4, 3, 6)（第二個陣列自動擴展為 (1, 3, 6)，再複製到 (4, 3, 6)）  
- (5, 4) + (1,) → (5, 4)（純量或一維長度為 1時可廣播到任意形狀）

**注意：**

- 若任一軸長度不相等且都不為 1，則無法廣播，會拋出錯誤。
- 廣播不會真正複製資料，只是調整資料讀取方式，效能高且節省記憶體。

**實務建議：**

---

[來源: tools_numpy] 為 1時可廣播到任意形狀）

**注意：**

- 若任一軸長度不相等且都不為 1，則無法廣播，會拋出錯誤。
- 廣播不會真正複製資料，只是調整資料讀取方式，效能高且節省記憶體。

**實務建議：**

- 若遇到形狀不符，可用 `reshape` 或 `np.newaxis` 調整陣列形狀以符合廣播規則。

```python

---

[來源: tools_numpy] # 模擬資料：2 個樣本，每個樣本有 3 個特徵 (Shape: 2, 3)
data = np.array([[1, 2, 3],
                 [10, 20, 30]])

---

[來源: tools_numpy] # 模擬偏置：要加到每個樣本上的數值 (Shape: 3,)
bias = np.array([1, 0, 1])

print("Data shape:", data.shape)
print("Bias shape:", bias.shape)

---

[來源: tools_numpy] # 觸發廣播：Bias 會自動「擴展」應用到 Data 的每一列
result = data + bias
print("Broadcast result =\n", result)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] 會自動「擴展」應用到 Data 的每一列
result = data + bias
print("Broadcast result =\n", result)
```

**✅ 程式碼逐行解析：**

1. `data` 是形狀 `(2, 3)` 的矩陣。
2. `bias` 是形狀 `(3,)` 的向量。
3. NumPy 廣播規則：
    - 從右往左比對維度。
    - 最後一個維度：`data` 和 `bias` 都是 3，直接匹配。
    - 倒數第二個維度：`data` 是 2，`bias` 沒有（視為 1），1 可廣播擴展為 2。
4. 廣播結果：
    - `bias` 被邏輯上複製為兩行，分別加到 `data` 的每一列。
    - 計算如下：
        - 第一列：[1, 2, 3] + [1, 0, 1] = [2, 2, 4]
        - 第二列：[10, 20, 30] + [1, 0, 1] = [11, 20, 31]

---

[來源: tools_numpy] - 第一列：[1, 2, 3] + [1, 0, 1] = [2, 2, 4]
        - 第二列：[10, 20, 30] + [1, 0, 1] = [11, 20, 31]

**🎯 重點摘要：**

- **核心規則**：形狀從**右向左**對齊。只要對應軸的長度**相等**，或其中一方為 **1**，即可進行廣播運算。
- **優勢**：廣播不會真正複製資料，只是調整資料讀取方式，因此能節省記憶體並提升運算速度。
- **常見錯誤**：若維度不匹配且都不為 1（例如 (2, 3) 加 (2,)），會拋出 ValueError。此時可用 `reshape` 或 `np.newaxis` 調整維度以符合廣播規則。

---

[來源: tools_numpy] ### 更多廣播範例

💡 **實際應用情境：** 這些不同的廣播方式在資料處理中超級實用：

- **列向量（形狀為 (N, 1)）**：適合「列級」運算，例如對每一**列 (Axis 0)**（樣本）乘上不同的正規化因子或權重。這種方式常見於批次資料的標準化或特徵縮放。

- **一維向量（形狀為 (M,)）**：適合「行級」運算，例如將偏置（Bias）或特徵平均值加到所有樣本的每一**行 (Axis 1)**。這在特徵工程、資料中心化等場景非常常見。

- **純量（單一數值）**：可自動廣播到整個陣列，適用於所有元素的統一加減或乘除。

這些廣播機制讓 NumPy 能夠用簡潔的語法完成複雜的批次運算，大幅提升程式效率與可讀性。

```python
b = np.arange(6).reshape(2, 3)
print("b =\n", b,"\n")

---

[來源: tools_numpy] # Column vector broadcasting (shape: (2, 1))
result2 = b + [[10], [20]]
print("b + [[10], [20]] =\n", result2,"\n")

---

[來源: tools_numpy] # Row vector broadcasting (shape: (3,))
result3 = b + [100, 200, 300]
print("b + [100, 200, 300] =\n", result3,"\n")

---

[來源: tools_numpy] # Scalar broadcasting
result4 = b + 10000
print("b + 10000 =\n", result4)
```

**✅ 程式碼逐行解析：**

1. `b = np.arange(6).reshape(2, 3)`：建立 2x3 陣列，形狀為 (Axis 0: 2, Axis 1: 3)。
2. `result2 = b + [[10], [20]]`：`b` 為 (2, 3)，`[[10], [20]]` 為 (2, 1)，廣播沿著Axis 1 擴展 (因為其長度為 1，可以擴展)。
3. `result3 = b + [100, 200, 300]`：`b` 為 (2, 3)，`[100, 200, 300]` 為 (3,)，廣播沿著Axis 0 擴展。
4. `result4 = b + 10000`：所有元素加同一值。

---

[來源: tools_numpy] , 200, 300]`：`b` 為 (2, 3)，`[100, 200, 300]` 為 (3,)，廣播沿著Axis 0 擴展。
4. `result4 = b + 10000`：所有元素加同一值。

**🎯 重點摘要：**

- **核心功能**：廣播的核心是**從右向左對齊**且**長度必須相等或為1**。
- **潛在問題**：複雜廣播可能難以理解，建議用簡單規則。
- **關鍵區分 (軸)**：在二維陣列中， `(N, 1)` 和 `(N,)` 的效果截然不同：
    - **`(N, 1)` 向量**：強迫擴展發生在 Axis 1，常用於對 Axis 0 進行獨立操作。
    - **`(N,)` 向量**：強迫擴展發生在 Axis 0，常用於對 Axis 1 進行獨立操作（如加 Bias）。
- **最佳使用情境**：特徵標準化、批次偏移等需要一致運算的情況。

---

[來源: tools_numpy] ### 類型提升

💡實際應用情境： 類型提升是 NumPy 確保混合資料類型運算正確性的機制。它會自動將所有運算元提升到一個「能安全容納所有可能結果」的通用類型。這在處理來自不同來源（如資料庫或檔案）的數據時尤其重要，但需注意它對記憶體和性能的影響。

```python
c1 = np.arange(0, 5, dtype=np.uint8)

---

[來源: tools_numpy] # c1: [0 1 2 3 4], dtype=uint8 (無符號整數)

c2 = c1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)

---

[來源: tools_numpy] # c2: [ 5 7 9 11 13]

c3 = c1 + 1.5

---

[來源: tools_numpy] # c3: [1.5 2.5 3.5 4.5 5.5]

print("Type of c1 is", c1.dtype)
print("Type of c2 is", c2.dtype) # 預期: int16 (或更高)
print("Type of c3 is", c3.dtype) # 預期: float64
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] is", c2.dtype) # 預期: int16 (或更高)
print("Type of c3 is", c3.dtype) # 預期: float64
```

**✅ 程式碼逐行解析：**

1. `c1 = np.arange(0, 5, dtype=np.uint8)`：建立 uint8 (8 位元無符號整數) 陣列，範圍 0-255。
2. `c2 = c1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)`：與 int8 (8 位元有符號整數) 陣列相加，觸發類型提升以避免溢位。
3. `c3 = c1 + 1.5`：與浮點數相加，進一步提升為 float64 以保持精確度。
4. `print(...)`：輸出各陣列的資料類型，觀察類型提升結果。

**🎯 重點摘要：**

---

[來源: tools_numpy] 。
3. `c3 = c1 + 1.5`：與浮點數相加，進一步提升為 float64 以保持精確度。
4. `print(...)`：輸出各陣列的資料類型，觀察類型提升結果。

**🎯 重點摘要：**

- **核心功能**：NumPy 的類型提升機制自動選擇能安全容納運算結果的通用類型，確保數值正確性，但可能導致記憶體使用量增加（如從 uint8 提升至 float64）。

- **潛在問題**：
    1. **效能與記憶體**：意外提升至高位元類型（如 float64）會大幅增加記憶體佔用和運算成本。
    2. **精確度**：提升雖常為精確度考量，但浮點數運算可能引入近似誤差。

- **最佳使用情境**：適用於混合類型運算的自動處理；為效能優化，建議預先統一資料類型，如統一使用 np.float32。

---

---

[來源: tools_numpy] ## 條件運算子 (Conditional Operators)
💡 **實際應用情境：** 條件運算是資料過濾和布林索引的基礎。在資料分析中，它用於快速識別和選擇滿足特定標準（例如，找出所有收入超過 $50000 且年齡小於 30 歲的用戶）的值，是取代傳統 Python 迴圈進行資料清理和篩選的關鍵。

```python
m = np.array([23, -15, 32, 57])
print(f"m = {m}")
print("-" * 25)

---

[來源: tools_numpy] # 1. 元素級比較 (Element-wise Comparison)
mask = m < [16, 0, 35, 20]
print("m < [16, 0, 35, 20] =", mask)

---

[來源: tools_numpy] # Output: [False  True  True False]

---

[來源: tools_numpy] # 2. 純量比較 (Scalar Comparison) - 觸發廣播
mask2 = m < 30
print("Mask of m < 30 =", mask2)

---

[來源: tools_numpy] # Output: [ True  True False False]

---

[來源: tools_numpy] # 3. 布林索引 (Boolean Indexing)
filtered = m[mask2]
print("m[m < 30] =", filtered)

---

[來源: tools_numpy] # 4. 進階：結合多個條件 (使用 & 進行 AND 運算)

---

[來源: tools_numpy] # 注意：必須使用 & (位元運算子) 而非 Python 的 and
mask_combined = (m > 0) & (m < 30)
print("\nMask of (m > 0) & (m < 30) =", mask_combined)

---

[來源: tools_numpy] # Output: [ True False False False]
filtered_combined = m[mask_combined]
print("Combined filter =", filtered_combined)

---

[來源: tools_numpy] # Output: [23]
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] 1. `m = np.array([23, -15, 32, 57])`：建立測試陣列，包含四個整數元素。
2. `mask = m < [16, 0, 35, 20]`：進行元素級比較，將 m 的每個元素與對應位置的比較值進行小於比較，返回布林陣列。
    - `print("m < [16, 0, 35, 20] =", mask)`：輸出比較結果的布林陣列。
3. `mask2 = m < 30`：與純量 30 比較，觸發廣播機制，將 30 與 m 的所有元素比較。
    - `print("Mask of m < 30 =", mask2)`：輸出純量比較的布林陣列。
4. `filtered = m[mask2]`：使用布林索引，根據 mask2 選擇 m 中對應 True 的元素。
    - `print("m[m < 30] =", filtered)`：輸出過濾後的陣

---

[來源: tools_numpy] ltered = m[mask2]`：使用布林索引，根據 mask2 選擇 m 中對應 True 的元素。
    - `print("m[m < 30] =", filtered)`：輸出過濾後的陣列。
5. `mask_combined = (m > 0) & (m < 30)`：結合多個條件，使用位元運算子 & 進行 AND 運算，必須用括號包裹每個條件。
    - `print("\nMask of (m > 0) & (m < 30) =", mask_combined)`：輸出結合條件的布林陣列。
    - `filtered_combined = m[mask_combined]`：使用結合的布林遮罩進行索引。
    - `print("Combined filter =", filtered_combined)`：輸出結合過濾後的陣列。

**🎯 重點摘要：**

---

[來源: tools_numpy] ined]`：使用結合的布林遮罩進行索引。
    - `print("Combined filter =", filtered_combined)`：輸出結合過濾後的陣列。

**🎯 重點摘要：**

- **核心功能**：條件運算子（如 `<`, `>`, `==`, `!=` 等）返回布林陣列（遮罩），是進行資料過濾和條件選擇的關鍵工具。
- **布林索引特性**：使用布林遮罩進行索引時，返回的結果總是一維陣列，會丟失原始陣列的形狀（例如無法保留 `(N, M)` 的結構）。
- **多條件組合**：必須使用 NumPy 的位元邏輯運算子（`&` 而非 `and`，`|` 而非 `or`），並以括號包裹每個條件以確保正確的運算優先級。
- **最佳使用情境**：資料清理、條件式元素選擇、統計過濾和條件聚合。

---

---

[來源: tools_numpy] ## 數學和統計函數 (Math and Statistical Functions)

💡 **實際應用情境：** 這些函數在資料分析和統計計算中不可或缺，如計算平均值、標準差等。

---

[來源: tools_numpy] ### 陣列方法

💡 實際應用情境： 在資料分析中，經常需要計算整體或特定維度的統計量（如平均值、總和、最大值等）。NumPy 提供的陣列方法能高效地完成這些任務，避免使用迴圈，提高運算速度。

---

[來源: tools_numpy] ```python
a_stat = np.array([[-2.2, 3.0, 86], [3, 300, -3]])
print("a_stat =", a_stat)
print("mean =", a_stat.mean()) 
print("max =", a_stat.max())
print("sum =", a_stat.sum())
print("std =", a_stat.std())
print("var =", a_stat.var())
print("mean along axis 1 =", a_stat.mean(axis=1).reshape(-1, 1))

```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] a_stat.var())
print("mean along axis 1 =", a_stat.mean(axis=1).reshape(-1, 1))

```

**✅ 程式碼逐行解析：**

1. `a_stat = np.array([[-2.2, 3.0, 86], [3, 300, -3]])`：建立二維陣列，維度為 (2, 3)。
2. `print("a_stat =", a_stat)`：輸出陣列。
3. `print("mean =", a_stat.mean())`：計算所有元素的平均值。
4. `print("max =", a_stat.max())`：找到最大值。
5. `print("sum =", a_stat.sum())`：計算總和。
6. `print("std =", a_stat.std())`：計算標準差。
7. `print("var ="

---

[來源: tools_numpy] 。
5. `print("sum =", a_stat.sum())`：計算總和。
6. `print("std =", a_stat.std())`：計算標準差。
7. `print("var =", a_stat.var())`：計算變異數。
8. `print("mean along axis 1 =", a_stat.mean(axis=1).reshape(-1, 1))`：沿著第 1 軸計算平均值，並重塑為行向量 (column vector)（即形狀為 (n, 1) 的二維陣列）。

**🎯 重點摘要：**

---

[來源: tools_numpy] .mean(axis=1).reshape(-1, 1))`：沿著第 1 軸計算平均值，並重塑為行向量 (column vector)（即形狀為 (n, 1) 的二維陣列）。

**🎯 重點摘要：**

- **核心功能**：NumPy 陣列方法（如 `mean()`、`sum()`、`max()` 等）提供統計運算，*預設對所有元素進行計算*，支援向量化處理以提升效能。
- **潛在問題**：若不指定 `axis` 參數，運算會將多維陣列展平為一維後進行，可能導致意外結果或效能損失。
- **最佳使用情境**：適用於整體資料統計分析，如計算整個資料集的平均值、總和或標準差。

---

[來源: tools_numpy] ### 指定軸運算

在機器學習中，資料通常是多維的（例如：`[樣本, 時間步, 特徵]` 或 `[批次, 頻道, 高度, 寬度]`）。`axis` 參數允許我們計算特定維度的統計量，例如：計算所有樣本的平均值 (Mean)、或某個時間序列的總和 (Sum)。

---

[來源: tools_numpy] `[樣本, 時間步, 特徵]` 或 `[批次, 頻道, 高度, 寬度]`）。`axis` 參數允許我們計算特定維度的統計量，例如：計算所有樣本的平均值 (Mean)、或某個時間序列的總和 (Sum)。

```python
mat_axis = np.arange(30).reshape(2, 3, 5)
print("mat_axis =", mat_axis)
print("="*30)
print("mat_axis.shape =", mat_axis.shape)
print("sum axis 0 =", mat_axis.sum(axis=0))
print("sum axis 1 =", mat_axis.sum(axis=1))
print("sum axis (0,2) =", mat_axis.sum(axis=(0, 2)))
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] 1 =", mat_axis.sum(axis=1))
print("sum axis (0,2) =", mat_axis.sum(axis=(0, 2)))
```

**✅ 程式碼逐行解析：**

1. `mat_axis = np.arange(30).reshape(2, 3, 5)`：建立 2x3x5 三維陣列。
2. `print("mat_axis.shape =", mat_axis.shape)`：輸出形狀。
3. `print("sum axis 0 =", mat_axis.sum(axis=0))`：沿第 0 軸求和，代表著將兩個「3x5 平面」對應相加。結果陣列的形狀會是 (3, 5)。
4. `print("sum axis 1 =", mat_axis.sum(axis=1))`：沿第 1 軸求和，代表著將三個「2x5 平面」對應相加。結果陣列的形狀會是 (

---

[來源: tools_numpy] 狀會是 (3, 5)。
4. `print("sum axis 1 =", mat_axis.sum(axis=1))`：沿第 1 軸求和，代表著將三個「2x5 平面」對應相加。結果陣列的形狀會是 (2, 5)。
5. `print("sum axis (0,2) =", mat_axis.sum(axis=(0, 2)))`：沿 (0,2) 軸求和，將第 0 軸和第 2 軸的元素相加，結果陣列的形狀會是 (3, )。

**🎯 重點摘要：**

---

[來源: tools_numpy] is (0,2) =", mat_axis.sum(axis=(0, 2)))`：沿 (0,2) 軸求和，將第 0 軸和第 2 軸的元素相加，結果陣列的形狀會是 (3, )。

**🎯 重點摘要：**

- **核心功能**：聚合函數（如 sum, mean, max, std）搭配 axis 參數，能計算特定維度的統計量。
- **軸消失原則**：運算時，被指定的 axis 會在結果的形狀中消失。例如，對 (2, 3, 4) 沿 axis=0 求和，結果為 (3, 4)。
- **軸索引**：軸索引從 0 開始，依序往上數。使用負數（如 axis=-1）則表示從最後一個維度開始算。
- **最佳使用情境**：多維資料分析，例如在影像處理中，對 (高, 寬, 頻道) 的陣列沿 axis=2 求平均值，可以得到一張灰階影像。

---

[來源: tools_numpy] ### 通用函數(Universal Functions / ufunc)

💡 **實際應用情境：** 在 Python 原生語法中，若要計算百萬筆數據的平方或對數，需要寫 `for` 迴圈，速度極慢。NumPy 的 `ufunc` 是在底層 C 語言實作的「向量化」函數，能對陣列中的每個元素進行快速的並行運算。

---

[來源: tools_numpy] hon 原生語法中，若要計算百萬筆數據的平方或對數，需要寫 `for` 迴圈，速度極慢。NumPy 的 `ufunc` 是在底層 C 語言實作的「向量化」函數，能對陣列中的每個元素進行快速的並行運算。

```python
a = np.array([[-12.5, 3.14, 99], [10, 100, 1000]])
print("Original array:")
print(a)
print("-" * 30)
print("np.sin(a) =", np.sin(a))
print("np.square(a) =", np.square(a))
print("np.abs(a) =", np.abs(a))
print("np.exp(a) =", np.exp(a))
print("np.log(a) =", np.log(a))
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] ) =", np.abs(a))
print("np.exp(a) =", np.exp(a))
print("np.log(a) =", np.log(a))
```

**✅ 程式碼逐行解析：**

- `np.sin(a)`：計算正弦值，輸入值被視為弧度 (Radians) 而非角度。
- `np.square(a)`：計算元素級平方 ($x^2$)。
- `np.abs(a)`：計算絕對值 ($

|x|$)，負數轉正。


- `np.exp(a)`：計算指數函數 ($e^x$)。
- `np.log(a)`：計算自然對數 ($\ln x$)。  
  重要細節：由於陣列包含負數，該位置的結果會變成 `nan`，並且 Python 介面通常會顯示 `RuntimeWarning: invalid value encountered in log`。這是處理真實數據時常見的情況

---

[來源: tools_numpy] ，該位置的結果會變成 `nan`，並且 Python 介面通常會顯示 `RuntimeWarning: invalid value encountered in log`。這是處理真實數據時常見的情況

**🎯 重點摘要：**

---

[來源: tools_numpy] an`，並且 Python 介面通常會顯示 `RuntimeWarning: invalid value encountered in log`。這是處理真實數據時常見的情況

**🎯 重點摘要：**

- **核心功能**：ufunc 提供高效的「向量化」數學運算，直接作用於整個陣列，效能遠優於 Python 迴圈。
- **常見數學定義**：
    - `np.log` 是**自然對數 ($\ln$)**；若需底數為 10，請用 `np.log10`。
    - 三角函數（如 `sin`, `cos`）皆使用**弧度**。
- **異常處理**：當數學運算定義域不符（如對負數取對數、除以零）時，NumPy 通常不會報錯中斷程式，而是返回 `nan` (Not a Number) 或 `inf` (Infinity) 並給出警告。
- **最佳使用情境**：科學運算、訊號處理、特徵轉換（如將偏斜分佈的數據取 log 轉常態分佈）。

---

[來源: tools_numpy] ### 二元通用函數 (Binary ufuncs)

💡**實際應用情境：** 二元通用函數接受兩個陣列作為輸入，並進行元素級的運算。這在比較兩組數據（例如：比較預測值與真實值的大小，或取兩張圖像中較亮的像素）時非常有用。

```python
a = np.array([-1, -2, 3, 4])
b = np.array([7, 8, -9, 10])
print("a =", a)
print("b =", b)
print("-" * 30)

---

[來源: tools_numpy] # 1. 加法 (等同於 a + b)
print("np.add(a, b) =", np.add(a, b))

---

[來源: tools_numpy] # 2. 比較運算 (等同於 a > b)
print("np.greater(a, b) =", np.greater(a, b))

---

[來源: tools_numpy] # Output: [False False  True False]

---

[來源: tools_numpy] # 3. 元素級最大值 (注意：這不是找出整個陣列的最大值)
print("np.maximum(a, b) =", np.maximum(a, b))

---

[來源: tools_numpy] # Output: [ 7  8  3 10] (取兩者中較大者)
```

**✅ 程式碼逐行解析：**

1. `np.add(a, b)`：元素級相加。這與運算子 `a + b` 功能完全相同。
2. `np.greater(a, b)`：元素級比較。這與運算子 `a > b` 功能完全相同，返回布林陣列。
3. `np.maximum(a, b)`：逐一比較 `a` 和 `b` 對應位置的元素，並保留較大的那個值。例如第一個位置：-1 vs 7，取 7。

**🎯 重點摘要：**

---

[來源: tools_numpy] 全相同，返回布林陣列。
3. `np.maximum(a, b)`：逐一比較 `a` 和 `b` 對應位置的元素，並保留較大的那個值。例如第一個位置：-1 vs 7，取 7。

**🎯 重點摘要：**

- **核心功能**：二元 ufunc 進行「一對一」的元素級運算，若陣列形狀不同，會自動觸發廣播機制。
- **常見陷阱 (maximum vs max)**：
    - `np.maximum(a, b)`：是二元運算，比較兩個陣列，返回一個新陣列。
    - `np.max(a)` (或 `a.max()`)：是聚合運算，找出單一陣列中的最大值，返回一個純量。
    - 請勿混用。
- **最佳使用情境**：需要對兩組數據進行邏輯比較或數值篩選（例如 ReLU 函數的實作就是 `np.maximum(0, x)`）。

---

---

[來源: tools_numpy] ## 陣列索引 (Array Indexing)

💡 **實際應用情境：** 索引是存取和修改陣列元素的關鍵，在資料選擇和特徵工程中廣泛應用。

例如：

- 影像處理，你可能需要提取特定像素區域（如裁剪臉部特徵）；
- 在時間序列分析中，索引能快速篩選特定時間段的資料（如過去一週的銷售記錄）；
- 在機器學習，索引用於交叉驗證 (Cross Validation) 的資料分割或特徵子集選擇。

此外，索引支援就地修改，能快速更新大型資料集而不需重建陣列，這在即時資料處理和記憶體受限的環境中特別重要。

NumPy 支援多種索引方式，包括基本切片（slicing）、花式索引（fancy indexing）和布林索引（boolean indexing），可靈活存取和操作一維向量到多維張量。這些語法不僅提升程式碼可讀性，也優化資料處理效能，適用於各種科學計算與資料分析場景。

---

[來源: tools_numpy] ### 一維陣列

💡 **實際應用情境：** 一維陣列（1D Array）是 NumPy 最基礎的資料結構，常用於表示序列資料，如時間序列、訊號波形或簡單的數值列表。在資料科學和機器學習中，一維陣列的索引操作是處理大型資料集的關鍵，能讓你快速存取特定元素、進行切片操作或就地修改數據，提升程式效率。

掌握 NumPy 一維陣列索引技巧，包括基本索引、切片（slicing）、負索引和步長（step），是學習多維陣列操作的基礎。這些操作不僅支援高效的資料存取，還能避免使用低效的 Python 迴圈，適用於資料預處理、特徵工程和演算法實作等場景。

---

[來源: tools_numpy] 索引、切片（slicing）、負索引和步長（step），是學習多維陣列操作的基礎。這些操作不僅支援高效的資料存取，還能避免使用低效的 Python 迴圈，適用於資料預處理、特徵工程和演算法實作等場景。

```python
a = np.linspace(100, 109, 10, dtype=int)
print("Example 1-D array a =", a)
print("-" * 30)

print("a[3] =", a[3])
print("a[3:5] =", a[3:5])
print("a[3:-1] =", a[3:-1])
print("a[3:] =", a[3:])
print("a[:3] =", a[:3])
print("a[3::2] =", a[3::2])
print("a[::-1] =", a[::-1])

a[3] = 99
print(f"a after modification = {a}")

a[1:4] = [-87, 88, 89]
print(f"a after slice assignment = {a}")
```

---

[來源: tools_numpy] (f"a after modification = {a}")

a[1:4] = [-87, 88, 89]
print(f"a after slice assignment = {a}")
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] ication = {a}")

a[1:4] = [-87, 88, 89]
print(f"a after slice assignment = {a}")
```

**✅ 程式碼逐行解析：**

1. `a = np.linspace(100, 109, 10, dtype=int)`：使用 linspace 建立從 100 到 109 的 10 個整數陣列。
2. `print("a[3] =", a[3])`：存取第 4 個元素（索引從 0 開始）。
3. `print("a[3:5] =", a[3:5])`：切片索引 3 到 4（不含 5）。
4. `print("a[3:-1] =", a[3:-1])`：從索引 3 到倒數第 2 個。
5. `print("a[3:] =", a[3:])`： 從索引 3 到結尾。
6. `print("a[:3] =", a[:3])

---

[來源: tools_numpy] ] =", a[3:-1])`：從索引 3 到倒數第 2 個。
5. `print("a[3:] =", a[3:])`： 從索引 3 到結尾。
6. `print("a[:3] =", a[:3])`：前 3 個元素。
7. `print("a[3::2] =", a[3::2])`：從索引 3 開始，取步長(Step)為 2。
8. `print("a[::-1] =", a[::-1])`：反轉陣列。
9. `a[3] = 99`：修改第 4 個元素為 99。
10. `print(f"a after modification = {a}")`：輸出修改後陣列。
11. `a[1:4] = [-87, 88, 89]`：將索引 1 到 3 的元素替換為新值。
12. `print(f"a after slice assignment = {a}")`：輸出切片賦值後陣列。

---

[來源: tools_numpy] [1:4] = [-87, 88, 89]`：將索引 1 到 3 的元素替換為新值。
12. `print(f"a after slice assignment = {a}")`：輸出切片賦值後陣列。

**🎯 重點摘要：**

---

[來源: tools_numpy] 88, 89]`：將索引 1 到 3 的元素替換為新值。
12. `print(f"a after slice assignment = {a}")`：輸出切片賦值後陣列。

**🎯 重點摘要：**

- **核心功能**：NumPy 一維陣列支援完整的 Python 索引語法，包括正索引、負索引、切片（含起始、結束和步長）、以及就地修改和切片賦值。這些操作都是 O(1) 或 O(k) 時間複雜度，非常有效率。
- **索引特性**：索引從 0 開始，支援負數從末尾計數；切片返回視圖（view）而非複製，提升記憶體效率；步長允許靈活的資料抽樣。
- **修改機制**：單元素修改和切片賦值都支援，切片賦值時若新值數量不匹配會觸發廣播，但在此例中數量相等。
- **效能優勢**：相較於 Python 列表，NumPy 陣列的索引和切片操作更快，尤其在大資料集上；視圖機制避免不必要的記憶體複製。

---

[來源: tools_numpy] ，切片賦值時若新值數量不匹配會觸發廣播，但在此例中數量相等。
- **效能優勢**：相較於 Python 列表，NumPy 陣列的索引和切片操作更快，尤其在大資料集上；視圖機制避免不必要的記憶體複製。
- **常見應用**：資料預處理中的子集選取、特徵工程的資料轉換、演算法實作中的陣列操作等。
- **潛在問題**：切片返回視圖，若修改視圖會影響原陣列；大型陣列的切片可能消耗額外記憶體；索引超出範圍會引發 IndexError。
- **最佳使用情境**：需要高效能資料存取和修改的科學計算、資料分析和機器學習任務；適合處理結構化資料如時間序列、訊號或影像的一維表示。

---

[來源: tools_numpy] ### 多維陣列

💡 **實際應用情境：**  
在處理影像、時間序列或機器學習資料集時，多維陣列（Multi-dimensional Arrays）是 NumPy 的核心結構，能有效儲存和操作高維資料。掌握多維索引技巧，能讓你快速存取特定子集、進行資料切片或重塑，對於資料預處理和特徵工程至關重要。本節將深入探討二維及更高維度的陣列索引方法，包含基本索引、切片和進階技巧，幫助你提升程式碼效率並避免常見錯誤。無論是影像處理中的像素操作，或是神經網路中的張量運算，多維索引都是不可或缺的技能。

---

[來源: tools_numpy] 關重要。本節將深入探討二維及更高維度的陣列索引方法，包含基本索引、切片和進階技巧，幫助你提升程式碼效率並避免常見錯誤。無論是影像處理中的像素操作，或是神經網路中的張量運算，多維索引都是不可或缺的技能。

```python
b = np.arange(15).reshape(3, 5)
print("Original Shape of this example =", b.shape)
print("b[1, 3] =", b[1, 3])      # 取第 2 列第 4 行元素（定義：列=axis 0，行=axis 1）
print("b[1, :] =", b[1, :])      # 取第 2 列所有行
print("b[:, -1] =", b[:, -1])    # 取所有列的最後一行
```

**✅ 程式碼逐行解析（台灣定義：列=axis 0，行=axis 1）：**

---

[來源: tools_numpy] # 取第 2 列所有行
print("b[:, -1] =", b[:, -1])    # 取所有列的最後一行
```

**✅ 程式碼逐行解析（台灣定義：列=axis 0，行=axis 1）：**

1. `b = np.arange(15).reshape(3, 5)`：建立 3 列 5 行的二維陣列（shape = (3, 5)）。
2. `print("b.shape =", b.shape)`：輸出陣列形狀，顯示為 (3, 5)。
3. `print("b[1, 3] =", b[1, 3])`：存取第 2 列第 4 行的元素（索引從 0 開始）。
4. `print("b[1, :] =", b[1, :])`：取第 2 列的所有行（即第 2 橫列）。
5. `print("b[:, -1] =", b[:, -1])`：取所有列的最後一行（即每一列的最右側元素）。

---

[來源: tools_numpy] b[1, :] =", b[1, :])`：取第 2 列的所有行（即第 2 橫列）。
5. `print("b[:, -1] =", b[:, -1])`：取所有列的最後一行（即每一列的最右側元素）。

**🎯 重點摘要：**

- **核心功能**：多維索引使用逗號分隔的索引或切片。
- **潛在問題**：切片返回不同維度的視圖。
- **最佳使用情境**：矩陣和張量操作，如影像處理。

---

[來源: tools_numpy] ### 花式索引 (Fancy Indexing)

在 NumPy 的世界中，花式索引 (Fancy Indexing) 是解鎖進階資料操作的關鍵。它允許我們使用整數陣列作為索引，輕鬆選取陣列中不連續的元素或重新排列資料順序。這不僅超越了傳統切片的限制，還能大幅提升程式碼的效率和可讀性。在資料科學、機器學習和影像處理等領域，花式索引是處理複雜資料篩選、特徵工程和隨機取樣的必備工具。掌握花式索引，你就能像魔法師一樣精準操控多維陣列，實現更靈活的資料分析與轉換。

💡**實際應用情境：** 一般的切片（如 3:5）只能選取連續的區域。當我們需要選取不連續的特定列或行（例如：選取第 1、3、5 個樣本，或打亂資料順序）時，就需要使用整數陣列來進行索引，這被稱為「花式索引」。

```python

---

[來源: tools_numpy] # Axis 1 (行): 0~5
b = np.arange(24).reshape(4, 6)
print("b (Shape: 4, 6) =\n", b)
print("-" * 30)

---

[來源: tools_numpy] # 1. 混合索引：花式索引 (Axis 0) + 切片 (Axis 1)

---

[來源: tools_numpy] # 選取 Axis 0 的索引 0 和 2 (第1, 3列)

---

[來源: tools_numpy] # 結果形狀預期: (2, 2)
print("b[[0,2], 3:5] =\n", b[[0, 2], 3:5])

---

[來源: tools_numpy] # 選取所有列，並依序選取 Axis 1 的索引 -1(最後), 2, -3(倒數第3)

---

[來源: tools_numpy] # 結果形狀預期: (4, 3)
print("b[:, [-1, 2, -3]] =\n", b[:, [-1, 2, -3]])

---

[來源: tools_numpy] # 3. 雙軸花式索引：點對點選取 (Point-wise Selection)

---

[來源: tools_numpy] # 它不是選取「列(-1, 2) 與 行(3, 4) 的交叉區域」，而是選取座標點：

---

[來源: tools_numpy] # 結果形狀預期: (2,) -> 一維陣列
print("b[[-1, 2], [3, 4]] =", b[[-1, 2], [3, 4]])
```

**✅ 程式碼逐行解析：**

1. `b[[0, 2], 3:5]`：

- 選取 Axis 0（列）索引 0 和 2，並對 Axis 1（行）切片 3:5。
- 結果是 shape (2, 2) 的子矩陣，保留原有的矩陣結構。
- 適合同時選取多列的連續行區段。

2. `b[:, [-1, 2, -3]]`：

- 選取所有列（:），但行索引順序重組為最後一行（-1）、索引 2、倒數第三行（-3）。
- 結果 shape 為 (4, 3)，可用於重排或抽取特定行。

3. `b[[-1, 2], [3, 4]]`：

---

[來源: tools_numpy] 取所有列（:），但行索引順序重組為最後一行（-1）、索引 2、倒數第三行（-3）。
- 結果 shape 為 (4, 3)，可用於重排或抽取特定行。

3. `b[[-1, 2], [3, 4]]`：

- 在兩個軸都提供整數列表時，NumPy 會將它們配對為座標點 (axis0_indices, axis1_indices)。
- 只選取這些座標上的元素，結果是一維陣列（此例 shape 為 (2,)）。
- 適合點對點選取，不會返回子矩陣。
- 例如：選取座標 (-1,3) 和 (2,4) 的值（即最後一列的第4行元素和第3列的第5行元素）。

**🎯 重點摘要：**

**核心差異與最佳實踐：**

---

[來源: tools_numpy] - 適合點對點選取，不會返回子矩陣。
- 例如：選取座標 (-1,3) 和 (2,4) 的值（即最後一列的第4行元素和第3列的第5行元素）。

**🎯 重點摘要：**

**核心差異與最佳實踐：**

- **切片 (Slicing)**：使用 `start:end` 語法，選取連續區域，返回的是原陣列的「視圖 (View)」，修改視圖會影響原陣列。
- **花式索引 (Fancy Indexing)**：使用整數列表 `[list]`，選取不連續元素，返回的是「副本 (Copy)」，修改不會影響原陣列。
- **座標選取陷阱**：同時在兩個軸使用列表索引（如 `b[[r1, r2], [c1, c2]]`）時，行為是「點對點 (Point-wise)」選取，只取 `(r1, c1)` 和 `(r2, c2)`，而非交叉區域。
- **最佳使用情境**：花式索引適合資料打亂、重新排序、或根據複雜條件選取特定樣本。

---

[來源: tools_numpy] ### 布林索引 (Boolean Indexing)

💡 **實際應用情境：** 布林索引（Boolean Indexing）是 NumPy 中最強大的資料篩選工具之一，能根據條件動態選擇陣列元素。在資料科學和機器學習中，這種索引方式無處不在，例如：過濾出所有年齡大於 30 歲的用戶、選取收入超過某閾值的樣本，或移除含有缺失值的資料行。掌握布林索引技巧，能讓你高效處理大型資料集，避免使用低效的迴圈，提升程式效能。

布林索引的核心是使用布林陣列（True/False）作為遮罩，快速篩選符合條件的元素。這不僅支援簡單的數值比較，還能結合多個條件進行複雜的邏輯運算，適用於資料清理、特徵選擇和條件式分析等場景。無論是一維向量還是多維張量，布林索引都能靈活應用，是 NumPy 向量化操作的典範。

---

[來源: tools_numpy] 件的元素。這不僅支援簡單的數值比較，還能結合多個條件進行複雜的邏輯運算，適用於資料清理、特徵選擇和條件式分析等場景。無論是一維向量還是多維張量，布林索引都能靈活應用，是 NumPy 向量化操作的典範。

```python
b = np.arange(24).reshape(4, 6)
print("b (Shape: 4, 6) =\n", b)
print("-" * 30)

---

[來源: tools_numpy] # 陣列有 4 列，遮罩長度必須為 4
mask_axis0 = np.array([True, False, True, False])

---

[來源: tools_numpy] # 選取第 0 和第 2 列 (Axis 0)
print("b[mask_axis0, :] =\n", b[mask_axis0, :])

---

[來源: tools_numpy] # 產生 [False, True, False, True, False, True]
mask_axis1 = np.array([False, True] * 3)

---

[來源: tools_numpy] # 選取第 1, 3, 5 行 (Axis 1)
print("b[:, mask_axis1] =\n", b[:, mask_axis1])

---

[來源: tools_numpy] # 注意：這種過濾方式會破壞形狀，返回一維陣列
print("b[b % 2 == 1] =", b[b % 2 == 1])
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] 1.  `mask_axis0 = ...`：建立對應 **Axis 0** 長度的布林遮罩。因為 `b` 有 4 列，所以遮罩長度必須是 **4**。
2.  `b[mask_axis0, :]`：將遮罩應用於第一個維度。`True` 的位置保留，`False` 的位置剔除。結果保留了原本的二維結構。
3.  `mask_axis1 = np.array([False, True] * 3)`：建立對應 **Axis 1** 長度的布林遮罩（長度為 6）。
4.  `b[:, mask_axis1]`：將遮罩應用於第二個維度。
5.  `b[b % 2 == 1]`：這是不指定軸的過濾。NumPy 會先計算 `b % 2 == 1` 得到一個與 `b` 形狀相同的布林矩陣，然後選取所有 `True` 的元素，並**展平為一維陣列**返回。

**🎯 重點摘要：**

---

[來源: tools_numpy] 這是不指定軸的過濾。NumPy 會先計算 `b % 2 == 1` 得到一個與 `b` 形狀相同的布林矩陣，然後選取所有 `True` 的元素，並**展平為一維陣列**返回。

**🎯 重點摘要：**

- **核心規則 (Shape Matching)**：若要對特定軸進行布林索引，**遮罩的長度必須嚴格等於該軸的長度**，否則會報錯 `IndexError`。
- **維度保留 vs 展平**：
    - `b[mask, :]` (切片式布林索引)：通常會**保留**維度結構（例如從 4x6 變成 2x6）。
    - `b[condition]` (直接條件索引)：總是返回**一維陣列**（因為符合條件的元素在記憶體中可能不連續，無法維持矩陣形狀）。
- **最佳使用情境**：資料清理（剔除無效樣本）、特徵選擇（只保留特定欄位）。

---

---

[來源: tools_numpy] ## 疊加與串接陣列 (Stacking & Concatenation)

💡 **實際應用情境：**
在機器學習中，我們常需要將多個資料集（例如訓練集 A 和訓練集 B）合併，或者將多張 2D 圖片堆疊成一個 3D 的批次 (Batch)。NumPy 提供了多種方法來處理這些需求。

```python
import numpy as np

---

[來源: tools_numpy] # m1: Shape (3, 5)
m1 = np.full((3, 5), 1.0)

---

[來源: tools_numpy] # m2: Shape (5, 5) -> 注意：Axis 0 (高度) 與 m1 不同
m2 = np.full((5, 5), 2.0)

---

[來源: tools_numpy] # m3: Shape (3, 5) -> 與 m1 形狀完全相同
m3 = np.full((3, 5), 3.0)

print("Original shapes:", m1.shape, m2.shape, m3.shape)
print("-" * 30)

---

[來源: tools_numpy] # 1. 垂直串接 (Vertical Stack) - 沿 Axis 0

---

[來源: tools_numpy] # 要求：Axis 1 (寬度) 必須一致
q4 = np.vstack((m1, m2, m3))
print(f"vstack (m1, m2, m3): {q4.shape}")

---

[來源: tools_numpy] # 2. 水平串接 (Horizontal Stack) - 沿 Axis 1

---

[來源: tools_numpy] # 注意：這裡不能放 m2，因為 m2 的 Axis 0 是 5，與其他人(3)不同
q5 = np.hstack((m1, m3))
print(f"hstack (m1, m3):     {q5.shape}")

---

[來源: tools_numpy] # 3. 通用串接 (Concatenate) - 指定 Axis

---

[來源: tools_numpy] # axis=0 等同於 vstack
q6 = np.concatenate((m1, m2, m3), axis=0)
print(f"concatenate axis=0:  {q6.shape}")

---

[來源: tools_numpy] # 要求：所有輸入陣列的形狀必須「完全一致」
q7 = np.stack((m1, m3)) 
print(f"stack (m1, m3):      {q7.shape}")

---

[來源: tools_numpy] # 結果: 產生新的 Axis 0 -> (2, 3, 5)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] 1.  `m1, m2, m3`：建立三個陣列。注意 `m2` 的高度 (Axis 0) 是 5，與其他兩個不同，這會限制它只能參與垂直串接。
2.  `np.vstack(...)`：沿 **Axis 0 (垂直方向)** 串接。它將陣列「上下」接在一起。
    * 條件：**Axis 1 (寬度)** 必須相同。
    * 結果：列數增加 (3+5+3 = 11)，行數不變。
3.  `np.hstack(...)`：沿 **Axis 1 (水平方向)** 串接。它將陣列「左右」接在一起。
    * 條件：**Axis 0 (高度)** 必須相同。
    * 結果：列數不變，行數增加 (5+5 = 10)。
4.  `np.concatenate(..., axis=0)`：這是最底層的函數，`vstack` 其實就是 `axis=0` 的特例，`hstack` 是 `axis

---

[來源: tools_numpy] 加 (5+5 = 10)。
4.  `np.concatenate(..., axis=0)`：這是最底層的函數，`vstack` 其實就是 `axis=0` 的特例，`hstack` 是 `axis=1` 的特例。
5.  `np.stack(...)`：**這是與前三者最大的不同**。它不是延伸現有維度，而是**插入一個新的 Axis 0**。
    * 條件：輸入陣列的形狀必須**完全相同**。
    * 結果：從 2D 變成 3D。原來的 `(3, 5)` 變成了 `(2, 3, 5)`，代表「2 個 (3, 5) 的矩陣」。

**🎯 重點摘要：**

---

[來源: tools_numpy] 輸入陣列的形狀必須**完全相同**。
    * 結果：從 2D 變成 3D。原來的 `(3, 5)` 變成了 `(2, 3, 5)`，代表「2 個 (3, 5) 的矩陣」。

**🎯 重點摘要：**

* **功能區分**：
    * `concatenate` / `vstack` / `hstack`：**延伸 (Extend)** 現有的維度（2D 拼完還是 2D）。
    * `stack`：**增加 (Increase)** 維度（2D 拼完變 3D）。
  * **形狀匹配規則**：
    * 串接 (Concatenate)：除了「串接軸」之外，其他所有軸的長度必須相等。
    * 堆疊 (Stack)：所有軸的長度都必須嚴格相等。
  * **最佳使用情境**：
    * 合併資料集（增加樣本數）➝ `vstack` / `concatenate`。
    * 合併特徵（增加欄位）➝ `hstack`。
    * 將圖片打包成 Batch（增加批次維度）➝ `stack`。

---

[來源: tools_numpy] 集（增加樣本數）➝ `vstack` / `concatenate`。
    * 合併特徵（增加欄位）➝ `hstack`。
    * 將圖片打包成 Batch（增加批次維度）➝ `stack`。

---

---

[來源: tools_numpy] ## 分割陣列 (Splitting Arrays)

💡 **實際應用情境：** 分割用於資料分批處理，如將大資料集分成小批次進行訓練。

```python
m = np.arange(48).reshape(6,8)
print("Original example is m =\n", m)

v1, v2, v3 = np.vsplit(m, 3)
print("v1 =\n", v1)
print("v2 =\n", v2)
print("v3 =\n", v3)

h1, h2 = np.hsplit(m, 2)
print("h1 =\n", h1)
print("h2 =\n", h2)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] ("v3 =\n", v3)

h1, h2 = np.hsplit(m, 2)
print("h1 =\n", h1)
print("h2 =\n", h2)
```

**✅ 程式碼逐行解析：**

1. `m = np.arange(48).reshape(6,8)`：建立 6x8 陣列。
2. `print("m =\n", m)`：輸出原始陣列。
3. `v1, v2, v3 = np.vsplit(m, 3)`：垂直分割成 3 等份。
4. `print("v1 =\n", v1)`：輸出第一份2x8陣列。
5. `print("v2 =\n", v2)`：輸出第二份2x8陣列。
6. `print("v3 =\n", v3)`：輸出第三份2x8陣列。
7. `h1, h2 = np.hsplit(m, 2)`：水平分割成 2 等份。
8. `print("h1 =\n", h1)`：輸出左半6x4陣列。
9. `print("h2 =\n", h2)`：輸出右半6x4陣列。

---

[來源: tools_numpy] np.hsplit(m, 2)`：水平分割成 2 等份。
8. `print("h1 =\n", h1)`：輸出左半6x4陣列。
9. `print("h2 =\n", h2)`：輸出右半6x4陣列。

**🎯 重點摘要：**

- **核心功能**：vsplit 垂直分割，hsplit 水平分割。
- **潛在問題**：分割數必須能整除對應維度。
- **最佳使用情境**：資料分批，如交叉驗證或記憶體管理。

---

---

[來源: tools_numpy] ## 轉置陣列 (Transposing Arrays)

💡 **實際應用情境：** 轉置在線性代數中至關重要，如矩陣運算和座標變換。

```python
t = np.arange(24).reshape(4,2,3)
print("t.shape =", t.shape)

t1 = t.transpose((1,2,0))
print("t1.shape =", t1.shape)

t2 = t.transpose()
print("t2.shape =", t2.shape)

t3 = t.swapaxes(0,1)
print("t3.shape =", t3.shape)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] rint("t2.shape =", t2.shape)

t3 = t.swapaxes(0,1)
print("t3.shape =", t3.shape)
```

**✅ 程式碼逐行解析：**

1. `t = np.arange(24).reshape(4,2,3)`：建立 4x2x3 三維陣列。
2. `print("t.shape =", t.shape)`：輸出原始形狀。
3. `t1 = t.transpose((1,2,0))`：重新排列軸順序。
    - 原本軸順序為 (0,1,2)，轉置後變為 (1,2,0)。
4. `print("t1.shape =", t1.shape)`：輸出轉置後形狀。
5. `t2 = t.transpose()`：預設反轉軸順序。
    - 原本 (0,1,2) 變為 (2,1,0)。
6. `print("t2.shape ="

---

[來源: tools_numpy] hape)`：輸出轉置後形狀。
5. `t2 = t.transpose()`：預設反轉軸順序。
    - 原本 (0,1,2) 變為 (2,1,0)。
6. `print("t2.shape =", t2.shape)`：輸出預設轉置形狀。
7. `t3 = t.swapaxes(0,1)`：交換指定兩軸。
    - 將軸 0 和軸 1 互換，變為 (1,0,2)。
8. `print("t3.shape =", t3.shape)`：輸出軸交換後形狀。

**🎯 重點摘要：**

- **核心功能**：transpose 重新排列軸，swapaxes 交換兩軸。
- **潛在問題**：軸索引超出範圍會錯誤。
- **最佳使用情境**：矩陣運算、資料重塑。

---

---

[來源: tools_numpy] ## 線性代數 (Linear Algebra)

💡 **實際應用情境：** 線性代數運算是機器學習和科學計算的核心，如解線性方程組和特徵分解。

---

[來源: tools_numpy] ### 矩陣運算

```python
m1 = np.arange(15).reshape(3, 5)
print("m1 =", m1)
print("Transpose of m1 (m1.T) =\n", m1.T)

n1 = np.arange(10).reshape(2, 5)
n2 = np.arange(15).reshape(5, 3)
print("n1.dot(n2) =", n1.dot(n2))
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] reshape(2, 5)
n2 = np.arange(15).reshape(5, 3)
print("n1.dot(n2) =", n1.dot(n2))
```

**✅ 程式碼逐行解析：**

1. `m1 = np.arange(15).reshape(3, 5)`：建立 3x5 矩陣。
2. `print("m1 =", m1)`：輸出矩陣。
3. `print("Transpose of m1 (m1.T) =\n", m1.T)`：輸出轉置矩陣。
4. `n1 = np.arange(10).reshape(2, 5)`：建立第一個2x5矩陣。
5. `n2 = np.arange(15).reshape(5, 3)`：建立第二個5x3矩陣。
6. `print("n1.dot(n2) =", n1.dot(n2))`：計算矩陣乘法，得到2x3矩陣。
    - 注意：矩陣乘法 `n1.dot(n2)` 需要 `n1` 的**行數** (5) 與 `n2` 的**列數** (5) 相同，本例中是正確的，因此可以進行矩陣乘法運算，結果為形狀 (2,3) 的矩陣。

---

[來源: tools_numpy] 注意：矩陣乘法 `n1.dot(n2)` 需要 `n1` 的**行數** (5) 與 `n2` 的**列數** (5) 相同，本例中是正確的，因此可以進行矩陣乘法運算，結果為形狀 (2,3) 的矩陣。

**🎯 重點摘要：**

---

[來源: tools_numpy] ot(n2)` 需要 `n1` 的**行數** (5) 與 `n2` 的**列數** (5) 相同，本例中是正確的，因此可以進行矩陣乘法運算，結果為形狀 (2,3) 的矩陣。

**🎯 重點摘要：**

- **核心功能**：
    - `T` 屬性可快速取得矩陣的轉置（即將行與列互換），常用於線性代數運算。
    - `dot` 方法或 `@` 運算子用於執行矩陣乘法（Matrix Multiplication），遵循線性代數規則：若 A 形狀為 (m, n)，B 形狀為 (n, k)，則 A.dot(B) 結果為 (m, k)。
    - 注意：`*` 運算子僅執行元素級相乘（Element-wise Multiplication），而非矩陣乘法。
- **潛在問題**：
    - 若參與矩陣乘法的陣列維度不相容（如 A 的列數不等於 B 的行數），會引發 ValueError

---

[來源: tools_numpy] ment-wise Multiplication），而非矩陣乘法。
- **潛在問題**：
    - 若參與矩陣乘法的陣列維度不相容（如 A 的列數不等於 B 的行數），會引發 ValueError。
    - 初學者常誤用 `*` 進行矩陣乘法，導致結果錯誤。
- **最佳使用情境**：
    - 線性變換（如旋轉、縮放）、特徵工程（如主成分分析）、神經網路層的權重計算、解線性方程組等科學計算場景。

---

[來源: tools_numpy] ### 求反矩陣和分解 (Matrix Inversion & Decomposition)

💡 **實際應用情境：**  
在機器學習、統計分析和工程計算中，計算反矩陣和分解是解線性方程組、特徵分解、主成份分析（PCA）等核心步驟。例如，解 $Ax = b$ 時需計算 $A$ 的逆矩陣 ($A^{-1}$)，特徵分解則用於資料降維和模式識別。NumPy 的 `linalg` 模組提供的矩陣運算工具，能處理大規模數據並支援多種分解方法。

---

[來源: tools_numpy] 解 $Ax = b$ 時需計算 $A$ 的逆矩陣 ($A^{-1}$)，特徵分解則用於資料降維和模式識別。NumPy 的 `linalg` 模組提供的矩陣運算工具，能處理大規模數據並支援多種分解方法。

- **逆矩陣 (`inv`)**：計算方陣的反矩陣，僅適用於非奇異（可逆; non-singular、invertible）矩陣。
- **行列式 (`det`)**：判斷矩陣是否可逆，行列式為 0 表示矩陣奇異。
- **特徵分解 (`eig`)**：取得矩陣的特徵值 (eigenvalues) 與特徵向量 (eigenvectors)，常用於資料降維與系統分析。

這些運算在數值分析、物理模擬、金融建模等領域都非常重要，能幫助你深入理解資料結構與系統行為。

---

[來源: tools_numpy] 特徵值 (eigenvalues) 與特徵向量 (eigenvectors)，常用於資料降維與系統分析。

這些運算在數值分析、物理模擬、金融建模等領域都非常重要，能幫助你深入理解資料結構與系統行為。

```python
import numpy.linalg as linalg

m3 = np.array([[1,2,3],[0,1,4],[5,6,0]])
print("m3 =", m3)
det_m3 = linalg.det(m3) # Calculate determinant
if det_m3 == 0:
    print("Matrix m3 is singular and cannot be inverted.")
else:
    inv_m3 = linalg.inv(m3)
    print("The determinant is", det_m3)
    print("Inverse is", inv_m3)
    print("Verification (m3.dot(inv_m3)) =\n", m3.dot(inv_m3))

eigenvalues, eigenvectors = linalg.eig(m3)
print("eigenvalues =", eigenvalues)
print("eigenvectors =", eigenvectors)
```

---

[來源: tools_numpy] ctors = linalg.eig(m3)
print("eigenvalues =", eigenvalues)
print("eigenvectors =", eigenvectors)
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] ig(m3)
print("eigenvalues =", eigenvalues)
print("eigenvectors =", eigenvectors)
```

**✅ 程式碼逐行解析：**

1. `import numpy.linalg as linalg`：匯入 NumPy 的線性代數模組，提供矩陣運算函數。
2. `m3 = np.array([[1,2,3],[0,1,4],[5,6,0]])`：建立一個 3x3 矩陣（注意：此矩陣非奇異，可求逆）。
3. `print("m3 =", m3)`：輸出原始矩陣以供檢查。
4. `det_m3 = linalg.det(m3)`：計算矩陣的行列式，用於判斷是否可逆。
5. `if det_m3 == 0:`：條件檢查：若行列式為零，矩陣奇異，無法求逆。
6. `print("Matrix m3 is singular an

---

[來源: tools_numpy] `：計算矩陣的行列式，用於判斷是否可逆。
5. `if det_m3 == 0:`：條件檢查：若行列式為零，矩陣奇異，無法求逆。
6. `print("Matrix m3 is singular and cannot be inverted.")`：輸出警告訊息。
7. `else:`：若行列式非零，進入求逆分支。
8. `inv_m3 = linalg.inv(m3)`：計算逆矩陣。
9. `print("The determinant is", det_m3)`：輸出行列式值。
10. `print("Inverse is", inv_m3)`：輸出逆矩陣。
11. `print("Verification (m3.dot(inv_m3)) =\n", m3.dot(inv_m3))`：驗證逆矩陣正確性，結果應接近單位矩陣。
12. `eigenvalues, eigenvectors

---

[來源: tools_numpy] ification (m3.dot(inv_m3)) =\n", m3.dot(inv_m3))`：驗證逆矩陣正確性，結果應接近單位矩陣。
12. `eigenvalues, eigenvectors = linalg.eig(m3)`：計算矩陣的特徵值和特徵向量。
13. `print("eigenvalues =", eigenvalues)`：輸出特徵值。
14. `print("eigenvectors =", eigenvectors)`：輸出特徵向量。

**🎯 重點摘要：**

- **核心功能**：矩陣求逆和分解是線性代數的核心，`inv` 計算逆矩陣，`det` 檢查可逆性，`eig` 分解為特徵值和向量。
- **潛在問題**：奇異矩陣（det=0）無逆矩陣；浮點數精度可能導致近似奇異矩陣誤判。
- **最佳使用情境**：解線性方程組、主成分分析、變換矩陣等。

---

[來源: tools_numpy] 查可逆性，`eig` 分解為特徵值和向量。
- **潛在問題**：奇異矩陣（det=0）無逆矩陣；浮點數精度可能導致近似奇異矩陣誤判。
- **最佳使用情境**：解線性方程組、主成分分析、變換矩陣等。

- **核心功能**：inv 求逆，det 行列式，eig 特徵分解。
- **潛在問題**：奇異矩陣無反矩陣。
- **最佳使用情境**：解方程組、主成份分析。

---

[來源: tools_numpy] ### 奇異值分解 (Singular Value Decomposition, SVD)

奇異值分解（SVD）是線性代數中最重要的矩陣分解技術之一。它能將任意形狀的矩陣 $A$ 分解為三個部分：$A = U \Sigma V^T$，其中 $U$ 和 $V$ 是正交矩陣，$\Sigma$ 是只含奇異值的對角矩陣。SVD 廣泛應用於資料降維（如主成份分析 PCA）、壓縮、去噪、推薦系統等領域。NumPy 的 `linalg.svd` 函數可計算 SVD，適合處理大型資料集和稀疏矩陣。

```python

---

[來源: tools_numpy] # 建立 4x5 的非方陣
m4 = np.array([[1, 0, 0, 0, 2], 
               [0, 3, 0, 0, 0], 
               [0, 0, 0, 2, 0], 
               [0, 0, 0, 0, 0]])

---

[來源: tools_numpy] # U: (4, 4), S: (4,), Vt: (5, 5)

---

[來源: tools_numpy] # 注意：NumPy 回傳的第三個值已經是 V 的轉置 (Vt)
U, S, Vt = np.linalg.svd(m4)

print("U Matrix =\n", U)
print("S (Singular Values) =", S)
print("Vt Matrix =\n", Vt)
print("-" * 30)

---

[來源: tools_numpy] # 原始矩陣是 (4, 5)，所以中間的 Sigma 矩陣必須建立為 (4, 5) 才能與 Vt 相乘
Sigma = np.zeros((4, 5))

---

[來源: tools_numpy] # 將 S 填入對角線 (S 的長度為 4，填入前 4x4 的區域)
Sigma[:4, :4] = np.diag(S)

---

[來源: tools_numpy] # 注意：不需要對 Vt 再次轉置
reconstructed = U @ Sigma @ Vt

print("SVD 重建矩陣 =\n", reconstructed)
print("重建成功？", np.allclose(m4, reconstructed))
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] nt("SVD 重建矩陣 =\n", reconstructed)
print("重建成功？", np.allclose(m4, reconstructed))
```

**✅ 程式碼逐行解析：**

1. `m4 = ...`：建立原始矩陣。觀察可知第 2 行（索引 1）元素 3 最大，這通常會反映在最大的奇異值上。
2. `U, S, Vt = np.linalg.svd(m4)`：
    - `S`：奇異值陣列。計算結果約為 `[3., 2.236, 2., 0.]`，分別對應原矩陣各行的特徵（如 3 來自第二列，2.236 來自第一列 $\sqrt{1^2+2^2}$，2 來自第三列）。
    - `Vt`：已是 $V^T$，不需再轉置。
3. `Sigma = np.zeros((4, 5))`：重建步驟關鍵。直接用 `np.diag(S)` 只會得到 (4, 4) 方陣，

---

[來源: tools_numpy] 列）。
    - `Vt`：已是 $V^T$，不需再轉置。
3. `Sigma = np.zeros((4, 5))`：重建步驟關鍵。直接用 `np.diag(S)` 只會得到 (4, 4) 方陣，無法與 (5, 5) 的 Vt 相乘。需手動建立形狀為 (4, 5) 的 Sigma，並將 S 填入前 4x4 的對角線。
4. `reconstructed = U @ Sigma @ Vt`：矩陣乘法順序為 (4, 4) @ (4, 5) @ (5, 5)，最終得到 (4, 5) 的重建矩陣，與原始 m4 形狀一致。

**🎯 重點摘要：**

---

[來源: tools_numpy] ucted = U @ Sigma @ Vt`：矩陣乘法順序為 (4, 4) @ (4, 5) @ (5, 5)，最終得到 (4, 5) 的重建矩陣，與原始 m4 形狀一致。

**🎯 重點摘要：**

- **核心功能**：SVD 分解矩陣為 $U \Sigma V^T$。
- **維度對齊**：當處理非方陣時，必須小心構建 $\Sigma$ 矩陣，通常需要補零行或補零列來匹配維度。
- **物理意義**：觀察輸出的 S，最後一個值為 0（或極小的浮點數），這精確地反映了 m4 的最後一行是全零行（線性相依/無效資訊）。

---

---

[來源: tools_numpy] ## 向量化 (Vectorization)

💡 **實際應用情境：**  
向量化是 NumPy 提升效能的核心技巧，能將原本需要巢狀 Python 迴圈的運算（如逐元素計算、資料轉換、數學函數套用）轉換為一次性批次運算。這種方式充分利用底層 C 語言優化，顯著加快資料處理速度並減少記憶體存取延遲。

在資料科學、影像處理、機器學習等領域，向量化能讓你用一行程式碼完成複雜的數值運算，例如：

- 對整張影像進行濾鏡處理（如 `img = np.clip(img * 1.2 + 10, 0, 255)`），無需逐像素迴圈  
- 批次計算所有樣本的特徵轉換（如 `X_norm = (X - X.mean(axis=0)) / X.std(axis=0)`）  
- 建立座標網格並進行函數運算（如 `Z = np.sin(X * Y / 40.5)`）

---

[來源: tools_numpy] 換（如 `X_norm = (X - X.mean(axis=0)) / X.std(axis=0)`）  
- 建立座標網格並進行函數運算（如 `Z = np.sin(X * Y / 40.5)`）

這不僅提升程式碼可讀性，也讓大規模資料分析和模型訓練變得可行。實務上，建議盡量將所有資料處理步驟向量化，僅在必要時才使用 Python 迴圈。

```python
import math
import numpy as np

---

[來源: tools_numpy] # 低效率方式
data_loop = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data_loop[y, x] = math.tan(x + y)

---

[來源: tools_numpy] # 高效率方式
x_coords = np.arange(0, 1024)
y_coords = np.arange(0, 768)
X, Y = np.meshgrid(x_coords, y_coords)
data_vec = np.tan(X + Y)

print("Shapes match:", data_loop.shape == data_vec.shape)
print("Results match:", np.allclose(data_loop, data_vec))

---

[來源: tools_numpy] atch:", data_loop.shape == data_vec.shape)
print("Results match:", np.allclose(data_loop, data_vec))

```

**✅ 程式碼逐行解析：**

1. `import math`：匯入數學模組。
2. `data_loop = np.empty((768, 1024))`：建立空陣列。
3. 迴圈賦值：使用巢狀迴圈計算每個元素（效率低）。
4. `x_coords = np.arange(0, 1024)`：建立 x 座標陣列。
5. `y_coords = np.arange(0, 768)`：建立 y 座標陣列。
6. `X, Y = np.meshgrid(x_coords, y_coords)`：建立座標網格。
7. `data_vet = np.tan(X + Y)`：向量化計算。
8. 比較結果：確認兩種方法結果相同。

**🎯 重點摘要：**

- **核心功能**：meshgrid 建立座標網格，向量化運算避免迴圈。
- **潛在問題**：記憶體使用量大於迴圈方式。
- **最佳使用情境**：數值計算、影像處理。

**補充說明：**
使用 Matplotlib 視覺化向量化結果：

```

---

[來源: tools_numpy] ，向量化運算避免迴圈。
- **潛在問題**：記憶體使用量大於迴圈方式。
- **最佳使用情境**：數值計算、影像處理。

**補充說明：**
使用 Matplotlib 視覺化向量化結果：

```

python
import matplotlib.pyplot as plt

---

[來源: tools_numpy] # 視覺化結果
fig = plt.figure(1, figsize=(8, 6))
plt.imshow(data_vec, cmap="autumn")
plt.show()

---

[來源: tools_numpy] ## 儲存和載入 (Saving and Loading)

💡 **實際應用情境：**  
在機器學習和資料科學工作流程中，資料持久化 (persistence) 是不可或缺的步驟。常見用途包括：  

- 儲存訓練好的模型權重，以便日後載入並進行預測或微調。
- 保存中間計算結果（如特徵工程後的陣列、驗證集預測值），避免重複運算，提升效率。
- 交換資料給團隊成員或跨平台工具（如將 NumPy 陣列匯出為 CSV 供 Excel 或 R 使用）。
- 長期保存原始資料集或處理後的資料，方便版本管理與重現實驗。

選擇合適的儲存格式（如二進位、文字或壓縮）能兼顧效能、精確度與可攜性，是專業資料處理流程的重要一環。

---

[來源: tools_numpy] ### 二進位格式 (Binary Format)

NumPy 的二進位格式（.npy）能完整保存陣列的形狀、資料類型和內容，適合高效率地儲存與載入大型科學資料。與文字格式相比，二進位格式讀寫速度更快且不會有精度損失，特別適合模型權重、特徵矩陣等需精確重現的場景。若需儲存多個陣列，可使用壓縮格式（.npz），將多個命名陣列打包於同一檔案，方便資料管理與交換。

---

[來源: tools_numpy] 。與文字格式相比，二進位格式讀寫速度更快且不會有精度損失，特別適合模型權重、特徵矩陣等需精確重現的場景。若需儲存多個陣列，可使用壓縮格式（.npz），將多個命名陣列打包於同一檔案，方便資料管理與交換。

```python
a = np.random.rand(3,3)*2-1
print("a =", a)
np.save("saved_array", a)

loaded_array = np.load("saved_array.npy")
print("loaded_array =", loaded_array)
print("Arrays equal:", np.array_equal(a, loaded_array))
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] _array =", loaded_array)
print("Arrays equal:", np.array_equal(a, loaded_array))
```

**✅ 程式碼逐行解析：**

1. `a = np.random.rand(3,3)*2-1`：建立隨機陣列，範圍在 -1 到 1 之間。
2. `print("a =", a)`：輸出原始陣列。
3. `np.save("saved_array", a)`：將陣列 `a` 儲存為二進位格式的 `.npy` 檔案。
    - 此方法會自動在檔名後加上 `.npy` 副檔名，能完整保存陣列的形狀、資料類型和內容
    - 注意：若檔案已存在，將直接覆蓋且不會警告。
4. `loaded_array = np.load("saved_array.npy")`：載入檔案。
5. `print("loaded_array =", loaded_array)`：輸出載入陣列。
6. `print("Arrays equal:", np.array_equal(a, loaded_array))`：檢查是否相同。

---

[來源: tools_numpy] rray =", loaded_array)`：輸出載入陣列。
6. `print("Arrays equal:", np.array_equal(a, loaded_array))`：檢查是否相同。

**🎯 重點摘要：**

- **核心功能**：save/load 以二進位格式儲存單一陣列。
- **潛在問題**：檔案覆蓋無警告。
- **最佳使用情境**：單一陣列持久化。

---

[來源: tools_numpy] ### 文字格式 (Text Format)

NumPy 支援將陣列儲存為純文字格式（如 CSV、TXT），方便與 Excel、R、Pandas 等其他工具交換資料。這種格式可直接用文字編輯器檢視，但儲存速度較慢且可能有精度損失（尤其是浮點數）。適合小型資料集或跨平台資料交換。

- `np.savetxt`：將陣列儲存為文字檔（可指定分隔符、格式）。
- `np.loadtxt`：從文字檔載入陣列（可指定分隔符、資料類型）。

常見應用：

- 匯出分析結果給非 Python 用戶
- 與資料庫、試算表或其他程式語言互通
- 檢查資料內容或進行手動編輯

注意事項：

---

[來源: tools_numpy] dtxt`：從文字檔載入陣列（可指定分隔符、資料類型）。

常見應用：

- 匯出分析結果給非 Python 用戶
- 與資料庫、試算表或其他程式語言互通
- 檢查資料內容或進行手動編輯

注意事項：

- 儲存時可用 `fmt` 參數控制數值格式（如 `fmt="%.6f"` 保留 6 位小數）。
- 載入時若資料有標題列，可用 `skiprows` 跳過。
- 文字格式不會保存陣列形狀資訊，僅儲存元素本身，載入時需自行重塑形狀（如 `.reshape()`）。
- 若資料包含非數值型態（如字串），需額外指定 `dtype=str`。

範例：

---

[來源: tools_numpy] ` 跳過。
- 文字格式不會保存陣列形狀資訊，僅儲存元素本身，載入時需自行重塑形狀（如 `.reshape()`）。
- 若資料包含非數值型態（如字串），需額外指定 `dtype=str`。

範例：

- 儲存為 CSV：`np.savetxt("data.csv", arr, delimiter=",")`
- 載入 CSV：`arr = np.loadtxt("data.csv", delimiter=",")`
- 儲存為 TXT：`np.savetxt("data.txt", arr, fmt="%.6f")`
- 載入 TXT：`arr = np.loadtxt("data.txt")`
- 若需儲存多維陣列，建議先展平或重塑為 2D，再儲存。

---

[來源: tools_numpy] t("data.txt", arr, fmt="%.6f")`
- 載入 TXT：`arr = np.loadtxt("data.txt")`
- 若需儲存多維陣列，建議先展平或重塑為 2D，再儲存。

```python
np.savetxt("saved_array.csv", a, delimiter=",")
with open("saved_array.csv", "rt") as f:
    print("CSV content:")
    print(f.read())

loaded_array_txt = np.loadtxt("saved_array.csv", delimiter=",")
print("Loaded from CSV:", loaded_array_txt)
print("CSV arrays equal:", np.allclose(a, loaded_array_txt))
```

---

[來源: tools_numpy] oaded from CSV:", loaded_array_txt)
print("CSV arrays equal:", np.allclose(a, loaded_array_txt))
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] , loaded_array_txt)
print("CSV arrays equal:", np.allclose(a, loaded_array_txt))
```

**✅ 程式碼逐行解析：**

1. `np.savetxt("saved_array.csv", a, delimiter=",")`：以逗號分隔符儲存陣列為 CSV 格式。
2. `with open("saved_array.csv", "rt") as f:`：以文字模式開啟 CSV 檔案。
3. `print("CSV content:")`：印出標題。
4. `print(f.read())`：讀取並印出檔案內容。
5. `loaded_array_txt = np.loadtxt("saved_array.csv", delimiter=",")`：從 CSV 載入陣列，使用逗號分隔符。
6. `print

---

[來源: tools_numpy] 。
5. `loaded_array_txt = np.loadtxt("saved_array.csv", delimiter=",")`：從 CSV 載入陣列，使用逗號分隔符。
6. `print("Loaded from CSV:", loaded_array_txt)`：輸出載入的陣列。
7. `print("CSV arrays equal:", np.allclose(a, loaded_array_txt))`：檢查載入陣列是否與原始陣列相等。

**🎯 重點摘要：**

- **核心功能**：savetxt/loadtxt 以文字格式儲存，可讀但較慢。
- **潛在問題**：精確度可能損失。
- **最佳使用情境**：與其他工具交換資料。

---

[來源: tools_numpy] ### 壓縮格式 (Compressed Format)

NumPy 的壓縮格式（`.npz`）允許一次儲存多個陣列於單一檔案，並以壓縮方式減少磁碟空間使用。這對於保存模型參數、特徵集合或多組資料特別有用。每個陣列都可指定名稱（key），載入時可依名稱存取，方便管理和交換資料。

**主要特點：**

- 支援多個陣列同時儲存，並以鍵值（key）管理。
- 檔案自動壓縮，適合大型資料集。
- 載入後為類字典物件（dict-like），可用 `keys()` 查詢所有陣列名稱。
- 適合保存模型多層權重、資料集分批結果等。

---

[來源: tools_numpy] 存，並以鍵值（key）管理。
- 檔案自動壓縮，適合大型資料集。
- 載入後為類字典物件（dict-like），可用 `keys()` 查詢所有陣列名稱。
- 適合保存模型多層權重、資料集分批結果等。

```python
b = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
np.savez("saved_arrays", array_a=a, array_b=b)

saved_arrays = np.load("saved_arrays.npz")
print("Keys:", list(saved_arrays.keys()))
print("array_a =", saved_arrays["array_a"])
```

**✅ 程式碼逐行解析：**

---

[來源: tools_numpy] ("Keys:", list(saved_arrays.keys()))
print("array_a =", saved_arrays["array_a"])
```

**✅ 程式碼逐行解析：**

1. `b = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)`：建立另一個陣列。
2. `np.savez("saved_arrays", array_a=a, array_b=b)`：儲存多個陣列到壓縮檔案。
3. `saved_arrays = np.load("saved_arrays.npz")`：載入壓縮檔案。
4. `print("Keys:", list(saved_arrays.keys()))`：輸出鍵名。
5. `print("array_a =", saved_arrays["array_a"])`：存取特定陣列。

---

[來源: tools_numpy] ("Keys:", list(saved_arrays.keys()))`：輸出鍵名。
5. `print("array_a =", saved_arrays["array_a"])`：存取特定陣列。

**🎯 重點摘要：**

- **核心功能**：savez 儲存多個陣列到單一壓縮檔案。
- **潛在問題**：鍵名管理複雜。
- **最佳使用情境**：
    - 儲存多個模型權重或中間結果。
    - 打包多組特徵資料，方便跨平台交換。
    - 長期保存多個相關陣列，便於版本管理。
- **注意事項**：
    - 儲存時建議使用具意義的鍵名（如 `train_X`, `train_y`）。
    - 載入後需以鍵名存取陣列（如 `arrays["train_X"]`）。
    - 若未指定鍵名，則以 `arr_0`, `arr_1`...自動命名。

---

---

[來源: tools_numpy] ## 常見問答 (FAQ)

**Q: NumPy 陣列和 Python 列表有什麼區別？**  
A: NumPy 陣列更有效率，支援向量化運算和廣播，所有元素必須同類型。

**Q: 如何選擇適當的資料類型？**  
A: 根據資料範圍選擇，如影像用 uint8，科學計算用 float64。

**Q: 廣播規則是什麼？**  
A: 較小陣列自動擴展以匹配較大陣列形狀，從後往前比較維度。

**Q: 為什麼要避免 Python 迴圈？**  
A: 迴圈在 Python 中效率低，NumPy 向量化運算利用 C 語言優化。

**Q: 如何處理記憶體不足？**  
A: 使用適當資料類型，考慮記憶體映射或分批處理。

---

---

[來源: tools_numpy] ## 最佳實踐

- **效能優化**：優先使用向量化運算，避免 Python 迴圈
- **記憶體管理**：選擇最小適當資料類型，使用 in-place 操作
- **程式碼清晰**：使用有意義的變數名，添加註釋
- **錯誤處理**：檢查陣列形狀相容性，處理數值異常
- **測試驗證**：使用 np.allclose 比較浮點數結果
- **文件記錄**：記錄陣列形狀和資料類型資訊

---

---

[來源: tools_numpy] ## 推薦標籤 (Suggested Hashtags)

#Python #程式設計 #教學 #NumPy #陣列 #線性代數 #科學計算 #編程 #學習筆記 #程式開發者 #軟體工程

---

[來源: tutorial_instruction] [標題: [SEO-optimized title for Threads platform] | 描述: [Comprehensive description optimized for engagement] | 關鍵字: [Relevant keywords including Python, 程式設計, 教學]]
# Python Tutorial Content Creation Instructions

This document outlines the process for creating comprehensive Python tutorial content, from initial draft to final documentation and executable code examples, with specialized guidelines for Threads social media platform optimization.

---

[來源: tutorial_instruction] ### Phase 1: Draft to Final Tutorial (.md)

**Transformation Process:**

1. **Structure Enhancement**

---

[來源: tutorial_instruction] ## Phase 1: Draft to Final Tutorial (.md)

**Transformation Process:**

1. **Structure Enhancement**

- Add SEO-optimized meta tags (``, ``, ``)
   - Create clear table of contents with anchor links. Use Markdown-style anchors for section headers (e.g., `##  Section Title`) to ensure cross-platform compatibility.
   - Add emoji icons for visual hierarchy (🐍 🎯 🔄 🚀 💡 ❓)
   - Optimize for Threads platform engagement with compelling headlines

---

[來源: tutorial_instruction] sual hierarchy (🐍 🎯 🔄 🚀 💡 ❓)
   - Optimize for Threads platform engagement with compelling headlines

2. **Content Refinement**
   - Replace casual tone with professional, educational language in Traditional Chinese (Taiwan conventions)
   - Add "Key Takeaways" section at the beginning
   - Include practical application scenarios for each concept
   - Provide detailed code explanations with "✅ 程式碼

---

[來源: tutorial_instruction] practical application scenarios for each concept
   - Provide detailed code explanations with "✅ 程式碼逐行解析" sections
   - Add "🎯 重點摘要" for each major section
   - Provide clear explanations and real-world application scenarios
   - Include comprehensive comments within code for enhanced readability
   - **Bilingual terminology (required):** include the English original in parentheses immediately aft

---

[來源: tutorial_instruction]  - **Bilingual terminology (required):** include the English original in parentheses immediately after the first Chinese mention of any specialised technical term (example: `梯度消失 (Vanishing Gradient)`).
     - Apply to: section headings, TOC entries, the first in-paragraph occurrence, figure/table captions, and FAQ items.
     - Rules: annotate only at first mention (and in headings/TOC); avoid re

---

[來源: tutorial_instruction] aptions, and FAQ items.
     - Rules: annotate only at first mention (and in headings/TOC); avoid repeating the parenthetical on every subsequent occurrence to reduce visual clutter.
     - Exception: glossary/terminology sections may include repeated bilingual labels.
     - Benefit: improves clarity for bilingual readers and boosts SEO/discoverability on social platforms.

---

[來源: tutorial_instruction] Benefit: improves clarity for bilingual readers and boosts SEO/discoverability on social platforms.

3. **Code Example Standards**
   - Use realistic, practical examples rather than abstract ones
   - Include comprehensive comments in Traditional Chinese
   - Show both correct and incorrect approaches where applicable
   - Demonstrate real-world use cases (config files, data processing, etc.)
   -

---

[來源: tutorial_instruction] es where applicable
   - Demonstrate real-world use cases (config files, data processing, etc.)
   - **Mandatory line-by-line breakdown after each code example**
   - **Include Key Points Summary highlighting:**
     - Core features of the code
     - Potential issues or considerations
     - Optimal use cases

---

[來源: tutorial_instruction] - Core features of the code
     - Potential issues or considerations
     - Optimal use cases

4. **Educational Enhancement**
   - Add FAQ section addressing common questions
   - Include best practices and development recommendations
   - Provide troubleshooting tips and common pitfalls
   - Add cross-references to related concepts
   - **Optimize content for Threads SEO and engagement**

---

[來源: tutorial_instruction] - Add cross-references to related concepts
   - **Optimize content for Threads SEO and engagement**

5. **Markdown Lint Compliance**
   - To avoid common lint warnings when generating .md files, follow these rules:
     - **MD012/no-multiple-blanks**: Ensure no more than one consecutive blank line between sections.
     - **MD022/blanks-around-headings**: Surround all headings with blank lines (on

---

[來源: tutorial_instruction] etween sections.
     - **MD022/blanks-around-headings**: Surround all headings with blank lines (one before and one after).
     - **MD032/blanks-around-lists**: Surround all lists with blank lines (one before and one after).
     - **MD033/no-inline-html**: Avoid inline HTML where possible; use Markdown equivalents (e.g., for anchors, use `##  Section Title` sparingly or replace with pure Markdo

---

[來源: tutorial_instruction] rkdown equivalents (e.g., for anchors, use `##  Section Title` sparingly or replace with pure Markdown if feasible).
     - **MD007/ul-indent**: Use 2 spaces for each level of unordered list indentation.
     - **MD018/no-missing-space-atx**: Always add a space after `#` in atx-style headings (e.g., `# Heading`).
     - **MD047/single-trailing-newline**: Ensure the file ends with a single newline 

---

[來源: tutorial_instruction]  `# Heading`).
     - **MD047/single-trailing-newline**: Ensure the file ends with a single newline character.
     - **Other**: Define all link references if using reference-style links; ensure all headers have corresponding anchors if referenced in TOC.

---

[來源: tutorial_instruction] ### Phase 2: Tutorial to Executable Code (.py)

**Code Generation Process:**

1. **Module Structure**

```python
   """
   [Module Name] Module
   
   This module demonstrates [brief description].
   
   Key features:
   - [Feature 1]
   - [Feature 2]
   - [Feature 3]
   
   Examples are based on the markdown documentation [filename].md
   """
   ```

---

[來源: tutorial_instruction] - [Feature 3]
   
   Examples are based on the markdown documentation [filename].md
   """
   ```

2. **Section Organization**
   - Use `# =============================================================================` separators
   - Group related examples under clear section headers
   - Maintain the same example numbering as the tutorial
   - Add descriptive print statements for output clarity in Traditional Chinese

---

[來源: tutorial_instruction] ring as the tutorial
   - Add descriptive print statements for output clarity in Traditional Chinese

3. **Code Documentation Standards**
   - Follow the annotation style guide for comments
   - Use type hints for all function parameters and returns
   - Include inline comments explaining complex operations in Traditional Chinese
   - Add docstrings for functions when applicable
   - **Preserve original code logic and structure exactly**

---

[來源: tutorial_instruction] docstrings for functions when applicable
   - **Preserve original code logic and structure exactly**

4. **Example Implementation**
   - Extract exact code blocks from tutorial markdown without modification
   - Ensure all examples are executable and produce meaningful output
   - Add setup code where necessary (imports, sample data creation)
   - Include error handling for robust demonstrations
   - **Add comprehensive inline comments for enhanced readability**

---

[來源: tutorial_instruction] ### Phase 3: Tutorial to HTML Document (.html)

**HTML Generation Process:**

---

[來源: tutorial_instruction] 1. **Template Structure**
   - Start with the standard HTML5 boilerplate from `tutorial_html_template.html`.
   - Populate SEO meta tags (`title`, `description`, `keywords`) from the Markdown's `` comments.
   - The template includes a comprehensive `<style>` block with light/dark mode support and Google Fonts integration.
   - **Bilingual-term rendering:** the HTML generator MUST preserve Chinese

---

[來源: tutorial_instruction] oogle Fonts integration.
   - **Bilingual-term rendering:** the HTML generator MUST preserve Chinese + English parentheticals at first mention. Concretely:
     - emit the English original in parentheses in the `<h1>`/`<h2>` text and the corresponding TOC entry;
     - preserve the first in-paragraph parenthetical and the first caption/figure/table mention;
     - include the English original in t

---

[來源: tutorial_instruction] h parenthetical and the first caption/figure/table mention;
     - include the English original in the `<title>` and `<meta description>` when it improves clarity/SEO but avoid excessive length;
     - do NOT duplicate the parenthetical on every subsequent occurrence (use glossary or tooltip if repeated clarification is needed).

---

[來源: tutorial_instruction] etical on every subsequent occurrence (use glossary or tooltip if repeated clarification is needed).

2. **MathJax Integration**
   - The template already includes the necessary scripts for MathJax library, configuration, and a polyfill for compatibility.
   - Ensure LaTeX code from Markdown is preserved as-is in the HTML.

---

[來源: tutorial_instruction] d a polyfill for compatibility.
   - Ensure LaTeX code from Markdown is preserved as-is in the HTML.

3. **Content Conversion**
   - **Header**: The main `<h1>` and the introductory paragraph (`<p class="meta">`) should be placed in the `<header>`.
   - **Table of Contents**: Generate a nested `<ul>` list from the Markdown's ToC and place it inside `<nav class="toc">`.
   - **Main Content**: Each 

---

[來源: tutorial_instruction] ` list from the Markdown's ToC and place it inside `<nav class="toc">`.
   - **Main Content**: Each major section from Markdown should be converted into a `<section class="section">` tag within `<main>`.
   - **Section Headers**: Use `<h2>`, `<h3>`, etc., for section titles. Each `<section>` should have an `id` and `aria-labelledby` attribute, and the corresponding `<h2>` should have a matching `i

---

[來源: tutorial_instruction] have an `id` and `aria-labelledby` attribute, and the corresponding `<h2>` should have a matching `id`.
   - **Summary Boxes**: Convert "🎯 重點摘要" blocks into `<div class="summary-box">`.
   - **Footer**: Place the hashtags in the `<footer class="hashtags">`.
   - Convert standard Markdown (lists, tables, bold text) to semantic HTML.

---

[來源: tutorial_instruction] ### HTML Document Template

請直接使用 `tutorial_html_template.html` 作為生成 HTML 文件的標準範本。該範本已包含所有必要的結構、CSS 樣式（含深色模式）、以及 MathJax 數學公式的設定。

---

[來源: tutorial_instruction] 直接使用 `tutorial_html_template.html` 作為生成 HTML 文件的標準範本。該範本已包含所有必要的結構、CSS 樣式（含深色模式）、以及 MathJax 數學公式的設定。

在轉換過程中，請根據 Markdown 內容動態填寫範本中的預留位置，例如：
- **`<title>`**: 來自 ``
- **`<meta name="description">`**: 來自 ``
- **`<meta name="keywords">`**: 來自 ``
- **`<header>`**: 包含 `<h1>` 和介紹性段落 `<p class="meta">`
- **`<main>`**: 包含所有從 Markdown 轉換而來的 `<section>` 內容
- **`<footer>`**: 包含 `` 的內容

---

[來源: tutorial_instruction] ### Python Code Template

```python
"""
[Module Name] Module

This module demonstrates [topic].

Key features:
- [Feature list]

Examples are based on the markdown documentation [filename].md
"""

import [required modules]
from typing import [type hints]

---

[來源: tutorial_instruction] # =============================================================================

---

[來源: tutorial_instruction] # EXAMPLE [N]: [SECTION TITLE]

---

[來源: tutorial_instruction] # =============================================================================

print("=== Example [N]: [Description] ===")

---

[來源: tutorial_instruction] # [Example code with detailed comments]
[variable] = [value]  # [Explanation of purpose]

---

[來源: tutorial_instruction] # [More complex operations with explanations]
if [condition]:  # [Why this condition matters]
    [action]  # [What this accomplishes]

print(f"[Descriptive output]: {[variable]}")

print("\n=== [Section] Examples Complete ===")
```

---

[來源: tutorial_instruction] ## Threads Platform Optimization Guidelines

---

[來源: tutorial_instruction] ### Content Strategy for Threads

1. **Engaging Headlines**
   - Use compelling, action-oriented titles
   - Include relevant emojis for visual appeal
   - Keep titles concise but informative
   - Target Taiwan developer community interests

---

[來源: tutorial_instruction] ual appeal
   - Keep titles concise but informative
   - Target Taiwan developer community interests

2. **Content Structure**
   - Start with practical problem statement
   - Use progressive disclosure of information
   - Include visual breaks with emojis and formatting
   - End with clear call-to-action or engagement hook

---

[來源: tutorial_instruction] clude visual breaks with emojis and formatting
   - End with clear call-to-action or engagement hook

3. **Language Optimization**
   - Use Traditional Chinese with Taiwan linguistic conventions
   - Include technical terms in both Chinese and English — **English originals must appear in parentheses at first mention and in section headings/TOC** to aid clarity and SEO
   - Maintain professional ye

---

[來源: tutorial_instruction]  at first mention and in section headings/TOC** to aid clarity and SEO
   - Maintain professional yet approachable tone
   - Use culturally relevant examples and scenarios
   - Threads-specific: include at most one bilingual term per short post to maximise clarity without clutter; prefer `中文 (English)` at the top or in the first sentence.

---

[來源: tutorial_instruction] post to maximise clarity without clutter; prefer `中文 (English)` at the top or in the first sentence.

4. **Hashtag Strategy**
   - Primary tags: #Python #程式設計 #教學
   - Secondary tags: #編程 #開發 #技術分享
   - Engagement tags: #學習筆記 #程式開發者 #軟體工程
   - Platform-specific: #ThreadsTech #開發者社群

---

[來源: tutorial_instruction] : #編程 #開發 #技術分享
   - Engagement tags: #學習筆記 #程式開發者 #軟體工程
   - Platform-specific: #ThreadsTech #開發者社群

5. **Engagement Optimization**
   - Include questions to encourage comments
   - Provide practical challenges or exercises
   - Reference current tech trends and applications
   - Create shareable, valuable content snippets

---

[來源: tutorial_instruction] - **SEO Optimization**: Meta tags, keywords, structured headings optimized for Threads platform
- **Accessibility**: Clear navigation, logical flow, comprehensive explanations
- **Practical Focus**: Real-world scenarios, actionable examples
- **Cross-platform Compatibility**: Consider Windows/Linux/macOS differences
- **Language Localization**: Traditional Chinese optimized for Taiwan linguistic c

---

[來源: tutorial_instruction] macOS differences
- **Language Localization**: Traditional Chinese optimized for Taiwan linguistic conventions
- **Social Media Optimization**: Engaging content designed for Threads platform virality

---

[來源: tutorial_instruction] ### Code Requirements

- **Executable**: All code must run without errors
- **Commented**: Comprehensive Traditional Chinese comments explaining logic
- **Structured**: Clear section divisions and logical progression
- **Educational**: Code demonstrates concepts progressively from basic to advanced
- **Enhanced Readability**: Include inline comments for better understanding

---

[來源: tutorial_instruction] ### Documentation Requirements

---

[來源: tutorial_instruction] - **Comprehensive**: Cover all major aspects of the topic
- **Progressive**: Build from simple concepts to complex applications
- **Practical**: Include real-world use cases and best practices
- **Interactive**: Encourage readers to experiment and modify examples
- **Detailed Analysis**: Mandatory line-by-line code breakdown
- **Key Points Summary**: Core features, potential issues, and optimal us

---

[來源: tutorial_instruction] ine-by-line code breakdown
- **Key Points Summary**: Core features, potential issues, and optimal use cases
- **Hashtag Optimization**: Include suggested hashtags for increased visibility

---

[來源: tutorial_instruction] ### Tutorial Markdown Template

```markdown

---

[來源: tutorial_instruction] # [Title]: [Subtitle]

[Introduction paragraph with practical context in Traditional Chinese]

---

[來源: tutorial_instruction] ## 關鍵重點 (Key Takeaways)
- [Key point 1 in Traditional Chinese]
- [Key point 2 in Traditional Chinese]

---

[來源: tutorial_instruction] ## [Section 1]
💡 **實際應用情境：** [Real-world scenario explanation]

---

[來源: tutorial_instruction] ### 範例 [N]: [Example title]
```python

---

[來源: tutorial_instruction] # [Enhanced comments in Traditional Chinese for readability]
[original_code_here]
```

**✅ 程式碼逐行解析：**

1. `第 X 行`: [Detailed explanation of each line's functionality]
2. `第 Y 行`: [Clear breakdown of logic and purpose]
3. `第 Z 行`: [Real-world context and application]

**🎯 重點摘要:**

---

[來源: tutorial_instruction] [Clear breakdown of logic and purpose]
3. `第 Z 行`: [Real-world context and application]

**🎯 重點摘要:**

- **核心功能**: [Core functionality explanation]
- **潛在問題**: [Potential issues and considerations]
- **最佳使用情境**: [Optimal use cases and scenarios]

---

[來源: tutorial_instruction] ## 總結與最佳實踐

[Summary and recommendations in Traditional Chinese]

---

[來源: tutorial_instruction] ## 常見問答 (FAQ)

[Common questions and answers]

---

[來源: tutorial_instruction] ## 推薦標籤 (Suggested Hashtags)

#Python #程式設計 #教學 #編程 #開發 #技術分享 #學習筆記
```

---

[來源: tutorial_instruction] ### Python Code Template

```python
"""
[Module Name] Module

This module demonstrates [topic].

Key features:
- [Feature list]

Examples are based on the markdown documentation [filename].md
"""

import [required modules]
from typing import [type hints]

---

[來源: tutorial_instruction] # =============================================================================

---

[來源: tutorial_instruction] # EXAMPLE [N]: [SECTION TITLE]

---

[來源: tutorial_instruction] # =============================================================================

print("=== Example [N]: [Description] ===")

---

[來源: tutorial_instruction] # [Example code with detailed comments]
[variable] = [value]  # [Explanation of purpose]

---

[來源: tutorial_instruction] # [More complex operations with explanations]
if [condition]:  # [Why this condition matters]
    [action]  # [What this accomplishes]

print(f"[Descriptive output]: {[variable]}")

print("\n=== [Section] Examples Complete ===")
```

---

[來源: tutorial_instruction] 1. Start with practical motivation for each concept
2. Use progressive complexity in examples
3. Include both positive and negative examples
4. Provide comprehensive explanations for each code segment in Traditional Chinese
5. End each section with actionable takeaways
6. **Include mandatory line-by-line breakdown for every code example**
7. **Add Key Points Summary highlighting core features, iss

---

[來源: tutorial_instruction] -line breakdown for every code example**
7. **Add Key Points Summary highlighting core features, issues, and use cases**
8. **Optimize content for Threads platform engagement**
9. **Include suggested hashtags for increased visibility**

---

[來源: tutorial_instruction] ### For Code Generation

1. Extract all code blocks from the tutorial without modification
2. Ensure proper imports and dependencies
3. Add setup code for demonstrations
4. Include meaningful output statements in Traditional Chinese
5. Test all examples for correctness
6. **Add comprehensive inline comments for enhanced readability**
7. **Maintain original code structure and logic**

---

[來源: tutorial_instruction] ### Quality Assurance

1. Verify all links and anchors work correctly. Ensure anchors use the compatible Markdown-style format.
2. Verify bilingual parentheticals: confirm the first occurrence of each specialised technical term includes an English original in parentheses (headings, TOC, first in-paragraph occurrence). Example automated QA (heuristic):

---

[來源: tutorial_instruction] nal in parentheses (headings, TOC, first in-paragraph occurrence). Example automated QA (heuristic):

```python
   # quick CI-style sanity check (heuristic)
   import re
   from pathlib import Path
   s = Path('11_training_deep_neural_networks.md').read_text(encoding='utf-8')
   # find Chinese phrases followed by ASCII English in parentheses
   pattern = re.compile(r"([\u4e00-\u9fff\u3000-\u303F]{2,})\s*\(([A-Za-z0-9 \-\_/\.]+)\)")
   matches = pattern.findall(s)
   if not matches:
       raise SystemExit('QA FAIL: no bilingual parentheticals found — check headings/first mentions')
   # optional: surface first 10 matches for reviewer
   print('bilingual examples (sample):', matches[:10])
   ```

---

[來源: tutorial_instruction] surface first 10 matches for reviewer
   print('bilingual examples (sample):', matches[:10])
   ```

3. Test code examples in clean Python environment
4. Ensure content flows logically from basic to advanced
5. Check that explanations match the code exactly
6. Validate that examples demonstrate real-world utility
7. **Verify Mathematical Equations**: When generating HTML, confirm that all LaTeX eq

---

[來源: tutorial_instruction] -world utility
7. **Verify Mathematical Equations**: When generating HTML, confirm that all LaTeX equations are rendered correctly by MathJax.
8. **Verify Traditional Chinese linguistic conventions for Taiwan**
9. **Test hashtag effectiveness for Threads platform**
10. **Ensure line-by-line breakdowns are comprehensive and accurate**
11. **Run Markdown lint checks**: Use tools like `markdownlint` 

---

[來源: tutorial_instruction] ns are comprehensive and accurate**
11. **Run Markdown lint checks**: Use tools like `markdownlint` to scan for issues such as MD012, MD022, MD032, MD033, MD007, MD018, MD047, and resolve them before finalizing the .md file.

---

[來源: tutorial_instruction] ## Success Metrics

**Tutorial Quality Indicators:**

---

[來源: tutorial_instruction] - Clear learning progression from basic to advanced concepts
- Practical examples that readers can immediately apply
- Comprehensive explanations that anticipate common questions
- SEO optimization for discoverability on Threads platform
- **Effective use of Traditional Chinese linguistic conventions**
- **Bilingual clarity:** specialised technical terms include the English original in parentheses

---

[來源: tutorial_instruction] s**
- **Bilingual clarity:** specialised technical terms include the English original in parentheses at first mention (headings/TOC/first in-paragraph occurrence); ≥95% of tutorials should pass the automated parenthetical QA check
- **High engagement through compelling content and hashtags**
- **Mandatory line-by-line analysis for all code examples**
- **Comprehensive Key Points Summary for each s

---

[來源: tutorial_instruction] datory line-by-line analysis for all code examples**
- **Comprehensive Key Points Summary for each section**

**Code Quality Indicators:**

---

[來源: tutorial_instruction] ode examples**
- **Comprehensive Key Points Summary for each section**

**Code Quality Indicators:**

- All examples execute without errors
- Code demonstrates best practices and modern Python idioms
- Comments enhance understanding without being redundant
- Examples build upon each other logically
- **Enhanced readability through comprehensive inline comments**
- **Clear correlation between code and detailed explanations**

---

[來源: tutorial_instruction] rough comprehensive inline comments**
- **Clear correlation between code and detailed explanations**

**Threads Platform Optimization:**

---

[來源: tutorial_instruction] *
- **Clear correlation between code and detailed explanations**

**Threads Platform Optimization:**

- **Engaging headlines and content structure**
- **Strategic use of emojis and visual hierarchy**
- **Effective hashtag strategy for maximum visibility**
- **Traditional Chinese content optimized for Taiwan audience**
- **Real-world application scenarios that resonate with developers**

---

[來源: tutorial_instruction] optimized for Taiwan audience**
- **Real-world application scenarios that resonate with developers**

This instruction set ensures consistent, high-quality Python educational content that serves both as comprehensive learning material and as practical, executable examples for readers to engage with. By following these guidelines, we can create a rich learning experience that empowers Python developers at all levels while maximizing engagement on the Threads social media platform.