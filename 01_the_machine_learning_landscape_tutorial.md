<!-- meta-title: 機器學習全景導覽：監督、非監督與強化學習實戰入門 -->
<!-- meta-description: 完整介紹機器學習的分類體系、核心術語、訓練流程與常見挑戰。用 Scikit-Learn 實作線性回歸、K-Means 和 Pipeline，適合 ML 初學者到進階開發者。 -->
<!-- meta-keywords: Python, 機器學習, 監督學習, 非監督學習, Scikit-Learn, 資料科學, 教學, 人工智慧 -->
<!-- meta-hashtags: #Python #機器學習 #監督學習 #Scikit學習 #資料科學 #AI入門 #程式設計 #教學 #深度學習 #技術分享 -->

# 🐍 機器學習全景導覽：系統性學習 ML 的第一步

機器學習（Machine Learning）是人工智慧的核心技術，讓電腦從資料中學習，而非靠人工撰寫每一條規則。本教學帶您系統性地了解 ML 的分類體系、核心術語和常見挑戰，並搭配 Scikit-Learn 實作，從第一行程式碼開始建立扎實的 ML 基礎。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🤔 什麼是機器學習？](#what-is-ml)
- [📚 監督式學習](#supervised-learning)
- [🔍 非監督式學習](#unsupervised-learning)
- [🤸 半監督與強化學習](#semi-rl)
- [📦 批次學習與線上學習](#batch-vs-online)
- [📐 基於實例與基於模型的學習](#instance-vs-model)
- [⚠️ 機器學習的主要挑戰](#challenges)
- [🔬 測試與驗證](#testing-validation)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- 機器學習依**是否有標籤**分為監督、非監督、半監督學習
- 依**訓練時機**分為批次（離線）學習與線上（增量）學習
- 依**學習方法**分為基於實例（相似度比較）與基於模型（參數學習）
- **過擬合（Overfitting）** 是 ML 最常見的挑戰之一，正則化（Regularization）是主要解法
- 使用**交叉驗證（Cross-Validation）** 才能得到可靠的效能評估

---

## <a id="what-is-ml"></a>🤔 什麼是機器學習？

💡 **實際應用情境：** 垃圾郵件過濾器是最典型的 ML 應用。傳統方式需要人工列出所有關鍵詞規則；而 ML 方式是讓程式從大量標記過的郵件中**自動學習**規則，並隨著新垃圾郵件的出現持續進化。

機器學習（Machine Learning）是讓電腦從**經驗（資料）** 中學習，在**任務（Task）** 上表現越來越好的技術，由 Arthur Samuel（1959）提出。

核心三要素：

- **任務 T**：例如分類郵件是否為垃圾
- **經驗 E**：例如大量已標記的郵件
- **效能衡量 P**：例如分類準確率

### 範例 1: 基礎環境設定

```python
import sys
from packaging import version
import sklearn
import numpy as np
import matplotlib.pyplot as plt

# 確認 Python 版本 ≥ 3.7
assert sys.version_info >= (3, 7), "Python 版本不符合要求"

# 確認 Scikit-Learn ≥ 1.0.1
assert version.parse(sklearn.__version__) >= version.parse("1.0.1"), "Scikit-Learn 版本不符"

# 設定 Matplotlib 預設樣式
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

## <a id="supervised-learning"></a>📚 監督式學習 (Supervised Learning)

💡 **實際應用情境：** 台灣的信用評分模型即為典型監督學習——銀行有歷史客戶的還款記錄（標籤），用這些資料訓練模型預測新客戶的違約風險。

監督式學習使用帶有**標籤（Label）** 的訓練資料，學習從輸入 X 到輸出 y 的映射。

| 任務類型 | 輸出 y | 典型演算法 | 範例 |
|----------|--------|-----------|------|
| 分類（Classification） | 離散類別 | SVM、Random Forest | 垃圾郵件偵測 |
| 迴歸（Regression） | 連續數值 | Linear Regression、SVR | 房價預測 |

### 範例 2: 線性回歸基礎示範

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 生成模擬資料：房屋面積（坪）→ 房價（萬元）
np.random.seed(42)
X = np.random.rand(100, 1) * 50 + 10   # 面積：10~60 坪
y = 35 * X.ravel() + np.random.randn(100) * 100  # 每坪 35 萬 + 雜訊

# 訓練線性回歸模型
model = LinearRegression()
model.fit(X, y)

# 預測新樣本
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

- **核心功能**: 從歷史資料學習 X→y 的線性映射
- **潛在問題**: 若 X 與 y 非線性關係，線性回歸表現不佳
- **最佳使用情境**: 特徵與目標有線性關係且資料量適中

---

## <a id="unsupervised-learning"></a>🔍 非監督式學習 (Unsupervised Learning)

💡 **實際應用情境：** 電商平台對消費者分群（Customer Segmentation）——沒有「正確答案」，讓演算法從消費行為資料中自動找出有意義的族群，再針對不同族群設計行銷策略。

非監督學習**沒有標籤**，演算法從原始資料中自行發現結構和模式。

主要技術：

- **聚類（Clustering）**：K-Means、DBSCAN
- **降維（Dimensionality Reduction）**：PCA、t-SNE、UMAP
- **密度估計（Density Estimation）**：GMM
- **異常偵測（Anomaly Detection）**：Isolation Forest

### 範例 3: K-Means 客群分析

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 模擬消費者資料：購買頻率（次/月）× 平均消費金額（百元）
X_customers = np.r_[
    np.random.randn(100, 2) * 0.5 + [2, 5],   # 低頻低消費族群
    np.random.randn(100, 2) * 0.5 + [8, 15],  # 高頻高消費族群
    np.random.randn(100, 2) * 0.5 + [5, 8],   # 中頻中消費族群
]

# K-Means 分成 3 群
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

## <a id="semi-rl"></a>🤸 半監督學習與強化學習

### 半監督學習 (Semi-supervised Learning)

💡 **實際應用情境：** 醫學影像標注成本高昂（需要醫師判讀），但收集大量未標注影像相對容易。半監督學習利用**少量標注資料 + 大量未標注資料**，在資料稀缺時仍能訓練出優秀模型。

- 先用非監督學習（如聚類）發現資料結構
- 再用少量標籤「解釋」每個群的含義
- 典型應用：Google Photos 的人臉辨識、語音辨識

### 強化學習 (Reinforcement Learning)

💡 **實際應用情境：** AlphaGo、ChatGPT 的 RLHF 訓練都使用強化學習。Agent 在環境中採取動作、獲得獎勵，透過無數次嘗試和錯誤找到最優策略。

核心概念：**Agent**（智能體）→ **Action**（動作）→ **Environment**（環境）→ **Reward**（獎勵）→ **Policy**（策略）

---

## <a id="batch-vs-online"></a>📦 批次學習與線上學習

### 批次學習 (Batch Learning)

- 使用全部可用資料**離線訓練一次**
- 若有新資料需**重新訓練整個模型**
- 適合：資料量固定、模型無需頻繁更新

### 線上學習 (Online Learning)

```python
from sklearn.linear_model import SGDRegressor

# SGD 迴歸支援增量學習（partial_fit）
model_online = SGDRegressor(random_state=42, max_iter=1)

# 模擬串流資料：每次收到一批新資料
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

## <a id="challenges"></a>⚠️ 機器學習的主要挑戰

### 過擬合 (Overfitting) vs 欠擬合 (Underfitting)

| 問題 | 症狀 | 解法 |
|------|------|------|
| 過擬合（模型太複雜） | 訓練集準確率高，測試集低 | 正則化、更多資料、Dropout |
| 欠擬合（模型太簡單） | 訓練集和測試集準確率都低 | 更複雜模型、更多特徵 |

### 範例 4: 學習曲線診斷過擬合/欠擬合

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge

# 高次多項式 → 過擬合
overfitting_model = make_pipeline(
    PolynomialFeatures(degree=20),  # 20 次多項式，幾乎記住訓練資料
    LinearRegression()
)

# 加入 Ridge 正則化 → 緩解過擬合
regularized_model = make_pipeline(
    PolynomialFeatures(degree=20),
    Ridge(alpha=10)  # alpha 越大，正則化越強
)

# 資料量不足時，兩個模型都有問題
# 解法：增加更多訓練資料或降低模型複雜度
```

**🎯 重點摘要:**

- **核心功能**: 識別並解決過擬合與欠擬合問題
- **重要原則**: 在驗證集上評估，而非訓練集
- **最佳使用情境**: 任何 ML 模型訓練後的必要診斷步驟

---

## <a id="testing-validation"></a>🔬 測試與驗證

### 範例 5: 交叉驗證最佳實踐

```python
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# 切分資料（先保留測試集，不碰它！）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 建立包含預處理的 Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # 特徵標準化（μ=0, σ=1）
    ("model", Ridge(alpha=1.0))
])

# 5-Fold 交叉驗證（在訓練集上）
cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"  # 使用 RMSE
)
print(f"CV RMSE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# 最終評估（只在最後用測試集一次）
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

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: 監督學習與非監督學習如何選擇？**

A: 關鍵問題是「是否有標籤資料」。若業務問題有明確的輸入-輸出對應（如信用評分），用監督學習。若只有原始資料且目標是「發現結構」（如客群分析），用非監督學習。

**Q2: 為什麼要用 Pipeline 而非手動呼叫 `fit_transform`？**

A: Pipeline 防止**資料洩漏（Data Leakage）**。若手動在訓練集上 `fit_transform` 後再分割，測試集的統計資訊（如均值、標準差）已污染了 Scaler，導致驗證結果過於樂觀。

**Q3: 過擬合和高偏差/高方差的關係是什麼？**

A: 過擬合對應**高方差（High Variance）**——模型對訓練資料的微小變化過度敏感；欠擬合對應**高偏差（High Bias）**——模型假設太簡單，無法捕捉真實規律。這兩者的取捨稱為**偏差-方差權衡（Bias-Variance Tradeoff）**。

**Q4: 什麼時候用 `cross_val_score` vs 保留驗證集？**

A: 資料量充足時兩者皆可；資料量少時，交叉驗證能充分利用每個樣本，是更好的選擇。注意：無論如何，測試集（最終評估集）應與這個過程完全隔離。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #機器學習 #監督學習 #非監督學習 #Scikit學習 #資料科學 #AI入門 #程式設計 #深度學習 #技術分享 #MachineLearning #DataScience #學習筆記 #軟體工程
