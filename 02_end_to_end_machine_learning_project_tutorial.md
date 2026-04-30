<!-- meta-title: 端到端機器學習專案實戰：加州房價預測完整指南 -->
<!-- meta-description: 跟著完整流程走一遍 ML 專案：資料下載、EDA、特徵工程、Pipeline 建構、交叉驗證、GridSearch 調參，到最終模型評估。使用 Scikit-Learn 實戰加州房價資料集。 -->
<!-- meta-keywords: Python, 機器學習, Scikit-Learn, 端到端, 資料科學, 房價預測, Pipeline, GridSearch, 交叉驗證 -->
<!-- meta-hashtags: #Python #機器學習 #ScikitLearn #端到端 #資料科學 #房價預測 #Pipeline #GridSearch #程式設計 #教學 -->

# 🐍 端到端機器學習專案實戰：加州房價預測

一個真實的 ML 專案需要哪些步驟？本教學帶您走完**完整的工作流程**——從原始資料到可部署的模型，以加州房價資料集為例，示範資料科學家的真實日常工作。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🚀 環境設定與資料下載](#setup)
- [📊 探索性資料分析](#eda)
- [🔧 特徵工程與資料前處理](#feature-engineering)
- [🏗️ 建構訓練 Pipeline](#pipeline)
- [🤖 模型選擇與訓練](#model-training)
- [🎛️ 超參數調優](#hyperparameter-tuning)
- [✅ 最終模型評估](#final-evaluation)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **Pipeline** 是防止資料洩漏（Data Leakage）的關鍵工具
- **分層抽樣（Stratified Sampling）** 確保測試集代表性
- **交叉驗證（Cross-Validation）** 提供更穩健的效能估計
- **GridSearchCV / RandomizedSearchCV** 系統性搜尋最佳超參數
- 最終評估應只用**測試集一次**，避免過度擬合評估指標

---

## <a id="setup"></a>🚀 環境設定與資料下載

💡 **實際應用情境：** 在正式啟動任何 ML 專案前，建立可重現的環境和自動化資料下載流程，能讓團隊成員快速接手繼續工作。

### 範例 1: 自動下載資料集

```python
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

## <a id="eda"></a>📊 探索性資料分析 (EDA)

💡 **實際應用情境：** 在建模前花時間理解資料，能避免後續的許多陷阱。例如：若收入欄位被人為限制在 [0.5, 15]，直接用它訓練可能導致模型無法預測極端值。

### 範例 2: 資料基本分析

```python
# 基本統計資訊
print(housing.info())        # 資料類型、缺失值概況
print(housing.describe())    # 數值型欄位的統計量

# 檢查缺失值
missing = housing.isnull().sum()
print(f"\n缺失值統計:\n{missing[missing > 0]}")

# 分類變數的值分佈
print(f"\nocean_proximity 分佈:\n{housing['ocean_proximity'].value_counts()}")

# 繪製數值型欄位的直方圖
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

### 範例 3: 分層抽樣建立訓練/測試集

```python
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# 根據收入等級（income_cat）分層抽樣
# 確保測試集的收入分佈代表整體
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

# 分層抽樣：每個 income_cat 按比例取 20% 進入測試集
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(housing, housing["income_cat"]):
    strat_train_set = housing.iloc[train_idx].drop("income_cat", axis=1)
    strat_test_set  = housing.iloc[test_idx].drop("income_cat", axis=1)

print(f"訓練集大小: {len(strat_train_set)}")   # ~16512
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

## <a id="feature-engineering"></a>🔧 特徵工程與資料前處理

### 範例 4: 自訂特徵組合轉換器

```python
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

**✅ 程式碼逐行解析：**

1. 繼承 `BaseEstimator, TransformerMixin`: 自動取得 `get_params()` 和 `fit_transform()` 方法
2. `fit(...)` 回傳 `self`: 此轉換器無需學習任何統計量，直接傳回自身
3. `np.c_[...]`: 按列（column）串接原始特徵和新特徵

**🎯 重點摘要:**

- **核心功能**: 從原有特徵組合出更具預測力的新特徵
- **設計原則**: 繼承 sklearn 基礎類別，確保與 Pipeline 無縫整合
- **最佳使用情境**: 需要將領域知識（Domain Knowledge）編碼為特徵時

---

## <a id="pipeline"></a>🏗️ 建構完整訓練 Pipeline

### 範例 5: 完整資料前處理 Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# 數值型特徵處理流程
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # 填補缺失值（用中位數）
    ("attribs_adder", FeatureAdder()),               # 新增組合特徵
    ("std_scaler", StandardScaler()),                # 標準化（μ=0, σ=1）
])

# 識別數值型和類別型欄位
housing_num = strat_train_set.drop("median_house_value", axis=1)
num_attribs = list(housing_num.select_dtypes(include=[np.number]))
cat_attribs = ["ocean_proximity"]

# ColumnTransformer：對不同欄位套用不同前處理
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),                   # 數值欄位
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),  # 類別欄位
])

# 準備訓練資料
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

## <a id="model-training"></a>🤖 模型選擇與訓練

### 範例 6: 比較多種模型

```python
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

**🎯 重點摘要:**

- **核心功能**: 用統一標準比較多種候選模型
- **選擇依據**: 在交叉驗證 RMSE 最低的模型上進行超參數調優

---

## <a id="hyperparameter-tuning"></a>🎛️ 超參數調優

### 範例 7: GridSearchCV 系統搜尋

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# 定義超參數搜尋空間
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

# 特徵重要性分析
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

- **核心功能**: 系統性找到最佳超參數組合
- **注意**: GridSearch 複雜度是 O(參數組合數 × CV折數)，參數空間大時改用 `RandomizedSearchCV`

---

## <a id="final-evaluation"></a>✅ 最終模型評估

### 範例 8: 測試集最終評估

```python
from sklearn.metrics import mean_squared_error

# 取出最佳模型
final_model = grid_search.best_estimator_

# 前處理測試集（只 transform，不 fit！）
X_test = full_pipeline.transform(strat_test_set.drop("median_house_value", axis=1))
y_test = strat_test_set["median_house_value"].copy()

# 最終預測
y_pred = final_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"測試集最終 RMSE: {final_rmse/1000:.1f}K 美元")

# 95% 置信區間
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

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: 為什麼一定要用 Pipeline？**

A: Pipeline 防止**資料洩漏（Data Leakage）**。若手動先 `fit_transform` 整個資料集再分割，則測試集的統計資訊（如 StandardScaler 的均值/標準差）已污染了轉換，導致評估結果虛高。Pipeline 確保 `scaler.fit` 只在訓練集上執行。

**Q2: GridSearch 和 RandomizedSearch 怎麼選？**

A: 超參數組合少（< 100）時用 GridSearch；組合多時用 RandomizedSearch（`n_iter` 參數控制嘗試次數）。RandomizedSearch 的優點是對連續型超參數效果更好，且計算成本可控。

**Q3: 特徵重要性分析有什麼用？**

A: 它能告訴你哪些特徵對模型預測最重要，幫助你：(1) 移除不重要的特徵降低維度；(2) 聚焦於最有價值的資料收集方向；(3) 向業務部門解釋模型決策依據。

**Q4: RMSE 和 MAE 有什麼不同？**

A: RMSE（均方根誤差）對大誤差懲罰更重（平方放大了大誤差的影響）；MAE（平均絕對誤差）對所有誤差一視同仁。若異常值/大誤差在業務上代價很高，用 RMSE；若想要更具解釋性的指標，用 MAE。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #機器學習 #ScikitLearn #端到端 #資料科學 #房價預測 #Pipeline #GridSearch #特徵工程 #交叉驗證 #程式設計 #教學 #DataScience #MachineLearning
