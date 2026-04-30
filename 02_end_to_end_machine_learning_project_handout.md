# 課程講義：端到端機器學習專案實戰 (Chapter 02)

恭喜你進入第一個完整的機器學習專案！本章以**加州房價預測**為主軸，帶你走過一位資料科學家的完整工作流程：從取得資料、探索、清洗、特徵工程、模型訓練，到超參數調整與最終評估。這套流程是工業界的通用範本，每一步都有不可省略的原因。

---

## 1. 問題框架化與資料取得

### 理論背景

開始動手前，務必回答三個問題：

1. **目標**：預測房價中位數（連續值 → 迴歸問題）
2. **評估指標**：RMSE（Root Mean Squared Error）對大誤差懲罰更重；若 outlier 多可改用 MAE
3. **下游影響**：預測結果用於何處？允許的誤差範圍是多少？

$$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2}$$

**分層抽樣 (Stratified Sampling)**：若收入中位數（`median_income`）對房價影響最大，應按收入分層後再切分訓練/測試集，確保兩者分佈一致，避免抽樣偏差（Sampling Bias）。

### 核心代碼

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

housing = pd.read_csv("datasets/housing/housing.csv")

# 建立收入類別，用於分層
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

# 分層抽樣
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(housing, housing["income_cat"]):
    train_set = housing.loc[train_idx]
    test_set  = housing.loc[test_idx]

# 移除輔助欄位
for s in (train_set, test_set):
    s.drop("income_cat", axis=1, inplace=True)
```

### ⚡ 補充練習 1

**理論題：** 為什麼要用分層抽樣而非隨機抽樣？若資料量很大（例如 100 萬筆），分層抽樣還有必要嗎？

**實作題：** 比較 `StratifiedShuffleSplit` 與 `train_test_split(random_state=42)` 在各收入類別比例上的差異，使用 `value_counts(normalize=True)` 量化。

---

## 2. 探索性資料分析 (EDA)

### 理論背景

EDA 的目的是發現：資料分佈、異常值、特徵間相關性，以及需要什麼前處理。

**相關係數矩陣 (Correlation Matrix)**：

$$r_{XY} = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2} \cdot \sqrt{\sum (Y_i - \bar{Y})^2}}$$

$r$ 範圍在 $[-1, 1]$：$|r|$ 越大相關越強；正值為正相關，負值為負相關。注意：相關係數只衡量**線性**相關。

**特徵工程的黃金法則**：先 EDA，再動手創造特徵。

### 核心代碼

```python
# 繪製地理分布（圓的大小=人口，顏色=房價）
housing.plot(kind="scatter", x="longitude", y="latitude",
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             alpha=0.4, figsize=(10, 7))

# 相關係數矩陣
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 創造衍生特徵（通常比原始特徵更有用）
housing["rooms_per_house"]  = housing["total_rooms"]  / housing["households"]
housing["bedrooms_ratio"]   = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"]  / housing["households"]
```

### ⚡ 補充練習 2

**理論題：** 散點圖顯示 `median_income` 與 `median_house_value` 高度相關，但存在水平條紋（房價被限制在 500,000 美元）。這對模型有何影響？如何處理？

**實作題：** 使用 `scatter_matrix` 或 `seaborn.pairplot` 視覺化 `median_income`、`housing_median_age`、`median_house_value`、`rooms_per_house` 四個特徵的兩兩相關圖，觀察哪些特徵組合最值得關注。

---

## 3. 特徵工程與資料前處理 Pipeline

### 理論背景

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

### 核心代碼

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_features = ["longitude", "latitude", "housing_median_age",
                "total_rooms", "total_bedrooms", "population",
                "households", "median_income",
                "rooms_per_house", "bedrooms_ratio", "people_per_house"]
cat_features = ["ocean_proximity"]

# 數值特徵 Pipeline：填補缺失值 → 縮放
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# 完整前處理管線
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])
```

### ⚡ 補充練習 3

**理論題：** 為什麼 `SimpleImputer` 的 `strategy="median"` 比 `strategy="mean"` 更適合有偏態分佈的特徵？

**實作題：** 手動建立一個 `FunctionTransformer`，將 `total_rooms` 除以 `households` 得到 `rooms_per_house`，並整合進 `ColumnTransformer`。確認前處理後的資料形狀正確。

---

## 4. 模型選擇、訓練與超參數調整

### 理論背景

模型選擇策略：先選幾種不同類型的候選模型（線性、樹狀、集成），用交叉驗證快速篩選，再對最佳候選深入調參。

**超參數搜索**：

| 方法 | 說明 | 適用場景 |
|------|------|---------|
| `GridSearchCV` | 窮舉所有組合 | 超參數少（< 5）|
| `RandomizedSearchCV` | 隨機取樣指定次數 | 超參數多 |
| `HalvingRandomSearchCV` | 進化式篩選，速度最快 | 資源有限時 |

最終模型評估：計算測試集 RMSE，並用 95% 信賴區間量化估計的不確定性：

$$\text{95\% CI} \approx \bar{e} \pm 1.96 \cdot \frac{\sigma_e}{\sqrt{m}}$$

### 核心代碼

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 完整的端到端 Pipeline
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", RandomForestRegressor(random_state=42))
])

# 超參數搜索
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

# 最終測試集評估
final_model = rnd_search.best_estimator_
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()
final_rmse = mean_squared_error(y_test, final_model.predict(X_test), squared=False)
print(f"測試集 RMSE: {final_rmse:.0f}")
```

### ⚡ 補充練習 4

**理論題：** `RandomizedSearchCV` 與 `GridSearchCV` 的主要差異是什麼？在哪些情況下 Randomized 更有優勢？

**實作題：** 使用 `feature_importances_` 提取 `RandomForestRegressor` 的特徵重要性，繪製條形圖，指出重要性最高的前 5 個特徵是否符合你的直覺。

---

## 結論

本章展示了一個完整的機器學習工作流程，其關鍵里程碑：

1. **問題定義** → 確定任務類型與評估指標
2. **資料取得與分層切分** → 防止測試集偏差
3. **EDA + 特徵工程** → 發現有用訊號
4. **Pipeline 封裝前處理** → 防止資料洩漏
5. **交叉驗證模型選擇 + 超參數搜索** → 找到最佳模型
6. **測試集最終評估** → 誠實估計泛化能力

下一章（Ch03）將深入分類問題的評估指標，你會發現準確率（Accuracy）有時是具有誤導性的指標。

---

## 課後作業

**作業：California Housing 完整專案**

在本章的 California Housing 資料集上，完成以下擴展：

1. 在 `preprocessing` Pipeline 中加入一個 `FunctionTransformer`，新增 `rooms_per_house`、`bedrooms_ratio`、`people_per_house` 三個衍生特徵。
2. 用 `RandomizedSearchCV` 比較 `RandomForestRegressor` 與 `GradientBoostingRegressor` 的交叉驗證 RMSE，哪個更好？
3. 計算最佳模型在測試集上的 95% 信賴區間（提示：使用 `scipy.stats.t.interval`）。
