# Ch02 速查表：End-to-End Machine Learning Project

> **核心主旨**：完整 ML 專案流程 —— 從資料獲取、探索、前處理到模型選擇與超參數調優。

---

## 1. 核心概念一覽

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

## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
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

## 3. 必備代碼片段

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# 分層切分
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(df, df["income_cat"]):
    train_set = df.iloc[train_idx]
    test_set = df.iloc[test_idx]

# 定義 Pipeline
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

# 完整模型 Pipeline
full_model = Pipeline([
    ("preprocessing", full_pipeline),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# 交叉驗證
scores = cross_val_score(full_model, X_train, y_train,
                         cv=10, scoring="neg_root_mean_squared_error")
print(f"RMSE: {-scores.mean():.0f} ± {scores.std():.0f}")

# 隨機搜索超參數
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

## 4. 常見陷阱

- **Pipeline 外做 Scaling**：會造成 data leakage，scaling 的 `fit()` 必須只看訓練資料。
- **直接用 `train_test_split` 不做分層**：若目標分佈不均，小型測試集的評估會有偏差。
- **GridSearch 看 `refit=True`**：預設會在全部訓練資料上 refit 最佳模型，直接呼叫 `.predict()` 即可。
- **特徵工程做在 Pipeline 外**：若手動加特徵（例如 `rooms_per_household`），新資料也要同樣處理，最好包進 `FunctionTransformer` 或自訂 Transformer。

---

## 5. 決策指南

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
