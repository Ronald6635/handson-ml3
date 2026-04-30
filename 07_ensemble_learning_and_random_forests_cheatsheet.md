# Ch07 速查表：Ensemble Learning & Random Forests

> **核心主旨**：集成多個弱學習器往往勝過單一強模型 —— Random Forest 是最佳預設選擇，GBM 通常更強但需謹慎調參。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
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

## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
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

## 3. 必備代碼片段

```python
from sklearn.ensemble import (VotingClassifier, BaggingClassifier,
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, GradientBoostingClassifier,
    GradientBoostingRegressor, StackingClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

# 軟投票集成（不同類型模型）
voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(probability=True, random_state=42))
    ],
    voting="soft"  # 用 predict_proba 平均，通常優於 hard voting
)
voting_clf.fit(X_train, y_train)

# Random Forest（最實用的基準線）
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",   # 每次分割只看 sqrt(n_features) 個特徵
    oob_score=True,        # 免費的 OOB 驗證
    n_jobs=-1,             # 並行使用所有 CPU
    random_state=42
)
rf_clf.fit(X_train, y_train)
print(f"OOB score: {rf_clf.oob_score_:.4f}")

# 特徵重要性（排序）
importances = rf_clf.feature_importances_
feature_names = X_train.columns.tolist()
sorted_idx = np.argsort(importances)[::-1]
for i in sorted_idx[:10]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Gradient Boosting（調參關鍵：少棵樹 + 低學習率）
gbm_clf = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,  # 學習率越低需要越多樹，但泛化越好
    max_depth=3,         # 淺樹（2-5）是 GBM 的慣例
    subsample=0.8,       # 隨機取 80% 樣本，防 overfitting
    random_state=42
)
gbm_clf.fit(X_train, y_train)

# Stacking（競賽提分利器）
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

## 4. 常見陷阱

- **Voting 需要 `probability=True`**：`SVC` 預設不輸出機率，soft voting 時必須加 `probability=True`（訓練慢）。
- **GBM learning_rate 與 n_estimators 互相制約**：降低 `learning_rate` 必須同時增加 `n_estimators`；常用配合：`lr=0.05, n_estimators=500`。
- **Random Forest 不需要 Scaling**：基於決策樹，特徵尺度無關。
- **AdaBoost 對 outlier 敏感**：outlier 會被反覆增加權重，影響後續模型。
- **`n_jobs=-1` 別忘加**：RF 和 GBM 原生支援並行，加了訓練速度倍增。

---

## 5. 決策指南

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
