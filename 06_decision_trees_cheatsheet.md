# Ch06 速查表：Decision Trees

> **核心主旨**：決策樹直觀可解釋，但容易 overfitting —— 正則化（限制深度）是必要的。

---

## 1. 核心概念一覽

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

## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `DecisionTreeClassifier` | `max_depth=5`, `min_samples_leaf=10`, `criterion="gini"` | 分類樹 |
| `DecisionTreeRegressor` | `max_depth=5`, `min_samples_split=20` | 迴歸樹 |
| `export_graphviz` | `feature_names=`, `class_names=`, `filled=True` | 輸出 .dot 圖形檔 |
| `.feature_importances_` | – | 特徵重要性陣列 |
| `.predict_proba()` | – | 輸出類別機率 |
| `plot_tree` | `filled=True`, `feature_names=`, `max_depth=3` | 直接在 matplotlib 上繪製 |

---

## 3. 必備代碼片段

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_graphviz
import matplotlib.pyplot as plt
import numpy as np

# 訓練分類樹（加正則化）
tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_clf.fit(X_train, y_train)

# 視覺化（最直觀的方法）
plt.figure(figsize=(12, 6))
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names, rounded=True)
plt.show()

# 匯出 graphviz（需要 graphviz 套件）
export_graphviz(tree_clf, out_file="tree.dot",
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                filled=True, rounded=True)

# 預測機率
iris = load_iris(as_frame=True)
X, y = iris.data[["petal length (cm)", "petal width (cm)"]], iris.target
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)
print(tree_clf.predict_proba([[5, 1.5]]))  # 各類別機率
print(tree_clf.predict([[5, 1.5]]))         # 類別預測

# 特徵重要性
importances = tree_clf.feature_importances_
for name, imp in zip(iris.feature_names, importances):
    print(f"{name}: {imp:.3f}")

# 迴歸樹
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train, y_train)

# 存取樹結構（進階）
n_nodes = tree_clf.tree_.node_count
print(f"節點數: {n_nodes}")
```

---

## 4. 常見陷阱

- **不需要 Scaling**：決策樹基於排序（分割點），特徵尺度不影響結果，**不需要 StandardScaler**。
- **不穩定性（High Variance）**：資料略微改動可能產生完全不同的樹結構，這是決策樹的天然缺陷 —— 改用 Random Forest 更穩定。
- **對旋轉敏感**：決策樹只做軸向切割，若類別邊界是斜線，需要 PCA 預處理或改用其他模型。
- **過深的樹 = 過擬合**：通常 `max_depth=4~8` 是合理起點，用 CV 調整。

---

## 5. 決策指南

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
