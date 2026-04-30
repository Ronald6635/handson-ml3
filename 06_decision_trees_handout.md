# 課程講義：決策樹 (Chapter 06)

決策樹是機器學習中最「可解釋」的模型之一——它的決策過程就像一系列的「是/否」問題，任何人都能讀懂。本章帶你理解決策樹的訓練演算法（CART）、Gini 不純度、以及為何決策樹對訓練集的「邊界形狀」非常敏感。這些知識是理解第七章集成學習（尤其是隨機森林）的必要基礎。

---

## 1. CART 演算法與節點分裂準則

### 理論背景

**CART (Classification and Regression Trees)** 演算法：

在每個節點，CART 窮舉所有特徵 $j$ 和所有閾值 $t_k$，找到能最小化加權不純度的分裂點：

$$J(j, t_k) = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}$$

**Gini 不純度**：

$$G_i = 1 - \sum_{k=1}^{K} p_{i,k}^2$$

其中 $p_{i,k}$ 是節點 $i$ 中類別 $k$ 的樣本比例。$G_i = 0$ 表示節點完全純淨（只含一種類別）。

**信息增益（Entropy 基準）**：

$$H_i = -\sum_{k=1, p_{i,k} \neq 0}^{K} p_{i,k} \log_2(p_{i,k})$$

Gini 計算較快，Entropy 有時能產生更平衡的樹；實際差異通常不大，Gini 是預設值。

**時間複雜度**：訓練 $O(n \cdot m \log m)$，預測 $O(\log m)$（$m$ 為訓練樣本數，$n$ 為特徵數）。

### 核心代碼

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

# 訓練決策樹（限制深度防止過擬合）
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

# 視覺化樹結構
plt.figure(figsize=(12, 5))
plot_tree(tree_clf,
          feature_names=["petal length", "petal width"],
          class_names=iris.target_names,
          filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree (max_depth=2)")
plt.tight_layout()
plt.show()
```

### ⚡ 補充練習 1

**理論題：** 手動計算根節點（所有 150 筆 Iris 資料，三類各 50 筆）的 Gini 不純度。再計算根節點按「花瓣長度 ≤ 2.45 cm」分裂後，左子節點的 Gini 不純度（左節點包含 50 筆 setosa）。

**實作題：** 訓練 `max_depth=3` 的決策樹，用 `plot_tree` 視覺化，找出在哪個節點 Gini 不純度最低，對應的分裂條件是什麼？

---

## 2. 節點解讀與預測機率

### 理論背景

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

**白盒 vs 黑盒**：決策樹是「白盒」模型——可以追蹤任何預測的決策路徑，清楚說明「為何」做出此預測，對高風險應用（醫療、法律）非常重要。

### 核心代碼

```python
# 預測機率和類別
X_new = [[5, 1.5]]
print(tree_clf.predict_proba(X_new))  # [[0., 0.907, 0.093]]
print(tree_clf.predict(X_new))        # [1] → Iris versicolor

# 查看決策路徑（某樣本走哪條路）
from sklearn.tree import _tree

def get_decision_path(clf, X_sample):
    node_indicator = clf.decision_path(X_sample)
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    for node_id in node_indicator.indices:
        if feature[node_id] != _tree.TREE_UNDEFINED:
            print(f"  節點 {node_id}: X[{feature[node_id]}] "
                  f"{'<=' if X_sample[0, feature[node_id]] <= threshold[node_id] else '>'} "
                  f"{threshold[node_id]:.2f}")

import numpy as np
get_decision_path(tree_clf, np.array([[5, 1.5]]))
```

### ⚡ 補充練習 2

**理論題：** 決策樹預測機率時，葉節點中某類別只有 1 筆樣本，但被預測為 100% 機率，這合理嗎？Laplace Smoothing 如何解決這個問題？

**實作題：** 用 `DecisionTreeClassifier(max_depth=4)` 繪製決策邊界（在 petal length × petal width 的 2D 空間）。注意邊界是水平和垂直線組成的「直角」形狀——這正是決策樹偏好軸對齊分裂的特點。

---

## 3. 正則化超參數與過擬合

### 理論背景

未加限制的決策樹會生長到每個葉節點只含一個樣本（訓練準確率 100%，嚴重過擬合）。

**主要正則化超參數**：

| 超參數 | 說明 | 增大的效果 |
|--------|------|-----------|
| `max_depth` | 樹的最大深度 | 降低複雜度（防過擬合） |
| `min_samples_split` | 節點分裂所需最少樣本數 | 降低複雜度 |
| `min_samples_leaf` | 葉節點最少樣本數 | 降低複雜度 |
| `max_features` | 每次分裂考慮的最多特徵數 | 降低複雜度（增加隨機性） |
| `max_leaf_nodes` | 最多葉節點數 | 降低複雜度 |

**剪枝 (Pruning)**：先訓練完整的樹，再移除對測試集效能貢獻不大的節點（後剪枝）。`ccp_alpha` 參數控制最小代價複雜度剪枝。

### 核心代碼

```python
from sklearn.model_selection import train_test_split, GridSearchCV

# 各種深度對比
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris,
                                                     test_size=0.3, random_state=42)
for depth in [None, 2, 3, 5]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    print(f"max_depth={str(depth):4s}  "
          f"train acc={clf.score(X_train, y_train):.3f}  "
          f"test acc={clf.score(X_test, y_test):.3f}")

# 最小代價複雜度剪枝
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # 去掉最後一個（退化的樹）
clfs = [DecisionTreeClassifier(ccp_alpha=a, random_state=42).fit(X_train, y_train)
        for a in ccp_alphas]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
best_alpha = ccp_alphas[test_scores.index(max(test_scores))]
print(f"最佳 ccp_alpha: {best_alpha:.6f}")
```

### ⚡ 補充練習 3

**理論題：** `min_samples_leaf=5` 意味著什麼？為什麼設定葉節點的最小樣本數比限制深度更「均勻」地正則化樹？

**實作題：** 在 `make_moons(n_samples=500, noise=0.3)` 資料集上，用 `GridSearchCV` 搜索 `max_depth` ∈ [3, 5, 7, 10] 與 `min_samples_leaf` ∈ [1, 3, 5, 10] 的最佳組合。繪製最佳模型的決策邊界。

---

## 4. 決策樹迴歸

### 理論背景

**迴歸樹 (Regression Tree)**：葉節點的預測值是該節點訓練樣本的**平均值**。

CART 迴歸的分裂準則：最小化 MSE（而非 Gini 不純度）：

$$J(j, t_k) = \frac{m_{\text{left}}}{m} \text{MSE}_{\text{left}} + \frac{m_{\text{right}}}{m} \text{MSE}_{\text{right}}$$

**決策樹的根本弱點**：

1. **對旋轉敏感**：決策邊界只能軸對齊；若資料旋轉 45°，需要更深的樹才能近似斜線邊界。
2. **高變異數 (High Variance)**：訓練資料稍有變動，整棵樹可能截然不同。
3. **貪婪演算法**：每次只選當前最好的分裂，不保證全域最優。

這些弱點正是為什麼**集成方法（隨機森林、Gradient Boosting）**能大幅超越單棵決策樹。

### 核心代碼

```python
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X_reg = np.sort(6 * np.random.rand(200, 1) - 3, axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.randn(200) * 0.2

tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_reg, y_reg)

# 視覺化預測曲線（決策樹迴歸是分段常數函數）
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
plt.scatter(X_reg, y_reg, s=10, alpha=0.5)
plt.plot(X_plot, tree_reg.predict(X_plot), "r-", linewidth=2)
plt.title("Decision Tree Regression (max_depth=3)")
plt.show()
```

### ⚡ 補充練習 4

**理論題：** 決策樹迴歸預測的是分段常數函數（每個葉節點的輸出是一個固定值）。這意味著決策樹**無法外推（Extrapolate）**——超出訓練資料範圍的預測會怎樣？

**實作題：** 用 `max_depth=None`（未剪枝）和 `max_depth=3` 分別訓練迴歸樹，比較兩者在訓練集和測試集的 RMSE，並繪製預測曲線，直觀觀察過擬合現象。

---

## 結論

決策樹以直觀的樹狀結構實現分類與迴歸：

- **CART 演算法**透過貪婪地最小化 Gini 不純度（或 MSE）分裂節點
- **葉節點**給出預測類別（多數決）或預測值（平均）
- **正則化超參數**（`max_depth`、`min_samples_leaf`、`ccp_alpha`）是防止過擬合的關鍵
- 決策樹的**高變異數弱點**促成了第七章的集成學習

下一章（Ch07）展示如何將許多棵弱決策樹組合成強大的集成模型。

---

## 課後作業

**作業：月亮資料集完整分析**

1. 使用 `make_moons(n_samples=10000, noise=0.4)` 產生資料集，切分為訓練集（8000）和測試集（2000）。
2. 用 `GridSearchCV` 找到最佳的 `DecisionTreeClassifier`（搜索 `max_leaf_nodes` ∈ [10, 50, 100, 500]、`min_samples_split` ∈ [2, 5, 10]）。
3. 在測試集評估最佳模型的準確率。
4. **進階**：訓練 1000 個決策樹，每個樹用 100 個隨機抽取的訓練樣本（帶回放），每棵樹對測試集預測，最終用多數投票決定類別。這就是隨機森林的原型！比較這個手工隨機森林與 `RandomForestClassifier` 的準確率。
