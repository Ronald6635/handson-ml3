<!-- meta-title: 決策樹完整指南：CART 演算法、Gini 不純度與剪枝策略實戰 -->
<!-- meta-description: 深入理解決策樹的 CART 演算法、Gini 不純度、熵、資訊增益，學習如何用 Scikit-Learn 訓練、視覺化、正規化決策樹，包含分類和回歸完整實戰。 -->
<!-- meta-keywords: Python, 決策樹, CART, Gini, 資訊增益, Scikit-Learn, 機器學習, 正規化, 過擬合 -->
<!-- meta-hashtags: #Python #決策樹 #CART #機器學習 #ScikitLearn #Gini #資訊增益 #程式設計 #教學 #DataScience -->

# 🐍 決策樹完整指南：從 CART 到剪枝策略

決策樹（Decision Trees）是機器學習中最直觀的演算法——它的決策過程就像人類的問答遊戲。決策樹不僅能單獨使用，更是隨機森林（Random Forest）和梯度提升（Gradient Boosting）等集成方法的核心組件。本教學帶你深入理解決策樹的建構原理，並學習如何避免它天生的過擬合傾向。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🌳 訓練與視覺化決策樹](#train-visualize)
- [🔮 進行預測](#making-predictions)
- [📊 Gini 不純度與熵](#gini-entropy)
- [✂️ 正規化：控制樹的複雜度](#regularization)
- [📈 決策樹回歸](#regression)
- [⚠️ 決策樹的限制](#limitations)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- 決策樹使用 **CART 演算法**（分類與回歸樹），每次分裂選擇最能降低不純度的特徵和閾值
- **Gini 不純度**（預設）和**熵（Entropy）** 都是衡量節點不純度的指標，通常結果相似
- 決策樹本身有**高方差（High Variance）** 傾向——對訓練集的微小變化非常敏感
- 透過 `max_depth`、`min_samples_leaf` 等參數**預剪枝**是控制過擬合的主要方式
- 決策樹的決策邊界**平行於座標軸**，對旋轉資料敏感（可用 PCA 前處理改善）

---

## <a id="train-visualize"></a>🌳 訓練與視覺化決策樹

💡 **實際應用情境：** 台灣銀行的信用評估系統常使用決策樹，因為樹的結構可以直接向客戶解釋「為何拒絕貸款」（例如：年收入 < 60 萬 且 無房產 → 拒絕）。這種可解釋性在金融監管合規中尤其重要。

### 範例 1: 訓練鳶尾花分類樹

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import matplotlib.pyplot as plt

# 載入資料
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

# 訓練決策樹（限制深度防止過擬合）
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)
print(f"訓練準確率: {tree_clf.score(X_iris, y_iris):.4f}")

# 視覺化：Matplotlib 版（不需要 Graphviz）
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

1. `DecisionTreeClassifier(max_depth=2)`: 限制樹的最大深度為 2，避免過擬合
2. `plot_tree(filled=True)`: 以顏色深淺表示節點的主要類別和純度（越深表示越純）
3. 每個節點顯示：分裂條件、Gini 不純度、樣本數、各類別分佈

**🎯 重點摘要:**

- `export_graphviz()` 可生成 `.dot` 文件，用 Graphviz 渲染更美觀的圖表
- `plot_tree()` 無需額外安裝，適合快速視覺化

---

## <a id="making-predictions"></a>🔮 進行預測：機率估計

### 範例 2: 類別機率估計

```python
# 預測類別機率（而非只預測類別）
sample = [[5.0, 1.5]]  # 花瓣長 5cm，花瓣寬 1.5cm

proba = tree_clf.predict_proba(sample)
print("各類別機率:")
for cls, p in zip(iris.target_names, proba[0]):
    print(f"  {cls}: {p:.2%}")

# 預測類別（機率最高的那個）
pred_class = tree_clf.predict(sample)
print(f"預測類別: {iris.target_names[pred_class[0]]}")

# 查看預測路徑（Decision Path）
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

## <a id="gini-entropy"></a>📊 Gini 不純度與熵

### 理論說明：CART 分裂準則

**Gini 不純度（預設）**：

$$G_i = 1 - \sum_{k=1}^{K} p_{i,k}^2$$

- $p_{i,k}$：節點 $i$ 中類別 $k$ 的比例
- 純節點（只有一種類別）：$G_i = 0$
- 最不純（所有類別均等分佈）：$G_i = 1 - 1/K$

**熵（Entropy）**：

$$H_i = -\sum_{k=1}^{K} p_{i,k} \log_2(p_{i,k})$$

### 範例 3: Gini vs Entropy 的比較

```python
# Gini vs Entropy 的決策樹比較
tree_gini = DecisionTreeClassifier(criterion="gini",    max_depth=3, random_state=42)
tree_entr = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)

from sklearn.model_selection import cross_val_score

for name, clf in [("Gini", tree_gini), ("Entropy", tree_entr)]:
    scores = cross_val_score(clf, X_iris, y_iris, cv=5, scoring="accuracy")
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# 實際上兩者效果通常差不多（Gini 稍快，Entropy 稍準）
```

**🎯 重點摘要:**

- Gini 計算比熵快（不需要 log），通常是首選
- 兩種準則在大多數情況下產生相似的樹，差異不大
- **訊息增益（Information Gain）** = 父節點熵 − 加權子節點熵，即 CART 最大化的目標

---

## <a id="regularization"></a>✂️ 正規化：控制樹的複雜度

💡 **實際應用情境：** 未限制的決策樹會完美記憶訓練資料（訓練準確率 100%），但在新資料上效果很差。正規化參數就像「剪枝」，讓樹更簡潔、更泛化。

### 範例 4: 正規化超參數的影響

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score

X_moons, y_moons = make_moons(n_samples=200, noise=0.25, random_state=42)

# 各種正規化設定
configs = {
    "無限制": DecisionTreeClassifier(random_state=42),
    "max_depth=3": DecisionTreeClassifier(max_depth=3, random_state=42),
    "min_samples_leaf=5": DecisionTreeClassifier(min_samples_leaf=5, random_state=42),
    "min_samples_split=10": DecisionTreeClassifier(min_samples_split=10, random_state=42),
    "max_leaf_nodes=10": DecisionTreeClassifier(max_leaf_nodes=10, random_state=42),
}

print("正規化參數比較（5-Fold CV 準確率）:")
for name, clf in configs.items():
    scores = cross_val_score(clf, X_moons, y_moons, cv=5, scoring="accuracy")
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

## <a id="regression"></a>📈 決策樹回歸

### 範例 5: DecisionTreeRegressor

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

np.random.seed(42)
X_reg = np.sort(5 * np.random.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.randn(80) * 0.1

# 兩種深度的回歸樹
tree_reg_2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg_5 = DecisionTreeRegressor(max_depth=5, random_state=42)

tree_reg_2.fit(X_reg, y_reg)
tree_reg_5.fit(X_reg, y_reg)

print(f"depth=2 訓練 MSE: {((tree_reg_2.predict(X_reg) - y_reg)**2).mean():.4f}")
print(f"depth=5 訓練 MSE: {((tree_reg_5.predict(X_reg) - y_reg)**2).mean():.4f}")
# depth=5 的訓練誤差遠低，但可能過擬合
```

**✅ 程式碼逐行解析：**

1. 決策樹回歸的節點分裂準則：最小化加權 MSE（均方誤差）
2. 葉節點的預測值 = 落入該葉節點的所有訓練樣本的**均值**
3. `max_depth=5` 產生的階梯形預測曲線過度擬合了訓練資料的噪音

**🎯 重點摘要:**

- 決策樹回歸預測**分段常數函數**（階梯形），不能外推
- 回歸樹和分類樹除了分裂準則不同（MSE vs Gini），結構完全相同

---

## <a id="limitations"></a>⚠️ 決策樹的限制

### 範例 6: 決策邊界平行軸的限制

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

np.random.seed(6)
X_square = np.random.rand(100, 2) - 0.5
y_square = ((X_square[:, 0] > 0) ^ (X_square[:, 1] > 0)).astype(int)

# 旋轉 45 度後，決策樹需要更多層才能分類
angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle),  np.cos(angle)]])
X_rotated = X_square @ rotation_matrix

# 對比：旋轉前後的決策樹深度需求
tree_original = DecisionTreeClassifier(random_state=42)
tree_rotated  = DecisionTreeClassifier(random_state=42)

tree_original.fit(X_square, y_square)
tree_rotated.fit(X_rotated, y_square)

print(f"原始資料 - 樹深度: {tree_original.get_depth()}, 葉節點: {tree_original.get_n_leaves()}")
print(f"旋轉資料 - 樹深度: {tree_rotated.get_depth()}, 葉節點: {tree_rotated.get_n_leaves()}")
# 旋轉後需要更多分裂才能完成相同的分類任務
```

**🎯 重點摘要:**

- 決策樹的邊界**只能平行於特徵軸**，對角線邊界需要很多分裂
- 解法：用 PCA 預處理（旋轉特徵），或改用隨機森林（多棵樹的平均消除這個限制）

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: 決策樹需要特徵縮放嗎？**

A: 不需要！決策樹使用閾值分裂（`feature > threshold`），不依賴特徵之間的距離或尺度。這是決策樹相較於 SVM、KNN 的一大優點。

**Q2: Gini 和 Entropy 哪個更好？**

A: 通常差別不大。Gini 計算稍快（不需要 log），是預設值。若想精確測試，用交叉驗證比較兩者。

**Q3: 為什麼決策樹是「高方差」模型？**

A: 決策樹對訓練資料的微小變化非常敏感——移除幾個樣本或加入少量雜訊，可能完全改變樹的結構。這就是為什麼隨機森林（多棵決策樹的集成）通常比單棵決策樹效果好得多。

**Q4: `max_features` 參數有什麼用？**

A: 在隨機森林中使用，每次分裂只從隨機選出的部分特徵中搜索最佳分裂點。這引入了隨機性，讓每棵樹更獨立，集成效果更好。對單棵決策樹通常不需要設定。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #決策樹 #CART #機器學習 #ScikitLearn #Gini #資訊增益 #隨機森林 #程式設計 #教學 #DataScience #MachineLearning #AI #特徵重要性
