# 課程講義：集成學習與隨機森林 (Chapter 07)

「三個臭皮匠，勝過一個諸葛亮。」集成學習（Ensemble Learning）正是這個哲學的機器學習體現：將**許多個弱模型組合**成一個強大的預測器。本章介紹四大集成技術：投票、Bagging、Boosting 和 Stacking，以及它們的理論基礎——為何一群「差不多但互不相同」的模型，組合後可以大幅超越其中任何一個。

---

## 1. 投票分類器：硬投票與軟投票

### 理論背景

**弱學習器的集體智慧**：設每個分類器的準確率為 51%，且它們的錯誤**相互獨立**。1000 個這樣的分類器多數投票的準確率是多少？

$$P(\text{多數正確}) = \sum_{k=501}^{1000} \binom{1000}{k} (0.51)^k (0.49)^{1000-k} \approx 75\%$$

即使每個分類器只比隨機略好一點，大量集成後效果顯著提升！

**硬投票 (Hard Voting)**：每個分類器投一票，取多數類別。

**軟投票 (Soft Voting)**：取所有分類器**預測機率的平均值**，選機率最高的類別。通常優於硬投票（因為高確信度的預測有更大的發言權）。

$$\hat{y} = \arg\max_k \frac{1}{N} \sum_{i=1}^{N} \hat{p}_{i,k}$$

**關鍵前提**：分類器的錯誤需要**多樣化（多樣性）**——犯的錯誤盡量不同。若所有分類器犯同樣的錯誤，集成無效！

### 核心代碼

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 三種不同類型的分類器（多樣性的來源）
log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
svm_clf = SVC(probability=True, random_state=42)  # 軟投票需要 probability=True

# 軟投票集成
voting_clf = VotingClassifier(
    estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
    voting="soft"
)
voting_clf.fit(X_train, y_train)

# 比較各個分類器和集成的準確率
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    print(f"{clf.__class__.__name__:25s}: {clf.score(X_test, y_test):.4f}")
```

### ⚡ 補充練習 1

**理論題：** 軟投票通常優於硬投票，但有一個例外情況：若某個分類器的機率輸出未校正（systematically overconfident），會對軟投票造成什麼影響？可以用什麼方法校正機率輸出？

**實作題：** 分別建立「3 個完全相同的 `LogisticRegression`」和「`LogisticRegression` + `RandomForest` + `SVC`」的硬投票集成，比較兩者的測試集準確率，驗證「多樣性」對集成效果的重要性。

---

## 2. Bagging 與 Pasting

### 理論背景

**Bagging (Bootstrap Aggregating)**：用**有放回抽樣 (Bootstrap)** 從訓練集中抽取子集，訓練同一類型的多個分類器。

**Pasting**：用**無放回抽樣**。

Bagging 的優點：
- 每個基學習器看到的資料不同 → 多樣性
- Bootstrap 抽樣讓每個樣本有約 $1 - (1-1/m)^m \approx 63.2\%$ 的機率被選中；剩下 ~36.8% 的樣本作為天然的**袋外樣本 (OOB)**，可用於驗證而無需額外驗證集

**OOB 評估 (Out-of-Bag Evaluation)**：每個訓練樣本對沒有用到它的那些基學習器進行預測，聚合後得到 OOB 分數，是對泛化誤差的合理估計。

### 核心代碼

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging：500 棵決策樹，各用 100 個樣本（有放回）
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

print(f"OOB 評估準確率: {bag_clf.oob_score_:.4f}")     # 接近測試集準確率
print(f"測試集準確率:   {bag_clf.score(X_test, y_test):.4f}")
print(f"OOB 決策機率（前 3 筆）:\n{bag_clf.oob_decision_function_[:3]}")
```

### ⚡ 補充練習 2

**理論題：** 解釋 OOB 評估為何能作為測試集的替代品。若 `n_estimators` 很小（比如 10），OOB 評估還可靠嗎？

**實作題：** 比較 Bagging（`bootstrap=True`）和 Pasting（`bootstrap=False`）在 `make_moons` 上的 OOB 分數和測試集準確率，哪種方法通常表現更好？為什麼？

---

## 3. 隨機森林與特徵重要性

### 理論背景

**隨機森林 (Random Forest)**：Bagging 的特化版，在每次分裂時，從**隨機抽取的 $\sqrt{n}$ 個特徵**中選最佳分裂點。這增加了樹之間的多樣性，通常優於純 Bagging。

**極端隨機樹 (Extra-Trees)**：進一步隨機化——不只隨機選特徵，連閾值也隨機選（而非最佳）。訓練更快，偏差略高但變異數更低。

**特徵重要性**：每個特徵在所有樹中造成的平均不純度下降量：

$$\text{Feature Importance}_j = \frac{\sum_{\text{nodes using } j} w_i \cdot \Delta G_i}{\text{sum of all node importances}}$$

### 核心代碼

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

# 特徵重要性
feature_importances = rf_clf.feature_importances_
for name, score in zip(iris.feature_names, feature_importances):
    print(f"{name:25s}: {score:.4f}")

# 視覺化
plt.bar(iris.feature_names, feature_importances)
plt.title("Feature Importances (Random Forest)")
plt.ylabel("Mean Decrease in Impurity")
plt.show()

# Extra-Trees（速度更快）
et_clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)
et_clf.fit(iris.data, iris.target)
```

### ⚡ 補充練習 3

**理論題：** 隨機森林在每次分裂時只考慮 $\sqrt{n}$ 個特徵，這帶來了偏差-變異數的什麼取捨？為何減少考慮的特徵數量反而能提升泛化效果？

**實作題：** 在 MNIST 數據集（10000 筆子集）上訓練 `RandomForestClassifier`，計算 `feature_importances_`，將重要性值 reshape 為 28×28 並顯示為熱力圖，觀察哪些像素對手寫數字分類最重要。

---

## 4. Boosting：AdaBoost 與 Gradient Boosting

### 理論背景

**Boosting**：循序訓練分類器，每個分類器**修正前一個的錯誤**。

**AdaBoost**：提高被前一個分類器誤分類的樣本的權重：

$$\hat{y}(\mathbf{x}) = \text{sign}\left(\sum_{j=1}^{N} \alpha_j h_j(\mathbf{x})\right)$$

其中 $\alpha_j = \eta \log\frac{1 - r_j}{r_j}$（$r_j$ 是加權誤差率，誤差越低 $\alpha_j$ 越大）。

**Gradient Boosting**：每個新樹擬合**前一個集成的殘差（偽殘差）**：

$$\hat{y}^{(i)}_{\text{new}} = \hat{y}^{(i)}_{\text{old}} + \eta h_t(\mathbf{x}^{(i)})$$

其中 $h_t$ 是擬合殘差 $y^{(i)} - \hat{y}^{(i)}_{\text{old}}$ 的決策樹，$\eta$ 是學習率（縮水因子）。

**過擬合防治**：
- `n_estimators` + `learning_rate`：小學習率需要更多樹（通常效果更好）
- `subsample < 1.0`：每棵樹用隨機子集 → **Stochastic GBM**
- 早停（`n_iter_no_change`）

### 核心代碼

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor

# AdaBoost（使用深度 1 的決策樹作為基學習器）
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),  # 決策樹樁（stump）
    n_estimators=30,
    learning_rate=0.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)
print(f"AdaBoost 測試準確率: {ada_clf.score(X_test, y_test):.4f}")

# Gradient Boosting（迴歸）
from sklearn.datasets import make_regression
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

### ⚡ 補充練習 4

**理論題：** Gradient Boosting 的 `learning_rate` 和 `n_estimators` 之間的關係是什麼？為什麼「小學習率 + 多棵樹」通常優於「大學習率 + 少棵樹」？

**實作題：** 在 `make_regression` 的資料上，分別訓練 `n_estimators=10/50/200/500`（固定 `learning_rate=0.1`）的 GBM，繪製訓練和驗證損失隨 `n_estimators` 的變化曲線，找到最佳的樹的數量。

---

## 5. Stacking 集成

### 理論背景

**Stacking (Stacked Generalization)**：用一個**元學習器 (Meta-learner)** 來組合基學習器的預測：

1. 訓練集切為 $k$ 個 fold（類似 K-fold CV）
2. 每個基學習器在 $k-1$ 個 fold 上訓練，對剩餘的 1 個 fold 做 OOF（Out-of-Fold）預測
3. 將所有 OOF 預測拼接，作為元學習器的訓練集
4. 元學習器學習「如何組合基學習器的輸出」

Stacking 比固定權重的投票更靈活，可以學習到「在什麼情況下信任哪個模型」。

### 核心代碼

```python
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

### ⚡ 補充練習 5

**理論題：** 為什麼 Stacking 的基學習器必須使用 OOF 預測（而非訓練集預測）來訓練元學習器？若用訓練集預測會發生什麼問題？

**實作題：** 在 `make_moons` 上比較 VotingClassifier（軟投票）、BaggingClassifier、RandomForestClassifier、GradientBoostingClassifier 和 StackingClassifier 的測試集準確率，製成比較表。

---

## 結論

集成學習的核心思想：

- **多樣性** + **數量** = 更好的泛化能力
- **Bagging**（隨機森林）：並行訓練，高變異數降低
- **Boosting**（AdaBoost、GBM、XGBoost）：循序訓練，高偏差降低
- **Stacking**：最靈活，元學習器自動調整組合權重

下一章（Ch08）轉向無監督學習的前置工作：降維，解決高維資料的「維度詛咒」。

---

## 課後作業

**作業：集成方法全面比較**

在 MNIST 資料集（取前 5000 筆）上：

1. 分別訓練以下模型，記錄準確率和訓練時間：
   - 單棵 `DecisionTreeClassifier`
   - `RandomForestClassifier(n_estimators=100)`
   - `GradientBoostingClassifier(n_estimators=100, max_depth=3)`
   - `StackingClassifier`（基學習器：DecisionTree + KNN + LogisticRegression）

2. 分析：哪種方法「性價比」最高（準確率提升 / 訓練時間）？

3. 對最佳 `RandomForestClassifier`，取出 `feature_importances_`，找出最重要的前 10 個像素位置，討論這些像素的分布是否合理。
