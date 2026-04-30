<!-- meta-title: 集成學習與隨機森林完整指南：投票、Bagging、Boosting 到 Stacking -->
<!-- meta-description: 深入集成學習的核心概念：投票分類器、Bagging/Pasting、Out-of-Bag 評估、隨機森林、Extra-Trees、特徵重要性、AdaBoost、Gradient Boosting、XGBoost、Stacking。含完整 Scikit-Learn 實戰。 -->
<!-- meta-keywords: Python, 集成學習, 隨機森林, Boosting, AdaBoost, XGBoost, Bagging, Stacking, Scikit-Learn, 機器學習 -->
<!-- meta-hashtags: #Python #集成學習 #隨機森林 #XGBoost #Boosting #機器學習 #ScikitLearn #程式設計 #教學 #DataScience -->

# 🐍 集成學習完整指南：從投票到 Stacking 的群體智慧

集成學習（Ensemble Learning）的核心思想：**眾多弱模型的智慧，勝過單一強模型**。就像台股分析師的委員會比單一分析師更準確——只要每個成員犯的錯不同，集體投票就能超越個人。本教學帶你掌握從簡單投票到複雜 Stacking 的所有主流集成方法。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🗳️ 投票分類器](#voting)
- [📦 Bagging 與 Pasting](#bagging)
- [🌲 隨機森林](#random-forest)
- [💡 特徵重要性](#feature-importance)
- [🚀 Boosting：AdaBoost 與 Gradient Boosting](#boosting)
- [⚡ XGBoost 與 HistGradientBoosting](#xgboost)
- [🔗 Stacking](#stacking)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **Bagging**（有放回取樣）和 **Pasting**（無放回）建立多個模型並行訓練，再投票
- **隨機森林**是 Bagging + 特徵隨機化，預設幾乎不需要調參，是最佳「入手」演算法
- **Boosting（提升）** 串行訓練，每個模型專注糾正前一個的錯誤（AdaBoost / Gradient Boosting）
- **XGBoost** 是實務上最強的 Gradient Boosting 實作，擁有正則化和早停止機制
- **Stacking** 用「元學習器（Meta Learner）」學習如何結合各基礎模型的預測

---

## <a id="voting"></a>🗳️ 投票分類器

💡 **實際應用情境：** 醫療診斷的第二意見制度——不同科別醫師（SVM、Random Forest、Logistic Regression）各自給出診斷，最終以多數意見決定。只要醫師的失誤不相關，集體判斷比個人更可靠。

### 範例 1: 硬投票 vs 軟投票

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 三個多樣化的基礎分類器（多樣性是關鍵！）
log_clf   = LogisticRegression(random_state=42)
rnd_clf   = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf   = SVC(probability=True, random_state=42)  # probability=True 用於軟投票

# 硬投票（Hard Voting）：多數決
hard_voting_clf = VotingClassifier(
    estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
    voting="hard"
)

# 軟投票（Soft Voting）：加權平均機率（通常更好）
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

- **硬投票**簡單但丟失了機率信息；**軟投票**保留機率，通常效果更好
- 集成效果的上限：基礎模型的多樣性和每個模型的個別準確率

---

## <a id="bagging"></a>📦 Bagging 與 Pasting

### 範例 2: BaggingClassifier 與 OOB 評估

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging：有放回取樣（bootstrap=True）
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

1. `bootstrap=True`: 有放回取樣，平均每個樣本被選中的機率 ≈ 63.2%（剩下 36.8% 成為 OOB 樣本）
2. `oob_score=True`: 利用「從未訓練過該樣本」的估計器評估那個樣本，相當於**免費**的交叉驗證
3. `n_jobs=-1`: 使用所有 CPU 核心平行訓練 500 棵樹

**🎯 重點摘要:**

- OOB（Out-of-Bag）評估是 Bagging 獨有的免費驗證機制，與 5-Fold CV 效果相當
- Bagging（有放回）通常比 Pasting（無放回）效果更好，因為取樣多樣性更高

---

## <a id="random-forest"></a>🌲 隨機森林

💡 **實際應用情境：** 台灣金融監理機關使用隨機森林預測金融機構的違規風險——因為它的特徵重要性分析可以告訴監理人員「哪些指標最能預測違規」，而且對異常值不敏感。

### 範例 3: RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# 隨機森林 = Bagging 決策樹 + 特徵隨機化
rnd_clf = RandomForestClassifier(
    n_estimators=500,      # 500 棵樹
    max_leaf_nodes=16,     # 限制葉節點數（降低複雜度）
    max_features="sqrt",   # 每次分裂只考慮 √n 個特徵（引入隨機性）
    n_jobs=-1,
    random_state=42
)
rnd_clf.fit(X_train, y_train)
print(f"隨機森林準確率: {rnd_clf.score(X_test, y_test):.4f}")

# Extra-Trees（極端隨機樹）：更隨機，更快
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

## <a id="feature-importance"></a>💡 特徵重要性

### 範例 4: 分析 MNIST 的像素重要性

```python
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

# 載入 MNIST（用小型子集示範）
mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X_mnist, y_mnist = mnist["data"][:10000] / 255.0, mnist["target"][:10000]

# 訓練隨機森林
rnd_clf_mnist = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rnd_clf_mnist.fit(X_mnist, y_mnist)

# 特徵重要性即為每個像素對分類的貢獻
importances = rnd_clf_mnist.feature_importances_
importance_image = importances.reshape(28, 28)

plt.figure(figsize=(6, 6))
plt.imshow(importance_image, cmap="hot")
plt.colorbar()
plt.title("MNIST 隨機森林像素重要性熱力圖")
plt.axis("off")
plt.show()
# 圖中較亮的像素（中央）對分類更重要，邊緣（黑色）幾乎不重要
```

**✅ 程式碼逐行解析：**

1. `feature_importances_`: Gini 重要性——每個特徵在所有樹中所有分裂降低不純度的加權平均
2. `.reshape(28, 28)`: 將 784 維特徵重塑為 28×28 圖像，便於視覺化

**🎯 重點摘要:**

- 特徵重要性可用於**特徵選擇**（移除重要性低的特徵）和**模型解釋**
- 注意：高相關性特徵的重要性會被稀釋（彼此「搶分」）

---

## <a id="boosting"></a>🚀 Boosting：AdaBoost 與 Gradient Boosting

💡 **實際應用情境：** 銀行反欺詐系統用 Gradient Boosting 逐步學習「欺詐交易的模式」——每個新的決策樹專注糾正前一棵樹犯錯的交易，最終形成對欺詐行為的精細識別能力。

### 範例 5: AdaBoostClassifier

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost：每輪調高被誤分類樣本的權重
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

### 範例 6: GradientBoostingRegressor

```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

np.random.seed(42)
X_reg = np.random.rand(100, 1) - 0.5
y_reg = 3 * X_reg.ravel()**2 + 0.05 * np.random.randn(100)

# Gradient Boosting：每棵樹擬合前一棵樹的「殘差（Residuals）」
gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=120,
    learning_rate=0.1,    # 縮減（Shrinkage）：學習率越小需要越多樹
    subsample=0.25,       # 隨機取 25% 樣本訓練（Stochastic GB）
    random_state=42
)
gbrt.fit(X_reg, y_reg)

# 早停止：找到最佳樹數
errors = [((y_reg - gbrt.staged_predict(X_reg)).__next__())**2 for _ in range(1)]
# 使用 staged_predict 追蹤訓練誤差
```

**🎯 重點摘要:**

- AdaBoost 調整**樣本權重**；Gradient Boosting 擬合**殘差（偽殘差）**
- 兩者的 `learning_rate` 越小，需要越多的 `n_estimators`（可用早停止自動決定）

---

## <a id="xgboost"></a>⚡ XGBoost 與 HistGradientBoosting

### 範例 7: XGBoost 與 HistGradientBoostingClassifier

```python
# HistGradientBoosting（sklearn 內建，速度接近 XGBoost）
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

# XGBoost（需 pip install xgboost）
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

**✅ 程式碼逐行解析：**

1. `HistGradientBoostingClassifier`: sklearn 內建的高效 GBDT，支援缺失值，不需要填補
2. `early_stopping=True`: 當驗證集效能不再改善時自動停止，防止過擬合
3. `colsample_bytree=0.8`: XGBoost 每棵樹隨機取 80% 特徵，類似隨機森林的特徵隨機化

**🎯 重點摘要:**

- XGBoost 是 Kaggle 競賽的常勝將軍，優點：速度快、正則化（L1/L2）、缺失值處理
- HistGradientBoosting 無需安裝額外套件，是 sklearn 的首選 Boosting 實作

---

## <a id="stacking"></a>🔗 Stacking（堆疊集成）

### 範例 8: StackingClassifier

```python
from sklearn.ensemble import StackingClassifier

# 定義基礎學習器（Layer 1）
base_estimators = [
    ("lr",  LogisticRegression(random_state=42)),
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
    ("svm", SVC(probability=True, random_state=42)),
]

# 元學習器（Meta Learner / Layer 2）：學習如何組合基礎模型的輸出
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

- Stacking 理論上最強，但需要更多計算資源和更仔細的設計
- 避免元學習器過於複雜（常用簡單的 LogReg 或 Ridge 作為元學習器）

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: 隨機森林 vs Gradient Boosting 如何選擇？**

A: 隨機森林是並行訓練（快速，調參少），適合快速建立 baseline；Gradient Boosting 是串行訓練（更準確，需要調參），適合追求最高準確率。通常先用 RF 作 baseline，再用 XGBoost 挑戰。

**Q2: Boosting 的 learning_rate 和 n_estimators 如何平衡？**

A: 這是「縮減（Shrinkage）」的核心思想：`learning_rate` 越小（如 0.01），需要越多的 `n_estimators`（如 1000），但泛化效果通常更好。搭配早停止（Early Stopping）使用。

**Q3: 為什麼集成方法效果更好？**

A: 數學依據：若每個分類器的錯誤是**相互獨立**的，集成模型的錯誤率呈指數下降。如 500 個準確率 0.75 的獨立分類器的集成，準確率 > 0.97（大數法則）。

**Q4: Stacking 一定比其他集成方法好嗎？**

A: 不一定。Stacking 的優勢是「讓元學習器自動學習如何結合各模型」，但也帶來過擬合風險和計算成本。實際上 XGBoost 的效果往往就已經很好了。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #集成學習 #隨機森林 #XGBoost #Boosting #機器學習 #ScikitLearn #Stacking #AdaBoost #程式設計 #教學 #DataScience #MachineLearning #特徵重要性
