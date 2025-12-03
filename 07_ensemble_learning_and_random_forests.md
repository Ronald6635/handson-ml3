<!-- meta-title: 第7章 – 集成學習與隨機森林 -->
<!-- meta-description: 本章探討集成學習技術，結合多個機器學習模型以提升預測效能與穩健性。我們將涵蓋投票分類器、裝袋與粘貼、隨機森林、提升方法如AdaBoost與梯度提升，以及堆疊。這些方法利用群眾智慧來減少過擬合並提升準確度。 -->
<!-- meta-keywords: 機器學習, 集成學習, 隨機森林, 投票分類器, 裝袋, 提升, 堆疊, Python, Scikit-Learn -->
<!-- meta-hashtags: #Python #機器學習 #集成學習 #隨機森林 #教學 #編程 #開發 #技術分享 #學習筆記 -->

# 🐍 第7章 – 集成學習與隨機森林

本章探討集成學習技術，結合多個機器學習模型以提升預測效能與穩健性。我們將涵蓋投票分類器、裝袋與粘貼、隨機森林、提升方法如AdaBoost與梯度提升，以及堆疊。這些方法利用群眾智慧來減少過擬合並提升準確度。

_此筆記本包含第7章所有範例程式碼與練習解答。_

## 📝 本文目錄
- [設定](#設定)
- [投票分類器](#投票分類器)
- [裝袋與粘貼](#裝袋與粘貼)
- [隨機森林](#隨機森林)
- [提升](#提升)
- [堆疊](#堆疊)
- [練習解答](#練習解答)

## 🎯 關鍵重點 (Key Takeaways)
- 集成學習結合多個模型以提升效能與穩健性。
- 投票分類器使用硬投票或軟投票來聚合預測。
- 裝袋與粘貼通過隨機子集訓練減少過擬合。
- 隨機森林是裝袋決策樹的擴展，具有特徵重要性。
- 提升方法如AdaBoost與梯度提升依次修正錯誤。
- 堆疊使用元學習器組合多個模型的預測。

## <a id="設定"></a>設定

此專案需要Python 3.7或以上：

```python
import sys

assert sys.version_info >= (3, 7)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 匯入sys模組以檢查Python版本。
2. `assert sys.version_info >= (3, 7)`: 確保Python版本至少為3.7，否則引發AssertionError。

**🎯 重點摘要:**

- **核心功能**: 版本檢查確保相容性。
- **潛在問題**: 舊版Python可能導致相容性問題。
- **最佳使用情境**: 在腳本開始處進行環境驗證。

它也需要Scikit-Learn ≥ 1.0.1：

```python
from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
```

**✅ 程式碼逐行解析：**

1. `from packaging import version`: 匯入version模組以比較版本。
2. `import sklearn`: 匯入sklearn模組。
3. `assert version.parse(sklearn.__version__) >= version.parse("1.0.1")`: 確保sklearn版本至少為1.0.1。

**🎯 重點摘要:**

- **核心功能**: 檢查sklearn版本。
- **潛在問題**: 舊版可能缺少功能。
- **最佳使用情境**: 確保依賴項版本正確。

讓我們定義預設字體大小以使圖表更美觀：

```python
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

1. `import matplotlib.pyplot as plt`: 匯入matplotlib.pyplot。
2. `plt.rc('font', size=14)`: 設定預設字體大小為14。
3. `plt.rc('axes', labelsize=14, titlesize=14)`: 設定軸標籤和標題字體大小。
4. `plt.rc('legend', fontsize=14)`: 設定圖例字體大小。
5. `plt.rc('xtick', labelsize=10)`: 設定x軸刻度字體大小。
6. `plt.rc('ytick', labelsize=10)`: 設定y軸刻度字體大小。

**🎯 重點摘要:**

- **核心功能**: 統一圖表字體大小。
- **潛在問題**: 可能影響圖表可讀性。
- **最佳使用情境**: 製作出版品質圖表。

並建立`images/ensembles`資料夾（如果不存在），並定義`save_fig()`函數在本筆記本中使用以將圖表以高解析度儲存：

```python
from pathlib import Path

IMAGES_PATH = Path() / "images" / "ensembles"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `from pathlib import Path`: 匯入Path類別。
2. `IMAGES_PATH = Path() / "images" / "ensembles"`: 定義圖片儲存路徑。
3. `IMAGES_PATH.mkdir(parents=True, exist_ok=True)`: 建立目錄如果不存在。
4. `def save_fig(...)`: 定義儲存圖表函數。
5. `path = IMAGES_PATH / f"{fig_id}.{fig_extension}"`: 建構檔案路徑。
6. `if tight_layout: plt.tight_layout()`: 調整佈局。
7. `plt.savefig(...)`: 儲存圖表。

**🎯 重點摘要:**

- **核心功能**: 自動建立目錄並儲存圖表。
- **潛在問題**: 檔案覆蓋。
- **最佳使用情境**: 批次儲存圖表。

## <a id="投票分類器"></a>投票分類器

投票分類器是集成方法，結合多個個別分類器的預測以提升整體準確度和穩健性。通過聚合不同模型（如邏輯回歸、隨機森林和支援向量機）的輸出，它們利用「群眾智慧」來減少過擬合並提升未見資料的效能。本節示範如何在Scikit-Learn中實作硬投票和軟投票。

為了說明大數法則，我們可以模擬多個序列的偏斜硬幣投擲，並繪製每個序列的運行頭部比例。

```python
# extra code – this cell generates and saves Figure 7–3

import matplotlib.pyplot as plt
import numpy as np

heads_proba = 0.51
np.random.seed(42)
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
cumulative_heads = coin_tosses.cumsum(axis=0)
cumulative_heads_ratio = cumulative_heads / np.arange(1, 10001).reshape(-1, 1)

plt.figure(figsize=(8, 3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
plt.grid()
save_fig("law_of_large_numbers_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `heads_proba = 0.51`: 設定頭部機率為0.51。
2. `np.random.seed(42)`: 設定隨機種子。
3. `coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)`: 生成投擲結果。
4. `cumulative_heads = coin_tosses.cumsum(axis=0)`: 計算累積頭部數。
5. `cumulative_heads_ratio = cumulative_heads / np.arange(1, 10001).reshape(-1, 1)`: 計算運行比例。
6. `plt.figure(figsize=(8, 3.5))`: 建立圖表。
7. `plt.plot(cumulative_heads_ratio)`: 繪製運行比例。
8. `plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")`: 繪製51%參考線。
9. `plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")`: 繪製50%參考線。
10. `plt.xlabel("Number of coin tosses")`: 設定x軸標籤。
11. `plt.ylabel("Heads ratio")`: 設定y軸標籤。
12. `plt.legend(loc="lower right")`: 設定圖例。
13. `plt.axis([0, 10000, 0.42, 0.58])`: 設定軸範圍。
14. `plt.grid()`: 顯示網格。
15. `save_fig("law_of_large_numbers_plot")`: 儲存圖表。
16. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 模擬大數法則。
- **潛在問題**: 隨機性可能導致變異。
- **最佳使用情境**: 示範統計收斂。

讓我們建立一個投票分類器：

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import make_moons`: 匯入資料集生成器。
2. `from sklearn.ensemble import RandomForestClassifier, VotingClassifier`: 匯入分類器。
3. `from sklearn.linear_model import LogisticRegression`: 匯入邏輯回歸。
4. `from sklearn.model_selection import train_test_split`: 匯入資料分割。
5. `from sklearn.svm import SVC`: 匯入支援向量機。
6. `X, y = make_moons(n_samples=500, noise=0.30, random_state=42)`: 生成資料。
7. `X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)`: 分割資料。
8. `voting_clf = VotingClassifier(...)`: 建立投票分類器。
9. `voting_clf.fit(X_train, y_train)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 建立並訓練投票分類器。
- **潛在問題**: 模型可能過擬合。
- **最佳使用情境**: 結合多個分類器。

```python
print(voting_clf.estimators_) # list of all base classifiers
print(voting_clf.named_estimators_) # dictionary of all base classifiers with their names
print(voting_clf.named_estimators_.items()) # items view of the dictionary
```

**✅ 程式碼逐行解析：**

1. `print(voting_clf.estimators_)`: 列印基礎分類器列表。
2. `print(voting_clf.named_estimators_)`: 列印具名分類器字典。
3. `print(voting_clf.named_estimators_.items())`: 列印字典項目。

**🎯 重點摘要:**

- **核心功能**: 檢查分類器結構。
- **潛在問題**: 無。
- **最佳使用情境**: 除錯與檢查。

```python
# Evaluate each base classifier
for name, clf in voting_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test, y_test))
```

**✅ 程式碼逐行解析：**

1. `for name, clf in voting_clf.named_estimators_.items()`: 遍歷具名分類器。
2. `print(name, "=", clf.score(X_test, y_test))`: 列印每個分類器的準確度。

**🎯 重點摘要:**

- **核心功能**: 評估個別分類器效能。
- **潛在問題**: 測試資料洩漏。
- **最佳使用情境**: 比較模型效能。

```python
X_test[:1] # the first instance in the test set
```

**✅ 程式碼逐行解析：**

1. `X_test[:1]`: 取得測試集第一個樣本。

**🎯 重點摘要:**

- **核心功能**: 檢查樣本資料。
- **潛在問題**: 無。
- **最佳使用情境**: 資料檢查。

```python
voting_clf.predict(X_test[:1]) # the ensemble's prediction for that instance
```

**✅ 程式碼逐行解析：**

1. `voting_clf.predict(X_test[:1])`: 預測第一個樣本。

**🎯 重點摘要:**

- **核心功能**: 集成預測。
- **潛在問題**: 單一樣本預測。
- **最佳使用情境**: 測試預測。

使用個別分類器對測試集進行預測：

```python
[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]
```

**✅ 程式碼逐行解析：**

1. `[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]`: 每個基礎分類器的預測。

**🎯 重點摘要:**

- **核心功能**: 比較個別預測。
- **潛在問題**: 無。
- **最佳使用情境**: 理解投票機制。

```python
voting_clf.score(X_test, y_test)
```

**✅ 程式碼逐行解析：**

1. `voting_clf.score(X_test, y_test)`: 計算集成準確度。

**🎯 重點摘要:**

- **核心功能**: 評估集成效能。
- **潛在問題**: 過擬合。
- **最佳使用情境**: 模型評估。

現在讓我們使用軟投票：

```python
voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
```

**✅ 程式碼逐行解析：**

1. `voting_clf.voting = "soft"`: 設定為軟投票。
2. `voting_clf.named_estimators["svc"].probability = True`: 啟用SVM機率預測。
3. `voting_clf.fit(X_train, y_train)`: 重新訓練。
4. `voting_clf.score(X_test, y_test)`: 評估準確度。

**🎯 重點摘要:**

- **核心功能**: 實作軟投票。
- **潛在問題**: SVM機率計算成本高。
- **最佳使用情境**: 需要機率預測時。

```python
voting_clf
```

**✅ 程式碼逐行解析：**

1. `voting_clf`: 顯示分類器物件。

**🎯 重點摘要:**

- **核心功能**: 檢查分類器配置。
- **潛在問題**: 無。
- **最佳使用情境**: 除錯。

## <a id="裝袋與粘貼"></a>裝袋與粘貼

裝袋與粘貼是集成學習技術，通過在訓練資料的隨機子集上訓練相同基礎估計器的多個實例來提升模型穩定性和準確性。裝袋（Bootstrap Aggregating）使用替換抽樣，允許實例重複，而粘貼使用不替換抽樣。這隨機性減少過擬合和變異，使集成比個別模型更穩健。在Scikit-Learn中，這些方法通過`BaggingClassifier`和`BaggingRegressor`類實作。

## Scikit-Learn中的裝袋與粘貼

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import BaggingClassifier`: 匯入裝袋分類器。
2. `from sklearn.tree import DecisionTreeClassifier`: 匯入決策樹。
3. `bag_clf = BaggingClassifier(...)`: 建立裝袋分類器。
4. `bag_clf.fit(X_train, y_train)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 實作裝袋集成。
- **潛在問題**: 計算成本高。
- **最佳使用情境**: 減少過擬合。

```python
# extra code – this cell generates and saves Figure 7–5

def plot_decision_boundary(clf, X, y, alpha=1.0):
    """Plot the decision boundary of a classifier.

    This function visualizes the decision boundary of a given classifier by
    creating a meshgrid over the feature space, predicting class labels for
    each point, and plotting filled contours and scatter points for the data.

    Args:
        clf: A trained classifier with a predict method.
        X: array-like of shape (n_samples, 2), feature matrix.
        y: array-like of shape (n_samples,), target labels.
        alpha: float, optional, default=1.0
            Transparency level for the contour plots.
    """
    axes=[-1.5, 2.4, -1, 1.5]
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    
    plt.contourf(x1, x2, y_pred, alpha=0.3 * alpha, cmap='Wistia')
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8 * alpha)
    colors = ["#78785c", "#c47b27"]
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf, X_train, y_train)
plt.title("Decision Tree")
plt.sca(axes[1])
plot_decision_boundary(bag_clf, X_train, y_train)
plt.title("Decision Trees with Bagging")
plt.ylabel("")
save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `def plot_decision_boundary(...)`: 定義繪圖函數。
2. `axes=[-1.5, 2.4, -1, 1.5]`: 設定軸範圍。
3. `x1, x2 = np.meshgrid(...)`: 建立網格。
4. `X_new = np.c_[x1.ravel(), x2.ravel()]`: 建立預測點。
5. `y_pred = clf.predict(X_new).reshape(x1.shape)`: 預測。
6. `plt.contourf(...)`: 繪製填充輪廓。
7. `plt.contour(...)`: 繪製輪廓線。
8. `for idx in (0, 1): plt.plot(...)`: 繪製資料點。
9. `plt.axis(axes)`: 設定軸。
10. `plt.xlabel(r"$x_1$")`: 設定x軸標籤。
11. `plt.ylabel(r"$x_2$", rotation=0)`: 設定y軸標籤。
12. `tree_clf = DecisionTreeClassifier(random_state=42)`: 建立單一決策樹。
13. `tree_clf.fit(X_train, y_train)`: 訓練決策樹。
14. `fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)`: 建立子圖。
15. `plt.sca(axes[0])`: 選擇第一個子圖。
16. `plot_decision_boundary(tree_clf, X_train, y_train)`: 繪製決策樹邊界。
17. `plt.title("Decision Tree")`: 設定標題。
18. `plt.sca(axes[1])`: 選擇第二個子圖。
19. `plot_decision_boundary(bag_clf, X_train, y_train)`: 繪製裝袋邊界。
20. `plt.title("Decision Trees with Bagging")`: 設定標題。
21. `plt.ylabel("")`: 清除y軸標籤。
22. `save_fig("decision_tree_without_and_with_bagging_plot")`: 儲存圖表。
23. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 可視化決策邊界比較。
- **潛在問題**: 計算密集。
- **最佳使用情境**: 比較模型差異。

## 袋外評估

袋外（Out-of-Bag）評估是裝袋集成中的技術，用於在沒有單獨驗證集的情況下評估模型效能。通過利用每個引導樣本中留出的實例，它提供對泛化誤差的無偏估計。

```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            oob_score=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_
```

**✅ 程式碼逐行解析：**

1. `bag_clf = BaggingClassifier(..., oob_score=True, ...)`: 啟用OOB評估。
2. `bag_clf.fit(X_train, y_train)`: 訓練分類器。
3. `bag_clf.oob_score_`: 取得OOB分數。

**🎯 重點摘要:**

- **核心功能**: 無需額外驗證集的評估。
- **潛在問題**: 僅適用於裝袋。
- **最佳使用情境**: 資源有限時。

```python
bag_clf.oob_decision_function_[:3]  # probas for the first 3 instances
```

**✅ 程式碼逐行解析：**

1. `bag_clf.oob_decision_function_[:3]`: 取得前3個樣本的OOB決策函數。

**🎯 重點摘要:**

- **核心功能**: OOB機率估計。
- **潛在問題**: 無。
- **最佳使用情境**: 檢查預測信心。

然後，我們可以評估OOB準確度：

```python
from sklearn.metrics import accuracy_score

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.metrics import accuracy_score`: 匯入準確度函數。
2. `y_pred = bag_clf.predict(X_test)`: 預測測試集。
3. `accuracy_score(y_test, y_pred)`: 計算準確度。

**🎯 重點摘要:**

- **核心功能**: 評估測試準確度。
- **潛在問題**: 過擬合。
- **最佳使用情境**: 最終模型評估。

## <a id="隨機森林"></a>隨機森林

隨機森林是集成學習技術，通過建構多個決策樹並組合其預測來建立更穩健和準確的模型。通過在特徵選擇和引導樣本中引入隨機性，它們減少過擬合並提升泛化優於單一決策樹。本節示範如何在Scikit-Learn中實作和使用隨機森林。

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import RandomForestClassifier`: 匯入隨機森林。
2. `rnd_clf = RandomForestClassifier(...)`: 建立分類器。
3. `rnd_clf.fit(X_train, y_train)`: 訓練分類器。
4. `y_pred_rf = rnd_clf.predict(X_test)`: 預測測試集。

**🎯 重點摘要:**

- **核心功能**: 實作隨機森林。
- **潛在問題**: 計算成本。
- **最佳使用情境**: 高維資料分類。

隨機森林等同於裝袋決策樹：

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)
```

**✅ 程式碼逐行解析：**

1. `bag_clf = BaggingClassifier(...)`: 建立等效裝袋分類器。
2. `DecisionTreeClassifier(max_features="sqrt", ...)`: 限制特徵數量。

**🎯 重點摘要:**

- **核心功能**: 模擬隨機森林行為。
- **潛在問題**: 手動配置。
- **最佳使用情境**: 理解隨機森林機制。

```python
# extra code – verifies that the predictions are identical
bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)
np.all(y_pred_bag == y_pred_rf)  # same predictions
```

**✅ 程式碼逐行解析：**

1. `bag_clf.fit(X_train, y_train)`: 訓練裝袋分類器。
2. `y_pred_bag = bag_clf.predict(X_test)`: 預測。
3. `np.all(y_pred_bag == y_pred_rf)`: 檢查預測是否相同。

**🎯 重點摘要:**

- **核心功能**: 驗證等效性。
- **潛在問題**: 無。
- **最佳使用情境**: 測試配置。

## 特徵重要性

機器學習模型中的特徵重要性，特別是在集成方法如隨機森林中，量化每個特徵對模型預測的貢獻程度。通過分析特徵如何在決策樹中減少雜質（如基尼雜質），我們可以識別最有影響力的變數，有助於特徵選擇、模型解釋和資料理解。本節示範如何使用Scikit-Learn計算和可視化特徵重要性。

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(f'importance: {round(score, 2)}, feature: {name}')
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import load_iris`: 匯入鳶尾花資料集。
2. `iris = load_iris(as_frame=True)`: 載入資料。
3. `rnd_clf = RandomForestClassifier(...)`: 建立隨機森林。
4. `rnd_clf.fit(iris.data, iris.target)`: 訓練分類器。
5. `for score, name in zip(...)`: 遍歷特徵重要性。
6. `print(f'importance: {round(score, 2)}, feature: {name}')`: 列印重要性。

**🎯 重點摘要:**

- **核心功能**: 計算特徵重要性。
- **潛在問題**: 相對而非絕對重要性。
- **最佳使用情境**: 特徵選擇。

隨機森林中的特徵重要性通過測量每個特徵在森林的所有樹中減少雜質的程度來計算。**導致更大雜質減少的特徵被認為更重要。**這提供有價值的見解，了解哪些特徵對模型的預測最有影響，並有助於特徵選擇和理解資料集。

接下來，讓我們可視化特徵重要性

此單元在MNIST影像上訓練隨機森林（平坦化為784特徵），計算特徵重要性，並將其作為28×28熱圖可視化：

- 目的  
    - 顯示隨機森林認為哪些像素（特徵）最重要進行分類。

- 程式碼所做的事  
    - 在MNIST資料上訓練RandomForestClassifier（100棵樹）。  
    - 讀取`rnd_clf.feature_importances_`並重新塑形為28×28以匹配影像佈局。  
        - 每個像素的重要性反映它在所有樹中減少雜質的貢獻程度。
    - 使用`plt.imshow(..., cmap="hot")`顯示重要性熱圖。更亮的像素=更重要。  
    - 新增色條與標籤「Not important」→「Very important」並使用`save_fig()`儲存高解析度影像。

- 解釋  
    - 更高強度的區域表示對模型決策貢獻最大的像素（例如，數字的筆畫和邊緣）。這有助於理解模型學到了什麼以及它在區分數字時關注的位置。

```python
# extra code – this cell generates and saves Figure 7–6

from sklearn.datasets import fetch_openml

X_mnist, y_mnist = fetch_openml('mnist_784', return_X_y=True, as_frame=False,
                                parser='auto')

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(X_mnist, y_mnist)

heatmap_image = rnd_clf.feature_importances_.reshape(28, 28)
plt.imshow(heatmap_image, cmap="hot")
cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(),
                           rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'], fontsize=14)
plt.axis("off")
save_fig("mnist_feature_importance_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import fetch_openml`: 匯入MNIST資料集。
2. `X_mnist, y_mnist = fetch_openml(...)`: 載入資料。
3. `rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)`: 建立隨機森林。
4. `rnd_clf.fit(X_mnist, y_mnist)`: 訓練分類器。
5. `heatmap_image = rnd_clf.feature_importances_.reshape(28, 28)`: 重塑重要性為影像。
6. `plt.imshow(heatmap_image, cmap="hot")`: 顯示熱圖。
7. `cbar = plt.colorbar(...)`: 新增色條。
8. `cbar.ax.set_yticklabels(['Not important', 'Very important'], fontsize=14)`: 設定色條標籤。
9. `plt.axis("off")`: 隱藏軸。
10. `save_fig("mnist_feature_importance_plot")`: 儲存圖表。
11. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 可視化像素重要性。
- **潛在問題**: 高維資料。
- **最佳使用情境**: 影像特徵分析。

## <a id="提升"></a>提升

提升是集成學習技術，依次訓練弱學習器，其中每個後續模型關注前一個的錯誤。這方法通常導致強預測效能，因為它結合簡單模型成為強集成。常見提升演算法包括AdaBoost和梯度提升。

## AdaBoost

**AdaBoost（Adaptive Boosting）**是集成學習技術，結合多個弱分類器以建立強分類器。它通過依次訓練弱學習器運作，其中每個學習器*更關注前一個誤分類的實例*。最終預測通過結合所有弱學習器的加權輸出進行，導致提升準確度和穩健性。

```python
# extra code – this cell generates and saves Figure 7–8

m = len(X_train)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
for subplot, learning_rate in ((0, 1), (1, 0.5)):
    sample_weights = np.ones(m) / m # initialize sample weights
    plt.sca(axes[subplot])
    for i in range(5): # 5 boosting rounds
        # train weak classifier with current sample weights
        svm_clf = SVC(C=0.2, gamma=0.6, random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
        y_pred = svm_clf.predict(X_train)

        # compute weighted error rate
        error_weights = sample_weights[y_pred != y_train].sum()
        r = error_weights / sample_weights.sum()  # equation 7-1
        # compute classifier weight
        alpha = learning_rate * np.log((1 - r) / r)  # equation 7-2
        # update sample weights
        sample_weights[y_pred != y_train] *= np.exp(alpha)  # equation 7-3
        sample_weights /= sample_weights.sum()  # normalization step

        plot_decision_boundary(svm_clf, X_train, y_train, alpha=0.4)
        plt.title(f"learning_rate = {learning_rate}")
    if subplot == 0:
        plt.text(-0.75, -0.95, "1", fontsize=16)
        plt.text(-1.05, -0.95, "2", fontsize=16)
        plt.text(1.0, -0.95, "3", fontsize=16)
        plt.text(-1.45, -0.5, "4", fontsize=16)
        plt.text(1.36,  -0.95, "5", fontsize=16)
    else:
        plt.ylabel("")

save_fig("boosting_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `m = len(X_train)`: 取得訓練樣本數。
2. `fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)`: 建立子圖。
3. `for subplot, learning_rate in ((0, 1), (1, 0.5))`: 遍歷學習率。
4. `sample_weights = np.ones(m) / m`: 初始化樣本權重。
5. `plt.sca(axes[subplot])`: 選擇子圖。
6. `for i in range(5)`: 5輪提升。
7. `svm_clf = SVC(...)`: 建立SVM分類器。
8. `svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)`: 訓練SVM。
9. `y_pred = svm_clf.predict(X_train)`: 預測訓練集。
10. `error_weights = sample_weights[y_pred != y_train].sum()`: 計算加權錯誤。
11. `r = error_weights / sample_weights.sum()`: 計算錯誤率。
12. `alpha = learning_rate * np.log((1 - r) / r)`: 計算分類器權重。
13. `sample_weights[y_pred != y_train] *= np.exp(alpha)`: 更新樣本權重。
14. `sample_weights /= sample_weights.sum()`: 正規化權重。
15. `plot_decision_boundary(svm_clf, X_train, y_train, alpha=0.4)`: 繪製決策邊界。
16. `plt.title(f"learning_rate = {learning_rate}")`: 設定標題。
17. `if subplot == 0: plt.text(...)`: 新增文字標籤。
18. `else: plt.ylabel("")`: 清除y軸標籤。
19. `save_fig("boosting_plot")`: 儲存圖表。
20. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 手動實作AdaBoost。
- **潛在問題**: 複雜且容易出錯。
- **最佳使用情境**: 理解AdaBoost機制。

```python
plt.figure()
plt.plot(sample_weights)
plt.title("Sample Weights")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `plt.figure()`: 建立新圖表。
2. `plt.plot(sample_weights)`: 繪製樣本權重。
3. `plt.title("Sample Weights")`: 設定標題。
4. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 可視化權重變化。
- **潛在問題**: 無。
- **最佳使用情境**: 檢查提升過程。

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30,
    learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import AdaBoostClassifier`: 匯入AdaBoost。
2. `ada_clf = AdaBoostClassifier(...)`: 建立AdaBoost分類器。
3. `ada_clf.fit(X_train, y_train)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 使用Scikit-Learn實作AdaBoost。
- **潛在問題**: 過擬合風險。
- **最佳使用情境**: 簡單分類任務。

```python
# extra code – in case you're curious to see what the decision boundary
#              looks like for the AdaBoost classifier
plot_decision_boundary(ada_clf, X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `plot_decision_boundary(ada_clf, X_train, y_train)`: 繪製AdaBoost決策邊界。

**🎯 重點摘要:**

- **核心功能**: 可視化AdaBoost邊界。
- **潛在問題**: 無。
- **最佳使用情境**: 比較模型。

同樣地，我們可以使用SVM作為AdaBoost的基礎估計器：

```python
svm_ada_clf = AdaBoostClassifier(
    SVC(probability=True), n_estimators=30,
    learning_rate=0.5, random_state=42)
svm_ada_clf.fit(X_train, y_train)
plot_decision_boundary(svm_ada_clf, X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `svm_ada_clf = AdaBoostClassifier(SVC(probability=True), ...)`: 使用SVM作為基礎估計器。
2. `svm_ada_clf.fit(X_train, y_train)`: 訓練分類器。
3. `plot_decision_boundary(svm_ada_clf, X_train, y_train)`: 繪製決策邊界。

**🎯 重點摘要:**

- **核心功能**: SVM與AdaBoost結合。
- **潛在問題**: 計算成本高。
- **最佳使用情境**: 非線性分類。

使用SVM作為AdaBoost中的基礎估計器時，重要的是設定`probability=True`以啟用機率估計，這在提升過程中使用。

## 梯度提升

梯度提升是集成學習技術，依次建構模型，其中*每個新模型修正前一個集成的錯誤（殘差）*。通過迭代擬合弱學習器到殘差，它最小化可微損失函數，通常導致高度準確的預測。本節示範手動實作和使用Scikit-Learn的GradientBoostingRegressor。

讓我們建立一個簡單的二次資料集並擬合`DecisionTreeRegressor`：

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5 # 100 instances in [-0.5, 0.5]
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)  # y = 3x² + Gaussian noise

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `import numpy as np`: 匯入NumPy。
2. `from sklearn.tree import DecisionTreeRegressor`: 匯入決策樹回歸。
3. `np.random.seed(42)`: 設定隨機種子。
4. `X = np.random.rand(100, 1) - 0.5`: 生成特徵。
5. `y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)`: 生成目標。
6. `tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)`: 建立回歸器。
7. `tree_reg1.fit(X, y)`: 訓練回歸器。

**🎯 重點摘要:**

- **核心功能**: 建立二次資料集並訓練第一棵樹。
- **潛在問題**: 資料簡單。
- **最佳使用情境**: 示範梯度提升。

現在讓我們訓練另一個決策樹回歸器在前一個預測器的殘差錯誤上：

```python
y2 = y - tree_reg1.predict(X) # residuals
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)
```

**✅ 程式碼逐行解析：**

1. `y2 = y - tree_reg1.predict(X)`: 計算殘差。
2. `tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)`: 建立第二棵樹。
3. `tree_reg2.fit(X, y2)`: 訓練第二棵樹。

**🎯 重點摘要:**

- **核心功能**: 訓練第二棵樹於殘差上。
- **潛在問題**: 無。
- **最佳使用情境**: 梯度提升步驟。

```python
y3 = y2 - tree_reg2.predict(X) # residuals
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)
```

**✅ 程式碼逐行解析：**

1. `y3 = y2 - tree_reg2.predict(X)`: 計算新殘差。
2. `tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)`: 建立第三棵樹。
3. `tree_reg3.fit(X, y3)`: 訓練第三棵樹。

**🎯 重點摘要:**

- **核心功能**: 繼續梯度提升過程。
- **潛在問題**: 無。
- **最佳使用情境**: 多輪提升。

```python
X_new = np.array([[-0.4], [0.], [0.5]])
sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

**✅ 程式碼逐行解析：**

1. `X_new = np.array([[-0.4], [0.], [0.5]])`: 建立新樣本。
2. `sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))`: 總和所有樹的預測。

**🎯 重點摘要:**

- **核心功能**: 集成預測。
- **潛在問題**: 無。
- **最佳使用情境**: 測試集成。

```python
y_new = 3 * X_new ** 2
```

**✅ 程式碼逐行解析：**

1. `y_new = 3 * X_new ** 2`: 計算真實值。

**🎯 重點摘要:**

- **核心功能**: 計算真實二次函數值。
- **潛在問題**: 無。
- **最佳使用情境**: 比較預測與真實值。

```python
# extra code – this cell generates and saves Figure 7–9

def plot_predictions(regressors, X, y, axes, style,
                     label=None, data_style="b.", data_label=None):
    """Plot predictions of an ensemble of regressors along with the data.
    Args:
        regressors: list of regressors in the ensemble
        X: array-like of shape (n_samples, n_features), feature matrix
        y: array-like of shape (n_samples,), target values
        axes: list of four floats, [x_min, x_max, y_min, y_max] for the plot
        style: string, style for the prediction line
        label: string, label for the prediction line
        data_style: string, style for the data points
        data_label: string, label for the data points
    """
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1))
                 for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center")
    plt.axis(axes)

plt.figure(figsize=(11, 11))

plt.subplot(3, 2, 1)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], style="g-",
                 label="$h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$  ", rotation=0)
plt.title("Residuals and tree predictions")

plt.subplot(3, 2, 2)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                 label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.title("Ensemble predictions")

plt.subplot(3, 2, 3)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                 label="$h_2(x_1)$", data_style="k+",
                 data_label="Residuals: $y - h_1(x_1)$")
plt.ylabel("$y$  ", rotation=0)

plt.subplot(3, 2, 4)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.2, 0.8],
                  style="r-", label="$h(x_1) = h_1(x_1) + h_2(x_1)$")

plt.subplot(3, 2, 5)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                 label="$h_3(x_1)$", data_style="k+",
                 data_label="Residuals: $y - h_1(x_1) - h_2(x_1)$")
plt.xlabel("$x_1$")
plt.ylabel("$y$  ", rotation=0)

plt.subplot(3, 2, 6)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y,
                 axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                 label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$")

save_fig("gradient_boosting_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `def plot_predictions(...)`: 定義繪圖函數。
2. `plt.figure(figsize=(11, 11))`: 建立大圖表。
3. `plt.subplot(3, 2, 1)`: 選擇第一個子圖。
4. `plot_predictions([tree_reg1], X, y, ...)`: 繪製第一棵樹。
5. `plt.ylabel("$y$  ", rotation=0)`: 設定y軸標籤。
6. `plt.title("Residuals and tree predictions")`: 設定標題。
7. `plt.subplot(3, 2, 2)`: 選擇第二個子圖。
8. `plot_predictions([tree_reg1], X, y, ...)`: 繪製集成預測。
9. `plt.title("Ensemble predictions")`: 設定標題。
10. `plt.subplot(3, 2, 3)`: 選擇第三個子圖。
11. `plot_predictions([tree_reg2], X, y2, ...)`: 繪製第二棵樹於殘差。
12. `plt.subplot(3, 2, 4)`: 選擇第四個子圖。
13. `plot_predictions([tree_reg1, tree_reg2], X, y, ...)`: 繪製前兩棵樹集成。
14. `plt.subplot(3, 2, 5)`: 選擇第五個子圖。
15. `plot_predictions([tree_reg3], X, y3, ...)`: 繪製第三棵樹於殘差。
16. `plt.xlabel("$x_1$")`: 設定x軸標籤。
17. `plt.subplot(3, 2, 6)`: 選擇第六個子圖。
18. `plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, ...)`: 繪製完整集成。
19. `plt.xlabel("$x_1$")`: 設定x軸標籤。
20. `save_fig("gradient_boosting_plot")`: 儲存圖表。
21. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 可視化梯度提升過程。
- **潛在問題**: 圖表複雜。
- **最佳使用情境**: 教育目的。

### 梯度提升中的殘差錯誤解釋

在梯度提升中，每個新模型被訓練來預測*前一個集成錯誤的殘差（差異）*。這個迭代過程通過專注於先前模型的錯誤來最小化整體錯誤。

#### 範例中的關鍵步驟：
- **資料集建立**：生成二次資料集與`y = 3x² + 高斯雜訊`以模擬非線性關係。
- **第一棵樹（`tree_reg1`）**：擬合原始資料`(X, y)`，擷取初始近似。
- **殘差計算**：`y2 = y - tree_reg1.predict(X)`計算第一棵樹的錯誤（殘差）。
- **第二棵樹（`tree_reg2`）**：訓練於`(X, y2)`以預測這些殘差，提升擬合。
- **第三棵樹（`tree_reg3`）**：進一步精煉通過計算`y3 = y2 - tree_reg2.predict(X)`並擬合於`(X, y3)`。
- **集成預測**：最終預測總和所有樹：`tree_reg1.predict(X_new) + tree_reg2.predict(X_new) + tree_reg3.predict(X_new)`，近似真實二次函數。

#### 圖7–9的解釋：
圖表使用3×2網格可視化梯度提升過程：
- **頂列**：顯示個別樹於殘差的預測（左：`tree_reg1`於原始資料；右：前兩棵樹集成）。
- **中列**：`tree_reg2`於`y2`殘差（左），以及前三棵樹集成（右）。
- **底列**：`tree_reg3`於`y3`殘差（左），以及最終集成預測（右）。
- **觀察**：集成逐步減少錯誤，收斂至真實二次曲線。這示範提升如何從弱學習器建構強回歸器通過迭代修正殘差。

現在讓我們嘗試梯度提升回歸器：

```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                 learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import GradientBoostingRegressor`: 匯入梯度提升回歸。
2. `gbrt = GradientBoostingRegressor(...)`: 建立回歸器。
3. `gbrt.fit(X, y)`: 訓練回歸器。

**🎯 重點摘要:**

- **核心功能**: 使用Scikit-Learn實作梯度提升。
- **潛在問題**: 學習率高可能過擬合。
- **最佳使用情境**: 回歸任務。

梯度提升通過依次新增預測器來建構集成，每個新預測器修正當前集成的錯誤。這個程式碼訓練GradientBoostingRegressor與3棵樹，每棵最大深度2，以及學習率1.0。演算法擬合每個新樹於當前集成的殘差錯誤，逐步改善整體預測。

然後，以下程式碼片段示範另一個Gradient Boosting Regressor模型的實例化和訓練以供比較，具有不同的超參數：

#### 程式碼分解
- **實例化**：建立名為`gbrt_best`的實例，具有調校的超參數以提升效能。
- **訓練**：`gbrt_best.fit(X, y)`在特徵矩陣`X`和目標向量`y`上訓練模型。這個方法迭代建構集成，擬合每個樹於先前預測的殘差（錯誤）。

#### 關鍵參數
- `max_depth=2`：限制每個決策樹深度為2，防止過擬合通過保持樹淺且訓練快（儘管這可能低估複雜資料）。
- `learning_rate=0.05`：控制每個新樹對集成的貢獻；較低率需要更多樹以收斂但通常導致更好泛化。
- `n_estimators=500`：設定提升階段（樹）數量為500，高於先前範例，允許模型學習更複雜模式但增加計算時間。
- `n_iter_no_change=10`：啟用早期停止；如果驗證分數在10個連續迭代中沒有改善，訓練提前停止以節省時間並防止過擬合。
- `random_state=42`：通過植入隨機數生成器確保可重複結果。

#### 運作方式
- 內部使用指定損失函數（預設平方誤差）計算梯度並更新模型。
- 每個新樹訓練於當前集成的殘差，逐步改善預測。

#### 陷阱
- `n_iter_no_change`需要驗證集，自動分割10%訓練資料（透過`validation_fraction=0.1`在類別預設）除非覆蓋，這可能影響小資料集效能。
- 確保資料預處理（例如，處理缺失值或縮放特徵）事先進行，因為`X`或`y`格式不當（例如，非數值型別）可能引發內部驗證錯誤。
- 對於較大資料集（n_samples >= 10,000），考慮更快的`HistGradientBoostingRegressor`變體在類別文件提及。

```python
gbrt_best = GradientBoostingRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=500,
    n_iter_no_change=10, random_state=42)
gbrt_best.fit(X, y)
```

**✅ 程式碼逐行解析：**

1. `gbrt_best = GradientBoostingRegressor(...)`: 建立調校回歸器。
2. `gbrt_best.fit(X, y)`: 訓練回歸器。

**🎯 重點摘要:**

- **核心功能**: 調校梯度提升參數。
- **潛在問題**: 早期停止可能過早停止。
- **最佳使用情境**: 平衡偏差與變異。

```python
gbrt_best.n_estimators_
```

**✅ 程式碼逐行解析：**

1. `gbrt_best.n_estimators_`: 取得實際使用的估計器數量。

**🎯 重點摘要:**

- **核心功能**: 檢查早期停止效果。
- **潛在問題**: 無。
- **最佳使用情境**: 評估訓練過程。

```python
# extra code – this cell generates and saves Figure 7–10

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

plt.sca(axes[0])
plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], style="r-",
                 label="Ensemble predictions")
plt.title(f"learning_rate={gbrt.learning_rate}, "
          f"n_estimators={gbrt.n_estimators_}")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)

plt.sca(axes[1])
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8], style="r-")
plt.title(f"learning_rate={gbrt_best.learning_rate}, "
          f"n_estimators={gbrt_best.n_estimators_}")
plt.xlabel("$x_1$")

save_fig("gbrt_learning_rate_plot")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)`: 建立子圖。
2. `plt.sca(axes[0])`: 選擇第一個子圖。
3. `plot_predictions([gbrt], X, y, ...)`: 繪製第一個模型。
4. `plt.title(f"learning_rate={gbrt.learning_rate}, n_estimators={gbrt.n_estimators_}")`: 設定標題。
5. `plt.xlabel("$x_1$")`: 設定x軸標籤。
6. `plt.ylabel("$y$", rotation=0)`: 設定y軸標籤。
7. `plt.sca(axes[1])`: 選擇第二個子圖。
8. `plot_predictions([gbrt_best], X, y, ...)`: 繪製第二個模型。
9. `plt.title(f"learning_rate={gbrt_best.learning_rate}, n_estimators={gbrt_best.n_estimators_}")`: 設定標題。
10. `plt.xlabel("$x_1$")`: 設定x軸標籤。
11. `save_fig("gbrt_learning_rate_plot")`: 儲存圖表。
12. `plt.show()`: 顯示圖表。

**🎯 重點摘要:**

- **核心功能**: 比較不同學習率的影響。
- **潛在問題**: 無。
- **最佳使用情境**: 超參數調校。

此節示範如何準備加州住房資料集以進行機器學習實驗。程式碼下載資料集如果需要，載入到pandas DataFrame，分割成訓練和測試集，並分離特徵從目標變數（`median_house_value`）。這個資料準備步驟對於使用集成方法評估如裝袋、提升和堆疊等至關重要，以可重複方式在真實世界回歸問題上進行。

```python
# extra code – at least not in this chapter, it's presented in chapter 2

import pandas as pd
from sklearn.model_selection import train_test_split
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing_labels = train_set["median_house_value"]
housing = train_set.drop("median_house_value", axis=1)
```

**✅ 程式碼逐行解析：**

1. `import pandas as pd`: 匯入pandas。
2. `from sklearn.model_selection import train_test_split`: 匯入資料分割。
3. `import tarfile`: 匯入tarfile。
4. `import urllib.request`: 匯入urllib。
5. `def load_housing_data()`: 定義資料載入函數。
6. `tarball_path = Path("datasets/housing.tgz")`: 定義壓縮檔案路徑。
7. `if not tarball_path.is_file()`: 檢查檔案是否存在。
8. `Path("datasets").mkdir(parents=True, exist_ok=True)`: 建立目錄。
9. `urllib.request.urlretrieve(url, tarball_path)`: 下載檔案。
10. `with tarfile.open(tarball_path) as housing_tarball`: 開啟壓縮檔案。
11. `housing_tarball.extractall(path="datasets")`: 解壓縮。
12. `return pd.read_csv(Path("datasets/housing/housing.csv"))`: 載入CSV。
13. `housing = load_housing_data()`: 載入資料。
14. `train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)`: 分割資料。
15. `housing_labels = train_set["median_house_value"]`: 提取標籤。
16. `housing = train_set.drop("median_house_value", axis=1)`: 移除標籤列。

**🎯 重點摘要:**

- **核心功能**: 載入並準備住房資料集。
- **潛在問題**: 網路下載失敗。
- **最佳使用情境**: 資料準備。

```python
housing.head()
```

**✅ 程式碼逐行解析：**

1. `housing.head()`: 顯示資料前幾行。

**🎯 重點摘要:**

- **核心功能**: 檢查資料結構。
- **潛在問題**: 無。
- **最佳使用情境**: 資料探索。

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder 

hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough"),
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)
hgb_reg.fit(housing, housing_labels)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.pipeline import make_pipeline`: 匯入pipeline。
2. `from sklearn.compose import make_column_transformer`: 匯入column transformer。
3. `from sklearn.ensemble import HistGradientBoostingRegressor`: 匯入直方圖梯度提升。
4. `from sklearn.preprocessing import OrdinalEncoder`: 匯入序數編碼器。
5. `hgb_reg = make_pipeline(...)`: 建立pipeline。
6. `hgb_reg.fit(housing, housing_labels)`: 訓練模型。

**🎯 重點摘要:**

- **核心功能**: 建立並訓練直方圖梯度提升回歸器。
- **潛在問題**: 分類特徵處理。
- **最佳使用情境**: 處理混合資料型別。

```python
# extra code – evaluate the RMSE stats for the hgb_reg model

from sklearn.model_selection import cross_val_score

hgb_rmses = -cross_val_score(hgb_reg, housing, housing_labels,
                             scoring="neg_root_mean_squared_error", cv=10)
pd.Series(hgb_rmses).describe()
```

**✅ 程式碼逐行解析：**

1. `from sklearn.model_selection import cross_val_score`: 匯入交叉驗證。
2. `hgb_rmses = -cross_val_score(...)`: 計算RMSE。
3. `pd.Series(hgb_rmses).describe()`: 描述統計。

**🎯 重點摘要:**

- **核心功能**: 評估模型效能。
- **潛在問題**: 負分數處理。
- **最佳使用情境**: 模型驗證。

## 直方圖基礎梯度提升（HistGradientBoostingRegressor）

`HistGradientBoostingRegressor`是梯度提升的高效實作，適用於大型資料集。與傳統梯度提升不同，它將連續特徵離散化為區間（直方圖），顯著加速訓練並減少記憶體使用。這個方法特別適用於具有數萬或更多樣本的資料集。

**關鍵特點：**
- 快速訓練和預測，即使在大型資料集上。
- 原生支援數值和分類特徵。
- 自動處理缺失值。
- 通常以最小調校達成最先進效能。

在Scikit-Learn中，`HistGradientBoostingRegressor`可用作`GradientBoostingRegressor`的即插即用替代品，提供更好可擴充性和效能以進行真實世界機器學習任務。

## <a id="堆疊"></a>堆疊

**堆疊（Stacked Generalization）**是集成學習技術，結合多個模型通過訓練元學習器（混成器）於其預測上。這方法可以提升效能，因為它利用不同模型的優勢並通常優於簡單集成方法如投票或裝袋。本節示範如何在Scikit-Learn中實作堆疊。

```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43), # meta-classifier
    cv=5  # number of cross-validation folds
)
stacking_clf.fit(X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import StackingClassifier`: 匯入堆疊分類器。
2. `stacking_clf = StackingClassifier(...)`: 建立堆疊分類器。
3. `stacking_clf.fit(X_train, y_train)`: 訓練分類器。

**🎯 重點摘要:**

- **核心功能**: 實作堆疊集成。
- **潛在問題**: 計算成本高。
- **最佳使用情境**: 結合多樣模型。

```python
stacking_clf.score(X_test, y_test)
```

**✅ 程式碼逐行解析：**

1. `stacking_clf.score(X_test, y_test)`: 評估堆疊分類器。

**🎯 重點摘要:**

- **核心功能**: 測試準確度。
- **潛在問題**: 過擬合。
- **最佳使用情境**: 最終評估。

```python
for name, clf in stacking_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test, y_test))
```

**✅ 程式碼逐行解析：**

1. `for name, clf in stacking_clf.named_estimators_.items()`: 遍歷基礎估計器。
2. `print(name, "=", clf.score(X_test, y_test))`: 列印各估計器準確度。

**🎯 重點摘要:**

- **核心功能**: 比較個別效能。
- **潛在問題**: 無。
- **最佳使用情境**: 分析貢獻。

```python
stacking_clf.final_estimator_.n_features_in_ # number of features used by the final estimator
```

**✅ 程式碼逐行解析：**

1. `stacking_clf.final_estimator_.n_features_in_`: 取得最終估計器特徵數。

**🎯 重點摘要:**

- **核心功能**: 檢查元學習器輸入。
- **潛在問題**: 無。
- **最佳使用情境**: 除錯。

## <a id="練習解答"></a>練習解答

## 1. 到 7.

1. 如果您訓練了五個不同的模型，每個都達到95%準確度，您可以嘗試將它們結合到投票集成中，這通常會給您更好的結果。它在模型非常不同時效果更好（例如，SVM分類器、決策樹分類器、邏輯回歸分類器等）。如果它們在不同訓練實例上訓練甚至更好（這是裝袋和粘貼集成的全部要點），但如果不是這仍然有效只要模型非常不同。
2. 硬投票分類器只計算集成中每個分類器的投票，並選取獲得最多投票的類別。軟投票分類器計算每個類別的平均估計類別機率，並選取具有最高機率的類別。這給予高信心投票更多權重，通常表現更好，但只在每個分類器能夠估計類別機率時運作（例如，Scikit-Learn中的SVM分類器您必須設定`probability=True`）。
3. 相當可能通過將裝袋集成分散到多個伺服器來加速訓練，因為集成中的每個預測器獨立於其他預測器。同樣適用於粘貼集成和隨機森林，基於相同原因。然而，提升集成中的每個預測器基於前一個預測器建構，所以訓練必然是順序的，您不會通過將訓練分散到多個伺服器獲得任何東西。關於堆疊集成，一層中的所有預測器彼此獨立，所以它們可以在多個伺服器上平行訓練。然而，一層中的預測器只能在前一層的所有預測器訓練後訓練。
4. 使用袋外評估，裝袋集成中的每個預測器使用未訓練它的實例評估（它們被留出）。這使得可能在沒有額外驗證集需求的情況下對集成有相當無偏評估。因此，您有更多實例可用於訓練，您的集成可以稍微表現更好。
5. 在隨機森林中生長樹時，只考慮每個節點分裂的隨機特徵子集。這也適用於極端隨機樹，但它們更進一步：而不是像常規決策樹那樣尋找最佳可能門檻，它們對每個特徵使用隨機門檻。這額外隨機性作為正規化形式：如果隨機森林過擬合訓練資料，極端隨機樹可能表現更好。此外，由於極端隨機樹不尋找最佳可能門檻，它們訓練快得多。然而，它們既不比隨機森林快也不慢用於預測。
6. 如果您的AdaBoost集成低擬合訓練資料，您可以嘗試增加估計器數量或減少基礎估計器的正規化超參數。您也可以稍微增加學習率嘗試。
7. 如果您的梯度提升集成過擬合訓練集，您應該嘗試減少學習率。您可以使用早期停止來找到正確的預測器數量（您可能有太多）。

## 8. 投票分類器

練習：_載入MNIST資料並分割成訓練集、驗證集和測試集（例如，使用50,000個實例訓練、10,000驗證、10,000測試）。_

MNIST資料集之前已載入。資料集已經分割成訓練集（前60,000個實例）和測試集（最後10,000個實例），訓練集已經洗牌。所以我們只需要從新訓練集中取前50,000個實例，下一個10,000驗證，最後10,000測試：

```python
X_train, y_train = X_mnist[:50_000], y_mnist[:50_000]
X_valid, y_valid = X_mnist[50_000:60_000], y_mnist[50_000:60_000]
X_test, y_test = X_mnist[60_000:], y_mnist[60_000:]
```

**✅ 程式碼逐行解析：**

1. `X_train, y_train = X_mnist[:50_000], y_mnist[:50_000]`: 設定訓練集。
2. `X_valid, y_valid = X_mnist[50_000:60_000], y_mnist[50_000:60_000]`: 設定驗證集。
3. `X_test, y_test = X_mnist[60_000:], y_mnist[60_000:]`: 設定測試集。

**🎯 重點摘要:**

- **核心功能**: 分割MNIST資料。
- **潛在問題**: 資料順序。
- **最佳使用情境**: 標準分割。

練習：_然後訓練各種分類器，例如隨機森林分類器、極端隨機樹分類器和SVM。_

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import ExtraTreesClassifier`: 匯入極端隨機樹。
2. `from sklearn.svm import LinearSVC`: 匯入線性SVM。
3. `from sklearn.neural_network import MLPClassifier`: 匯入多層感知器。

**🎯 重點摘要:**

- **核心功能**: 匯入分類器。
- **潛在問題**: 無。
- **最佳使用情境**: 多樣分類器。

注意：`LinearSVC`有`dual`超參數，其預設值將從Scikit-Learn 1.5中的`True`變更為`"auto"`。為了確保此筆記本繼續產生相同輸出，我明確設定它為`True`。請參閱[文件](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)以取得更多詳細資訊。

```python
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, dual=True, random_state=42)
mlp_clf = MLPClassifier(random_state=42)
```

**✅ 程式碼逐行解析：**

1. `random_forest_clf = RandomForestClassifier(...)`: 建立隨機森林。
2. `extra_trees_clf = ExtraTreesClassifier(...)`: 建立極端隨機樹。
3. `svm_clf = LinearSVC(...)`: 建立線性SVM。
4. `mlp_clf = MLPClassifier(...)`: 建立MLP。

**🎯 重點摘要:**

- **核心功能**: 實例化分類器。
- **潛在問題**: 收斂問題。
- **最佳使用情境**: 比較模型。

```python
estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `estimators = [...]`: 建立估計器列表。
2. `for estimator in estimators`: 遍歷估計器。
3. `print("Training the", estimator)`: 列印訓練訊息。
4. `estimator.fit(X_train, y_train)`: 訓練估計器。

**🎯 重點摘要:**

- **核心功能**: 訓練所有分類器。
- **潛在問題**: 時間成本。
- **最佳使用情境**: 批次訓練。

```python
[estimator.score(X_valid, y_valid) for estimator in estimators]
```

**✅ 程式碼逐行解析：**

1. `[estimator.score(X_valid, y_valid) for estimator in estimators]`: 計算驗證準確度。

**🎯 重點摘要:**

- **核心功能**: 評估驗證效能。
- **潛在問題**: 無。
- **最佳使用情境**: 模型選擇。

線性SVM遠遠落後其他分類器。然而，讓我們暫時保留它因為它可能改善投票分類器的效能。

練習：_接下來，嘗試將\[分類器\]結合到優於它們全部的集成，使用硬或軟投票分類器。_

```python
from sklearn.ensemble import VotingClassifier
```

**✅ 程式碼逐行解析：**

1. `from sklearn.ensemble import VotingClassifier`: 匯入投票分類器。

**🎯 重點摘要:**

- **核心功能**: 匯入投票分類器。
- **潛在問題**: 無。
- **最佳使用情境**: 集成建構。

```python
named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]
```

**✅ 程式碼逐行解析：**

1. `named_estimators = [...]`: 建立具名估計器列表。

**🎯 重點摘要:**

- **核心功能**: 準備具名估計器。
- **潛在問題**: 無。
- **最佳使用情境**: 堆疊準備。

```python
voting_clf = VotingClassifier(named_estimators)
```

**✅ 程式碼逐行解析：**

1. `voting_clf = VotingClassifier(named_estimators)`: 建立投票分類器。

**🎯 重點摘要:**

- **核心功能**: 實例化投票分類器。
- **潛在問題**: 無。
- **最佳使用情境**: 集成。

```python
voting_clf.fit(X_train, y_train)
```

**✅ 程式碼逐行解析：**

1. `voting_clf.fit(X_train, y_train)`: 訓練投票分類器。

**🎯 重點摘要:**

- **核心功能**: 訓練集成。
- **潛在問題**: 時間成本。
- **最佳使用情境**: 最終訓練。

```python
voting_clf.score(X_valid, y_valid)
```

**✅ 程式碼逐行解析：**

1. `voting_clf.score(X_valid, y_valid)`: 評估驗證準確度。

**🎯 重點摘要:**

- **核心功能**: 檢查集成效能。
- **潛在問題**: 無。
- **最佳使用情境**: 驗證。

`VotingClassifier`複製了每個分類器，並使用類別索引作為標籤訓練複製品，而不是原始類別名稱。因此，要評估這些複製品我們需要提供類別索引以及。為了將類別轉換為類別索引，我們可以使用`LabelEncoder`：

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_valid_encoded = encoder.fit_transform(y_valid)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.preprocessing import LabelEncoder`: 匯入標籤編碼器。
2. `encoder = LabelEncoder()`: 建立編碼器。
3. `y_valid_encoded = encoder.fit_transform(y_valid)`: 編碼標籤。

**🎯 重點摘要:**

- **核心功能**: 編碼類別標籤。
- **潛在問題**: 順序問題。
- **最佳使用情境**: 分類準備。

然而，在MNIST案例中，更簡單地將類別名稱轉換為整數，因為數字匹配類別id：

```python
y_valid_encoded = y_valid.astype(np.int64)
```

**✅ 程式碼逐行解析：**

1. `y_valid_encoded = y_valid.astype(np.int64)`: 轉換為整數。

**🎯 重點摘要:**

- **核心功能**: 簡單類別轉換。
- **潛在問題**: 無。
- **最佳使用情境**: MNIST特有。

現在讓我們評估分類器複製品：

```python
[estimator.score(X_valid, y_valid_encoded)
 for estimator in voting_clf.estimators_]
```

**✅ 程式碼逐行解析：**

1. `[estimator.score(X_valid, y_valid_encoded) for estimator in voting_clf.estimators_]`: 評估複製品。

**🎯 重點摘要:**

- **核心功能**: 檢查個別效能。
- **潛在問題**: 無。
- **最佳使用情境**: 比較。

讓我們移除SVM看看效能是否改善。它是可能的移除估計器通過設定它為`"drop"`使用`set_params()`像這樣：

```python
voting_clf.set_params(svm_clf="drop")
```

**✅ 程式碼逐行解析：**

1. `voting_clf.set_params(svm_clf="drop")`: 移除SVM。

**🎯 重點摘要:**

- **核心功能**: 動態移除估計器。
- **潛在問題**: 無。
- **最佳使用情境**: 調校集成。

這更新了估計器列表：

```python
voting_clf.estimators
```

**✅ 程式碼逐行解析：**

1. `voting_clf.estimators`: 檢查估計器列表。

**🎯 重點摘要:**

- **核心功能**: 驗證變更。
- **潛在問題**: 無。
- **最佳使用情境**: 除錯。

然而，它沒有更新_訓練_估計器列表：

```python
voting_clf.estimators_
```

**✅ 程式碼逐行解析：**

1. `voting_clf.estimators_`: 檢查訓練估計器。

**🎯 重點摘要:**

- **核心功能**: 檢查訓練狀態。
- **潛在問題**: 不一致。
- **最佳使用情境**: 除錯。

```python
voting_clf.named_estimators_
```

**✅ 程式碼逐行解析：**

1. `voting_clf.named_estimators_`: 檢查具名估計器。

**🎯 重點摘要:**

- **核心功能**: 檢查具名估計器。
- **潛在問題**: 無。
- **最佳使用情境**: 除錯。

所以我們可以重新擬合`VotingClassifier`，或只是從列表移除SVM，在`estimators_`和`named_estimators_`中：

```python
svm_clf_trained = voting_clf.named_estimators_.pop("svm_clf")
voting_clf.estimators_.remove(svm_clf_trained)
```

**✅ 程式碼逐行解析：**

1. `svm_clf_trained = voting_clf.named_estimators_.pop("svm_clf")`: 移除具名估計器。
2. `voting_clf.estimators_.remove(svm_clf_trained)`: 移除估計器。

**🎯 重點摘要:**

- **核心功能**: 手動移除訓練估計器。
- **潛在問題**: 無。
- **最佳使用情境**: 調校。

現在讓我們再次評估`VotingClassifier`：

```python
voting_clf.score(X_valid, y_valid)
```

**✅ 程式碼逐行解析：**

1. `voting_clf.score(X_valid, y_valid)`: 重新評估。

**🎯 重點摘要:**

- **核心功能**: 檢查改善。
- **潛在問題**: 無。
- **最佳使用情境**: 驗證調校。

稍微更好！SVM正在傷害效能。現在讓我們嘗試使用軟投票分類器。我們實際上不需要重新訓練分類器，我們可以只是設定`voting`為`"soft"`：

```python
voting_clf.voting = "soft"
```

**✅ 程式碼逐行解析：**

1. `voting_clf.voting = "soft"`: 切換到軟投票。

**🎯 重點摘要:**

- **核心功能**: 變更投票策略。
- **潛在問題**: 無。
- **最佳使用情境**: 比較策略。

```python
voting_clf.score(X_valid, y_valid)
```

**✅ 程式碼逐行解析：**

1. `voting_clf.score(X_valid, y_valid)`: 評估軟投票。

**🎯 重點摘要:**

- **核心功能**: 比較軟投票效能。
- **潛在問題**: 無。
- **最佳使用情境**: 策略選擇。

不，硬投票在這種情況贏得。

_一旦您找到\[優於個別預測器的集成\]，在測試集上嘗試它。與個別分類器比較，它改善多少？_

```python
voting_clf.voting = "hard"
voting_clf.score(X_test, y_test)
```

**✅ 程式碼逐行解析：**

1. `voting_clf.voting = "hard"`: 切回硬投票。
2. `voting_clf.score(X_test, y_test)`: 測試集評估。

**🎯 重點摘要:**

- **核心功能**: 最終測試評估。
- **潛在問題**: 無。
- **最佳使用情境**: 最終效能。

```python
[estimator.score(X_test, y_test.astype(np.int64))
 for estimator in voting_clf.estimators_]
```

**✅ 程式碼逐行解析：**

1. `[estimator.score(X_test, y_test.astype(np.int64)) for estimator in voting_clf.estimators_]`: 個別測試準確度。

**🎯 重點摘要:**

- **核心功能**: 比較個別與集成。
- **潛在問題**: 無。
- **最佳使用情境**: 效能分析。

投票分類器將最佳模型的錯誤率從約3%減少到2.7%，意味著10%更少錯誤。

## 9. 堆疊集成

練習：_執行前一個練習中的個別分類器以在驗證集上進行預測，並建立新訓練集與所有分類器對影像的預測集合：每個訓練實例是向量，包含來自所有分類器的影像類別預測集合，目標是影像的類別。訓練分類器在此新訓練集上。_

```python
X_valid_predictions = np.empty((len(X_valid), len(estimators)), dtype=object)

for index, estimator in enumerate(estimators):
    X_valid_predictions[:, index] = estimator.predict(X_valid)
```

**✅ 程式碼逐行解析：**

1. `X_valid_predictions = np.empty((len(X_valid), len(estimators)), dtype=object)`: 建立預測陣列。
2. `for index, estimator in enumerate(estimators)`: 遍歷估計器。
3. `X_valid_predictions[:, index] = estimator.predict(X_valid)`: 收集預測。

**🎯 重點摘要:**

- **核心功能**: 建立堆疊訓練資料。
- **潛在問題**: 記憶體使用。
- **最佳使用情境**: 堆疊準備。

```python
X_valid_predictions
```

**✅ 程式碼逐行解析：**

1. `X_valid_predictions`: 顯示預測陣列。

**🎯 重點摘要:**

- **核心功能**: 檢查預測資料。
- **潛在問題**: 無。
- **最佳使用情境**: 驗證。

```python
rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True,
                                            random_state=42)
rnd_forest_blender.fit(X_valid_predictions, y_valid)
```

**✅ 程式碼逐行解析：**

1. `rnd_forest_blender = RandomForestClassifier(...)`: 建立混成器。
2. `rnd_forest_blender.fit(X_valid_predictions, y_valid)`: 訓練混成器。

**🎯 重點摘要:**

- **核心功能**: 訓練元學習器。
- **潛在問題**: 過擬合。
- **最佳使用情境**: 堆疊。

```python
rnd_forest_blender.oob_score_
```

**✅ 程式碼逐行解析：**

1. `rnd_forest_blender.oob_score_`: 取得OOB分數。

**🎯 重點摘要:**

- **核心功能**: 評估混成器。
- **潛在問題**: 無。
- **最佳使用情境**: 驗證。

您可以微調這個混成器或嘗試其他型別混成器（例如，`MLPClassifier`），然後使用交叉驗證選擇最佳一個，一如往常。

練習：_恭喜，您現在訓練了混成器，並與分類器一起形成堆疊集成！現在在測試集上評估集成。對於測試集中的每個影像，使用所有分類器進行預測，然後餵給混成器以取得集成的預測。與您之前訓練的投票分類器比較如何？_

```python
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=object)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
```

**✅ 程式碼逐行解析：**

1. `X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=object)`: 建立測試預測陣列。
2. `for index, estimator in enumerate(estimators)`: 遍歷估計器。
3. `X_test_predictions[:, index] = estimator.predict(X_test)`: 收集測試預測。

**🎯 重點摘要:**

- **核心功能**: 準備測試預測。
- **潛在問題**: 無。
- **最佳使用情境**: 最終評估。

```python
y_pred = rnd_forest_blender.predict(X_test_predictions)
```

**✅ 程式碼逐行解析：**

1. `y_pred = rnd_forest_blender.predict(X_test_predictions)`: 堆疊預測。

**🎯 重點摘要:**

- **核心功能**: 產生堆疊預測。
- **潛在問題**: 無。
- **最佳使用情境**: 測試。

```python
accuracy_score(y_test, y_pred)
```

**✅ 程式碼逐行解析：**

1. `accuracy_score(y_test, y_pred)`: 計算堆疊準確度。

**🎯 重點摘要:**

- **核心功能**: 評估堆疊效能。
- **潛在問題**: 無。
- **最佳使用情境**: 比較。

這個堆疊集成不如我們之前嘗試的自訂堆疊表現得好。

練習：_現在嘗試再次使用`StackingClassifier`代替：您得到更好效能嗎？如果是的，為什麼？_

因為`StackingClassifier`使用K-Fold交叉驗證，我們不需要單獨驗證集，所以讓我們將訓練集和驗證集合併成更大的訓練集：

```python
X_train_full, y_train_full = X_mnist[:60_000], y_mnist[:60_000]
```

**✅ 程式碼逐行解析：**

1. `X_train_full, y_train_full = X_mnist[:60_000], y_mnist[:60_000]`: 合併訓練資料。

**🎯 重點摘要:**

- **核心功能**: 建立完整訓練集。
- **潛在問題**: 無。
- **最佳使用情境**: 交叉驗證堆疊。

現在讓我們建立並訓練完整訓練集上的堆疊分類器：

**警告**：以下單元將花費相當長時間執行（15-30分鐘取決於您的硬體），因為它使用預設5折交叉驗證。對於每個基礎分類器，它將訓練5次在完整訓練集的80%上以進行預測，加上完整訓練集上的一次，最後在預測上訓練最終模型。總共25個模型訓練！

```python
stack_clf = StackingClassifier(named_estimators,
                               final_estimator=rnd_forest_blender)
stack_clf.fit(X_train_full, y_train_full)
```

**✅ 程式碼逐行解析：**

1. `stack_clf = StackingClassifier(named_estimators, final_estimator=rnd_forest_blender)`: 建立堆疊分類器。
2. `stack_clf.fit(X_train_full, y_train_full)`: 訓練堆疊分類器。

**🎯 重點摘要:**

- **核心功能**: 訓練完整堆疊。
- **潛在問題**: 計算密集。
- **最佳使用情境**: 最佳堆疊。

```python
stack_clf.score(X_test, y_test)
```

**✅ 程式碼逐行解析：**

1. `stack_clf.score(X_test, y_test)`: 評估堆疊分類器。

**🎯 重點摘要:**

- **核心功能**: 最終堆疊效能。
- **潛在問題**: 無。
- **最佳使用情境**: 比較。

`StackingClassifier`顯著優於我們之前嘗試的自訂堆疊實作！這主要是兩個原因：

* 因為我們能夠回收驗證集，`StackingClassifier`在更大資料集上訓練。
* 它使用了`predict_proba()`如果可用，或`decision_function()`如果可用，或`predict()`。這給混成器更細緻輸入以運作。

這就是今天的一切，恭喜完成章節和練習！

```python

```

**✅ 程式碼逐行解析：**

1. 空單元。

**🎯 重點摘要:**

- **核心功能**: 結束筆記本。
- **潛在問題**: 無。
- **最佳使用情境**: 完成。

## 💡 總結與最佳實踐

集成學習是機器學習中的強大技術，通過結合多個模型來提升預測效能與穩健性。本章涵蓋了投票分類器、裝袋與粘貼、隨機森林、提升方法和堆疊等關鍵技術。每種方法都有其優勢，適用於不同情境。

## ❓ 常見問答 (FAQ)

**Q: 何時使用裝袋而不是提升？**  
A: 裝袋適合減少變異和過擬合，提升適合處理偏差和逐步改善模型。

**Q: 隨機森林的特徵重要性可靠嗎？**  
A: 是的，但它是相對的，適用於特徵選擇而非絕對重要性。

**Q: 如何處理集成中的過擬合？**  
A: 使用交叉驗證、早期停止和適當的正規化參數。

## 🏷️ 推薦標籤 (Suggested Hashtags)

#Python #機器學習 #集成學習 #隨機森林 #裝袋 #提升 #堆疊 #ScikitLearn #資料科學 #人工智慧