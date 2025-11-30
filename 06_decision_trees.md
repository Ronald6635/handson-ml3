<!-- meta-title: 決策樹在機器學習中的應用：完整 Python 教學 -->
<!-- meta-description: 學習如何使用 Scikit-Learn 建構決策樹模型，包括訓練、可視化、預測和正則化技巧。包含實用範例和逐步解析，適合初學者到進階開發者。 -->
<!-- meta-keywords: Python, 機器學習, 決策樹, Scikit-Learn, 教學, 程式設計, 資料科學 -->
<!-- meta-hashtags: #Python #機器學習 #決策樹 #ScikitLearn #程式教學 #資料科學 -->

# 🐍 決策樹在機器學習中的應用：完整 Python 教學

學習決策樹（Decision Trees）是機器學習中的基礎演算法之一。本教學將帶您從基礎概念開始，逐步深入到實際應用，使用 Scikit-Learn 建構決策樹模型，並探討其在分類和回歸問題中的應用。

## 📝 本文目錄

- [設定與環境準備](#設定與環境準備)
- [訓練與視覺化決策樹](#訓練與視覺化決策樹)
- [進行預測](#進行預測)
- [估計類別機率](#估計類別機率)
- [正則化超參數](#正則化超參數)
- [回歸問題](#回歸問題)
- [決策樹的敏感性](#決策樹的敏感性)
- [決策樹的高變異性](#決策樹的高變異性)
- [額外內容：存取樹狀結構](#額外內容存取樹狀結構)
- [練習解答](#練習解答)
- [總結與最佳實踐](#總結與最佳實踐)
- [常見問答](#常見問答)

## 🎯 關鍵重點 (Key Takeaways)

- 決策樹是一種強大的監督學習演算法，能處理分類和回歸問題
- Scikit-Learn 的 `DecisionTreeClassifier` 和 `DecisionTreeRegressor` 提供簡單易用的 API
- 正則化技巧如限制樹深度和葉節點數量有助於避免過擬合
- 決策樹對軸向旋轉敏感，可使用 PCA 預處理改善
- 決策樹具有高變異性，容易過擬合訓練資料

## <a id="設定與環境準備"></a>設定與環境準備

在開始之前，我們需要確保 Python 環境正確設定，並匯入必要的程式庫。

### 範例 1: 環境檢查與設定

```python
import sys

# 檢查 Python 版本是否為 3.7 或以上
assert sys.version_info >= (3, 7)

# 匯入並檢查 Scikit-Learn 版本
from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

# 設定 Matplotlib 字體大小
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# 建立圖片儲存目錄
from pathlib import Path

IMAGES_PATH = Path() / "images" / "decision_trees"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

# 定義圖片儲存函數
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 匯入 sys 模組，用於檢查 Python 版本
2. `assert sys.version_info >= (3, 7)`: 確保 Python 版本至少為 3.7
3. `from packaging import version`: 匯入版本比較工具
4. `assert version.parse(sklearn.__version__) >= version.parse("1.0.1")`: 檢查 Scikit-Learn 版本
5. `plt.rc(...)`: 設定 Matplotlib 的預設字體和大小
6. `IMAGES_PATH.mkdir(...)`: 建立圖片儲存目錄
7. `def save_fig(...)`: 定義儲存圖片的輔助函數

**🎯 重點摘要:**

- **核心功能**: 環境設定和圖片儲存工具
- **潛在問題**: 版本檢查失敗會引發 AssertionError
- **最佳使用情境**: 在機器學習專案開始時進行環境驗證

## <a id="訓練與視覺化決策樹"></a>訓練與視覺化決策樹

讓我們使用鳶尾花（Iris）資料集來訓練第一個決策樹模型。

### 範例 2: 載入資料並訓練決策樹

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# 載入鳶尾花資料集
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

# 建立並訓練決策樹分類器
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import load_iris`: 匯入鳶尾花資料集
2. `iris = load_iris(as_frame=True)`: 載入資料並轉為 DataFrame 格式
3. `X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values`: 選取花瓣長度和寬度作為特徵
4. `y_iris = iris.target`: 取得目標標籤
5. `tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)`: 建立決策樹分類器，限制深度為 2
6. `tree_clf.fit(X_iris, y_iris)`: 在訓練資料上擬合模型

**🎯 重點摘要:**

- **核心功能**: 使用鳶尾花資料集訓練決策樹
- **潛在問題**: 資料預處理和特徵選擇很重要
- **最佳使用情境**: 分類問題的基礎模型訓練

### 範例 3: 匯出決策樹圖形

```python
from sklearn.tree import export_graphviz

# 匯出決策樹為 dot 格式
export_graphviz(
        tree_clf,
        out_file=str(IMAGES_PATH / "iris_tree.dot"),
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
```

**✅ 程式碼逐行解析：**

1. `from sklearn.tree import export_graphviz`: 匯入圖形匯出函數
2. `export_graphviz(...)`: 將決策樹匯出為 GraphViz dot 檔案
3. `out_file=...`: 指定輸出檔案路徑
4. `feature_names=...`: 設定特徵名稱
5. `class_names=...`: 設定類別名稱
6. `rounded=True, filled=True`: 設定視覺化樣式

**🎯 重點摘要:**

- **核心功能**: 將決策樹視覺化為圖形檔案
- **潛在問題**: 需要安裝 GraphViz 軟體才能轉換為圖片
- **最佳使用情境**: 模型解釋和教學用途

## <a id="進行預測"></a>進行預測

訓練好模型後，我們可以使用它進行預測。

### 範例 4: 預測新資料

```python
# 使用訓練好的模型進行預測
predictions = tree_clf.predict([[5, 1.5]])
print("預測結果:", predictions)
```

**✅ 程式碼逐行解析：**

1. `tree_clf.predict([[5, 1.5]])`: 使用模型預測新樣本
2. `print(...)`: 輸出預測結果

**🎯 重點摘要:**

- **核心功能**: 使用決策樹進行單一樣本預測
- **潛在問題**: 輸入資料格式必須與訓練時一致
- **最佳使用情境**: 即時預測應用

## <a id="估計類別機率"></a>估計類別機率

決策樹不僅能進行預測，還能估計每個類別的機率。

### 範例 5: 估計類別機率

```python
# 估計類別機率
probabilities = tree_clf.predict_proba([[5, 1.5]])
print("類別機率:", probabilities.round(3))
```

**✅ 程式碼逐行解析：**

1. `tree_clf.predict_proba(...)`: 計算每個類別的預測機率
2. `.round(3)`: 將機率四捨五入到小數點後三位
3. `print(...)`: 輸出機率分佈

**🎯 重點摘要:**

- **核心功能**: 獲取決策樹的機率預測
- **潛在問題**: 機率估計依賴於樹的結構
- **最佳使用情境**: 需要不確定性估計的應用

## <a id="正則化超參數"></a>正則化超參數

為了避免過擬合，我們需要調整決策樹的正則化參數。

### 範例 6: 比較不同正則化設定的效果

```python
from sklearn.datasets import make_moons

# 生成月亮形資料集
X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

# 建立兩個決策樹：一個無限制，一個有限制
tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)

# 訓練模型
tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import make_moons`: 匯入月亮形資料集生成器
2. `X_moons, y_moons = make_moons(...)`: 生成非線性可分的資料
3. `tree_clf1 = DecisionTreeClassifier(random_state=42)`: 無限制的決策樹
4. `tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, ...)`: 限制最小葉節點數量的決策樹
5. `tree_clf1.fit(...)`: 訓練無限制模型
6. `tree_clf2.fit(...)`: 訓練正則化模型

**🎯 重點摘要:**

- **核心功能**: 比較過擬合和正則化模型的差異
- **潛在問題**: 無限制模型容易過擬合複雜資料
- **最佳使用情境**: 調整模型複雜度以平衡偏差和變異

## <a id="回歸問題"></a>回歸問題

決策樹不僅可用於分類，也可用於回歸問題。

### 範例 7: 決策樹回歸

```python
from sklearn.tree import DecisionTreeRegressor

# 生成二次函數資料
np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5
y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)

# 訓練決策樹回歸器
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.tree import DecisionTreeRegressor`: 匯入決策樹回歸器
2. `np.random.seed(42)`: 設定隨機種子確保重現性
3. `X_quad = np.random.rand(200, 1) - 0.5`: 生成輸入特徵
4. `y_quad = X_quad ** 2 + 0.025 * np.random.randn(...)`: 生成二次函數目標值加雜訊
5. `tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)`: 建立回歸樹
6. `tree_reg.fit(...)`: 訓練回歸模型

**🎯 重點摘要:**

- **核心功能**: 使用決策樹進行回歸預測
- **潛在問題**: 回歸樹對離群值敏感
- **最佳使用情境**: 非線性回歸問題

## <a id="決策樹的敏感性"></a>決策樹的敏感性

決策樹對資料的軸向方向很敏感。

### 範例 8: 展示軸向敏感性

```python
# 生成方形資料並旋轉
np.random.seed(6)
X_square = np.random.rand(100, 2) - 0.5
y_square = (X_square[:, 0] > 0).astype(np.int64)

# 旋轉資料
angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
X_rotated_square = X_square.dot(rotation_matrix)

# 訓練兩個模型
tree_clf_square = DecisionTreeClassifier(random_state=42)
tree_clf_square.fit(X_square, y_square)
tree_clf_rotated_square = DecisionTreeClassifier(random_state=42)
tree_clf_rotated_square.fit(X_rotated_square, y_square)
```

**✅ 程式碼逐行解析：**

1. `X_square = np.random.rand(100, 2) - 0.5`: 生成方形分佈的資料
2. `y_square = (X_square[:, 0] > 0).astype(np.int64)`: 基於 x 軸建立二元標籤
3. `rotation_matrix = ...`: 定義旋轉矩陣
4. `X_rotated_square = X_square.dot(rotation_matrix)`: 旋轉資料
5. `tree_clf_square.fit(...)`: 在原始資料上訓練
6. `tree_clf_rotated_square.fit(...)`: 在旋轉資料上訓練

**🎯 重點摘要:**

- **核心功能**: 展示決策樹對軸向的敏感性
- **潛在問題**: 資料旋轉會導致完全不同的決策邊界
- **最佳使用情境**: 理解模型對特徵尺度化的需求

## <a id="決策樹的高變異性"></a>決策樹的高變異性

決策樹具有高變異性，即使在相同資料上也可能產生不同模型。

### 範例 9: 展示高變異性

```python
# 使用不同隨機種子訓練模型
tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked.fit(X_iris, y_iris)
```

**✅ 程式碼逐行解析：**

1. `tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)`: 使用不同隨機種子
2. `tree_clf_tweaked.fit(X_iris, y_iris)`: 在相同資料上訓練

**🎯 重點摘要:**

- **核心功能**: 展示決策樹的隨機性
- **潛在問題**: CART 演算法的隨機性導致模型變異
- **最佳使用情境**: 理解為何需要集成方法如隨機森林

## <a id="額外內容存取樹狀結構"></a>額外內容：存取樹狀結構

可以直接存取決策樹的內部結構。

### 範例 10: 存取樹狀結構

```python
tree = tree_clf.tree_
print("節點數量:", tree.node_count)
print("最大深度:", tree.max_depth)
print("葉節點數量:", tree.n_leaves)
```

**✅ 程式碼逐行解析：**

1. `tree = tree_clf.tree_`: 取得樹的內部結構
2. `tree.node_count`: 總節點數
3. `tree.max_depth`: 最大深度
4. `tree.n_leaves`: 葉節點數

**🎯 重點摘要:**

- **核心功能**: 檢查決策樹的結構屬性
- **潛在問題**: 直接存取內部結構需要了解實作細節
- **最佳使用情境**: 模型分析和除錯

## <a id="總結與最佳實踐"></a>💡 總結與最佳實踐

決策樹是機器學習中的重要工具，具有直觀、可解釋的優點。本教學涵蓋了從基礎訓練到進階應用的完整流程。

**最佳實踐：**

- 使用交叉驗證選擇超參數
- 考慮使用 PCA 處理軸向敏感性
- 對於高變異性問題，考慮使用隨機森林
- 正確處理分類和回歸問題的不同需求

## <a id="常見問答"></a>❓ 常見問答 (FAQ)

**Q: 決策樹是否需要特徵標準化？**

A: 不需要，決策樹對特徵尺度不敏感。

**Q: 如何避免決策樹過擬合？**

A: 限制樹深度、葉節點數量，或使用最小樣本分割數。

**Q: 決策樹適合什麼類型的資料？**

A: 適合處理分類和回歸問題，尤其在需要模型解釋時。

## 🏷️ 推薦標籤

推薦標籤：#Python #機器學習 #決策樹 #ScikitLearn #程式教學 #資料科學 #人工智慧 #演算法
 
 