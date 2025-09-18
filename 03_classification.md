<!-- meta-title: 🐍 第三章：分類 (Classification) - Hands-On Machine Learning 實作筆記 -->
<!-- meta-description: 本章深入探討了機器學習中的分類問題，從 MNIST 資料集入手，涵蓋了二元分類器、多類別分類、多標籤分類以及多輸出分類的理論與實作。我們將學習如何評估模型效能，包括準確率、混淆矩陣、精確率、召回率、F1 分數、ROC 曲線與 AUC。 -->
<!-- meta-keywords: Python, 機器學習, 分類, MNIST, Scikit-Learn, SGD, SVM, 隨機森林, 效能評估, ROC曲線, Hands-On Machine Learning -->
<!-- meta-hashtags: #Python #MachineLearning #Classification #ScikitLearn #DataScience #程式設計 #教學 -->

# 🐍 第三章：分類 (Classification)

本章節將帶您深入了解機器學習中的「分類」任務。我們將使用经典的 MNIST 手寫數字資料集，從頭開始建立並評估一個分類模型。您將學習到如何訓練二元分類器、多類別分類器，並探索多標籤與多輸出分類等進階主題。

## 📝 本文目錄

- [設定](#setup)
- [MNIST 資料集](#mnist)
- [訓練一個二元分類器](#binary-classifier)
- [效能衡量](#performance-measures)
- [多類別分類](#multiclass-classification)
- [錯誤分析](#error-analysis)
- [多標籤分類](#multilabel-classification)
- [多輸出分類](#multioutput-classification)
- [練習題解答](#exercise-solutions)
- [總結與最佳實踐](#best-practices)
- [常見問答](#faq)
- [推薦標籤](#hashtags)

## 🎯 關鍵重點 (Key Takeaways)

- **分類器種類**: 了解二元、多類別、多標籤與多輸出分類的區別與應用場景。
- **效能指標**: 學習準確率的陷阱，並掌握精確率、召回率、F1 分數、ROC 曲線與 AUC 等關鍵評估指標。
- **模型選擇**: 透過 `SGDClassifier`、`RandomForestClassifier` 與 `SVC` 等不同模型，了解其在分類任務中的優劣。
- **資料前處理**: 學習如何使用 `StandardScaler` 對資料進行特徵縮放，以提升模型效能。
- **錯誤分析**: 透過混淆矩陣視覺化模型的錯誤，找出改進方向。

---

## <a id="setup"></a>設定

在開始之前，我們需要確認 Python 環境與必要的套件版本。

### 範例 1: 確認 Python 版本

```python
# 載入 sys 模組
import sys

# 斷言 Python 版本是否大於等於 3.7
assert sys.version_info >= (3, 7)
```

**✅ 程式碼逐行解析：**

1. `import sys`: 載入 Python 的 `sys` 模組，它提供了對 Python 解譯器本身的存取。
2. `assert sys.version_info >= (3, 7)`: 使用 `assert` 語句檢查當前的 Python 版本是否為 3.7 或更高。`sys.version_info` 會回傳一個包含主版號、次版號等的元組。如果條件不成立，程式將會中斷並拋出 `AssertionError`。

**🎯 重點摘要:**

- **核心功能**: 確保執行環境符合最低 Python 版本要求，避免因版本不相容導致的錯誤。
- **最佳使用情境**: 在專案或腳本的開頭進行環境檢查，確保程式在預期的環境中運行。

### 範例 2: 確認 Scikit-Learn 版本

```python
# 從 packaging 套件中載入 version
from packaging import version
# 載入 scikit-learn
import sklearn

# 斷言 scikit-learn 版本是否大於等於 1.0.1
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
```

**✅ 程式碼逐行解析：**

1. `from packaging import version`: 從 `packaging` 套件中匯入 `version` 模組，它提供了一個強大的版本號解析與比較工具。
2. `import sklearn`: 載入 `scikit-learn` 函式庫。
3. `assert version.parse(sklearn.__version__) >= version.parse("1.0.1")`: 使用 `version.parse()` 將字串形式的版本號（如 `"1.0.1"`）轉換為可比較的物件，然後斷言當前安裝的 `scikit-learn` 版本是否符合最低要求。

**🎯 重點摘要:**

- **核心功能**: 確保 `scikit-learn` 函式庫的版本足夠新，以支援本筆記中使用的所有功能。
- **潛在問題**: 如果未使用 `version.parse()`，直接比較字串版本號（如 `"1.2"` > `"1.10"`）可能會得到錯誤的結果。

### 範例 3: 設定 Matplotlib 圖表樣式

```python
# 載入 matplotlib.pyplot
import matplotlib.pyplot as plt

# 設定全域字體大小
plt.rc('font', size=14)
# 設定座標軸標籤與標題大小
plt.rc('axes', labelsize=14, titlesize=14)
# 設定圖例字體大小
plt.rc('legend', fontsize=14)
# 設定 x, y 軸刻度標籤大小
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**

1. `import matplotlib.pyplot as plt`: 載入 `matplotlib` 的 `pyplot` 模組，這是繪製圖表的主要介面。
2. `plt.rc(...)`: `rc` 是 "runtime configuration" 的縮寫，用於設定 `matplotlib` の全域參數，讓圖表風格保持一致。這裡我們統一設定了字體、座標軸、圖例等元素的預設大小。

**🎯 重點摘要:**

- **核心功能**: 統一圖表視覺風格，提升可讀性。
- **最佳使用情境**: 在筆記本或專案的開頭設定，確保所有後續生成的圖表都遵循相同的樣式規範。

### 範例 4: 建立圖片儲存路徑與輔助函式

```python
# 載入 pathlib 中的 Path
from pathlib import Path

# 定義圖片儲存路徑
IMAGES_PATH = Path() / "images" / "classification"
# 建立路徑 (如果不存在)
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

# 定義儲存圖片的函式
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

**✅ 程式碼逐行解析：**

1. `from pathlib import Path`: 載入 `pathlib` 模組的 `Path` 物件，它提供了物件導向的檔案系統路徑操作。
2. `IMAGES_PATH = Path() / "images" / "classification"`: 使用 `/` 運算子串接路徑，建立一個指向 `images/classification` 資料夾的 `Path` 物件。
3. `IMAGES_PATH.mkdir(parents=True, exist_ok=True)`: 建立資料夾。`parents=True` 表示如果上層資料夾不存在，也會一併建立；`exist_ok=True` 表示如果資料夾已存在，則不會拋出錯誤。
4. `def save_fig(...)`: 定義一個輔助函式，用於儲存 `matplotlib` 圖表。
5. `plt.tight_layout()`: 自動調整子圖參數，使之緊密排列，避免標籤重疊。
6. `plt.savefig(...)`: 將當前圖表儲存到指定路徑，並設定格式與解析度。

**🎯 重點摘要:**

- **核心功能**: 自動化圖片的儲存流程，並統一管理圖片路徑。
- **最佳使用情境**: 在需要重複儲存多張圖表的筆記本中，定義此類輔助函式可以讓程式碼更簡潔。

---

## <a id="mnist"></a>MNIST 資料集

💡 **實際應用情境：**
MNIST 資料集是機器學習领域的 "Hello, World!"。它包含了 70,000 張手寫數字的灰階圖片（0 到 9），每張圖片大小為 28x28 像素。這個資料集常用於評估與比較各種分類演算法的效能。

### 範例 5: 載入 MNIST 資料集

```python
# 從 sklearn.datasets 載入 fetch_openml
from sklearn.datasets import fetch_openml

# 從 OpenML 抓取 mnist_784 資料集
mnist = fetch_openml('mnist_784', as_frame=False)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.datasets import fetch_openml`: `fetch_openml` 是一個方便的函式，可以直接從 [OpenML.org](https://www.openml.org/) 網站下載公開資料集。
2. `mnist = fetch_openml('mnist_784', as_frame=False)`: 下載名為 `mnist_784` 的資料集。`as_frame=False` 參數表示我們希望資料以 NumPy 陣列的形式回傳，而不是 Pandas DataFrame。回傳的 `mnist` 物件是一個類似字典的 `Bunch` 物件，包含了資料、標籤與描述資訊。

**🎯 重點摘要:**

- **核心功能**: 方便地從網路獲取標準資料集，無需手動下載與解壓縮。
- **潛在問題**: `fetch_openml` 需要網路連線。第一次下載後，Scikit-Learn 會將資料快取在本地，後續載入會變快。

### 範例 6: 探索資料結構

```python
# 將資料與標籤分離
X, y = mnist.data, mnist.target
# 顯示特徵資料
X
```

![image](images/classification/some_digit_plot.png)

**✅ 程式碼逐行解析：**

1. `X, y = mnist.data, mnist.target`: `mnist.data` 包含了所有的圖片資料（特徵），`mnist.target` 則是對應的標籤（0-9 的數字）。我們將它們分別指派給 `X` 和 `y`。
2. `X`: `X` 是一個 NumPy 陣列，形狀為 `(70000, 784)`。每一行代表一張圖片，784 個特徵對應 28x28 像素的灰階值。

**🎯 重點摘要:**

- **核心功能**: 將資料集的特徵（`X`）與目標（`y`）分離，這是機器學習工作流程的標準步驟。
- **資料結構**: `X` 是二維陣列（樣本數 x 特徵數），`y` 是一維陣列（樣本數）。

### 範例 7: 視覺化單一數字

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()
```

![image](images/classification/some_digit_plot.png)

**✅ 程式碼逐行解析：**

1. `def plot_digit(image_data)`: 定義一個函式，用於將一維的像素陣列視覺化。
2. `image = image_data.reshape(28, 28)`:將 784 個元素的一維陣列重塑為 28x28 的二維陣列，以符合圖片的原始尺寸。
3. `plt.imshow(image, cmap="binary")`: 使用 `imshow` 函式顯示圖片。`cmap="binary"` 表示使用黑白色彩映射，像素值低的點顯示為黑色，高的點顯示為白色。
4. `plt.axis("off")`: 關閉座標軸，讓圖片更清晰。
5. `some_digit = X[0]`: 選取資料集中的第一張圖片。
6. `plot_digit(some_digit)`: 呼叫函式繪製該圖片。

**🎯 重點摘要:**

- **核心功能**: 將扁平化的特徵向量還原成二維圖像並顯示，方便我們直觀地理解資料內容。
- **最佳使用情境**: 在進行資料分析或模型除錯時，視覺化樣本有助於檢查資料是否正確載入或模型預測是否合理。

### 範例 8: 分割訓練集與測試集

```python
# 分割資料為訓練集與測試集
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

**✅ 程式碼逐行解析：**

1. `X_train, X_test = X[:60000], X[60000:]`: MNIST 資料集已經預先打亂，且通常前 60,000 筆作為訓練集，後 10,000 筆作為測試集。這裡我們使用 Python 的切片語法來進行分割。
2. `y_train, y_test = y[:60000], y[60000:]`: 同樣地，對標籤 `y` 進行相同的分割。

**🎯 重點摘要:**

- **核心功能**: 將資料集劃分為訓練集和測試集，是所有監督式學習任務的標準流程。模型在訓練集上學習，在測試集上評估其泛化能力。
- **潛在問題**: 確保訓練集與測試集沒有重疊，否則會導致評估結果過於樂觀，無法反映模型在未知資料上的真實表現。

---

## <a id="binary-classifier"></a>訓練一個二元分類器

💡 **實際應用情境：**
二元分類器用於解決「是」或「否」的問題。例如，判斷一封郵件是否為垃圾郵件、一張圖片是否為貓、或一個交易是否為詐騙。在這裡，我們將簡化問題：只判斷一個數字**是否為 `5`**。

### 範例 9: 建立目標向量

```python
# 建立二元分類的目標向量
y_train_5 = (y_train == '5')  # 如果是 '5'，則為 True，否則為 False
y_test_5 = (y_test == '5')
```

**✅ 程式碼逐行解析：**

1. `y_train_5 = (y_train == '5')`: 這行程式碼會對 `y_train` 中的每個元素進行比較。如果元素等於字串 `'5'`，則新陣列 `y_train_5` 在該位置的值為 `True`，否則為 `False`。
2. `y_test_5 = (y_test == '5')`: 同樣地，為測試集建立二元標籤。

**🎯 重點摘要:**

- **核心功能**: 將一個多類別分類問題（0-9）轉換為一個二元分類問題（是 5 vs 不是 5）。
- **資料型態**: `y_train_5` 和 `y_test_5` 是布林值（Boolean）陣列，這在 Scikit-Learn 中可以被直接當作 `0` 和 `1` 來處理。

### 範例 10: 訓練 SGD 分類器

```python
# 從 sklearn.linear_model 載入 SGDClassifier
from sklearn.linear_model import SGDClassifier

# 建立 SGDClassifier 實例
sgd_clf = SGDClassifier(random_state=42)
# 使用訓練資料擬合模型
sgd_clf.fit(X_train, y_train_5)
```

**✅ 程式碼逐行解析：**

1. `from sklearn.linear_model import SGDClassifier`: `SGDClassifier` 是一個使用隨機梯度下降（Stochastic Gradient Descent）演算法來訓練的線性分類器。
2. `sgd_clf = SGDClassifier(random_state=42)`: 建立一個 `SGDClassifier` 的實例。`random_state=42` 確保了每次執行程式碼時，SGD 的隨機過程都是一樣的，使得結果可以重現。
3. `sgd_clf.fit(X_train, y_train_5)`: 使用訓練集的特徵 `X_train` 和二元目標 `y_train_5` 來訓練模型。模型會學習如何從 784 個像素值中辨識出數字 `5`。

**🎯 重點摘要:**

- **核心功能**: `SGDClassifier` 是一個高效的分類器，特別適合處理像 MNIST 這樣的大型資料集。
- **最佳使用情境**: 當資料量很大，無法一次性載入記憶體時，SGD 是一個很好的選擇，因為它可以進行增量學習（online learning）。

### 範例 11: 進行預測

```python
# 預測 `some_digit` 是否為 5
sgd_clf.predict([some_digit])
```

**✅ 程式碼逐行解析：**

1. `sgd_clf.predict([some_digit])`: 使用訓練好的 `sgd_clf` 模型來預測 `some_digit`（我們之前選取的第一張圖片）。`predict` 方法會回傳 `True` 或 `False`。因為 `some_digit` 確實是數字 `5`，所以模型正確地預測為 `True`。

**🎯 重點摘要:**

- **核心功能**: `.predict()` 方法是使用訓練好的模型進行預測的標準介面。
- **輸入格式**: 注意 `[some_digit]` 使用了方括號，因為 Scikit-Learn 的 `predict` 方法期望收到一個二維陣列（即使只有一個樣本）。

---

## <a id="performance-measures"></a>效能衡量

💡 **實際應用情境：**
模型訓練完後，我們如何知道它做得好不好？單純看「準確率」有時會誤導人，尤其是在資料不平衡的情況下。本節將介紹更全面的評估方法。

### 範例 12: 使用交叉驗證衡量準確率

```python
# 從 sklearn.model_selection 載入 cross_val_score
from sklearn.model_selection import cross_val_score

# 使用 3-fold 交叉驗證計算準確率
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

**✅ 程式碼逐行解析：**

1. `from sklearn.model_selection import cross_val_score`: `cross_val_score` 是一個方便的函式，可以自動完成交叉驗證的整個流程。
2. `cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")`:
   - `sgd_clf`: 要評估的模型。
   - `X_train`, `y_train_5`: 訓練資料。
   - `cv=3`: `cv` 代表 "cross-validation"，這裡設定為 3，表示要進行 3-fold 交叉驗證。資料會被分成 3 折，輪流將其中 1 折作為驗證集，另外 2 折作為訓練集，重複 3 次。
   - `scoring="accuracy"`: 指定評估指標為「準確率」。
   - **回傳值**: 一個包含 3 次驗證準確率的陣列。

**🎯 重點摘要:**

- **核心功能**: 交叉驗證提供比單次訓練/驗證分割更穩健的模型效能評估。它可以減少因偶然的資料分割所帶來的評估偏差。
- **潛在問題**: 交叉驗證的計算成本較高，因為模型需要被訓練 `cv` 次。

### 範例 13: 混淆矩陣 (Confusion Matrix)

```python
# 從 sklearn.model_selection 載入 cross_val_predict
from sklearn.model_selection import cross_val_predict
# 從 sklearn.metrics 載入 confusion_matrix
from sklearn.metrics import confusion_matrix

# 取得交叉驗證的預測結果
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# 計算混淆矩陣
cm = confusion_matrix(y_train_5, y_train_pred)
cm
```

**✅ 程式碼逐行解析：**

1. `from sklearn.model_selection import cross_val_predict`: 與 `cross_val_score` 不同，`cross_val_predict` 回傳的是在交叉驗證過程中，對每個樣本的「預測值」，而不是效能分數。
2. `y_train_pred = cross_val_predict(...)`: 這裡的 `y_train_pred` 是一個與 `y_train_5` 大小相同的陣列，其中每個預測都是由一個「沒有看過」該樣本的模型所做出的。
3. `cm = confusion_matrix(y_train_5, y_train_pred)`: `confusion_matrix` 函式接收真實標籤 `y_train_5` 和預測標籤 `y_train_pred`，並回傳一個 2x2 的矩陣。

**🎯 重點摘要:**

- **核心功能**: 混淆矩陣提供了比單一準確率更詳細的效能資訊。它顯示了模型在各類別上的預測分佈情況。
- **矩陣解讀**:
  - 左上 (TN): 正確預測為「非 5」。
  - 右上 (FP): 錯誤預測為「是 5」（偽陽性）。
  - 左下 (FN): 錯誤預測為「非 5」（偽陰性）。
  - 右下 (TP): 正確預測為「是 5」。
- **最佳使用情境**: 混淆矩陣是計算精確率、召回率等指標的基礎，對於理解模型的錯誤類型至關重要。

### 範例 14: 精確率與召回率 (Precision and Recall)

```python
# 從 sklearn.metrics 載入 precision_score, recall_score
from sklearn.metrics import precision_score, recall_score

# 計算精確率
precision_score(y_train_5, y_train_pred)
# 計算召回率
recall_score(y_train_5, y_train_pred)
```

**✅ 程式碼逐行解析：**

1. `precision_score(y_train_5, y_train_pred)`: 計算精確率，公式為 `TP / (TP + FP)`。在所有被模型預測為「是 5」的樣本中，有多少是真的 `5`？
2. `recall_score(y_train_5, y_train_pred)`: 計算召回率，公式為 `TP / (TP + FN)`。在所有真正的 `5` 中，有多少被模型成功找出來了？

**🎯 重點摘要:**

- **核心功能**: 精確率和召回率是衡量分類器效能的兩個核心指標，特別是在类别不平衡的資料集上。
- **權衡 (Trade-off)**: 通常情況下，提高精確率會導致召回率下降，反之亦然。這被稱為「精確率/召回率權衡」。
- **最佳使用情境**:
  - **高精確率**: 當「偽陽性」的代價很高時（例如：將正常郵件誤判為垃圾郵件）。
  - **高召回率**: 當「偽陰性」的代價很高時（例如：未能檢測出癌症病患）。

### 範例 15: F1 分數

```python
# 從 sklearn.metrics 載入 f1_score
from sklearn.metrics import f1_score

# 計算 F1 分數
f1_score(y_train_5, y_train_pred)
```

**✅ 程式碼逐行解析：**

1. `f1_score(y_train_5, y_train_pred)`: F1 分數是精確率和召回率的「調和平均數」。公式為 `2 * (precision * recall) / (precision + recall)`。

**🎯 重點摘要:**

- **核心功能**: F1 分數提供了一個結合精確率與召回率的單一指標。當你希望兩者都有不錯的表現時，F1 分數是一個很好的綜合評估指標。
- **特性**: F1 分數對於較低的值更為敏感。只有當精確率和召回率都很高時，F1 分數才會高。

### 範例 16: 精確率/召回率權衡與決策邊界

```python
# 取得單一樣本的決策分數
y_scores = sgd_clf.decision_function([some_digit])
y_scores

# 設定不同的閾值
threshold = 0
y_some_digit_pred = (y_scores > threshold) # > True

threshold = 3000
y_some_digit_pred = (y_scores > threshold) # > False
```

**✅ 程式碼逐行解析：**

1. `y_scores = sgd_clf.decision_function([some_digit])`: `decision_function` 回傳每個樣本的「決策分數」。對於線性分類器，這代表樣本到決策邊界的帶符號距離。分數越高，模型越確信樣本屬於正類。
2. `y_some_digit_pred = (y_scores > threshold)`: `predict` 方法的內部機制其實就是比較決策分數和一個閾值（預設為 0）。透過手動調整 `threshold`，我們可以改變模型的決策邊界。

**🎯 重點摘要:**

- **核心功能**: 透過調整決策閾值，我們可以在精確率和召回率之間进行權衡。
- **閾值影響**:
  - **提高閾值**: 模型變得更「嚴格」，只在非常有把握時才預測為正類。這會提高精確率，但可能漏掉一些正樣本，從而降低召回率。
  - **降低閾值**: 模型變得更「寬鬆」，更容易將樣本預測為正類。這會提高召回率，但可能引入更多偽陽性，從而降低精確率。

### 範例 17: 繪製精確率-召回率曲線 (Precision-Recall Curve)

```python
# 取得所有訓練樣本的決策分數
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

# 從 sklearn.metrics 載入 precision_recall_curve
from sklearn.metrics import precision_recall_curve

# 計算不同閾值下的精確率與召回率
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# 繪製曲線
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend()
plt.grid()
plt.xlabel("Threshold")
plt.show()
```

![image](images/classification/precision_recall_vs_threshold_plot.png)

**✅ 程式碼逐行解析：**

1. `y_scores = cross_val_predict(..., method="decision_function")`: 這次我們使用 `cross_val_predict` 來取得每個樣本的決策分數。
2. `precisions, recalls, thresholds = precision_recall_curve(...)`: 這個函式會計算在所有可能的閾值下，對應的精確率和召回率。
3. `plt.plot(...)`:我們將精確率和召回率作為閾值的函式繪製出來。注意 `precisions` 和 `recalls` 的長度比 `thresholds` 多一，所以我們使用 `[:-1]` 來切片。

**🎯 重點摘要:**

- **核心功能**: PR 曲線視覺化了精確率和召回率之間的權衡。
- **曲線解讀**: 隨著閾值的升高，精確率曲線（藍色虛線）趨於上升，而召回率曲線（綠色實線）趟於下降。你可以根據業務需求，在這條曲線上選擇一個最優的平衡點（閾值）。

### 範例 18: ROC 曲線 (Receiver Operating Characteristic Curve)

```python
# 從 sklearn.metrics 載入 roc_curve
from sklearn.metrics import roc_curve

# 計算 FPR, TPR
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# 繪製 ROC 曲線
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.legend()
plt.show()
```

![image](images/classification/roc_curve_plot.png)

**✅ 程式碼逐行解析：**

1. `from sklearn.metrics import roc_curve`: 載入計算 ROC 曲線所需的函式。
2. `fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)`: `roc_curve` 計算在不同閾值下的「偽陽性率 (FPR)」和「真陽性率 (TPR)」。
   - **TPR (召回率)**: `TP / (TP + FN)`
   - **FPR**: `FP / (FP + TN)`，即在所有負樣本中，被錯誤預測為正樣本的比例。
3. `plt.plot(fpr, tpr, ...)`: 繪製 TPR vs FPR 的曲線。
4. `plt.plot([0, 1], [0, 1], 'k:')`: 繪製一條對角虛線，它代表一個完全隨機的分類器。

**🎯 重點摘要:**

- **核心功能**: ROC 曲線是另一個評估分類器效能的重要工具。它展示了 TPR 和 FPR 之間的權衡。
- **曲線解讀**: 一個好的分類器，其 ROC 曲線會盡量往左上角靠近（TPR 高，FPR 低）。曲線離對角線越遠，模型效能越好。
- **AUC (Area Under the Curve)**: ROC 曲線下的面積（AUC）是一個綜合性的效能指標。AUC 為 1.0 代表完美分類器，0.5 代表隨機分類器。

### 範例 19: 比較不同模型的 ROC 曲線

```python
# 訓練一個隨機森林分類器
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

# 取得隨機森林的分數 (使用正類的機率)
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# 繪製兩條 ROC 曲線
plt.plot(fpr, tpr, "b:", label="SGD")
plt.plot(fpr_forest, tpr_forest, label="Random Forest")
plt.legend(loc="lower right")
plt.show()
```

![image](images/classification/pr_curve_comparison_plot.png)

**✅ 程式碼逐行解析：**

1. `y_probas_forest = cross_val_predict(..., method="predict_proba")`: 對於 `RandomForestClassifier`，我們使用 `predict_proba` 方法來取得每個樣本屬於各類別的「機率」。
2. `y_scores_forest = y_probas_forest[:, 1]`: 我們選取屬於正類（'是 5'）的機率作為決策分數。
3. `fpr_forest, tpr_forest, ... = roc_curve(...)`: 計算隨機森林模型的 ROC 曲線數據。
4. `plt.plot(...)`: 在同一張圖上繪製 SGD 和隨機森林的 ROC 曲線。

**🎯 重點摘要:**

- **核心功能**: 透過比較不同模型的 ROC 曲線與 AUC 分數，我們可以客觀地判斷哪個模型在該分類任務上表現更好。
- **結論**: 從圖中可以看出，隨機森林的 ROC 曲線更靠近左上角，且其 AUC 分數（可透過 `roc_auc_score` 計算）也更高，表明它是一個比 SGD 更好的分類器。

---

## <a id="multiclass-classification"></a>多類別分類

💡 **實際應用情境：**
當分類任務的目標超過兩個類別時，就稱為多類別分類。例如，將手寫數字辨識為 0 到 9 中的任何一個，或者將新聞文章分類到體育、政治、娛樂等多個頻道。

Scikit-Learn 會自動偵測到你傳入的 `y` 包含了多個類別，並自動採用相應的策略。

### 範例 20: 訓練多類別 SVM 分類器

```python
# 從 sklearn.svm 載入 SVC
from sklearn.svm import SVC

# 建立 SVC 實例
svm_clf = SVC(random_state=42)
# 使用原始的多類別標籤 y_train 進行訓練
# 為了速度，這裡只使用前 2000 個樣本
svm_clf.fit(X_train[:2000], y_train[:2000])
```

**✅ 程式碼逐行解析：**

1. `from sklearn.svm import SVC`: `SVC` (Support Vector Classifier) 是一個強大的分類模型，常用於複雜但中小型規模的資料集。
2. `svm_clf.fit(X_train[:2000], y_train[:2000])`: 我們直接使用包含 0-9 標籤的 `y_train` 來訓練模型。Scikit-Learn 的 `SVC` 在底層會自動採用 **OvO (One-vs-One)** 策略。

**🎯 重點摘要:**

- **核心功能**: Scikit-Learn 的分類器大多支援多類別分類，無需手動轉換。
- **OvO vs OvR**:
  - **OvO (一對一)**: 為每一對類別訓練一個二元分類器（例如，為 '0' vs '1', '0' vs '2', ..., '8' vs '9' 都訓練一個）。對於 N 個類別，需要訓練 `N * (N-1) / 2` 個分類器。`SVC` 預設使用此策略。
  - **OvR (一對多)**: 為每個類別訓練一個二元分類器，用以區分該類別與所有其他類別（例如，'0' vs '非 0', '1' vs '非 1', ...）。對於 N 個類別，需要訓練 N 個分類器。`SGDClassifier` 和 `LogisticRegression` 預設使用此策略。

### 範例 21: 檢視多類別決策分數

```python
# 取得單一樣本的決策分數
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)

# 找出分數最高的類別索引
class_id = some_digit_scores.argmax()
# 根據索引找到對應的類別名稱
svm_clf.classes_[class_id]
```

**✅ 程式碼逐行解析：**

1. `some_digit_scores = svm_clf.decision_function([some_digit])`: 對於多類別分類，`decision_function` 會回傳每個類別的分數。
2. `class_id = some_digit_scores.argmax()`: `argmax()` 函式會回傳分數最高（最可能）的類別的「索引」。
3. `svm_clf.classes_[class_id]`: `svm_clf.classes_` 屬性儲存了模型學習到的所有類別標籤的列表。我們使用 `class_id` 索引來取得最終的預測類別。

**🎯 重點摘要:**

- **核心功能**: 多類別分類的預測過程，是基於比較所有類別的決策分數，並選擇分數最高者作為最終結果。

---

## <a id="error-analysis"></a>錯誤分析

💡 **實際应用情境：**
模型效能達到瓶頸時，我們需要深入分析它「錯在哪裡」，才能找到改進的方向。錯誤分析可以幫助我們識別模型的弱點，例如，模型是否經常混淆某幾個特定的類別？

### 範例 22: 繪製多類別混淆矩陣

```python
# 從 sklearn.metrics 載入 ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay

# 取得交叉驗證的預測結果
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

# 繪製混淆矩陣
plt.rc('font', size=9)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()
```

![image](images/classification/confusion_matrix_plot_1.png)

**✅ 程式碼逐行解析：**

1. `y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)`: 首先，我們需要一組「乾淨」的預測。這裡使用 `cross_val_predict` 確保每個樣本的預測都是由未見過該樣本的模型做出的。注意，這裡使用了經過特徵縮放的 `X_train_scaled`，這對 `SGDClassifier` 的效能至關重要。
2. `ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)`: 這是 Scikit-Learn 1.0 之後新增的便捷函式，可以直接從真實標籤和預測標籤繪製出帶有標籤和顏色條的混淆矩陣圖。

**🎯 重點摘要:**

- **核心功能**: 視覺化多類別分類的混淆矩陣。
- **矩陣解讀**:
  - **對角線**: 對角線上的數字代表「正確預測」的樣本數。數字越亮（或越大），表示該類別的預測效果越好。
  - **非對角線**: 非對角線上的數字代表「錯誤預測」。例如，第 3 行第 5 列的數字表示有多少個真實的 `3` 被錯誤地預測成了 `5`。
- **分析**: 從圖中可以看出，數字 `5` 的對角線格子看起來比其他數字暗一些，表示模型在辨識 `5` 方面的準確率較低。此外，許多數字被錯誤地分類為 `8`。

### 範例 23: 正規化混淆矩陣以分析錯誤率

```python
# 繪製按真實標籤正規化的混淆矩陣
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
plt.show()
```

![image](images/classification/confusion_matrix_plot_2.png)

**✅ 程式碼逐行解析：**

1. `normalize="true"`: 這個參數會將混淆矩陣的每一行（代表每個「真實類別」）除以該行的總和。這樣，矩陣中的每個值就變成了「比例」或「錯誤率」。
2. `values_format=".0%"`: 這個參數用於格式化矩陣中的數字，使其以百分比形式顯示。

**🎯 重點摘要:**

- **核心功能**: 透過正規化，我們可以更清楚地看到每個類別的「錯誤分佈」。
- **分析**:
  - **行分析**: 查看第 8 行，我們可以看到只有 85% 的 `8` 被正確分類，而有 10% 的 `8` 被錯誤地預測為 `1`, `3`, `5` 等。
  - **列分析**: 查看第 8 列，我們可以看到許多其他數字（特別是 `1`, `2`, `3`, `5`）都被錯誤地預測成了 `8`。
  - **對稱性**: 矩陣在 `3` 和 `5` 之間看起來也有些對稱，表示 `3` 和 `5` 經常被互相混淆。

### 範例 24: 檢視個別錯誤範例

```python
# 找出真實為 3 但預測為 5 的圖片
X_35 = X_train[(y_train == '3') & (y_train_pred == '5')]
# 找出真實為 5 但預測為 3 的圖片
X_53 = X_train[(y_train == '5') & (y_train_pred == '3')]

# 繪製這些被混淆的數字
# (此處省略繪圖程式碼，詳見 notebook)
```

![image](images/classification/error_analysis_digits_plot.png)

**✅ 程式碼逐行解析：**

1. `X_35 = X_train[(y_train == '3') & (y_train_pred == '5')]`: 使用布林索引來篩選出所有「真實標籤是 '3' 但模型預測是 '5'」的圖片。
2. `X_53 = X_train[(y_train == '5') & (y_train_pred == '3')]`: 使用布林索引來篩選出所有「真實標籤是 '5' 但模型預測是 '3'」的圖片。

**🎯 重點摘要:**

- **核心功能**: 深入到實際的錯誤樣本中，直觀地理解模型為什麼會犯錯。
- **分析**: 透過觀察這些被混淆的圖片，我們發現許多 `3` 和 `5` 的寫法確實非常相似，甚至人眼也很難區分。這表明這個問題本身具有一定的模糊性。對於線性模型 `SGDClassifier` 來說，這些細微的差異很難捕捉。
- **改進方向**: 針對這些特定的混淆對，我們可以嘗試收集更多難以區分的樣本來增強訓練集，或者使用更能捕捉複雜特徵的模型（如 CNN）。

---

## <a id="multilabel-classification"></a>多標籤分類

💡 **實際應用情境：**
在某些情況下，我們希望一個樣本可以同時擁有多個標籤。例如，在一張包含多個人物的照片中，我們可能希望模型能辨識出所有出現的人物（例如，同時標記出 "愛麗絲"、"鮑勃" 和 "查理"）。這就是多標籤分類。

### 範例 25: 建立多標籤目標

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 標籤 1: 數字是否大於等於 7
y_train_large = (y_train.astype('int8') >= 7)
# 標籤 2: 數字是否為奇數
y_train_odd = (y_train.astype('int8') % 2 == 1)

# 將兩個標籤合併成一個多標籤陣列
y_multilabel = np.c_[y_train_large, y_train_odd]

# 訓練一個 KNN 分類器
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

**✅ 程式碼逐行解析：**

1. `y_train_large = (y_train.astype('int8') >= 7)`: 建立第一個標籤，判斷數字是否 >= 7。
2. `y_train_odd = (y_train.astype('int8') % 2 == 1)`: 建立第二個標籤，判斷數字是否為奇數。
3. `y_multilabel = np.c_[y_train_large, y_train_odd]`: `np.c_` 是 NumPy 中一個方便的工具，可以將兩個一維陣列按列合併成一個二維陣列。現在，`y_multilabel` 的每一行都包含了兩個布林標籤。
4. `knn_clf.fit(X_train, y_multilabel)`: `KNeighborsClassifier` 本身就支援多標籤分類，所以我們可以像平常一樣直接訓練它。

**🎯 重點摘要:**

- **核心功能**: 建立一個多標籤的目標 `y`，其中每個樣本可以對應多個 `True`/`False` 標籤。
- **模型支援**: 并非所有 Scikit-Learn 分類器都原生支援多標籤分類，但 `KNeighborsClassifier` 和 `RandomForestClassifier` 等模型可以。

### 範例 26: 進行多標籤預測與評估

```python
# 預測 `some_digit` (數字 5) 的多個標籤
knn_clf.predict([some_digit])
# > array([[False,  True]])  (不大於等於 7, 是奇數)

# 評估多標籤分類的 F1 分數
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
```

**✅ 程式碼逐行解析：**

1. `knn_clf.predict([some_digit])`: 預測結果是一個包含兩個布林值的陣列。對於數字 `5`，模型正確地預測出它「不是大數字」(`False`) 且「是奇數」(`True`)。
2. `f1_score(..., average="macro")`: 在評估多標籤分類時，我們需要指定 `average` 參數。
   - `"macro"`: 分別計算每個標籤的 F1 分數，然後取其「未加權」的平均值。所有標籤同等重要。
   - `"weighted"`: 同樣計算每個標籤的 F1 分數，但會根據每個標籤的樣本數（support）進行加權平均。

**🎯 重點摘要:**

- **核心功能**: 多標籤分類的評估需要考慮所有標籤的綜合表現。
- **`average` 參數的選擇**: 如果你認為所有標籤同等重要，使用 `"macro"`；如果某些標籤的樣本數遠多於其他標籤，且你希望給樣本數多的標籤更大權重，則使用 `"weighted"`。

---

## <a id="multioutput-classification"></a>多輸出分類

💡 **實際應用情境：**
多輸出分類（Multioutput Classification）是多標籤分類的一般化形式，其中每個標籤可以是多類別的（而不僅僅是 `True`/`False`）。一個典型的例子是圖像去噪：模型的輸入是一張帶有噪點的圖片，輸出是「乾淨」的圖片。在這個場景下，模型的每個輸出（像素點）都可以有多個值（0-255 的灰階值）。

### 範例 27: 建立多輸出資料集

```python
# 為訓練集和測試集添加噪點
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

# 目標是還原成原始的、乾淨的圖片
y_train_mod = X_train
y_test_mod = X_test
```

![image](images/classification/noisy_digit_example_plot.png)

**✅ 程式碼逐行解析：**

1. `noise = np.random.randint(0, 100, ...)`: 生成一個與 `X_train` 形狀相同的隨機整數矩陣，值域在 0 到 100 之間。
2. `X_train_mod = X_train + noise`: 將噪點加到原始圖片上，生成帶噪點的輸入特徵。
3. `y_train_mod = X_train`: 設定目標 `y` 為「原始的、乾淨的」圖片。

**🎯 重點摘要:**

- **核心功能**: 建立一個多輸出任務。輸入是 784 維的帶噪點像素，輸出是 784 維的乾淨像素。每個輸出維度（像素）都是一個多類別問題（0-255）。

### 範例 28: 訓練並預測

```python
# 訓練 KNN 分類器來去噪
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)

# 預測一張帶噪點圖片的乾淨版本
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
```

![image](images/classification/cleaned_digit_example_plot.png)

**✅ 程式碼逐行解析：**

1. `knn_clf.fit(X_train_mod, y_train_mod)`: 訓練 KNN 模型。模型學習從帶噪點的圖片 `X_train_mod` 映射到乾淨的圖片 `y_train_mod`。
2. `clean_digit = knn_clf.predict([X_test_mod[0]])`: 使用訓練好的模型對一張帶噪點的測試圖片進行預測。
3. `plot_digit(clean_digit)`: 繪製出去噪後的圖片。

**🎯 重點摘要:**

- **核心功能**: 多輸出分類器可以處理複雜的結構化輸出，例如圖像。
- **結論**: 儘管這是一個非常困難的任務，但 KNN 分類器仍然成功地移除了大部分噪點，並還原了數字的基本形狀。

---

## <a id="exercise-solutions"></a>練習題解答

### 1. MNIST 97% 準確率分類器

嘗試建立一個在 MNIST 測試集上達到 97% 準確率的分類器。一個提示是 `KNeighborsClassifier` 在這個任務上表現得相當不錯。你可能需要使用網格搜尋（`GridSearchCV`）來找到最佳的超參數組合。

```python
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

```python
# 最佳超參數
grid_search.best_params_

# 最佳分數
grid_search.best_score_

# 使用最佳估計器在測試集上評估
from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)
accuracy_score(y_test, y_pred)
```

透過網格搜尋，我們可以找到 `weights='distance'` 和 `n_neighbors=4` 的組合，在交叉驗證中達到了約 97.1% 的準確率，並在最終的測試集上達到了 97.14% 的準確率。

### 2. 資料增強 (Data Augmentation)

編寫一個函式，可以將 MNIST 圖像向任何方向（上、下、左、右）移動一個像素。然後，對於訓練集中的每張圖片，創建四個移動過的副本（每個方向一個），並將它們添加到訓練集中。最後，在這個擴展的訓練集上訓練你的最佳模型，並在測試集上評估它。你應該會觀察到模型效能有顯著提升。

```python
from scipy.ndimage import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])
```

```python
# 創建增強後的資料集
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)
```

```python
# 在增強後的資料集上重新訓練
knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(X_train_augmented, y_train_augmented)

y_pred = knn_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```

透過資料增強，我們將訓練集的大小擴大了五倍。在這個擴展的資料集上重新訓練 `KNeighborsClassifier`，測試集上的準確率從 97.14% 提升到了 97.56%。這看似微小的提升，實際上讓模型的錯誤率降低了約 17%，效果顯著。

### 3. 鐵達尼號資料集挑戰

挑戰：處理鐵達尼號資料集。目標是預測一個乘客是否能夠存活（`Survived`）。

**步驟：**

1. **載入資料**: 使用 Pandas 載入 `train.csv` 和 `test.csv`。
2. **探索資料**:
    - `head()`: 查看前幾行。
    - `info()`: 檢查缺失值和資料類型。`Age`, `Cabin`, `Embarked` 有缺失值。`Cabin` 缺失嚴重，可能需要放棄。
    - `describe()`: 查看數值屬性的統計摘要。
    - `value_counts()`: 查看類別屬性的分佈，如 `Sex`, `Pclass`, `Embarked`。
3. **建立前處理管線**:
    - **數值管線**: 處理 `Age`, `SibSp`, `Parch`, `Fare`。使用 `SimpleImputer` 填充 `Age` 的缺失值（使用中位數），然後使用 `StandardScaler` 進行特徵縮放。
    - **類別管線**: 處理 `Pclass`, `Sex`, `Embarked`。使用 `SimpleImputer` 填充 `Embarked` 的缺失值（使用最頻繁值），然後使用 `OneHotEncoder` 進行獨熱編碼。
    - **合併管線**: 使用 `ColumnTransformer` 將數值和類別管線合併。

```python
# (此處省略了詳細的資料載入與前處理管線程式碼，詳見 notebook)
# ...

# 訓練一個隨機森林分類器
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train_prepared, y_train)

# 評估模型
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_clf, X_train_prepared, y_train, cv=10)
forest_scores.mean()
# > 約 81.2% 的準確率
```

我們建立了一個完整的前處理管線，並使用 `RandomForestClassifier` 進行訓練。透過 10-fold 交叉驗證，模型的平均準確率約為 81.2%。

### 4. 垃圾郵件分類器

挑戰：建立一個垃圾郵件分類器。

**步驟：**

1. **獲取資料**: 下載並解壓縮 [SpamAssassin 公開資料集](https://spamassassin.apache.org/old/publiccorpus/)。
2. **探索與前處理**: 探索電子郵件的結構，提取有用的特徵（如字詞頻率、主題長度等）。
3. **建立管線**: 建立一個完整的前處理管線，包括文字清理、特徵提取和縮放。
4. **訓練模型**: 使用多種不同的分類器進行訓練和比較。
5. **評估**: 使用交叉驗證評估模型的效能。

透過以上步驟，我們可以建立一個有效的垃圾郵件分類器，並深入了解文字分類的挑戰與技巧。

---

## <a id="best-practices"></a>💡 總結與最佳實踐

本章我們深入探討了機器學習中的分類任務，從基礎的二元分類到進階的多標籤、多輸出分類。以下是關鍵的總結和最佳實踐建議：

### 🎯 關鍵學習重點

1. **選擇適當的評估指標**:
   - **準確率**並非萬能，在不平衡資料集上可能會誤導
   - **精確率 vs 召回率**：根據業務需求選擇重點
   - **F1 分數**：當你需要平衡精確率和召回率時
   - **ROC 曲線與 AUC**：適合比較不同模型的整體表現

2. **模型選擇指南**:
   - **SGDClassifier**：適合大型資料集，訓練速度快
   - **SVM (SVC)**：適合中小型資料集，支援非線性分類
   - **RandomForestClassifier**：通常表現良好，較少需要調參
   - **KNeighborsClassifier**：簡單有效，適合作為基準模型

3. **資料前處理的重要性**:
   - **特徵縮放**：對於距離導向的演算法（如 SVM、KNN）特別重要
   - **資料增強**：可以有效提升模型的泛化能力
   - **交叉驗證**：確保模型評估的可信度

### 🚀 實務開發建議

1. **從簡單開始**：
   - 先建立一個簡單的基準模型
   - 逐步增加複雜度，觀察效能提升

2. **重視錯誤分析**：
   - 使用混淆矩陣識別模型的弱點
   - 分析錯誤案例，指導特徵工程

3. **適當的模型選擇**：
   - 考慮資料大小、維度和問題複雜度
   - 平衡模型複雜度與可解釋性

4. **持續監控與改進**：
   - 定期檢查模型在新資料上的表現
   - 根據業務回饋調整評估指標

### ⚠️ 常見陷阱與注意事項

1. **過度擬合**：使用交叉驗證和適當的正則化
2. **資料洩漏**：確保測試集完全獨立
3. **評估偏差**：在不平衡資料上選擇合適的評估指標
4. **特徵工程**：領域知識比演算法選擇更重要

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

### Q1: 何時應該使用精確率，何時使用召回率？

**A**: 這取決於「偽陽性」和「偽陰性」的代價：
- **高精確率優先**：當偽陽性代價很高時（例如：垃圾郵件過濾，將重要郵件誤判為垃圾會很糟糕）
- **高召回率優先**：當偽陰性代價很高時（例如：癌症檢測，漏診比誤診更危險）
- **F1 分數**：當兩者同等重要時

### Q2: SGD 與 SVM 有何區別？何時選擇哪一個？

**A**: 
- **SGD（隨機梯度下降）**：
  - 是一種**最佳化演算法**，可以訓練多種模型（包括線性 SVM）
  - 適合**大型資料集**，記憶體效率高
  - 支援**線上學習**（incremental learning）

- **SVM（支援向量機）**：
  - 是一種**特定的機器學習演算法**
  - 使用核技巧支援**非線性分類**
  - 適合**中小型資料集**，在高維空間表現優異

**選擇建議**：
- 資料量大（>100K 樣本）→ SGDClassifier
- 資料量中等且需要非線性邊界 → SVC with RBF kernel
- 需要機率輸出 → LogisticRegression 或 RandomForestClassifier

### Q3: 如何處理不平衡的資料集？

**A**: 多種策略可以組合使用：

1. **評估指標調整**：
   - 使用精確率、召回率、F1 分數而非準確率
   - 關注 ROC-AUC 或 Precision-Recall AUC

2. **資料層面**：
   - **過度採樣**：SMOTE、ADASYN
   - **欠採樣**：隨機欠採樣、Tomek links
   - **資料增強**：針對少數類生成更多樣本

3. **演算法層面**：
   - 調整 `class_weight='balanced'` 參數
   - 使用成本敏感學習
   - 集成方法（如 BalancedRandomForestClassifier）

### Q4: 交叉驗證的 fold 數應該設多少？

**A**: 常見的選擇：
- **k=5**：計算效率與評估品質的良好平衡，適合大多數情況
- **k=10**：更穩健的評估，但計算成本較高
- **留一交叉驗證 (LOOCV)**：資料集很小時（<1000 樣本）
- **分層交叉驗證**：不平衡資料集的首選

**經驗法則**：資料越少，fold 數可以越高；資料越多，fold 數可以適當減少。

### Q5: 如何選擇合適的決策閾值？

**A**: 根據業務需求調整：

1. **分析 PR 曲線和 ROC 曲線**：找到最佳的精確率/召回率平衡點
2. **使用 GridSearchCV**：將閾值當作超參數進行搜尋
3. **成本分析**：計算不同閾值下的業務成本
4. **驗證集調參**：在驗證集上測試不同閾值的效果

```python
# 範例：找到 90% 精確率對應的閾值
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
idx_90_precision = np.argmax(precisions >= 0.90)
threshold_90_precision = thresholds[idx_90_precision]
```

### Q6: 多類別分類中，OvO 和 OvR 哪個更好？

**A**: 各有優缺點：

**One-vs-One (OvO)**：
- ✅ 每個分類器只需要處理兩個類別的資料
- ✅ 對不平衡資料更穩健
- ❌ 需要訓練 N(N-1)/2 個分類器
- 🎯 適合：SVM 等訓練時間與樣本數呈超線性關係的演算法

**One-vs-Rest (OvR)**：
- ✅ 只需要訓練 N 個分類器
- ✅ 訓練和預測都比較快
- ❌ 對不平衡資料較敏感
- 🎯 適合：線性模型或大型資料集

**Scikit-Learn 的預設選擇通常是最佳的**，但你可以手動指定：
```python
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# 強制使用 OvO
ovo_clf = OneVsOneClassifier(SGDClassifier())
# 強制使用 OvR  
ovr_clf = OneVsRestClassifier(SGDClassifier())
```

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

### 主要標籤

`#Python` `#程式設計` `#教學` `#MachineLearning` `#Classification` `#ScikitLearn` `#DataScience`

### 技術標籤

`#SGD` `#SVM` `#RandomForest` `#ConfusionMatrix` `#Precision` `#Recall` `#ROC` `#AUC` `#F1Score`

### 學習標籤

`#編程` `#開發` `#技術分享` `#學習筆記` `#程式開發者` `#軟體工程` `#AI` `#人工智慧`

### 社群標籤

`#ThreadsTech` `#開發者社群` `#台灣開發者` `#程式學習` `#ML入門` `#實戰教學`

### 專題標籤

`#MNIST` `#HandsOnML` `#分類演算法` `#模型評估` `#特徵工程` `#資料科學實戰`
