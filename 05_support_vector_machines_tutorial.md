<!-- meta-title: 支持向量機完整指南：從線性分類到核技巧與 SVR 實戰 -->
<!-- meta-description: 深入理解 SVM 的最大間隔分類、軟間隔 C 參數、核技巧（RBF/多項式）、SVM 回歸 (SVR)。含 Scikit-Learn 實戰程式碼、逐行解析，以台灣製造業與金融情境為例。 -->
<!-- meta-keywords: Python, SVM, 支持向量機, 核技巧, RBF, Scikit-Learn, SVR, 分類, 迴歸, 機器學習 -->
<!-- meta-hashtags: #Python #SVM #支持向量機 #核技巧 #機器學習 #ScikitLearn #SVR #程式設計 #教學 #DataScience -->

# 🐍 支持向量機完整指南：從最大間隔到核技巧

支持向量機（Support Vector Machine，SVM）以「最大化分類間隔」為核心，在中小型資料集上效能出色，是二元分類的強力工具。本教學帶你從線性 SVM 開始，逐步深入核技巧（Kernel Trick），讓線性模型具備處理非線性問題的能力。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🔲 線性 SVM 分類](#linear-svm)
- [🔄 軟間隔 SVM](#soft-margin)
- [🔮 核技巧與非線性分類](#kernel-trick)
- [📡 RBF 核超參數調優](#rbf-tuning)
- [📈 SVM 回歸 (SVR)](#svr)
- [⚡ LinearSVC vs SVC vs SGDClassifier](#comparison)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **特徵縮放**是 SVM 成敗關鍵——必須在 Pipeline 中使用 `StandardScaler`
- **C 參數**控制軟間隔寬度：C 小→更寬間隔（更正則化）；C 大→更緊間隔（更擬合）
- **核技巧（Kernel Trick）** 讓線性 SVM 隱式在高維空間分類，計算效率高
- **RBF 核的 gamma**：高 gamma → 決策邊界複雜（過擬合風險）；低 gamma → 平滑邊界
- 大資料集（> 10 萬）改用 `LinearSVC` 或 `SGDClassifier`，訓練速度更快

---

## <a id="linear-svm"></a>🔲 線性 SVM 分類

💡 **實際應用情境：** 在 PCB 板良品/不良品檢測中，若兩類的特徵（如缺陷面積、灰度均值）線性可分，SVM 能找到「最大間隔」的決策邊界，對邊界附近的樣本分類更穩健。

SVM 的核心思想：找到**最寬的分類間隔（Margin）**。決策函數：

$$\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$$

**支持向量（Support Vectors）**：位於間隔邊界上的訓練樣本，決定決策邊界的位置。

### 範例 1: Iris 資料集線性 SVM

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# 載入 Iris 資料（只取後兩類：Versicolor vs Virginica）
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values[50:]  # 後 100 筆
y = (iris.target[50:] == 2).astype(int)  # 1 = Virginica, 0 = Versicolor

# SVM 對特徵尺度敏感，必須標準化！
svm_clf = Pipeline([
    ("scaler", StandardScaler()),   # 特徵縮放（必要！）
    ("linear_svc", LinearSVC(C=1.0, max_iter=2000, random_state=42))
])
svm_clf.fit(X, y)
print(f"預測結果: {svm_clf.predict([[5.5, 1.7], [4.0, 1.0]])}")
```

**✅ 程式碼逐行解析：**

1. `StandardScaler()`: SVM 對特徵尺度極敏感——若特徵 1 的範圍是 [0, 1000]，特徵 2 是 [0, 1]，SVM 會嚴重偏向特徵 1
2. `LinearSVC(C=1.0)`: C=1 是預設軟間隔強度，增大 C 讓 SVM 更努力分對訓練樣本（但可能過擬合）
3. `max_iter=2000`: LinearSVC 使用迭代優化，預設可能不夠收斂，適當增加

**🎯 重點摘要:**

- **Pipeline 的重要性**: Scaler 必須在 Pipeline 內，確保測試集只 `transform`
- **C 參數直覺**: 想像 C 是違規停車的罰款——C 高，每個誤分類代價高，SVM 努力避免（間隔變窄）

---

## <a id="soft-margin"></a>🔄 軟間隔 SVM (Soft Margin SVM)

💡 **實際應用情境：** 真實資料幾乎不可能完美線性可分（雜訊、異常值很常見）。軟間隔允許部分樣本落在間隔內或甚至越過邊界，透過 C 參數控制這種「容忍度」。

硬間隔 vs 軟間隔：

- **硬間隔（Hard Margin）**：要求所有樣本都在正確側，對離群值不容忍
- **軟間隔（Soft Margin）**：允許一些「鬆弛（Slack）」，更泛化

### 範例 2: C 參數對決策邊界的影響

```python
from sklearn.svm import SVC
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, C in zip(axes, [0.01, 100]):
    svm_c = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", C=C))  # 線性核 + 不同 C
    ])
    svm_c.fit(X, y)
    ax.set_title(f"C = {C}")
    # （此處省略決策邊界繪製代碼）

plt.show()
# C=0.01: 更寬間隔，允許更多誤分類（高偏差/欠擬合）
# C=100:  更窄間隔，努力分對所有點（高方差/過擬合）
```

**🎯 重點摘要:**

- **C 小（強正則化）**：更寬間隔，更多樣本可能在間隔內 → 更泛化
- **C 大（弱正則化）**：更窄間隔，努力分對訓練集 → 可能過擬合
- 用 `GridSearchCV` 或 `RandomizedSearchCV` 搜尋最佳 C

---

## <a id="kernel-trick"></a>🔮 核技巧與非線性分類

💡 **實際應用情境：** 如圓形邊界的問題（如用半徑和角度分類衛星軌道），在原始空間線性不可分，但映射到高維空間後可能線性可分。核技巧讓我們**不需要顯式計算高維映射**。

核函數類型：

| 核函數 | 特點 | 超參數 |
|--------|------|--------|
| 線性核（linear） | 等同 LinearSVC | 無 |
| 多項式核（poly） | 處理多項式邊界 | degree, coef0 |
| RBF（高斯）核 | 最通用，處理任意邊界 | gamma |
| Sigmoid 核 | 類似神經網路 | gamma, coef0 |

### 範例 3: RBF 核處理非線性問題

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 月牙形資料（線性不可分）
X_moons, y_moons = make_moons(n_samples=100, noise=0.15, random_state=42)

# RBF 核 SVM
rbf_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", gamma=5, C=0.001)  # gamma 大 → 決策邊界更複雜
)
rbf_kernel_svm_clf.fit(X_moons, y_moons)
print(f"RBF SVM 訓練準確率: {rbf_kernel_svm_clf.score(X_moons, y_moons):.4f}")
```

**✅ 程式碼逐行解析：**

1. `make_moons(noise=0.15)`: 生成帶雜訊的月牙形資料，是測試非線性分類器的標準資料集
2. `SVC(kernel="rbf", gamma=5)`: RBF 核的 gamma 控制「影響半徑」——gamma 越大，每個支持向量的影響範圍越小，邊界越複雜
3. `C=0.001`: 搭配大 gamma，使用小 C 平衡複雜度

**🎯 重點摘要:**

- **核技巧（Kernel Trick）** 計算技巧：$K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \phi(\mathbf{x}^{(i)})^T \phi(\mathbf{x}^{(j)})$，不需顯式計算映射
- RBF 核是預設首選，對大多數問題效果良好

---

## <a id="rbf-tuning"></a>📡 RBF 核超參數調優

### 範例 4: GridSearchCV 搜尋最佳 C 和 gamma

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {"svc__kernel": ["rbf"],
     "svc__gamma": [0.01, 0.1, 1, 10, 100],
     "svc__C":     [0.001, 0.01, 0.1, 1, 10, 100]}
]

svm_pipeline = make_pipeline(StandardScaler(), SVC(random_state=42))
grid_search = GridSearchCV(
    svm_pipeline, param_grid,
    cv=5, scoring="accuracy",
    n_jobs=-1, verbose=1
)
grid_search.fit(X_moons, y_moons)

print(f"最佳參數: {grid_search.best_params_}")
print(f"最佳準確率: {grid_search.best_score_:.4f}")
```

**🎯 重點摘要:**

- gamma 和 C 通常在對數尺度上搜尋（`np.logspace(-3, 3, 7)` 或 `[0.001, 0.01, ..., 1000]`）
- 若計算資源有限，改用 `RandomizedSearchCV` 加快搜尋

---

## <a id="svr"></a>📈 SVM 回歸 (SVR)

💡 **實際應用情境：** SVR 用於預測連續值，如廠房能耗預測。它建立一個寬度為 $2\varepsilon$ 的「容忍區」，在區內的預測誤差不懲罰。

### 範例 5: SVR 房價預測

```python
from sklearn.svm import SVR
import numpy as np

np.random.seed(42)
X_svr = 2 * np.random.rand(50, 1)
y_svr = (4 + 3 * X_svr + np.random.randn(50, 1)).ravel()

# SVR：在 epsilon 容忍區內的誤差不懲罰
svm_poly_reg = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)  # epsilon 容忍帶
)
svm_poly_reg.fit(X_svr, y_svr)

# 評估
from sklearn.metrics import mean_squared_error
y_pred_svr = svm_poly_reg.predict(X_svr)
rmse = np.sqrt(mean_squared_error(y_svr, y_pred_svr))
print(f"SVR 訓練集 RMSE: {rmse:.3f}")
```

**✅ 程式碼逐行解析：**

1. `SVR(epsilon=0.1)`: 誤差在 0.1 以內不計入損失（容忍帶寬度 = 2 × epsilon）
2. `C=100`: 對超出容忍帶的誤差施加高懲罰（等同分類中的 C）

**🎯 重點摘要:**

- SVR = 找到一條讓**最多點落在容忍帶內**的回歸線
- `epsilon` 越大，容忍帶越寬，支持向量越少，模型越平滑

---

## <a id="comparison"></a>⚡ LinearSVC vs SVC vs SGDClassifier

```python
# 三種 SVM 實作的比較
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier

# LinearSVC：用優化演算法直接求解，O(m × n)，大資料集快
linear_svc = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC(C=1, max_iter=2000))])

# SVC(kernel="linear")：用核演算法，O(m² × n) 到 O(m³ × n)，小資料集準確
svc_linear = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear", C=1))])

# SGDClassifier(loss="hinge")：等效線性 SVM，支援線上學習
sgd_hinge = Pipeline([("scaler", StandardScaler()),
                      ("sgd", SGDClassifier(loss="hinge", alpha=1/(len(X)*1.0)))])
```

| 方法 | 複雜度 | 特點 | 適用場景 |
|------|--------|------|---------|
| `LinearSVC` | O(m × n) | 快速線性 SVM | 大型資料集、線性問題 |
| `SVC` | O(m² ~ m³ × n) | 支援各種核 | 小型非線性問題 |
| `SGDClassifier(loss="hinge")` | O(m) per epoch | 支援線上學習 | 超大資料集、串流資料 |

**🎯 重點摘要:**

- 預設先試 `LinearSVC`（速度快）
- 若線性不夠，改用 `SVC(kernel="rbf")`
- 資料量 > 100 萬，用 `SGDClassifier`

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: 為什麼 SVM 對特徵縮放那麼敏感？**

A: SVM 最大化分類間隔，間隔的計算依賴各特徵的「距離」。若特徵 A 的範圍是 [0, 1000] 而特徵 B 是 [0, 1]，SVM 會認為特徵 A 更重要，導致偏頗的決策邊界。

**Q2: 核技巧為什麼能提高計算效率？**

A: 若要將 n 維特徵映射到 p 維空間，顯式計算需要 O(p) 空間。但核函數 $K(a, b) = \phi(a)^T \phi(b)$ 只需計算一個純量（不需要顯式計算 $\phi$），對無限維映射（如 RBF 的隱式映射）特別有效。

**Q3: C 和 gamma 同時調，方向有何規律？**

A: (1) 先用小 gamma（平滑邊界），觀察 C 的影響；(2) 找到合適的 C 後，再調整 gamma；(3) 或直接用 `GridSearchCV` 二維搜尋。高 C + 高 gamma 通常嚴重過擬合。

**Q4: SVR 和 Ridge 回歸有什麼差別？**

A: Ridge 懲罰所有誤差；SVR 對 epsilon 容忍帶內的誤差不懲罰（只懲罰大誤差），這使 SVR 對異常值更健壯。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #SVM #支持向量機 #核技巧 #機器學習 #ScikitLearn #SVR #RBF #GridSearch #程式設計 #教學 #DataScience #MachineLearning #AI #特徵工程
