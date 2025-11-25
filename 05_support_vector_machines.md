<!-- meta-title: 支持向量機實戰指南 | 從線性分類到核技巧與迴歸 -->
<!-- meta-description: 以 Scikit-Learn 完成 SVM 線性分類、非線性核技巧、迴歸與高維調參，涵蓋 Iris、make_moons、Wine 與 California Housing 實務案例。 -->
<!-- meta-keywords: Python, SVM, 支持向量機, Scikit-Learn, 分類, 迴歸, 核技巧, RBF, 線性分類, 教學 -->
<!-- meta-hashtags: #Python #SVM #支持向量機 #機器學習 #ScikitLearn #資料科學 #技術教學 #ThreadsTech #程式設計 #AI -->

# 🐍 支持向量機實戰：從線性分類到核技巧與迴歸

SVM（Support Vector Machine）以「最大化間隔」為核心，能在中小型資料集上取得高準確度。本文以台灣常見的風控、製造良品檢測與房價預估情境為例，示範如何透過 Scikit-Learn 打造可量產的 SVM 模型，並同時提供操作程式碼、逐行解析與實務建議。

## 📝 本文目錄
- [🎯 關鍵重點 (Key Takeaways)](#key-takeaways)
- [🚀 環境與資料基礎設置](#setup)
- [🧭 線性 SVM 分類策略](#linear-svm)
- [🔄 核技巧與非線性分類](#kernel)
- [📡 RBF 超參數調優實戰](#rbf)
- [💡 SVM 迴歸應用](#regression)
- [🧪 多分類與調參技巧：Wine 案例](#wine)
- [🏙️ California Housing 迴歸挑戰](#housing)
- [❓ 常見問答 (FAQ)](#faq)
- [💼 總結與最佳實踐](#best-practices)
- [🏷️ 推薦標籤](#hashtags)

<h2 id="key-takeaways">🎯 關鍵重點 (Key Takeaways)</h2>

- 特徵縮放是 SVM 成敗關鍵，務必以 `StandardScaler` 納入 `Pipeline`，避免資料洩漏。
- 調整 `C` 影響軟邊界寬度，`gamma` 決定 RBF 核的感受野；兩者需搭配交叉驗證尋找平衡點。
- Kernel Trick 讓線性模型具備非線性能力，優先考慮 `rbf`，再視需求測試 `poly` 或 `sigmoid`。
- SVM 迴歸透過 `epsilon` 建立「容忍區」，適用於需忽略微小噪音的價格或感測信號預測。
- 大型資料或高維特徵下，`LinearSVC` 與 `SGDClassifier` 較能保持速度；核 SVM 則適合資料量較小但邏輯複雜的專案。

<h2 id="setup">🚀 環境與資料基礎設置</h2>

💡 **實際應用情境：** 在導入 SVM 至既有專案前，需確認 Python、Scikit-Learn 版本並建立圖像輸出目錄，確保開發與報表生產流程一致。

```python
# ✅ 安裝檢查與共用設定 (Traditional Chinese 註解)
import sys
from pathlib import Path
from packaging import version
import sklearn
import matplotlib.pyplot as plt

assert sys.version_info >= (3, 8), "請升級至 Python 3.8 以上"
assert version.parse(sklearn.__version__) >= version.parse("1.2.0"), "需要較新的 Scikit-Learn"

IMAGES_PATH = Path("images") / "svm"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

plt.rc("font", size=13)
plt.rc("axes", labelsize=13, titlesize=13)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
```

**✅ 程式碼逐行解析：**
1. 第 1-6 行：匯入標準函式庫與 Scikit-Learn，用 `assert` 避免舊版環境造成 API 不相容。
2. 第 8-9 行：確保筆記型電腦或 CI 環境具備需求版本，提前避免訓練階段錯誤。
3. 第 11-12 行：建立專屬圖像資料夾，與團隊共用輸出格式。
4. 第 14-15 行：設定 Matplotlib 字型大小，報表輸出更易閱讀。

**🎯 重點摘要：**
- **核心功能**：提供最小可行設定與版本鎖定。
- **潛在問題**：忽略版本檢查將導致 `dual="auto"` 等新參數行為改變。
- **最佳使用情境**：專案初始化、CI 腳本或技術分享前置作業。

<h2 id="linear-svm">🧭 線性 SVM 分類策略</h2>

💡 **實際應用情境：** 以花瓣尺寸判斷 Iris 品種，或用信用卡交易金額/時段區分正常與異常。

### 範例 1：`LinearSVC` + `Pipeline` 建立穩健基準線
```python
# 線性 SVM：以 Pipeline 確保流程可重現
import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)  # True 代表 Iris virginica

svm_clf = make_pipeline(
    StandardScaler(),  # 先做特徵縮放，避免單位差異
    LinearSVC(C=1.0, dual=True, random_state=42)
)
svm_clf.fit(X, y)

new_samples = [[5.5, 1.7], [5.0, 1.5]]
print("預測結果:", svm_clf.predict(new_samples))
print("決策分數:", svm_clf.decision_function(new_samples))
```

**✅ 程式碼逐行解析：**
1. 第 1-6 行：載入資料集、建構標準化與線性分類器的組合。
2. 第 8-10 行：選取最具區別力的兩個特徵並建立二元標籤。
3. 第 12-16 行：`Pipeline` 保障縮放流程與模型一同部署，`C` 控制錯誤懲罰。
4. 第 18-21 行：示範推論，同步輸出類別與決策距離，利於風控閾值設定。

**🎯 重點摘要：**
- **核心功能**：以最少特徵建立高可讀性分類器。
- **潛在問題**：若 `dual=True` 時特徵遠大於樣本數需調整設定；未縮放會導致決策邊界偏移。
- **最佳使用情境**：特徵維度少、分類邊界接近線性的稽核、醫療初篩。

### 範例 2：調整 `C` 理解軟邊界
```python
# 對照不同 C 值的邊界寬度
import matplotlib.pyplot as plt
C_grid = [0.5, 100]
models = []
for C in C_grid:
    model = make_pipeline(StandardScaler(), LinearSVC(C=C, dual=True, random_state=42))
    model.fit(X, y)
    models.append(model)
    print(f"C={C}，支持向量數量估計：", np.mean(np.abs(model.decision_function(X)) < 1))
```

**✅ 程式碼逐行解析：**
1. 第 1-4 行：建立不同 `C` 值以觀察正則化強度。
2. 第 5-7 行：維持相同資料流程，只調整懲罰係數確保公平比較。
3. 第 8-9 行：利用距離 < 1 近似支持向量比例，理解邊界緊迫程度。

**🎯 重點摘要：**
- **核心功能**：展示 `C` 對容錯的影響。
- **潛在問題**：`C` 過大導致過擬合且收斂較慢；過小則出現欠擬合。
- **最佳使用情境**：需要在靈敏與穩健之間快速試錯的金融風險模型。

<h2 id="kernel">🔄 核技巧與非線性分類</h2>

💡 **實際應用情境：** 製造瑕疵圖像或 IoT 感測資料多呈環狀、曲線型可分，此時需要 Kernel Trick。

### 範例 3：多項式特徵 + 線性分類器
```python
# 多項式特徵把曲線拉直
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
poly_linear_clf = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    LinearSVC(C=10, dual=True, random_state=42, max_iter=15_000)
)
poly_linear_clf.fit(X_moons, y_moons)
```

**✅ 程式碼逐行解析：**
1. 第 1-4 行：生成「兩輪月亮」資料，模擬有明顯曲線邊界的感測場景。
2. 第 6-9 行：`PolynomialFeatures` 直接建構高維特徵，再透過線性 SVM 切分。
3. 第 10-11 行：`max_iter` 適度提高避免收斂警告。

**🎯 重點摘要：**
- **核心功能**：用特徵工程方式達成非線性分類。
- **潛在問題**：高次多項式易造成維度爆炸與共線性。
- **最佳使用情境**：特徵數少但需要處理弧形決策邏輯的品質檢查儀表。

### 範例 4：多項式核 `SVC`
```python
# 以 kernel="poly" 精準控制高次互動關係
from sklearn.svm import SVC
poly_kernel_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)
poly_kernel_clf.fit(X_moons, y_moons)
```

**✅ 程式碼逐行解析：**
1. 第 1-5 行：`coef0` 控制高次與低次項比重，`degree=3` 足以擬合月亮資料。
2. 第 6 行：訓練後即可直接用於推論或繪製決策邊界。

**🎯 重點摘要：**
- **核心功能**：核技巧免去手動創造特徵。
- **潛在問題**：`degree` 過高計算成本大且容易過擬合。
- **最佳使用情境**：邏輯複雜、樣本 < 1 萬筆的舊系統升級案。

<h2 id="rbf">📡 RBF 超參數調優實戰</h2>

💡 **實際應用情境：** 在布建詐欺偵測服務時，RBF 核常能兼顧準確率與彈性。

### 範例 5：探索 `gamma` 與 `C`
```python
# 建立小型網格觀察 RBF 行為
import itertools
rbf_candidates = list(itertools.product([0.1, 1, 5], [0.1, 1, 10]))
for gamma, C in rbf_candidates:
    rbf_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=gamma, C=C))
    rbf_clf.fit(X_moons, y_moons)
    score = rbf_clf.score(X_moons, y_moons)
    print(f"gamma={gamma:<4} C={C:<4} -> 訓練準確率 {score:.3f}")
```

**✅ 程式碼逐行解析：**
1. 第 1-3 行：列舉多種 `gamma` 與 `C`，掌握感受野與懲罰的交互作用。
2. 第 4-6 行：每組參數皆建立獨立 `Pipeline`，確保縮放與模型同步。
3. 第 7-8 行：即時計算訓練準確率，做為後續交叉驗證的參考。

**🎯 重點摘要：**
- **核心功能**：快速探索超參數影響。
- **潛在問題**：只觀察訓練集會高估效能，務必再以 `GridSearchCV` 驗證。
- **最佳使用情境**：專案前期需要直覺理解 `gamma` 的工程團隊。

<h2 id="regression">💡 SVM 迴歸應用</h2>

💡 **實際應用情境：** 半導體晶圓厚度、能源即時報價等連續數值，容忍小幅誤差但需偵測異常尖峰。

### 範例 6：`LinearSVR` 的 ε-不敏感區
```python
# 線性 SVR：偏好線性趨勢且希望忽略噪音
from sklearn.svm import LinearSVR
np.random.seed(42)
X_reg = 2 * np.random.rand(60, 1)
y_reg = 4 + 3 * X_reg[:, 0] + np.random.randn(60)

lin_svr = make_pipeline(
    StandardScaler(),
    LinearSVR(epsilon=0.5, dual=True, random_state=42)
)
lin_svr.fit(X_reg, y_reg)
print("訓練 RMSE:", root_mean_squared_error(y_reg, lin_svr.predict(X_reg)))
```

**✅ 程式碼逐行解析：**
1. 第 1-4 行：建立線性資料並加入雜訊模擬感測器漂移。
2. 第 6-9 行：`epsilon=0.5` 代表 ±0.5 內的誤差不計入懲罰。
3. 第 10-11 行：訓練並立即計算 RMSE，快速檢查模型是否過度受噪音影響。

**🎯 重點摘要：**
- **核心功能**：提供對離群較不敏感的迴歸器。
- **潛在問題**：`epsilon` 過小會造成支持向量極多，模型緩慢。
- **最佳使用情境**：生產線趨勢監控、需求量近似線性成長的報表。

### 範例 7：多項式核 SVR 擬合曲線
```python
# 非線性價格曲線或排程需求
from sklearn.svm import SVR
np.random.seed(42)
X_curve = 2 * np.random.rand(80, 1) - 1
y_curve = 0.2 + 0.1 * X_curve[:, 0] + 0.5 * X_curve[:, 0] ** 2 + np.random.randn(80) / 10

poly_svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
)
poly_svr.fit(X_curve, y_curve)
```

**✅ 程式碼逐行解析：**
1. 第 1-4 行：模擬二次函數趨勢並加入白噪音。
2. 第 6-9 行：`degree=2` 恰好對應曲線階次，`C=0.01` 保持強正則化。
3. 第 10 行：訓練完成後即可用於預測或視覺化。

**🎯 重點摘要：**
- **核心功能**：以少量資料擬合平滑曲線。
- **潛在問題**：`C` 過小可能欠擬合，需針對資料噪音調整。
- **最佳使用情境**：頻寬需求、用電負載等存在平滑趨勢的序列。

<h2 id="wine">🧪 多分類與調參技巧：Wine 案例</h2>

💡 **實際應用情境：** 食品化驗或化工原料分類，需要在 3 類以上中分辨供應來源。

### 範例 8：Randomized Search + `LinearSVC`
```python
# Wine 資料：示範多分類 OvR 與調參流程
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import loguniform

wine = load_wine(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

base_pipeline = make_pipeline(
    StandardScaler(),
    LinearSVC(dual=True, random_state=42, max_iter=20_000)
)

param_distrib = {"linearsvc__C": loguniform(1e-2, 1e2)}
rnd_search = RandomizedSearchCV(base_pipeline, param_distrib, n_iter=20, cv=5, random_state=42)
rnd_search.fit(X_train, y_train)
print("最佳參數:", rnd_search.best_params_)
print("測試準確率:", rnd_search.score(X_test, y_test))
```

**✅ 程式碼逐行解析：**
1. 第 1-6 行：載入 Wine 特徵並拆分訓練/測試集。
2. 第 8-12 行：`Pipeline` 自動完成縮放與線性多分類（OvR 內建）。
3. 第 14-17 行：使用對數均勻分布搜尋 `C`，平衡探索範圍。
4. 第 18-20 行：輸出最佳參數與測試集分數，供後續部署參考。

**🎯 重點摘要：**
- **核心功能**：示範如何在多分類下調整 `C`。
- **潛在問題**：若特徵未縮放會導致 `LinearSVC` 無法收斂。
- **最佳使用情境**：化工、食品或製藥領域的品管分類。

<h2 id="housing">🏙️ California Housing 迴歸挑戰</h2>

💡 **實際應用情境：** 預估台灣類似房價資料時，可參考加州公開數據建模策略。

### 範例 9：RBF SVR + 隨機搜尋
```python
# 大型資料需子抽樣加速調參
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

svr_pipeline = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
param_distrib = {
    "svr__gamma": loguniform(1e-4, 1e-1),
    "svr__C": loguniform(1, 1e3)
}
rnd_search_reg = RandomizedSearchCV(
    svr_pipeline, param_distrib, n_iter=40, cv=3, random_state=42
)
rnd_search_reg.fit(X_train[:3000], y_train[:3000])  # 以 3k 筆加速搜尋

best_model = rnd_search_reg.best_estimator_
y_pred = best_model.predict(X_test)
print("測試 RMSE (單位:10萬美金):", root_mean_squared_error(y_test, y_pred))
```

**✅ 程式碼逐行解析：**
1. 第 1-6 行：載入加州房價並拆分資料，目標值為 10 萬美元為單位。
2. 第 8-13 行：建立 RBF SVR `Pipeline` 並設定對數搜尋空間。
3. 第 14-18 行：僅抽樣 3000 筆加速超參數搜尋，再以全資料重新訓練。
4. 第 20-21 行：評估 RMSE 以千美元為單位，便於與實際預算對照。

**🎯 重點摘要：**
- **核心功能**：示範大資料下的調參策略。
- **潛在問題**：SVR 在 2 萬筆以上資料會相當緩慢，可考慮 `LinearSVR` 或樹模型。
- **最佳使用情境**：需要更平滑預測且樣本仍可抽樣的房價或能源需求。

<h2 id="faq">❓ 常見問答 (FAQ)</h2>

1. **何時選擇 `LinearSVC`、`SVC` 或 `SGDClassifier`？**  
   - 大量特徵/樣本：`LinearSVC` 或 `SGDClassifier`。  
   - 需要 kernel：`SVC`。  
   - 線上學習：`SGDClassifier` 支援 partial fit。
2. **如何取得概率？**  
   `SVC(probability=True)` 會在訓練結束後以交叉驗證擬合 Platt scaling，成本較高但可得到 `predict_proba()`。
3. **為什麼模型不收斂？**  
   通常是未縮放、`C` 過大或 `max_iter` 過低。先確認 `StandardScaler` 已放入 `Pipeline`。
4. **`gamma` 調整方向？**  
   欠擬合 → 增加 `gamma`。過擬合 → 降低 `gamma` 或 `C`。
5. **SVM 迴歸 `epsilon` 如何設定？**  
   以商業可接受誤差為基準，例如價格可容忍 ±5 萬，就設定 `epsilon=0.5` (單位 10 萬)。

<h2 id="best-practices">💼 總結與最佳實踐</h2>

- **流程化**：所有 SVM 範例均包於 `Pipeline`，避免資料洩漏並便於部署。
- **縮放優先**：無論線性或核 SVM，都先做標準化；對稀疏向量可改用 `MaxAbsScaler`。
- **交叉驗證**：小型資料建議 `StratifiedKFold`，大資料則使用 `ShuffleSplit` 加速。
- **解釋性**：線性 SVM 可檢視 `coef_` 排序，協助法遵或商務單位理解判斷依據。
- **效能取捨**：當資料量 > 5 萬筆時，優先評估線性方法或樹模型，以避免核 SVM 訓練時間過長。

<h2 id="hashtags">🏷️ 推薦標籤 (Suggested Hashtags)</h2>

#Python #SVM #支持向量機 #機器學習 #ScikitLearn #資料科學 #程式設計 #技術教學 #ThreadsTech #開發者社群
