"""
Support Vector Machines Module

This module demonstrates Support Vector Machine (SVM) for classification and regression using Scikit-Learn.

Key features:
- Linear SVM Classification with LinearSVC and SVC
- Nonlinear SVM Classification with Polynomial and RBF kernels
- SVM Regression with LinearSVR and SVR
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- Multi-class classification on Wine dataset
- Regression on California Housing dataset

Examples are based on the markdown documentation 05_support_vector_machines.md
"""

import sys
import numpy as np
from pathlib import Path
from packaging import version
import sklearn
import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_FONT_CANDIDATES = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "Heiti TC",
]

available_fonts = []
for font_name in CJK_FONT_CANDIDATES:
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
    except ValueError:
        continue
    available_fonts.append(font_name)

if not available_fonts:
    available_fonts.append("DejaVu Sans")  # guaranteed by Matplotlib

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = available_fonts
plt.rcParams["axes.unicode_minus"] = False

from sklearn.datasets import load_iris, make_moons, load_wine, fetch_california_housing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, SVR, LinearSVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    print("sklearn.metrics 中沒有 root_mean_squared_error，改用 mean_squared_error 計算 RMSE。")
    def root_mean_squared_error(labels, predictions):
        return mean_squared_error(labels, predictions, squared=False)

from scipy.stats import loguniform
import itertools


# =============================================================================
# 環境設置：版本檢查與圖像輸出配置
# =============================================================================

print("=== 環境設置：版本檢查與圖像輸出配置 ===")

# 檢查 Python 版本必須 >= 3.8
assert sys.version_info >= (3, 8), "請升級至 Python 3.8 以上"

# 檢查 Scikit-Learn 版本必須 >= 1.2.0
assert version.parse(sklearn.__version__) >= version.parse("1.2.0"), "需要較新的 Scikit-Learn"

# 建立圖像輸出資料夾
IMAGES_PATH = Path("images") / "svm"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

# 設定 Matplotlib 圖表樣式，提升可讀性
plt.rc("font", size=13)
plt.rc("axes", labelsize=13, titlesize=13)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_FONT_CANDIDATES = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "Heiti TC",
]

available_fonts = []
for font_name in CJK_FONT_CANDIDATES:
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
    except ValueError:
        continue
    available_fonts.append(font_name)

if not available_fonts:
    available_fonts.append("DejaVu Sans")  # guaranteed by Matplotlib

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = available_fonts
plt.rcParams["axes.unicode_minus"] = False 

print("環境設置完成。")
print(f"Python 版本: {sys.version}")
print(f"Scikit-Learn 版本: {sklearn.__version__}")
print(f"圖像輸出路徑: {IMAGES_PATH}\n")


# =============================================================================
# 範例 1：LinearSVC + Pipeline 建立穩健基準線
# =============================================================================

print("=== 範例 1：LinearSVC + Pipeline 建立穩健基準線 ===")

# 載入 Iris 資料集
iris = load_iris(as_frame=True)

# 選取 'petal length (cm)' 和 'petal width (cm)' 作為特徵
X = iris.data[["petal length (cm)", "petal width (cm)"]].values

# 建立二元分類目標：判斷是否為 Iris virginica (target == 2)
y = (iris.target == 2)

# 建立包含特徵縮放和 LinearSVC 的 Pipeline
# StandardScaler() 先進行特徵標準化，避免單位差異影響模型
# C=1.0 控制正則化強度，dual=True 適合樣本數 > 特徵數的情況
svm_clf = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1.0, dual=True, random_state=42)
)

# 訓練模型
svm_clf.fit(X, y)

# 準備新資料點進行預測
new_samples = [[5.5, 1.7], [5.0, 1.5]]

# 進行預測並輸出結果
predictions = svm_clf.predict(new_samples)
print(f"預測結果: {predictions}")

# 顯示決策函數分數（距離決策邊界的距離）
# 正值代表預測為 True (Iris virginica)，負值為 False
decision_scores = svm_clf.decision_function(new_samples)
print(f"決策分數: {decision_scores}")

# 視覺化決策邊界
plt.figure(figsize=(9, 4))
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris virginica")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Not Iris virginica")
plt.plot(new_samples[0][0], new_samples[0][1], "ro", markersize=10, label="New sample 1")
plt.plot(new_samples[1][0], new_samples[1][1], "mo", markersize=10, label="New sample 2")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("範例 1：LinearSVC 決策邊界")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.savefig(IMAGES_PATH / "example_01_linearsvc.png", dpi=300, bbox_inches='tight')
print(f"圖表已儲存至: {IMAGES_PATH / 'example_01_linearsvc.png'}")
plt.show()
print()


# =============================================================================
# 範例 2：調整 C 理解軟邊界
# =============================================================================

print("=== 範例 2：調整 C 理解軟邊界 ===")

# 對照不同 C 值的邊界寬度
C_grid = [0.5, 100]
models = []

for C in C_grid:
    # 每個 C 值建立獨立的 Pipeline
    model = make_pipeline(StandardScaler(), LinearSVC(C=C, dual=True, random_state=42))
    model.fit(X, y)
    models.append(model)
    
    # 利用距離 < 1 近似支持向量的比例，理解邊界緊迫程度
    # 距離決策邊界 < 1 的點被視為支持向量或靠近邊界的點
    support_vector_ratio = np.mean(np.abs(model.decision_function(X)) < 1)
    print(f"C={C}，支持向量比例估計：{support_vector_ratio:.3f}")

# 視覺化不同 C 值的影響
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for idx, (C, model) in enumerate(zip(C_grid, models)):
    ax = axes[idx]
    ax.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris virginica")
    ax.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Not Iris virginica")
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_title(f"C = {C}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

plt.suptitle("範例 2：不同 C 值對軟邊界的影響", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(IMAGES_PATH / "example_02_soft_margin.png", dpi=300, bbox_inches='tight')
print(f"圖表已儲存至: {IMAGES_PATH / 'example_02_soft_margin.png'}")
plt.show()
print()


# =============================================================================
# 範例 3：多項式特徵 + 線性分類器
# =============================================================================

print("=== 範例 3：多項式特徵 + 線性分類器 ===")

# 產生「兩輪月亮」資料集，模擬有明顯曲線邊界的場景
X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)

# 使用 PolynomialFeatures 將原始特徵轉換為高維多項式特徵
# degree=3 創建三次多項式特徵，include_bias=False 不包含截距項
# 再透過 LinearSVC 在高維空間中進行線性分類
poly_linear_clf = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    LinearSVC(C=10, dual=True, random_state=42, max_iter=15_000)
)

# 訓練模型
poly_linear_clf.fit(X_moons, y_moons)

# 計算訓練準確率
train_accuracy = poly_linear_clf.score(X_moons, y_moons)
print(f"使用多項式特徵的線性 SVM 訓練準確率：{train_accuracy:.3f}")

# 視覺化決策邊界
def plot_decision_boundary(clf, X, y, title, filename):
    """繪製決策邊界的輔助函數"""
    # 建立網格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 預測網格點
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 繪圖
    plt.figure(figsize=(9, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0][y==0], X[:, 1][y==0], c='blue', marker='o', 
                edgecolors='k', s=50, label='Class 0')
    plt.scatter(X[:, 0][y==1], X[:, 1][y==1], c='red', marker='s', 
                edgecolors='k', s=50, label='Class 1')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.savefig(IMAGES_PATH / filename, dpi=300, bbox_inches='tight')
    print(f"圖表已儲存至: {IMAGES_PATH / filename}")
    plt.show()

plot_decision_boundary(poly_linear_clf, X_moons, y_moons, 
                      "範例 3：多項式特徵 + 線性 SVM", 
                      "example_03_poly_features.png")
print()


# =============================================================================
# 範例 4：多項式核 SVC
# =============================================================================

print("=== 範例 4：多項式核 SVC ===")

# 使用 kernel="poly" 精準控制高次互動關係
# coef0=1 控制高次項與低次項的比重
# degree=3 設定多項式次數，C=5 控制正則化強度
poly_kernel_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)

# 訓練模型
poly_kernel_clf.fit(X_moons, y_moons)

# 計算訓練準確率
train_accuracy_poly = poly_kernel_clf.score(X_moons, y_moons)
print(f"使用多項式核的 SVM 訓練準確率：{train_accuracy_poly:.3f}")

# 視覺化決策邊界
plot_decision_boundary(poly_kernel_clf, X_moons, y_moons, 
                      "範例 4：多項式核 SVC (degree=3)", 
                      "example_04_poly_kernel.png")
print()


# =============================================================================
# 範例 5：探索 gamma 與 C（RBF 核）
# =============================================================================

print("=== 範例 5：探索 gamma 與 C（RBF 核）===")

# 建立小型網格觀察 RBF 行為
# gamma 控制 RBF 核的感受野，C 控制正則化強度
rbf_candidates = list(itertools.product([0.1, 1, 5], [0.1, 1, 10]))

results = []
for gamma, C in rbf_candidates:
    # 每組參數皆建立獨立 Pipeline，確保縮放與模型同步
    rbf_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=gamma, C=C))
    rbf_clf.fit(X_moons, y_moons)
    
    # 即時計算訓練準確率，做為後續交叉驗證的參考
    score = rbf_clf.score(X_moons, y_moons)
    results.append((gamma, C, score))
    print(f"gamma={gamma:<4} C={C:<4} -> 訓練準確率 {score:.3f}")

# 視覺化不同 gamma 和 C 組合的效果（選擇 4 個代表性組合）
selected_configs = [(0.1, 0.1), (0.1, 10), (5, 0.1), (5, 10)]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (gamma, C) in enumerate(selected_configs):
    rbf_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=gamma, C=C))
    rbf_clf.fit(X_moons, y_moons)
    
    # 建立網格
    x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
    y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = rbf_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # 繪圖
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    axes[idx].scatter(X_moons[:, 0][y_moons==0], X_moons[:, 1][y_moons==0], 
                     c='blue', marker='o', edgecolors='k', s=50)
    axes[idx].scatter(X_moons[:, 0][y_moons==1], X_moons[:, 1][y_moons==1], 
                     c='red', marker='s', edgecolors='k', s=50)
    axes[idx].set_xlabel("Feature 1")
    axes[idx].set_ylabel("Feature 2")
    axes[idx].set_title(f"gamma={gamma}, C={C}")
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("範例 5：不同 gamma 與 C 組合的 RBF 核效果", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(IMAGES_PATH / "example_05_rbf_grid.png", dpi=300, bbox_inches='tight')
print(f"圖表已儲存至: {IMAGES_PATH / 'example_05_rbf_grid.png'}")
plt.show()
print()


# =============================================================================
# 範例 6：LinearSVR 的 ε-不敏感區
# =============================================================================

print("=== 範例 6：LinearSVR 的 ε-不敏感區 ===")

# 建立線性資料並加入雜訊，模擬感測器漂移
np.random.seed(42)
X_reg = 2 * np.random.rand(60, 1)
y_reg = 4 + 3 * X_reg[:, 0] + np.random.randn(60)

# 建立 LinearSVR Pipeline
# epsilon=0.5 代表 ±0.5 內的誤差不計入懲罰（容忍區）
# dual=True 適合樣本數 > 特徵數的情況
lin_svr = make_pipeline(
    StandardScaler(),
    LinearSVR(epsilon=0.5, dual=True, random_state=42)
)

# 訓練模型
lin_svr.fit(X_reg, y_reg)

# 計算訓練 RMSE，快速檢查模型是否過度受噪音影響
train_rmse = root_mean_squared_error(y_reg, lin_svr.predict(X_reg))
print(f"訓練 RMSE: {train_rmse:.4f}")

# 視覺化迴歸結果與 epsilon 容忍區
X_test_plot = np.linspace(0, 2, 100).reshape(-1, 1)
y_pred_plot = lin_svr.predict(X_test_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, c='blue', s=50, alpha=0.6, edgecolors='k', label='Training data')
plt.plot(X_test_plot, y_pred_plot, 'r-', linewidth=2, label='LinearSVR prediction')
plt.fill_between(X_test_plot.ravel(), 
                 y_pred_plot - 0.5, 
                 y_pred_plot + 0.5, 
                 alpha=0.2, color='red', 
                 label=f'ε-insensitive zone (ε=0.5)')
plt.xlabel("X")
plt.ylabel("y")
plt.title("範例 6：LinearSVR 的 ε-不敏感區")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.savefig(IMAGES_PATH / "example_06_linear_svr.png", dpi=300, bbox_inches='tight')
print(f"圖表已儲存至: {IMAGES_PATH / 'example_06_linear_svr.png'}")
plt.show()
print()


# =============================================================================
# 範例 7：多項式核 SVR 擬合曲線
# =============================================================================

print("=== 範例 7：多項式核 SVR 擬合曲線 ===")

# 模擬二次函數趨勢並加入白噪音
np.random.seed(42)
X_curve = 2 * np.random.rand(80, 1) - 1
y_curve = 0.2 + 0.1 * X_curve[:, 0] + 0.5 * X_curve[:, 0] ** 2 + np.random.randn(80) / 10

# 建立使用多項式核的 SVR Pipeline
# degree=2 恰好對應曲線階次，C=0.01 保持強正則化
# epsilon=0.1 定義容忍區寬度
poly_svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
)

# 訓練模型
poly_svr.fit(X_curve, y_curve)

# 計算訓練 RMSE
train_rmse_poly = root_mean_squared_error(y_curve, poly_svr.predict(X_curve))
print(f"使用多項式核 SVR 的訓練 RMSE: {train_rmse_poly:.4f}")

# 視覺化多項式迴歸曲線
X_test_curve = np.linspace(-1, 1, 100).reshape(-1, 1)
y_pred_curve = poly_svr.predict(X_test_curve)

plt.figure(figsize=(10, 6))
plt.scatter(X_curve, y_curve, c='blue', s=50, alpha=0.6, edgecolors='k', label='Training data')
plt.plot(X_test_curve, y_pred_curve, 'r-', linewidth=2, label='Polynomial SVR (degree=2)')
plt.fill_between(X_test_curve.ravel(), 
                 y_pred_curve - 0.1, 
                 y_pred_curve + 0.1, 
                 alpha=0.2, color='red', 
                 label=f'ε-insensitive zone (ε=0.1)')
# 繪製真實的二次函數曲線
X_true = np.linspace(-1, 1, 100)
y_true = 0.2 + 0.1 * X_true + 0.5 * X_true ** 2
plt.plot(X_true, y_true, 'g--', linewidth=1.5, alpha=0.7, label='True function')
plt.xlabel("X")
plt.ylabel("y")
plt.title("範例 7：多項式核 SVR 擬合二次曲線")
plt.legend(loc="upper center")
plt.grid(True, alpha=0.3)
plt.savefig(IMAGES_PATH / "example_07_poly_svr.png", dpi=300, bbox_inches='tight')
print(f"圖表已儲存至: {IMAGES_PATH / 'example_07_poly_svr.png'}")
plt.show()
print()


# =============================================================================
# 範例 8：Randomized Search + LinearSVC（Wine 資料集）
# =============================================================================

print("=== 範例 8：Randomized Search + LinearSVC（Wine 資料集）===")

# 載入 Wine 資料集，示範多分類 OvR 與調參流程
wine = load_wine(as_frame=True)

# 拆分訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# 建立基礎 Pipeline，包含標準化與 LinearSVC
base_pipeline = make_pipeline(
    StandardScaler(),
    LinearSVC(dual=True, random_state=42, max_iter=20_000)
)

# 設定超參數搜尋空間
# 使用對數均勻分布搜尋 C，平衡探索範圍
param_distrib = {"linearsvc__C": loguniform(1e-2, 1e2)}

# 使用 RandomizedSearchCV 進行超參數搜尋
# n_iter=20 隨機嘗試 20 組參數，cv=5 使用 5 折交叉驗證
rnd_search = RandomizedSearchCV(base_pipeline, param_distrib, n_iter=20, cv=5, random_state=42)
rnd_search.fit(X_train, y_train)

# 輸出最佳參數與測試集分數
print(f"最佳參數: {rnd_search.best_params_}")
print(f"測試準確率: {rnd_search.score(X_test, y_test):.4f}")

# 視覺化超參數搜尋結果
results = rnd_search.cv_results_
C_values = [params['linearsvc__C'] for params in results['params']]
mean_scores = results['mean_test_score']
std_scores = results['std_test_score']

plt.figure(figsize=(10, 6))
plt.semilogx(C_values, mean_scores, 'bo-', linewidth=2, markersize=8, label='Mean CV score')
plt.fill_between(C_values, 
                 mean_scores - std_scores, 
                 mean_scores + std_scores, 
                 alpha=0.2, color='blue')
plt.axvline(x=rnd_search.best_params_['linearsvc__C'], 
           color='red', linestyle='--', linewidth=2, 
           label=f"Best C = {rnd_search.best_params_['linearsvc__C']:.4f}")
plt.xlabel("C (regularization parameter)")
plt.ylabel("Cross-validation score")
plt.title("範例 8：Wine 資料集 - RandomizedSearchCV 超參數調優")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(IMAGES_PATH / "example_08_wine_cv.png", dpi=300, bbox_inches='tight')
print(f"圖表已儲存至: {IMAGES_PATH / 'example_08_wine_cv.png'}")
plt.show()

# 繪製混淆矩陣
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = rnd_search.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine.target_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("範例 8：Wine 資料集分類混淆矩陣")
plt.savefig(IMAGES_PATH / "example_08_wine_confusion.png", dpi=300, bbox_inches='tight')
print(f"圖表已儲存至: {IMAGES_PATH / 'example_08_wine_confusion.png'}")
plt.show()
print()


# =============================================================================
# 範例 9：RBF SVR + 隨機搜尋（California Housing）
# =============================================================================

print("=== 範例 9：RBF SVR + 隨機搜尋（California Housing）===")

# 載入加州房價資料集
housing = fetch_california_housing()

# 拆分資料，目標值為 10 萬美元為單位
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# 建立 RBF SVR Pipeline
svr_pipeline = make_pipeline(StandardScaler(), SVR(kernel="rbf"))

# 設定超參數搜尋空間
# gamma 控制 RBF 核的感受野，C 控制正則化強度
param_distrib_svr = {
    "svr__gamma": loguniform(1e-4, 1e-1),
    "svr__C": loguniform(1, 1e3)
}

# 使用 RandomizedSearchCV 進行超參數搜尋
# 僅抽樣 3000 筆加速搜尋，n_iter=40，cv=3
rnd_search_reg = RandomizedSearchCV(
    svr_pipeline, param_distrib_svr, n_iter=40, cv=3, random_state=42
)

# 以 3000 筆資料進行超參數搜尋
rnd_search_reg.fit(X_train_h[:3000], y_train_h[:3000])

# 使用最佳模型在測試集上預測
best_model = rnd_search_reg.best_estimator_
y_pred = best_model.predict(X_test_h)

# 評估 RMSE（單位：10 萬美元）
test_rmse = root_mean_squared_error(y_test_h, y_pred)
print(f"最佳參數: {rnd_search_reg.best_params_}")
print(f"測試 RMSE (單位:10萬美金): {test_rmse:.4f}")

# 視覺化預測結果 vs 實際值
plt.figure(figsize=(10, 6))
plt.scatter(y_test_h, y_pred, alpha=0.3, s=20, edgecolors='k', linewidths=0.5)
plt.plot([y_test_h.min(), y_test_h.max()], 
         [y_test_h.min(), y_test_h.max()], 
         'r--', linewidth=2, label='Perfect prediction')
plt.xlabel("實際房價 (單位: 10萬美元)")
plt.ylabel("預測房價 (單位: 10萬美元)")
plt.title(f"範例 9：California Housing 預測結果\nTest RMSE = {test_rmse:.4f}")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.savefig(IMAGES_PATH / "example_09_housing_predictions.png", dpi=300, bbox_inches='tight')
print(f"圖表已儲存至: {IMAGES_PATH / 'example_09_housing_predictions.png'}")
plt.show()

# 視覺化預測誤差分布
residuals = y_test_h - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
plt.xlabel("預測誤差 (實際 - 預測)")
plt.ylabel("頻率")
plt.title("範例 9：California Housing 預測誤差分布")
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.savefig(IMAGES_PATH / "example_09_housing_residuals.png", dpi=300, bbox_inches='tight')
print(f"圖表已儲存至: {IMAGES_PATH / 'example_09_housing_residuals.png'}")
plt.show()
print()

print("=== 所有 SVM 範例完成 ===")

