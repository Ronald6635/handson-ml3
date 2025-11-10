"""
訓練線性模型 (Training Linear Models) 模組

本模組示範了各種線性模型的訓練技術，包含線性迴歸、梯度下降、
多項式迴歸、正規化方法以及邏輯迴歸。

主要功能:
- 線性迴歸 (正規方程式與 SVD 方法)
- 梯度下降 (批次、隨機、小批次)
- 多項式迴歸與過擬合分析
- 正規化技術 (Ridge、Lasso、Elastic Net、Early Stopping)
- 邏輯迴歸與 Softmax 迴歸

範例來自於 Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd Edition)
對應筆記本: 04_training_linear_models.ipynb
"""

# =============================================================================
# 匯入必要的函式庫
# =============================================================================
import sys
from tkinter import N  # 系統相關功能，用於版本檢查
from matplotlib.pylab import unicode_
import numpy as np  # 數值計算與陣列操作
import matplotlib.pyplot as plt  # 資料視覺化
import matplotlib as mpl  # Matplotlib 進階設定
from packaging import version  # 版本比較工具
from regex import F, T
import sklearn  # Scikit-Learn 機器學習函式庫
from pathlib import Path  # 路徑操作工具
from math import ceil  # 無條件進位函數
from copy import deepcopy  # 深度複製物件
from matplotlib import rcParams # Matplotlib 繪圖參數設定

def setup_chinese_font() -> None:
    """Configure matplotlib to display Traditional Chinese characters."""
    # Set multiple fallback fonts (matplotlib will use the first available one)
    rcParams['font.sans-serif'] = [
        'Microsoft JhengHei',  # Windows Traditional Chinese
        'Microsoft YaHei',     # Windows Simplified Chinese
        'Noto Sans TC',        # Cross-platform Traditional Chinese
        'SimHei',              # Linux/Mac Simplified Chinese
        'Arial'                # Latin fallback
    ]
    rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    
# setup_chinese_font()

# 資料前處理相關
from sklearn.preprocessing import add_dummy_feature, PolynomialFeatures, StandardScaler

# 線性模型相關
from sklearn.linear_model import (
    LinearRegression,  # 線性迴歸
    SGDRegressor,      # 隨機梯度下降迴歸器
    Ridge,             # Ridge 迴歸 (L2 正規化)
    Lasso,             # Lasso 迴歸 (L1 正規化)
    ElasticNet,        # Elastic Net (L1 + L2 正規化)
    LogisticRegression # 邏輯迴歸
)

# 模型評估與選擇相關
from sklearn.model_selection import learning_curve, train_test_split
# 為了計算 RMSE，嘗試從 sklearn.metrics 匯入 root_mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    print("sklearn.metrics 中沒有 root_mean_squared_error，改用 mean_squared_error 計算 RMSE。")
    def root_mean_squared_error(labels, predictions):
        return mean_squared_error(labels, predictions, squared=False)

# 資料集相關
from sklearn.datasets import load_iris

# 管線相關
from sklearn.pipeline import make_pipeline

# 忽略警告訊息
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 版本檢查與環境設定
# =============================================================================
# 確保 Python 版本 >= 3.7
assert sys.version_info >= (3, 7), "請使用 Python 3.7 或更新版本"
# 確保 Scikit-Learn 版本 >= 1.0.1
assert version.parse(sklearn.__version__) >= version.parse("1.0.1"), "請使用 Scikit-Learn 1.0.1 或更新版本"

# 設定 Matplotlib 繪圖參數，提升圖表的美觀度與可讀性
plt.rc('font', size=14)  # 預設字體大小
plt.rc('font', family='Microsoft JhengHei')  # 設定字體為微軟正黑體
plt.rc('axes', labelsize=14, titlesize=14)  # 座標軸標籤與標題大小
plt.rc('axes', unicode_minus=False)  # 負號正常顯示
plt.rc('legend', fontsize=14)  # 圖例字體大小
plt.rc('xtick', labelsize=10)  # x 軸刻度標籤大小
plt.rc('ytick', labelsize=10)  # y 軸刻度標籤大小

# 建立圖片儲存資料夾
IMAGES_PATH = Path() / "images" / "training_linear_models"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id: str, tight_layout: bool = True, fig_extension: str = "png", resolution: int = 300) -> None:
    """
    儲存圖表到指定路徑
    
    Args:
        fig_id: 圖表檔案名稱 (不含副檔名)
        tight_layout: 是否使用緊湊佈局
        fig_extension: 圖片格式 (預設 png)
        resolution: 解析度 (預設 300 DPI)
    """
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# =============================================================================
# 線性迴歸 (LINEAR REGRESSION)
# =============================================================================

print("\n" + "=" * 80)
print("線性迴歸 - 正規方程式 (Normal Equation)")
print("=" * 80)

# 設定隨機種子以確保結果可重現
np.random.seed(42)
m = 100  # 訓練樣本數量
X = 2 * np.random.rand(m, 1)  # 生成 100 個介於 0 到 2 之間的隨機數作為特徵

# 根據線性方程式 y = 4 + 3x + noise 生成目標值
y = 4 + 3 * X + np.random.randn(m, 1)  # 加入高斯雜訊

print(f"X 的形狀: {X.shape}")
print(f"X 的前 10 個樣本:\n{X[:10]}")
print(f"\ny 的形狀: {y.shape}")
print(f"y 的前 10 個樣本:\n{y[:10]}")

# 繪製生成的資料
plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")  # 藍色點
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis((0, 2, 0, 15))
plt.grid()
save_fig("generated_data_plot")
plt.show()

# 使用正規方程式計算最佳參數
# X_b 是在 X 前面添加一行 1 (偏差項 x0 = 1)
X_b = add_dummy_feature(X)

print(f"\n檢視添加偏差項前後的資料:")
print(f"X (前 10 個樣本):\n{X[:10]}")
print(f"\nX_b (前 10 個樣本，已添加 x0=1):\n{X_b[:10]}")

# 正規方程式: θ = (X^T · X)^(-1) · X^T · y
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"\n使用正規方程式計算的最佳參數:")
print(f"theta_best = \n{theta_best}")
print(f"θ₀ (截距): {theta_best[0][0]:.4f}")
print(f"θ₁ (斜率): {theta_best[1][0]:.4f}")

# 使用計算出的參數進行預測
X_new = np.array([[0], [2]])  # 新的資料點
X_new_b = add_dummy_feature(X_new)
y_predict = X_new_b @ theta_best

print(f"\n對新資料點的預測:")
print(f"X_new:\n{X_new}")
print(f"X_new_b (已添加 x0=1):\n{X_new_b}")
print(f"Shape of X_new: {X_new.shape}")
print(f"Shape of X_new_b: {X_new_b.shape}")
print(f"y_predict:\n{y_predict}")
print(f"x = 0 時，預測 y = {y_predict[0][0]:.4f}")
print(f"x = 2 時，預測 y = {y_predict[1][0]:.4f}")

# 繪製預測結果
plt.figure(figsize=(6, 4))
plt.plot(X_new, y_predict, "r-", linewidth=2, label="預測線")
plt.plot(X, y, "b.", label="訓練資料")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis((0, 2, 0, 15))
plt.grid()
plt.legend(loc="upper left")
save_fig("linear_model_predictions_plot")
plt.show()

# =============================================================================
# 使用 Scikit-Learn 的 LinearRegression
# ============================================================================ 
print("\n" + "=" * 80)
print("線性迴歸 - 使用 Scikit-Learn")
print("=" * 80)

# 使用 Scikit-Learn 的 LinearRegression 類別
# 內部使用 SVD (奇異值分解) 方法，比正規方程式更有效率且數值穩定
lin_reg = LinearRegression()
lin_reg.fit(X, y)

print(f"Scikit-Learn 計算的參數:")
print(f"截距 (intercept_): {lin_reg.intercept_[0]:.4f}")
print(f"係數 (coef_): {lin_reg.coef_[0][0]:.4f}")
print(f"\n對新資料點 {X_new} 的預測: {lin_reg.predict(X_new).ravel()}")
print(f'Note: the shape of intercept_ is {lin_reg.intercept_.shape}, coef_ is {lin_reg.coef_.shape}')

# 直接使用 numpy 的 lstsq 函數 (使用 SVD 分解的最小平方法)
# lstsq = Least Squares，內部使用 SVD (Singular Value Decomposition) 來求解
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(f"\n使用 SVD 方法計算的參數:")
print(f"θ = {theta_best_svd.ravel()}")
# 返回值說明: residuals=殘差平方和, rank=矩陣秩, s=奇異值

# 使用 pseudoinverse (偽逆矩陣/Moore-Penrose 逆矩陣)
# 偽逆矩陣也是基於 SVD 分解實現的，適用於奇異或非方陣
theta_pseudoinverse = np.linalg.pinv(X_b) @ y
print(f"\n使用偽逆矩陣計算的參數:")
print(f"θ = {theta_pseudoinverse.ravel()}")

# 比較以上三種方法計算的參數是否相同
print("\n比較三種方法計算的參數是否相同:")
print(f"正規方程式 vs SVD 相同: {np.allclose(theta_best, theta_best_svd)}")
print(f"正規方程式 vs 偽逆矩陣 相同: {np.allclose(theta_best, theta_pseudoinverse)}")
print(f"SVD vs 偽逆矩陣 相同: {np.allclose(theta_best_svd, theta_pseudoinverse)}")

# =============================================================================
# 梯度下降 (GRADIENT DESCENT)
# =============================================================================

print("\n" + "=" * 80)
print("批次梯度下降 (Batch Gradient Descent)")
print("=" * 80)

# 批次梯度下降的超參數
eta = 0.1  # 學習率 (learning rate)
n_epochs = 1000  # 迭代次數 (epochs)
m = len(X_b)  # 訓練樣本數量

# 隨機初始化模型參數
np.random.seed(42)
theta = np.random.randn(2, 1)
print(f"初始參數:\n{theta}")

# 儲存訓練過程中的梯度和成本歷史
gradient_history = []
cost_history = []

# 批次梯度下降訓練迴圈
for epoch in range(n_epochs):
    # 計算梯度: 2/m * X^T · (X·θ - y)
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    # 更新參數: θ = θ - η · ∇J(θ)
    theta = theta - eta * gradients
    
    # 記錄梯度
    gradient_history.append(gradients)
    
    # 計算均方誤差 (MSE)
    cost = (1/m) * np.sum((X_b @ theta - y) ** 2)
    cost_history.append(cost)

print(f"\n訓練完成後的參數:")
print(f"θ = {theta.ravel()}")

# 轉換梯度歷史為 numpy 陣列以便繪圖
gradient_history = np.array(gradient_history)
print(f"\n梯度歷史的形狀: {gradient_history.shape}")

# 重塑梯度歷史資料以便繪圖 (從 (1000, 2, 1) 到 (1000, 2))
y_to_plot = gradient_history.squeeze(-1) if gradient_history.ndim == 3 else gradient_history.reshape(gradient_history.shape[0], -1)
print(f'重塑後梯度歷史的形狀: {y_to_plot.shape}')

# 繪製梯度收斂圖
plt.figure(figsize=(10, 6))
plt.plot(y_to_plot)
plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("梯度 (Gradient)")
plt.title("參數收斂過程 (Convergence of Parameters)")
plt.legend([r"$\theta_0$", r"$\theta_1$"])
plt.grid()
plt.show()

# 繪製最後 100 次迭代的梯度變化 (放大檢視)
start_iter = 900
plt.figure(figsize=(10, 6))

plt.plot(np.arange(start_iter, len(y_to_plot)), y_to_plot[start_iter:, 0], 'b-', label=r"$\theta_0$")
plt.plot(np.arange(start_iter, len(y_to_plot)), y_to_plot[start_iter:, 1], 'r-', label=r"$\theta_1$")

# 在終點添加標記點並標註
final_theta0 = y_to_plot[-1, 0]
final_theta1 = y_to_plot[-1, 1]
final_iter = len(y_to_plot) - 1

# 繪製終點標記
plt.plot(final_iter, final_theta0, 'bo', markersize=8, label=r'$\theta_0$ 終值: {:.6e}'.format(final_theta0))
plt.plot(final_iter, final_theta1, 'ro', markersize=8, label=r'$\theta_1$ 終值: {:.6e}'.format(final_theta1))

plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("梯度 (Gradient)")
plt.title("參數收斂過程 (放大) - Zoomed In")
plt.legend(loc='best')
plt.grid()
plt.show()

# 繪製成本函數收斂圖
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("成本 (Cost - MSE)")
plt.title("批次梯度下降的成本函數收斂")
plt.grid()
plt.show()

# 繪製成本函數收斂圖 (放大檢視後期)
cost_start_iter = 300
plt.figure(figsize=(10, 6))
plt.plot(np.arange(cost_start_iter, len(cost_history)), cost_history[cost_start_iter:], "g-")
plt.xlabel("迭代次數 (Iterations)")
plt.ylabel("成本 (Cost - MSE)")
plt.title("批次梯度下降的成本函數收斂 (放大)")
# plt.axis([cost_start_iter, n_epochs, 0, cost_history[cost_start_iter]*1.1])     
plt.axis((cost_start_iter, n_epochs, min(cost_history[cost_start_iter:])*0.99, max(cost_history[cost_start_iter:])*1.01))
plt.grid()
plt.show()

# 繪製不同學習率的比較圖
def plot_gradient_descent(theta, eta):
    """
    繪製梯度下降過程中的預測線變化
    
    Args:
        theta: 初始參數
        eta: 學習率
    
    Returns:
        theta_path: 參數的更新路徑
    """
    m = len(X_b)
    plt.plot(X, y, "b.")  # 繪製訓練資料點
    n_epochs = 1000
    n_shown = 20  # 只顯示前 20 條預測線
    theta_path = []
    
    for epoch in range(n_epochs):
        if epoch < n_shown:
            y_predict = X_new_b @ theta
            # Use get_cmap() to retrieve the colormap object by name
            color = mpl.colors.rgb2hex(plt.cm.get_cmap("OrRd")(epoch / n_shown + 0.15))
            plt.plot(X_new, y_predict, linestyle="solid", color=color)
        gradients = 2 / m * X_b.T @ (X_b @ theta - y)
        theta = theta - eta * gradients
        theta_path.append(theta)
    
    plt.xlabel("$x_1$")
    plt.axis((0, 2, 0, 15))
    plt.grid()
    plt.title(fr"$\eta = {eta}$")
    return theta_path

# 重新初始化參數
np.random.seed(42)
theta = np.random.randn(2, 1)

# 比較三種不同學習率的效果
plt.figure(figsize=(10, 4))

plt.subplot(131)
plot_gradient_descent(theta, eta=0.02)  # 學習率太小
plt.ylabel("$y$", rotation=0)

plt.subplot(132)
theta_path_bgd = plot_gradient_descent(theta, eta=0.1)  # 適當的學習率
plt.gca().axes.yaxis.set_ticklabels([])

plt.subplot(133)
plot_gradient_descent(theta, eta=0.5)  # 學習率太大
plt.gca().axes.yaxis.set_ticklabels([])

save_fig("gradient_descent_plot")
plt.show()

# =============================================================================
# 隨機梯度下降 (STOCHASTIC GRADIENT DESCENT; SGD)
# =============================================================================
print("\n" + "=" * 80)
print("隨機梯度下降 (Stochastic Gradient Descent - SGD)")
print("=" * 80)

# 初始化儲存 SGD 訓練路徑的變數
theta_path_sgd = []  # 參數更新路徑
eta_path_sgd = []    # 學習率變化路徑
cost_path_sgd = []   # 成本變化路徑

# SGD 超參數
n_epochs = 50
t0, t1 = 5, 50  # 學習率排程的超參數

def learning_schedule(t):
    """
    學習率排程函數，隨時間遞減學習率
    
    Args:
        t: 當前迭代步數
    
    Returns:
        當前學習率
    """
    return t0 / (t + t1)

# 隨機初始化參數
np.random.seed(42)
theta = np.random.randn(2, 1)

n_shown = 20  # 只繪製前 20 條預測線
plt.figure(figsize=(6, 4))

# SGD 訓練迴圈
for epoch in range(n_epochs):
    for iteration in range(m):
        # 繪製前 20 條預測線 (僅在第一個 epoch)
        if epoch == 0 and iteration < n_shown:
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.get_cmap("OrRd")(iteration / n_shown + 0.15))
            plt.plot(X_new, y_predict, color=color)
        
        # 隨機選擇一個訓練樣本
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]  # 單一樣本的特徵
        yi = y[random_index:random_index + 1]    # 單一樣本的目標值
        
        # 計算該樣本的梯度 (注意：不除以 m)
        gradients = 2 * xi.T @ (xi @ theta - yi)
        
        # 根據學習率排程更新學習率
        eta = learning_schedule(epoch * m + iteration)
        
        # 更新參數
        theta = theta - eta * gradients
        
        # 計算並記錄當前的成本 (使用整個訓練集)
        cost = (1/m) * np.sum((X_b @ theta - y) ** 2)
        
        # 記錄訓練過程
        theta_path_sgd.append(theta)
        eta_path_sgd.append(eta)
        cost_path_sgd.append(cost)

# 完成 SGD 繪圖
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis((0, 2, 0, 15))
plt.grid()
save_fig("sgd_plot")
plt.show()

print(f"\nSGD 訓練完成後的參數:")
print(f"θ = {theta.ravel()}")

print(f"使用正規方程式計算的最佳參數:")
print(f"θ = {theta_best.ravel()}")

# 將 SGD 路徑轉換為 numpy 陣列
theta_path_sgd = np.array(theta_path_sgd)
eta_path_sgd = np.array(eta_path_sgd)

# 重塑參數路徑以便繪圖
from numpy import shape as np_shape
print(f'SGD 參數路徑的形狀: {np_shape(theta_path_sgd)}')
theta_path_sgd_reshaped = theta_path_sgd.squeeze(-1) if theta_path_sgd.ndim == 3 else theta_path_sgd.reshape(theta_path_sgd.shape[0], -1)
print(f'重塑後 SGD 參數路徑的形狀: {np_shape(theta_path_sgd_reshaped)}')

# 繪製參數空間中的 SGD 路徑
plt.figure(figsize=(8, 6))
plt.plot(theta_path_sgd_reshaped[:, 0], theta_path_sgd_reshaped[:, 1], "r-s", linewidth=1, markersize=2, alpha=0.5)
plt.plot(theta_path_sgd_reshaped[-1, 0], theta_path_sgd_reshaped[-1, 1], "go", markersize=10, label="最終參數 (Final Parameters)")
plt.plot(theta_best[0], theta_best[1], "b*", markersize=15, label="最佳參數 (Optimal Parameters)")
plt.legend(loc="best")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$", rotation=0)
plt.title("SGD 在參數空間中的路徑 (Path in Parameter Space)")
plt.grid()
plt.show()

# 繪製參數空間中的 SGD 路徑
plt.figure(figsize=(8, 6))
plt.plot(theta_path_sgd_reshaped[:, 0], theta_path_sgd_reshaped[:, 1], "r-s", linewidth=1, markersize=2, alpha=0.5)
plt.plot(theta_path_sgd_reshaped[-1, 0], theta_path_sgd_reshaped[-1, 1], "go", markersize=10, label="最終參數 (Final Parameters)")
plt.plot(theta_best[0], theta_best[1], "b*", markersize=15, label="最佳參數 (Optimal Parameters)")
plt.legend(loc="best")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$", rotation=0)
plt.title("SGD 在參數空間中的路徑 (Path in Parameter Space): 放大檢視")
plt.axis((theta_best[0, 0]-0.3, theta_best[0, 0]+0.3, theta_best[1, 0]-0.3, theta_best[1, 0]+0.3))
plt.grid()
plt.show()


# 繪製學習率排程圖
print(f"學習率路徑的形狀: {eta_path_sgd.shape}")
plt.figure(figsize=(8, 6))
plt.plot(eta_path_sgd, "b-", linewidth=1, alpha=0.7)
plt.xlabel("迭代次數 (Iteration)")
plt.ylabel(r"學習率 ($\eta$)")
plt.title("SGD 學習率排程 (Learning Rate Schedule)")
plt.grid()
plt.show()

# 繪製 SGD 的成本函數收斂圖
start_iter = 50  # 跳過前幾次迭代以獲得更好的視覺效果
plt.figure(figsize=(8, 6))
plt.plot(np.arange(start_iter, len(cost_path_sgd)), cost_path_sgd[start_iter:], "g-", linewidth=1, alpha=0.7)
plt.xlabel("迭代次數 (Iteration)")
plt.ylabel("成本 (Cost)")
plt.title("SGD 成本函數收斂")
plt.grid()
plt.show()

# =============================================================================
# 使用 Scikit-Learn 的 SGDRegressor
# =============================================================================
print("\n使用 Scikit-Learn 的 SGDRegressor:")
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
print(f'y 在 ravel() 前的形狀: {y.shape}')
sgd_reg.fit(X, y.ravel())  # fit() 需要 1D 目標值
print(f'y 在 ravel() 後的形狀: {y.ravel().shape}')

print(f"\nSGDRegressor 的參數:")
print(f"截距: {sgd_reg.intercept_[0]:.4f}")
print(f"係數: {sgd_reg.coef_[0]:.4f}")

print("比較 Scikit-Learn SGDRegressor 與 自行實作 SGD 的參數:")
print(f"\nSGD 訓練完成後的參數:")
print(f"截距: {theta[0][0]:.4f}")
print(f"係數: {theta[1][0]:.4f}")

# =============================================================================
# 小批次梯度下降 (MINI-BATCH GRADIENT DESCENT; MGD)
# =============================================================================
print("\n" + "=" * 80)
print("小批次梯度下降 (Mini-batch Gradient Descent - MGD)")
print("=" * 80)

# 小批次梯度下降的超參數
n_epochs = 50
minibatch_size = 20 # 小批次大小
n_batches_per_epoch = ceil(m / minibatch_size) # 每個 epoch 的批次數量
print(f'訓練樣本數量: {m}')
print(f'小批次大小: {minibatch_size}')
print(f'每個 epoch 的批次數量: {n_batches_per_epoch}')

# 隨機初始化參數
np.random.seed(42)
theta = np.random.randn(2, 1)

# 學習率排程的超參數 (與 SGD 不同)
t0, t1 = 200, 1000

# def learning_schedule(t):
#     """小批次梯度下降的學習率排程"""
#     return t0 / (t + t1)

# 儲存訓練路徑
theta_path_mgd = []
cost_path_mgd = []
eta_path_mgd = []

# 小批次梯度下降訓練迴圈
for epoch in range(n_epochs):
    # 在每個 epoch 開始時打亂資料
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]  # 打亂特徵
    y_shuffled = y[shuffled_indices]      # 打亂目標值 (保持對應關係)
    
    # 遍歷所有小批次
    for iteration in range(0, n_batches_per_epoch):
        idx = iteration * minibatch_size  # 小批次的起始索引
        xi = X_b_shuffled[idx:idx + minibatch_size]  # 小批次的特徵
        yi = y_shuffled[idx:idx + minibatch_size]    # 小批次的目標值
        
        # 計算小批次的梯度 (注意：除以 minibatch_size)
        gradients = 2 / minibatch_size * xi.T @ (xi @ theta - yi)
        
        # 更新學習率
        eta = learning_schedule(epoch * n_batches_per_epoch + iteration)
        
        # 更新參數
        theta = theta - eta * gradients
        
        # 記錄訓練過程
        theta_path_mgd.append(theta)
        cost = (1/m) * np.sum((X_b @ theta - y) ** 2)
        cost_path_mgd.append(cost)
        eta_path_mgd.append(eta)

print(f"\n小批次梯度下降訓練完成後的參數:")
print(f"θ = {theta.ravel()}")

# 轉換路徑為 numpy 陣列
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

# 比較三種梯度下降方法在參數空間中的路徑
plt.figure(figsize=(7, 4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="隨機 (Stochastic)")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="小批次 (Mini-batch)")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="批次 (Batch)")
plt.title("三種梯度下降方法在參數空間中的路徑")
plt.legend(loc="upper left")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$   ", rotation=0)
plt.axis((2.6, 4.6, 2.3, 3.4))
plt.grid()
save_fig("gradient_descent_paths_plot")
plt.show()

# 比較三種梯度下降方法在參數空間中的路徑: 放大檢視
plt.figure(figsize=(10, 6))
plt.plot(theta_path_sgd[-20:, 0], theta_path_sgd[-20:, 1], "r-s", linewidth=1, label="隨機 (Stochastic)")
plt.plot(theta_path_mgd[-20:, 0], theta_path_mgd[-20:, 1], "g-+", linewidth=2, label="小批次 (Mini-batch)")
plt.plot(theta_path_bgd[:,0], theta_path_bgd[:,1], "b-o", linewidth=3, label="批次 (Batch)")
plt.plot(theta_best[0], theta_best[1], "k*", markersize=15, label="最佳參數 (Optimal Parameters)")
plt.plot(theta_path_bgd[-1, 0], theta_path_bgd[-1, 1], "b*", markersize=10, label="批次 最終參數")
plt.plot(theta_path_sgd[-1, 0], theta_path_sgd[-1, 1], "r*", markersize=10, label="隨機 最終參數")
plt.plot(theta_path_mgd[-1, 0], theta_path_mgd[-1, 1], "g*", markersize=10, label="小批次 最終參數")
plt.title("三種梯度下降方法在參數空間中的路徑 (放大檢視)")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$   ", rotation=0)
plt.axis((theta_best[0, 0]-0.1, theta_best[0, 0]+0.1, theta_best[1, 0]-0.1, theta_best[1, 0]+0.1))
plt.grid()
save_fig("gradient_descent_paths_plot_magnified")
plt.show()

# 繪製小批次梯度下降的成本函數收斂圖
print(f"成本路徑的長度: {len(cost_path_mgd)}")
start_iter = 5
plt.figure(figsize=(8, 6))
plt.plot(np.arange(start_iter, len(cost_path_mgd)), cost_path_mgd[start_iter:], "g-", linewidth=1, alpha=0.7)
plt.xlabel("迭代次數 (Iteration)")
plt.ylabel("成本 (Cost)")
plt.title("小批次梯度下降的成本函數收斂")
plt.grid()
plt.show()

# 繪製小批次梯度下降的學習率收斂圖
print(f"學習率路徑的形狀: {len(eta_path_mgd)}")
plt.figure(figsize=(8, 6))
plt.plot(eta_path_mgd, "b-", linewidth=1, alpha=0.7)
plt.xlabel("迭代次數 (Iteration)")
plt.ylabel(r"學習率 ($\eta$)")
plt.title("小批次梯度下降的學習率排程")
plt.grid()
plt.show()

# =============================================================================
# 多項式迴歸 (POLYNOMIAL REGRESSION)
# =============================================================================

print("\n" + "=" * 80)
print("多項式迴歸 (Polynomial Regression)")
print("=" * 80)

# 生成二次方程式的資料集
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3  # 範圍 [-3, 3)

# 真實函數: y = 0.5*x^2 + x + 2 + 雜訊
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# 繪製二次資料
plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis((-3, 3, 0, 10))
plt.grid()
save_fig("quadratic_data_plot")
plt.show()

# 使用 PolynomialFeatures 建立多項式特徵
# degree=2 表示建立到 2 次方的特徵: [x, x^2]
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print(f"\n原始特徵 X[0]: {X[0]}")
print(f"多項式特徵 X_poly[0]: {X_poly[0]}")  # [x, x^2]

# 在多項式特徵上訓練線性迴歸模型
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

print(f"\n多項式迴歸的參數:")
print(f"截距 (θ₀): {lin_reg.intercept_[0]:.4f}")
print(f"係數 (θ₁, θ₂): {lin_reg.coef_[0]}")
print(f"  θ₁ (x 的係數): {lin_reg.coef_[0][0]:.4f}")
print(f"  θ₂ (x² 的係數): {lin_reg.coef_[0, 1]:.4f}")

# 生成平滑的預測曲線
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

# 繪製預測結果
plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.", label="訓練資料")
plt.plot(X_new, y_new, "r-", linewidth=2, label="預測曲線")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.legend(loc="upper left")
plt.axis((-3, 3, 0, 10))
plt.grid()
save_fig("quadratic_predictions_plot")
plt.show()

# 比較不同次數的多項式迴歸
print("\n比較不同次數的多項式迴歸:")
plt.figure(figsize=(6, 4))

for style, width, degree in (("k-+", 2, 1), ("r--", 2, 2), ("g-", 1, 300)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()  # 標準化對高次多項式很重要
    lin_reg = LinearRegression()
    # 建立管線：多項式特徵 -> 標準化 -> 線性迴歸
    polynomial_regression = make_pipeline(polybig_features, std_scaler, lin_reg)
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    label = f"{degree} 次" if degree > 1 else f"{degree} 次 (線性)"
    plt.plot(X_new, y_newbig, style, label=label, linewidth=width)
    
    if degree == 1:
        print(f"{degree} 次多項式 (線性) - 欠擬合")
    elif degree == 2:
        print(f"{degree} 次多項式 - 良好擬合")
    elif degree == 300:
        print(f"{degree} 次多項式 - 過擬合")

plt.plot(X, y, "b.", linewidth=3, label="訓練資料")
plt.legend(loc="upper left")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis((-3, 3, 0, 10))
plt.grid()
save_fig("high_degree_polynomials_plot")
plt.show()

# =============================================================================
# 學習曲線 (LEARNING CURVES)
# =============================================================================

print("\n" + "=" * 80)
print("學習曲線 (Learning Curves)")
print("=" * 80)

# 使用學習曲線診斷模型效能
# 簡單線性迴歸模型 (會欠擬合)
lin_reg = LinearRegression()
train_sizes, train_scores, valid_scores, fit_times, score_times = learning_curve(
    lin_reg, X, y, 
    train_sizes=np.linspace(0.01, 1.0, 40),  # 使用不同比例的訓練資料
    cv=5,  # 5 折交叉驗證
    scoring="neg_root_mean_squared_error",  # 使用 RMSE 作為評分指標
    return_times=True # 返回訓練和評分時間
)

# 計算平均誤差 (轉為正值)
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

# 繪製學習曲線
plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="訓練集")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="驗證集")
plt.xlabel("訓練集大小 (Training set size)")
plt.ylabel("RMSE")
plt.grid()
plt.legend(loc="upper right")
plt.axis((0, 80, 0, 2.5))
plt.title("簡單線性模型 - 欠擬合 (Underfitting)")
save_fig("underfitting_learning_curves_plot")
plt.show()

print("簡單線性模型的學習曲線特徵:")
print("- 訓練誤差和驗證誤差都很高且接近")
print("- 表示模型過於簡單，無法捕捉資料的複雜性 (欠擬合)")
print("- 增加更多訓練資料無法顯著改善效能")
# 10 次多項式迴歸模型 (可能過擬合)
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    StandardScaler(),
    LinearRegression()
)

train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y,
    train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5,
    scoring="neg_root_mean_squared_error"
)

# 計算平均誤差 (轉為正值)
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

# 繪製 10 次多項式的學習曲線
plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="訓練集")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="驗證集")
plt.legend(loc="upper right")
plt.xlabel("訓練集大小 (Training set size)")
plt.ylabel("RMSE")
plt.grid()
plt.axis((0, 80, 0, 2.5))
plt.title("10 次多項式模型 - 過擬合 (Overfitting)")
save_fig("learning_curves_plot")
plt.show()

print("\n10 次多項式模型的學習曲線特徵:")
print("- 訓練誤差很低，但驗證誤差較高")
print("- 兩者之間有明顯差距")
print("- 表示模型在訓練集上擬合得太好，但泛化能力差 (過擬合)")
print("- 增加訓練資料可能有助於改善驗證效能")

# 2次多項式迴歸模型 (良好擬合)
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LinearRegression()
)
train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y,
    train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5,
    scoring="neg_root_mean_squared_error"
)
# 計算平均誤差 (轉為正值)
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)
# 繪製 2 次多項式的學習曲線
plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="訓練集")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="驗證集")
plt.legend(loc="upper right")
plt.xlabel("訓練集大小 (Training set size)")
plt.ylabel("RMSE")
plt.grid()
plt.axis((0, 80, 0, 2.5))
plt.title("2 次多項式模型 - 良好擬合 (Good Fit)")
save_fig("good_fit_learning_curves_plot")
plt.show()

print("\n2 次多項式模型的學習曲線特徵:")
print("- 訓練誤差和驗證誤差都較低且接近")
print("- 表示模型能夠適當擬合資料且具有良好的泛化能力")

# =============================================================================
# 正規化線性模型 (REGULARIZED LINEAR MODELS)
# =============================================================================

print("\n" + "=" * 80)
print("Ridge 迴歸 (L2 正規化)")
print("=" * 80)

# 生成新的資料集用於正規化示範
np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1) # 範圍 [0, 3)
# 真實函數: y = 1 + 0.5*x + 雜訊
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1) # 用於預測的平滑輸入

# Ridge 迴歸 (L2 正規化)
# alpha 控制正規化強度，alpha 越大，正規化程度越高
# solver="cholesky" 使用解析解法求解
ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)

print(f"Ridge 迴歸對 x=1.5 的預測: {ridge_reg.predict(np.array([[1.5]]))[0][0]:.4f}")
print(f"Ridge 迴歸截距: {ridge_reg.intercept_[0]:.4f}")
print(f"Ridge 迴歸係數: {ridge_reg.coef_.ravel()[0]:.4f}")

# 使用 SGD 實現 Ridge 迴歸
# penalty="l2" 表示 L2 正規化
sgd_reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None,
                       max_iter=1000, eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())

print(f"\nSGD Ridge 迴歸對 x=1.5 的預測: {sgd_reg.predict(np.array([[1.5]]))[0]:.4f}")
print(f"SGD Ridge 迴歸截距: {sgd_reg.intercept_[0]:.4f}")
print(f"SGD Ridge 迴歸係數: {sgd_reg.coef_[0]:.4f}")

print("\n" + "=" * 80)
print("Lasso 迴歸 (L1 正規化)")
print("=" * 80)

# Lasso 迴歸 (L1 正規化)
# L1 正規化可以將某些係數完全歸零，達到特徵選擇的效果
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)

print(f"Lasso 迴歸對 x=1.5 的預測: {lasso_reg.predict(np.array([[1.5]]))[0]:.4f}")
print(f"Lasso 迴歸截距: {lasso_reg.intercept_[0]:.4f}")
print(f"Lasso 迴歸係數: {lasso_reg.coef_.ravel()[0]:.4f}")

print("\n" + "=" * 80)
print("Elastic Net (L1 + L2 正規化)")
print("=" * 80)

# Elastic Net 結合 L1 和 L2 正規化
# l1_ratio 控制 L1 和 L2 的混合比例
# l1_ratio=0 相當於 Ridge，l1_ratio=1 相當於 Lasso
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)

print(f"Elastic Net 對 x=1.5 的預測: {elastic_net.predict(np.array([[1.5]]))[0]:.4f}")
print(f"Elastic Net 截距: {elastic_net.intercept_[0]:.4f}")
print(f"Elastic Net 係數: {elastic_net.coef_[0]:.4f}")
print(f"l1_ratio=0.5 表示 L1 和 L2 正規化各佔 50%")

# 使用 SGD 實現 Elastic Net 迴歸
# penalty="elasticnet" 表示 Elastic Net 正規化
sgd_en_reg = SGDRegressor(penalty="elasticnet", alpha=0.1 / m, tol=None,
                       max_iter=1000, eta0=0.01, random_state=42)
sgd_en_reg.fit(X, y.ravel())

print(f"\nSGD Elastic Net 迴歸對 x=1.5 的預測: {sgd_en_reg.predict(np.array([[1.5]]))[0]:.4f}")
print(f"SGD Elastic Net 迴歸截距: {sgd_en_reg.intercept_[0]:.4f}")
print(f"SGD Elastic Net 迴歸係數: {sgd_en_reg.coef_[0]:.4f}")

# =============================================================================
# Early Stopping
# =============================================================================
print("\n" + "=" * 80)
print("Early Stopping")
print("=" * 80)

# 重新生成二次方程式資料集
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3 # 範圍 [-3, 3)
# 真實函數: y = 0.5*x^2 + x + 2 + 雜訊
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# 分割訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, shuffle=False)
print(f"Shape of X_train: {X_train.shape}, X_val: {X_val.shape}")
print(f"Shape of y_train: {y_train.shape}, y_val: {y_val.shape}")
y_train = y_train.ravel()
y_val = y_val.ravel()

print(f"訓練集大小: {len(X_train)}")
print(f"驗證集大小: {len(X_val)}")

# 建立預處理管線 (90次多項式 + 標準化)
preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_val_prep = preprocessing.transform(X_val)

# 初始化 SGD 迴歸器
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_val_rmse = float('inf')
train_errors, val_errors = [], []
best_model = None # initialize best model
initial_model = None  # 用於儲存初始模型

# 手動實現 Early Stopping
for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)  # 增量訓練
    y_val_predict = sgd_reg.predict(X_val_prep) # 在驗證集上進行預測
    val_error = root_mean_squared_error(y_val, y_val_predict) # 計算驗證集RMSE
    val_errors.append(val_error)
    
    # 如果找到更好的模型，就儲存它
    if val_error < best_val_rmse:
        best_val_rmse = val_error
        best_model = deepcopy(sgd_reg) # 深拷貝最佳模型參數

    if epoch == 0:
        initial_model = deepcopy(sgd_reg)  # 儲存初始模型

    # 記錄訓練和驗證誤差以供繪圖
    y_train_predict = sgd_reg.predict(X_train_prep)
    train_error = root_mean_squared_error(y_train, y_train_predict) # 計算訓練集的RMSE
    train_errors.append(train_error)

# 找到最佳 epoch
best_epoch = np.argmin(val_errors)
print(f"\n最佳 Epoch: {best_epoch}, 對應的驗證 RMSE: {val_errors[best_epoch]:.4f}")
print(f"最佳驗證 RMSE: {best_val_rmse:.4f}")

# 繪製 Early Stopping 圖
plt.figure(figsize=(6, 4))
plt.annotate('最佳模型',
             xy=(best_epoch, best_val_rmse),
             xytext=(best_epoch, best_val_rmse + 0.5),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
plt.plot(val_errors, "b-", linewidth=3, label="驗證集")
plt.plot(best_epoch, best_val_rmse, "bo")
plt.plot(train_errors, "r--", linewidth=2, label="訓練集")
plt.legend(loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.axis((0, n_epochs, 0, 3.5))
plt.grid()
save_fig("early_stopping_plot")
plt.show()

# 檢視最佳模型的係數
print(f"\n最佳模型的截距: {best_model.intercept_[0]:.4f}")
print(f"最佳模型的係數數量: {len(best_model.coef_)}")

# 繪製最佳模型的係數大小
plt.figure(figsize=(8, 4))
plt.plot(best_model.coef_, "bo")
plt.xlabel("係數索引")
plt.ylabel("係數大小")
plt.title("Early Stopping 找到的最佳模型係數")
plt.grid()
plt.show()

# 繪製最佳模型的預測結果
X_new = np.linspace(-3, 3, 300).reshape(300, 1)
X_new_prep = preprocessing.transform(X_new)
y_new = best_model.predict(X_new_prep)

plt.figure(figsize=(6, 4))
plt.plot(X_new, y_new, "r-", label="預測結果")
plt.scatter(X, y, s=10, label="訓練資料")
plt.xlabel("X")
plt.ylabel("y", rotation=0)
plt.axis((-3, 3, 0, 10))
save_fig("early_stopping_predictions_plot")
plt.title("最佳模型的預測結果, RMSE: {:.4f}".format(best_val_rmse))
plt.legend()
plt.grid()
plt.show()

# Predict with the initial model for comparison
y_ini = initial_model.predict(X_new_prep)

# Plot predictions vs. training data
plt.figure(figsize=(6, 4))
plt.plot(X_new, y_new, "r-", label="Predictions")
plt.plot(X_new, y_ini, "g--", label="Initial model") # optional: plot initial model
plt.scatter(X, y, s=10, label="Training data")
plt.xlabel("X")
plt.ylabel("y", rotation=0)
plt.axis((-3, 3, min(y_new.min(), y_ini.min(), y.min()), 10))
plt.title("Best Model Predictions, RMSE: {:.4f}".format(best_val_rmse))  # use existing best_val_rmse
plt.legend()
plt.grid()
save_fig("early_stopping_predictions_plot")
plt.show()

# =============================================================================
# 邏輯迴歸 (LOGISTIC REGRESSION)
# =============================================================================
print("\n" + "=" * 80)
print("邏輯迴歸 (Logistic Regression)")
print("=" * 80)

# 繪製 Logistic (Sigmoid) 函數
lim = 6
t = np.linspace(-lim, lim, 100)
sig = 1 / (1 + np.exp(-t))

plt.figure(figsize=(8, 3))
plt.plot([-lim, lim], [0, 0], "k-") # x 軸 (y=0 線)
plt.plot([-lim, lim], [0.5, 0.5], "k:") # y=0.5 線
plt.plot([-lim, lim], [1, 1], "k:") # y=1 線
plt.plot([0, 0], [-0.1, 1.1], "k-") # y 軸 (x=0 線)
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \dfrac{1}{1 + e^{-t}}$") # Logistic 函數
plt.xlabel("t")
plt.legend(loc="best")
plt.axis((-lim, lim, -0.1, 1.1))
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.grid()
plt.title("Logistic 函數")
save_fig("logistic_function_plot")
plt.show()

# 載入鳶尾花資料集
iris = load_iris(as_frame=True) # 使用 DataFrame 格式
for k in list(iris):
    print(f"{k}: {type(iris[k])}, shape: {getattr(iris[k], 'shape', 'N/A')}")

print("\n鳶尾花資料集的前 5 筆資料:")
print(iris.data.head())
print(f"\n目標類別: {iris.target_names}")

# 準備二元分類資料 (是否為 Iris virginica)
X = iris.data[["petal width (cm)"]].values  # 使用花瓣寬度作為特徵
# 解釋: 使用雙括號 [[...]] 確保得到 2D DataFrame，.values 轉換為 NumPy 陣列
# 也可使用 .to_numpy() (pandas 較新的方法)
y = (iris.target_names[iris.target] == 'virginica')  # 目標: True/False

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Note: 對於train_test_split，X 和 y 都是 NumPy 陣列 (numpy.ndarray)

# 訓練邏輯迴歸模型
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# 產生用於繪製機率曲線的資料點
X_new = np.linspace(0, 3, 1000).reshape(-1, 1) # shape=(1000, 1)
# 預測新資料點的類別機率
y_proba = log_reg.predict_proba(X_new) # 預測機率, shape=(1000, 2)
# 找出決策邊界 (機率 = 0.5)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]

print(f"\n決策邊界位於花瓣寬度: {decision_boundary:.2f} cm")
print(f"預測花瓣寬度為 1.7cm 的類別: {log_reg.predict(np.array([[1.7]]))[0]}")
print(f"預測花瓣寬度為 1.5cm 的類別: {log_reg.predict(np.array([[1.5]]))[0]}")

# 繪製機率曲線與決策邊界
plt.figure(figsize=(8, 3))
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="非 Iris virginica 機率")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica 機率")
plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2, label="決策邊界")
plt.plot(X_train[y_train == 0], y_train[y_train == 0], "bs", label="非 Iris virginica")
plt.plot(X_train[y_train == 1], y_train[y_train == 1], "g^", label="Iris virginica")
plt.xlabel("花瓣寬度 (cm)")
plt.ylabel("機率")
plt.legend(loc="center left")
plt.axis((0, 3, -0.02, 1.02))
plt.grid()
save_fig("logistic_regression_plot")
plt.show()

# ==========================================
# 範例 13.1: 使用兩個特徵的邏輯迴歸 (機率等高線圖與決策邊界)
# ==========================================
# 💡 實際應用情境:
# 此範例訓練一個二元邏輯迴歸分類器來區分 Iris virginica，使用兩個特徵
# 並視覺化機率表面和線性決策邊界。此技術用於理解模型如何在 2D 特徵空間中劃分類別。

# ✅ 程式碼逐行解析:

# 1) 資料設定與訓練
# 使用兩個特徵: 花瓣長度和花瓣寬度
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
# 解釋: 使用雙括號 [[...]] 確保得到 2D DataFrame，.values 轉換為 NumPy 陣列
# 也可使用 .to_numpy() (pandas 較新的方法)

# 建立二元目標: True 表示 virginica, False 表示非 virginica
y = (iris.target_names[iris.target] == 'virginica')

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 訓練邏輯迴歸模型
# C 是正規化強度的倒數: C 越大 = 正規化越弱 = 模型越複雜
log_reg = LogisticRegression(C=2, random_state=42)
log_reg.fit(X_train, y_train)  # 學習係數 (coef_) 和截距 (intercept_)

print(f"\n模型係數: {log_reg.coef_}")
print(f"模型截距: {log_reg.intercept_}")

# 2) 建立機率網格用於等高線圖
# 使用 np.meshgrid 建立密集的 2D 網格以繪製平滑的等高線
x0, x1 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1, 1),      # 500 個花瓣長度座標 (2.9 到 7 cm)
    np.linspace(0.8, 2.7, 200).reshape(-1, 1),    # 200 個花瓣寬度座標 (0.8 到 2.7 cm)
)
# 結果: x0 和 x1 的形狀都是 (200, 500) = (Ny, Nx)
# .reshape(-1, 1) 將 1D 向量轉換為列向量 (非嚴格必要)
print(f"\n x0 的形狀: {x0.shape}")  # 顯示 x0 的形狀
print(f"x0 的前 10 行和前 5 列:\n{x0[:10, :5]}")
print(f"x1 的前 10 行和前 5 列:\n{x1[:10, :5]}")  # 顯示 x1 的前 10 行和前 5 列

# 將網格點轉換為樣本矩陣用於預測
X_new = np.c_[x0.ravel(), x1.ravel()]
# ravel() 將 2D 陣列展平為 1D: (200, 500) -> (100000,)
# np.c_ 將兩個 1D 陣列堆疊為 2D: (100000,) + (100000,) -> (100000, 2)
# X_new 現在包含 100,000 個 (花瓣長度, 花瓣寬度) 的點
print(f"\nX_new 的形狀: {X_new.shape}")  # 顯示 X_new 的形狀
print(f"X_new 的前 10 行:\n{X_new[:10]}")  # 顯示 X_new 的前 10 行
print(f"X_new 的後 10 行:\n{X_new[-10:]}")  # 顯示 X_new 的後 10 行

print(f"\n網格點數量: {X_new.shape[0]} (200×500 網格)")
print(f"網格形狀: x0.shape = {x0.shape}")

# 計算每個網格點的類別機率
y_proba = log_reg.predict_proba(X_new)  # 形狀: (100000, 2)
# y_proba[:, 0] = P(非 virginica | 特徵)
# y_proba[:, 1] = P(virginica | 特徵)

# 將機率重塑回 2D 網格形狀用於等高線繪圖
zz = y_proba[:, 1].reshape(x0.shape)  # 形狀: (200, 500)
# zz[i, j] = P(virginica | 網格點 (i, j) 處的特徵)

# 3) 計算決策邊界 (P = 0.5 的線)
# 對於邏輯迴歸，決策邊界是線性的:
# 邊界方程式: coef_[0] * x0 + coef_[1] * x1 + intercept_ = 0
# 改寫為: x1 = -(coef_[0] * x0 + intercept_) / coef_[1]
left_right = np.array([2.9, 7])  # 定義線的 x 範圍 (花瓣長度)
boundary = -(log_reg.coef_[0, 0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0, 1]
# boundary 是對應的 y 值 (花瓣寬度)

print(f"\n決策邊界方程式: x1 = -{log_reg.coef_[0, 0]:.3f}*x0 + {-log_reg.intercept_[0]/log_reg.coef_[0, 1]:.3f}")

# 4) 視覺化: 等高線圖 + 決策邊界 + 訓練點
plt.figure(figsize=(10, 4))

# 繪製訓練點
plt.plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], "bs", label="非 Iris virginica")  # 藍色方塊
plt.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], "g^", label="Iris virginica")     # 綠色三角形

# 繪製機率等高線 (顏色表示 P(virginica))
# Use get_cmap() to retrieve the colormap object by name
contour = plt.contour(x0, x1, zz, cmap=plt.cm.get_cmap("brg"))  # brg = blue-red-green 顏色映射
plt.clabel(contour, inline=1)  # 在等高線上顯示機率值

# 繪製決策邊界 (P = 0.5 的線)
plt.plot(left_right, boundary, "k--", linewidth=3, label="決策邊界")  # 黑色虛線

# 添加文字標籤
plt.text(3.5, 1.27, "非 Iris virginica", color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", color="g", ha="center")

# 設定座標軸標籤和範圍
plt.xlabel("花瓣長度 (cm)")
plt.ylabel("花瓣寬度 (cm)")
plt.axis((2.9, 7, 0.8, 2.7))  # [xmin, xmax, ymin, ymax]
plt.grid()
plt.legend()
save_fig("logistic_regression_contour_plot")
plt.show()

# 🎯 重點摘要:
# - 使用 np.meshgrid 建立密集網格 (200×500 = 100,000 個點) 以繪製平滑的等高線
# - 使用 ravel() 展平 2D 陣列為 1D 用於批量預測 (記憶體效率高)
# - 決策邊界是線性的: petal_width = -(w_len × petal_length + b) / w_wid
# - 等高線顏色顯示 P(virginica) 的機率分布，虛線是 P = 0.5 的決策邊界
# - 此方法適用於任何二元分類問題的視覺化分析

# =============================================================================
# Softmax 迴歸 (SOFTMAX REGRESSION)
# =============================================================================
print("\n" + "=" * 80)
print("Softmax 迴歸 (Softmax Regression)")
print("=" * 80)

# ==========================================
# 範例 14.1: Softmax 迴歸用於多類別分類
# ==========================================
# 💡 實際應用情境:
# Softmax 迴歸 (也稱為多項式邏輯迴歸, multinomial logistic regression) 是邏輯迴歸的自然推廣，
# 用於同時分類多個互斥的類別。此範例展示如何分類所有三個 Iris 物種。

# ✅ 程式碼逐行解析:

# 1) 資料準備
# 使用兩個特徵提供更多資訊以區分三個物種
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
print(f"\n特徵矩陣形狀: {X.shape}")

# 使用原始的三類別目標變數 (而非二元分類)
# 0 = setosa, 1 = versicolor, 2 = virginica
y = iris.target
print(f"目標變數形狀: {y.shape}")
print(f"各類別數量: setosa={np.sum(y==0)}, versicolor={np.sum(y==1)}, virginica={np.sum(y==2)}")

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2) 模型配置
# C=30: 高正規化參數 (正規化強度的倒數)，允許模型更靈活地擬合訓練資料
# 當 y 有超過 2 個類別時，LogisticRegression 自動使用 multinomial (Softmax) 迴歸
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

print(f"\n模型已訓練完成")
print(f"模型係数形狀: {softmax_reg.coef_.shape}")  # (3, 2) - 每個類別有一組係數
print(f"模型截距形狀: {softmax_reg.intercept_.shape}")  # (3,) - 每個類別有一個截距

# 3) 預測方法演示
# predict(): 返回最高機率的類別標籤 (0, 1 或 2)
predicted_class = softmax_reg.predict(np.array([[5, 2]]))[0]
print(f"\n對於花瓣長度 5cm、寬度 2cm 的預測類別: {predicted_class}")
print(f"對應的物種名稱: {iris.target_names[predicted_class]}")

# predict_proba(): 返回所有三個類別的機率估計 (總和為 1.0)
proba = softmax_reg.predict_proba(np.array([[5, 2]])).round(2)
print(f"各類別的預測機率: {proba}")
print(f"  - P(setosa)     = {proba[0, 0]:.2f}")
print(f"  - P(versicolor) = {proba[0, 1]:.2f}")
print(f"  - P(virginica)  = {proba[0, 2]:.2f}")

# decision_function(): 返回 softmax 轉換前的原始類別分數
scores = softmax_reg.decision_function(np.array([[5, 2]]))
print(f"原始類別分數: {scores.round(2)}")

# ==========================================
# 範例 14.2: Softmax 迴歸決策邊界視覺化
# ==========================================
# 💡 實際應用情境:
# 此視覺化展示 Softmax 迴歸如何使用花瓣測量值分類三個 Iris 物種，
# 幫助理解模型的決策邊界和機率分布。

# ✅ 程式碼逐行解析:

# 1) 視覺化設定
# 建立自訂顏色映射: 淡黃色=setosa, 淡藍色=versicolor, 淡綠色=virginica
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

# 2) 網格建立
# 建立覆蓋特徵空間的細緻網格
x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500),      # 花瓣長度: 0 到 8 cm (500 個點)
    np.linspace(0, 3.5, 200)    # 花瓣寬度: 0 到 3.5 cm (200 個點)
)
print(f"\n網格形狀: x0.shape = {x0.shape}, x1.shape = {x1.shape}")

# 將 2D 網格轉換為適合模型預測的格式
X_new = np.c_[x0.ravel(), x1.ravel()]
print(f"用於預測的網格點數量: {X_new.shape[0]} (200×500 網格)")

# 3) 整個特徵空間的模型預測
# 計算網格中每個點的類別機率
y_proba = softmax_reg.predict_proba(X_new)
print(f"機率矩陣形狀: {y_proba.shape}")  # (100000, 3)

# 確定每個網格點的預測類別
y_predict = softmax_reg.predict(X_new)

# 提取 Iris versicolor 機率y_proba[:, 1]用於等高線 (中間類別，最能顯示漸變)
zz1 = y_proba[:, 1].reshape(x0.shape)
print(f"Versicolor 機率網格形狀: {zz1.shape}")

# 將預測重塑回網格格式用於視覺化
zz = y_predict.reshape(x0.shape)

# 4) 繪製決策區域和資料點
plt.figure(figsize=(10, 4))

# 繪製訓練資料點 (使用完整資料集以顯示所有點)
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica", markersize=8)
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor", markersize=8)
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa", markersize=8)

# 繪製決策區域: 用顏色填充區域，顯示預測每個類別的位置
plt.contourf(x0, x1, zz, cmap=custom_cmap, alpha=0.3)

# 繪製機率等高線: 顯示 Iris versicolor 機率等級
# "hot" 顏色映射建立顯示機率轉變的漸層線
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1, fontsize=8)  # 在等高線上顯示機率值

# 設定座標軸標籤和範圍
plt.xlabel("花瓣長度 (cm)")
plt.ylabel("花瓣寬度 (cm)")
plt.legend(loc="center left")
plt.axis((0.5, 7, 0, 3.5))
plt.grid(alpha=0.3)
save_fig("softmax_regression_contour_plot")
plt.show()

# 🎯 重點摘要:
print("\n" + "=" * 80)
print("Softmax 迴歸重點摘要:")
print("=" * 80)
print("✓ Softmax 迴歸將邏輯迴歸擴展到多類別分類")
print("✓ 使用 softmax 函數確保機率總和為 1")
print("✓ 每個類別有獨立的參數集 (coef_ 和 intercept_)")
print("✓ 建立複雜的決策邊界分離多個類別")
print("✓ 視覺化顯示:")
print("  - Iris setosa (黃色區域) 以較小的花瓣測量值容易區分")
print("  - Iris versicolor 和 virginica 之間的邊界更複雜")
print("  - 等高線顯示平滑的機率轉變")
print("  - 接近訓練資料的區域顯示更高的模型信心")
print("=" * 80)

# ==========================================
# 範例 14.3: Softmax 迴歸不同類別機率視覺化
# ==========================================

# Extract probabilities for Iris virginica (class 2) instead of versicolor (class 1)
zz2 = y_proba[:, 2].reshape(x0.shape) # Virginica 機率網格形狀: (200, 500)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz2, cmap="hot")
plt.clabel(contour, inline=1)
plt.xlabel("花瓣長度 (cm)")
plt.ylabel("花瓣寬度 (cm)")
plt.legend(loc="center left")
plt.axis((0.5, 7, 0, 3.5))
plt.grid(alpha=0.3)
save_fig("softmax_regression_contour_plot_virginica")
plt.show()

# 🎯 重點摘要:
print("\n" + "=" * 80)
print("Softmax 迴歸不同類別機率視覺化重點摘要:")
print("=" * 80)
print("✓ 此圖顯示 Iris virginica (class 2) 的機率分布")
print("✓ 決策區域顯示模型如何區分三個物種")
print("✓ 等高線顯示 virginica 機率的平滑轉變")
print("✓ 與 versicolor 機率圖相比，顯示不同的信心區域")
print("✓ 幫助理解模型在多類別分類中的行為")
print("=" * 80)

# =============================================================================
# 程式結束
# =============================================================================
print("\n" + "=" * 80)
print("✅ 訓練線性模型範例完成")
print("=" * 80)
print("\n涵蓋主題:")
print("  1. 線性迴歸 (Normal Equation)")
print("  2. 梯度下降 (Batch, Stochastic, Mini-batch)")
print("  3. 多項式迴歸 (Polynomial Regression)")
print("  4. 學習曲線 (Learning Curves)")
print("  5. 正規化模型 (Ridge, Lasso, Elastic Net)")
print("  6. 早停法 (Early Stopping)")
print("  7. 邏輯迴歸 (Logistic Regression)")
print("  8. Softmax 迴歸 (Softmax Regression)")
print("\n所有圖表已儲存至:", IMAGES_PATH)
print("=" * 80)
