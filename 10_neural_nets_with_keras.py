"""
神經網路與 Keras 實作模組

此模組示範如何使用 Keras 建構和訓練神經網路，包括分類和回歸任務。
涵蓋從基礎感知器到複雜深度網路的完整實作。

關鍵功能：
- 感知器實作
- 多層感知器 (MLP)
- Keras 序列 API
- Keras 函數式 API
- Keras 子類化 API
- 模型儲存與載入
- 回呼函數使用
- TensorBoard 視覺化
- 超參數調優

範例基於 markdown 文件 10_neural_nets_with_keras.md
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from scipy.special import expit as sigmoid
import tensorflow as tf
import keras_tuner as kt
from time import strftime

# =============================================================================
# 範例 1: 環境檢查與設定
# =============================================================================

print("=== 範例 1: 環境檢查與設定 ===")

# 檢查 Python 版本 - 確保版本至少為 3.7
assert sys.version_info >= (3, 7)
print("✓ Python 版本檢查通過")

# 設定圖表字體大小 - 使圖表更美觀
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
print("✓ Matplotlib 設定完成")

# 建立圖片儲存目錄
IMAGES_PATH = Path() / "images" / "ann"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """儲存圖表到檔案"""
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

print("✓ 圖片儲存目錄建立完成")
print("\n=== 環境設定完成 ===\n")

# =============================================================================
# 範例 2: 使用 Scikit-Learn 實作感知器
# =============================================================================

print("=== 範例 2: 使用 Scikit-Learn 實作感知器 ===")

# 載入鳶尾花資料集 - 選擇花瓣長度和寬度作為特徵
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)  # 將問題轉為二元分類：是否為 Iris setosa

# 分割資料 - 分為訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練感知器 - 使用隨機種子確保結果可重現
per_clf = Perceptron(random_state=42)
per_clf.fit(X_train, y_train)

# 進行預測 - 測試兩個新樣本
X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)

print(f"新樣本預測結果: {y_pred}")
print("✓ 感知器訓練和預測完成")
print("\n=== 感知器範例完成 ===\n")

# =============================================================================
# 範例 3: 常見激活函數視覺化
# =============================================================================

print("=== 範例 3: 常見激活函數視覺化 ===")

def relu(z):
    """ReLU 激活函數 -  rectified linear unit"""
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    """數值微分函數 - 用於計算導數"""
    return (f(z + eps) - f(z - eps))/(2 * eps)

# 設定輸入範圍
max_z = 4.5
z = np.linspace(-max_z, max_z, 200)

# 建立圖表
plt.figure(figsize=(11, 3.1))

# 繪製激活函數
plt.subplot(121)
plt.plot([-max_z, 0], [0, 0], "r-", linewidth=2, label="Heaviside")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.plot([0, 0], [0, 1], "r-", linewidth=0.5)
plt.plot([0, max_z], [1, 1], "r-", linewidth=2)
plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, np.tanh(z), "b-", linewidth=1, label="Tanh")
plt.grid(True)
plt.title("激活函數")
plt.axis([-max_z, max_z, -1.65, 2.4])
plt.legend(loc="lower right", fontsize=13)

# 繪製導數
plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Heaviside")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=1, label="Tanh")
plt.grid(True)
plt.title("導數")
plt.axis([-max_z, max_z, -0.2, 1.2])

save_fig("activation_functions_plot")
plt.show()
print("✓ 激活函數視覺化完成")
print("\n=== 激活函數範例完成 ===\n")

# =============================================================================
# 範例 4: 使用 Scikit-Learn 建構回歸 MLP
# =============================================================================

print("=== 範例 4: 使用 Scikit-Learn 建構回歸 MLP ===")

# 載入加州房價資料集 - 這是一個經典的回歸資料集
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

# 建立 MLP 回歸模型 - 使用三層隱藏層，每層 50 個神經元
mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)

# 計算均方根誤差 (RMSE)
rmse = root_mean_squared_error(y_valid, y_pred)
print(f"驗證集 RMSE: {rmse:.4f}")
print("✓ MLP 回歸訓練完成")
print("\n=== 回歸 MLP 範例完成 ===\n")

# =============================================================================
# 範例 5: 使用 Scikit-Learn 建構分類 MLP
# =============================================================================

print("=== 範例 5: 使用 Scikit-Learn 建構分類 MLP ===")

# 載入鳶尾花資料集 - 多類別分類問題
iris = load_iris()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    iris.data, iris.target, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42)

# 建立 MLP 分類器 - 單隱藏層，5 個神經元
mlp_clf = MLPClassifier(hidden_layer_sizes=[5], max_iter=10_000, random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_clf)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_valid, y_valid)
print(f"驗證集準確率: {accuracy:.4f}")
print("✓ MLP 分類訓練完成")
print("\n=== 分類 MLP 範例完成 ===\n")

# =============================================================================
# 範例 6: 載入和預處理 Fashion MNIST 資料集
# =============================================================================

print("=== 範例 6: 載入和預處理 Fashion MNIST 資料集 ===")

# 載入 Fashion MNIST 資料集 - 這是 Zalando 的服飾影像資料集
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

print(f"訓練集形狀: {X_train.shape}")
print(f"驗證集形狀: {X_valid.shape}")
print(f"測試集形狀: {X_test.shape}")

# 正規化像素值 - 將 0-255 範圍縮放到 0-1
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

# 定義類別名稱
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"第一個訓練樣本的類別: {class_names[y_train[0]]}")
print("✓ Fashion MNIST 資料載入和預處理完成")
print("\n=== 資料載入範例完成 ===\n")

# =============================================================================
# 範例 7: 建構和編譯序列模型
# =============================================================================

print("=== 範例 7: 建構和編譯序列模型 ===")

# 設定隨機種子確保結果可重現
tf.random.set_seed(42)

# 建構序列模型 - 展平層 + 兩個隱藏層 + 輸出層
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),  # 將 28x28 影像展平為 784 維向量
    tf.keras.layers.Dense(300, activation="relu"),   # 第一隱藏層，300 個神經元
    tf.keras.layers.Dense(100, activation="relu"),   # 第二隱藏層，100 個神經元
    tf.keras.layers.Dense(10, activation="softmax")  # 輸出層，10 個類別
])

# 編譯模型 - 指定損失函數、優化器和評估指標
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# 顯示模型摘要
model.summary()
print("✓ Keras 序列模型建構完成")
print("\n=== 模型建構範例完成 ===\n")

# =============================================================================
# 範例 8: 訓練和評估模型
# =============================================================================

print("=== 範例 8: 訓練和評估模型 ===")

# 訓練模型 - 使用 30 個 epoch 和驗證資料
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

# 評估模型在測試集上的表現
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"測試損失: {test_loss:.4f}")
print(f"測試準確率: {test_accuracy:.4f}")
print("✓ 模型訓練和評估完成")
print("\n=== 訓練評估範例完成 ===\n")

# =============================================================================
# 範例 9: 建構回歸 MLP
# =============================================================================

print("=== 範例 9: 建構回歸 MLP ===")

# 載入加州房價資料
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

# 建構回歸模型
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)  # 回歸任務輸出層無激活函數
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)

# 訓練模型
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

# 評估和預測
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

print(f"測試集 RMSE: {rmse_test:.4f}")
print(f"前三個樣本預測值: {y_pred.flatten()}")
print("✓ 回歸 MLP 訓練完成")
print("\n=== 回歸 MLP 範例完成 ===\n")

# =============================================================================
# 範例 10: Wide & Deep 模型
# =============================================================================

print("=== 範例 10: Wide & Deep 模型 ===")

# 清除 session 並設定種子
tf.keras.backend.clear_session()
tf.random.set_seed(42)

# 定義各層
normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

# 建構函數式模型
input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])

# 編譯和訓練
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
normalization_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

print("✓ Wide & Deep 模型訓練完成")
print("\n=== Wide & Deep 範例完成 ===\n")

# =============================================================================
# 範例 11: 自訂 WideAndDeepModel 類別
# =============================================================================

print("=== 範例 11: 自訂 WideAndDeepModel 類別 ===")

class WideAndDeepModel(tf.keras.Model):
    """自訂的 Wide & Deep 模型類別"""
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # 呼叫父類別初始化
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """前向傳播邏輯"""
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output

# 建立和訓練模型
tf.random.set_seed(42)
model = WideAndDeepModel(30, activation="relu", name="my_cool_model")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=optimizer,
              metrics=["RootMeanSquaredError", "RootMeanSquaredError"])

print("✓ 自訂模型類別定義完成")
print("\n=== 自訂模型範例完成 ===\n")

# =============================================================================
# 範例 12: 儲存和載入 Keras 模型
# =============================================================================

print("=== 範例 12: 儲存和載入 Keras 模型 ===")

# 儲存完整模型
model.save("my_model.keras")
print("✓ 模型已儲存為 my_model.keras")

# 載入模型
loaded_model = tf.keras.models.load_model("my_model.keras")
print("✓ 模型已從 my_model.keras 載入")

# 儲存權重
model.save_weights("my_weights.weights.h5")
print("✓ 權重已儲存為 my_weights.weights.h5")

# 載入權重
model.load_weights("my_weights.weights.h5")
print("✓ 權重已從 my_weights.weights.h5 載入")

print("\n=== 模型儲存載入範例完成 ===\n")

# =============================================================================
# 範例 13: 使用常見回呼函數
# =============================================================================

print("=== 範例 13: 使用常見回呼函數 ===")

# 檢查點回呼 - 儲存最佳權重
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints.weights.h5",
                                                   save_weights_only=True)

# 早期停止回呼 - 當驗證損失不再改善時停止訓練
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

# 自訂回呼 - 印出驗證/訓練損失比率
class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")

print("✓ 回呼函數定義完成")
print("\n=== 回呼函數範例完成 ===\n")

# =============================================================================
# 範例 14: 設定 TensorBoard 回呼
# =============================================================================

print("=== 範例 14: 設定 TensorBoard 回呼 ===")

def get_run_logdir(root_logdir="my_logs"):
    """取得唯一的執行日誌目錄"""
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir()
print(f"TensorBoard 日誌目錄: {run_logdir}")

# 建立 TensorBoard 回呼
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir,
                                                profile_batch=(100, 200))

print("✓ TensorBoard 回呼設定完成")
print("注意：實際使用時需要載入 tensorboard 擴充並啟動伺服器")
print("\n=== TensorBoard 範例完成 ===\n")

# =============================================================================
# 範例 15: 使用 Keras Tuner 進行超參數搜尋
# =============================================================================

print("=== 範例 15: 使用 Keras Tuner 進行超參數搜尋 ===")

def build_model(hp):
    """建構模型的函數，接受超參數物件"""
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model

# 建立隨機搜尋調優器
random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)

print("✓ Keras Tuner 設定完成")
print("注意：實際執行搜尋需要呼叫 random_search_tuner.search()")
print("\n=== 超參數調優範例完成 ===\n")

print("🎉 所有範例執行完成！")
print("此模組示範了使用 Keras 建構神經網路的完整流程。")