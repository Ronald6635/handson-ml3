"""
訓練深度神經網路模組

此模組示範訓練深度神經網路的關鍵技術，包括梯度問題解決、
初始化方法、激活函數、批次正規化、優化器和正規化。

關鍵特點：
- 梯度消失/爆炸問題的處理
- Xavier和He初始化
- 非飽和激活函數
- 批次正規化
- 梯度裁剪
- 遷移學習
- 進階優化器
- 學習率調度
- 正規化技術

範例基於markdown文件11_training_deep_neural_networks.md
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from packaging import version
from pathlib import Path
from functools import partial
import math

# =============================================================================
# 設定與初始化
# =============================================================================

print("=== 設定與初始化 ===")

# 檢查Python版本
assert sys.version_info >= (3, 7)
print("Python版本檢查通過")

# 檢查TensorFlow版本
assert version.parse(tf.__version__) >= version.parse("2.8.0")
print("TensorFlow版本檢查通過")

# 設定圖表字體大小
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
print("圖表設定完成")

# 建立圖片目錄
IMAGES_PATH = Path() / "images" / "deep"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """儲存圖表到檔案"""
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

print("圖片儲存函數定義完成")

# =============================================================================
# 範例1: 繪製Sigmoid激活函數的飽和問題
# =============================================================================

print("=== 範例1: Sigmoid激活函數飽和問題 ===")

def sigmoid(z):
    """計算sigmoid函數值"""
    return 1 / (1 + np.exp(-z))

# 建立z值的範圍
z = np.linspace(-5, 5, 200)

# 繪製圖表
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [1, 1], 'k--')
plt.plot([0, 0], [-0.2, 1.2], 'k-')
plt.plot([-5, 5], [-3/4, 7/4], 'g--')
plt.plot(z, sigmoid(z), "b-", linewidth=2,
         label=r"$\sigma(z) = \dfrac{1}{1+e^{-z}}$")

# 設定圖表屬性並儲存
plt.axis([-5, 5, -0.2, 1.2])
plt.xlabel("$z$")
plt.legend(loc="upper left", fontsize=16)
save_fig("sigmoid_saturation_plot")
plt.show()

print("Sigmoid飽和問題圖表已儲存")

# =============================================================================
# 範例2: 使用He初始化建立Dense層
# =============================================================================

print("=== 範例2: He初始化Dense層 ===")

# 使用He正態初始化建立Dense層，適用於ReLU激活函數
dense = tf.keras.layers.Dense(50, activation="relu",
                              kernel_initializer="he_normal")
print("He初始化Dense層建立完成")

# =============================================================================
# 範例3: 自訂He初始化變體
# =============================================================================

print("=== 範例3: 自訂He初始化變體 ===")

# 使用VarianceScaling初始化器自訂He初始化
he_avg_init = tf.keras.initializers.VarianceScaling(scale=2., mode="fan_avg",
                                                    distribution="uniform")
dense = tf.keras.layers.Dense(50, activation="sigmoid",
                              kernel_initializer=he_avg_init)
print("自訂He初始化變體建立完成")

# =============================================================================
# 範例4: Leaky ReLU激活函數
# =============================================================================

print("=== 範例4: Leaky ReLU激活函數 ===")

def leaky_relu(z, alpha):
    """計算Leaky ReLU函數值"""
    return np.maximum(alpha * z, z)

# 繪製Leaky ReLU曲線
z = np.linspace(-5, 5, 200)
plt.plot(z, leaky_relu(z, 0.1), "b-", linewidth=2,
         label=r"$LeakyReLU(z) = max(\alpha z, z)$")
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k:', linewidth=2)
plt.axis([-5, 5, -1, 3.7])
plt.xlabel("$z$")
plt.legend()
save_fig("leaky_relu_plot")
plt.show()

print("Leaky ReLU圖表已儲存")

# =============================================================================
# 範例5: 使用TensorFlow的LeakyReLU
# =============================================================================

print("=== 範例5: TensorFlow LeakyReLU層 ===")

# 使用TensorFlow的LeakyReLU層
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)  # 預設為alpha=0.3
dense = tf.keras.layers.Dense(50, activation=leaky_relu,
                              kernel_initializer="he_normal")
print("TensorFlow LeakyReLU層建立完成")

# =============================================================================
# 範例6: 批次正規化Sequential模型
# =============================================================================

print("=== 範例6: 批次正規化模型 ===")

# 清除之前的模型
tf.keras.backend.clear_session()
tf.random.set_seed(42)

# 建立包含批次正規化的模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.BatchNormalization(),  # 輸入標準化
    tf.keras.layers.Dense(300, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),  # 隱藏層標準化
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),  # 另一隱藏層標準化
    tf.keras.layers.Dense(10, activation="softmax")
])

print("批次正規化模型建立完成")

# =============================================================================
# 範例7: 梯度裁剪優化器
# =============================================================================

print("=== 範例7: 梯度裁剪優化器 ===")

# 使用clipvalue進行梯度裁剪
optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)
print("梯度裁剪優化器設定完成")

# =============================================================================
# 範例8: 重用預訓練層
# =============================================================================

print("=== 範例8: 重用預訓練層 ===")

# 注意：此範例需要預訓練模型檔案，實際執行時需先訓練model_A
# model_A = tf.keras.models.load_model("my_model_A.keras")
# model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])
# model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))
print("重用預訓練層範例（需要預訓練模型）")

# =============================================================================
# 範例9: Adam優化器
# =============================================================================

print("=== 範例9: Adam優化器 ===")

# 使用Adam優化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                     beta_2=0.999)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
print("Adam優化器設定完成")

# =============================================================================
# 範例10: 學習率調度
# =============================================================================

print("=== 範例10: 學習率調度 ===")

# 使用指數衰減調度器
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=20_000,
    decay_rate=0.1,
    staircase=False
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
print("學習率調度器設定完成")

# =============================================================================
# 範例11: Dropout正規化
# =============================================================================

print("=== 範例11: Dropout正規化 ===")

# 清除之前的模型
tf.keras.backend.clear_session()
tf.random.set_seed(42)

# 在模型中加入Dropout層
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dropout(rate=0.2),  # 輸入Dropout
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),  # 隱藏層Dropout
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),  # 另一隱藏層Dropout
    tf.keras.layers.Dense(10, activation="softmax")
])

# 編譯並訓練模型
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])

print("Dropout正規化模型建立完成")

# 注意：實際訓練需要資料集，此處僅建立模型
print("所有範例建立完成！")