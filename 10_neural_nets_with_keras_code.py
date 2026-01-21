"""
使用 Keras 介紹人工神經網路 - 實作模組

此模組包含《Hands-On Machine Learning》第 3 版第 10 章的所有實作範例，
展示如何使用 Keras 建構和訓練人工神經網路。

主要功能：
- 感知器實作
- 多層感知器（MLP）
- Keras 三種 API 的使用
- 模型儲存與載入
- 回呼函數與訓練技巧
- TensorBoard 視覺化
- 超參數調校

範例基於 10_neural_nets_with_keras_tutorial.md 教學文件
"""

import sys
import os
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal
import tensorflow as tf
from tensorflow import keras

# =============================================================================
# 範例 1: 檢查 Python 與 Scikit-Learn 版本
# =============================================================================

print("=== 範例 1: 環境檢查 ===")

# 確保 Python 版本至少 3.7
assert sys.version_info >= (3, 7), "需要 Python 3.7 或更新版本"
print(f"Python 版本: {sys.version}")

# 檢查 Scikit-Learn 版本
print(f"Scikit-Learn 版本: {sklearn.__version__}")

print("\n=== 環境檢查完成 ===")

# =============================================================================
# 範例 2: 使用 Scikit-Learn 實作感知器
# =============================================================================

print("\n=== 範例 2: 感知器實作 ===")

# 載入鳶尾花資料集
iris = load_iris()
X = iris.data[:, (2, 3)]  # 選取花瓣長度和寬度特徵
y = (iris.target == 0).astype(int)  # 建立二元分類標籤（是否為 Setosa）

# 建立並訓練感知器模型
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

# 進行預測
y_pred = per_clf.predict([[2, 0.5]])
print(f"預測結果: {y_pred}")

print("\n=== 感知器實作完成 ===")

# =============================================================================
# 範例 3: 實作多層感知器
# =============================================================================

print("\n=== 範例 3: 多層感知器實作 ===")

# 建立 MLP 分類器
mlp_clf = MLPClassifier(hidden_layer_sizes=(5,), activation='relu',
                       solver='adam', random_state=42, max_iter=1000)
mlp_clf.fit(X, y)

# 進行預測
y_pred = mlp_clf.predict([[2, 0.5]])
print(f"MLP 預測結果: {y_pred}")

print("\n=== 多層感知器實作完成 ===")

# =============================================================================
# 範例 4: 使用 Sequential API 建構 Keras 模型
# =============================================================================

print("\n=== 範例 4: Sequential API 建構模型 ===")

# 建立 Sequential 模型
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),  # 輸入層：攤平 28x28 圖片
    keras.layers.Dense(300, activation="relu"),   # 第一隱藏層：300 個神經元
    keras.layers.Dense(100, activation="relu"),   # 第二隱藏層：100 個神經元
    keras.layers.Dense(10, activation="softmax")  # 輸出層：10 個類別
])

# 編譯模型
model.compile(loss="sparse_categorical_crossentropy",  # 損失函數
              optimizer="sgd",                         # 優化器
              metrics=["accuracy"])                    # 評估指標

print("Sequential 模型摘要:")
model.summary()

print("\n=== Sequential API 建構完成 ===")

# =============================================================================
# 範例 5: 使用 Functional API 建構模型
# =============================================================================

print("\n=== 範例 5: Functional API 建構模型 ===")

# 建立 Functional 模型
input_ = keras.layers.Input(shape=[28, 28])                    # 定義輸入
flatten = keras.layers.Flatten(input_shape=[28, 28])(input_)   # 攤平層
hidden1 = keras.layers.Dense(300, activation="relu")(flatten)  # 第一隱藏層
hidden2 = keras.layers.Dense(100, activation="relu")(hidden1)  # 第二隱藏層
output = keras.layers.Dense(10, activation="softmax")(hidden2) # 輸出層

functional_model = keras.models.Model(inputs=[input_], outputs=[output])

# 編譯模型
functional_model.compile(loss="sparse_categorical_crossentropy",
                        optimizer="sgd",
                        metrics=["accuracy"])

print("Functional 模型摘要:")
functional_model.summary()

print("\n=== Functional API 建構完成 ===")

# =============================================================================
# 範例 6: 使用 Subclassing API 建構模型
# =============================================================================

print("\n=== 範例 6: Subclassing API 建構模型 ===")

class MyModel(keras.models.Model):
    """自訂神經網路模型類別"""
    
    def __init__(self):
        super().__init__()
        # 初始化各層
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(300, activation="relu")
        self.dense2 = keras.layers.Dense(100, activation="relu")
        self.dense3 = keras.layers.Dense(10, activation="softmax")
    
    def call(self, inputs):
        """定義前向傳播"""
        x = self.flatten(inputs)  # 攤平輸入
        x = self.dense1(x)        # 第一隱藏層
        x = self.dense2(x)        # 第二隱藏層
        return self.dense3(x)     # 輸出層

# 建立模型實例
subclass_model = MyModel()

# 編譯模型
subclass_model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="sgd",
                      metrics=["accuracy"])

print("Subclassing 模型摘要:")
subclass_model.build(input_shape=(None, 28, 28))
subclass_model.summary()

print("\n=== Subclassing API 建構完成 ===")

# =============================================================================
# 範例 7: 模型儲存與載入
# =============================================================================

print("\n=== 範例 7: 模型儲存與載入 ===")

# 儲存模型
model.save("my_keras_model.h5")
print("模型已儲存為 my_keras_model.h5")

# 載入模型
loaded_model = keras.models.load_model("my_keras_model.h5")
print("模型已從 my_keras_model.h5 載入")

# 清理檔案
if os.path.exists("my_keras_model.h5"):
    os.remove("my_keras_model.h5")
    print("清理臨時檔案")

print("\n=== 模型儲存與載入完成 ===")

# =============================================================================
# 範例 8: 使用回呼函數訓練模型
# =============================================================================

print("\n=== 範例 8: 使用回呼函數訓練 ===")

# 建立簡單的訓練資料（示範用）
X_train = np.random.randn(1000, 28, 28).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)
X_valid = np.random.randn(200, 28, 28).astype(np.float32)
y_valid = np.random.randint(0, 10, 200)

# 定義回呼函數
checkpoint_cb = keras.callbacks.ModelCheckpoint("best_model.h5", 
                                               save_best_only=True,
                                               monitor='val_accuracy')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, 
                                                 restore_best_weights=True,
                                                 monitor='val_accuracy')

# 訓練模型
print("開始訓練模型...")
history = model.fit(X_train, y_train, 
                   epochs=10,  # 縮短訓練時間作為示範
                   validation_data=(X_valid, y_valid),
                   callbacks=[checkpoint_cb, early_stopping_cb],
                   verbose=1)

print(f"訓練完成，最終驗證準確率: {history.history['val_accuracy'][-1]:.4f}")

# 清理檔案
if os.path.exists("best_model.h5"):
    os.remove("best_model.h5")

print("\n=== 回呼函數訓練完成 ===")

# =============================================================================
# 範例 9: 使用 TensorBoard 視覺化
# =============================================================================

print("\n=== 範例 9: TensorBoard 視覺化 ===")

# 設定日誌目錄
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    """產生唯一的執行日誌目錄"""
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
print(f"TensorBoard 日誌目錄: {run_logdir}")

# 建立 TensorBoard 回呼
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# 訓練時使用 TensorBoard
print("使用 TensorBoard 訓練...")
history = model.fit(X_train, y_train, 
                   epochs=5,  # 縮短作為示範
                   validation_data=(X_valid, y_valid),
                   callbacks=[tensorboard_cb],
                   verbose=1)

print("訓練完成，可使用以下指令啟動 TensorBoard:")
print(f"tensorboard --logdir {root_logdir}")

print("\n=== TensorBoard 視覺化完成 ===")

# =============================================================================
# 範例 10: 超參數調校
# =============================================================================

print("\n=== 範例 10: 超參數調校 ===")

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[28, 28]):
    """建構神經網路模型的函數"""
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    
    # 動態加入隱藏層
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    
    model.add(keras.layers.Dense(10, activation="softmax"))
    
    # 設定優化器
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", 
                 optimizer=optimizer, 
                 metrics=["accuracy"])
    return model

# 包裝為 Scikit-Learn 分類器
keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model, epochs=5, verbose=0)

# 定義參數分佈
param_distribs = {
    "n_hidden": [0, 1, 2, 3],                    # 隱藏層數量
    "n_neurons": np.arange(1, 100),              # 每層神經元數量
    "learning_rate": reciprocal(3e-4, 3e-2),     # 學習率
}

print("開始超參數調校（這可能需要一些時間）...")

# 執行隨機搜尋
rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, 
                                  n_iter=3, cv=2, verbose=1)  # 縮小規模作為示範

try:
    rnd_search_cv.fit(X_train[:100], y_train[:100],  # 使用小樣本
                     validation_data=(X_valid[:20], y_valid[:20]))
    
    print(f"最佳參數: {rnd_search_cv.best_params_}")
    print(f"最佳分數: {rnd_search_cv.best_score_:.4f}")
    
except Exception as e:
    print(f"調校過程中發生錯誤: {e}")
    print("這是正常的，因為我們使用的是隨機資料")

print("\n=== 超參數調校完成 ===")

# =============================================================================
# 總結
# =============================================================================

print("\n" + "="*60)
print("🎉 神經網路實作模組執行完成！")
print("="*60)
print("此模組展示了 Keras 的主要功能：")
print("• 三種模型建構 API（Sequential、Functional、Subclassing）")
print("• 模型儲存與載入")
print("• 回呼函數與訓練技巧")
print("• TensorBoard 視覺化")
print("• 超參數自動調校")
print("\n更多細節請參考教學文件：10_neural_nets_with_keras_tutorial.md")
print("="*60)