# 課程講義：深度神經網路訓練優化與技術 (Chapter 11)

**學習目標：** 本章節旨在幫助電機系同學掌握克服「梯度消失與爆炸」的核心技術，並學會如何透過進階優化器與正則化手段，穩定且高效地訓練深層神經網路。

---

你好，各位同學。我是你們這學期「深度學習實務」課程的教授。

今天我們要探討的是深度神經網路（DNN）訓練中最核心的挑戰：**如何讓深層模型穩定且快速地收斂**。在建構深層網路時，我們經常會遇到梯度消失（Vanishing Gradients）或梯度爆炸（Exploding Gradients）的問題，這會讓模型難以訓練。我們將從權重初始化、活化函數 (Activation Function)、歸一化、遷移學習、優化器以及正則化這六大面向出發，打造你的「深度學習工具箱」。

---

## 1. 權重初始化與梯度問題 (Initialization & Gradient Problems)

當訊號在深層網路中反向傳播時，梯度可能會趨近於 0（梯度消失）或是無窮大（梯度爆炸）

* **關鍵概念：**
  * **He 初始化 (He Initialization)**：專門為 **ReLU** 及其變體（Leaky ReLU, ELU, SELU）設計。
  * **Xavier (Glorot) 初始化**：適用於 **Sigmoid** 或 **Tanh** 活化函數。
  * **核心目標**：在每一層的輸入與輸出之間保持信號的變異數（Variance）一致。

* **⚡ 補充練習 1：**
  1. 請說明為什麼在深層網路中將所有權重初始化為 0 會導致模型無法學習？（提示：對稱性打破，Symmetry Breaking）。
  2. 撰寫程式碼比較：在一個 20 層的 Dense 網路中，分別使用 `random_normal`（預設變異數）與 `he_normal` 初始化，觀察第一層梯度的數值分佈差異。

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fan_in = 20

# He Normal Initialization
init_he = tf.keras.initializers.he_normal()
w = init_he(shape=(fan_in, 50))
print(w.numpy().std(), np.sqrt(2.0 / fan_in))

# Random Normal Initialization
init_random = tf.keras.initializers.random_normal()
w_random = init_random(shape=(fan_in, 50))
print(w_random.numpy().std())

plt.hist(w.numpy().flatten(), bins=50, alpha=0.5, label='He Normal')
plt.hist(w_random.numpy().flatten(), bins=50, alpha=0.5, label='Random Normal (常態分佈)')
plt.legend()
plt.title('Weight Distribution Comparison')
plt.show()
```

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def make_model(init):
    inp = tf.keras.layers.Input(shape=(100,))
    x = inp
    for _ in range(20):
        x = tf.keras.layers.Dense(50,
                                  activation="relu",
                                  kernel_initializer=init)(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inp, out)

# fake batch
x_batch = tf.random.normal((32, 100))
y_batch = tf.random.normal((32, 1))

plt.figure()
for name, init in [("random_normal", "random_normal"),
                   ("he_normal", "he_normal")]:
    model = make_model(init)
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        loss = tf.reduce_mean(tf.square(preds - y_batch))
    # 第一個 Dense 層的權重梯度
    grads = tape.gradient(loss, [model.layers[1].kernel, model.layers[-1].kernel])
    vals = grads[0].numpy().flatten()
    plt.hist(vals, bins=50, alpha=0.5, label=name)
    print(f"{name} - grad std: {vals.std():.4f}")
    print(f"{name} - grad mean: {vals.mean():.4f}")
    print(f"{name} - grad max: {vals.max():.4f}")
    print(f"{name} - grad min: {vals.min():.4f}")
    print("-" * 20)

    grad_last = grads[1]
    print(f"{name} - last layer grad std: {grad_last.numpy().flatten().std():.4f}")
    print(f"{name} - last layer grad mean: {grad_last.numpy().flatten().mean():.4f}")
    print(f"{name} - last layer grad max: {grad_last.numpy().flatten().max():.4f}")
    print(f"{name} - last layer grad min: {grad_last.numpy().flatten().min():.4f}")
    print("-" * 30)

plt.legend()
plt.title("Comparison of Gradients for Different Initializations")
plt.show()
```

---

在深層神經網路中，**若將所有權重初始化為 $0$，模型將無法學習**，其根本原因在於「對稱性無法被打破（Symmetry Breaking）」。

假設某一層的輸出為 $h_i$，則
$$
h_i = \sigma\left(\sum_j w_{ij} x_j + b_i\right)
$$
若所有權重 $w_{ij} = 0$，且偏置 $b_i$ 也相同，則所有神經元的輸出 $h_i$ 都完全一樣。

在反向傳播時，權重的梯度為
$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial h_i} \cdot x_j
$$
由於所有 $h_i$ 都一樣，$\frac{\partial \mathcal{L}}{\partial h_i}$ 也會一樣，導致所有 $w_{ij}$ 的梯度完全相同。

因此，權重更新步驟
$$
w_{ij} \leftarrow w_{ij} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}}
$$
會讓每個權重依然保持一致，無法產生差異。這種「對稱性」會讓每一層的所有神經元永遠學到相同的特徵，等同於網路只有一個有效神經元，**模型表現力大幅受限**。

**結論：**  
為了讓每個神經元能學習不同特徵，必須用隨機初始化（如 He、Glorot 等），讓每個權重一開始就有微小差異，才能有效打破對稱性，讓網路具備學習能力。

---

## 2. 非飽和活化函數 (Nonsaturating Activation Functions)

Sigmoid 在輸入值極大或極小時，梯度幾乎為 0（飽和區），這會導致訓練停滯。

* **關鍵概念：**
  * **ReLU**：計算最快，但可能有「Dead ReLU」問題（神經元輸出永遠為 0）。
  * **Leaky ReLU**：透過負數端的一個小斜率（如 0.01）解決 Dead ReLU 問題。
  * **SELU / GELU / Swish**：平滑且具有自我歸一化（Self-Normalizing）特性，在 Transformer 或深層視覺模型中表現卓越。

* **⚡ 補充練習 2：**
  1. 什麼是「自我歸一化（Self-Normalization）」？在使用 SELU 活化函數時，需要滿足哪些條件（例如輸入特徵需標準化、初始化方法等）？
  2. 實作練習：定義一個 `LeakyReLU` 活化函數，將斜率 $\alpha$ 設為 0.2，並套用至一個隱藏層中。

```python
import tensorflow as tf
leaky_relu_layer = tf.keras.layers.LeakyReLU(alpha=0.2)

# 方法 1：作為獨立層插入
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, input_shape=(100,)),
  leaky_relu_layer,
  tf.keras.layers.Dense(10, activation='softmax')
])

# 方法 2：直接在 Dense 層指定 activation 參數
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation=leaky_relu_layer, input_shape=(100,)),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

---

## 3. 批次歸一化 (Batch Normalization, BN)

BN 是訓練深層模型最常用的技術之一，它在每一層活化函數前後對資料進行標準化。

* **關鍵概念：**
  * **目的**：減輕內部協方差偏移（Internal Covariate Shift），允許使用更高的學習率（Learning Rate）。
  * **運算**：使用小批次的均值與變異數進行縮放，並引入可學習的偏移量（beta）與縮放因子（gamma）。
  * **缺點**：會增加每輪訓練的時間成本，且在推論（Inference）時需要處理統計量的差異。

* **⚡ 補充練習 3：**
  1. 批次歸一化（BN）通常放在活化函數「之前」還是「之後」？請查閱最新文獻並簡述不同學派的看法。
  2. 在 Keras 中實作一個含有 10 個 Hidden Layers 的模型，在有 BN 與沒有 BN 的情況下，觀察模型對學習率（Learning Rate）的耐受度差異。

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Prepare dataset
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
X_train, X_valid, X_test = X_train / 255, X_valid / 255, X_test / 255

# Standardize the dataset
pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

# Build model with and without Batch Normalization
def build_model(use_bn=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[28,28])) # 以 MNIST 或 Fashion MNIST 為例
    model.add(tf.keras.layers.Flatten())
    for _ in range(10):
        model.add(tf.keras.layers.Dense(256, kernel_initializer="he_normal"))
        if use_bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU()) # 或者 tf.keras.layers.Activation('relu')
    model.add(tf.keras.layers.Dense(10, activation='softmax')) # 輸出層依任務而定
    return model

# visualize the learning curve of both models to compare their training speed and performance with and without batch normalization for the various learning rates
epochs = 30
for lr in [0.1, 0.01, 0.001, 0.0001]:
    # rebuild the models on each loop iteration to ensure independence
    model_no_bn = build_model(use_bn=False)
    model_with_bn = build_model(use_bn=True)

    model_no_bn.compile(loss="sparse_categorical_crossentropy",
                        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                        metrics=["accuracy"])
    model_with_bn.compile(loss="sparse_categorical_crossentropy",
                          optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                    metrics=["accuracy"])
    print(f"\n--- Training with Learning Rate: {lr} ---") # Added for clarity
    print("Training model_no_bn...")
    history_no_bn = model_no_bn.fit(X_train_scaled, y_train, epochs=epochs,
                                    validation_data=(X_valid_scaled, y_valid),
                                    verbose=0) # Set verbose to 0 to avoid printing all epoch logs
    print("Training model_with_bn...")
    history_with_bn = model_with_bn.fit(X_train_scaled, y_train, epochs=epochs,
                                        validation_data=(X_valid_scaled, y_valid),
                                        verbose=0) # Set verbose to 0 to avoid printing all epoch logs
    
    print("\nEvaluation results:")
    loss_no_bn, acc_no_bn = model_no_bn.evaluate(X_test_scaled, y_test, verbose=0)
    loss_with_bn, acc_with_bn = model_with_bn.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Model No BN (LR={lr}) - Test Loss: {loss_no_bn:.4f}, Test Accuracy: {acc_no_bn:.4f}")
    print(f"Model With BN (LR={lr}) - Test Loss: {loss_with_bn:.4f}, Test Accuracy: {acc_with_bn:.4f}")

    # plot the learning curves for both models here
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # use each model's history to draw the curves
    plt.plot(history_no_bn.history['accuracy'], '--' ,label='No BN - Train Acc')
    plt.plot(history_no_bn.history['val_accuracy'], '--', label='No BN - Val Acc')
    plt.plot(history_with_bn.history['accuracy'], label='BN - Train Acc')
    plt.plot(history_with_bn.history['val_accuracy'], label='BN - Val Acc')
    plt.title(f'Learning Curves (LR={lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
```

---

## 4. 遷移學習 (Transfer Learning)

在電機工程中，「重用」是提高效率的關鍵。遷移學習讓我們可以站在巨人的肩膀上。

* **關鍵概念：**
  * **凍結（Freezing）**：在初期訓練時，固定預訓練層的權重，只訓練頂層的分類器。
  * **微調（Fine-tuning）**：在模型收斂後，以極低學習率解凍部分底層進行細微調整。

* **⚡ 補充練習 4：**
  1. 為什麼在進行微調（Fine-tuning）時，建議使用較小的學習率（如 $10^{-5}$）？
  2. 練習：載入 `VGG16` 的權重，凍結所有卷積層，僅更換最後的 Dense Layer 來進行 CIFAR-10 的圖像分類，並觀察準確率。

```python
import tensorflow as tf

# 載入 CIFAR-10 資料集
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0

# 1. 載入預訓練模型，不包含原本的 1000 類輸出層 (include_top=False)
base_model = tf.keras.applications.VGG16(weights='imagenet', 
                                          include_top=False, 
                                          input_shape=(32, 32, 3))

# 2. 凍結卷積層，防止權重在訓練初期被破壞
base_model.trainable = False

# 3. 建立新模型，串接自定義的 Dense Layer
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # CIFAR-10 有 10 個類別
])

# 4. 編譯模型並觀察準確率
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# 5. 訓練模型
history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_valid, y_valid))

# 6. 評估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

---

## 5. 進階優化器 (Advanced Optimizers)

除了隨機梯度下降（SGD），我們有更多工具來加速在複雜地形（Loss Landscape）中的搜索。

* **關鍵概念：**
  * **Momentum (動量)**：模擬物理慣性，加速越過局部最小值與鞍點。
  * **Adam / AdamW**：自動調整每個參數的學習率。**AdamW** 修復了 Adam 在權重衰減（Weight Decay）上的缺陷，是目前最推薦的優化器。
  * **Nadam**：Adam 加上 Nesterov 加速動量。

* **⚡ 補充練習 5：**
  1. 簡述 Adam 與 SGD + Momentum 的權衡（Trade-off）：哪一個在初期收斂快？哪一個通常能達到更好的最終泛化（Generalization）？
  2. 實作比較：使用同一個模型分別配置 `Adam(learning_rate=0.001)` 與 `SGD(learning_rate=0.01, momentum=0.9)` 訓練 20 個 Epochs，繪製 Loss 曲線圖。

---

## 6. 學習率排程與正則化 (LR Scheduling & Regularization)

控制學習率的變化以及防止模型「過度擬合」訓練資料。

* **關鍵概念：**
  * **ReduceLROnPlateau**：當驗證集準確率 (accuracy of the validation set) 不再提升時，自動降低學習率。
  * **1Cycle Scheduling**：先快速提高學習率再緩慢降低，能顯著縮短訓練時間（Super-convergence）。
  * **Dropout / MC Dropout**：訓練時隨機關閉神經元。MC Dropout 則是在預測時也開啟 Dropout 來估算模型的不確定性。

* **⚡ 補充練習 6：**
  1. 解釋為什麼「權重衰減（L2 Regularization）」可以防止模型權重過大？（從 Loss Function 的懲罰項角度說明）。
  2. 實作練習：定義一個 `MCDropout` 類別，並對測試集進行 50 次預測，計算其預測結果的標準差，視覺化模型在哪些影像上最「猶豫」。

---

## 結語

掌握這些技術，代表你已經具備了處理「深度」模型的能力。下週我們將進入第 14 章：卷積神經網路（CNN），並結合今天所學的 BN、Dropout 與 He 初始化。

**課後作業：** 請完成筆記本（Notebook）最後的 CIFAR-10 綜合練習，目標是在不使用卷積層的情況下，透過本章優化技術讓準確率達到 50% 以上。
