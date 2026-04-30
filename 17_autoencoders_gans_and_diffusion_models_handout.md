# 課程講義：自動編碼器、GAN 與擴散模型 (Chapter 17)

生成模型（Generative Models）的目標是學習資料的分佈，讓模型能夠**生成從未見過的新樣本**。本章介紹三大生成模型技術：**自動編碼器（AE）**用於降維與去噪、**變分自動編碼器（VAE）**用於連續潛空間採樣、**生成對抗網路（GAN）**用於超高品質圖片生成，最後簡介擴散模型（Diffusion Models）的基本原理。

---

## 1. 自動編碼器：編碼、解碼與重建

### 理論背景

**自動編碼器 (Autoencoder)**：一種無監督學習的神經網路，以**重建輸入**為目標，學習資料的壓縮表示。

$$\text{Encoder}: \mathbf{x} \mapsto \mathbf{z} = f(\mathbf{x})$$
$$\text{Decoder}: \mathbf{z} \mapsto \hat{\mathbf{x}} = g(\mathbf{z})$$

訓練目標：最小化重建誤差：

$$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} \|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\|^2$$

**潛空間 (Latent Space) / 瓶頸 (Bottleneck)**：中間層維度 $d \ll$ 輸入維度 $n$，迫使模型學習最重要的特徵表示。

**主要應用**：

| 應用 | 說明 |
|------|------|
| 非線性降維 | 比 PCA 更強的表示能力 |
| 去噪 | 去噪自動編碼器學習移除雜訊 |
| 異常偵測 | 正常樣本重建誤差小，異常樣本大 |
| 預訓練 | 為分類器提供良好的初始特徵 |

### 核心代碼

```python
import tensorflow as tf
import numpy as np

# 堆疊自動編碼器（Stacked Autoencoder）
tf.random.set_seed(42)

# Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full = X_train_full.astype("float32") / 255.

# 建立 Encoder + Decoder
stacked_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),   # 瓶頸層（30 維）
])

stacked_decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=[30]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),  # 還原為 784 維
    tf.keras.layers.Reshape([28, 28])
])

stacked_ae = tf.keras.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy", optimizer="nadam")
stacked_ae.fit(X_train_full, X_train_full,  # 輸入 = 標籤 = 原圖
               epochs=20, validation_split=0.1)

# 重建效果
X_reconstructed = stacked_ae.predict(X_test[:5])
```

### ⚡ 補充練習 1

**理論題：** 自動編碼器與 PCA 的關係：線性自動編碼器（沒有活化函數、一個瓶頸層）與 PCA 等價嗎？若使用非線性活化函數，自動編碼器能學習到什麼 PCA 學不到的表示？

**實作題：** 訓練一個在 Fashion MNIST 上的堆疊自動編碼器，將瓶頸層維度分別設為 2、30、100，比較重建圖片的視覺品質，並繪製 2D 潛空間的散點圖（顏色=類別），觀察哪些類別在潛空間中聚集。

---

## 2. 去噪自動編碼器與稀疏自動編碼器

### 理論背景

**去噪自動編碼器 (Denoising Autoencoder)**：

訓練時加入雜訊，要求輸出乾淨的重建：

$$\mathcal{L}_{\text{denoise}} = \|\mathbf{x} - g(f(\tilde{\mathbf{x}}))\|^2$$

其中 $\tilde{\mathbf{x}} = \mathbf{x} + \varepsilon$（高斯雜訊）或隨機置零某些輸入（Dropout 雜訊）。

好處：迫使編碼器學習**魯棒的特徵表示**，而非簡單地記憶輸入。

**稀疏自動編碼器 (Sparse Autoencoder)**：

對瓶頸層的激活值加入稀疏懲罰（L1 正則化），迫使大多數神經元「靜默」：

$$\mathcal{L}_{\text{sparse}} = \text{Reconstruction Loss} + \alpha \sum_j |a_j|$$

用 `activity_regularizer=tf.keras.regularizers.L1(1e-4)` 實現。

### 核心代碼

```python
# 去噪自動編碼器（使用 GaussianNoise 層）
tf.random.set_seed(42)

denoising_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.GaussianNoise(stddev=0.2),  # 訓練時加入雜訊，推理時自動關閉
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),
])

denoising_ae = tf.keras.Sequential([
    denoising_encoder,
    stacked_decoder  # 與前面共用 decoder（不含雜訊層）
])
denoising_ae.compile(loss="binary_crossentropy", optimizer="nadam")
denoising_ae.fit(X_train_full, X_train_full, epochs=10)

# 稀疏自動編碼器
sparse_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu",
                          activity_regularizer=tf.keras.regularizers.L1(1e-4))
])
```

### ⚡ 補充練習 2

**理論題：** 稀疏自動編碼器的潛空間向量通常大多數值接近 0，只有少數維度被激活。這與人腦神經元的「稀疏啟動」假說有何相似之處？這種稀疏性對下游分類任務有何好處？

**實作題：** 用去噪自動編碼器對 Fashion MNIST 測試圖片加入高斯雜訊（`stddev=0.5`），輸入給訓練好的模型，展示去噪前後的圖片對比。

---

## 3. 變分自動編碼器（VAE）與重參數化技巧

### 理論背景

**VAE (Variational Autoencoder)**：將潛空間從**點**（確定性向量）改為**分布**（高斯分布的均值和變異數）。

Encoder 輸出 $\boldsymbol{\mu}$ 和 $\log \boldsymbol{\sigma}^2$（對數變異數）。

**重參數化技巧 (Reparameterization Trick)**：

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

這樣隨機採樣步驟的梯度可以流過 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$（不能對隨機採樣求梯度，但可以對參數求梯度）。

**VAE 損失函數**（ELBO 的負值）：

$$\mathcal{L}_{\text{VAE}} = \underbrace{\text{Reconstruction Loss}}_{\text{重建品質}} + \underbrace{KL\left(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \| \mathcal{N}(\mathbf{0}, \mathbf{I})\right)}_{\text{正則化潛空間}}$$

KL 散度（封閉形式）：

$$KL = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**VAE 的關鍵特性**：潛空間平滑且連續，可以在潛空間中插值或採樣，生成新圖片。

### 核心代碼

```python
# VAE 編碼器
tf.random.set_seed(42)

codings_size = 10

inputs = tf.keras.layers.Input(shape=[28, 28])
Z = tf.keras.layers.Flatten()(inputs)
Z = tf.keras.layers.Dense(150, activation="relu")(Z)
Z = tf.keras.layers.Dense(100, activation="relu")(Z)
codings_mean    = tf.keras.layers.Dense(codings_size)(Z)       # μ
codings_log_var = tf.keras.layers.Dense(codings_size)(Z)       # log(σ²)
codings         = Sampling()([codings_mean, codings_log_var])  # z = μ + σε

# 自訂 Sampling 層（重參數化技巧）
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return mean + tf.exp(log_var / 2) * tf.random.normal(tf.shape(mean))

# VAE 損失（含 KL 散度）
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, X):
        codings_mean, codings_log_var, codings = self.encoder(X)
        reconstructions = self.decoder(codings)
        # KL 散度損失
        kl_loss = -0.5 * tf.reduce_sum(
            1 + codings_log_var - tf.square(codings_mean) - tf.exp(codings_log_var),
            axis=-1
        )
        self.add_loss(tf.reduce_mean(kl_loss) / (28 * 28))
        return reconstructions

# 生成新圖片（從標準常態分布採樣）
codings_random = tf.random.normal([12, codings_size])
generated_images = vae_decoder.predict(codings_random)
```

### ⚡ 補充練習 3

**理論題：** VAE 的 KL 散度項有什麼正則化效果？若 KL 散度的係數設為 0（只保留重建損失），VAE 退化成什麼？若係數設得太大，潛空間的特性會有什麼問題（「後驗崩潰」）？

**實作題：** 在 Fashion MNIST 的 2D VAE（`codings_size=2`）訓練完成後，繪製潛空間的 2D 網格：對 $z_1, z_2 \in [-3, 3]$ 均勻採樣，解碼每個點，組成 $10 \times 10$ 的圖片矩陣，觀察潛空間的語意插值效果。

---

## 4. 生成對抗網路（GAN）

### 理論背景

**GAN (Generative Adversarial Network)**：Generator 和 Discriminator 的雙人零和博弈。

- **Generator $G$**：從潛空間隨機雜訊 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 生成假圖片
- **Discriminator $D$**：判斷圖片是真實（1）還是假的（0）

訓練目標：

$$\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z} [\log(1 - D(G(\mathbf{z})))]$$

**交替訓練流程**：

1. **訓練 Discriminator**：固定 Generator，用真假資料更新 D（最大化分辨能力）
2. **訓練 Generator**：固定 Discriminator，更新 G（欺騙 D）

**GAN 的訓練難點**：

- **模式崩潰 (Mode Collapse)**：Generator 只生成少數幾種樣本
- **訓練不穩定**：D 太強或太弱都會使訓練崩潰
- **解法**：Wasserstein GAN（WGAN）、漸進式訓練（ProGAN）、歸一化技術

### 核心代碼

```python
tf.random.set_seed(42)

codings_size = 30

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=[codings_size]),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # 真/假
])
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

# GAN（訓練 Generator 時，Discriminator 凍結）
discriminator.trainable = False
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

# 交替訓練
def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            # 訓練 Discriminator
            noise = tf.random.normal([batch_size, codings_size])
            fake_images = generator(noise, training=True)
            X_combined = tf.concat([X_batch, fake_images], axis=0)
            y_combined  = tf.constant([[1.]] * batch_size + [[0.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_combined, y_combined)
            # 訓練 Generator
            noise = tf.random.normal([batch_size, codings_size])
            y_mislead = tf.ones([batch_size, 1])  # 標記為「真實」來欺騙 D
            discriminator.trainable = False
            gan.train_on_batch(noise, y_mislead)
```

### ⚡ 補充練習 4

**理論題：** GAN 的「均衡點」在哪裡？理想情況下，訓練結束時 Discriminator 的輸出是什麼（對真實和假圖片各為多少）？這個均衡點是穩定的嗎？

**實作題：** 訓練 Fashion MNIST GAN，每 10 個 epoch 保存 Generator 生成的 10 張圖片，製成動畫或拼圖，觀察生成品質隨訓練進行的改善（注意模式崩潰的跡象）。

---

## 結論

生成模型的三大典範：

- **AE（自動編碼器）**：確定性壓縮，適合去噪和降維；不適合生成多樣樣本
- **VAE**：機率性潛空間，平滑可插值；生成圖片略模糊但多樣
- **GAN**：對抗訓練，生成品質最高（可達照片級）；訓練不穩定
- **Diffusion**：現代 SOTA，透過迭代去噪生成高品質圖片（Stable Diffusion、DALL-E 2）

下一章（Ch18）轉向強化學習，讓 Agent 在環境中學習最優策略。

---

## 課後作業

**作業：潛空間操作實驗**

在 Fashion MNIST 上訓練 VAE（`codings_size=10`）：

1. 視覺化**潛空間插值**：找到一件T恤（類別0）和一件連衣裙（類別3）在潛空間的均值 $\boldsymbol{\mu}$，在兩點之間線性插值（10個均勻步驟），解碼每個插值點，觀察圖片的「變形」過程。

2. 實作**潛空間算術**：計算某個類別的平均潛空間向量（如所有「鞋子」的均值），將另一個樣本的潛空間向量加上這個方向，觀察解碼後的圖片是否呈現鞋子的特徵。

3. **對比 AE vs VAE 的潛空間**：訓練相同架構但潛空間為點（非分布）的 AE，比較 AE 和 VAE 的潛空間在 2D（`codings_size=2`）下的散點圖分布，解釋為何 VAE 的潛空間更「均勻」。
