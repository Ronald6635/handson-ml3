<!-- meta-title: 自動編碼器、GAN 與擴散模型完整指南：生成式 AI 原理與實作 -->
<!-- meta-description: 深入生成式 AI：堆疊自動編碼器、去噪自動編碼器、稀疏自動編碼器、變分自動編碼器（VAE 與重參數化技巧）、生成對抗網路（GAN 交替訓練）與現代擴散模型的原理。 -->
<!-- meta-keywords: Python, 自動編碼器, VAE, GAN, 生成式AI, 擴散模型, Keras, TensorFlow, 深度學習, 降維 -->
<!-- meta-hashtags: #Python #自動編碼器 #VAE #GAN #生成式AI #擴散模型 #Keras #深度學習 #教學 #DataScience -->

# 🐍 自動編碼器、GAN 與擴散模型：生成式 AI 從零開始

Midjourney、Stable Diffusion、DALL-E——這些令人驚嘆的圖像生成 AI 背後都依賴本章的技術。從最基礎的**自動編碼器（Autoencoder）**到**變分自動編碼器（VAE）**、**生成對抗網路（GAN）**，最後認識現代的**擴散模型（Diffusion Model）**——帶你從原理到實作，理解生成式 AI 的本質。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🔧 堆疊自動編碼器](#stacked-ae)
- [🎨 卷積自動編碼器](#conv-ae)
- [🌀 去噪自動編碼器](#denoising-ae)
- [💡 稀疏自動編碼器](#sparse-ae)
- [🎲 變分自動編碼器 (VAE)](#vae)
- [⚔️ 生成對抗網路 (GAN)](#gan)
- [🌫️ 擴散模型概念](#diffusion)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **自動編碼器**強制網路學習資料的**壓縮表示（潛在空間）**，捕捉最重要的特徵
- **VAE** 的核心是**重參數化技巧（Reparameterization Trick）**：讓採樣操作可微分，使梯度能反向傳播
- **GAN** 的訓練是博弈（Generator vs Discriminator）：困難之處在於維持二者的平衡
- **模式崩潰（Mode Collapse）** 是 GAN 的主要問題：Generator 只生成幾種模式，忽略資料的多樣性
- 現代**擴散模型**逐步加噪然後反向去噪，效果優於 GAN，且訓練更穩定

---

## <a id="stacked-ae"></a>🔧 堆疊自動編碼器

💡 **實際應用情境：** 工廠感測器資料的異常檢測——用正常資料訓練自動編碼器後，異常資料的重建誤差會明顯偏高（因為編碼器沒見過這類模式），以此識別設備故障。

### 範例 1: 基礎堆疊自動編碼器

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

# 載入 Fashion-MNIST（示範用）
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype(np.float32) / 255.0  # 正規化到 [0, 1]
X_test  = X_test.astype(np.float32) / 255.0
X_train_flat = X_train.reshape(-1, 28 * 28)   # 展平
X_test_flat  = X_test.reshape(-1, 28 * 28)

# 對稱結構的自動編碼器
# Encoder：784 → 256 → 128 → 32（潛在空間）
# Decoder：32 → 128 → 256 → 784（重建）

# 方法 1：Sequential API（簡潔）
autoencoder = keras.Sequential([
    # Encoder
    keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32),       # 潛在空間（線性激活）
    # Decoder
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid")  # 輸出在 [0,1]
])

# 輸入 = 輸出（重建任務）
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

# 訓練
history = autoencoder.fit(
    X_train_flat, X_train_flat,    # X 和 y 都是 X（自監督）
    epochs=20,
    batch_size=256,
    validation_data=(X_test_flat, X_test_flat),
    verbose=0
)
print(f"驗證重建損失: {min(history.history['val_loss']):.4f}")
```

### 範例 2: 分離 Encoder 和 Decoder（方法 2：Functional API）

```python
# 使用 Functional API 分別建立 Encoder 和 Decoder
encoder_input = keras.Input(shape=(784,), name="encoder_input")
z_encoded = keras.layers.Dense(256, activation="relu")(encoder_input)
z_encoded = keras.layers.Dense(128, activation="relu")(z_encoded)
z_encoded = keras.layers.Dense(32, name="latent_space")(z_encoded)
encoder = keras.Model(encoder_input, z_encoded, name="encoder")

decoder_input = keras.Input(shape=(32,), name="decoder_input")
x_decoded = keras.layers.Dense(128, activation="relu")(decoder_input)
x_decoded = keras.layers.Dense(256, activation="relu")(x_decoded)
x_decoded = keras.layers.Dense(784, activation="sigmoid")(x_decoded)
decoder = keras.Model(decoder_input, x_decoded, name="decoder")

# 組合成完整自動編碼器
ae_input = keras.Input(shape=(784,))
ae_output = decoder(encoder(ae_input))
autoencoder_full = keras.Model(ae_input, ae_output, name="autoencoder")
autoencoder_full.compile(optimizer="adam", loss="binary_crossentropy")

# 異常檢測
def detect_anomalies(model, X: np.ndarray,
                     threshold_percentile: float = 95) -> np.ndarray:
    """基於重建誤差的異常檢測"""
    reconstructed = model.predict(X, verbose=0)
    recon_errors = np.mean(np.square(X - reconstructed), axis=1)  # MSE per sample
    threshold = np.percentile(recon_errors, threshold_percentile)
    return recon_errors > threshold  # True = 異常

print("自動編碼器異常檢測框架建立完成！")
```

**✅ 程式碼逐行解析：**

1. 損失函數 `binary_crossentropy`: 輸入像素被視為 Bernoulli 分佈，重建誤差用交叉熵而非 MSE（更適合 [0,1] 輸出）
2. Encoder 和 Decoder 分開定義：可以單獨使用 Encoder 進行降維，或單獨使用 Decoder 從潛在向量生成圖像

---

## <a id="conv-ae"></a>🎨 卷積自動編碼器

### 範例 3: 用於圖像的卷積自動編碼器

```python
# 卷積自動編碼器（圖像資料）
conv_autoencoder = keras.Sequential([
    # Encoder（卷積下採樣）
    keras.layers.Reshape((28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2), padding="same"),  # 28×28 → 14×14
    keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2), padding="same"),  # 14×14 → 7×7

    # Decoder（轉置卷積上採樣）
    keras.layers.Conv2DTranspose(16, (3, 3), activation="relu", padding="same"),
    keras.layers.UpSampling2D((2, 2)),                  # 7×7 → 14×14
    keras.layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same"),
    keras.layers.UpSampling2D((2, 2)),                  # 14×14 → 28×28
    keras.layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same"),
    keras.layers.Reshape((28 * 28,))
], name="conv_autoencoder")

conv_autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
```

---

## <a id="denoising-ae"></a>🌀 去噪自動編碼器

### 範例 4: 去噪自動編碼器

```python
# 去噪自動編碼器：輸入帶噪聲的圖像，輸出乾淨的圖像
def add_gaussian_noise(X: np.ndarray, noise_factor: float = 0.3) -> np.ndarray:
    """添加高斯噪聲"""
    noisy = X + noise_factor * np.random.randn(*X.shape)
    return np.clip(noisy, 0., 1.)  # 確保值在 [0, 1]

X_train_noisy = add_gaussian_noise(X_train_flat)
X_test_noisy  = add_gaussian_noise(X_test_flat)

# 訓練：輸入帶噪圖像，目標是乾淨圖像
denoising_ae = keras.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid")
])
denoising_ae.compile(optimizer="adam", loss="binary_crossentropy")

# 訓練：X=帶噪圖像，y=乾淨圖像（去噪任務）
# denoising_ae.fit(X_train_noisy, X_train_flat, epochs=10, ...)
print("去噪自動編碼器可用於圖像去噪和缺失資料填補！")
```

**🎯 重點摘要:**

- 去噪 AE 比普通 AE 學到更**健壯的特徵**——必須學會忽略噪聲，只關注本質結構
- 應用：圖像修復、缺失值填補（在 ID 欄位添加噪聲後重建）

---

## <a id="vae"></a>🎲 變分自動編碼器 (VAE)

💡 **實際應用情境：** 普通自動編碼器的潛在空間不連續——在兩個點之間插值可能產生無意義的結果。VAE 強制潛在空間呈高斯分佈，插值生成有意義的新圖像（如：在「運動鞋」和「靴子」之間生成過渡鞋款）。

### 範例 5: VAE（含重參數化技巧）

```python
class Sampling(keras.layers.Layer):
    """重參數化技巧：z = μ + ε * σ（ε ~ N(0,1)）

    將採樣轉化為可微分操作：隨機性來自 ε，而非 z
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]

        # 從標準常態分佈採樣噪聲
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))

        # 重參數化：z = μ + σ * ε = μ + exp(log_var/2) * ε
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ── VAE Encoder ──
latent_dim = 16
encoder_input = keras.Input(shape=(784,), name="vae_encoder_input")
x = keras.layers.Dense(256, activation="relu")(encoder_input)
x = keras.layers.Dense(128, activation="relu")(x)
z_mean    = keras.layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
z         = Sampling()([z_mean, z_log_var])  # 重參數化採樣

vae_encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="vae_encoder")

# ── VAE Decoder ──
decoder_input = keras.Input(shape=(latent_dim,))
x = keras.layers.Dense(128, activation="relu")(decoder_input)
x = keras.layers.Dense(256, activation="relu")(x)
vae_decoder_output = keras.layers.Dense(784, activation="sigmoid")(x)
vae_decoder = keras.Model(decoder_input, vae_decoder_output, name="vae_decoder")


# ── 自訂 VAE 訓練邏輯（覆寫 train_step）──
class VAE(keras.Model):
    """變分自動編碼器（Variable Autoencoder）"""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # 重建損失
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            ) * 784

            # KL 散度損失（讓潛在空間接近標準正態分佈）
            # KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )

            # 總損失 = 重建損失 + KL 散度損失
            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"total_loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss}


vae = VAE(vae_encoder, vae_decoder)
vae.compile(optimizer=keras.optimizers.Adam(1e-3))
# vae.fit(X_train_flat, epochs=30, batch_size=128)
print("VAE 定義完成！潛在空間維度:", latent_dim)
```

**✅ 程式碼逐行解析：**

1. `Sampling` 層的重參數化技巧：$z = \mu + \sigma \cdot \varepsilon$，$\varepsilon \sim N(0,I)$——隨機性來自 ε（固定），梯度通過 μ 和 σ 反向傳播
2. KL 散度損失：懲罰潛在分佈偏離標準正態分佈，確保潛在空間是連續的（可插值）
3. `reconstruction_loss + kl_loss`：重建損失讓 VAE 學會重建；KL 損失讓潛在空間有意義

**🎯 重點摘要:**

- VAE 的潛在空間是**連續的**：從 N(0, I) 採樣 z，通過 decoder 生成新圖像
- 調整 KL 損失的權重（β-VAE）：更大的 KL 係數 → 更解耦的潛在因子，但重建質量下降

---

## <a id="gan"></a>⚔️ 生成對抗網路 (GAN)

### 範例 6: GAN 交替訓練迴圈

```python
# GAN = Generator（生成器）+ Discriminator（判別器）的博弈
# Generator：雜訊 → 假圖像（試圖騙過 Discriminator）
# Discriminator：圖像 → 真/假分類（試圖分辨真假）

latent_dim_gan = 32

# ── Generator ──
generator = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(latent_dim_gan,)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid"),
    keras.layers.Reshape((28, 28))
], name="generator")

# ── Discriminator ──
discriminator = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.4),      # 正則化（防止 Discriminator 過擬合）
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1, activation="sigmoid")  # 1=真, 0=假
], name="discriminator")


class GAN(keras.Model):
    """生成對抗網路（交替訓練）"""

    def __init__(self, discriminator, generator, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator     = generator
        self.latent_dim    = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # ── 訓練 Discriminator ──
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator(noise, training=False)

        # 合併真實和假圖像
        combined_images = tf.concat([fake_images, real_images], axis=0)
        labels = tf.concat([
            tf.zeros((batch_size, 1)),  # 假圖像 → 0
            tf.ones((batch_size, 1))    # 真圖像 → 1
        ], axis=0)
        # 添加標籤噪聲（提升穩定性）
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )

        # ── 訓練 Generator（讓 Discriminator 認為假圖像是真的）──
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))  # Generator 目標：讓 D 輸出 1（真）

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            predictions = self.discriminator(fake_images, training=False)
            g_loss = self.loss_fn(misleading_labels, predictions)
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {"d_loss": self.d_loss_tracker.result(),
                "g_loss": self.g_loss_tracker.result()}


gan = GAN(discriminator, generator, latent_dim=latent_dim_gan)
gan.compile(
    d_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),  # β₁=0.5 是 GAN 的常見設定
    g_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy()
)
print("GAN 建立完成！訓練時注意 d_loss ≈ g_loss 才代表平衡。")
```

**🎯 重點摘要:**

- GAN 訓練訣竅：如果 d_loss → 0，Generator 已被 Discriminator 完全壓制（模式崩潰前兆）
- 標籤噪聲（Label Smoothing）：將真實標籤從 1 改為 0.9~1，防止 Discriminator 過度自信

---

## <a id="diffusion"></a>🌫️ 擴散模型概念

### 範例 7: 擴散過程概念（前向加噪）

```python
# 擴散模型的直覺：
# 前向過程：原始圖像 x₀ → 逐步添加噪聲 → 純雜訊 xₜ
# 反向過程：訓練一個 U-Net 預測每步的噪聲，逆向還原圖像

def forward_diffusion(x0: np.ndarray, t: int, T: int = 1000) -> tuple:
    """前向擴散過程：在時間步 t 添加噪聲

    x_t = sqrt(α̅_t) * x₀ + sqrt(1 - α̅_t) * ε
    """
    # 線性噪聲排程（β 從 0.0001 到 0.02）
    betas  = np.linspace(0.0001, 0.02, T)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)  # 累積乘積

    alpha_bar_t = alpha_bars[t]
    noise = np.random.randn(*x0.shape).astype(np.float32)

    # 加噪公式（可以直接從 x₀ 跳到任意時間步 t）
    x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise

# 示範不同時間步的加噪效果
sample_image = X_train_flat[0:1]
for t_step in [0, 100, 500, 999]:
    x_t, noise = forward_diffusion(sample_image, t_step)
    signal_ratio = np.var(np.sqrt(1-1e-4) * sample_image) / np.var(x_t)
    print(f"t={t_step:4d}: 訊噪比 ≈ {signal_ratio:.4f}")

print("\n擴散模型的訓練：預測每個時間步添加的噪聲 ε（MSE 損失）")
print("推論：從純雜訊開始，反覆去噪 T 步，生成新圖像")
```

**🎯 重點摘要:**

- 擴散模型 vs GAN：擴散模型訓練更穩定（無博弈問題），但推論速度慢（需要反覆去噪 T 步）
- DDPM（Denoising Diffusion Probabilistic Models）是 Stable Diffusion 的基礎架構
- 現代加速方法（DDIM、DPM-Solver）將推論步數從 1000 降至 20~50

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: VAE 和普通自動編碼器有什麼差別？**

A: 普通 AE 的潛在空間是任意的（不連續、不規則）；VAE 強制潛在空間服從高斯分佈（通過 KL 損失），使潛在空間連續且可插值。VAE 可以從潛在空間採樣**生成新圖像**，普通 AE 不能。

**Q2: GAN 的模式崩潰（Mode Collapse）如何緩解？**

A: (1) 使用 Wasserstein GAN（WGAN）改變損失函數；(2) Mini-Batch Discrimination（讓 Discriminator 看到一個 batch 的多個樣本）；(3) 漸進式增長 GAN（Progressive Growing GAN）；(4) 添加標籤噪聲。

**Q3: 什麼情況下用 VAE vs GAN？**

A: VAE 適合需要**插值/探索潛在空間**的場景（生成過渡圖像），且訓練穩定；GAN 生成的圖像視覺質量通常更高（更清晰），但訓練不穩定。現代最佳方案：擴散模型（兩者的優點）。

**Q4: 自動編碼器適合什麼任務？**

A: (1) **降維/視覺化**（比 PCA 更強）；(2) **異常檢測**（重建誤差高 = 異常）；(3) **去噪**；(4) **特徵學習**（作為預訓練的 encoder）。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #自動編碼器 #VAE #GAN #生成式AI #擴散模型 #Keras #TensorFlow #深度學習 #重參數化技巧 #程式設計 #教學 #DataScience #MachineLearning
