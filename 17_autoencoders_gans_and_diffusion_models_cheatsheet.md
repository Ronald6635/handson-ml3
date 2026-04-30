# Ch17 速查表：Autoencoders, GANs & Diffusion Models

> **核心主旨**：生成模型家族 —— Autoencoder 學習壓縮表示，VAE 採樣生成，GAN 對抗生成，Diffusion 逐步去噪生成。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Autoencoder | Encoder 壓縮 → Bottleneck → Decoder 重建，無監督學習表示 | 特徵萃取、去噪、異常偵測 |
| Denoising AE | 輸入加入雜訊，訓練重建乾淨影像，學習魯棒特徵 | 資料去噪、魯棒表示學習 |
| Sparse AE | Bottleneck 加 L1 正則，強迫稀疏啟動 | 更具解釋性的潛在表示 |
| Variational AE (VAE) | Encoder 輸出 μ 和 σ，用重參數化技巧取樣 z | 生成新樣本、插值 |
| KL Divergence Loss | VAE 的正則化項，讓潛在空間接近標準常態 | VAE 訓練必要損失 |
| GAN | Generator 造假 vs Discriminator 判真偽，對抗訓練 | 高品質影像生成 |
| DCGAN | GAN + 轉置卷積，用於影像生成 | 影像合成的基礎 GAN |
| Diffusion Model | 正向加噪 → 反向去噪，學習逐步還原資料 | 最先進的影像生成（DALL-E, SD） |

---

## 2. 關鍵 API 速查

| Keras API | 重點參數 | 用途 |
|-----------|---------|------|
| `tf.keras.layers.Dense` | `units=coding_size`, `activation="relu"` | AE 的 Encoder/Decoder 層 |
| `tf.keras.layers.Conv2DTranspose` | `filters=32`, `kernel_size=3`, `strides=2`, `padding="same"` | 卷積 AE 的 Decoder 上採樣 |
| `tf.keras.layers.UpSampling2D` | `size=(2, 2)` | 上採樣（不含可訓練參數） |
| `tf.keras.regularizers.l1(activity)` | `l1=1e-3` | 稀疏自編碼器的稀疏正則 |
| `model.add_loss()` | – | VAE 中添加 KL 散度損失 |
| `tf.keras.losses.binary_crossentropy` | – | AE/GAN 重建損失 |

---

## 3. 必備代碼片段

```python
import tensorflow as tf
import numpy as np

# 基本 Stacked Autoencoder
coding_dim = 30

encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(coding_dim, activation="selu")
])
decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="selu", input_shape=[coding_dim]),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])
autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(loss="binary_crossentropy", optimizer="nadam")
autoencoder.fit(X_train, X_train, epochs=20, validation_data=(X_val, X_val))

# Denoising Autoencoder（加入高斯雜訊層）
denoising_ae = tf.keras.Sequential([
    tf.keras.layers.GaussianNoise(0.2),  # 只在訓練時有效
    encoder,
    decoder
])
denoising_ae.compile(loss="binary_crossentropy", optimizer="nadam")
denoising_ae.fit(X_train, X_train, epochs=20, validation_data=(X_val, X_val))

# Variational Autoencoder (VAE)
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean

# Encoder
codings_size = 10
inputs = tf.keras.layers.Input(shape=[28, 28])
Z = tf.keras.layers.Flatten()(inputs)
Z = tf.keras.layers.Dense(150, activation="selu")(Z)
codings_mean = tf.keras.layers.Dense(codings_size)(Z)
codings_log_var = tf.keras.layers.Dense(codings_size)(Z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(inputs=inputs, outputs=[codings_mean, codings_log_var, codings])

# Decoder
decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
x = tf.keras.layers.Dense(150, activation="selu")(decoder_inputs)
x = tf.keras.layers.Dense(28 * 28, activation="sigmoid")(x)
outputs = tf.keras.layers.Reshape([28, 28])(x)
variational_decoder = tf.keras.Model(inputs=decoder_inputs, outputs=outputs)

# VAE Model
class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        codings_mean, codings_log_var, codings = self.encoder(inputs)
        reconstructions = self.decoder(codings)
        # KL 散度損失：讓 latent space 接近 N(0,1)
        kl_loss = -0.5 * tf.reduce_sum(
            1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
            axis=-1)
        self.add_loss(tf.reduce_mean(kl_loss) / (28 * 28))
        return reconstructions

vae = VariationalAutoEncoder(variational_encoder, variational_decoder)
vae.compile(loss="binary_crossentropy", optimizer="nadam")
vae.fit(X_train, X_train, epochs=20, validation_data=(X_val, X_val))

# 從 VAE 採樣生成新圖片
codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings).numpy()

# GAN（基本框架）
codings_size = 100
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    tf.keras.layers.Dense(150, activation="selu"),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(150, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

gan = tf.keras.Sequential([generator, discriminator])
discriminator.trainable = False  # GAN 訓練時凍結 Discriminator
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

# GAN 訓練步驟
def train_gan(X_train, gan, generator, discriminator, batch_size=32, codings_size=100):
    half_batch = batch_size // 2
    # Phase 1: 訓練 Discriminator
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    X_real = X_train[idx]
    noise = tf.random.normal(shape=[half_batch, codings_size])
    X_fake = generator(noise)
    X_combined = tf.concat([X_real, X_fake], axis=0)
    y_combined = tf.concat([tf.ones((half_batch, 1)), tf.zeros((half_batch, 1))], axis=0)
    discriminator.trainable = True
    discriminator.train_on_batch(X_combined, y_combined)

    # Phase 2: 訓練 Generator（通過 GAN，凍結 Discriminator）
    noise = tf.random.normal(shape=[batch_size, codings_size])
    y_gen = tf.ones((batch_size, 1))  # 希望 Discriminator 判為真
    discriminator.trainable = False
    gan.train_on_batch(noise, y_gen)
```

---

## 4. 常見陷阱

- **GAN Mode Collapse**：Generator 只生成少數幾種輸出（多樣性消失）。對策：使用 minibatch discrimination 或 Wasserstein GAN。
- **GAN 訓練不穩定**：Discriminator 太強 → Generator 梯度消失；Discriminator 太弱 → Generator 學不到東西。平衡兩者是 GAN 的藝術。
- **VAE 生成圖片模糊**：重建損失（binary_crossentropy）和 KL 損失的權重比例影響品質；KL 太強 → 潛在空間太緊，重建差。
- **`GaussianNoise` 只在訓練時有效**：推論時自動關閉，不需手動處理。

---

## 5. 決策指南

```
選哪種生成模型？
├── 學習資料的壓縮表示（特徵萃取） → Autoencoder
├── 資料去噪                         → Denoising AE
├── 異常偵測                          → AE（重建誤差作為異常分數）
├── 從連續潛在空間採樣生成            → VAE（生成效果較 AE 好）
├── 高品質影像/影片生成               → GAN（或 Diffusion 現代方法）
└── 最先進的影像生成（2023+）         → Diffusion Models (Stable Diffusion)

VAE Loss = Reconstruction Loss + β × KL Divergence
├── β 小 → 重建清晰但潛在空間雜亂
└── β 大 → 潛在空間規則但圖片模糊
```
