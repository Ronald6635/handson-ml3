# 課程講義：使用 RNN 與 Attention 的 NLP (Chapter 16)

自然語言是人類最複雜的資訊載體，而本章帶你用深度學習征服它。從**字元級語言模型**生成莎士比亞風格文字，到情感分析、神經機器翻譯，再到**Attention 機制**——現代 GPT、BERT 等大語言模型的核心。理解本章，你將掌握現代 NLP 的技術基礎。

---

## 1. 字元級 RNN 與文字生成

### 理論背景

**字元級語言模型**：學習在給定前 $t$ 個字元後，預測第 $t+1$ 個字元的條件機率：

$$P(\text{char}_{t+1} \mid \text{char}_1, \ldots, \text{char}_t)$$

**TextVectorization**：將原始文字轉換為整數序列（字元ID或詞彙ID）。

**Embedding 層**：將離散的字元ID映射到連續的稠密向量：

$$\text{embedding}(i) = \mathbf{E}[i, :] \in \mathbb{R}^d$$

Embedding 層的優點：相似的字元/詞彙在嵌入空間中位置相近，且向量可學習。

**文字採樣策略（Temperature）**：

$$P(\text{char}_k \propto) = \frac{\exp(\log p_k / T)}{\sum_j \exp(\log p_j / T)}$$

- $T = 1$：正常機率採樣
- $T < 1$：更保守（偏好高機率字元）
- $T > 1$：更多樣、創意（但可能亂碼）

### 核心代碼

```python
import tensorflow as tf
import numpy as np

# 下載莎士比亞文本
shakespeare_url = "https://homl.info/shakespeare"
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

# 字元級向量化
text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                    standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]
encoded -= 2  # 移除 <PAD> 和 <UNK>
n_tokens = text_vec_layer.vocabulary_size() - 2  # 約 39 個字元

# 建立資料集（滑動視窗）
def to_dataset(sequence, length, shuffle=False, seed=42, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(10_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda w: (w[:, :-1], w[:, 1:])).prefetch(1)

train_set = to_dataset(encoded[:1_000_000], length=100, shuffle=True)

# Char-RNN 模型
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens + 2, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens + 2, activation="softmax")
])
model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy")

# 文字生成（溫度採樣）
def next_char(text, temperature=1):
    y_proba = model.predict([text_vec_layer([text])])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]  # +2 for offset

def generate_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text
```

### ⚡ 補充練習 1

**理論題：** 語言模型的「困惑度 (Perplexity)」定義為：

$$\text{Perplexity} = 2^{H} = 2^{-\frac{1}{N}\sum_t \log_2 P(w_t \mid w_{<t})}$$

困惑度越低代表模型越好。若字元集大小為 39，一個「隨機猜測」的模型困惑度是多少？

**實作題：** 分別用 `temperature=0.5`、`1.0`、`2.0` 生成 200 個字元的莎士比亞風格文字，比較三種溫度下文字的多樣性和可讀性。

---

## 2. Stateful RNN 與情感分析

### 理論背景

**Stateful RNN**：跨批次保留隱藏狀態（上一個批次的最終狀態傳給下一個批次的初始狀態）。適合非常長的序列（如音訊、書籍文本），讓 RNN 能學習跨批次的長期模式。

設定要點：
1. `stateful=True` 建立帶狀態的 RNN 層
2. 每個 epoch 結束後需要 `model.reset_states()`
3. 批次必須**有序且不打亂**（每個批次繼接上一個）

**情感分析 (Sentiment Analysis)**：文字分類問題，常用於電影評論、社交媒體情緒分析。

**Masking**：序列長度不一時，需要填充（Padding）到相同長度。`Masking` 層告訴模型哪些位置是填充，不應計入損失計算。

### 核心代碼

```python
# 情感分析：IMDB 資料集
import tensorflow_datasets as tfds

raw_train_ds = tfds.load("imdb_reviews", split="train", as_supervised=True)
raw_test_ds  = tfds.load("imdb_reviews", split="test",  as_supervised=True)

# 詞彙級向量化
vocab_size = 1000
max_length  = 600

text_vec = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_sequence_length=max_length
)
text_vec.adapt(raw_train_ds.map(lambda reviews, labels: reviews))

# 帶 Masking 的情感分析模型
tf.random.set_seed(42)
sentiment_model = tf.keras.Sequential([
    text_vec,
    tf.keras.layers.Embedding(input_dim=vocab_size + 2, output_dim=16,
                               mask_zero=True),  # mask_zero=True 自動產生 mask
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
sentiment_model.compile(optimizer="nadam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"])
```

### ⚡ 補充練習 2

**理論題：** `Embedding(mask_zero=True)` 會自動生成 mask，讓後續的 LSTM 忽略填充位置。若不使用 Masking，LSTM 將填充位置的 `0` 視為真實輸入，這對模型有何影響？

**實作題：** 在 IMDB 資料集上比較三種模型的驗證集準確率：(1) `LSTM(32)`；(2) `Bidirectional(LSTM(32))`；(3) `LSTM(32)` + `LSTM(16)`（堆疊）。

---

## 3. Encoder-Decoder 與神經機器翻譯

### 理論背景

**Encoder-Decoder 架構**（序列到序列）：

- **Encoder**：讀取輸入序列（源語言），將整個序列壓縮為一個固定長度的**上下文向量 (Context Vector)** $\mathbf{c}$（最後時間步的隱藏狀態）
- **Decoder**：以 $\mathbf{c}$ 為初始狀態，逐步生成目標序列（目標語言）

**訓練策略（Teacher Forcing）**：
- 訓練時：Decoder 的輸入是**真實的**目標詞彙（即使前一步預測錯誤）
- 推理時：Decoder 的輸入是**自己前一步的預測**
- 使用特殊標記：`startofseq` 作為解碼起始符，`endofseq` 作為結束符

**限制**：固定長度的 Context Vector 是瓶頸——輸入序列越長，資訊越容易丟失。這正是 **Attention 機制**誕生的動機。

### 核心代碼

```python
# Encoder-Decoder NMT 架構
tf.random.set_seed(42)

# 準備雙語語料對 (英語-西班牙語)
import urllib.request, os
url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
# （假設資料已下載）

encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

# Encoder
enc_text_vec = tf.keras.layers.TextVectorization(max_tokens=1000,
                                                  output_sequence_length=50)
enc_embed = tf.keras.layers.Embedding(input_dim=1002, output_dim=128, mask_zero=True)
enc_lstm = tf.keras.layers.LSTM(512, return_state=True)

enc_encoded = enc_text_vec(encoder_inputs)
enc_emb = enc_embed(enc_encoded)
enc_out, enc_h, enc_c = enc_lstm(enc_emb)
# enc_h, enc_c 是 Context Vector（傳給 Decoder 的初始狀態）

# Decoder（訓練時用 Teacher Forcing）
dec_text_vec = tf.keras.layers.TextVectorization(max_tokens=1000,
                                                  output_sequence_length=50)
dec_embed = tf.keras.layers.Embedding(input_dim=1002, output_dim=128, mask_zero=True)
dec_lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
dec_dense = tf.keras.layers.Dense(1000, activation="softmax")

dec_encoded = dec_text_vec(decoder_inputs)
dec_emb = dec_embed(dec_encoded)
dec_out, _, _ = dec_lstm(dec_emb, initial_state=[enc_h, enc_c])  # 注入 Context
dec_proba = dec_dense(dec_out)

model_nmt = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
                           outputs=dec_proba)
```

### ⚡ 補充練習 3

**理論題：** 在 Encoder-Decoder 架構中，「Context Vector 瓶頸」問題是什麼？隨著輸入序列長度從 10 個詞增加到 50 個詞，翻譯品質如何變化？Attention 機制如何解決此問題？

**實作題：** 實作貪婪解碼（Greedy Decoding）函式：給定一個英文句子，循環呼叫 Decoder，每步取機率最高的詞彙，直到輸出 `endofseq` 或達到最大長度。測試 10 個英文句子的翻譯結果。

---

## 4. Attention 機制與 Transformer

### 理論背景

**Attention 機制**：讓 Decoder 在每個時間步能夠「關注」Encoder 輸出序列的不同部分，而不只依賴固定的 Context Vector。

$$\text{attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

- **Query (Q)**：當前 Decoder 隱藏狀態
- **Key (K)**：所有 Encoder 輸出
- **Value (V)**：所有 Encoder 輸出
- $\sqrt{d_k}$：縮放因子，避免內積過大（Scaled Dot-Product Attention）

**Multi-Head Attention**：在多個子空間平行執行 Attention，再拼接：

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O$$

**Positional Encoding**：Transformer 無序列順序感知，需注入位置資訊：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### 核心代碼

```python
tf.random.set_seed(42)

# 使用 Keras 的 MultiHeadAttention 層
encoder_inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int64)
decoder_inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int64)

embed_size = 128

# Positional Encoding（自訂層）
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, embed_size, **kwargs):
        super().__init__(**kwargs)
        p, i = np.meshgrid(np.arange(max_length),
                           2 * np.arange(embed_size // 2))
        angles = p / 10_000 ** (i / embed_size)
        sin_cos = np.empty((1, max_length, embed_size))
        sin_cos[0, :, ::2]  = np.sin(angles).T
        sin_cos[0, :, 1::2] = np.cos(angles).T
        self.positional_encoding = tf.constant(sin_cos.astype(np.float32))

    def call(self, X):
        return X + self.positional_encoding[:, :tf.shape(X)[1], :]

# Transformer Encoder Block
enc_emb = tf.keras.layers.Embedding(input_dim=1002, output_dim=embed_size)(encoder_inputs)
enc_pos = PositionalEncoding(max_length=512, embed_size=embed_size)(enc_emb)

mha = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=embed_size // 8)
enc_out = mha(enc_pos, enc_pos)  # Self-Attention
enc_out = tf.keras.layers.LayerNormalization()(enc_out + enc_pos)
enc_out = tf.keras.layers.Dense(256, activation="relu")(enc_out)
enc_out = tf.keras.layers.Dense(embed_size)(enc_out)
enc_out = tf.keras.layers.LayerNormalization()(enc_out + enc_pos)
```

### ⚡ 補充練習 4

**理論題：** Scaled Dot-Product Attention 為何需要除以 $\sqrt{d_k}$？若不縮放，當 $d_k$ 很大時，Softmax 的梯度會有什麼問題？

**實作題：** 用 `tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)` 建立一個簡單的 Transformer Encoder（2 層），在 IMDB 情感分析任務上訓練，比較其與 LSTM 的收斂速度和最終準確率。

---

## 結論

NLP 的深度學習演進：

- **Char-RNN**：文字生成的入門，理解語言模型訓練
- **Embedding + LSTM**：情感分析的標準做法
- **Encoder-Decoder**：序列到序列（機器翻譯）的基礎架構
- **Attention + Transformer**：現代 LLM（GPT、BERT）的基石

下一章（Ch17）轉向生成模型：自動編碼器、VAE 和 GAN。

---

## 課後作業

**作業：建立一個多語言情感分析器**

1. 使用 `tensorflow_datasets.load("imdb_reviews")` 訓練一個帶 Masking 的 LSTM 情感分析模型（Embedding + LSTM + Dense），在驗證集達到至少 85% 準確率。

2. 比較以下三種輸入表示方式的效果：
   - `TextVectorization(max_tokens=1000)`（詞彙索引）
   - `TextVectorization(max_tokens=10000)`（更大詞彙表）
   - Bidirectional LSTM

3. 用你的模型預測 5 條自己寫的英文影評，判斷其情感正負面。
