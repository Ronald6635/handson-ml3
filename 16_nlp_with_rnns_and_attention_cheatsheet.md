# Ch16 速查表：NLP with RNNs & Attention

> **核心主旨**：從字元 RNN 到 Transformer —— Attention 機制是現代 NLP 的核心；實務上優先考慮 Hugging Face 預訓練模型。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| `TextVectorization` | 將文字轉為 token 索引 | NLP 管線的第一層 |
| `Embedding` | 將 token 索引映射為密集向量 | 所有 NLP 任務 |
| Stateful RNN | 跨批次保留隱藏狀態，適合長文字串流 | 字元級語言模型 |
| Encoder-Decoder (seq2seq) | 編碼輸入序列 → 解碼輸出序列 | 機器翻譯、摘要 |
| Attention Mechanism | Decoder 每步「查詢」Encoder 所有輸出 | seq2seq 改進 |
| Multi-Head Attention | 多組 Q/K/V，並行學習多種注意力模式 | Transformer 核心 |
| Positional Encoding | 為 Transformer 注入位置資訊（無序列歸納偏置） | Transformer 必備 |
| Transformer | MHA + Feed-Forward + Residual + LayerNorm | 現代 NLP 標準架構 |
| BERT / GPT | 預訓練 Transformer；BERT 雙向，GPT 單向 | 幾乎所有 NLP 任務 |

---

## 2. 關鍵 API 速查

| Keras / HuggingFace API | 重點參數 | 用途 |
|-------------------------|---------|------|
| `tf.keras.layers.TextVectorization` | `max_tokens=`, `output_mode="int"`, `output_sequence_length=` | 文字前處理 |
| `tf.keras.layers.Embedding` | `input_dim=vocab_size`, `output_dim=16`, `mask_zero=True` | 嵌入層 |
| `tf.keras.layers.Masking` | `mask_value=0.0` | 遮蔽填充 token |
| `tf.keras.layers.Bidirectional` | `layer=LSTM(...)` | 雙向 RNN |
| `tf.keras.layers.MultiHeadAttention` | `num_heads=8`, `key_dim=64` | Multi-Head Attention |
| `tf.keras.layers.LayerNormalization` | – | Transformer 中的歸一化 |
| `transformers.pipeline()` | `task`, `model=` | HF 快速推論 |
| `transformers.AutoTokenizer` | `from_pretrained("bert-base-uncased")` | 載入 tokenizer |
| `transformers.TFAutoModelForSequenceClassification` | `from_pretrained(...)` | 載入預訓練模型 |

---

## 3. 必備代碼片段

```python
import tensorflow as tf

# 文字前處理管線
vocab_size = 1000
max_length = 200

text_vec_layer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size, output_sequence_length=max_length)
text_vec_layer.adapt(train_texts)  # 建立詞彙表

# 情感分析（Embedding + LSTM）
model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Embedding(input_dim=vocab_size + 2, output_dim=16, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Transformer Block（自訂）
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.mha(x, x, training=training)  # Self-attention
        x = self.ln1(x + self.drop1(attn_output, training=training))
        ffn_output = self.ffn(x)
        return self.ln2(x + self.drop2(ffn_output, training=training))

# Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        P = tf.Variable(tf.zeros((1, max_len, d_model)), trainable=False)
        positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angles = positions / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        P = tf.concat([tf.sin(angles[:, 0::2]), tf.cos(angles[:, 1::2])], axis=-1)
        self.P = tf.expand_dims(P, 0)

    def call(self, x):
        return x + self.P[:, :tf.shape(x)[1]]

# HuggingFace Pipeline（最快的 NLP 起點）
from transformers import pipeline

# 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love TensorFlow!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]

# 文字生成
generator = pipeline("text-generation", model="gpt2")
generated = generator("The future of AI is", max_length=50, num_return_sequences=1)

# Fine-tune BERT（TensorFlow）
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = TFAutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)

inputs = tokenizer(train_texts, truncation=True, padding=True,
                   return_tensors="tf", max_length=128)
dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), train_labels)).batch(16)

model_bert.compile(optimizer=tf.keras.optimizers.Adam(2e-5),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_bert.fit(dataset, epochs=3)
```

---

## 4. 常見陷阱

- **`mask_zero=True` 要在 Embedding 層設定**：告訴後續層忽略填充 token，不然 LSTM/Attention 會把 padding 也計算進去。
- **Stateful RNN 要手動 reset state**：`model.reset_states()` 要在每個文件/序列開始時呼叫，批次大小必須固定。
- **Transformer 不包含歸納偏置（位置）**：`Positional Encoding` 是必要的，否則模型不知道 token 的順序。
- **`TextVectorization.adapt()` 只能看訓練集**：測試集的罕見詞會映射到 `[UNK]` token，這是正常行為。

---

## 5. 決策指南

```
NLP 任務選模型：
├── 文字分類/情感分析       → 用 HuggingFace BERT fine-tune（首選）
│                            → 或 TextVec + Embedding + Bidirectional LSTM
├── 文字生成                → HuggingFace GPT-2/GPT-3.5（API）
├── 機器翻譯/摘要            → HuggingFace seq2seq 模型（T5/mBART）
├── 命名實體識別（NER）      → HuggingFace pipeline("ner")
└── 從頭訓練（罕見）         → Transformer + 大量資料 + 大 GPU

Attention 類型：
├── Self-Attention (Encoder)    → 理解句子上下文（BERT）
├── Causal Self-Attention (Decoder) → 生成任務（GPT）
└── Cross-Attention             → seq2seq（翻譯、摘要）
```
