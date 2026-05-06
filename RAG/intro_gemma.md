# 💎 Gemma 基礎入門指南

## 1. 這是什麼？ (What is Gemma?)

**Gemma** 是由 Google DeepMind 團隊打造的全新一代「開放權重 (Open-weights)」大型語言模型。你可以把它視為是你之前在雲端使用的 **Gemini 模型的「開源輕量版親戚」**。

它採用了與 Gemini 相同的底層技術與研究基礎，但被特別設計成可以**在你自己的個人電腦、筆電甚至樹莓派上順暢運行**。

### 核心優勢與定位：
* **系出名門的理解力**：繼承了 Google 訓練 Gemini 的優良基因，在同等級大小的模型中，邏輯推理與程式碼生成能力名列前茅。
* **專為本地端打造的大小**：
  * `gemma2` (預設為 9B，即 90 億參數)：目前本地端的高效能首選，適合用來做複雜的 RAG 推理與統整，建議電腦有 **8GB 以上**的記憶體 (RAM 或 VRAM)。
  * `gemma:2b` (20 億參數)：極度輕量的版本，即使是文書筆電也能飛速運行，適合簡單的問答或硬體受限的環境。
* **RAG 系統的完美大腦**：在我們的架構中，如果說 `nomic-embed-text` 是負責幫你翻書找資料的「圖書管理員」，那 Gemma 就是負責閱讀這些資料並向你解釋的「專屬家教」。

---

## 2. 如何下載與安裝？ (Downloading via Ollama)

因為你已經安裝了 Ollama，所以要在電腦裡「安裝」Gemma，只需要一句指令來下載它的權重檔 (Weights)。

請打開終端機 (Terminal)，輸入以下指令：

```bash
# 下載主流效能版 Gemma 2 (約需要 5.5 GB 硬碟空間)
ollama pull gemma2

# (備用選項) 如果你的電腦跑 9B 覺得太卡，可以下載輕量版
# ollama pull gemma:2b
```

下載完成後，你可以直接在終端機輸入 `ollama run gemma2` 來進行純文字介面的快速盲測。

---

## 3. 如何在 Python 中使用？ (Python Integration)

在實際開發 Micro SaaS 或 RAG 系統時，我們會透過 Python 來呼叫它。這裡提供兩種最經典的使用情境：

### 模式 A：單次完整生成 (Standard Generation)
這是最標準的用法，適合用在 RAG 系統中，等待模型看完你的筆記後給出一個完整的總結。

```python
import ollama

print("🧠 Gemma 正在閱讀與思考...")
response = ollama.chat(
    model='gemma2', 
    messages=[
        {'role': 'system', 'content': '你是一個精通繁體中文的 Python 助教。'},
        {'role': 'user', 'content': '請簡短解釋 Pandas DataFrame 是什麼？'}
    ]
)

print("\n✅ 回答：")
print(response['message']['content'])
```

### 模式 B：打字機流式輸出 (Streaming Generation)
因為本地端模型沒有網路延遲，運算出來的每一個字可以直接印在螢幕上。這在 Streamlit 前端能提供極佳的使用者體驗，不用盯著轉圈圈的 Loading 畫面！

```python
import ollama
import sys

print("🧠 Gemma (流式輸出模式): \n")

# 設定 stream=True
stream_response = ollama.chat(
    model='gemma2',
    messages=[{'role': 'user', 'content': '請列出三個學習機器學習的好處。'}],
    stream=True
)

# 像打字機一樣，將接收到的每一個小片段立刻印出來
for chunk in stream_response:
    sys.stdout.write(chunk['message']['content'])
    sys.stdout.flush()
    
print("\n\n✅ 生成完畢！")
```