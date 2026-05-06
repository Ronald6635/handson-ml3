# 🦙 Ollama 基礎入門指南

## 1. 這是什麼？ (What is Ollama?)

**Ollama** 是一個開源的本地端大型語言模型 (LLM) 運行工具。你可以把它想像成是 **「架設在你個人電腦上的專屬 AI 伺服器」**。

在過去，要跑起一個強大的 AI 模型，你需要複雜的環境設定、編譯程式碼，甚至要懂很多底層架構。而 Ollama 將這一切極度簡化，讓你像安裝普通軟體一樣，輕鬆把 AI 裝進電腦裡。

### 核心優勢：

*   **絕對的隱私安全**：所有的對話、文本與運算都在你的本機端（CPU/GPU）進行，資料「零」外流。
*   **完全免費且無速限**：沒有 API 的計費機制，也沒有每分鐘 100 次的請求限制 (Rate Limit)，只要你的電腦沒關機，愛算多少就算多少。
*   **豐富的開源生態**：支援一鍵下載 Google 的 `Gemma`、Meta 的 `Llama`、微軟的 `Phi` 等頂級開源模型，以及專做文本向量化的 `nomic-embed-text`。
    - 文本向量化 (Embedding) 是 RAG 系統的核心技術，Ollama 讓你直接在本地端就能使用專業級的向量模型，無需再依賴外部 API。
*   **簡單的 Python 整合**：透過官方提供的 `ollama` Python 套件，你可以在自己的程式碼裡輕鬆呼叫本地端的 AI 模型，無論是生成文本還是做向量轉換，都能一行程式碼搞定。
    - 透過 Ollama，你可以在本地端同時運行多個模型，並且在 Python 程式碼裡自由切換使用。

---

## 2. 如何安裝？ (Installation)

要讓 Ollama 完美結合你的 Python 開發環境，安裝分為兩個層次：**「核心伺服器」**與 **「Python 溝通橋樑」**。

### 步驟 2.1：安裝 Ollama 核心應用程式 (Server)

這是驅動 AI 運算的底層引擎，必須安裝在你電腦的作業系統上。
1. 前往 [Ollama 官方網站 (ollama.com)](https://ollama.com/)。
2. 點擊 **Download**，選擇適合你作業系統 (Windows / macOS / Linux) 的安裝檔。
3. 下載後執行安裝程式，一路點擊下一步直到完成。安裝後，Ollama 會以一個常駐背景程式的形式在你的電腦上運行（預設佔用 `localhost:11434` Port）。

### 步驟 2.2：安裝 Python 套件 (Client)

為了讓你的 Python 專案（如 FastAPI 或 Streamlit）能跟 Ollama 伺服器對話，需要安裝專用的通訊套件。
打開終端機，啟動你的 Conda 虛擬環境後執行：
```bash
# 1. 確保進入正確的虛擬環境
conda activate <你的環境名稱>

# 2. 安裝 Python 套件
pip install ollama
```

---

## 3. 如何啟用與測試？ (Getting Started)

安裝完成後，開啟全新的終端機 (Terminal / 命令提示字元) 來進行第一次的喚醒測試。

### 動作一：下載並運行第一個生成模型 (以 Gemma 2 為例)

在終端機輸入以下指令。Ollama 會自動從開源模型庫將 Google 的 Gemma 4 模型權重下載到你的硬碟中，並直接啟動一個終端機聊天室：
```bash
ollama run gemma4
```
*(註：第一次下載通常需要幾分鐘，檔案大小約數 GB。下載完畢後，你就可以直接在終端機裡用純文字跟它聊天。輸入 `/bye` 即可退出聊天室。)*

### 動作二：下載專屬向量模型 (Embedding)

如果你要做 RAG (檢索增強生成) 系統，你需要一個專門把文字轉成數字的數學模型。在終端機輸入：
```bash
ollama pull nomic-embed-text
```

### 動作三：透過 Python 呼叫 (整合進你的程式碼)

現在，你的本機 AI 伺服器已經啟動，且擁有大腦了。你可以寫一小段 Python 程式碼來測試串接：

```python
import ollama

# 測試 1：生成對話 (Generation)
print("🤖 Gemma 思考中...")
response = ollama.chat(
    model='gemma2', 
    messages=[{'role': 'user', 'content': '請用一句話解釋什麼是機器學習？'}]
)
print("回答：", response['message']['content'])
print("-" * 30)

# 測試 2：文字轉向量 (Embedding)
print("🔢 正在將文字轉換為 768 維向量...")
embed_response = ollama.embeddings(
    model='nomic-embed-text', 
    prompt="隨機森林是一種基於決策樹的演算法。"
)
print(f"轉換成功！向量長度為: {len(embed_response['embedding'])} 維")
```