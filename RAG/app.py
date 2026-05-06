"""
RAG Streamlit App (Hybrid Backend)

此模組實作了一個結合「本地端 Ollama」與「雲端 Google Gemini」的檢索增強生成 (RAG) 系統。

核心功能：
- 雙模組支援：可透過 LOCAL_OLLAMA 開關切換本地運算（隱私免費用量）或雲端 API。
- 自動化知識庫：自動讀取本地 Markdown 筆記並進行切塊與向量化 (Embedding)。
- 智慧檢索：利用餘弦相似度 (Cosine Similarity) 從筆記中尋找最相關的知識片段。
- 嚴謹生成：AI 必須嚴格根據提供的筆記內容進行回答，防止幻覺 (Hallucination)。

使用說明：
1. 本地模式：需安裝 Ollama 並下載 nomic-embed-text 與 gemma4 模型。
2. 雲端模式：需在側邊欄輸入有效的 Google Gemini API Key。
"""
import streamlit as st
import google.generativeai as genai
import ollama
import numpy as np
import os
import time
from typing import Literal, List
from chunk import load_local_markdown_notes, build_knowledge_base

NOTE_DIR = "../"  # 預設筆記資料夾路徑
UPDATE_KB_EMBED = True  # 是否每次啟動都重新生成知識庫向量（開發階段建議開啟，正式使用可關閉以加速啟動）
LOCAL_OLLAMA_EMBED = True  # 是否使用本地 Ollama 進行向量化（如果你已經安裝並運行了 Ollama，建議開啟以完全免費且無速限的向量化！）
LOCAL_OLLAMA_GEN = True  # 是否使用本地 Ollama 進行生成（如果你已經安裝並運行了 Ollama，建議開啟以完全免費且無速限的生成！）
LOCAL_OLLAMA_GEN_MODEL = "gemma2:9b"  # 本地 Ollama 使用的生成模型名稱（確保已透過 `ollama pull gemma2` 下載）

# ==========================================
# 1. 系統設定與 UI 初始化
# ==========================================
st.set_page_config(page_title="專屬 ML 筆記大腦", page_icon="🧠", layout="centered")
st.title("🤖 專屬 ML 讀書筆記大腦 (Phase 1)")

# 在側邊欄顯示狀態與設定
with st.sidebar:
    st.header("系統狀態")
    
    # 顯示目前使用的模式
    if LOCAL_OLLAMA_EMBED and LOCAL_OLLAMA_GEN:
        st.success("🟢 目前運行於：本地 Ollama 模式")
        st.info("向量模型: `nomic-embed-text`  \n生成模型: `gemma2:9b`")
        api_key = st.text_input("Gemini API Key (選用，雲端備援用):", type="password")
    else:
        st.warning("🔵 目前運行於：雲端 Gemini 模式")
        api_key = st.text_input("請輸入 Gemini API Key 以啟動:", type="password")
        
        if not api_key:
            st.warning("請輸入 API Key 以繼續。")
            st.stop()
    
    st.markdown("---")
    st.caption("這是一個微型 RAG 測試系統。預設使用本地算力，保護您的隱私。")

# 只有在有輸入 Key 的情況下才初始化 Google AI，否則跳過
if api_key:
    genai.configure(api_key=api_key)
elif not LOCAL_OLLAMA_EMBED and not LOCAL_OLLAMA_GEN:
    # 雙重防呆：如果不是本地模式又沒給 Key，就停下來
    st.error("雲端模式需要 API Key 才能運作。")
    st.stop()

# ==========================================
# 2. 自動化知識庫建立 (Data Ingestion & Chunking)
# ==========================================

# 2.1 切塊器與讀取器來自 chunk.py（支援 *_*.md 根目錄篩選）
# chunk_text, load_local_markdown_notes 已移至 RAG/chunk.py

# ========== [核心修正區：動態獲獲取模型與防呆] ==========
@st.cache_data(show_spinner=False)
def get_embedding_model_name() -> str:
    """
    動態獲取當前帳戶可用的 Embedding 模型名稱。
    優先選擇較新的模型，若無則回退至預設值。
    """
    try:
        # 遍歷所有模型，尋找支援 embedContent 的模型
        for m in genai.list_models():
            if 'embedContent' in m.supported_generation_methods:
                # 確保名稱不包含 'models/' 前綴（genai.embed_content 偏好純模型名）
                return m.name.replace("models/", "")
    except Exception as e:
        print(f"無法列出模型清單: {e}")
        
    # 若失敗，回退至最穩定的模型
    return "text-embedding-004"

def safe_embed_content(text: str, task_type_str: str) -> List[float]:
    """
    包裝 Embedding 請求，根據模型名稱自動調整參數。
    """
    model_name = get_embedding_model_name()
    
    # 根據模型世代決定是否需要內建 task_type 參數
    # 新一代模型 (如 v2) 參數結構可能不同
    if "gemini-embedding-2" in model_name:
        result = genai.embed_content(model=model_name, content=text)
    else:
        # 傳統模型需要明確指定任務類型 (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT)
        result = genai.embed_content(
            model=model_name, 
            content=text, 
            task_type=task_type_str
        )
    return result['embedding']

@st.cache_data(show_spinner=False)
def create_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    將文字區塊轉換為數值向量表示。

    根據 LOCAL_OLLAMA 標記，自動選擇使用本地端 Ollama 或雲端 Google Gemini 進行向量化。
    整合了 Streamlit 進度條與快取機制。

    Args:
        chunks: 需要轉換的字串列表。

    Returns:
        包含所有潛入向量 (Embeddings) 的列表。
    """
    total: int = len(chunks)
    embeddings: List[List[float]] = []
    
    # UI 提示
    mode_text: str = "本地 Ollama" if LOCAL_OLLAMA_EMBED else "Google Gemini"
    progress_bar = st.progress(0, text=f"🧠 正在使用 {mode_text} 建立大腦索引...")

    for i, chunk in enumerate(chunks):
        if LOCAL_OLLAMA_EMBED:
            # 本地端呼叫
            response = ollama.embeddings(model='nomic-embed-text', prompt=chunk)
            embeddings.append(response['embedding'])
        else:
            # 雲端端呼叫 (需確保已定義 safe_embed_content)
            embeddings.append(safe_embed_content(chunk, "RETRIEVAL_DOCUMENT"))
            # 雲端版限制頻率
            if (i + 1) % 10 == 0:
                time.sleep(0.5)
        
        # 更新進度條
        progress_bar.progress((i + 1) / total)
    
    progress_bar.empty()
    return embeddings

# ==========================================
# 2.3 啟動自動讀取與轉換
# ==========================================
# 如果「保險箱是空的」或者是「開發者強制要求更新」，就執行 Embedding
if "kb_embeddings" not in st.session_state or UPDATE_KB_EMBED:
    with st.spinner(f'正在從 {NOTE_DIR} 讀取筆記並轉化為大腦...'):
        loaded_chunks = load_local_markdown_notes(NOTE_DIR)
        
        if loaded_chunks:
            # 確保在此處才正確呼叫函數並存入 session_state
            st.session_state.kb_texts = loaded_chunks # 先存原始文本，後面檢索用
            st.session_state.kb_embeddings = create_embeddings(loaded_chunks) # 這裡會呼叫 create_embeddings，並且根據 LOCAL_OLLAMA_EMBED 決定使用哪個模型
            build_knowledge_base(loaded_chunks)  # 將已載入的切塊結果持久化至 knowledge_base.md，避免重複讀取磁碟
            st.success(f"知識庫載入完成！共讀取了 {len(loaded_chunks)} 個知識區塊。")
        else:
            st.warning("未找到任何有效筆記，請確認路徑或檔案名稱。")
            st.stop()

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 檢查本地模型是否存在 (僅在本地模式) ---
if LOCAL_OLLAMA_EMBED or LOCAL_OLLAMA_GEN:
    try:
        models_info = ollama.list()
        available_models = [m['name'] for m in models_info.get('models', [])]
        
        required_models = ['nomic-embed-text:latest', 'gemma4:latest'] + [LOCAL_OLLAMA_GEN_MODEL]  # 確保生成模型也在檢查範圍內
        for rm in required_models:
            # Ollama 的名稱匹配有時包含或不包含 :latest
            if not any(rm.split(':')[0] in m for m in available_models):
                st.warning(f"⚠️ 找不到本地模型 `{rm}`，請確保已執行 `ollama pull {rm}`")
    except Exception:
        # 如果無法連線，由後續的錯誤處理流程捕捉
        pass

# ==========================================
# 3. 定義核心 RAG 檢索邏輯
# ==========================================
def retrieve_relevant_chunks(query, texts, embeddings, top_k=2):
    """
    檢索最相關的知識區塊。
    生產環境優化：先獲取較多候選者 (Candidate Generation)，再進行相似度過濾與排序 (Reranking/Filtering)。
    """
    # 根據開關選擇如何將「問題」轉為向量
    if LOCAL_OLLAMA_EMBED:
        response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_vec = response['embedding']
    else:
        query_vec = safe_embed_content(query, "RETRIEVAL_QUERY")

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # 1. 計算所有相似度
    similarities = np.array([cosine_similarity(query_vec, kb_vec) for kb_vec in embeddings])
    
    # 2. 獲取較多的候選者 (例如 top_k * 3) 以進行精確排序
    candidate_size = min(top_k * 3, len(texts))
    ranked_indices = np.argsort(similarities)[::-1][:candidate_size]
    
    # 3. 執行嚴格過濾與精簡
    # 門檻值 0.35 定義了「相關」的底線
    final_indices = [idx for idx in ranked_indices if similarities[idx] >= 0.35]
    
    if not final_indices:
        return None, []

    # 4. 只取最終要求的前 top_k 個最優解
    top_indices = final_indices[:top_k]
    top_chunks = [texts[i] for i in top_indices]
    top_scores = [float(similarities[i]) for i in top_indices]

    return top_chunks, top_scores

# ==========================================
# 4. Streamlit 對話介面與生成
# ==========================================
# 顯示歷史訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收使用者輸入
if prompt := st.chat_input("你想問關於 Scikit-Learn、Pandas 或 NumPy 的什麼問題？"):
    # 顯示使用者訊息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('大腦檢索中...'):
        # 步驟 A：檢索 (Retrieval)
        retrieved_contexts, scores = retrieve_relevant_chunks(
            prompt,
            st.session_state.kb_texts,
            st.session_state.kb_embeddings,
            top_k=2
        )
        
        # 判斷是否檢索到相關內容
        if retrieved_contexts is None:
            answer_text = "抱歉，根據目前的筆記，我找不到與此問題相關的資訊。"
            with st.chat_message("assistant"):
                st.markdown(answer_text)
                st.warning("檢索分數過低，已停止生成以防止幻覺。")
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
            st.stop()

        retrieved_context = "\n\n---\n\n".join(
            [f"[檢索片段 {idx + 1} | 相似度 {scores[idx]:.4f}]\n{chunk}"
             for idx, chunk in enumerate(retrieved_contexts)]
        )
        top_score = scores[0]
        
        # 步驟 B：組裝 Prompt
        system_prompt = f"""
        你是一位嚴謹的機器學習技術導師。
        請「嚴格」根據以下 [參考筆記] 的內容，來回答使用者的 [問題]。
        如果參考筆記中的資訊不足以回答問題，請明確告知「根據目前的筆記，我無法回答這個問題」，不要自行編造。
        請使用繁體中文，並保持專業的語氣。

        [參考筆記]
        {retrieved_context}

        [問題]
        {prompt}
        """

        # 步驟 C：生成 (Generation)
        if LOCAL_OLLAMA_GEN:
            # 100% 本地端 Gemma 4 大腦
            try:
                print(f"🤖 [系統日誌] 正在喚醒本地 {LOCAL_OLLAMA_GEN_MODEL} 進行思考...")
                
                # 呼叫本地端的 Ollama 聊天 API
                response = ollama.chat(
                    model=LOCAL_OLLAMA_GEN_MODEL, 
                    messages=[
                        {
                            'role': 'user',
                            'content': system_prompt # 把包含筆記和問題的長字串交給它
                        }
                    ]
                )
                
                # 提取 Gemma 的回答文字
                answer_text = response['message']['content']
                
                # 在終端機印出檢索結果，方便 Debug
                print(f"User Query: {prompt}")
                print(f"Retrieved Score: {top_score:.4f}")
                print(f"Retrieved Chunk: {retrieved_context}\n---")

                # 顯示 AI 回應，並偷偷附上是參考了哪一段筆記
                with st.chat_message("assistant"):
                    st.markdown(answer_text)
                    with st.expander("🔍 點擊查看背後檢索到的筆記片段 (Debug 區)"):
                        st.info(f"**相似度分數:** {top_score:.4f}\n\n**原始筆記:** {retrieved_context}")
                        
                st.session_state.messages.append({"role": "assistant", "content": answer_text})

            except Exception as e:
                # 檢查是否為根本沒安裝或沒啟動 Ollama
                error_msg: str = str(e)
                if "connection" in error_msg.lower():
                    st.error("❌ 無法連線至 Ollama。請確保 Ollama 應用程式已啟動 (localhost:11434)。")
                else:
                    st.error(f"本地生成過程中發生錯誤，請確認 Ollama 與 gemma4 是否正常運作: {e}")
        else:
            # 100% 雲端 Google Gemini 大腦 (備用方案)
            try:
                # 動態尋找可用的 Flash 模型
                target_model_name = "gemini-1.5-flash-latest" # 備用預設值
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods and 'flash' in m.name.lower():
                        # 取出模型名稱 (GenerativeModel 比較挑剔，通常不需要 'models/' 前綴)
                        target_model_name = m.name.replace("models/", "")
                        break # 找到第一個支援的 flash 模型就採用
                
                print(f"🤖 [系統日誌] 正在使用生成模型: {target_model_name}")
                
                # 建立模型並生成回答
                model = genai.GenerativeModel(target_model_name)
                response = model.generate_content(system_prompt)
                
                # 在終端機印出檢索結果，方便你 Debug 觀察相似度
                print(f"User Query: {prompt}")
                print(f"Retrieved Score: {top_score:.4f}")
                print(f"Retrieved Chunk: {retrieved_context}\n---")

                # 顯示 AI 回應，並偷偷附上是參考了哪一段筆記
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                    with st.expander("🔍 點擊查看背後檢索到的筆記片段 (Debug 區)"):
                        st.info(f"**相似度分數:** {top_score:.4f}\n\n**原始筆記:** {retrieved_context}")
                        
                st.session_state.messages.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"生成過程中發生錯誤: {e}")