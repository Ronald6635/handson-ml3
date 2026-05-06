## Streamlit 是什麼？

Streamlit 是一個用 Python 快速建立互動式網頁應用程式的框架，特別適合資料分析、機器學習、模型展示和原型開發。

### 主要特點

- 只要寫 Python，不用寫 HTML/CSS/JavaScript
- 直接把程式碼變成網頁介面
- 內建常用 UI 元件：按鈕、輸入框、滑桿、下拉選單、圖表、表格、Markdown
- 對於資料科學和 ML 工程師非常友善
- 每次修改程式碼後，網頁會自動重新整理

### 適合用在什麼場合

- 展示機器學習模型預測結果
- 做資料探索報告
- 建立內部工具或 demo
- 做快速原型驗證

### 你現在這個 app.py 用到 Streamlit 做什麼

- `st.set_page_config(...)`：設定頁面標題、圖示、版面
- `st.title(...)`：顯示主標題
- `st.sidebar`：建立側邊欄輸入 API key
- `st.chat_input(...)`、`st.chat_message(...)`：做對話式輸入與訊息顯示
- `st.spinner(...)`：顯示等待動畫
- `st.expander(...)`：可展開的 debug 區塊

### 總結

Streamlit 的核心價值是：用最少的 Python 程式碼，把資料或模型變成互動式網頁應用。對於你這種想做 RAG demo、筆記小助手、機器學習介面的人特別方便。