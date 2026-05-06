"""
Chunk Module

此模組提供 RAG 系統的文字切塊與 Markdown 筆記讀取功能。

Key features:
- 從 Markdown 內容進行語義化切塊，優先以標題與段落分割
- GIGO 防護：過濾 HTML 元數據、TOC 區塊、錨點標籤、短碎片
- 保持程式碼區塊（``` fence）與 Markdown 表格完整，支援切塊重疊（overlap）
- 移除標題行前導 Emoji，確保向量化品質
- 依檔案類型（tutorial/handout/cheatsheet）動態調整 max_chars
- 自動保留並前置 HTML 元數據（標題、描述、關鍵字）至首個 chunk
- 只讀取根目錄下符合 *_*.md 命名規則的 Markdown 檔案
- 提供 build_knowledge_base() 函數，將切塊結果輸出至 knowledge_base.md
"""
import os
import glob
import re
from typing import List

# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum characters a chunk must have to be kept; filters out header-only stubs
_MIN_CHUNK_LEN: int = 30

# Compiled regex for stripping leading emoji from Markdown heading lines.
# Covers the most common emoji Unicode blocks; leaves CJK and ASCII untouched.
_HEADING_EMOJI_RE: re.Pattern = re.compile(
    r'^(#{1,6}\s+)[\U0001F300-\U0001FAFF\U00002600-\U000027BF\uFE00-\uFE0F\u2600-\u27BF ]+',
    re.MULTILINE,
)

# =============================================================================
# HELPERS
# =============================================================================

def _extract_metadata_prefix(raw: str) -> tuple:
    """
    從原始 Markdown 文字開頭提取 HTML 元數據欄位，回傳 (prefix_str, cleaned_text)。

    解析的欄位：meta-title（標題）、meta-description（描述）、meta-keywords（關鍵字）。
    cleaned_text 已移除全部 HTML 元數據註解（<!-- ... -->）。
    若無任何元數據，prefix_str 為空字串。

    Args:
        raw (str): 原始 Markdown 文字

    Returns:
        tuple: (prefix_str, cleaned_text)
    """
    field_map = [
        ('meta-title', '標題'),
        ('meta-description', '描述'),
        ('meta-keywords', '關鍵字'),
    ]
    parts = []
    for key, label in field_map:
        m = re.search(rf'<!--\s*{re.escape(key)}:\s*(.*?)\s*-->', raw, re.DOTALL)
        if m:
            parts.append(f"{label}: {m.group(1).strip()}")

    cleaned = re.sub(r'<!--.*?-->', '', raw, flags=re.DOTALL).strip()
    prefix = f"[{' | '.join(parts)}]" if parts else ''
    return prefix, cleaned


def _build_source_label(basename: str) -> str:
    """
    從 Markdown 檔名建立簡潔的來源標籤。

    範例：
    - ``02_end_to_end_machine_learning_project_tutorial.md`` → ``[來源: ch02 | 類型: tutorial]``
    - ``tools_numpy.md`` → ``[來源: tools_numpy]``

    Args:
        basename (str): 檔案名稱（不含路徑）

    Returns:
        str: 格式化的來源標籤字串
    """
    name = os.path.splitext(basename)[0]
    ch_match = re.match(r'^(\d+)_', name)
    ch_label = f"ch{ch_match.group(1)}" if ch_match else name
    for file_type in ('cheatsheet', 'tutorial', 'handout'):
        if name.endswith(f'_{file_type}'):
            return f"[來源: {ch_label} | 類型: {file_type}]"
    return f"[來源: {ch_label}]"


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_text(text: str, max_chars: int = 350, overlap_chars: int = 100) -> List[str]:
    """
    將 Markdown 內容依據標題與段落切塊，並過濾低品質內容以防止 GIGO。

    GIGO 防護措施：
    1. 清理標題中的 HTML 錨點標籤（<a id="..."></a>），減少檢索到的文字雜訊
    2. 移除 Markdown 標題行中的前導 Emoji，避免符號影響向量化品質
    3. 跳過目錄（TOC）區塊（>50% 非空行為 `- [text](#anchor)` 格式），避免純導覽連結污染
    4. 保持程式碼區塊（``` fence）與 Markdown 表格完整，不在中間切割，避免產生殘缺 chunk
    5. 過濾過短的 chunk（< _MIN_CHUNK_LEN 字元），避免獨立標題等低信號內容
    6. 支援 Overlap (重疊)：在切塊邊界保留部分重複文字，防止語義斷層。

    切塊優先順序：
    1. 優先以 Markdown 標題分割
    2. 若單一章節過長，依段落切分，並保持程式碼區塊完整
    3. 若段落仍過長，依字元截斷，並加上 overlap

    Args:
        text (str): 原始 Markdown 文字內容
        max_chars (int): 每個 chunk 的最大字元數（預設 350）
        overlap_chars (int): 相鄰 chunk 之間的重疊字元數（預設 100）

    Returns:
        List[str]: 切塊後的文字清單，已過濾低品質 chunk
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    # 1. 清理標題中的 HTML 錨點標籤（例如 ## <a id="setup"></a>設定）
    text = re.sub(r'<a\s[^>]*></a>', '', text)

    # 2. 移除 Markdown 標題行中的前導 Emoji（保留純文字標題供向量化）
    text = _HEADING_EMOJI_RE.sub(r'\1', text)

    sections = re.split(r'(?m)(?=^#{1,6}\s+)', text)
    chunks: List[str] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # 3. 跳過目錄（TOC）區塊：超過 50% 的非空行為 Markdown 清單連結
        non_empty = [l for l in section.splitlines() if l.strip()]
        toc_lines = sum(1 for l in non_empty if re.match(r'\s*[-*]\s*\[.+?\]\(#.+?\)', l))
        if non_empty and toc_lines / len(non_empty) > 0.5:
            continue

        # 5. 過濾過短的 chunk（獨立標題或空節）
        if len(section) <= max_chars:
            if len(section) >= _MIN_CHUNK_LEN:
                chunks.append(section)
            continue

        # 4. 將段落切分，但保持程式碼區塊（``` ... ```）與 Markdown 表格完整不被截斷
        parts = re.split(r'(```[\s\S]*?```|(?:[ \t]*\|[^\n]+\n?){2,})', section)
        current = ""

        def add_chunk_with_overlap(chunk_content: str) -> str:
            """輔助函數：添加 chunk 並為下一個 chunk 預留 overlap"""
            if len(chunk_content) >= _MIN_CHUNK_LEN:
                chunks.append(chunk_content)
            # 返回 overlap 部分作為下個 chunk 的開頭
            return chunk_content[-overlap_chars:] if len(chunk_content) > overlap_chars else ""

        for part in parts:
            # 程式碼區塊或 Markdown 表格均視為不可分割的原子單元
            is_atomic = part.startswith('```') or bool(re.match(r'[ \t]*\|', part))

            if is_atomic:
                if not current:
                    current = part
                elif len(current) + len(part) + 2 <= max_chars:
                    current += "\n\n" + part
                else:
                    current = add_chunk_with_overlap(current)
                    # 如果原子區塊本身就超過 max_chars，強制放入但不再細分 (保持完整性)
                    if not current:
                        current = part
                    else:
                        current += "\n\n" + part
            else:
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n', part) if p.strip()]
                for para in paragraphs:
                    if not current:
                        current = para
                        continue

                    if len(current) + len(para) + 2 <= max_chars:
                        current += "\n\n" + para
                    else:
                        current = add_chunk_with_overlap(current)
                        if len(para) <= max_chars:
                            current = (current + "\n\n" + para).strip() if current else para
                        else:
                            # 極端情況：單一無空格段落長於 max_chars
                            temp_para = (current + "\n\n" + para).strip() if current else para
                            while len(temp_para) > max_chars:
                                split_point = max_chars
                                chunks.append(temp_para[:split_point])
                                temp_para = temp_para[split_point - overlap_chars:]
                            current = temp_para

        if current and len(current) >= _MIN_CHUNK_LEN:
            chunks.append(current)

    return chunks

# =============================================================================
# PREPROCESSING
# =============================================================================

def build_knowledge_base(
    chunks: List[str],
    output_path: str = "knowledge_base.md",
) -> int:
    """
    將已切塊的 chunk 清單寫出至 knowledge_base.md。

    接收 load_local_markdown_notes() 回傳的 chunk 清單，
    以 '---' 分隔線串接後寫出至 output_path，避免重複讀取磁碟。
    每個 chunk 已包含來源標籤（[來源: chXX | 類型: ...]）。

    Args:
        chunks (List[str]): 已切塊的文字清單（由 load_local_markdown_notes() 產生）
        output_path (str): 輸出檔案路徑（預設 "knowledge_base.md"，相對於 CWD）

    Returns:
        int: 寫出的 chunk 數量；若 chunks 為空則回傳 0

    Example:
        >>> chunks = load_local_markdown_notes("../")
        >>> n = build_knowledge_base(chunks, output_path="knowledge_base.md")
        >>> print(f"寫出了 {n} 個 chunk")
    """
    if not chunks:
        return 0

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("\n\n---\n\n".join(chunks))

    return len(chunks)

# =============================================================================
# LOADING
# =============================================================================

def load_local_markdown_notes(directory_path: str = "../") -> List[str]:
    """
    讀取指定根目錄下符合 *_*.md 命名規則的 Markdown 檔案，並切塊後回傳。

    只處理根目錄（非遞迴），且檔名必須含有底線（_），例如：
    02_end_to_end_machine_learning_project.md

    Args:
        directory_path (str): 根目錄路徑（預設 "../"）

    Returns:
        List[str]: 所有符合條件的 chunk，格式為 "[來源: chXX | 類型: TYPE] chunk 內容"。
                   各檔案的首個 chunk 會前置元數據標籤（若檔案有 HTML 元數據）。
                   若目錄不存在或無符合檔案，回傳空清單。
    """
    all_chunks: List[str] = []

    # 檢查資料夾是否存在
    if not os.path.exists(directory_path):
        return []

    # 只抓根目錄（非遞迴）的 *.md 檔案，再過濾檔名含底線者
    candidate_files = glob.glob(os.path.join(directory_path, "*.md"))
    md_files = sorted(
        f for f in candidate_files
        if "_" in os.path.basename(f)
    )

    if not md_files:
        return []

    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

            basename = os.path.basename(filepath)

            # 依檔案類型調整 max_chars（cheatsheet 表格密集需較大空間；tutorial 內容詳盡；handout 適中）
            if '_cheatsheet' in basename:
                max_chars = 500
            elif '_tutorial' in basename:
                max_chars = 600
            else:  # _handout 及其他
                max_chars = 400

            # 提取元數據前置標籤，並傳入已清除 HTML 元數據註解的內容給 chunk_text
            meta_prefix, cleaned_content = _extract_metadata_prefix(content)
            file_chunks = chunk_text(cleaned_content, max_chars=max_chars)

            # 將元數據前置標籤加到第一個 chunk，保留關鍵上下文
            if file_chunks and meta_prefix:
                file_chunks[0] = meta_prefix + "\n" + file_chunks[0]

            source_label = _build_source_label(basename)
            for chunk in file_chunks:
                all_chunks.append(f"{source_label} {chunk}")

        except Exception as e:  # noqa: BLE001 — log and continue on per-file errors
            print(f"[chunk.py] 讀取檔案 {filepath} 時發生錯誤: {e}")

    return all_chunks
