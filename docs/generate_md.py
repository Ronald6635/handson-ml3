import json
import re

# Load the JSON data
with open('chapters.json', 'r', encoding='utf-8') as f:
    chapters = json.load(f)

markdown_content = "# 機器學習與深度學習實戰速查手冊\n\n"
markdown_content += "基於《Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition》\n\n"

for chapter_title, rows in chapters.items():
    markdown_content += f"## {chapter_title}\n\n"
    markdown_content += "| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |\n"
    markdown_content += "|:---|:---|:---|:---|:---|\n"
    for row in rows:
        concept = row['關鍵概念']
        functions = row['重要函數']
        description = row['功能簡介與應用場景']
        params = row['Fine-tune 用參數']
        code = row['範例代碼片段']
        
        # Format concept: add ** if not already
        if concept and not concept.startswith('**'):
            concept = f"**{concept}**"
        
        # Format functions: add ` if not
        if functions and not functions.startswith('`'):
            functions = f"`{functions}`"
        
        # Description: keep as is
        
        # Params: format as **<code>param</code>** (desc)
        # But in the data, it's already formatted
        
        # Code: clean and format
        code = code.replace('<br>', '\n').replace('Copy', '').strip()
        if code and code != 'N/A':
            code = f"`{code}`"
        
        markdown_content += f"| {concept} | {functions} | {description} | {params} | {code} |\n"
    markdown_content += "\n"

with open('new_md.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)