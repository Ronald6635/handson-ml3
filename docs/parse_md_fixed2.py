import re
import html

def parse_markdown_tables(md_file_path):
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all tables in the markdown
    table_pattern = r'\|.*\|\n\|[-\s|:]+\|\n(?:\|.*\|\n)+'
    tables = re.findall(table_pattern, content)

    chapter_tables = {}

    for table in tables:
        lines = table.strip().split('\n')
        # Skip header separator
        data_lines = lines[2:]

        for line in data_lines:
            if line.strip():
                # Split by | and strip whitespace
                cells = [cell.strip() for cell in line.split('|')[1:-1]]

                if len(cells) >= 7:  # Ensure we have enough columns
                    chapter_ref = cells[6]  # Last column is chapter reference
                    # Extract chapter number
                    chapter_match = re.search(r'Chapter (\d+)', chapter_ref)
                    if chapter_match:
                        chapter = f"Chapter {chapter_match.group(1)}"
                    else:
                        chapter = chapter_ref

                    if chapter not in chapter_tables:
                        chapter_tables[chapter] = []

                    # Map to 5 columns: 關鍵概念 (cells[0]), 重要函數 (cells[1]), 功能簡介與應用場景 (cells[3]), Fine-tune 用參數 (cells[4]), 範例代碼片段 (cells[5])
                    # Convert markdown to HTML
                    key_concept = convert_md_to_html(cells[0])
                    functions = convert_md_to_html(cells[1])
                    description = convert_md_to_html(cells[3])
                    params = convert_md_to_html(cells[4])
                    code = cells[5].replace('\n', '<br>')  # Replace newlines with <br> in code

                    row = f'<tr><td>{key_concept}</td><td>{functions}</td><td>{description}</td><td>{params}</td><td class="code-snippet"><code>{html.escape(code)}</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td></tr>'
                    chapter_tables[chapter].append(row)

    return chapter_tables

def convert_md_to_html(text):
    # Convert **bold** to <strong>bold</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Convert `code` to <code>code</code>
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return text

if __name__ == '__main__':
    md_file = '機器學習與深度學習實戰速查手冊.md'
    chapter_tables = parse_markdown_tables(md_file)

    for chapter, rows in chapter_tables.items():
        print(f"{chapter}:")
        for row in rows:
            print(row)
        print()