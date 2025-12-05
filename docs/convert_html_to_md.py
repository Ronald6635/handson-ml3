import re
from bs4 import BeautifulSoup

# Read the HTML file
with open('機器學習與深度學習實戰速查手冊.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Parse HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Find all sections with id starting with 'chapter-'
chapters = soup.find_all('section', id=re.compile(r'^chapter-\d+$'))

markdown_content = '# 《Hands-on ML, 3rd Edition》程式範例速查表格 (依章節分類)\n\n'

# Generate table of contents
markdown_content += '## 目錄 (Table of Contents)\n\n'
for chapter in chapters:
    title_elem = chapter.find('h2')
    if title_elem:
        title = title_elem.get_text()
        chapter_num = re.search(r'第(\d+)章', title).group(1)
        chapter_title = title.split('：')[1] if '：' in title else title
        markdown_content += f'- [第{chapter_num}章：{chapter_title}](#chapter-{chapter_num})\n'
markdown_content += '\n'

for chapter in chapters:
    # Get chapter title
    title_elem = chapter.find('h2')
    if title_elem:
        title = title_elem.get_text()
        chapter_num = re.search(r'第(\d+)章', title).group(1)
        markdown_content += f'<a id="chapter-{chapter_num}"></a>\n## {title}\n\n'
    
    # Find the table
    table = chapter.find('table')
    if table:
        # Get headers
        headers = [th.get_text() for th in table.find_all('th')]
        markdown_content += '| ' + ' | '.join(headers) + ' |\n'
        markdown_content += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
        
        # Get rows
        rows = table.find_all('tr')[1:]  # Skip header row
        for row in rows:
            cells = row.find_all('td')
            row_data = []
            for cell in cells:
                cell_text = cell.get_text().strip()
                # Remove 'Copy' from end if present
                cell_text = re.sub(r'Copy$', '', cell_text).strip()
                # Clean up HTML artifacts
                cell_text = re.sub(r'\s+', ' ', cell_text)  # Replace multiple whitespace with single space
                # If it's a code snippet cell (last column), format as code block if multiline
                if cell.find('code') and '\n' in cell_text:
                    # Multi-line code
                    cell_text = f'```\n{cell_text}\n```'
                elif cell.find('code'):
                    # Single line code
                    cell_text = f'`{cell_text}`'
                row_data.append(cell_text)
            markdown_content += '| ' + ' | '.join(row_data) + ' |\n'
        
        markdown_content += '\n'

# Write to Markdown file
with open('機器學習與深度學習實戰速查手冊.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print("Conversion completed!")