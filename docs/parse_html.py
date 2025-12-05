import sys
from bs4 import BeautifulSoup

# Read the HTML file
with open('機器學習與深度學習實戰速查手冊.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')

chapters = {}

for i in range(1, 20):
    section_id = f'chapter-{i}'
    section = soup.find('section', id=section_id)
    print(f"Chapter {i}: section found: {section is not None}")
    if section:
        h2 = section.find('h2')
        chapter_title = h2.get_text() if h2 else f'第{i}章'
        print(f"Title: {chapter_title}")
        table = section.find('table')
        print(f"Table found: {table is not None}")
        if table:
            rows = table.find_all('tr')
            print(f"Rows: {len(rows)}")
            data = []
            for row in rows[1:]:  # Skip header
                cols = row.find_all('td')
                print(f"Cols in row: {len(cols)}")
                if len(cols) == 5:
                    concept = cols[0].get_text().strip()
                    functions = cols[1].get_text().strip()
                    description = cols[2].get_text().strip()
                    params = cols[3].get_text().strip()
                    code = cols[4].get_text().strip()
                    data.append({
                        '關鍵概念': concept,
                        '重要函數': functions,
                        '功能簡介與應用場景': description,
                        'Fine-tune 用參數': params,
                        '範例代碼片段': code
                    })
            chapters[chapter_title] = data

import json
print(json.dumps(chapters, ensure_ascii=False, indent=2))