import os
import json
import glob
from collections import defaultdict

chapters = defaultdict(set)

for file in glob.glob("*.ipynb"):
    if file.startswith(('01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19')):
        chapter = file.split('_')[0]
        with open(file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                for line in cell['source']:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        chapters[chapter].add(line)

for ch in sorted(chapters.keys()):
    print(f"Chapter {ch}:")
    for imp in sorted(chapters[ch]):
        print(f"  {imp}")
    print()