import json

with open('docstore.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('docstore_formatted.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)