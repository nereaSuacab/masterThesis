import json

# Read the original file
with open('data/docstore.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Loop through nested structure and decode "__data__" strings if possible
for key, value in data.get("index_store/data", {}).items():
    if "__data__" in value:
        try:
            # Parse the inner JSON string into a dictionary
            inner_data = json.loads(value["__data__"])
            value["__data__"] = inner_data  # Replace the string with dict
        except json.JSONDecodeError:
            pass  # Ignore if it’s not valid JSON

# Write the fully formatted JSON to a new file
with open('data/index_formatted.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Fully formatted JSON written to data/index_formatted.json")