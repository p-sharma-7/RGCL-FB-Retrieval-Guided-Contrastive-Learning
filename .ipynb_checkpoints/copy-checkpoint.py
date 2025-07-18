import json
import os

file1 = '/workspace/RGCL/data/HarMeme/memes/annotations/val.jsonl'
file2 = '/workspace/RGCL/data/HarMeme/memesP/defaults/annotations/val.jsonl'

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

# Load both JSONL files (each line is a JSON object)
list1 = load_jsonl(file1)
list2 = load_jsonl(file2)

# Merge the lists
merged = list1 + list2

# Save as .json (array of objects) — OR as .jsonl if needed
output_path = '/workspace/RGCL/data/gt/HarMeme/merged_val.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save as JSON array
with open(output_path, 'w') as f:
    json.dump(merged, f, indent=4)

print(f"Merged JSON saved to {output_path}")
