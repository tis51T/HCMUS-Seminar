# from format_data.strategy import format_entry
import json
from sklearn.model_selection import train_test_split
import os
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
from format_data.strategy import format_entry
import torch
from tqdm import tqdm

# =====================================================================================
with open("./data/text_image_dataset.json", "r") as f:
    raw_datasets = json.load(f)

# load again dataset for formatting
text_image_dir = os.path.join(".", "data", "text_image")
os.makedirs(text_image_dir, exist_ok=True)

train_path = os.path.join(text_image_dir, "train_dataset.json")
val_path = os.path.join(text_image_dir, "val_dataset.json")
test_path = os.path.join(text_image_dir, "test_dataset.json")

# If split files are missing, create them from the full dataset
if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
    raw_idx = [entry['text_id'].split("_")[0] for entry in raw_datasets]
    train_idx, test_idx = train_test_split(list(set(raw_idx)), test_size=0.2, random_state=42)
    test_idx, val_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    train_dataset = [entry for entry in raw_datasets if entry['text_id'].split("_")[0] in train_idx]
    val_dataset = [entry for entry in raw_datasets if entry['text_id'].split("_")[0] in val_idx]
    test_dataset = [entry for entry in raw_datasets if entry['text_id'].split("_")[0] in test_idx]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, indent=4, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_dataset, f, indent=4, ensure_ascii=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, indent=4, ensure_ascii=False)
else:
    with open(train_path, "r", encoding="utf-8") as f:
        train_dataset = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_dataset = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)


# format
strategy = "vlp-mabsa"  # or "vlp-mabsa"
device = "cuda" if torch.cuda.is_available() else "cpu"

formatted_train, formatted_test, formatted_val = [], [], []

for entry in tqdm(train_dataset[:10]):
    formatted_train.append(format_entry(entry, device=device, strategy="vlp-mabsa", set_type="train"))

for entry in tqdm(val_dataset[:10]):
    formatted_val.append(format_entry(entry, device=device, strategy="vlp-mabsa", set_type="val"))

for entry in tqdm(test_dataset[:10]):
    formatted_test.append(format_entry(entry, device=device, strategy="vlp-mabsa", set_type="test"))

with open(os.path.join(text_image_dir, "train.json"), "w", encoding="utf-8") as f:
    json.dump(list(set(formatted_train)), f, indent=4, ensure_ascii=False)
with open(os.path.join(text_image_dir, "val.json"), "w", encoding="utf-8") as f:
    json.dump(list(set(formatted_val)), f, indent=4, ensure_ascii=False)
with open(os.path.join(text_image_dir, "test.json"), "w", encoding="utf-8") as f:
    json.dump(list(set(formatted_test)), f, indent=4, ensure_ascii=False)