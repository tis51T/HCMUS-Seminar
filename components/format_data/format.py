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
from format_data.strategy import format_entry, tokenize_review
import torch
from tqdm import tqdm
import pandas as pd


# helper to deduplicate list of dicts while preserving order
def get_unique_dicts(lst):
    seen = set()
    out = []
    for d in lst:
        # create a stable key from important fields
        key = (
            tuple(d.get('words', [])),
            tuple(d.get('term', [])),
            d.get('from'),
            d.get('to'),
            d.get('polarity'),
            d.get('field')
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

# deduplicate whole-entry JSON-serializable objects (preserve order)
def unique_entries(lst):
    seen = set()
    out = []
    for obj in lst:
        try:
            key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        except Exception:
            # fallback: string conversion
            key = str(obj)
        if key in seen:
            continue
        seen.add(key)
        out.append(obj)
    return out
# =====================================================================================
with open("./data/text_image/text_image_dataset.json", "r") as f:
    raw_datasets = json.load(f)

# load again dataset for formatting
text_image_dir = os.path.join(".", "data", "text_image")
os.makedirs(text_image_dir, exist_ok=True)

train_path = os.path.join(text_image_dir,  "train_dataset.json")
dev_path = os.path.join(text_image_dir, "dev_dataset.json")
test_path = os.path.join(text_image_dir, "test_dataset.json")

# If split files are missing, create them from the full dataset
if not (os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path)):
    raw_idx = [entry['text_id'].split("_")[0] for entry in raw_datasets]
    train_idx, test_idx = train_test_split(list(set(raw_idx)), test_size=0.2, random_state=42)
    test_idx, dev_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    train_dataset = [entry for entry in raw_datasets if entry['text_id'].split("_")[0] in train_idx]
    dev_dataset = [entry for entry in raw_datasets if entry['text_id'].split("_")[0] in dev_idx]
    test_dataset = [entry for entry in raw_datasets if entry['text_id'].split("_")[0] in test_idx]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, indent=4, ensure_ascii=False)
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_dataset, f, indent=4, ensure_ascii=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, indent=4, ensure_ascii=False)
else:
    with open(train_path, "r", encoding="utf-8") as f:
        train_dataset = json.load(f)
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_dataset = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)


# format
strategy = "dtca"  # or 
device = "cuda" if torch.cuda.is_available() else "cpu"
if strategy == "vlp-mabsa":
    # Collect outputs as plain lists (do not use DataFrame per request)
    train_outputs, dev_outputs, test_outputs = [], [], []

    out_dir_strategy = os.path.join(text_image_dir, strategy)
    os.makedirs(out_dir_strategy, exist_ok=True)

    for entry in tqdm(train_dataset):
        try:
            train_out = format_entry(entry, device=device, strategy=strategy, set_type="train")
        except Exception as e:
            print(f"format_entry failed for text_id={entry.get('text_id')} : {e}")
            continue

        if train_out:
            if isinstance(train_out, dict):
                train_outputs.append(train_out)
            elif isinstance(train_out, (list, tuple)):
                # two-list variant -> wrap into one entry
                if len(train_out) == 2 and isinstance(train_out[0], list) and isinstance(train_out[1], list):
                    train_outputs.append({
                        "words": tokenize_review(entry.get("review", "") ),
                        "image_id": entry.get("image_id"),
                        "aspects": train_out[0],
                        "opinions": train_out[1]
                    })
                else:
                    # assume flat list of aspect/opinion dicts — wrap as aspects
                    train_outputs.append({
                        "words": tokenize_review(entry.get("review", "") ),
                        "image_id": entry.get("image_id"),
                        "aspects": list(train_out),
                        "opinions": []
                    })
    # write JSON outputs
    with open(os.path.join(out_dir_strategy,"train.json"), "w", encoding="utf-8") as f:
        json.dump(unique_entries(train_outputs), f, indent=4, ensure_ascii=False)



    for entry in tqdm(dev_dataset):
        try:
            dev_out = format_entry(entry, device=device, strategy=strategy, set_type="dev")
        except Exception as e:
            print(f"format_entry failed for text_id={entry.get('text_id')} : {e}")
            continue

        if dev_out:
            if isinstance(dev_out, dict):
                dev_outputs.append(dev_out)
            elif isinstance(dev_out, (list, tuple)):
                if len(dev_out) == 2 and isinstance(dev_out[0], list) and isinstance(dev_out[1], list):
                    dev_outputs.append({
                        "words": tokenize_review(entry.get("review", "") ),
                        "image_id": entry.get("image_id"),
                        "aspects": dev_out[0],
                        "opinions": dev_out[1]
                    })
                else:
                    dev_outputs.append({
                        "words": tokenize_review(entry.get("review", "") ),
                        "image_id": entry.get("image_id"),
                        "aspects": list(dev_out),
                        "opinions": []
                    })
    with open(os.path.join(out_dir_strategy,"dev.json"), "w", encoding="utf-8") as f:
        json.dump(unique_entries(dev_outputs), f, indent=4, ensure_ascii=False)


    for entry in tqdm(test_dataset):
        try:
            test_out = format_entry(entry, device=device, strategy=strategy, set_type="test")
        except Exception as e:
            print(f"format_entry failed for text_id={entry.get('text_id')} : {e}")
            continue

        if test_out:
            if isinstance(test_out, dict):
                test_outputs.append(test_out)
            elif isinstance(test_out, (list, tuple)):
                if len(test_out) == 2 and isinstance(test_out[0], list) and isinstance(test_out[1], list):
                    test_outputs.append({
                        "words": tokenize_review(entry.get("review", "") ),
                        "image_id": entry.get("image_id"),
                        "aspects": test_out[0],
                        "opinions": test_out[1]
                    })
                else:
                    test_outputs.append({
                        "words": tokenize_review(entry.get("review", "") ),
                        "image_id": entry.get("image_id"),
                        "aspects": list(test_out),
                        "opinions": []
                    })

    with open(os.path.join(out_dir_strategy,"test.json"), "w", encoding="utf-8") as f:
        json.dump(unique_entries(test_outputs), f, indent=4, ensure_ascii=False)


elif strategy == "dtca":
        # Collect outputs as plain lists (do not use DataFrame per request)
    train_outputs, dev_outputs, test_outputs = "", "", ""

    out_dir_strategy = os.path.join(text_image_dir, strategy)
    os.makedirs(out_dir_strategy, exist_ok=True)

    for entry in tqdm(train_dataset):
        try:
            train_out = format_entry(entry, device=device, strategy=strategy, set_type="train", term_type="aspect")
        except Exception as e:
            print(f"format_entry failed for text_id={entry.get('text_id')} : {e}")
            continue

        if train_out:
            for aspect in train_out["aspects"]:
                out = f'{aspect[0]}\n{aspect[1]}\n{aspect[2]}\n{train_out["image_id"]}\n'
                if out not in train_outputs:
                    train_outputs += out

    with open(os.path.join(out_dir_strategy,"train.txt"), "w", encoding="utf-8") as f:
        f.write(train_outputs)

    for entry in tqdm(dev_dataset):
        try:
            dev_out = format_entry(entry, device=device, strategy=strategy, set_type="dev", term_type="aspect")
        except Exception as e:
            print(f"format_entry failed for text_id={entry.get('text_id')} : {e}")
            continue

        if dev_out:
            for aspect in dev_out["aspects"]:
                out = f'{aspect[0]}\n{aspect[1]}\n{aspect[2]}\n{dev_out["image_id"]}\n'
                if out not in dev_outputs:
                    dev_outputs += out

    with open(os.path.join(out_dir_strategy,"dev.txt"), "w", encoding="utf-8") as f:
        f.write(dev_outputs)

    for entry in tqdm(test_dataset):
        try:
            test_out = format_entry(entry, device=device, strategy=strategy, set_type="test", term_type="aspect")
        except Exception as e:
            print(f"format_entry failed for text_id={entry.get('text_id')} : {e}")
            continue

        if test_out:
            for aspect in test_out["aspects"]:
                out = f'{aspect[0]}\n{aspect[1]}\n{aspect[2]}\n{test_out["image_id"]}\n'
                if out not in test_outputs:
                    test_outputs += out

    with open(os.path.join(out_dir_strategy,"test.txt"), "w", encoding="utf-8") as f:
        f.write(test_outputs)

