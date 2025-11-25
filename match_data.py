import json

with open("../fine_dataset.json", "r", encoding="utf-8") as f:
    fine_data = json.load(f)

with open("../images_with_caption_dataset.json", "r", encoding="utf-8") as f:
    image_data = json.load(f)

fine_samples = fine_data[:10]
image_samples = image_data[:5]

matched_data = []
only_text = []
only_image = []

for fine_entry in fine_samples:
    fine_id = fine_entry["id"].split("_")[0]
    matched = False
    fine_entry_review = fine_entry.get("review", {})
    for image_entry in image_samples:
        image_id = image_entry["id"]
        if fine_id == image_id:
            



            
            matched_data.append({
                "id": fine_id,
                "text": fine_entry["text"],
                "photo_url": image_entry["photo_url"],
                "photo_caption": image_entry["photo_caption"]
            })
            matched = True
            break
    if not matched:
        only_text.append({
            "id": fine_id,
            "text": fine_entry["text"]
        })