import json

with open("../split_sentences_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
only_images = []
appended_ids = set()
for entry in data:
    simple_id = entry["id"].split("_")[0]
    new_entry = {"id": simple_id, "photo": entry["review_photo"]}
    if simple_id not in appended_ids:
        only_images.append({"id": simple_id, "photo_url": list(entry["review_photo"].keys())[0], "photo_caption": list(entry["review_photo"].values())[0]})
        appended_ids.add(simple_id)

with open("../only_images.json", "w") as f:
    json.dump(only_images, f)
    