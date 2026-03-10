import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def matching(inp_path, similarity_path, out_path):
    with open(inp_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(similarity_path, "r", encoding="utf-8") as f:
        similarity_scores = json.load(f)

    thd = 0.3
    k = min(similarity_scores.get("no_of_review", 0), similarity_scores.get("no_of_captions", 0))
    topk_scores = similarity_scores.get("similarity_scores", [])[:k]
    filter_scores = [entry for entry in topk_scores if entry["similarity"] >= thd]
    other_scores = [entry for entry in similarity_scores.get("similarity_scores", []) if entry not in filter_scores]
    
    text_image_matches = []
    image_matches, text_matches = set(), set()
    for entry in filter_scores:
        sentence = entry["sentence"]
        caption = entry["caption"]
        
        text_matches.add(sentence)
        image_matches.add(caption)

        four_ext = data.get("four_extraction", {})
        review_photo = data.get("review_photo", {})

        # index
        sentence_idx = list(four_ext.keys()).index(sentence)
        image_idx = list(review_photo.values()).index(caption)

        # image path & id
        image_path = list(review_photo.keys())[image_idx]
        image_basename = os.path.splitext(os.path.basename(image_path))[0]

        text_image_matches.append({
            "text_id": f"{data['id']}_{sentence_idx}",
            "image_id": f"{image_basename}",
            "image_path": image_path,
            "text": sentence,
            "label": four_ext.get(sentence, []),
        })
        
    # export text-image matches
    out_text_image_matches_path = os.path.join(out_path, "text_image", str(data['id']) + ".json")
    os.makedirs(os.path.dirname(out_text_image_matches_path), exist_ok=True)
    with open(out_text_image_matches_path, "w", encoding="utf-8") as f:
        json.dump(text_image_matches, f, indent=4, ensure_ascii=False)
        
        
    text_only = []; image_only = []
    for entry in other_scores:
        # save for text only first
        if entry["sentence"] not in text_matches:
            out_text_only = {
                "text_id": f"{data['id']}_{list(data.get('four_extraction', {}).keys()).index(entry['sentence'])}",
                "text": entry["sentence"],
                "label": data.get("four_extraction", {}).get(entry["sentence"], []),
            }
            if out_text_only not in text_only:
                text_only.append(out_text_only)
            
        if entry["caption"] not in image_matches:
            image_idx = list(data.get("review_photo", {}).values()).index(entry["caption"])
            image_path = list(data.get("review_photo", {}).keys())[image_idx]
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            
            out_image_only = {
                "image_id": f"{image_basename}",
                "image_path": image_path,
                "caption": entry["caption"],
            }
            if out_image_only not in image_only:
                image_only.append(out_image_only)
            
            
    out_text_only_path =  os.path.join(out_path, "text_only", str(data['id']) + ".json")
    os.makedirs(os.path.dirname(out_text_only_path), exist_ok=True)
    with open(out_text_only_path, "w", encoding="utf-8") as f:
        json.dump(text_only, f, indent=4, ensure_ascii=False)
        
    out_image_only_path = os.path.join(out_path, "image_only", str(data['id']) + ".json")
    os.makedirs(os.path.dirname(out_image_only_path), exist_ok=True)
    with open(out_image_only_path, "w", encoding="utf-8") as f:
        json.dump(image_only, f, indent=4, ensure_ascii=False)
        
        
        
# def get_matching_info(similarity_scores, threshold=0.3):
#     # turn to all positive
#     new_similarity_scores = [
#         {"text_id": entry["text_id"], "image_id": entry["image_id"], "similarity": abs(entry["similarity"])}
#         for entry in similarity_scores
#     ]

#     # save to new dict
#     similarity_info = {}
#     for entry in new_similarity_scores:
#         original_id = entry["image_id"].split("_")[0]
#         if original_id not in similarity_info:
#             similarity_info[original_id] = {
#                 "similarities": [],
#                 "text_ids": [],
#                 "image_ids": []
#             }
#         # if entry["text_id"] not in similarity_info[original_id]["text_ids"] and entry["image_id"] not in similarity_info[original_id]["image_ids"]:
#         similarity_info[original_id]["similarities"].append(entry["similarity"])
#         similarity_info[original_id]["text_ids"].append(entry["text_id"])
#         similarity_info[original_id]["image_ids"].append(entry["image_id"])


#     # match
#     matching_info = {}
#     for key, entry in similarity_info.items():
#         # find top similarity
#         sim_list = np.array(entry["similarities"])
#         k = min(len(set(entry["text_ids"])), len(set(entry["image_ids"])))
#         # filter by threshold then take top-k (descending)
#         indices_above = np.where(sim_list >= threshold)[0]
#         if indices_above.size:
#             filtered_sorted = indices_above[np.argsort(-sim_list[indices_above])]
#             topk_indices = filtered_sorted[:k]
#         else:
#             topk_indices = np.array([], dtype=int)
#         topk_similarities = sim_list[topk_indices]
#         match_texts = [entry["text_ids"][i] for i in topk_indices]
#         match_images = [entry["image_ids"][i] for i in topk_indices]
        
#         # match_info[key] = list(zip(match_texts, match_images, topk_similarities))
#         unmatched_texts = set(entry["text_ids"]) - set(match_texts)
#         unmatched_images = set(entry["image_ids"]) - set(match_images)

#         # only_text_info[key] = list(unmatched_texts)
#         # only_image_info[key] = list(unmatched_images)
#         matching_info[key] = {
#             "match_info": list(zip(match_texts, match_images, topk_similarities)),
#             "only_text_info": list(unmatched_texts),
#             "only_image_info": list(unmatched_images),
#         }
#     return matching_info

# def matching(text_path = "./model_call/data/fine_dataset.json",
# image_path = "./model_call/data/object_with_caption_dataset_blip_large.json",
# macthing_info_path = "./model_call/data/matching_info.json",):

#     with open(text_path, "r", encoding="utf-8") as f:
#         text_data = json.load(f)

#     with open(image_path, "r", encoding="utf-8") as f:
#         image_data = json.load(f)

#     with open(macthing_info_path, "r", encoding="utf-8") as f:
#         matching_info = json.load(f)

#     text_samples = text_data
#     image_samples = image_data


#     # turn to dict for easy tracking
#     text_samples_by_ids = {text["id"]: text for text in text_samples}

#     image_samples_by_ids = {}
#     for image in image_samples:
#         original_image_id = image["id"].split("_")[0]
#         object_detections = image.get("detected_objects", [])
#         if len(object_detections) > 0:
#             for i, obj in enumerate(object_detections):
#                 image_samples_by_ids[f"{original_image_id}_{i}"] = {
#                     "id": original_image_id + f"_{i}",
#                     "photo_url": image["photo_url"],
#                     "photo_caption": obj.get("sub_image_caption", ""),
#                     "label": obj.get("label", ""),
#                     "bbox": obj.get("bbox", [None, None, None, None]),
#                     "confidence": obj.get("confidence", None),
#                 }
        
#         image_samples_by_ids[f"{original_image_id}_main"] = {
#             "id": original_image_id + f"_main",
#             "photo_url": image["photo_url"],
#             "photo_caption": image.get("original_caption", ""),
#             "label": "main_image",
#             "bbox": [None, None, None, None],
#             "confidence": 1,
#             } # remove text sample if no object detected

#     match_dataset = []
#     only_text_dataset = []
#     only_image_dataset = []

#     for info in tqdm(matching_info):
#         match_info = matching_info[info]["match_info"]
#         only_text_info = matching_info[info]["only_text_info"]
#         only_image_info = matching_info[info]["only_image_info"]

#         for text_id, image_id, sim in match_info:
#             text_sample = text_samples_by_ids.get(text_id, None)
#             image_sample = image_samples_by_ids.get(image_id, None)
#             if text_sample and image_sample:
#                 match_dataset.append({
#                     "text_id": text_id,
#                     "image_id": image_id,
#                     "similarity": sim,
#                     "review": text_sample.get("review", ""),
#                     "review_aspects": text_sample.get("review_aspects", []),
#                     "review_aspect_categories": text_sample.get("review_aspect_categories", []),
#                     "review_opinions": text_sample.get("review_opinions", []),
#                     "review_opinion_categories": text_sample.get("review_opinion_categories", []),
#                     "photo_url": image_sample.get("photo_url", ""),
#                     "photo_caption": image_sample.get("photo_caption", ""),
#                     "label": image_sample.get("label", ""),
#                     "bbox": image_sample.get("bbox", [None, None, None, None]),
#                     "confidence": image_sample.get("confidence", None),
#                 })

#         for text_id in only_text_info:
#             text_sample = text_samples_by_ids.get(text_id, None)
#             if text_sample:
#                 only_text_dataset.append(text_sample)

#         for image_id in only_image_info:
#             image_sample = image_samples_by_ids.get(image_id, None)
#             if image_sample:
#                 only_image_dataset.append(image_sample)

#     with open("./model_call/data/text_image_dataset.json", "w", encoding="utf-8") as f:
#         json.dump(match_dataset, f, ensure_ascii=False, indent=4)
#     with open("./model_call/data/only_text_dataset.json", "w", encoding="utf-8") as f:
#         json.dump(only_text_dataset, f, ensure_ascii=False, indent=4)
#     with open("./model_call/data/only_image_dataset.json", "w", encoding="utf-8") as f:
#         json.dump(only_image_dataset, f, ensure_ascii=False, indent=4)


#     # print(image_samples_by_ids["0_main"])







    
if __name__ == "__main__":
    inp_path = "source/data/image_caption_batch/0.json"
    similarity_path = "source/data/similarity_scores/0.json"
    out_path = "source/data"
    
    matching(inp_path, similarity_path, out_path)