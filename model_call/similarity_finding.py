import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

# Load once (DO NOT reload inside function)
MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # fast, works on CPU

def compute_similarity(text, caption):
    # Compute embeddings
    embeddings = MODEL.encode([text, caption], convert_to_tensor=True)

    text_emb, caption_emb = embeddings

    # Cosine similarity
    sim = F.cosine_similarity(text_emb.unsqueeze(0), caption_emb.unsqueeze(0))
    
    return sim.item()

def get_similarity_scores(
text_path = "./model_call/fine_dataset.json",
image_path = "./model_call/object_with_caption_dataset_blip_large.json"):

    with open(text_path, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    with open(image_path, "r", encoding="utf-8") as f:
        image_data = json.load(f)

    text_samples = text_data
    image_samples = image_data


    # turn to dict for easy tracking
    text_samples_by_ids = {}
    for text in text_samples:
        original_text_id = text["id"].split("_")[0]
        if original_text_id not in text_samples_by_ids:
            text_samples_by_ids[original_text_id] = []
        text_samples_by_ids[original_text_id].append(text)

    image_samples_by_ids = {}
    for image in image_samples:
        original_image_id = image["id"].split("_")[0]
        image_samples_by_ids[original_image_id]= image  


    # match
    sim_scores = []
    for id in tqdm(text_samples_by_ids.keys(), desc="Computing similarity scores"):
        text_sample = text_samples_by_ids.get(id, [])
        image_sample = image_samples_by_ids.get(id, None)

        if text_sample and image_sample:
            main_caption = image_sample.get("original_caption", "")
            sub_captions = [entry.get("sub_image_caption", "") for entry in image_sample.get("detected_objects", [])]

            # mactch sup caption first
            for text in text_sample:
                review=text.get("review", "")
                aspects=[a.get("term", "") for a in text.get("review_aspects", [])]

                for i, caption in enumerate(sub_captions + [main_caption]):
                    sim = compute_similarity(review, caption)
                    text_id = text["id"]
                    image_id = image_sample["id"] + "_" + str(i if i < len(sub_captions) else "main")
                    
                    sim_scores.append({
                        "text_id": text_id,
                        "image_id": image_id,
                        "similarity": sim,
                    })

    return sim_scores

if __name__ == "__main__":
    scores = get_similarity_scores()
    with open("./model_call/similarity_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)