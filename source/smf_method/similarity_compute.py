import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import os

# Load once (DO NOT reload inside function)
MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # fast, works on CPU

def compute_similarity(text, caption):
    # Compute embeddings
    embeddings = MODEL.encode([text, caption], convert_to_tensor=True)

    text_emb, caption_emb = embeddings
    # Cosine similarity
    sim = F.cosine_similarity(text_emb.unsqueeze(0), caption_emb.unsqueeze(0))
    
    return sim.item()

def compute_similarity_scores(inp_path, out_path):

    with open(inp_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    
    sentence_list = data.get("four_extraction", {}).keys()
    caption_list = data.get("review_photo", {}).values()
    
    sim_scores = []
    for sentence in sentence_list:
        for caption in caption_list:
            sim = compute_similarity(sentence, caption)
            sim_scores.append({
                "sentence": sentence,
                "caption": caption,
                "similarity": sim,
            })

    # sort by similarity descending
    sim_scores = sorted(sim_scores, key=lambda x: x["similarity"], reverse=True)
    output = {
        "no_of_review": len(sentence_list),
        "no_of_captions": len(caption_list),
        "similarity_scores": sim_scores
    }


    # save output
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
  
if __name__ == "__main__":
    inp = "./source/image_caption_batch/0.json"
    outp = "./source/similarity_scores/0.json"
    compute_similarity_scores(inp, outp)