from typing import Dict
import requests
from PIL import Image
from io import BytesIO
import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.ops import roi_align
import torch.nn as nn
import json
import pandas as pd
import re
from copy import deepcopy
from source.prepare_data.image_downloading import download_sample_entry   

# simple cache for heavyweight models so format_entry can be called repeatedly
_MODEL_CACHE = {}

# def read_image_from_url(url: str):
#     """
#     Read image from URL and return PIL Image object
    
#     Args:
#         url: Image URL
        
#     Returns:
#         PIL Image object or None if failed
#     """
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         image = Image.open(BytesIO(response.content))
#         return image
#     except Exception as e:
#         print(f"Error reading image from {url}: {e}")
#         return None
    
def read_image_from_file(path: str):
    """
    Read image from a local file path and return PIL Image object.
    """
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error reading image from file {path}: {e}")
        return None
    
def tokenize_review(text: str):
    # Keep contractions (don't -> don't) by allowing internal apostrophes.
    # Matches sequences of letters/digits/apostrophes or any single non-word non-space char.
    token_re = re.compile(r"[A-Za-z0-9']+|[^\w\s]")
    return token_re.findall(text)

def is_containing_punctation(text: str):
    # Return True if any punctuation (except apostrophe) exists in text.
    # Punctuation defined as any char that's NOT a word char, whitespace, or apostrophe.
    return bool(re.search(r"[^\w\s']", text))

def remove_punctuation(text: str, keep_apostrophe: bool = False) -> str:
    """
    Remove punctuation from text by replacing punctuation with spaces and normalizing whitespace.
    If keep_apostrophe=True, preserves internal apostrophes (e.g. don't).
    """
    if keep_apostrophe:
        cleaned = re.sub(r"[^\w\s']+", " ", text)
    else:
        cleaned = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def compute_true_token_span(text: str, term: str):
    """
    Compute the start/end token indices for `term` inside `text` when counting only
    word tokens (i.e. punctuation tokens are excluded). Indices are 0-based and
    measured over the word-only token sequence.

    Returns (start_idx, end_idx) or (None, None) if the term cannot be located.
    """
    # Tokenize the text (keeps punctuation as separate tokens)
    tokens = tokenize_review(text)

    # Build list mapping from original token index -> normalized word (skip pure-punctuation)
    indexed_norm = []  # list of (orig_index, normalized_word)
    for idx, tok in enumerate(tokens):
        # normalize token by removing non-word except apostrophe
        norm = re.sub(r"[^\w']+", "", tok).lower()
        if norm:
            indexed_norm.append((idx, norm))

    # Normalize the term into tokens (preserve apostrophes inside words)
    term_norm = re.sub(r"[^\w']+", " ", term).strip().lower()
    term_tokens = [t for t in term_norm.split() if t]

    if not term_tokens:
        return None, None

    L = len(term_tokens)
    # find subsequence match in indexed_norm and return original token indices
    for i in range(len(indexed_norm) - L + 1):
        window = [w for (_, w) in indexed_norm[i:i + L]]
        if window == term_tokens:
            start_orig = indexed_norm[i][0]
            end_orig = indexed_norm[i + L - 1][0]
            return start_orig, end_orig
    return None, None

# ==================================================
# Format VLP-MABSA
# ==================================================
def prepare_faster_rcnn_model(device="cuda"):
    # Use ResNet-101 backbone with FPN from torchvision
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()
    return model

def extract_region_features(image, model, transform, device="cuda", num_regions=36):
    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Run detection
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # Get boxes and scores
    boxes = outputs["boxes"]
    scores = outputs["scores"]

    if len(boxes) == 0:
        return np.zeros((num_regions, 2048)), np.zeros((num_regions, 4))

    # Select top-N detections
    top_indices = scores.topk(k=min(num_regions, len(scores))).indices
    selected_boxes = boxes[top_indices]
    # Prepare ROI Align (need batch index for each box)
    box_indices = torch.zeros((len(selected_boxes),), dtype=torch.int64).to(device)
    rois = torch.cat([box_indices[:, None], selected_boxes], dim=1)
    # Extract FPN feature maps
    with torch.no_grad():
        feature_maps = model.backbone(img_tensor)  # dict of feature maps from FPN

    # Use the highest resolution feature map (often '0')
    fmap = list(feature_maps.values())[0]  # using first key's feature map from FPN
    # ROI Align: output size = (7, 7)
    pooled = roi_align(fmap, rois, output_size=(7, 7), spatial_scale=1.0)
    # Global average pooling → (num_regions, 2048)
    features = pooled.mean(dim=[2, 3])

    return features.cpu().numpy(), selected_boxes.cpu().numpy()

def project_features(region_features, output_dim=2048, device=None):
    # dynamically handle input dim and device instead of assuming 2048/CUDA
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # region_features may be numpy array; convert to tensor on the chosen device
    region_tensor = torch.tensor(region_features, dtype=torch.float32, device=device)
    in_dim = region_tensor.shape[1] if region_tensor.ndim == 2 else int(region_tensor.numel())
    projection_layer = nn.Linear(in_dim, output_dim).to(device)
    with torch.no_grad():
        projected = projection_layer(region_tensor).cpu().numpy()
    return projected

def format_vlp_mabsa(entry: Dict, image_extract_model, device, term_type="all", out_dir="./data/text_image_set/vlp-mabsa/region_box", set_type="train"):
    review = entry["review"]
    words = tokenize_review(review)

    # ensure outputs exist even if review aspects/opinions are missing
    aspect_outputs, opinion_outputs = [], []

    if all([
        entry.get("review_aspects", None) is not None,
        entry.get("review_opinions", None) is not None,
        entry.get("review_aspect_categories", None) is not None,
        entry.get("review_opinion_categories", None) is not None
    ]):
 
        aspect_outputs, opinion_outputs = [], []
        for i in range(len(entry["review_aspects"])):
            aspect = entry["review_aspects"][i]
            opinion = entry["review_opinions"][i]
            polarity = entry["review_opinion_categories"][i]
            if polarity == "Positive":
                polarity = "POS"
            elif polarity == "Negative":
                polarity = "NEG"
            else:
                polarity = "NEU"

            # compute true positions over word-only tokens (skip punctuation tokens)
            a_start, a_end = compute_true_token_span(review, aspect["term"])
            o_start, o_end = compute_true_token_span(review, opinion["term"])

            # if either term cannot be located, skip this pair
            if a_start is None or o_start is None:
                continue

            aspect_outputs.append(
                {   
                    "term": remove_punctuation(aspect["term"]).split(),
                    "from": a_start,
                    "to": a_end + 1,
                    "polarity": polarity,
                    "field": entry["review_aspect_categories"][i]
                }
            )

            opinion_outputs.append(
                {
                    "term": remove_punctuation(opinion["term"]).split(),
                    "from": o_start,
                    "to": o_end + 1,
                    "polarity": polarity,
                    "field": entry["review_opinion_categories"][i]
                }
            )
    else:
        print(f"Warning: Missing aspects/opinions in entry {entry['text_id']}/{entry['image_id']}")

    text_outputs = {
            "words": words,
            "image_id": entry["image_id"]+".jpg",
            "aspects": aspect_outputs,
            "opinions": opinion_outputs
        }
    
    if term_type == "aspect":
        text_outputs.pop("opinions", None)
    elif term_type == "opinion":
        text_outputs.pop("aspects", None)
    elif term_type == "all":
        pass
    else: 
        raise ValueError(f"Invalid term_type: {term_type}. Must be one of 'all', 'aspect', 'opinion'.")
    
    # format image
    # Ensure output directory exists
    out_dir = os.path.join(out_dir, set_type)
    os.makedirs(out_dir, exist_ok=True)
    
    att_dir = os.path.join(out_dir, '_att')
    os.makedirs(att_dir, exist_ok=True) 

    box_dir = os.path.join(out_dir, '_box')
    os.makedirs(box_dir, exist_ok=True)

    if os.path.exists(os.path.join(att_dir, f"{entry['image_id']}.npz")) and \
       os.path.exists(os.path.join(box_dir, f"{entry['image_id']}.npy")):
        return text_outputs
    # image_url = entry["photo_url"]
    # image = read_image_from_url(image_url)
    image_path = "./data/hotel_data/images/" + entry["image_id"] + ".jpg"
    image = read_image_from_file(image_path)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # If image download failed, fall back to zero features so saving still works
    num_regions = 36
    if image is None:
        region_feat = np.zeros((0, 0), dtype=np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
    else:
        region_feat, boxes = extract_region_features(image, image_extract_model, transform=transform, device=device, num_regions=num_regions)

    # Project features to fixed dimension 2048
    region_proj_feat = project_features(region_feat, output_dim=2048, device=device)
    region_proj_feat = np.asarray(region_proj_feat, dtype=np.float32)
    if region_proj_feat.ndim == 1:
        region_proj_feat = region_proj_feat.reshape(1, -1)

    # Pad/truncate projected features to exactly (num_regions, 2048)
    if region_proj_feat.size == 0:
        region_proj_feat = np.zeros((num_regions, 2048), dtype=np.float32)
    else:
        rows, cols = region_proj_feat.shape
        if cols != 2048:
            # re-project to 2048 if projection returned a different dim
            region_proj_feat = project_features(region_proj_feat, output_dim=2048, device=device)
            region_proj_feat = np.asarray(region_proj_feat, dtype=np.float32)
            if region_proj_feat.ndim == 1:
                region_proj_feat = region_proj_feat.reshape(1, -1)
            rows, cols = region_proj_feat.shape
        if rows < num_regions:
            pad = np.zeros((num_regions - rows, 2048), dtype=np.float32)
            region_proj_feat = np.vstack([region_proj_feat, pad])
        elif rows > num_regions:
            region_proj_feat = region_proj_feat[:num_regions]

    # Normalize boxes to shape (num_regions, 4)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        boxes = np.zeros((num_regions, 4), dtype=np.float32)
    else:
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, 4)
        brow = boxes.shape[0]
        if brow < num_regions:
            padb = np.zeros((num_regions - brow, 4), dtype=np.float32)
            boxes = np.vstack([boxes, padb])
        elif brow > num_regions:
            boxes = boxes[:num_regions]

    # Save to .npy (boxes)
    np.save(os.path.join(out_dir, '_box', f"{entry['image_id']}.npy"), boxes)
    # Save to .npz (features)
    np.savez(os.path.join(out_dir, '_att', f"{entry['image_id']}.npz"),
             feat=region_proj_feat)

    return text_outputs

def format_dtca(entry: Dict, device, term_type, out_dir="./data/text_image/dtca", set_type="train"):
    review = entry["review"]
    words = tokenize_review(review)

    aspects = entry.get("review_aspects", [])
    aspect_outputs = []
    if aspects:
        for i, aspect in enumerate(aspects):
            polarity = entry["review_opinion_categories"][i]
            encoded_polarity = 1 if polarity == "Positive" else -1 if polarity == "Negative" else 0

            aspect = remove_punctuation(aspect["term"])
            index_in_review = words.index(aspect) if aspect in words else -1
            if index_in_review == -1:
                continue
            else:
                temp_words = deepcopy(words)
                temp_words[index_in_review] = "$AT$"
                temp_sentence = " ".join(temp_words)
                if (temp_sentence, aspect, encoded_polarity, f'{entry["image_id"]}.jpg') not in aspect_outputs:
                    aspect_outputs.append((temp_sentence, aspect, encoded_polarity))

    opinions = entry.get("review_opinions", [])
    opinion_outputs = []
    if opinions:
        for opinion in opinions:
            polarity = entry["review_opinion_categories"][i]
            encoded_polarity = 1 if polarity == "Positive" else -1 if polarity == "Negative" else 0

            opinion = remove_punctuation(opinion["term"])
            index_in_review = words.index(opinion) if opinion in words else -1
            if index_in_review == -1:
                continue
            else:
                temp_words = deepcopy(words)
                temp_words[index_in_review] = "$OT$"
                temp_sentence = " ".join(temp_words)
                if (temp_sentence, opinion, encoded_polarity, ) not in opinion_outputs:
                    opinion_outputs.append((temp_sentence, opinion, encoded_polarity))

    text_outputs = {
        "words": words,
        "image_id": f'{entry["image_id"]}.jpg',
        }
    
    if term_type == "aspect":
        text_outputs["aspects"] = aspect_outputs
    elif term_type == "opinion":
        text_outputs["opinions"] = opinion_outputs
    elif term_type == "all":
        text_outputs["aspects"] = aspect_outputs
        text_outputs["opinions"] = opinion_outputs
    else:
        raise ValueError(f"Invalid term_type: {term_type}. Must be one of 'all', 'aspect', 'opinion'.")
    
    # no needs to process image for DTCA format
    return text_outputs


# =================================================
# Format wrapper
# =================================================

def format_entry(entry: Dict, strategy, term_type="aspect", device="cuda", set_type="train"):
    # check if images already downloaded
    if os.path.exists(f"./data/hotel_data/images/{entry['image_id']}.jpg"):
        print("Image already exists, skipping download.")
        pass
    else:

        output_dir = "./data/hotel_data/images"
        downloaded_path = download_sample_entry(entry, output_dir)
        print(f"Downloaded image to: {downloaded_path}")

    
    if strategy == "vlp-mabsa":
        if device not in _MODEL_CACHE:
            _MODEL_CACHE[device] = prepare_faster_rcnn_model(device)
        image_extract_model = _MODEL_CACHE[device]
        return format_vlp_mabsa(entry, image_extract_model = image_extract_model, device=device, 
                                term_type=term_type, set_type=set_type)
    
    if strategy == "dtca":
        return format_dtca(entry, device=device, term_type=term_type, set_type=set_type)

    return None
