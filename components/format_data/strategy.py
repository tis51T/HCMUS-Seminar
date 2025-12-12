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

# simple cache for heavyweight models so format_entry can be called repeatedly
_MODEL_CACHE = {}

def read_image_from_url(url: str):
    """
    Read image from URL and return PIL Image object
    
    Args:
        url: Image URL
        
    Returns:
        PIL Image object or None if failed
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error reading image from {url}: {e}")
        return None
    
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

def format_vlp_mabsa(entry: Dict, image_extract_model, device, out_dir="./data/text_image/vlp-mabsa/region_box", set_type="train"):
    review = entry["review"]
    # Tokenize the review so punctuation stays as separate tokens.
    # This will split words and keep punctuation like . , ! ? : ; as separate tokens.


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
                    "field": entry["review_aspect_categories"][i]
                }
            )


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
        return {
            "words": words,
            "image_id": entry["image_id"],
            "aspects": aspect_outputs,
            "opinions": opinion_outputs
        }
    
    

    # image_url = entry["photo_url"]
    # image = read_image_from_url(image_url)
    image_path = "./data/images/" + entry["image_id"] + ".jpg"
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

    return {
        "words": words,
        "image_id": entry["image_id"],
        "aspects": aspect_outputs,
        "opinions": opinion_outputs
    }
# =================================================
# Format wrapper
# =================================================

def format_entry(entry: Dict, device="cuda", strategy="vlp-mabsa", set_type="train"):
    if strategy == "vlp-mabsa":
        # create and cache model per device so repeated calls (e.g. in a loop)
        # don't reinstantiate the heavyweight detection model each time.
        if device not in _MODEL_CACHE:
            _MODEL_CACHE[device] = prepare_faster_rcnn_model(device)
        image_extract_model = _MODEL_CACHE[device]
        return format_vlp_mabsa(entry, image_extract_model, device, set_type=set_type)
    return None


entry =[
    {
        "text_id": "10_1",
        "image_id": "10_main",
        "similarity": 0.5631591081619263,
        "review": "Lovely room and pool",
        "review_aspects": [
            {
                "term": "room",
                "from": 1,
                "to": 1
            },
            {
                "term": "pool",
                "from": 3,
                "to": 3
            }
        ],
        "review_aspect_categories": [
            "Facility",
            "Facility"
        ],
        "review_opinions": [
            {
                "term": "Lovely",
                "from": 0,
                "to": 0
            },
            {
                "term": "Lovely",
                "from": 0,
                "to": 0
            }
        ],
        "review_opinion_categories": [
            "Positive",
            "Positive"
        ],
        "photo_url": "https://q-xx.bstatic.com/xdata/images/xphoto/max1280x900/226822174.jpg?k=342642af4a2d9824a9115c0f8064f1e3e2a62e91a323261f796b8ec487ac99b2&o=",
        "photo_caption": "there is a small pool in the middle of a large house",
        "label": "main_image",
        "bbox": [
            None,
            None,
            None,
            None
        ],
        "confidence": 1
    },
    {
        "text_id": "10_1",
        "image_id": "10_1",
        "similarity": 0.45163166522979736,
        "review": "Lovely room and pool",
        "review_aspects": [
            {
                "term": "room",
                "from": 1,
                "to": 1
            },
            {
                "term": "pool",
                "from": 3,
                "to": 3
            }
        ],
        "review_aspect_categories": [
            "Facility",
            "Facility"
        ],
        "review_opinions": [
            {
                "term": "Lovely",
                "from": 0,
                "to": 0
            },
            {
                "term": "Lovely",
                "from": 0,
                "to": 0
            }
        ],
        "review_opinion_categories": [
            "Positive",
            "Positive"
        ],
        "photo_url": "https://q-xx.bstatic.com/xdata/images/xphoto/max1280x900/226822174.jpg?k=342642af4a2d9824a9115c0f8064f1e3e2a62e91a323261f796b8ec487ac99b2&o=",
        "photo_caption": "this is an image of a patio with a pool and a blue umbrella",
        "label": "umbrella",
        "bbox": [
            92,
            286,
            301,
            463
        ],
        "confidence": 0.3957480490207672
    },
    {
        "text_id": "24_0",
        "image_id": "24_14",
        "similarity": 0.49013298749923706,
        "review": "Going out is the beach, the sea view, very comfortable",
        "review_aspects": [
            {
                "term": "beach,",
                "from": 4,
                "to": 4
            },
            {
                "term": "sea view,",
                "from": 6,
                "to": 7
            }
        ],
        "review_aspect_categories": [
            "Facility",
            "Facility"
        ],
        "review_opinions": [
            {
                "term": "comfortable",
                "from": 9,
                "to": 9
            },
            {
                "term": "comfortable",
                "from": 9,
                "to": 9
            }
        ],
        "review_opinion_categories": [
            "Positive",
            "Positive"
        ],
        "photo_url": "https://q-xx.bstatic.com/xdata/images/xphoto/max1280x900/149729069.jpg?k=f26b720eb990944e252c044cb28af81896d851f5505532b93fe14ad16fa6e228&o=",
        "photo_caption": "there are many lounge chairs and umbrellas on the beach near the water",
        "label": "umbrella",
        "bbox": [
            0,
            179,
            369,
            830
        ],
        "confidence": 0.31494948267936707
    },
    {
        "text_id": "24_0",
        "image_id": "24_main",
        "similarity": 0.48541101813316345,
        "review": "Going out is the beach, the sea view, very comfortable",
        "review_aspects": [
            {
                "term": "beach,",
                "from": 4,
                "to": 4
            },
            {
                "term": "sea view,",
                "from": 6,
                "to": 7
            }
        ],
        "review_aspect_categories": [
            "Facility",
            "Facility"
        ],
        "review_opinions": [
            {
                "term": "comfortable",
                "from": 9,
                "to": 9
            },
            {
                "term": "comfortable",
                "from": 9,
                "to": 9
            }
        ],
        "review_opinion_categories": [
            "Positive",
            "Positive"
        ],
        "photo_url": "https://q-xx.bstatic.com/xdata/images/xphoto/max1280x900/149729069.jpg?k=f26b720eb990944e252c044cb28af81896d851f5505532b93fe14ad16fa6e228&o=",
        "photo_caption": "several lounge chairs and umbrellas on a sandy beach near the ocean",
        "label": "main_image",
        "bbox": [
            None,
            None,
            None,
            None
        ],
        "confidence": 1
    },
]

# for e in entry:
#     outputs= format_entry(e, device="cpu", strategy="vlp-mabsa", set_type="train")
#     print(outputs)
