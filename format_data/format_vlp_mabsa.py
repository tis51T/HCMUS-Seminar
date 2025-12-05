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


# ==================================================
# Format VLP-MABSA
# ==================================================
def prepare_faster_rcnn_model(device="cuda"):
    # Use ResNet-50 backbone with FPN from torchvision.
    # Prefer the new `weights=` API when available to avoid deprecation warnings.
    try:
        weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights).to(device)
    except Exception:
        # Fall back to older API if `weights` enum isn't available in this torchvision
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

def project_features(region_features, output_dim=768, device=None):
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

def format_vlp_mabsa(entry: Dict, image_extract_model, device, out_dir="./format_data/region_box", set_type="train"):
    review = entry["review"]
    words = entry["review"].split()
    aspects = entry["review_aspects"]
    opinions = entry["review_opinions"]

    formatted_aspects_outputs = []
    for aspect in aspects:
        review_without_aspect = " ".join(words[: aspect["from"]] + ["$AT$"] + words[aspect["to"] + 1 :]).strip()
        formatted_aspects_outputs.append(
            {
                "text_input": review_without_aspect,
                "aspect": aspect["term"],
            }
        )

    formatted_opinions_outputs = []
    for opinion in opinions:
        review_without_opinion = " ".join(words[: opinion["from"]] + ["$OT$"] + words[opinion["to"] + 1 :]).strip()
        formatted_opinions_outputs.append(
            {
                "text_input": review_without_opinion,
                "opinion": opinion["term"],
            }
        )

    # format image
    image_url = entry["photo_url"]
    image = read_image_from_url(image_url)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # If image download failed, fall back to zero features so saving still works
    if image is None:
        num_regions = 36
        region_feat = np.zeros((num_regions, 2048), dtype=np.float32)
        boxes = np.zeros((num_regions, 4), dtype=np.float32)
    else:
        region_feat, boxes = extract_region_features(image, image_extract_model, transform=transform, device=device, num_regions=36)

    region_proj_feat = project_features(region_feat, output_dim=768, device=device)

    # save formatted data: text into json and image features into npz/npy
    formatted_entry = {
        "text_id": entry["text_id"],
        "image_id": entry["image_id"],
        "formatted_aspects": formatted_aspects_outputs,
        "formatted_opinions": formatted_opinions_outputs,
    }
    

    # Ensure output directory exists
    out_dir = os.path.join(out_dir, set_type)
    os.makedirs(out_dir, exist_ok=True)
    
    att_dir = os.path.join(out_dir, '_att')
    os.makedirs(att_dir, exist_ok=True) 

    box_dir = os.path.join(out_dir, '_box')
    os.makedirs(box_dir, exist_ok=True)

    # Save to .npy (just features)
    np.save(os.path.join(out_dir, '_box', f"{entry['image_id']}.npy"), region_proj_feat)

    # Save to .npz (features + text + boxes)
    np.savez(os.path.join(out_dir, '_att', f"{entry['image_id']}.npz"),
             img_features=region_proj_feat,
             boxes=boxes,
             text=review)

    return formatted_entry


device = "cuda" if torch.cuda.is_available() else "cpu"
entry={
        "text_id": "5609_2",
        "image_id": "5609_0",
        "similarity": 0.3973047137260437,
        "review": "Beautiful sea view and fully equipped apartment",
        "review_aspects": [
            {
                "term": "sea view",
                "from": 1,
                "to": 2
            },
            {
                "term": "apartment",
                "from": 6,
                "to": 6
            }
        ],
        "review_aspect_categories": [
            "Facility",
            "Facility"
        ],
        "review_opinions": [
            {
                "term": "Beautiful",
                "from": 0,
                "to": 0
            },
            {
                "term": "Beautiful",
                "from": 0,
                "to": 0
            }
        ],
        "review_opinion_categories": [
            "Positive",
            "Positive"
        ],
        "photo_url": "https://q-xx.bstatic.com/xdata/images/xphoto/max1280x900/360793783.jpg?k=671089861fa5c6b253b3d0e6812e2a67f370522a9045c25aaa6fec89b2989f65&o=",
        "photo_caption": "there is a table and chair on a balcony overlooking the beach",
        "label": "chair",
        "bbox": [
            31,
            466,
            450,
            900
        ],
        "confidence": 0.5115492939949036
    }



# =================================================
# Format wrapper
# =================================================

def format_data(entry: Dict, device="cuda", strategy="vlp-mabsa"):
    if strategy == "vlp-mabsa":
         image_extract_model = prepare_faster_rcnn_model(device)
         return format_vlp_mabsa(entry, image_extract_model,device)
    return None
    


if __name__ == "__main__":
    formatted_entry = format_data(entry, device=device, strategy="vlp-mabsa")
    print(json.dumps(formatted_entry, indent=4))