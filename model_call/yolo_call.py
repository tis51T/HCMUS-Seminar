import os
import cv2
import json
import requests
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


# ============================
# Sample input data
# ============================
data = [
    {
        "id": "10443",
        "photo": {
            "https://q-xx.bstatic.com/xdata/images/xphoto/max1280x900/246549059.jpg?k=81fd6facd589c47386892f9742fc63d88e893a316186346a61d997a66f494c6f&o=":
            "Swimming pool in/near Tam Coc Village Bungalow"
        }
    }
]


# ============================
# Load image from URL
# ============================
def read_image_from_url(url):
    """Download an image and decode into OpenCV format."""
    resp = requests.get(url)
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ============================
# Get bounding boxes only
# ============================
def get_bboxes_from_image(img, model):
    results = model(img, verbose=False)

    detected_objects = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            label = model.names[cls_id]
            confidence = float(box.conf)

            detected_objects.append(
                {
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence
                }
            )

    return detected_objects


# ============================
# Process the whole dataset
# ============================
def process_dataset(data, batch_size = 200):
    print(f"Data will process and save after every {batch_size} images.")
    
    model = YOLO("yolov8n.pt")   # Load YOLO once

    batch_id = 0
    count = 0
    for item in tqdm(data, desc="Processing images", unit="image"):
        url = item["photo_url"]

        img = read_image_from_url(url)
        if img is None:
            # print(f"Failed to load image from {url}")
            continue

        detected = get_bboxes_from_image(img, model)
        # Save results back into data
        item["detected_objects"] = detected

        count += 1
        if count % batch_size == 0:
            batch_id += 1
            # Optionally save intermediate results
            with open(f"../images_dataset_batch_{batch_id}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"\nSaved batch {batch_id} after processing {count} images.")

    # save the last batch
    batch_id += 1
    with open(f"../images_dataset_batch_{batch_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # return data


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    with open("../only_images.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    updated_data = process_dataset(data)

    # print(json.dumps(updated_data, indent=4, ensure_ascii=False))

    # # Optionally save to a JSON file
    # with open("../images_dataset.json", "w", encoding="utf-8") as f:
    #     json.dump(updated_data, f, indent=4, ensure_ascii=False)

    # print("\nProcessing complete. Bounding boxes added to dataset.")
