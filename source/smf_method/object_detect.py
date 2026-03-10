import os
import cv2
import json
import requests
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import logging


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

def download_image(url, save_path):
    """Download an image from a URL and save it to a specified path."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        logging.info(f"Image downloaded and saved to {save_path}")
    else:
        logging.error(f"Failed to download image from {url}. Status code: {response.status_code}")
        
    return response.status_code == 200

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

def crop_and_save(
    img: np.ndarray,
    bbox: list,
    save_dir: str,
    save_name: str
):
    """
    Crop image by bbox and save to disk.

    Args:
        img (np.ndarray): original image
        bbox (list): [x1, y1, x2, y2]
        save_dir (str): directory to save cropped image
        save_name (str): filename of cropped image

    Returns:
        str | None: saved crop path if success, else None
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # clamp bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop_img = img[y1:y2, x1:x2]
    if crop_img.size == 0:
        return None

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, crop_img)

    return save_path


def detect_object(yolo_model:YOLO, inp_data_path, outp_data_path):
    if not os.path.exists(outp_data_path):
    
        out_image_folder = os.path.join(os.path.dirname(outp_data_path), "images")
        os.makedirs(out_image_folder, exist_ok=True)
        
        with open(inp_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            

        photo_dict = data.get("review_photo", {})
        new_photo_dict = {}
        for idx, url in photo_dict.items():
            # download image
            img_save_path = os.path.join(out_image_folder, f"{data['id']}_{idx}.jpg")
            new_photo_dict[img_save_path] = True
            
            # check if image already exists to avoid redundant downloads
            if not os.path.exists(img_save_path):
                check_img = download_image(url, img_save_path) # also checks if download was successful
            else:
                check_img = True
            
            # stop if image download failed
            if not check_img:
                new_photo_dict[img_save_path] = False
            else:
                # read image
                img = cv2.imread(img_save_path)
                if img is None:
                    new_photo_dict[img_save_path] = False
                    continue
                
                # get bboxes
                detected_objects = get_bboxes_from_image(img, yolo_model)
                
                # crop and save
                crop_paths = [""]
                for obj_idx, obj in enumerate(detected_objects):
                    bbox = obj["bbox"]
                    crop_save_name = f"{data['id']}_{idx}_{obj_idx}.jpg"
                    
                    crop_save_path = crop_and_save(
                        img,
                        bbox,
                        out_image_folder,
                        crop_save_name
                    )
                    if crop_save_path:
                        crop_paths.append(crop_save_path)
                        
                    new_photo_dict[crop_save_path] = True
                
            del data["review_photo"]
            data["review_photo"] = new_photo_dict

        with open(outp_data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        # print(f"Processed file {inp_data_path} and saved output to {outp_data_path}")   
        
        
if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
    yolo_model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)
    total_files = 121584
    
    
    files = [f for f in os.listdir("D:/MABSA/source/data/train") if f.endswith(".json")]
    logging.info(f"Number of files to process: {len(files)}")
    
    for file in tqdm(files, desc="Processing files"):
    
        inp_path = os.path.join("D:/MABSA/source/data/train", file)
        out_path = os.path.join("D:/MABSA/source/data/object_detected_batch", file)
        
        detect_object(
            yolo_model,
            inp_data_path=inp_path,
            outp_data_path=out_path
        )