import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import os
from tqdm import tqdm


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load BLIP-large
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)

model.eval()


def _generate_caption(image_path, max_length=60):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length
        )

    caption = processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    return caption


def caption_image(data_path, output_path, max_length=60):
    with open(data_path, 'rb') as f:
        data = json.load(f)

    if not os.path.exists(output_path):
        review_photo = data.get("review_photo", {})
        if review_photo:
            for key, value in review_photo.items():
                if value != False: # only caption if the image is valid (not False)
                    caption = _generate_caption(image_path = key, max_length=max_length)
                    review_photo[key] = caption

    
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Process file {output_path}")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

# ===== TEST =====
if __name__ == "__main__":
    # inp_path = "./source/data/object_detected_batch/1.json"
    # output_path = inp_path.replace("object_detected_batch", "image_caption_batch")
    # caption_image(data_path=inp_path, output_path=output_path)
    
    files = [f for f in os.listdir("D:/MABSA/source/data/object_detected_batch") if f.endswith(".json")]
    
    for file in tqdm(files, desc="Processing files"):
    
        inp_path = os.path.join("D:/MABSA/source/data/object_detected_batch", file)
        out_path = os.path.join("D:/MABSA/source/data/image_caption_batch", file)
        
        caption_image(  
            data_path=inp_path,
            output_path=out_path
        )