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

if device == "cuda":
    model.half()

model.eval()


def _load_image(image_path):
    return Image.open(image_path).convert("RGB")


def _generate_captions(image_paths, max_length=60, batch_size=8):
    captions = {}
    if not image_paths:
        return captions

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        try:
            images = [_load_image(p) for p in batch_paths]
        except Exception as e:
            print(f"[Image Load Error] {batch_paths} -> {e}")
            continue

        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output_ids = model.generate(
                        **inputs,
                        max_length=max_length
                    )
            else:
                output_ids = model.generate(
                    **inputs,
                    max_length=max_length
                )

        captions_batch = processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )

        for path, caption in zip(batch_paths, captions_batch):
            captions[path] = caption

    return captions


def caption_image(data_path, output_path, error_files, max_length=60, batch_size=8):
    # ❗ skip file rỗng
    if os.path.getsize(data_path) == 0:
        print(f"[Empty File] {data_path}")
        error_files.append(data_path)
        return

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[JSON Error] {data_path} -> {e}")
        error_files.append(data_path)
        return

    try:
        if not os.path.exists(output_path):
            review_photo = data.get("review_photo", {})

            if review_photo:
                valid_paths = [
                    key for key, value in review_photo.items()
                    if value != False
                ]

                captions = _generate_captions(
                    image_paths=valid_paths,
                    max_length=max_length,
                    batch_size=batch_size
                )

                for key, caption in captions.items():
                    review_photo[key] = caption

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print(f"Process file {output_path}")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"[Processing Error] {data_path} -> {e}")
        error_files.append(data_path)


# ===== MAIN =====
if __name__ == "__main__":

    input_dir = "D:/MABSA/source/data/object_detected_batch"
    output_dir = "D:/MABSA/source/data/image_caption_batch"

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    error_files = []  # ✅ collect lỗi ở đây

    for file in tqdm(files, desc="Processing files"):

        inp_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)

        caption_image(
            data_path=inp_path,
            output_path=out_path,
            error_files=error_files,
            batch_size=8
        )

    # ===== REPORT =====
    print("\n========== ERROR FILES ==========")
    for f in error_files:
        print(f)

    print(f"\nTotal error files: {len(error_files)}")

    # (optional) save ra file để check sau
    with open("error_files.json", "w", encoding="utf-8") as f:
        json.dump(error_files, f, indent=2, ensure_ascii=False)