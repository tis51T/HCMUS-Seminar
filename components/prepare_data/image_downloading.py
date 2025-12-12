"""Utilities to download and crop images used by the data preparation step.

Functions:
  - download_and_crop(photo_url, bbox, save_path, timeout=10)
      Download an image from `photo_url`, optionally crop by `bbox` (x1,y1,x2,y2),
      and save to `save_path` (creating parent directories if needed).

  - download_sample_entry(entry, output_dir)
      Convenience wrapper that takes a sample dictionary (like those in `samples`)
      and saves the (cropped) image to `output_dir` using a generated filename.

Requires: requests, Pillow
"""

from io import BytesIO
import os
from typing import Optional, Sequence

import requests
from PIL import Image

from tqdm import tqdm

def download_and_crop(photo_url: str, bbox: Optional[Sequence], save_path: str, timeout: int = 10) -> str:
    """Download image from `photo_url`, crop by `bbox` and save to `save_path`.

    Args:
        photo_url: URL of the image to download.
        bbox: Sequence of four ints (x1, y1, x2, y2). If None or contains nulls, the
              full image is saved.
        save_path: Path to save the resulting image. Parent directories will be created.
        timeout: Request timeout in seconds.

    Returns:
        The absolute path to the saved image.

    Raises:
        requests.RequestException on download errors.
        OSError / PIL.UnidentifiedImageError on image/crop/save errors.
    """
    if not photo_url:
        raise ValueError("photo_url must be provided")

    resp = requests.get(photo_url, stream=True, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")

    # Determine whether bbox is usable (all four values present and not None)
    use_bbox = False
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            # allow numeric-like strings or floats, coerce to int
            coords = [None if v is None else int(v) for v in bbox]
            if all(v is not None for v in coords):
                use_bbox = True
            else:
                coords = None
        except Exception:
            coords = None
    else:
        coords = None

    # Prepare save directory
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if use_bbox and coords:
        # Clamp bbox to image bounds
        w, h = img.size
        x1 = max(0, min(coords[0], w))
        y1 = max(0, min(coords[1], h))
        x2 = max(0, min(coords[2], w))
        y2 = max(0, min(coords[3], h))
        # Ensure valid box
        if x2 > x1 and y2 > y1:
            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(save_path, format="JPEG", quality=95)
        else:
            # fallback to saving full image
            img.save(save_path, format="JPEG", quality=95)
    else:
        # Save full image
        img.save(save_path, format="JPEG", quality=95)

    return os.path.abspath(save_path)


def download_sample_entry(entry: dict, output_dir: str) -> Optional[str]:
    """Save image for a single sample entry.

    The filename is generated from `text_id` and `image_id` fields if present.
    Returns the saved file path on success, or None on failure.
    """
    url = entry.get("photo_url")
    bbox = entry.get("bbox")
    text_id = entry.get("text_id", "no_text")
    image_id = entry.get("image_id", "no_image")
    filename = f"{image_id}.jpg"
    save_path = os.path.join(output_dir, filename)
    try:
        path = download_and_crop(url, bbox, save_path)
        # print(f"Saved image to: {path}")
        return path
    except Exception as e:
        print(f"Failed to save image for {text_id}/{image_id}: {e}")
        return None


if __name__ == "__main__":
    # Example usage: download sample entries to ./data/downloaded_images
    input_dir = "./data/text_image"
    for set_type in ["train", "test", "dev"]:
        import json
        set_path = os.path.join(input_dir, f"{set_type}_dataset.json")
        output_dir = os.path.join("./data/downloaded_images", set_type)
        os.makedirs(output_dir, exist_ok=True)
        with open(set_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for entry in tqdm(dataset):
            download_sample_entry(entry, output_dir)
