"""
BLIP Image Captioning Script

This script uses Salesforce/blip-image-captioning-base to generate captions for:
1. Original images
2. Sub-images cropped from detected object bounding boxes

Input: JSON file with image URLs and detected objects with bounding boxes
Output: Enhanced JSON with captions for original and sub-images
"""

import json
import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from io import BytesIO
import os
from typing import List, Dict, Any
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BLIPCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", use_fast: bool = False):
        """
        Initialize BLIP model and processor
        
        Args:
            model_name: HuggingFace model name for BLIP
            use_fast: whether to load a fast processor/tokenizer when available. If False, a "slow" processor
                      will be used (keeps previous behaviour). In Transformers v4.52 the default may change
                      to use_fast=True which can cause minor differences in outputs.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Pass use_fast explicitly so behavior is reproducible regardless of HF default changes
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=use_fast)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    
    def download_image(self, url: str) -> Image.Image:
        """
        Download image from URL
        
        Args:
            url: Image URL
            
        Returns:
            PIL Image object
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:

            raise
    
    def crop_image(self, image: Image.Image, bbox: List[int]) -> Image.Image:
        """
        Crop image based on bounding box
        
        Args:
            image: PIL Image object
            bbox: Bounding box as [x1, y1, x2, y2]
            
        Returns:
            Cropped PIL Image object
        """
        x1, y1, x2, y2 = bbox
        # Ensure coordinates are within image bounds
        width, height = image.size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        
        if x2 <= x1 or y2 <= y1:

            return None
            
        return image.crop((x1, y1, x2, y2))
    
    def generate_caption(self, image: Image.Image, max_length: int = 50) -> str:
        """
        Generate caption for an image
        
        Args:
            image: PIL Image object
            max_length: Maximum length of generated caption
            
        Returns:
            Generated caption string
        """
        try:
            # Process image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=max_length, num_beams=5)
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:

            return "Caption generation failed"
    
    def process_image_data(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single image data entry
        
        Args:
            image_data: Dictionary containing image info and detected objects
            
        Returns:
            Enhanced dictionary with captions
        """
        result = image_data.copy()
        
        try:
            # Download original image
            original_image = self.download_image(image_data['photo_url'])
            
            # Generate caption for original image
            original_caption = self.generate_caption(original_image)
            result['original_caption'] = original_caption
            
            # Process detected objects
            if 'detected_objects' in image_data:
                result['detected_objects'] = []
                
                for i, obj in enumerate(image_data['detected_objects']):
                    obj_result = obj.copy()
                    
                    # Crop sub-image based on bounding box
                    bbox = obj['bbox']
                    cropped_image = self.crop_image(original_image, bbox)
                    
                    if cropped_image is not None:
                        # Generate caption for cropped image

                        sub_caption = self.generate_caption(cropped_image)
                        obj_result['sub_image_caption'] = sub_caption
                    else:
                        obj_result['sub_image_caption'] = "Invalid bounding box"
                    
                    result['detected_objects'].append(obj_result)
            
        except Exception as e:

            result['error'] = str(e)
        
        return result
    
    def process_json_file(self, input_file: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        Process JSON file with image data
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file (optional)
            
        Returns:
            List of processed image data dictionaries
        """

        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        results = []
        total_images = len(data)
        
        count = 0
        batch_id = 0
        batch_size = 100
        for i, image_data in enumerate(tqdm(data, desc="Images", unit="img", total=total_images), 1):

            result = self.process_image_data(image_data)
            results.append(result)
            count += 1
            if count % batch_size == 0:
                out_path = f"./model_call/obj_capt_batch_{batch_id}.json"

                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                batch_id += 1
        
        # Save results if output file specified
        out_path = f"./model_call/obj_capt_batch_{batch_id}.json"
        # logger.info(f"Saving results to: {out_path}")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results


def main():
    """Main function to run the captioning script"""
    
    # Initialize captioner
    # To preserve the slow-processor behavior use use_fast=False. To opt in to faster processors (may
    # change outputs slightly), set use_fast=True.
    captioner = BLIPCaptioner(use_fast=False, model_name="Salesforce/blip-image-captioning-large")
    
    # Example usage
    input_file = "./model_call/images_dataset.json"
    output_file = "./model_call/example_with_captions.json"
    
    if os.path.exists(input_file):
        try:
            results = captioner.process_json_file(input_file, output_file)
        except:
            logger.error("balba")
    else:
        logger.error(f"Input file not found: {input_file}")
        print("Please make sure 'example.json' exists in the same directory.")


if __name__ == "__main__":
    main()