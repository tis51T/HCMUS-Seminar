from pyabsa import AspectTermExtraction as ATEPC
from typing import Dict, Tuple, List
# Extract aspect terms from a single sentence
import json
from tqdm import tqdm

def extract_term_from_file(file_path: str, aspect_extractor) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pos_reviews, neg_reviews, merge_reviews = [], [], []
    for entry in data:
        # Ensure "review_positive" and "review_negative" are strings
        pos_rw = entry.get("review_positive", "") if isinstance(entry.get("review_positive", ""), str) else ""
        neg_rw = entry.get("review_negative", "") if isinstance(entry.get("review_negative", ""), str) else ""
        merge_rw = f"{pos_rw}. {neg_rw}".strip()
        
        merge_reviews.append(merge_rw)
        pos_reviews.append(pos_rw)
        neg_reviews.append(neg_rw)
        
    file_name = file_path.split("/")[-1].replace(".json", "")
    aspect_extractor.predict(merge_reviews, print_result=False)
    # rename json data
    with open("Aspect Term Extraction and Polarity Classification.FAST_LCF_ATEPC.result.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(f"../translated_with_extracted_term_data/extracted/{file_name}_merge.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Aspect term extraction results saved to {file_name}_merge.json")
        
    aspect_extractor.predict(pos_reviews, print_result=False)
    with open("Aspect Term Extraction and Polarity Classification.FAST_LCF_ATEPC.result.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(f"../translated_with_extracted_term_data/extracted/{file_name}_pos.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Aspect term extraction results saved to {file_name}_pos.json")
        
    aspect_extractor.predict(neg_reviews, print_result=False)
    with open("Aspect Term Extraction and Polarity Classification.FAST_LCF_ATEPC.result.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(f"../translated_with_extracted_term_data/extracted/{file_name}_neg.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Aspect term extraction results saved to {file_name}_neg.json")
        
    return (f"../translated_with_extracted_term_data/extracted/{file_name}_pos.json", 
              f"../translated_with_extracted_term_data/extracted/{file_name}_neg.json",
                f"../translated_with_extracted_term_data/extracted/{file_name}_merge.json")
    


def process_results(meta_path: str, extracted_path: Tuple):
    
    def process_result(result: Dict):
        new_result = {}
        new_result["words"] = result["tokens"]
        
        new_result["aspects"] = []
        for idx, aspect in enumerate(result["aspect"]):
            from_pos = result["position"][idx][0]
            to_pos = result["position"][idx][-1] + 1
            
            new_result["aspects"].append(
                {"from": from_pos, "to": to_pos, "term": aspect.split(), "polarity": result["sentiment"][idx]}
            )
            
        return new_result
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    
    pos_path, neg_path, merge_path = extracted_path
    with open(pos_path, "r", encoding="utf-8") as f:
        pos_data = json.load(f)
    with open(neg_path, "r", encoding="utf-8") as f:
        neg_data = json.load(f)
    with open(merge_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)
        
        
    for idx, data in tqdm(enumerate(meta_data)):
        pos_result, neg_result, merge_result = pos_data[idx], neg_data[idx], merged_data[idx]
        pos_processed_data, neg_processed_data, merge_processed_data = process_result(pos_result), process_result(neg_result), process_result(merge_result)
        
        pos_sentence = data.get("review_positive", "") if isinstance(data.get("review_positive", ""), str) else ""
        neg_sentence = data.get("review_negative", "") if isinstance(data.get("review_negative", ""), str) else ""
        merge_sentence = f"{pos_sentence}. {neg_sentence}".strip()
        
        pos_processed_data["sentence"] = pos_sentence
        neg_processed_data["sentence"] = neg_sentence
        merge_processed_data["sentence"] = merge_sentence
        
        data["review_positive"] = pos_processed_data
        data["review_negative"] = neg_processed_data
        data["review_merged"] = merge_processed_data
        
    with open(meta_path.replace(".json", "_with_extracted_terms.json").replace("../translated_data", "../translated_with_extracted_term_data"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)
    
    return meta_path.replace(".json", "_with_extracted_terms.json").replace("../translated_data", "../translated_with_extracted_term_data")


def main():
    import os
    
    aspect_extractor = ATEPC.AspectExtractor(
        checkpoint="english",
        auto_device=True,
    )
    
    all_meta_paths = [f"../translated_data/reviews_photo_english_{i*1000}.json" for i in range(0, 122)]
    for meta_path in all_meta_paths:
        print(f"\n\n\n===================================== Processing file: {meta_path} ================================================")
        result_path = meta_path.replace(".json", "_with_extracted_terms.json").replace("../translated_data", "../translated_with_extracted_term_data")
        if os.path.exists(result_path):
            print(f"File {result_path} already exists. Skipping...")
            continue
        else:
            
            extracted_paths = extract_term_from_file(meta_path, aspect_extractor)
            result_path = process_results(meta_path, extracted_paths)
            
if __name__ == "__main__":
    
    from pyabsa import AspectSentimentTripletExtraction as ASTE
    config = (
        ASTE.ASTEConfigManager.get_aste_config_english()
    )  # this config contains 'pretrained_bert', it is based on pretrained models
    config.model = ASTE.ASTEModelList.EMCGCN  # improved version of LCF-ATEPC
