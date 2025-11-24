from pyabsa import AspectSentimentTripletExtraction as ASTE
import json
from tqdm import tqdm
import time

import os
import sys
import contextlib



# find a suitable checkpoint and use the name:

triplet_extractor = ASTE.AspectSentimentTripletExtractor(
    checkpoint="english"
)  # here I use the english checkpoint which is trained on all English datasets in PyABSA


def extract_triplets(extractor):
    triplet_extractor = extractor

    # Load data
    with open("../../data/hotel_review_10k_only_merge.json", "r") as f:
        data = json.load(f)

    # batch_i = 0
    batch_size = 100
    batch_save = []

    idx = 0
    for i, entry in tqdm(enumerate(data), total=len(data)):
        # check file exists
        if os.path.exists(f"../../data/hotel_review_10k_triplet_extracted_batch_{idx}.json"):
            idx += 1
            continue

        
        review = entry["review_merged"]
        sentence_split = [s for s in review.split(".")if len(s.strip()) > 5] 
        result = []
        for s in sentence_split:
            s = s.strip()
            result.append({s: triplet_extractor.predict(s, print_result=False)["Triplets"]})   
        
        entry["result"] = result
        batch_save.append(entry)

        if batch_save and len(batch_save) >= batch_size:
            with open(
                f"../../data/hotel_review_10k_triplet_extracted_batch_{idx}.json",
                "w",
            ) as f:
                json.dump(batch_save, f, indent=4)
            batch_save = []
            idx+=1

        # time.sleep(0.01)  # to prevent potential overload

    with open(
        f"../../data/hotel_review_10k_triplet_extracted_batch_{idx}.json",
        "w",
        ) as f:
        json.dump(batch_save, f, indent=4)
        batch_save = []


def re_extract_triplets(extractor, retry=3,
                        data_path="../../data/hotel_review_10k_triplet_extracted_full.json"):
    triplet_extractor = extractor

    # Load data
    with open(data_path, "r") as f:
        data = json.load(f)

    for i, entry in tqdm(enumerate(data), total=len(data)):
        results = entry.get("result", None)
        if results:
            for res in results:
                for key, value in res.items():
                    if value == []:
                        for r in range(retry):
                            s = key.strip()
                            new_res = triplet_extractor.predict(s, print_result=False)["Triplets"]
                            if new_res:
                                results[key] = new_res
                                break

    with open(
        data_path,
        "w",
        ) as f:
        json.dump(data, f, indent=4)

def check_missing_triplets():
    # Load data
    with open("../../data/hotel_review_10k_triplet_extracted_full.json", "r") as f:
        data = json.load(f)

    missing_count = 0
    missing_triplet_samples = []
    for i, entry in tqdm(enumerate(data), total=len(data)):
        results = entry.get("result", None)
        if results:
            for res in results:
                for key, value in res.items():
                    word_count = len(key.split(" "))
                    if value == [] and word_count < 2:
                        missing_count += 1
                        missing_triplet_samples.append(entry)
                        
                        # print(f"Missing triplet in review {i}: '{entry}'")
                        break
    print(f"Total missing triplets: {missing_count}")

    with open(
        f"../../data/hotel_review_10k_missing_triplet_samples.json",
        "w",
        ) as f:
        json.dump(missing_triplet_samples, f, indent=4)



if __name__ == "__main__":
    # check_missing_triplets()
    re_extract_triplets(triplet_extractor, retry=3, data_path="../../data/hotel_data/hotel_review_10k_missing_triplet_samples.json")