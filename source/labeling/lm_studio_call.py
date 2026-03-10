import os
from tqdm import tqdm
import json
import lmstudio as lms


class LmStudio:
    def __init__(self, model_name="gemma-3-4b"):
        # assume lms server đã start và model đã load
        self.model = lms.llm(model_name)

    def get_response(self, user_prompt, system_prompt: str = ""):
        if system_prompt:
            prompt = system_prompt + "\n" + user_prompt
        else:
            prompt = user_prompt
        response = self.model.respond(prompt)
  
        return response.content
   

    
    
def classify_aspect_categories(llm_call, input):
    # if system_prompt == "":
    system_prompt = '''
        # Role: Aspect Category Identifier 
        # Description:
        You are an expert in identifying aspect categories in hotel reviews. You will be provided a review text and a list of words extracted from the review, and then check which category each word belongs to.
        There are 6 categories to choose from:
        - "Facility": Includes factors such as facilities, room furniture, decoration, pool area, etc.
        - "Amenity": Includes public services such as parking, wifi, nearby attractions, security, etc.
        - "Service": Includes service-related factors such as staff behavior, food quality, room service, check-in/check-out process, etc.
        - "Experience": Includes overall experience factors such as value for money, comfort, ambiance, etc.
        - "Branding": Overall satisfaction of customer compared to brand expectations.
        - "Loyalty": Customer's willingness to return or recommend the hotel.
        
        # Contraints:
        - Your output will be in a JSON with key is each word from input list, and value is the corresponding aspect category.
        - Only return a JSON, no need any other explanations or comments.
        
        # Output:
        A JSON file
        
        # Input:
        
        '''
    response = llm_call.get_response(user_prompt = json.dumps(input), system_prompt=system_prompt)
    response = response.replace("", "").replace("json", "")
    
        # Convert the response string to a JSON object
    try:
        response_json = json.loads(response)
        return response_json
    except json.JSONDecodeError:
        return response
    
    
    
def _process_input_for_extract_category(sample):

    result = sample["triplet_extraction"]
    out = []
    for key, value in result.items():
        list_word = [entry["Aspect"] for entry in value]
        
        out.append({
            "sentence": key,
            "list": list_word if list_word else []
        })
    # print(out)  
    return out

def _process_output_for_extract_category(aspect_category_with_sentence, sample):

    result = sample["triplet_extraction"]
    for key, value in result.items():
        aspect_category = aspect_category_with_sentence.get(key, {})
        for entry in value:
            aspect = entry["Aspect"]
            try:
                entry["Category"] = aspect_category[aspect]
            except:
                entry["Category"] = "Unknown"
            
    sample["four_extraction"] = result
    return sample
    
    
    
def extract_aspect_categories(llm_call, inp_path: str):
    with open(inp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Configure output folder    
    out_folder_path = os.path.dirname(inp_path).replace("triplet_batch", "four_batch")
    os.makedirs(out_folder_path, exist_ok=True)
    output_path = os.path.join(out_folder_path, os.path.basename(inp_path))
    if not os.path.exists(output_path):
        # Extract aspect categories
        inputs = _process_input_for_extract_category(data)
        aspect_category_with_sentence = {}
        for inp in inputs:
            response = classify_aspect_categories(llm_call, inp["list"])
            aspect_category_with_sentence[inp["sentence"]] = response
            
        new_entry = _process_output_for_extract_category(aspect_category_with_sentence, data)
        del new_entry["triplet_extraction"]


        with open(os.path.join(out_folder_path, os.path.basename(inp_path)), "w", encoding="utf-8") as f:
            json.dump(new_entry, f, indent=4, ensure_ascii=False)
        print(f"\nProcessed file {inp_path} and saved output to {output_path}")
        
        
if __name__ == "__main__":
    lmstudio_call = LmStudio(model_name="google/gemma-3-4b")
    # with open("data/hotel_review_10k/new_hotel_review_10k_triplet_extracted_full_processed.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)
    
    files = os.listdir("D:\\MABSA\\source\\data\\triplet_batch")
    
    files = sorted(files, reverse=True)
    for file in tqdm(files):
        inp_path = os.path.join("D:\\MABSA\\source\\data\\triplet_batch", file)
        extract_aspect_categories(lmstudio_call, inp_path)
    
    # extract_aspect_categories(lmstudio_call, "D:\\MABSA\\source\\data\\triplet_batch\\8.json")