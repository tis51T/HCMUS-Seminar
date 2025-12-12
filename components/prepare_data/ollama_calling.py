from tqdm import tqdm
import ollama
import json

class OllamaLLM:
    def __init__(self, model_name="gemma3:4b", host="http://127.0.0.1:3128"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=self.host)
        
    def get_response(self, user_prompt, system_prompt: str = ""):
        # Combine system and user prompts into a single string
        big_prompt = system_prompt + "\n" + user_prompt
        
        result = self.client.generate(
            model=self.model_name, prompt=big_prompt  # Pass as a string
        )
        
        # Replace direct key access with .get to avoid KeyError
        return result.get("response", "No response available")
    
    
def translate_text(ollama_call, text, system_prompt = ""):
    system_prompt = '''
    Expecting you are a professional translator, your task is translating the following text to English. Your response should match these requirements:
    - Make sure the translation is accurate and preserves the original meaning.
    - Make sure name of people, places, and organizations should be unchanged except name of countries.
    - In some cases, the text have already been in English, so just return the original text.
    - Your response should be in English.
    - Your response only contains the translated text, without any additional explanations or comments.    
    '''
    response = ollama_call.get_response(text, system_prompt=system_prompt)
    return response
    
def translate_dataset():
    ollama_call = OllamaLLM(host="http://localhost:11434", model_name="gemma3:4b")
    
    with open("./reviews_photo.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    
    batch = 1000
    start = 76000
    end = start + batch*50
    print("Total batchs:", len(data)//batch + (1 if len(data) % batch != 0 else 0))
    print("Number of batch:", (end-start)//batch)
    for i in range(start, end, batch):
        batch_data = data[i:i+batch]
        print(f"Processing batch {i//batch}:")
        # Process batch_data as needed
    
        new_data = []
        
        for item in tqdm(batch_data, desc="Translating reviews...", unit="review"):
            new_item = item.copy()

            
            if item.get("review_title", None) is not None:
                new_item["review_title"] = translate_text(ollama_call, item.get("review_title"))
            else:
                new_item["review_title"] = None

            if item.get("review_score", None) is not None:
                new_item["review_score"] = translate_text(ollama_call, item.get("review_score"))
            else:
                new_item["review_score"] = None

            if item.get("review_positive", None) is not None:
                new_item["review_positive"] = translate_text(ollama_call, item.get("review_positive"))
            else:
                new_item["review_positive"] = None

            if item.get("review_negative", None) is not None:
                new_item["review_negative"] = translate_text(ollama_call, item.get("review_negative"))
            else:
                new_item["review_negative"] = None

            new_item["review_photo"] = item.get("review_photo", None)
            for key, value in new_item["review_photo"].items():
                if value is not None:
                    new_item["review_photo"][key] = translate_text(ollama_call, value)
                else:
                    new_item["review_photo"][key] = None
            
            new_data.append(new_item)
            
        with open(f"./translated_data/reviews_photo_english_{i}.json", "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        print(f"Translation completed and saved to review_photo_english_{i}.json")   
    
def exctract_aspect_categories(ollama_call, input, system_prompt=""):
    # if system_prompt == "":
    user_prompt = '''
        You are an expert in identifying aspect categories in hotel reviews. 
        You will be provided a review text and a list of words extracted from the review, and then check which category each word belongs to.
        There are 6 categories to choose from:
        - "Facility": Includes factors such as facilities, room furniture, decoration, pool area, etc.
        - "Amenity": Includes public services such as parking, wifi, nearby attractions, security, etc.
        - "Service": Includes service-related factors such as staff behavior, food quality, room service, check-in/check-out process, etc.
        - "Experience": Includes overall experience factors such as value for money, comfort, ambiance, etc.
        - "Branding": Overall satisfaction of customer compared to brand expectations.
        - "Loyalty": Customer's willingness to return or recommend the hotel.
        
        
        Please follow these guidelines:
        - Your output will be in a JSON with key is each word from input list, and value is the corresponding aspect category.
        - Only return a JSON, no need any other explanations or comments.
        
        Review: 
        '''
        
    response = ollama_call.get_response(user_prompt + json.dumps(input), system_prompt=system_prompt)
    response = response.replace("`", "").replace("json", "")
    
        # Convert the response string to a JSON object
    try:
        response_json = json.loads(response)
        return response_json
    except json.JSONDecodeError:
        return response
    
def process_input_for_extract_category(sample):
    result = sample["result"]
    out = []
    for key, value in result.items():
        list_word = [entry["Aspect"] for entry in value]
        # list_word = []
        # for entry in value:
        #     try:
        #         list_word.append(entry["Aspect"])
        #     except:
        #         pass
        
        out.append({
            "sentence": key,
            "list": list_word if list_word else []
        })
        
    return out

def process_output_for_extract_category(aspect_category_with_sentence, sample):
    result = sample["result"]
    for key, value in result.items():
        aspect_category = aspect_category_with_sentence.get(key, {})
        for entry in value:
            # print(entry)
            aspect = entry["Aspect"]
            try:
                entry["Category"] = aspect_category[aspect]
            except:
                entry["Category"] = "Unknown"
            
    sample["result"] = result
    return sample
    
    
if __name__ == "__main__":
    ollama_call = OllamaLLM(host="http://localhost:11434", model_name="gemma3:4b")
    with open("data/hotel_review_10k/new_hotel_review_10k_triplet_extracted_full_processed.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    
    batch = 100
    idx = 0
    outdata = []
    start = 102
    end = 3
    
    for i, entry in enumerate(tqdm(data[:end*batch], desc="Processing dataset...", unit="entry")):
        inputs = process_input_for_extract_category(entry)
        aspect_category_with_sentence = {}
        for inp in inputs:
            response = exctract_aspect_categories(ollama_call, inp)
            aspect_category_with_sentence[inp["sentence"]] = response
            
            
        new_entry = process_output_for_extract_category(aspect_category_with_sentence, entry)
        outdata.append(new_entry)
        
        if len(outdata) >= batch:
            with open(f"data/hotel_review_10k_four/hotel_review_10k_four_components_{idx}.json", "w", encoding="utf-8") as f:
                json.dump(outdata, f, indent=4, ensure_ascii=False)
            print(f"\nSaved batch {idx} with {len(outdata)} entries.")
            idx += 1
            outdata = []
    
    
    if outdata:
        with open(f"data/hotel_review_10k_four/hotel_review_10k_four_components_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(outdata, f, indent=4, ensure_ascii=False)
        print(f"\nSaved final batch {idx} with {len(outdata)} entries.")
    
