import os
import json
import os
import json

def remove_file_by_meta_string(folder_path, save_path="removed_files.json"):
    
    files = os.listdir(folder_path)
    removed_files = []

    for f in files:
        file_path = os.path.join(folder_path, f)

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        review_text = data.get("review", "").lower()

        if "translation" in review_text or "translator" in review_text:
            removed_files.append(f)
            os.remove(file_path)

    # lưu danh sách file bị xóa
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(removed_files, f, indent=4, ensure_ascii=False)

    return removed_files


import os
import json

def remove_empty_four_extraction_elements(folder_path):

    files = os.listdir(folder_path)

    for f in files:
        file_path = os.path.join(folder_path, f)

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if "four_extraction" in data and isinstance(data["four_extraction"], dict):

            fe = data["four_extraction"]

            # giữ lại những key có list không rỗng
            fe = {k: v for k, v in fe.items() if isinstance(v, list) and len(v) > 0}

            data["four_extraction"] = fe

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
    
    
    
    

def main():

    for set_type in ["train", "test", "val"]:


        folder_path = f"D:/MABSA/source/data/{set_type}"
        save_path = f"{set_type}_removed_files.json"

        print("Removing files containing translation meta...")
        removed = remove_file_by_meta_string(folder_path, save_path=    save_path)
        print(f"Removed {len(removed)} files")

        print("Cleaning empty four_extraction elements...")
        remove_empty_four_extraction_elements(folder_path)

        print("Done.")


if __name__ == "__main__":
    main()