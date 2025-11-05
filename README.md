## Instrucstion
Install pretrain model

```
import shutil
from huggingface_hub import hf_hub_download

# Replace 'repo_id' with the model's repository ID (e.g., "facebook/bart-large")
repo_id = "facebook/bart-base"
filename = "pytorch_model.bin"

# Download the file
file_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Define the target folder
target_folder = "./checkpoint"
target_path = f"{target_folder}/{filename}"

# Move the file to the checkpoint folder
shutil.copy(file_path, target_path)

print(f"Model downloaded to: {target_path}")

```