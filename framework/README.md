This folder contains multiple framework for testing. To get a framework, go to git and clone.
The directory will look like:
```
framework/
    |- VLP-MABSA/ 
    |- DTCA/
    |- ...
```
Detail of each framework is shown below:
|Framework|Link|
|-|-|
|VLP-MABSA||
|DTCA||


<!-- 
### Download BART model
BART model must be downloaded for running.
```
from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "facebook/bart-base"  # or "facebook/bart-base"
path="your_path"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# After loading or training your model
model.save_pretrained(path)
tokenizer.save_pretrained(path)
``` -->