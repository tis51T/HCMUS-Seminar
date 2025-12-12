# %%
import pandas as pd
import matplotlib.pyplot as plt
# !pip install seaborn
import seaborn as sns

path = "../hotel_data/translated_with_extracted_term_data/reviews_photo_english_with_extracted_terms.json"

# %%
df = pd.read_json(path)
df

# %% [markdown]
# # Extract some features

# %%
from typing import Dict
def count_words(review: Dict) -> int:
    words = review["words"]
    count = 0
    for w in words:
        if len(w) >= 2:
            count += 1
    return count

def count_photos(review_photos: Dict) -> int:
    return len(review_photos.keys())

# %%
df["num_of_words"] = df["review_merged"].apply(count_words)
df["num_of_images"] = df["review_photo"].apply(count_photos)

# %% [markdown]
# # Check word count (pos + neg)

# %%
# Filter using IQR
Q1 = df["num_of_words"].quantile(0.25)
Q3 = df["num_of_words"].quantile(0.75)
IQR = Q3 - Q1
filtered_df = df[(df["num_of_words"] >= Q1 - 1.5 * IQR) & (df["num_of_words"] <= Q3 + 1.5 * IQR)] # shape is (114645, 13)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df["num_of_words"], bins=50, kde=False)
plt.title("Before IQR Filtering")
plt.xlabel("num_of_words")
plt.ylabel("Count")

plt.subplot(1,2,2)
sns.histplot(filtered_df["num_of_words"], bins=50, kde=False)
plt.title("After IQR Filtering")
plt.xlabel("num_of_words")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Filter sample which is more than 5 words count

# %%
filtered_df = filtered_df[filtered_df["num_of_words"] >=6]
sns.histplot(filtered_df["num_of_words"], bins=50, kde=False)
plt.title("After IQR Filtering")
plt.xlabel("num_of_words")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# %% [markdown]
# # Count images

# %%
filtered_df["num_of_images"].value_counts()

# %%
# take only samples with one images first
filtered_df = filtered_df[filtered_df["num_of_images"] == 1]
filtered_df

# %%
exported_df = filtered_df.sample(frac=0.25).reset_index(drop=True)
with open("hotel_review_10k.json", "w", encoding="utf-8") as f:
    exported_df.to_json(f, orient="records")

# %%



