# Papers
## Checking

|Name|Paper|Code|Status|
|-|-|-|-|
|Few-shot Joint Multimodal Aspect-Sentiment Analysis Based on Generative Multimodal Prompt (GMP)|[Paper](https://arxiv.org/pdf/2305.10169) | [Code](https://github.com/YangXiaocui1215/GMP) |✅|
|Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis (VLP-MABSA)|[Paper](https://arxiv.org/pdf/2204.07955)| [Code](https://github.com/NUSTM/VLP-MABSA) |🛑|
|Dual-Encoder Transformers with Cross-modal Alignment for Multimodal Aspect-based Sentiment Analysis (DCTA)|-|[Code]( https://github.com/windforfurture/DTCA)|▶️|
|Joint Multi-modal Aspect-Sentiment Analysis with Auxiliary Cross-modal Relation Detection (JML)|-|[Code](https://github.com/MANLP-suda/JML)|▶️|
|M2DF: Multi-grained Multi-curriculum Denoising Framework for Multimodal Aspect-based Sentiment Analysis (M2DF)|-|[Code]( https://github.com/grandchicken/M2DF)|▶️|
| | |[Code](https://github.com/pengts/DQPSA)|

## Paper Review
### I. Few-shot Joint MABSA Based on Generative Multimodal Prompt

#### Idea
Few-shot Joint MABSA based on Generative Multimodal Prompt (GMP) proposes a unified framework for Multimodal Aspect-Based Sentiment Analysis (MABSA) using generative models. It leverages both text and image modalities and prompt-based learning to jointly solve multiple tasks (aspect extraction, sentiment classification, etc.) in a few-shot setting.

#### Input
- **Text:** Social media posts (e.g., tweets)
- **Image:** Associated images with the posts
- **Prompt:** Task-specific prompts to guide the model (e.g., "Extract aspects", "Classify sentiment")

#### Output
- **Aspect Extraction:** List of aspect terms from the post
- **Sentiment Classification:** Sentiment polarity (positive, negative, neutral) for each aspect
- **Joint Output:** For each aspect, the corresponding sentiment

#### How does it work for each task?
##### 1. Aspect Extraction
- **Input:** Text + image + prompt ("Extract aspects")
- **Process:** The model encodes both modalities and generates aspect terms as output.

##### 2. Sentiment Classification
- **Input:** Text + image + prompt ("Classify sentiment for aspects")
- **Process:** The model encodes the input and generates sentiment labels for each aspect.

##### 3. Joint Aspect-Sentiment Extraction
- **Input:** Text + image + prompt ("Extract aspect-sentiment pairs")
- **Process:** The model generates aspect-sentiment pairs directly, leveraging multimodal context.

#### How does it work?
- The model uses a multimodal encoder (text + image) and a generative decoder.
- Prompts are used to specify the task (few-shot learning).
- The framework supports joint training and inference for multiple tasks, improving performance in low-resource scenarios.
