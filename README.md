# Papers
Legend: 🔄️: Redo; 🛑: Not started; ✅: Success; ❌: Failed

|Name|Paper|Code|Status|Note|Task|
|-|-|-|-|-|-|
|Few-shot Joint Multimodal Aspect-Sentiment Analysis Based on Generative Multimodal Prompt (GMP)|[Paper](https://arxiv.org/pdf/2305.10169) | [Code](https://github.com/YangXiaocui1215/GMP) |🔄️|Framework is able to run but give a low performance on Twitter (in paper, it say the method is high)|MASC, MATE, JMASA|
|Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis (VLP-MABSA)|[Paper](https://arxiv.org/pdf/2204.07955)| [Code](https://github.com/NUSTM/VLP-MABSA) |✅|
|Dual-Encoder Transformers with Cross-modal Alignment for Multimodal Aspect-based Sentiment Analysis (DCTA)|[Paper](https://aclanthology.org/2022.aacl-main.32.pdf)|[Code]( https://github.com/windforfurture/DTCA)|✅|
|Joint Multi-modal Aspect-Sentiment Analysis with Auxiliary Cross-modal Relation Detection (JML)|[Paper](https://aclanthology.org/2021.emnlp-main.360.pdf)|[Code](https://github.com/MANLP-suda/JML)|❌|Different format of raw data, and don't show how to convert to expected format
|M2DF: Multi-grained Multi-curriculum Denoising Framework for Multimodal Aspect-based Sentiment Analysis (M2DF)|[Paper](https://aclanthology.org/2023.emnlp-main.561.pdf)|[Code]( https://github.com/grandchicken/M2DF)|🔄️|Not a Framework - can be used as reference for finding new ways
|CORSA|[Paper](https://aclanthology.org/2025.coling-main.22.pdf)|[Code](https://github.com/Liuxj-Anya/CORSA)|🛑
|Detecting Aspect-oriented Information for Multimodal Aspect-Based Sentiment Analysis (AoM)|[Paper](https://aclanthology.org/2023.findings-acl.519.pdf)|[Code](https://github.com/SilyRab/AoM/)|🛑

## Framework
Create a folder `framework`, then access to git version on above table


# Datasets
## One sample look like
### Twitter Dataset
```
  {
    "words": [ "RT", "@", "FundsOverBuns", ":", "Tyga", "went", "from", "pedophile", "to", "messing", "with", "cougars", "all", "within", "a", "week"
    ],
    "image_id": "975807.jpg",
    "aspects": [{ "from": 4, "to": 5, "polarity": "NEU", "term": ["Tyga"] }],
    "opinions": [{ "term": [] }]
  },
```
Then turn into look like this
```
1	1	975807.jpg	RT @ FundsOverBuns : $T$ went from pedophile to messing with cougars all within a week	Tyga
```
or
```
RT @ FundsOverBuns : $T$ went from pedophile to messing with cougars all within a week
Tyga
0
975807.jpg
```

### Hotel Dataset
```
{
  "name": "My",
  "country": "Việt Nam",
  "room": "Bungalow Deluxe Nhìn Ra Khu Vườn",
  "date": "1 đêm · tháng 9/2023",
  "state": "Nhóm",
  "review_title": "nếu có dịp quay trở lại Ninh Bình sẽ lại ghé qua",
  "review_date": "Ngày đánh giá: ngày 5 tháng 9 năm 2023",
  "review_score": "Đạt điểm 10",
  "review_positive": "thiên nhiên, đẹp, không khí mát mẻ, trong lành, thích hợp bỏ phố bụi về nơi relax, phòng đầy đủ tiện nghi, riêng tư, anh chủ siêu nhiệt tình",
  "review_negative": null,
  "review_photo": {
 "https://q-xx.bstatic.com/xdata/images/xphoto/max1280x900/269491127.jpg?k=2336478e680599e52a1de044c47a178983ec2346d9967c47f7c2f9ff91cee09e&o=": "Vườn quanh Fairy Mountain Retreat",
 "https://r-xx.bstatic.com/xdata/images/xphoto/max1280x900/269491145.jpg?k=21499bb35d9d2ef36ce631287ec2c664997d8c65202c9718be08369c05250e1d&o=": "Ảnh của Fairy Mountain Retreat Ninh Bình được người dùng đăng tải",
 "https://q-xx.bstatic.com/xdata/images/xphoto/max1280x900/269491173.jpg?k=848cdc6fd9f94c8bbc2dd5c65bb460e97f0be6b2d8ca79069523db656cf02a66&o=": "Vườn quanh Fairy Mountain Retreat",
 "https://r-xx.bstatic.com/xdata/images/xphoto/max1280x900/269491192.jpg?k=b2c64ba1de2f1aa0c30937f8c1a15380afc5fa120eac83f72993b6a3af44c00b&o=": "Phong cảnh thiên nhiên gần resort"
  }
}
```

## Difference
- `term`: Term of Twitter Dataset are very various while Hotel Dataset may be use a pair of `(aspect_name, aspect_term)`. E.g.: `("service", "Dang Thanh An")`.
- In Hotel Dataset, `sentiment` and `aspect` will share a `term` (or `aspect_term` = `sentiment_term`). 
- Twitter Dataset has only 1 image while Hotel Dataset have multi-images or nothing. -> Need to find a way to concanate them:
  - Multiple images: 
    - Using Pooling (max/mean) to convert into 1 image
    - Treat images to be distinct sample. (text1, [img1, img2. img3]) -> (text1, img1), (text1, img2), (text1, img3)
  - No image: Add a dummy image (all black)

- Twitter Dataset has the location of aspect
