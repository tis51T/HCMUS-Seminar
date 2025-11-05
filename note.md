# Note
## One sample look like
### Twitter Dataset
```
{
    "words": [      "And",      "this",      "is",      "proof",      "that",      "the",      "US",      "is",      "behind",
      "in",      "fashion",      ",",      "I",      "bought",      "this",      "sweater",      "in",      "Europe",      "last",
      "winter",      "(",      "except",      "in",      "grey",      ")"
    ],
    "image_id": "15649.jpg",
    "aspects": [
      { "from": 6, "to": 7, "polarity": "NEG", "term": ["US"] },
      { "from": 17, "to": 18, "polarity": "POS", "term": ["Europe"] }
    ],
    "opinions": [{ "term": [] }],
    "caption": "im not a fan of the black t  shirt but i love the black",
    "image_path": "./data/twitter2015_images/15649.jpg",
    "aspects_num": 2
}
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
