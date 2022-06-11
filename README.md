# ModerTool

The ModerTool enables users to analyze posts from Reddit platform by using topic modeling technique(BertTopic and LDA).
This tool It allows orderly reading of the data (posts) from Reddit using Pushshift.io and Reddit API.
ModerTool allows to extract special features from post's text to use Machine Learning.
## Features
- Sentiment 
- Offensive 
- Hate
- Ner
- Emotion:  Anger ,Fear ,joy ,Sadness , Surprise and Natural

## Installation
    pip install ModerTool
    
## Embedding Models
ModerTool contains many embedding models that can be used to anylsis the documents and words from Reddit platform:

- BertTweet
- BertTopic
- LDA
- GoEmotion

## The Method
![alt text](https://imgur.com/lGeX8yj.png)


## Getting Started

| Name | Link |
| ------ | ------ |
| Create BERTopic Modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mO6Zq1Skd_CMQ5UMDFdfmjTmsVyhBv3F?usp=sharing)  |
| Read data from mongo | [plugins/dropbox/README.md][PlDb] |
| Optimization of BertTopic Model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A3_2l42Td0Hg9U8kQKbS1Z798OIvvgKu?usp=sharing)  |
| Extract Features | [plugins/googledrive/README.md][PlGd] |
| Vizualization | [plugins/onedrive/README.md][PlOd] |
