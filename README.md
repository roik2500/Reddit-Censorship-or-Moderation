##                                Reddit-Censorship-or-Moderation
In the project, we used the following methods of characterizing moderation, which combines a number of different NLP techniques for text analysis:
(1) BertTopic - A tool that creates Topic models for each subreddit segmented into different topics.
(2) Bertweet - The language model, trained over millions of tweets, which is able to assess hate, abuse, and sentiment (positive, negative, or neutral) within the text.
(3) Go Emotion - A model trained on 58,000 comments from Reddit allows extraction of Ekman's 2 emotions.
(4) spaCy NER - A tool for recognition of entities from text.

    
## The Method
![alt text](https://imgur.com/lGeX8yj.png)


## Getting Started

| Name | Link |
| ------ | ------ |
| Create BERTopic Modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mO6Zq1Skd_CMQ5UMDFdfmjTmsVyhBv3F?usp=sharing)  |
| Read data from mongo | [plugins/dropbox/README.md][PlDb] |
| Optimization of BertTopic Model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A3_2l42Td0Hg9U8kQKbS1Z798OIvvgKu?usp=sharing)  |
| Extract Features | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PqCK3jLP6yLSXqBHv-MCjXK-V8S1IANz)  |
| Vizualization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a-000bp33oCqFUbWV65W4llKqAPTnu7N?usp=sharing)  |



Optimization of BertTopic
```python
torch.cuda.empty_cache()
from collections import Counter
index_save=0
for n_neighbor, min_topic_size  in tqdm(product(n_neighbors, min_topic_sizes), total=36):
    #if n_neighbors==20  and min_topic_size==50:continue
   
    model = get_topic_model(n_neighbor, min_topic_size)
    model.save(#you path)
    print("finish model")

    topics, probas = model.fit_transform(df[text]) 
    df['Topic']=topics
    print("finish transform")

    #Saving a data for this divition of topics
    path_to_save= # your path
    df.to_pickle(path_to_save,protocol=4)

    #Calaulate the amount of each topic
    get_topic=model.get_topic_info()
    c = Counter(topics)
    soret_c=sorted(c.items(),key=lambda x:x[0])
    count=[i[1] for i in soret_c]

    #update the dataframe with the ammount of each topic after transform
    get_topic['Count']=count

    sum_ = sum(count)
    get_topic['percentage']=get_topic['Count'].apply(lambda x: str(round((x/sum_)*100,2))+'%')

    print("start coh")
    coh = get_coherence(df, model)

    #Adding to the dataframe
    new_record=[str(n_neighbor),str(min_topic_size),str(len(get_topic)),coh['c_npmi'],coh['c_uci'],coh['u_mass'],coh['c_v'],str(count[0]),get_topic['percentage']   [0],str(sum_)]
    res_tabel.loc[len(res_tabel.index)]=new_record


    #save in any 2 iteration
    if index_save % 2==0:
        print("save")
        res_tabel.to_csv("/home/roikreme/BertTopic/{}/random optimization/final_tabel_df.csv".format(subreddit),index=False)
    index_save+=1
   
```

```python
>>> res_tabel

Number Of Negihbor	Min Topic Size   Num of Topic	        c_npmi          c_uci           u_mass          c_v
20                      50	                 189	        0.021770	-2.307029	-0.989312	0.430001	
20  	                100	                 69	        0.042159	-1.939629	-0.646382	0.510711	
20	                150	                 49	        0.031635	-2.048354	-0.453446	0.535289	
20	                200	                 37	        0.065220	-1.110054	-0.288773	0.635015
15	                50	                 182	        0.019615	-2.310290	-0.996556	0.431170	
15	                100	                 86	        0.018527	-2.374765	-0.718971	0.471135
```
## Features Extraction
- Emotion:  Anger ,Fear ,joy ,Sadness , Surprise and Natural
- Sentiment
- Offensive 
- Hate
- Ner


1. GoEmotion 
```python
from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

res = []
for txt in tqdm(batches(df[text],32), total=len(df)//32):   
    res.append(get_emotions(model,tokenizer, list(txt)))
emo = torch.cat(res).cpu().numpy()
for k, v in model.config.id2label.items():
    df[v] = emo[:,k]

```

2.BertTweet - sentiment
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/bertweet-base-sentiment")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

res = []
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
for txt in tqdm(batches(df[text],32)):
    res.append(get_sentiment(model,tokenizer, list(txt)))
sent = torch.cat(res).cpu().numpy()
df["bertweet_neg"] = sent[:,0]
df["bertweet_neu"] = sent[:,1]
df["bertweet_pos"] = sent[:,2]
```

3.Hate
```python
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/bertweet-base-hate")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
res = []
for txt in tqdm(batches(df[text],32), total=len(df)//32):
    res.append(get_sentiment(model,tokenizer, list(txt)))
hate = torch.cat(res).cpu().numpy()
df["not_hate"] = hate[:,0]
df["hate"] = hate[:,1]
```
4.Offensive
```python
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/bertweet-base-offensive")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
res = []
for txt in tqdm(batches(df[text],32), total=len(df)//32):
    res.append(get_sentiment(model,tokenizer, list(txt)))
offensive = torch.cat(res).cpu().numpy()
df["not_offensive"] = offensive[:,0]
df["offensive"] = offensive[:,1]
```
5.SPACY - NER
```python
import spacy
res = []
for doc in tqdm(nlp.pipe(list(df[text]), disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]), total=len(df)):
    res.append([(ent.text, ent.label_) for ent in doc.ents])
df["ner"] = res
```


## Vizualization

![alt text](https://i.postimg.cc/P5DDTjhM/Whats-App-Image-2022-06-27-at-18-58-35.jpg)

![Example of Features analysis using Altair](https://user-images.githubusercontent.com/74155541/176166137-4b042f77-bb50-43a8-9bd6-f2ae70672f25.gif)


