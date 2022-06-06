import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from bertopic import BERTopic
import pandas as pd
import pickle
import random
from tqdm.notebook import tqdm
import re
from langdetect import detect
import torch
print(torch.cuda.is_available())
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import en_core_web_lg
import en_core_web_sm
import unidecode
import os
import sys
import nltk
from nltk.tokenize import sent_tokenize


def batches(l, n):
    # For item i in a range that is a length of l,
    size = l.shape[-1]
    for i in range(0, size, n):
        # Create an index range for l of n items:
        if len(l.shape) == 1:
            yield l[i:i + n]

        else:
            yield l[:, i:i + n]


# Transform input tokens
def get_sentiment(model, inputs):
    outputs = model(**inputs.to(device))
    return torch.nn.functional.softmax(outputs.logits.detach())


# Transform input tokens
def get_emotions(model, inputs):
    outputs = model(**inputs.to(device))
    return torch.nn.functional.sigmoid(outputs[0].detach())


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def bert_tweet_base_model(batch_size, model_name, data):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    model = AutoModelForSequenceClassification.from_pretrained(f"cardiffnlp/bertweet-base-{model_name}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    res = []
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    for txt in tqdm(batches(data, batch_size)):
        inputs = tokenizer(list(txt), padding=True, return_tensors="pt", truncation=True)
        res.append(get_sentiment(model, inputs, device))
    res = torch.cat(res).cpu().numpy()
    res_df = pd.DataFrame(res, index=data.index)
    return res_df


def extract_spacy_ner(data):
    nlp = spacy.load("en_core_web_lg")
    spacy.prefer_gpu()
    res = []
    for doc in tqdm(nlp.pipe(list(data), disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]),
                    total=len(df)):
        res.append([(ent.text, ent.label_) for ent in doc.ents])
    # res_df = pd.DataFrame(res, index=data.index)
    return res


def extract_monologg_models(data, batch_size):
    tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
    model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    res = []
    torch.cuda.empty_cache()
    for txt in tqdm(batches(data, batch_size), total=len(df) // batch_size):
        inputs = tokenizer(list(txt), padding=True, return_tensors="pt", truncation=True)
        res.append(get_emotions(model, inputs, device))
    res = torch.cat(res).cpu().numpy()
    res_df = pd.DataFrame(res, index=data.index, columns=model.config.id2label.values())

    #     for k, v in model.config.id2label.items():
    #         res_df[v] = res[:, k]
    return res_df


# load data
df = pd.read_pickle(f'/home/{user}/BertTopic/dt-reddit/uncleaned_data/{subreddit}_{sub_kind}_{year}.pickle')
df = df.reset_index()

# split text
original_df = df.copy()
df[text] = df[text].apply(sent_tokenize)
df = df.explode(text)

df = df[df[text].notna()]
df[text] = df[text].apply(unidecode.unidecode)
df = df[df[text].notna()]

torch.cuda.empty_cache()

# emotion
emotion = extract_monologg_models(df[text], 16)
df[emotion.columns] = emotion

# sentiment
sentiment = bert_tweet_base_model(32, 'sentiment', df[text])
df[["bertweet_neg", "bertweet_neu", "bertweet_pos"]] = sentiment.loc[:, 0:3]

# hate
hate = bert_tweet_base_model(32, 'hate', df[text])
df[["not_hate", "hate"]] = hate.loc[:, 0:2]

# offensive
offensive = bert_tweet_base_model(32, 'offensive', df[text])
df[["not_offensive", "offensive"]] = hate.loc[:, 0:2]

# merge
dff_agg = df.groupby("post_id").mean()
dff_agg.reset_index(inplace=True)
df = pd.merge(original_df, dff_agg, left_on="post_id", right_on="post_id")

# ner
df["ner"] = extract_spacy_ner(df[text])





#saving
path = f"/home/{user}/BertTopic/dt-reddit/finall_features_try2/{subreddit}_{sub_kind}_{year}.csv"
df.to_csv(path)
print("ended successfully")



