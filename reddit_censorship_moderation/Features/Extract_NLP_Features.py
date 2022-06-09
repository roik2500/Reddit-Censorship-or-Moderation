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

# print(torch.cuda.is_available())
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import en_core_web_lg
import en_core_web_sm
import unidecode
import os
import sys
import nltk
from nltk.tokenize import sent_tokenize


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


class Features:

    def __init__(self, data):
        self.data = data  # should be dataframe
        self.data[self.text].apply(lambda x: [x for x in self.split_rows(x)])
        self.data.explode(self.text)
        self.text = 'title_selftext'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def get_sentiment(self, model, inputs):
        outputs = model(**inputs.to(self.device))
        return torch.nn.functional.softmax(outputs.logits.detach())

    # Transform input tokens
    def get_emotions(self, model, inputs):
        outputs = model(**inputs.to(self.device))
        return torch.nn.functional.sigmoid(outputs[0].detach())

    # Transform input tokens
    def batches(self, l, n):
        # For item i in a range that is a length of l,
        size = l.shape[-1]
        for i in range(0, size, n):
            # Create an index range for l of n items:
            if len(l.shape) == 1:
                yield l[i:i + n]
            else:
                yield l[:, i:i + n]

    def bert_tweet_base_model(self, batch_size, model_name):
        '''
        :param batch_size: usually 32
        :param model_name: sentiment offensive or hate
        :param data: self.data[self.text]
        :return:
        '''
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
        model = AutoModelForSequenceClassification.from_pretrained(f"cardiffnlp/bertweet-base-{model_name}")
        model = model.to(self.device)
        res = []
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        for txt in tqdm(self.batches(self.data, batch_size)):
            inputs = tokenizer(list(txt), padding=True, return_tensors="pt", truncation=True)
            res.append(self.get_sentiment(model, inputs, self.device))
        res = torch.cat(res).cpu().numpy()
        res_df = pd.DataFrame(res, index=self.data.index)
        return res_df

    def extract_spacy_ner(self):
        nlp = spacy.load("en_core_web_lg")
        spacy.prefer_gpu()
        res = []
        for doc in tqdm(nlp.pipe(list(self.data), disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]),
                        total=len(self.data)):
            res.append([(ent.text, ent.label_) for ent in doc.ents])
        res_df = pd.DataFrame(res, index=self.data.index)
        return res_df


    # Emotion
    def extract_monologg_models(self, batch_size):
        tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
        model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
        model = model.to(self.device)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        res = []
        torch.cuda.empty_cache()
        for txt in tqdm(self.batches(self.data, batch_size), total=len(self.data // batch_size)):
            res.append(self.get_emotions(model, tokenizer, list(txt)))
        emo = torch.cat(res).cpu().numpy()
        res_df = pd.DataFrame(res, index=self.data.index, columns=model.config.id2label.values())

        for k, v in model.config.id2label.items():
            res_df[v] = emo[:, k]
        return res_df

    def split_rows(self, s):
        start = 0
        end = 0
        sen = nltk.sent_tokenize(s)
        for s in sen:
            j_s = ''
            if len(s) < 128:
                yield s
            else:
                split_sen = s.split()
                for ss in split_sen:
                    if len(ss) > 10:
                        ss = ''
                    tmp = ' '.join([j_s, ss])
                    if len(tmp) < 128:
                        j_s = tmp
                    elif len(j_s) < 128:
                        yield j_s
                        j_s = ss
        if j_s != '':
            yield j_s

    def get_features(self, lst_feature):
        '''
        This function will extract features by user decision
        :param lst_feature: sentiment offensive or hate, ner
        :return:
        '''

        original_df = self.data.copy()
        for feature in lst_feature:
            if feature == 'sentiment':
                sentiment = self.bert_tweet_base_model(32, 'sentiment')
                self.data[["bertweet_neg", "bertweet_neu", "bertweet_pos"]] = sentiment.loc[:, 0:3]

            elif feature == 'hate':
                hate = self.bert_tweet_base_model(32, 'hate')
                self.data[["not_hate", "hate"]] = hate.loc[:, 0:2]

            elif feature == 'offensive':
                offensive = self.bert_tweet_base_model(32, 'offensive')
                self.data[["not_offensive", "offensive"]] = offensive.loc[:, 0:2]

        # merge
        dff_agg = self.data.groupby("post_id").mean()
        dff_agg.reset_index(inplace=True)
        self.data = pd.merge(original_df, dff_agg, left_on="post_id", right_on="post_id")

        if 'ner' in lst_feature:
            self.data["ner"] = self.extract_spacy_ner()

    # split text
    def split_text(self):
        original_df = self.data.copy()
        self.data[self.text] = self.data[self.text].apply(sent_tokenize)
        df = self.data.explode(self.text)

    def get_data(self):
        return self.data

    def save_data(self, path):
        # path = f"/home/{user}/BertTopic/dt-reddit/finall_features_try2/{subreddit}_{sub_kind}_{year}.csv"
        self.data.to_csv(path)

    # df = df[df[text].notna()]
    # df[text] = df[text].apply(unidecode.unidecode)
    # df = df[df[text].notna()]

    # torch.cuda.empty_cache()
