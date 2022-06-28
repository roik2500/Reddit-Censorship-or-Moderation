from pathlib import Path
import time
import os
from Optimization.Optimization import Optimization
from sklearn.cluster import KMeans

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch

torch.cuda.is_available()
import pandas as pd
import pickle
import numpy as np
from itertools import product
from bertopic import BERTopic
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from nltk.stem import WordNetLemmatizer
from collections import Counter
import sys
import Topic_model


class BertTopic(Topic_model.Topic_Model):
    def __init__(self, **kwargs):
        Topic_model.__init__(self)
        self.model = BERTopic(**kwargs)

    @classmethod
    def recommended_conf(cls, n_neighbors, min_topic_size, calculate_probabilities, k=None):
        models = {'vectorizer_model': CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=15),
                  'umap_model': UMAP(n_neighbors=n_neighbors, n_components=10, min_dist=0.0,
                                     metric='cosine')}  # models for bert args
        if k:
            models['hdbscan'] = KMeans(n_clusters=k)
        cls(**models, calculate_probabilities=calculate_probabilities, verbose=True,
            min_topic_size=min_topic_size,
            nr_topics="auto")

    def fit(self, documents):
        self.model.fit(documents)

    def transform(self, documents):
        self.__data = pd.DataFrame({})
        self.__data['text'] = documents
        topics, probs = self.model.transform(documents)
        self.update_data_topic_prob(topics=topics, probs=probs)
        return self.__data

    def fit_transform(self, documents):
        self.model.fit(documents)
        return self.transform(documents)

    # def optimize(self, data):
    #     self.optimize = Optimization(data)
    #     return self.optimize
    #
    # def get_updated_data(self):
    #     return self.data

    def insert_topic_word(self):
        topics = set(self.__data['Topic'].to_list())
        size = len(set(topics))
        for t in range(-1, size - 1):
            t_w = set()
            topic = self.model.get_topic(t)
            if not isinstance(topic, bool) and str(topic) != 'NaN':
                for words in topic:
                    t_w.add(words[0])
                self.__data.loc[self.__data.Topic == t, "topic_words"] = ', '.join(t_w)

    # def fix_topic_outline(self, probs):
    #     probability_threshold = 0.01
    #     topics = [np.argmax(prob) if max(prob) >= probability_threshold else -1 for prob in probs]
    #     return topics

    # def build_umap_model(self,):
    #     umap_model = UMAP(n_neighbors=n_neighbors, n_components=10, min_dist=0.0, metric='cosine')
    #     return umap_model

    def update_data_topic_prob(self, topics, probs):
        self.__data["Topic"] = topics
        prob = [round(p[np.argmax(p)], 4) for p in probs]
        self.__data["Topic"] = topics
        self.__data['probas'] = prob
        # Change the get_topic's Quantities
        c = Counter(topics)
        sorted_c = sorted(c.items(), key=lambda x: x[0])
        get_topics = self.model.get_topic_info()
        get_topics['Count'] = [i[1] for i in sorted_c]
        self.__data = self.__data.merge(get_topics, left_on='Topic', right_on='Topic')
        self.data['ID'] = len(self.__data)
        self.insert_topic_word()
        print("The data is updated successfully!")

    # def create_model(self, text_lst, n_neighbors, min_topic_size, calculate_probabilities, path=None):
    #     vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=15)
    #     umap_model = UMAP(n_neighbors=n_neighbors, n_components=10, min_dist=0.0, metric='cosine')
    #     self.model = self.get_topic_model(map_model=umap_model, vectorizer_model=vectorizer_model,
    #                                       calculate_probabilities=calculate_probabilities, verbose=True,
    #                                       min_topic_size=min_topic_size,
    #                                       nr_topics="auto")
    #     topics, probs = self.model.fit_transform(text_lst)
    #     self.update_data_topic_prob(topics=topics, probs=probs)
    #     return self.model

