from pathlib import Path
import time
import os
from Optimization.Optimization import Optimization
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


class Create_Model(Topic_model.Topic_Model):

    def __init__(self, data):
        Topic_model.__init__(self)
        self.model = None
        self.data = None
        self.optimize = None

    def optimize(self, data):
        self.optimize = Optimization(data)
        return self.optimize

    def get_updated_data(self):
        return self.data

    def insert_topic_word(self):
        topics = set(self.data['Topic'].to_list())
        size = len(set(topics))
        for t in range(-1, size - 1):
            t_w = set()
            topic = self.model.get_topic(t)
            if not isinstance(topic, bool) and str(topic) != 'NaN':
                for words in topic:
                    t_w.add(words[0])
                self.data.loc[self.data.Topic == t, "topic_words"] = ', '.join(t_w)

    def fix_topic_outline(self, probs):
        probability_threshold = 0.01
        topics = [np.argmax(prob) if max(prob) >= probability_threshold else -1 for prob in probs]
        return topics

    # def build_umap_model(self,):
    #     umap_model = UMAP(n_neighbors=n_neighbors, n_components=10, min_dist=0.0, metric='cosine')
    #     return umap_model

    def get_topic_model(self, **kwargs):
        print("start creation the optimize model")
        text = "title_selftext"
        # df = df.reset_index()
        topic_model = BERTopic(**kwargs)
        return topic_model

    def update_data_topic_prob(self, topics, probs):
        self.data["Topic"] = topics
        prob = [round(p[np.argmax(p)], 4) for p in probs]
        self.data["Topic"] = topics
        self.data['probas'] = prob
        # Change the get_topic's Quantities
        c = Counter(topics)
        sorted_c = sorted(c.items(), key=lambda x: x[0])
        get_topics = self.model.get_topic_info()
        get_topics['Count'] = [i[1] for i in sorted_c]
        self.data = self.data.merge(get_topics, left_on='Topic', right_on='Topic')
        self.data['ID'] = len(self.data)
        self.insert_topic_word(self.data)
        print("The data is updated successfully!")

    def create_model(self, text_lst, n_neighbors, min_topic_size, calculate_probabilities, path=None):
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=15)
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=10, min_dist=0.0, metric='cosine')
        self.model = self.get_topic_model(map_model=umap_model, vectorizer_model=vectorizer_model,
                                          calculate_probabilities=calculate_probabilities, verbose=True,
                                          min_topic_size=min_topic_size,
                                          nr_topics="auto")
        topics, probs = self.model.fit_transform(text_lst)
        self.update_data_topic_prob(topics=topics, probs=probs)
        return self.model
