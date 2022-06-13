from pathlib import Path
import time
import os

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


class Create_Model:

    def __init__(self, n_neighbor, min_topic_size, subreddit, sub_kind, year):
        self.n_neighbor = n_neighbor
        self.min_topic_size = min_topic_size
        self.subreddit = subreddit
        self.sub_kind = sub_kind
        self.year = year
        self.path_clean_data = f'/home/roikreme/BertTopic/cleaned_data/{self.subreddit}_{self.sub_kind}_{self.year}.pickle'
        # self.path_to_save_model = f"/home/roikreme/BertTopic/models/{self.subreddit}_{self.sub_kind}_{self.year}"
        self.path_to_save_model = f"/dt/puzis/dt-reddit/project_code/BertTopic/{self.subreddit}_{self.sub_kind}_{self.year}"
        # self.model_path = f"{self.path_to_save_model}/optimal_model_{self.n_neighbor}_{self.min_topic_size}"
        self.model_path = f"/dt/puzis/dt-reddit/project_code/BertTopic/optimal_model_{self.n_neighbor}_{self.min_topic_size}"
        # self.path_to_save_topic_output = f'/home/roikreme/BertTopic/topic_output/{self.subreddit}_{self.sub_kind}_{self.year}.pickle'
        self.path_to_save_topic_output = f'/dt/puzis/dt-reddit/project_code/BertTopic/{self.subreddit}_{self.sub_kind}_{self.year}.pickle'

    def get_topic_model(n_neighbors, min_topic_size, calculate_probabilities=True):
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=10, min_dist=0.0, metric='cosine')
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=15)
        topic_model = BERTopic(umap_model=umap_model, vectorizer_model=vectorizer_model,
                               calculate_probabilities=calculate_probabilities, verbose=True,
                               min_topic_size=min_topic_size,
                               nr_topics="auto")
        # topic_model.fit(documents)
        return topic_model

    def insert_topic_word(dff):
        topics = set(dff['Topic'].to_list())
        size = len(set(topics))
        for t in range(-1, size - 1):
            t_w = set()
            topic = model.get_topic(t)
            if not isinstance(topic, bool) and str(topic) != 'NaN':
                for words in topic:
                    t_w.add(words[0])
                dff.loc[dff.Topic == t, "topic_words"] = ', '.join(t_w)

    def fix_topic_outline(self, probs):
        probability_threshold = 0.01
        topics = [np.argmax(prob) if max(prob) >= probability_threshold else -1 for prob in probs]
        return topics

    Reddit.extract_featurs(topic_modeling=[LDA()], haggingface_features=["GoEmotion", "BerTweetEmotion"], ner=True)

    def create_model(self):
        print("start creation the optimize model")
        text = "title_selftext"
        df = pd.read_pickle(self.path_clean_data)
        df = df.reset_index()
        # Create the model

        model = self.get_topic_model(self.n_neighbor, self.min_topic_size)

        # Fit and transform by all the data
        start_time = time.time()
        #model.fit(df.sample(min(300000, len(df)))[text].to_list())
        topics, probs = model.fit_transform(df[text].to_list())

        print("--- %s seconds ---" % (time.time() - start_time))
        Path(self.path_to_save_model).mkdir(parents=True, exist_ok=True)

        model.save(self.model_path)
        topics = self.fix_topic_outline(probs)

        df["Topic"] = topics
        prob = [round(p[np.argmax(p)], 4) for p in probs]
        df["Topic"] = topics
        df['probas'] = prob

        # Change the get_topic's Quantities
        c = Counter(topics)
        sorted_c = sorted(c.items(), key=lambda x: x[0])
        get_topics = model.get_topic_info()
        get_topics['Count'] = [i[1] for i in sorted_c]

        df = df.merge(get_topics, left_on='Topic', right_on='Topic')
        df['ID'] = len(df)

        self.insert_topic_word(df)

        df.to_pickle(self.path_to_save_topic_output, protocol=4)
        print("end")
