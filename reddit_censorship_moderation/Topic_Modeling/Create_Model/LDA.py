import logging
import os
import time
from wordcloud import WordCloud
import spacy
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
import pathlib
import Topic_model


class LDA(Topic_model.Topic_Model):
    def __init__(self, **kwargs):
        Topic_model.__init__(self)
        self.model = LatentDirichletAllocation(**kwargs)
        self.vectorizer = None

    @classmethod
    def recommended_conf(cls, n_topics=20,  # Number of topics
                         max_iter=10,  # Max learning iterations
                         learning_method='online',
                         random_state=100,  # Random state
                         batch_size=128,  # n docs in each learning iter
                         evaluate_every=-1,  # compute perplexity every n iters, default: Don't
                         n_jobs=-1,  # Use all available CPUs
                         ):
        cls(n_topics=n_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            random_state=random_state,
            batch_size=batch_size,
            evaluate_every=evaluate_every,
            n_jobs=n_jobs)

    @classmethod
    def create_vectorizer(cls, **kwargs):
        return CountVectorizer(analyzer='word',
                               min_df=10,
                               stop_words='english',
                               lowercase=True,
                               token_pattern='[a-zA-Z0-9]{3,}',
                               **kwargs)

    def fit(self, documents, **kwargs):
        self.vectorizer = LDA.create_vectorizer(**kwargs)
        data_vectorized = self.vectorizer.fit_transform(documents)
        self.model.fit(data_vectorized)

    def transform(self, documents, n_words=20):
        data_vectorized = self.vectorizer.fit_transform(documents)
        lda_output = self.model.transform(data_vectorized)
        # column names
        topicnames = ["Topic" + str(i) for i in range(self.model.n_topics)]

        # index names
        docnames = ["Doc" + str(i) for i in range(len(documents))]

        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        dominant_topic_probs = np.max(df_document_topic.values, axis=1)

        # Show top n keywords for each topic
        keywords = np.array(self.vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in self.model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))

        df_document_topic['Topic'] = dominant_topic
        df_document_topic['probs'] = dominant_topic_probs
        df_document_topic['topic_words'] = topic_keywords
        return df_document_topic

    def fit_transform(self, documents, **kwargs):
        self.fit(documents, **kwargs)
        return self.transform(documents)

    def save_model(self, path):
        pass

    # def create_model(self, **kwargs):
    #     # Build LDA Model
    #     lda_model = LatentDirichletAllocation(**kwargs)
    #     self.model = lda_model
    #     # lda_model = LatentDirichletAllocation(n_topics=20,  # Number of topics
    #     #                                       max_iter=10,  # Max learning iterations
    #     #                                       learning_method='online',
    #     #                                       random_state=100,  # Random state
    #     #                                       batch_size=128,  # n docs in each learning iter
    #     #                                       evaluate_every=-1,  # compute perplexity every n iters, default: Don't
    #     #                                       n_jobs=-1,  # Use all available CPUs
    #     #                                       )
