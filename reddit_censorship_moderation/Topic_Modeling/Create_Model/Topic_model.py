import pandas as pd


class Topic_Model:

    def __init__(self):
        self.__data = None

    @classmethod
    def recommended_conf(cls):
        pass

    def get_data(self):
        return self.__data

    def get_topic_model(self):
        return self.model

    def fit(self, documents):
        pass

    def transform(self, documents):
        pass

    def fit_transform(self, documents):
        pass

    def save_model(self, path):
        pass
