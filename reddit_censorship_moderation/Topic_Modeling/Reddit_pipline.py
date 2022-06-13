from .Create_Model.LDA import LDA
from .Create_Model.BertTopic import BertTopic
from ..Features.Extract_NLP_Features import Features
import pandas as pd


class Reddit_pipline:

    def __init__(self, data):
        self.data = data
        self.data_berttopic = None

    def extract_features(self, topic_modeling=[LDA(), BertTopic()],
                         haggingface_features=["GoEmotion", "BerTweetEmotion"], ner=True):
        '''

        :param topic_modeling:
        :param haggingface_features:
        :param ner:
        :return:
        '''
        final_df = pd.DataFrame({})
        for model in topic_modeling:
            model.fit(self.data)
            df = model.transform(self.data)
            final_df.merge(df, axis=1, suffixes=type(model).__name__)

        if haggingface_features:
            nlp_features = Features(self.data)
            df = nlp_features.get_features(haggingface_features, ner)
            final_df.merge(df, axis=1, suffixes=type(model).__name__)

        return final_df
