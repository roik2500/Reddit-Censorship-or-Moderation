from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import TfidfModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora import Dictionary
import numpy as np
import xgboost
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, auc, roc_curve, roc_auc_score, \
    mean_squared_error
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from db_utils.FileReader import FileReader
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
from collections import Counter
from fast_ml.model_development import train_valid_test_split
import datetime
load_dotenv()

''' supervised model'''

'''
Creating a new folder in Google drive. 
:argument folderName - name of the new folder
:return full path to the new folder
'''


def create_new_folder_drive(path, new_folder):
    updated_path = "{}{}/".format(path, new_folder)
    if not os.path.exists(updated_path):
        os.makedirs(updated_path)
    return updated_path


class doc_2_vec:
    def __init__(self, corpus):
        self.corpus_tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
        self.model = None

    def train_model(self):
        self.model = Doc2Vec(self.corpus_tagged, vector_size=5, window=2, min_count=1, workers=4)

    def save_model_to_disk(self):
        self.model.save("test_doc2vec.model")

    def load_model_from_disk(self):
        self.model = Doc2Vec.load("test_doc2vec.model")


class tfidf:
    def __init__(self, corpus):
        self.file_reader = FileReader()
        corpus = [d.split() for d in corpus]
        self.dct = Dictionary(corpus)  # fit dictionary
        self._corpus = [self.dct.doc2bow(line) for line in corpus]  # convert corpus to BoW format
        self.model = TfidfModel(self._corpus)  # fit model

    def get_tfidf_by_post_text(self, post_text):
        post_split = [d.split() for d in [post_text]]
        doc_2_bow = [self.dct.doc2bow(line) for line in post_split]
        tfidf_score = self.model[doc_2_bow]
        return [x for x in zip(post_split, tfidf_score)]

    def explore_rare_words_in_removed_posts(self, df_removed_posts):
        removed_tfidf_dict = {}
        for index, row in tqdm(df_removed_posts.iterrows()):
            removed_tfidf_dict[row["post_id"]] = self.get_tfidf_by_post_text(
                row["title_selftext"])  # {post_id: (word, word_id, tf_idf_score)}
        csv_record_path = "G:\\.shortcut-targets-by-id\\1lJuBfy-iW6jibopA67C65lpds3B1Topb\\Reddit Censorship Analysis\\final_project\\Features\\testing\\"
        self.file_reader.write_dict_to_json(path=csv_record_path, file_name="tfidf_removed_posts",
                                            dict_to_write=removed_tfidf_dict)


class Model:

    def __init__(self, year, subreddit, sub_kind):
        self.df_train = pd.DataFrame()
        self.df_train_balanced = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.df_valid = pd.DataFrame()
        self.train_labels = None
        self.test_labels = None
        self.valid_labels = None
        self.file_reader = FileReader()
        self.data = self.read_dataset(year, subreddit, sub_kind)
        self.MAX_POST_NUMBER = self.data.shape[0] 
        self.post_or_comment_model = sub_kind

    # read embedding features     
    def read_embedding_dataset(self, year, subreddit, sub_kind):
        df_data = pd.read_pickle(f"/dt/puzis/dt-reddit/uncleaned_data/{subreddit}_{sub_kind}_{year}.pickle")
        df_embed = pd.read_csv(f"/dt/puzis/dt-reddit/embeeding_scores/{subreddit}_{sub_kind}_{year}.csv")
        data = [df_data ,df_embed.T]
        df = pd.concat(data).drop(columns=['author_fullname','link_flair_text','num_comments','retrieved','title_selftext']).status.dropna()
        df = pd.concat([df_embed.T, df, df_data.date], axis=1, join="inner")
        return df
        
    # read clean data without filters and domain features
    def read_dataset(self, year, subreddit, sub_kind):

        data_df = pd.read_pickle(f"/dt/puzis/dt-reddit/cleaned_data/{subreddit}_{sub_kind}_{year}.pickle")
        features_df = pd.read_csv(f"/dt/puzis/dt-reddit/finall_features_try/{subreddit}_{sub_kind}_{year}.csv", encoding='latin-1')
        simple_features_df = pd.read_csv(f"/sise/home/shai1/reddit_code_shai/simple_features/{year}/{subreddit}_{sub_kind}_{year}.csv", encoding='latin-1')

        data_df.drop(columns=['title_selftext', 'author_fullname','link_flair_text','retrieved', 'num_comments', 'author_fullname'], inplace=True)
        features_df.index = features_df["post_id"]
        features_df.drop(columns=['status', 'date'], inplace=True) #, 'link_flair_text', 'retrieved', 'author_fullname'
        simple_features_df.index = simple_features_df["post_id"]
        simple_features_df.drop(columns=['num_comments', 'author_fullname', 'status', 'date'], inplace=True)
        futures_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 'bertweet_neg','bertweet_neu', 'bertweet_pos', 'hate', 'offensive']

        df = pd.concat((data_df, features_df[futures_list], simple_features_df), axis=1, join="inner")
        df.drop(columns=['post_id'], inplace=True)

        return df
        
    def sort_data_set_by_dates(self):
        
        train_union = pd.concat((self.df_train, self.train_labels), axis=1, join="inner")
        train_union.sort_values(by=['created_date'], inplace=True)
        self.train_labels = None
        self.train_labels = train_union.status
        self.df_train = train_union.drop(columns=['status'])
        
        valid_union = pd.concat((self.df_valid, self.valid_labels), axis=1, join="inner")
        valid_union.sort_values(by=['created_date'], inplace=True)
        self.valid_labels = None
        self.valid_labels = valid_union.status
        self.df_valid = valid_union.drop(columns=['status'])
        
        test_union = pd.concat((self.df_test, self.test_labels), axis=1, join="inner")
        test_union.sort_values(by=['created_date'], inplace=True)
        self.test_labels = None
        self.test_labels = test_union.status
        self.df_test = test_union.drop(columns=['status'])
        

    # use this method after splitting the corpus
    def balance_data_Undersample_the_biggest_dataset(self, class_name):
        under_sampler = RandomUnderSampler(random_state=42)
        if len(self.train_labels.value_counts()) == 1 or len(self.test_labels.value_counts()) == 1 or len(self.valid_labels.value_counts()) == 1:
            print("Canceled balance_data_Undersample_the_biggest_dataset func")
            return False

        self.df_train, self.train_labels = under_sampler.fit_resample(self.df_train, self.train_labels)
        self.df_test, self.test_labels = under_sampler.fit_resample(self.df_test, self.test_labels)
        self.df_valid, self.valid_labels = under_sampler.fit_resample(self.df_valid, self.valid_labels)
        self.sort_data_set_by_dates()
        
        self.df_train.drop(columns=['created_date'], inplace=True)
        self.df_valid.drop(columns=['created_date'], inplace=True)
        self.df_test.drop(columns=['created_date'], inplace=True)
     
    # test of spliting dates for doreen saga
    def train_valid_test_split_local(self, result):

        result.sort_values(by='created_date',inplace=True)

        start_date_before='[2021-01-01 00:00]'
        end_date_before='[2022-26-01 21:12]'

        start_date_during='[2022-24-01 20:00]'
        end_date_during='[2022-28-01 23:59]'

        start_date_after='[2022-29-01 00:00]'
        end_date_after='[2022-15-05 21:12]'


        self.df_train = self.data.iloc[0:140000,:] 
        self.train_labels = self.df_train.status
        
        self.df_valid = self.data.iloc[140000:150000,:]
        self.valid_labels = self.df_valid.status
        
        self.df_test = self.data.iloc[150000:,:]
        self.test_labels = self.df_test.status


    def split_corpus_binary(self, class_name, k_best_features):
        if 'status' not in k_best_features:
            k_best_features.append('created_date')
            k_best_features.append('status')
        self.make_class_as_binary(class_name)
        self.data = self.data.loc[:, k_best_features]
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        self.df_train, self.train_labels, self.df_valid, self.valid_labels, self.df_test, self.test_labels = \
            train_valid_test_split(self.data, target='status', method='sorted', sort_by_col='created_date',
                                   train_size=0.7, valid_size=0.2, test_size=0.1)

        print((self.df_train.shape, self.train_labels.shape))
        print((self.df_valid.shape, self.valid_labels.shape))
        print((self.df_test.shape, self.test_labels.shape))
        
        
    #  change the status to be binary
    def make_class_as_binary(self, class_name):
        print("class_name", class_name)
        if "exists" in self.data.status.value_counts():  
            self.data.status[self.data.status == "exists"] = "exist"
        names = ["removed", "exist", "shadow_ban"] 
        names.remove(class_name)
        neg_class = "not_" + class_name
        self.data["status"].replace(names, neg_class, inplace=True)

    def split_corpus_basic(self):

        self.data = self.data.sort_values(by="created_date")

        border = int(self.MAX_POST_NUMBER * 0.8)
        self.df_train = self.data.iloc[:border, :]
        self.train_labels = self.df_train["status"]
        self.df_train.drop(columns=['status'], inplace=True)
        
        self.df_test = self.data.iloc[border:, :]
        self.test_labels = self.df_test["status"]
        self.df_test.drop(columns=['status'], inplace=True)



#     def split_corpus_all(self, k_best_features):
#         self.data["date"] = pd.to_datetime(self.data["created_date"])

#         self.data = self.data.sort_values(by="created_date")
#         self.MAX_POST_NUMBER = self.data.shape[0]
#         border = int(self.MAX_POST_NUMBER * 0.8)
#         self.df_train = self.data.iloc[:border, :]
#         self.train_labels = self.df_train["status"]
#         self.df_train = self.df_train[k_best_features]

#         self.df_test = self.data.iloc[border:, :]
#         self.test_labels = self.df_test["status"]
#         self.df_test = self.df_test[k_best_features]

    def k_best_features(self, k, feature_list):
        selector = SelectKBest(chi2, k=k)
        selector.fit_transform(self.df_train[feature_list], self.train_labels)
        cols = selector.get_support(indices=True)
        features_df_new = [feature_list[c] for c in cols]

        return features_df_new  

    def train_model(self, model_name, dec_tree_params=None):


        if model_name == 'LogisticRegression':
            return LogisticRegression(random_state=0).fit(self.df_train, self.train_labels)

        elif model_name == 'RandomForestClassifier':
            return RandomForestClassifier(max_depth=5, random_state=0).fit(self.df_train, self.train_labels)

        elif model_name == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(random_state=0, max_depth=dec_tree_params["max_depth"], criterion=dec_tree_params["criterion"])
            return model.fit(self.df_train, self.train_labels)

        elif model_name == "XGBClassifier":
            model_xgboost = xgboost.XGBClassifier(random_state=42, max_depth=5,
                                                  objective='binary:logistic')  # , use_label_encoder=False)
            eval_set = [(self.df_train, self.train_labels,), (self.df_test, self.test_labels)]
            model_xgboost.fit(self.df_train,
                              self.train_labels,
                              early_stopping_rounds=10,
                              eval_set=eval_set,
                              verbose=True)

            return model_xgboost

        else:
            return "invalid model"

    def get_class_size(self, data):
        return list(data.value_counts().to_dict().items())

    def evaluation_indices(self, prediction, prediction_prob, _class, predict_train, predict_prob_train,
                           predict_valid, predict_prob_vaild):
        
       
        # TRAIN - section
        accuracy = accuracy_score(self.train_labels, predict_train)
        precision_recall_fscore = precision_recall_fscore_support(self.train_labels, predict_train, average='weighted')
        f1 = f1_score(self.train_labels, predict_train, average="macro")
        if _class == "all": 
            fpr = 0
            tpr = 0
            thresholds = 0
            _auc = 0
        else:
            _auc = roc_auc_score(self.train_labels, predict_prob_train)
        print("---TRAIN---")
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("precision: %.2f%%" % (precision_recall_fscore[0] * 100),
              "\nrecall: %.2f%%" % (precision_recall_fscore[1] * 100)
              , "\nfscore: %.2f%%" % (f1 * 100), "\nAUC %.4f%%" % (_auc * 100))

        train = [accuracy, precision_recall_fscore[0], precision_recall_fscore[1], f1, _auc]  # [fpr, tpr, thresholds]

        # VALID - section

        accuracy = accuracy_score(self.valid_labels, predict_valid)
        precision_recall_fscore = precision_recall_fscore_support(self.valid_labels, predict_valid, average='weighted')
        f1 = f1_score(self.valid_labels, predict_valid, average="macro")
        if _class == "all":  # mean class is all
            fpr = 0
            tpr = 0
            thresholds = 0
            _auc = 0
        else:
            _auc = roc_auc_score(self.valid_labels, predict_prob_vaild)
            
        print("---VALID---")
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("precision: %.2f%%" % (precision_recall_fscore[0] * 100),
              "\nrecall: %.2f%%" % (precision_recall_fscore[1] * 100)
              , "\nfscore: %.2f%%" % (f1 * 100), "\nAUC %.4f%%" % (_auc * 100))

        valid = [accuracy, precision_recall_fscore[0], precision_recall_fscore[1], f1, _auc]  # [fpr, tpr, thresholds]

        # TEST - SECTION
        accuracy = accuracy_score(self.test_labels, prediction)
        precision_recall_fscore = precision_recall_fscore_support(self.test_labels, prediction, average='weighted')
        f1 = f1_score(self.test_labels, prediction, average="macro")
        if _class == "all": 
            fpr = 0
            tpr = 0
            thresholds = 0
            _auc = 0
        else:
            _auc = roc_auc_score(self.test_labels, prediction_prob)
            
        print("---TEST---")
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("precision: %.2f%%" % (precision_recall_fscore[0] * 100),
              "\nrecall: %.2f%%" % (precision_recall_fscore[1] * 100)
              , "\nfscore: %.2f%%" % (f1 * 100), "\nAUC %.4f%%" % (_auc * 100))

        test = [accuracy, precision_recall_fscore[0], precision_recall_fscore[1], f1, _auc]

        return test, valid, train
