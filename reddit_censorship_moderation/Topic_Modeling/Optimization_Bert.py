from Optimization_Shimon import Optimization
from itertools import product
from sklearn.cluster import KMeans
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

class Optimization_Bert(Optimization):

    def __init__(self, grid_params):
        self.grid_params = grid_params
        self.best_params_ = None
        self.best_model = None

    def fit(self, documents):
        keys = self.grid_params.keys()
        vals = self.grid_params.values()
        for instance in product(*vals):
            curr_params = dict(zip(keys, instance))
            print(curr_params)
            final_params = {
                "vectorizer_model": CountVectorizer(ngram_range=(1, 2), stop_words="english", max_df=1, min_df=1)}
            if 'n_neighbors' in curr_params and curr_params['n_neighbors']:
                umap_model = UMAP(n_neighbors=curr_params['n_neighbors'], n_components=10, min_dist=0.0, metric='cosine')
                final_params['umap_model'] = umap_model
            if 'k' in curr_params and curr_params['k']:
                cluster_model = KMeans(n_clusters=curr_params['k'])
                final_params['hdbscan'] = cluster_model
            if 'min_topic_size' in curr_params and curr_params['min_topic_size']:
                final_params['min_topic_size'] = curr_params['min_topic_size']
            if 'calculate_probabilities' in curr_params and curr_params['calculate_probabilities']:
                final_params['calculate_probabilities'] = curr_params['calculate_probabilities']
            BERTopic(final_params)
    def get_best_model(self):
        pass

    def get_best_params(self):
        pass

    def get_best_score(self):
        pass
