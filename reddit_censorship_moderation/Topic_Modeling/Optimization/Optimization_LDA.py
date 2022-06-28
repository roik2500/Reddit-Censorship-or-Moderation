from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

from Optimization import Optimization
from reddit_censorship_moderation.Topic_Modeling.Create_Model.LDA import LDA


class Optimization_LDA(Optimization):

    def __init__(self, grid_params):
        Optimization.__init__(grid_params)
        self.model = GridSearchCV(LatentDirichletAllocation(), param_grid=self.grid_params)

    def fit(self, documents, **kwargs):
        vectorizer = LDA.create_vectorizer(**kwargs)
        data_vectorized = vectorizer.fit_transform(documents)
        self.model.fit(data_vectorized)
        self.best_params = self.model.best_params_
        self.best_model = LDA(self.grid_params)
        self.best_model.model = self.model.best_estimator_

    def get_best_model(self):
        return self.best_model

    def get_best_params(self):
        pass

    def get_best_score(self):
        pass
