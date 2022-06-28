from collections import Counter
import gensim.corpora as corpora
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
from Optimization import Optimization
from itertools import product
from sklearn.cluster import KMeans
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from reddit_censorship_moderation.Topic_Modeling.Create_Model.BertTopic import BertTopic


class Optimization_Bert(Optimization):

    def __init__(self, grid_params):
        Optimization.__init__(grid_params)
        self.optimization_table = pd.DataFrame(self.grid_params)

    def fit(self, documents):
        keys = self.grid_params.keys()
        vals = self.grid_params.values()
        for instance in product(*vals):
            curr_params = dict(zip(keys, instance))
            print(curr_params)
            bert_model = BertTopic.recommended_conf(curr_params)
            topics, probs = bert_model.fit_transform(documents)
            self.optimization_table['Topic'] = topics
            print("finish transform")

            # Calaulate the amount of each topic
            get_topic = bert_model.get_topic_info()
            c = Counter(topics)
            sorted_c = sorted(c.items(), key=lambda x: x[0])
            count = [i[1] for i in sorted_c]

            # update the dataframe with the ammount of each topic after transform
            get_topic['Count'] = count

            sum_ = sum(count)
            get_topic['percentage'] = get_topic['Count'].apply(lambda x: str(round((x / sum_) * 100, 2)) + '%')

            print("start coh")
            coh = self.get_coherence(sorted_c, topics)

            # Adding to the dataframe
            self.optimization_table[all([(x == y) for x, y in curr_params.items()])] = \
                curr_params.update({'Num of Topic': str(len(get_topic)), 'coherence': coh,
                                    'Quantity of topic -1': str(count[0]),
                                    'topic -1 %': get_topic['percentage'][0], 'Total Amount': str(sum_)})

            print("coh is:{}, {}".format(coh, curr_params))

    def get_best_model(self):
        pass

    def get_best_params(self):
        pass

    def get_best_score(self):
        pass

    def get_coherence(self, topic_model, topics):
        documents_per_topic = self.data.groupby(['Topic'], as_index=False).agg({'title_selftext': ' '.join})
        cleaned_docs = topic_model._preprocess_text(documents_per_topic.title_selftext.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = []
        for t in range(len(set(topics)) - 2):
            t_w = []
            topic = topic_model.get_topic(t)
            if not isinstance(topic, bool):
                for words in topic:
                    if words[0] not in tokens[0]: continue
                    t_w.append(words[0])
                topic_words.append(t_w)

        # Evaluate
        coh_list = ['c_npmi', 'c_uci', 'u_mass', 'c_v']
        coherence = {}
        for c in coh_list:
            coh = CoherenceModel(topics=list(topic_words),
                                 texts=tokens,
                                 corpus=corpus,
                                 dictionary=dictionary,
                                 coherence=c)
            coherence[c] = coh.get_coherence()

        return coherence
