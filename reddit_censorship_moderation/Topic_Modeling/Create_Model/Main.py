import sys
import os
sys.path.append(os.path.abspath('/dt/puzis/dt-reddit/project_code/Topic_Modeling/Create_model/'))
from Create_Model import BertTopic

if __name__ == '__main__':
    argv = sys.argv
    subreddit = argv[1]  # "UkrainianConflict"
    sub_kind = argv[2]  # "post"
    year = argv[3]  # 2022

    # create model
    n_neighbor = int(argv[4])
    min_topic_size = int(argv[5])
    model = BertTopic(n_neighbor, min_topic_size, subreddit, sub_kind, year)
    model.create_model()

