# Load cleaned data
import sys
import os
sys.path.append(os.path.abspath('/dt/puzis/dt-reddit/project_code/Topic_Modeling/Optimization/'))
from Optimization import Optimization

if __name__ == '__main__':

    argv = sys.argv
    subreddit = argv[1]  # "UkrainianConflict"
    sub_kind = argv[2]  # "post"
    year = argv[3]  # 2022
    
    # Optimization
    Optimization = Optimization(subreddit, sub_kind, year)
    df = Optimization.get_data()
    Optimization.optimize(df)