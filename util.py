import numpy as np
import pandas as pd
import datetime

def make_submission(estimator, data=None, outfile=None):
    time = datetime.datetime.now()\
        .strftime('%d_%m_%Y__%H_%M_%S')
    if data is None:
        data = pd.read_csv('data/test.csv', index_col='id')
    if outfile is None:
        outfile = 'data/submission_{}.csv'.format(time)
    
    submission = pd.DataFrame(index=data.index,
                              data=estimator.predict(data),
                              columns=['target'])
    submission.to_csv(outfile)
    
def best_score(grid):
    results = pd.DataFrame(grid.cv_results_)
    best = results.loc[results.rank_test_score == 1, 'mean_test_score']
    return np.sqrt(-best)

def hist(*args, **kwargs):
    counts, edges = np.histogram(*args, **kwargs)
    centers = (edges[1:] + edges[:-1]) / 2
    return pd.Series(index=centers, data=counts)