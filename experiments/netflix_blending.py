import numpy as np
import pandas as pd


def netflix(es, ps, e0, la=.0001):
    """Combine predictions with the optimal weights to minimize RMSE.
    Args:
        es (list of float): RMSEs of predictions
        ps (list of np.array): predictions
        e0 (float): RMSE of all zero prediction
        la (float): lambda as in the ridge regression
    Returns:
        (tuple):
            - (np.array): ensemble predictions
            - (np.array): weights for input predictions
    """
    m = len(es)
    n = len(ps[0])

    X = np.stack(ps).T
    pTy = .5 * (n * e0**2 + (X**2).sum(axis=0) - n * np.array(es)**2)

    w = np.linalg.pinv(X.T.dot(X) + la * n * np.eye(m)).dot(pTy)
    return X.dot(w), w


lgbm_0 = pd.read_csv('../output/submissions/submission_lgbm_0.csv')['market_cap_indexed'].values
lgbm_1 = pd.read_csv('../output/submissions/submission_lgbm_1.csv')['market_cap_indexed'].values
cat_0 = pd.read_csv('../output/submissions/submission_cat_0.csv')['market_cap_indexed'].values

es = [
    0.303460,
    0.329844,
    0.303750
]
ps = [
    lgbm_0,
    lgbm_1,
    cat_0
]
e0 = 0.926908

pred, w = netflix(es, ps, e0, la=.0001)
print(w)

exp_name = 'netflix'
sub = pd.read_csv('../input/data/sample_submission.csv')
sub['market_cap_indexed'] = pred
sub.to_csv(f'../output/submissions/submission_{exp_name}.csv', index=False)
