import numpy as np
import pandas as pd
import yaml

from ayniy.model.model_lgbm import ModelLGBM
from ayniy.model.runner import Runner
from ayniy.utils import Data


X_train = Data.load('../input/X_train_00.pkl')
y_train = Data.load('../input/y_train.pkl')
X_test = Data.load('../input/X_test_00.pkl')

X_train.drop(['fiscal_year'], axis=1, inplace=True)
X_test.drop(['fiscal_year'], axis=1, inplace=True)
y_train = np.log(np.sqrt(y_train))

f = open("configs/fe_00.yml", "r+")
configs = yaml.load(f)
categorical_cols = configs['cols_definition']['categorical_col']

params_lgbm = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 14,
    'max_depth': 6,
    "feature_fraction": 0.8,
    'subsample_freq': 1,
    "bagging_fraction": 0.7,
    'min_data_in_leaf': 10,
    'learning_rate': 0.1,
    "boosting": "gbdt",
    "lambda_l1": 0.4,
    "lambda_l2": 0.4,
    "verbosity": -1,
    "random_state": 42,
    "num_boost_round": 50000,
    "early_stopping_rounds": 100
}

evaluation_metric = 'rmse'
exp_name = 'lgbm'

runner = Runner(exp_name, ModelLGBM, X_train, X_test, y_train, evaluation_metric, params_lgbm, categorical_cols)
runner.run_train_cv()
runner.run_predict_cv()

pred = Data.load(f'../output/pred/{exp_name}-test.pkl')
oof = Data.load(f'../output/pred/{exp_name}-train.pkl')
sub = pd.read_csv('../input/data/sample_submission.csv')

sub['market_cap_indexed'] = np.exp(pred)
sub.to_csv(f'../output/submissions/submission_{exp_name}_0.csv', index=False)

sub['market_cap_indexed'] = np.exp(pred)**2
sub.to_csv(f'../output/submissions/submission_{exp_name}_1.csv', index=False)
