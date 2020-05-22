import numpy as np
import pandas as pd
import yaml

from ayniy.model.model_cat import ModelCatRegressor
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

params_cat = {
    'depth': 6,
    'learning_rate': 0.1,
    'iterations': 10000,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 777,
    'allow_writing_files': False,
    'task_type': "CPU",
    'early_stopping_rounds': 50
}

evaluation_metric = 'rmse'
exp_name = 'cat'

runner = Runner(exp_name, ModelCatRegressor, X_train, X_test, y_train, evaluation_metric, params_cat, categorical_cols)
runner.run_train_cv()
runner.run_predict_cv()

pred = Data.load(f'../output/pred/{exp_name}-test.pkl')
oof = Data.load(f'../output/pred/{exp_name}-train.pkl')
sub = pd.read_csv('../input/data/sample_submission.csv')

sub['market_cap_indexed'] = np.exp(pred)
sub.to_csv(f'../output/submissions/submission_{exp_name}_0.csv', index=False)
