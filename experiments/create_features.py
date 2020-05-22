import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import yaml

from ayniy.preprocessing.tabular import label_encoding, count_encoding, count_encoding_interact, matrix_factorization, frequency_encoding
from ayniy.preprocessing.tabular import aggregation, numeric_interact
from ayniy.preprocessing.tabular import count_null
from ayniy.preprocessing.tabular import save_as_pickle, delete_cols, detect_delete_cols
from ayniy.mkfold import mkStratifiedKFold


data2014 = pd.read_csv('../input/data/2014/documents.csv')
data2015 = pd.read_csv('../input/data/2015/documents.csv')
data2016 = pd.read_csv('../input/data/2016/documents.csv')
data2017 = pd.read_csv('../input/data/2017/documents.csv')
sub = pd.read_csv('../input/data/sample_submission.csv')

f = open("configs/fe_00.yml", "r+")
configs = yaml.load(f)
id_col = configs['cols_definition']['id_col']
target_col = configs['cols_definition']['target_col']
categorical_cols = configs['cols_definition']['categorical_col']
numerical_cols = configs['cols_definition']['numerical_col']

data2014 = pd.merge(data2014, data2015[[id_col, target_col]], on=id_col, how='inner', suffixes=('_before', ''))
data2015 = pd.merge(data2015, data2016[[id_col, target_col]], on=id_col, how='inner', suffixes=('_before', ''))
data2016 = pd.merge(data2016, data2017[[id_col, target_col]], on=id_col, how='inner', suffixes=('_before', ''))

data2014[target_col] = data2014[target_col] / data2014[target_col + '_before']
data2015[target_col] = data2015[target_col] / data2015[target_col + '_before']
data2016[target_col] = data2016[target_col] / data2016[target_col + '_before']
data2017 = pd.merge(data2017.drop([target_col], axis=1), sub[[id_col]], on=id_col, how='inner')

data = pd.concat([data2014, data2015, data2016]).reset_index(drop=True)
data.dropna(subset=[target_col], inplace=True)

train = data[categorical_cols + numerical_cols + [target_col]]
test = data2017[categorical_cols + numerical_cols]

train['close_minus_open'] = train['close'] - train['open']
test['close_minus_open'] = test['close'] - test['open']

# Count null
encode_col = list(train.columns)
encode_col.remove(target_col)
train, test = count_null(train, test, {'encode_col': encode_col})

# Categorical
train, test = label_encoding(train, test, {'encode_col': categorical_cols})
print(f'train.shape: {train.shape}')
train, test = frequency_encoding(train, test, {'encode_col': categorical_cols})
print(f'train.shape: {train.shape}')
train, test = count_encoding(train, test, {'encode_col': categorical_cols})
print(f'train.shape: {train.shape}')
train, test = count_encoding_interact(train, test, {'encode_col': categorical_cols})
print(f'train.shape: {train.shape}')
train, test = matrix_factorization(train, test, {'encode_col': categorical_cols}, {'n_components_lda': 5, 'n_components_svd': 3})
print(f'train.shape: {train.shape}')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
for c in categorical_cols:
    train['te_' + c] = np.nan
    test['te_' + c] = 0
    for i, (train_index, valid_index) in enumerate(cv.split(train, train['fiscal_year'])):
        mean_val = train.loc[train_index].groupby(c)[target_col].mean().reset_index()
        mean_val.columns = [c, '_te']
        trn_arr = train[[c]].merge(mean_val, on=c, how='left')['_te']
        tst_arr = test[[c]].merge(mean_val, on=c, how='left')['_te']
        train.loc[valid_index, 'te_' + c] = trn_arr.loc[valid_index]
        test['te_' + c] += tst_arr / cv.n_splits

# Numerical
train, test = aggregation(train, test,
                          col_definition={'groupby_dict': configs['aggregation']['groupby_dict'], 'nunique_dict': configs['aggregation']['nunique_dict']})
print(f'train.shape: {train.shape}')
train, test = numeric_interact(train, test, {'encode_col': [
    'close_minus_open',
    'average',
    'high',
    'open',
    'PBR',
    'low',
    'net_assets_per_share',
    'close',
    'operating_income',
    'market_cap',
    'PER']})
print(f'train.shape: {train.shape}')

unique_cols, duplicated_cols, high_corr_cols = detect_delete_cols(train, test, {'encode_col': numerical_cols}, option={'threshold': 0.995})
train, test = delete_cols(train, test, {'encode_col': unique_cols + duplicated_cols + high_corr_cols})

save_as_pickle(train, test, {'target_col': target_col}, option={'exp_id': '_00'})
mkStratifiedKFold(train, 'fiscal_year', n_splits=5, shuffle=True, name='fold_id')
