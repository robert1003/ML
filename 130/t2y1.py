import numpy as np
import pandas as pd

ids = [28, 29, 32, 35, 51, 54, 55]

feature_importances = np.zeros(shape=(10000, 3))
for i in ids:
    a = np.array(pd.read_csv(f'../tree_based_model/features/{i}.csv', header=None))
    feature_importances += a

# load datas 
X = np.load('../../X_train.npz')['arr_0']
y = np.load('../../Y_train.npz')['arr_0'][:, 1]
TX = np.load('../../X_test.npz')['arr_0']
print('finish loading')

# search params
params = {
    'num_leaves': 240,
    'min_data_in_leaf': 100,
    'bagging_fraction': 0.75,
    'bagging_freq': 3,
    'learning_rate': 0.1,
    'boosting': 'dart',
    'num_iterations': 1000
}
print(params)

# search

import itertools
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

per = 1e-3

X_train = X[:, feature_importances[:, 1] > per]
X_test = TX[:, feature_importances[:, 1] > per]
w_train = 1.0 / y

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y, sample_weight=w_train)

pd.DataFrame(model.predict(X_train).reshape(-1, 1)).to_csv('train_t2y1.csv', index=None, header=None)
pd.DataFrame(model.predict(X_test).reshape(-1, 1)).to_csv('test_t2y1.csv', index=None, header=None)

print(np.mean(np.abs(y - model.predict(X_train)) / y))
print(np.mean(np.abs(y - model.predict(X_train))))

