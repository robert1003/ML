import numpy as np
import pandas as pd

ids = [28, 29, 32, 35, 51, 54, 55]

feature_importances = np.zeros(shape=(10000, 3))
for i in ids:
    a = np.array(pd.read_csv(f'./features/{i}.csv', header=None))
    feature_importances += a

# load datas 
X = np.load('../../X_train.npz')['arr_0']
y = np.load('../../Y_train.npz')['arr_0'][:, 2]

# search params
params = {
    'num_leaves': [240, 180, 120, 60],
    'min_data_in_leaf': [20, 100, 500],
    'bagging_fraction': [1.0, 0.75, 0.5],
    'bagging_freq': [12, 7, 3],
    'learning_rate': [0.1, 0.05, 0.01],
    'boosting': ['dart', 'gbdt'],
    'num_iterations': [1000]
}

# search

import itertools
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

feature_percent = [1e-2, 5e-3, 1e-3]
params_list = []
Ein = []
Eval = []

f = open('status.txt', 'w')
keys, values = zip(*params.items())
for per in feature_percent:
    X_train, X_test, y_train, y_test = train_test_split(X[:, feature_importances[:, 2] > per], y, test_size=0.2, random_state=1126)
    w_train, w_test = 1.0 / y_train, 1.0 / y_test
    for v in itertools.product(*values):
        param = dict(zip(keys, v))
        print(per, param, file=f)
        f.flush()
        print(per, param)
        params_list.append((per, param))

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train,
                  sample_weight=w_train,
                  eval_set=[(X_test, y_test)],
                  eval_sample_weight=[w_test],
                  eval_metric='l1',
                  early_stopping_rounds=100,
                  verbose=50)

        Ein.append(np.mean(np.abs(y_train - model.predict(X_train)) / y_train))
        Eval.append(np.mean(np.abs(y_test - model.predict(X_test)) / y_test))

        print('Ein', Ein[-1], file=f)
        f.flush()
        print('Ein', Ein[-1])
        print('Eval', Eval[-1], file=f)
        f.flush()
        print('Eval', Eval[-1])

        del(param, model)

    del(X_train, X_test, y_train, y_test)

print('', file=f)
print(params_list[np.argmin(Eval)], file=f)
f.flush()
print(params_list[np.argmin(Eval)])
f.close()
