# specify id
y_id = 1
track_id = 1
file_id = 58
prefix = './'

# import modules
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '../utils/')
from training_utils import *

# load files
test_x = np.load('../../X_test.npz')['arr_0']
train_x = np.load('../../X_train.npz')['arr_0']
train_y = np.load('../../Y_train.npz')['arr_0'][:, y_id]
all_x = train_x.copy()
all_y = train_y.copy()
print(test_x.shape, train_x.shape, train_y.shape)

'''
# pick only important data
idx = {}
with open('../29/adaboost' + str(y_id) + '_feature.csv', 'r') as f:
    i = 0
    for lines in f:
        importance = float(lines.replace('\n', '').split(',')[y_id])
        if i not in idx:
            idx[i] = 0
        idx[i] += importance
        i += 1
with open('../28/adaboost' + str(y_id) + '_feature.csv', 'r') as f:
    i = 0
    for lines in f:
        importance = float(lines.replace('\n', '').split(',')[y_id])
        if i not in idx:
            idx[i] = 0
        idx[i] += importance
        i += 1

with open('../32/adaboost' + str(y_id) + '_feature.csv', 'r') as f:
    i = 0
    for lines in f:
        importance = float(lines.replace('\n', '').split(',')[y_id])
        if i not in idx:
            idx[i] = 0
        idx[i] += importance
        i += 1
with open('../35/random_forest' + str(y_id) + '_feature.csv', 'r') as f:
    i = 0
    for lines in f:
        importance = float(lines.replace('\n', '').split(',')[y_id])
        if i not in idx:
            idx[i] = 0
        idx[i] += importance
        i += 1
    
idxx = [i[0] for i in idx.items() if i[1] > 1e-3]
print(len(idxx))
train_x = train_x[:, idxx]
test_x = test_x[:, idxx]
all_x = all_x[:, idxx]
print(train_x.shape, all_x.shape, test_x.shape)
'''

# split data
train_x, mytest_x, train_y, mytest_y = train_test_split(train_x, train_y, test_size=0.052631578947368, random_state=1126)
# train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.055555555555554, random_state=1126)
# print(train_x.shape, val_x.shape, mytest_x.shape, train_y.shape, val_y.shape, mytest_y.shape)
print(train_x.shape, mytest_x.shape, train_y.shape, mytest_y.shape)

# define my own scorer for t2
# actually it should be called error function here
w = [300.0, 1.0, 200.0]
def scorer2(y_pred, y_true):
    return 'error', 1.0 * np.sum(np.abs(y_true.get_label() - y_pred) / y_true.get_label()) / len(y_pred)

# define my own scorer for t1
def scorer(y_pred, y_true):
    return 'error', w[y_id] * np.sum(np.abs(y_true.get_label() - y_pred)) / len(y_pred)
  
# define my own error function
def mae(y_true, y_pred): # ln(cosh(x))
    grad = np.tanh(y_pred - y_true)
    hess = 1 - grad * grad
    return grad, hess

def mae2(y_true, y_pred): # 2/k*ln(1+exp(kx))-x-2/k*ln(2), k=10
    grad = (np.exp(10 * (y_pred - y_true)) - 1) / (np.exp(10 * (y_pred - y_true)) + 1)
    hess = (20 * np.exp(10 * (y_pred - y_true))) / (np.exp(10 * (y_pred - y_true)) + 1) ** 2
    return grad, hess
  
params = {
    'objective': mae,
    'max_depth': 6,
    'learning_rate': 0.01,
    'verbosity': 20,
    'tree_method': 'hist',
    'predictor': 'cpu_predictor',
    'n_estimators': 12000,
    'n_jobs': -1,
    'subsample': 0.5,
    'colsample_bytree': 0.05,
    'booster': 'dart'
}
params2 = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'verbosity': 20,
    'tree_method': 'hist',
    'predictor': 'cpu_predictor',
    'n_estimators': 100,
    'n_jobs': -1,
    'subsample': 0.5,
    'colsample_bytree': 0.05,
    'booster': 'dart'
}

# fit
model = XGBRegressor(**params2)
model.fit(train_x, train_y)
model.save_model('tmp.model')
print('tmp finish')
model = XGBRegressor(**params)
# model.fit(scaled_x, scaled_y, eval_set=[(val_x, val_y)], eval_metric=scorer, early_stopping_rounds=None)
# model.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric=scorer, early_stopping_rounds=None)
model.fit(train_x, train_y, xgb_model='tmp.model')
# model.fit(scaled_x, scaled_y)

# write result

f = open(f'{prefix}{file_id}_result_y{y_id}_{track_id}.txt', 'w')

print("ein1:", err1_calc(model.predict(train_x), train_y, y_id), file=f)
# print("eval1:", err1_calc(model.predict(val_x), val_y, y_id), file=f)
print("etest1:", err1_calc(model.predict(mytest_x), mytest_y, y_id), file=f)
print("eall1:", err1_calc(model.predict(all_x), all_y, y_id), file=f)

print("ein2:", err2_calc(model.predict(train_x), train_y), file=f)
# print("eval2:", err2_calc(model.predict(val_x), val_y), file=f)
print("etest2:", err2_calc(model.predict(mytest_x), mytest_y), file=f)
print("eall2:", err2_calc(model.predict(all_x), all_y), file=f)

f.close()

# write files
write_prediction(f'{prefix}{file_id}_train_y{y_id}_{track_id}.txt', 'w', model.predict(all_x).reshape((47500, 1)).astype('str'))
write_prediction(f'{prefix}{file_id}_test_y{y_id}_{track_id}.txt', 'w', model.predict(test_x).reshape((2500, 1)).astype('str'))
model.save_model(f'{prefix}{file_id}_model_y{y_id}_{track_id}.model')


