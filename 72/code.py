# import modules
import numpy as np 
from sklearn.ensemble import ExtraTreesRegressor
import sys

# parameters
machine_id = sys.argv[1]
y_id = 1
track_id = 1

n_estimators = int(sys.argv[2])
criterion = 'mae'
n_jobs = -1
oob_score = True
bootstrap = True
information_file_name = f'y{y_id}_t{track_id}_m{machine_id}_info.txt'
predict_y_file_name = f'y{y_id}_t{track_id}_m{machine_id}_predict.txt'
data_file_path_prefix = '../../'
features_file_path_prefix = '../'

# load data
test_x = np.load(data_file_path_prefix + 'X_test.npz')['arr_0']
train_x = np.load(data_file_path_prefix + 'X_train.npz')['arr_0']
train_y = np.load(data_file_path_prefix + 'Y_train.npz')['arr_0'][:, y_id]
print('data loaded')

# calculate type 1 error
w = [200.0, 1.0, 300.0]
def err1_calc(predict, real, idx):
    return np.sum(w[idx] * np.abs(predict - real)) / len(predict)

# calculate type 2 errr
def err2_calc(predict, real):
    return np.sum(np.abs(predict - real) / real) / len(predict)

# write prediction
def write_prediction(name, mode, data):
    assert mode == 'a' or mode == 'w'
    with open(name, mode) as f:
        for lines in data:
            print(','.join(list(lines)), file=f)

# get important features
idx = {}
with open(features_file_path_prefix + '29/adaboost' + str(y_id) + '_feature.csv', 'r') as f:
    i = 0
    for lines in f:
        importance = float(lines.replace('\n', '').split(',')[y_id])
        if i not in idx:
            idx[i] = 0
        idx[i] += importance
        i += 1
with open(features_file_path_prefix + '28/adaboost' + str(y_id) + '_feature.csv', 'r') as f:
    i = 0
    for lines in f:
        importance = float(lines.replace('\n', '').split(',')[y_id])
        if i not in idx:
            idx[i] = 0
        idx[i] += importance
        i += 1
with open(features_file_path_prefix + '32/adaboost' + str(y_id) + '_feature.csv', 'r') as f:
    i = 0
    for lines in f:
        importance = float(lines.replace('\n', '').split(',')[y_id])
        if i not in idx:
            idx[i] = 0
        idx[i] += importance
        i += 1
with open(features_file_path_prefix + '35/random_forest' + str(y_id) + '_feature.csv', 'r') as f:
    i = 0
    for lines in f:
        importance = float(lines.replace('\n', '').split(',')[y_id])
        if i not in idx:
            idx[i] = 0
        idx[i] += importance
        i += 1
features_idx = np.array([i[0] for i in idx.items() if i[1] > 1e-2])

# rescale data for track 2
datas_idx = []
if track_id == 2:
    if y_id == 0:
        for i in range(train_x.shape[0]):
            v = int(1 / train_y[i] // 5 + 1)
            for _ in range(v):
                datas_idx.append(i)
    elif y_id == 1:
        for i in range(train_x.shape[0]):
            v = int(1 / train_y[i] * 100 + 1)
            for _ in range(v):
                datas_idx.append(i)
    else:
        for i in range(train_x.shape[0]):
            v = int(1 / train_y[i] + 1)
            for _ in range(v):
                datas_idx.append(i)
else:
    for i in range(train_x.shape[0]):
        datas_idx.append(i)
datas_idx = np.array(datas_idx)

print(f'features_idx shape {features_idx.shape}')
print(f'datas_idx shape {datas_idx.shape}')

# modify train_x, train_y
train_x = train_x[datas_idx][:, features_idx]
train_y = train_y[datas_idx]

# modify test_x
test_x = test_x[:, features_idx]

print(f'train_x shape {train_x.shape}')
print(f'train_y shape {train_y.shape}')
print(f'test_x shape {test_x.shape}')

# define model
model = ExtraTreesRegressor(n_estimators=n_estimators, criterion=criterion, n_jobs=n_jobs, oob_score=oob_score, bootstrap=bootstrap)

# train model
model.fit(train_x, train_y)

# print informations
with open(information_file_name, 'w') as f:
    if oob_score:
        print(f'oob_score: {model.oob_score_}', file=f)
    print(f'track 1 ein: {err1_calc(model.predict(train_x), train_y, y_id)}', file=f)
    print(f'track 2 ein: {err2_calc(model.predict(train_x), train_y)}', file=f)

# write prediction
write_prediction(predict_y_file_name, 'w', model.predict(test_x).reshape((2500, 1)).astype('str'))
