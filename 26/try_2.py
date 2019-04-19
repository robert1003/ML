import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

test_x = np.load('../../X_test.npz')['arr_0']
train_x = np.load('../../X_train.npz')['arr_0']
train_y = np.load('../../Y_train.npz')['arr_0'][:, 2]
all_x = train_x.copy()

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=1126)
val = [(val_x, val_y)]

# stage 1
model = xgb.XGBRegressor(learning_rate=0.1126, max_depth=6, subsample=0.5, n_jobs=32, tree_method='hist', n_estimators=1000)
model.fit(train_x, train_y, eval_set=val, eval_metric='mae', early_stopping_rounds=5)
model.save_model('m2.model')
with open('status.txt', 'w') as f:
    print("stage 2 saved", file=f)

# stage 1 prediction
prediction_train_1 = model.predict(all_x)
prediction_test_1 = model.predict(test_x)
prediction_valid_1 = model.predict(val_x)
with open('train_stage_1_2.txt', 'w') as f:
    for i in prediction_train_1:
        print(i, file=f)
with open('test_stage_1_2.txt', 'w') as f:
    for i in prediction_test_1:
        print(i, file=f)
err1 = 0.0
err2 = 0.0
for i in range(len(val_x)):
    err1 += 300.0 * np.abs(prediction_valid_1[i] - val_y[i])
    err2 += np.abs(prediction_valid_1[i] - val_y[i]) / val_y[i]
with open('status.txt', 'a') as f:
    print('stage 1 predicted', file=f)
    print('err1 valid error:', err1, file=f)
    print('err2 valid error:', err2, file=f)

# stage 2
model = xgb.XGBRegressor(learning_rate=0.01126, max_depth=4, subsample=0.75, n_jobs=32, tree_method='hist', n_estimators=1000)
model.fit(train_x, train_y, eval_set=val, eval_metric='mae', early_stopping_rounds=10, xgb_model='m2.model')
model.save_model('m2.model')
with open('status.txt', 'a') as f:
    print("stage 2 saved", file=f)

# stage 2 prediciton
prediction_train_2 = model.predict(all_x)
prediction_test_2 = model.predict(test_x)
prediction_valid_2 = model.predict(val_x)
with open('train_stage_2_2.txt', 'w') as f:
    for i in prediction_train_2:
        print(i, file=f)
with open('test_stage_2_2.txt', 'w') as f:
    for i in prediction_test_2:
        print(i, file=f)
err1 = 0.0
err2 = 0.0
for i in range(len(val_x)):
    err1 += 300.0 * np.abs(prediction_valid_2[i] - val_y[i])
    err2 += np.abs(prediction_valid_2[i] - val_y[i]) / val_y[i]
with open('status.txt', 'a') as f:
    print('stage 2 predicted', file=f)
    print('err1 valid error:', err1, file=f)
    print('err2 valid error:', err2, file=f)

