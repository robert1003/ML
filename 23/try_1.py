import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

test_x = np.load('../../X_test.npz')['arr_0']
train_x = np.load('../../X_train.npz')['arr_0']
train_y = np.load('../../Y_train.npz')['arr_0'][:, 1]
all_x = train_x.copy()

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=1126)
val = [(val_x, val_y)]

# stage 1
model = xgb.XGBRegressor(learning_rate=0.1126, max_depth=5, subsample=0.75, n_jobs=32, tree_method='hist', n_estimators=1000)
'''
model.fit(train_x, train_y, eval_set=val, eval_metric='mae', early_stopping_rounds=6)
model.save_model('m1.model')
'''
model.load_model('m1.model')
'''
# stage 1 prediction
prediction_train_1 = model.predict(all_x)
prediction_test_1 = model.predict(test_x)
print("satge 1 over")
# stage 2
model = xgb.XGBRegressor(learning_rate=0.01126, max_depth=5, subsample=0.75, n_jobs=32, tree_method='hist', n_estimators=1000)
model.fit(train_x, train_y, eval_set=val, eval_metric='mae', early_stopping_rounds=10, xgb_model='m1.model')
model.save_model('m1.model')

# stage 2 prediciton
'''
prediction_train_2 = model.predict(all_x)
prediction_test_2 = model.predict(test_x)

with open('train_stage_2_1.txt', 'w') as f:
    for i in prediction_train_2:
        print(i, file=f)
with open('test_stage_2_1.txt', 'w') as f:
    for i in prediction_test_2:
        print(i, file=f)

