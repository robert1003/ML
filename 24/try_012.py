import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

test_x = np.load('../../X_test.npz')['arr_0']
train_x = np.load('../../X_train.npz')['arr_0']
train_y = np.load('../../Y_train.npz')['arr_0']

# stage 1
model = RandomForestRegressor(n_estimators=1000, criterion='mae', max_features='sqrt', bootstrap=True, oob_score=True, n_jobs=32, random_state=1126, verbose=1, warm_start=False)
model.fit(train_x, train_y)
dump(model, 'm012.joblib')

# stage 1 prediction
prediction_train_1 = model.predict(all_x)
prediction_test_1 = model.predict(test_x)

with open('train_stage_1_0.txt', 'w') as f:
    for i in range(len(prediction_train_1)):
        print(i, file=f)
with open('test_stage_1_0.txt', 'w') as f:
    for i in range(len(prediction_test_1)):
        print(i, file=f)
