import numpy as np
import sys

train_y_name, predict_y_name = sys.argv[1], sys.argv[2]
train_y = np.load(train_y_name)['arr_0']
predict_y = []
with open(predict_y_name, 'r') as f:
    for lines in f:
        li = list(map(float, lines.replace('\n', '').split(',')))
        predict_y.append(li)

err1 = np.array([0.0, 0.0, 0.0])
err2 = np.array([0.0, 0.0, 0.0])
w = [200.0, 1.0, 300.0]
for i in range(len(predict_y)):
    for j in range(3):
        err1[j] += w[j] * np.abs(train_y[i][j] - predict_y[i][j])
        err2[j] += np.abs(train_y[i][j] - predict_y[i][j]) / train_y[i][j]

print("err1:", err1 / len(predict_y))
print("err2:", err2 / len(predict_y))
print("e1:", np.sum(err1) / len(predict_y), ",e2:", np.sum(err2) / len(predict_y)) 

