import numpy as np
import sys

train_y_name, predict_y_name, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
train_y = np.load(train_y_name)['arr_0']
predict_y = []
with open(predict_y_name, 'r') as f:
    for lines in f:
        li = float(lines.replace('\n', ''))
        predict_y.append(li)

err1 = 0.0
err2 = 0.0
w = [200.0, 1, 300.0]
for i in range(len(predict_y)):
    err1 += w[idx] * np.abs(train_y[i][idx] - predict_y[i])
    err2 += np.abs(train_y[i][idx] - predict_y[i]) / train_y[i][idx]

print("err1:", err1 / len(predict_y))
print("err2:", err2 / len(predict_y))

