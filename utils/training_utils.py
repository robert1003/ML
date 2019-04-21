import numpy as np

# load data
def load_data(idx):
    test_x = np.load('/tmp2/b07902047/X_test.npz')['arr_0']
    train_x = np.load('../../X_train.npz')['arr_0']
    train_y = np.load('../../Y_train.npz')['arr_0']
    assert idx >= 0 and idx <= 3
    if idx != 3:
        train_y = train_y[:, idx]
    return (test_x, train_x, train_y)

# predict data
def predict(model, *args):
    result = []
    for arg in args:
        result.append(model.predict(arg))
    return result

# calculate type 1 error
w = [200.0, 1.0, 300.0]
def err1_calc(predict, real, idx):
    return np.sum(w[idx] * np.abs(predict - real)) / len(predict)

# calculate type 2 errr
def err2_calc(predict, real):
    return np.sum(np.abs(predict - real) / real) / len(predict)

# write status
def write_status(name, mode, message):
    assert mode == 'a' or mode == 'w'
    with open(name, mode) as f:
        print(message, file=f)

# write prediction
def write_prediction(name, mode, data):
    assert mode == 'a' or mode == 'w'
    with open(name, mode) as f:
        for lines in data:
            print(','.join(list(lines)), file=f)
