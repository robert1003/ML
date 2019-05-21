import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_x = np.load('../../X_test.npz')['arr_0']
train_x = np.load('../../X_train.npz')['arr_0']
train_y = np.load('../../Y_train.npz')['arr_0']
print(f'test_x shape: {test_x.shape}, train_x shape: {train_x.shape}, train_y shape: {train_y.shape}')
train_num = train_x.shape[0]
tmp_x = np.concatenate((train_x, test_x), axis=0)
print(f'concatenated: {tmp_x.shape}')

fig = plt.figure(figsize=(30, 10))
for i in range(10000):
    if i % 100 == 0:
        print(i)
    plt.subplot(2500, 4, (i + 1))
    sns.distplot(tmp_x[:, i], norm_hist=False, kde=False)
    plt.title(f'feature {i + 1}')
fig.savefig(f'distplot.pdf', bbox_inches='tight')
