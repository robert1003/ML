import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

test_x = np.load('../../X_test.npz')['arr_0']
train_x = np.load('../../X_train.npz')['arr_0']
train_y = np.load('../../Y_train.npz')['arr_0']
print(f'test_x shape: {test_x.shape}, train_x shape: {train_x.shape}, train_y shape: {train_y.shape}')
train_num = train_x.shape[0]
tmp_x = np.concatenate((train_x, test_x), axis=0)
print(f'concatenated: {tmp_x.shape}')

with PdfPages('distplot.pdf') as pdf:
    for i in range(625):
        plt.figure(figsize=(20, 10))
        plt.subplots_adjust(wspace=None, hspace=0.3)
        for j in range(16 * i, 16 * (i + 1)):
            plt.subplot(4, 4, j % 16 + 1)
            sns.distplot(tmp_x[:, j], norm_hist=False, kde=False)
            plt.title(f'feature {j + 1}')
        pdf.savefig()
        plt.close()
