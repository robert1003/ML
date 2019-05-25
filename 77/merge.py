import numpy as np

a = np.zeros((2500, 3))

idd=0
with open('y0name.txt', 'r') as f:
    for name in f:
        with open('../71/' + name[:-1], 'r') as ff:
            i = 0
            for num in ff:
                a[i][0] += float(num)/(20 if idd<12 else 10)
                i += 1
        idd += 1

with open('y1name.txt', 'r') as f:
    for name in f:
        with open('../72/' + name[:-1], 'r') as ff:
            i = 0
            for num in ff:
                a[i][1] += float(num)/10
                i += 1

with open('y2name.txt', 'r') as f:
    for name in f:
        with open('../73/' + name[:-1], 'r') as ff:
            i = 0
            for num in ff:
                a[i][2] += float(num)/10
                i += 1

import pandas as pd
a = pd.DataFrame(a)
a.to_csv('all.csv', index=False)
