import sys
name_0, name_1, name_2, name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

y = [[], [], []]

with open(name_0, 'r') as f:
    for l in f:
        y[0].append(float(l.replace('\n', '')))
with open(name_1, 'r') as f:
    for l in f:
        y[1].append(float(l.replace('\n', '')))
with open(name_2, 'r') as f:
    for l in f:
        y[2].append(float(l.replace('\n', '')))

with open(name, 'w') as f:
    for i in range(len(y[0])):
        print(y[0][i], y[1][i], y[2][i], sep=',', file=f)
