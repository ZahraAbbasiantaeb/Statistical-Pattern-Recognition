import numpy as np
import matplotlib.pyplot as plt

with open('/q6/dense_data.dat') as fp:
    lines = fp.readlines()

data = []

for line in lines:
    data.append(float(line))

N = len(data)

K = 30

x = np.arange(0, 1, 0.001).tolist()

def find_KNN(elem, data):

    res = []

    for row in data:

        res.append((row, np.linalg.norm(np.array(row) - np.array(elem))))

    res = np.array(res)

    res = res[np.argsort(res[:, 1])]

    tmp = res[0:K,0]

    tmp = list(tmp)

    return tmp

y = []

for elem in x:

    arr = find_KNN(elem, data)
    h = max(arr) - min(arr)
    y.append(h)

y = np.array(y)

y = (y * K) / N

plt.plot(x, y)

plt.show()