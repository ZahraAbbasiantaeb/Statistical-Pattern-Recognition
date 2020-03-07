import numpy as np
import matplotlib.pyplot as plt

with open('/q6/dense_data.dat') as fp:
    lines = fp.readlines()

data = []

for line in lines:
    data.append(float(line))

N = len(data)

h = 0.075

sigma = 0.06

min_x = min(data)

max_x = max(data)

x = np.arange(0, 1, 0.001).tolist()

y = []

for elem in x:

    count = 0

    for row in data:
        count += np.exp(-1 * (np.linalg.norm(row - elem) ** 2) / (2* (sigma ** 2)))


    y.append(count)

y = np.array(y)

y = y / (N*h)

plt.plot(x, y)

plt.title('')

plt.show()