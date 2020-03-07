import numpy as np
import matplotlib.pyplot as plt

with open('/q6/dense_data.dat') as fp:
    lines = fp.readlines()

data = []

for line in lines:
    data.append(float(line))

N = len(data)

h = 0.0375

min_x = min(data)

max_x = max(data)

x = np.arange(0, 1, 0.001).tolist()

y = []

for elem in x:

    count = 0

    for row in data:

        if(row >= elem - h/2 and row <= elem + h/2):
            count += 1

    y.append(count)

y = np.array(y)

y = y / (N*h)

plt.plot(x, y)

plt.title('')

plt.show()