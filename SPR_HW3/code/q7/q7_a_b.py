import numpy as np
import matplotlib.pyplot as plt

fp = open('/q7/animals_weight.txt', 'r')

lines = fp.readlines()

brain_w = []
body_w = []

for line in lines:

    tmp = line.split(',')
    brain_w.append(float(tmp[1]))
    body_w.append(float(tmp[2].strip()))



def Knn_for_regression(k, sample):

    res = []

    for i in range (0, len(brain_w)):
        res.append((body_w[i], np.abs(brain_w[i]-sample)))

    res = np.array(res)

    res = res[np.argsort(res[:, 1])]

    return np.mean(res[0:k,0])


def apply_knn(input, k):

    body_w_test = []

    for point in input:
        body_w_test.append(Knn_for_regression(k, point))

    return body_w_test


brain_w_test = [0.041, 0.583, 4.859, 53.298, 1247.122]


body_w_test = apply_knn(brain_w_test, 5)

print(body_w_test)

plt.plot(brain_w_test, body_w_test)

body_w_test = apply_knn(brain_w_test, 3)

print(body_w_test)

plt.plot(brain_w_test, body_w_test)

body_w_test = apply_knn(brain_w_test, 1)

print(body_w_test)

plt.plot(brain_w_test, body_w_test)

plt.legend(['5-NN', '3-NN', '1-NN'])

plt.show()