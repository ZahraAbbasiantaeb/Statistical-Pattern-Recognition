import numpy as np
import matplotlib.pyplot as plt


# X = [-8, -7, -5, -2, 0, 2, 3, 4, 5, 7]
#
# Y = []
#
# range_= np.arange(-10, 10, 0.001)
#
# H = 1/np.sqrt(11)
#
# for elem in range_:
#     count = 0
#
#     for point in X:
#
#         if(np.abs(point-elem)<= H/2):
#             count += 1
#
#     if(count>1):
#         print(elem)
#
#     Y.append(count)
#
# plt.plot(range_, Y)
#
# plt.show()
#
# for i in range(1, len(Y)):
#     if Y[i]!= Y[i-1]:
#         print(range_[i])
#

from sklearn.neighbors import KNeighborsClassifier

Y = [2,3,5,6.5,7.5,8,8.5,9,9.2,9.5,10,10.8,11.2,11.3,13,15]

N = len(Y)

K_s = [3,4,5]

point_s = np.arange(-2, 20, 0.001)



for K in K_s:

    Y_ =[]

    for point in point_s:

        dist = []

        for elem in Y:

            dist.append((elem, np.abs(point-elem)))

        dist = np.array(dist)

        dist = dist[np.argsort(dist[:, 1])]

        V = np.max(dist[0:K, 1])

        prob = K/(N*V)

        Y_.append(prob)

    plt.plot(point_s, Y_)

    plt.show()







