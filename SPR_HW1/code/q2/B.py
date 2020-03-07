import numpy as np
from scipy.stats import multivariate_normal
from q2.data import user_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_binomial_distribution(data, i):

    x, y = np.mgrid[0:145:.5, 0:90:.5]

    pos = np.empty(x.shape + (2,))

    pos[:, :, 0] = x

    pos[:, :, 1] = y

    X = np.stack((data['x'], data['y']), axis=0)

    cov = np.cov(X)

    mean = [np.mean(data['x']), np.mean(data['y'])]

    bnormald = multivariate_normal(mean, cov)

    text = 'player ' + str(i)

    z = bnormald.pdf(pos)

    plt.contourf(x, y, z, label=text)

    plt.xlabel('X')

    plt.ylabel('Y')

    plt.legend()

    plt.show()

    fig = plt.figure()

    ax = Axes3D(fig)

    ax.scatter(x, y, z, label=text)

    ax.set_xlabel('X')

    ax.set_ylabel('Y')

    ax.set_zlabel('Prob')

    ax.legend()

    plt.show()

    return


for i in user_data:

    plot_binomial_distribution(user_data[i], i)
