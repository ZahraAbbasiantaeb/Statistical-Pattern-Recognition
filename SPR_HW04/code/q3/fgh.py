import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

N = 1000

mean_1 = [10, 10]
mean_2 = [10, 22]

cov = [[9, 4], [4, 4]]

# Part F

x_1, y_1 = np.random.multivariate_normal(mean_1, cov, N).T

x_2, y_2 = np.random.multivariate_normal(mean_2, cov, N).T

data = []

for i in range(0, N):

    data.append([x_1[i], y_1[i]])

for i in range(0, N):

    data.append([x_2[i], y_2[i]])

data = np.array(data)


def inverse_transform(lda, x):

    if lda.solver == 'lsqr':
        raise NotImplementedError("(inverse) transform not implemented for 'lsqr' "
                                  "solver (use 'svd' or 'eigen').")
    check_is_fitted(lda, ['xbar_', 'scalings_'], all_or_any=any)

    inv = np.linalg.pinv(lda.scalings_)

    x = check_array(x)

    if lda.solver == 'svd':
        x_back = np.dot(x, inv) + lda.xbar_

    elif lda.solver == 'eigen':
        x_back = np.dot(x, inv)

    return x_back


clf = LinearDiscriminantAnalysis()

y = []

for i in range(0, N):
    y.append(1)

for i in range(0, N):
    y.append(2)

# Part G

clf.fit(data, y)

LinearDiscriminantAnalysis(solver='eigen', n_components=1)

# LDA_data = clf.predict(data)

transferred_data = clf.transform(data)

inverse_transfer = inverse_transform(clf, transferred_data)

eigenvector = clf.coef_[0]

fig, ax = plt.subplots()

mu = data.mean(axis=0)

sigma = inverse_transfer.std(axis=0).mean()/2

start, end = mu - sigma * clf.coef_[0], mu + sigma * clf.coef_[0]

plt.scatter(x_1, y_1, alpha=0.8)
plt.scatter(x_2, y_2, alpha=0.8)

plt.scatter(inverse_transfer[:, 0][0:N], inverse_transfer[:, 1][0:N], alpha=0.8)
plt.scatter(inverse_transfer[:, 0][N:2*N], inverse_transfer[:, 1][N:2*N], alpha=0.8)

plt.legend(['class1', 'class2', 'projected 1', 'projected 2'])
plt.title('LDA')

ax.annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))

plt.show()