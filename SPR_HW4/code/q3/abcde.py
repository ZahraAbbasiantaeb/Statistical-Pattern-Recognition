import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

N = 1000

mean_1 = [10, 10]
mean_2 = [10, 22]

cov = [[9, 4], [4, 4]]

# Part A)

x_1, y_1 = np.random.multivariate_normal(mean_1, cov, N).T

x_2, y_2 = np.random.multivariate_normal(mean_2, cov, N).T

pca = PCA(n_components=1)

data = []

for i in range(0, N):

    data.append([x_1[i], y_1[i]])

for i in range(0, N):

    data.append([x_2[i], y_2[i]])

data = np.array(data)

# Part B and C)

pca.fit(data)

X_pca = pca.transform(data)

X_new = pca.inverse_transform(X_pca)

fig, ax = plt.subplots()

mu = data.mean(axis=0)

sigma = X_new.std(axis=0).mean() * 2

start, end = mu - sigma * pca.components_[0], mu + sigma * pca.components_[0]

ax.annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))


plt.scatter(x_1, y_1, alpha=0.8)
plt.scatter(x_2, y_2, alpha=0.8)

plt.scatter(X_new[:, 0][0:N], X_new[:, 1][0:N], alpha=0.8)
plt.scatter(X_new[:, 0][N:2*N], X_new[:, 1][N:2*N], alpha=0.8)

plt.legend(['class1', 'class2', 'projected 1', 'projected 2'])
plt.title('PCA')

plt.show()

# part E)
# reconstruction error

print('Reconstruction error is: ')
res = X_new-data

error = np.sum([[x**2+y**2] for x,y in res])/N

print(error)