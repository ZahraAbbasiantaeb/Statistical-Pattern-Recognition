import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

train_data = np.load('/q3/dataset/train-ubyte.npz')
test_data  = np.load('/q3/dataset/test-ubyte.npz')

x_train, y_train = train_data['a'], train_data['b']
x_test,  y_test  = test_data['a'],  test_data['b']

row = np.shape(x_train)[1]
col = np.shape(x_train)[2]

x_train = x_train.reshape(np.shape(x_train)[0],
                          np.shape(x_train)[1]*np.shape(x_train)[2])

x_test = x_test.reshape(np.shape(x_test)[0],
                          np.shape(x_test)[1]*np.shape(x_test)[2])

fig, ax = plt.subplots()


# Part I

pca = PCA(n_components=2)

pca.fit(x_train)

X_pca = pca.transform(x_train)

print(set(y_train))

print(np.shape(X_pca))

for i in range(0, 10):

    data = []

    for j in range(0, len(X_pca)):

        if y_train[j] == i:

            data.append(X_pca[j])

    data = np.array(data)

    plt.scatter(data[:, 0], data[:, 1])

plt.legend(['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'])

plt.show()

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)

plt.show()