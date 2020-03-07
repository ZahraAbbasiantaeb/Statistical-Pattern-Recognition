import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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


# Part J

def return_given_classes(X, Y, classes):

    new_X = []

    new_Y = []

    for j in range(0, len(X)):

        if Y[j] in classes:

            new_X.append(X[j])

            new_Y.append(Y[j])

    return new_X, new_Y

def rotate( point, angle):

    px, py = point

    qx =  math.cos(angle) * (px) - math.sin(angle) * (py)
    qy =  math.sin(angle) * (px) + math.cos(angle) * (py)

    return np.array([qx, qy])


train_2c, train_label_2c = return_given_classes(x_train, y_train, [2,6])

test_2c, test_label_2c = return_given_classes(x_test, y_test, [2,6])

pca = PCA(n_components=2)

pca.fit(train_2c)

X_pca_train = pca.transform(train_2c)

X_pca_test = pca.transform(test_2c)

lda = LinearDiscriminantAnalysis()

lda.fit(X_pca_train, train_label_2c)

LinearDiscriminantAnalysis(solver='eigen', n_components=1)

one_d_data = lda.transform(X_pca_train)

y = np.zeros(len(one_d_data))

y_pred_train = lda.predict(X_pca_train)

print('train confusion matrix: ')

print(confusion_matrix(train_label_2c, y_pred_train))

print(accuracy_score(train_label_2c, y_pred_train))

y_test_pred = lda.predict(X_pca_test)

print('test confusion matrix: ')

print(confusion_matrix(test_label_2c, y_test_pred))

print(accuracy_score(test_label_2c, y_test_pred))

# Part K

fig, ax = plt.subplots()

plt.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c = train_label_2c)

vec = rotate(lda.coef_[0], 90)

start, end = - vec *100000, + vec*100000

ax.annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))
plt.show()