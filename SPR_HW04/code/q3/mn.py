import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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

def return_given_classes(X, Y, classes):

    new_X = []

    new_Y = []

    for j in range(0, len(X)):

        if Y[j] in classes:

            new_X.append(X[j])

            new_Y.append(Y[j])

    return new_X, new_Y

train_2c, train_label_2c = return_given_classes(x_train, y_train, [2,6])

test_2c, test_label_2c = return_given_classes(x_test, y_test, [2,6])

pca = PCA(n_components=170)

pca.fit(train_2c)

var = pca.explained_variance_ratio_

# Let's first find the enough components for covering 90% variance

def find_90_percent_var(var):

    cum_var = 0

    for i in range(0, len(var)):
        cum_var += var[i]

        if (cum_var >= 0.9):
            return (i +1 )

    return len(var)

components = find_90_percent_var(var)

pca.fit(train_2c)

X_pca_train = pca.transform(train_2c)

X_pca_test = pca.transform(test_2c)

lda = LinearDiscriminantAnalysis()

lda.fit(X_pca_train, train_label_2c)

LinearDiscriminantAnalysis(solver='eigen', n_components=1)

one_d_train = lda.transform(X_pca_train)

y = np.zeros(len(one_d_train))

y_pred_train = lda.predict(X_pca_train)

print('train confusion matrix: ')

print(confusion_matrix(train_label_2c, y_pred_train))

print(accuracy_score(train_label_2c, y_pred_train))

y_test_pred = lda.predict(X_pca_test)

print('test confusion matrix: ')

print(confusion_matrix(test_label_2c, y_test_pred))

print(accuracy_score(test_label_2c, y_test_pred))


# Part N

def Knn(point, train_data, train_label, K):

    res = []

    for elem, label in zip(train_data, train_label):
        res.append([np.abs(elem[0]-point), label])

    res = sorted(res, key=lambda x: x[0])

    res = np.array(res)

    labels = []

    for k in K:

        labels.append(np.argmax(np.bincount(res[0:k, 1].astype(int))))

    return  np.array(labels)


def Knn_for_training(point, train_data, train_label, K, index):

    res = []
    i = 0
    for elem, label in zip(train_data, train_label):

        if(i != index):

            res.append([np.abs(elem[0]-point), label])

        i+=1

    res = sorted(res, key=lambda x: x[0])

    res = np.array(res)

    labels = []

    for k in K:

        labels.append(np.argmax(np.bincount(res[0:k, 1].astype(int))))

    return  np.array(labels)


# one_d_test = lda.transform(X_pca_test)
#
# pred_knn_train = []
#
# k = [1, 3, 5, 7, 9]
#
# for point in one_d_test:
#
#     pred_knn_train.append(Knn(point[0], one_d_train, train_label_2c, k))
#
# pred_knn_train= np.array(pred_knn_train)
#
# print(confusion_matrix(test_label_2c, pred_knn_train[:,0]))
# print(confusion_matrix(test_label_2c, pred_knn_train[:,1]))
# print(confusion_matrix(test_label_2c, pred_knn_train[:,2]))
# print(confusion_matrix(test_label_2c, pred_knn_train[:,3]))
# print(confusion_matrix(test_label_2c, pred_knn_train[:,4]))

pred_knn_train = []
k = [1, 3, 5, 7, 9]
index = 0

one_d_train = np.array(one_d_train)

for point in one_d_train:
    pred_knn_train.append(Knn_for_training(point[0], one_d_train, train_label_2c, k, index))

    index += 1

pred_knn_train= np.array(pred_knn_train)

print(confusion_matrix(train_label_2c, pred_knn_train[:,0]))
print(accuracy_score(train_label_2c, pred_knn_train[:,0]))

print(confusion_matrix(train_label_2c, pred_knn_train[:,1]))
print(accuracy_score(train_label_2c, pred_knn_train[:,1]))

print(confusion_matrix(train_label_2c, pred_knn_train[:,2]))
print(accuracy_score(train_label_2c, pred_knn_train[:,2]))

print(confusion_matrix(train_label_2c, pred_knn_train[:,3]))
print(accuracy_score(train_label_2c, pred_knn_train[:,3]))

print(confusion_matrix(train_label_2c, pred_knn_train[:,4]))
print(accuracy_score(train_label_2c, pred_knn_train[:,4]))


def plot_image(img):

    plt.imshow(img.reshape(row, col), cmap='gray')
    plt.show()

    return

for i in range(0, len(train_label_2c)):

    if train_label_2c[i] == 6 and pred_knn_train[i,3] == 2:
        plot_image(train_2c[i])
        break


for i in range(0, len(train_label_2c)):

    if train_label_2c[i] == 2 and pred_knn_train[i,3] == 6:
        plot_image(train_2c[i])
        break