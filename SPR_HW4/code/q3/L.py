import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import math
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

def find_most_incorrectly_classified(pred_val, act_label):

    pant_idx = 0
    pant_val = 0

    eyeglass_idx = 0
    eyeglass_val = 0

    for i in range(0, len(act_label)):

        if act_label[i] == 2:
            if (pred_val[i] > pant_val):
                pant_val = pred_val[i]
                pant_idx = i

        elif act_label[i] == 6:
            if (pred_val[i] < eyeglass_val):
                eyeglass_val = pred_val[i]
                eyeglass_idx = i

    print(pant_val)
    print(eyeglass_val)

    return pant_idx, eyeglass_idx


def plot_image(img):

    plt.imshow(img.reshape(row, col), cmap='gray')
    plt.show()

    return

# Find and plot the most incorrectly classified object in test set

test_pred_val = lda.decision_function(X_pca_test)


pant_idx, eyeglass_idx = find_most_incorrectly_classified(test_pred_val, test_label_2c)

print(test_label_2c[pant_idx])

print(test_label_2c[eyeglass_idx])

plot_image(x_test[pant_idx])

plot_image(x_test[eyeglass_idx])
