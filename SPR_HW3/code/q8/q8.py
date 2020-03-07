import numpy as np
import matplotlib.pyplot as plt

pic = plt.imread('/q8/frame_a.jpg')/255

plt.imshow(pic)
plt.show()

train_x = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
print(train_x.shape)


train_y = plt.imread('/q8/frame_a_mask.png')
#
X_ = train_y.shape[0]
Y_ = train_y.shape[1]

train_y = train_y.reshape(train_y.shape[0]*train_y.shape[1])


test_x = plt.imread('/q8/frame_b.jpg')/255
test_x = test_x.reshape(test_x.shape[0]*test_x.shape[1], test_x.shape[2])

print(test_x.shape)


def find_dist(point1, point2):

    return np.linalg.norm(np.array(point1) - np.array(point2))


def KNN(K, data, point, label):

    res = []
    i = 0

    for elem in data:

        res.append((find_dist(elem,  point), label[i]))
        i += 1

    res = np.array(res)

    res = res[np.argsort(res[:, 0])]

    tmp = res[0:K,1]

    tmp =list(tmp)

    return max(tmp, key=tmp.count)


def predict_KNN(data,  K):

    pred = []

    for point in data:
        pred.append(KNN(K, train_x, point, train_y))

    return pred

pred = predict_KNN(test_x, 5)

print(pred)

pred = pred.reshape(X_, Y_)

plt.imshow(pred)

plt.show()