import numpy as np

text_file = open("/q7/wine_quality.csv", "r")

train_data = []

lines = text_file.readlines()

for line in lines:

    tmp  = line.split(';')

    res = [float(elem.strip()) for elem in tmp]

    train_data.append(res)


test = [[7.5, 0.9, 0.26, 2.3, 0.054, 19, 13, 0.99708, 3.78, 0.55, 9.7],
        [5.4, 0.78, 0.17, 3.2, 0.084, 11, 58, 0.9987, 2.94, 0.83, 11.8],
        [8.2, 0.56, 0.46, 1.7, 0.069, 25, 15, 0.997, 3.39, 0.65, 12.5],
        [6.0, 0.7, 0.01, 4.6, 0.093, 6, 104, 0.99746, 3.12, 0.52, 10.5],
        [10.8, 0.43, 0.31, 2.5, 0.105, 35, 31, 1.0001, 3.22, 0.48, 11.1]]


def find_dist(point1, point2):

    return np.linalg.norm(np.array(point1) - np.array(point2))


def KNN(K, data, point):

    res = []

    for elem in data:
        res.append((find_dist(elem[0:11], point), elem[11]))

    res = np.array(res)

    res = res[np.argsort(res[:, 0])]

    tmp = res[0:K,1]

    tmp =list(tmp)

    return max(tmp, key=tmp.count)

def predict_KNN(data, K):

    label = []

    for point in data:
        label.append(KNN(K, train_data, point))

    return label


label = predict_KNN(test, 5)

print(label)

label = predict_KNN(test, 3)

print(label)

label = predict_KNN(test, 1)

print(label)
