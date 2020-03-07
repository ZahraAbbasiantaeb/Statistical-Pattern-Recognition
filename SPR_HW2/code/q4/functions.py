import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def read_image(path):

    img = plt.imread(path)

    rows, cols = img.shape

    img_size = rows*cols

    img_1D_vector = img.reshape(img_size)

    img_1D_vector = np.array(img_1D_vector)

    return img_1D_vector


def convert_image_to_array(count, directory):

    train_images = {}

    for i in range(0, count):
        train_images[i]= []

    for filename in os.listdir(directory):

        for i in range(0, count):

            if filename.startswith(str(i)+"_"):
                train_images[i].append(read_image(directory + filename))

    return train_images


def get_center_of_images(train_images, count):

    center = {}

    for i in range(0, count):

        center[i] = np.mean(train_images[i], axis=0)

        img_2D = center[i].reshape(60, 60)

        plt.imshow(img_2D)

        plt.show()

    return center


def find_class(vec, centers):

    dist = np.linalg.norm(vec - centers[0])

    id = 0

    for i in range(1, len(centers)):

        new_dist = np.linalg.norm(vec - centers[i])

        if new_dist< dist:

            dist = new_dist

            id = i

    return id


def classify(test_images, centers, count):

    actual = []
    pred = []

    for i in range(0, count):

        for vec in test_images[i]:
            id = find_class(vec, centers)
            actual.append(i)
            pred.append(id)

    print(confusion_matrix(actual, pred))

    return actual, pred


