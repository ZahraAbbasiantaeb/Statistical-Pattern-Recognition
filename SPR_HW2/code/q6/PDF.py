import numpy as np
import matplotlib.pyplot as plt
from q6.ROC import retrieve_data

filepath = '/q6/Datasets/dataset_1.txt'


def get_PDF(dataset):

    x_min = min(dataset)

    x_max =  max(dataset)

    threshold = 0.05

    range_ = np.arange(x_min, x_max, threshold).tolist()

    Point = []

    Dist = []

    for elem in range_:

        Point.append((elem+threshold/2))
        count = 0

        for num in dataset:

            if (num >= elem and num < elem + threshold ):
                count += 1

        Dist.append(count)


    return Point, Dist


for i in range(1, 5):

    filepath = '/q6/Datasets/dataset_' + str(i) + '.txt'

    feature1, feature2 = retrieve_data(filepath)

    Point, Dist = get_PDF(feature1)

    plt.plot(Point, Dist)

    Point, Dist = get_PDF(feature2)

    plt.plot(Point, Dist)

    plt.xlabel('False Positive')

    plt.ylabel('True Positive')

    plt.title(('PDF of dataset '+ str(i)))

    plt.legend(('class_1', 'class_2'))

    plt.show()

