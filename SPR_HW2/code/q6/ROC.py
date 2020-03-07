import numpy as np
import matplotlib.pyplot as plt


def retrieve_data(filepath):

    feature1 = []
    feature2 = []

    with open(filepath) as fp:

        lines = fp.readlines()
        for i in range(0, 1000):

            line = lines[i]
            tmp = line.strip().split('   ')
            feature1.append(float(tmp[0]))
            feature2.append(float(tmp[1]))

    return feature1, feature2


def get_TP(x_star):

    count = 0

    for elem in feature2:

        if elem>x_star:
            count += 1

    return count/n


def get_FP(x_star):

    count = 0

    for elem in feature1:

        if elem>x_star:
            count += 1

    return count/n


def get_ROC_value():

    x_min = min(min(feature2), min(feature1))

    x_max = max(max(feature2), max(feature1))

    threshold = 0.05

    range_ = np.arange(x_min, x_max, threshold).tolist()

    TP = []

    FP = []

    for elem in range_:
        TP.append(get_TP(elem))
        FP.append(get_FP(elem))

    return FP, TP


for i in range(1, 5):

    filepath = '/q6/Datasets/dataset_'+str(i)+'.txt'

    feature1, feature2 = retrieve_data(filepath)

    d_prime = np.abs(np.mean(feature2) - np.mean(feature1)) / np.sqrt(np.var(feature1) + np.var(feature2))

    print(d_prime)

    n = len(feature2)

    FP, TP = get_ROC_value()

    plt.plot(FP, TP)



plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC Curve')
plt.legend(('dataset_1', 'dataset_2', 'dataset_3', 'dataset_4'))
plt.show()