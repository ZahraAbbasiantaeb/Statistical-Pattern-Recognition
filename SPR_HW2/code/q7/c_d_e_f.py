import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from q7.a import prior_true, prior_false
from q7.b import read_all_rgb_images, normal_true, normal_false, make_file_name
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

test_data_path = '/q7/Dataset/Test/'

test_length = 50


def plot_mask(path, new_mask, begin_index):

    img = plt.imread(path)

    rows, cols = img.shape

    size = rows * cols

    image = (new_mask[begin_index:(begin_index+size)]).reshape(rows, cols)

    plt.imshow(image)

    plt.show()

    return size


def plot_ROC(masks, labels):

    fpr, tpr, threshold = metrics.roc_curve(masks, labels)

    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

    plt.legend(loc='lower right')

    plt.plot([0, 1], [0, 1], 'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    return


def plot_all_mask(mask_path, mask_array):

    directory = os.fsencode(mask_path)

    index = 0

    length = len(os.listdir(directory))

    for i in range(1, length + 1):

        filename = mask_path + make_file_name('mask_', i, '.png')

        tmp = plot_mask(filename, mask_array, index)

        index += tmp

    return


masks, red_col, green_col, blue_col \
    = read_all_rgb_images( (test_data_path+'Images/')  ,(test_data_path+'Masks/'))


labels = []

size = len(masks)

for i in range(0, size):

    if (prior_true * normal_true.pdf([red_col[i], green_col[i], blue_col[i]])>
    prior_false * normal_false.pdf([red_col[i], green_col[i], blue_col[i]])):

        labels.append(1)

    else:

        labels.append(0)

labels = np.array(labels)

print(confusion_matrix(masks, labels))

print(1-accuracy_score(masks, labels))

plot_all_mask((test_data_path+'/Masks/') , labels)

plot_ROC(masks, labels)
