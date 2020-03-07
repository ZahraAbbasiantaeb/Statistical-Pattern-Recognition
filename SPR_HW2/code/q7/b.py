import numpy as np
from scipy.stats import multivariate_normal
from q4.functions import read_image
import os
import matplotlib.pyplot as plt


def read_image_rgb(path):

    img = plt.imread(path)

    rows, cols, r = img.shape

    red = img[:,:,0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    img_size = rows*cols

    red = red.reshape(img_size)
    green = green.reshape(img_size)
    blue = blue.reshape(img_size)

    return red, green, blue


# def rescale(red_color, green_color, blue_color):
#
#     for i in range(0, len(red_color)):
#
#         tmp = red_color[i] + green_color[i] + blue_color[i]
#
#         if(tmp > 0):
#
#             red_color[i] = red_color[i] / tmp
#             green_color[i] = green_color[i] / tmp
#             blue_color[i] = blue_color[i] / tmp
#
#     return red_color, green_color, blue_color


def make_file_name(param, i, param1):

    if(i<=9):

        return param+'0'+str(i)+ param1

    else:

        return param+str(i)+ param1


def read_all_rgb_images(img_path, img_mask):

    masks = []
    red_color = []
    green_color = []
    blue_color = []


    directory = os.fsencode(img_path)

    length = len(os.listdir(directory))

    for i in range(1, length+1):

        filename = img_path + make_file_name('img_', i, '.jpg')
        red, green, blue = read_image_rgb(filename)
        red_color = np.append(red_color, red)
        blue_color = np.append(blue_color, blue)
        green_color = np.append(green_color, green)

    directory = os.fsencode(img_mask)

    length = len(os.listdir(directory))

    for i in range(1, length+1):

        filename = img_mask + make_file_name('mask_', i, '.png')
        mask = read_image(filename)
        masks = np.append(masks, mask)

    # red_color, green_color, blue_color = rescale(red_color, green_color, blue_color)

    return masks, red_color, green_color, blue_color


masks, red_color, green_color, blue_color = read_all_rgb_images('/q7/Dataset/Train/Images/'
                                                                ,'/q7/Dataset/Train/Masks/')


red_true = []
green_true = []
blue_true = []


red_false = []
green_false = []
blue_false = []


for i in range(0, len(masks)):

    if masks[i] == 1:

        red_true.append(red_color[i])
        blue_true.append(blue_color[i])
        green_true.append(green_color[i])

    else:

        red_false.append(red_color[i])
        blue_false.append(blue_color[i])
        green_false.append(green_color[i])

cov_true = np.cov([red_true, green_true, blue_true])
cov_false = np.cov([red_false, green_false, blue_false])

mean_true = [np.mean(red_true), np.mean(green_true), np.mean(blue_true)]
mean_false = [np.mean(red_false), np.mean(green_false), np.mean(blue_false)]

normal_true = multivariate_normal(mean_true, cov_true)
normal_false = multivariate_normal(mean_false, cov_false)

print(cov_true)
print(cov_false)

print(mean_true)
print(mean_false)
