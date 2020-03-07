from scipy.stats import multivariate_normal

from q2.data import user_data
import numpy as np
import random

for i in range(0, 3):

    index = random.randint(1, len(user_data)+1)
    print('player ID is: '+ str(index))
    X = np.stack((user_data[index]['x'], user_data[index]['y']), axis=0)
    cov = np.cov(X)
    mean = [np.mean(user_data[index]['x']), np.mean(user_data[index]['y'])]

    for i in range(0, 3):
        index2 = random.randint(0, len(user_data[index]['x'])+1)
        print('loc is: '+ str(user_data[index]['x'][index2])+" : "+ str(user_data[index]['y'][index2]))
        print(multivariate_normal.pdf([user_data[index]['x'][index2], user_data[index]['y'][index2]], mean=mean, cov=cov))

