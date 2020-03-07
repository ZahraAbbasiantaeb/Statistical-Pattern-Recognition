import numpy as np

X = [2,3,5,6,6,7,8,8,8,11,12,12,14,18,20,20]

Y = [4,9,14]

prob =[]

for y in Y:

    count  = 0

    for x in X:
        count += 1/(np.sqrt(2*np.pi))*np.exp(-1 * (np.linalg.norm(x - y) ** 2)/2)

    prob.append(count)

    print(count)

