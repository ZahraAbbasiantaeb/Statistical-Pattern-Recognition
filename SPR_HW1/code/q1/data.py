
import numpy as np

with open('Data/iris.text') as f:
    lines = f.readlines()

F1 = []
F2 = []
F3 = []
F4 = []
id = []

for line in lines:

    tokens = line.split(',')
    F1.append(float(tokens[0]))
    F2.append(float(tokens[1]))
    F3.append(float(tokens[2]))
    F4.append(float(tokens[3]))
    id.append(tokens[4].replace('\n',''))


F1 = np.array(F1)
F2 = np.array(F2)
F3 = np.array(F3)
F4 = np.array(F4)