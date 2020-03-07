import matplotlib.pyplot as plt
from q1.data import F1, F4, id, F2, F3

def plotTransposedFeatures(F1, F2, group, T1, T2):

    plt.scatter(T1*F1+T2*F2, group, alpha=0.8, c=group)
    plt.show()

    return

plotTransposedFeatures(F3, F4, id, -3, 1)



