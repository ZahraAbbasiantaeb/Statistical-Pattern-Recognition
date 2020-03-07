import matplotlib.pyplot as plt
from q1.data import F1, F2, id, F4, F3


def plotTransposedFeatures(F1, F2, group, T1, T2):
    # X = range(1, len(F1)+1)
    plt.scatter(T1*F1+T2*F2, group, alpha=0.8, c=group)
    plt.show()

    return

plotTransposedFeatures(F1, F2, id, 1.2, -0.3)
plotTransposedFeatures(F1, F2, id, -1.8, 0.6)
plotTransposedFeatures(F1, F2, id, 1.4, 0.5)
plotTransposedFeatures(F1, F2, id, -0.5, -1.0)


plotTransposedFeatures(F1, F3, id, 1.2, -0.3)
plotTransposedFeatures(F1, F3, id, -1.8, 0.6)
plotTransposedFeatures(F1, F3, id, 1.4, 0.5)
plotTransposedFeatures(F1, F3, id, -0.5, -1.0)


plotTransposedFeatures(F2, F4, id, 1.2, -0.3)
plotTransposedFeatures(F2, F4, id, -1.8, 0.6)
plotTransposedFeatures(F2, F4, id, 1.4, 0.5)
plotTransposedFeatures(F2, F4, id, -0.5, -1.0)