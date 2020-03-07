import matplotlib.pyplot as plt

from q1.data import F1, F4, id, F3, F2

plt.scatter(F1, F4, alpha=0.8, c=id)
plt.title('feature1 and feature4')
plt.show()


plt.scatter(F2, F3, alpha=0.8, c=id)
plt.title('feature2 and feature3')
plt.show()

plt.scatter(F3, F4, alpha=0.8, c=id)
plt.title('feature3 and feature4')
plt.show()

