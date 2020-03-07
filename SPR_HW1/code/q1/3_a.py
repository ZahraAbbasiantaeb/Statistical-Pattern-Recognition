import matplotlib.pyplot as plt
from q1.data import F1, F2, F3, F4, id

plt.scatter(F1, F2, alpha=0.8, c=id)
plt.title('feature1 and feature2')
plt.show()


plt.scatter(F1, F3, alpha=0.8, c=id)
plt.title('feature1 and feature3')
plt.show()

plt.scatter(F2, F4, alpha=0.8, c=id)
plt.title('feature2 and feature4')
plt.show()

