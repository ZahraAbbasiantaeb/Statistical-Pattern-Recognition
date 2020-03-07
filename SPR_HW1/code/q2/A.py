import numpy as np
from q2.data import user_data
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for i in user_data:
    color = "player" + str(i)
    mean = [np.mean(user_data[i]['x']), np.mean(user_data[i]['y'])]
    print(i)
    print(mean)
    ax.scatter(np.mean(user_data[i]['x']), np.mean(user_data[i]['y']), label=color, alpha=0.8
               , edgecolors='none')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.show()
