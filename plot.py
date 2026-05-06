import numpy as np
import matplotlib.pyplot as plt

f_all = np.load("results/final_test/1/f.npy")   # shape (pop, 3)
x_all = np.load("results/final_test/1/x.npy")   # shape (pop, n_params)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(f_all[:, 0], f_all[:, 1], f_all[:, 2])
ax.set_xlabel("Flat"); ax.set_ylabel("Ice"); ax.set_zlabel("Hill")
plt.show()