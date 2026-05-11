import numpy as np
import matplotlib.pyplot as plt

f_all = np.load("best_folder/big_test/f.npy")   # shape (pop, 3)
x_all = np.load("best_folder/big_test/x.npy")   # shape (pop, n_params)

# --- Tunable weights for each objective ---
w_flat = 1.0
w_ice  = 1.0
w_hill = 1.0

#scores = w_flat * f_all[:, 0] + w_ice * f_all[:, 1] + w_hill * f_all[:, 2]
scores = np.min(f_all, axis=1)

best_idx = np.argmax(scores)
best_x = x_all[best_idx]
best_f = f_all[best_idx]

print(f"Best individual index : {best_idx}")
print(f"Flat={best_f[0]:.2f}  Ice={best_f[1]:.2f}  Hill={best_f[2]:.2f}  Score={scores[best_idx]:.2f}")

np.save("best_folder/big_test/x_best_alt.npy", best_x)
print("Saved to best_folder/big_test/x_best_alt.npy")

# --- Plot ---
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

mask = np.ones(len(f_all), dtype=bool)
mask[best_idx] = False

ax.scatter(f_all[mask, 0], f_all[mask, 1], f_all[mask, 2], c="steelblue", alpha=0.6, label="Population")
ax.scatter(best_f[0], best_f[1], best_f[2], c="red", s=100, zorder=5, label="Best")

ax.set_xlabel("Flat"); ax.set_ylabel("Ice"); ax.set_zlabel("Hill")
ax.legend()
plt.tight_layout()
plt.show()