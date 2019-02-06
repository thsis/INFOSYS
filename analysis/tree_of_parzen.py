import os
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt

figpath = os.path.join("blog", "pictures", "tpe.png")

centers = [[50, 160], [150, 90]]
X, _ = make_blobs(n_samples=200, centers=centers, n_features=2,
                  random_state=0, cluster_std=15.0)
good = X[X[:, 1] < 120]
bad = X[X[:, 1] >= 120]

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
fig.suptitle("Tree of Parzen Estimators")
ax[0].scatter(x=good[:, 0], y=good[:, 1], c="blue")
ax[0].scatter(x=bad[:, 0], y=bad[:, 1], c="red")
ax[0].axhline(y=120, xmin=-100, xmax=100, linestyle="--", c="k")
ax[0].set_xlim([0, 250])
ax[0].set_title("Random Forest Results")
ax[0].set_xlabel("Number of Estimators")
ax[0].set_ylabel("Loss")

sns.kdeplot(good[:, 0], c="blue", ax=ax[1], label=r"$l(x)$")
sns.kdeplot(bad[:, 0], c="red", ax=ax[1], label=r"$g(x)$")
ax[1].set_title("Probability Distributions")
ax[1].set_xlabel("Number of Estimators")
ax[1].set_xlim([0, 250])
ax[1].set_ylabel("Density")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend()
plt.savefig(figpath)
