from groot.model import GrootTree
from groot.visualization import plot_adversary
from groot.adversary import DecisionTreeAdversary

from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white", context="paper")

attack_models = [
    (0, 0),
    (0.1, 0.1),
    ("<", 0.1),
    (0.1, "<>"),
]

X, y = make_moons(n_samples=100, noise=0.3, random_state=1)
X = MinMaxScaler().fit_transform(X)

_, ax = plt.subplots(2, 2, figsize=(5, 5))

for i, attack_model in enumerate(attack_models):
    tree = GrootTree(max_depth=3, attack_model=attack_model, random_state=1)
    tree.fit(X, y)

    adversary = DecisionTreeAdversary(tree, "groot")
    print(adversary.adversarial_accuracy(X, y))
    axis = ax[i // 2, i % 2]
    axis.set_title(str(attack_model), {"fontsize": 12.0})
    plot_adversary(X, y, adversary, ax=axis)

plt.tight_layout()
plt.savefig("out/threat_models_vis.png", dpi=200)
plt.show()
