from groot.model import GrootRandomForest
from groot.util import sklearn_forest_to_xgboost_json
from groot.datasets import load_mnist, load_fashion_mnist
from groot.provably_robust_boosting.wrapper import fit_provably_robust_boosting

from xgbKantchelianAttack import (
    optimal_adversarial_example,
    attack_binary_dataset_epsilon,
    score_dataset,
    attack_epsilon_feasibility,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

sns.set(style="whitegrid")

mnist_epsilon = 0.4
fashion_epsilon = 0.1
n_trees = 500
sample_limit = 500
cached_mnist = False
# dataset = "mnist"
dataset = "fmnist"

output_dir = "out/"
mnist_dir = output_dir + dataset + "_ensembles/"
mnist_normal_path = mnist_dir + "rf.json"
mnist_groot_path = mnist_dir + "groot_rf.json"
mnist_chen_path = mnist_dir + "chen_rf.json"
mnist_provably_path = mnist_dir + "provably_robust_boosting.json"

if dataset == "mnist":
    X, y = load_mnist()[1:3]
    X = X[(y == 2) | (y == 6)]
    y = y[(y == 2) | (y == 6)]
    y = np.where(y == 6, 1, 0)
elif dataset == "fmnist":
    X, y = load_fashion_mnist()[1:3]
    X = X[(y == 7) | (y == 9)]
    y = y[(y == 7) | (y == 9)]
    y = np.where(y == 9, 1, 0)
else:
    raise Exception("dataset should be mnist or fmnist")

X = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=1
)

if not cached_mnist:
    # print("Fitting RF...")
    # normal_rf = RandomForestClassifier(
    #     n_estimators=n_trees,
    #     max_depth=None,
    #     min_samples_split=10,
    #     min_samples_leaf=5,
    #     verbose=True,
    #     random_state=1,
    # )
    # normal_rf.fit(X_train, y_train)
    # sklearn_forest_to_xgboost_json(normal_rf, mnist_normal_path)

    print("Fitting GROOT RF...")
    groot_rf = GrootRandomForest(
        n_estimators=n_trees,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        one_adversarial_class=False,
        attack_model=[mnist_epsilon] * X.shape[1],
        n_jobs=1,
        verbose=True,
        random_state=1,
    )
    groot_rf.fit(X_train, y_train)
    groot_rf.to_xgboost_json(mnist_groot_path)

    # print("Fitting Chen et al. RF...")
    # chen_rf = GrootRandomForest(
    #     n_estimators=n_trees,
    #     max_depth=None,
    #     min_samples_split=10,
    #     min_samples_leaf=5,
    #     one_adversarial_class=False,
    #     attack_model=[mnist_epsilon] * X.shape[1],
    #     verbose=True,
    #     random_state=1,
    #     chen_heuristic=True,
    # )
    # chen_rf.fit(X_train, y_train)
    # chen_rf.to_xgboost_json(mnist_chen_path)

    # print("Fitting provably robust boosting...")
    # fit_provably_robust_boosting(
    #     X_train,
    #     y_train,
    #     n_trees=n_trees,
    #     max_depth=8,
    #     epsilon=mnist_epsilon,
    #     filename=mnist_provably_path,
    #     verbose=True,
    # )

    print("Done fitting.")


# normal_acc = score_dataset(mnist_normal_path, X_test, y_test, sample_limit=10000)
groot_acc = score_dataset(mnist_groot_path, X_test, y_test, sample_limit=10000)
# chen_acc = score_dataset(mnist_chen_path, X_test, y_test, sample_limit=10000)
# provably_acc = score_dataset(mnist_provably_path, X_test, y_test, sample_limit=10000, pred_threshold=0.0)

# normal_adv_acc = attack_epsilon_feasibility(mnist_normal_path, X_test, y_test, mnist_epsilon, sample_limit=sample_limit)
groot_adv_acc = attack_epsilon_feasibility(
    mnist_groot_path, X_test, y_test, mnist_epsilon, sample_limit=sample_limit
)
# chen_adv_acc = attack_epsilon_feasibility(mnist_chen_path, X_test, y_test, mnist_epsilon, sample_limit=sample_limit)
# provably_adv_acc = attack_epsilon_feasibility(mnist_provably_path, X_test, y_test, mnist_epsilon, sample_limit=sample_limit, pred_threshold=0.0)

# print("Accuracy", normal_acc, groot_acc, chen_acc, provably_acc)
# print("Adversarial accuracy", normal_adv_acc, groot_adv_acc, chen_adv_acc, provably_adv_acc)

print(groot_acc, groot_adv_acc)
exit()

with open(f"{mnist_dir}scores.txt", "w") as file:
    file.writelines(
        [
            str(s) + "\n"
            for s in (
                normal_acc,
                groot_acc,
                chen_acc,
                provably_acc,
                normal_adv_acc,
                groot_adv_acc,
                chen_adv_acc,
                provably_adv_acc,
            )
        ]
    )

fig, ax = plt.subplots(1, 2)
ax[0].bar([0, 1, 2, 3], [normal_acc, groot_acc, chen_acc, provably_acc])
ax[1].bar([0, 1, 2, 3], [normal_adv_acc, groot_adv_acc, chen_adv_acc, provably_adv_acc])
plt.savefig(f"{mnist_dir}ensembles_scores.pdf")
plt.close()

if dataset == "mnist":
    plot_samples = [
        (X_test[0], y_test[0]),
        (X_test[2], y_test[2]),
        (X_test[3], y_test[3]),
        (X_test[4], y_test[4]),
    ]
elif dataset == "fmnist":
    plot_samples = [
        (X_test[0], y_test[0]),
        (X_test[1], y_test[1]),
        (X_test[3], y_test[3]),
        (X_test[4], y_test[4]),
    ]

_, ax = plt.subplots(4, 5)

for row, (original, label) in enumerate(plot_samples):
    normal_adv_sample = optimal_adversarial_example(mnist_normal_path, original, label)
    groot_adv_sample = optimal_adversarial_example(mnist_groot_path, original, label)
    chen_adv_sample = optimal_adversarial_example(mnist_chen_path, original, label)
    provably_adv_sample = optimal_adversarial_example(
        mnist_provably_path, original, label
    )

    ax[row][0].imshow(original.reshape(28, 28), cmap="gray")
    ax[row][0].set_title("original")

    ax[row][1].imshow(normal_adv_sample.reshape(28, 28), cmap="gray")
    distance = round(np.linalg.norm(original - normal_adv_sample, ord=np.inf), 3)
    ax[row][1].set_title(f"$L_\infty$ dist. {distance}")

    ax[row][2].imshow(groot_adv_sample.reshape(28, 28), cmap="gray")
    distance = round(np.linalg.norm(original - groot_adv_sample, ord=np.inf), 3)
    ax[row][2].set_title(f"$L_\infty$ dist. {distance}")

    ax[row][3].imshow(chen_adv_sample.reshape(28, 28), cmap="gray")
    distance = round(np.linalg.norm(original - chen_adv_sample, ord=np.inf), 3)
    ax[row][3].set_title(f"$L_\infty$ dist. {distance}")

    ax[row][4].imshow(provably_adv_sample.reshape(28, 28), cmap="gray")
    distance = round(np.linalg.norm(original - provably_adv_sample, ord=np.inf), 3)
    ax[row][4].set_title(f"$L_\infty$ dist. {distance}")

plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout()
plt.savefig(f"{mnist_dir}adversarial_examples.pdf")
plt.savefig(f"{mnist_dir}adversarial_examples.png")
plt.close()
