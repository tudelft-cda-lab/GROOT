from groot.model import GrootTree
from groot.datasets import epsilon_attacker, load_all
from groot.treant import RobustDecisionTree
from groot.util import sklearn_tree_to_xgboost_json

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from multiprocessing import Pool

import time

sns.set_style("whitegrid")

MAX_DEPTH = 4


def train_groot_tree(X, y, epsilon, filename):
    attack_model = [epsilon] * X.shape[1]

    groot_tree = GrootTree(
        max_depth=MAX_DEPTH,
        attack_model=attack_model,
        one_adversarial_class=False,
        random_state=1,
    )

    start_time = time.time()
    groot_tree.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime)

    groot_tree.to_xgboost_json(filename)

    return runtime


def train_chen_tree(X, y, epsilon, filename):
    attack_model = [epsilon] * X.shape[1]

    chen_tree = GrootTree(
        max_depth=MAX_DEPTH,
        attack_model=attack_model,
        one_adversarial_class=False,
        chen_heuristic=True,
        random_state=1,
    )

    start_time = time.time()
    chen_tree.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime)

    chen_tree.to_xgboost_json(filename)

    return runtime


def train_sklearn_tree(X, y, _, filename):
    tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=1,)

    start_time = time.time()
    tree.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime)

    sklearn_tree_to_xgboost_json(tree, filename)

    return runtime


def train_treant_tree(X, y, epsilon, filename):
    attacker = epsilon_attacker(X.shape[1], epsilon)

    treant_tree = RobustDecisionTree(
        max_depth=MAX_DEPTH,
        attacker=attacker,
        affine=False,
        min_instances_per_node=2,
        seed=1,
    )

    start_time = time.time()
    treant_tree.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime)

    treant_tree.to_xgboost_json(filename)

    return runtime


if __name__ == "__main__":
    epsilon = 0.1  # fraction of feature range
    k_folds = 5
    n_processes = 8
    output_dir = "trees/"

    datasets = load_all()

    # Create tuples of (samples, labels, epsilon, filename) to call train
    # functions with
    groot_arguments = []
    sklearn_arguments = []
    treant_arguments = []
    chen_arguments = []
    for name, X, y in datasets:
        k_folds_cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)
        for fold_i, (train_index, test_index) in enumerate(k_folds_cv.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            filename = f"{name}_fold_{fold_i}.json"

            groot_arguments.append(
                (X_train, y_train, epsilon, output_dir + "groot_" + filename)
            )
            sklearn_arguments.append(
                (X_train, y_train, epsilon, output_dir + "sklearn_" + filename)
            )
            treant_arguments.append(
                (X_train, y_train, epsilon, output_dir + "treant_" + filename)
            )
            chen_arguments.append(
                (X_train, y_train, epsilon, output_dir + "chen_" + filename)
            )

    # Fit all the trees in order of Scikit-learn, GROOT, TREANT such that
    # the fast results come early
    with Pool(n_processes) as pool:
        sklearn_times = pool.starmap(train_sklearn_tree, sklearn_arguments)
        groot_times = pool.starmap(train_groot_tree, groot_arguments)
        chen_times = pool.starmap(train_chen_tree, chen_arguments)
        treant_times = pool.starmap(train_treant_tree, treant_arguments)

    runtimes = []
    for dataset, sklearn_time, groot_time, treant_time, chen_time in zip(
        datasets, sklearn_times, groot_times, treant_times, chen_times
    ):
        name = dataset[0]
        runtimes.append((name, sklearn_time, groot_time, treant_time, chen_time))

    runtimes = []
    for i_dataset, dataset in enumerate(datasets):
        name = dataset[0]
        for i_fold in range(k_folds):
            sklearn_time = sklearn_times[i_dataset * k_folds + i_fold]
            groot_time = groot_times[i_dataset * k_folds + i_fold]
            treant_time = treant_times[i_dataset * k_folds + i_fold]
            chen_time = chen_times[i_dataset * k_folds + i_fold]
            runtimes.append((name, sklearn_time, groot_time, treant_time, chen_time))

    runtimes_df = pd.DataFrame(
        runtimes, columns=["Name", "Scikit-learn", "GROOT", "TREANT", "Chen et al."]
    )
    runtimes_df.to_csv(output_dir + "runtimes.csv")
