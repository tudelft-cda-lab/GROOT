from groot.datasets import epsilon_attacker, load_all, load_epsilons_dict
from groot.model import GrootTree, GrootRandomForest
from groot.treant import RobustDecisionTree
from groot.util import (
    sklearn_tree_to_xgboost_json,
    sklearn_forest_to_xgboost_json,
    sklearn_booster_to_xgboost_json,
)
from groot.provably_robust_boosting.wrapper import fit_provably_robust_boosting

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

import numba

numba.set_num_threads(1)

import pandas as pd

from multiprocessing import Pool

import time

TREE_MAX_DEPTH = 4
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
FOREST_N_TREES = 100
BOOSTING_MAX_DEPTH = 8


def train_groot_tree(X, y, epsilon, filename):
    attack_model = [epsilon] * X.shape[1]

    groot_tree = GrootTree(
        max_depth=TREE_MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        attack_model=attack_model,
        one_adversarial_class=False,
        random_state=1,
    )

    start_time = time.time()
    groot_tree.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime, flush=True)

    groot_tree.to_xgboost_json(filename)

    return runtime


def train_chen_tree(X, y, epsilon, filename):
    attack_model = [epsilon] * X.shape[1]

    chen_tree = GrootTree(
        max_depth=TREE_MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        attack_model=attack_model,
        one_adversarial_class=False,
        chen_heuristic=True,
        random_state=1,
    )

    start_time = time.time()
    chen_tree.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime, flush=True)

    chen_tree.to_xgboost_json(filename)

    return runtime


def train_sklearn_tree(X, y, _, filename):
    tree = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=1,
    )

    start_time = time.time()
    tree.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime, flush=True)

    sklearn_tree_to_xgboost_json(tree, filename)

    return runtime


def train_treant_tree(X, y, epsilon, filename):
    if "spambase" in filename:
        # After multiple days TREANT did not finish fitting on this dataset.
        # Return the number of seconds in a day.
        return 86400.0

    attacker = epsilon_attacker(X.shape[1], epsilon)

    treant_tree = RobustDecisionTree(
        max_depth=TREE_MAX_DEPTH,
        attacker=attacker,
        affine=False,
        min_instances_per_node=MIN_SAMPLES_SPLIT,
        seed=1,
    )

    start_time = time.time()
    treant_tree.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime, flush=True)

    treant_tree.to_xgboost_json(filename)

    return runtime


def train_groot_forest(X, y, epsilon, filename):
    attack_model = [epsilon] * X.shape[1]

    groot_forest = GrootRandomForest(
        n_estimators=FOREST_N_TREES,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        attack_model=attack_model,
        one_adversarial_class=False,
        random_state=1,
    )

    start_time = time.time()
    groot_forest.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime, flush=True)

    groot_forest.to_xgboost_json(filename)

    return runtime


def train_chen_forest(X, y, epsilon, filename):
    attack_model = [epsilon] * X.shape[1]

    chen_forest = GrootRandomForest(
        n_estimators=FOREST_N_TREES,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        attack_model=attack_model,
        one_adversarial_class=False,
        chen_heuristic=True,
        random_state=1,
    )

    start_time = time.time()
    chen_forest.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime, flush=True)

    chen_forest.to_xgboost_json(filename)

    return runtime


def train_sklearn_forest(X, y, _, filename):
    forest = RandomForestClassifier(
        n_estimators=FOREST_N_TREES,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=1,
    )

    start_time = time.time()
    forest.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime, flush=True)

    sklearn_forest_to_xgboost_json(forest, filename)

    return runtime


def train_sklearn_boosting(X, y, _, filename):
    booster = GradientBoostingClassifier(
        n_estimators=FOREST_N_TREES,
        max_depth=BOOSTING_MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=1,
    )

    start_time = time.time()
    booster.fit(X, y)
    runtime = time.time() - start_time
    print(filename, runtime, flush=True)

    sklearn_booster_to_xgboost_json(booster, filename)

    return runtime


def train_provably_robust_boosting(X, y, epsilon, filename):
    start_time = time.time()
    fit_provably_robust_boosting(
        X,
        y,
        epsilon=epsilon,
        n_trees=FOREST_N_TREES,
        max_depth=BOOSTING_MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        filename=filename,
    )
    runtime = time.time() - start_time
    print(filename, runtime, flush=True)

    return runtime


if __name__ == "__main__":
    k_folds = 5
    n_processes = 16
    output_dir = "out/"
    tree_dir = "out/trees/"
    forest_dir = "out/forests/"

    datasets = load_all()
    epsilons = load_epsilons_dict()

    # Create tuples of (samples, labels, epsilon, filename) to call train
    # functions with
    sklearn_arguments = []
    groot_arguments = []
    chen_arguments = []
    treant_arguments = []

    sklearn_forest_arguments = []
    groot_forest_arguments = []
    chen_forest_arguments = []
    sklearn_boosting_arguments = []
    provable_boosting_arguments = []
    for name, X, y in datasets:
        X = MinMaxScaler().fit_transform(X)  # Scale all features to [0,1]

        epsilon = epsilons[name]  # Get the epsilon for this dataset

        # Create K folds, for each fold keep track of training arguments
        k_folds_cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)
        for fold_i, (train_index, test_index) in enumerate(k_folds_cv.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            filename = f"{name}_fold_{fold_i}.json"

            sklearn_arguments.append(
                (X_train, y_train, epsilon, tree_dir + "sklearn_" + filename)
            )
            groot_arguments.append(
                (X_train, y_train, epsilon, tree_dir + "groot_" + filename)
            )
            chen_arguments.append(
                (X_train, y_train, epsilon, tree_dir + "chen_" + filename)
            )
            treant_arguments.append(
                (X_train, y_train, epsilon, tree_dir + "treant_" + filename)
            )

            sklearn_forest_arguments.append(
                (X_train, y_train, epsilon, forest_dir + "sklearn_" + filename)
            )
            groot_forest_arguments.append(
                (X_train, y_train, epsilon, forest_dir + "groot_" + filename)
            )
            chen_forest_arguments.append(
                (X_train, y_train, epsilon, forest_dir + "chen_" + filename)
            )
            sklearn_boosting_arguments.append(
                (X_train, y_train, epsilon, forest_dir + "boost_" + filename)
            )
            provable_boosting_arguments.append(
                (X_train, y_train, epsilon, forest_dir + "provable_" + filename)
            )

    def export_runtimes():
        runtimes = []
        for i_dataset, dataset in enumerate(datasets):
            name = dataset[0]
            for i_fold in range(k_folds):
                i_time = i_dataset * k_folds + i_fold

                runtimes.append(
                    (
                        name,
                        i_fold,
                        sklearn_times[i_time],
                        groot_times[i_time],
                        chen_times[i_time],
                        treant_times[i_time],
                        sklearn_forest_times[i_time],
                        sklearn_boosting_times[i_time],
                        groot_forest_times[i_time],
                        chen_forest_times[i_time],
                        provable_boosting_times[i_time],
                    )
                )

        runtimes_df = pd.DataFrame(
            runtimes,
            columns=[
                "Dataset",
                "Fold",
                "Decision tree",
                "GROOT tree",
                "Chen et al. tree",
                "TREANT tree",
                "Random forest",
                "Gradient boosting",
                "GROOT forest",
                "Chen et al. forest",
                "Provably robust boosting",
            ],
        )
        runtimes_df.to_csv(output_dir + "runtimes.csv", index=False)

    # Fit all models in parallel per algorithm on the training arguments
    with Pool(n_processes) as pool:
        # Fit tree ensembles
        sklearn_forest_times = pool.starmap(
            train_sklearn_forest, sklearn_forest_arguments
        )
        groot_forest_times = pool.starmap(train_groot_forest, groot_forest_arguments)
        chen_forest_times = pool.starmap(train_chen_forest, chen_forest_arguments)
        sklearn_boosting_times = pool.starmap(
            train_sklearn_boosting, sklearn_boosting_arguments
        )

        # Fit single trees
        sklearn_times = pool.starmap(train_sklearn_tree, sklearn_arguments)
        groot_times = pool.starmap(train_groot_tree, groot_arguments)
        chen_times = pool.starmap(train_chen_tree, chen_arguments)

        # Export to get fast partial results in
        print("Exporting partial results...", flush=True)
        provable_boosting_times = [1] * (len(datasets) * k_folds)
        treant_times = [1] * (len(datasets) * k_folds)
        export_runtimes()

        # Run provably robust boosting
        print("Running provably robust boosting...", flush=True)
        provable_boosting_times = pool.starmap(
            train_provably_robust_boosting, provable_boosting_arguments
        )

        # Export to get fast partial results in
        print("Exporting partial results...", flush=True)
        export_runtimes()

        # Run TREANT
        print("Running TREANT...", flush=True)
        treant_times = pool.starmap(train_treant_tree, treant_arguments)

        # Export after TREANT to get all results in
        print("Exporting full results...", flush=True)
        export_runtimes()

        print("Done", flush=True)
