from groot.datasets import load_all, load_epsilons_dict
from groot.util import numpy_to_chensvmlight

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

import subprocess

import time

import pandas as pd

k_folds = 5

data_dir = "data/"
output_dir = "out/"
forest_dir = "out/forests/"

epsilons = load_epsilons_dict()

train_config = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "tree_method": "robust_exact",
    "eta": 0.2,
    "gamma": 1.0,
    "min_child_weight": 1,
    "max_depth": 8,
    "num_round": 100,
    "save_period": 0,
    "nthread": 1,
}

dump_config = {
    "task": "dump",
    "dump_format": "json",
}

runtimes = []
for name, X, y in load_all():
    X = MinMaxScaler().fit_transform(X)  # Scale all features to [0,1]

    epsilon = epsilons[name]  # Get the epsilon for this dataset

    k_folds_cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)
    for fold_i, (train_index, test_index) in enumerate(k_folds_cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Write each dataset's folds to a libSVM file in the data directory
        numpy_to_chensvmlight(X_train, y_train, f"{data_dir}{name}_{fold_i}.train")
        numpy_to_chensvmlight(X_test, y_test, f"{data_dir}{name}_{fold_i}.test")

        # Write a config file for each dataset fold
        train_config["data"] = f'"{data_dir}{name}_{fold_i}.train"'
        train_config["eval[test]"] = f'"{data_dir}{name}_{fold_i}.test"'
        train_config["test:data"] = f'"{data_dir}{name}_{fold_i}.test"'
        train_config[
            "model_out"
        ] = f'"{forest_dir}chenboost_{name}_fold_{fold_i}.model"'
        train_config["robust_eps"] = epsilon
        with open(f"{data_dir}{name}_{fold_i}.conf", "w") as file:
            for key, value in train_config.items():
                file.write(f"{key} = {value}\n")

        start_time = time.time()
        # Train Chen et al.'s xgboost model on it using the CLI
        subprocess.run(["./xgboost", f"{data_dir}{name}_{fold_i}.conf"])

        total_time = time.time() - start_time
        runtimes.append((name, fold_i, total_time))

        # Write a dumping config file for each dataset fold
        dump_config["model_in"] = f'"{forest_dir}chenboost_{name}_fold_{fold_i}.model"'
        dump_config["name_dump"] = f'"{forest_dir}chenboost_{name}_fold_{fold_i}.json"'
        with open(f"{data_dir}{name}_{fold_i}_dump.conf", "w") as file:
            for key, value in dump_config.items():
                file.write(f"{key} = {value}\n")

        # Dump the trained models to JSON format
        subprocess.run(["./xgboost", f"{data_dir}{name}_{fold_i}_dump.conf"])

runtimes_df = pd.DataFrame(
    runtimes, columns=["Dataset", "Fold", "Chen et al. boosting"]
)
runtimes_df.to_csv(f"{output_dir}chenboost_runtimes.csv", index=False)
