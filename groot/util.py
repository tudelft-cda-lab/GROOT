from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import json

import numpy as np


def convert_numpy(obj):
    """
    Convert numpy ints and floats to python types. Useful when converting objects to JSON.

    Parameters
    ----------
    obj : {np.int32, np.int64, np.float32, np.float64, np.longlong}
        Number to convert to python int or float.
    """
    if isinstance(obj, np.int32) or isinstance(obj, np.int64) or isinstance(obj, np.longlong):
        return int(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    raise TypeError(f"Cannot convert type {type(obj)} to int or float")


def _sklearn_tree_to_dict(tree, scale=1.0, classifier=True):
    if classifier:
        assert hasattr(tree, "classes_"), "Not a classifier but classifier=True"
        assert tree.classes_.shape == (2,), "Currently only binary classifiers are supported"

    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value

    def dfs(node_id, depth):
        left_id = children_left[node_id]
        right_id = children_right[node_id]

        if left_id == right_id:
            # If leaf node
            if classifier:
                # A decision tree classifier contains the counts of samples
                # that reach the leaf
                class_counts = value[node_id][0]
                prediction = class_counts[1] / (class_counts[0] + class_counts[1])
                return {
                    "nodeid": node_id,
                    "leaf": prediction * scale,
                }
            else:
                # A decision tree regressor contains the raw prediction value
                prediction = value[node_id][0][0]
                return {
                    "nodeid": node_id,
                    "leaf": prediction * scale,
                }
        else:
            # If decision node
            left_dict = dfs(left_id, depth + 1)
            right_dict = dfs(right_id, depth + 1)

            return {
                "nodeid": node_id,
                "depth": depth,
                "split": feature[node_id],
                "split_condition": threshold[node_id],
                "yes": left_id,
                "no": right_id,
                "missing": left_id,
                "children": [left_dict, right_dict],
            }

    return dfs(0, 0)


def sklearn_tree_to_xgboost_json(tree: DecisionTreeClassifier, filename: str):
    """
    Export a scikit-learn decision tree to a JSON file in xgboost format.

    Parameters
    ----------
    tree : sklearn.tree.DecisionTreeClassifier
        Decision tree to export
    filename : str
        Exported JSON filename or path.
    """
    tree_dict = _sklearn_tree_to_dict(tree)

    with open(filename, "w") as file:
        json.dump([tree_dict], file, indent=2, default=convert_numpy)


def sklearn_forest_to_xgboost_json(forest: RandomForestClassifier, filename: str):
    """
    Export a scikit-learn random forest to a JSON file in xgboost format.

    Parameters
    ----------
    forest : sklearn.ensemble.RandomForestClassifier
        Random forest to export
    filename : str
        Exported JSON filename or path.
    """
    scale = 1 / forest.n_estimators
    forest_dict = [
        _sklearn_tree_to_dict(tree, scale=scale, classifier=True)
        for tree in forest.estimators_
    ]

    with open(filename, "w") as file:
        json.dump(forest_dict, file, indent=2, default=convert_numpy)


def sklearn_booster_to_xgboost_json(booster: GradientBoostingClassifier, filename: str):
    """
    Export a scikit-learn gradient boosting classifier to a JSON file in xgboost format.

    Parameters
    ----------
    booster : sklearn.ensemble.GradientBoostingClassifier
        Gradient boosting ensemble to export
    filename : str
        Exported JSON filename or path.
    """
    if booster.loss_.K == 1:
        json_trees = [
            _sklearn_tree_to_dict(tree[0], classifier=False) for tree in booster.estimators_
        ]
    else:
        json_trees = []
        for round_estimators in booster.estimators_:
            for tree in round_estimators:
                json_tree = _sklearn_tree_to_dict(tree, classifier=False)
                json_trees.append(json_tree)

    with open(filename, "w") as file:
        json.dump(json_trees, file, indent=2, default=convert_numpy)


def numpy_to_chensvmlight(X, y, filename):
    """
    Export a numpy dataset to the SVM-Light format that is needed for Chen et al. (2019).

    The difference between SVM-Light and this format is that zero values are also included.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Array of amples.
    y : array-like of shape (n_samples,)
        Array of class labels as integers.
    filename : str
        Exported SVM-Light dataset filename or path.
    """
    lines = []
    for sample, label in zip(X, y):
        terms = [str(label)]
        for i, value in enumerate(sample):
            terms.append(f"{i}:{value}")
        lines.append(" ".join(terms))
    svmlight_string = "\n".join(lines)

    with open(filename, "w") as file:
        file.write(svmlight_string)
