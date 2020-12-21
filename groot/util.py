from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import json

import numpy as np


def convert_numpy(obj):
    """Convert numpy ints and floats to python types."""
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    raise TypeError(f"Cannot convert type {type(obj)} to int or float")


def _sklearn_tree_to_dict(tree: DecisionTreeClassifier, scale=1.0):
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
            class_counts = value[node_id][0]
            prediction = class_counts[1] / (class_counts[0] + class_counts[1])
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
    tree_dict = _sklearn_tree_to_dict(tree)

    with open(filename, "w") as file:
        json.dump([tree_dict], file, indent=2, default=convert_numpy)


def sklearn_forest_to_xgboost_json(forest: RandomForestClassifier, filename: str):
    scale = 1 / forest.n_estimators
    forest_dict = [_sklearn_tree_to_dict(tree, scale) for tree in forest.estimators_]

    with open(filename, "w") as file:
        json.dump(forest_dict, file, indent=2, default=convert_numpy)
