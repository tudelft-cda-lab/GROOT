from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

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


def _sklearn_tree_to_dict(tree, classifier=True, one_vs_all_class=1, learning_rate=1.0):
    if classifier:
        assert len(tree.classes_.shape) == 1, "Multi-output is not supported"

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

                # Map the prediction probability to a value in the range [-1, 1]
                leaf_value = (class_counts[one_vs_all_class] / np.sum(class_counts)) * 2 - 1
                return {
                    "nodeid": node_id,
                    "leaf": leaf_value,
                }
            else:
                # A decision tree regressor contains the raw prediction value
                prediction = value[node_id][0][0]
                return {
                    "nodeid": node_id,
                    "leaf": learning_rate * prediction,
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
    Export a scikit-learn decision tree to a JSON file in xgboost format. A multiclass tree gets turned into a one-vs-all representation inside the JSON file.

    Parameters
    ----------
    tree : sklearn.tree.DecisionTreeClassifier
        Decision tree to export
    filename : str
        Exported JSON filename or path.
    """
    if tree.n_classes_ == 2:
        json_trees = [_sklearn_tree_to_dict(tree, classifier=True)]
    else:
        json_trees = []
        for class_label in range(tree.n_classes_):
            json_tree = _sklearn_tree_to_dict(tree, classifier=True, one_vs_all_class=class_label)
            json_trees.append(json_tree)

    with open(filename, "w") as file:
        json.dump(json_trees, file, indent=2, default=convert_numpy)


def sklearn_forest_to_xgboost_json(forest: RandomForestClassifier, filename: str):
    """
    Export a scikit-learn random forest to a JSON file in xgboost format. A multiclass forest gets turned into a one-vs-all representation inside the JSON file.

    Parameters
    ----------
    forest : sklearn.ensemble.RandomForestClassifier
        Random forest to export
    filename : str
        Exported JSON filename or path.
    """
    if forest.n_classes_ == 2:
        json_trees = [
            _sklearn_tree_to_dict(tree, classifier=True)
            for tree in forest.estimators_
        ]
    else:
        json_trees = []
        for tree in forest.estimators_:
            for class_label in range(forest.n_classes_):
                json_tree = _sklearn_tree_to_dict(tree, classifier=True, one_vs_all_class=class_label)
                json_trees.append(json_tree)

    with open(filename, "w") as file:
        json.dump(json_trees, file, indent=2, default=convert_numpy)


def _sigmoid_inverse(proba: float):
    return np.log(proba / (1 - proba))


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
    init = booster.init_
    if not (isinstance(init, DummyClassifier) and init.strategy == "prior") and not init == "zero":
        raise ValueError("Only 'zero' or prior DummyClassifier init is supported")

    json_trees = []
    if booster.loss_.K == 1:
        if init != "zero":
            # For the binary case sklearn inverts the sigmoid function
            json_trees.append({
                "nodeid": 0,
                "leaf": _sigmoid_inverse(init.class_prior_[1]),
            })

        json_trees.extend([
            _sklearn_tree_to_dict(tree[0], classifier=False, learning_rate=booster.learning_rate) for tree in booster.estimators_
        ])
    else:
        json_trees = []

        if init != "zero":
            for i in range(booster.loss_.K):
                # For the multiclass case sklearn uses the log prior probability
                json_trees.append({
                    "nodeid": 0,
                    "leaf": np.log(init.class_prior_[i]),
                })

        for round_estimators in booster.estimators_:
            for tree in round_estimators:
                json_tree = _sklearn_tree_to_dict(tree, classifier=False, learning_rate=booster.learning_rate)
                json_trees.append(json_tree)

    with open(filename, "w") as file:
        json.dump(json_trees, file, indent=2, default=convert_numpy)


def predict_json_file(X, filename, n_classes=2):
    """
    Make class predictions based on a json_file. The prediction values are turned into class labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples to predict.
    filename : str
        Name of the JSON file to predict on.
    n_classes : int, optional
        Number of classes that the encoded model predicts.

    Returns
    -------
    ndarray of shape (n_samples)
        Predicted class labels.
    """
    prediction_values = decision_function_json_file(X, filename, n_classes)
    if n_classes == 2:
        return (prediction_values >= 0).astype(int)
    else:
        return np.argmax(prediction_values, axis=1)


def decision_function_json_file(X, filename, n_classes=2):
    """
    Compute prediction values based on a json_file. These values are the sum of leaf values in which the samples end up.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples to predict.
    filename : str
        Name of the JSON file to predict on.
    n_classes : int, optional
        Number of classes that the encoded model predicts.

    Returns
    -------
    ndarray of shape (n_samples) or ndarray of shape (n_samples, n_classes)
        Predicted values. Returns a 1-dimensional array if n_classes=2, else a 2-dimensional array.
    """
    json_trees = json.load(open(filename))

    if n_classes == 2:
        values = []
        for sample in X:
            value = 0
            for tree in json_trees:
                value += _predict_proba_json_tree(tree, sample)
            values.append(value)
    else:
        values = []
        for sample in X:
            class_values = np.zeros(n_classes)
            for i, tree in enumerate(json_trees):
                class_values[i % n_classes] += _predict_proba_json_tree(tree, sample)
            values.append(class_values)
    
    return np.array(values)

def _predict_proba_json_tree(json_tree, sample):
    """
    Recursively follow the path of a sample through the JSON tree and return the resulting leaf's value.
    """
    if "leaf" in json_tree:
        return json_tree["leaf"]
    
    if sample[json_tree["split"]] <= json_tree["split_condition"]:
        next_node_id = json_tree["yes"]
    else:
        next_node_id = json_tree["no"]

    for sub_tree in json_tree["children"]:
        if sub_tree["nodeid"] == next_node_id:
            return _predict_proba_json_tree(sub_tree, sample)


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
