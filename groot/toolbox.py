import json

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from groot.verification.kantchelian_attack import KantchelianAttackWrapper
from groot.util import convert_numpy

import numpy as np

class Model:
    def __init__(self, json_model, n_classes):
        self.json_model = json_model
        self.n_classes = n_classes
    
    @staticmethod
    def from_json_file(filename, n_classes):
        with open(filename, "r") as file:
            json_model = json.load(file)

        return Model(json_model, n_classes)

    @staticmethod
    def from_sklearn(classifier):
        if isinstance(classifier, DecisionTreeClassifier):
            return _sklearn_tree_to_model(classifier)
        elif isinstance(classifier, RandomForestClassifier):
            return _sklearn_forest_to_model(classifier)
        elif isinstance(classifier, GradientBoostingClassifier):
            return _sklearn_booster_to_model(classifier)
        else:
            raise ValueError("Only decision tree, random forest and gradient boosting classifiers are supported, not " + type(classifier))
    
    @staticmethod
    def from_groot(classifier):
        json_trees = classifier.to_xgboost_json(output_file=None)
        if not isinstance(json_trees, list):
            json_trees = [json_trees]

        return Model(json_trees, 2)

    
    def predict(self, X):
        """
        Predict classes for some samples. The raw prediction values are turned into class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples)
            Predicted class labels.
        """
        prediction_values = self.decision_function(X)
        if self.n_classes == 2:
            return (prediction_values >= 0).astype(int)
        else:
            return np.argmax(prediction_values, axis=1)


    def decision_function(self, X):
        """
        Compute prediction values for some samples. These values are the sum of leaf values in which the samples end up.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples) or ndarray of shape (n_samples, n_classes)
            Predicted values. Returns a 1-dimensional array if n_classes=2, else a 2-dimensional array.
        """
        values = []
        if self.n_classes == 2:
            for sample in X:
                value = 0
                for tree in self.json_model:
                    value += self.__predict_proba_tree_sample(tree, sample)
                values.append(value)
        else:
            for sample in X:
                class_values = np.zeros(self.n_classes)
                for i, tree in enumerate(self.json_model):
                    class_values[i % self.n_classes] += self.__predict_proba_tree_sample(tree, sample)
                values.append(class_values)
        
        return np.array(values)


    def __predict_proba_tree_sample(self, json_tree, sample):
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
                return self.__predict_proba_tree_sample(sub_tree, sample)

    def __get_attack_wrapper(self, attack_name):
        if attack_name in {"milp", "kantchelian", "gurobi"}:
            return KantchelianAttackWrapper(self.json_model, self.n_classes)
        else:
            raise ValueError(f"Attack '{attack_name}' not supported.")

    def attack_feasibility(self, X, y, attack="milp",order=np.inf, epsilon=0.0):
        attack_wrapper = self.__get_attack_wrapper(attack)
        return attack_wrapper.attack_feasibility(X, y, order=order, epsilon=epsilon)
        
    def attack_distance(self, X, y, attack="milp", order=np.inf):
        attack_wrapper = self.__get_attack_wrapper(attack)
        return attack_wrapper.attack_distance(X, y, order=order)

    def adversarial_examples(self, X, y, attack="milp", order=np.inf):
        attack_wrapper = self.__get_attack_wrapper(attack)
        return attack_wrapper.adversarial_examples(X, y, order=order)

    def to_json(self, filename, indent=2):
        """
        Export the model object to a JSON file.

        Parameters
        ----------
        filename : str
            Name of the JSON file to export to.
        indent : int, optional
            Number of spaces to use for indentation in the JSON file. Can be reduced to save storage.
        """
        with open(filename, "w") as file:
            json.dump(self.json_model, file, indent=indent, default=convert_numpy)


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


def _sklearn_tree_to_model(tree: DecisionTreeClassifier):
    """
    Load a scikit-learn decision tree as a Model instance. A multiclass tree gets turned into a one-vs-all representation inside the JSON.

    Parameters
    ----------
    tree : sklearn.tree.DecisionTreeClassifier
        Decision tree to export
    """
    if tree.n_classes_ == 2:
        json_trees = [_sklearn_tree_to_dict(tree, classifier=True)]
    else:
        json_trees = []
        for class_label in range(tree.n_classes_):
            json_tree = _sklearn_tree_to_dict(tree, classifier=True, one_vs_all_class=class_label)
            json_trees.append(json_tree)

    return Model(json_trees, tree.n_classes_)


def _sklearn_forest_to_model(forest: RandomForestClassifier):
    """
    Load a scikit-learn random forest as a Model instance. A multiclass forest gets turned into a one-vs-all representation inside the JSON.

    Parameters
    ----------
    forest : sklearn.ensemble.RandomForestClassifier
        Random forest to export
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

    return Model(json_trees, forest.n_classes_)


def _sigmoid_inverse(proba: float):
    return np.log(proba / (1 - proba))


def _sklearn_booster_to_model(booster: GradientBoostingClassifier):
    """
    Load a scikit-learn gradient boosting classifier as a Model instance. A multiclass booster gets turned into a one-vs-all representation inside the JSON.
.
    Parameters
    ----------
    booster : sklearn.ensemble.GradientBoostingClassifier
        Gradient boosting ensemble to export
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

    return Model(json_trees, booster.n_classes_)
