import numpy as np
import numbers
from copy import deepcopy
import json


def convert_numpy(obj):
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    raise TypeError


class Leaf:
    """Representation of a decision leaf by its bounding box and value."""

    def __init__(self, conditions, value, is_numeric, attack_model, n_categories):
        # Conditions is a list with at every position the condition on
        # a single feature, numerical features are bounded by [low, high] and
        # categorical features have a set of categories that do not reach it.
        # value is the probability for a malicious sample.
        self.conditions = conditions
        self.value = value
        self.is_numeric = is_numeric
        self.attack_model = attack_model
        self.n_categories = n_categories

    def __can_reach_numerical_feature(self, position, condition, attack):
        # If the point is already in the leaf
        if position > condition[0] and position <= condition[1]:
            return True

        # Otherwise check if an attack can move it there
        if attack == ">":
            if position < condition[0]:
                return True
        elif attack == "<":
            if position > condition[1]:
                return True
        if attack == "<>":
            return True
        elif isinstance(attack, numbers.Number):
            if position <= condition[0] and condition[0] < position + attack:
                return True
            elif position > condition[1] and position - attack <= condition[1]:
                return True
        elif isinstance(attack, tuple):
            if position <= condition[0] and condition[0] < position + attack[1]:
                return True
            elif position > condition[1] and position - attack[0] <= condition[1]:
                return True

        # If the point is not in the leaf and cannot be perturbed to be in it
        return False

    def __can_reach_categorical_feature(self, category, condition, attack):
        # If the point is already in the leaf
        if category not in condition:
            return True

        # If no defined attack or not for this category
        if attack == "" or category not in attack:
            return False

        # Otherwise check if an attack can move it there
        attack_categories = attack[category]
        if isinstance(attack_categories, int) and attack_categories not in condition:
            return True
        elif isinstance(attack_categories, list) or isinstance(
            attack_categories, tuple
        ):
            for attack_category in attack_categories:
                if attack_category not in condition:
                    return True

        # If the point is not in the leaf and cannot be perturbed to be in it
        return False

    def __can_reach_feature(self, position, numeric, condition, attack):
        if numeric:
            return self.__can_reach_numerical_feature(position, condition, attack)
        return self.__can_reach_categorical_feature(
            int(round(position)), condition, attack
        )

    def can_reach(self, point):
        """
        Checks whether this leaf is in reach of the given point by the attacker.

        Parameters
        ----------
        point : array-like of shape (n_features,)
            Point's unperturbed values.

        Returns
        -------
        in_reach : bool
            Whether or not the point is in reach of this leaf.
        """
        for position, numeric, condition, attack in zip(
            point, self.is_numeric, self.conditions, self.attack_model
        ):
            if not self.__can_reach_feature(position, numeric, condition, attack):
                return False
        return True

    def get_bounding_box(self):
        """
        Get the bounding box of this leaf.

        Returns
        -------
        bbox : ndarray of shape (n_features, 2)
            Bounding box given by [low, high] for each feature.
        value : float
            Prediction value of this leaf.
        """
        if not all(self.is_numeric):
            raise Exception("Can only generate bounding box for numerical variables")

        return np.array(self.conditions), self.value

    def minimal_distance(self, point, order):
        """
        Compute the minimum perturbation distance between this leaf and the given sample in the given L-p norm.

        Parameters
        ----------
        point : array-like of shape (n_features,)
            Point's unperturbed values.
        order : {0, 1, 2, np.inf}
            L-p norm to compute distance in.

        Returns
        -------
        in_reach : bool
            Whether or not the point is in reach of this leaf.
        """
        bounds, _ = self.get_bounding_box()

        nearest_point = np.clip(point, bounds[:, 0], bounds[:, 1])
        distance = np.linalg.norm(point - nearest_point, ord=order)
        return distance

    def __numerical_conditions_intersect(self, condition, other_condition):
        # Return true if the two conditions ([low, high]) overlap
        if condition[0] < other_condition[1] and other_condition[0] < condition[1]:
            return True
        return False

    def __categorical_conditions_intersect(
        self, condition, other_condition, categories, sample_categories
    ):
        # Return false if the two leaves together refuse all categories
        combined_conditions = condition.union(other_condition)
        if len(combined_conditions) == categories:
            return False

        # Return false if all remaining categories are unreachable
        if combined_conditions.intersection(sample_categories) == len(
            sample_categories
        ):
            return False

        # Else the leaves intersect and are reachable by the given sample
        return True

    def compute_intersection(self, other):
        """
        Computes the intersection (a new Leaf object) of this leaf with another leaf. The intersection leaf represents the overlapping region of the two leaves. The new Leaf's value is the average of the original values.

        Parameters
        ----------
        other : Leaf
            Leaf to compute intersection with.

        Returns
        -------
        intersection : Leaf
            Leaf representing the intersection between this leaf and the other leaf.
        """
        intersection_conditions = []
        for this_condition, other_condition, numeric in zip(
            self.conditions, other.conditions, self.is_numeric
        ):
            if numeric:
                condition = [
                    max(this_condition[0], other_condition[0]),
                    min(this_condition[1], other_condition[1]),
                ]
            else:
                condition = this_condition.intersection(other_condition)
            intersection_conditions.append(condition)

        intersection_value = 0.5 * (self.value + other.value)
        intersection = Leaf(
            intersection_conditions,
            intersection_value,
            self.is_numeric,
            self.attack_model,
            self.n_categories,
        )
        return intersection

    def to_json(self):
        summary = {}
        summary["value"] = self.value
        summary["conditions"] = [list(condition) for condition in self.conditions]

        return summary


class DecisionTreeAdversary:
    """Adversary that can attack and score decision trees against adversarial examples."""

    def __init__(
        self,
        decision_tree,
        kind,
        attack_model=None,
        is_numeric=None,
        n_categories=None,
        one_adversarial_class=False,
    ):
        """
        Parameters
        ----------
        decision_tree : groot.model.GrootTree or sklearn.tree.DecisionTreeClassifier or groot.treant.RobustDecisionTree
            The decision tree to attack following our decision tree
            implementation.
        kind : {"ours", "groot", "sklearn", "treant"}
            The kind of decision tree to attack, different kinds require different conditions for categorical variables.
        attack_model : array-like of shape (n_features,), optional
            Attacker capabilities for perturbing X, it is only required for when kind is 'sklearn', 'treant' or 'robust'. The attack model describes for every feature in which way it can be perturbed. By default, all features are considered not perturbable.
        is_numeric : array-like of shape (n_features,), optional
            Boolean mask for whether each feature is numerical or categorical.
        n_categories : array-like of shape (n_features,), optional
            Number of categories per feature, entries for numerical features are ignored.
        one_adversarial_class : bool, optional
            Whether one class (malicious, 1) perturbs their samples or if both classes (benign and malicious, 0 and 1) do so.
        """

        self.decision_tree = decision_tree
        self.kind = kind
        self.one_adversarial_class = one_adversarial_class

        if is_numeric is not None:
            self.is_numeric = is_numeric

        if attack_model is not None:
            self.attack_model = attack_model

        if n_categories is not None:
            self.n_categories = n_categories

        if kind == "ours" or kind == "groot":
            if is_numeric is None:
                self.is_numeric = self.decision_tree.is_numerical
            if attack_model is None:
                self.attack_model = self.decision_tree.attack_model
            if n_categories is None:
                self.n_categories = self.decision_tree.n_categories_
            self.__calculate_leaves_ours()

        elif kind == "json":
            if is_numeric is None:
                self.is_numeric = self.decision_tree.is_numerical
            if attack_model is None:
                self.attack_model = self.decision_tree.attack_model
            if n_categories is None:
                self.n_categories = [None] * len(self.is_numeric)
            self.__calculate_leaves_ours()

        elif kind == "treant":
            assert attack_model is not None

            self.__calculate_leaves_treant()

        elif kind == "robust":
            assert attack_model is not None
            assert is_numeric is not None

            self.__calculate_leaves_robust()

        elif kind == "sklearn":
            assert attack_model is not None
            assert is_numeric is not None

            self.__calculate_leaves_sklearn()

    def __calculate_leaves_ours(self):
        inf = 10.0 ** 10
        conditions = []
        for numeric in self.is_numeric:
            if numeric:
                conditions.append([-inf, inf])
            else:
                conditions.append(set())
        self.leaves = self.__calculate_leaves_ours_rec(
            self.decision_tree.root_, conditions
        )

    def __calculate_leaves_ours_rec(self, node, conditions):
        if node.is_leaf():
            return [
                Leaf(
                    deepcopy(conditions),
                    node.value[1],
                    self.is_numeric,
                    self.attack_model,
                    self.n_categories,
                )
            ]

        feature = node.feature
        if self.is_numeric[feature]:
            old_bound = conditions[feature][1]
            conditions[feature][1] = node.threshold

            left_leaves = self.__calculate_leaves_ours_rec(node.left_child, conditions)

            conditions[feature][1] = old_bound
            old_bound = conditions[feature][0]
            conditions[feature][0] = node.threshold

            right_leaves = self.__calculate_leaves_ours_rec(
                node.right_child, conditions
            )

            conditions[feature][0] = old_bound
            return left_leaves + right_leaves
        else:
            old_categories = conditions[feature]
            conditions[feature] = old_categories | node.categories_right

            left_leaves = self.__calculate_leaves_ours_rec(node.left_child, conditions)

            conditions[feature] = old_categories | node.categories_left

            right_leaves = self.__calculate_leaves_ours_rec(
                node.right_child, conditions
            )

            conditions[feature] = old_categories
            return left_leaves + right_leaves

    def __calculate_leaves_sklearn(self):
        inf = 10.0 ** 10
        conditions = []
        for numeric in self.is_numeric:
            if numeric:
                conditions.append([-inf, inf])
            else:
                conditions.append(set())
        self.leaves = self.__calculate_leaves_sklearn_rec(0, conditions)

    def __calculate_leaves_sklearn_rec(self, node_id, conditions):
        tree = self.decision_tree.tree_
        left_node_id = tree.children_left[node_id]
        right_node_id = tree.children_right[node_id]

        # If left and right node id are the same we have a leaf
        if left_node_id == right_node_id:
            samples_in_leaf = tree.value[node_id][0]

            n_samples_in_leaf = np.sum(samples_in_leaf)
            if n_samples_in_leaf == 0:
                prediction = 0.5
            else:
                prediction = samples_in_leaf[1] / n_samples_in_leaf
            return [
                Leaf(
                    deepcopy(conditions),
                    prediction,
                    self.is_numeric,
                    self.attack_model,
                    self.n_categories,
                )
            ]

        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        if self.is_numeric[feature]:
            # <= threshold goes left, > goes right
            old_bound = conditions[feature][1]
            conditions[feature][1] = threshold

            left_leaves = self.__calculate_leaves_sklearn_rec(left_node_id, conditions)

            conditions[feature][1] = old_bound
            old_bound = conditions[feature][0]
            conditions[feature][0] = threshold

            right_leaves = self.__calculate_leaves_sklearn_rec(
                right_node_id, conditions
            )

            conditions[feature][0] = old_bound
            return left_leaves + right_leaves
        else:
            # Scikit learn trees treat categorical values the same as numerical
            # <= threshold goes left, > goes right
            threshold = int(threshold)
            old_categories = conditions[feature]
            conditions[feature] = old_categories | set(
                range(threshold + 1, self.n_categories[feature])
            )

            left_leaves = self.__calculate_leaves_sklearn_rec(left_node_id, conditions)

            conditions[feature] = old_categories | set(range(threshold + 1))

            right_leaves = self.__calculate_leaves_sklearn_rec(
                right_node_id, conditions
            )

            conditions[feature] = old_categories
            return left_leaves + right_leaves

    def __calculate_leaves_treant(self):
        inf = 10.0 ** 10
        conditions = []
        self.is_numeric = self.decision_tree.numerical_idx
        for numeric in self.is_numeric:
            if numeric:
                conditions.append([-inf, inf])
            else:
                conditions.append(set())
        self.leaves = self.__calculate_leaves_treant_rec(
            self.decision_tree.root, conditions
        )

    def __calculate_leaves_treant_rec(self, node, conditions):
        if node.is_leaf():
            return [
                Leaf(
                    deepcopy(conditions),
                    node.get_node_prediction()[1],
                    self.is_numeric,
                    self.attack_model,
                    self.n_categories,
                )
            ]

        feature = node.best_split_feature_id
        if self.is_numeric[feature]:
            old_bound = conditions[feature][1]
            conditions[feature][1] = node.best_split_feature_value

            left_leaves = self.__calculate_leaves_treant_rec(node.left, conditions)

            conditions[feature][1] = old_bound
            old_bound = conditions[feature][0]
            conditions[feature][0] = node.best_split_feature_value

            right_leaves = self.__calculate_leaves_treant_rec(node.right, conditions)

            conditions[feature][0] = old_bound
            return left_leaves + right_leaves
        else:
            # TREANT trees compare categorical feature splits using '=='
            threshold = int(node.best_split_feature_value)
            old_categories = conditions[feature]
            conditions[feature] = (
                old_categories
                | set(range(threshold))
                | set(range(threshold + 1, self.n_categories[feature]))
            )

            left_leaves = self.__calculate_leaves_treant_rec(node.left, conditions)

            conditions[feature] = old_categories | set([threshold])

            right_leaves = self.__calculate_leaves_treant_rec(node.right, conditions)

            conditions[feature] = old_categories
            return left_leaves + right_leaves

    def __count_misclassifications(self, X, y):
        X_0 = X[y == 0]
        X_1 = X[y == 1]

        # Count benign misclassifications
        if self.one_adversarial_class:
            y_0_pred = self.decision_tree.predict(X_0)
            false_positives = np.sum(y_0_pred == 1)
            true_negatives = len(y_0_pred) - false_positives
        else:
            false_positives = 0
            for sample in X_0:
                for leaf in self.leaves:
                    if round(leaf.value) == 1 and leaf.can_reach(sample):
                        false_positives += 1
                        break
            true_negatives = X_0.shape[0] - false_positives

        # Count the amount misclassifications caused by the adversary
        false_negatives = 0
        for sample in X_1:
            for leaf in self.leaves:
                if round(leaf.value) == 0 and leaf.can_reach(sample):
                    false_negatives += 1
                    break
        true_positives = X_1.shape[0] - false_negatives

        return true_negatives, false_negatives, true_positives, false_positives

    def get_bounding_boxes(self):
        return [leaf.get_bounding_box() for leaf in self.leaves]

    def adversarial_accuracy(self, X, y):
        """
        Computes the accuracy under an adversary with given attack model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        adv_accuracy : float
            Adversarial accuracy score.
        """

        _, false_negatives, _, false_positives = self.__count_misclassifications(X, y)

        # Return the accuracy under the effects of an adversary
        return 1 - ((false_negatives + false_positives) / len(y))

    def adversarial_f1_score(self, X, y):
        """
        Computes the f1 score under an adversary with given attack model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        adv_f1 : float
            Adversarial f1 score.
        """

        _, fn, tp, fp = self.__count_misclassifications(X, y)

        # Return the f1 score under the effects of an adversary
        if tp + fn == 0:
            return 0
        else:
            recall = tp / (tp + fn)

        if tp + fp == 0:
            return 0
        else:
            precision = tp / (tp + fp)

        if recall == 0 or precision == 0:
            return 0

        return 2 / ((1 / recall) + (1 / precision))

    def average_attack_distance(self, X, y, order=np.inf):
        """
        Computes the average perturbation distance when perturbing each sample
        optimally. Here optimally means by the shortest possible distance
        such that the predicted class is different than the sample's label.

        The order parameter is fed straight into numpy.linalg.norm. See the
        numpy documentation for explanations and examples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        mean_distance : np.float
            Mean perturbation distance.
        """

        distances = []
        for sample, label in zip(X, y):
            # Skip benign samples if only malicious samples can be perturbed
            if self.one_adversarial_class and label == 0:
                continue

            best_distance = np.inf
            for leaf in self.leaves:
                if round(leaf.value) != label:
                    distance = leaf.minimal_distance(sample, order)
                    if distance < best_distance:
                        best_distance = distance

            distances.append(best_distance)

        return np.mean(distances)

    def to_file(self, filename):
        summary = {}

        summary["attack_model"] = self.attack_model
        summary["is_numeric"] = self.is_numeric

        summary["leaves"] = [leaf.to_json() for leaf in self.leaves]

        with open(filename, "w") as file:
            json.dump(summary, file, indent=2, default=convert_numpy)
