import json
import numbers
import numpy as np
from collections import defaultdict
from itertools import product
from numba import jit
from numpy.lib.function_base import iterable
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from sklearn.base import clone
from .util import convert_numpy


_TREE_LEAF = -1
_TREE_UNDEFINED = -2

LEFT = 0
LEFT_INTERSECT = 1
RIGHT_INTERSECT = 2
RIGHT = 3

NOGIL = True


class Node:
    """Base class for decision tree nodes, also functions as leaf."""

    def __init__(self, feature, left_child, right_child, value):
        self.feature = feature
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

    def predict(self, _):
        assert self.left_child == _TREE_LEAF
        assert self.right_child == _TREE_LEAF

        return self.value

    def pretty_print(self, _, depth=0):
        indentation = depth * "  "
        if isinstance(self.value, np.ndarray):
            return f"{indentation}return [{self.value[0]:.3f}, {self.value[1]:.3f}]"
        else:
            return f"{indentation}return {self.value:.3f}"

    def to_json(self):
        if isinstance(self.value, np.ndarray):
            return {
                "value": [self.value[0], self.value[1]],
            }
        else:
            return {
                "value": self.value,
            }

    def to_xgboost_json(self, node_id, depth):
        if isinstance(self.value, np.ndarray):
            # Return leaf value in range [-1, 1]
            return {"nodeid": node_id, "leaf": self.value[1] * 2 - 1}, node_id
        else:
            return {"nodeid": node_id, "leaf": self.value}, node_id

    def is_leaf(self):
        return self.left_child == _TREE_LEAF and self.right_child == _TREE_LEAF

    def prune(self, _):
        return self


class NumericalNode(Node):
    """
    Decision tree node for numerical decision (threshold).
    """

    def __init__(self, feature, threshold, left_child, right_child, value):
        super().__init__(feature, left_child, right_child, value)
        self.threshold = threshold

    def predict(self, sample):
        """
        Predict the class label of the given sample. Follow the left subtree
        if the sample's value is lower or equal to the threshold, else follow
        the right sub tree.
        """
        comparison = sample[self.feature] <= self.threshold
        if comparison:
            return self.left_child.predict(sample)
        else:
            return self.right_child.predict(sample)

    def pretty_print(self, feature_names, depth=0):
        indentation = depth * "  "

        if feature_names:
            feature_name = feature_names[self.feature]
        else:
            feature_name = f"x[{self.feature}]"

        return f"""{indentation}if {feature_name} <= {self.threshold}:
{self.left_child.pretty_print(feature_names, depth + 1)}
{indentation}else:
{self.right_child.pretty_print(feature_names, depth + 1)}"""

    def to_json(self):
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "left_child": self.left_child.to_json(),
            "right_child": self.right_child.to_json(),
        }

    def to_xgboost_json(self, node_id, depth):
        left_id = node_id + 1
        left_dict, new_node_id = self.left_child.to_xgboost_json(left_id, depth + 1)

        right_id = new_node_id + 1
        right_dict, new_node_id = self.right_child.to_xgboost_json(right_id, depth + 1)

        return (
            {
                "nodeid": node_id,
                "depth": depth,
                "split": self.feature,
                "split_condition": self.threshold,
                "yes": left_id,
                "no": right_id,
                "missing": left_id,
                "children": [left_dict, right_dict],
            },
            new_node_id,
        )

    def prune(self, bounds=defaultdict(lambda: [-np.inf, np.inf])):
        old_high = bounds[self.feature][1]
        bounds[self.feature][1] = self.threshold

        self.left_child = self.left_child.prune(bounds)

        bounds[self.feature][1] = old_high
        old_low = bounds[self.feature][0]
        bounds[self.feature][0] = self.threshold

        self.right_child = self.right_child.prune(bounds)

        bounds[self.feature][0] = old_low

        if self.threshold >= bounds[self.feature][1] or self.threshold == np.inf:
            # If no sample can reach this node's right side
            return self.left_child
        elif self.threshold <= bounds[self.feature][0] or self.threshold == -np.inf:
            # If no sample can reach this node's left side
            return self.right_child
        elif (
            self.left_child.is_leaf()
            and self.right_child.is_leaf()
            and self.left_child.value[1] == self.right_child.value[1]
        ):
            # If both children are leaves and they predict the same value
            return self.left_child
        else:
            return self


def node_tree_to_arrays(node: Node):
    xgboost_json, n_nodes = node.to_xgboost_json(0, 0)
    n_nodes += 1

    left_ids = np.empty(n_nodes, dtype=np.int32)
    right_ids = np.empty(n_nodes, dtype=np.int32)
    features = np.empty(n_nodes, dtype=np.int32)
    thresholds = np.empty(n_nodes, dtype=np.float32)
    values = np.empty(n_nodes, dtype=np.float32)

    def _recurse(json_node):
        node_id = json_node["nodeid"]

        if "leaf" in json_node:
            left_ids[node_id] = _TREE_LEAF
            right_ids[node_id] = _TREE_LEAF
            features[node_id] = _TREE_LEAF
            thresholds[node_id] = _TREE_LEAF
            values[node_id] = json_node["leaf"]
        else:
            left_ids[node_id] = json_node["yes"]
            right_ids[node_id] = json_node["no"]
            features[node_id] = json_node["split"]
            thresholds[node_id] = json_node["split_condition"]
            values[node_id] = _TREE_UNDEFINED

            _recurse(json_node["children"][0])
            _recurse(json_node["children"][1])

    _recurse(xgboost_json)

    return left_ids, right_ids, features, thresholds, values


@jit(nopython=True, nogil=NOGIL)
def _predict_compiled(X, left_ids, right_ids, features, thresholds, values):
    n_samples = X.shape[0]

    # Initialize the output to -1
    y_pred = np.empty(n_samples, dtype=np.float32)

    # Iterate over the samples
    for i, sample in enumerate(X):
        # Initialize the current node to the root node
        node_id = 0

        # Iterate over the nodes until we reach a leaf
        while True:
            # Get the feature and threshold of the current node
            feature = features[node_id]
            threshold = thresholds[node_id]

            # If the feature is -1, we have reached the leaf node
            if feature == _TREE_LEAF:
                break

            # If the sample is lower or equal to the threshold, follow the left child
            if sample[feature] <= threshold:
                node_id = left_ids[node_id]
            else:
                node_id = right_ids[node_id]

        # Store the prediction of the leaf node
        y_pred[i] = values[node_id]

    return y_pred


class CompiledTree:
    def __init__(self, node):
        (
            self.left_ids,
            self.right_ids,
            self.features,
            self.thresholds,
            self.values,
        ) = node_tree_to_arrays(node)

    def predict_classification(self, X):
        pred_values = _predict_compiled(
            X,
            self.left_ids,
            self.right_ids,
            self.features,
            self.thresholds,
            self.values,
        )
        return (pred_values > 0.0).astype(np.int32)

    def predict_classification_proba(self, X):
        pred_values = _predict_compiled(
            X,
            self.left_ids,
            self.right_ids,
            self.features,
            self.thresholds,
            self.values,
        )

        # Rescale [-1, 1] values to probabilities in range [0, 1]
        pred_values += 1
        pred_values *= 0.5

        return np.vstack([1 - pred_values, pred_values]).T

    def predict_regression(self, X):
        return _predict_compiled(
            X,
            self.left_ids,
            self.right_ids,
            self.features,
            self.thresholds,
            self.values,
        )


def _attack_model_to_tuples(attack_model, n_features):
    if isinstance(attack_model, numbers.Number):
        return [(attack_model, attack_model) for _ in range(n_features)]
    elif iterable(attack_model):
        new_attack_model = []
        for attack_mode in attack_model:
            if attack_mode == "":
                new_attack_model.append((0, 0))
            elif attack_mode == ">":
                new_attack_model.append((0, 10e9))
            elif attack_mode == "<":
                new_attack_model.append((10e9, 0))
            elif attack_mode == "<>":
                new_attack_model.append((10e9, 10e9))
            elif isinstance(attack_mode, numbers.Number):
                new_attack_model.append((attack_mode, attack_mode))
            elif isinstance(attack_mode, tuple) and len(attack_mode) == 2:
                new_attack_model.append(attack_mode)
            else:
                raise Exception("Unknown attack model spec:", attack_mode)
        return new_attack_model
    else:
        raise Exception(
            "Unknown attack model spec, needs to be perturbation radius or perturbation"
            " per feature:",
            attack_model,
        )


@jit(nopython=True, nogil=NOGIL)
def _scan_numerical_feature_fast(
    samples,
    y,
    dec,
    inc,
    left_bound,
    right_bound,
    chen_heuristic,
    one_adversarial_class,
):
    sort_order = samples.argsort()
    sorted_labels = y[sort_order]
    sample_queue = samples[sort_order]
    dec_queue = sample_queue - dec
    inc_queue = sample_queue + inc

    # Initialize sample counters
    l_0 = l_1 = li_0 = li_1 = ri_0 = ri_1 = 0
    label_counts = np.bincount(y)
    r_0 = label_counts[0]
    r_1 = label_counts[1]

    # Initialize queue values and indices
    sample_i = dec_i = inc_i = 0
    sample_val = sample_queue[0]
    dec_val = dec_queue[0]
    inc_val = inc_queue[0]

    best_score = 10e9
    best_split = None
    adv_gini = None
    while True:
        smallest_val = min(sample_val, dec_val, inc_val)

        # Find the current point and label from the queue with smallest value.
        # Also update the sample counters
        if sample_val == smallest_val:
            point = sample_val
            label = sorted_labels[sample_i]

            if label == 0:
                if one_adversarial_class:
                    r_0 -= 1
                    l_0 += 1
                else:
                    ri_0 -= 1
                    li_0 += 1
            else:
                ri_1 -= 1
                li_1 += 1

            # Update sample_val and i to the values belonging to the next
            # sample in queue. If we reached the end of the queue then store
            # a high number to make sure the sample_queue does not get picked
            if sample_i < sample_queue.shape[0] - 1:
                sample_i += 1
                sample_val = sample_queue[sample_i]
            else:
                sample_val = 10e9
        elif dec_val == smallest_val:
            point = dec_val
            label = sorted_labels[dec_i]

            if label == 0:
                if not one_adversarial_class:
                    r_0 -= 1
                    ri_0 += 1
            else:
                r_1 -= 1
                ri_1 += 1

            # Update dec_val and i to the values belonging to the next
            # sample in queue. If we reached the end of the queue then store
            # a high number to make sure the dec_queue does not get picked
            if dec_i < dec_queue.shape[0] - 1:
                dec_i += 1
                dec_val = dec_queue[dec_i]
            else:
                dec_val = 10e9
        else:
            point = inc_val
            label = sorted_labels[inc_i]

            if label == 0:
                if not one_adversarial_class:
                    li_0 -= 1
                    l_0 += 1
            else:
                li_1 -= 1
                l_1 += 1

            # Update inc_val and i to the values belonging to the next
            # sample in queue. If we reached the end of the queue then store
            # a high number to make sure the inc_queue does not get picked
            if inc_i < inc_queue.shape[0] - 1:
                inc_i += 1
                inc_val = inc_queue[inc_i]
            else:
                inc_val = 10e9

        if point >= right_bound:
            break

        # If the next point is not the same as this one
        next_point = min(sample_val, dec_val, inc_val)
        if next_point != point:
            if one_adversarial_class:
                if chen_heuristic:
                    adv_gini, _ = chen_adversarial_gini_gain_one_class(
                        l_0, l_1, r_0, r_1, li_1, ri_1
                    )
                else:
                    adv_gini, _ = adversarial_gini_gain_one_class(
                        l_0, l_1, r_0, r_1, li_1 + ri_1
                    )
            else:
                if chen_heuristic:
                    adv_gini, _, __ = chen_adversarial_gini_gain_two_class(
                        l_0, l_1, li_0, li_1, ri_0, ri_1, r_0, r_1
                    )
                else:
                    adv_gini, _, __ = adversarial_gini_gain_two_class(
                        l_0, l_1, li_0, li_1, ri_0, ri_1, r_0, r_1
                    )

            # Maximize the margin of the split
            split = (point + next_point) * 0.5

            if (
                adv_gini is not None
                and adv_gini < best_score
                and split > left_bound
                and split < right_bound
            ):
                best_score = adv_gini
                best_split = split

    return best_score, best_split


@jit(nopython=True, nogil=NOGIL)
def _scan_numerical_feature_fast_regression(
    samples,
    y,
    dec,
    inc,
    left_bound,
    right_bound,
    chen_heuristic,
):
    unique_samples = np.unique(samples)

    if dec == 0 and inc == 0:
        thresholds = np.sort(unique_samples)
    else:
        thresholds = np.sort(
            np.unique(np.concatenate((unique_samples - dec, unique_samples + inc)))
        )

    samples_inc = samples + inc
    samples_dec = samples - dec

    best_score = 10e9
    best_split = None
    adv_sse = None
    for point, next_point in zip(thresholds[:-1], thresholds[1:]):

        if point >= right_bound:
            break

        y_left = y[samples_inc <= point]
        y_right = y[samples_dec > point]

        if chen_heuristic:
            y_left_intersect = y[(samples <= point) & (samples_inc > point)]
            y_right_intersect = y[(samples > point) & (samples_dec <= point)]

            adv_sse, _ = chen_adversarial_sum_absolute_errors(
                y_left,
                y_left_intersect,
                y_right_intersect,
                y_right,
            )
        else:
            y_intersect = y[~((samples_inc <= point) | (samples_dec > point))]

            adv_sse, _ = adversarial_sum_absolute_errors(
                y_left,
                y_right,
                y_intersect,
            )

        # Maximize the margin of the split
        split = (point + next_point) * 0.5

        if (
            adv_sse is not None
            and adv_sse < best_score
            and split > left_bound
            and split < right_bound
        ):
            best_score = adv_sse
            best_split = split

    return best_score, best_split


@jit(nopython=True, nogil=NOGIL)
def chen_adversarial_gini_gain_one_class(l_0, l_1, r_0, r_1, li_1, ri_1):
    i_1 = li_1 + ri_1

    s1 = weighted_gini(l_0, l_1 + li_1, r_0, r_1 + ri_1)
    s2 = weighted_gini(l_0, l_1, r_0, r_1 + i_1)
    s3 = weighted_gini(l_0, l_1 + i_1, r_0, r_1)
    s4 = weighted_gini(l_0, l_1 + ri_1, r_0, r_1 + li_1)

    worst_case = max(s1, s2, s3, s4)

    # Return the worst found weighted Gini impurity, the number of class 1
    # samples that move to the left and the number of class 0 samples that
    # move to the left
    if s1 == worst_case:
        return s1, li_1

    if s2 == worst_case:
        return s2, 0

    if s3 == worst_case:
        return s3, i_1

    if s4 == worst_case:
        return s4, ri_1


@jit(nopython=True, nogil=NOGIL)
def chen_adversarial_gini_gain_two_class(l_0, l_1, li_0, li_1, ri_0, ri_1, r_0, r_1):
    i_0 = li_0 + ri_0
    i_1 = li_1 + ri_1

    s1 = weighted_gini(l_0 + li_0, l_1 + li_1, r_0 + ri_0, r_1 + ri_1)
    s2 = weighted_gini(l_0, l_1, r_0 + i_0, r_1 + i_1)
    s3 = weighted_gini(l_0 + i_0, l_1 + i_1, r_0, r_1)
    s4 = weighted_gini(l_0 + ri_0, l_1 + ri_1, r_0 + li_0, r_1 + li_1)

    worst_case = max(s1, s2, s3, s4)

    # Return the worst found weighted Gini impurity, the number of class 1
    # samples that move to the left and the number of class 0 samples that
    # move to the left
    if s1 == worst_case:
        return s1, li_1, li_0

    if s2 == worst_case:
        return s2, 0, 0

    if s3 == worst_case:
        return s3, i_1, i_0

    if s4 == worst_case:
        return s4, ri_1, ri_0


@jit(nopython=True, nogil=NOGIL)
def adversarial_gini_gain_one_class(l_0, l_1, r_0, r_1, i_1):
    # Fast implementation of the adversarial Gini gain, it finds the
    # analytical maximum and rounds to the nearest two ints, then returns
    # the highest of those two. x is limited by the range [0, i_1].
    x = max(min((l_0 * r_1 + l_0 * i_1 - l_1 * r_0) / (l_0 + r_0), i_1), 0)

    x_floor = int(np.floor(x))
    x_ceil = int(np.ceil(x))
    adv_gini_floor = weighted_gini(l_0, l_1 + x_floor, r_0, r_1 + i_1 - x_floor)
    adv_gini_ceil = weighted_gini(l_0, l_1 + x_ceil, r_0, r_1 + i_1 - x_ceil)
    if adv_gini_floor > adv_gini_ceil:
        return adv_gini_floor, x_floor
    else:
        return adv_gini_ceil, x_ceil


@jit(nopython=True, nogil=NOGIL)
def adversarial_gini_gain_two_class(l_0, l_1, li_0, li_1, ri_0, ri_1, r_0, r_1):
    i_0 = li_0 + ri_0
    i_1 = li_1 + ri_1

    if i_0 == 0 and i_1 == 0:
        return weighted_gini(l_0, l_1, r_0, r_1), 0, 0

    if l_1 + r_1 + i_1 == 0:
        return (
            weighted_gini(l_0 + li_0, l_1 + li_1, r_0 + i_0 - li_0, r_1 + i_1 - li_1),
            li_1,
            li_0,
        )

    # Compute these before since we use their values multiple times
    x_coef = (l_0 + r_0 + i_0) / (l_1 + r_1 + i_1)
    intercept = (l_1 * r_0 - l_0 * r_1 - l_0 * i_1 + l_1 * i_0) / (l_1 + r_1 + i_1)
    denominator = x_coef**2 + 1

    # In the paper we refer to m1, m0 here they are li_1 and li_0
    x_prime = round((li_1 + x_coef * (li_0 - intercept)) / denominator)
    y_prime = round((x_coef * (li_1 + x_coef * li_0) + intercept) / denominator)

    # Unfortunately the best solution often lies outside our region of interest
    # (x in [0, i_1] and y in [0, i_0]) so we have to check for corner cases
    if x_prime < 0 and y_prime > i_0:
        # If the point (x', y') is out the top-left corner of our region of
        # interest then the line does not pass the region, we use the best
        # point which is that corner
        x_prime = 0
        y_prime = i_0
    elif x_prime < 0:
        # If x' is smaller than 0 we try the closest point on the solution line
        # in the region x \in [0, i_1] which is x = 0
        x_prime = 0
        y_prime = (
            l_1 * r_0 - l_0 * r_1 - l_0 * i_1 + l_1 * i_0 + (l_0 + r_0 + i_0) * x_prime
        ) / (l_1 + r_1 + i_1)
        if y_prime > i_0:
            # If y is still not in the region than the line is completely
            # outside of the region
            x_prime = 0
            y_prime = i_0
    elif x_prime > i_1 and y_prime < 0:
        # If the point (x', y') is out the bottom-right corner of our region of
        # interest then the line does not pass the region, we use the best
        # point which is that corner
        x_prime = i_1
        y_prime = 0
    elif x_prime > i_1:
        # If x' is larger than i_10 we try the closest point on the solution
        # line in the region x \in [0, i_1] which is x = i_1
        x_prime = i_1
        y_prime = (
            l_1 * r_0 - l_0 * r_1 - l_0 * i_1 + l_1 * i_0 + (l_0 + r_0 + i_0) * x_prime
        ) / (l_1 + r_1 + i_1)
        if y_prime < 0:
            # If y is still not in the region than the line is completely
            # outside of the region
            x_prime = i_1
            y_prime = 0
    elif y_prime < 0:
        # If y' is smaller than 0 we try the closest point on the solution line
        # in the region y \in [0, i_1] which is y = 0
        y_prime = 0
        x_prime = (
            l_0 * r_1 + l_0 * i_1 - l_1 * r_0 - l_1 * i_0 + (l_1 + r_1 + i_1) * y_prime
        ) / (l_0 + r_0 + i_0)
        if x_prime > i_1:
            x_prime = i_1
            y_prime = 0
    elif y_prime > i_0:
        # If y' is smaller than 0 we try the closest point on the solution line
        # in the region y \in [0, i_1] which is y = 0
        y_prime = i_0
        x_prime = (
            l_0 * r_1 + l_0 * i_1 - l_1 * r_0 - l_1 * i_0 + (l_1 + r_1 + i_1) * y_prime
        ) / (l_0 + r_0 + i_0)
        if x_prime < 0:
            x_prime = 0
            y_prime = i_0

    x_prime = int(round(x_prime))
    y_prime = int(round(y_prime))

    assert x_prime >= 0 and x_prime <= i_1
    assert y_prime >= 0 and y_prime <= i_0

    # Return the gini gain given the rounded x and y prime
    return (
        weighted_gini(
            l_0 + y_prime, l_1 + x_prime, r_0 + i_0 - y_prime, r_1 + i_1 - x_prime
        ),
        x_prime,
        y_prime,
    )


@jit(nopython=True, nogil=NOGIL)
def gini_impurity(i_0, i_1):
    if i_0 + i_1 == 0:
        return 1.0

    ratio = i_0 / (i_0 + i_1)
    return 1.0 - (ratio**2) - ((1 - ratio) ** 2)


@jit(nopython=True, nogil=NOGIL)
def sum_absolute_errors(y):
    if len(y) == 0:
        return 0.0

    return np.sum(np.abs(y - np.median(y)))


@jit(nopython=True, nogil=NOGIL)
def adversarial_sum_absolute_errors(y_l, y_r, y_i):
    if len(y_i) == 0:
        return sum_absolute_errors(y_l) + sum_absolute_errors(y_r), (0, 0)

    y_i = np.sort(y_i)
    max_error = 0
    indices = None
    for i in range(len(y_i)):
        error = 0
        error += sum_absolute_errors(np.concatenate((y_l, y_i[:i])))
        error += sum_absolute_errors(np.concatenate((y_r, y_i[i:])))

        if error > max_error:
            max_error = error
            indices = (0, i)

    for i in range(len(y_i)):
        error = 0
        error += sum_absolute_errors(np.concatenate((y_l, y_i[i:])))
        error += sum_absolute_errors(np.concatenate((y_r, y_i[:i])))

        if error > max_error:
            max_error = error
            indices = (i, len(y_i))

    return max_error, indices


@jit(nopython=True, nogil=NOGIL)
def chen_adversarial_sum_absolute_errors(y_l, y_li, y_ri, y_r):
    if len(y_li) == 0 and len(y_ri) == 0:
        return sum_absolute_errors(y_l) + sum_absolute_errors(y_r), 1

    s1 = sum_absolute_errors(np.concatenate((y_l, y_li))) + sum_absolute_errors(
        np.concatenate((y_r, y_ri))
    )
    s2 = sum_absolute_errors(y_l) + sum_absolute_errors(
        np.concatenate((y_li, y_ri, y_r))
    )
    s3 = sum_absolute_errors(np.concatenate((y_l, y_li, y_ri))) + sum_absolute_errors(
        y_r
    )
    s4 = sum_absolute_errors(np.concatenate((y_l, y_ri))) + sum_absolute_errors(
        np.concatenate((y_r, y_li))
    )

    worst_case = max(s1, s2, s3, s4)

    if s1 == worst_case:
        return s1, 1
    elif s2 == worst_case:
        return s2, 2
    elif s3 == worst_case:
        return s3, 3
    else:
        return s4, 4


@jit(nopython=True, nogil=NOGIL)
def weighted_gini(l_0, l_1, r_0, r_1):
    l_t = l_0 + l_1
    r_t = r_0 + r_1

    # Prevent division by 0
    if l_t == 0:
        l_p = 1.0
    else:
        l_p = l_0 / (l_0 + l_1)
    if r_t == 0:
        r_p = 1.0
    else:
        r_p = r_0 / (r_0 + r_1)

    gini = l_t * (1 - (l_p**2) - ((1 - l_p) ** 2)) + r_t * (
        1 - (r_p**2) - ((1 - r_p) ** 2)
    )

    total = l_t + r_t
    if total != 0:
        gini /= total
        return gini
    else:
        return 1.0


@jit(nopython=True, nogil=NOGIL)
def _counts_to_one_class_adv_gini(counts, rho, chen_heuristic):
    # Apply rho by moving a number of samples back from intersect
    rho_inv = 1.0 - rho
    left_mal = counts[LEFT][1] + int(round(rho_inv * counts[LEFT_INTERSECT][1]))
    right_mal = counts[RIGHT][1] + int(round(rho_inv * counts[RIGHT_INTERSECT][1]))
    left_i_mal = int(round(rho * counts[LEFT_INTERSECT][1]))
    right_i_mal = int(round(rho * counts[RIGHT_INTERSECT][1]))

    # Compute the adversarial gini gain
    if chen_heuristic:
        adv_gini, _ = chen_adversarial_gini_gain_one_class(
            counts[LEFT][0],
            left_mal,
            counts[RIGHT][0],
            right_mal,
            left_i_mal,
            right_i_mal,
        )
    else:
        adv_gini, _ = adversarial_gini_gain_one_class(
            counts[LEFT][0],
            left_mal,
            counts[RIGHT][0],
            right_mal,
            left_i_mal + right_i_mal,
        )

    return adv_gini


@jit(nopython=True, nogil=NOGIL)
def _counts_to_two_class_adv_gini(counts, rho, chen_heuristic):
    # Apply rho by moving a number of samples back from intersect
    rho_inv = 1.0 - rho
    left = counts[LEFT] + np.rint(rho_inv * counts[LEFT_INTERSECT]).astype(np.int64)
    right = counts[RIGHT] + np.rint(rho_inv * counts[RIGHT_INTERSECT]).astype(np.int64)
    left_i = np.rint(rho * counts[LEFT_INTERSECT]).astype(np.int64)
    right_i = np.rint(rho * counts[RIGHT_INTERSECT]).astype(np.int64)

    # Compute the adversarial gini gain
    if chen_heuristic:
        adv_gini, _, _ = chen_adversarial_gini_gain_two_class(
            left[0],
            left[1],
            left_i[0],
            left_i[1],
            right_i[0],
            right_i[1],
            right[0],
            right[1],
        )
    else:
        adv_gini, _, _ = adversarial_gini_gain_two_class(
            left[0],
            left[1],
            left_i[0],
            left_i[1],
            right_i[0],
            right_i[1],
            right[0],
            right[1],
        )

    return adv_gini


class BaseGrootTree(BaseEstimator):
    """
    Base class for GROOT decision trees.

    Implements high level fitting operation and exporting to strings/JSON.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        robust_weight=1.0,
        attack_model=None,
        chen_heuristic=False,
        compile=True,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.robust_weight = robust_weight
        self.attack_model = attack_model
        self.chen_heuristic = chen_heuristic
        self.compile = compile
        self.random_state = random_state

    def fit(self, X, y, check_input=True):
        """
        Build a robust and fair binary decision tree from the training set
        (X, y) using greedy splitting according to the weighted adversarial
        Gini impurity and fairness impurity.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training samples.
        y : array-like of shape (n_samples,)
            The class labels as integers 0 (benign) or 1 (malicious)

        Returns
        -------
        self : object
            Fitted estimator.
        """

        if check_input:
            X, y = check_X_y(X, y)
            y = self._check_target(y)

        self.n_samples_, self.n_features_in_ = X.shape

        if self.attack_model is None:
            attack_model = [""] * X.shape[1]
        else:
            attack_model = self.attack_model

        # Turn numerical features in attack model into tuples to make fitting
        # code simpler
        self.attack_model_ = np.array(
            _attack_model_to_tuples(attack_model, X.shape[1]), dtype=X.dtype
        )

        self.random_state_ = check_random_state(self.random_state)

        if self.max_features == "sqrt":
            self.max_features_ = int(np.sqrt(self.n_features_in_))
        elif self.max_features == "log2":
            self.max_features_ = int(np.log2(self.n_features_in_))
        elif self.max_features is None:
            self.max_features_ = self.n_features_in_
        else:
            self.max_features_ = self.max_features

        if self.max_features_ == 0:
            self.max_features_ = 1

        # Keep track of the minimum and maximum split value for each feature
        constraints = np.concatenate(
            (np.min(X, axis=0).reshape(-1, 1), np.max(X, axis=0).reshape(-1, 1)), axis=1
        )

        self.root_ = self.__fit_recursive(X, y, constraints)

        # Compile the tree into a representation that is faster when predicting
        if self.compile:
            self.compiled_root_ = CompiledTree(self.root_)

        return self

    def __fit_recursive(self, X, y, constraints, depth=0):
        """
        Recursively fit the decision tree on the training dataset (X, y).

        The constraints make sure that leaves are well formed, e.g. don't
        cross an earlier split. Stop when the depth has reached self.max_depth,
        when a leaf is pure or when the leaf contains too few samples.
        """
        if (
            (self.max_depth is not None and depth == self.max_depth)
            or len(y) < self.min_samples_split
            or np.all(y == y[0])
        ):
            return self._create_leaf(y)

        current_score = self._score(y)

        rule, feature, split_score = self.__best_adversarial_decision(X, y, constraints)

        score_gain = current_score - split_score

        if rule is None or score_gain <= 0.00:
            return self._create_leaf(y)

        # Assert that the split obeys constraints made by previous splits
        assert rule >= constraints[feature][0]
        assert rule < constraints[feature][1]

        X_left, y_left, X_right, y_right = self._split_left_right(
            X,
            y,
            rule,
            feature,
        )

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return self._create_leaf(y)

        # Set the right bound and store old one for after recursion
        old_right_bound = constraints[feature][1]
        constraints[feature][1] = rule

        left_node = self.__fit_recursive(X_left, y_left, constraints, depth + 1)

        # Reset right bound, set left bound, store old one for after recursion
        constraints[feature][1] = old_right_bound
        old_left_bound = constraints[feature][0]
        constraints[feature][0] = rule

        right_node = self.__fit_recursive(X_right, y_right, constraints, depth + 1)

        # Reset the left bound
        constraints[feature][0] = old_left_bound

        node = NumericalNode(feature, rule, left_node, right_node, _TREE_UNDEFINED)

        return node

    def __best_adversarial_decision(self, X, y, constraints):
        """
        Find the best split by iterating through each feature and scanning
        it for that feature's optimal split.
        """

        best_score = 10e9
        best_rule = None
        best_feature = None

        # If there is a limit on features to consider in a split then choose
        # that number of random features.
        all_features = np.arange(self.n_features_in_)
        features = self.random_state_.choice(
            all_features, size=self.max_features_, replace=False
        )

        for feature in features:
            score, decision_rule = self._scan_feature(X, y, feature, constraints)

            if decision_rule is not None and score < best_score:
                best_score = score
                best_rule = decision_rule
                best_feature = feature

        return best_rule, best_feature, best_score

    def to_string(self, feature_names=None):
        result = ""
        result += f"Parameters: {self.get_params()}\n"

        if feature_names is None:
            feature_names = [f"x[{i}]" for i in range(self.n_features_in_)]

        if hasattr(self, "root_"):
            result += f"Tree:\n{self.root_.pretty_print(feature_names)}"
        else:
            result += "Tree has not yet been fitted"

        return result

    def to_json(self, output_file="tree.json"):
        dictionary = {
            "params": self.get_params(),
        }
        if hasattr(self, "root_"):
            dictionary["tree"] = self.root_.to_json()
        else:
            dictionary["tree"] = None

        if output_file is None:
            return dictionary
        else:
            with open(output_file, "w") as fp:
                json.dump(dictionary, fp, indent=2, default=convert_numpy)

    def to_xgboost_json(self, output_file="tree.json"):
        check_is_fitted(self, "root_")

        dictionary, _ = self.root_.to_xgboost_json(0, 0)

        if output_file is None:
            return dictionary
        else:
            with open(output_file, "w") as fp:
                # If saving to file then surround dict in list brackets
                json.dump([dictionary], fp, indent=2, default=convert_numpy)


class GrootTreeClassifier(BaseGrootTree, ClassifierMixin):
    """
    A robust decision tree for binary classification.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        robust_weight=1.0,
        attack_model=None,
        one_adversarial_class=False,
        chen_heuristic=False,
        compile=True,
        random_state=None,
    ):
        """
        Parameters
        ----------
        max_depth : int, optional
            The maximum depth for the decision tree once fitted.
        min_samples_split : int, optional
            The minimum number of samples required to split a node.
        min_samples_leaf : int, optional
            The minimum number of samples required to make a leaf.
        max_features : int or {"sqrt", "log2"}, optional
            The number of features to consider while making each split, if None then all features are considered.
        robust_weight : float, optional
            The ratio of samples that are actually moved by an adversary.
        attack_model : array-like of shape (n_features,), optional
            Attacker capabilities for perturbing X. By default, all features are considered not perturbable.
        one_adversarial_class : bool, optional
            Whether one class (malicious, 1) perturbs their samples or if both classes (benign and malicious, 0 and 1) do so.
        chen_heuristic : bool, optional
            Whether to use the heuristic for the adversarial Gini impurity from Chen et al. (2019) instead of GROOT's adversarial Gini impurity.
        compile : bool, optional
            Whether to compile the tree for faster predictions.
        random_state : int, optional
            Controls the sampling of the features to consider when looking for the best split at each node.

        Attributes
        ----------
        classes_ : ndarray of shape (n_classes,)
            The class labels.
        max_features_ : int
            The inferred value of max_features.
        n_samples_ : int
            The number of samples when `fit` is performed.
        n_features_ : int
            The number of features when `fit` is performed.
        root_ : Node
            The root node of the tree after fitting.
        compiled_root_ : CompiledTree
            The compiled root node of the tree after fitting.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.robust_weight = robust_weight
        self.attack_model = attack_model
        self.one_adversarial_class = one_adversarial_class
        self.chen_heuristic = chen_heuristic
        self.compile = compile
        self.random_state = random_state

    def _check_target(self, y):
        target_type = type_of_target(y)
        if target_type != "binary":
            raise ValueError(
                "Unknown label type: classifier only supports binary labels but found"
                f" {target_type}"
            )

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        return y

    def _score(self, y):
        return gini_impurity(np.sum(y == 0), np.sum(y == 1))

    def _create_leaf(self, y):
        """
        Create a leaf object that predicts according to the ratio of benign
        and malicious labels in the array y.
        """

        # Count the number of points that fall into this leaf including
        # adversarially moved points
        label_counts = np.bincount(y, minlength=2)

        # Set the leaf's prediction value to the weighted average of the
        # prediction with and without moving points
        value = label_counts / np.sum(label_counts)

        return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)

    def _scan_feature(self, X, y, feature, constraints):
        """
        Scan feature to find the locally optimal split.
        """

        samples = X[:, feature]
        attack_mode = self.attack_model_[feature]
        constraint = constraints[feature]

        # If possible, use the faster scan implementation
        if self.robust_weight == 1:
            return _scan_numerical_feature_fast(
                samples,
                y,
                *attack_mode,
                *constraint,
                self.chen_heuristic,
                self.one_adversarial_class,
            )
        else:
            return self.__scan_feature_numerical(samples, y, attack_mode, *constraint)

    def __initialize_scan(self, samples, y, attack_mode):
        queue = []
        counts = np.array(
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            dtype=np.int64,
        )

        if attack_mode == "":
            counts[RIGHT] = np.bincount(y)

            for sample, label in zip(samples, y):
                queue.append((sample, label, RIGHT, LEFT))

        elif attack_mode == ">":
            counts[RIGHT] = np.bincount(y)

            for sample, label in zip(samples, y):
                if label == 0 and self.one_adversarial_class:
                    queue.append((sample, label, RIGHT, LEFT))
                else:
                    queue.append((sample, label, RIGHT, LEFT_INTERSECT))

        elif attack_mode == "<":
            if self.one_adversarial_class:
                counts[RIGHT][0] = np.sum(y == 0)
                counts[RIGHT_INTERSECT][1] = np.sum(y == 1)
            else:
                counts[RIGHT_INTERSECT] = np.bincount(y)

            for sample, label in zip(samples, y):
                if label == 0 and self.one_adversarial_class:
                    queue.append((sample, label, RIGHT, LEFT))
                else:
                    queue.append((sample, label, RIGHT_INTERSECT, LEFT))

        elif attack_mode == "<>":
            if self.one_adversarial_class:
                counts[RIGHT][0] = np.sum(y == 0)
                counts[RIGHT_INTERSECT][1] = np.sum(y == 1)
            else:
                counts[RIGHT_INTERSECT] = np.bincount(y)

            for sample, label in zip(samples, y):
                if label == 0 and self.one_adversarial_class:
                    queue.append((sample, label, RIGHT, LEFT))
                else:
                    queue.append((sample, label, RIGHT_INTERSECT, LEFT_INTERSECT))

        elif isinstance(attack_mode, numbers.Number):
            counts[RIGHT] = np.bincount(y)

            for sample, label in zip(samples, y):
                if label == 0 and self.one_adversarial_class:
                    queue.append((sample, label, RIGHT, LEFT))
                else:
                    queue.append(
                        (sample - attack_mode, label, RIGHT, RIGHT_INTERSECT),
                    )
                    queue.append(
                        (sample, label, RIGHT_INTERSECT, LEFT_INTERSECT),
                    )
                    queue.append(
                        (sample + attack_mode, label, LEFT_INTERSECT, LEFT),
                    )

        elif isinstance(attack_mode, tuple):
            counts[RIGHT] = np.bincount(y)

            for sample, label in zip(samples, y):
                if label == 0 and self.one_adversarial_class:
                    queue.append((sample, label, RIGHT, LEFT))
                else:
                    queue.append(
                        (sample - attack_mode[0], label, RIGHT, RIGHT_INTERSECT),
                    )
                    queue.append(
                        (sample, label, RIGHT_INTERSECT, LEFT_INTERSECT),
                    )
                    queue.append(
                        (sample + attack_mode[1], label, LEFT_INTERSECT, LEFT),
                    )

        # Sort queue in reverse order since popping from end is faster
        queue.sort(reverse=True)

        return queue, counts

    def __scan_feature_numerical(
        self,
        samples,
        y,
        attack_mode,
        left_bound,
        right_bound,
    ):
        """
        Scan a numerical feature for the optimal split by identifying every
        potential split, sorting these and iterating through them.

        While iterating from the left to the right, remember counts for the
        right / right_intersect / left_intersect / left positions and label.
        """
        best_score = 10e9
        best_split = None

        queue, counts = self.__initialize_scan(samples, y, attack_mode)

        while len(queue) > 0 and queue[-1][0] < left_bound:
            point, label, move_from, move_to = queue.pop()
            counts[move_from][label] -= 1
            counts[move_to][label] += 1

        adv_gini = None
        while queue:
            point, label, move_from, move_to = queue.pop()
            counts[move_from][label] -= 1
            counts[move_to][label] += 1

            if point >= right_bound:
                break

            if len(queue) > 0 and queue[-1][0] != point:
                # Compute the adversarial Gini gain
                if self.one_adversarial_class:
                    adv_gini = _counts_to_one_class_adv_gini(
                        counts, self.robust_weight, self.chen_heuristic
                    )
                else:
                    adv_gini = _counts_to_two_class_adv_gini(
                        counts, self.robust_weight, self.chen_heuristic
                    )

                # Maximize the margin of the split
                split = (point + queue[-1][0]) * 0.5

                if (
                    adv_gini is not None
                    and adv_gini < best_score
                    and split < right_bound
                ):
                    best_score = adv_gini
                    best_split = split

            if len(queue) == 0:
                break

        return best_score, best_split

    def _split_left_right(self, X, y, rule, feature):
        """
        Split the dataset (X, y) into a left and right dataset according to the
        optimal split determined by rule and feature.

        Only a ratio 'robust_weight' of moving (malicious) points are actually
        transfered to the other side.
        """

        # Get perturbation range for this feature
        dec, inc = self.attack_model_[feature]

        if self.one_adversarial_class:
            # Determine the indices of samples on each side of the split
            label_0 = y == 0
            label_1 = np.invert(label_0)

            i_left = np.where(
                (label_0 & (X[:, feature] <= rule))
                | (label_1 & (X[:, feature] + inc <= rule))
            )[0]
            i_left_intersection = np.where(
                label_1 & (X[:, feature] + inc > rule) & (X[:, feature] <= rule)
            )[0]
            i_right_intersection = np.where(
                label_1 & (X[:, feature] > rule) & (X[:, feature] - dec <= rule)
            )[0]
            i_right = np.where(
                (label_0 & (X[:, feature] > rule))
                | (label_1 & (X[:, feature] - dec > rule))
            )[0]
        else:
            # Determine the indices of samples on each side of the split
            i_left = np.where(X[:, feature] + inc <= rule)[0]
            i_left_intersection = np.where(
                (X[:, feature] + inc > rule) & (X[:, feature] <= rule)
            )[0]
            i_right_intersection = np.where(
                (X[:, feature] > rule) & (X[:, feature] - dec <= rule)
            )[0]
            i_right = np.where(X[:, feature] - dec > rule)[0]

        # Count samples with labels 0 and 1 left and right
        l_0, l_1 = np.bincount(y[i_left], minlength=2)
        r_0, r_1 = np.bincount(y[i_right], minlength=2)

        # Determine labels on the left and right intersection
        y_left_intersection = y[i_left_intersection]
        y_right_intersection = y[i_right_intersection]

        i_left_intersection_0 = i_left_intersection[y_left_intersection == 0]
        i_left_intersection_1 = i_left_intersection[y_left_intersection == 1]
        i_right_intersection_0 = i_right_intersection[y_right_intersection == 0]
        i_right_intersection_1 = i_right_intersection[y_right_intersection == 1]

        li_0 = len(i_left_intersection_0)
        li_1 = len(i_left_intersection_1)
        ri_0 = len(i_right_intersection_0)
        ri_1 = len(i_right_intersection_1)

        # Compute optimal movement
        if self.one_adversarial_class:
            # Compute numbers of samples after applying rho
            assert li_0 == 0
            assert ri_0 == 0
            l_1 = l_1 + round((1.0 - self.robust_weight) * li_1)
            r_1 = r_1 + round((1.0 - self.robust_weight) * ri_1)
            li_1 = round(self.robust_weight * li_1)
            ri_1 = round(self.robust_weight * ri_1)
            i_1 = li_1 + ri_1

            # Determine optimal movement
            if self.chen_heuristic:
                _, m1 = chen_adversarial_gini_gain_one_class(
                    l_0, l_1, r_0, r_1, li_1, ri_1
                )
            else:
                _, m1 = adversarial_gini_gain_one_class(l_0, l_1, r_0, r_1, i_1)
            m0 = None
        else:
            # Compute numbers of samples after applying rho
            l_0 = l_0 + round((1.0 - self.robust_weight) * li_0)
            l_1 = l_1 + round((1.0 - self.robust_weight) * li_1)
            r_0 = r_0 + round((1.0 - self.robust_weight) * ri_0)
            r_1 = r_1 + round((1.0 - self.robust_weight) * ri_1)
            li_0 = round(self.robust_weight * li_0)
            li_1 = round(self.robust_weight * li_1)
            ri_0 = round(self.robust_weight * ri_0)
            ri_1 = round(self.robust_weight * ri_1)

            # Determine optimal movement
            if self.chen_heuristic:
                _, m1, m0 = chen_adversarial_gini_gain_two_class(
                    l_0, l_1, li_0, li_1, ri_0, ri_1, r_0, r_1
                )
            else:
                _, m1, m0 = adversarial_gini_gain_two_class(
                    l_0, l_1, li_0, li_1, ri_0, ri_1, r_0, r_1
                )

        # Move label 1 samples according to m1
        if m1 > li_1:
            n_move_left = m1 - li_1

            i_left_intersection_1 = np.concatenate(
                (i_left_intersection_1, i_right_intersection_1[:n_move_left])
            )
            i_right_intersection_1 = i_right_intersection_1[n_move_left:]
        elif m1 < li_1:
            n_move_right = li_1 - m1

            i_right_intersection_1 = np.concatenate(
                (i_left_intersection_1[:n_move_right], i_right_intersection_1)
            )
            i_left_intersection_1 = i_left_intersection_1[n_move_right:]

        # Move label 0 samples according to m0 (not used if one_adversarial_class=True)
        if m0:
            if m0 > li_0:
                n_move_left = m0 - li_0

                i_left_intersection_0 = np.concatenate(
                    (i_left_intersection_0, i_right_intersection_0[:n_move_left])
                )
                i_right_intersection_0 = i_right_intersection_0[n_move_left:]
            elif m0 < li_0:
                n_move_right = li_0 - m0

                i_right_intersection_0 = np.concatenate(
                    (i_left_intersection_0[:n_move_right], i_right_intersection_0)
                )
                i_left_intersection_0 = i_left_intersection_0[n_move_right:]

        i_left = np.concatenate(
            (
                i_left,
                i_left_intersection_0,
                i_left_intersection_1,
            )
        )

        i_right = np.concatenate(
            (
                i_right,
                i_right_intersection_0,
                i_right_intersection_1,
            )
        )

        # Move the resulting intersection arrays to the left and right sides
        X_left = X[i_left]
        y_left = y[i_left]
        X_right = X[i_right]
        y_right = y[i_right]

        # Assert that we are not losing any samples in this process
        assert len(X_left) + len(X_right) == len(X)

        return X_left, y_left, X_right, y_right

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.

        The class probability is the fraction of samples of the same class in
        the leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        proba : array of shape (n_samples,)
            The probability for each input sample of being malicious.
        """

        check_is_fitted(self, "root_")

        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Received different number of features during predict than during fit"
            )

        # If model has been compiled, use compiled predict_proba
        if self.compile:
            return self.compiled_root_.predict_classification_proba(X)

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        predictions = np.array(predictions)
        predictions /= np.sum(predictions, axis=1)[:, np.newaxis]

        return predictions

    def predict(self, X):
        """
        Predict the classes of the input samples X.

        The predicted class is the most frequently occuring class label in a
        leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """

        # If model has been compiled, use compiled predict
        if self.compile:
            check_is_fitted(self, "root_")

            X = check_array(X)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    "Received different number of features during predict than"
                    " during fit"
                )

            return self.classes_.take(self.compiled_root_.predict_classification(X))

        y_pred_proba = self.predict_proba(X)

        return self.classes_.take(np.argmax(y_pred_proba, axis=1))


class GrootTreeRegressor(BaseGrootTree, RegressorMixin):
    """
    A robust decision tree for single target regression.

    Optimizes the adversarial sum of absolute errors at each split.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        robust_weight=1.0,
        attack_model=None,
        chen_heuristic=False,
        compile=True,
        random_state=None,
    ):
        """
        Parameters
        ----------
        max_depth : int, optional
            The maximum depth for the decision tree once fitted.
        min_samples_split : int, optional
            The minimum number of samples required to split a node.
        min_samples_leaf : int, optional
            The minimum number of samples required to make a leaf.
        max_features : int or {"sqrt", "log2"}, optional
            The number of features to consider while making each split, if None then all features are considered.
        robust_weight : float, optional
            The ratio of samples that are actually moved by an adversary.
        attack_model : array-like of shape (n_features,), optional
            Attacker capabilities for perturbing X. By default, all features are considered not perturbable.
        one_adversarial_class : bool, optional
            Whether one class (malicious, 1) perturbs their samples or if both classes (benign and malicious, 0 and 1) do so.
        chen_heuristic : bool, optional
            Whether to use the heuristic for the adversarial Gini impurity from Chen et al. (2019) instead of GROOT's adversarial Gini impurity.
        compile : bool, optional
            Whether to compile the tree for faster predictions.
        random_state : int, optional
            Controls the sampling of the features to consider when looking for the best split at each node.

        Attributes
        ----------
        classes_ : ndarray of shape (n_classes,)
            The class labels.
        max_features_ : int
            The inferred value of max_features.
        n_samples_ : int
            The number of samples when `fit` is performed.
        n_features_ : int
            The number of features when `fit` is performed.
        root_ : Node
            The root node of the tree after fitting.
        compiled_root_ : CompiledTree
            The compiled root node of the tree after fitting.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.robust_weight = robust_weight
        self.attack_model = attack_model
        self.chen_heuristic = chen_heuristic
        self.compile = compile
        self.random_state = random_state

    def _check_target(self, y):
        target_type = type_of_target(y)
        if target_type not in {"continuous", "multiclass", "binary"}:
            raise ValueError(
                "Unknown label type: regressor only supports"
                f" continuous/multiclass/binary targets but found {target_type}"
            )

        # Make a copy of y if it is readonly to prevent errors
        if not y.flags.writeable:
            y = np.copy(y)

        return y

    def _score(self, y):
        return sum_absolute_errors(y)

    def _create_leaf(self, y):
        """
        Create a leaf object that predicts according to the ratio of benign
        and malicious labels in the array y.
        """

        # For optimal mean absolute error predict the median
        value = np.median(y)

        return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)

    def _scan_feature(self, X, y, feature, constraints):
        """
        Scan a feature to find the locally optimal split.
        """

        samples = X[:, feature]
        attack_mode = self.attack_model_[feature]
        constraint = constraints[feature]

        # If possible, use the faster scan implementation
        if self.robust_weight == 1:
            return _scan_numerical_feature_fast_regression(
                samples, y, *attack_mode, *constraint, self.chen_heuristic
            )
        else:
            raise ValueError("Robust weight not yet supported")

    def _split_left_right(self, X, y, rule, feature):
        """
        Split the dataset (X, y) into a left and right dataset according to the
        optimal split determined by rule and feature.

        Only a ratio 'robust_weight' of moving (malicious) points are actually
        transfered to the other side.
        """

        # Get perturbation range for this feature
        dec, inc = self.attack_model_[feature]

        # Determine the indices of samples on each side of the split
        i_left = np.where(X[:, feature] + inc <= rule)[0]
        i_right = np.where(X[:, feature] - dec > rule)[0]

        y_left = y[i_left]
        y_right = y[i_right]

        if self.chen_heuristic:
            i_left_intersection = np.where(
                (X[:, feature] + inc > rule) & (X[:, feature] <= rule)
            )[0]
            i_right_intersection = np.where(
                (X[:, feature] - dec <= rule) & (X[:, feature] > rule)
            )[0]

            y_left_intersection = y[i_left_intersection]
            y_right_intersection = y[i_right_intersection]

            # Compute optimal movement
            _, configuration = chen_adversarial_sum_absolute_errors(
                y_left,
                y_left_intersection,
                y_right_intersection,
                y_right,
            )

            # Move the resulting intersection arrays to the left and right sides
            if configuration == 1:
                i_left = np.concatenate((i_left, i_left_intersection))
                i_right = np.concatenate((i_right, i_right_intersection))
            elif configuration == 2:
                i_right = np.concatenate(
                    (i_right, i_left_intersection, i_right_intersection)
                )
            elif configuration == 3:
                i_left = np.concatenate(
                    (i_left, i_left_intersection, i_right_intersection)
                )
            else:
                i_left = np.concatenate((i_left, i_right_intersection))
                i_right = np.concatenate((i_right, i_left_intersection))
        else:
            i_intersection = np.where(
                (X[:, feature] + inc > rule) & (X[:, feature] - dec <= rule)
            )[0]
            y_intersection = np.sort(y[i_intersection])

            # Compute optimal movement
            _, (l_start, l_end) = adversarial_sum_absolute_errors(
                y_left, y_right, y_intersection
            )
            if l_start == 0:
                r_start = l_end
                r_end = -1
            else:
                r_start = 0
                r_end = l_start

            # Move the resulting intersection arrays to the left and right sides
            i_left = np.concatenate((i_left, i_intersection[l_start:l_end]))
            i_right = np.concatenate((i_right, i_intersection[r_start:r_end]))

        X_left = X[i_left]
        y_left = y[i_left]
        X_right = X[i_right]
        y_right = y[i_right]

        return X_left, y_left, X_right, y_right

    def predict(self, X):
        """
        Predict the targets of the input samples X.

        The predicted value is the median of values in a leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """

        check_is_fitted(self, "root_")

        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Received different number of features during predict than during fit"
            )

        # If model has been compiled, use compiled predict
        if self.compile:
            return self.compiled_root_.predict_regression(X)

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)


def _build_tree_parallel(base_tree, X, y, indices, seed, n_samples):
    tree = clone(base_tree)

    random_state = check_random_state(seed)
    tree.random_state = random_state

    i_bootstrap = random_state.choice(indices, n_samples)
    X_bootstrap = X[i_bootstrap]
    y_bootstrap = y[i_bootstrap]

    tree.fit(X_bootstrap, y_bootstrap, check_input=False)
    return tree


class BaseGrootRandomForest(BaseEstimator):
    """
    Base class for robust (GROOT) random forests.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        robust_weight=1.0,
        attack_model=None,
        verbose=False,
        chen_heuristic=False,
        max_samples=None,
        n_jobs=None,
        compile=True,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_estimators : int, optional
            The number of decision trees to fit in the forest.
        max_depth : int, optional
            The maximum depth for the decision trees once fitted.
        max_features : int or {"sqrt", "log2", None}, optional
            The number of features to consider while making each split, if None then all features are considered.
        min_samples_split : int, optional
            The minimum number of samples required to split a tree node.
        min_samples_leaf : int, optional
            The minimum number of samples required to make a tree leaf.
        robust_weight : float, optional
            The ratio of samples that are actually moved by an adversary.
        attack_model : array-like of shape (n_features,), optional
            Attacker capabilities for perturbing X. The attack model needs to describe for every feature in which way it can be perturbed.
        verbose : bool, optional
            Whether to print fitting progress on screen.
        chen_heuristic : bool, optional
            Whether to use the heuristic for the adversarial Gini impurity from Chen et al. (2019) instead of GROOT's adversarial Gini impurity.
        max_samples : float, optional
            The fraction of samples to draw from X to train each decision tree. If None (default), then draw X.shape[0] samples.
        n_jobs : int, optional
            The number of jobs to run in parallel when fitting trees. See joblib.
        compile : bool, optional
            Whether to compile decision trees for faster predictions.
        random_state : int, optional
            Controls the sampling of the features to consider when looking for the best split at each node.

        Attributes
        ----------
        estimators_ : list of GrootTree
            The collection of fitted sub-estimators.
        n_samples_ : int
            The number of samples when `fit` is performed.
        n_features_ : int
            The number of features when `fit` is performed.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.robust_weight = robust_weight
        self.attack_model = attack_model
        self.verbose = verbose
        self.chen_heuristic = chen_heuristic
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.compile = compile
        self.random_state = random_state

    def fit(self, X, y):
        """
        Build a robust and fair random forest of binary decision trees from
        the training set (X, y) using greedy splitting according to the
        adversarial Gini combined with fair gini under the given attack model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training samples.
        y : array-like of shape (n_samples,)
            The class labels as integers 0 (benign) or 1 (malicious)

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X, y = check_X_y(X, y)
        y = self._check_target(y)

        self.n_samples_, self.n_features_in_ = X.shape

        random_state = check_random_state(self.random_state)

        # Generate seeds for the random states of each tree to prevent each
        # of them from fitting exactly the same way, but use the random state
        # to keep the forest reproducible
        seeds = [
            random_state.randint(np.iinfo(np.int32).max)
            for _ in range(self.n_estimators)
        ]

        tree = self._get_estimator()

        if self.max_samples:
            n_bootstrap_samples = int(self.n_samples_ * self.max_samples)
        else:
            n_bootstrap_samples = self.n_samples_

        indices = np.arange(X.shape[0])
        self.estimators_ = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, prefer="processes"
        )(
            delayed(_build_tree_parallel)(
                tree, X, y, indices, seed, n_bootstrap_samples
            )
            for seed in seeds
        )

        return self

    def __str__(self):
        result = ""
        result += f"Parameters: {self.get_params()}\n"

        if hasattr(self, "estimators_"):
            for tree in self.estimators_:
                result += f"Tree:\n{tree.root_.pretty_print()}\n"
        else:
            result += "Forest has not yet been fitted"

        return result

    def to_json(self, output_file="forest.json"):
        with open(output_file, "w") as fp:
            dictionary = {
                "params": self.get_params(),
            }
            if hasattr(self, "estimators_"):
                dictionary["trees"] = [tree.to_json(None) for tree in self.estimators_]
                json.dump(dictionary, fp, indent=2, default=convert_numpy)
            else:
                dictionary["trees"] = None
                json.dump(dictionary, fp)

    def to_xgboost_json(self, output_file="forest.json"):
        if hasattr(self, "estimators_"):
            dictionary = [tree.to_xgboost_json(None) for tree in self.estimators_]

            if output_file:
                with open(output_file, "w") as fp:
                    json.dump(dictionary, fp, indent=2, default=convert_numpy)
            else:
                return dictionary
        else:
            raise Exception("Forest not yet fitted")


class GrootRandomForestClassifier(BaseGrootRandomForest, ClassifierMixin):
    """
    A robust random forest for binary classification.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        robust_weight=1.0,
        attack_model=None,
        one_adversarial_class=False,
        verbose=False,
        chen_heuristic=False,
        max_samples=None,
        n_jobs=None,
        compile=True,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_estimators : int, optional
            The number of decision trees to fit in the forest.
        max_depth : int, optional
            The maximum depth for the decision trees once fitted.
        max_features : int or {"sqrt", "log2", None}, optional
            The number of features to consider while making each split, if None then all features are considered.
        min_samples_split : int, optional
            The minimum number of samples required to split a tree node.
        min_samples_leaf : int, optional
            The minimum number of samples required to make a tree leaf.
        robust_weight : float, optional
            The ratio of samples that are actually moved by an adversary.
        attack_model : array-like of shape (n_features,), optional
            Attacker capabilities for perturbing X. The attack model needs to describe for every feature in which way it can be perturbed.
        one_adversarial_class : bool, optional
            Whether one class (malicious, 1) perturbs their samples or if both classes (benign and malicious, 0 and 1) do so.
        verbose : bool, optional
            Whether to print fitting progress on screen.
        chen_heuristic : bool, optional
            Whether to use the heuristic for the adversarial Gini impurity from Chen et al. (2019) instead of GROOT's adversarial Gini impurity.
        max_samples : float, optional
            The fraction of samples to draw from X to train each decision tree. If None (default), then draw X.shape[0] samples.
        n_jobs : int, optional
            The number of jobs to run in parallel when fitting trees. See joblib.
        compile : bool, optional
            Whether to compile decision trees for faster predictions.
        random_state : int, optional
            Controls the sampling of the features to consider when looking for the best split at each node.

        Attributes
        ----------
        estimators_ : list of GrootTree
            The collection of fitted sub-estimators.
        n_samples_ : int
            The number of samples when `fit` is performed.
        n_features_ : int
            The number of features when `fit` is performed.
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            robust_weight=robust_weight,
            attack_model=attack_model,
            verbose=verbose,
            chen_heuristic=chen_heuristic,
            max_samples=max_samples,
            n_jobs=n_jobs,
            compile=compile,
            random_state=random_state,
        )
        self.one_adversarial_class = one_adversarial_class

    def _check_target(self, y):
        target_type = type_of_target(y)
        if target_type != "binary":
            raise ValueError(
                "Unknown label type: classifier only supports binary labels but found"
                f" {target_type}"
            )

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        return y

    def _get_estimator(self):
        return GrootTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            robust_weight=self.robust_weight,
            attack_model=self.attack_model,
            one_adversarial_class=self.one_adversarial_class,
            chen_heuristic=self.chen_heuristic,
            compile=self.compile,
        )

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.
        The class probability is the average of the probabilities predicted by
        each decision tree. The probability prediction of each tree is the
        fraction of samples of the same class in the leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        proba : array of shape (n_samples,)
            The probability for each input sample of being malicious.
        """

        check_is_fitted(self, "estimators_")
        X = check_array(X)

        probability_sum = np.zeros((len(X), 2))

        for tree in self.estimators_:
            probabilities = tree.predict_proba(X)
            probability_sum += probabilities

        return probability_sum / self.n_estimators

    def predict(self, X):
        """
        Predict the classes of the input samples X.
        The predicted class is the rounded average of the class labels in
        each predicted leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """

        y_pred_proba = self.predict_proba(X)

        return self.classes_.take(np.argmax(y_pred_proba, axis=1))


class GrootRandomForestRegressor(BaseGrootRandomForest, RegressorMixin):
    """
    A robust random forest for binary classification.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        robust_weight=1.0,
        attack_model=None,
        verbose=False,
        chen_heuristic=False,
        max_samples=None,
        n_jobs=None,
        compile=True,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_estimators : int, optional
            The number of decision trees to fit in the forest.
        max_depth : int, optional
            The maximum depth for the decision trees once fitted.
        max_features : int or {"sqrt", "log2", None}, optional
            The number of features to consider while making each split, if None then all features are considered.
        min_samples_split : int, optional
            The minimum number of samples required to split a tree node.
        min_samples_leaf : int, optional
            The minimum number of samples required to make a tree leaf.
        robust_weight : float, optional
            The ratio of samples that are actually moved by an adversary.
        attack_model : array-like of shape (n_features,), optional
            Attacker capabilities for perturbing X. The attack model needs to describe for every feature in which way it can be perturbed.
        verbose : bool, optional
            Whether to print fitting progress on screen.
        chen_heuristic : bool, optional
            Whether to use the heuristic for the adversarial Gini impurity from Chen et al. (2019) instead of GROOT's adversarial Gini impurity.
        max_samples : float, optional
            The fraction of samples to draw from X to train each decision tree. If None (default), then draw X.shape[0] samples.
        n_jobs : int, optional
            The number of jobs to run in parallel when fitting trees. See joblib.
        compile : bool, optional
            Whether to compile decision trees for faster predictions.
        random_state : int, optional
            Controls the sampling of the features to consider when looking for the best split at each node.

        Attributes
        ----------
        estimators_ : list of GrootTree
            The collection of fitted sub-estimators.
        n_samples_ : int
            The number of samples when `fit` is performed.
        n_features_ : int
            The number of features when `fit` is performed.
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            robust_weight=robust_weight,
            attack_model=attack_model,
            verbose=verbose,
            chen_heuristic=chen_heuristic,
            max_samples=max_samples,
            n_jobs=n_jobs,
            compile=compile,
            random_state=random_state,
        )

    def _check_target(self, y):
        target_type = type_of_target(y)
        if target_type not in {"continuous", "multiclass", "binary"}:
            raise ValueError(
                "Unknown label type: regressor only supports"
                f" continuous/multiclass/binary targets but found {target_type}"
            )

        # Make a copy of y if it is readonly to prevent errors
        if not y.flags.writeable:
            y = np.copy(y)

        return y

    def _get_estimator(self):
        return GrootTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            robust_weight=self.robust_weight,
            attack_model=self.attack_model,
            chen_heuristic=self.chen_heuristic,
            compile=self.compile,
        )

    def predict(self, X):
        """
        Predict the values of the input samples X.

        The predicted values are the means of all individual predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """

        check_is_fitted(self, "estimators_")
        X = check_array(X)

        predictions_sum = np.zeros(len(X))

        for tree in self.estimators_:
            predictions_sum += tree.predict(X)

        return predictions_sum / self.n_estimators
