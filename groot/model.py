import json
import numbers
import time
import numpy as np
from collections import defaultdict
from itertools import product
from numba import jit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
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

        n_samples_in_leaf = np.sum(self.value)
        if n_samples_in_leaf == 0:
            # By default predict malicious
            return [0.0, 1.0]
        else:
            return self.value / n_samples_in_leaf

    def pretty_print(self, depth=0):
        indentation = depth * "  "
        return f"{indentation}return [{self.value[0]:.3f}, {self.value[1]:.3f}]"

    def to_json(self):
        return {
            "value": [self.value[0], self.value[1]],
        }

    def to_xgboost_json(self, node_id, depth):
        # Return leaf value in range [-1, 1]
        return {"nodeid": node_id, "leaf": self.value[1] * 2 - 1}, node_id

    def is_leaf(self):
        return self.left_child == _TREE_LEAF and self.right_child == _TREE_LEAF

    def prune(self, _):
        return self


class CategoricalNode(Node):
    """
    Decision tree node for categorical decision (category filter).
    """

    def __init__(self, feature, category_split, left_child, right_child, value):
        super().__init__(feature, left_child, right_child, value)
        self.categories_left = category_split[0]
        self.categories_right = category_split[1]

    def predict(self, sample):
        """
        Predict the class label of the given sample. Follow the left sub tree
        if the sample's category points to a 0 in the mask, follow the right
        if the sample's category points to a 1.
        """
        sample_feature = int(sample[self.feature])
        if sample_feature in self.categories_left:
            return self.left_child.predict(sample)
        else:
            return self.right_child.predict(sample)

    def pretty_print(self, depth=0):
        indentation = depth * "  "
        return f"""{indentation}if x{self.feature} in {self.categories_left}:
{self.left_child.pretty_print(depth + 1)}
{indentation}elif x{self.feature} in {self.categories_right}:
{self.right_child.pretty_print(depth + 1)}"""

    def to_json(self):
        return {
            "feature": self.feature,
            "categories_left": list(self.categories_left),
            "categories_right": list(self.categories_right),
            "left_child": self.left_child.to_json(),
            "right_child": self.right_child.to_json(),
        }

    def to_xgboost_json(self, node_id, depth):
        raise NotImplementedError(
            "XGBoost JSON is not yet supported for categorical features"
        )


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

    def pretty_print(self, depth=0):
        indentation = depth * "  "
        return f"""{indentation}if x{self.feature} <= {self.threshold}:
{self.left_child.pretty_print(depth + 1)}
{indentation}else:
{self.right_child.pretty_print(depth + 1)}"""

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


def _attack_model_to_tuples(attack_model):
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
        elif isinstance(attack_mode, dict):
            new_attack_model.append(attack_mode)
        else:
            raise Exception("Unknown attack model spec:", attack_mode)
    return new_attack_model


@jit(nopython=True, nogil=NOGIL)
def _split_left_right_fast(self, X, y, rule, feature, inc, dec, chen_heuristic):
    # Find the boolean mask of samples in the sets L, LI, RI and R
    b_L = X[:, feature] <= rule - dec
    b_LI = (X[:, feature] <= rule) & (X[:, feature] > rule - dec)
    b_RI = (X[:, feature] <= rule + inc) & (X[:, feature] > rule)
    b_R = X[:, feature] > rule + inc

    # Find the indices of the samples in each set-label combination
    i_L_0 = np.where(b_L & (y == 0))[0]
    i_L_1 = np.where(b_L & (y == 1))[0]
    i_LI_0 = np.where(b_LI & (y == 0))[0]
    i_LI_1 = np.where(b_LI & (y == 1))[0]
    i_RI_0 = np.where(b_RI & (y == 0))[0]
    i_RI_1 = np.where(b_RI & (y == 1))[0]
    i_R_0 = np.where(b_R & (y == 0))[0]
    i_R_1 = np.where(b_R & (y == 1))[0]

    # Determine optimal movement for intersection samples
    if chen_heuristic:
        _, x, y = chen_adversarial_gini_gain_two_class(
            len(i_L_0),
            len(i_L_1),
            len(i_LI_0),
            len(i_LI_1),
            len(i_RI_0),
            len(i_RI_1),
            len(i_R_0),
            len(i_R_1),
        )
    else:
        _, x, y = adversarial_gini_gain_two_class(
            len(i_L_0),
            len(i_L_1),
            len(i_LI_0),
            len(i_LI_1),
            len(i_RI_0),
            len(i_RI_1),
            len(i_R_0),
            len(i_R_1),
        )

    # TODO: add randomization

    # If there are fewer samples in LI_0 than y, we need to move some samples
    # from RI_0 into it
    if len(i_LI_0) < y:
        i_LI_0 = np.append(i_LI_0, i_RI_0[: y - len(i_LI_0)])
        i_RI_0 = i_RI_0[y - len(i_LI_0) :]
    elif len(i_LI_0) > y:
        i_RI_0 = np.append(i_RI_0, i_LI_0[: len(i_LI_0) - y])
        i_LI_0 = i_LI_0[: len(i_LI_0) - y]

    # If there are fewer samples in LI_1 than x, we need to move some samples
    # from RI_1 into it
    if len(i_LI_1) < x:
        i_LI_1 = np.append(i_LI_1, i_RI_1[: x - len(i_LI_1)])
        i_RI_1 = i_RI_1[x - len(i_LI_1) :]
    elif len(i_LI_1) > x:
        i_RI_1 = np.append(i_RI_1, i_LI_1[: len(i_LI_1) - x])
        i_LI_1 = i_LI_1[: len(i_LI_1) - x]

    i_left = np.concatenate((i_RI_0, i_RI_1, i_R_0, i_R_1))
    i_right = np.concatenate((i_RI_0, i_RI_1, i_R_0, i_R_1))

    return X[i_left], y[i_left], X[i_right], y[i_right]


@jit(nopython=True, nogil=NOGIL)
def _scan_numerical_feature_fast(
    samples,
    y,
    dec,
    inc,
    left_bound,
    right_bound,
    chen_heuristic,
):
    # TODO: so far we assume attack_mode is a tuple (dec, inc), and both
    # classes can move
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

    # Returns True to indicate numeric decision
    return True, best_score, best_split


@jit(nopython=True, nogil=NOGIL)
def chen_adversarial_gini_gain_one_class(l_0, l_1, r_0, r_1, i_1):
    raise NotImplementedError("Not discussed in the paper by Chen et al.")


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

    x_floor = np.floor(x)
    x_ceil = np.ceil(x)
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
    denominator = x_coef ** 2 + 1

    # In the paper we refer to a, b here they are li_1 and li_0
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
    return 1.0 - (ratio ** 2) - ((1 - ratio) ** 2)


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

    gini = l_t * (1 - (l_p ** 2) - ((1 - l_p) ** 2)) + r_t * (
        1 - (r_p ** 2) - ((1 - r_p) ** 2)
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
            left_i_mal + right_i_mal,
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


@jit(nopython=True, nogil=NOGIL)
def _categorical_counts_to_one_class_adv_gini(
    left_counts,
    left_intersection_counts,
    right_intersection_counts,
    right_counts,
    rho,
    chen_heuristic,
):
    # Apply rho by moving back a number of intersection samples
    inv_rho = 1.0 - rho
    left_counts += np.rint(inv_rho * left_intersection_counts).astype(np.int64)
    right_counts += np.rint(inv_rho * right_intersection_counts).astype(np.int64)
    left_intersection_counts = np.rint(rho * left_intersection_counts).astype(np.int64)
    right_intersection_counts = np.rint(rho * right_intersection_counts).astype(
        np.int64
    )

    # Compute adversarial Gini gain
    if chen_heuristic:
        adv_gini, _ = chen_adversarial_gini_gain_one_class(
            left_counts[0],
            left_counts[1],
            right_counts[0],
            right_counts[1],
            left_intersection_counts[1] + right_intersection_counts[1],
        )
    else:
        adv_gini, _ = adversarial_gini_gain_one_class(
            left_counts[0],
            left_counts[1],
            right_counts[0],
            right_counts[1],
            left_intersection_counts[1] + right_intersection_counts[1],
        )

    return adv_gini


@jit(nopython=True, nogil=NOGIL)
def _categorical_counts_to_two_class_adv_gini(
    left_counts,
    left_intersection_counts,
    right_intersection_counts,
    right_counts,
    rho,
    chen_heuristic,
):
    # Apply rho by moving back a number of intersection samples
    inv_rho = 1.0 - rho
    left_counts += np.rint(inv_rho * left_intersection_counts).astype(np.int64)
    right_counts += np.rint(inv_rho * right_intersection_counts).astype(np.int64)
    left_intersection_counts = np.rint(rho * left_intersection_counts).astype(np.int64)
    right_intersection_counts = np.rint(rho * right_intersection_counts).astype(
        np.int64
    )

    # Compute adversarial Gini gain
    if chen_heuristic:
        adv_gini, _, _ = chen_adversarial_gini_gain_two_class(
            left_counts[0],
            left_counts[1],
            left_intersection_counts[0],
            left_intersection_counts[1],
            right_intersection_counts[0],
            right_intersection_counts[1],
            right_counts[0],
            right_counts[1],
        )
    else:
        adv_gini, _, _ = adversarial_gini_gain_two_class(
            left_counts[0],
            left_counts[1],
            left_intersection_counts[0],
            left_intersection_counts[1],
            right_intersection_counts[0],
            right_intersection_counts[1],
            right_counts[0],
            right_counts[1],
        )

    return adv_gini


def _identify_intersection_categories(
    left_categories,
    right_categories,
    categories_counts,
    attack_mode_array,
    one_adversarial_class,
):
    # Sum category counts if the category can be perturbed to the other side
    left_intersection_mask = np.any(attack_mode_array[:, right_categories], axis=1)
    left_intersection_mask[right_categories] = 0
    left_intersection_counts = np.sum(categories_counts[left_intersection_mask], axis=0)

    right_intersection_mask = np.any(attack_mode_array[:, left_categories], axis=1)
    right_intersection_mask[left_categories] = 0
    right_intersection_counts = np.sum(
        categories_counts[right_intersection_mask], axis=0
    )

    # If only one class moves then reset the number of benign intersection
    # samples back to zero
    if one_adversarial_class:
        left_intersection_counts[0] = 0
        right_intersection_counts[0] = 0

    return left_intersection_counts, right_intersection_counts


class GrootTree(BaseEstimator, ClassifierMixin):
    """
    A robust and fair decision tree for binary classification. Use class 0 for a negative and class 1 for a positive label.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        robust_weight=1.0,
        attack_model=None,
        is_numerical=None,
        one_adversarial_class=False,
        chen_heuristic=False,
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
        is_numerical : array-like of shape (n_features,), optional
            Boolean mask for whether each feature is numerical or categorical. By default, all features are considered numerical.
        one_adversarial_class : bool, optional
            Whether one class (malicious, 1) perturbs their samples or if both classes (benign and malicious, 0 and 1) do so.
        chen_heuristic : bool, optional
            Whether to use the heuristic for the adversarial Gini impurity from Chen et al. (2019) instead of GROOT's adversarial Gini impurity.
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
        n_categories_ : list of length n_features
            The number of categories occuring in each feature. Each position
            in the list is a positive integer for if that feature is
            categorical and None if numerical.
        root_ : Node
            The root node of the tree after fitting.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.robust_weight = robust_weight
        self.attack_model = attack_model
        self.is_numerical = is_numerical
        self.one_adversarial_class = one_adversarial_class
        self.chen_heuristic = chen_heuristic
        self.random_state = random_state

        # Turn numerical features in attack model into tuples to make fitting
        # code simpler
        self.attack_model_ = _attack_model_to_tuples(attack_model)

    def fit(self, X, y):
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

        X, y = check_X_y(X, y)
        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = np.unique(y)

        if self.attack_model is None:
            self.attack_model = [""] * X.shape[1]

        if self.is_numerical is None:
            self.is_numerical = [True] * X.shape[1]

        self.n_categories_ = []
        for feature, numeric in enumerate(self.is_numerical):
            if numeric:
                self.n_categories_.append(None)
            else:
                self.n_categories_.append(int(np.max(X[:, feature])) + 1)

        self.random_state_ = check_random_state(self.random_state)

        if self.max_features == "sqrt":
            self.max_features_ = int(np.sqrt(self.n_features_))
        elif self.max_features == "log2":
            self.max_features_ = int(np.log2(self.n_features_))
        elif self.max_features is None:
            self.max_features_ = self.n_features_
        else:
            self.max_features_ = self.max_features

        if self.max_features_ == 0:
            self.max_features_ = 1

        # For each feature set the initial constraints for splitting
        constraints = []
        for feature_i, numeric in enumerate(self.is_numerical):
            if numeric:
                # Numeric splits can occur anywhere in the space of the feature
                constraints.append([np.min(X[:, feature_i]), np.max(X[:, feature_i])])
            else:
                # Categorical filters can contain every category
                constraints.append(set())

        self.root_ = self.__fit_recursive(X, y, constraints)

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
            or np.sum(y == 0) == 0
            or np.sum(y == 1) == 0
        ):
            return self.__create_leaf(y)

        current_gini = gini_impurity(np.sum(y == 0), np.sum(y == 1))

        numeric, rule, feature, split_gini = self.__best_adversarial_decision(
            X, y, constraints
        )

        gini_gain = current_gini - split_gini

        if rule is None or gini_gain <= 0.00:
            return self.__create_leaf(y)

        if numeric:
            # Assert that the split obeys constraints made by previous splits
            assert rule >= constraints[feature][0]
            assert rule < constraints[feature][1]
        else:
            # Assert that category split does not contain previous categories
            assert rule[0].isdisjoint(constraints[feature])
            assert rule[1].isdisjoint(constraints[feature])

        # TODO: fix this
        X_left, y_left, X_right, y_right = self.__split_left_right(
            X, y, rule, feature, numeric, self.attack_model_[feature]
        )
        # if self.robust_weight == 1 and all(self.is_numerical):
        #     X_left, y_left, X_right, y_right = _split_left_right_fast(
        #         X, y, rule, feature, numeric, self.attack_model_[feature]
        #     )
        # else:
        #     X_left, y_left, X_right, y_right = self.__split_left_right(
        #         X, y, rule, feature, numeric, self.attack_model_[feature]
        #     )

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return self.__create_leaf(y)

        if numeric:
            # Set the right bound and store old one for after recursion
            old_right_bound = constraints[feature][1]
            constraints[feature][1] = rule
        else:
            # Ignore categories moving to the right and store old constraints
            constraints[feature].update(rule[1])

        left_node = self.__fit_recursive(X_left, y_left, constraints, depth + 1)

        if numeric:
            # Reset right bound, set left bound, store old one for after recursion
            constraints[feature][1] = old_right_bound
            old_left_bound = constraints[feature][0]
            constraints[feature][0] = rule
        else:
            # Ignore categories going to the left
            constraints[feature].difference_update(rule[1])
            constraints[feature].update(rule[0])

        right_node = self.__fit_recursive(X_right, y_right, constraints, depth + 1)

        if numeric:
            # Reset the left bound
            constraints[feature][0] = old_left_bound

            node = NumericalNode(feature, rule, left_node, right_node, _TREE_UNDEFINED)
        else:
            # Reset the categories to ignore
            constraints[feature].difference_update(rule[0])

            node = CategoricalNode(
                feature, rule, left_node, right_node, _TREE_UNDEFINED
            )

        return node

    def __create_leaf(self, y):
        """
        Create a leaf object that predicts according to the ratio of benign
        and malicious labels in the array y.
        """

        # Count the number of points that fall into this leaf including
        # adversarially moved points
        label_counts = np.bincount(y)

        # Fix leaves with only 0 (benign) labels
        if len(label_counts) == 1:
            label_counts = np.array([label_counts[0], 0])

        # Set the leaf's prediction value to the weighted average of the
        # prediction with and without moving points
        value = label_counts / np.sum(label_counts)

        return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)

    def __transfer_n_samples(self, X_from, y_from, X_to, y_to, n):
        n_from = len(X_from)
        indices_to = self.random_state_.choice(n_from, size=int(n), replace=False)
        indices_from = np.setdiff1d(np.arange(n_from), indices_to).astype(np.int64)
        X_to = np.append(X_to, X_from[indices_to], axis=0)
        y_to = np.append(y_to, y_from[indices_to])
        X_from = X_from[indices_from]
        y_from = y_from[indices_from]
        return X_from, y_from, X_to, y_to

    def __split_left_right(self, X, y, rule, feature, numeric, attack_mode):
        """
        Split the dataset (X, y) into a left and right dataset according to the
        optimal split determined by rule, feature, numeric and attack_mode.

        Only a ratio 'robust_weight' of moving (malicious) points are actually
        transfered to the other side.
        """

        X_left = []
        y_left = []
        X_left_intersection = []
        y_left_intersection = []
        X_right_intersection = []
        y_right_intersection = []
        X_right = []
        y_right = []

        if numeric:
            samples = X[:, feature]

            for sample, value, label in zip(X, samples, y):
                sample_in_intersection = False

                if label == 1 or not self.one_adversarial_class:
                    # TODO: move this to a method of an attack model class and
                    # clean this up

                    if attack_mode == ">":
                        if value <= rule:
                            sample_in_intersection = True
                    elif attack_mode == "<":
                        if value >= rule:
                            sample_in_intersection = True
                    elif attack_mode == "<>":
                        sample_in_intersection = True
                    elif isinstance(attack_mode, numbers.Number):
                        if abs(value - rule) <= attack_mode:
                            sample_in_intersection = True
                    elif isinstance(attack_mode, tuple):
                        if value <= rule and value + attack_mode[1] > rule:
                            sample_in_intersection = True
                        elif value > rule and value - attack_mode[0] <= rule:
                            sample_in_intersection = True

                if sample_in_intersection:
                    if value <= rule:
                        X_left_intersection.append(sample)
                        y_left_intersection.append(label)
                    else:
                        X_right_intersection.append(sample)
                        y_right_intersection.append(label)
                else:
                    # Place points that are not in the intersection normally
                    if value <= rule:
                        X_left.append(sample)
                        y_left.append(label)
                    else:
                        X_right.append(sample)
                        y_right.append(label)
        else:
            values = X[:, feature].astype(np.int64)

            for sample, value, label in zip(X, values, y):
                sample_in_intersection = False

                if label == 1 or not self.one_adversarial_class:
                    # TODO: move this to a method of an attack model class and
                    # clean this up

                    side = 0 if value in rule[0] else 1

                    if isinstance(attack_mode, dict):
                        if value in attack_mode:
                            attack = attack_mode[value]
                            if isinstance(attack, int):
                                if attack in rule[1 - side]:
                                    sample_in_intersection = True
                            elif isinstance(attack, list) or isinstance(attack, tuple):
                                for attack_category in attack:
                                    if attack_category in rule[1 - side]:
                                        sample_in_intersection = True
                                        break

                if sample_in_intersection:
                    if value in rule[0]:
                        X_left_intersection.append(sample)
                        y_left_intersection.append(label)
                    else:
                        X_right_intersection.append(sample)
                        y_right_intersection.append(label)
                else:
                    # Place points that are not in the intersection normally
                    if value in rule[0]:
                        X_left.append(sample)
                        y_left.append(label)
                    else:
                        X_right.append(sample)
                        y_right.append(label)

        # TODO: refactor this entire part, we can do all operations on indices
        # and only create new arrays at the end
        X_left = np.array(X_left).reshape(-1, self.n_features_)
        y_left = np.array(y_left)
        X_left_intersection = np.array(X_left_intersection).reshape(
            -1, self.n_features_
        )
        y_left_intersection = np.array(y_left_intersection)
        X_right_intersection = np.array(X_right_intersection).reshape(
            -1, self.n_features_
        )
        y_right_intersection = np.array(y_right_intersection)
        X_right = np.array(X_right).reshape(-1, self.n_features_)
        y_right = np.array(y_right)

        # Compute optimal movement
        if self.one_adversarial_class:
            # Compute numbers of samples after applying rho
            assert np.sum(y_left_intersection == 0) == 0
            li_1 = np.sum(y_left_intersection == 1)
            assert np.sum(y_right_intersection == 0) == 0
            ri_1 = np.sum(y_right_intersection == 1)
            l_0 = np.sum(y_left == 0)
            l_1 = np.sum(y_left == 1) + round((1.0 - self.robust_weight) * li_1)
            r_0 = np.sum(y_right == 0)
            r_1 = np.sum(y_right == 1) + round((1.0 - self.robust_weight) * ri_1)
            li_1 = round(self.robust_weight * li_1)
            ri_1 = round(self.robust_weight * ri_1)
            i_1 = li_1 + ri_1

            # Determine optimal movement
            if self.chen_heuristic:
                _, x = chen_adversarial_gini_gain_one_class(l_0, l_1, r_0, r_1, i_1)
            else:
                _, x = adversarial_gini_gain_one_class(l_0, l_1, r_0, r_1, i_1)

            # Move samples accordingly
            if x > li_1:
                n_move_left = x - li_1
                (
                    X_right_intersection,
                    y_right_intersection,
                    X_left_intersection,
                    y_left_intersection,
                ) = self.__transfer_n_samples(
                    X_right_intersection,
                    y_right_intersection,
                    X_left_intersection,
                    y_left_intersection,
                    n_move_left,
                )
            elif x < li_1:
                n_move_right = li_1 - x
                (
                    X_left_intersection,
                    y_left_intersection,
                    X_right_intersection,
                    y_right_intersection,
                ) = self.__transfer_n_samples(
                    X_left_intersection,
                    y_left_intersection,
                    X_right_intersection,
                    y_right_intersection,
                    n_move_right,
                )
        else:
            # Compute numbers of samples after applying rho
            li_0 = np.sum(y_left_intersection == 0)
            li_1 = np.sum(y_left_intersection == 1)
            ri_0 = np.sum(y_right_intersection == 0)
            ri_1 = np.sum(y_right_intersection == 1)
            l_0 = np.sum(y_left == 0) + round((1.0 - self.robust_weight) * li_0)
            l_1 = np.sum(y_left == 1) + round((1.0 - self.robust_weight) * li_1)
            r_0 = np.sum(y_right == 0) + round((1.0 - self.robust_weight) * ri_0)
            r_1 = np.sum(y_right == 1) + round((1.0 - self.robust_weight) * ri_1)
            li_0 = round(self.robust_weight * li_0)
            li_1 = round(self.robust_weight * li_1)
            ri_0 = round(self.robust_weight * ri_0)
            ri_1 = round(self.robust_weight * ri_1)

            # Determine optimal movement
            if self.chen_heuristic:
                _, x, y = chen_adversarial_gini_gain_two_class(
                    l_0, l_1, li_0, li_1, ri_0, ri_1, r_0, r_1
                )
            else:
                _, x, y = adversarial_gini_gain_two_class(
                    l_0, l_1, li_0, li_1, ri_0, ri_1, r_0, r_1
                )

            # Split intersection arrays on label
            X_left_intersection_0 = X_left_intersection[y_left_intersection == 0]
            y_left_intersection_0 = y_left_intersection[y_left_intersection == 0]
            X_left_intersection_1 = X_left_intersection[y_left_intersection == 1]
            y_left_intersection_1 = y_left_intersection[y_left_intersection == 1]
            X_right_intersection_0 = X_right_intersection[y_right_intersection == 0]
            y_right_intersection_0 = y_right_intersection[y_right_intersection == 0]
            X_right_intersection_1 = X_right_intersection[y_right_intersection == 1]
            y_right_intersection_1 = y_right_intersection[y_right_intersection == 1]

            # Move samples according to x
            if x > li_1:
                n_move_left = x - li_1
                (
                    X_right_intersection_1,
                    y_right_intersection_1,
                    X_left_intersection_1,
                    y_left_intersection_1,
                ) = self.__transfer_n_samples(
                    X_right_intersection_1,
                    y_right_intersection_1,
                    X_left_intersection_1,
                    y_left_intersection_1,
                    n_move_left,
                )
            elif x < li_1:
                n_move_right = li_1 - x
                (
                    X_left_intersection_1,
                    y_left_intersection_1,
                    X_right_intersection_1,
                    y_right_intersection_1,
                ) = self.__transfer_n_samples(
                    X_left_intersection_1,
                    y_left_intersection_1,
                    X_right_intersection_1,
                    y_right_intersection_1,
                    n_move_right,
                )

            # Move samples according to y
            if y > li_0:
                n_move_left = y - li_0
                (
                    X_right_intersection_0,
                    y_right_intersection_0,
                    X_left_intersection_0,
                    y_left_intersection_0,
                ) = self.__transfer_n_samples(
                    X_right_intersection_0,
                    y_right_intersection_0,
                    X_left_intersection_0,
                    y_left_intersection_0,
                    n_move_left,
                )
            elif y < li_0:
                n_move_right = li_0 - y
                (
                    X_left_intersection_0,
                    y_left_intersection_0,
                    X_right_intersection_0,
                    y_right_intersection_0,
                ) = self.__transfer_n_samples(
                    X_left_intersection_0,
                    y_left_intersection_0,
                    X_right_intersection_0,
                    y_right_intersection_0,
                    n_move_right,
                )

            # Merge intersection arrays again
            X_left_intersection = np.append(
                X_left_intersection_0, X_left_intersection_1, axis=0
            )
            y_left_intersection = np.append(
                y_left_intersection_0, y_left_intersection_1
            )
            X_right_intersection = np.append(
                X_right_intersection_0, X_right_intersection_1, axis=0
            )
            y_right_intersection = np.append(
                y_right_intersection_0, y_right_intersection_1
            )

        # Move the resulting intersection arrays to the left and right sides
        X_left = np.append(X_left, X_left_intersection, axis=0)
        y_left = np.append(y_left, y_left_intersection).astype(int)
        X_right = np.append(X_right, X_right_intersection, axis=0)
        y_right = np.append(y_right, y_right_intersection).astype(int)

        return X_left, y_left, X_right, y_right

    def __best_adversarial_decision(self, X, y, constraints):
        """
        Find the best split by iterating through each feature and scanning
        it for that feature's optimal split.
        """

        best_gini = 10e9
        best_rule = None
        best_feature = None
        best_is_numerical = None

        # If there is a limit on features to consider in a split then choose
        # that number of random features.
        all_features = np.arange(self.n_features_)
        features = self.random_state_.choice(
            all_features, size=self.max_features_, replace=False
        )

        for feature in features:
            numeric, gini, decision_rule = self.__scan_feature(
                X, y, feature, constraints
            )

            if decision_rule is not None and gini < best_gini:
                best_gini = gini
                best_rule = decision_rule
                best_feature = feature
                best_is_numerical = numeric

        return best_is_numerical, best_rule, best_feature, best_gini

    def __scan_feature(self, X, y, feature, constraints):
        """
        Depending on the type of feature (numerical / categorical) scan it for
        the optimal split.
        """

        samples = X[:, feature]
        attack_mode = self.attack_model_[feature]
        numeric = self.is_numerical[feature]
        constraint = constraints[feature]

        if numeric:
            # If possible, use the faster scan implementation
            if self.robust_weight == 1:
                return _scan_numerical_feature_fast(
                    samples, y, *attack_mode, *constraint, self.chen_heuristic
                )
            else:
                return self.__scan_feature_numerical(
                    samples, y, attack_mode, *constraint
                )
        else:
            n_categories = self.n_categories_[feature]
            return self.__scan_feature_categorical(
                samples,
                y,
                attack_mode,
                n_categories,
                constraint,
            )

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

        # Returns True to indicate numeric decision
        return True, best_score, best_split

    def __scan_feature_categorical(
        self,
        samples,
        y,
        attack_mode,
        n_categories,
        ignore_categories,
    ):
        """
        Scan a categorical feature for the optimal split either by brute
        forcing (there are usually few categories) or by searching through
        their sorted order.

        Their sorted order is determined by the ratio of benign / malicious
        samples for each category.
        """
        samples = samples.astype(np.int64)
        categories_counts = np.zeros((n_categories, 2), dtype=int)

        # Count the number of occurences of each category by label
        categories_count_0 = np.bincount(samples[y == 0])
        categories_counts[: len(categories_count_0), 0] += categories_count_0

        categories_count_1 = np.bincount(samples[y == 1])
        categories_counts[: len(categories_count_1), 1] += categories_count_1

        remaining_categories = np.setdiff1d(
            np.arange(n_categories), list(ignore_categories), assume_unique=True
        )
        categories_counts = categories_counts[remaining_categories]
        n_remaining_categories = len(remaining_categories)

        best_gini = float("inf")
        best_left_categories = None

        # Compute the matrix that describes what categories can be perturbed
        # and to what category read as: row x can be perturbed to column y
        if attack_mode == "":
            attack_mode_matrix = None
        else:
            category_mapping = {}
            for i, category in enumerate(remaining_categories):
                category_mapping[category] = i

            attack_mode_matrix = np.zeros(
                (n_remaining_categories, n_remaining_categories), dtype=bool
            )
            for category in remaining_categories:
                if category in attack_mode:
                    for attack_category in attack_mode[category]:
                        if attack_category in remaining_categories:
                            cat_i = category_mapping[category]
                            attack_cat_i = category_mapping[attack_category]
                            attack_mode_matrix[cat_i, attack_cat_i] = True

        # Use the linear time (in categories) algorithm for the best
        # categorical split if there is no attack or if there are too many
        # features for an exponential-time search, else perform the search.
        if n_remaining_categories == 1:
            return False, best_gini, None
        if attack_mode == "" or n_remaining_categories > 20:
            total_per_category = np.sum(categories_counts, axis=1)
            total_per_category[total_per_category == 0] = 1  # Prevent division by 0

            probabilities = categories_counts[:, 1] / total_per_category

            category_probability_order = np.argsort(probabilities)
            # TODO: if possible reduce the length of this loop by 1
            for split in range(1, len(category_probability_order)):
                # Category values on each side of the split
                left_categories = category_probability_order[:split]
                right_categories = category_probability_order[split:]

                score = self.__score_categorical_split(
                    left_categories,
                    right_categories,
                    attack_mode_matrix,
                    categories_counts,
                )

                if score < best_gini:
                    best_gini = score
                    best_left_categories = left_categories
                    best_right_categories = right_categories

        else:
            for category_split in product([0, 1], repeat=n_remaining_categories):
                category_split = np.array(category_split)

                # Category values on each side of the split
                left_categories = np.arange(n_remaining_categories)[category_split == 0]
                right_categories = np.arange(n_remaining_categories)[
                    category_split == 1
                ]

                if len(left_categories) == 0 or len(right_categories) == 0:
                    continue

                score = self.__score_categorical_split(
                    left_categories,
                    right_categories,
                    attack_mode_matrix,
                    categories_counts,
                )

                if score < best_gini:
                    best_gini = score
                    best_left_categories = left_categories
                    best_right_categories = right_categories

        # If no good split is found
        if best_left_categories is None:
            return False, best_gini, None

        # Map the best found remaining category split to all category IDs
        left_categories = set(remaining_categories[best_left_categories])
        right_categories = set(remaining_categories[best_right_categories])

        # Returns False to indicate categorical decision
        return (
            False,
            best_gini,
            (left_categories, right_categories),
        )

    def __score_categorical_split(
        self,
        left_categories,
        right_categories,
        attack_mode_matrix,
        categories_counts,
    ):
        # Count 0 and 1 labels on each side of the split
        left_counts = np.sum(categories_counts[left_categories], axis=0)
        right_counts = np.sum(categories_counts[right_categories], axis=0)

        if attack_mode_matrix is None:
            left_intersection_counts = np.zeros(2, dtype=np.int64)
            right_intersection_counts = np.zeros(2, dtype=np.int64)
        else:
            (
                left_intersection_counts,
                right_intersection_counts,
            ) = _identify_intersection_categories(
                left_categories,
                right_categories,
                categories_counts,
                attack_mode_matrix,
                self.one_adversarial_class,
            )

            # Intersection samples were counted twice (e.g. both in left and
            # left_intersection) so remove them from left and right
            left_counts -= left_intersection_counts
            right_counts -= right_intersection_counts

        if self.one_adversarial_class:
            return _categorical_counts_to_one_class_adv_gini(
                left_counts,
                left_intersection_counts,
                right_intersection_counts,
                right_counts,
                self.robust_weight,
                self.chen_heuristic,
            )
        else:
            return _categorical_counts_to_two_class_adv_gini(
                left_counts,
                left_intersection_counts,
                right_intersection_counts,
                right_counts,
                self.robust_weight,
                self.chen_heuristic,
            )

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

        X = check_array(X)

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)

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

        X = check_array(X)

        return np.round(self.predict_proba(X)[:, 1])

    def to_string(self):
        result = ""
        result += f"Parameters: {self.get_params()}\n"

        if hasattr(self, "root_"):
            result += f"Tree:\n{self.root_.pretty_print()}"
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
        if hasattr(self, "root_"):
            dictionary, _ = self.root_.to_xgboost_json(0, 0)
        else:
            raise Exception("Tree is not yet fitted")

        if output_file is None:
            return dictionary
        else:
            with open(output_file, "w") as fp:
                # If saving to file then surround dict in list brackets
                json.dump([dictionary], fp, indent=2, default=convert_numpy)


def parse_json_attack_model(json_attack_model):
    attack_model = []
    for attack in json_attack_model:
        if isinstance(attack, list):
            attack_model.append(tuple(attack))
        elif isinstance(attack, dict):
            attack_model.append({int(key): value for key, value in attack.items()})
        else:
            attack_model.append(attack)
    return attack_model


def load_groot_from_json(filename, tree_dict=None):
    """Load GROOT tree from json file."""

    def dict_to_tree_rec(d):
        is_leaf = "value" in d
        if is_leaf:
            return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, np.array(d["value"]))
        else:
            is_numerical = "threshold" in d
            if is_numerical:
                left_child = dict_to_tree_rec(d["left_child"])
                right_child = dict_to_tree_rec(d["right_child"])
                return NumericalNode(
                    d["feature"],
                    d["threshold"],
                    left_child,
                    right_child,
                    _TREE_UNDEFINED,
                )
            else:
                left_child = dict_to_tree_rec(d["left_child"])
                right_child = dict_to_tree_rec(d["right_child"])
                return CategoricalNode(
                    d["feature"],
                    (set(d["categories_left"]), set(d["categories_right"])),
                    left_child,
                    right_child,
                    _TREE_UNDEFINED,
                )

    if tree_dict is None:
        with open(filename, "r") as fp:
            tree_dict = json.load(fp)

    tree = GrootTree()
    tree.set_params(**tree_dict["params"])
    tree.attack_model = parse_json_attack_model(tree_dict["params"]["attack_model"])
    tree.root_ = dict_to_tree_rec(tree_dict["tree"])
    return tree


def _build_tree_parallel(base_tree, X, y, indices, seed, verbose, n_samples):
    tree = clone(base_tree)

    random_state = check_random_state(seed)
    tree.random_state = random_state

    i_bootstrap = random_state.choice(indices, n_samples)
    X_bootstrap = X[i_bootstrap]
    y_bootstrap = y[i_bootstrap]

    tree.fit(X_bootstrap, y_bootstrap)
    return tree


class GrootRandomForest(BaseEstimator, ClassifierMixin):
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
        is_numerical=None,
        one_adversarial_class=False,
        verbose=False,
        chen_heuristic=False,
        max_samples=None,
        n_jobs=None,
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
        is_numerical : array-like of shape (n_features,), optional
            Boolean mask for whether each feature is numerical or categorical.
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
        self.is_numerical = is_numerical
        self.one_adversarial_class = one_adversarial_class
        self.verbose = verbose
        self.chen_heuristic = chen_heuristic
        self.max_samples = max_samples
        self.n_jobs = n_jobs
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

        self.n_samples_, self.n_features_ = X.shape

        random_state = check_random_state(self.random_state)

        # Generate seeds for the random states of each tree to prevent each
        # of them from fitting exactly the same way, but use the random state
        # to keep the forest reproducible
        seeds = [
            random_state.randint(np.iinfo(np.int32).max)
            for _ in range(self.n_estimators)
        ]

        tree = GrootTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            robust_weight=self.robust_weight,
            attack_model=self.attack_model,
            is_numerical=self.is_numerical,
            one_adversarial_class=self.one_adversarial_class,
            chen_heuristic=self.chen_heuristic,
            random_state=random_state,
        )

        if self.max_samples:
            n_bootstrap_samples = int(self.n_samples_ * self.max_samples)
        else:
            n_bootstrap_samples = self.n_samples_

        indices = np.arange(n_bootstrap_samples)
        self.estimators_ = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, prefer="processes"
        )(
            delayed(_build_tree_parallel)(
                tree, X, y, indices, seed, self.verbose, n_bootstrap_samples
            )
            for seed in seeds
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
        probability_sum = np.zeros((self.n_samples_, 2))

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

        return np.round(self.predict_proba(X)[:, 1])

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


class JsonTree(BaseEstimator, ClassifierMixin):
    def __init__(
        self, one_adversarial_class=False, attack_model=None, is_numerical=None
    ):
        self.one_adversarial_class = one_adversarial_class
        self.attack_model = attack_model
        self.is_numerical = is_numerical

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

        if not hasattr(self, "root_"):
            raise Exception("Tree has not been fitted")

        X = check_array(X)

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)

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

        X = check_array(X)

        return np.round(self.predict_proba(X))


def _tree_dict_to_nodes_rec(tree_dict, round_digits):
    if "leaf" in tree_dict:
        proba = tree_dict["leaf"]
        value = np.array([1 - proba, proba])
        return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)
    else:
        for child in tree_dict["children"]:
            if child["nodeid"] == tree_dict["yes"]:
                left_child = _tree_dict_to_nodes_rec(child, round_digits)
            if child["nodeid"] == tree_dict["no"]:
                right_child = _tree_dict_to_nodes_rec(child, round_digits)

        return NumericalNode(
            tree_dict["split"],
            round(tree_dict["split_condition"], round_digits),
            left_child,
            right_child,
            _TREE_UNDEFINED,
        )


def json_tree_from_file(
    filename,
    one_adversarial_class=False,
    attack_model=None,
    is_numerical=None,
    round_digits=6,
):
    with open(filename, "r") as file:
        loaded_json = json.load(file)

        assert isinstance(loaded_json, list)  # Check that it's list of trees
        assert len(loaded_json) == 1  # Check that we only have one tree

        tree_dict = loaded_json[0]

    tree = JsonTree(
        one_adversarial_class=one_adversarial_class,
        attack_model=attack_model,
        is_numerical=is_numerical,
    )

    tree.root_ = _tree_dict_to_nodes_rec(tree_dict, round_digits)
    return tree
