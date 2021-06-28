# encoding: utf-8

"""
Created by Gabriele Tolomei on 2019-01-23.

Code adapted for comparison with GROOT from src/parallel_robust_forest.py
at https://github.com/gtolomei/treant

Also see: https://arxiv.org/abs/1907.01197
"""

import sys
import os
import logging
import dill
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin

# Added to turn TREANT trees into GROOT trees
from .model import Node as GrootNode
from .model import NumericalNode as GrootNumericalNode
from .model import _TREE_LEAF, _TREE_UNDEFINED

"""
Logging setup
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

LOGGING_FORMAT = "%(asctime)-15s *** %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s()] *** %(message)s"
formatter = logging.Formatter(LOGGING_FORMAT)

# log to stdout console
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)
# log to file
file_handler = logging.FileHandler(filename="out/treant.log", mode="w")
file_handler.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# CONSTANTS
EPSILON = 1e-10
# SEED = np.random.seed(73)


# JSON exporting for numpy values
def convert_numpy(obj):
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    raise TypeError


# <code>Attacker Rule</code>


class AttackerRule:
    """
    Class AttackerRule represents a rule of attack.
    """

    def __init__(self, pre_conditions, post_condition, cost, is_numerical=True):
        """
        Class constructor.

        Args:
            pre_conditions (dict): set of pre-conditions which must be met in order for this rule to be applied.
            post_condition (dict): post-condition indicating the outcome of this rule once applied.
            cost (float): cost of rule application.
            is_numerical (boolean): flag to indicate whether the attack specified by this rule operates on
                                    a numerical (perturbation) or a categorical (assignment) feature.
        """
        # pre_conditions = {feature_id: (value_left, value_right)}
        self.pre_conditions = pre_conditions
        # post_condition = {feature_id: new_value}
        self.post_condition = post_condition
        self.cost = cost
        self.is_numerical = is_numerical
        if not self.is_numerical:
            if type(self.pre_conditions[1]) == str:
                # fix single element
                self.pre_conditions = (
                    self.pre_conditions[0],
                    set((self.pre_conditions[1],)),
                )
            else:
                self.pre_conditions = (
                    self.pre_conditions[0],
                    set(self.pre_conditions[1]),
                )

        self.logger = logging.getLogger(__name__)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        if "logger" in d:
            d["logger"] = logging.getLogger(d["logger"])
        else:
            self.logger = logging.getLogger(__name__)
        self.__dict__.update(d)

    def get_cost(self):
        """
        Return the cost of this rule.
        """
        return self.cost

    def get_target_feature(self):
        """
        Return the feature (id) targeted by this rule.
        """
        return self.post_condition[0]

    def get_pre_interval(self):
        if self.is_numerical:
            return self.pre_conditions[1]
        else:
            return None

    def is_num(self):
        return self.is_numerical

    def is_applicable(self, x):
        """
        Returns whether the rule can be applied to the input instance x or not.

        Args:
            x (numpy.array): 1-dimensional array representing an instance.
            numerical_idx (list): binary array which indicates whether a feature is numerical or not;
                                  numerical_idx[i] = 1 iff feature id i is numerical, 0 otherwise.

        Return:
            True iff this rule is applicable to x (i.e., if x satisfies ALL the pre-conditions of this rule).
        """
        feature_id = self.pre_conditions[0]
        if self.is_numerical:  # the feature is numeric
            left, right = self.pre_conditions[1]
            return left <= x[feature_id] <= right
        else:  # the feature is categorical
            valid_set = self.pre_conditions[1]
            return x[feature_id] in valid_set

    def apply(self, x):
        """
        Application of the rule to the input instance x.

        Args:
            x (numpy.array): 1-dimensional array representing an instance.

        Return:
            x_prime (numpy.array): A (deep) copy of x yet modified according to the post-condition of this rule.
        """
        x_prime = x.copy()
        feature_id, feature_attack = self.post_condition
        if self.is_numerical:
            x_prime[feature_id] += feature_attack
        else:
            x_prime[feature_id] = feature_attack
        return x_prime


# <code>Attacker</code>
#
# This class represents an **attacker**. Informally, this is made of a **set of rules** (i.e., <code>AttackerRule</code>s) and a **budget** which it can spend on modifying instances.


def load_attack_rules(attack_rules_filename, colnames, encodings=None):

    attack_rules = []

    with open(attack_rules_filename) as json_file:
        json_data = json.load(json_file)
        json_attacks = json_data["attacks"]
        for attack in json_attacks:
            for feature in attack:
                feature_atk_list = attack[feature]
                for feature_atk in feature_atk_list:
                    pre = eval(feature_atk["pre"])
                    post = feature_atk["post"]
                    cost = feature_atk["cost"]
                    is_numerical = feature_atk["is_numerical"]
                    feature_id = colnames.index(feature)

                    if encodings:
                        encoding = encodings[feature_id]
                        if encoding:
                            pre = (encoding[item] for item in pre)
                            post = encoding[post]

                    attack_rules.append(
                        AttackerRule(
                            (feature_id, pre),
                            (feature_id, post),
                            cost=cost,
                            is_numerical=is_numerical,
                        )
                    )

    return attack_rules


class Attacker:
    """
    Class Attacker represents an attacker.
    """

    def __init__(self, rules, budget):

        """
        Class constructor.

        Args:
            rules (:obj:`AttackerRule`): set of AttackerRule objects.
            budget (float): total budget of the attacker (per instance).
        """
        self.rules = rules
        self.budget = budget
        self.attacks = {}

        self.logger = logging.getLogger(__name__)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        if "logger" in d:
            d["logger"] = logging.getLogger(d["logger"])
        else:
            self.logger = logging.getLogger(__name__)
        self.__dict__.update(d)

    ################################################### PUBLIC API ###################################################

    def is_filled(self):
        if self.attacks:
            for atk in self.attacks:
                if len(self.attacks[atk]) > 1:
                    return True
        return False

    def attack_dataset(self, X, attacks_filename=None):
        """
        This function is responsible for attacking the whole input dataset.
        It either loads all the attacks from the attack file provided as input
        or it computes all the attacks from scratch.
        """
        # infer index of numerical features
        self.numerical_idx = self.__infer_numerical_features(X)

        if attacks_filename is None:  # check if the attack filename is None
            # if that is the case, just compute all the attacks from scratch
            self.__compute_attacks(X, attacks_filename)
        else:  # otherwise, try to load the attacks from the input file
            self.logger.info(
                "Loading attacks to the dataset from file: {}".format(attacks_filename)
            )
            try:
                with open(attacks_filename, "rb") as attacks_file:
                    self.attacks = dill.load(attacks_file)
            except Exception as dill_ex:
                self.logger.error(
                    "Unable to load attacks to the dataset from file using dill: {}\nException: {}".format(
                        attacks_filename, dill_ex
                    )
                )
                self.logger.info(
                    "Eventually, recompute the attacks from scratch and store them to: {}".format(
                        attacks_filename
                    )
                )
                self.__compute_attacks(X, attacks_filename)

    def attack(self, x, feature_id, cost):
        """
        This function retrieves the list of attacks to a given instance,
        on a given feature, subject to a given cost.
        """
        fid = []
        for r in self.rules:
            ids = r.get_target_feature()
            fid.append(ids)
        if feature_id in fid:
            attack_key = (tuple(x.tolist()), feature_id)  # get the key of this attack
            # if attacks to this instance have NOT been already computed
            if attack_key not in self.attacks:
                self.logger.info(
                    "Attacks for this instance have not been computed yet! Let's compute those from scratch"
                )
                # compute the attack from scratch and store it as a new entry in the dictionary of attacks
                self.attacks[attack_key] = self.__compute_attack(x, feature_id, cost)

            attacks_xf = self.attacks[attack_key]
            attacks_xf = [x for x in attacks_xf if x[1] <= self.budget - cost]
            attacks_xf = [(x[0], x[1] + cost) for x in attacks_xf]

        else:
            attacks_xf = [(x, 0 + cost)]

        return attacks_xf

    ################################################### PRIVATE FUNCTIONS ###################################################

    def __infer_numerical_features(self, X, numerics=["integer", "floating"]):
        if X is not None:

            def infer_type(x):
                return pd.api.types.infer_dtype(x, skipna=True)

            X_types = list(np.apply_along_axis(infer_type, 0, X))
            return np.isin(X_types, numerics).tolist()

    def __is_equal_perturbation(self, a, b):
        return np.array_equal(a[0], b[0]) and a[1] <= b[1]

    def __compute_attacks(self, X, attacks_filename):
        """
        Return all the attacks of all the instances of the original dataset X, assuming starting cost is 0.
        """
        self.logger.info("Compute all the attacks to the dataset from scratch...")
        # index list for features to be perturbed
        f = []
        for r in self.rules:
            ids = r.get_target_feature()
            f.append(ids)
        # print(f)
        for i in range(X.shape[0]):
            for (
                j
            ) in (
                f
            ):  # range(X.shape[1]):# restrict here to only columns which are involved in the attack(feature index)
                key = (tuple(X[i, :].tolist()), j)
                self.attacks[key] = self.__compute_attack(X[i, :], j, 0)
        self.logger.info(
            "Finally, store all the attacks to file: {}".format(attacks_filename)
        )
        with open(attacks_filename, "wb") as attacks_file:
            dill.dump(self.attacks, attacks_file, protocol=dill.HIGHEST_PROTOCOL)

    def __compute_attack(self, x, feature_id, cost):
        """
        Return the set of attacks generated by this attacker on a particular instance.

        Args:
            x (numpy.array): 1-dimensional array representing an instance.
            cost (float): cost associated with the instance that has been spent so far.
            feature_id (int, optional): id of the feature targeted by this attack
        """

        queue = [
            (x, cost)
        ]  # enqueue the instance as it is, along with its associated cost computed so far
        attacks = []  # prepare the list of attacks to be eventually returned
        # the index of features to be attacked
        # loop until the queue is not empty

        while len(queue) != 0:
            # dequeue the first inserted element (i.e., an instance and its updated cost spent)
            x, b = queue.pop()
            # append the current attacked instance to the list of attacks
            attacks.append((x, b))
            # check the rules applicable to the current feature_id
            applicables = [
                r for r in self.rules if r.get_target_feature() == feature_id
            ]

            # extract the list of applicable rules out of the set of all rules
            applicables = [r for r in applicables if r.is_applicable(x)]

            #   print(feature_id)
            for r in applicables:  # for each applicable rule
                # check if the current budget of the attacker is large enough to apply the rule
                if self.budget >= b + r.get_cost():
                    # if so, just apply the rule and get back the attacked instance
                    x_prime = r.apply(x)
                    cost_prime = b + r.get_cost()
                    if not any(
                        atk
                        for atk in attacks
                        if self.__is_equal_perturbation(atk, (x_prime, cost_prime))
                    ):
                        # insert such a new instance in the queue with its updated cost
                        queue.insert(0, (x_prime, cost_prime))
                        attacks.append((x_prime, cost_prime))

                    # if numerical check extremes !
                    ##############
                    # WARNING!!! # This should be for every applicable rule !!!
                    ##############
                    if r.is_num():
                        # Evaluate extremes of validity interval
                        f = r.get_target_feature()
                        low, high = sorted([x[f], x_prime[f]])
                        extremes = r.get_pre_interval()
                        z = set([t for t in extremes if low < t < high])
                        # apply modifications
                        for zi in z:
                            x_prime = x.copy()
                            x_prime[f] = zi
                            if not any(
                                atk
                                for atk in attacks
                                if self.__is_equal_perturbation(
                                    atk, (x_prime, cost_prime)
                                )
                            ):
                                # insert such a new instance in the queue with its updated cost
                                queue.insert(0, (x_prime, cost_prime))
                                attacks.append((x_prime, cost_prime))
            break

        # eventually, return all the (unique) attacks generated
        return attacks


# ## <code>Constraint</code>
#
# This class represents a **constraint**, which applies to a specific labeled instance.


class Constraint(object):
    """
    Class Constraint represents a constraint.
    """

    def __init__(self, x, y, cost, ineq, bound):
        """
        Class constructor.

        Args:
            x (int): current instance.
            y (int/float): label associated with this instance.
            cost (float): cost associated with this instance (so far).
            ineq (int): flag to encode the direction of the inequality represented by this constraint;
                        0 = 'less than', 1 = 'greater than or equal to'.
            bound (float): constraint value on the loss function
        """
        self.x = x
        self.y = y
        self.cost = cost
        self.ineq = ineq  # 0 = less than or equal to; 1 = greater than or equal to
        self.bound = bound

        self.logger = logging.getLogger(__name__)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        if "x" in d:
            del d["x"]
        if "y" in d:
            del d["y"]
        return d

    def __setstate__(self, d):
        if "logger" in d:
            d["logger"] = logging.getLogger(d["logger"])
        else:
            self.logger = logging.getLogger(__name__)
        self.__dict__.update(d)

    def propagate_left(self, attacker, feature_id, feature_value, is_numerical):
        """
        Propagate the constraint to the left.
        """
        # retrieve all the attacks
        attacks = attacker.attack(self.x, feature_id, self.cost)
        if is_numerical:
            # retain only those attacks whose feature value is less than or equal to feature value
            attacks = [atk for atk in attacks if atk[0][feature_id] <= feature_value]
        else:
            # retain only those attacks whose feature value is equal to feature value
            attacks = [atk for atk in attacks if atk[0][feature_id] == feature_value]
        if not attacks:
            return None
        return Constraint(
            self.x, self.y, np.min([atk[1] for atk in attacks]), self.ineq, self.bound
        )

    def propagate_right(self, attacker, feature_id, feature_value, is_numerical):
        """
        Propagate the constraint to the right.
        """
        # generate all the attacks
        attacks = attacker.attack(self.x, feature_id, self.cost)
        if is_numerical:
            # retain only those attacks whose feature value is greater than feature value
            attacks = [atk for atk in attacks if atk[0][feature_id] > feature_value]
        else:
            # retain only those attacks whose feature value is different from feature value
            attacks = [atk for atk in attacks if atk[0][feature_id] != feature_value]
        if not attacks:
            return None
        return Constraint(
            self.x, self.y, np.min([atk[1] for atk in attacks]), self.ineq, self.bound
        )

    def encode_for_optimizer(self, direction):
        """
        Encode this constraint according to the format used by the optimizer.
        """
        encoded_constraint = {}
        encoded_constraint["type"] = "ineq"

        if direction == "L":
            if self.ineq == 0:  # encoding less than or equal to
                encoded_constraint["fun"] = (
                    lambda pred: -((pred[0] - self.y) ** 2) + (self.bound - self.y) ** 2
                )
            else:  # encoding greater than or equal to
                encoded_constraint["fun"] = (
                    lambda pred: (pred[0] - self.y) ** 2 - (self.bound - self.y) ** 2
                )

        if direction == "R":
            if self.ineq == 0:  # encoding less than or equal to
                encoded_constraint["fun"] = (
                    lambda pred: -((pred[1] - self.y) ** 2) + (self.bound - self.y) ** 2
                )
            else:  # encoding greater than or equal to
                encoded_constraint["fun"] = (
                    lambda pred: (pred[1] - self.y) ** 2 - (self.bound - self.y) ** 2
                )

        if direction == "U":
            if self.ineq == 0:  # encoding less than or equal to
                encoded_constraint["fun"] = (
                    lambda pred: -np.max((pred - self.y) ** 2)
                    + (self.bound - self.y) ** 2
                )
            else:  # encoding greater than or equal to
                encoded_constraint["fun"] = (
                    lambda pred: np.min((pred - self.y) ** 2)
                    - (self.bound - self.y) ** 2
                )

        return encoded_constraint

    def __str__(self):
        return "::".join([str(self.y), str(self.ineq), str((self.bound - self.y) ** 2)])


# # <code>RobustForest</code>

# ## <code>Node</code>
#
# This class represents an individual node of each tree eventually composing the forest.
# Intuitively, each node contains a horizontal slice of the original matrix of data; in other words, it contains a subset <code>X'</code> of the rows of <code>X</code>.
# In addition to that, each node keep track of the <code>feature_id</code> along with the corresponding <code>feature_value</code> which lead to the **best splitting** of the data "located" at that node (i.e., <code>X'</code>). The best splitting is computed on the basis of a specific function utilized to generate the tree/forest.<br />
# Furthermore, at each node we are able to tell what would be the prediction of the label/target value, on the basis of the subset of instances located at that node. More specifically, at each node we also keep track of the labels/target values associated with the instances located at that node. Prediction can be simply obtained by:
# -  _(i)_ take the most representative class label (i.e., majority voting) in case of classification;
# -  _(ii)_ take the average of the target values in case of regression;
#
# Finally, each node contains a reference to its left and right child, respectively. A **leaf node** is a special node whose left and right childs are both equal to <code>None</code>, as well as its best splitting feature id/value.


class Node(object):
    """
    Class Node represents a node of a decision tree.
    """

    def __init__(
        self,
        node_id,
        values,
        n_values,
        left=None,
        right=None,
        best_split_feature_id=None,
        best_split_feature_value=None,
    ):
        """
        Class constructor.

        Args:
            node_id (int): node identifier.
            values (int): number of instances
            n_values (int): maximum number of unique y values.
            left (:obj:`Node`, optional): left child node. Defaults to None.
            right (:obj:`Node`, optional): left child node. Defaults to None.
            best_split_feature_id (int, optional): index of the feature associated with the best split of this Node. Defaults to None.
            best_split_feature_value (float, optional): value of the feature associated with the best split of this Node. Defaults to None.

        """
        self.node_id = node_id
        self.values = values
        self.n_values = n_values
        self.left = left
        self.right = right
        self.best_split_feature_id = best_split_feature_id
        self.best_split_feature_value = best_split_feature_value
        self.prediction_score = None
        self.prediction = None
        self.loss_value = None
        self.gain_value = None
        self.constraints = None
        self.instance = None

    def set_node_prediction(self, prediction_score, threshold=0.5):
        self.prediction_score = prediction_score
        if self.prediction_score > threshold:
            self.prediction = 1
        else:
            self.prediction = 0

    def set_loss_value(self, loss_value):
        self.loss_value = loss_value

    def set_gain_value(self, gain_value):
        self.gain_value = gain_value

    def get_loss_value(self):
        return self.loss_value

    def get_gain_value(self):
        return self.gain_value

    def set_constraint(self, constraints):
        self.constraints = len(constraints)

    def get_constraint(self):
        return self.constraints

    def set_instance(self, min_instances_per_node):
        self.min_instances_per_node = min_instances_per_node

    def get_instance(self):
        return self.min_instances_per_node

    def get_node_prediction(self):
        """
        Get the prediction as being computed at this node.
        """
        return self.prediction, self.prediction_score

    def is_leaf(self):
        """
        Returns True iff the current node is a leaf (i.e., if it doesn't have neither a left nor a right child)
        """
        return self.left is None and self.right is None

    def to_json(self):
        if self.is_leaf():
            return {
                "prediction": self.get_node_prediction()[0],
                "prediction_score": self.get_node_prediction()[1],
                "values": self.values,
                "loss_value": self.loss_value,
                "gain_value": self.gain_value,
            }
        else:
            return {
                "best_split_feature_id": self.best_split_feature_id,
                "best_split_feature_value": self.best_split_feature_value,
                "values": self.values,
                "loss_value": self.loss_value,
                "gain_value": self.gain_value,
                "constraints": self.constraints,
                "left": self.left.to_json(),
                "right": self.right.to_json(),
            }

    def to_xgboost_json(self, node_id, depth):
        if self.is_leaf():
            return (
                {"nodeid": node_id, "leaf": self.get_node_prediction()[1] * 2 - 1},
                node_id,
            )
        else:
            left_id = node_id + 1
            left_dict, new_node_id = self.left.to_xgboost_json(left_id, depth + 1)

            right_id = new_node_id + 1
            right_dict, new_node_id = self.right.to_xgboost_json(right_id, depth + 1)

            return (
                {
                    "nodeid": node_id,
                    "depth": depth,
                    "split": self.best_split_feature_id,
                    "split_condition": self.best_split_feature_value,
                    "yes": left_id,
                    "no": right_id,
                    "missing": left_id,
                    "children": [left_dict, right_dict],
                },
                new_node_id,
            )

    def to_groot_node(self):
        # This method assumes the tree is numerical!!!
        if self.is_leaf():
            value = [1.0 - self.prediction_score, self.prediction_score]
            return GrootNode(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)
        else:
            return GrootNumericalNode(
                self.best_split_feature_id,
                self.best_split_feature_value,
                self.left.to_groot_node(),
                self.right.to_groot_node(),
                _TREE_UNDEFINED,
            )

    def pretty_print(self, out, tabs=""):

        leaf_txt = "{}Prediction: {}; Score: {:.5f}; N. instances:{};Loss:{:.5f}; gain:{:.5f}".format(
            tabs,
            self.get_node_prediction()[0],
            self.get_node_prediction()[1],
            self.values,
            self.loss_value,
            self.gain_value,
        )
        internal_node_txt = "{}Feature ID: {}; Threshold: {}; N. instances: {};Loss:{:.5f}; gain: {:.5f},N.constraints:{:3d}".format(
            tabs,
            self.best_split_feature_id,
            self.best_split_feature_value,
            self.values,
            self.loss_value,
            self.gain_value,
            self.constraints,
        )

        if self.is_leaf():  # base case
            out.write(leaf_txt + "\n")
        else:  # recursive case
            out.write(internal_node_txt + "\n")
            self.left.pretty_print(out, tabs + "\t")
            self.right.pretty_print(out, tabs + "\t")


# ## <code>SplitOptimizer</code>


class SplitOptimizer(object):
    """
    Class used for determining the best splitting strategy, accoriding to a specific splitting function.
    The class comes with few splitting functions already implemented. In particular, those are as follows:

    - __gini_impurity (classification);
    - __entropy (classification);
    - __logloss (classificattion);
    - __mse (regression);
    - __sse (regression);
    - __mae (regression).

    Of course this class can be instantiated with custom, user-defined splitting functions.
    """

    def __init__(self, split_function_name=None, icml2019=False):
        """
        Class constructor.

        Args:
            split_function (func): The function used as splitting criterion.
                                     Defaults to None, if so it falls back to __gini_impurity implemented internally.


        if split_function is None:
            self.split_function = SplitOptimizer._SplitOptimizer__sse
            self.split_function_name = "SSE"

        else:
            self.split_function = split_function
            if split_function_name is None:
                split_function_name = split_function.__name__
            self.split_function_name = split_function_name
        """

        self.split_function_name = split_function_name

        if split_function_name == "logloss":
            self.split_function = SplitOptimizer._SplitOptimizer__logloss
            self.split_function_name = "logloss"
            self.mysplit = SplitOptimizer._SplitOptimizer__logloss_under_max_attack

        elif split_function_name == "sse":
            self.split_function = SplitOptimizer._SplitOptimizer__sse
            self.split_function_name = "sse"
            self.mysplit = SplitOptimizer._SplitOptimizer__sse_under_max_attack

        else:
            if split_function_name is None:
                self.split_function = SplitOptimizer._SplitOptimizer__logloss
                self.mysplit = SplitOptimizer._SplitOptimizer__logloss_under_max_attack

        self.logger = logging.getLogger(__name__)

        self.icml2019 = icml2019

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        if "logger" in d:
            d["logger"] = logging.getLogger(d["logger"])
        else:
            self.logger = logging.getLogger(__name__)
        self.__dict__.update(d)

    @staticmethod
    def __gini_impurity(y_true, y_pred):
        """
        This function computes the Gini impurity.

        Given K classes, encoded with integers in [0, ..., K-1], and a vector y of M elements where y[i] in [0, ..., K-1]
        let N_k be the number of entries in y equal to k in [0, ..., K-1].
        In other words:

                    N_k = |{i in [0, ..., M-1] s.t. y[i] = k, k in [0, ..., K-1]}|

        Clearly, N_0 + N_1 + ... + N_K-1 = M.

        In addition, we let P_k = N_k/M be the probability of an item belonging to class k.

        The Gini impurity can be therefore computed as follows:

                    G(y) = 1 - [(P_0)^2 + (P_1)^2 + ... + (P_K-1)^2] = 1 - SUM_{k=0}^{K-1} (P_k)^2

        Gini impurity is used by the CART (classification and regression tree) algorithm for classification trees.

        Args:
            y_true (numpy.array): array of true label values.
            y_pred (int): unused when computing Gini impurity, left for API compatibility.

        Returns:
            The Gini impurity
        """
        if len(y_true) == 0:
            return 0

        freqs = np.bincount(y_true)
        return 1 - np.sum(np.square(freqs / len(y_true)))

    @staticmethod
    def __entropy(y_true, y_pred):
        """
        This function computes the Entropy which information gain is based on.
        Using the same notation as above, we define the Entropy H(y) as follows:

                    H(y) = - (P_0 * log_2(P_0)) + (P_1 * log_2(P_1)) + ... + (P_K-1 * log_2(P_K-1)) =

                         = - SUM_{k=0}^{K-1} (P_k * log_2(P_k))

        Entropy (in combination with information gain) is used by the ID3, C4.5 and C5.0 tree-generation algorithms

        Args:
            y_true (numpy.array): array of true label values
            y_pred (int): unused when computing Entropy, left for API compatibility.

        Returns:
            The Entropy score
        """
        if len(y_true) == 0:
            return 0

        freqs = np.bincount(y_true)
        # to avoid computing log_2(0) (i.e., P_k > 0)
        return -np.sum(
            (freqs + EPSILON) / len(y_true) * np.log2((freqs + EPSILON) / len(y_true))
        )

    @staticmethod
    def __mse(y_true, y_pred):
        """
        This function computes the Mean Squared Error (MSE), and can used as the splitting criterion for
        generating regression trees.

        Args:
            y_true (numpy.array): array of true label values
            y_pred (float): predicted value

        Returns:
            The MSE
        """
        if len(y_true) == 0:
            return 0

        return 1 / len(y_true) * np.sum(np.square(y_true - y_pred))

    @staticmethod
    def __sse(y_true, y_pred):
        """
        This function computes the Sum of Squared Error (SSE), and can used as the splitting criterion for
        generating regression trees.

        Args:
            y_true (numpy.array): array of true label values
            y_pred (float): predicted value

        Returns:
            The SSE
        """
        if len(y_true) == 0:
            return 0

        return np.sum(np.square(y_true - y_pred))

    def __sse_under_max_attack(self, L, R, U, left, right):
        """
        Compute SSE Under Max Attack.
        """
        return (
            np.sum((L - left) ** 2.0)
            + np.sum((R - right) ** 2.0)
            + np.sum(np.maximum((U - left) ** 2.0, (U - right) ** 2.0))
        )

    @staticmethod
    def __logloss(y_true, y_pred):

        """
        This function computes the logloss, and can be used as the splitting criterion for
        generating classification trees.

        Args:
            y_true (numpy.array): array of true label values
            y_pred (float): predicted value

        Returns:
            The logloss
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        if len(y_true) == 0:
            return 0

        return -np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def __logloss_under_max_attack(self, L, R, U, left, right):

        """
        Compute logloss Under Max Attack.
        """

        eps = 1e-15
        lp = np.clip(left, eps, 1 - eps)
        rp = np.clip(right, eps, 1 - eps)
        # np.sum( np.max( U*np.log(lp)+(1-U)*np.log(1-lp), U*np.log(rp) +(1-U)*np.log(1-rp) ) )
        # print(( U*np.log(lp)+(1-U)*np.log(1-lp), U*np.log(rp) +(1-U)*np.log(1-rp) ))
        # print(np.maximum( -U*np.log(lp)-(1-U)*np.log(1-lp), -U*np.log(rp) -(1-U)*np.log(1-rp) ))

        return (
            -np.sum(L * np.log(lp) + (1 - L) * np.log(1 - lp))
            + -np.sum(R * np.log(rp) + (1 - R) * np.log(1 - rp))
            + np.sum(
                np.maximum(
                    -U * np.log(lp) - (1 - U) * np.log(1 - lp),
                    -U * np.log(rp) - (1 - U) * np.log(1 - rp),
                )
            )
        )

    @staticmethod
    def __mae(y_true, y_pred):
        """
        This function computes the Mean Absolute Error (MAE), and can used as the splitting criterion for
        generating regression trees.

        Args:
            y_true (numpy.array): array of true label values
            y_pred (float): predicted value

        Returns:
            The MAE
        """
        if len(y_true) == 0:
            return 0

        return 1 / len(y_true) * np.sum(np.abs(y_true - y_pred))

    def __icml_split_loss(self, y, L, R):
        if len(L) == 0:
            icml_pred_right = np.mean(y[R])
            icml_loss = self.__sse(y[R], icml_pred_right)
            return None, icml_pred_right, icml_loss
        elif len(R) == 0:
            icml_pred_left = np.mean(y[L])
            icml_loss = self.__sse(y[L], icml_pred_left)
            return icml_pred_left, None, icml_loss
        else:
            icml_pred_left = np.mean(y[L])
            icml_pred_right = np.mean(y[R])
            icml_loss = self.__sse(y[L], icml_pred_left) + self.__sse(
                y[R], icml_pred_right
            )
            return icml_pred_left, icml_pred_right, icml_loss

    def __split_icml2019(
        self, X, y, rows, numerical_idx, attacker, costs, feature_id, feature_value
    ):
        is_numerical = numerical_idx[feature_id]
        split_left = (
            []
        )  # indices of instances which surely DO satisfy the boolean spitting predicate, disregarding the attacker
        split_right = (
            []
        )  # indices of instances which surely DO NOT satisfy the boolean spitting predicate, disregarding the attacker
        # indices of instances which may or may not satisfy the boolean splitting predicate
        split_unknown_left = []
        split_unknown_right = []

        # loop through every instance
        for row_id in rows:
            # x = X[row_id, :]  # get the i-th instance
            # get the i-th cost spent on the i-th instance so far
            cost = costs[row_id]
            # collect all the attacks the attacker can do on the i-th instance
            attacks = attacker.attack(X[row_id, :], feature_id, cost)

            # apply the splitting predicates to all the attacks of the i-th instance, limited to feature_id of interest
            # this will get a boolean mask (i.e., a binary vector containing as many elements as the number of attacks for this instance)
            all_left = True
            all_right = True

            for atk in attacks:
                if is_numerical:
                    if atk[0][feature_id] <= feature_value:
                        all_right = False
                    else:
                        all_left = False
                else:
                    if atk[0][feature_id] == feature_value:
                        all_right = False
                    else:
                        all_left = False

                if not all_left and not all_right:
                    break

            if all_left:
                # it means the splitting predicate is ALWAYS satisfied by this instance, no matter what the attacker does
                # as such, we can safely place this instance among those which surely go to the true (left) branch
                split_left.append(row_id)

            elif all_right:
                # it means the splitting predicate is NEVER satisfied by this instance, no matter what the attacker does
                # as such, we can safely place this instance among those which surely go to the false (right) branch
                split_right.append(row_id)

            else:
                # it means the splitting predicate MAY or MAY NOT be satisfied, depending on what the attacker does
                # as such, we place this instance among the unknowns
                if is_numerical:
                    if X[row_id, feature_id] <= feature_value:
                        split_unknown_left.append(row_id)
                    else:
                        split_unknown_right.append(row_id)
                else:
                    if X[row_id, feature_id] == feature_value:
                        split_unknown_left.append(row_id)
                    else:
                        split_unknown_right.append(row_id)

        icml_options = []

        # case 1: no perturbations
        icml_left = split_left + split_unknown_left
        icml_right = split_right + split_unknown_right
        # if len(icml_left)!=0 and len(icml_right)!=0:
        icml_options.append(self.__icml_split_loss(y=y, L=icml_left, R=icml_right))

        # case 2: swap
        icml_left = split_left + split_unknown_right
        icml_right = split_right + split_unknown_left
        # if len(icml_left)!=0 and len(icml_right)!=0:
        icml_options.append(self.__icml_split_loss(y=y, L=icml_left, R=icml_right))

        # case 3: all left
        icml_left = split_left + split_unknown_right + split_unknown_left
        icml_right = split_right
        # if len(icml_left)!=0 and len(icml_right)!=0:
        icml_options.append(self.__icml_split_loss(y=y, L=icml_left, R=icml_right))

        # case 4: all right
        icml_left = split_left
        icml_right = split_right + split_unknown_right + split_unknown_left
        # if len(icml_left)!=0 and len(icml_right)!=0:
        icml_options.append(self.__icml_split_loss(y=y, L=icml_left, R=icml_right))

        if len(icml_options) == 0:
            # this is not happening any more
            return (
                split_left,
                split_right,
                split_unknown_right + split_unknown_left,
                None,
            )
        elif (len(split_left) + len(split_unknown_left)) == 0 or (
            len(split_right) + len(split_unknown_right)
        ) == 0:
            return (
                split_left,
                split_right,
                split_unknown_right + split_unknown_left,
                None,
            )
        else:
            # eventually, we return the 3 list of instance indices distributed across the 3 possible branches
            y_pred_left, y_pred_right, sse = sorted(icml_options, key=lambda x: x[-1])[
                -1
            ]
            # overwrite pred_left and right
            y_pred_left = np.mean(y[split_left + split_unknown_left])
            y_pred_right = np.mean(y[split_right + split_unknown_right])
            return (
                split_left,
                split_right,
                split_unknown_right + split_unknown_left,
                (y_pred_left, y_pred_right, sse),
            )

    def __simulate_split(
        self, X, rows, numerical_idx, attacker, costs, feature_id, feature_value
    ):
        """
        This function emulates splitting data X on feature_id using feature_value.

        Args:
            X (numpy.array): 2-dimensional array containing the horizontal slice of input data matrix located at this node (i.e., the actual rows indexed by self.rows)
            rows (numpy.array): 1-dimensional array containing the indices of a subset of n_samples (i.e., a subset of the rows of X and y).
            numerical_idx (list): binary array which indicates whether a feature is numerical or not;
                                  numerical_idx[i] = 1 iff feature id i is numerical, 0 otherwise.
            attacker (:obj:`Attacker`): attacker.
            costs (dict): cost associated with each instance (indexed by rows).
            feature_id (int): index of the feature tested
            feature_value (int/float/str): feature value used to simulate the split.
                                           Numerical features contain int/float values, whilst categorical features are represented as string

        Returns:
            (numpy.array, numpy.array, numpy.array):
                                        three 1-dimensional boolean array:
                                        the first one (split_left) indicating the subset of rows whose feature value ALWAYS agrees with the splitting predicate, even upon attacks
                                        the second one (split_right) indicating the subset of rows whose feature value NEVER agrees with the splitting predicate, even upon attacks
                                        the third one (split_unknown) indicating the subset of rows whose feature value MAY or MAY NOT agree with the splitting predicate, depending on the attack
        """

        is_numerical = numerical_idx[feature_id]

        split_left = (
            []
        )  # indices of instances which surely DO satisfy the boolean spitting predicate, disregarding the attacker
        split_right = (
            []
        )  # indices of instances which surely DO NOT satisfy the boolean spitting predicate, disregarding the attacker
        # indices of instances which may or may not satisfy the boolean splitting predicate
        split_unknown = []

        # loop through every instance
        for row_id in rows:
            # x = X[row_id, :]  # get the i-th instance
            # get the i-th cost spent on the i-th instance so far
            cost = costs[row_id]
            # collect all the attacks the attacker can do on the i-th instance
            attacks = attacker.attack(X[row_id, :], feature_id, cost)

            # apply the splitting predicates to all the attacks of the i-th instance, limited to feature_id of interest
            # this will get a boolean mask (i.e., a binary vector containing as many elements as the number of attacks for this instance)
            all_left = True
            all_right = True

            for atk in attacks:
                if is_numerical:
                    if atk[0][feature_id] <= feature_value:
                        all_right = False
                    else:
                        all_left = False
                else:
                    if atk[0][feature_id] == feature_value:
                        all_right = False
                    else:
                        all_left = False

                if not all_left and not all_right:
                    break

            if all_left:
                # it means the splitting predicate is ALWAYS satisfied by this instance, no matter what the attacker does
                # as such, we can safely place this instance among those which surely go to the true (left) branch
                split_left.append(row_id)

            elif all_right:
                # it means the splitting predicate is NEVER satisfied by this instance, no matter what the attacker does
                # as such, we can safely place this instance among those which surely go to the false (right) branch
                split_right.append(row_id)

            else:
                # it means the splitting predicate MAY or MAY NOT be satisfied, depending on what the attacker does
                # as such, we place this instance among the unknowns
                split_unknown.append(row_id)

        # eventually, we return the 3 list of instance indices distributed across the 3 possible branches
        # self.logger.info("number of unknown instances:{}".format(len(split_unknown)))
        return split_left, split_right, split_unknown

    def __optimize_sse_under_max_attack(
        self,
        y,
        current_prediction_score,
        split_left,
        split_right,
        split_unknown,
        loss_function,
        C=[],
    ):
        """
        Solver which optimizes the loss function specified as input and constrained with the list of constraints.
        """

        L = y[split_left]
        R = y[split_right]
        U = y[split_unknown]

        # seed
        # x_0 = np.array([np.mean(L), np.mean(R)])
        x_0 = np.array([current_prediction_score, current_prediction_score])

        # loss function to be minimized

        def fun(x):
            return loss_function(self, L, R, U, x[0], x[1])  # self

        # constrained optimization
        res = minimize(fun, x_0, method="SLSQP", constraints=C)

        # check result
        if not res.success:
            self.logger.error(
                "!!!!!!!!!!!!!!!!!! Solver Error: {} !!!!!!!!!!!!!!!!!!".format(
                    res.message
                )
            )
            return None

        return res.x[0], res.x[1], res.fun

    def evaluate_split(self, y_true, y_pred):
        """
        This function is a meta-function which calls off to the actual splitting function along with input arguments.
        """
        return self.split_function(y_true, y_pred)

    def optimize_gain(
        self,
        X,
        y,
        rows,
        numerical_idx,
        feature_blacklist,
        n_sample_features,
        replace_features,
        attacker,
        costs,
        constraints,
        current_score,
        current_prediction_score,
    ):
        """
        This function is responsible for finding the splitting which optimizes the gain (according to the splitting function)
        among all the possibile splittings.

        Args:
            X (numpy.array): 2-dimensional array of shape (n_samples, n_features) representing the feature matrix;
            y (numpy.array): 1-dimensional array of shape (n_samples, ) representing class labels (classification) or target values (regression).
            rows (numpy.array): 1-dimensional array containing the indices of a subset of n_samples (i.e., a subset of the rows of X and y).
            numerical_idx (list): binary array which indicates whether a feature is numerical or not;
                                  numerical_idx[i] = 1 iff feature id i is numerical, 0 otherwise.
            feature_blacklist (set): set of (integer) indices corresponding to blacklisted features.
            n_sample_features (int): number of features to be randomly sampled at each node split.
            attacker (:obj:`Attacker`): attacker.
            costs (dict): cost associated with each instance (indexed by rows).
            constraints (list): list of `Constraint` objects.
            current_score (float): is the score before any splitting is done; this must be compared with the best splitting found.
                                     Whenever the current_score is greater than the one computed after splitting there will be a gain.
        Returns:
            best_gain (float): The highest gain obtained after all the possible splittings have been tested
                                 (may be 0, in which case the splitting will be not worth it)
            best_split_left_id (numpy.array): 1-dimensional array containing all the indices of rows going on the left branch.
            best_split_right_id (numpy.array): 1-dimensional array containing all the indices of rows going on the right branch.
            best_split_feature_id (int): index of the feature which led to the best splitting.
            best_split_feature_value (int/float): value of the feature which led to the best splitting.
            next_best_split_feature_value (int/float): next-observed value of the feature which led to the best splitting.
            constraints_left (numpy.array): array of constraints if propagated to left.
            constraints_right (numpy.array): array of constraints if propagated to right.
            costs_left (numpy.array): array of costs if propagated left.
            costs_right (numpy.array): array of cost if propagated right.
        """

        # prepare locations where best splitting information will be eventually stored
        best_gain = 0.0
        best_split_feature_id = None
        best_split_feature_value = None
        next_best_split_feature_value = None
        best_split_left_id = None
        best_split_right_id = None
        best_split_unknown_id = None
        best_pred_left = None
        best_pred_right = None
        best_sse_uma = None
        constraints_left = None
        constraints_right = None
        costs_left = None
        costs_right = None

        assert (not self.icml2019) or len(
            constraints
        ) == 0, "!!! ICML ERROR: Non empty constraints !!!"

        # create a dictionary containing individual values for each feature_id (limited to the slice of data located at this node)
        # {'feature_1': [val_1,1, ..., val_1,k1], ..., 'feature_n': [val_1,n, ..., val_1,kn]}
        # 1. filter out any blacklisted features from the list of features actually considered
        actual_features = [
            f_id for f_id in range(np.size(X, 1)) if f_id not in feature_blacklist
        ]
        # 2. randomly sample a subset of n features out of the actual features
        self.logger.info(
            "Randomly sample (without replacement) {} out of the total number of features ({})".format(
                n_sample_features, len(actual_features)
            )
        )
        actual_features = sorted(
            np.random.choice(
                actual_features,
                size=min(n_sample_features, len(actual_features)),
                replace=replace_features,
            )
        )
        feature_map = dict(
            (f_id, sorted(list(set(X[:, f_id])))) for f_id in actual_features
        )  # range(np.size(X, 1)) if f_id not in feature_blacklist)

        # MAIN OUTER LOOP
        # for each feature_id in the dictionary of features
        for feature_id, feature_values in feature_map.items():
            # INNER LOOP
            # for each feature value observed for a given feature_id
            for feature_value_idx, feature_value in enumerate(feature_values):
                self.logger.debug(
                    "Simulate splitting on feature id {} using value = {}".format(
                        feature_id, feature_value
                    )
                )

                if self.icml2019:
                    (
                        split_left,
                        split_right,
                        split_unknown,
                        optimizer_res,
                    ) = self.__split_icml2019(
                        X,
                        y,
                        rows,
                        numerical_idx,
                        attacker,
                        costs,
                        feature_id,
                        feature_value,
                    )

                else:
                    split_left, split_right, split_unknown = self.__simulate_split(
                        X,
                        rows,
                        numerical_idx,
                        attacker,
                        costs,
                        feature_id,
                        feature_value,
                    )

                    self.logger.debug(
                        "Solve constrained optimization problem to deal with unknown instances..."
                    )

                    updated_constraints = []

                    for c in constraints:
                        c_left = c.propagate_left(
                            attacker,
                            feature_id,
                            feature_value,
                            numerical_idx[feature_id],
                        )
                        c_right = c.propagate_right(
                            attacker,
                            feature_id,
                            feature_value,
                            numerical_idx[feature_id],
                        )
                        if c_left and c_right:
                            updated_constraints.append(c.encode_for_optimizer("U"))
                        else:
                            if c_left:
                                updated_constraints.append(c.encode_for_optimizer("L"))
                            if c_right:
                                updated_constraints.append(c.encode_for_optimizer("R"))

                    optimizer_res = self.__optimize_sse_under_max_attack(
                        y,
                        current_prediction_score,
                        split_left,
                        split_right,
                        split_unknown,
                        self.mysplit,  # __sse_under_max_attack,
                        # [c.encode_for_optimizer() for c in constraints]
                        C=updated_constraints,
                    )

                # ONLY IF THE OPTIMIZER RETURNS SOMETHING...
                if optimizer_res:

                    y_pred_left, y_pred_right, sse_uma = optimizer_res

                    self.logger.debug(
                        "Optimization result\nFeature ID: {}; Threshold:{}\n[L={:.3f}; R={:.3f}; SSE_UMA={:.3f}]".format(
                            feature_id,
                            feature_value,
                            y_pred_left,
                            y_pred_right,
                            sse_uma,
                        )
                    )

                    n_left = len(split_left)
                    n_right = len(split_right)

                    self.logger.debug(
                        "Number of instances ending up in the left branch: {}".format(
                            n_left
                        )
                    )
                    self.logger.debug(
                        "Number of instances ending up in the right branch: {}".format(
                            n_right
                        )
                    )

                    # compute gain
                    gain = current_score - sse_uma
                    self.logger.debug(
                        "Gain (Current score - Optimizer's score) = {:.5f}".format(gain)
                    )

                    # if gain obtained with this split simulation is greater than the best gain so far
                    if gain > best_gain:

                        best_gain = gain  # update best gain with the one just computed
                        # save the feature id which leads to such a best gain
                        best_split_feature_id = feature_id
                        best_split_feature_value = (
                            feature_value  # save the best splitting value
                        )
                        # if we are considering all but the largest feature value (for that specific feature id)
                        if feature_value_idx < len(feature_values) - 1:
                            # get also the next best value
                            next_best_split_feature_value = feature_values[
                                feature_value_idx + 1
                            ]
                        else:
                            # otherwise, just stick to the single best splitting point
                            next_best_split_feature_value = best_split_feature_value

                        # save the indices of instances that will go on the left branch
                        best_split_left_id = split_left
                        # save the indices of instances that will go on the right branch
                        best_split_right_id = split_right
                        # save the indices of instances that will go on the unknown branch
                        best_split_unknown_id = split_unknown
                        # save the left prediction as obtained by the optimizer
                        best_pred_left = y_pred_left
                        # save the right prediction as obtained by the optimizer
                        best_pred_right = y_pred_right
                        # save best SSE under max attack
                        best_sse_uma = sse_uma

                        if numerical_idx[best_split_feature_id]:
                            self.logger.debug(
                                "Best splitting strategy so far: [feature id {}; feature value = {:.5f}; next feature value = {:.5f}; actual split value = {:.5f}]".format(
                                    best_split_feature_id,
                                    best_split_feature_value,
                                    next_best_split_feature_value,
                                    best_split_feature_value,
                                )
                            )
                        else:
                            self.logger.debug(
                                "Best splitting strategy so far: [feature id {}; feature value = {}; next feature value = {}; actual split value = {}]".format(
                                    best_split_feature_id,
                                    best_split_feature_value,
                                    next_best_split_feature_value,
                                    best_split_feature_value,
                                )
                            )

        # Continue iff there's an actual gain
        if best_gain > 0:

            if self.icml2019:
                self.logger.debug(
                    "Assign unknown instance either to left or right split, according to ICML2019 strategy"
                )

                constraints_left = []
                constraints_right = []

                for u in best_split_unknown_id:
                    if X[u, best_split_feature_id] <= best_split_feature_value:
                        best_split_left_id.append(u)
                    else:
                        best_split_right_id.append(u)

            else:
                self.logger.debug(
                    "Assign unknown instance either to left or right split, according to the worst-case scenario..."
                )
                # get the unknown-y values
                y_true_unknown = y[best_split_unknown_id]
                unknown_to_left = np.abs(y_true_unknown - best_pred_left)
                unknown_to_right = np.abs(y_true_unknown - best_pred_right)
                constraints_left = np.array(
                    [
                        c.propagate_left(
                            attacker,
                            best_split_feature_id,
                            best_split_feature_value,
                            numerical_idx[best_split_feature_id],
                        )
                        for c in constraints
                    ]
                )
                constraints_left = constraints_left[
                    constraints_left != np.array(None)
                ].tolist()

                constraints_right = np.array(
                    [
                        c.propagate_right(
                            attacker,
                            best_split_feature_id,
                            best_split_feature_value,
                            numerical_idx[best_split_feature_id],
                        )
                        for c in constraints
                    ]
                )
                constraints_right = constraints_right[
                    constraints_right != np.array(None)
                ].tolist()

                for i, u in enumerate(best_split_unknown_id):
                    self.logger.debug(
                        "Label of unknown instance ID {}: {}".format(
                            i, y_true_unknown[i]
                        )
                    )

                    self.logger.debug(
                        "Distance to left prediction: {:.3f}".format(unknown_to_left[i])
                    )
                    self.logger.debug(
                        "Distance to right prediction: {:.3f}".format(
                            unknown_to_right[i]
                        )
                    )

                    attacks = attacker.attack(X[u, :], best_split_feature_id, costs[u])
                    min_left = None
                    min_right = None

                    if numerical_idx[best_split_feature_id]:
                        min_left = np.min(
                            [
                                atk[1]
                                for atk in attacks
                                if atk[0][best_split_feature_id]
                                <= best_split_feature_value
                            ]
                        )
                        min_right = np.min(
                            [
                                atk[1]
                                for atk in attacks
                                if atk[0][best_split_feature_id]
                                > best_split_feature_value
                            ]
                        )
                    else:
                        min_left = np.min(
                            [
                                atk[1]
                                for atk in attacks
                                if atk[0][best_split_feature_id]
                                == best_split_feature_value
                            ]
                        )
                        min_right = np.min(
                            [
                                atk[1]
                                for atk in attacks
                                if atk[0][best_split_feature_id]
                                != best_split_feature_value
                            ]
                        )

                    if unknown_to_left[i] > unknown_to_right[i]:
                        self.logger.debug(
                            "Assign unknown instance ID {} to left split as the distance is larger".format(
                                i
                            )
                        )

                        best_split_left_id.append(u)
                        costs[u] = min_left
                        constraints_left.append(
                            Constraint(X[u, :], y[u], costs[u], 1, best_pred_right)
                        )
                        constraints_right.append(
                            Constraint(X[u, :], y[u], costs[u], 0, best_pred_right)
                        )
                    else:
                        self.logger.debug(
                            "Assign unknown instance ID {} to right split as the distance is larger".format(
                                i
                            )
                        )
                        best_split_right_id.append(u)
                        costs[u] = min_right
                        constraints_left.append(
                            Constraint(X[u, :], y[u], costs[u], 0, best_pred_left)
                        )
                        constraints_right.append(
                            Constraint(X[u, :], y[u], costs[u], 1, best_pred_left)
                        )

            costs_left = {key: costs[key] for key in best_split_left_id}
            costs_right = {key: costs[key] for key in best_split_right_id}

        return (
            best_gain,
            best_split_left_id,
            best_split_right_id,
            best_split_feature_id,
            best_split_feature_value,
            next_best_split_feature_value,
            best_pred_left,
            best_pred_right,
            best_sse_uma,
            constraints_left,
            constraints_right,
            costs_left,
            costs_right,
        )


# ## <code>RobustDecisionTree</code>
#
# This class represents a single decision tree, which can be used either for classification or regression.
# It exposes a quite familiar API which basically consists of two main methods:
#
# -  <code>**fit(X, y)**</code>
# -  <code>**predict(X)**</code>
#
# The former is invoked to train (i.e., learn) a decision tree by fitting it on a dataset of input observations (<code>X</code>) and associated labels/target values (<code>y</code>). To do so, this class makes use of a <code>SplitOptimizer</code>.<br />
# When creating a <code>RobustDecisionTree</code> reference, few **hyperparameters** can be also specified. For example, one can choose the maximum depth of the tree.


class RobustDecisionTree(BaseEstimator, ClassifierMixin):
    """
    This class implements a single Robust Decision Tree.
    Inspired by sklearn API, it is a sublcass of the sklearn.base.BaseEstimator class and exposes two main methods:
    - fit(X, y)
    - predict(X)
    The former is used at training time for learning a single decision tree;
    the latter is used at inference (testing) time for computing predictions using the learned tree.

    """

    def __init__(
        self,
        tree_id=0,
        attacker=Attacker([], 0),
        split_optimizer=SplitOptimizer(split_function_name="sse"),
        max_depth=8,
        min_instances_per_node=20,
        max_samples=1.0,
        max_features=1.0,
        replace_samples=False,
        replace_features=False,
        feature_blacklist={},
        affine=True,
        seed=0,
    ):
        """
        Class constructor.

        Args:
            tree_id (int): tree identifier.
            attacker (:obj:`Attacker`): the attacker under which this tree must grow (default = empty attacker).
            split_optimizer (:obj:`SplitOptimizer`): the optimizer used by this tree (default = SSE).
            max_depth (int, optional): maximum depth of the tree to be generated (default = 10).
            min_instances_per_node (int, optional): minimum number of instances per node (default = 20).
            max_samples (float): proportion of instances sampled without replacement (default = 1.0, i.e., 100%)
            max_features (float): proportion of features sampled without replacement (default = 1.0, i.e., 100%)
            feature_blacklist (dict): dictionary of features excluded during tree growth (default = {}), i.e., empty).
            replace_samples (bool): whether the random sampling of instances should be with replacement or not (default = False).
            replace_features (bool): whether the random sampling of features should be with replacement or not (default = False).
            seed (int): integer used by randomized processes.
        """

        self.tree_id = tree_id
        self.attacker = attacker
        self.split_optimizer = split_optimizer
        self.max_depth = max_depth
        self.min_instances_per_node = min_instances_per_node
        self.max_samples = max_samples
        self.max_features = max_features
        self.replace_samples = replace_samples
        self.replace_features = replace_features
        self.feature_blacklist = feature_blacklist
        self.feature_blacklist_ids = set(list(feature_blacklist.keys()))
        self.feature_blacklist_names = set(list(feature_blacklist.values()))
        self.is_affine = affine  # self.attacker.is_filled()
        self.seed = seed

        np.random.seed(self.seed)

        self.root = None
        # flag indicating whether this tree should bootstrap samples using max_sample * 100 proportions of the original dataset
        self.bootstrap_samples = True
        self.is_trained = False

        self.logger = logging.getLogger(__name__)

        self.logger.info("***** Robust Decision Tree successfully created *****")
        self.logger.info("*\tTree ID: {}".format(self.tree_id))
        self.logger.info("*\tAttacker: {}".format(self.attacker))
        self.logger.info(
            "*\tSplitting criterion: {}".format(
                self.split_optimizer.split_function_name
            )
        )
        self.logger.info("*\tMax depth: {}".format(self.max_depth))
        self.logger.info(
            "*\tMin instances per tree node: {}".format(self.min_instances_per_node)
        )
        self.logger.info("*\tMax samples: {:.1f}%".format(self.max_samples * 100))
        self.logger.info("*\tMax features: {:.1f}%".format(self.max_features * 100))
        self.logger.info(
            "*\tFeature blacklist: {}".format(self.feature_blacklist_names)
        )
        self.logger.info("*\tAffinity: {}".format(self.is_affine))
        self.logger.info("*****************************************************")

    def __getstate__(self):
        return dict((k, v) for (k, v) in self.__dict__.items() if k not in ["logger"])

    def __setstate__(self, d):
        if "logger" in d:
            d["logger"] = logging.getLogger(d["logger"])
        else:
            self.logger = logging.getLogger(__name__)
        self.__dict__.update(d)

    def __infer_numerical_features(self, X, numerics=["integer", "floating"]):
        if X is not None:

            def infer_type(x):
                return pd.api.types.infer_dtype(x, skipna=True)

            X_types = np.apply_along_axis(infer_type, 0, X)
            return np.isin(X_types, numerics).tolist()

    def __fit(
        self,
        X_train,
        y_train,
        rows,
        attacker,
        costs,
        node_prediction,
        feature_blacklist,
        n_sample_features,
        replace_features,
        constraints=[],
        node_id=[-1],
        depth=0,
    ):
        """
        This function is a private method used to actually train a single Robust Decision Tree on (a slice of) the input data matrix X indexed by rows.

        Args:
            X_train
            y_train
            rows (numpy.array): boolean mask used for indexing in the subset of the input data matrix.
            attacker (:obj:`Attacker`): an attacker object.
            costs (dict): cost associated with each instance (indexed by rows).
            constraints (list, optional): list of `Constraint` objects (default=[])
            node_id (list, optional): list of node_ids for encoding a path.
            depth (int, optional): current depth of the tree.

        Returns:
            node (Node): a reference to the root of the trained tree.
        """

        # base case 1: if X doesn't contain any record at all, just return None
        if np.size(X_train, 0) == 0:
            self.logger.info("No more data available")
            return None

        # get the current subset of rows indexed
        X = X_train[rows, :]
        # get the corresponding subset of labels/targets indexed
        y = y_train[rows]
        self.logger.debug(
            "Input data shape X = ({} x {})".format(X.shape[0], X.shape[1])
        )
        self.logger.debug("Input target shape y = ({}, )".format(y.shape[0]))
        # create a new Node using the subset of rows indexed along with the corresponding values
        self.logger.debug(
            "Create node ID: [{}]".format("->".join([str(n_id) for n_id in node_id]))
        )

        node = Node(node_id, len(y), self.y_n_uniques)
        # set current node prediction
        node.set_node_prediction(node_prediction)

        self.logger.info("Current tree depth: {}".format(depth))
        # compute the current prediction considering the current node
        current_prediction = node.get_node_prediction()[0]
        self.logger.info("Current node's prediction: {}".format(current_prediction))
        # compute the current prediction score considering the current node
        current_prediction_score = node.get_node_prediction()[1]
        self.logger.info(
            "Current node's prediction score: {}".format(current_prediction_score)
        )

        # call off to the optimizer to compute the current score considering the current node
        current_score = self.split_optimizer.evaluate_split(y, current_prediction_score)
        node.set_loss_value(current_score)
        self.logger.info("Current node's loss: {:.5f}".format(current_score))
        # number of instances per node
        node.set_instance(X.shape[0])
        self.logger.info(
            "number of instances on current node:{:.5f}".format(node.get_instance())
        )
        # base case 2: if we have already reached the maximum depth of the tree,
        # return the node just created without trying to further split it
        if depth == self.max_depth:
            self.logger.info(
                "Current depth {} is equal to maximum depth of this tree {}".format(
                    depth, self.max_depth
                )
            )
            return node

        # base case 3: if the number of instances in the current node is less than the minimum number of instances allowed by this tree,
        # return the node just created without trying to further split it
        if X.shape[0] < self.min_instances_per_node:
            self.logger.info(
                "Number of instances ended up in the current node {} are less than the minimum number of instances at each node of this tree {}".format(
                    X.shape[0], self.min_instances_per_node
                )
            )
            return node

        # ASK THE OPTIMIZER TO FIND THE BEST SPLIT
        (
            best_gain,
            best_split_left_id,
            best_split_right_id,
            best_split_feature_id,
            best_split_feature_value,
            next_best_split_feature_value,
            best_pred_left,
            best_pred_right,
            best_sse_uma,
            constraints_left,
            constraints_right,
            costs_left,
            costs_right,
        ) = self.split_optimizer.optimize_gain(
            X_train,
            y_train,
            rows,
            self.numerical_idx,
            feature_blacklist,
            n_sample_features,
            replace_features,
            attacker,
            costs,
            constraints,
            current_score,
            current_prediction_score,
        )

        # Check if there has been an actual best gain
        # (NOTE: if the best gain returned by the optimizer is 0 it means no further split is actually worth it
        # and therefore the current node will become a leaf)
        if best_gain > EPSILON:
            # assign current node SSE under max attack
            node.set_loss_value(best_sse_uma)
            node.set_gain_value(best_gain)
            self.logger.info(
                "Current node's loss (after best splitting): {:.5f}".format(
                    node.get_loss_value()
                )
            )

            self.logger.info(
                "current node's gain :{:.5f}".format(node.get_gain_value())
            )

            # node.set_constraint(constraints)
            # self.logger.info("current node's constraints  :{:3d}".format(node.get_constraint()))

            # assign to the current node the best feature id
            node.best_split_feature_id = best_split_feature_id

            # if the feature is numerical
            if self.numerical_idx[best_split_feature_id]:
                # assign to the current node the best splitting feature value as the average of the two values
                # (best_split_feature_value  + next_best_split_feature_value) / 2
                node.best_split_feature_value = best_split_feature_value
                self.logger.info(
                    "The best gain obtained is: {:.5f} using [feature id {}; feature value = {:.5f}; next feature value = {:.5f}; actual split value = {:.5f}]".format(
                        best_gain,
                        best_split_feature_id,
                        best_split_feature_value,
                        next_best_split_feature_value,
                        best_split_feature_value,
                    )
                )
            # the feature is categorical
            else:
                # assign to the current node the best splitting feature value as the actual categorical value
                node.best_split_feature_value = best_split_feature_value
                self.logger.info(
                    "The best gain obtained is: {:.5f} using [feature id {}; feature value = {}; next feature value = {}; actual split value = {}]".format(
                        best_gain,
                        best_split_feature_id,
                        best_split_feature_value,
                        next_best_split_feature_value,
                        best_split_feature_value,
                    )
                )

            # Left recursive call
            self.logger.debug(
                'Recursively call the "fit" method on the left child node'
            )
            # deep copy of the list used to reference nodes
            left_node_id = deepcopy(node_id)
            # go left (encoded by appending 1 which reflects the fact that left branch is the True branch)
            left_node_id.append(1)
            self.logger.info("Number of left rows: {}".format(len(best_split_left_id)))

            # save the current feature blacklist
            updated_feature_blacklist = feature_blacklist

            if self.is_affine:
                self.logger.debug(
                    "Update feature blacklist removing the best split feature id found [{}]".format(
                        best_split_feature_id
                    )
                )
                # add the best feature id just found to the feature blacklist,
                # so that at the next level of the tree this won't be considered
                updated_feature_blacklist = updated_feature_blacklist | set(
                    [best_split_feature_id]
                )

            # assign to the left node of the current node the result of the recursive call on the left branch
            node.left = self.__fit(
                X_train,
                y_train,
                best_split_left_id,
                attacker,
                costs_left,
                best_pred_left,  # assign left prediction to node's left child
                updated_feature_blacklist,
                n_sample_features,
                replace_features,
                constraints=constraints_left,
                node_id=left_node_id,
                depth=depth + 1,
            )

            self.logger.debug(
                'Recursively call the "fit" method on the right child node'
            )
            # deep copy of the list used to reference nodes
            right_node_id = deepcopy(node_id)
            # go right (encoded by appending 0 which reflects the fact that right branch is the False branch)
            right_node_id.append(0)
            self.logger.info(
                "Number of right rows: {}".format(len(best_split_right_id))
            )
            # assign to the right node of the current node the result of the recursive call on the right branch
            node.right = self.__fit(
                X_train,
                y_train,
                best_split_right_id,
                attacker,
                costs_right,
                best_pred_right,  # assign right prediction to node's right child
                updated_feature_blacklist,
                n_sample_features,
                replace_features,
                constraints=constraints_right,
                node_id=right_node_id,
                depth=depth + 1,
            )

        # compute current node constraint

        node.set_constraint(constraints)
        self.logger.info(
            "current node's constraints  :{:3d}".format(node.get_constraint())
        )

        # In the end, return the node
        self.logger.info("Eventually, return the node...")

        return node

    def zip_fit(self, args):
        return self.fit(*args)

    def fit(self, X, y=None, numerical_idx=None):
        """
        This function is the public API's entry point for client code to start training a single Robust Decision Tree.
        It saves both the input data (X) and labels/targets (y) in the internals of the tree and delegates off to
        the private self.__fit method. The result being a reference to the root node of the trained tree.

        Args:
            X (numpy.array): 2-dimensional array of shape (n_samples, n_features)
            y (numpy.array): 1-dimensional array of values of shape (n_samples, )
        """

        np.random.seed(self.seed)

        self.logger.info(
            "Fitting Tree ID {}...  [process ID: {}]".format(self.tree_id, os.getpid())
        )

        if numerical_idx is None:
            # infer the index of numerical features
            self.numerical_idx = self.__infer_numerical_features(X)
        else:
            self.numerical_idx = numerical_idx

        if self.max_samples <= 0:
            self.max_samples = 1.0

        # otherwise, take the minimum between the two numbers
        self.n_sample_instances = min(
            np.size(X, 0), int(self.max_samples * np.size(X, 0))
        )

        # set the number of classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if (
            self.max_features <= 0
        ):  # proportion of features to be randomly sampled at each split
            self.max_features = 1.0

        # otherwise, take the minimum between the two numbers
        self.n_sample_features = min(
            np.size(X, 1), int(self.max_features * np.size(X, 1))
        )

        # get the number of unique y values
        self.y_n_uniques = np.unique(y).size
        self.logger.info(
            "Successfully loaded input data X = ({} x {}) and y values = ({}, )".format(
                X.shape[0], X.shape[1], y.shape[0]
            )
        )

        # if this tree has to randomly select random samples from the original dataset
        if self.bootstrap_samples:
            # randomly select rows (i.e., instances)
            self.logger.info(
                "Randomly sample (with replacement) {:.1f}% of the total number of instances ({}) = {}".format(
                    self.max_samples * 100, np.size(X, 0), self.n_sample_instances
                )
            )
            rows = sorted(
                np.random.choice(
                    range(np.size(X, 0)),
                    size=self.n_sample_instances,
                    replace=self.replace_samples,
                )
            )
        else:  # otherwise (i.e., this will be a single tree of the ensemble)
            rows = [x for x in range(np.size(X, 0))]

        # assign to the internal root reference the result of self.__fit on the whole input data matrix X
        self.root = self.__fit(
            X,
            y,
            rows,
            self.attacker,
            dict(zip([x for x in range(np.size(X, 0))], np.zeros(X.shape[0]))),
            np.mean(y),  # default node prediction
            self.feature_blacklist_ids,
            self.n_sample_features,
            self.replace_features,
        )

        if self.root is not None:
            self.is_trained = True
            self.logger.info(
                "Fitting Tree ID {} completed (is_trained = {})! [process ID: {}]".format(
                    self.tree_id, self.is_trained, os.getpid()
                )
            )

        # Clean
        self.clean_after_training()

        return self

    def clean_after_training(self):
        self.attacker = None
        self.split_optimizer = None

    def __predict(self, x, node):
        """
        This function provides the prediction for a single instance x.

        Args:
            x (numpy.array): 1-dimensional array containing a single instance.
            node (:obj:`Node`): the current node.
        """

        # self.logger.debug("Node ID: [{}]".format(
        #     "->".join([str(n_id) for n_id in node.node_id])))

        # base case: the current node has no left nor right child (i.e., it is a leaf)
        if node.is_leaf():
            # just return the prediction stored at the current node
            return node.get_node_prediction()

        # otherwise, get the best splitting feature id and value stored at this node
        best_feature_id = node.best_split_feature_id
        self.logger.debug("Best feature id = {}".format(best_feature_id))
        best_feature_value = node.best_split_feature_value
        self.logger.debug("Best feature value = {}".format(best_feature_value))
        # get the feature value indexed by the best feature id of this node
        x_feature_value = x[best_feature_id]
        self.logger.debug("X feature value = {}".format(x_feature_value))

        if self.numerical_idx[best_feature_id]:  # the feature is numeric
            # compare this value with the best splitting value to decide which branch to take (either the left or the right one)
            if x_feature_value <= best_feature_value:
                # go left as the predicate is True
                return self.__predict(x, node.left)
            else:
                # go right as the predicate is False
                return self.__predict(x, node.right)
        else:  # the feature is categorical
            # compare this value with the best splitting value to decide which branch to take (either the left or the right one)
            if x_feature_value == best_feature_value:
                # go left as the predicate is True
                return self.__predict(x, node.left)
            else:
                # go right as the predicate is False
                return self.__predict(x, node.right)

    def predict(self, X, y=None):
        """
        This function is the public API's entry point for client code to obtain predictions from an already trained tree.
        If this tree hasn't been trained yet, predictions cannot be made; otherwise, for each instance in X, the tree is traversed
        until a leaf node is met: the prediction stored at that leaf node is the one returned to the caller.

        Args:
            X (numpy.array): 2-dimensional array of shape (n_test_samples, n_features) containing
                             samples which we want to know the predictions of.

        Returns:
            predictions (numpy.array): 1-dimensional array of shape (n_test_samples, ).
        """

        # prepare the array of predictions
        predictions = np.empty(X.shape[0])

        # Check if the current tree is trained
        if self.is_trained:
            # Get the prediction for all the instances
            predictions = np.asarray(
                [
                    self.__predict(x=X[i, :], node=self.root)[0]
                    for i in range(X.shape[0])
                ]
            )

        return predictions

    def predict_proba(self, X, y=None):
        """
        This function is the public API's entry point for client code to obtain predictions from an already trained tree.
        If this tree hasn't been trained yet, predictions cannot be made; otherwise, for each instance in X, the tree is traversed
        until a leaf node is met: the prediction stored at that leaf node is the one returned to the caller.

        Args:
            X (numpy.array): 2-dimensional array of shape (n_test_samples, n_features) containing
                             samples which we want to know the predictions of.

        Returns:
            probs (numpy.array): 2-dimensional array of shape (n_test_samples, 2) containing probability scores both for class 0 (1st column) and class 1 (2nd column).
        """
        probs_0 = np.empty(X.shape[0])
        probs_1 = np.empty(X.shape[0])
        # bbc = []

        # Check if the current tree is trained
        if self.is_trained:
            # Get the prediction scores for class 1
            probs_1 = np.asarray(
                [
                    self.__predict(x=X[i, :], node=self.root)[1]
                    for i in range(X.shape[0])
                ]
            )
            # Get the prediction scores for class 0
            probs_0 = 1 - probs_1

        return np.column_stack((probs_0, probs_1))

    def save(self, filename):
        """
        This function is used to persist this RobustDecisionTree object to file on disk using dill.
        """
        # this is never used by the scikilearn bagging forest
        with open(filename, "wb") as model_file:
            dill.dump(self, model_file)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as model_file:
            return dill.load(model_file)

    def to_json(self, output_file="treant_tree.json"):
        with open(output_file, "w") as fp:
            if self.is_trained:
                json.dump(self.root.to_json(), fp, indent=2, default=convert_numpy)
            else:
                json.dump({}, fp)

    def to_xgboost_json(self, output_file="treant_tree_xgboost.json"):
        if hasattr(self, "root"):
            dictionary, _ = self.root.to_xgboost_json(0, 0)
        else:
            raise Exception("Tree is not yet fitted")

        if output_file is None:
            return dictionary
        else:
            with open(output_file, "w") as fp:
                # If saving to file then surround dict in list brackets
                json.dump([dictionary], fp, indent=2, default=convert_numpy)

    def pretty_print(self, output_file=None):
        if self.is_trained:
            out = None
            if output_file:
                out = open(output_file, "a")
            else:
                out = sys.stdout
            self.root.pretty_print(out)
            if out:
                out.close()
        else:
            self.logger.error("Tree has not been learned yet!")

    def to_groot_root(self):
        if self.is_trained:
            return self.root.to_groot_node()
        else:
            self.logger.error("Tree has not been learned yet!")


def load_treant_from_json(filename):
    def dict_to_tree_rec(d, i):
        is_leaf = "left" not in d
        if is_leaf:
            leaf = Node(i, d["values"], 2)
            leaf.prediction = d["prediction"]
            leaf.prediction_score = d["prediction_score"]
            return leaf, i + 1
        else:
            left, i = dict_to_tree_rec(d["left"], i)
            right, i = dict_to_tree_rec(d["right"], i)
            return (
                Node(
                    i,
                    d["values"],
                    2,
                    left=left,
                    right=right,
                    best_split_feature_id=d["best_split_feature_id"],
                    best_split_feature_value=d["best_split_feature_value"],
                ),
                i + 1,
            )

    with open(filename, "r") as fp:
        tree_dict = json.load(fp)
        tree = RobustDecisionTree()
        tree.root = dict_to_tree_rec(tree_dict, 0)[0]
        tree.is_trained = True
        return tree
