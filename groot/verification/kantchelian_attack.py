import pprint
import json
import time

from gurobipy import *

from tqdm import tqdm

import numpy as np

from ..attack import AttackWrapper


"""
This code is for the most part written by Hongge Chen and is taken and
adapted from the repository https://github.com/chenhongge/RobustTrees.

It is an implementation of the MILP attack from:
Kantchelian, Alex, J. Doug Tygar, and Anthony Joseph. "Evasion and hardening of
tree ensemble classifiers." International Conference on Machine Learning.
PMLR, 2016.

Feasibility idea from:
Andriushchenko, Maksym, and Matthias Hein. "Provably robust boosted decision 
stumps and trees against adversarial attacks." arXiv preprint arXiv:1906.03526
(2019).

The changes made were related to:
- Default guard_val, round_digits values
- Removing dependency on xgboost
- Taking only a JSON file as input
- Solving a feasibility encoding for fixed epsilon
- Removing print statements
- Removing excessive model updates
- Only keeping binary classification attacks
"""


GUARD_VAL = 5e-6
ROUND_DIGITS = 6


class node_wrapper(object):
    def __init__(
        self,
        treeid,
        nodeid,
        attribute,
        threshold,
        left_leaves,
        right_leaves,
        root=False,
    ):
        # left_leaves and right_leaves are the lists of leaf indices in self.leaf_v_list
        self.attribute = attribute
        self.threshold = threshold
        self.node_pos = []
        self.leaves_lists = []
        self.add_leaves(treeid, nodeid, left_leaves, right_leaves, root)

    def print(self):
        print(
            "node_pos{}, attr:{}, th:{}, leaves:{}".format(
                self.node_pos, self.attribute, self.threshold, self.leaves_lists
            )
        )

    def add_leaves(self, treeid, nodeid, left_leaves, right_leaves, root=False):
        self.node_pos.append({"treeid": treeid, "nodeid": nodeid})
        if root:
            self.leaves_lists.append((left_leaves, right_leaves, "root"))
        else:
            self.leaves_lists.append((left_leaves, right_leaves))

    def add_grb_var(self, node_grb_var, leaf_grb_var_list):
        self.p_grb_var = node_grb_var
        self.l_grb_var_list = []
        for item in self.leaves_lists:
            left_leaf_grb_var = [leaf_grb_var_list[i] for i in item[0]]
            right_leaf_grb_var = [leaf_grb_var_list[i] for i in item[1]]
            if len(item) == 3:
                self.l_grb_var_list.append(
                    (left_leaf_grb_var, right_leaf_grb_var, "root")
                )
            else:
                self.l_grb_var_list.append((left_leaf_grb_var, right_leaf_grb_var))


class KantchelianAttack(object):
    def __init__(
        self,
        json_model,
        epsilon=None,
        order=np.inf,
        guard_val=GUARD_VAL,
        round_digits=ROUND_DIGITS,
        pos_json_input=None,
        neg_json_input=None,
        pred_threshold=0.0,
        verbose=False,
        n_threads=1,
    ):
        assert (
            epsilon is None or order == np.inf
        ), "feasibility epsilon can only be used with order inf"

        self.pred_threshold = pred_threshold
        self.epsilon = epsilon
        self.binary = (pos_json_input == None) or (neg_json_input == None)
        self.pos_json_input = pos_json_input
        self.neg_json_input = neg_json_input
        self.guard_val = guard_val
        self.round_digits = round_digits
        self.json_model = json_model
        self.order = order
        self.verbose = verbose
        self.n_threads = n_threads

        # two nodes with identical decision are merged in this list, their left and right leaves and in the list, third element of the tuple
        self.node_list = []
        self.leaf_v_list = []  # list of all leaf values
        self.leaf_pos_list = []  # list of leaves' position in xgboost model
        self.leaf_count = [0]  # total number of leaves in the first i trees
        node_check = (
            {}
        )  # track identical decision nodes. {(attr, th):<index in node_list>}

        def dfs(tree, treeid, root=False, neg=False):
            if "leaf" in tree.keys():
                if neg:
                    self.leaf_v_list.append(-tree["leaf"])
                else:
                    self.leaf_v_list.append(tree["leaf"])
                self.leaf_pos_list.append({"treeid": treeid, "nodeid": tree["nodeid"]})
                return [len(self.leaf_v_list) - 1]
            else:
                attribute, threshold, nodeid = (
                    tree["split"],
                    tree["split_condition"],
                    tree["nodeid"],
                )
                if type(attribute) == str:
                    attribute = int(attribute[1:])
                threshold = round(threshold, self.round_digits)
                # XGBoost can only offer precision up to 8 digits, however, minimum difference between two splits can be smaller than 1e-8
                # here rounding may be an option, but its hard to choose guard value after rounding
                # for example, if round to 1e-6, then guard value should be 5e-7, or otherwise may cause mistake
                # xgboost prediction has a precision of 1e-8, so when min_diff<1e-8, there is a precision problem
                # if we do not round, xgboost.predict may give wrong results due to precision, but manual predict on json file should always work
                left_subtree = None
                right_subtree = None
                for subtree in tree["children"]:
                    if subtree["nodeid"] == tree["yes"]:
                        left_subtree = subtree
                    if subtree["nodeid"] == tree["no"]:
                        right_subtree = subtree
                if left_subtree == None or right_subtree == None:
                    pprint.pprint(tree)
                    raise ValueError("should be a tree but one child is missing")
                left_leaves = dfs(left_subtree, treeid, False, neg)
                right_leaves = dfs(right_subtree, treeid, False, neg)
                if (attribute, threshold) not in node_check:
                    self.node_list.append(
                        node_wrapper(
                            treeid,
                            nodeid,
                            attribute,
                            threshold,
                            left_leaves,
                            right_leaves,
                            root,
                        )
                    )
                    node_check[(attribute, threshold)] = len(self.node_list) - 1
                else:
                    node_index = node_check[(attribute, threshold)]
                    self.node_list[node_index].add_leaves(
                        treeid, nodeid, left_leaves, right_leaves, root
                    )
                return left_leaves + right_leaves

        if self.binary:
            for i, tree in enumerate(self.json_model):
                dfs(tree, i, root=True)
                self.leaf_count.append(len(self.leaf_v_list))
            if len(self.json_model) + 1 != len(self.leaf_count):
                print("self.leaf_count:", self.leaf_count)
                raise ValueError("leaf count error")
        else:
            for i, tree in enumerate(self.pos_json_input):
                dfs(tree, i, root=True)
                self.leaf_count.append(len(self.leaf_v_list))
            for i, tree in enumerate(self.neg_json_input):
                dfs(tree, i + len(self.pos_json_input), root=True, neg=True)
                self.leaf_count.append(len(self.leaf_v_list))
            if len(self.pos_json_input) + len(self.neg_json_input) + 1 != len(
                self.leaf_count
            ):
                print("self.leaf_count:", self.leaf_count)
                raise ValueError("leaf count error")

        self.m = Model("attack")

        if not self.verbose:
            self.m.setParam(
                "OutputFlag", 0
            )  # suppress Gurobi output, gives a small speed-up and prevents huge logs

        self.m.setParam("Threads", self.n_threads)

        # Most datasets require a very low tolerance
        self.m.setParam("IntFeasTol", 1e-9)
        self.m.setParam("FeasibilityTol", 1e-9)

        self.P = self.m.addVars(len(self.node_list), vtype=GRB.BINARY, name="p")
        self.L = self.m.addVars(len(self.leaf_v_list), lb=0, ub=1, name="l")
        if epsilon:
            self.B = self.m.addVar(name="b", lb=0.0, ub=self.epsilon - 0.0001)
        elif self.order == np.inf:
            self.B = self.m.addVar(name="b")
        self.llist = [self.L[key] for key in range(len(self.L))]
        self.plist = [self.P[key] for key in range(len(self.P))]

        # p dictionary by attributes, {attr1:[(threshold1, gurobiVar1),(threshold2, gurobiVar2),...],attr2:[...]}
        self.pdict = {}
        for i, node in enumerate(self.node_list):
            node.add_grb_var(self.plist[i], self.llist)
            if node.attribute not in self.pdict:
                self.pdict[node.attribute] = [(node.threshold, self.plist[i])]
            else:
                self.pdict[node.attribute].append((node.threshold, self.plist[i]))

        # sort each feature list
        # add p constraints
        for key in self.pdict.keys():
            min_diff = 1000
            if len(self.pdict[key]) > 1:
                self.pdict[key].sort(key=lambda tup: tup[0])
                for i in range(len(self.pdict[key]) - 1):
                    self.m.addConstr(
                        self.pdict[key][i][1] <= self.pdict[key][i + 1][1],
                        name="p_consis_attr{}_{}th".format(key, i),
                    )
                    min_diff = min(
                        min_diff, self.pdict[key][i + 1][0] - self.pdict[key][i][0]
                    )

                if min_diff < 2 * self.guard_val:
                    self.guard_val = min_diff / 3
                    print(
                        "guard value too large, change to min_diff/3:", self.guard_val
                    )

        # all leaves sum up to 1
        for i in range(len(self.leaf_count) - 1):
            leaf_vars = [
                self.llist[j] for j in range(self.leaf_count[i], self.leaf_count[i + 1])
            ]
            self.m.addConstr(
                LinExpr([1] * (self.leaf_count[i + 1] - self.leaf_count[i]), leaf_vars)
                == 1,
                name="leaf_sum_one_for_tree{}".format(i),
            )

        # node leaves constraints
        for j in range(len(self.node_list)):
            p = self.plist[j]
            for k in range(len(self.node_list[j].leaves_lists)):
                left_l = [self.llist[i] for i in self.node_list[j].leaves_lists[k][0]]
                right_l = [self.llist[i] for i in self.node_list[j].leaves_lists[k][1]]
                if len(self.node_list[j].leaves_lists[k]) == 3:
                    self.m.addConstr(
                        LinExpr([1] * len(left_l), left_l) - p == 0,
                        name="p{}_root_left_{}".format(j, k),
                    )
                    self.m.addConstr(
                        LinExpr([1] * len(right_l), right_l) + p == 1,
                        name="p_{}_root_right_{}".format(j, k),
                    )
                else:
                    self.m.addConstr(
                        LinExpr([1] * len(left_l), left_l) - p <= 0,
                        name="p{}_left_{}".format(j, k),
                    )
                    self.m.addConstr(
                        LinExpr([1] * len(right_l), right_l) + p <= 1,
                        name="p{}_right_{}".format(j, k),
                    )
        self.m.update()

    def attack_feasible(self, sample, label):
        if self.binary:
            pred = (
                1 if self.check(sample, self.json_model) >= self.pred_threshold else 0
            )
        else:
            pred = (
                1
                if self.check(sample, self.pos_json_input)
                >= self.check(sample, self.neg_json_input)
                else 0
            )
        x = np.copy(sample)

        if pred != label:
            # Wrong prediction, no attack needed
            return True

        # model mislabel
        try:
            c = self.m.getConstrByName("mislabel")
            self.m.remove(c)
        except Exception:
            pass
        if (not self.binary) or label == 1:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist)
                <= self.pred_threshold - self.guard_val,
                name="mislabel",
            )
        else:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist)
                >= self.pred_threshold + self.guard_val,
                name="mislabel",
            )

        # Generate constraints for self.B, the l-infinity distance.
        for key in self.pdict.keys():
            if len(self.pdict[key]) == 0:
                raise ValueError("self.pdict list empty")
            axis = [-np.inf] + [item[0] for item in self.pdict[key]] + [np.inf]
            w = [0] * (len(self.pdict[key]) + 1)
            for i in range(len(axis) - 1, 0, -1):
                if x[key] < axis[i] and x[key] >= axis[i - 1]:
                    w[i - 1] = 0
                elif x[key] < axis[i] and x[key] < axis[i - 1]:
                    w[i - 1] = np.abs(x[key] - axis[i - 1])
                elif x[key] >= axis[i] and x[key] >= axis[i - 1]:
                    w[i - 1] = np.abs(x[key] - axis[i] + self.guard_val)
                else:
                    print("x[key]:", x[key])
                    print("axis:", axis)
                    print("axis[i]:{}, axis[i-1]:{}".format(axis[i], axis[i - 1]))
                    raise ValueError("wrong axis ordering")
            for i in range(len(w) - 1):
                w[i] -= w[i + 1]
            else:
                try:
                    c = self.m.getConstrByName("linf_constr_attr{}".format(key))
                    self.m.remove(c)
                except Exception:
                    pass
                self.m.addConstr(
                    LinExpr(w[:-1], [item[1] for item in self.pdict[key]]) + w[-1]
                    <= self.B,
                    name="linf_constr_attr{}".format(key),
                )

        self.m.setObjective(0, GRB.MINIMIZE)

        self.m.update()
        self.m.optimize()

        return not self.m.status == 3  # 3 -> infeasible -> no adv example -> False

    def optimal_adversarial_example(self, sample, label):
        if self.binary:
            pred = (
                1 if self.check(sample, self.json_model) >= self.pred_threshold else 0
            )
        else:
            pred = (
                1
                if self.check(sample, self.pos_json_input)
                >= self.check(sample, self.neg_json_input)
                else 0
            )
        x = np.copy(sample)

        if pred != label:
            # Wrong prediction, no attack needed
            return x

        # model mislabel
        # this is for binary
        try:
            c = self.m.getConstrByName("mislabel")
            self.m.remove(c)
        except Exception:
            pass
        if (not self.binary) or label == 1:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist)
                <= self.pred_threshold - self.guard_val,
                name="mislabel",
            )
        else:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist)
                >= self.pred_threshold + self.guard_val,
                name="mislabel",
            )

        if self.order == np.inf:
            rho = 1
        else:
            rho = self.order

        if self.order != np.inf:
            self.obj_coeff_list = []
            self.obj_var_list = []
            self.obj_c = 0
        # model objective
        for key in self.pdict.keys():
            if len(self.pdict[key]) == 0:
                raise ValueError("self.pdict list empty")
            axis = [-np.inf] + [item[0] for item in self.pdict[key]] + [np.inf]
            w = [0] * (len(self.pdict[key]) + 1)
            for i in range(len(axis) - 1, 0, -1):
                if x[key] < axis[i] and x[key] >= axis[i - 1]:
                    w[i - 1] = 0
                elif x[key] < axis[i] and x[key] < axis[i - 1]:
                    w[i - 1] = np.abs(x[key] - axis[i - 1]) ** rho
                elif x[key] >= axis[i] and x[key] >= axis[i - 1]:
                    w[i - 1] = np.abs(x[key] - axis[i] + self.guard_val) ** rho
                else:
                    print("x[key]:", x[key])
                    print("axis:", axis)
                    print("axis[i]:{}, axis[i-1]:{}".format(axis[i], axis[i - 1]))
                    raise ValueError("wrong axis ordering")
            for i in range(len(w) - 1):
                w[i] -= w[i + 1]
            if self.order != np.inf:
                self.obj_c += w[-1]
                self.obj_coeff_list += w[:-1]
                self.obj_var_list += [item[1] for item in self.pdict[key]]
            else:
                try:
                    c = self.m.getConstrByName("linf_constr_attr{}".format(key))
                    self.m.remove(c)
                except Exception:
                    pass
                self.m.addConstr(
                    LinExpr(w[:-1], [item[1] for item in self.pdict[key]]) + w[-1]
                    <= self.B,
                    name="linf_constr_attr{}".format(key),
                )

        if self.order != np.inf:
            self.m.setObjective(
                LinExpr(self.obj_coeff_list, self.obj_var_list) + self.obj_c,
                GRB.MINIMIZE,
            )
        else:
            self.m.setObjective(self.B, GRB.MINIMIZE)

        self.m.update()
        self.m.optimize()

        # If infeasible
        if self.m.status == 3:
            return None

        # Assert that the adversarial example causes a misclassification
        for key in self.pdict.keys():
            for node in self.pdict[key]:
                if node[1].x > 0.5 and x[key] >= node[0]:
                    x[key] = node[0] - self.guard_val
                if node[1].x <= 0.5 and x[key] < node[0]:
                    x[key] = node[0] + self.guard_val

        if self.binary:
            pred = 1 if self.check(x, self.json_model) >= self.pred_threshold else 0
        else:
            pos_value = self.check(x, self.pos_json_input)
            neg_value = self.check(x, self.neg_json_input)
            pred = 1 if pos_value >= neg_value else 0

        if pred == label and self.verbose:
            print("!" * 50)
            print("MILP result did not cause a misclassification!")
            print("!" * 50)

        return x

    def check(self, x, json_file):
        # Due to XGBoost precision issues, some attacks may not succeed if tested using model.predict.
        # We manually run the tree on the json file here to make sure those attacks are actually successful.
        leaf_values = []
        for item in json_file:
            tree = item.copy()
            while "leaf" not in tree.keys():
                attribute, threshold, nodeid = (
                    tree["split"],
                    tree["split_condition"],
                    tree["nodeid"],
                )
                if type(attribute) == str:
                    attribute = int(attribute[1:])
                if x[attribute] < threshold:
                    if tree["children"][0]["nodeid"] == tree["yes"]:
                        tree = tree["children"][0].copy()
                    elif tree["children"][1]["nodeid"] == tree["yes"]:
                        tree = tree["children"][1].copy()
                    else:
                        pprint.pprint(tree)
                        print("x[attribute]:", x[attribute])
                        raise ValueError("child not found")
                else:
                    if tree["children"][0]["nodeid"] == tree["no"]:
                        tree = tree["children"][0].copy()
                    elif tree["children"][1]["nodeid"] == tree["no"]:
                        tree = tree["children"][1].copy()
                    else:
                        pprint.pprint(tree)
                        print("x[attribute]:", x[attribute])
                        raise ValueError("child not found")
            leaf_values.append(tree["leaf"])
        manual_res = np.sum(leaf_values)
        return manual_res


class KantchelianAttackMultiClass(object):
    def __init__(
        self,
        json_model,
        n_classes,
        order=np.inf,
        epsilon=None,
        guard_val=GUARD_VAL,
        round_digits=ROUND_DIGITS,
        pred_threshold=0.0,
        low_memory=False,
        verbose=False,
        n_threads=1,
    ):
        if n_classes <= 2:
            raise ValueError("multiclass attack must be used when number of class > 2")

        assert (
            epsilon is None or order == np.inf
        ), "feasibility epsilon can only be used with order inf"

        self.n_classes = n_classes
        self.order = order
        self.epsilon = epsilon
        self.guard_val = guard_val
        self.round_digits = round_digits
        self.pred_threshold = pred_threshold
        self.low_memory = low_memory
        self.verbose = verbose
        self.n_threads = n_threads

        # Create all attacker models, this takes quadratic space in terms
        # of n_classes, but speeds up attacks for many samples.
        self.one_vs_all_models = [[] for _ in range(self.n_classes)]
        for i, json_tree in enumerate(json_model):
            self.one_vs_all_models[i % n_classes].append(json_tree)

        if not low_memory:
            self.__create_cached_attackers()

    def __create_cached_attackers(self):
        self.binary_attackers = []
        for class_label in range(self.n_classes):
            attackers = []
            for other_label in range(self.n_classes):
                if class_label == other_label:
                    attackers.append(None)
                    continue

                attacker = KantchelianAttack(
                    None,
                    epsilon=self.epsilon,
                    order=self.order,
                    guard_val=self.guard_val,
                    round_digits=self.round_digits,
                    pred_threshold=self.pred_threshold,
                    verbose=self.verbose,
                    n_threads=self.n_threads,
                    pos_json_input=self.one_vs_all_models[class_label],
                    neg_json_input=self.one_vs_all_models[other_label],
                )

                attackers.append(attacker)
            self.binary_attackers.append(attackers)
        return self.binary_attackers

    def optimal_adversarial_example(self, sample, label):
        best_distance = float("inf")
        best_adv_example = None

        for other_label in range(self.n_classes):
            if other_label == label:
                continue

            # Create new attacker or use a cached attacker
            if self.low_memory:
                attacker = KantchelianAttack(
                    None,
                    epsilon=self.epsilon,
                    order=self.order,
                    guard_val=self.guard_val,
                    round_digits=self.round_digits,
                    pred_threshold=self.pred_threshold,
                    verbose=self.verbose,
                    n_threads=self.n_threads,
                    pos_json_input=self.one_vs_all_models[label],
                    neg_json_input=self.one_vs_all_models[other_label],
                )
            else:
                attacker = self.binary_attackers[label][other_label]

            # Generate adversarial example on this binary attacker
            adv_example = attacker.optimal_adversarial_example(sample, 1)

            # If this binary attacker example was better than the previous ones, keep it
            if adv_example is not None:
                distance = np.linalg.norm(sample - adv_example, ord=self.order)
                if distance < best_distance:
                    best_adv_example = adv_example
                    best_distance = distance

        if best_adv_example is None:
            raise Exception(
                "No adversarial example found, does your model predict a constant value?"
            )

        return best_adv_example

    def attack_feasible(self, sample, label):
        for other_label in range(self.n_classes):
            if other_label == label:
                continue

            # Create new attacker or use a cached attacker
            if self.low_memory:
                attacker = KantchelianAttack(
                    None,
                    epsilon=self.epsilon,
                    order=self.order,
                    guard_val=self.guard_val,
                    round_digits=self.round_digits,
                    pred_threshold=self.pred_threshold,
                    verbose=self.verbose,
                    n_threads=self.n_threads,
                    pos_json_input=self.one_vs_all_models[label],
                    neg_json_input=self.one_vs_all_models[other_label],
                )
            else:
                attacker = self.binary_attackers[label][other_label]

            # Check if the binary attacker can create an adversarial example
            if attacker.attack_feasible(sample, 1):
                return True

        return False


DEFAULT_OPTIONS = {
    "epsilon": None,
    "guard_val": GUARD_VAL,
    "round_digits": ROUND_DIGITS,
    "pred_threshold": 0.0,
    "order": np.inf,
    "low_memory": False,
    "verbose": False,
    "n_threads": 1,
}


class KantchelianAttackWrapper(AttackWrapper):
    def __init__(self, json_model, n_classes):
        self.json_model = json_model
        self.n_classes = n_classes

    def __get_attacker(self, order, options):
        if self.n_classes == 2:
            attack = KantchelianAttack(
                self.json_model,
                order=order,
                epsilon=options["epsilon"],
                guard_val=options["guard_val"],
                round_digits=options["round_digits"],
                pred_threshold=options["pred_threshold"],
                verbose=options["verbose"],
                n_threads=options["n_threads"],
            )
        else:
            attack = KantchelianAttackMultiClass(
                self.json_model,
                self.n_classes,
                order=order,
                epsilon=options["epsilon"],
                guard_val=options["guard_val"],
                round_digits=options["round_digits"],
                pred_threshold=options["pred_threshold"],
                low_memory=options["low_memory"],
                verbose=options["verbose"],
                n_threads=options["n_threads"],
            )
        return attack

    def attack_feasibility(self, X, y, order, epsilon, options={}):
        default_options = DEFAULT_OPTIONS.copy()
        default_options.update(options)
        options = default_options
        options["epsilon"] = epsilon

        attack = self.__get_attacker(order, options)

        attack_feasible = []
        start_time = time.time()

        for sample, label in tqdm(zip(X, y), total=X.shape[0]):
            attack_feasible.append(attack.attack_feasible(sample, label))

        total_time = time.time() - start_time

        if options["verbose"]:
            print("Total time:", total_time)
            print("Avg time per instance:", total_time / len(X))

        return np.array(attack_feasible)

    def adversarial_examples(self, X, y, order, options={}):
        default_options = DEFAULT_OPTIONS.copy()
        default_options.update(options)
        options = default_options

        attack = self.__get_attacker(order, options)

        start_time = time.time()

        X_adv = []
        for sample, label in tqdm(zip(X, y), total=X.shape[0]):
            optimal_example = attack.optimal_adversarial_example(sample, label)
            X_adv.append(optimal_example)

        total_time = time.time() - start_time

        if options["verbose"]:
            print("Total time:", total_time)
            print("Avg time per instance:", total_time / len(X))

        return np.array(X_adv)
