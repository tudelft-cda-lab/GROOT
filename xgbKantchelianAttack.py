import pprint
from gurobipy import *
import numpy as np
import json
import time
from tqdm import tqdm


GUARD_VAL = 5e-7  # 2e-7
ROUND_DIGITS = 6  # 20


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


class KantchelianAttackLinfEpsilon(object):
    def __init__(
        self,
        json_model,
        epsilon,
        guard_val=GUARD_VAL,
        round_digits=ROUND_DIGITS,
        LP=False,
        binary=True,
        pos_json_input=None,
        neg_json_input=None,
        pred_threshold=0,
    ):
        self.pred_threshold = pred_threshold
        self.epsilon = epsilon
        self.LP = LP
        self.binary = binary or (pos_json_input == None) or (neg_json_input == None)
        # print('binary: ', self.binary)
        # if LP:
        # print('USING LINEAR PROGRAMMING APPROXIMATION!!')
        # else:
        # print('USING MILP EXACT SOLVER!!')
        self.guard_val = guard_val
        self.round_digits = round_digits

        # print('round features to {} digits'.format(self.round_digits))
        # print('guard value is :', guard_val)
        # print('feature values are rounded to {} digits'.format(round_digits))
        self.json_file = json_model

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
            for i, tree in enumerate(self.json_file):
                dfs(tree, i, root=True)
                self.leaf_count.append(len(self.leaf_v_list))
            if len(self.json_file) + 1 != len(self.leaf_count):
                print("self.leaf_count:", self.leaf_count)
                raise ValueError("leaf count error")
        else:
            for i, tree in enumerate(self.pos_json_file):
                dfs(tree, i, root=True)
                self.leaf_count.append(len(self.leaf_v_list))
            for i, tree in enumerate(self.neg_json_file):
                dfs(tree, i + len(self.pos_json_file), root=True, neg=True)
                self.leaf_count.append(len(self.leaf_v_list))
            if len(self.pos_json_file) + len(self.neg_json_file) + 1 != len(
                self.leaf_count
            ):
                print("self.leaf_count:", self.leaf_count)
                raise ValueError("leaf count error")

        self.m = Model("attack")
        # self.m.setParam('Threads', 1)
        self.m.setParam(
            "OutputFlag", 0
        )  # suppress Gurobi output, gives a small speed-up and prevents huge logs
        if self.LP:
            self.P = self.m.addVars(len(self.node_list), lb=0, ub=1, name="p")
        else:
            self.P = self.m.addVars(len(self.node_list), vtype=GRB.BINARY, name="p")
        self.L = self.m.addVars(len(self.leaf_v_list), lb=0, ub=1, name="l")
        self.B = self.m.addVar(
            name="b", lb=0.0, ub=self.epsilon - 0.0001
        )  # TODO: experimental
        self.llist = [self.L[key] for key in range(len(self.L))]
        self.plist = [self.P[key] for key in range(len(self.P))]

        # print('leaf value list:',self.leaf_v_list)
        # print('number of leaves in the first k trees:',self.leaf_count)

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
                # print('attr {} min difference between thresholds:{}'.format(key,min_diff))
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

    def attack(self, X, label):
        # pred = 1 if self.check(X, self.json_file) >= self.pred_threshold else 0
        x = np.copy(X)
        # print('\n\n==================================')

        # if pred != label:
        #     print('wrong prediction, no need to attack')
        #     return False

        # model mislabel
        # this is for binary
        try:
            c = self.m.getConstrByName("mislabel")
            self.m.remove(c)
            # self.m.update()
        except Exception:
            pass
        if (not self.binary) or label == 1:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist) <= self.pred_threshold,
                name="mislabel",
            )
        else:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist)
                >= self.pred_threshold + self.guard_val,
                name="mislabel",
            )
        # self.m.update()

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
                    # self.m.update()
                except Exception:
                    pass
                self.m.addConstr(
                    LinExpr(w[:-1], [item[1] for item in self.pdict[key]]) + w[-1]
                    <= self.B,
                    name="linf_constr_attr{}".format(key),
                )
                # self.m.update()

        self.m.setObjective(0, GRB.MINIMIZE)

        self.m.update()
        self.m.optimize()

        # print("status:", self.m.status)

        return self.m.status == 3  # 3 -> infeasible -> no adv example -> True

    def check(self, x, json_file):
        # Due to XGBoost precision issues, some attacks may not succeed if tested using model.predict.
        # We manually run the tree on the json file here to make sure those attacks are actually successful.
        # print('-------------------------------------\nstart checking')
        # print('manually run trees')
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
        # print('leaf values:{}, \nsum:{}'.format(leaf_values, manual_res))
        return manual_res


def attack_epsilon_feasibility(
    json_filename,
    X,
    y,
    epsilon,
    guard_val=GUARD_VAL,
    round_digits=ROUND_DIGITS,
    sample_limit=None,
    pred_threshold=0.5,
):
    json_model = json.load(open(json_filename, "r"))

    attack = KantchelianAttackLinfEpsilon(
        json_model,
        epsilon,
        guard_val=guard_val,
        round_digits=round_digits,
        pred_threshold=pred_threshold,
    )

    if sample_limit:
        X = X[:sample_limit]
        y = y[:sample_limit]

    n_correct_within_epsilon = 0
    global_start = time.time()
    progress_bar = tqdm(total=X.shape[0])
    for sample, label in zip(X, y):
        # predict = 1 if attack.check(sample, json_model) >= pred_threshold else 0
        # if predict != label:

        correct_within_epsilon = attack.attack(sample, label)
        if correct_within_epsilon:
            n_correct_within_epsilon += 1

        progress_bar.update()
    progress_bar.close()

    total_time = time.time() - global_start
    print("Total time:", total_time)
    print("Avg time per instance:", total_time / len(X))

    adv_accuracy = n_correct_within_epsilon / len(X)

    return adv_accuracy


class xgbKantchelianAttack(object):
    def __init__(
        self,
        json_model,
        order=np.inf,
        guard_val=GUARD_VAL,
        round_digits=ROUND_DIGITS,
        LP=False,
        binary=True,
        pos_json_input=None,
        neg_json_input=None,
        pred_threshold=0,
    ):
        self.pred_threshold = pred_threshold

        self.LP = LP
        self.binary = binary or (pos_json_input == None) or (neg_json_input == None)
        # print('binary: ', self.binary)
        # if LP:
        # print('USING LINEAR PROGRAMMING APPROXIMATION!!')
        # else:
        # print('USING MILP EXACT SOLVER!!')
        self.guard_val = guard_val
        self.round_digits = round_digits
        # print('order is:',order)
        # print('round features to {} digits'.format(self.round_digits))
        # print('guard value is :', guard_val)
        # print('feature values are rounded to {} digits'.format(round_digits))
        self.json_file = json_model

        self.order = order
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
            for i, tree in enumerate(self.json_file):
                dfs(tree, i, root=True)
                self.leaf_count.append(len(self.leaf_v_list))
            if len(self.json_file) + 1 != len(self.leaf_count):
                print("self.leaf_count:", self.leaf_count)
                raise ValueError("leaf count error")
        else:
            for i, tree in enumerate(self.pos_json_file):
                dfs(tree, i, root=True)
                self.leaf_count.append(len(self.leaf_v_list))
            for i, tree in enumerate(self.neg_json_file):
                dfs(tree, i + len(self.pos_json_file), root=True, neg=True)
                self.leaf_count.append(len(self.leaf_v_list))
            if len(self.pos_json_file) + len(self.neg_json_file) + 1 != len(
                self.leaf_count
            ):
                print("self.leaf_count:", self.leaf_count)
                raise ValueError("leaf count error")

        self.m = Model("attack")
        self.m.setParam("Threads", 8)
        if self.LP:
            self.P = self.m.addVars(len(self.node_list), lb=0, ub=1, name="p")
        else:
            self.P = self.m.addVars(len(self.node_list), vtype=GRB.BINARY, name="p")
        self.L = self.m.addVars(len(self.leaf_v_list), lb=0, ub=1, name="l")
        if self.order == np.inf:
            self.B = self.m.addVar(name="b")
        self.llist = [self.L[key] for key in range(len(self.L))]
        self.plist = [self.P[key] for key in range(len(self.P))]

        # print('leaf value list:',self.leaf_v_list)
        # print('number of leaves in the first k trees:',self.leaf_count)

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
                # print('attr {} min difference between thresholds:{}'.format(key,min_diff))
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

    def attack(self, X, label):
        pred = 1 if self.check(X, self.json_file) >= self.pred_threshold else 0
        x = np.copy(X)
        print("\n\n==================================")

        if pred != label:
            print("wrong prediction, no need to attack")
            return X

        print("X:", x)
        print("label:", label)
        print("prediction:", pred)
        # model mislabel
        # this is for binary
        try:
            c = self.m.getConstrByName("mislabel")
            self.m.remove(c)
            self.m.update()
        except Exception:
            pass
        if (not self.binary) or label == 1:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist) <= self.pred_threshold,
                name="mislabel",
            )
        else:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist)
                >= self.pred_threshold + self.guard_val,
                name="mislabel",
            )
        self.m.update()

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
                    self.m.update()
                except Exception:
                    pass
                self.m.addConstr(
                    LinExpr(w[:-1], [item[1] for item in self.pdict[key]]) + w[-1]
                    <= self.B,
                    name="linf_constr_attr{}".format(key),
                )
                self.m.update()

        if self.order != np.inf:
            self.m.setObjective(
                LinExpr(self.obj_coeff_list, self.obj_var_list) + self.obj_c,
                GRB.MINIMIZE,
            )
        else:
            self.m.setObjective(self.B, GRB.MINIMIZE)

        self.m.update()
        self.m.optimize()

        print("Obj: %g" % self.m.objVal)
        for key in self.pdict.keys():
            for node in self.pdict[key]:
                if node[1].x > 0.5 and x[key] >= node[0]:
                    x[key] = node[0] - self.guard_val
                if node[1].x <= 0.5 and x[key] < node[0]:
                    x[key] = node[0] + self.guard_val

        print("\n-------------------------------------\n")
        print("result for this point:", x)
        self.check(x, self.json_file)
        return x

    def attack_epsilon(self, X, label, epsilon):
        pred = 1 if self.check(X, self.json_file) >= self.pred_threshold else 0
        x = np.copy(X)

        if pred != label:
            # Wrong prediction, no attack
            return False

        # model mislabel
        # this is for binary
        try:
            c = self.m.getConstrByName("mislabel")
            self.m.remove(c)
            self.m.update()
        except Exception:
            pass
        if (not self.binary) or label == 1:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist) <= self.pred_threshold,
                name="mislabel",
            )
        else:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist)
                >= self.pred_threshold + self.guard_val,
                name="mislabel",
            )
        self.m.update()

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
                    self.m.update()
                except Exception:
                    pass
                self.m.addConstr(
                    LinExpr(w[:-1], [item[1] for item in self.pdict[key]]) + w[-1]
                    <= self.B,
                    name="linf_constr_attr{}".format(key),
                )
                self.m.update()

        if self.order != np.inf:
            self.m.setObjective(
                LinExpr(self.obj_coeff_list, self.obj_var_list) + self.obj_c,
                GRB.MINIMIZE,
            )
        else:
            self.m.setObjective(self.B, GRB.MINIMIZE)

        self.m.setParam("BestObjStop", epsilon * 0.999)
        self.m.setParam("BestBdStop", epsilon * 1.001)

        self.m.update()
        self.m.optimize()

        # This <= has been changed to < for provably robust boosting
        if self.m.objVal < epsilon - 0.001:
            # TODO: not sure if it was okay commenting this out
            # # Assert that the adversarial example causes a misclassification
            # for key in self.pdict.keys():
            #     for node in self.pdict[key]:
            #         if node[1].x > 0.5 and x[key] >= node[0]:
            #             x[key] = node[0] - self.guard_val
            #         if node[1].x <= 0.5 and x[key] < node[0]:
            #             x[key] = node[0] + self.guard_val

            # # assert int(self.check(x, self.json_file) >= self.pred_threshold) != label
            # if int(self.check(x, self.json_file) >= self.pred_threshold) == label:
            #     print("NOT A MISCLASSIFICATION", self.check(x, self.json_file), label)
            # return False

            return False

        return True

    def optimal_adversarial_example(self, sample, label):
        pred = 1 if self.check(sample, self.json_file) >= self.pred_threshold else 0
        x = np.copy(sample)

        if pred != label:
            # Wrong prediction, no attack
            return x

        # model mislabel
        # this is for binary
        try:
            c = self.m.getConstrByName("mislabel")
            self.m.remove(c)
            self.m.update()
        except Exception:
            pass
        if (not self.binary) or label == 1:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist) <= self.pred_threshold,
                name="mislabel",
            )
        else:
            self.m.addConstr(
                LinExpr(self.leaf_v_list, self.llist)
                >= self.pred_threshold + self.guard_val,
                name="mislabel",
            )
        self.m.update()

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
                    self.m.update()
                except Exception:
                    pass
                self.m.addConstr(
                    LinExpr(w[:-1], [item[1] for item in self.pdict[key]]) + w[-1]
                    <= self.B,
                    name="linf_constr_attr{}".format(key),
                )
                self.m.update()

        if self.order != np.inf:
            self.m.setObjective(
                LinExpr(self.obj_coeff_list, self.obj_var_list) + self.obj_c,
                GRB.MINIMIZE,
            )
        else:
            self.m.setObjective(self.B, GRB.MINIMIZE)

        self.m.update()
        self.m.optimize()

        # Assert that the adversarial example causes a misclassification
        for key in self.pdict.keys():
            for node in self.pdict[key]:
                if node[1].x > 0.5 and x[key] >= node[0]:
                    x[key] = node[0] - self.guard_val
                if node[1].x <= 0.5 and x[key] < node[0]:
                    x[key] = node[0] + self.guard_val

        pred_proba = self.check(x, self.json_file)
        pred = 1 if self.check(x, self.json_file) >= self.pred_threshold else 0
        print("New prediction:", pred_proba, pred)
        print("Original label:", label)

        if pred == label:
            print("MILP result did not cause a misclassification")
        #     raise Exception("MILP result did not cause a misclassification")

        return x

    def check(self, x, json_file):
        # Due to XGBoost precision issues, some attacks may not succeed if tested using model.predict.
        # We manually run the tree on the json file here to make sure those attacks are actually successful.
        # print('-------------------------------------\nstart checking')
        # print('manually run trees')
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
        # print('leaf values:{}, \nsum:{}'.format(leaf_values, manual_res))
        return manual_res


def attack_binary_dataset_epsilon(
    json_filename,
    X,
    y,
    epsilon,
    guard_val=GUARD_VAL,
    round_digits=ROUND_DIGITS,
    sample_limit=500,
    pred_threshold=0.5,
):
    json_model = json.load(open(json_filename, "r"))

    attack = xgbKantchelianAttack(
        json_model,
        guard_val=guard_val,
        round_digits=round_digits,
        pred_threshold=pred_threshold,
    )

    X = X[:sample_limit]
    y = y[:sample_limit]

    n_correct_within_epsilon = 0
    global_start = time.time()
    for sample, label in zip(X, y):

        predict = 1 if attack.check(sample, json_model) >= pred_threshold else 0
        if label != predict:
            print("Wrong prediction")
            continue

        correct_within_epsilon = attack.attack_epsilon(sample, label, epsilon)
        if correct_within_epsilon:
            n_correct_within_epsilon += 1

    total_time = time.time() - global_start
    print("Total time:", total_time)
    print("Avg time per instance:", total_time / len(X))

    adv_accuracy = n_correct_within_epsilon / len(X)

    return adv_accuracy


def score_dataset(
    json_filename,
    X,
    y,
    guard_val=GUARD_VAL,
    round_digits=ROUND_DIGITS,
    sample_limit=500,
    pred_threshold=0.5,
):
    json_model = json.load(open(json_filename, "r"))

    attack = xgbKantchelianAttack(
        json_model,
        guard_val=guard_val,
        round_digits=round_digits,
        pred_threshold=pred_threshold,
    )

    X = X[:sample_limit]
    y = y[:sample_limit]

    n_correct = 0
    for sample, label in zip(X, y):
        predict = 1 if attack.check(sample, json_model) >= pred_threshold else 0
        if label != predict:
            # print("Wrong prediction", predict)
            continue
        else:
            # print("Correct prediction", predict)
            n_correct += 1

    return n_correct / len(X)


def optimal_adversarial_example(
    json_filename,
    sample,
    label,
    guard_val=GUARD_VAL,
    round_digits=ROUND_DIGITS,
    pred_threshold=0.5,
):
    json_model = json.load(open(json_filename, "r"))

    attack = xgbKantchelianAttack(
        json_model,
        guard_val=guard_val,
        round_digits=round_digits,
        pred_threshold=pred_threshold,
    )

    return attack.optimal_adversarial_example(sample, label)


def attack_binary_dataset(
    json_filename,
    X,
    y,
    epsilon=0.1,
    guard_val=GUARD_VAL,
    round_digits=ROUND_DIGITS,
    pred_threshold=0.5,
):
    np.random.seed(8)

    json_model = json.load(open(json_filename, "r"))

    attack = xgbKantchelianAttack(
        json_model,
        guard_val=guard_val,
        round_digits=round_digits,
        pred_threshold=pred_threshold,
    )

    samples = np.arange(len(X))
    num_attacks = len(
        samples
    )  # real number of attacks cannot be larger than test data size
    avg_dist = 0
    counter = 0
    n_correct_within_epsilon = 0
    distances = []
    global_start = time.time()
    for n, idx in enumerate(samples):
        print(
            "\n\n\n\n======== Point {} ({}/{}) starts =========".format(
                idx, n + 1, num_attacks
            )
        )
        predict = 1 if attack.check(X[idx], json_model) >= pred_threshold else 0
        if y[idx] == predict:
            counter += 1
        else:
            print("true label:{}, predicted label:{}".format(y[idx], predict))
            print("prediction not correct, skip this one.")
            continue
        adv = attack.attack(X[idx], y[idx])
        dist = np.max(np.abs(adv - X[idx]))
        avg_dist += dist
        if dist > epsilon:
            n_correct_within_epsilon += 1
        distances.append(dist)
        print(
            "\n======== Point {} ({}/{}) finished, distortion:{} =========".format(
                idx, n + 1, num_attacks, dist
            )
        )
    print(
        "\n\nattacked {}/{} points, average linf distortion: {}, total time:{}".format(
            counter, num_attacks, avg_dist / counter, time.time() - global_start
        )
    )

    print(
        "\n\nattacked {}/{} points, average linf distortion including wrong predictions: {}, total time:{}".format(
            counter, num_attacks, avg_dist / num_attacks, time.time() - global_start
        )
    )

    print(f"{n_correct_within_epsilon}/{num_attacks} correct within epsilon {epsilon}")

    adv_accuracy = n_correct_within_epsilon / len(samples)

    return adv_accuracy, distances
