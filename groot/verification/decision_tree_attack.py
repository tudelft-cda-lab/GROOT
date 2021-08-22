from ..attack import AttackWrapper

import numpy as np

from tqdm import tqdm

from collections import defaultdict

from copy import deepcopy


def _extract_bounding_boxes(tree, bounds):
    if "leaf" in tree:
        return [(deepcopy(bounds), tree["leaf"])]
    else:
        leaves = []

        # If the split's new right threshold (so the node on the left) is more specific
        # than the previous one, update the bound and recurse
        old_bound = bounds[tree["split"]][1]
        if (
            tree["split_condition"] < bounds[tree["split"]][1]
            and tree["split_condition"] > bounds[tree["split"]][0]
        ):
            bounds[tree["split"]][1] = tree["split_condition"]

        if tree["split_condition"] >= bounds[tree["split"]][0]:
            for subtree in tree["children"]:
                if subtree["nodeid"] == tree["yes"]:
                    leaves.extend(_extract_bounding_boxes(subtree, bounds))

        bounds[tree["split"]][1] = old_bound

        # If the split's new left threshold (so the node on the right) is more specific
        # than the previous one, update the bound and recurse
        old_bound = bounds[tree["split"]][0]
        if (
            tree["split_condition"] > bounds[tree["split"]][0]
            and tree["split_condition"] < bounds[tree["split"]][1]
        ):
            bounds[tree["split"]][0] = tree["split_condition"]

        if tree["split_condition"] <= bounds[tree["split"]][1]:
            for subtree in tree["children"]:
                if subtree["nodeid"] == tree["no"]:
                    leaves.extend(_extract_bounding_boxes(subtree, bounds))

        bounds[tree["split"]][0] = old_bound

        return leaves


class DecisionTreeAttackWrapper(AttackWrapper):
    def __init__(self, json_model, n_classes):
        if len(json_model) != 1:
            raise ValueError("This attack can only be used with single decision trees")
        self.json_model = json_model

        if n_classes != 2:
            raise ValueError("Currently only binary classification is supported")
        self.n_classes = n_classes

        bounds = defaultdict(lambda: np.array([-np.inf, np.inf]))
        self.leaves = _extract_bounding_boxes(json_model[0], bounds)

    def adversarial_examples(self, X, y, order, options={}):
        # Turn 'leaves' into bounding boxes and leaf prediction values
        bound_dicts, leaf_values = zip(*self.leaves)
        predictions = [value > 0 for value in leaf_values]
        bounding_boxes = []
        for bound_dict in bound_dicts:
            bounding_box = np.tile(np.array([-np.inf, np.inf]), (X.shape[1], 1))
            for i, bound in bound_dict.items():
                bounding_box[i] = bound
            bounding_boxes.append(bounding_box)

        X_adv = []
        for sample, label in tqdm(zip(X, y), total=X.shape[0]):
            # Create a minimal adversarial example for each leaf
            # then choose the one with minimal distance
            best_distance = np.inf
            for bounding_box, prediction in zip(bounding_boxes, predictions):
                if prediction != label:
                    adv_example = np.clip(
                        sample, bounding_box[:, 0], bounding_box[:, 1]
                    )
                    distance = np.linalg.norm(adv_example - sample, ord=order)

                    if distance < best_distance:
                        best_distance = distance
                        best_adv_example = adv_example

            if best_distance == np.inf:
                # If no adversarial example is possible (tree predicts one value everywhere)
                # then return a vector of NaNs
                X_adv.append(np.full(len(sample), np.nan))
            else:
                X_adv.append(best_adv_example)

        return np.array(X_adv)
