from .tree_ensemble import TreeEnsemble
from ..util import convert_numpy

import numpy as np

import time

import json


def fit_provably_robust_boosting(
    X,
    y,
    n_trees=100,
    epsilon=0.1,
    lr=0.2,
    min_samples_split=10,
    min_samples_leaf=5,
    max_depth=4,
    max_weight=1.0,
    model="robust_bound",
    verbose=False,
    filename=None,
    data_augment=False,
):
    # Map 0,1 labels to -1,1
    y = np.where(y == 0, -1, 1)

    if data_augment:
        print("Using data augmentation")
        X = extend_dataset(X)
        y = np.tile(y, X.shape[0] // y.shape[0])

    ensemble = TreeEnsemble(
        weak_learner="tree",
        n_trials_coord=X.shape[1],
        lr=lr,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_weight=max_weight,
        idx_clsf=0,
    )

    if verbose:
        print("Tree\tTime")

    gamma = np.ones(X.shape[0])
    for i in range(1, n_trees + 1):
        start_time = time.time()

        # fit a new tree in order to minimize the robust loss of the whole ensemble
        weak_learner = ensemble.fit_tree(X, y, gamma, model, epsilon, depth=1)
        margin_prev = ensemble.certify_treewise(X, y, epsilon)  # needed for pruning
        ensemble.add_weak_learner(weak_learner)
        ensemble.prune_last_tree(X, y, margin_prev, epsilon, model)

        # calculate per-example weights for the next iteration
        gamma = np.exp(-ensemble.certify_treewise(X, y, epsilon))

        # If verbose print number of tree and time taken
        if verbose:
            total_time = time.time() - start_time
            print(i, total_time, sep="\t")

    if filename:
        with open(filename, "w") as file:
            json.dump(
                [
                    tree.get_json_dict(counter_terminal_nodes=-10)[0]
                    for tree in ensemble.trees
                ],
                file,
                default=convert_numpy,
            )
    else:
        return ensemble


def extend_dataset(X):
    num, dim = X.shape
    img_shape = (28, 28)
    X_img = np.reshape(np.copy(X), [num, *img_shape])
    if len(img_shape) == 2:  # introduce a fake last dimension for grayscale datasets
        X_img = X_img[:, :, :, None]

    n_crop = 2
    X_img_pad = np.pad(
        X_img,
        [(0, 0), (n_crop // 2, n_crop // 2), (n_crop // 2, n_crop // 2), (0, 0)],
        "constant",
        constant_values=0,
    )

    # Note: (1, 1) is the original image
    X_img_l = crop_batch(X_img_pad, 1, 0, n_crop)
    X_img_r = crop_batch(X_img_pad, 1, 2, n_crop)
    X_img_t = crop_batch(X_img_pad, 0, 1, n_crop)
    X_img_b = crop_batch(X_img_pad, 2, 1, n_crop)

    X_img_extended = np.vstack([X_img, X_img_l, X_img_r, X_img_t, X_img_b])

    X_final = np.reshape(X_img_extended, [-1, dim])
    return X_final


def crop_batch(X_img, n_h, n_w, n_crop):
    _, h, w, _ = X_img.shape
    bottom, right = h - (n_crop - n_h), w - (n_crop - n_w)
    return X_img[:, n_h:bottom, n_w:right, :]
