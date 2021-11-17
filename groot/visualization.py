from .toolbox import Model

import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np

from sklearn.base import is_classifier


def plot_estimator(X, y, estimator, ax=None, steps=100, colors=("b", "r")):
    """
    Plot a scikit-learn estimator and samples for a 2D dataset. Uses matplotlib.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature values.
    y : array-like of shape (n_samples,)
        Ground truth targets.
    estimator : Scikit-learn compatible estimator
        Estimator to visualize
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.
    """
    if X.shape[1] != 2:
        raise ValueError("X must be 2D")

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    x_low = np.min(X[:, 0])
    x_high = np.max(X[:, 0])
    y_low = np.min(X[:, 1])
    y_high = np.max(X[:, 1])

    x_extra = (x_high - x_low) * 0.1
    y_extra = (y_high - y_low) * 0.1

    x_low -= x_extra
    x_high += x_extra
    y_low -= y_extra
    y_high += y_extra

    xx, yy = np.meshgrid(
        np.linspace(x_low, x_high, steps), np.linspace(y_low, y_high, steps)
    )

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if isinstance(estimator, Model) or is_classifier(estimator):
        ax.contourf(xx, yy, Z, alpha=0.25, levels=1, colors=colors)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], marker="_", c="b")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], marker="+", c="r")
    else:
        vmin = np.min(y)
        vmax = np.max(y)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="bwr", levels=10, vmin=vmin, vmax=vmax)
        ax.scatter(X[:, 0], X[:, 1], marker=".", c=y, cmap="bwr", vmin=vmin, vmax=vmax)

    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)


def plot_adversary(X, y, adversary, ax=None):
    """
    Plot the decision tree and samples for a 2D dataset using the adversary. Uses matplotlib.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature values.
    y : array-like of shape (n_samples,)
        Class labels as integers 0 and 1.
    adversary : groot.adversary.DecisionTreeAdversary
        Adversary for this decision tree.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.
    """
    if X.shape[1] != 2:
        raise ValueError("X must be 2D")

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    x_low = np.min(X[:, 0])
    x_high = np.max(X[:, 0])
    y_low = np.min(X[:, 1])
    y_high = np.max(X[:, 1])

    x_extra = (x_high - x_low) * 0.1
    y_extra = (y_high - y_low) * 0.1

    bounding_boxes = adversary.get_bounding_boxes()

    for box, probability in bounding_boxes:
        box[0] = np.clip(box[0], x_low - x_extra, x_high + x_extra)
        box[1] = np.clip(box[1], y_low - y_extra, y_high + y_extra)

        start_x, start_y = box[:, 0]
        width, height = box[:, 1] - box[:, 0]
        class_label = round(probability)

        colors = {0: "b", 1: "r"}
        color = colors[class_label]
        ax.add_patch(
            patches.Rectangle(
                (start_x, start_y),
                width,
                height,
                color=color,
                fill=True,
                alpha=0.25,
                linewidth=0,
            )
        )

    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker="_", c="b")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker="+", c="r")

    ax.set_xlim(x_low - x_extra, x_high + x_extra)
    ax.set_ylim(y_low - y_extra, y_high + y_extra)
