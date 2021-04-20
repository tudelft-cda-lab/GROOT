import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


def plot_adversary(X, y, adversary, ax=None):
    """
    Plot the decision tree and samples for a 2D dataset using the adversary. Uses matplotlib.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature values.
    y : array-like of shape (n_samples,)
        Class labels as integers 0 (benign) or 1 (malicious).
    adversary : groot.adversary.DecisionTreeAdversary
        Adversary for this decision tree.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    for box, probability in adversary.get_bounding_boxes():
        start_x, start_y = box[:, 0]
        width, height = box[:, 1] - box[:, 0]
        class_label = round(probability)
        # colors = {0: "#9aadd0", 1: "#e3b6a1"}
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

    x_low = np.min(X[:, 0])
    x_high = np.max(X[:, 0])
    y_low = np.min(X[:, 1])
    y_high = np.max(X[:, 1])

    x_extra = (x_high - x_low) * 0.1
    y_extra = (y_high - y_low) * 0.1

    ax.set_xlim(x_low - x_extra, x_high + x_extra)
    ax.set_ylim(y_low - y_extra, y_high + y_extra)
