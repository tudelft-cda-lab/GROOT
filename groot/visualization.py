import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from groot.model import _TREE_UNDEFINED


def plot_adversary(X, y, adversary, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    for box, probability in adversary.get_bounding_boxes():
        start_x, start_y = box[:, 0]
        width, height = box[:, 1] - box[:, 0]
        class_label = round(probability)
        colors = {0: "#9aadd0", 1: "#e3b6a1"}
        color = colors[class_label]
        ax.add_patch(
            patches.Rectangle(
                (start_x, start_y),
                width,
                height,
                color=color,
                fill=True,
                alpha=0.5,
                linewidth=0,
            )
        )

    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker="_")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker="+")

    x_low = np.min(X[:, 0])
    x_high = np.max(X[:, 0])
    y_low = np.min(X[:, 1])
    y_high = np.max(X[:, 1])

    x_extra = (x_high - x_low) * 0.1
    y_extra = (y_high - y_low) * 0.1

    ax.set_xlim(x_low - x_extra, x_high + x_extra)
    ax.set_ylim(y_low - y_extra, y_high + y_extra)
