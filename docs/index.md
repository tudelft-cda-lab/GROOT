# GROOT: Growing Robust Trees

Welcome to the GROOT documentation! :evergreen_tree:

Growing Robust Trees (GROOT) is a fast algorithm that fits binary classification decision trees such that they are robust against user-specified adversarial examples. It can be used to fit interpretable robust trees or stronger robust random forests.

GROOT closely resembles algorithms used for fitting normal decision trees (i.e. CART) but changes the splitting criterion and the way samples propagate when creating a split. It is based on the algorithm by [Chen et al. (2019)](https://arxiv.org/abs/1902.10660) but speeds it up tremendously by computing the adversarial Gini impurity in constant time.

[Get started](getting_started){ .md-button .md-button--primary}
{.center}
