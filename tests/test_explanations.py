from groot.toolbox import Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np

import pytest

import importlib

@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="Gurobi not installed")
def test_counterfactual_explanations():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32)
    y = np.array([0, 0, 0, 1])

    for classifier in (
        DecisionTreeClassifier(random_state=0),
        RandomForestClassifier(n_estimators=10, random_state=0),
    ):
        classifier.fit(X, y)

        model = Model.from_sklearn(classifier)

        counterfactuals = model.counterfactual_explanations(X, y)
        assert len(counterfactuals) == len(X)
        assert np.allclose(np.linalg.norm(counterfactuals - X, ord=0, axis=1),  np.array([2, 1, 1, 1])), np.linalg.norm(counterfactuals - X, ord=0, axis=1)

        counterfactuals = model.counterfactual_explanations(X)
        assert len(counterfactuals) == len(X)
        assert np.allclose(np.linalg.norm(counterfactuals - X, ord=0, axis=1),  np.array([2, 1, 1, 1])), np.linalg.norm(counterfactuals - X, ord=0, axis=1)

        natural_language_explanations = model.natural_language_explanations(
            X,
            y,
            feature_names=["x", "y"],
            class_names=["A", "B"],
        )
        assert len(natural_language_explanations) == len(X)
        assert "x was changed from 0.000 to" in natural_language_explanations[0]
        assert "y was changed from 0.000 to" in natural_language_explanations[0]

        natural_language_explanations = model.natural_language_explanations(
            X,
            feature_names=["x", "y"],
            class_names=["A", "B"],
        )
        assert len(natural_language_explanations) == len(X)
        assert "x was changed from 0.000 to" in natural_language_explanations[0]
        assert "y was changed from 0.000 to" in natural_language_explanations[0]
