from groot.model import (
    GrootTreeClassifier,
    GrootTreeRegressor,
    GrootRandomForestClassifier,
    GrootRandomForestRegressor,
)

from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.utils.estimator_checks import check_estimator

import numpy as np


def assert_fit_and_predict_groot(X_train, y_train, X_test, y_test, model, kwargs):
    model = clone(model)
    model.set_params(**kwargs)
    model.fit(X_train, y_train)

    assert model.n_samples_ == X_train.shape[0]
    assert model.n_features_in_ == X_train.shape[1]

    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape


def test_groot_tree_parameters():
    attack_models = [
        (0, 0),
        (0.2, 0.2),
        (">", ">"),
        (0.1, "<>"),
    ]

    X_train, y_train = make_moons(
        n_samples=100, shuffle=True, noise=0.2, random_state=1
    )
    X_test, y_test = make_moons(n_samples=100, shuffle=True, noise=0.2, random_state=2)

    models = (
        GrootTreeClassifier(),
        GrootRandomForestClassifier(n_estimators=10),
        GrootTreeRegressor(),
        GrootRandomForestRegressor(n_estimators=10),
    )
    classification_models = (True, True, False, False)

    for model, is_classification in zip(models, classification_models):
        for attack_model in attack_models:
            for chen_heuristic in (False, True):
                if is_classification:
                    for one_adversarial_class in (False, True):
                        assert_fit_and_predict_groot(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            model,
                            {
                                "attack_model": attack_model,
                                "chen_heuristic": chen_heuristic,
                                "one_adversarial_class": one_adversarial_class,
                            },
                        )
                else:
                    assert_fit_and_predict_groot(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        model,
                        {
                            "attack_model": attack_model,
                            "chen_heuristic": chen_heuristic,
                        },
                    )


def test_groot_regressors_sklearn_estimator():
    check_estimator(GrootTreeRegressor())
    check_estimator(GrootRandomForestRegressor(n_estimators=10))


def test_groot_classifiers_sklearn_estimator():
    # Skip tests that contain non-binary classification datasets for GROOT classifiers
    classifier_checks_to_skip = {
        "check_estimators_dtypes",
        "check_fit_score_takes_y",
        "check_estimators_fit_returns_self",
        "check_dtype_object",
        "check_estimators_overwrite_params",
        "check_classifier_data_not_an_array",
        "check_classifiers_classes",
        "check_classifiers_train",
        "check_supervised_y_2d",
        "check_methods_sample_order_invariance",
        "check_methods_subset_invariance",
        "check_dont_overwrite_parameters",
        "check_fit2d_predict1d",
    }
    for estimator in (GrootTreeClassifier(), GrootRandomForestClassifier(n_estimators=10)):
        for _, check in check_estimator(estimator, generate_only=True):
            if check.func.__name__ in classifier_checks_to_skip:
                continue

            check(estimator)


def test_groot_tree_known_dataset():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

    # Fit a tree of depth 1, it cannot be completely accurate
    tree = GrootTreeClassifier(max_depth=1, attack_model=[0.1, 0.1], random_state=1)
    tree.fit(X, y)

    y_pred = tree.predict(X)
    y_known_pred = np.zeros(9)
    assert np.array_equal(y_pred, y_known_pred)

    y_pred_proba = tree.predict_proba(X)
    y_known_proba = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    assert np.array_equal(y_pred_proba, y_known_proba)

    # Fit a tree of depth 2, it should be completely accurate
    tree = GrootTreeClassifier(max_depth=2, attack_model=[0.1, 0.1], random_state=1)
    tree.fit(X, y)

    y_pred = tree.predict(X)
    assert np.array_equal(y_pred, y)

    y_pred_proba = tree.predict_proba(X)
    y_known_proba = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    assert np.array_equal(y_pred_proba, y_known_proba)


def test_groot_tree_known_dataset_one_adversarial_class():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
        ]
    )
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])

    # Check that when class 1 moves we get a completely correct tree
    tree = GrootTreeClassifier(
        max_depth=1, attack_model=[">", ""], one_adversarial_class=True, random_state=1
    )
    tree.fit(X, y)
    y_pred = tree.predict(X)
    assert np.array_equal(y_pred, y)

    # Check that when two classes move we get a single leaf
    tree = GrootTreeClassifier(
        max_depth=1, attack_model=[">", ""], one_adversarial_class=False, random_state=1
    )
    tree.fit(X, y)
    assert np.array_equal(tree.root_.value, [2 / 3, 1 / 3])
