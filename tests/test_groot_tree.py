from groot.model import GrootTree

from sklearn.datasets import make_moons
from sklearn.utils.estimator_checks import check_estimator

def assert_fit_and_score_tree(X_train, y_train, X_test, y_test, kwargs):
    tree = GrootTree(**kwargs)
    tree.fit(X_train, y_train)

    assert hasattr(tree, "root_")
    assert tree.n_samples_ == X_train.shape[0]
    assert tree.n_features_in_ == X_train.shape[1]

    score = tree.score(X_test, y_test)
    assert score >= 0.1

def test_groot_tree_sklearn_estimator():
    estimator = GrootTree()

    # Skip tests that contain non-binary classification datasets
    checks_to_skip = {
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

    for _, check in check_estimator(GrootTree(), generate_only=True):
        if check.func.__name__ in checks_to_skip:
            continue

        check(estimator)


def test_groot_tree_parameters():
    attack_models = [
        (0, 0),
        (0.2, 0.2),
        (">", ">"),
        (0.1, "<>"),
    ]

    X_train, y_train = make_moons(n_samples=100, shuffle=True, noise=0.2, random_state=1)
    X_test, y_test = make_moons(n_samples=100, shuffle=True, noise=0.2, random_state=2)

    for attack_model in attack_models:
        for chen_heuristic in (False, True):
            for one_adversarial_class in (False, True):
                assert_fit_and_score_tree(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        {
                            "attack_model": attack_model,
                            "chen_heuristic": chen_heuristic,
                            "one_adversarial_class": one_adversarial_class,
                        },
                    )
