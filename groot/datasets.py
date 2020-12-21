from groot.treant import AttackerRule, Attacker

import numpy as np

from sklearn.datasets import fetch_openml, make_moons
from sklearn.datasets import load_breast_cancer as load_breast_cancer_sklearn

import json
from numbers import Number
from collections import defaultdict


def attack_model_to_treant_attacker(attack_model, is_numerical):
    attacks = []
    for i, (attack, numerical) in enumerate(zip(attack_model, is_numerical)):
        if attack == "":
            continue

        if numerical:
            if attack == ">":
                posts = [np.inf]
            elif attack == "<":
                posts = [-np.inf]
            elif isinstance(attack, Number):
                posts = [-attack, attack]
            elif isinstance(attack, tuple):
                posts = []
                if attack[0] != 0:
                    posts.append(-attack[0])
                if attack[1] != 0:
                    posts.append(attack[1])

            for post in posts:
                attacks.append(
                    AttackerRule(
                        pre_conditions=(i, (-np.inf, np.inf)),
                        post_condition=(i, post),
                        cost=1,
                        is_numerical=True,
                    )
                )
        else:
            # For each postcondition (category that can be perturbed to)
            # collect a list of categories that map to it
            preconditions = defaultdict(list)
            for key in attack:
                postconditions = attack[key]
                for postcondition in postconditions:
                    preconditions[postcondition].append(key)

            # Create the attack rules
            for post, pres in preconditions.items():
                attacks.append(
                    AttackerRule(
                        pre_conditions=(i, tuple(pres)),
                        post_condition=(i, post),
                        cost=1,
                        is_numerical=False,
                    )
                )

    for attack in attacks:
        print(attack.pre_conditions, attack.post_condition)

    # Create an attacker with enough budget to use all attack rules
    attacker = Attacker(attacks, len(attacks))
    return attacker


def epsilon_attacker(n_features, radius=0.1, max_depth=4):
    attacks = []
    for feature in range(n_features):
        attacks.append(
            AttackerRule(
                pre_conditions=(feature, (-np.inf, np.inf)),
                post_condition=(feature, radius),
                cost=1,
                is_numerical=True,
            )
        )
        attacks.append(
            AttackerRule(
                pre_conditions=(feature, (-np.inf, np.inf)),
                post_condition=(feature, -radius),
                cost=1,
                is_numerical=True,
            )
        )

    # Budget must be large enough, max_depth works for all depths <= max_depth
    return Attacker(attacks, max_depth)


def load_adult():
    # Refered to as 'census'
    data = fetch_openml("adult", version=2, return_X_y=False)
    X = data.data
    y = data.target
    is_numeric = [
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]
    y = np.where(y == "<=50K", 0, 1)

    return (
        "census",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_wine():
    # Refered to as 'wine'
    data = fetch_openml("wine_quality", version=1, return_X_y=False)
    X = data.data
    y = data.target

    is_numeric = [True, True, True, True, True, True, True, True, True, True, True]
    y = np.where(y >= 6, 0, 1)  # Larger or equal to a 6 is a 'good' wine

    return (
        "wine",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_credit():
    # Refered to as 'credit'
    data = fetch_openml("default_credit_card_p", version=1, return_X_y=False)
    X = data.data
    y = np.copy(X[:, 21]).astype(int)  # Extract target variable

    # Remove the index columns and target variable from the data
    X = np.delete(X, [0, 21, 25, 26], 1)
    for index in reversed([0, 21, 25, 26]):
        del data.feature_names[index]

    is_numeric = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]

    return (
        "credit",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_diabetes():
    # Refered to as 'diabetes'
    data = fetch_openml("diabetes", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "tested_negative", 0, 1)

    is_numeric = [True, True, True, True, True, True, True, True]

    return (
        "diabetes",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_cod_rna():
    # Refered to as 'cod-rna'
    data = fetch_openml("codrnaNorm", version=1, return_X_y=False)
    X = data.data
    y = data.target

    X = X.toarray()
    y = np.where(y == "-1", 0, 1)

    is_numeric = [True, True, True, True, True, True, True, True]

    return (
        "cod-rna",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_ionosphere():
    # Refered to as 'ionosphere'
    data = fetch_openml("ionosphere", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "b", 1, 0)

    is_numeric = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    return (
        "ionosphere",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_ijcnn():
    # Refered to as 'ijcnn'
    data = fetch_openml("ijcnn", version=1, return_X_y=False)
    X = data.data
    y = data.target

    X = X.toarray()
    y = np.where(y == -1, 0, 1)

    is_numeric = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    return (
        "ijcnn",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_spambase():
    # Refered to as 'spambase'
    data = fetch_openml("spambase", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = y.astype(int)

    is_numeric = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    return (
        "spambase",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_diabetes():
    # Refered to as 'diabetes'
    data = fetch_openml("diabetes", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "tested_positive", 1, 0)
    y = y.astype(int)

    is_numeric = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    return (
        "diabetes",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_balance_scale():
    # Refered to as 'balance-scale'
    data = fetch_openml("balance-scale", version=2, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "P", 1, 0)
    y = y.astype(int)

    is_numeric = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    return (
        "balance-scale",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_banknote_authentication():
    # Refered to as 'banknote-authentication'
    data = fetch_openml("banknote-authentication", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "2", 1, 0)

    is_numeric = [
        True,
        True,
        True,
        True,
    ]

    return (
        "banknote-authentication",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_cylinder_bands():
    # Refered to as 'cylinder-bands'
    data = fetch_openml("cylinder-bands", version=2, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "band", 1, 0)
    y = y.astype(int)

    # Remove rows with missing values
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]

    is_numeric = [True] * 37

    return (
        "cylinder-bands",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_blood_transfusion():
    # Refered to as 'blood-transfusion'
    data = fetch_openml("blood-transfusion-service-center", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "2", 1, 0)
    y = y.astype(int)

    is_numeric = [True] * 4

    return (
        "blood-transfusion",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_climate_model_simulation():
    # Refered to as 'climate-model-simulation'
    data = fetch_openml("climate-model-simulation-crashes", version=4, return_X_y=False)
    X = data.data
    y = data.target

    y = y.astype(int)

    is_numeric = [True] * 18

    return (
        "climate-model-simulation",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_sonar():
    # Refered to as 'sonar'
    data = fetch_openml("sonar", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "Mine", 1, 0)
    y = y.astype(int)

    is_numeric = [True] * 60

    return (
        "sonar",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_haberman():
    # Refered to as 'haberman'
    data = fetch_openml("haberman", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "2", 1, 0)
    y = y.astype(int)

    is_numeric = [True] * 3

    return (
        "haberman",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_parkinsons():
    # Refered to as 'parkinsons'
    data = fetch_openml("parkinsons", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "2", 1, 0)
    y = y.astype(int)

    is_numeric = [True] * 22

    return (
        "parkinsons",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_planning_relax():
    # Refered to as 'planning-relax'
    data = fetch_openml("planning-relax", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = np.where(y == "2", 1, 0)
    y = y.astype(int)

    is_numeric = [True] * 12

    return (
        "planning-relax",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_SPECTF():
    # Refered to as 'SPECTF'
    data = fetch_openml("SPECTF", version=2, return_X_y=False)
    X = data.data
    y = data.target

    y = y.astype(int)

    is_numeric = [True] * 44

    return (
        "SPECTF",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_moons():
    X, y = make_moons(n_samples=100, noise=0.3, random_state=0)

    return "moons", X, y, [True, True], []


def load_breast_cancer():
    X, y = load_breast_cancer_sklearn(return_X_y=True)

    y = 1 - y  # Flip labels so 1=malicious

    return "breast-cancer", X, y, [True] * 30, []


def load_mnist():
    # Refered to as 'MNIST'
    data = fetch_openml("mnist_784", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = y.astype(int)

    is_numeric = [True] * 784

    return (
        "MNIST",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_fashion_mnist():
    # Refered to as 'MNIST'
    data = fetch_openml("Fashion-MNIST", version=1, return_X_y=False)
    X = data.data
    y = data.target

    y = y.astype(int)

    is_numeric = [True] * 784

    return (
        "Fashion-MNIST",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_all():
    return [
        load_haberman()[:3],
        load_blood_transfusion()[:3],
        load_planning_relax()[:3],
        load_cylinder_bands()[:3],
        load_SPECTF()[:3],
        load_diabetes()[:3],
        load_parkinsons()[:3],
        load_ionosphere()[:3],
        load_sonar()[:3],
        load_climate_model_simulation()[:3],
        load_banknote_authentication()[:3],
        load_breast_cancer()[:3],
        load_wine()[:3],
        load_spambase()[:3],
    ]
