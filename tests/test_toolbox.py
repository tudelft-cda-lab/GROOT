from groot.model import GrootTreeClassifier
from groot.toolbox import Model

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

def test_toolbox_model_adversarial_accuracy():
    # Load the dataset
    X, y = make_moons(noise=0.3, random_state=0)
    X_test, y_test = make_moons(noise=0.3, random_state=1)

    # Define the attacker's capabilities (L-inf norm radius 0.3)
    epsilon = 0.3
    attack_model = [epsilon, epsilon]

    # Create and fit a GROOT tree
    trees = (
        GrootTreeClassifier(
            max_depth=3,
            attack_model=attack_model,
            random_state=0
        ),
        DecisionTreeClassifier(
            max_depth=3,
            random_state=0
        ),
    )
    for tree in trees:
        tree.fit(X, y)

        # Determine the accuracy and accuracy against attackers
        accuracy = tree.score(X_test, y_test)
        if isinstance(tree, GrootTreeClassifier):
            model = Model.from_groot(tree)
        else:
            model = Model.from_sklearn(tree)
        
        adversarial_accuracy_auto = model.adversarial_accuracy(X_test, y_test, attack="auto", epsilon=0.3)
        adversarial_accuracy_tree = model.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=0.3)

        assert adversarial_accuracy_tree == adversarial_accuracy_auto
        assert accuracy >= adversarial_accuracy_tree
