from groot.model import GrootTreeClassifier
from groot.toolbox import Model

from sklearn.datasets import make_moons

def test_getting_started_example():
    # Load the dataset
    X, y = make_moons(noise=0.3, random_state=0)
    X_test, y_test = make_moons(noise=0.3, random_state=1)

    # Define the attacker's capabilities (L-inf norm radius 0.3)
    epsilon = 0.3
    attack_model = [epsilon, epsilon]

    # Create and fit a GROOT tree
    tree = GrootTreeClassifier(
        attack_model=attack_model,
        random_state=0
    )
    tree.fit(X, y)

    # Determine the accuracy and accuracy against attackers
    accuracy = tree.score(X_test, y_test)
    model = Model.from_groot(tree)
    adversarial_accuracy = model.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=0.3)

    print("Accuracy:", accuracy)
    print("Adversarial Accuracy:", adversarial_accuracy)
