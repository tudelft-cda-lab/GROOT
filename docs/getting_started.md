The GROOT repository contains several useful modules for fitting and scoring decision trees and ensembles against adversarial examples:

- Implementations of robust decision tree learning algorithms:
    * [GROOT decision tree](../reference/models/groot_tree/)[^1]
    * [GROOT random forest](../reference/models/groot_forest/)[^1]
    * [TREANT decision tree](../reference/models/treant/)[^2]
    * [Provably robust boosting](../reference/models/boosting/)[^3]
- [Adversary](../reference/adversary/) for attacking and scoring decision trees
- [Kantchelian's MILP attack](../reference/verification/) for attacking and scoring trees / ensembles[^4]
- Easy functions to import [datasets](../reference/datasets/)
- [Utilities](../reference/util) for exporting scikit-learn models
- 2D decision tree [visualizer](../reference/visualization/)

For an introduction to decision trees and adversarial examples see [this blogpost](https://cyber-analytics.nl/blogposts/2021-03-15-trees-adversarial-examples/).

## Installing
GROOT can be directly install from PyPi:
```pip install groot-trees```

To use Kantchelian's MILP attack it is required that you have [GUROBI](https://www.gurobi.com/) installed along with their python package:
```python -m pip install -i https://pypi.gurobi.com gurobipy```

## Toolbox
The [GROOT toolbox](../reference/toolbox/) contains the useful `Model` class which can take models of different types (e.g. Scikit-learn, GROOT, TREANT) and turn these into a general JSON representation. It can also be used to easily evaluate the performance of these models against adversarial attacks.

## GROOT Example
Below is a very simple example demonstrating how to train and score a GROOT tree on a toy dataset. We start by creating a 2D dataset using scikit-learn then split it into a train and test set.

``` python
from groot.model import GrootTreeClassifier
from groot.toolbox import Model

from sklearn.datasets import make_moons

# Load the dataset
X, y = make_moons(noise=0.3, random_state=0)
X_test, y_test = make_moons(noise=0.3, random_state=1)
```

To encode an attacker that can increase/decrease a sample by 0.3 for both features we set the attack_model to `[0.3, 0.3]`.

``` python
# Define the attacker's capabilities (L-inf norm radius 0.3)
epsilon = 0.3
attack_model = [epsilon, epsilon]
```

We train the `GrootTreeClassifier` using `.fit()` just like other scikit-learn models.

``` python
# Create and fit a GROOT tree
tree = GrootTreeClassifier(
    attack_model=attack_model,
    random_state=0
)
tree.fit(X, y)
```

Lastly, we test the performance. To get the regular accuracy we use `.score()` and to determine adversarial accuracy we can use the `Model` class that exposes some useful functionality.

``` python
# Determine the accuracy and accuracy against attackers
accuracy = tree.score(X_test, y_test)
model = Model.from_groot(tree)
adversarial_accuracy = model.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=0.3)

print("Accuracy:", accuracy)
print("Adversarial Accuracy:", adversarial_accuracy)
```

See the [groot.toolbox](../reference/toolbox) for more information on how to use the `Model` class.

### Putting it all together
The full script is given below.

``` python
from groot.model import GrootTreeClassifier
from groot.toolbox import Model

from sklearn.datasets import make_moons

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
```

Which evaluates to:

```
Accuracy: 0.83
Adversarial Accuracy: 0.65
```

[^1]: Vos, Daniël, and Sicco Verwer. "Efficient Training of Robust Decision Trees Against Adversarial Examples." arXiv preprint arXiv:2012.10438 (2020).
[^2]: Calzavara, Stefano, et al. "Treant: training evasion-aware decision trees." Data Mining and Knowledge Discovery 34.5 (2020): 1390-1420.
[^3]: Andriushchenko, Maksym, and Matthias Hein. "Provably robust boosted decision stumps and trees against adversarial attacks." arXiv preprint arXiv:1906.03526 (2019).
[^4]: Kantchelian, Alex, J. Doug Tygar, and Anthony Joseph. "Evasion and hardening of tree ensemble classifiers." International Conference on Machine Learning. PMLR, 2016.