# GROOT: Growing Robust Trees
Growing Robust Trees (GROOT) is an algorithm that fits binary classification decision trees such that they are robust against user-specified adversarial examples. The algorithm closely resembles algorithms used for fitting normal decision trees (i.e. CART) but changes the splitting criterion and the way samples propagate when creating a split. 

This repository contains the module `groot` that implements GROOT as a Scikit-learn compatible classifier, an adversary for model evaluation and easy functions to import datasets. For documentation see https://groot.cyber-analytics.nl

## Simple example
To train and evaluate GROOT on a toy dataset against an attacker that can move samples by 0.5 in each direction one can use the following code:

```python
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

## Installation
`groot` can be installed from PyPi:
```pip install groot-trees```

To use Kantchelian's MILP attack it is required that you have [GUROBI](https://www.gurobi.com/) installed along with their python package:
```python -m pip install -i https://pypi.gurobi.com gurobipy```

## Reproducing 'Efficient Training of Robust Decision Trees Against Adversarial Examples' (article)
Code and details for reproducing the paper's experiments are given in a [separate repository](https://github.com/tudelft-cda-lab/GROOT-experiment-code).

### Implementation details
The TREANT implementation (`groot.treant.py`) is copied almost completely from the authors of TREANT at https://github.com/gtolomei/treant with small modifications to better interface with the experiments. The heuristic by Chen et al. runs in the GROOT code, only with a different score function. This score function can be enabled by setting `chen_heuristic=True` on a `GrootTreeClassifier` before calling `.fit(X, y)`. The provably robust boosting implementation comes almost completely from their code at https://github.com/max-andr/provably-robust-boosting and we use a small wrapper around their code (`groot.provably_robust_boosting.wrapper.py`) to use it. When we recorded the runtimes we turned off all parallel options in the `@jit` annotations from the code. The implementation of Chen et al. boosting can be found in their own repo https://github.com/chenhongge/RobustTrees, from whic we need to compile and copy the binary `xgboost` to the current directory. The script `fit_chen_xgboost.py` then calls this binary and uses the command line interface to fit all models.

## Important note on TREANT
To encode L-infinity norms correctly we had to modify TREANT to NOT apply rules recursively. This means we added a single `break` statement in the `treant.Attacker.__compute_attack()` method. If you are planning on using TREANT with recursive attacker rules then you should remove this statement or use TREANT's unmodified code at https://github.com/gtolomei/treant .

# Citation
If you want to cite GROOT please reference the original paper:
```
@inproceedings{vos2021efficient,
  title={Efficient Training of Robust Decision Trees Against Adversarial Examples},
  author={Vos, Dani{\"e}l and Verwer, Sicco},
  booktitle={International Conference on Machine Learning},
  pages={10586--10595},
  year={2021},
  organization={PMLR}
}
```

# Contact
For any questions or comments please create an issue or contact [me](https://github.com/daniel-vos) directly.
