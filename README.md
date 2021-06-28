# GROOT: Growing Robust Trees
Growing Robust Trees (GROOT) is an algorithm that fits binary classification decision trees such that they are robust against user-specified adversarial examples. The algorithm closely resembles algorithms used for fitting normal decision trees (i.e. CART) but changes the splitting criterion and the way samples propagate when creating a split. 

This repository contains the module `groot` that implements GROOT as a Scikit-learn compatible classifier, an adversary for model evaluation and easy functions to import datasets. For documentation see https://groot.cyber-analytics.nl

## Simple example
To train and evaluate GROOT on a toy dataset against an attacker that can move samples by 0.5 in each direction one can use the following code:

```python
from groot.adversary import DecisionTreeAdversary
from groot.model import GrootTree

from sklearn.datasets import make_moons

X, y = make_moons(noise=0.3, random_state=0)
X_test, y_test = make_moons(noise=0.3, random_state=1)

attack_model = [0.5, 0.5]
is_numerical = [True, True]
tree = GrootTree(attack_model=attack_model, is_numerical=is_numerical, random_state=0)

tree.fit(X, y)
accuracy = tree.score(X_test, y_test)
adversarial_accuracy = DecisionTreeAdversary(tree, "groot").adversarial_accuracy(X_test, y_test)

print("Accuracy:", accuracy)
print("Adversarial Accuracy:", adversarial_accuracy)
```

## Installation
`groot` can be installed from PyPi:
```pip install groot-trees```

To use Kantchelian's MILP attack it is required that you have [GUROBI](https://www.gurobi.com/) installed along with their python package:
```python -m pip install -i https://pypi.gurobi.com gurobipy```

### Specific dependency versions
To reproduce our experiments with exact package versions you can clone the repository and run:
```pip install -r requirements.txt```

We recommend using virtual environments.

## Reproducing 'Efficient Training of Robust Decision Trees Against Adversarial Examples' (article)
To reproduce the results from the paper we provide `generate_k_fold_results.py`, a script that takes the trained models (from JSON format) and generates tables and figures. The resulting figures generate under `/out/`.

To not only generate the results but to also retrain all models we include the scripts `train_kfold_models.py` and `fit_chen_xgboost.py`. The first script runs the algorithms in parallel for each dataset then outputs to `/out/trees/` and `/out/forests/`. Warning: the script can take a long time to run (about a day given 16 cores). The second script train specifically the Chen et al. boosting ensembles. `/out/results.zip` contains all results from when we ran the scripts.

To experiment on image datasets we have a script `image_experiments.py` that fits and output the results. In this script, one can change the `dataset` variable to 'mnist' or 'fmnist' to switch between the two.

The scripts `summarize_datasets.py` and `visualize_threat_models.py` output some figures we used in the text. 

### Implementation details
The TREANT implementation (`groot.treant.py`) is copied almost completely from the authors of TREANT at https://github.com/gtolomei/treant with small modifications to better interface with the experiments. The heuristic by Chen et al. runs in the GROOT code, only with a different score function. This score function can be enabled by setting `chen_heuristic=True` on a `GrootTree` before calling `.fit(X, y)`. The provably robust boosting implementation comes almost completely from their code at https://github.com/max-andr/provably-robust-boosting and we use a small wrapper around their code (`groot.provably_robust_boosting.wrapper.py`) to use it. When we recorded the runtimes we turned off all parallel options in the `@jit` annotations from the code. The implementation of Chen et al. boosting can be found in their own repo https://github.com/chenhongge/RobustTrees, from whic we need to compile and copy the binary `xgboost` to the current directory. The script `fit_chen_xgboost.py` then calls this binary and uses the command line interface to fit all models.

## Important note on TREANT
To encode L-infinity norms correctly we had to modify TREANT to NOT apply rules recursively. This means we added a single `break` statement in the `treant.Attacker.__compute_attack()` method. If you are planning on using TREANT with recursive attacker rules then you should remove this statement or use TREANT's unmodified code at https://github.com/gtolomei/treant .

# Contact
For any questions or comments please create an issue or contact [me](https://github.com/daniel-vos) directly.
