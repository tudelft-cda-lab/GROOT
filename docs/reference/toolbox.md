The `groot.toolbox` package exposes the `Model` class that allows easy loading, converting and attacking decision tree ensembles of different formats. The `Model` class can load tree ensembles from the following formats using:

- Scikit-learn: `Model.from_sklearn`
- JSON file: `Model.from_json_file`
- GROOT: `Model.from_groot`
- TREANT: `Model.from_treant`
- Provably robust boosting: `Model.from_provably_robust_boosting`

After loading you can then easily determine metrics such as accuracy and adversarial accuracy (against a given perturbation radius epsilon). It is also possible to get access to more information about adversarial robustness than just a metric. The model class has three methods for this:

- `attack_feasibility`: Compute for each sample whether or not an adversarial example exists within an radius around it.
- `attack_distances`: Compute for each sample the distance it needs to move to turn into an adversarial example.
- `adversarial_examples`: Generate adversarial examples for each input sample.

These three methods are theoretically listed in order of increasing complexity. That means that when you only need to know e.g. `attack_feasibility` and not `attack_distances` calling only the first function might be faster than calling the second and computing the 'feasibility' from that. For example for the default `'milp'` attack, `attack_feasibility` is orders of magnitude faster than `attack_distances` and `adversarial_examples`.

## Example
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from groot.toolbox import Model

X, y = load_iris(return_X_y=True)
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X, y)

model = Model.from_sklearn(tree)
print("Accuracy:", model.accuracy(X, y))

epsilon = 0.3
print("Adversarial accuracy:", model.adversarial_accuracy(X, y, epsilon=epsilon))

X_adv = model.adversarial_examples(X, y)
print("Adversarial examples:")
print(X_adv)
```

## Code reference
::: groot.toolbox