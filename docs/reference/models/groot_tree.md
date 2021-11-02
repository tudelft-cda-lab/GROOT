The main class in the GROOT repository is the `GrootTreeClassifier`, this class implements GROOT as a Scikit-learn compatible classifier. That means you initialize it with all important hyperparameters, then fit it using `.fit(X, y)` and predict with `.predict(X)` or `.predict_proba(X)`. The `GrootTreeClassifier` is also used within the `GrootRandomForestClassifier`.

**Example:**
```python
from sklearn.datasets import make_moons
X, y = make_moons(random_state=1)

from groot.model import GrootTreeClassifier
tree = GrootTreeClassifier(max_depth=3, attack_model=[0.1, 0.1])
tree.fit(X, y)
print(tree.score(X, y))
```
```
0.9
```

::: groot.model:GrootTreeClassifier