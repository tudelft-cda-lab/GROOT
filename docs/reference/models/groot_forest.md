The `GrootRandomForestClassifier` class uses bootstrap aggregation and partially random feature selection to train an ensemble of `GrootTreeClassifier`s. On datasets with many features, a `GrootRandomForestClassifier` might perform better than a `GrootTreeClassifier` as it is not limited in the number of features it can use by a maximum size.

**Example:**
```python
from sklearn.datasets import make_moons
X, y = make_moons(random_state=1)

from groot.model import GrootRandomForestClassifier
forest = GrootRandomForestClassifier(attack_model=[0.1, 0.1], random_state=1)
forest.fit(X, y)
print(forest.score(X, y))
```
```
1.0
```

::: groot.model:GrootRandomForestClassifier