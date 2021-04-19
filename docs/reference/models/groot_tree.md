The main class in the GROOT repository is the `GrootTree`, this class implements GROOT as a Scikit-learn compatible classifier. That means you initialize it with all important hyperparameters, then fit it using `.fit(X, y)` and predict with `.predict(X)` or `.predict_proba(X)`. The `GrootTree` is also used within the `GrootRandomForest`.

::: groot.model:GrootTree