The `GrootRandomForest` class uses bootstrap aggregation and partially random feature selection to train an ensemble of `GrootTree`s. On datasets with many features, a `GrootRandomForest` might perform better than a `GrootTree` as it is not limited in the number of features it can use by a maximum size.

::: groot.model:GrootRandomForest