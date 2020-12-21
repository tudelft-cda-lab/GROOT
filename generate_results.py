import os

from groot.adversary import DecisionTreeAdversary
from groot.datasets import load_all
from groot.model import json_tree_from_file

from sklearn.model_selection import StratifiedKFold

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid", context="paper")

trees_dir = "trees/"
output_dir = "out/"

runtime_file = trees_dir + "runtimes.csv"

epsilon = 0.1
k_folds = 5

# Load and process runtimes file
runtime_df = pd.read_csv(runtime_file, index_col=0)

runtime_df = pd.melt(
    runtime_df, id_vars="Name", var_name="Model", value_name="Runtime (s)"
)
runtime_df = runtime_df.rename({"Name": "Dataset"}, axis="columns")
runtime_df = runtime_df.replace({"Scikit-learn": "Natural"})
runtime_df["Runtime (s)"] = runtime_df["Runtime (s)"].clip(1).round()

dataset_order = (
    runtime_df[runtime_df["Model"] == "TREANT"]
    .sort_values(by="Runtime (s)")["Dataset"]
    .tolist()
)

_, ax = plt.subplots(figsize=(5, 4))

g = sns.barplot(
    x="Dataset",
    y="Runtime (s)",
    hue="Model",
    hue_order=["Natural", "Chen et al.", "TREANT", "GROOT"],
    order=dataset_order,
    data=runtime_df,
    ax=ax,
)
g.set_yscale("log")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(output_dir + "runtimes.pdf")
plt.close()

# Load all test sets
test_sets = {}
for name, X, y in load_all():
    k_folds_cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)
    for fold_i, (train_index, test_index) in enumerate(k_folds_cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        test_sets[(name, fold_i)] = (X_test, y_test)

# Evaluate the performance of all JSON tree files
scores = []
for filename in os.listdir(trees_dir):
    if not filename.endswith(".json"):
        continue

    tree = json_tree_from_file(trees_dir + filename)

    model, data_name, _, fold = filename[:-5].split("_")
    X_test, y_test = test_sets[(data_name, int(fold))]

    adversary = DecisionTreeAdversary(
        tree,
        "json",
        attack_model=[epsilon] * X_test.shape[1],
        is_numeric=[True] * X_test.shape[1],
        n_categories=[None] * X_test.shape[1],
        one_adversarial_class=False,
    )

    accuracy = tree.score(X_test, y_test)
    adv_accuracy = adversary.adversarial_accuracy(X_test, y_test)

    scores.append((model, data_name, fold, accuracy, "accuracy"))
    scores.append((model, data_name, fold, adv_accuracy, "adversarial accuracy"))

# Process and save performance results
scores_df = pd.DataFrame(
    scores, columns=["Model", "Dataset", "Fold", "Score", "Metric"]
)
scores_df["Model"] = scores_df["Model"].replace(
    {"groot": "GROOT", "sklearn": "Natural", "treant": "TREANT", "chen": "Chen et al."}
)
scores_df.to_csv(output_dir + "json_results.csv")

# Plot average accuracy and adversarial accuracy scores
g = sns.catplot(
    x="Model",
    y="Score",
    col="Metric",
    kind="bar",
    data=scores_df,
    height=4,
    aspect=0.675,
    order=["Natural", "Chen et al.", "TREANT", "GROOT"],
)
plt.tight_layout()
plt.savefig(output_dir + "average_scores.pdf")
plt.close()

# Output latex results table
datasets = []
for name, X, y in load_all():
    datasets.append((name, X.shape[0], X.shape[1]))

datasets_df = pd.DataFrame(datasets, columns=["Dataset", "Samples", "Features"])

accuracy_df = scores_df[scores_df["Metric"] == "adversarial accuracy"]

grouped = accuracy_df.groupby(by=["Model", "Dataset"])
mean_scores = (
    grouped.mean()
    .reset_index()
    .pivot_table(index="Dataset", columns="Model", values="Score")
    .reset_index()
    .round(3)
)
std_scores = (
    grouped.std()
    .reset_index()
    .pivot_table(index="Dataset", columns="Model", values="Score")
    .reset_index()
    .round(3)
)
std_scores.columns = [
    "Dataset",
    "Chen et al. std",
    "GROOT std",
    "Natural std",
    "TREANT std",
]

combined = mean_scores.merge(std_scores, on="Dataset").merge(datasets_df, on="Dataset")
column_order = [
    "Dataset",
    "Samples",
    "Features",
    "Chen et al.",
    "Chen et al. std",
    "GROOT",
    "GROOT std",
    "Natural",
    "Natural std",
    "TREANT",
    "TREANT std",
]
combined = combined[column_order]

combined.to_latex(
    output_dir + "result_table.tex", index=False,
)
