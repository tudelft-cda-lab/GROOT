import os

from groot.adversary import DecisionTreeAdversary
from groot.toolbox import Model
from groot.datasets import load_all, load_epsilons_dict
from groot.model import json_tree_from_file

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import seaborn as sns

sns.set(style="whitegrid", font_scale=0.65)

trees_dir = "out/trees/"
forests_dir = "out/forests/"
output_dir = "out/"

runtime_file = output_dir + "runtimes.csv"
chen_runtime_file = output_dir + "chenboost_runtimes.csv"

sample_limit = -1
k_folds = 5
use_cached_scores_df = False
clip_runtimes = 0.01  # Clip runtimes below this value to the value

# Load and process runtimes file
runtime_df = pd.read_csv(runtime_file)
chen_runtime_df = pd.read_csv(chen_runtime_file)
runtime_df = pd.merge(runtime_df, chen_runtime_df, on=["Dataset", "Fold"])
runtime_df.drop(columns="Fold", inplace=True)

runtime_df = pd.melt(
    runtime_df, id_vars="Dataset", var_name="Model", value_name="Runtime (s)"
)
runtime_df["Runtime (s)"] = runtime_df["Runtime (s)"].clip(clip_runtimes)

dataset_order = (
    runtime_df[runtime_df["Model"] == "GROOT tree"]
    .sort_values(by="Runtime (s)")["Dataset"]
    .unique()
    .tolist()
)

# Define matching of model to color
palette = {
    "Decision tree": sns.color_palette()[0],
    "Chen et al. tree": sns.color_palette()[1],
    "TREANT tree": sns.color_palette()[2],
    "GROOT tree": sns.color_palette()[3],
    "Random forest": sns.color_palette()[4],
    "Gradient boosting": sns.color_palette()[5],
    "Chen et al. forest": sns.color_palette()[6],
    "Chen et al. boosting": sns.color_palette()[7],
    "Provably robust boosting": sns.color_palette()[8],
    "GROOT forest": sns.color_palette()[9],
}

# Plot the single decision tree runtimes
_, ax = plt.subplots(figsize=(6.75, 1.5))
g = sns.barplot(
    x="Dataset",
    y="Runtime (s)",
    hue="Model",
    hue_order=["Decision tree", "Chen et al. tree", "GROOT tree", "TREANT tree"],
    palette=palette,
    order=dataset_order,
    data=runtime_df.loc[
        runtime_df["Model"].isin(
            ["Decision tree", "Chen et al. tree", "GROOT tree", "TREANT tree"]
        )
    ],
    errwidth=1.0,
    ax=ax,
)
g.set_yscale("log")
plt.yticks([10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])
g.set_xticklabels([""] * 14)
g.tick_params(pad=-2.0)
plt.xlabel("")
plt.legend(bbox_to_anchor=(0.5, 1.0), loc=8, borderaxespad=0.5, ncol=4)
plt.tight_layout(pad=0.4)
plt.savefig(output_dir + "tree_runtimes.pdf")
plt.savefig(output_dir + "tree_runtimes.png")
plt.close()

# Plot the ensemble runtimes
_, ax = plt.subplots(figsize=(6.75, 2.35))
g = sns.barplot(
    x="Dataset",
    y="Runtime (s)",
    hue="Model",
    hue_order=[
        "Chen et al. boosting",
        "Random forest",
        "Gradient boosting",
        "GROOT forest",
        "Chen et al. forest",
        "Provably robust boosting",
    ],
    palette=palette,
    order=dataset_order,
    data=runtime_df.loc[
        runtime_df["Model"].isin(
            [
                "Chen et al. boosting",
                "Random forest",
                "Gradient boosting",
                "Chen et al. forest",
                "GROOT forest",
                "Provably robust boosting",
            ]
        )
    ],
    errwidth=1.0,
    ax=ax,
)
g.set_yscale("log")
plt.yticks([10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4])
plt.xticks(rotation=20, ha="right")
g.tick_params(pad=-2.0)
plt.xlabel("")
plt.legend(bbox_to_anchor=(0.5, 1.0), loc=8, borderaxespad=0.5, ncol=3)
plt.tight_layout(pad=0.4)
plt.savefig(output_dir + "forest_runtimes.pdf")
plt.savefig(output_dir + "forest_runtimes.png")
plt.close()

if not use_cached_scores_df:
    # Load all test sets
    epsilons = load_epsilons_dict()
    test_sets = {}
    for name, X, y in load_all():
        X = MinMaxScaler().fit_transform(X)

        k_folds_cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)
        for fold_i, (_, test_index) in enumerate(k_folds_cv.split(X, y)):
            X_test = X[test_index][:sample_limit]
            y_test = y[test_index][:sample_limit]

            test_sets[(name, fold_i)] = (X_test, y_test)

    # Evaluate the performance of all JSON decision tree files
    scores = []
    filename_to_alg = {
        "chen": "Chen et al. tree",
        "groot": "GROOT tree",
        "treant": "TREANT tree",
        "sklearn": "Decision tree",
    }
    for filename in os.listdir(trees_dir):
        if not filename.endswith(".json"):
            continue

        path = trees_dir + filename

        print(filename[:-5].split("_"))
        algorithm, data_name, _, fold = filename[:-5].split("_")
        X_test, y_test = test_sets[(data_name, int(fold))]
        epsilon = epsilons[data_name]

        # Use modified kantchelian attack to compute accuracy and adv_accuracy
        model = Model.from_json_file(path, 2)
        accuracy = model.accuracy(X_test, y_test)
        adv_accuracy = model.adversarial_accuracy(X_test, y_test, epsilon=epsilon)

        algorithm = filename_to_alg[algorithm]
        scores.append((algorithm, data_name, fold, accuracy, "accuracy"))
        scores.append(
            (algorithm, data_name, fold, adv_accuracy, "adversarial accuracy")
        )

    # Evaluate the performance of all JSON ensemble files
    filename_to_alg = {
        "chen": "Chen et al. forest",
        "groot": "GROOT forest",
        "sklearn": "Random forest",
        "boost": "Gradient boosting",
        "provable": "Provably robust boosting",
        "chenboost": "Chen et al. boosting",
    }
    for i, filename in enumerate(os.listdir(forests_dir)):
        if not filename.endswith(".json"):
            continue

        print(filename, i)

        path = forests_dir + filename

        algorithm, data_name, _, fold = filename[:-5].split("_")
        X_test, y_test = test_sets[(data_name, int(fold))]
        epsilon = epsilons[data_name]

        # Use modified Kantchelian attack to compute accuracy and adv_accuracy
        model = Model.from_json_file(path, 2)
        accuracy = model.accuracy(X_test, y_test)

        options = {"n_threads": 8}
        adv_accuracy = model.adversarial_accuracy(
            X_test, y_test, epsilon=epsilon, options=options
        )

        algorithm = filename_to_alg[algorithm]
        scores.append((algorithm, data_name, fold, accuracy, "accuracy"))
        scores.append(
            (algorithm, data_name, fold, adv_accuracy, "adversarial accuracy")
        )

    # Process and save performance results
    scores_df = pd.DataFrame(
        scores, columns=["Model", "Dataset", "Fold", "Score", "Metric"]
    )
    scores_df.to_csv(output_dir + "all_results.csv", index=False)
else:
    scores_df = pd.read_csv(output_dir + "all_results.csv")

# Plot average accuracy and adversarial accuracy scores
scores_df["Algorithm"] = scores_df["Model"].replace(
    {
        "Chen et al. tree": "Chen et al.",
        "GROOT tree": "GROOT",
        "TREANT tree": "TREANT",
        "Decision tree": "Scikit-learn",
        "Chen et al. forest": "Chen et al.",
        "Chen et al. boosting": "Chen et al.",
        "GROOT forest": "GROOT",
        "Random forest": "Scikit-learn",
    }
)

# Remove spambase results from the scores since TREANT did not run on this
scores_df = scores_df[scores_df["Dataset"] != "spambase"]

# Plot average tree scores
g = sns.catplot(
    x="Model",
    y="Score",
    col="Metric",
    kind="bar",
    palette=palette,
    data=scores_df.loc[
        scores_df["Model"].isin(
            ["Decision tree", "Chen et al. tree", "TREANT tree", "GROOT tree"]
        )
    ],
    height=4,
    aspect=0.675,
    order=["Decision tree", "Chen et al. tree", "TREANT tree", "GROOT tree"],
    dodge=False,
    legend=False,
)
g.set_xticklabels(rotation=30, ha="right")
for ax in g.axes.ravel():
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig(output_dir + "average_tree_scores.pdf")
plt.savefig(output_dir + "average_tree_scores.png")
plt.close()
scores_df.drop(["Algorithm"], axis=1)

# Plot average ensemble scores
g = sns.catplot(
    x="Model",
    y="Score",
    col="Metric",
    kind="bar",
    palette=palette,
    data=scores_df.loc[
        scores_df["Model"].isin(
            [
                "Random forest",
                "Gradient boosting",
                "Chen et al. forest",
                "Chen et al. boosting",
                "Provably robust boosting",
                "GROOT forest",
            ]
        )
    ],
    height=4,
    aspect=0.675,
    order=[
        "Random forest",
        "Gradient boosting",
        "Chen et al. boosting",
        "Chen et al. forest",
        "Provably robust boosting",
        "GROOT forest",
    ],
    dodge=False,
    legend=False,
)
g.set_xticklabels(rotation=30, ha="right")
for ax in g.axes.ravel():
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig(output_dir + "average_forest_scores.pdf")
plt.savefig(output_dir + "average_forest_scores.png")
plt.close()

# Plot average adversarial accuracy scores
g = sns.catplot(
    x="Model",
    y="Score",
    kind="bar",
    palette=palette,
    data=scores_df[scores_df["Metric"] == "adversarial accuracy"],
    height=3,
    aspect=1.3,
    order=[
        "Decision tree",
        "Chen et al. tree",
        "TREANT tree",
        "GROOT tree",
        "Gradient boosting",
        "Random forest",
        "Chen et al. boosting",
        "Chen et al. forest",
        "Provably robust boosting",
        "GROOT forest",
    ],
    dodge=False,
    legend=False,
)
g.set_xticklabels(rotation=30, ha="right")
for ax in g.axes.ravel():
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
g.ax.vlines(3.5, 0, 0.8, color="lightgray")
plt.xlabel("")
plt.ylabel("Mean adversarial accuracy")
plt.tight_layout()
plt.savefig(output_dir + "average_adversarial_accuracy.pdf")
plt.savefig(output_dir + "average_adversarial_accuracy.png")
plt.close()

# Plot average accuracy scores
g = sns.catplot(
    x="Model",
    y="Score",
    kind="bar",
    palette=palette,
    data=scores_df[scores_df["Metric"] == "accuracy"],
    height=3,
    aspect=1.3,
    order=[
        "Decision tree",
        "Chen et al. tree",
        "TREANT tree",
        "GROOT tree",
        "Gradient boosting",
        "Random forest",
        "Chen et al. boosting",
        "Chen et al. forest",
        "Provably robust boosting",
        "GROOT forest",
    ],
    dodge=False,
    legend=False,
)
g.set_xticklabels(rotation=30, ha="right")
for ax in g.axes.ravel():
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
g.ax.vlines(3.5, 0, 0.8, color="lightgray")
plt.xlabel("")
plt.ylabel("Mean accuracy")
plt.tight_layout()
plt.savefig(output_dir + "average_accuracy.pdf")
plt.savefig(output_dir + "average_accuracy.png")
plt.close()

scores_df.drop(["Algorithm"], axis=1)

# Output latex results table
datasets = []
for name, X, y in load_all():
    datasets.append((name, X.shape[0], X.shape[1]))

datasets_df = pd.DataFrame(datasets, columns=["Dataset", "Samples", "Features"])

# Output result tables of adversarial accuracy
adv_accuracy_df = scores_df[scores_df["Metric"] == "adversarial accuracy"]
grouped = adv_accuracy_df.groupby(by=["Model", "Dataset"])
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
std_scores.rename(columns=lambda x: x if x == "Dataset" else x + " std", inplace=True)

combined = mean_scores.merge(std_scores, on="Dataset").merge(datasets_df, on="Dataset")

# print summary results
# TODO: output this differently
print(
    (
        mean_scores.drop(["Dataset"], axis=1).rank(
            method="min", ascending=False, axis=1
        )
        == 1
    ).sum(axis=0)
)
print(
    mean_scores.drop(["Dataset"], axis=1)
    .rank(method="min", ascending=False, axis=1)
    .mean(axis=0)
)
print(mean_scores.drop(["Dataset"], axis=1).mean(axis=0))

# Export only decision trees
column_order = [
    "Dataset",
    "Samples",
    "Features",
    "Decision tree",
    "Decision tree std",
    "Chen et al. tree",
    "Chen et al. tree std",
    "TREANT tree",
    "TREANT tree std",
    "GROOT tree",
    "GROOT tree std",
]
combined_trees = combined[column_order]
combined_trees.to_latex(
    output_dir + "result_table_trees.tex",
    index=False,
)

# Export only ensembles
column_order = [
    "Dataset",
    "Samples",
    "Features",
    "Random forest",
    "Random forest std",
    "Gradient boosting",
    "Gradient boosting std",
    "Chen et al. forest",
    "Chen et al. forest std",
    "GROOT forest",
    "GROOT forest std",
    "Chen et al. boosting",
    "Chen et al. boosting std",
    "Provably robust boosting",
    "Provably robust boosting std",
]
combined_forests = combined[column_order]
combined_forests.to_latex(
    output_dir + "result_table_forests.tex",
    index=False,
)
