# 05_classification.py — decision tree, naive Bayes, KNN, MLP and logistic on the structured features (Lab 4)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

plt.rcParams.update({"font.size": 11})   # figures are placed at 4-6.2 in wide in the report

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

SEED = 7
MIN_SCORE = 0.90          # fixed in advance, the same for 05, 07 and 08: a method votes when score >= 0.90
CLASSES = ["non_human", "human"]


# 1. Load
# the split written by 02 (stratified on is_human, 60/20/20, SEED = 7), plus every row for the product
train = pd.read_csv(PROC / "twitter_train.csv")
val = pd.read_csv(PROC / "twitter_validation.csv")
test = pd.read_csv(PROC / "twitter_test.csv")
df = pd.read_csv(PROC / "twitter_full.csv")
print("train:", train.shape, "validation:", val.shape, "test:", test.shape, "all rows:", df.shape)


# 2. Features
# gender:confidence and label_conflict describe the label, so they are not features
NOT_FEATURES = {"_unit_id", "is_human", "label", "name", "gender", "gender:confidence", "label_conflict"}
FEATURES = [c for c in train.columns if c not in NOT_FEATURES]
print("features:", len(FEATURES))

# KNN is given a NumPy array; a DataFrame raised an attribute error with these library versions
X_train, y_train = train[FEATURES].to_numpy(float), train["is_human"].astype(int).to_numpy()
X_val, y_val = val[FEATURES].to_numpy(float), val["is_human"].astype(int).to_numpy()
X_test, y_test = test[FEATURES].to_numpy(float), test["is_human"].astype(int).to_numpy()
baseline = y_train.mean()
print("share human (train): %.3f (the accuracy of always guessing human)" % baseline)


# 3. Tree depth and number of neighbours
# 5-fold CV on the training split only; the depth and k with the best mean are used below
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
MAX_DEPTH = 6             # best of the depth sweep
K_NEIGHBOURS = 25         # best of the k sweep


def sweep(param, values, make, chosen, file_name, title):
    rows = []
    for v in values:
        scores = cross_val_score(make(v), X_train, y_train, cv=cv, scoring="accuracy")
        rows.append({param: "None" if v is None else v, "cv_accuracy": scores.mean(), "cv_std": scores.std()})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / ("classification_%s_sweep.csv" % file_name), index=False)
    print("\n%s sweep, 5-fold CV accuracy (train):\n%s" % (param, table.round(4).to_string(index=False)))
    # evenly spaced points, so max_depth=None can sit at the right-hand end
    x = np.arange(len(values))
    best = values.index(chosen)
    plt.figure(figsize=(4.0, 3.0))
    plt.errorbar(x, table["cv_accuracy"], yerr=table["cv_std"], color="tab:gray", marker="o", capsize=3)
    plt.plot(x[best], table["cv_accuracy"][best], "o", color="tab:red", markersize=9,
             label="used: %s = %s" % (param, chosen))
    plt.xticks(x, table[param], fontsize=9 if len(values) > 8 else None)
    plt.xlabel(param)
    plt.ylabel("CV accuracy (train)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT / ("fig_classification_%s_sweep.png" % file_name), dpi=150, bbox_inches="tight")
    plt.show()


sweep("max_depth", [2, 4, 6, 8, 10, 12, 15, 20, None],
      lambda d: DecisionTreeClassifier(max_depth=d, random_state=SEED), MAX_DEPTH, "depth", "decision tree depth")
sweep("n_neighbors", [5, 11, 15, 25, 35, 51, 75, 101],
      lambda k: KNeighborsClassifier(n_neighbors=k), K_NEIGHBOURS, "k", "KNN number of neighbours")


# 4. Model comparison
# 5-fold CV on the training split, confusion matrices on the validation split
models = {
    "logistic":      LogisticRegression(max_iter=2000),
    "decision_tree": DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=SEED),
    "naive_bayes":   GaussianNB(),
    "knn":           KNeighborsClassifier(n_neighbors=K_NEIGHBOURS),
    "mlp":           MLPClassifier(random_state=SEED, early_stopping=True, max_iter=500),
}

cv_scores, val_pred, rows = {}, {}, []
for name, clf in models.items():
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    precision, recall, _, _ = precision_recall_fscore_support(y_val, pred, labels=[0, 1], zero_division=0)
    cv_scores[name], val_pred[name] = scores, pred
    rows.append({"model": name, "cv_accuracy": scores.mean(), "cv_std": scores.std(),
                 "validation_accuracy": accuracy_score(y_val, pred),
                 "non_human_precision": precision[0], "non_human_recall": recall[0],
                 "human_precision": precision[1], "human_recall": recall[1]})
    print("%-14s CV accuracy (train) %.4f +/- %.4f | validation accuracy %.4f"
          % (name, scores.mean(), scores.std(), rows[-1]["validation_accuracy"]))

comparison = pd.DataFrame(rows).sort_values("validation_accuracy", ascending=False)
print(comparison.round(4).to_string(index=False))

# y-axis starts near the baseline so the gaps between models are visible
fig, ax = plt.subplots(figsize=(4.8, 3.6))
x = np.arange(len(comparison))
ax.bar(x - 0.2, comparison["cv_accuracy"], 0.4, yerr=comparison["cv_std"], capsize=3, color="tab:gray",
       label="5-fold CV (train)")
ax.bar(x + 0.2, comparison["validation_accuracy"], 0.4, color="tab:green", label="validation")
ax.axhline(baseline, color="red", linestyle="--", label="always guess human")
ax.set_xticks(x, [m.replace("_", "\n") for m in comparison["model"]])
ax.set_ylim(baseline - 0.02, comparison[["cv_accuracy", "validation_accuracy"]].max().max() + 0.03)
ax.set_xlabel("model")
ax.set_ylabel("accuracy")
ax.set_title("classification: model comparison")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2)
plt.tight_layout()
plt.savefig(OUT / "fig_classification_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# sized to be read at 6.2 in wide: every number at the base font size (11 pt)
fig, axes = plt.subplots(2, 3, figsize=(6.2, 5.6))
axes[1, 2].axis("off")   # five models on a 3 + 2 grid
for k, (ax, name) in enumerate(zip(axes.ravel(), comparison["model"])):
    cm = confusion_matrix(y_val, val_pred[name])          # rows recorded, columns predicted
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["non-\nhuman", "human"])       # two lines, so the labels do not touch
    ax.set_yticks([0, 1], ["non-\nhuman", "human"]) if k % 3 == 0 else ax.set_yticks([])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    if k % 3 == 0:
        ax.set_ylabel("recorded")
    ax.set_title(name.replace("_", " "))
fig.suptitle("confusion matrices on the validation split")
plt.tight_layout()
plt.savefig(OUT / "fig_classification_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()


# 5. t-test between the two best models
# Lab 4: is the gap in CV accuracy real, or noise?
# logistic regression votes in 07, so the test compares the two best models eligible for this vote
eligible = comparison.loc[comparison["model"] != "logistic"].sort_values("cv_accuracy", ascending=False)
top_two = eligible["model"].iloc[:2].tolist()
t, p = ttest_ind(cv_scores[top_two[0]], cv_scores[top_two[1]])
print("\nt-test on CV accuracy, %s vs %s: t %.4f, p %.4f" % (top_two[0], top_two[1], t, p))
pd.DataFrame([{"model_a": top_two[0], "model_b": top_two[1], "t": t, "p": p,
               "significant": p < 0.05}]).to_csv(OUT / "classification_ttest.csv", index=False)


# 6. Choose on validation, test once
# logistic regression votes through 07_logistic_regression.py, so the vote here goes to the best other model
best_name = comparison.loc[comparison["model"] != "logistic", "model"].iloc[0]
best_model = models[best_name]                            # already fitted on the training split
print("\nselected on validation:", best_name)

pred_test = best_model.predict(X_test)
test_acc = accuracy_score(y_test, pred_test)
precision, recall, _, _ = precision_recall_fscore_support(y_test, pred_test, labels=[0, 1])
print("test accuracy: %.4f" % test_acc)
print("confusion matrix (test) [rows recorded, cols predicted, non_human first]:\n",
      confusion_matrix(y_test, pred_test))
print(pd.DataFrame({"precision": precision, "recall": recall}, index=CLASSES).round(4))

comparison["test_accuracy"] = np.where(comparison["model"] == best_name, test_acc, np.nan)
comparison.to_csv(OUT / "classification_model_comparison.csv", index=False)


# 7. Predict every profile and nominate
split_of = pd.concat([pd.Series(name, index=part["_unit_id"])
                      for name, part in [("train", train), ("validation", val), ("test", test)]])
prob = best_model.predict_proba(df[FEATURES].to_numpy(float))[:, 1]
score = np.maximum(prob, 1 - prob).round(3)             # rounded first, so the file obeys votes == (score >= 0.90)
predictions = pd.DataFrame({
    "_unit_id": df["_unit_id"],
    "says": np.where(prob >= 0.5, "human", "non_human"),
    "score": score,
    "recorded": df["label"],
    "votes": (score >= MIN_SCORE).astype(int),
    "split": df["_unit_id"].map(split_of).fillna("unknown"),
})
predictions.to_csv(OUT / "classification_predictions.csv", index=False)
print("\nclassification_predictions.csv:", len(predictions), "rows")
print(predictions["split"].value_counts().to_string())

labelled = predictions["recorded"] != "unknown"
disagree = labelled & (predictions["says"] != predictions["recorded"])
# printed for information only; the cut-off above was fixed before looking at it
sweep = pd.DataFrame([{"min_score": cut, "profiles": int((disagree & (score >= cut)).sum())}
                      for cut in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]])
sweep.to_csv(OUT / "classification_threshold_sweep.csv", index=False)
print(sweep.to_string(index=False))

hit = disagree & (predictions["votes"] == 1)
result = predictions.loc[hit, ["_unit_id", "says", "score", "recorded", "split"]].join(
    df.loc[hit, ["gender", "name", "gender:confidence"]]).sort_values("score", ascending=False)
result.to_csv(OUT / "classification_flagged.csv", index=False)

print("\nprofiles nominated (classification_flagged.csv):", len(result))
print(pd.crosstab(result["recorded"], result["says"]))
# training rows are judged by a model that saw their label, so flags there are conservative
print("nominations per split:\n" + result["split"].value_counts().to_string())
print(result.head(10)[["name", "gender", "recorded", "says", "score"]])


# 8. Crowd-confidence check
# the model never used gender:confidence, so it is an after-the-fact check
conf = df.loc[labelled, "gender:confidence"]
unsure_by_label = (conf < 1).groupby(df.loc[labelled, "label"]).mean()
nominated = (result["gender:confidence"] < 1).mean()
same_mix = (result["recorded"].value_counts(normalize=True) * unsure_by_label).sum()
everyone = (conf < 1).mean()
print("\nshare below full crowd confidence - nominated %.3f | all labelled with the same recorded-label mix %.3f"
      " | all labelled %.3f" % (nominated, same_mix, everyone))

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
groups = [result["gender:confidence"], conf]
axes[0].hist(groups, bins=20, weights=[np.full(len(g), 100 / len(g)) for g in groups],
             color=["tab:red", "grey"], label=["nominated by classification", "all labelled profiles"])
axes[0].set_xlabel("gender:confidence")
axes[0].set_ylabel("% of the group")
axes[0].set_title("crowd confidence, nominated vs all")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22))
# middle bar: the same recorded-label mix as the nominations, split by recorded label
mix = result["recorded"].value_counts(normalize=True)
axes[1].bar(0, nominated, color="tab:red")
bottom = 0
for name, colour in [("human", "tab:blue"), ("non_human", "tab:orange")]:
    part = mix.get(name, 0) * unsure_by_label[name]
    axes[1].bar(1, part, bottom=bottom, color=colour, label="recorded " + name)
    bottom += part
axes[1].bar(2, everyone, color="grey")
for x, value in enumerate([nominated, same_mix, everyone]):
    axes[1].text(x, value + 0.01, "%.3f" % value, ha="center")
axes[1].set_ylim(0, max(nominated, same_mix) * 1.2)
axes[1].set_xticks([0, 1, 2], ["nominated", "same label\nmix", "all\nlabelled"])
axes[1].set_ylabel("share below full confidence")
axes[1].set_title("below full crowd confidence")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3))
plt.tight_layout()
plt.savefig(OUT / "fig_classification_crowd_confidence.png", dpi=150, bbox_inches="tight")
plt.show()
