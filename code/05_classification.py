# CSCI446/946 Big Data Analytics - Assignment 2
# 05 - Classification

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
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

SEED = 7
MIN_SCORE = 0.90          # chosen from the sweep below, targets a few hundred profiles


# Task 1: load the processed data and build the feature matrix
df = pd.read_csv(PROC / "twitter_full.csv")
labelled = df["is_human"].notna()

NOT_FEATURES = {"_unit_id", "is_human", "label", "name", "gender", "gender:confidence",
                 "label_conflict", "description", "text", "desc_clean", "text_clean",
                 "link_color", "sidebar_color", "fav_number", "tweet_count",
                 "tweets_per_day", "favs_per_day"}
FEATURES = [c for c in df.columns if c not in NOT_FEATURES]
print("features:", len(FEATURES))

# KNeighborsClassifier chokes on a DataFrame in this environment, so use arrays
X = df.loc[labelled, FEATURES].to_numpy(float)
y = df.loc[labelled, "is_human"].astype(int).to_numpy()
recorded = np.where(y == 1, "human", "non_human")   # the target is is_human, not the label column
print("labelled rows:", len(X), "| human rate: %.3f" % y.mean())


# Task 2: five algorithms, same CV split for all so the comparison is fair
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
models = [
    ("logistic",      LogisticRegression(max_iter=2000, class_weight="balanced")),
    ("decision_tree", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
    ("naive_bayes",   GaussianNB()),
    ("knn",           KNeighborsClassifier(n_neighbors=25)),
    ("mlp",           MLPClassifier(random_state=SEED, early_stopping=True, max_iter=500)),
]

scores_all, prob_all, rows = {}, {}, []
for name, clf in models:
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")            # credential
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]  # product: P(human)
    pred = (prob >= 0.5).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    scores_all[name] = scores
    prob_all[name] = prob
    rows.append({"model": name, "accuracy": scores.mean(), "std": scores.std(),
                 "macro_f1": f1_score(y, pred, average="macro"),
                 "non_human_precision": precision[0], "non_human_recall": recall[0],
                 "human_precision": precision[1], "human_recall": recall[1]})
    print("%-14s acc %.4f (+/- %.4f)" % (name, scores.mean(), scores.std()))

comparison = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
comparison.to_csv(OUT / "classification_model_comparison.csv", index=False)
print(comparison.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(comparison["model"], comparison["accuracy"], yerr=comparison["std"])
ax.axhline(y.mean(), color="red", linestyle="--", label="always guess human")
ax.set_ylabel("5-fold CV accuracy")
ax.set_title("classification: model comparison")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_classification_model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()

# one confusion matrix per algorithm, side by side
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, (name, _) in zip(axes, models):
    pred = (prob_all[name] >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["non_human", "human"])
    ax.set_yticks([0, 1], ["non_human", "human"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(name)
plt.tight_layout()
plt.savefig(OUT / "fig_classification_confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.show()


# Task 3: is the gap between algorithms real, or just noise?
# same cv split for every model, so these are paired samples, not independent -
# ttest_ind (as Lab 4 uses) ignores that pairing; report p-values with that caveat
names = list(scores_all)
ttest_rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        t, p = ttest_ind(scores_all[names[i]], scores_all[names[j]])
        ttest_rows.append({"model_a": names[i], "model_b": names[j], "t": t, "p": p,
                            "significant": p < 0.05})
ttest_table = pd.DataFrame(ttest_rows)
ttest_table.to_csv(OUT / "classification_ttest.csv", index=False)
print("\nt-test on CV accuracy (paired folds, see comment above):")
print(ttest_table.round(4).to_string(index=False))


# Task 4: nominate profiles whose predicted class contradicts the recorded label
# logistic regression votes through 07_logistic_regression.py,
# so the classification vote goes to the best Lab 4 model
best_name = comparison.loc[comparison["model"] != "logistic", "model"].iloc[0]
prob = prob_all[best_name]
says = np.where(prob >= 0.5, "human", "non_human")
score = np.maximum(prob, 1 - prob)
disagree = says != recorded
print("\nbest model:", best_name)

conf_all = df.loc[labelled, "gender:confidence"].to_numpy()
print("threshold sweep (crowd was unsure about %.3f of all labelled profiles)"
      % (conf_all < 1).mean())
sweep = []
for cut in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    hit = disagree & (score >= cut)
    sweep.append({"min_score": cut, "profiles": int(hit.sum()),
                  "crowd_unsure": (conf_all[hit] < 1).mean() if hit.sum() else np.nan})
sweep_table = pd.DataFrame(sweep)
sweep_table.to_csv(OUT / "classification_threshold_sweep.csv", index=False)
print(sweep_table.round(3).to_string(index=False))

hit = disagree & (score >= MIN_SCORE)
lab_rows = df.loc[labelled]

# output contract: _unit_id, says, score
result = pd.DataFrame({
    "_unit_id": lab_rows["_unit_id"].to_numpy()[hit],
    "says": says[hit],
    "score": score[hit].round(3),
    "recorded": recorded[hit],
    "gender": lab_rows["gender"].to_numpy()[hit],
    "name": lab_rows["name"].to_numpy()[hit],
    "gender:confidence": conf_all[hit],
}).sort_values("score", ascending=False)
result.to_csv(OUT / "classification_flagged.csv", index=False)

print("\nprofiles nominated:", len(result))
print(pd.crosstab(result["recorded"], result["says"]))
print(result.head(10)[["name", "gender", "recorded", "says", "score"]])

# check the nominations against crowd confidence, which the model never used
plt.figure(figsize=(8, 4))
groups = [result["gender:confidence"], pd.Series(conf_all)]
plt.hist(groups, bins=20, weights=[np.full(len(g), 100 / len(g)) for g in groups],
         label=["nominated by classification", "all labelled profiles"])
plt.xlabel("gender:confidence")
plt.ylabel("% of the group")
plt.title("crowd confidence, nominated vs all")
plt.legend()
plt.savefig(OUT / "fig_classification_crowd_confidence.png", dpi=120, bbox_inches="tight")
plt.show()


# Task 5: predict the 1,037 unknown profiles, and write one file covering every row
unknown = ~labelled
X_unknown = df.loc[unknown, FEATURES].to_numpy(float)
best_model = dict(models)[best_name]
best_model.fit(X, y)                                    # refit on all labelled rows
unknown_prob = best_model.predict_proba(X_unknown)[:, 1]
unknown_says = np.where(unknown_prob >= 0.5, "human", "non_human")
unknown_score = np.maximum(unknown_prob, 1 - unknown_prob)
print("\npredicted a class for", unknown.sum(), "unknown profiles")

all_predictions = pd.concat([
    pd.DataFrame({"_unit_id": lab_rows["_unit_id"].to_numpy(), "says": says,
                  "score": score.round(3), "recorded": recorded, "disagrees": disagree}),
    pd.DataFrame({"_unit_id": df.loc[unknown, "_unit_id"].to_numpy(), "says": unknown_says,
                  "score": unknown_score.round(3), "recorded": "unknown", "disagrees": np.nan}),
], ignore_index=True)
all_predictions.to_csv(OUT / "classification_predictions.csv", index=False)
print("classification_predictions.csv:", len(all_predictions), "rows (labelled OOF + unknown)")


# Task 6: worth testing - does a cleaner teacher (confidence == 1.0 only) sharpen the detector?
# fair comparison: same 17,678 rows scored either way, only the training data changes
reliable = df.loc[labelled, "gender:confidence"].ge(1.0).to_numpy()
clf = dict(models)[best_name]
prob_reliable = np.empty(len(y))
prob_reliable[reliable] = cross_val_predict(
    clf, X[reliable], y[reliable], cv=cv, method="predict_proba")[:, 1]   # OOF within the reliable rows
clf.fit(X[reliable], y[reliable])
prob_reliable[~reliable] = clf.predict_proba(X[~reliable])[:, 1]          # rows this fit never saw

acc_baseline = ((prob_all[best_name] >= 0.5).astype(int) == y).mean()
acc_reliable = ((prob_reliable >= 0.5).astype(int) == y).mean()
print("\n%s accuracy on all 17,678 labelled rows - trained on everyone %.4f, "
      "trained on confidence==1.0 only (%d rows) %.4f"
      % (best_name, acc_baseline, reliable.sum(), acc_reliable))
