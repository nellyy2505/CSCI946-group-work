# 07_logistic_regression.py — logistic regression on is_human with RFE; the regression vote (Lab 5, task 2)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

plt.rcParams.update({"font.size": 11})   # figures are placed at 4-6.2 in wide in the report

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

TARGET = "is_human"
MIN_SCORE = 0.90          # fixed in advance, the same for 05, 07 and 08: a method votes when score >= 0.90

# same features as 06_linear_regression.py, so the two regression views stay comparable
NUM_COLS = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days", "text_len", "desc_len", "text_n_urls",
            "text_n_mentions", "text_n_hashtags", "name_n_digits",
            "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]


# 1. Load
# the split written by 02 (stratified on is_human, 60/20/20), and the class balance
train = pd.read_csv(PROC / "twitter_train.csv")
val = pd.read_csv(PROC / "twitter_validation.csv")
test = pd.read_csv(PROC / "twitter_test.csv")
tz_cols = [c for c in train.columns if c.startswith("tz_")]
FEATURES = NUM_COLS + FLAG_COLS + tz_cols      # gender:confidence and label_conflict stay out: they describe the label
print("train:", train.shape, "validation:", val.shape, "test:", test.shape)

print(train[TARGET].value_counts())
print("share human: %.3f (the accuracy of always guessing human)" % train[TARGET].mean())

X_train, y_train = train[FEATURES], train[TARGET].astype(int)
X_val, y_val = val[FEATURES], val[TARGET].astype(int)
X_test, y_test = test[FEATURES], test[TARGET].astype(int)


# 2. Full model
# compare train with validation accuracy to check for overfitting
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
print("\naccuracy - train %.4f | validation %.4f | gap %.4f" % (train_acc, val_acc, abs(train_acc - val_acc)))
print("confusion matrix (validation) [rows recorded, cols predicted, non_human first]:\n",
      confusion_matrix(y_val, model.predict(X_val)))


# 3. RFE
# drop the weakest features one at a time and see whether a smaller model holds up
def rfe_model(n_features):
    rfe = RFE(estimator=LogisticRegression(max_iter=2000), n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)
    acc = accuracy_score(y_val, rfe.predict(X_val))
    print("RFE %2d features -> validation accuracy %.4f" % (n_features, acc))
    return rfe, acc


print("\nfull model (%d features) -> validation accuracy %.4f" % (len(FEATURES), val_acc))
candidates = {"full": (model, val_acc)}
for n in [10, 5]:
    candidates["rfe_%d" % n] = rfe_model(n)


# 4. Choose on validation, test once
best_name = max(candidates, key=lambda k: candidates[k][1])
best_model = candidates[best_name][0]
if best_name != "full":
    print("features kept:", [f for f, keep in zip(FEATURES, best_model.support_) if keep])
print("selected on validation:", best_name)

pred_test = best_model.predict(X_test)
test_acc = accuracy_score(y_test, pred_test)
cm = confusion_matrix(y_test, pred_test)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred_test, labels=[0, 1])
print("\ntest accuracy: %.4f" % test_acc)
print("confusion matrix (test):\n", cm)
print(pd.DataFrame({"precision": precision, "recall": recall, "f1": f1},
                   index=["non_human", "human"]).round(4))

comparison = pd.DataFrame({"model": list(candidates),
                           "validation_accuracy": [acc for _, acc in candidates.values()]})
comparison["test_accuracy"] = np.where(comparison["model"] == best_name, test_acc, np.nan)
comparison.to_csv(OUT / "regression_logistic_model_comparison.csv", index=False)
print("regression_logistic_model_comparison.csv:", len(comparison), "candidates")

fig, ax = plt.subplots(figsize=(4.4, 3.8))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["non_human", "human"])
ax.set_yticks([0, 1], ["non_human", "human"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("predicted")
plt.ylabel("recorded")
plt.title("logistic regression (%s):\nconfusion matrix (test)" % best_name)
plt.tight_layout()
plt.savefig(OUT / "fig_regression_logistic_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# 5. Coefficients and odds ratios
# log-odds of human per unit of each feature (Week 6)
kept = FEATURES if best_name == "full" else [f for f, keep in zip(FEATURES, best_model.support_) if keep]
fitted = best_model if best_name == "full" else best_model.estimator_
coefs = pd.DataFrame({"feature": kept, "coef": fitted.coef_[0]})
coefs["odds_ratio"] = np.exp(coefs["coef"])
coefs = coefs.sort_values("coef", ascending=False)
coefs.round(6).to_csv(OUT / "regression_logistic_coefficients.csv", index=False)   # rounded so reruns match
print("\nintercept %.4f" % fitted.intercept_[0])
print("coefficients and odds ratios (exp(coef)), sorted:")
print(coefs.round(4).to_string(index=False))


# 6. Predict every profile and nominate
# training rows are judged by a model that saw their label, so flags there are conservative
df = pd.read_csv(PROC / "twitter_full.csv")
split_of = pd.concat([pd.Series(name, index=part["_unit_id"])
                      for name, part in [("train", train), ("validation", val), ("test", test)]])
prob = best_model.predict_proba(df[FEATURES])[:, 1]
score = np.maximum(prob, 1 - prob).round(3)             # rounded first, so the file obeys votes == (score >= 0.90)
predictions = pd.DataFrame({
    "_unit_id": df["_unit_id"],
    "says": np.where(prob >= 0.5, "human", "non_human"),
    "score": score,
    "recorded": df["label"],
    "votes": (score >= MIN_SCORE).astype(int),
    "split": df["_unit_id"].map(split_of).fillna("unknown"),
})
predictions.to_csv(OUT / "regression_predictions.csv", index=False)
print("\nregression_predictions.csv:", len(predictions), "rows")
print(predictions["split"].value_counts().to_string())

labelled = predictions["recorded"] != "unknown"
disagree = labelled & (predictions["says"] != predictions["recorded"])
# printed for information only; the cut-off above was fixed before looking at it
print(pd.DataFrame([{"min_score": cut, "profiles": int((disagree & (score >= cut)).sum())}
                    for cut in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]).to_string(index=False))

hit = disagree & (predictions["votes"] == 1)
result = predictions.loc[hit, ["_unit_id", "says", "score", "recorded", "split"]].join(
    df.loc[hit, ["gender", "name", "gender:confidence"]]).sort_values("score", ascending=False)
result.to_csv(OUT / "regression_flagged.csv", index=False)

print("\nprofiles nominated (regression_flagged.csv):", len(result))
print(pd.crosstab(result["recorded"], result["says"]))
print("nominations per split:\n" + result["split"].value_counts().to_string())
print(result.head(10)[["name", "gender", "recorded", "says", "score"]])


# 7. Crowd-confidence check
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
             color=["tab:red", "grey"], label=["nominated by logistic regression", "all labelled profiles"])
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
plt.savefig(OUT / "fig_regression_logistic_crowd_confidence.png", dpi=150, bbox_inches="tight")
plt.show()
