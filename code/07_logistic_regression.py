# CSCI446/946 Big Data Analytics - Assignment 2
# 07 - Logistic regression: predict is_human from the profile features (Lab 5, task 2)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

TARGET = "is_human"
SEED = 7
MIN_SCORE = 0.90          # chosen from the sweep below, targets a few hundred profiles

# same features as 06_linear_regression.py, so the two regression views stay comparable
NUM_COLS = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days", "text_len", "desc_len", "text_n_urls",
            "text_n_mentions", "text_n_hashtags", "name_n_digits",
            "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]


# 1. load the split written by 02 (stratified on is_human, 60/20/20, SEED = 7) and check the classes
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


# 2. fit, then compare train with validation accuracy to check for overfitting
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
print("\naccuracy - train %.4f | validation %.4f | gap %.4f" % (train_acc, val_acc, abs(train_acc - val_acc)))
print("confusion matrix (validation) [rows actual, cols predicted, non_human first]:\n",
      confusion_matrix(y_val, model.predict(X_val)))


# 3. RFE: drop the weakest features one at a time and see whether a smaller model holds up
def rfe_model(n_features):
    rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)
    acc = accuracy_score(y_val, rfe.predict(X_val))
    print("RFE %2d features -> validation accuracy %.4f" % (n_features, acc))
    return rfe, acc


print("\nfull model (%d features) -> validation accuracy %.4f" % (len(FEATURES), val_acc))
candidates = {"full": (model, val_acc)}
for n in [10, 5]:
    candidates["rfe_%d" % n] = rfe_model(n)


# 4. choose on validation, then evaluate the winner once on test
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

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["non_human", "human"])
ax.set_yticks([0, 1], ["non_human", "human"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("logistic regression (%s): confusion matrix (test)" % best_name)
plt.tight_layout()
plt.savefig(OUT / "fig_regression_logistic_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()


# 5. the product: an out-of-fold verdict for every labelled profile with the candidate chosen above
#    (an RFE candidate re-selects its features inside each fold, so nothing leaks between folds)
# (Lab 4's k-fold applied to Lab 5's model, so no profile is judged by a model that trained on it)
df = pd.read_csv(PROC / "twitter_full.csv")
labelled = df["is_human"].notna()
X_all = df.loc[labelled, FEATURES].to_numpy(float)
y_all = df.loc[labelled, TARGET].astype(int).to_numpy()
recorded = df.loc[labelled, "label"].to_numpy()
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
prob = cross_val_predict(clone(best_model), X_all, y_all, cv=cv, method="predict_proba")[:, 1]
says = np.where(prob >= 0.5, "human", "non_human")
score = np.maximum(prob, 1 - prob)
disagree = says != recorded
print("\n5-fold out-of-fold accuracy on all %d labelled profiles: %.4f" % (len(y_all), (says == recorded).mean()))

conf_all = df.loc[labelled, "gender:confidence"].to_numpy()
print("threshold sweep (crowd was unsure about %.3f of all labelled profiles)" % (conf_all < 1).mean())
sweep = []
for cut in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    hit = disagree & (score >= cut)
    sweep.append({"min_score": cut, "profiles": int(hit.sum()),
                  "crowd_unsure": (conf_all[hit] < 1).mean() if hit.sum() else np.nan})
print(pd.DataFrame(sweep).round(3).to_string(index=False))


# 6. output contract: regression_predictions.csv for every profile, regression_flagged.csv for the nominations
final = clone(best_model).fit(X_all, y_all)                  # refit on every labelled row for the unknown profiles
unknown = ~labelled
prob_unknown = final.predict_proba(df.loc[unknown, FEATURES].to_numpy(float))[:, 1]
predictions = pd.concat([
    pd.DataFrame({"_unit_id": df.loc[labelled, "_unit_id"].to_numpy(), "says": says,
                  "score": score.round(3), "recorded": recorded}),
    pd.DataFrame({"_unit_id": df.loc[unknown, "_unit_id"].to_numpy(),
                  "says": np.where(prob_unknown >= 0.5, "human", "non_human"),
                  "score": np.maximum(prob_unknown, 1 - prob_unknown).round(3), "recorded": "unknown"}),
], ignore_index=True)
predictions.to_csv(OUT / "regression_predictions.csv", index=False)
print("\nregression_predictions.csv:", len(predictions), "rows (labelled out-of-fold + unknown)")

hit = disagree & (score >= MIN_SCORE)
lab_rows = df.loc[labelled]
result = pd.DataFrame({
    "_unit_id": lab_rows["_unit_id"].to_numpy()[hit],
    "says": says[hit],
    "score": score[hit].round(3),
    "recorded": recorded[hit],
    "gender": lab_rows["gender"].to_numpy()[hit],
    "name": lab_rows["name"].to_numpy()[hit],
    "gender:confidence": conf_all[hit],
}).sort_values("score", ascending=False)
result.to_csv(OUT / "regression_flagged.csv", index=False)

print("profiles nominated (regression_flagged.csv):", len(result))
print(pd.crosstab(result["recorded"], result["says"]))
print(result.head(10)[["name", "gender", "recorded", "says", "score"]])

# check the nominations against crowd confidence, which the model never used
plt.figure(figsize=(8, 4))
groups = [result["gender:confidence"], pd.Series(conf_all)]
plt.hist(groups, bins=20, weights=[np.full(len(g), 100 / len(g)) for g in groups],
         label=["nominated by logistic regression", "all labelled profiles"])
plt.xlabel("gender:confidence")
plt.ylabel("% of the group")
plt.title("crowd confidence, nominated vs all")
plt.legend()
plt.savefig(OUT / "fig_regression_logistic_crowd_confidence.png", dpi=120, bbox_inches="tight")
plt.show()
