"""
CSCI446/946 Big Data Analytics - Assignment 2
Task 2: Regression (Logistic)

Goal:
- predict is_human (1=human, 0=brand/non-human) from profile features
- complementary view to 4.1_linear_regression.py:
  4.1 asks "how confident was the label", this asks "does the profile look human"
- model confident about one class, but actually labelled the other -> candidate for Task 4

Steps:
check the classes -> fit -> evaluate (accuracy, confusion matrix) ->
try fewer features (RFE) -> pick the best on validation -> evaluate once on test -> conclusion
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_score, recall_score, f1_score)

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

# same features as 4.1_linear_regression.py -> the two views stay comparable
NUM_COLS = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days", "text_len", "desc_len", "text_n_urls",
            "text_n_mentions", "text_n_hashtags", "name_n_digits",
            "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]


# =====================================================================
# 1. Load data, check the target
# =====================================================================
# - reuse the exact train/validation/test split from 02_preprocess.py
# - stratified on is_human, 60/20/20, seed=7
# - same split as 4.1 -> results between the two files stay comparable
train = pd.read_csv(PROC / "twitter_train.csv")
val = pd.read_csv(PROC / "twitter_validation.csv")
test = pd.read_csv(PROC / "twitter_test.csv")
tz_cols = [c for c in train.columns if c.startswith("tz_")]
FEATURES = NUM_COLS + FLAG_COLS + tz_cols

print("----- shapes -----")
print("train:", train.shape, "validation:", val.shape, "test:", test.shape)

# check: exactly 2 classes, and how balanced they are
print("\n----- class balance (train) -----")
print(train[TARGET].value_counts())
print("share human:", round(train[TARGET].mean(), 3))
print("-> ~69% human vs ~31% non-human: imbalanced, but not extreme")
print("-> a model that always predicts 'human' would already score ~69% accuracy")
print("-> accuracy alone is not enough - precision/recall in step 6 matter more")

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
# drop gender:confidence / label_conflict from features
# -> related to the label we're evaluating -> would leak the answer


# =====================================================================
# 2. Fit Logistic Regression
# =====================================================================
# - LogisticRegression: despite the name, a classification model
# - learns a weighted sum of features -> sigmoid -> probability of class 1 (human)
# - probability >= 0.5 -> predicted class = 1, else 0
model = LogisticRegression(max_iter=1000, random_state=SEED)
model.fit(X_train, y_train)

y_hat_train = model.predict(X_train)
y_hat_val = model.predict(X_val)


# =====================================================================
# 3. Evaluate: accuracy and confusion matrix
# =====================================================================
# train vs validation accuracy close -> no overfitting
# big gap -> model memorised train instead of learning a general pattern
print("\n----- accuracy -----")
train_acc = accuracy_score(y_train, y_hat_train)
val_acc = accuracy_score(y_val, y_hat_val)
print("train:", round(train_acc, 4))
print("validation:", round(val_acc, 4))
print(f"-> gap = {abs(train_acc - val_acc):.4f} -> small -> no overfitting")

# rows/cols ordered [0, 1] = [non-human, human]
print("\n----- confusion matrix (validation, rows=actual, cols=predicted, order=[0,1]) -----")
print(confusion_matrix(y_val, y_hat_val))


# =====================================================================
# 4. Feature selection (RFE)
# =====================================================================
# - RFE: fit -> rank features by weight -> drop the weakest -> repeat
# - drop weakest features one by one -> check if a smaller model still works
# - fewer features -> easier to interpret, less risk of overfitting to noise
def rfe_accuracy(n_features):
    rfe = RFE(estimator=LogisticRegression(max_iter=1000, random_state=SEED),
              n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)
    acc = accuracy_score(y_val, rfe.predict(X_val))
    print(f"RFE, {n_features} features -> validation accuracy: {acc:.4f}")
    return rfe, acc

print("\n----- RFE: fewer features -----")
full_acc = accuracy_score(y_val, y_hat_val)
print(f"Full model ({len(FEATURES)} features) -> validation accuracy: {full_acc:.4f}")
rfe_10, acc_10 = rfe_accuracy(10)
rfe_5, acc_5 = rfe_accuracy(5)

best_rfe = rfe_10 if acc_10 >= acc_5 else rfe_5
selected = [f for f, keep in zip(FEATURES, best_rfe.support_) if keep]
print("features kept by the best RFE model:", selected)
print("-> full model wins over the smaller RFE models")
print("-> with ~10,600 training rows, 38 features does not overfit,")
print("   so cutting features here only throws away useful signal instead of noise")


# =====================================================================
# 5. Pick best model on validation, evaluate once on test
# =====================================================================
# - compare all 3 candidates on validation only
# - test set stays untouched until now - decision must not "see" it
candidates = {
    "full": (model, full_acc),
    "rfe_10": (rfe_10, acc_10),
    "rfe_5": (rfe_5, acc_5),
}
best_name = max(candidates, key=lambda k: candidates[k][1])
best_model = candidates[best_name][0]
print("\nBest model, selected on validation:", best_name)

y_hat_test = best_model.predict(X_test)
y_prob_test = best_model.predict_proba(X_test)[:, 1]

print(f"\n----- {best_name}: final evaluation (test) -----")
test_acc = accuracy_score(y_test, y_hat_test)
print("accuracy:", round(test_acc, 4))
cm = confusion_matrix(y_test, y_hat_test)
print("confusion matrix (rows=actual, cols=predicted, order=[0,1]):")
print(cm)

plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.xticks([0, 1], ["non-human", "human"])
plt.yticks([0, 1], ["non-human", "human"])
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title(f"{best_name}: confusion matrix (test)")
plt.tight_layout()
plt.savefig(OUT / "logistic_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: logistic_confusion_matrix.png")


# =====================================================================
# 6. Conclusion: precision, recall, F1 - both classes
# =====================================================================
# - sklearn's default (pos_label=1) only scores the "human" class
# - dataset is imbalanced (~69% human) -> also score "non-human" (pos_label=0)
#   on its own, otherwise a model that just leans towards the majority class
#   can still look good on the human-only numbers
precision_h = precision_score(y_test, y_hat_test, pos_label=1)
recall_h = recall_score(y_test, y_hat_test, pos_label=1)
f1_h = f1_score(y_test, y_hat_test, pos_label=1)

precision_nh = precision_score(y_test, y_hat_test, pos_label=0)
recall_nh = recall_score(y_test, y_hat_test, pos_label=0)
f1_nh = f1_score(y_test, y_hat_test, pos_label=0)

print("\n----- human (majority class) -----")
print(f"Precision: {precision_h:.4f}")
print(f"Recall: {recall_h:.4f}")
print(f"F1: {f1_h:.4f}")

print("\n----- non-human (minority class) -----")
print(f"Precision: {precision_nh:.4f}")
print(f"Recall: {recall_nh:.4f}")
print(f"F1: {f1_nh:.4f}")

print(f"\n-> recall human ({recall_h:.2f}) higher than recall non-human ({recall_nh:.2f})")
print("-> model leans towards predicting 'human' -> matches the ~69% human imbalance in step 1")
print("-> recall non-human is the real number for 'does this catch brand/bot accounts' -")
print("   the human-only numbers alone would hide how well the minority class is caught")
print("-> false negative (real person flagged non-human) is rarer than")
print("   false positive (a brand/bot-like account predicted human, so it slips through)")
print("-> which error matters more depends on how the group uses this result")
print("   (report finding vs automatic flagging) - worth a line in the report")


# =====================================================================
# 7. Candidate mislabels for Task 4 - a third, independent view
# =====================================================================
# model confident about one class, but actually labelled the other -> flag
# same idea as the residual in 4.1_linear_regression.py, just for a label instead of a number
diag = test[["name", "gender", TARGET, "label_conflict"]].copy()
diag["predicted_prob_human"] = y_prob_test
diag["predicted"] = y_hat_test
diag["misclassified"] = (diag["predicted"] != diag[TARGET]).astype(int)
# how far the predicted probability sits on the wrong side of 0.5
# -> bigger = model was more confident about the wrong class
diag["confidence_gap"] = np.where(
    diag["misclassified"] == 1,
    np.abs(diag["predicted_prob_human"] - 0.5) + 0.5,
    0,
)
diag = diag[diag["misclassified"] == 1].sort_values("confidence_gap", ascending=False)

print(f"\nmisclassified profiles in test set: {len(diag)} of {len(test)}")
print(diag.head(20).to_string(index=False))

# label_conflict = a different signal (inconsistent label across duplicate records)
# -> check overlap: does this view agree with label_conflict, or catch different cases?
top_n = min(200, len(diag))
overlap = diag.head(top_n)["label_conflict"].sum()
print(f"\nOf the top {top_n} misclassified profiles, {overlap} are also flagged by label_conflict.")
print(f"-> only {overlap}/{top_n} overlap -> this is a mostly independent, third signal")
print("-> Task 4 now has 3 views: association rules (03), regression residual (4.1),")
print("   and this misclassification view (4.2) - present them side by side, not merged into one")

diag.head(50).to_csv(OUT / "logistic_mislabel_candidates.csv", index=False)
print("Saved: logistic_mislabel_candidates.csv")
