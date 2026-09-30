# 06_linear_regression.py — linear regression on is_human: why a 0/1 outcome needs logistic

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

plt.rcParams.update({"font.size": 11})   # figures are placed at 4-6.2 in wide in the report

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

TARGET = "is_human"
SEED = 7                  # the seed 02 used for the split; nothing in this script is random

# same 38 features as 07
NUM_COLS = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days", "text_len", "desc_len", "text_n_urls",
            "text_n_mentions", "text_n_hashtags", "name_n_digits",
            "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]


# 1. Load
# the split written by 02 (stratified on is_human, 60/20/20, SEED = 7)
train = pd.read_csv(PROC / "twitter_train.csv")
val = pd.read_csv(PROC / "twitter_validation.csv")
test = pd.read_csv(PROC / "twitter_test.csv")
print("train:", train.shape, "validation:", val.shape, "test:", test.shape)

tz_cols = [c for c in train.columns if c.startswith("tz_")]
FEATURES = NUM_COLS + FLAG_COLS + tz_cols      # gender:confidence and label_conflict stay out: they describe the label
print("features:", len(FEATURES))

X_train, y_train = train[FEATURES], train[TARGET].astype(int)
X_val, y_val = val[FEATURES], val[TARGET].astype(int)
X_test, y_test = test[FEATURES], test[TARGET].astype(int)


# 2. Why crowd confidence is not a regression target
# gender:confidence is an agreement score that sits on a few values, mostly exactly 1.0
confidence = train["gender:confidence"]
print("\ngender:confidence (train), most common values:")
print(confidence.round(2).value_counts().head(6).to_string())
print("share exactly 1.0: %.3f" % (confidence == 1).mean())

plt.figure(figsize=(6.2, 3.6))
plt.hist(confidence, bins=40, color="grey")
plt.xlabel("gender:confidence")
plt.ylabel("profiles (train)")
plt.title("gender:confidence is not a continuous target")
plt.tight_layout()
plt.savefig(OUT / "fig_regression_linear_target.png", dpi=150, bbox_inches="tight")
plt.show()


# 3. Linear regression on is_human
# the linear probability model: ordinary least squares on a 0/1 target
model = LinearRegression()
model.fit(X_train, y_train)
coefs = pd.DataFrame({"feature": FEATURES, "coef": model.coef_}).sort_values("coef", ascending=False)
coefs.to_csv(OUT / "regression_linear_coefficients.csv", index=False)
print("\nintercept %.4f" % model.intercept_)
print("coefficients, sorted:")
print(coefs.round(4).to_string(index=False))

# top 10 each way; positive pushes towards human, negative towards non-human
top = pd.concat([coefs.head(10), coefs.tail(10)]).sort_values("coef")
plt.figure(figsize=(6.2, 5.5))
plt.barh(top["feature"], top["coef"], color=np.where(top["coef"] > 0, "tab:blue", "tab:orange"))
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("coefficient (change in predicted is_human per unit)")
plt.ylabel("feature")
plt.title("linear regression on is_human: top 10 coefficients each way")
plt.tight_layout()
plt.savefig(OUT / "fig_regression_linear_coefficients.png", dpi=150, bbox_inches="tight")
plt.show()


# 4. Evaluation
# R2 and MSE as in Lab 5, plus accuracy with predictions cut at 0.5 to compare with logistic regression
rows = []
for split, X, y in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
    pred = model.predict(X)
    rows.append({"model": "LinearRegression", "split": split, "r2": r2_score(y, pred),
                 "mse": mean_squared_error(y, pred), "accuracy_at_0.5": accuracy_score(y, (pred >= 0.5).astype(int))})
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "regression_linear_model_comparison.csv", index=False)
print("\n" + comparison.round(4).to_string(index=False))


# 5. Where the linear model breaks
# a probability must lie in [0, 1]; a straight line does not respect that
pred_test = model.predict(X_test)
outside = (pred_test < 0) | (pred_test > 1)
print("\ntest predictions below 0: %d | above 1: %d | outside [0, 1]: %d of %d (%.1f%%)"
      % ((pred_test < 0).sum(), (pred_test > 1).sum(), outside.sum(), len(pred_test), 100 * outside.mean()))
print("test prediction min %.3f | max %.3f" % (pred_test.min(), pred_test.max()))

plt.figure(figsize=(6.2, 3.8))
bins = np.linspace(pred_test.min(), pred_test.max(), 40)
plt.hist(pred_test[y_test == 1], bins=bins, alpha=0.7, color="tab:blue", label="recorded human")
plt.hist(pred_test[y_test == 0], bins=bins, alpha=0.7, color="tab:orange", label="recorded non_human")
for x in [0, 0.5, 1]:
    plt.axvline(x, color="black", linestyle="--" if x == 0.5 else "-", linewidth=1)
plt.xlabel("predicted is_human (test)")
plt.ylabel("profiles")
plt.title("linear predictions escape [0, 1]")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig_regression_linear_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

# with a 0/1 outcome every residual is either -fitted or 1 - fitted, so the points fall on two lines
residual = y_test - pred_test
plt.figure(figsize=(5, 3.8))
plt.scatter(pred_test, residual, s=6, alpha=0.3, color="grey")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("fitted value (test)")
plt.ylabel("residual")
plt.title("residuals vs fitted: two bands")
plt.tight_layout()
plt.savefig(OUT / "fig_regression_linear_residuals.png", dpi=150, bbox_inches="tight")
plt.show()

