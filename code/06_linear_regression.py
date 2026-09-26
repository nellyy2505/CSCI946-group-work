# CSCI446/946 Big Data Analytics - Assignment 2
# 06 - Linear regression: predict gender:confidence from the profile features (Lab 5, task 1)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

TARGET = "gender:confidence"
SEED = 7

# numeric features are already log-transformed and standardised in 02_preprocess.py
NUM_COLS = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days", "text_len", "desc_len", "text_n_urls",
            "text_n_mentions", "text_n_hashtags", "name_n_digits",
            "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]


# 1. load the split written by 02 (stratified on is_human, 60/20/20, SEED = 7)
train = pd.read_csv(PROC / "twitter_train.csv")
val = pd.read_csv(PROC / "twitter_validation.csv")
test = pd.read_csv(PROC / "twitter_test.csv")
tz_cols = [c for c in train.columns if c.startswith("tz_")]
FEATURES = NUM_COLS + FLAG_COLS + tz_cols      # gender, is_human and label_conflict stay out: they are the label
print("train:", train.shape, "validation:", val.shape, "test:", test.shape)

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]


# 2. the target: mostly exactly 1.0, so predicting the mean is the baseline every model must beat
print(y_train.describe().round(4))
target_var = y_train.var()
print("share of train exactly 1.0: %.3f | baseline MSE (predict the mean): %.4f"
      % ((y_train == 1).mean(), target_var))

plt.figure()
plt.hist(y_train, bins=40)
plt.xlabel("gender:confidence")
plt.ylabel("profiles")
plt.title("distribution of gender:confidence (train)")
plt.savefig(OUT / "fig_regression_linear_target.png", dpi=120, bbox_inches="tight")
plt.show()


# 3. correlation of each numeric feature with the target, before fitting anything
corr = train[NUM_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
print(corr.round(3))

plt.figure(figsize=(6, 6))
corr.plot.barh()
plt.xlabel("correlation with gender:confidence")
plt.title("feature correlation with the target")
plt.tight_layout()
plt.savefig(OUT / "fig_regression_linear_correlation.png", dpi=120, bbox_inches="tight")
plt.show()

top = corr.abs().idxmax()
slope, intercept = np.polyfit(train[top], y_train, 1)
x_line = np.linspace(train[top].min(), train[top].max(), 100)
plt.figure()
plt.scatter(train[top], y_train, s=6, alpha=0.15)
plt.plot(x_line, slope * x_line + intercept, "r-")
plt.xlabel(top)
plt.ylabel("gender:confidence")
plt.title("strongest single feature: %s (corr %.2f)" % (top, corr[top]))
plt.savefig(OUT / "fig_regression_linear_top_feature.png", dpi=120, bbox_inches="tight")
plt.show()


def evaluate(model, name, split, X, y):
    # MSE, RMSE, MAE and R2 for one model on one split
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    row = {"model": name, "split": split, "MSE": mse, "RMSE": mse ** 0.5,
           "MAE": mean_absolute_error(y, pred), "R2": r2_score(y, pred)}
    print("%-13s %-11s MSE %.4f  RMSE %.4f  MAE %.4f  R2 %.4f"
          % (name, split, row["MSE"], row["RMSE"], row["MAE"], row["R2"]))
    return pred, row


# 4. ridge regression: a linear model with an L2 penalty, since several features are correlated
ridge = Ridge(alpha=1.0, random_state=SEED)
ridge.fit(X_train, y_train)
rows = [evaluate(ridge, "Ridge", "train", X_train, y_train)[1],
        evaluate(ridge, "Ridge", "validation", X_val, y_val)[1]]

coefs = pd.Series(ridge.coef_, index=FEATURES).sort_values()
plt.figure(figsize=(6, 9))
coefs.plot.barh()
plt.xlabel("ridge coefficient")
plt.title("ridge coefficients")
plt.tight_layout()
plt.savefig(OUT / "fig_regression_linear_ridge_coefficients.png", dpi=120, bbox_inches="tight")
plt.show()


# 5. random forest: a non-linear comparison, with importances to check against the coefficients
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5,
                           random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
rows += [evaluate(rf, "RandomForest", "train", X_train, y_train)[1],
         evaluate(rf, "RandomForest", "validation", X_val, y_val)[1]]

importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(6, 9))
importance.plot.barh()
plt.xlabel("feature importance")
plt.title("random forest feature importance")
plt.tight_layout()
plt.savefig(OUT / "fig_regression_linear_rf_importance.png", dpi=120, bbox_inches="tight")
plt.show()


# 6. choose on validation, then evaluate the winner once on test
val_rmse = {r["model"]: r["RMSE"] for r in rows if r["split"] == "validation"}
best_name = min(val_rmse, key=val_rmse.get)
best_model = {"Ridge": ridge, "RandomForest": rf}[best_name]
print("selected on validation:", best_name)
best_pred, row = evaluate(best_model, best_name, "test", X_test, y_test)
rows.append(row)
print("test MSE is %.1f%% of the baseline" % (100 * row["MSE"] / target_var))

comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "regression_linear_model_comparison.csv", index=False)
print(comparison.round(4).to_string(index=False))

plt.figure()
plt.scatter(y_test, best_pred, s=8, alpha=0.4)
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("actual gender:confidence")
plt.ylabel("predicted gender:confidence")
plt.title(best_name + ": predicted vs actual (test)")
plt.savefig(OUT / "fig_regression_linear_predicted_vs_actual.png", dpi=120, bbox_inches="tight")
plt.show()


# 7. why the residuals are not a mislabel list: |residual| is almost the target itself, re-sorted
residual = y_test - best_pred
print("mislabel check - |residual| is the target re-sorted: corr(|residual|, gender:confidence) on test = %.3f"
      % np.corrcoef(np.abs(residual), y_test)[0, 1])
