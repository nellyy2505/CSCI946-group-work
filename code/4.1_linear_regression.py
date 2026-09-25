"""
CSCI446/946 Big Data Analytics - Assignment 2
Task 2: Regression (Linear)

Goal:
- predict gender:confidence (crowd-labeller agreement score) from profile features
- big gap between actual vs predicted confidence -> candidate mislabelled profile (Task 4)

Steps:
look at the target -> check the features actually relate to it -> fit 2 models ->
compare them on validation -> evaluate the winner once on test -> flag suspicious profiles
"""

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

# feature groups: activity, text, colour, account age, flags, timezone
# -> all already cleaned/scaled upstream in 02_preprocess.py
NUM_COLS = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days", "text_len", "desc_len", "text_n_urls",
            "text_n_mentions", "text_n_hashtags", "name_n_digits",
            "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]


# =====================================================================
# 1. Load data
# =====================================================================
# - train/validation/test already split in 02_preprocess.py
# - stratified on is_human, 60/20/20, seed=7
# - reuse this exact split everywhere -> every script compares apples to apples
train = pd.read_csv(PROC / "twitter_train.csv")
val = pd.read_csv(PROC / "twitter_validation.csv")
test = pd.read_csv(PROC / "twitter_test.csv")

tz_cols = [c for c in train.columns if c.startswith("tz_")]
FEATURES = NUM_COLS + FLAG_COLS + tz_cols

print("----- shapes -----")
print("train:", train.shape, "validation:", val.shape, "test:", test.shape)

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
# drop gender / is_human / label_conflict from features
# -> they come from the label we are trying to evaluate -> would leak the answer


# =====================================================================
# 2. Summarize the target
# =====================================================================
print("\n----- y_train.describe() -----")
print(y_train.describe())

# baseline: MSE if we always predict the mean, no features at all
# -> every model below must beat this number to be worth anything
target_var = y_train.var()
print("\nVar(y_train), baseline MSE:", round(target_var, 4))

plt.hist(y_train, bins=40)
plt.xlabel("gender:confidence")
plt.ylabel("number of profiles")
plt.title("Distribution of gender:confidence (train)")
plt.savefig(OUT / "regression_target_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: regression_target_distribution.png")


# =====================================================================
# 3. Sanity-check the relationship, before fitting anything
# =====================================================================
# - look at the data before fitting a model to it
# - check correlation of each feature with the target
corr = train[NUM_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
print("\n----- correlation with gender:confidence -----")
print(corr)

corr.plot.barh(figsize=(6, 6))
plt.xlabel("correlation with gender:confidence")
plt.title("Feature correlation with the target (sanity check)")
plt.tight_layout()
plt.savefig(OUT / "regression_feature_correlation.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: regression_feature_correlation.png")

# plot the strongest single feature vs target, with a fitted line
top = corr.abs().idxmax()
slope, intercept = np.polyfit(train[top], y_train, 1)
x_line = np.linspace(train[top].min(), train[top].max(), 100)

plt.scatter(train[top], y_train, s=6, alpha=0.15)
plt.plot(x_line, slope * x_line + intercept, "r-")
plt.xlabel(top)
plt.ylabel("gender:confidence")
plt.title(f"Strongest single feature: {top} (corr={corr[top]:.2f})")
plt.savefig(OUT / "regression_top_feature.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: regression_top_feature.png")
print(f"\n-> even the strongest feature only reaches |corr|={corr[top]:.2f}")
print("-> expect a low R^2 below - that's the data, not a bug")


def evaluate(name, model, X, y):
    # predict -> compute MSE / RMSE / MAE / R2 -> print -> return
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"{name}: MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return pred, mse, rmse, mae, r2


# =====================================================================
# 4. Fit Ridge Regression (linear model)
# =====================================================================
# - 27 features, some correlated with each other -> plain LinearRegression unstable
# - Ridge = linear regression + L2 regularization -> keeps coefficients stable
ridge = Ridge(alpha=1.0, random_state=SEED)
ridge.fit(X_train, y_train)

print("\n----- Ridge: train -----")
_, ridge_train_mse, ridge_train_rmse, ridge_train_mae, ridge_train_r2 = evaluate(
    "Ridge (train)", ridge, X_train, y_train)

print("\n----- Ridge: validation (held-out) -----")
_, ridge_mse, ridge_rmse, ridge_mae, ridge_r2 = evaluate(
    "Ridge (validation)", ridge, X_val, y_val)

# coefficients -> which feature pushes confidence up / down
coefs = pd.Series(ridge.coef_, index=FEATURES).sort_values()
coefs.plot.barh(figsize=(6, 9))
plt.xlabel("Ridge coefficient")
plt.title("Ridge coefficients")
plt.tight_layout()
plt.savefig(OUT / "regression_ridge_coefficients.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: regression_ridge_coefficients.png")


# =====================================================================
# 5. Fit Random Forest (second model, for comparison)
# =====================================================================
# - target is skewed (most values = 1.0) -> try a non-linear model too
# - also gives feature importances -> cross-check against Ridge coefficients
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5,
                           random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)

print("\n----- RandomForest: train -----")
_, rf_train_mse, rf_train_rmse, rf_train_mae, rf_train_r2 = evaluate(
    "RandomForest (train)", rf, X_train, y_train)

print("\n----- RandomForest: validation (held-out) -----")
_, rf_mse, rf_rmse, rf_mae, rf_r2 = evaluate(
    "RandomForest (validation)", rf, X_val, y_val)

importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
importance.plot.barh(figsize=(6, 9))
plt.xlabel("feature importance")
plt.title("Random Forest feature importance")
plt.tight_layout()
plt.savefig(OUT / "regression_rf_importance.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: regression_rf_importance.png")


# =====================================================================
# 6. Pick the best model on validation
# =====================================================================
# - compare Ridge vs RandomForest on validation only
# - test set stays untouched until step 7 - decision must not "see" it
print("\n----- model selection (validation set) -----")
print(f"Ridge        validation MSE={ridge_mse:.4f} ({ridge_mse / target_var:.1%} of baseline), R2={ridge_r2:.4f}")
print(f"RandomForest validation MSE={rf_mse:.4f} ({rf_mse / target_var:.1%} of baseline), R2={rf_r2:.4f}")

best_name, best_model = ("RandomForest", rf) if rf_rmse < ridge_rmse else ("Ridge", ridge)
print("Best model, selected on validation:", best_name)


# =====================================================================
# 7. Final evaluation on test (one time only)
# =====================================================================
print(f"\n----- {best_name}: final evaluation (test) -----")
best_pred, best_mse, best_rmse, best_mae, best_r2 = evaluate(
    f"{best_name} (test - final)", best_model, X_test, y_test)
print(f"-> test MSE is {best_mse / target_var:.1%} of the baseline")
print("-> profile features barely beat guessing the mean confidence for everyone")
print("-> confidence mostly reflects labeller agreement, not something visible in the profile")

comparison = pd.DataFrame({
    "model": ["Ridge", "Ridge", "RandomForest", "RandomForest", best_name],
    "split": ["train", "validation", "train", "validation", "test (final)"],
    "MSE": [ridge_train_mse, ridge_mse, rf_train_mse, rf_mse, best_mse],
    "RMSE": [ridge_train_rmse, ridge_rmse, rf_train_rmse, rf_rmse, best_rmse],
    "MAE": [ridge_train_mae, ridge_mae, rf_train_mae, rf_mae, best_mae],
    "R2": [ridge_train_r2, ridge_r2, rf_train_r2, rf_r2, best_r2],
})
print("\n----- summary table -----")
print(comparison)
comparison.to_csv(OUT / "regression_model_comparison.csv", index=False)
print("Saved: regression_model_comparison.csv")

plt.scatter(y_test, best_pred, s=8, alpha=0.4)
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("actual gender:confidence")
plt.ylabel("predicted gender:confidence")
plt.title(f"{best_name}: predicted vs actual (test)")
plt.savefig(OUT / "regression_prediction_evaluation.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: regression_prediction_evaluation.png")


# =====================================================================
# 8. Candidate mislabels for Task 4
# =====================================================================
# - residual = actual - predicted
# - big |residual| -> profile "looks normal" but confidence disagrees -> flag as candidate
# - not a confirmed error - just worth a manual look
diag = test[["name", "gender", TARGET, "label_conflict"]].copy()
diag["predicted"] = best_pred
diag["residual"] = diag[TARGET] - diag["predicted"]
diag = diag.reindex(diag["residual"].abs().sort_values(ascending=False).index)

print("\n----- top 20 profiles by |residual| -----")
print(diag.head(20).to_string(index=False))

# label_conflict = a different signal (inconsistent label across duplicate records)
# -> check overlap: do the two signals agree, or catch different cases?
overlap = diag.head(200)["label_conflict"].sum()
print(f"\nOf the top 200 profiles by residual, {overlap} are also flagged by label_conflict.")
print("-> low overlap = two different signals, not one confirming the other")
print("-> present both separately in Task 4")

diag.head(50).to_csv(OUT / "regression_mislabel_candidates.csv", index=False)
print("Saved: regression_mislabel_candidates.csv")
