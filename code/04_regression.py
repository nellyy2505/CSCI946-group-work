"""
CSCI446/946 Big Data Analytics - Assignment 2
Task 2: Regression

Goal: predict gender:confidence (how much the crowd-sourced labellers
agreed on a profile's gender label) from profile features. A profile
the model expects to have a normal confidence, but whose real
confidence disagrees a lot with the prediction, is a candidate for a
mislabelled profile.

Structure (Linear Regression):
summarise -> sanity-check the relationship before fitting -> fit ->
evaluate in-sample -> evaluate on a held-out split -> predict.
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

# The features used: activity (tweets/favourites, log-scaled since they
# are heavily skewed), text stats, colour customisation, account age,
# plus binary flags and one-hot encoded timezone. All of this was
# already cleaned/scaled upstream in 02_preprocess.py.
NUM_COLS = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days", "text_len", "desc_len", "text_n_urls",
            "text_n_mentions", "text_n_hashtags", "name_n_digits",
            "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]


# =====================================================================
# 1. Load data
# =====================================================================
# 02_preprocess.py already split the labelled profiles (is_human not
# missing) into train/validation/test, stratified on is_human so all
# three sets keep the same human vs non-human ratio (60/20/20, seed=7).
# Loading the ready-made split here instead of re-splitting keeps every
# script in the group using the exact same rows for each set.
train = pd.read_csv(PROC / "twitter_train.csv")
val = pd.read_csv(PROC / "twitter_validation.csv")
test = pd.read_csv(PROC / "twitter_test.csv")

# tz_* columns (one-hot encoded timezone) are not a fixed list - read
# them from the data itself.
tz_cols = [c for c in train.columns if c.startswith("tz_")]
FEATURES = NUM_COLS + FLAG_COLS + tz_cols

print("----- shapes -----")
print("train:", train.shape, "validation:", val.shape, "test:", test.shape)

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
# gender, is_human and label_conflict are deliberately NOT used as
# features: they are derived from (or are) the label this model is
# trying to evaluate, so using them would leak the answer into the
# model instead of testing whether the profile's own behaviour
# predicts its label confidence.


# =====================================================================
# 2. Summarize the target
# =====================================================================
# describe(): distribution of gender:confidence on the training set.
print("\n----- y_train.describe() -----")
print(y_train.describe())

# var(): the MSE a model would get by always predicting the mean of
# y_train, regardless of any feature. This is the baseline every model
# below needs to beat to be worth anything - same idea as comparing
# MSE to y.var() in Describe 6 of the lab.
target_var = y_train.var()
print("\nVar(y_train), i.e. the 'always predict the mean' baseline MSE:", round(target_var, 4))

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
# Same habit as the lab's Describe 5 (sns.lmplot before committing to a
# linear model): look at how strongly each feature relates to the
# target first, so a low R^2 later can be explained rather than
# mistaken for a bug.
corr = train[NUM_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
print("\n----- correlation of each numeric feature with gender:confidence -----")
print(corr)

corr.plot.barh(figsize=(6, 6))
plt.xlabel("correlation with gender:confidence")
plt.title("Feature correlation with the target (sanity check)")
plt.tight_layout()
plt.savefig(OUT / "regression_feature_correlation.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: regression_feature_correlation.png")

# Plot the single strongest-correlated feature against the target, with
# a fitted line - the same visual check as the lab's lmplot step, just
# drawn manually since we want it for one chosen feature.
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
print(f"\nEven the strongest single feature only has |corr|={corr[top]:.2f} with the target -"
      f" the point cloud above is wide and the fitted line is close to flat.")
print("This previews a low R^2 below: no single feature drives gender:confidence strongly.")


def evaluate(name, model, X, y):
    # Same 3 metrics the lab uses (MSE, R^2), plus RMSE (same unit as
    # the target, easier to read) and MAE (less sensitive to outliers
    # than MSE, useful given the target is skewed - see Describe 2).
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"{name}: MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return pred, mse, rmse, mae, r2


# =====================================================================
# 4. Fit a linear model (Ridge Regression)
# =====================================================================
# Plain LinearRegression is not used here: with 27 features (vs 1 in
# the lab's sea-ice example) and some of them correlated with each
# other (e.g. tweet_count_log and favs_per_day_log), an unregularised
# fit can become unstable. Ridge adds L2 regularisation (alpha=1.0),
# which keeps the coefficients small and stable without changing the
# underlying linear model.
ridge = Ridge(alpha=1.0, random_state=SEED)
ridge.fit(X_train, y_train)

# In-sample evaluation first (same as Describe 6: MSE/R^2 on the data
# the model was trained on).
print("\n----- Ridge: in-sample (train) -----")
_, ridge_train_mse, ridge_train_rmse, ridge_train_mae, ridge_train_r2 = evaluate(
    "Ridge (train)", ridge, X_train, y_train)

# Held-out evaluation on the validation set (same idea as Describe 7's
# held-out years: check performance on data the model has not seen).
print("\n----- Ridge: held-out (validation) -----")
_, ridge_mse, ridge_rmse, ridge_mae, ridge_r2 = evaluate(
    "Ridge (validation)", ridge, X_val, y_val)

# coef_: how much each feature pushes the predicted confidence up or
# down, holding the others fixed - the linear-model equivalent of
# est.coef_ in Describe 6.
coefs = pd.Series(ridge.coef_, index=FEATURES).sort_values()
coefs.plot.barh(figsize=(6, 9))
plt.xlabel("Ridge coefficient")
plt.title("Ridge coefficients")
plt.tight_layout()
plt.savefig(OUT / "regression_ridge_coefficients.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: regression_ridge_coefficients.png")


# =====================================================================
# 5. A second model for comparison (Random Forest)
# =====================================================================
# Beyond what the lab covers: gender:confidence is skewed, not normal
# (most values sit exactly at 1.0 - see Describe 2), so a model that
# does not assume a linear relationship is worth trying alongside
# Ridge. Random Forest also gives feature importances, which can be
# cross-checked against Ridge's coefficients.
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5,
                           random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)

print("\n----- RandomForest: in-sample (train) -----")
_, rf_train_mse, rf_train_rmse, rf_train_mae, rf_train_r2 = evaluate(
    "RandomForest (train)", rf, X_train, y_train)

print("\n----- RandomForest: held-out (validation) -----")
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
# 6. Pick the best model on the validation set
# =====================================================================
# The test set must only be used once, for the final report number
# below - not for deciding which model is "best". That decision is
# made here using validation performance only.
print("\n----- model selection (validation set) -----")
print(f"Ridge        validation MSE={ridge_mse:.4f} ({ridge_mse / target_var:.1%} of baseline), R2={ridge_r2:.4f}")
print(f"RandomForest validation MSE={rf_mse:.4f} ({rf_mse / target_var:.1%} of baseline), R2={rf_r2:.4f}")

best_name, best_model = ("RandomForest", rf) if rf_rmse < ridge_rmse else ("Ridge", ridge)
print("Best model, selected on validation:", best_name)


# =====================================================================
# 7. Predictions: final, one-time evaluation on the held-out test set
# =====================================================================
print(f"\n----- {best_name}: final evaluation (test) -----")
best_pred, best_mse, best_rmse, best_mae, best_r2 = evaluate(
    f"{best_name} (test - final)", best_model, X_test, y_test)
print(f"{best_name} test MSE is {best_mse / target_var:.1%} of the baseline Var(y_train) -"
      f" i.e. predicting from profile features only modestly beats guessing the mean confidence"
      f" for everyone. This matches the weak correlations found in step 3: gender:confidence"
      f" mostly reflects whether the labellers agreed, which is not strongly visible in the"
      f" profile's own features.")

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
# 8. Conclusion: profiles the model disagrees with most (Task 4)
# =====================================================================
# residual = actual - predicted. A large |residual| means the model
# expected a "normal" confidence from this profile's features, but the
# real confidence was very different - a candidate for a mislabelled
# profile, to be reviewed manually rather than treated as confirmed.
diag = test[["name", "gender", TARGET, "label_conflict"]].copy()
diag["predicted"] = best_pred
diag["residual"] = diag[TARGET] - diag["predicted"]
diag = diag.reindex(diag["residual"].abs().sort_values(ascending=False).index)

print("\n----- top 20 profiles by |residual| -----")
print(diag.head(20).to_string(index=False))

# label_conflict (from 02_preprocess.py) flags accounts whose name was
# labelled inconsistently across duplicate records - a different,
# independent signal of the same underlying problem. Checking overlap
# tells us whether the two signals agree or catch different cases.
overlap = diag.head(200)["label_conflict"].sum()
print(f"\nOf the top 200 profiles by residual, {overlap} are also flagged by label_conflict.")
print("Low/no overlap means the two signals catch different kinds of mislabelling - worth"
      " presenting both in Task 4 as separate, complementary views rather than one confirming"
      " the other.")

diag.head(50).to_csv(OUT / "regression_mislabel_candidates.csv", index=False)
print("Saved: regression_mislabel_candidates.csv")
