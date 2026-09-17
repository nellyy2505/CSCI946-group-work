# CSCI446/946 Big Data Analytics - Assignment 2
# 04 - Regression: predict gender:confidence to flag candidate mislabels

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

NUM_COLS = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days", "text_len", "desc_len", "text_n_urls",
            "text_n_mentions", "text_n_hashtags", "name_n_digits",
            "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]


# 1. load the pre-split train/test sets (see 02_preprocess.py - already stratified on is_human)
train = pd.read_csv(PROC / "twitter_train.csv")
test = pd.read_csv(PROC / "twitter_test.csv")
tz_cols = [c for c in train.columns if c.startswith("tz_")]
FEATURES = NUM_COLS + FLAG_COLS + tz_cols
print("train:", train.shape, "test:", test.shape)

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
# gender, is_human and label_conflict are not used as features - they come from the
# label this model is trying to evaluate, so using them would leak the answer


# 2. target distribution and a variance baseline (MSE if we always predicted the mean)
print(y_train.describe())
target_var = y_train.var()
print("Var(y_train):", round(target_var, 4))

plt.hist(y_train, bins=40)
plt.xlabel("gender:confidence")
plt.title("Distribution of gender:confidence (train)")
plt.savefig(OUT / "regression_target_distribution.png", dpi=120, bbox_inches="tight")
plt.show()


# 3. correlation with the target, and the strongest single feature as a sanity check
corr = train[NUM_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
print(corr)
corr.plot.barh(figsize=(6, 6))
plt.xlabel("correlation with gender:confidence")
plt.tight_layout()
plt.savefig(OUT / "regression_feature_correlation.png", dpi=120, bbox_inches="tight")
plt.show()

top = corr.abs().idxmax()
slope, intercept = np.polyfit(train[top], y_train, 1)
plt.scatter(train[top], y_train, s=6, alpha=0.15)
x_line = np.linspace(train[top].min(), train[top].max(), 100)
plt.plot(x_line, slope * x_line + intercept, "r-")
plt.xlabel(top)
plt.ylabel("gender:confidence")
plt.title(f"strongest single feature: {top} (corr={corr[top]:.2f})")
plt.savefig(OUT / "regression_top_feature.png", dpi=120, bbox_inches="tight")
plt.show()


def evaluate(name, model, X, y):
    pred = model.predict(X)
    rmse = mean_squared_error(y, pred) ** 0.5
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"{name}: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return pred, rmse, mae, r2


# 4. Ridge regression - linear baseline, regularised
ridge = Ridge(alpha=1.0, random_state=SEED)
ridge.fit(X_train, y_train)
_, ridge_train_rmse, _, _ = evaluate("Ridge (train)", ridge, X_train, y_train)
ridge_pred, ridge_rmse, ridge_mae, ridge_r2 = evaluate("Ridge (test)", ridge, X_test, y_test)

coefs = pd.Series(ridge.coef_, index=FEATURES).sort_values()
coefs.plot.barh(figsize=(6, 9))
plt.xlabel("Ridge coefficient")
plt.tight_layout()
plt.savefig(OUT / "regression_ridge_coefficients.png", dpi=120, bbox_inches="tight")
plt.show()


# 5. Random Forest - non-linear, no assumption of normal residuals
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5,
                           random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
_, rf_train_rmse, _, _ = evaluate("RandomForest (train)", rf, X_train, y_train)
rf_pred, rf_rmse, rf_mae, rf_r2 = evaluate("RandomForest (test)", rf, X_test, y_test)

importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
importance.plot.barh(figsize=(6, 9))
plt.xlabel("feature importance")
plt.tight_layout()
plt.savefig(OUT / "regression_rf_importance.png", dpi=120, bbox_inches="tight")
plt.show()


# 6. compare models: train vs test (overfitting check), and against the variance baseline
comparison = pd.DataFrame({
    "model": ["Ridge", "Ridge", "RandomForest", "RandomForest"],
    "split": ["train", "test", "train", "test"],
    "RMSE": [ridge_train_rmse, ridge_rmse, rf_train_rmse, rf_rmse],
    "MAE": [None, ridge_mae, None, rf_mae],
    "R2": [None, ridge_r2, None, rf_r2],
})
print(comparison)
comparison.to_csv(OUT / "regression_model_comparison.csv", index=False)

for name, rmse, r2 in [("Ridge", ridge_rmse, ridge_r2), ("RandomForest", rf_rmse, rf_r2)]:
    print(f"{name} test RMSE^2={rmse ** 2:.4f} vs Var(y_train)={target_var:.4f}"
         f" ({rmse ** 2 / target_var:.1%}), R2={r2:.4f}")

best_name, best_pred = ("RandomForest", rf_pred) if rf_rmse < ridge_rmse else ("Ridge", ridge_pred)
print("best model:", best_name)

plt.scatter(y_test, best_pred, s=8, alpha=0.4)
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("actual gender:confidence")
plt.ylabel("predicted gender:confidence")
plt.title(f"{best_name}: predicted vs actual (test)")
plt.savefig(OUT / "regression_prediction_evaluation.png", dpi=120, bbox_inches="tight")
plt.show()


# 7. profiles whose prediction disagrees most with the label - candidate mislabels for Task 4
diag = test[["name", "gender", TARGET, "label_conflict"]].copy()
diag["predicted"] = best_pred
diag["residual"] = diag[TARGET] - diag["predicted"]
diag = diag.reindex(diag["residual"].abs().sort_values(ascending=False).index)

print(diag.head(20).to_string(index=False))
print("of top 200 by residual, also flagged by label_conflict:", diag.head(200)["label_conflict"].sum())
diag.head(50).to_csv(OUT / "regression_mislabel_candidates.csv", index=False)
