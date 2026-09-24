"""Classification analysis for CSCI446/946 Assignment 2.

Trains four classifiers (Decision Tree, KNN, Naive Bayes, MLP) to predict
whether a Twitter profile is human or non-human from its metadata, compares
them, and screens for profiles whose recorded label looks inconsistent with
the pattern the models learned.

Inputs (created by 02_preprocess.py):
    data/processed/twitter_train.csv
    data/processed/twitter_validation.csv
    data/processed/twitter_test.csv
    data/processed/twitter_full.csv

Run from the repository root:
    python code/04_classification.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "is_human"
CLASS_NAMES = ["non-human", "human"]
NON_FEATURE_COLUMNS = {
    "_unit_id", "name", "gender", "gender:confidence", "label_conflict", TARGET,
    "description", "text", "desc_clean", "text_clean",
    "link_color", "sidebar_color",
    "fav_number", "tweet_count", "tweets_per_day", "favs_per_day",
}


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_data(root: Path) -> dict[str, pd.DataFrame]:
    """Read the four processed CSV files produced by 02_preprocess.py."""
    processed = root / "data" / "processed"
    return {
        "train": pd.read_csv(processed / "twitter_train.csv"),
        "validation": pd.read_csv(processed / "twitter_validation.csv"),
        "test": pd.read_csv(processed / "twitter_test.csv"),
        "full": pd.read_csv(processed / "twitter_full.csv"),
    }


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except identifiers, labels and raw text."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def reliable_mask(df: pd.DataFrame) -> pd.Series:
    """Rows with a fully confident, non-conflicting label, trustworthy for training."""
    confidence = pd.to_numeric(df["gender:confidence"], errors="coerce")
    conflict = pd.to_numeric(df["label_conflict"], errors="coerce").fillna(0)
    return df[TARGET].notna() & confidence.ge(1.0) & conflict.eq(0)


def build_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric feature matrix and raw target column (target may hold NaN)."""
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, df[TARGET]


def fit_imputer(X_fit: pd.DataFrame) -> SimpleImputer:
    """Median imputer fit once on the training pool and reused everywhere else."""
    return SimpleImputer(strategy="median").fit(X_fit)


def apply_imputer(imputer: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)


def scaled_pipeline(model) -> Pipeline:
    """Wraps a distance/gradient-based model with feature scaling, so
    cross-validation never scales a fold using information from itself."""
    return Pipeline([("scale", StandardScaler()), ("model", model)])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def classifier_diagnostics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Accuracy, macro F1, and per-class precision/recall from the test predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "non_human_precision": precision[0],
        "non_human_recall": recall[0],
        "human_precision": precision[1],
        "human_recall": recall[1],
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(name: str, cm: np.ndarray, out: Path) -> None:
    """Save a labelled heatmap of the confusion matrix for one model."""
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.matshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_title(f"Confusion matrix on test data - {name}", pad=14, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=200)
    plt.close(fig)


def tune_parameter(build_model, values: list, X: pd.DataFrame, y: pd.Series) -> tuple[list, list]:
    """Cross-validate one parameter value at a time; returns mean/std accuracy per value."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    means, stds = [], []
    for value in values:
        scores = cross_val_score(build_model(value), X, y, scoring="accuracy", cv=cv)
        means.append(scores.mean())
        stds.append(scores.std())
    return means, stds


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "output" / "classification"
    out.mkdir(parents=True, exist_ok=True)

    frames = load_data(root)
    features = get_feature_columns(frames["train"])
    print(f"Using {len(features)} features (identifiers, labels and text columns excluded).")

    # Training pool = reliable train + validation rows combined. Test is
    # kept aside and used once, at the end, for the reported comparison.
    train_reliable = frames["train"].loc[reliable_mask(frames["train"])]
    val_reliable = frames["validation"].loc[reliable_mask(frames["validation"])]
    test_reliable = frames["test"].loc[reliable_mask(frames["test"])]
    pool = pd.concat([train_reliable, val_reliable], ignore_index=True)
    print(f"Training pool (reliable labels only): {len(pool):,} rows")
    print(f"Test set (reliable labels only, untouched until final evaluation): {len(test_reliable):,} rows")

    X_pool_raw, y_pool = build_xy(pool, features)
    X_test_raw, y_test = build_xy(test_reliable, features)
    y_pool = y_pool.astype(int)
    y_test = y_test.astype(int)
    imputer = fit_imputer(X_pool_raw)
    X_pool = apply_imputer(imputer, X_pool_raw)
    X_test = apply_imputer(imputer, X_test_raw)

    # Majority-class baseline, to show what accuracy alone would look like
    # with no real learning.
    dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
    dummy.fit(X_pool, y_pool)
    dummy_pred = dummy.predict(X_test)

    # --- Decision Tree: depth, tree diagram, feature importance ---
    print("\n--- Decision Tree ---")
    depth_values = list(range(1, 21))
    depth_means, depth_stds = tune_parameter(
        lambda d: DecisionTreeClassifier(max_depth=d, random_state=SEED),
        depth_values, X_pool, y_pool,
    )
    best_depth = depth_values[int(np.argmax(depth_means))]
    print(f"Best max_depth by mean CV accuracy: {best_depth}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(depth_values, depth_means, yerr=depth_stds, marker="x")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("10-fold CV accuracy")
    ax.set_title("Decision Tree: accuracy vs max_depth")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "tuning_decision_tree_depth.png", dpi=200)
    plt.close(fig)

    dt_model = DecisionTreeClassifier(max_depth=best_depth, random_state=SEED)
    dt_model.fit(X_pool, y_pool)
    dt_pred = dt_model.predict(X_test)

    importance = pd.DataFrame({
        "feature": features, "importance": dt_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(out / "decision_tree_feature_importance.csv", index=False)

    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Gini importance")
    ax.set_title("Decision Tree feature importance")
    fig.tight_layout()
    fig.savefig(out / "decision_tree_feature_importance.png", dpi=200)
    plt.close(fig)

    # --- KNN: choose K ---
    print("\n--- KNN ---")
    k_values = list(range(1, 52, 4))
    k_means, _ = tune_parameter(
        lambda k: scaled_pipeline(KNeighborsClassifier(n_neighbors=k)), k_values, X_pool, y_pool,
    )
    best_k = k_values[int(np.argmax(k_means))]
    print(f"Best K by mean CV accuracy: {best_k}")

    knn_model = scaled_pipeline(KNeighborsClassifier(n_neighbors=best_k))
    knn_model.fit(X_pool, y_pool)
    knn_pred = knn_model.predict(X_test)

    # --- Gaussian Naive Bayes: no hyperparameter to sweep ---
    print("\n--- Naive Bayes ---")
    nb_model = GaussianNB()
    nb_model.fit(X_pool, y_pool)
    nb_pred = nb_model.predict(X_test)
    print("Estimated class priors P(non-human), P(human):", np.round(nb_model.class_prior_, 3))

    # --- MLP: fixed configuration ---
    # hidden_layer_sizes and alpha were chosen from an earlier sweep over
    # {(10,), (30,), (50,)} x {0.0001, 0.001, 0.01, 0.1}; re-running that
    # search here would cost ~70 slow MLP fits for an answer already known,
    # so the result is fixed instead of re-discovered on every run.
    print("\n--- MLP ---")
    mlp_hidden_layer_sizes = (10,)
    mlp_alpha = 0.1
    mlp_model = scaled_pipeline(
        MLPClassifier(
            hidden_layer_sizes=mlp_hidden_layer_sizes, alpha=mlp_alpha,
            max_iter=500, early_stopping=True, random_state=SEED,
        )
    )
    mlp_model.fit(X_pool, y_pool)
    mlp_pred = mlp_model.predict(X_test)

    # --- Final comparison on the held-out test set ---
    print("\n--- Final test-set comparison ---")
    predictions = {
        "Dummy": dummy_pred, "Decision Tree": dt_pred, "KNN": knn_pred,
        "Naive Bayes": nb_pred, "MLP": mlp_pred,
    }
    rows, confusion_matrices = [], {}
    for name, pred in predictions.items():
        diag = classifier_diagnostics(y_test, pred)
        confusion_matrices[name] = diag["confusion_matrix"]
        rows.append({"model": name, **{k: v for k, v in diag.items() if k != "confusion_matrix"}})
    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    comparison.to_csv(out / "test_model_comparison.csv", index=False)
    print(comparison.round(4).to_string(index=False))

    # Only the best-scoring model's confusion matrix is worth a figure -
    # the rest of the comparison is already in the table above.
    best_model_name = comparison.iloc[0]["model"]
    plot_confusion_matrix(best_model_name, confusion_matrices[best_model_name], out)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    positions = np.arange(len(comparison))
    ax.bar(positions - width / 2, comparison["accuracy"], width, label="Accuracy")
    ax.bar(positions + width / 2, comparison["macro_f1"], width, label="Macro F1")
    ax.set_xticks(positions)
    ax.set_xticklabels(comparison["model"])
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=200)
    plt.close(fig)

    # --- Screen for potentially mislabelled profiles ---
    # Only records the models never trained on are checked, so a model is
    # never judged against labels it has memorised: the test set, the
    # low-confidence/conflicting rows held out of training, and the
    # unknown-label rows.
    print("\n--- Screening for potentially mislabelled profiles ---")
    full = frames["full"]
    trained_ids = set(pool["_unit_id"])
    screening = full.loc[~full["_unit_id"].isin(trained_ids)].copy()
    X_screen_raw, _ = build_xy(screening, features)
    X_screen = apply_imputer(imputer, X_screen_raw)

    prob_columns = {}
    for key, model in [("decision_tree", dt_model), ("knn", knn_model),
                        ("naive_bayes", nb_model), ("mlp", mlp_model)]:
        prob_columns[key] = model.predict_proba(X_screen)[:, 1]  # P(human)

    result_cols = [c for c in ["_unit_id", "name", "gender", "gender:confidence", "label_conflict", TARGET]
                   if c in screening]
    result = screening[result_cols].copy()
    for key, prob in prob_columns.items():
        result[f"{key}_prob_human"] = prob
        result[f"{key}_prediction"] = np.where(prob >= 0.5, "human", "non-human")

    prob_matrix = np.column_stack(list(prob_columns.values()))
    result["mean_prob_human"] = prob_matrix.mean(axis=1)
    result["votes_for_human"] = (prob_matrix >= 0.5).sum(axis=1)
    result["consensus_prediction"] = np.where(result["votes_for_human"] >= 2, "human", "non-human")

    labelled = result[TARGET].notna()
    recorded_human = result[TARGET].fillna(-1).astype(int).eq(1)
    # Disagreement confidence: how strongly the models lean away from the
    # recorded label, so candidates can be ranked, not just counted.
    result["models_disagreeing_with_recorded_label"] = np.where(
        labelled,
        np.where(recorded_human, 4 - result["votes_for_human"], result["votes_for_human"]),
        np.nan,
    )
    result["disagreement_confidence"] = np.where(
        labelled,
        np.where(recorded_human, 1 - result["mean_prob_human"], result["mean_prob_human"]),
        np.nan,
    )
    result.to_csv(out / "all_screened_predictions.csv", index=False)

    candidates = result.loc[labelled & result["models_disagreeing_with_recorded_label"].ge(3)]
    candidates = candidates.sort_values(
        ["models_disagreeing_with_recorded_label", "disagreement_confidence"], ascending=False
    )
    candidates.to_csv(out / "potential_mislabels.csv", index=False)

    print(f"Screened {len(result):,} profiles never used in training.")
    print(f"Flagged {len(candidates):,} labelled profiles where at least 3 of 4 models "
          f"disagree with the recorded label.")

    unknown = result.loc[~labelled].sort_values("mean_prob_human", ascending=False)
    unknown.to_csv(out / "unknown_profile_predictions.csv", index=False)
    print(f"Predicted a class for {len(unknown):,} profiles with an unknown recorded label.")

    (out / "selected_hyperparameters.json").write_text(
        json.dumps({
            "decision_tree": {"max_depth": best_depth},
            "knn": {"n_neighbors": best_k},
            "mlp": {"hidden_layer_sizes": mlp_hidden_layer_sizes, "alpha": mlp_alpha,
                    "max_iter": 500, "early_stopping": True, "note": "fixed from an earlier sweep"},
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
