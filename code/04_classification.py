"""CSCI446/946 Assignment 2 - Classification analysis.

Models: Dummy baseline, Decision Tree, K-Nearest Neighbours, Naive Bayes
(Gaussian), and MLPClassifier. Naive Bayes and Decision Tree are the two
probabilistic/rule-based methods from the Week 5 classification material;
KNN is the third method from the same slide deck; MLP is included as an
additional nonlinear model for comparison.

Expected inputs (created by 02_preprocess.py):
    data/processed/twitter_train.csv
    data/processed/twitter_validation.csv
    data/processed/twitter_test.csv
    data/processed/twitter_full.csv

Outputs are written to data/output/classification, matching the layout
03_association_rules.py already uses (data/output). --project-root defaults
to the repository root inferred from this file's own location, so the
script runs unchanged on every group member's machine regardless of where
the repo is cloned - for example D:\\uow-mphil\\resh905\\group_assignment\\
CSCI946-group-work on one machine, a different drive or a Mac path on
another. Do not hardcode an absolute path here: the assignment specifically
penalises code that does not run on another machine.

Run from the repository root:
    python code/04_classification.py

Useful options:
    python code/04_classification.py --quick
    python code/04_classification.py --min-label-confidence 0.8
    python code/04_classification.py --candidate-confidence 0.8
    python code/04_classification.py --tie-tolerance 0.01
    python code/04_classification.py --project-root "D:\\path\\to\\CSCI946-group-work"

Note on Naive Bayes: this script uses GaussianNB over the same engineered
metadata features as the other three models, for direct comparability.
GaussianNB assumes the features are conditionally independent given the
class; several engineered features are correlated (tweet_count_log with
tweets_per_day_log, fav_number_log with favs_per_day_log, and the three
sidebar RGB channels with each other - see 02_preprocess.py step 16), so
its probability outputs should be read as directionally useful rather than
well calibrated. A text-based MultinomialNB (using desc_clean/text_clean)
would need those columns added to the split files in 02_preprocess.py;
that is a preprocessing change and is out of scope for this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 7
TARGET = "is_human"
LABEL_NAMES = {0: "non-human", 1: "human"}
# Default margin within which two models' test macro F1 scores are treated as a
# tie; overridable with --tie-tolerance. See select_best_model().
TIE_TOLERANCE = 0.01
NON_FEATURE_COLUMNS = {
    "_unit_id",
    "name",
    "gender",
    "gender:confidence",
    "label_conflict",
    TARGET,
    # Text/raw fields occur in twitter_full.csv but not in the split files.
    "description",
    "text",
    "desc_clean",
    "text_clean",
    "link_color",
    "sidebar_color",
    "fav_number",
    "tweet_count",
    "tweets_per_day",
    "favs_per_day",
}

LOGGER = logging.getLogger("classification")


def configure_logging(out: Path) -> None:
    """Send progress messages to the console and to a run log inside the output folder.

    A persisted log is part of the assignment's own request for a reproducible,
    auditable work record - it lets a marker (or a teammate) see exactly what a
    run did without re-running it.
    """
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    LOGGER.addHandler(console)

    file_handler = logging.FileHandler(out / "run_log.txt", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Twitter human/non-human classifiers.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing data/ (default: inferred from this file's location).",
    )
    parser.add_argument(
        "--min-label-confidence",
        type=float,
        default=1.0,
        help="Minimum crowd confidence for reliable train/validation/test labels (default: 1.0).",
    )
    parser.add_argument(
        "--candidate-confidence",
        type=float,
        default=0.80,
        help="Minimum predicted-class probability for a high-confidence contradiction.",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=TIE_TOLERANCE,
        help="Macro F1 margin within which the reporting model is chosen by balanced "
             "accuracy and non-human recall instead of macro F1 alone (default: 0.01).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Folds used for out-of-fold predictions (default: 5).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small parameter grids for a fast pipeline smoke test.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Fail fast on nonsensical CLI input rather than producing a silently wrong run."""
    if not 0.0 <= args.min_label_confidence <= 1.0:
        raise ValueError("--min-label-confidence must be between 0 and 1.")
    if not 0.0 <= args.candidate_confidence <= 1.0:
        raise ValueError("--candidate-confidence must be between 0 and 1.")
    if not 0.0 <= args.tie_tolerance <= 1.0:
        raise ValueError("--tie-tolerance must be between 0 and 1.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    processed = root / "data" / "processed"
    paths = {
        "train": processed / "twitter_train.csv",
        "validation": processed / "twitter_validation.csv",
        "test": processed / "twitter_test.csv",
        "full": processed / "twitter_full.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n  " + "\n  ".join(missing))

    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    for name in ("train", "validation", "test"):
        if TARGET not in frames[name]:
            raise ValueError(f"{paths[name]} does not contain {TARGET!r}.")
    return frames


def infer_feature_columns(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Use the split-file schema as the authority and prevent label/ID leakage."""
    features = [c for c in frames["train"].columns if c not in NON_FEATURE_COLUMNS]
    if not features:
        raise ValueError("No feature columns were found in twitter_train.csv.")

    for split in ("validation", "test", "full"):
        missing = sorted(set(features) - set(frames[split].columns))
        if missing:
            raise ValueError(f"{split} is missing model features: {missing}")

    suspicious = [c for c in features if "gender" in c.lower() or c.lower() == "human"]
    if suspicious:
        raise ValueError(f"Possible target leakage in feature columns: {suspicious}")
    return features


def coerce_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df.loc[:, features].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def reliable_mask(df: pd.DataFrame, min_confidence: float) -> pd.Series:
    confidence = pd.to_numeric(df["gender:confidence"], errors="coerce")
    conflict = pd.to_numeric(df["label_conflict"], errors="coerce").fillna(0)
    return df[TARGET].notna() & confidence.ge(min_confidence) & conflict.eq(0)


def build_search_spaces(quick: bool) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(random_state=SEED)),
    ])
    knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_jobs=-1)),
    ])
    # GaussianNB is scale-invariant (it fits a per-class mean/variance per
    # feature), so no StandardScaler step is needed here.
    nb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GaussianNB()),
    ])
    mlp = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            MLPClassifier(
                random_state=SEED,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                max_iter=500,
            ),
        ),
    ])

    if quick:
        return {
            "Decision Tree": (tree, {"model__max_depth": [8], "model__min_samples_leaf": [10], "model__class_weight": ["balanced"]}),
            "KNN": (knn, {"model__n_neighbors": [21], "model__weights": ["distance"], "model__p": [2]}),
            "Naive Bayes": (nb, {"model__var_smoothing": [1e-9]}),
            "MLP": (mlp, {"model__hidden_layer_sizes": [(64,)], "model__alpha": [0.001], "model__learning_rate_init": [0.001]}),
        }

    return {
        "Decision Tree": (
            tree,
            {
                "model__max_depth": [5, 8, 12, 16, None],
                "model__min_samples_leaf": [1, 5, 10, 25],
                "model__criterion": ["gini", "entropy"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            knn,
            {
                "model__n_neighbors": [5, 11, 21, 31, 51],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "Naive Bayes": (
            nb,
            {
                # var_smoothing is GaussianNB's analogue of the Laplace/additive
                # smoothing covered in the Week 5 slides: it adds a small amount
                # of variance to every feature so a class with zero observed
                # variance for a feature does not force P(a_j | c_i) to zero.
                "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            },
        ),
        "MLP": (
            mlp,
            {
                "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "model__alpha": [0.0001, 0.001, 0.01],
                "model__learning_rate_init": [0.0005, 0.001],
            },
        ),
    }


def positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def predictions_at_threshold(prob_human: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_human >= threshold).astype(int)


def find_best_threshold(y_true: pd.Series, prob_human: np.ndarray) -> tuple[float, float]:
    """Choose the human-probability threshold that maximises validation macro F1."""
    candidates = np.arange(0.20, 0.801, 0.01)
    scores = [f1_score(y_true, predictions_at_threshold(prob_human, t), average="macro") for t in candidates]
    best_index = int(np.argmax(scores))
    return float(candidates[best_index]), float(scores[best_index])


def tune_on_validation(
    name: str,
    estimator: Pipeline,
    grid: dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, dict, float, pd.DataFrame]:
    """Select hyperparameters and decision threshold using validation macro F1."""
    rows: list[dict] = []
    best: tuple[float, float, Pipeline, dict] | None = None

    for params in ParameterGrid(grid):
        model = clone(estimator).set_params(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        probability = positive_probability(model, X_val)
        threshold, score = find_best_threshold(y_val, probability)
        rows.append({"model": name, "validation_macro_f1": score, "threshold": threshold, **params})

        # If scores tie, prefer the threshold closest to the conventional 0.5.
        ranking = (score, -abs(threshold - 0.5))
        if best is None or ranking > (best[0], best[1]):
            best = (score, -abs(threshold - 0.5), model, params | {"decision_threshold": threshold})

    assert best is not None
    results = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    best_model = best[2]
    best_details = best[3]
    return best_model, best_details, float(best_details["decision_threshold"]), results


def metric_row(
    name: str,
    y_true: pd.Series,
    prob_human: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    pred = predictions_at_threshold(prob_human, threshold)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )
    y_brand = 1 - np.asarray(y_true, dtype=int)
    prob_brand = 1 - prob_human
    row = {
        "model": name,
        "decision_threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "weighted_f1": f1_score(y_true, pred, average="weighted"),
        "non_human_precision": precision[0],
        "non_human_recall": recall[0],
        "non_human_f1": f1[0],
        "human_precision": precision[1],
        "human_recall": recall[1],
        "human_f1": f1[1],
        "non_human_support": int(support[0]),
        "human_support": int(support[1]),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": roc_auc_score(y_true, prob_human),
        "non_human_average_precision": average_precision_score(y_brand, prob_brand),
        "brier_score": brier_score_loss(y_true, prob_human),
        "log_loss": log_loss(y_true, np.column_stack([prob_brand, prob_human]), labels=[0, 1]),
    }
    return row, pred


def select_best_model(metrics: pd.DataFrame, tolerance: float = TIE_TOLERANCE) -> str:
    """Pick the reporting model by test macro F1, breaking near-ties by class balance.

    Macro F1 is the primary criterion because the classes are imbalanced. But
    when two or more models land within `tolerance` of the top score, sorting
    by macro F1 alone can silently pick a model that reaches its score mostly
    by predicting the majority class well while under-recalling the minority
    (non-human) class - exactly the class this project cares about flagging.
    In that near-tie case, prefer the model with better balanced accuracy and
    non-human recall instead. This makes the "prefer non-human recall and
    interpretability when scores are close" rule from the project's design
    notes an explicit, reproducible rule in code rather than an ad-hoc
    judgement call made after the fact.
    """
    candidates = metrics.loc[metrics["model"].ne("Dummy")].sort_values("macro_f1", ascending=False)
    if candidates.empty:
        raise ValueError("No non-Dummy models available to select a reporting model from.")

    top_score = candidates.iloc[0]["macro_f1"]
    close = candidates.loc[candidates["macro_f1"] >= top_score - tolerance]
    if len(close) == 1:
        return str(close.iloc[0]["model"])

    ranked = close.sort_values(["balanced_accuracy", "non_human_recall"], ascending=False)
    return str(ranked.iloc[0]["model"])


def save_confusion_matrix(name: str, y_true: pd.Series, pred: np.ndarray, out: Path) -> None:
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    row_totals = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    annotations = np.array([
        [f"{cm[i, j]}\n{normalised[i, j]:.1%}" for j in range(2)] for i in range(2)
    ])
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    sns.heatmap(
        normalised,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=["Non-human", "Human"],
        yticklabels=["Non-human", "Human"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Recorded label")
    ax.set_title(f"{name} confusion matrix")
    fig.tight_layout()
    fig.savefig(out / f"confusion_matrix_{slug(name)}.png", dpi=220)
    plt.close(fig)


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def save_curves(
    y_true: pd.Series,
    model_probabilities: dict[str, np.ndarray],
    out: Path,
) -> None:
    y_brand = 1 - np.asarray(y_true, dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for name, prob_human in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, prob_human)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, prob_human):.3f})")

        prob_brand = 1 - prob_human
        precision, recall, _ = precision_recall_curve(y_brand, prob_brand)
        ap = average_precision_score(y_brand, prob_brand)
        axes[1].plot(recall, precision, label=f"{name} (AP={ap:.3f})")

        observed, predicted = calibration_curve(y_true, prob_human, n_bins=10, strategy="quantile")
        axes[2].plot(predicted, observed, marker="o", label=name)

    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].axhline(y_brand.mean(), linestyle="--", color="grey", label="Brand prevalence")
    axes[1].set(title="Precision-recall curve for non-human", xlabel="Recall", ylabel="Precision")
    axes[2].plot([0, 1], [0, 1], "--", color="grey")
    axes[2].set(title="Human-probability calibration", xlabel="Mean predicted probability", ylabel="Observed proportion")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "roc_pr_calibration_curves.png", dpi=220)
    plt.close(fig)


def save_model_comparison(metrics: pd.DataFrame, out: Path) -> None:
    chosen = ["balanced_accuracy", "macro_f1", "non_human_recall", "human_recall", "roc_auc"]
    plot_data = metrics.set_index("model")[chosen]
    ax = plot_data.plot.bar(figsize=(10, 5), ylim=(0, 1), width=0.8)
    ax.set_ylabel("Score")
    ax.set_title("Classification model comparison on the untouched test set")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "model_comparison.png", dpi=220)
    plt.close()


def save_classification_data_audit(
    frames: dict[str, pd.DataFrame],
    masks: dict[str, pd.Series],
    features: list[str],
    out: Path,
) -> None:
    """Document the exact data entering classification after shared preprocessing."""
    split_rows = []
    feature_rows = []
    for split in ("train", "validation", "test"):
        frame = frames[split]
        reliable = frame.loc[masks[split]]
        counts = reliable[TARGET].astype(int).value_counts()
        split_rows.append({
            "split": split,
            "all_rows": len(frame),
            "reliable_rows_used": len(reliable),
            "rows_held_out_as_uncertain": len(frame) - len(reliable),
            "non_human_rows": int(counts.get(0, 0)),
            "human_rows": int(counts.get(1, 0)),
            "human_rate": float(counts.get(1, 0) / len(reliable)) if len(reliable) else np.nan,
        })
        numeric = coerce_features(reliable, features)
        for feature in features:
            feature_rows.append({
                "split": split,
                "feature": feature,
                "missing_after_numeric_conversion": int(numeric[feature].isna().sum()),
                "unique_values": int(numeric[feature].nunique(dropna=True)),
                "mean": numeric[feature].mean(),
                "std": numeric[feature].std(),
                "minimum": numeric[feature].min(),
                "maximum": numeric[feature].max(),
            })
    pd.DataFrame(split_rows).to_csv(out / "classification_split_summary.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out / "classification_feature_audit.csv", index=False)


def save_error_analysis(
    test_output: pd.DataFrame,
    model_names: list[str],
    out: Path,
) -> None:
    """Summarise systematic errors by the original male/female/brand annotation."""
    rows = []
    for name in model_names:
        prefix = slug(name)
        pred_col = f"{prefix}_prediction"
        test_output[f"{prefix}_correct"] = test_output[pred_col].eq(test_output[TARGET]).astype(int)
        for gender, group in test_output.groupby("gender", dropna=False):
            rows.append({
                "model": name,
                "recorded_gender": gender,
                "records": len(group),
                "correct": int(group[f"{prefix}_correct"].sum()),
                "errors": int((1 - group[f"{prefix}_correct"]).sum()),
                "accuracy": group[f"{prefix}_correct"].mean(),
                "mean_probability_human": group[f"{prefix}_prob_human"].mean(),
            })
    pd.DataFrame(rows).to_csv(out / "test_error_analysis_by_recorded_gender.csv", index=False)


def save_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    out: Path,
) -> None:
    """Report Decision Tree feature importance regardless of which model is
    selected for headline metrics - interpretation stays anchored to the one
    model built to be read as rules, per the project's design notes."""
    tree_importance = pd.DataFrame({
        "feature": features,
        "tree_importance": model.named_steps["model"].feature_importances_,
    })
    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
    )
    tree_importance["permutation_importance_mean"] = permutation.importances_mean
    tree_importance["permutation_importance_std"] = permutation.importances_std
    tree_importance = tree_importance.sort_values("permutation_importance_mean", ascending=False)
    tree_importance.to_csv(out / "decision_tree_feature_importance.csv", index=False)

    top = tree_importance.head(15).sort_values("permutation_importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top["feature"],
        top["permutation_importance_mean"],
        xerr=top["permutation_importance_std"],
        color="#4472C4",
        alpha=0.9,
    )
    ax.set_xlabel("Decrease in test macro F1 after permutation")
    ax.set_title("Decision Tree feature importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "decision_tree_feature_importance.png", dpi=220)
    plt.close(fig)


def add_model_predictions(
    result: pd.DataFrame,
    row_index: pd.Index,
    name: str,
    prob_human: np.ndarray,
    threshold: float,
) -> None:
    prefix = slug(name)
    pred = predictions_at_threshold(prob_human, threshold)
    confidence = np.where(pred == 1, prob_human, 1 - prob_human)
    result.loc[row_index, f"{prefix}_prob_human"] = prob_human
    result.loc[row_index, f"{prefix}_prediction"] = pred
    result.loc[row_index, f"{prefix}_confidence"] = confidence


def _largest_vote_block(row: pd.Series) -> int:
    """Size of the largest agreeing group of model predictions for one profile.

    Using row.mode().iloc[0] here would silently favour whichever class
    pandas' mode() happens to list first on an exact tie (0 before 1), which
    only mattered once an even number of models (four, after adding Naive
    Bayes) made a 2-2 tie possible. Counting the largest block directly is
    correct regardless of which class it belongs to.
    """
    return int(row.value_counts().max())


def create_candidate_outputs(
    full: pd.DataFrame,
    features: list[str],
    selected_models: dict[str, Pipeline],
    thresholds: dict[str, float],
    min_label_confidence: float,
    candidate_confidence: float,
    cv_folds: int,
    out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create leakage-conscious OOF predictions and a ranked disagreement table."""
    identifiers = [c for c in ["_unit_id", "name", "gender", "gender:confidence", "label_conflict", TARGET] if c in full]
    result = full[identifiers].copy()
    X_full = coerce_features(full, features)
    labelled = full[TARGET].notna()
    reliable = reliable_mask(full, min_label_confidence)
    review_pool = labelled & ~reliable
    unknown = ~labelled

    y_reliable = full.loc[reliable, TARGET].astype(int)
    minimum_class = int(y_reliable.value_counts().min())
    folds = min(cv_folds, minimum_class)
    if folds < 2:
        raise ValueError("Not enough reliable examples in both classes for out-of-fold prediction.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for name, final_model in selected_models.items():
        threshold = thresholds[name]
        # Reliable labelled rows receive out-of-fold probabilities, avoiding in-sample predictions.
        oof = cross_val_predict(
            clone(final_model),
            X_full.loc[reliable],
            y_reliable,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )
        class_order = list(final_model.classes_)
        oof_prob_human = oof[:, class_order.index(1)]
        add_model_predictions(result, full.index[reliable], name, oof_prob_human, threshold)

        # Low-confidence/conflicting labels and unknown labels were excluded from fitting.
        external = review_pool | unknown
        if external.any():
            prob_human = positive_probability(final_model, X_full.loc[external])
            add_model_predictions(result, full.index[external], name, prob_human, threshold)

    prediction_cols = [f"{slug(name)}_prediction" for name in selected_models]
    confidence_cols = [f"{slug(name)}_confidence" for name in selected_models]
    probability_cols = [f"{slug(name)}_prob_human" for name in selected_models]

    result["ensemble_prob_human"] = result[probability_cols].mean(axis=1)
    # On an exact tie in prediction_cols, mean(...) >= 0.5 defaults the
    # consensus to human (the majority class), since a tie is inconclusive
    # either way and human is the better-supported prior.
    result["consensus_prediction"] = (result[prediction_cols].mean(axis=1) >= 0.5).astype(int)
    result["consensus_label"] = result["consensus_prediction"].map(LABEL_NAMES)
    result["consensus_votes"] = result[prediction_cols].apply(_largest_vote_block, axis=1)

    for name in selected_models:
        prefix = slug(name)
        result[f"{prefix}_contradiction"] = (
            labelled
            & result[f"{prefix}_prediction"].ne(result[TARGET])
        ).astype(int)
        result[f"{prefix}_strong_contradiction"] = (
            result[f"{prefix}_contradiction"].eq(1)
            & result[f"{prefix}_confidence"].ge(candidate_confidence)
        ).astype(int)

    contradiction_cols = [f"{slug(name)}_contradiction" for name in selected_models]
    strong_cols = [f"{slug(name)}_strong_contradiction" for name in selected_models]
    result["models_contradicting"] = result[contradiction_cols].sum(axis=1)
    result["strong_models_contradicting"] = result[strong_cols].sum(axis=1)
    result["recorded_label"] = result[TARGET].map(LABEL_NAMES)
    result["prediction_source"] = np.select(
        [reliable, review_pool, unknown],
        ["out-of-fold", "held-out candidate", "unknown-label prediction"],
        default="unavailable",
    )

    # "A majority of the compared models" rather than a hardcoded count: with
    # four models (Decision Tree, KNN, Naive Bayes, MLP) a majority is three,
    # not the "two" that was correct back when there were three models. This
    # threshold is derived from however many models were actually trained, so
    # adding or removing a model can never silently loosen the flagging rule.
    num_models = len(selected_models)
    majority_threshold = num_models // 2 + 1

    result["classification_review_status"] = "not flagged"
    result.loc[labelled & result["models_contradicting"].ge(majority_threshold), "classification_review_status"] = "review"
    result.loc[labelled & result["strong_models_contradicting"].ge(majority_threshold), "classification_review_status"] = "strong candidate"

    result.to_csv(out / "all_profile_classification_predictions.csv", index=False)
    unknown_output = result.loc[unknown].sort_values("consensus_votes", ascending=False)
    unknown_output.to_csv(out / "unknown_profile_predictions.csv", index=False)

    candidates = result.loc[result["classification_review_status"].ne("not flagged")].copy()
    candidates["original_confidence"] = pd.to_numeric(candidates["gender:confidence"], errors="coerce")
    candidates = candidates.sort_values(
        ["strong_models_contradicting", "models_contradicting", "original_confidence"],
        ascending=[False, False, True],
    )
    candidates.to_csv(out / "potential_mislabels_classification.csv", index=False)
    return result, candidates


def write_analysis_summary(
    metrics: pd.DataFrame,
    best_name: str,
    candidates: pd.DataFrame,
    counts: dict,
    feature_count: int,
    tie_tolerance: float,
    out: Path,
) -> None:
    best = metrics.set_index("model").loc[best_name]
    candidate_counts = candidates.get("recorded_label", pd.Series(dtype=str)).value_counts()
    lines = [
        "# Classification analysis summary",
        "",
        "## Experimental design",
        "",
        f"The analysis used {feature_count} engineered metadata features and treated human as male/female (1) and non-human as brand (0). "
        f"Only records with crowd-label confidence at or above {counts['minimum_confidence']:.2f} and no recorded label conflict were used as reliable supervised examples.",
        "",
        f"The fixed splits contained {counts['train']} reliable training records, {counts['validation']} reliable validation records and {counts['test']} reliable test records. "
        "Hyperparameters and decision thresholds were selected using validation macro F1; the test split was used once for final comparison.",
        "",
        "## Model comparison",
        "",
        "A Dummy Classifier established the majority-class baseline. Decision Tree, KNN, Naive Bayes and MLPClassifier were compared: they represent "
        "interpretable rule-based, local similarity-based, probabilistic and nonlinear learning approaches covered in the subject materials.",
        "",
        f"The reporting model was chosen by test macro F1, with any model scoring within {tie_tolerance:.3f} of the top score re-ranked by balanced accuracy "
        "and non-human recall rather than macro F1 alone, since the non-human class is the one this project is specifically trying to catch. "
        f"The selected model was **{best_name}**, with test macro F1 of {best['macro_f1']:.3f}, balanced accuracy of {best['balanced_accuracy']:.3f}, "
        f"ROC-AUC of {best['roc_auc']:.3f}, non-human recall of {best['non_human_recall']:.3f}, and human recall of {best['human_recall']:.3f}.",
        "",
        "Macro F1 was the primary selection metric because the classes were imbalanced and ordinary accuracy could favour the larger human class. "
        "Balanced accuracy and class-specific recall show whether performance is distributed across both classes. ROC-AUC and non-human average precision "
        "assess ranking quality, while Brier score and calibration curves assess whether predicted probabilities are reliable enough for candidate prioritisation.",
        "",
        "## Potentially mislabelled profiles",
        "",
        "A disagreement was treated as evidence for review rather than proof of an incorrect label. Reliable labelled profiles received out-of-fold predictions, "
        "while low-confidence, conflicting and unknown records were predicted by models that had not trained on those records. A strong candidate required "
        "a majority of the compared models to contradict the recorded label with predicted-class probability above the configured threshold.",
        "",
        f"The classification analysis flagged {len(candidates)} profiles for review, including {int(candidate_counts.get('human', 0))} recorded human profiles "
        f"and {int(candidate_counts.get('non-human', 0))} recorded non-human profiles.",
        "",
        "These candidates should be combined with association-rule, clustering and text-analysis evidence before recommending a label amendment. "
        "Model disagreement can also reflect unusual but correctly labelled accounts, incomplete profile information, or noise in a single sampled tweet.",
        "",
        "## Limitations",
        "",
        "The recorded labels are crowd annotations rather than verified ground truth. Each account contributes only one sampled tweet, and profile "
        "characteristics may have changed since collection. Colour and timezone variables may encode platform defaults rather than identity. "
        "Naive Bayes assumes the features are conditionally independent given the class; several engineered features are correlated (tweet rate with "
        "tweet count, favourites rate with favourite count, and the sidebar RGB channels with each other), so its probability estimates are less reliable "
        "than the other models' even in periods where its accuracy is competitive. Text content (tweet and description text) was not used at this "
        "classification stage; a text-based model would need those columns carried through the preprocessing split files. "
        "The results therefore identify profiles requiring review, not confirmed annotation errors.",
    ]
    (out / "classification_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_output_manifest(out: Path) -> None:
    """List every file this run produced, for traceability in the group submission."""
    files = sorted(
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file() and p.name != "output_manifest.json"
    )
    (out / "output_manifest.json").write_text(json.dumps(files, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    root = args.project_root.resolve()

    # Output lives under data/output, matching 03_association_rules.py, so both
    # scripts write into one predictable location instead of two separate
    # top-level output trees.
    out = root / "data" / "output" / "classification"
    model_dir = out / "models"
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out)
    sns.set_theme(style="whitegrid")

    LOGGER.info("Project root: %s", root)
    LOGGER.info("Output folder: %s", out)

    frames = load_inputs(root)
    features = infer_feature_columns(frames)
    (out / "feature_columns.json").write_text(json.dumps(features, indent=2), encoding="utf-8")

    masks = {
        name: reliable_mask(frames[name], args.min_label_confidence)
        for name in ("train", "validation", "test")
    }
    for name, mask in masks.items():
        dropped = len(mask) - int(mask.sum())
        LOGGER.info("%s: using %s reliable rows; holding out %s uncertain/conflicting rows", name, f"{int(mask.sum()):,}", f"{dropped:,}")
    save_classification_data_audit(frames, masks, features, out)

    X_train = coerce_features(frames["train"].loc[masks["train"]], features)
    y_train = frames["train"].loc[masks["train"], TARGET].astype(int)
    X_val = coerce_features(frames["validation"].loc[masks["validation"]], features)
    y_val = frames["validation"].loc[masks["validation"], TARGET].astype(int)
    X_test = coerce_features(frames["test"].loc[masks["test"]], features)
    y_test = frames["test"].loc[masks["test"], TARGET].astype(int)

    for split_name, y in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        if y.nunique() != 2:
            raise ValueError(f"Reliable {split_name} data must contain both human and non-human labels.")

    # Majority-class baseline. Its fixed threshold is included for a consistent output schema.
    dummy = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])
    dummy.fit(X_train, y_train)
    dummy_prob = positive_probability(dummy, X_test)
    dummy_metrics, dummy_pred = metric_row("Dummy", y_test, dummy_prob, 0.5)
    save_confusion_matrix("Dummy", y_test, dummy_pred, out)

    selected_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    tuning_tables: list[pd.DataFrame] = []
    test_rows = [dummy_metrics]
    test_probabilities = {"Dummy": dummy_prob}
    test_predictions: dict[str, np.ndarray] = {"Dummy": dummy_pred}
    best_parameters: dict[str, dict] = {}

    spaces = build_search_spaces(args.quick)
    for name, (estimator, grid) in spaces.items():
        LOGGER.info("Tuning %s (%s configurations)...", name, len(list(ParameterGrid(grid))))
        _, details, threshold, tuning = tune_on_validation(
            name, estimator, grid, X_train, y_train, X_val, y_val
        )
        tuning_tables.append(tuning)
        params = {k: v for k, v in details.items() if k != "decision_threshold"}

        # Final test model uses train + validation after all choices are fixed.
        final_model = clone(estimator).set_params(**params)
        X_train_final = pd.concat([X_train, X_val], axis=0)
        y_train_final = pd.concat([y_train, y_val], axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            final_model.fit(X_train_final, y_train_final)

        probability = positive_probability(final_model, X_test)
        row, pred = metric_row(name, y_test, probability, threshold)
        test_rows.append(row)
        test_probabilities[name] = probability
        test_predictions[name] = pred
        selected_models[name] = final_model
        thresholds[name] = threshold
        best_parameters[name] = details
        joblib.dump(final_model, model_dir / f"{slug(name)}.joblib")
        save_confusion_matrix(name, y_test, pred, out)

        report = classification_report(
            y_test,
            pred,
            labels=[0, 1],
            target_names=["non-human", "human"],
            digits=4,
            zero_division=0,
        )
        (out / f"classification_report_{slug(name)}.txt").write_text(report, encoding="utf-8")

    tuning_results = pd.concat(tuning_tables, ignore_index=True)
    tuning_results.to_csv(out / "validation_tuning_results.csv", index=False)
    (out / "selected_hyperparameters.json").write_text(
        json.dumps(best_parameters, indent=2, default=str), encoding="utf-8"
    )

    metrics = pd.DataFrame(test_rows).sort_values("macro_f1", ascending=False)
    metrics.to_csv(out / "test_model_comparison.csv", index=False)
    save_model_comparison(metrics, out)
    save_curves(y_test, test_probabilities, out)

    best_name = select_best_model(metrics, tolerance=args.tie_tolerance)
    save_feature_importance(selected_models["Decision Tree"], X_test, y_test, features, out)

    test_output_columns = [c for c in ["_unit_id", "name", "gender", "gender:confidence", "label_conflict", TARGET] if c in frames["test"]]
    test_output = frames["test"].loc[masks["test"], test_output_columns].copy()
    for name, probability in test_probabilities.items():
        threshold = 0.5 if name == "Dummy" else thresholds[name]
        test_output[f"{slug(name)}_prob_human"] = probability
        test_output[f"{slug(name)}_prediction"] = predictions_at_threshold(probability, threshold)
    save_error_analysis(test_output, list(test_probabilities), out)
    test_output.to_csv(out / "test_predictions.csv", index=False)

    all_predictions, candidates = create_candidate_outputs(
        frames["full"],
        features,
        selected_models,
        thresholds,
        args.min_label_confidence,
        args.candidate_confidence,
        args.cv_folds,
        out,
    )

    counts = {
        "train": int(masks["train"].sum()),
        "validation": int(masks["validation"].sum()),
        "test": int(masks["test"].sum()),
        "minimum_confidence": args.min_label_confidence,
    }
    write_analysis_summary(metrics, best_name, candidates, counts, len(features), args.tie_tolerance, out)
    write_output_manifest(out)

    LOGGER.info("Final test comparison:\n%s", metrics[["model", "accuracy", "balanced_accuracy", "macro_f1", "non_human_recall", "human_recall", "roc_auc"]].round(4).to_string(index=False))
    LOGGER.info("Selected model by test macro F1 (tie-tolerance %.3f) for reporting: %s", args.tie_tolerance, best_name)
    LOGGER.info("Classification review candidates: %s", f"{len(candidates):,}")
    LOGGER.info("Outputs written to: %s", out)


if __name__ == "__main__":
    main()
