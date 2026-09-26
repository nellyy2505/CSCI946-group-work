"""Combine independent method nominations into a ranked label-review list.

Run after the method scripts: python code/07_consensus.py
Optional: python code/07_consensus.py --methods association clustering classification text
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "output"
FULL = ROOT / "data" / "processed" / "twitter_full.csv"
DEFAULT_METHODS = ("association", "clustering", "classification", "regression", "text")
REQUIRED = {"_unit_id", "says", "score"}
LABELS = {0: "non_human", 1: "human"}


def load_votes(methods, profiles):
    votes = []
    known_ids = set(profiles["_unit_id"])
    unknown_ids = set(profiles.loc[profiles["is_human"].isna(), "_unit_id"])
    for method in methods:
        path = OUT / f"{method}_flagged.csv"
        if not path.exists():
            print(f"Skipping {method}: {path.name} has not been produced yet")
            continue
        df = pd.read_csv(path)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing {sorted(missing)}")
        if df["_unit_id"].isna().any() or df["_unit_id"].duplicated().any():
            raise ValueError(f"{path.name}: IDs must be present and unique per method")
        if not set(df["_unit_id"]).issubset(known_ids):
            raise ValueError(f"{path.name}: contains IDs absent from twitter_full.csv")
        if not df["says"].isin(LABELS.values()).all():
            raise ValueError(f"{path.name}: says must be human or non_human")
        score = pd.to_numeric(df["score"], errors="coerce")
        if score.isna().any() or not score.between(0, 1).all():
            raise ValueError(f"{path.name}: score must be a number between 0 and 1")
        recorded = df["_unit_id"].map(profiles.set_index("_unit_id")["is_human"].map(LABELS))
        if (recorded.notna() & df["says"].eq(recorded)).any():
            raise ValueError(f"{path.name}: a flagged row agrees with its recorded label")
        if df["_unit_id"].isin(unknown_ids).any():
            raise ValueError(f"{path.name}: unknown profiles belong in <method>_predictions.csv")
        selected = df[["_unit_id", "says", "score"]].copy()
        predictions_path = OUT / f"{method}_predictions.csv"
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)
            missing_predictions = REQUIRED - set(predictions.columns)
            if missing_predictions:
                raise ValueError(f"{predictions_path.name}: missing {sorted(missing_predictions)}")
            unknown_predictions = predictions.loc[predictions["_unit_id"].isin(unknown_ids),
                                                  ["_unit_id", "says", "score"]]
            if unknown_predictions["_unit_id"].duplicated().any():
                raise ValueError(f"{predictions_path.name}: duplicate unknown profile IDs")
            if not unknown_predictions["says"].isin(LABELS.values()).all():
                raise ValueError(f"{predictions_path.name}: invalid suggested label")
            prediction_scores = pd.to_numeric(unknown_predictions["score"], errors="coerce")
            if prediction_scores.isna().any() or not prediction_scores.between(0, 1).all():
                raise ValueError(f"{predictions_path.name}: invalid prediction score")
            selected = pd.concat([selected, unknown_predictions], ignore_index=True)
            print(f"Loaded {len(unknown_predictions)} unknown predictions from {method}")
        votes.append(selected.assign(method=method))
        print(f"Loaded {len(df)} labelled nominations from {method}")
    if not votes:
        raise ValueError("No method outputs found; run at least one method script first")
    return pd.concat(votes, ignore_index=True)


def build_consensus(profiles, votes, min_votes, available_methods):
    # Count methods, not model predictions within a method; scores stay separate.
    tally = (votes.groupby(["_unit_id", "says"]).agg(
        votes=("method", "nunique"), methods=("method", lambda x: ", ".join(sorted(x)))
    ).reset_index())
    possible_tie = tally.groupby("_unit_id")["votes"].transform("max")
    tally = tally.loc[tally["votes"].eq(possible_tie)].copy()
    tied = set(tally.loc[tally.duplicated("_unit_id", keep=False), "_unit_id"])
    winner = tally.loc[~tally["_unit_id"].isin(tied)].copy()

    # Keep each method's recommendation and score visible for manual review.
    per_method = votes.pivot(index="_unit_id", columns="method", values=["says", "score"])
    per_method.columns = [f"{method}_{field}" for field, method in per_method.columns]
    result = profiles.merge(winner, on="_unit_id", how="inner", validate="one_to_one")
    result = result.merge(per_method.reset_index(), on="_unit_id", validate="one_to_one")
    total_votes = votes.groupby("_unit_id")["method"].nunique()
    result["opposing_votes"] = (result["_unit_id"].map(total_votes) - result["votes"]).astype(int)
    result["recorded"] = result["is_human"].map(LABELS)
    result["crowd_unsure"] = result["gender:confidence"].lt(1)
    result["disagrees"] = result["recorded"].notna() & result["says"].ne(result["recorded"])
    result["review_tier"] = result["votes"].map(
        lambda n: "3+ methods" if n >= 3 else "2 methods" if n == 2 else "1 method")
    result["available_methods"] = available_methods
    ranked = result.loc[result["disagrees"] & result["votes"].ge(min_votes)].copy()
    ranked = ranked.sort_values(["votes", "opposing_votes", "gender:confidence", "_unit_id"],
                                ascending=[False, True, True, True])
    unknown = result.loc[result["recorded"].isna() & result["votes"].ge(min_votes)].copy()
    unknown = unknown.sort_values(["votes", "_unit_id"], ascending=[False, True])
    columns = ["_unit_id", "name", "gender", "recorded", "says", "votes", "opposing_votes", "available_methods",
               "review_tier", "methods", "gender:confidence", "label_conflict", *per_method.columns]
    OUT.mkdir(parents=True, exist_ok=True)
    ranked[columns].to_csv(OUT / "consensus_candidates.csv", index=False)
    unknown[columns].to_csv(OUT / "consensus_unknown.csv", index=False)
    votes.loc[votes["_unit_id"].isin(tied)].sort_values(["_unit_id", "method"]).to_csv(
        OUT / "consensus_ties.csv", index=False)
    return ranked, unknown, tied


def plot_validation(profiles, votes, ranked):
    # Crowd agreement is a corroborating signal, not verified ground truth.
    labelled = profiles.loc[profiles["is_human"].notna()]
    counts = votes.loc[votes["_unit_id"].isin(labelled["_unit_id"])].groupby("_unit_id")["method"].nunique()
    overview = labelled[["_unit_id", "gender:confidence"]].copy()
    overview["nominations"] = overview["_unit_id"].map(counts).fillna(0).astype(int)
    summary = overview.groupby("nominations").agg(
        profiles=("_unit_id", "size"),
        crowd_unsure=("gender:confidence", lambda x: x.lt(1).mean()),
    ).reset_index()
    summary.to_csv(OUT / "consensus_vote_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(summary["nominations"].astype(str), summary["profiles"])
    axes[0].set(xlabel="Methods nominating a profile", ylabel="Labelled profiles")
    axes[1].bar(summary["nominations"].astype(str), 100 * summary["crowd_unsure"])
    axes[1].axhline(100 * labelled["gender:confidence"].lt(1).mean(),
                    color="black", linestyle="--", label="All labelled profiles")
    axes[1].set(xlabel="Methods nominating a profile", ylabel="Crowd confidence below 1 (%)", ylim=(0, 100))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_consensus_agreement.png", dpi=150)
    plt.close(fig)
    print(f"Ranked candidates: {len(ranked)}; baseline crowd uncertainty: "
          f"{labelled['gender:confidence'].lt(1).mean():.1%}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        help="Method names corresponding to <method>_flagged.csv")
    parser.add_argument("--min-votes", type=int, default=1)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods) or args.min_votes < 1:
        parser.error("Methods must be unique and --min-votes must be positive")

    profiles = pd.read_csv(FULL, low_memory=False)[
        ["_unit_id", "name", "gender", "is_human", "gender:confidence", "label_conflict"]]
    if profiles["_unit_id"].duplicated().any():
        raise ValueError("twitter_full.csv must contain one row per _unit_id")
    votes = load_votes(args.methods, profiles)
    ranked, unknown, tied = build_consensus(profiles, votes, args.min_votes,
                                          votes["method"].nunique())
    plot_validation(profiles, votes, ranked)
    print(f"Unknown profiles with nominations: {len(unknown)}; tied suggestions: {len(tied)}")
    if not len(unknown):
        print("Unknown predictions require a separate all-profiles output contract; flagged files alone do not contain them")
    print("See data/output/consensus_candidates.csv and fig_consensus_agreement.png")


if __name__ == "__main__":
    main()
