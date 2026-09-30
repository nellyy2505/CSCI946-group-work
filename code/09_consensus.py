"""Combine independent method votes into a ranked label-review list.

Run after the method scripts: python code/09_consensus.py
Optional: python code/09_consensus.py --methods association clustering classification regression text

Each method writes <method>_predictions.csv with a `votes` column (1 = confident enough to vote).
A vote can agree or disagree with the recorded label; the majority is the suggestion.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update({"font.size": 11})   # figures are placed at 6.2 in wide in the report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "output"
FULL = ROOT / "data" / "processed" / "twitter_full.csv"
DEFAULT_METHODS = ("association", "clustering", "classification", "regression", "text")
REQUIRED = {"_unit_id", "says", "score", "votes"}
LABELS = {0: "non_human", 1: "human"}


def load_votes(methods, profiles):
    votes = []
    known_ids = set(profiles["_unit_id"])
    recorded_of = profiles.set_index("_unit_id")["is_human"].map(LABELS)
    for method in methods:
        path = OUT / f"{method}_predictions.csv"
        if not path.exists():
            # a missing voter is an error, not a silent skip: the vote would be wrong without it
            raise FileNotFoundError(f"{path.name} has not been produced yet; run the {method} script first "
                                    f"or leave it out with --methods")
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
        if not df["votes"].isin([0, 1]).all():
            raise ValueError(f"{path.name}: votes must be 0 or 1")
        # only confident rows vote; the rest abstain
        selected = df.loc[df["votes"].eq(1), ["_unit_id", "says", "score"]].copy()
        recorded = selected["_unit_id"].map(recorded_of)
        labelled = recorded.notna()
        against = labelled & selected["says"].ne(recorded)
        votes.append(selected.assign(method=method))
        print(f"Loaded {len(selected)} votes from {method}: {labelled.sum()} on labelled profiles "
              f"({against.sum()} against the recorded label), {(~labelled).sum()} on unknown profiles")
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
    result["voters"] = result["votes"] + result["opposing_votes"]   # methods that voted on this profile
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
    columns = ["_unit_id", "name", "gender", "recorded", "says", "votes", "opposing_votes", "voters", "available_methods",
               "review_tier", "methods", "gender:confidence", "label_conflict", *per_method.columns]
    OUT.mkdir(parents=True, exist_ok=True)
    ranked[columns].to_csv(OUT / "consensus_candidates.csv", index=False)
    unknown[columns].to_csv(OUT / "consensus_unknown.csv", index=False)
    votes.loc[votes["_unit_id"].isin(tied)].sort_values(["_unit_id", "method"]).to_csv(
        OUT / "consensus_ties.csv", index=False)
    return ranked, unknown, tied


def plot_validation(profiles, votes, ranked):
    # Crowd agreement is a corroborating signal, not verified ground truth.
    labelled = profiles.loc[profiles["is_human"].notna()].copy()
    labelled["recorded"] = labelled["is_human"].map(LABELS)
    # a nomination is a vote against the recorded label; agreeing votes are not nominations
    recorded_of = labelled.set_index("_unit_id")["recorded"]
    against = votes.loc[votes["_unit_id"].isin(labelled["_unit_id"])].copy()
    against = against.loc[against["says"].ne(against["_unit_id"].map(recorded_of))]
    counts = against.groupby("_unit_id")["method"].nunique()
    overview = labelled[["_unit_id", "recorded", "gender:confidence"]].copy()
    overview["nominations"] = overview["_unit_id"].map(counts).fillna(0).astype(int)
    overview["unsure"] = overview["gender:confidence"].lt(1)
    # recorded brands are below full confidence far more often than recorded humans, so the fair
    # baseline for a group is the share expected from its own recorded-label mix
    unsure_by_label = overview.groupby("recorded")["unsure"].mean()
    overview["expected"] = overview["recorded"].map(unsure_by_label)
    overview["candidate"] = overview["_unit_id"].isin(ranked["_unit_id"])
    summary = overview.groupby("nominations").agg(
        profiles=("_unit_id", "size"),
        candidates=("candidate", "sum"),
        recorded_non_human=("recorded", lambda r: r.eq("non_human").mean()),
        crowd_unsure=("unsure", "mean"),
        expected_from_label_mix=("expected", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "consensus_vote_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.4))
    x = summary["nominations"].astype(str)
    bars = axes[0].bar(x, summary["profiles"], color="tab:gray")
    axes[0].bar_label(bars, padding=2)
    axes[0].set_yscale("log")
    axes[0].set_ylim(top=summary["profiles"].max() * 3)
    axes[0].set(xlabel="Methods nominating a profile", ylabel="Labelled profiles (log scale)")
    width, pos = 0.4, range(len(summary))
    observed = axes[1].bar([p - width / 2 for p in pos], summary["crowd_unsure"], width,
                           color="tab:red", label="observed")
    bottom = 0
    for name, colour in [("human", "tab:blue"), ("non_human", "tab:orange")]:
        share = summary["recorded_non_human"] if name == "non_human" else 1 - summary["recorded_non_human"]
        part = share * unsure_by_label[name]
        axes[1].bar([p + width / 2 for p in pos], part, width, bottom=bottom, color=colour,
                    label=f"same label mix: recorded {name}")
        bottom = bottom + part
    axes[1].bar_label(observed, fmt="%.2f", padding=2)
    axes[1].axhline(overview["unsure"].mean(), color="black", linestyle="--", label="All labelled profiles")
    axes[1].set_xticks(list(pos), [f"{k}\n(n={n})" for k, n in zip(x, summary["profiles"])])
    axes[1].set(xlabel="Methods nominating a profile", ylabel="Crowd confidence below 1 (share)", ylim=(0, 1.1))
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_consensus_agreement.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Ranked candidates: {len(ranked)}; baseline crowd uncertainty: "
          f"{overview['unsure'].mean():.1%} (recorded human {unsure_by_label['human']:.1%}, "
          f"recorded non-human {unsure_by_label['non_human']:.1%})")
    cand = overview.loc[overview["candidate"]]
    print(f"Candidates below full confidence: {cand['unsure'].mean():.1%} observed vs "
          f"{cand['expected'].mean():.1%} expected from their recorded-label mix")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        help="Method names corresponding to <method>_predictions.csv")
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
    print(f"Unknown profiles with a suggestion: {len(unknown)}; tied profiles: {len(tied)}")
    print("See data/output/consensus_candidates.csv and fig_consensus_agreement.png")


if __name__ == "__main__":
    main()
