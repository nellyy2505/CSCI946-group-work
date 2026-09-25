# Consensus voting — draft for team review

## Purpose

Rank profiles for **manual label review**. A nomination is not proof of a wrong label.
This stage combines independent evidence rather than retraining or relabelling profiles.

## Inputs

- `data/processed/twitter_full.csv`: one row per `_unit_id`, containing `is_human`,
  `gender:confidence`, name and label-conflict metadata.
- One `data/output/<method>_flagged.csv` per participating method. Every row has
  `_unit_id`, `says` (`human` or `non_human`) and `score` in `[0, 1]`.
- Each method emits at most one nomination for a profile. A missing row is an
  **abstention**, not a vote for its recorded label.
- Classification may compare several models internally, but only its selected
  model casts the classification vote. Nancy confirmed that this model is
  currently MLP, so logistic regression may cast a separate regression vote.
  Recheck the selected model if classification is rerun. Text and structured
  classification use different feature views; their overlap should still be
  explained in the report.
- For unknown profiles, `data/output/<method>_predictions.csv` may supply
  `_unit_id`, `says`, and `score`. The script reads only its unknown rows;
  labelled profiles continue to use the method's flagged file. Nelly is
  updating `APPROACH.md` and method exports, so confirm final filenames and
  columns when those changes land.

## Decision rule

1. Give each included method one equal vote for its `says` label. Do not add
   probabilities, rule confidence and cluster purity: their scales differ.
2. For each profile, the label with more votes is the provisional suggestion.
   A tie is unresolved and listed separately. Show any votes against the winner.
3. For a labelled profile, include the suggestion as a candidate only if it
   differs from the recorded `is_human` label.
4. Review tiers: **3+ methods**, **2 methods**, **1 method**. Rank within the list
   by winning vote count (descending), opposing votes (ascending), then crowd
   confidence (ascending) and `_unit_id`. These are review priorities, not
   calibrated probabilities. Missing method files reduce the available number
   of voters; report that number with each result.
5. `gender:confidence` is used only for ranking and descriptive corroboration;
   it does not cast a vote. It is not an independent ground truth, particularly
   where a method chose its training rows using this field.

## Outputs

- `consensus_candidates.csv`: review list including all per-method votes and scores.
- `consensus_ties.csv`: unresolved opposing nominations.
- `consensus_unknown.csv`: provisional suggestions for unknown-labelled accounts
  from available full-prediction files. A one-method suggestion has no
  cross-method corroboration.
- `consensus_vote_summary.csv` and `fig_consensus_agreement.png`: number of
  labelled profiles at each nomination count and fraction with crowd confidence
  below one. The figure shows an association, not confirmed labelling accuracy.

Run `python code/07_consensus.py` from the repository root after the methods.
Use `--methods association clustering classification text` to name the intended
voters, or `--min-votes 2` to export only corroborated candidates.
