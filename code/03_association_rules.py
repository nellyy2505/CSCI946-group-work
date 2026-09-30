# 03_association_rules.py  Apriori rules on profile items; rules concluding a label vote on each profile

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

plt.rcParams.update({"font.size": 11})   # figures are placed at 4-6.2 in wide in the report

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

FLAGS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
         "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]
BINNED = ["tweet_count", "fav_number", "tweets_per_day", "favs_per_day",
          "account_age_days", "text_len", "desc_len", "text_n_urls"]
COLOURS = {"human": "tab:blue", "non_human": "tab:orange"}
MIN_SUPPORT = 0.05
MIN_CONF, MIN_LIFT = 0.90, 1.3   # confidence cut-off: see the sweep in section 5


# 1. Load
# every profile gets items; only the labelled ones are used to mine the rules
df = pd.read_csv(PROC / "twitter_full.csv")
label = df["label"]
labelled = label != "unknown"
print("shape:", df.shape, "| labelled:", labelled.sum())


# 2. Basket
# one row per profile, one column per property it has or has not
items = pd.DataFrame(index=df.index)
for col in FLAGS:
    items[col] = df[col] == 1
items["has_description"] = df["desc_missing"] == 0
items["has_location"] = df["location_missing"] == 0
items["custom_link_colour"] = df["link_default"] == 0
items["custom_sidebar_colour"] = df["sidebar_default"] == 0
items["uploaded_image"] = df["default_image"] == 0

# numeric features need discretising - top and bottom quartile become items
for col in BINNED:
    low, high = df[col].quantile([.25, .75])
    items[col + "_low"] = df[col] <= low
    items[col + "_high"] = df[col] >= high

# the label as an item, so rules can conclude with it
items["human"] = label == "human"
items["non_human"] = label == "non_human"
print("items:", items.shape[1])
print(items.mean().sort_values(ascending=False).round(3))


# 3. Frequent itemsets and rules
# labelled rows only: an unknown row has no label item and would only lower rule confidence
frequent = apriori(items[labelled], min_support=MIN_SUPPORT, use_colnames=True,
                   max_len=4)   # longer itemsets are padded copies of shorter rules
print("frequent itemsets:", len(frequent))
print(frequent.sort_values("support", ascending=False).head(15))

# generate rules
rules = association_rules(frequent, metric="confidence", min_threshold=0.6)
rules = rules[rules["lift"] > 1]
print("rules:", len(rules))
print(rules.sort_values("lift", ascending=False).head(10)[
    ["antecedents", "consequents", "support", "confidence", "lift"]])

# support vs confidence, coloured by lift
plt.figure(figsize=(4.8, 3.6))
points = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", alpha=0.6)
plt.colorbar(points, label="lift")
plt.xlabel("support")
plt.ylabel("confidence")
plt.title("all rules")
plt.savefig(OUT / "fig_association_support_confidence.png", dpi=150, bbox_inches="tight")
plt.show()


# 4. Label rules
# the main result: rules that conclude a label
TARGETS = [frozenset(["human"]), frozenset(["non_human"])]
label_rules = rules[rules["consequents"].isin(TARGETS)].copy()
label_rules["rule"] = [" & ".join(sorted(a)) for a in label_rules["antecedents"]]
label_rules["says"] = ["human" if c == TARGETS[0] else "non_human" for c in label_rules["consequents"]]
label_rules = label_rules.sort_values(["lift", "confidence"], ascending=False)
print("rules concluding a label:", len(label_rules))

for name in ["human", "non_human"]:
    print("\nstrongest rules for", name, "(base rate %.3f)" % (label[labelled] == name).mean())
    for _, r in label_rules[label_rules["says"] == name].head(8).iterrows():
        print("  %-62s conf %.3f  lift %.2f  support %.3f"
              % (r["rule"], r["confidence"], r["lift"], r["support"]))

label_rules[["rule", "says", "support", "confidence", "lift"]].to_csv(
    OUT / "association_rules.csv", index=False)

# top rules by lift
top = label_rules.head(12).iloc[::-1]


def wrap_rule(rule, width=32):
    # break a long itemset between items so the label stays short
    lines = [""]
    for item in rule.split(" & "):
        if lines[-1] and len(lines[-1]) + len(item) + 2 > width:
            lines[-1] += " &"
            lines.append(item)
        else:
            lines[-1] = lines[-1] + " & " + item if lines[-1] else item
    return "\n".join(lines)


plt.figure(figsize=(7.4, 6.5))
plt.barh(range(len(top)), top["lift"], color=[COLOURS[s] for s in top["says"]])
plt.yticks(range(len(top)), [wrap_rule(r) for r in top["rule"]])
plt.axvline(1, color="red", linestyle="--", label="lift = 1")
plt.xlabel("lift")
plt.title("strongest rules for the label")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_association_top_by_lift.png", dpi=150, bbox_inches="tight")
plt.show()


# 5. Votes
# apply the strong rules to every profile, labelled or unknown
def rule_votes(min_conf, min_lift):
    # one class fires: take it, score = best confidence; both fire: abstain (base rates differ)
    strong = label_rules[(label_rules["confidence"] >= min_conf) & (label_rules["lift"] >= min_lift)]
    best = pd.DataFrame(0.0, index=df.index, columns=["human", "non_human"])
    matched = pd.Series(0, index=df.index)
    for _, r in strong.iterrows():
        match = items[sorted(r["antecedents"])].all(axis=1)
        matched[match] += 1                                   # strong rules matched, for either class
        best.loc[match, r["says"]] = best.loc[match, r["says"]].clip(lower=r["confidence"])
    fired = (best > 0).sum(axis=1)
    says = best.idxmax(axis=1).where(fired == 1, "")
    score = best.max(axis=1).where(fired == 1, 0.0)
    return strong, says, score, matched, fired == 2


def below_full(rows):
    # share of the rows the crowd was not fully sure about
    return (df.loc[rows, "gender:confidence"] < 1).mean()


# baseline for any nomination list: recorded brands are below full confidence more often than humans
unsure_by_label = {name: below_full(label == name) for name in ["human", "non_human"]}


def expected_unsure(recorded):
    # what a list with this recorded-label mix would score by chance
    mix = recorded.value_counts(normalize=True)
    return sum(mix.get(name, 0) * unsure_by_label[name] for name in unsure_by_label)


# the sweep is printed so the cut-off can be judged by what it produces
print("\nthreshold sweep, lift >= %.1f (below full crowd confidence: all labelled %.3f, "
      "recorded human %.3f, recorded non_human %.3f)"
      % (MIN_LIFT, below_full(labelled), unsure_by_label["human"], unsure_by_label["non_human"]))
sweep = []
for conf in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    strong, says, score, _, conflict = rule_votes(conf, MIN_LIFT)
    hit = labelled & (score > 0) & (says != label)
    sweep.append({"min_confidence": conf, "rules": len(strong),
                  "human_rules": int((strong["says"] == "human").sum()),
                  "non_human_rules": int((strong["says"] == "non_human").sum()),
                  "abstained": int(conflict.sum()), "profiles": int(hit.sum()),
                  "crowd_unsure": below_full(hit), "expected_unsure": expected_unsure(label[hit])})
sweep = pd.DataFrame(sweep).set_index("min_confidence")
print(sweep.round(3).to_string())
# mined on labelled rows, 0.80 doubles the list; 0.90 keeps a few hundred and both classes' rules
print("chosen cut-off %.2f: 0.80 nominates %d profiles, too many to check by hand; %.2f nominates %d"
      " and keeps %d human and %d non_human rules"
      % (MIN_CONF, sweep.loc[0.80, "profiles"], MIN_CONF, sweep.loc[MIN_CONF, "profiles"],
         sweep.loc[MIN_CONF, "human_rules"], sweep.loc[MIN_CONF, "non_human_rules"]))

STRONG, says, score, matched, conflict = rule_votes(MIN_CONF, MIN_LIFT)
print("\nusing confidence >=", MIN_CONF, "lift >=", MIN_LIFT, "->", len(STRONG), "rules",
      "(human", int((STRONG["says"] == "human").sum()),
      "| non_human", int((STRONG["says"] == "non_human").sum()), ")")
print("abstained (strong rules for both classes):", int(conflict.sum()),
      "| labelled:", int((conflict & labelled).sum()))


# 6. Predictions and flags
# output contract: every profile a strong rule of one class covers, labelled or unknown
covered = score > 0
predictions = pd.DataFrame({
    "_unit_id": df.loc[covered, "_unit_id"],
    "says": says[covered],
    "score": score[covered].round(3),
    "recorded": label[covered],
    "votes": 1,                                    # a strong rule fired, so the method votes
    "rules_matched": matched[covered],
})
predictions.to_csv(OUT / "association_predictions.csv", index=False)
print("profiles covered by a strong rule:", len(predictions),
      "| unknown among them:", int((predictions["recorded"] == "unknown").sum()))

# output contract: association_flagged.csv - labelled profiles whose rule contradicts the label
hit = covered & labelled & (says != label)
result = pd.DataFrame({
    "_unit_id": df.loc[hit, "_unit_id"],
    "says": says[hit],
    "score": score[hit].round(3),
    "recorded": label[hit],
    "gender": df.loc[hit, "gender"],
    "name": df.loc[hit, "name"],
    "gender:confidence": df.loc[hit, "gender:confidence"],
    "rules_matched": matched[hit],
}).sort_values("score", ascending=False)
result.to_csv(OUT / "association_flagged.csv", index=False)

print("profiles nominated:", len(result))
print(pd.crosstab(result["recorded"], result["says"]))
print(result.head(10)[["name", "gender", "recorded", "says", "score"]])


# 7. Crowd confidence
# check the nominations against crowd confidence, which the rules never used
nominated = below_full(hit)
mix = label[hit].value_counts(normalize=True)
expected = expected_unsure(label[hit])
print("below full crowd confidence - nominated: %.3f | all labelled with the same recorded-label mix: "
      "%.3f | all labelled: %.3f" % (nominated, expected, below_full(labelled)))

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
groups = [result["gender:confidence"], df.loc[labelled, "gender:confidence"]]
axes[0].hist(groups, bins=20, weights=[np.full(len(g), 100 / len(g)) for g in groups],
             color=["tab:red", "grey"], label=["nominated by the rules", "all labelled profiles"])
axes[0].set_xlabel("gender:confidence")
axes[0].set_ylabel("% of the group")
axes[0].set_title("crowd confidence, nominated vs all")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22))
# middle bar: the same recorded-label mix as the nominations, split by recorded label
axes[1].bar(0, nominated, color="tab:red")
bottom = 0
for name, colour in [("human", "tab:blue"), ("non_human", "tab:orange")]:
    part = mix.get(name, 0) * unsure_by_label[name]
    axes[1].bar(1, part, bottom=bottom, color=colour, label="recorded " + name)
    bottom += part
axes[1].bar(2, below_full(labelled), color="grey")
for x, value in enumerate([nominated, expected, below_full(labelled)]):
    axes[1].text(x, value + 0.01, "%.3f" % value, ha="center")
axes[1].set_ylim(0, max(nominated, expected) * 1.2)
axes[1].set_xticks([0, 1, 2], ["nominated", "same label\nmix", "all\nlabelled"])
axes[1].set_ylabel("share below full confidence")
axes[1].set_title("below full crowd confidence")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3))
plt.tight_layout()
plt.savefig(OUT / "fig_association_crowd_confidence.png", dpi=150, bbox_inches="tight")
plt.show()
