# CSCI446/946 Big Data Analytics - Assignment 2
# 03 - Association rules

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

FLAGS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
         "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]
BINNED = ["tweet_count", "fav_number", "tweets_per_day", "favs_per_day",
          "account_age_days", "text_len", "desc_len", "text_n_urls"]
COLOURS = {"human": "tab:blue", "non_human": "tab:orange"}
MIN_SUPPORT = 0.05
MIN_CONF, MIN_LIFT = 0.80, 1.3

# load data - unsupervised, so the whole file is used
df = pd.read_csv(PROC / "twitter_full.csv")
label = df["label"]
labelled = label != "unknown"
print("shape:", df.shape, "| labelled:", labelled.sum())

# build the basket: one row per profile, one column per property it has or has not
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

# frequent itemsets - max_len=4, longer ones are padded copies of shorter rules
frequent = apriori(items, min_support=MIN_SUPPORT, use_colnames=True, max_len=4)
print("frequent itemsets:", len(frequent))
print(frequent.sort_values("support", ascending=False).head(15))

# generate rules
rules = association_rules(frequent, metric="confidence", min_threshold=0.6)
rules = rules[rules["lift"] > 1]
print("rules:", len(rules))
print(rules.sort_values("lift", ascending=False).head(10)[
    ["antecedents", "consequents", "support", "confidence", "lift"]])

# support vs confidence, coloured by lift
plt.figure(figsize=(8, 5))
points = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", alpha=0.6)
plt.colorbar(points, label="lift")
plt.xlabel("support")
plt.ylabel("confidence")
plt.title("all rules")
plt.savefig(OUT / "fig_rules_support_confidence.png", dpi=120, bbox_inches="tight")
plt.show()

# main result: the rules that conclude a label
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
plt.figure(figsize=(9, 5))
plt.barh(range(len(top)), top["lift"], color=[COLOURS[s] for s in top["says"]])
plt.yticks(range(len(top)), [r[:55] for r in top["rule"]], fontsize=8)
plt.axvline(1, color="red", linestyle="--", label="lift = 1")
plt.xlabel("lift")
plt.title("strongest rules for the label")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_rules_top_by_lift.png", dpi=120, bbox_inches="tight")
plt.show()


def contradicted(min_conf, min_lift):
    # profiles matching a rule for the class opposite to their label
    strong = label_rules[(label_rules["confidence"] >= min_conf) & (label_rules["lift"] >= min_lift)]
    hit = pd.Series(False, index=df.index)
    for _, r in strong.iterrows():
        hit |= items[sorted(r["antecedents"])].all(axis=1) & labelled & (label != r["says"])
    return strong, hit


# choose the threshold by what it produces, not by hand
print("\nthreshold sweep (crowd was unsure about %.3f of all labelled profiles)"
      % (df.loc[labelled, "gender:confidence"] < 1).mean())
sweep = []
for conf in [0.70, 0.75, 0.80, 0.85, 0.90]:
    strong, hit = contradicted(conf, MIN_LIFT)
    sweep.append({"min_confidence": conf, "rules": len(strong), "profiles": int(hit.sum()),
                  "crowd_unsure": (df.loc[hit, "gender:confidence"] < 1).mean()})
print(pd.DataFrame(sweep).round(3).to_string(index=False))

# 0.80 is the tightest cut keeping rules for both classes (best non_human rule is 0.846)
STRONG, hit = contradicted(MIN_CONF, MIN_LIFT)
print("\nusing confidence >=", MIN_CONF, "lift >=", MIN_LIFT, "->", len(STRONG), "rules")

# score each nominated profile by the strongest rule against it
score = pd.Series(0.0, index=df.index)
says = pd.Series("", index=df.index)
votes = pd.Series(0, index=df.index)
for _, r in STRONG.iterrows():
    match = items[sorted(r["antecedents"])].all(axis=1) & labelled & (label != r["says"])
    votes[match] += 1
    stronger = match & (r["confidence"] > score)
    says[stronger] = r["says"]
    score[stronger] = r["confidence"]

# output contract: _unit_id, says, score
result = pd.DataFrame({
    "_unit_id": df.loc[hit, "_unit_id"],
    "says": says[hit],
    "score": score[hit].round(3),
    "recorded": label[hit],
    "gender": df.loc[hit, "gender"],
    "name": df.loc[hit, "name"],
    "gender:confidence": df.loc[hit, "gender:confidence"],
    "rules_contradicting": votes[hit],
}).sort_values("score", ascending=False)
result.to_csv(OUT / "association_flagged.csv", index=False)

print("profiles nominated:", len(result))
print(pd.crosstab(result["recorded"], result["says"]))
print(result.head(10)[["name", "gender", "recorded", "says", "score"]])

# check the nominations against crowd confidence, which the rules never used
plt.figure(figsize=(8, 4))
groups = [result["gender:confidence"], df.loc[labelled, "gender:confidence"]]
plt.hist(groups, bins=20, weights=[np.full(len(g), 100 / len(g)) for g in groups],
         label=["nominated by the rules", "all labelled profiles"])
plt.xlabel("gender:confidence")
plt.ylabel("% of the group")
plt.title("crowd confidence, nominated vs all")
plt.legend()
plt.savefig(OUT / "fig_rules_crowd_confidence.png", dpi=120, bbox_inches="tight")
plt.show()
