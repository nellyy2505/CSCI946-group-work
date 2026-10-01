# 04_clustering.py — k-means, hierarchical and DBSCAN on the behaviour features; one-sided clusters vote

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})   # figures are placed at 4-6.2 in wide in the report

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 7
SAMPLE = 5000          # sample size for hierarchical clustering and DBSCAN
K_RANGE = range(1, 16)
K_MAIN = 4             # chosen by comparing k-1, k and k+1
K_TREE = 15            # where the hierarchical tree is cut
K_FINE = 12            # finer clusters used to flag profiles
PURE_HUMAN = 0.85      # human rate at or above this -> one-sided human
PURE_NON_HUMAN = 0.35  # human rate at or below this -> one-sided non-human
# same label colours in every chart
LABEL_COLOURS = {"human": "tab:blue", "non_human": "tab:orange"}

# features, grouped by what they describe
ACTIVITY = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days"]
CONTENT = ["text_len", "desc_len", "text_n_urls", "text_n_mentions", "text_n_hashtags",
           "name_n_digits"]
PROFILE = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
           "has_coord", "location_missing", "timezone_missing"]
COLOUR = ["link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b",
          "link_default", "sidebar_default"]
BEHAVIOUR = ACTIVITY + CONTENT + PROFILE


# 1. Load
# unsupervised, so the whole file is used (features already z-scored in 02)
df = pd.read_csv(PROC / "twitter_full.csv")
labelled = df["is_human"].notna()
truth = df.loc[labelled, "is_human"]
print("Twitter dataset size:", df.shape)
print("labelled:", labelled.sum(), "| human rate:", round(truth.mean(), 3))


# 2. Feature choice
# check for highly correlated features
corr = df[BEHAVIOUR].corr().abs()
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print("\nbehaviour attributes correlated above 0.7:\n", pairs[pairs > 0.7].round(2))
# count and per-day rate are correlated but both kept: the rate adjusts for account age

# compare feature sets by how much the human rate differs between clusters
spread = {}
for name, cols in [("behaviour + colour", BEHAVIOUR + COLOUR),
                   ("behaviour only", BEHAVIOUR),
                   ("colour only", COLOUR)]:
    km = KMeans(n_clusters=K_MAIN, n_init=10, random_state=SEED)
    km.fit(df[cols])
    rate = truth.groupby(km.predict(df.loc[labelled, cols])).agg("mean")
    spread[name] = rate.round(2).tolist()
    print(name, "- human rate per cluster:", spread[name])

# colour clusters barely differ in human rate, so colour is left out
FEATURES = BEHAVIOUR
X = df[FEATURES]
print("\nclustering on", len(FEATURES), "behaviour attributes:", FEATURES)


# 3. Choosing k
# WSS for k = 1 to 15 to look for the elbow (n_init=10 keeps the best of 10 starts)
wss = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(X)
    wss.append(km.inertia_)
plt.figure(figsize=(4.8, 3.6))
plt.plot(list(K_RANGE), wss, marker="o")
plt.axvline(K_MAIN, color="red", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sum of Squares")
plt.title("WSS for k = 1 to 15")
plt.savefig(OUT / "fig_clustering_wss.png", dpi=150, bbox_inches="tight")
plt.show()

# no sharp elbow - compare k-1, k, k+1 on cluster sizes and closest centroids
for k in [K_MAIN - 1, K_MAIN, K_MAIN + 1]:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(X)
    groups = km.predict(X)
    print("\nk =", k, "| WSS:", round(km.inertia_))
    print("  cluster sizes:", np.bincount(groups).tolist())
    print("  closest two centroids:", round(pdist(km.cluster_centers_, "euclidean").min(), 2))
    print("  human rate per cluster:", truth.groupby(groups[labelled]).agg("mean").round(2).tolist())

# k=4 has the most separated centroids and no tiny cluster


# 4. Final k-means
# each cluster described by its mean feature values
km = KMeans(n_clusters=K_MAIN, n_init=10, random_state=SEED)
km.fit(X)
df["cluster"] = km.predict(X)
print("\ncluster sizes:\n", df["cluster"].value_counts().sort_index())

# means are z-scores; for 0/1 flags the mean is the share of the cluster
cluster_mean = df.groupby(["cluster"])[FEATURES].agg("mean")
print("\ncluster means:\n", cluster_mean.T.round(2))

# sized to be read at 6.2 in wide on its own: every number at the base font size (11 pt)
fig, ax = plt.subplots(figsize=(6.2, 7))
image = ax.imshow(cluster_mean.T, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(K_MAIN), ["cluster " + str(c) for c in cluster_mean.index])
ax.set_yticks(range(len(FEATURES)), FEATURES)
for i in range(len(FEATURES)):
    for j in range(K_MAIN):
        ax.text(j, i, "%.2f" % cluster_mean.iloc[j, i], ha="center", va="center")
fig.colorbar(image, label="mean value (z-score, or share for a 0/1 flag)")
plt.title("what each cluster looks like")
plt.tight_layout()
plt.savefig(OUT / "fig_clustering_cluster_means.png", dpi=150, bbox_inches="tight")
plt.show()


# 5. Spread inside each cluster
# a mean can hide a mix of highs and lows
DESCRIBE = ["tweet_count", "tweets_per_day", "fav_number", "text_n_urls", "text_n_hashtags"]
for cluster in range(K_MAIN):
    print("\ndescribe cluster labeled " + str(cluster) + ": \n",
          df.loc[df["cluster"] == cluster, DESCRIBE].describe().round(2))

RAW_COUNTS = ["tweet_count", "tweets_per_day", "fav_number", "favs_per_day"]
BOX = RAW_COUNTS + ["account_age_days", "text_len", "desc_len"]
# sized to be read at 6.2 in wide on its own
fig, axes = plt.subplots(3, 3, figsize=(6.2, 7.6))
for ax, col in zip(axes.ravel(), BOX):
    ax.boxplot([df.loc[df["cluster"] == c, col] for c in range(K_MAIN)], showfliers=False)
    ax.set_xticks(range(1, K_MAIN + 1), range(K_MAIN))
    ax.set_xlabel("cluster")
    ax.set_title(col)
    # raw counts on a symlog scale (a log scale that can show zero), the rest are z-scores
    if col in RAW_COUNTS:
        ax.set_yscale("symlog")
        ax.set_ylim(bottom=0)   # counts cannot be negative
    else:
        ax.set_ylabel("z-score")
for ax in axes.ravel()[len(BOX):]:
    ax.axis("off")
plt.suptitle("spread of each attribute inside each cluster\n(outliers hidden)")
plt.tight_layout()
plt.savefig(OUT / "fig_clustering_spread.png", dpi=150, bbox_inches="tight")
plt.show()


# 6. Label mix per cluster
# the clustering never saw the label
mix = pd.crosstab(df["cluster"], df["is_human"].map({1: "human", 0: "non_human"}))
mix["human_rate"] = (mix["human"] / mix.sum(axis=1)).round(3)
mix["unlabelled"] = df.loc[~labelled, "cluster"].value_counts().sort_index()
print("\nlabel mix per cluster (human rate of the whole data:", round(truth.mean(), 3), ")")
print(mix)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
mix[["non_human", "human"]].plot.bar(stacked=True, ax=axes[0], rot=0, color=LABEL_COLOURS)
axes[0].set_ylabel("profiles")
axes[0].set_title("how the labels fall\nin each cluster")
axes[1].bar(mix.index, mix["human_rate"], color=LABEL_COLOURS["human"])
axes[1].set_xticks(mix.index)   # one tick per cluster, not a number scale
axes[1].axhline(truth.mean(), color="red", linestyle="--", label="rate of the whole data")
axes[1].set_xlabel("cluster")
axes[1].set_ylabel("human rate")
axes[1].set_title("share of each cluster\nlabelled human")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig(OUT / "fig_clustering_label_mix.png", dpi=150, bbox_inches="tight")
plt.show()


# 7. Pair plots
# the most distinct features in pairs, one colour per cluster
distinct = (cluster_mean.max() - cluster_mean.min()) / cluster_mean.abs().max()
print("\nhow different each attribute is between clusters:\n",
      distinct.sort_values(ascending=False).round(2))
top = list(distinct.sort_values(ascending=False).index[:4])
# biggest cluster drawn first so the small ones stay visible
by_size = df["cluster"].value_counts().index
CLUSTER_COLOURS = ["tab:purple", "tab:green", "tab:red", "tab:olive"]

fig, axs = plt.subplots(3, 2, figsize=(7.4, 10))
i2, j2 = 0, 0
for i in range(len(top) - 1):
    for j in range(i + 1, len(top)):
        if j2 > 1:
            j2 = 0
            i2 += 1
        for c in by_size:
            hit = df["cluster"] == c
            axs[i2, j2].scatter(df.loc[hit, top[i]], df.loc[hit, top[j]], s=3, alpha=0.3,
                                color=CLUSTER_COLOURS[c], label="cluster " + str(c))
        centres = km.cluster_centers_[:, [FEATURES.index(top[i]), FEATURES.index(top[j])]]
        axs[i2, j2].scatter(centres[:, 0], centres[:, 1], c="black", marker="X", s=120)
        axs[i2, j2].set_title("{}\nvs {}".format(top[i], top[j]))
        j2 += 1
axs[0, 0].legend(markerscale=4)
plt.suptitle("clusters on the four most distinct attributes\n(black X = centroid)")
plt.tight_layout()
plt.savefig(OUT / "fig_clustering_pairs.png", dpi=150, bbox_inches="tight")
plt.show()

# most distinct pair, coloured by cluster then by crowd label
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4))
for c in by_size:
    hit = df["cluster"] == c
    axes[0].scatter(df.loc[hit, top[0]], df.loc[hit, top[1]], s=4, alpha=0.3,
                    color=CLUSTER_COLOURS[c], label="cluster " + str(c))
axes[0].legend(markerscale=4)
axes[0].set_title("clusters found by k-means")
for value, name in [(1, "human"), (0, "non_human")]:
    hit = df["is_human"] == value
    axes[1].scatter(df.loc[hit, top[0]], df.loc[hit, top[1]], s=4, alpha=0.3, label=name,
                    color=LABEL_COLOURS[name])
axes[1].set_title("the crowd label, same points")
axes[1].legend(markerscale=4)
for ax in axes:
    ax.set_xlabel(top[0])
    ax.set_ylabel(top[1])
plt.tight_layout()
plt.savefig(OUT / "fig_clustering_vs_label.png", dpi=150, bbox_inches="tight")
plt.show()


# 8. Hierarchical clustering
# on a sample: pairwise distances for all 18,715 profiles are too large
part = np.random.RandomState(SEED).choice(len(df), SAMPLE, replace=False)
sample = df.iloc[part].copy()
dist = pdist(sample[FEATURES], "euclidean")

# the four linkage methods, each cut into K_MAIN clusters
for method in ["single", "complete", "average", "centroid"]:
    if method == "centroid":
        linkage_matrix = linkage(sample[FEATURES], method=method)   # needs the raw points
    else:
        linkage_matrix = linkage(dist, method=method)
    labels = cut_tree(linkage_matrix, n_clusters=K_MAIN).ravel()
    print(method, "linkage, cut into", K_MAIN, "- cluster sizes:", np.bincount(labels).tolist())

# at 4 clusters every linkage gives one big group plus outliers, so cut complete linkage lower
linkage_matrix = linkage(dist, method="complete")
cut = (linkage_matrix[-K_TREE, 2] + linkage_matrix[-K_TREE + 1, 2]) / 2
plt.figure(figsize=(7.4, 4))
dendrogram(linkage_matrix, truncate_mode="lastp", p=40, no_labels=True, color_threshold=cut)
plt.axhline(cut, color="red", linestyle="--", label="cut into " + str(K_TREE) + " clusters")
plt.ylabel("merge distance")
plt.title("complete linkage dendrogram (" + str(SAMPLE) + " profiles)")
plt.legend()
plt.savefig(OUT / "fig_clustering_dendrogram.png", dpi=150, bbox_inches="tight")
plt.show()

sample["tree_cluster"] = cut_tree(linkage_matrix, n_clusters=K_TREE).ravel()
branches = sample.groupby("tree_cluster").agg(
    size=("tree_cluster", "size"), human_rate=("is_human", "mean"),
    default_image=("default_image", "mean"), median_likes=("fav_number", "median"),
    median_tweets_per_day=("tweets_per_day", "median"))
print("\nhierarchical clusters (complete linkage, cut into", K_TREE, "):")
print(branches.sort_values("size", ascending=False).round(2))
# where each hierarchical cluster's profiles went in k-means
print("\nhierarchical cluster (rows) vs k-means cluster (columns):")
print(pd.crosstab(sample["tree_cluster"], sample["cluster"]))


# 9. DBSCAN
# try a range of radii (Eps) with MinPts = 20
MIN_PTS = 20
for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    dbscan = DBSCAN(eps=eps, min_samples=MIN_PTS).fit(sample[FEATURES])
    found = len(set(dbscan.labels_) - {-1})
    print("Eps", eps, "| clusters:", found,
          "| noise points:", str(round((dbscan.labels_ == -1).mean() * 100)) + "%")

# small Eps -> mostly noise, large Eps -> one cluster: no separate dense groups
dbscan = DBSCAN(eps=2.0, min_samples=MIN_PTS).fit(sample[FEATURES])
noise = dbscan.labels_ == -1
plt.figure(figsize=(4.8, 3.6))
plt.scatter(sample.loc[~noise, top[0]], sample.loc[~noise, top[1]], s=4, alpha=0.4,
            label="in a cluster")
plt.scatter(sample.loc[noise, top[0]], sample.loc[noise, top[1]], s=4, alpha=0.6,
            color="grey", label="noise")
plt.xlabel(top[0])
plt.ylabel(top[1])
plt.title("DBSCAN, Eps = 2.0, MinPts = " + str(MIN_PTS))
plt.legend(markerscale=4)
plt.savefig(OUT / "fig_clustering_dbscan.png", dpi=150, bbox_inches="tight")
plt.show()


# 10. Finer clusters
# to judge single profiles; first check k and the purity cut-offs
sweep = []
for k in [8, 10, 12, 14, 16]:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(X)
    groups = pd.Series(km.predict(X), index=df.index)
    rate = truth.groupby(groups[labelled]).agg("mean")
    for pure_h, pure_nh in [(0.85, 0.35), (0.85, 0.30), (0.90, 0.30), (0.90, 0.20)]:
        says = groups.map(pd.Series(np.where(rate >= pure_h, 1,
                                             np.where(rate <= pure_nh, 0, np.nan)), index=rate.index))
        hit = labelled & says.notna() & (says != df["is_human"])
        sweep.append({"k": k, "cut-offs": str(pure_h) + " / " + str(pure_nh),
                      "flagged": int(hit.sum())})
print("\nprofiles flagged for each k and purity cut-offs (human / non-human):")
print(pd.DataFrame(sweep).pivot(index="k", columns="cut-offs", values="flagged"))
# k=12 flags the same profiles under every cut-off, so it is used

fine = KMeans(n_clusters=K_FINE, n_init=10, random_state=SEED)
fine.fit(X)
df["fine_cluster"] = fine.predict(X)
rate = truth.groupby(df.loc[labelled, "fine_cluster"]).agg(human_rate="mean", labelled="size")
rate["size"] = df["fine_cluster"].value_counts()
# one-sided: nearly all labelled members share one label
rate["leans"] = np.where(rate["human_rate"] >= PURE_HUMAN, "human",
                         np.where(rate["human_rate"] <= PURE_NON_HUMAN, "non_human", "mixed"))
print("\n", K_FINE, "finer clusters, sorted by how human they look:")
print(rate.sort_values("human_rate").round(3))
print("one-sided clusters:", (rate["leans"] != "mixed").sum(), "of", K_FINE)


# 11. Flags
# profiles in a one-sided cluster that carry the opposite label
leaning = rate[rate["leans"] != "mixed"]
suspect = df["fine_cluster"].map(leaning["leans"]).where(labelled)
cluster_says = suspect.map({"human": 1, "non_human": 0})
flagged = df[cluster_says.notna() & (cluster_says != df["is_human"])].copy()
flagged["cluster_says"] = suspect[flagged.index]
flagged["recorded"] = flagged["label"]
flagged_purity = flagged["fine_cluster"].map(rate["human_rate"])
flagged["cluster_human_rate"] = flagged_purity.round(3)
# output contract: says = the cluster's label, score = cluster purity
flagged["says"] = flagged["cluster_says"]
flagged["score"] = np.where(flagged["says"] == "human", flagged_purity, 1 - flagged_purity).round(3)
# distance to the centroid - small means a typical member of the cluster
flagged["distance_to_centre"] = np.linalg.norm(
    X.loc[flagged.index].to_numpy() - fine.cluster_centers_[flagged["fine_cluster"]],
    axis=1).round(2)

print("\nprofiles whose cluster contradicts their label:", len(flagged))
print(flagged["gender"].value_counts())
print(flagged.groupby("cluster_says")[["cluster_human_rate", "gender:confidence"]].mean().round(3))


def flag_set(model):
    # the flag rule above, applied to another k-means run
    groups = pd.Series(model.predict(X), index=df.index)
    rate = truth.groupby(groups[labelled]).agg("mean")
    says = groups.map(pd.Series(np.where(rate >= PURE_HUMAN, 1,
                                         np.where(rate <= PURE_NON_HUMAN, 0, np.nan)), index=rate.index))
    return set(df.loc[labelled & says.notna() & (says != df["is_human"]), "_unit_id"])


# seed check: k-means only finds a local optimum, so rerun once with another seed and compare flags
other = flag_set(KMeans(n_clusters=K_FINE, n_init=10, random_state=SEED + 1).fit(X))
this = set(flagged["_unit_id"])
print("seed check - flagged with seed", SEED, ":", len(this), "| seed", SEED + 1, ":", len(other),
      "| Jaccard:", round(len(this & other) / len(this | other), 3))


# 12. Crowd confidence
# check the flags against crowd confidence, which the clustering never used
print("\ncrowd confidence, flagged vs all labelled profiles:")
print(pd.DataFrame({"flagged": flagged["gender:confidence"].describe(),
                    "all": df.loc[labelled, "gender:confidence"].describe()}).round(3))


def below_full(rows):
    # share of the rows the crowd was not fully sure about
    return (df.loc[rows, "gender:confidence"] < 1).mean()


# baseline: recorded brands are unsure more often than humans, so also compare with the same label mix
unsure_by_label = {name: below_full(df["label"] == name) for name in ["human", "non_human"]}
mix = flagged["recorded"].value_counts(normalize=True)
nominated = below_full(flagged.index)
expected = sum(mix.get(name, 0) * unsure_by_label[name] for name in unsure_by_label)
print("below full crowd confidence - nominated: %.3f | all labelled with the same recorded-label mix: "
      "%.3f | all labelled: %.3f" % (nominated, expected, below_full(labelled)))

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
# bars as % of each group, so different-sized groups can be compared
groups = [flagged["gender:confidence"], df.loc[labelled, "gender:confidence"]]
axes[0].hist(groups, bins=20, weights=[np.full(len(g), 100 / len(g)) for g in groups],
             color=["tab:red", "grey"], label=["nominated by clustering", "all labelled profiles"])
axes[0].set_xlabel("gender:confidence")
axes[0].set_ylabel("% of the group")
axes[0].set_title("crowd confidence, nominated vs all")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22))
# middle bar: the same recorded-label mix as the nominations, split by recorded label
axes[1].bar(0, nominated, color="tab:red")
bottom = 0
for name in ["human", "non_human"]:
    part = mix.get(name, 0) * unsure_by_label[name]
    axes[1].bar(1, part, bottom=bottom, color=LABEL_COLOURS[name], label="recorded " + name)
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
plt.savefig(OUT / "fig_clustering_crowd_confidence.png", dpi=150, bbox_inches="tight")
plt.show()

# overlap with the association rules flags (only if 03 has been run)
rules_file = OUT / "association_flagged.csv"
if rules_file.exists():
    by_rules = set(pd.read_csv(rules_file)["_unit_id"])
    both = by_rules & set(flagged["_unit_id"])
    flagged["also_by_rules"] = flagged["_unit_id"].isin(by_rules).astype(int)
    print("\nflagged by the association rules:", len(by_rules),
          "| by clustering:", len(flagged), "| by both:", len(both))


# 13. Predictions
# output contract: every profile in a one-sided cluster, labelled or not
leans = df["fine_cluster"].map(leaning["leans"])
purity = df["fine_cluster"].map(rate["human_rate"])
covered = leans.notna()
predictions = pd.DataFrame({
    "_unit_id": df.loc[covered, "_unit_id"],
    "says": leans[covered],
    "score": np.where(leans[covered] == "human", purity[covered], 1 - purity[covered]).round(3),
    "recorded": df.loc[covered, "label"],
    "votes": 1,                                    # a one-sided cluster, so the method votes
    "fine_cluster": df.loc[covered, "fine_cluster"],
})
print("\nprofiles in a one-sided cluster:", len(predictions))
print("suggestions for the unknown profiles:\n",
      predictions.loc[predictions["recorded"] == "unknown", "says"].value_counts())


# 14. Write outputs
# flagged list sorted by cluster purity, then distance to centroid
flagged = flagged.sort_values(["cluster_human_rate", "distance_to_centre"])
KEEP = ["_unit_id", "says", "score", "recorded", "name", "gender", "gender:confidence", "cluster",
        "fine_cluster", "cluster_human_rate", "distance_to_centre"]
if "also_by_rules" in flagged:
    KEEP.append("also_by_rules")
flagged[KEEP].to_csv(OUT / "clustering_flagged.csv", index=False)
predictions.to_csv(OUT / "clustering_predictions.csv", index=False)
df[["_unit_id", "name", "gender", "is_human", "cluster", "fine_cluster"]].to_csv(
    OUT / "clustering_assignments.csv", index=False)
print("\nwritten:", [f.name for f in sorted(OUT.glob("clustering_*.csv"))])
print("\nclearest candidates (typical members of clusters that disagree with their label):")
print(flagged.head(10)[["name", "gender", "gender:confidence", "says",
                        "cluster_human_rate", "distance_to_centre"]])
