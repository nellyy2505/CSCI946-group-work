# CSCI446/946 Big Data Analytics - Assignment 2
# 03 - Clustering

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)
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


# load data - unsupervised, so the whole file is used (features already z-scored in 02)
df = pd.read_csv(PROC / "twitter_full.csv")
labelled = df["is_human"].notna()
truth = df.loc[labelled, "is_human"]
print("Twitter dataset size:", df.shape)
print("labelled:", labelled.sum(), "| human rate:", round(truth.mean(), 3))


# check for highly correlated features
corr = df[BEHAVIOUR].corr().abs()
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print("\nbehaviour attributes correlated above 0.7:\n", pairs[pairs > 0.7].round(2))
# two pairs at 0.76 (count vs per-day rate) - both kept, the rate adjusts for account age

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


# WSS for k = 1 to 15 to look for the elbow (n_init=10 keeps the best of 10 starts)
wss = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(X)
    wss.append(km.inertia_)
plt.plot(list(K_RANGE), wss, marker="o")
plt.axvline(K_MAIN, color="red", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sum of Squares")
plt.title("WSS for k = 1 to 15")
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


# final k-means, each cluster described by its mean feature values
km = KMeans(n_clusters=K_MAIN, n_init=10, random_state=SEED)
km.fit(X)
df["cluster"] = km.predict(X)
print("\ncluster sizes:\n", df["cluster"].value_counts().sort_index())

# means are z-scores; for 0/1 flags the mean is the share of the cluster
cluster_mean = df.groupby(["cluster"])[FEATURES].agg("mean")
print("\ncluster means:\n", cluster_mean.T.round(2))

fig, ax = plt.subplots(figsize=(7, 8))
image = ax.imshow(cluster_mean.T, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(K_MAIN), ["cluster " + str(c) for c in cluster_mean.index])
ax.set_yticks(range(len(FEATURES)), FEATURES)
for i in range(len(FEATURES)):
    for j in range(K_MAIN):
        ax.text(j, i, round(cluster_mean.iloc[j, i], 2), ha="center", va="center", fontsize=7)
fig.colorbar(image, label="mean value")
plt.title("what each cluster looks like")
plt.tight_layout()
plt.show()


# spread inside each cluster - a mean can hide a mix of highs and lows
DESCRIBE = ["tweet_count", "tweets_per_day", "fav_number", "text_n_urls", "text_n_hashtags"]
for cluster in range(K_MAIN):
    print("\ndescribe cluster labeled " + str(cluster) + ": \n",
          df.loc[df["cluster"] == cluster, DESCRIBE].describe().round(2))

RAW_COUNTS = ["tweet_count", "tweets_per_day", "fav_number", "favs_per_day"]
BOX = RAW_COUNTS + ["account_age_days", "text_len", "desc_len"]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
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
axes.ravel()[-1].axis("off")
plt.suptitle("spread of each attribute inside each cluster (outliers hidden)")
plt.tight_layout()
plt.show()


# crowd label mix per cluster - the clustering never saw the label
mix = pd.crosstab(df["cluster"], df["is_human"].map({1: "human", 0: "non_human"}))
mix["human_rate"] = (mix["human"] / mix.sum(axis=1)).round(3)
mix["unlabelled"] = df.loc[~labelled, "cluster"].value_counts().sort_index()
print("\nlabel mix per cluster (human rate of the whole data:", round(truth.mean(), 3), ")")
print(mix)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
mix[["non_human", "human"]].plot.bar(stacked=True, ax=axes[0], rot=0, color=LABEL_COLOURS)
axes[0].set_ylabel("profiles")
axes[0].set_title("how the labels fall in each cluster")
axes[1].bar(mix.index, mix["human_rate"], color=LABEL_COLOURS["human"])
axes[1].set_xticks(mix.index)   # one tick per cluster, not a number scale
axes[1].axhline(truth.mean(), color="red", linestyle="--", label="rate of the whole data")
axes[1].set_xlabel("cluster")
axes[1].set_ylabel("human rate")
axes[1].set_title("share of each cluster labelled human")
axes[1].legend()
plt.tight_layout()
plt.show()


# plot the most distinct features in pairs, one colour per cluster
distinct = (cluster_mean.max() - cluster_mean.min()) / cluster_mean.abs().max()
print("\nhow different each attribute is between clusters:\n",
      distinct.sort_values(ascending=False).round(2))
top = list(distinct.sort_values(ascending=False).index[:4])
# biggest cluster drawn first so the small ones stay visible
by_size = df["cluster"].value_counts().index
CLUSTER_COLOURS = ["tab:purple", "tab:green", "tab:red", "tab:olive"]

fig, axs = plt.subplots(2, 3, figsize=(15, 9))
i2, j2 = 0, 0
for i in range(len(top) - 1):
    for j in range(i + 1, len(top)):
        if j2 > 2:
            j2 = 0
            i2 += 1
        for c in by_size:
            hit = df["cluster"] == c
            axs[i2, j2].scatter(df.loc[hit, top[i]], df.loc[hit, top[j]], s=3, alpha=0.3,
                                color=CLUSTER_COLOURS[c], label="cluster " + str(c))
        centres = km.cluster_centers_[:, [FEATURES.index(top[i]), FEATURES.index(top[j])]]
        axs[i2, j2].scatter(centres[:, 0], centres[:, 1], c="black", marker="X", s=120)
        axs[i2, j2].set_title("{} vs {}".format(top[i], top[j]))
        j2 += 1
axs[0, 0].legend(markerscale=4)
plt.suptitle("clusters on the four most distinct attributes (black X = centroid)")
plt.tight_layout()
plt.show()

# most distinct pair, coloured by cluster then by crowd label
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
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
plt.show()


# hierarchical clustering on a sample - pairwise distances for all 18,715 profiles are too large
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
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix, truncate_mode="lastp", p=40, no_labels=True, color_threshold=cut)
plt.axhline(cut, color="red", linestyle="--", label="cut into " + str(K_TREE) + " clusters")
plt.ylabel("merge distance")
plt.title("complete linkage dendrogram (" + str(SAMPLE) + " profiles)")
plt.legend()
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


# DBSCAN - try a range of radii (Eps) with MinPts = 20
MIN_PTS = 20
for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    dbscan = DBSCAN(eps=eps, min_samples=MIN_PTS).fit(sample[FEATURES])
    found = len(set(dbscan.labels_) - {-1})
    print("Eps", eps, "| clusters:", found,
          "| noise points:", str(round((dbscan.labels_ == -1).mean() * 100)) + "%")

# small Eps -> mostly noise, large Eps -> one cluster: no separate dense groups
dbscan = DBSCAN(eps=2.0, min_samples=MIN_PTS).fit(sample[FEATURES])
noise = dbscan.labels_ == -1
plt.scatter(sample.loc[~noise, top[0]], sample.loc[~noise, top[1]], s=4, alpha=0.4,
            label="in a cluster")
plt.scatter(sample.loc[noise, top[0]], sample.loc[noise, top[1]], s=4, alpha=0.6,
            color="grey", label="noise")
plt.xlabel(top[0])
plt.ylabel(top[1])
plt.title("DBSCAN, Eps = 2.0, MinPts = " + str(MIN_PTS))
plt.legend(markerscale=4)
plt.show()


# finer clusters to judge single profiles - first check k and the purity cut-offs
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


# flag profiles in a one-sided cluster that carry the opposite label
leaning = rate[rate["leans"] != "mixed"]
suspect = df["fine_cluster"].map(leaning["leans"]).where(labelled)
cluster_says = suspect.map({"human": 1, "non_human": 0})
flagged = df[cluster_says.notna() & (cluster_says != df["is_human"])].copy()
flagged["cluster_says"] = suspect[flagged.index]
flagged["cluster_human_rate"] = flagged["fine_cluster"].map(rate["human_rate"]).round(3)
# output contract: says = the cluster's label, score = cluster purity
flagged["says"] = flagged["cluster_says"]
flagged["score"] = np.where(flagged["says"] == "human", flagged["cluster_human_rate"],
                            1 - flagged["cluster_human_rate"]).round(3)
# distance to the centroid - small means a typical member of the cluster
flagged["distance_to_centre"] = np.linalg.norm(
    X.loc[flagged.index].to_numpy() - fine.cluster_centers_[flagged["fine_cluster"]],
    axis=1).round(2)

print("\nprofiles whose cluster contradicts their label:", len(flagged))
print(flagged["gender"].value_counts())
print(flagged.groupby("cluster_says")[["cluster_human_rate", "gender:confidence"]].mean().round(3))


# check the flags against crowd confidence, which the clustering never used
print("\ncrowd confidence, flagged vs all labelled profiles:")
print(pd.DataFrame({"flagged": flagged["gender:confidence"].describe(),
                    "all": df.loc[labelled, "gender:confidence"].describe()}).round(3))
print("below full confidence - flagged:", round((flagged["gender:confidence"] < 1).mean(), 3),
      "| all:", round((df.loc[labelled, "gender:confidence"] < 1).mean(), 3))

plt.figure(figsize=(8, 4))
# bars as % of each group, so different-sized groups can be compared
confidence = [flagged["gender:confidence"], df.loc[labelled, "gender:confidence"]]
plt.hist(confidence, bins=20, weights=[np.full(len(c), 100 / len(c)) for c in confidence],
         label=["flagged by clustering", "all labelled profiles"])
plt.xlabel("gender:confidence")
plt.ylabel("% of profiles in the group")
plt.title("the crowd was less sure about the profiles the clusters flagged")
plt.legend()
plt.show()

# overlap with the association rules flags
rules_file = OUT / "association_flagged.csv"
if rules_file.exists():
    by_rules = set(pd.read_csv(rules_file)["_unit_id"])
    both = by_rules & set(flagged["_unit_id"])
    flagged["also_by_rules"] = flagged["_unit_id"].isin(by_rules).astype(int)
    print("\nflagged by the association rules:", len(by_rules),
          "| by clustering:", len(flagged), "| by both:", len(both))


# suggest a label for the unknown profiles from their cluster
unlabelled = df.loc[~labelled].copy()
unlabelled["suggested"] = unlabelled["fine_cluster"].map(leaning["leans"])
unlabelled["cluster_human_rate"] = unlabelled["fine_cluster"].map(rate["human_rate"]).round(3)
print("\nsuggestions for the unlabelled profiles:")
print(unlabelled["suggested"].value_counts(dropna=False))


# write outputs - flagged list sorted by cluster purity, then distance to centroid
KEEP = ["_unit_id", "says", "score", "name", "gender", "gender:confidence", "cluster",
        "fine_cluster", "cluster_human_rate", "distance_to_centre"]
if "also_by_rules" in flagged:
    KEEP.append("also_by_rules")
flagged = flagged.sort_values(["cluster_human_rate", "distance_to_centre"])
flagged[KEEP].to_csv(OUT / "clustering_flagged.csv", index=False)

suggestions = unlabelled[unlabelled["suggested"].notna()]
suggestions[["_unit_id", "name", "cluster", "fine_cluster", "cluster_human_rate",
             "suggested"]].to_csv(OUT / "cluster_suggested_labels.csv", index=False)

df[["_unit_id", "name", "gender", "is_human", "cluster", "fine_cluster"]].to_csv(
    OUT / "cluster_assignments.csv", index=False)
print("\nwritten:", [f.name for f in sorted(OUT.glob("cluster*.csv"))])
print("\nclearest candidates (typical members of clusters that disagree with their label):")
print(flagged.head(10)[["name", "gender", "gender:confidence", "says",
                        "cluster_human_rate", "distance_to_centre"]])
