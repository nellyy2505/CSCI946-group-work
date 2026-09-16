# CSCI446/946 Big Data Analytics - Assignment 2
# 04 - Clustering: group the profiles by behaviour and use the groups to question the labels

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)
SEED = 7
SAMPLE = 5000          # silhouette on all 18k records is slow, a sample is enough
K_RANGE = range(2, 16)
K_MAIN = 4             # chosen in step 3
K_FINE = 12            # smaller clusters, used in step 8 to question the labels
PURE_HUMAN = 0.85      # a cluster this human-heavy is treated as one-sided
PURE_NON_HUMAN = 0.35  # and this is the other side, well below the 0.69 rate of the data

# the features, grouped by what they describe
ACTIVITY = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days"]
CONTENT = ["text_len", "desc_len", "text_n_urls", "text_n_mentions", "text_n_hashtags",
           "name_n_digits"]
PROFILE = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
           "has_coord", "location_missing", "timezone_missing"]
COLOUR = ["link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b",
          "link_default", "sidebar_default"]
BEHAVIOUR = ACTIVITY + CONTENT + PROFILE


# 1. load the full cleaned data - clustering is unsupervised, so no train/test split.
# The numeric columns were already log-transformed and standardised in 02_preprocess.py,
# which k-means needs because it measures distance
df = pd.read_csv(PROC / "twitter_full.csv")
labelled = df["is_human"].notna()
truth = df.loc[labelled, "is_human"]
print("shape:", df.shape)
print("labelled:", labelled.sum(), "| human rate:", round(truth.mean(), 3))


# 2. which features to cluster on: run k-means on three feature sets and compare.
# Silhouette says how tight and separated the clusters are; the crosstab against the
# known label says whether the clusters mean anything for the human/non-human question
def score(X, groups):
    # silhouette on a sample, the other two on everything
    part = np.random.RandomState(SEED).choice(len(X), SAMPLE, replace=False)
    return {"silhouette": silhouette_score(X[part], groups[part]),
            "davies_bouldin": davies_bouldin_score(X, groups),
            "calinski_harabasz": calinski_harabasz_score(X, groups),
            # spread of the human rate across clusters: 0 means every cluster looks the same
            "human_rate_spread": truth.groupby(groups[labelled]).mean().std()}


rows = []
for name, cols in [("behaviour + colour", BEHAVIOUR + COLOUR),
                   ("behaviour only", BEHAVIOUR),
                   ("colour only", COLOUR)]:
    X = df[cols].to_numpy(dtype=float)
    groups = KMeans(K_MAIN, n_init=10, random_state=SEED).fit_predict(X)
    rows.append({"features": name, "n_features": len(cols), **score(X, groups)})
print("\nfeature sets compared at k =", K_MAIN)
print(pd.DataFrame(rows).set_index("features").round(3))

# Colour alone gives by far the tightest clusters, because the repaired hex codes sit on a
# few fixed values - but those clusters hold no information about the label. Colour would
# dominate the distance and hide the behaviour, so it is left out from here on
FEATURES = BEHAVIOUR
X = df[FEATURES].to_numpy(dtype=float)
print("\nclustering on", len(FEATURES), "behaviour features:", FEATURES)


# 3. how many clusters: fit k-means across a range of k and read the three diagnostics
diag = []
for k in K_RANGE:
    model = KMeans(k, n_init=10, random_state=SEED).fit(X)
    diag.append({"k": k, "inertia": model.inertia_, **score(X, model.labels_)})
diag = pd.DataFrame(diag).set_index("k")
print("\nchoosing k:")
print(diag.round(3))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(diag.index, diag["inertia"], marker="o")
axes[0].set_title("elbow: within-cluster sum of squares")
axes[1].plot(diag.index, diag["silhouette"], marker="o")
axes[1].set_title("silhouette (higher is better)")
axes[2].plot(diag.index, diag["davies_bouldin"], marker="o")
axes[2].set_title("Davies-Bouldin (lower is better)")
for ax in axes:
    ax.axvline(K_MAIN, color="red", linestyle="--")
    ax.set_xlabel("k")
plt.tight_layout()
plt.show()

# k=2 scores best on every index, but it only repeats the split we already have a label for.
# The curve bends at k=4 and the extra clusters are still large and readable, so k=4 is used
# for the description of the data


# 4. fit the chosen model and describe each cluster by its average feature values
kmeans = KMeans(K_MAIN, n_init=10, random_state=SEED).fit(X)
df["cluster"] = kmeans.labels_
print("\ncluster sizes:\n", df["cluster"].value_counts().sort_index())

# the features are standardised, so a centre of +1 means "one standard deviation above average"
centres = df.groupby("cluster")[FEATURES].mean().T
print("\ncluster centres (standardised units for the counts, rate for the flags):")
print(centres.round(2))

fig, ax = plt.subplots(figsize=(7, 8))
image = ax.imshow(centres, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(K_MAIN), ["cluster " + str(c) for c in centres.columns])
ax.set_yticks(range(len(FEATURES)), FEATURES)
for i in range(len(FEATURES)):
    for j in range(K_MAIN):
        ax.text(j, i, round(centres.iloc[i, j], 2), ha="center", va="center", fontsize=7)
fig.colorbar(image, label="mean value")
plt.title("what each cluster looks like")
plt.tight_layout()
plt.show()


# 5. read the clusters against the crowd label. The clusters were built without the label,
# so any difference between them is something the behaviour alone found
mix = pd.crosstab(df["cluster"], df["is_human"].map({1: "human", 0: "non_human"}))
mix["human_rate"] = (mix["human"] / mix.sum(axis=1)).round(3)
mix["unlabelled"] = df.loc[~labelled, "cluster"].value_counts().sort_index()
print("\nlabel mix per cluster (human rate of the whole data:", round(truth.mean(), 3), ")")
print(mix)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
mix[["non_human", "human"]].plot.bar(stacked=True, ax=axes[0], rot=0)
axes[0].set_ylabel("profiles")
axes[0].set_title("how the labels fall in each cluster")
axes[1].bar(mix.index, mix["human_rate"])
axes[1].set_xticks(mix.index)   # one tick per cluster, not a number scale
axes[1].axhline(truth.mean(), color="red", linestyle="--", label="rate of the whole data")
axes[1].set_xlabel("cluster")
axes[1].set_ylabel("human rate")
axes[1].set_title("share of each cluster labelled human")
axes[1].legend()
plt.tight_layout()
plt.show()


# 6. show the clusters in two dimensions. PCA rotates the 19 features so that the first two
# components carry as much of the spread as possible, which makes the shape drawable
pca = PCA(n_components=2, random_state=SEED)
points = pca.fit_transform(X)
print("\nvariance kept by the two components:", pca.explained_variance_ratio_.round(3),
      "total", round(pca.explained_variance_ratio_.sum(), 3))
print("what the components are built from:")
print(pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1", "PC2"]).round(2))

# the same points twice: coloured by cluster, then by the crowd label
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for cluster in range(K_MAIN):
    hit = df["cluster"] == cluster
    axes[0].scatter(points[hit, 0], points[hit, 1], s=4, alpha=0.3, label="cluster " + str(cluster))
centre_points = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(centre_points[:, 0], centre_points[:, 1], c="black", marker="X", s=150)
axes[0].set_title("clusters found by k-means")
for value, name in [(1, "human"), (0, "non-human")]:
    hit = (df["is_human"] == value).to_numpy()
    axes[1].scatter(points[hit, 0], points[hit, 1], s=4, alpha=0.3, label=name)
axes[1].set_title("the crowd label, same points")
for ax in axes:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=4)
plt.tight_layout()
plt.show()


# 7. k-means assumes round, equally sized clusters. Three other algorithms are tried to
# check that the four groups are really in the data and not just an artefact of k-means.
# Ward and the dendrogram run on a sample because they need the distance between every pair
part = np.random.RandomState(SEED).choice(len(X), SAMPLE, replace=False)
X_part, truth_part = X[part], df["is_human"].to_numpy()[part]
has_label = ~np.isnan(truth_part)

# a dendrogram shows where the merges happen, which is a second opinion on the number of clusters
links = linkage(X[np.random.RandomState(SEED).choice(len(X), 2000, replace=False)], method="ward")
plt.figure(figsize=(11, 4))
# cut halfway between the merge that makes k groups and the one that makes k-1,
# and colour the branches below the cut so each colour is one group
cut = (links[-K_MAIN, 2] + links[-K_MAIN + 1, 2]) / 2
dendrogram(links, truncate_mode="lastp", p=30, no_labels=True, color_threshold=cut)
plt.axhline(cut, color="red", linestyle="--", label="cut for k = " + str(K_MAIN))
plt.ylabel("merge distance")
plt.title("Ward dendrogram (2000 profiles)")
plt.legend()
plt.tight_layout()
plt.show()

# DBSCAN needs a radius. The usual way to set it is to measure how far each profile is from
# its 20th neighbour and take the middle of those distances
MIN_SAMPLES = 20
neighbours = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X_part)
eps = float(np.median(neighbours.kneighbors(X_part)[0][:, -1]))
print("\nDBSCAN radius taken from the data:", round(eps, 2))

models = {
    "k-means": KMeans(K_MAIN, n_init=10, random_state=SEED).fit_predict(X_part),
    "hierarchical (Ward)": AgglomerativeClustering(K_MAIN, linkage="ward").fit_predict(X_part),
    # a mixture model allows stretched, overlapping clusters instead of round ones
    "gaussian mixture": GaussianMixture(K_MAIN, covariance_type="full",
                                        random_state=SEED).fit_predict(X_part),
    # DBSCAN finds dense regions and decides the number of clusters itself
    "DBSCAN": DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X_part),
}

rows = []
for name, groups in models.items():
    found = len(set(groups) - {-1})
    rows.append({
        "algorithm": name,
        "clusters": found,
        "unassigned": int((groups == -1).sum()),
        # the two indices need at least two clusters to mean anything
        "silhouette": silhouette_score(X_part, groups) if found > 1 else np.nan,
        "davies_bouldin": davies_bouldin_score(X_part, groups) if found > 1 else np.nan,
        # agreement with the crowd label, and with the k-means result
        "rand_vs_label": adjusted_rand_score(truth_part[has_label], groups[has_label]),
        "rand_vs_kmeans": adjusted_rand_score(models["k-means"], groups),
    })
print("\nalgorithms compared on", SAMPLE, "profiles:")
print(pd.DataFrame(rows).set_index("algorithm").round(3))

# k-means scores best on both indices and Ward agrees with it in part, so the groups are not
# an artefact of one algorithm. DBSCAN returns everything as a single cluster plus a ring of
# unassigned profiles around it: the data is one dense cloud with no empty space between the
# groups, so cutting it into parts works here and looking for dense regions does not
fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharex=True, sharey=True)
points_part = points[part]
for ax, (name, groups) in zip(axes, models.items()):
    ax.scatter(points_part[:, 0], points_part[:, 1], c=groups, cmap="tab10", s=5, alpha=0.4)
    ax.set_title(name)
    ax.set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.tight_layout()
plt.show()


# 8. finer clusters for the actual question. Four clusters describe the data well but they are
# too broad to judge a single profile: even the most one-sided of them is 19% human. Splitting
# the same features into more, smaller groups gives clusters that lean much harder one way
fine = KMeans(K_FINE, n_init=10, random_state=SEED).fit(X)
df["fine_cluster"] = fine.labels_
rate = truth.groupby(df.loc[labelled, "fine_cluster"]).agg(human_rate="mean", labelled="size")
rate["size"] = df["fine_cluster"].value_counts()
# a cluster is one-sided if nearly all of its labelled members agree with each other
rate["leans"] = np.where(rate["human_rate"] >= PURE_HUMAN, "human",
                         np.where(rate["human_rate"] <= PURE_NON_HUMAN, "non_human", "mixed"))
print("\n", K_FINE, "finer clusters, sorted by how human they look:")
print(rate.sort_values("human_rate").round(3))
print("one-sided clusters:", (rate["leans"] != "mixed").sum(), "of", K_FINE)


# 9. a profile sitting in a one-sided cluster but carrying the opposite label is a candidate
# mislabel: everything about its behaviour matches the profiles it was not grouped with
leaning = rate[rate["leans"] != "mixed"]
suspect = df["fine_cluster"].map(leaning["leans"]).where(labelled)
cluster_says = suspect.map({"human": 1, "non_human": 0})
flagged = df[cluster_says.notna() & (cluster_says != df["is_human"])].copy()
flagged["cluster_says"] = suspect[flagged.index]
flagged["cluster_human_rate"] = flagged["fine_cluster"].map(rate["human_rate"]).round(3)
# how far from its own cluster centre the profile sits: a small distance means it is a
# typical member of a cluster that disagrees with its label, so the flag is harder to dismiss
flagged["distance_to_centre"] = np.linalg.norm(
    X[flagged.index] - fine.cluster_centers_[flagged["fine_cluster"]], axis=1).round(2)

print("\nprofiles whose cluster contradicts their label:", len(flagged))
print(flagged["gender"].value_counts())
print(flagged.groupby("cluster_says")[["cluster_human_rate", "gender:confidence"]].mean().round(3))


# 10. check the flags against something the clustering never saw: how sure the crowd was.
# If the flags were noise the two distributions would sit on top of each other
print("\ncrowd confidence, flagged vs all labelled profiles:")
print(pd.DataFrame({"flagged": flagged["gender:confidence"].describe(),
                    "all": df.loc[labelled, "gender:confidence"].describe()}).round(3))
print("below full confidence - flagged:", round((flagged["gender:confidence"] < 1).mean(), 3),
      "| all:", round((df.loc[labelled, "gender:confidence"] < 1).mean(), 3))

plt.figure(figsize=(8, 4))
plt.hist([flagged["gender:confidence"], df.loc[labelled, "gender:confidence"]],
         bins=20, density=True, label=["flagged by clustering", "all labelled profiles"])
plt.xlabel("gender:confidence")
plt.ylabel("density")
plt.title("the crowd was less sure about the profiles the clusters flagged")
plt.legend()
plt.show()

# the association rules flagged profiles the same way, from different evidence.
# Profiles found by both methods are the strongest candidates
rules_file = OUT / "association_flagged.csv"
if rules_file.exists():
    by_rules = set(pd.read_csv(rules_file)["_unit_id"])
    both = by_rules & set(flagged["_unit_id"])
    flagged["also_by_rules"] = flagged["_unit_id"].isin(by_rules).astype(int)
    print("\nflagged by the association rules:", len(by_rules),
          "| by clustering:", len(flagged), "| by both:", len(both))


# 11. the 1037 profiles the crowd could not label still fall into a cluster, so the cluster
# they landed in is a suggestion for what they most likely are
unlabelled = df.loc[~labelled].copy()
unlabelled["suggested"] = unlabelled["fine_cluster"].map(leaning["leans"])
unlabelled["cluster_human_rate"] = unlabelled["fine_cluster"].map(rate["human_rate"]).round(3)
print("\nsuggestions for the unlabelled profiles:")
print(unlabelled["suggested"].value_counts(dropna=False))


# 12. write the three lists out for the report.
# The flagged list is sorted so the most one-sided cluster comes first, and inside a cluster
# the profiles closest to its centre come first
KEEP = ["_unit_id", "name", "gender", "gender:confidence", "cluster", "fine_cluster",
        "cluster_says", "cluster_human_rate", "distance_to_centre"]
if "also_by_rules" in flagged:
    KEEP.append("also_by_rules")
flagged = flagged.sort_values(["cluster_human_rate", "distance_to_centre"])
flagged[KEEP].to_csv(OUT / "cluster_flagged.csv", index=False)

suggestions = unlabelled[unlabelled["suggested"].notna()]
suggestions[["_unit_id", "name", "cluster", "fine_cluster", "cluster_human_rate",
             "suggested"]].to_csv(OUT / "cluster_suggested_labels.csv", index=False)

df[["_unit_id", "name", "gender", "is_human", "cluster", "fine_cluster"]].to_csv(
    OUT / "cluster_assignments.csv", index=False)
print("\nwritten:", [f.name for f in sorted(OUT.glob("cluster*.csv"))])
print("\nclearest candidates (typical members of clusters that disagree with their label):")
print(flagged.head(10)[["name", "gender", "gender:confidence", "cluster_says",
                        "cluster_human_rate", "distance_to_centre"]])
