# CSCI446/946 Big Data Analytics - Assignment 2
# 04 - Clustering: group the profiles by behaviour and use the groups to question the labels

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

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
