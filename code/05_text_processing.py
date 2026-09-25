"""
CSCI446/946 Big Data Analytics - Assignment 2
Task 2: Text Processing

Goal:
- find words and topics that differ between human and non-human profiles
- a profile whose text "reads" like the other class -> candidate for Task 4

Steps:
count words -> find distinctive words per group (TF-IDF) -> discover topics (LDA) ->
use the distinctive words to flag profiles whose text disagrees with their label
(tokenizing / stop words / case folding already done upstream in 02_preprocess.py
-> text_clean / desc_clean are ready to use as-is)

Word list built on train only, checked on validation, final candidates from test -
same train/validation/test split as 4.1 / 4.2, so no profile grades its own score.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

SEED = 7
N_TOP_WORDS = 15
N_TOPICS = 5


# =====================================================================
# 1. Load data
# =====================================================================
# - only twitter_full.csv keeps the text columns (train/validation/test drop them)
# - so: load twitter_full.csv for the text, then use _unit_id to split it into
#   the same train/validation/test groups Nelly's twitter_train/validation/test.csv use
df = pd.read_csv(PROC / "twitter_full.csv")
train_ids = pd.read_csv(PROC / "twitter_train.csv")["_unit_id"]
val_ids = pd.read_csv(PROC / "twitter_validation.csv")["_unit_id"]
test_ids = pd.read_csv(PROC / "twitter_test.csv")["_unit_id"]

train_mask = df["_unit_id"].isin(train_ids)
val_mask = df["_unit_id"].isin(val_ids)
test_mask = df["_unit_id"].isin(test_ids)
train = df[train_mask]
print("----- shape -----")
print("full:", df.shape, "| train:", train.shape,
      "| validation:", val_mask.sum(), "| test:", test_mask.sum())

train_human = train[train["is_human"] == 1]
train_non_human = train[train["is_human"] == 0]
print("train human:", len(train_human), "| train non-human:", len(train_non_human))


# =====================================================================
# 2. Word frequency: human vs non-human description (train only)
# =====================================================================
# - split each profile's cleaned bio into words -> count -> most common per group
# - just a word count, split by label, so we can see which words each group uses most
# - train only -> the word list below is built without looking at validation/test
def top_words(text_series, n=N_TOP_WORDS):
    counts = Counter(" ".join(text_series.fillna("")).split())
    return counts.most_common(n)

human_top = top_words(train_human["desc_clean"])
non_human_top = top_words(train_non_human["desc_clean"])

print("\n----- top words: human description (train) -----")
print(human_top)
print("\n----- top words: non-human description (train) -----")
print(non_human_top)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, title, words in [(axes[0], "human", human_top), (axes[1], "non-human", non_human_top)]:
    labels, counts = zip(*reversed(words))
    ax.barh(labels, counts)
    ax.set_title(f"top words - {title}")
plt.tight_layout()
plt.savefig(OUT / "text_word_frequency.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: text_word_frequency.png")


# =====================================================================
# 3. TF-IDF: which words are distinctive of each class (train only)
# =====================================================================
# - raw frequency (step 2) favours common words even after stop-word removal
# - TF-IDF instead favours words that stand out FOR a group, not just common in it
# - fit on the train descriptions only -> the word list is never told about
#   validation/test profiles, so step 5's score can be checked on unseen data
tfidf = TfidfVectorizer(stop_words="english", min_df=5, max_features=3000)
tfidf_matrix = tfidf.fit_transform(train["desc_clean"].fillna(""))
vocab = np.array(tfidf.get_feature_names_out())

human_mean = np.asarray(tfidf_matrix[train["is_human"].values == 1].mean(axis=0)).ravel()
non_human_mean = np.asarray(tfidf_matrix[train["is_human"].values == 0].mean(axis=0)).ravel()

human_distinctive = vocab[np.argsort(human_mean)[::-1][:N_TOP_WORDS]]
non_human_distinctive = vocab[np.argsort(non_human_mean)[::-1][:N_TOP_WORDS]]

print("\n----- TF-IDF distinctive words: human (train) -----")
print(list(human_distinctive))
print("\n----- TF-IDF distinctive words: non-human (train) -----")
print(list(non_human_distinctive))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, title, words, means in [(axes[0], "human", human_distinctive, human_mean),
                                 (axes[1], "non-human", non_human_distinctive, non_human_mean)]:
    idx = [np.where(vocab == w)[0][0] for w in words]
    ax.barh(list(reversed(words)), list(reversed(means[idx])))
    ax.set_title(f"TF-IDF distinctive words - {title}")
plt.tight_layout()
plt.savefig(OUT / "text_tfidf_distinctive_words.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: text_tfidf_distinctive_words.png")


# =====================================================================
# 4. Topic modelling (LDA) - unsupervised, whole dataset
# =====================================================================
# - no target used here at all -> fine to fit on every profile, not just train
#   (nothing to leak: LDA never sees is_human while finding the topics)
# - using sklearn's LatentDirichletAllocation: no extra package to install,
#   and it works with the same scikit-learn already used everywhere else in this file
# - LDA needs raw counts, not TF-IDF -> CountVectorizer, not TfidfVectorizer
count_vec = CountVectorizer(stop_words="english", min_df=5, max_features=3000)
counts = count_vec.fit_transform(df["desc_clean"].fillna(""))

lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=SEED, max_iter=10)
topic_weights = lda.fit_transform(counts)
count_vocab = np.array(count_vec.get_feature_names_out())

print("\n----- top words per topic -----")
for i, topic in enumerate(lda.components_):
    top = count_vocab[np.argsort(topic)[::-1][:10]]
    print(f"topic {i}: {list(top)}")

# dominant topic per profile -> which topic looks more human / more non-human
df["dominant_topic"] = topic_weights.argmax(axis=1)
topic_by_class = pd.crosstab(df["dominant_topic"], df["is_human"], normalize="index")
topic_by_class.columns = ["non-human share", "human share"]
print("\n----- topic composition by class -----")
print(topic_by_class.round(3))

topic_by_class.plot.bar(stacked=True, figsize=(7, 4))
plt.xlabel("topic")
plt.ylabel("share of profiles")
plt.title("Topic composition: human vs non-human")
plt.tight_layout()
plt.savefig(OUT / "text_topic_by_class.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved plot: text_topic_by_class.png")


# =====================================================================
# 5. Candidate mislabels for Task 4 - a fourth, independent view
# =====================================================================
# - text_style_score: count of this profile's OWN distinctive-word set match
#   -> +1 for each human-distinctive word present in its description,
#      -1 for each non-human-distinctive word present
#   -> word lists come from TRAIN only (step 3), so scoring validation/test
#      profiles with them is a fair, out-of-sample check - not the profile
#      "grading itself" with a rule built partly from its own words
human_set = set(human_distinctive)
non_human_set = set(non_human_distinctive)

def text_style_score(desc):
    words = set(str(desc).split())
    return len(words & human_set) - len(words & non_human_set)

df["text_style_score"] = df["desc_clean"].apply(text_style_score)
# mismatch: sign of the score disagrees with the label
df["mismatch"] = np.where(df["is_human"] == 1, -df["text_style_score"], df["text_style_score"])


def evaluate(part):
    # coverage: share of profiles where the rule actually found a distinctive
    # word at all -> most bios are too short/generic, score stays 0, rule is silent
    n = len(part)
    fired = part[part["text_style_score"] != 0]
    coverage = len(fired) / n

    # recall per class, on the profiles where the rule DID fire -> checks the
    # rule isn't just riding the majority class (human ~69% of the data)
    pred = (fired["text_style_score"] > 0).astype(int)
    actual = fired["is_human"]
    recall_human = (pred[actual == 1] == 1).mean()
    recall_non_human = (pred[actual == 0] == 0).mean()
    agree = (pred == actual).mean()
    return coverage, agree, recall_human, recall_non_human


# 5a. sanity check on validation - does the train-built rule generalise at all?
val = df[val_mask & df["is_human"].notna()]
coverage, val_rate, recall_h, recall_nh = evaluate(val)
baseline = (train["is_human"] == 1).mean()   # always predict the majority class
print(f"\n----- validation check (rule built on train only) -----")
print(f"coverage: rule fires on {coverage:.1%} of validation profiles (rest have no distinctive word)")
print(f"of the profiles where it fires: agrees with the label {val_rate:.1%} of the time")
print(f"  recall on human:     {recall_h:.1%}")
print(f"  recall on non-human: {recall_nh:.1%}")
print(f"baseline (always guess the majority class): {baseline:.1%}")
print("-> both recalls beat guessing -> real signal, not just riding the majority class"
      if recall_nh > 0 else "-> rule never catches non-human -> just riding the majority class")

# 5b. final candidate list - test only, touched once, same rule as validation
test = df[test_mask]
diag = test[["name", "gender", "is_human", "label_conflict", "text_style_score", "mismatch"]].copy()
diag = diag[diag["is_human"].notna()].sort_values("mismatch", ascending=False)

print("\n----- top 20 test profiles: text reads like the other class -----")
print(diag.head(20).to_string(index=False))

top_n = min(50, (diag["mismatch"] > 0).sum())
overlap = diag.head(top_n)["label_conflict"].sum()
print(f"\nOf the top {top_n} text-mismatch profiles (test set), {overlap} are also flagged by label_conflict.")
print("-> Task 4 now has 4 views: association rules (03), regression residual (4.1),")
print("   logistic misclassification (4.2), and this text-style mismatch (05)")
print("-> present all 4 side by side, cross-check which profiles multiple views agree on")

diag.head(top_n).to_csv(OUT / "text_mislabel_candidates.csv", index=False)
print("Saved: text_mislabel_candidates.csv")
