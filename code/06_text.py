# CSCI446/946 Big Data Analytics - Assignment 2
# 06 - Text processing

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from gensim import corpora
from gensim.models import LdaModel

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

SEED = 7
MAX_FEATURES = 2000
N_TOPICS = 8
MIN_SCORE = 0.85          # chosen from the sweep below

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
STOP = set(stopwords.words("english")) | {"http", "https", "com", "www", "amp"}


# Task 1: tokenise and remove stop words
df = pd.read_csv(PROC / "twitter_full.csv")
label = df["label"]
labelled = label != "unknown"
df["desc_clean"] = df["desc_clean"].fillna("")
df["text_clean"] = df["text_clean"].fillna("")
print("shape:", df.shape, "| labelled:", labelled.sum())


def tokens(s):
    return [w for w in nltk.word_tokenize(s) if w not in STOP and len(w) > 2]


df["desc_tokens"] = df["desc_clean"].map(tokens)
df["text_tokens"] = df["text_clean"].map(tokens)
print("mean words kept - description:", round(df["desc_tokens"].str.len().mean(), 1),
      "| tweet:", round(df["text_tokens"].str.len().mean(), 1))


# Task 2a: conditional frequency distribution - which words belong to which class
pairs = [(lab, w) for lab, toks in zip(label[labelled], df.loc[labelled, "desc_tokens"]) for w in toks]
cfd = nltk.ConditionalFreqDist(pairs)
print("\nmost common description words per class:")
cfd.tabulate(conditions=["human", "non_human"],
             samples=[w for w, _ in cfd["human"].most_common(12)])

# words that lean hardest to one class, relative to how often they appear overall
freq = pd.DataFrame({c: pd.Series(dict(cfd[c])) for c in ["human", "non_human"]}).fillna(0)
freq = freq[freq.sum(axis=1) >= 40]
freq["human_share"] = freq["human"] / freq.sum(axis=1)
print("\nmost non-human words:\n", freq.nsmallest(15, "human_share").round(2))
print("\nmost human words:\n", freq.nlargest(15, "human_share").round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, sub) in zip(axes, [("non_human", freq.nsmallest(15, "human_share")),
                                  ("human", freq.nlargest(15, "human_share"))]):
    ax.barh(range(len(sub)), sub["human_share"], color="tab:blue" if name == "human" else "tab:orange")
    ax.set_yticks(range(len(sub)), sub.index)
    ax.axvline((label[labelled] == "human").mean(), color="red", linestyle="--")
    ax.set_xlabel("share of uses that are human profiles")
    ax.set_title("words leaning " + name)
plt.tight_layout()
plt.savefig(OUT / "fig_text_words_by_class.png", dpi=120, bbox_inches="tight")
plt.show()


# Task 2b: TF-IDF, one block per text field
tf_desc = TfidfVectorizer(max_features=MAX_FEATURES, min_df=5, sublinear_tf=True)
tf_text = TfidfVectorizer(max_features=MAX_FEATURES, min_df=5, sublinear_tf=True)
Xd = tf_desc.fit_transform(df["desc_clean"])
Xt = tf_text.fit_transform(df["text_clean"])
from scipy.sparse import hstack
X = hstack([Xd, Xt]).tocsr()
vocab = np.concatenate([["desc:" + w for w in tf_desc.get_feature_names_out()],
                        ["tweet:" + w for w in tf_text.get_feature_names_out()]])
print("\nTF-IDF features:", X.shape[1], "(description", Xd.shape[1], "+ tweet", Xt.shape[1], ")")

y = df.loc[labelled, "is_human"].astype(int)
X_lab = X[labelled.to_numpy()]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
clf = LogisticRegression(max_iter=3000, class_weight="balanced")

# each field on its own, then together
for name, M in [("description only", Xd[labelled.to_numpy()]),
                ("tweet only", Xt[labelled.to_numpy()]),
                ("both", X_lab)]:
    s = cross_val_score(clf, M, y, cv=cv, scoring="accuracy")
    print("  %-18s acc %.4f  (+/- %.4f)" % (name, s.mean(), s.std()))
print("  %-18s acc %.4f" % ("always human", y.mean()))

prob = cross_val_predict(clf, X_lab, y, cv=cv, method="predict_proba")[:, 1]
pred = (prob >= 0.5).astype(int)
print("\naccuracy:", round(accuracy_score(y, pred), 4))
cm = confusion_matrix(y, pred)
print("confusion matrix [rows actual, cols predicted]:\n", cm)

fig, ax = plt.subplots(figsize=(5, 4.5))
image = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["non_human", "human"])
ax.set_yticks([0, 1], ["non_human", "human"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(image)
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("text model: confusion matrix")
plt.tight_layout()
plt.savefig(OUT / "fig_text_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()

# the words the model leans on most
clf.fit(X_lab, y)
weights = pd.Series(clf.coef_[0], index=vocab).sort_values()
print("\nstrongest non-human words:\n", weights.head(12).round(2).to_dict())
print("strongest human words:\n", weights.tail(12).round(2).to_dict())


# Task 3: LDA topics on the descriptions
docs = [t for t in df.loc[labelled, "desc_tokens"] if len(t) >= 3]
dictionary = corpora.Dictionary(docs)
dictionary.filter_extremes(no_below=10, no_above=0.4)
bags = [dictionary.doc2bow(d) for d in docs]
lda = LdaModel(bags, num_topics=N_TOPICS, id2word=dictionary, passes=3, random_state=SEED)
print("\nLDA topics in the descriptions:")
for i, words in lda.print_topics(num_words=8):
    print(" topic %d: %s" % (i, words))

# how human each topic is, using the dominant topic of each description
kept = df.loc[labelled][df.loc[labelled, "desc_tokens"].str.len() >= 3].copy()
kept["topic"] = [max(lda.get_document_topics(b), key=lambda t: t[1])[0] for b in bags]
topics = kept.groupby("topic").agg(profiles=("topic", "size"),
                                   human_rate=("is_human", "mean")).round(3)
topics["top_words"] = [", ".join(w for w, _ in lda.show_topic(i, topn=5)) for i in topics.index]
print("\ntopic composition (human rate of the whole data %.3f):" % y.mean())
print(topics.to_string())

plt.figure(figsize=(9, 4))
plt.bar(topics.index, topics["human_rate"], color="tab:blue")
plt.axhline(y.mean(), color="red", linestyle="--", label="rate of the whole data")
plt.xticks(topics.index)
plt.xlabel("LDA topic")
plt.ylabel("human rate")
plt.title("how human each description topic is")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_text_lda_topics.png", dpi=120, bbox_inches="tight")
plt.show()


# Task 4: nominate profiles whose words contradict their label
score_all = np.maximum(prob, 1 - prob)
says_all = np.where(prob >= 0.5, "human", "non_human")
disagree = says_all != label[labelled].to_numpy()

print("\nthreshold sweep (crowd was unsure about %.3f of all labelled profiles)"
      % (df.loc[labelled, "gender:confidence"] < 1).mean())
conf_all = df.loc[labelled, "gender:confidence"].to_numpy()
sweep = []
for cut in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    hit = disagree & (score_all >= cut)
    sweep.append({"min_score": cut, "profiles": int(hit.sum()),
                  "crowd_unsure": (conf_all[hit] < 1).mean() if hit.sum() else np.nan})
print(pd.DataFrame(sweep).round(3).to_string(index=False))

hit = disagree & (score_all >= MIN_SCORE)
lab_rows = df.loc[labelled]

# output contract: _unit_id, says, score
result = pd.DataFrame({
    "_unit_id": lab_rows["_unit_id"].to_numpy()[hit],
    "says": says_all[hit],
    "score": score_all[hit].round(3),
    "recorded": label[labelled].to_numpy()[hit],
    "gender": lab_rows["gender"].to_numpy()[hit],
    "name": lab_rows["name"].to_numpy()[hit],
    "gender:confidence": conf_all[hit],
}).sort_values("score", ascending=False)
result.to_csv(OUT / "text_flagged.csv", index=False)

print("\nprofiles nominated:", len(result))
print(pd.crosstab(result["recorded"], result["says"]))
print(result.head(10)[["name", "gender", "recorded", "says", "score"]])


# check the nominations against crowd confidence, which the text never used
plt.figure(figsize=(8, 4))
groups = [result["gender:confidence"], pd.Series(conf_all)]
plt.hist(groups, bins=20, weights=[np.full(len(g), 100 / len(g)) for g in groups],
         label=["nominated by the text model", "all labelled profiles"])
plt.xlabel("gender:confidence")
plt.ylabel("% of the group")
plt.title("crowd confidence, nominated vs all")
plt.legend()
plt.savefig(OUT / "fig_text_crowd_confidence.png", dpi=120, bbox_inches="tight")
plt.show()
