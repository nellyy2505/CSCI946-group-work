# 08_text.py — nltk tokens, TF-IDF + logistic regression and gensim LDA on the two text fields

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from gensim import corpora
from gensim.models import LdaModel

plt.rcParams.update({"font.size": 11})   # figures are placed at 4-6.2 in wide in the report

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

SEED = 7
MIN_SCORE = 0.90          # fixed in advance, the same vote cut-off as 05 and 07
MAX_FEATURES = 2000
N_TOPICS = 8
FIELDS = {"description": ["desc_clean"], "tweet": ["text_clean"], "both": ["desc_clean", "text_clean"]}


# 1. nltk resources
# use a local copy when there is one, download only when missing
for resource, package in [("corpora/stopwords", "stopwords"), ("tokenizers/punkt_tab", "punkt_tab")]:
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            nltk.download(package, quiet=True)
            nltk.data.find(resource)
        except Exception:          # no network, a blocked proxy, or a failed download all end here
            sys.exit("nltk resource '%s' is missing and could not be downloaded.\n"
                     "Run once, with internet access:  python -m nltk.downloader stopwords punkt_tab"
                     % package)
from nltk.corpus import stopwords
STOP = set(stopwords.words("english")) | {"http", "https", "com", "www", "amp"}


# 2. Load
# every profile, and which split 02 put each labelled profile in
df = pd.read_csv(PROC / "twitter_full.csv")
df["desc_clean"] = df["desc_clean"].fillna("")
df["text_clean"] = df["text_clean"].fillna("")
df["split"] = "unknown"
for name in ["train", "validation", "test"]:
    ids = pd.read_csv(PROC / ("twitter_" + name + ".csv"), usecols=["_unit_id"])["_unit_id"]
    df.loc[df["_unit_id"].isin(ids), "split"] = name
labelled = df["label"] != "unknown"
print("shape:", df.shape, "| labelled:", labelled.sum(), "| unknown:", (~labelled).sum())
print(df["split"].value_counts().to_string())

train, val, test = (df[df["split"] == s] for s in ["train", "validation", "test"])
y_train, y_val, y_test = (part["is_human"].astype(int) for part in [train, val, test])


# 3. Tokens
# tokenise and remove stop words
def tokens(s):
    return [w for w in nltk.word_tokenize(s) if w not in STOP and len(w) > 2]


df["desc_tokens"] = df["desc_clean"].map(tokens)
df["text_tokens"] = df["text_clean"].map(tokens)
print("mean words kept - description:", round(df["desc_tokens"].str.len().mean(), 1),
      "| tweet:", round(df["text_tokens"].str.len().mean(), 1))


# 4. Words by class
# conditional frequency distribution of the description words
label = df["label"]
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

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.8))
for ax, (name, sub) in zip(axes, [("non_human", freq.nsmallest(15, "human_share")),
                                  ("human", freq.nlargest(15, "human_share"))]):
    ax.barh(range(len(sub)), sub["human_share"], color="tab:blue" if name == "human" else "tab:orange")
    ax.set_yticks(range(len(sub)), sub.index)
    # dashed line: human share of all labelled profiles (explained in the report caption)
    ax.axvline((label[labelled] == "human").mean(), color="black", linestyle="--")
    ax.set_xlabel("share of uses that are\nhuman profiles")
    ax.set_title("description words\nleaning " + name)
plt.tight_layout()
plt.savefig(OUT / "fig_text_words_by_class.png", dpi=150, bbox_inches="tight")
plt.show()


# 5. Model comparison
# TF-IDF + logistic regression per text field, 5-fold CV on the training split
# the vectoriser sits inside the pipeline, so its vocabulary is learnt from training rows only
def text_model(columns):
    tfidf = ColumnTransformer([(c, TfidfVectorizer(max_features=MAX_FEATURES, min_df=5, sublinear_tf=True), c)
                               for c in columns])
    return Pipeline([("tfidf", tfidf), ("logistic", LogisticRegression(max_iter=3000))])


cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv_scores = {}
rows = []
for name, columns in FIELDS.items():
    cv_scores[name] = cross_val_score(text_model(columns), train[columns], y_train, cv=cv, scoring="accuracy")
    val_acc = accuracy_score(y_val, text_model(columns).fit(train[columns], y_train).predict(val[columns]))
    rows.append({"field": name, "cv_mean": cv_scores[name].mean(), "cv_std": cv_scores[name].std(),
                 "validation_acc": val_acc})
comparison = pd.DataFrame(rows).set_index("field")
print("\ntext fields - 5-fold CV on the training split, then accuracy on the validation split:")
print(comparison.round(4).to_string())
print("always human (validation): %.4f" % y_val.mean())

first, second = comparison["cv_mean"].nlargest(2).index
t, p = ttest_ind(cv_scores[first], cv_scores[second])
print("t-test on the CV scores, %s vs %s: t = %.3f, p = %.4f" % (first, second, t, p))


# 6. Test
# choose the field on validation, fit on the training split, test once
FIELD = comparison["validation_acc"].idxmax()
columns = FIELDS[FIELD]
model = text_model(columns).fit(train[columns], y_train)
pred_test = model.predict(test[columns])
print("\nchosen on validation: %s" % FIELD)
print("text model (%s), test accuracy: %.4f" % (FIELD, accuracy_score(y_test, pred_test)))
cm = confusion_matrix(y_test, pred_test)
print("confusion matrix on the test split [rows recorded, cols predicted; non_human, human]:\n", cm)
print(classification_report(y_test, pred_test, target_names=["non_human", "human"], digits=3))

fig, ax = plt.subplots(figsize=(4.6, 3.8))
image = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["non_human", "human"])
ax.set_yticks([0, 1], ["non_human", "human"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(image)
plt.xlabel("predicted")
plt.ylabel("recorded")
plt.title("text model (%s): test split" % FIELD)
plt.tight_layout()
plt.savefig(OUT / "fig_text_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# the words the model leans on most (stop words stay in the TF-IDF: pronouns carry the signal)
vocab = [n.replace("desc_clean__", "desc:").replace("text_clean__", "tweet:")
         for n in model.named_steps["tfidf"].get_feature_names_out()]
weights = pd.Series(model.named_steps["logistic"].coef_[0], index=vocab).sort_values()
print("strongest non-human words:\n", weights.head(12).round(2).to_dict())
print("strongest human words:\n", weights.tail(12).round(2).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.8))
for ax, (name, sub) in zip(axes, [("non_human", weights.head(15)), ("human", weights.tail(15)[::-1])]):
    ax.barh(range(len(sub)), sub.abs(), color="tab:blue" if name == "human" else "tab:orange")
    ax.set_yticks(range(len(sub)), sub.index)
    ax.invert_yaxis()
    ax.set_xlabel("|coefficient|")
    ax.set_title("words the model weights\ntowards " + name)
plt.tight_layout()
plt.savefig(OUT / "fig_text_model_words.png", dpi=150, bbox_inches="tight")
plt.show()


# 7. LDA topics
# topics of the labelled descriptions
docs = df.loc[labelled & (df["desc_tokens"].str.len() >= 3), ["is_human", "desc_tokens"]].copy()
dictionary = corpora.Dictionary(docs["desc_tokens"])
dictionary.filter_extremes(no_below=10, no_above=0.4)
docs["bag"] = [dictionary.doc2bow(d) for d in docs["desc_tokens"]]
# a bag emptied by filter_extremes gets a uniform topic mix, so it would all land in topic 0
empty = docs["bag"].str.len() == 0
print("\ndescriptions with 3+ words: %d | empty after filter_extremes, left without a topic: %d"
      % (len(docs), empty.sum()))
docs = docs[~empty]
lda = LdaModel(list(docs["bag"]), num_topics=N_TOPICS, id2word=dictionary, passes=3, random_state=SEED)
print("LDA topics in the descriptions:")
for i, words in lda.print_topics(num_words=8):
    print(" topic %d: %s" % (i, words))

# how human each topic is, using the dominant topic of each description
docs["topic"] = [max(lda.get_document_topics(b), key=lambda t: t[1])[0] for b in docs["bag"]]
topics = docs.groupby("topic").agg(profiles=("topic", "size"), human_rate=("is_human", "mean")).round(3)
topics["top_words"] = [", ".join(w for w, _ in lda.show_topic(i, topn=5)) for i in topics.index]
human_rate = df.loc[labelled, "is_human"].mean()
print("\ntopic composition (human rate of all labelled profiles %.3f):" % human_rate)
print(topics.to_string())
print("topic human rate ranges from %.3f to %.3f" % (topics["human_rate"].min(), topics["human_rate"].max()))

plt.figure(figsize=(7.4, 5))
plt.barh(range(len(topics)), topics["human_rate"], color="tab:blue")
plt.yticks(range(len(topics)), ["%d: %s\n(n=%d)" % (i, w, n) for i, w, n in
                                zip(topics.index, topics["top_words"], topics["profiles"])])
plt.gca().invert_yaxis()
plt.axvline(human_rate, color="black", linestyle="--")   # all labelled profiles (see caption)
plt.xlim(0, 1)
plt.xlabel("human rate of the descriptions in the topic")
plt.title("how human each description topic is\n(LDA, %d topics)" % N_TOPICS)
plt.tight_layout()
plt.savefig(OUT / "fig_text_lda_topics.png", dpi=150, bbox_inches="tight")
plt.show()


# 8. Predictions and flags
# apply the fitted model to every profile and vote where score >= MIN_SCORE
prob = model.predict_proba(df[columns])[:, 1]
df["says"] = np.where(prob >= 0.5, "human", "non_human")
df["score"] = np.maximum(prob, 1 - prob).round(3)
df["votes"] = (df["score"] >= MIN_SCORE).astype(int)
df["recorded"] = df["label"]

# printed for information only; the cut-off is fixed above
lab = df[labelled]
disagree = lab["says"] != lab["recorded"]
print("\nlabelled profiles the model disagrees with, by cut-off:")
print(pd.DataFrame({"min_score": [0.70, 0.80, 0.90, 0.95],
                    "profiles": [int((disagree & (lab["score"] >= c)).sum()) for c in [0.70, 0.80, 0.90, 0.95]]}
                   ).to_string(index=False))

predictions = df[["_unit_id", "says", "score", "recorded", "votes", "split"]]
predictions.to_csv(OUT / "text_predictions.csv", index=False)
print("text_predictions.csv:", len(predictions), "rows | votes cast:", predictions["votes"].sum())
print(predictions.groupby("split")["votes"].agg(["size", "sum"]).rename(columns={"size": "rows", "sum": "votes"}))

flagged = lab[disagree & (lab["votes"] == 1)]
result = flagged[["_unit_id", "says", "score", "recorded", "votes", "split",
                  "gender", "name", "gender:confidence"]].sort_values("score", ascending=False)
result.to_csv(OUT / "text_flagged.csv", index=False)
print("\nprofiles nominated (text_flagged.csv):", len(result))
print(pd.crosstab(result["recorded"], result["says"]))
print("by split (train rows were scored by a model that saw their label):")
print(result["split"].value_counts().reindex(["train", "validation", "test"]).to_string())
print(result.head(10)[["name", "gender", "recorded", "says", "score"]])


# 9. Crowd confidence
# check the nominations against crowd confidence, which the text model never used
def below_full(rows):
    # share of the rows the crowd was not fully sure about
    return (rows["gender:confidence"] < 1).mean()


unsure_by_label = {name: below_full(lab[lab["recorded"] == name]) for name in ["human", "non_human"]}
mix = result["recorded"].value_counts(normalize=True)
nominated = below_full(result)
expected = sum(mix.get(name, 0) * unsure_by_label[name] for name in unsure_by_label)
print("\nbelow full crowd confidence - nominated: %.3f | all labelled with the same recorded-label mix: "
      "%.3f | all labelled: %.3f" % (nominated, expected, below_full(lab)))

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
groups = [result["gender:confidence"], lab["gender:confidence"]]
axes[0].hist(groups, bins=20, weights=[np.full(len(g), 100 / len(g)) for g in groups],
             color=["tab:red", "grey"], label=["nominated by the text model", "all labelled profiles"])
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
axes[1].bar(2, below_full(lab), color="grey")
for x, value in enumerate([nominated, expected, below_full(lab)]):
    axes[1].text(x, value + 0.01, "%.3f" % value, ha="center")
axes[1].set_ylim(0, max(nominated, expected) * 1.2)
axes[1].set_xticks([0, 1, 2], ["nominated", "same label\nmix", "all\nlabelled"])
axes[1].set_ylabel("share below full confidence")
axes[1].set_title("below full crowd confidence")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3))
plt.tight_layout()
plt.savefig(OUT / "fig_text_crowd_confidence.png", dpi=150, bbox_inches="tight")
plt.show()
