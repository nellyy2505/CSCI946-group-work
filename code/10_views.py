# 10_views.py — how well each view of a profile predicts is_human on its own (Task 4, multiple views)

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

plt.rcParams.update({"font.size": 11})   # figures are placed at 4-6.2 in wide in the report

pd.set_option("display.width", 200)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

MAX_FEATURES = 2000       # same TF-IDF settings as 08_text.py


# 1. Load
# the training and validation splits written by 02, and the two text fields; the test split stays
# untouched here because it was already used once to score the chosen models in 05, 07 and 08
train = pd.read_csv(PROC / "twitter_train.csv")
validation = pd.read_csv(PROC / "twitter_validation.csv")
text = pd.read_csv(PROC / "twitter_full.csv", usecols=["_unit_id", "desc_clean", "text_clean"]).fillna("")
train = train.merge(text, on="_unit_id", how="left", validate="one_to_one")
validation = validation.merge(text, on="_unit_id", how="left", validate="one_to_one")
y_train, y_val = train["is_human"].astype(int), validation["is_human"].astype(int)
print("train:", train.shape, "validation:", validation.shape)


# 2. Views
# the structured features grouped by the 02_preprocess.py step that made them
VIEWS = {
    "activity": ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
                 "account_age_days"],                                                     # 02 step 7
    "profile flags": ["default_image", "has_retweets", "has_coord", "location_missing",
                      "timezone_missing", "name_n_digits"],                               # 02 step 8
    "colour": ["link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b",
               "link_default", "sidebar_default"],                                        # 02 step 6
    "text counts": ["text_len", "desc_len", "text_n_urls", "text_n_mentions", "text_n_hashtags",
                    "desc_missing", "desc_has_url", "text_has_emoji"],                    # 02 step 5
    "time zone": [c for c in train.columns if c.startswith("tz_")],                       # 02 step 9
}
id_cols = ["_unit_id", "is_human", "label", "name", "gender", "gender:confidence", "label_conflict",
           "desc_clean", "text_clean"]
structured = [c for c in train.columns if c not in id_cols]
grouped = [c for cols in VIEWS.values() for c in cols]
# every structured column of the processed files sits in exactly one view
assert sorted(grouped) == sorted(structured), set(grouped) ^ set(structured)
VIEWS["all structured"] = structured
for name, cols in VIEWS.items():
    print("%-15s %2d: %s" % (name, len(cols), ", ".join(cols)))


# 3. TF-IDF
# both text fields, fitted on the training split only
tf_desc = TfidfVectorizer(max_features=MAX_FEATURES, min_df=5, sublinear_tf=True).fit(train["desc_clean"])
tf_text = TfidfVectorizer(max_features=MAX_FEATURES, min_df=5, sublinear_tf=True).fit(train["text_clean"])


def tfidf(part):
    return hstack([tf_desc.transform(part["desc_clean"]), tf_text.transform(part["text_clean"])]).tocsr()


# 4. Accuracy per view
# plain logistic regression on the training split, accuracy on the validation split
X = {name: (train[cols].to_numpy(float), validation[cols].to_numpy(float)) for name, cols in VIEWS.items()}
X["text (TF-IDF)"] = (tfidf(train), tfidf(validation))
X["all structured + text"] = (hstack([csr_matrix(X["all structured"][0]), X["text (TF-IDF)"][0]]).tocsr(),
                              hstack([csr_matrix(X["all structured"][1]), X["text (TF-IDF)"][1]]).tocsr())

rows = []
for name, (X_train, X_val) in X.items():
    model = LogisticRegression(max_iter=3000).fit(X_train, y_train)
    rows.append({"view": name, "features": X_train.shape[1],
                 "validation_accuracy": accuracy_score(y_val, model.predict(X_val))})
views = pd.DataFrame(rows)
baseline = y_val.mean()
views.to_csv(OUT / "views_accuracy.csv", index=False)
print("\nlogistic regression per view, validation accuracy (always human: %.4f):" % baseline)
print(views.round(4).to_string(index=False))

plt.figure(figsize=(6.5, 4.2))
bars = plt.barh(views["view"], views["validation_accuracy"], color="tab:gray")
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.axvline(baseline, color="black", linestyle="--", label="always human (%.3f)" % baseline)
plt.gca().invert_yaxis()
plt.xlim(0.5, 0.95)
plt.xlabel("validation accuracy (logistic regression fitted on the training split)", fontsize=9)
plt.title("how well each view of a profile predicts\nhuman vs non-human")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22))   # below, clear of the bar labels
plt.tight_layout()
plt.savefig(OUT / "fig_views_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
