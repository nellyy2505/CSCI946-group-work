# CSCI446/946 Big Data Analytics - Assignment 2
# 02 - Data preprocessing: fix the problems found in 01_eda.py

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "twitter_user_data.csv"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DATE_FORMAT = "%m/%d/%y %H:%M"
SEED = 7

# the target every model predicts
TARGET = "is_human"          # 1 = human (male/female), 0 = non-human (brand), NaN = unknown
LABEL = "label"              # "human" / "non_human" / "unknown"


def repair_text(s):
    if not isinstance(s, str):
        return ""
    try:
        s = s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        s = re.sub("_Ÿ[^\x00-\x7f]{0,3}", " ", s)   # emoji remnants
        s = re.sub(r"[^\x00-\x7f]+", " ", s)                # other unrepairable characters
        s = re.sub(r"(?<!\w)_(?!\w)", " ", s)               # leftover placeholders
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def clean_words(s):
    # lower-case words only: no links, mentions, numbers or punctuation
    s = re.sub(r"https?://\S+|@\w+", " ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", s)).strip()


def fix_colour(code):
    # restore a six-character hex code; codes saved as numbers (e.g. 2.21E+09) cannot be restored
    code = str(code)
    if "E+" in code:
        return np.nan
    date = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{2})", code)   # e.g. 2FEB45 saved as 2-Feb-45
    if date:
        code = "".join(date.groups())
    return code.zfill(6).upper()


# 1. load the raw data
df = pd.read_csv(DATA, encoding="mac_roman")
print("raw:", df.shape)


# 2. remove records whose profile was unavailable (no label)
df = df[df["profile_yn"] == "yes"].copy()
print("profile available:", df.shape)


# 3. one record per account; flag accounts labelled both human and brand
group = df["gender"].map({"male": "human", "female": "human", "brand": "brand"})
df["label_conflict"] = group.groupby(df["name"]).transform("nunique").gt(1).astype(int)
df = df.sort_values(["name", "_golden", "gender:confidence", "_unit_id"],
                    ascending=[True, True, False, True])
df = df.drop_duplicates("name").sort_values("_unit_id")
print("one record per account:", df.shape, "| label conflicts:", df["label_conflict"].sum())


# 4. label: 1 human, 0 non-human, missing for unknown
df[TARGET] = df["gender"].map({"male": 1, "female": 1, "brand": 0})
df[LABEL] = df[TARGET].map({1: "human", 0: "non_human"}).fillna("unknown")
print("gender ->", TARGET)
print(pd.crosstab(df["gender"], df[LABEL]))


# 5. text: repair, cleaned words, and counts
df["text_has_emoji"] = df["text"].str.contains("_Ÿ").astype(int)
df["desc_missing"] = df["description"].isna().astype(int)
for col in ["text", "description"]:
    df[col] = df[col].map(repair_text)
df["text_clean"] = df["text"].map(clean_words)
df["desc_clean"] = df["description"].map(clean_words)
df["text_len"] = df["text"].str.len()
df["desc_len"] = df["description"].str.len()
df["text_n_urls"] = df["text"].str.count(r"https?://")
df["text_n_mentions"] = df["text"].str.count(r"@\w+")
df["text_n_hashtags"] = df["text"].str.count(r"#\w+")
df["desc_has_url"] = df["description"].str.contains(r"https?://|www\.").astype(int)


# 6. colour: repair the hex codes, split into red, green and blue (0-1)
for col in ["link_color", "sidebar_color"]:
    prefix = col.split("_")[0]
    df[col] = df[col].map(fix_colour)
    print(col, "unrecoverable:", df[col].isna().sum())
    for i, channel in enumerate("rgb"):
        value = df[col].str[2 * i:2 * i + 2].map(lambda h: int(h, 16) / 255, na_action="ignore")
        df[prefix + "_" + channel] = value.fillna(value.median())
df["link_default"] = (df["link_color"] == "0084B4").astype(int)
df["sidebar_default"] = (df["sidebar_color"] == "C0DEED").astype(int)


# 7. dates: account age at the time the sample was taken, and activity rates
sampled_at = pd.to_datetime(df["tweet_created"], format=DATE_FORMAT).max()
df["account_age_days"] = (sampled_at - pd.to_datetime(df["created"], format=DATE_FORMAT)).dt.days
df["tweets_per_day"] = df["tweet_count"] / df["account_age_days"].clip(lower=1)
df["favs_per_day"] = df["fav_number"] / df["account_age_days"].clip(lower=1)


# 8. other profile fields as flags
df["default_image"] = df["profileimage"].str.contains("default_profile_images").astype(int)
df["has_retweets"] = (df["retweet_count"] > 0).astype(int)
df["has_coord"] = df["tweet_coord"].notna().astype(int)
df["location_missing"] = df["tweet_location"].isna().astype(int)
df["timezone_missing"] = df["user_timezone"].isna().astype(int)
df["name_n_digits"] = df["name"].str.count(r"\d")


# 9. categorical: the ten most common time zones, the rest as Other, one-hot encoded
top_zones = df["user_timezone"].value_counts().index[:10]
zone = df["user_timezone"].where(df["user_timezone"].isin(top_zones), "Other")
zone[df["user_timezone"].isna()] = "Missing"
zones = pd.get_dummies(zone, prefix="tz", dtype=int).drop(columns="tz_Missing")
df = pd.concat([df, zones], axis=1)

# 10. assemble the feature matrix: numeric features to be scaled, binary features as they are
LOG_COLS = ["fav_number", "tweet_count", "tweets_per_day", "favs_per_day"]
RAW_COLS = ["account_age_days", "text_len", "desc_len", "text_n_urls",
            "text_n_mentions", "text_n_hashtags", "name_n_digits",
            "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
NUM_COLS = LOG_COLS + RAW_COLS
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]
BIN_COLS = FLAG_COLS + list(zones.columns)
# names of the numeric columns once the pipeline has logged the skewed counts
SCALED_COLS = [c + "_log" for c in LOG_COLS] + RAW_COLS


# 11. keep the identifiers, label, text and the features
ID_COLS = ["_unit_id", TARGET, LABEL, "name", "gender", "gender:confidence", "label_conflict"]
TEXT_COLS = ["description", "text", "desc_clean", "text_clean", "link_color", "sidebar_color"]
df = df[ID_COLS + TEXT_COLS + NUM_COLS + BIN_COLS]
print("clean:", df.shape)


# 12. split the labelled records into training, validation and test sets.
labelled = df[df[TARGET].notna()]
trn, rest = train_test_split(labelled, test_size=.4, random_state=SEED,
                             stratify=labelled[TARGET])
val, tst = train_test_split(rest, test_size=.5, random_state=SEED,
                            stratify=rest[TARGET])
print("train:", trn.shape, "validation:", val.shape, "test:", tst.shape)
print("class balance:\n", pd.DataFrame({"train": trn["is_human"].value_counts(normalize=True),
                                        "validation": val["is_human"].value_counts(normalize=True),
                                        "test": tst["is_human"].value_counts(normalize=True)}).round(3))


# 13. the numeric transform: log1p the skewed counts, then standardise.
# The pipeline is fitted on the training split only
prep = Pipeline([
    ("log", ColumnTransformer([("log1p", FunctionTransformer(np.log1p), LOG_COLS)],
                              remainder="passthrough")),
    ("scale", StandardScaler()),
])
prep.fit(trn[NUM_COLS])


def prepare(part, keep_text=False):
    scaled = pd.DataFrame(prep.transform(part[NUM_COLS]), columns=SCALED_COLS, index=part.index)
    keep = ID_COLS + TEXT_COLS + LOG_COLS if keep_text else ID_COLS
    return pd.concat([part[keep], scaled, part[BIN_COLS]], axis=1)


# 14. write the four files
full = prepare(df, keep_text=True)
full.to_csv(PROC / "twitter_full.csv", index=False)
for name, part in [("train", trn), ("validation", val), ("test", tst)]:
    out = prepare(part)
    out.to_csv(PROC / ("twitter_" + name + ".csv"), index=False)
    print(name, out.shape)
print("full:", full.shape)
print("features:", SCALED_COLS + BIN_COLS)
print("target:", TARGET, "- 1 human, 0 non-human, blank unknown", repr(LABEL))


# 15. exploratory analysis of the cleaned data: class balance and feature summary
print("is_human:\n", df["is_human"].value_counts(dropna=False))
print(df[NUM_COLS].describe().round(2).T)
print("skew before and after the log transform:")
print(pd.DataFrame({"raw": df[LOG_COLS].skew(), "log1p": np.log1p(df[LOG_COLS]).skew()}).round(2))

df["is_human"].map({1: "human", 0: "non-human"}).fillna("unknown").value_counts().plot.bar()
plt.ylabel("records")
plt.savefig(OUT / "fig_preprocess_class_balance.png", dpi=120, bbox_inches="tight")
plt.show()


# 16. correlation between the numeric features, to find redundant pairs
corr = full[SCALED_COLS].corr()
print(corr.round(2))

fig, ax = plt.subplots(figsize=(9, 8))
image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(SCALED_COLS)), SCALED_COLS, rotation=90)
ax.set_yticks(range(len(SCALED_COLS)), SCALED_COLS)
fig.colorbar(image)
plt.title("correlation between the numeric features")
plt.tight_layout()
plt.savefig(OUT / "fig_preprocess_feature_correlation.png", dpi=120, bbox_inches="tight")
plt.show()

pairs = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print("pairs correlated above 0.7:\n", pairs[pairs > 0.7].sort_values(ascending=False))


# 17. distribution of every numeric feature after the log transform and standardising
fig, axes = plt.subplots(4, 5, figsize=(16, 10))
for ax, col in zip(axes.ravel(), SCALED_COLS):
    ax.hist(full[col], bins=40)
    ax.set_title(col, fontsize=9)
for ax in axes.ravel()[len(SCALED_COLS):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "fig_preprocess_feature_distributions.png", dpi=120, bbox_inches="tight")
plt.show()


# 18. how the features separate human from non-human profiles
print("median by label:\n", labelled.groupby("is_human")[NUM_COLS].median().round(2).T)
print("flag rate by label:\n", labelled.groupby("is_human")[FLAG_COLS].mean().round(3).T)

labelled.boxplot(column=["tweet_count", "fav_number", "account_age_days"],
                 by="is_human", figsize=(11, 4))
plt.yscale("log")
plt.savefig(OUT / "fig_preprocess_counts_by_label.png", dpi=120, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col in zip(axes, ["link_r", "link_g", "link_b"]):
    for value, name in [(1, "human"), (0, "non-human")]:
        ax.hist(labelled.loc[labelled["is_human"] == value, col], bins=30, alpha=0.5, label=name)
    ax.set_title(col)
axes[0].legend()
plt.tight_layout()
plt.savefig(OUT / "fig_preprocess_link_colour_by_label.png", dpi=120, bbox_inches="tight")
plt.show()

# most common words in the cleaned descriptions, by label
words = labelled["desc_clean"].str.findall(r"[a-z]{3,}")
for value, name in [(1, "human"), (0, "non-human")]:
    counts = pd.Series(words[labelled["is_human"] == value].sum())
    print(name, counts[~counts.isin(ENGLISH_STOP_WORDS)].value_counts().head(20).to_dict())
