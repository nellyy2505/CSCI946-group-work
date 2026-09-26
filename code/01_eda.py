# CSCI446/946 Big Data Analytics - Assignment 2
# 01 - Initial exploratory analysis: inspect the raw data and record the problems

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "twitter_user_data.csv"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)
NUMERIC = ["fav_number", "retweet_count", "tweet_count"]


# 1. load the data (the file is not UTF-8; Mac Roman reads it without errors)
df = pd.read_csv(DATA, encoding="mac_roman")
print("shape:", df.shape)
print(df.dtypes)
print(df.head())


# 2. missing and unique values per column
summary = pd.DataFrame({"missing": df.isna().sum(),
                        "missing_%": (df.isna().mean() * 100).round(2),
                        "unique": df.nunique()})
print(summary)

summary.loc[summary["missing"] > 0, "missing_%"].sort_values().plot.barh()
plt.xlabel("missing (%)")
plt.savefig(OUT / "fig_eda_missing.png", dpi=120, bbox_inches="tight")
plt.show()


# 3. label: class balance, crowd confidence, and agreement with the gold standard
print(df["gender"].value_counts(dropna=False))
print(pd.crosstab(df["profile_yn"], df["gender"].fillna("missing")))
print(df.groupby("gender")["gender:confidence"].describe())
print("records below full confidence:", (df["gender:confidence"] < 1).sum())
gold = df[df["_golden"]]
print(pd.crosstab(gold["gender"], gold["gender_gold"]))


# 4. numeric columns: scale, skew and correlation
print(df[NUMERIC].describe().round(2))
print("skew:\n", df[NUMERIC].skew().round(2))
print("zeros:\n", (df[NUMERIC] == 0).sum())

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for j, col in enumerate(NUMERIC):
    axes[0, j].hist(df[col], bins=50)
    axes[0, j].set_yscale("log")
    axes[0, j].set_title(col)
    axes[1, j].hist(np.log1p(df[col]), bins=50)
    axes[1, j].set_title("log1p(" + col + ")")
plt.tight_layout()
plt.savefig(OUT / "fig_eda_numeric_distributions.png", dpi=120, bbox_inches="tight")
plt.show()

print(np.log1p(df[NUMERIC]).corr().round(3))


# 5. date columns, and columns that hold a single value
for col in ["created", "tweet_created", "_last_judgment_at"]:
    t = pd.to_datetime(df[col], format="%m/%d/%y %H:%M")
    print(col, "min", t.min(), "max", t.max(), "unique", t.nunique())
print("tweet_id unique values:", df["tweet_id"].unique())


# 6. colour columns: the hex codes were damaged when the file was saved
for col in ["link_color", "sidebar_color"]:
    codes = df[col].astype(str)
    print(col, "stored length:\n", codes.str.len().value_counts().sort_index())
    print(col, "malformed examples:\n", codes[codes.str.len() != 6].value_counts().head(10))


# 7. text columns: encoding damage
for col in ["text", "description"]:
    s = df[col].fillna("")
    print(col, "records with non-ASCII characters:", s.map(lambda x: any(ord(c) > 127 for c in x)).sum())
print(df.loc[df["text"].str.contains("â€"), "text"].head())


# 8. categorical columns, duplicated records and repeated accounts
print(df["user_timezone"].value_counts().head(10))
print(df["tweet_location"].value_counts().head(10))
print("tweet_coord present:", df["tweet_coord"].notna().sum())
print("duplicated rows (excluding _unit_id):", df.drop(columns="_unit_id").duplicated().sum())

copies = df["name"].value_counts()
print("accounts:", len(copies), "appearing more than once:", (copies > 1).sum())
print(df[df["name"].isin(copies[copies > 1].index)].groupby("name")["gender"].agg(
    lambda s: "+".join(sorted(s.dropna().unique()))).value_counts())

