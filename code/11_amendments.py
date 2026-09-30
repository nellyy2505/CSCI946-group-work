# 11_amendments.py — word groups, patterns and suggested amendments for the consensus candidates (report 10.6)

import re
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"


# 1. Load
# every profile with its raw description and tweet, and the three files written by 09
df = pd.read_csv(PROC / "twitter_full.csv")
for c in ["description", "text"]:
    df[c] = df[c].fillna("")
cols = ["_unit_id", "description", "text", "default_image", "desc_missing", "desc_has_url", "text_has_emoji",
        "has_retweets", "text_n_urls", "fav_number", "tweets_per_day"]
cand = pd.read_csv(OUT / "consensus_candidates.csv").merge(df[cols], on="_unit_id", how="left", validate="one_to_one")
unk = pd.read_csv(OUT / "consensus_unknown.csv").merge(df[cols], on="_unit_id", how="left", validate="one_to_one")
ties = pd.read_csv(OUT / "consensus_ties.csv")
lab = df[df["label"] != "unknown"].copy()
print("candidates:", len(cand), "| unknown with a suggestion:", len(unk), "| labelled profiles:", len(lab))


# 2. Word groups
# a word matches when a token of the lower-cased raw description (or tweet) equals a listed word
TOKEN = re.compile(r"[a-z']+")      # a token is a run of letters a-z and apostrophes; anything else splits
# first-person: I, I'm, me, my, mine, myself, plus the forms im, i've, i'll and i'd
FIRST = {"i", "i'm", "im", "me", "my", "mine", "myself", "i've", "i'll", "i'd"}
ORG = {"we", "our", "us", "official", "news", "updates", "subscribe", "shop", "store", "company", "customer",
       "services", "station", "channel", "magazine", "radio"}
PRONOUN = re.compile(r"\b(?:she/her|he/him|they/them|she/they|he/they)\b", re.I)
print("\nmatch rule: lower-case the field, turn the curly apostrophe into ', split into runs of [a-z'];")
print("  a profile has the word when one run equals it exactly (so we're, #we're and 'my' do not match we or my,")
print("  #news and @us do match). First-person and organisation words are looked for in `description`;")
print("  first-person words also in `text` (the tweet). Pronoun tag: she/her, he/him, they/them, she/they or")
print("  he/they anywhere in `description`, any case. Fields are the raw columns of twitter_full.csv.")
print("first-person words:", sorted(FIRST))
print("organisation words:", sorted(ORG))


def tokens(s):
    return set(TOKEN.findall(s.lower().replace("’", "'")))


def add_flags(f):
    f = f.copy()
    f["fp_desc"] = f["description"].map(lambda s: bool(tokens(s) & FIRST))
    f["fp_tweet"] = f["text"].map(lambda s: bool(tokens(s) & FIRST))
    f["org_desc"] = f["description"].map(lambda s: bool(tokens(s) & ORG))
    f["pronoun"] = f["description"].str.contains(PRONOUN)
    f["weather"] = f["text"].str.contains("Weather Channel", case=False)
    f["forevermore"] = f["text"].str.contains("Forevermore", case=False)
    f["tweet_link"] = f["text_n_urls"] > 0
    # organisation word and no first-person word anywhere; first-person word or pronoun tag and no organisation word
    f["org_only"] = f["org_desc"] & ~(f["fp_desc"] | f["fp_tweet"])
    f["person_words"] = (f["fp_desc"] | f["fp_tweet"] | f["pronoun"]) & ~f["org_desc"]
    return f


cand, unk, lab = add_flags(cand), add_flags(unk), add_flags(lab)


# 3. Recorded non-human, suggested human (Table 10.8)
# the 839 by number of agreeing methods and word group
to_h = cand[cand["recorded"] == "non_human"]
brands = lab[lab["label"] == "non_human"]
humans = lab[lab["label"] == "human"]


def tier_row(g, name):
    return {"methods_agreeing": name, "candidates": len(g), "first_person_word": int(g["person_words"].sum()),
            "organisation_word": int(g["org_only"].sum()),
            "neither_or_both": int(len(g) - g["person_words"].sum() - g["org_only"].sum()),
            "emoji_pct": round(100 * (g["text_has_emoji"] == 1).mean(), 1),
            "link_in_tweet_pct": round(100 * g["tweet_link"].mean(), 1), "median_favourites": g["fav_number"].median()}


rows = [tier_row(to_h[to_h["votes"] == v], str(v)) for v in [5, 4, 3, 2, 1]]
rows += [tier_row(to_h, "all"), tier_row(brands, "all recorded non-human")]
t108 = pd.DataFrame(rows)
t108.to_csv(OUT / "amendments_non_human_to_human.csv", index=False)
print("\nrecorded non-human -> human: %d | recorded human -> non-human: %d"
      % (len(to_h), (cand["recorded"] == "human").sum()))
print(t108.to_string(index=False))
print("uploaded image %.1f%% (all recorded non-human %.1f%%) | median favourites %.0f (non-human %.0f, human %.0f)"
      % (100 * (to_h["default_image"] == 0).mean(), 100 * (brands["default_image"] == 0).mean(),
         to_h["fav_number"].median(), brands["fav_number"].median(), humans["fav_number"].median()))
print("first-person word in description or tweet %.1f%% (non-human %.1f%%, human %.1f%%)"
      % tuple(100 * (g["fp_desc"] | g["fp_tweet"]).mean() for g in [to_h, brands, humans]))


# 4. Recorded human, suggested non-human (Table 10.9)
# five patterns in the tweet or description; each candidate falls in the first that matches
to_nh = cand[cand["recorded"] == "human"].copy()
NEWS = r"\bnews\b|feed|auto|updates|official"
APP = r"Transponder Snail|#GameInsight|#TreCru|I have completed|I played the|Enter to win|\bWIN\b|giveaway|#AvonRep|@AppBounty"
news = to_nh["description"].str.contains(NEWS, case=False) | to_nh["text"].str.contains("#news", case=False)
app = to_nh["text"].str.contains(APP, case=False)
to_nh["pattern"] = "other"
for name, m in [("forevermore", to_nh["forevermore"]), ("app-written", app), ("news/feed", news),
                ("weather channel", to_nh["weather"])]:
    to_nh.loc[m, "pattern"] = name                # later lines win; the four patterns do not overlap here
print("\npatterns overlap on %d candidates" % ((to_nh["weather"].astype(int) + news + app + to_nh["forevermore"]) > 1).sum())


# 5. Suggested amendments
# recorded non-human: 2+ methods unless organisation-only, 1 method only with person words; Foothill_Dance by hand
MANUAL_KEEP = {"Foothill_Dance": "school dance programme; 'Dance I' matched as a first-person word"}
amend_h = ((to_h["votes"] >= 2) & ~to_h["org_only"]) | ((to_h["votes"] == 1) & to_h["person_words"])
amend_h_word = amend_h.copy()
amend_h &= ~to_h["name"].isin(MANUAL_KEEP)
# recorded human: default image and no first-person description, plus feeds and spam accounts read by hand
FEEDS = ["khalidrafiq141", "Eddki885Mohamm", "dcgblog", "BenoistDaily", "Bamford_ID", "IPOSniffer", "iPeytonManning",
         "SachaIDK", "ShannaIDK", "YvetteIDK"]
SPAM = ["AdamCoo64216897", "PamelaW76653339", "ThomasH29137281", "Charlot02798732", "SalesForceReal"]
empty = (to_nh["default_image"] == 1) & ~to_nh["fp_desc"]
amend_nh = empty | to_nh["name"].isin(FEEDS) | to_nh["name"].isin(SPAM)


def reason_h(r):
    if r["name"] in MANUAL_KEEP:
        return "keep non-human", MANUAL_KEEP[r["name"]]
    if r["org_only"]:
        return "keep non-human", "organisation word in the description and no first-person word"
    if r["votes"] >= 2:
        words = "first-person word or pronoun tag" if r["person_words"] else "no organisation-only description"
        return "amend to human", "%d methods agree; %s" % (r["votes"], words)
    if r["person_words"]:
        return "amend to human", "1 method; first-person word or pronoun tag and no organisation word"
    return "keep non-human", "1 method; no first-person word or pronoun tag free of organisation words"


def reason_nh(r):
    if r["default_image"] == 1 and not r["fp_desc"]:
        return "amend to non-human", "default image and no first-person word in the description"
    if r["name"] in FEEDS:
        return "amend to non-human", "news or content feed (read by hand)"
    if r["name"] in SPAM:
        return "amend to non-human", "spam or company account (read by hand)"
    if r["fp_desc"] and not r["org_desc"]:
        return "keep human", "first-person word in the description and no organisation word"
    if r["forevermore"]:
        return "keep human", "Forevermore campaign tweet only"
    return "keep human", "uploaded image or first-person description, not a feed or spam account"


sug_h = to_h.assign(pattern="").join(pd.DataFrame(to_h.apply(reason_h, axis=1).tolist(), index=to_h.index,
                                                  columns=["action", "reason"]))
sug_nh = to_nh.join(pd.DataFrame(to_nh.apply(reason_nh, axis=1).tolist(), index=to_nh.index,
                                 columns=["action", "reason"]))
suggested = pd.concat([sug_h, sug_nh]).loc[cand.index]          # back in the ranked order of 09
keep = ["_unit_id", "name", "recorded", "says", "votes", "methods", "fp_desc", "fp_tweet", "pronoun", "org_desc",
        "pattern", "action", "reason"]
suggested[keep].to_csv(OUT / "amendments_suggested.csv", index=False)

t109 = pd.DataFrame([{"pattern": p, "candidates": int((to_nh["pattern"] == p).sum()),
                      "two_or_more_methods": int(((to_nh["pattern"] == p) & (to_nh["votes"] >= 2)).sum()),
                      "suggested_non_human": int(((to_nh["pattern"] == p) & amend_nh).sum())}
                     for p in ["weather channel", "news/feed", "app-written", "forevermore", "other"]])
t109.loc[len(t109)] = ["all", len(to_nh), int((to_nh["votes"] >= 2).sum()), int(amend_nh.sum())]
t109.to_csv(OUT / "amendments_human_to_non_human.csv", index=False)
print(t109.to_string(index=False))

two = (to_h["votes"] >= 2) & amend_h
print("\nnon-human -> human: word rule %d; without Foothill_Dance %d amend, %d stay non-human"
      % (amend_h_word.sum(), amend_h.sum(), len(to_h) - amend_h.sum()))
print("  2+ methods amended %d of %d (%d with person words, %d neither) | 1 method amended %d of %d"
      % (two.sum(), (to_h["votes"] >= 2).sum(), (two & to_h["person_words"]).sum(), (two & ~to_h["person_words"]).sum(),
         (amend_h & (to_h["votes"] == 1)).sum(), (to_h["votes"] == 1).sum()))
print("  organisation-only kept: %d (%d with 2+ methods)" % (to_h["org_only"].sum(), (to_h["org_only"] & (to_h["votes"] >= 2)).sum()))
print("human -> non-human: %d amend, %d stay human | default image rule %d (2+ %d, 1 method %d) | feeds outside it %d"
      " | spam %d" % (amend_nh.sum(), len(to_nh) - amend_nh.sum(), empty.sum(), (empty & (to_nh["votes"] >= 2)).sum(),
                      (empty & (to_nh["votes"] == 1)).sum(), (to_nh["name"].isin(FEEDS) & ~empty).sum(),
                      to_nh["name"].isin(SPAM).sum()))


# 6. Unknown profiles
# suggestions by agreement; ties and profiles with no vote stay unknown
n_unknown = int((df["label"] == "unknown").sum())
unk_ties = ties.loc[ties["_unit_id"].isin(df.loc[df["label"] == "unknown", "_unit_id"]), "_unit_id"].nunique()
h, nh = unk[unk["says"] == "human"], unk[unk["says"] == "non_human"]
print("\nunknown: %d of %d get a suggestion (%d human, %d non-human) | 3+ methods %d | 2 methods %d | 1 method %d"
      " | ties %d | no vote %d" % (len(unk), n_unknown, len(h), len(nh), (unk["votes"] >= 3).sum(),
                                   (unk["votes"] == 2).sum(), (unk["votes"] == 1).sum(), unk_ties,
                                   n_unknown - len(unk) - unk_ties))
nh3, h3 = nh[nh["votes"] >= 3], h[h["votes"] >= 3]
print("  3+ non-human %d, Weather Channel %d (default image %d) | 3+ human %d, uploaded image %d, median favourites %.0f,"
      " first-person %.1f%%" % (len(nh3), nh3["weather"].sum(), (nh3["weather"] & (nh3["default_image"] == 1)).sum(),
                                len(h3), (h3["default_image"] == 0).sum(), h3["fav_number"].median(),
                                100 * (h3["fp_desc"] | h3["fp_tweet"]).mean()))
doubt = nh3[nh3["forevermore"] | (nh3["text_says"] == "human")]
print("  3+ non-human that stay unknown (Forevermore or text voted human):", ", ".join(doubt["name"]))
h2, nh2 = h[h["votes"] == 2], nh[nh["votes"] == 2]
print("  2 methods: human %d (uploaded image %d, median favourites %.0f, first-person %.1f%%) | non-human %d,"
      " default image and no description %d" % (len(h2), (h2["default_image"] == 0).sum(), h2["fav_number"].median(),
                                                100 * (h2["fp_desc"] | h2["fp_tweet"]).mean(), len(nh2),
                                                ((nh2["default_image"] == 1) & (nh2["desc_missing"] == 1)).sum()))


# 7. Signals (Table 10.10)
# share non-human among the labelled profiles that show each signal
fav_q75 = df["fav_number"].quantile(0.75)
signals = [("first-person word in the description", lab["fp_desc"]),
           ("first-person word in the tweet", lab["fp_tweet"]),
           ("emoji in the tweet", lab["text_has_emoji"] == 1),
           ("favourites in the top quarter (%.0f or more)" % fav_q75, lab["fav_number"] >= fav_q75),
           ("organisation word in the description", lab["org_desc"]),
           ("official in the description", lab["description"].map(lambda s: "official" in tokens(s))),
           ("updates in the description", lab["description"].map(lambda s: "updates" in tokens(s))),
           ("Weather Channel in the tweet", lab["weather"]),
           ("default image", lab["default_image"] == 1),
           ("default image, no description, no favourites",
            (lab["default_image"] == 1) & (lab["desc_missing"] == 1) & (lab["fav_number"] == 0)),
           ("no favourites", lab["fav_number"] == 0),
           ("no description", lab["desc_missing"] == 1),
           ("link in the description", lab["desc_has_url"] == 1),
           ("retweet", lab["has_retweets"] == 1),
           ("all labelled profiles", pd.Series(True, index=lab.index))]
t110 = pd.DataFrame([{"signal": name, "labelled_profiles": int(m.sum()),
                      "non_human_pct": round(100 * (lab.loc[m, "label"] == "non_human").mean(), 1)}
                     for name, m in signals])
t110.to_csv(OUT / "amendments_signals.csv", index=False)
print("\n" + t110.to_string(index=False))
