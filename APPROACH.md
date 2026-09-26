# Assignment 2 — how the pieces fit together

Read this before writing your model. It explains what we are actually looking for, what your
program must output so the pieces join up, and what happens after everyone is done.

---

## 1. What the assignment is really asking

> *"identify profiles that are **mistakenly recorded** as human/non-human profiles"*

The `gender` column was filled in by crowd workers. **Some of it is wrong.** Our job is to produce
a list of which rows are wrong.

**The deliverable is a list of suspect profiles. It is not a model, and it is not an accuracy score.**

This is the part that trips everyone up, so to be explicit:

- We are **not** building a gender predictor.
- We are **not** trying to win on accuracy.
- Accuracy is a *credential* — it tells the reader whether to believe our flags. Nothing more.

### Why a model finds errors in the labels it was trained on

The obvious objection: *we train on the labels, so the model just learns the labels — how can it
find errors in them?*

Wrong labels are a **minority**, maybe 5–10%. The other 90–95% agree with each other, so the model
learns *their* pattern. The mislabelled rows are exactly the ones that pattern gets wrong.

> 100 photos labelled "cat", 5 are secretly dogs. Train on all 100 — the model still learns *cat*,
> because 95 of them are cats. Show it the 5 dogs and it says "not a cat", **disagreeing with the
> label it was trained on.** Those 5 disagreements are the error list.

A disagreement is a **nomination, not a verdict**. It becomes evidence only when something
independent agrees with it — which is why we run five methods instead of one.

---

## 2. The shape of the whole project

```
01_eda.py                  diagnose the raw data
02_preprocess.py           clean, engineer features, split  -> data/processed/*.csv
                                       │
     ┌──────────────┬──────────────┬───┴──────────┬──────────────────┬───────────┐
     │              │              │              │                  │           │
03_association  04_clustering  05_classification  06_linear_regression  07_logistic  08_text
  (Nelly)        (Binh)          (Nancy)          (Sun, negative result)  _regression  (Nelly)
     │              │              │                                     (Sun)        │
     └──────────────┴──────────────┴──────────────────┬──────────────────┴───────────┘
                                                      │
                                              09_consensus.py (Andrew)
                                                      │
                                     ranked list of suspect profiles = the Task 4 answer
```

**Every method is an independent voter.** They reach their answer by different mechanisms, which is
the whole point: a profile flagged by one method is a guess, a profile flagged by three is a finding.

We already have evidence this works. Clustering flagged 279 profiles, association rules flagged 827,
and **159 appear on both lists** from completely unrelated evidence. Where the two agree, 68% of
the profiles are ones the crowd was also unsure about — against 59% (rules) and 57% (clustering)
alone, and 26% of all labelled profiles. With all five methods voting (2026-09-26 rerun): 1,295
candidates, 513 nominated by two or more methods, and crowd uncertainty climbs with every extra
vote — 45% at one method, 64% at two, 79% at three, 84% at four, 100% for the 14 flagged by all five.

---

## 3. Ground rules — everyone follows these

### The target variable is `is_human`

Not `gender`. Not `gender:confidence`.

```python
df["is_human"] = df["gender"].map({"male": 1, "female": 1, "brand": 0})   # unknown -> NaN
```

| column | role |
|---|---|
| `gender` | the raw crowd label (male / female / brand / unknown) — the **source** |
| **`is_human`** | **the target.** 1 = human, 0 = brand, NaN = unknown |
| `gender:confidence` | **never a target.** How much the annotators agreed. We use it to *validate* flags |

Male vs female is a different problem (that was the original CrowdFlower project). A person recorded
as the wrong sex is not "misinformation" in the sense being asked about — a *company account
recorded as a person* is.

The 1,037 `unknown` rows have no label, so they sit out of training. At the end we predict them as
**suggestions**, which is a separate, smaller deliverable.

### Use the files 02 already produced

| file | rows | use it for |
|---|---|---|
| `twitter_full.csv` | 18,715 | unsupervised methods (clustering, rules, text) — **use every row** |
| `twitter_train.csv` | 10,606 | supervised training |
| `twitter_validation.csv` | 3,536 | choosing between models / tuning |
| `twitter_test.csv` | 3,536 | **final number only — touch once, at the end** |

Numeric features are already log-transformed and standardised; binary flags are left at 0/1. The
scaler was fitted on the training split only, so there is no leakage. `SEED = 7` everywhere.

### If you use cross-validation, you need both calls

```python
cross_val_score(clf, X, y, cv=cv)        # -> 5 numbers.       The CREDENTIAL
cross_val_predict(clf, X, y, cv=cv,      # -> 17,678 rows.     The PRODUCT
                  method="predict_proba")
```

`score` tells the reader whether to trust us. `predict` gives the per-row verdicts we actually spend.
The labs only ever show you `cross_val_score`, which is why this is easy to miss.

**Hard rule: never judge a row with a model that trained on it.** For that row the model may have
memorised the wrong label. `cross_val_predict` handles this automatically.

### Code style

One-line comments. The analysis and the justification go in the report, not in the source. Follow the
lab scripts' layout — numbered sections, top to bottom, `plt.show()` per figure.

---

## 4. The output contract — this is the important bit

**Every voting method writes two CSVs with the same first columns.** `09_consensus.py` reads the
flagged file for labelled profiles and only the unknown rows of the predictions file.

`data/output/<method>_predictions.csv` — every profile the method has an opinion about, labelled or unknown.

| column | type | meaning |
|---|---|---|
| `_unit_id` | int | the profile — the join key |
| `says` | str | `"human"` or `"non_human"` — what your method thinks it should be |
| `score` | float 0–1 | your method's confidence in that call |
| `recorded` | str | `"human"`, `"non_human"` or `"unknown"` — the `label` column from `02` |

Supervised methods cover all 18,715 rows: labelled rows out-of-fold (`cross_val_predict`), unknown
rows from a refit on every labelled row. Clustering covers rows in one-sided clusters. Association
rules cover rows matched by at least one strong rule, the strongest rule deciding `says`.

`data/output/<method>_flagged.csv` — the labelled rows of the predictions file where `says` differs
from `recorded` and `score` clears the method's threshold. Same four columns plus `gender`, `name`,
`gender:confidence`. A flagged row is always also a prediction row with the same `says` and `score`.

```csv
_unit_id,says,score,recorded
815719226,non_human,0.93,human
815719411,human,0.88,unknown
```

Add whatever extra columns are useful to you (cluster id, the rule that fired, the model's raw
probability) — they are ignored by the consensus step. The four above are the only ones that must
be there, and must be spelled exactly like that.

When the report needs a reason for a particular candidate, read it off your method's own output —
the rules table, the cluster summary, the feature importances. It does not belong in this file.

### What `score` means, per method

`score` = **how strongly your method believes `says`**, on 0–1. Roughly, 0.5 = no opinion,
1.0 = certain. Every method already computes this; you just have to pull it out.

| method | `score` is | how to get it |
|---|---|---|
| **classification** | model confidence in the predicted class | `np.maximum(prob, 1 - prob)` where `prob = cross_val_predict(..., method="predict_proba")[:, 1]` |
| **regression (logistic)** | same as above | same as above |
| **text** | same as above | same as above — it is a classifier |
| **clustering** | purity of the cluster the profile sits in | cluster human rate if `says == "human"`, else `1 - human_rate` |
| **association rules** | confidence of the rule that fired | `r["confidence"]` — literally P(class \| antecedent) |

```python
# classification / regression / text
prob  = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]   # P(human)
says  = np.where(prob >= 0.5, "human", "non_human")
score = np.maximum(prob, 1 - prob)          # P(human)=0.04 -> says non_human, score 0.96

# clustering
score = np.where(says == "human", cluster_human_rate, 1 - cluster_human_rate)
```

**These numbers are not comparable across methods.** A rule confidence of 0.925 and a logistic
probability of 0.925 do not mean the same thing — they come from different mechanisms with
different calibration. So in `09_consensus.py`:

> **Count votes first. Use `score` only to rank within a method, and as a tie-breaker.**

Do not average scores across methods and call it a combined confidence.

### Do not use one threshold for every model

Measured on our data, 5-fold CV, flagging at `score >= 0.9`:

| model | rows scoring ≥ 0.9 | distinct score values | profiles flagged |
|---|---|---|---|
| logistic | 36.7% | 4,551 | 336 |
| decision tree | 41.5% | 357 | 511 |
| knn (k=25) | 38.7% | **13** | 381 |
| **naive bayes** | **78.0%** | 3,381 | **3,126** |

**Naive Bayes flags 9× more profiles than logistic at the same threshold.** Its probabilities pile
up near 0 and 1 because of the independence assumption — it is not 9× better, it is overconfident.
Left uncorrected it would dominate the consensus with noise.

**KNN has only 13 distinct score values**, because `predict_proba` is just the fraction of
neighbours voting — with `k=25` nothing finer is possible.

So: pick the threshold **per model**, not globally. The practical rule is to choose each model's
cut-off so it nominates a comparable number of profiles (a few hundred), and say in the report
which cut-off you used for each and why.

### One gotcha that will cost you an hour

`KNeighborsClassifier` crashes on a pandas DataFrame in this environment
(`AttributeError: 'Flags' object has no attribute 'c_contiguous'` — sklearn checks
`X.flags.c_contiguous` and pandas has its own unrelated `.flags`). Pass arrays:

```python
X = lab[FEATURES].to_numpy(float)
```

---

## 5. What each person does

Everyone's section has the same shape: **fit the models, prove they are worth believing, then spend
that credibility on a flag list.** What differs is the mechanism.

### Nelly — Association rules (`03_association_rules.py`) ✅ done

**Lab 6.** Apriori via `mlxtend`.

A profile is not a shopping basket, so items have to be built: the binary profile flags, negations
where absence is informative, top/bottom quartile of each numeric feature, and the label itself as
an item so rules can conclude with it.

- **Main output is the rules**, not the flag list. Apriori is *descriptive, not predictive* — the
  Week 6 slide says so outright. A rule reads as a sentence a human can check by eye, which is what
  Task 4 needs.
- Flag rule: profile matches a high-confidence rule for the class opposite to its label.
- Thresholds chosen from a printed sweep, not by hand.

**Honest finding to report:** as a classifier it is the weakest of the three tested (0.7745 vs
0.7953 decision tree vs 0.8052 logistic), because it is unsupervised and does not optimise for the
label. That is a *justified method selection* result, not a failure — say it plainly.

### Binh — Clustering (`04_clustering.py`) ✅ done

**Lab 3.** k-means and hierarchical (four linkages), plus DBSCAN as a robustness check.

- Cluster on behaviour features only. Colour was tested and dropped: it gives the tightest clusters
  (silhouette 0.588) but carries almost no label information (`human_rate_spread` 0.057).
- Flag rule: profile sits in a one-sided cluster but carries the opposite label.

### Nancy — Classification (`05_classification.py`) ✅ done

**Lab 4.** Decision tree, KNN, naive Bayes, MLP — plus logistic regression, so five algorithms in
one loop.

```python
y  = labelled["is_human"].astype(int)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

for name, clf in [("logistic",      LogisticRegression(max_iter=2000, class_weight="balanced")),
                  ("decision tree", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
                  ("naive bayes",   GaussianNB()),
                  ("knn",           KNeighborsClassifier(n_neighbors=25)),
                  ("mlp",           MLPClassifier(random_state=SEED))]:
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")            # credential
    prob   = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:,1] # product
```

- Report accuracy **and** a confusion matrix per algorithm. The models are much worse on brands than
  on humans — that asymmetry matters and should be visible. Try `class_weight="balanced"`.
- Use `ttest_ind` on two score arrays to say whether a difference between algorithms is *real*,
  exactly as Lab 4 does. A 0.004 gap is noise.
- Flag rule: `(prob >= 0.9) & (label == brand)` or `(prob <= 0.1) & (label == human)`.
- **Measured (Task 6 of the script):** training the best model only on the 13,119 rows where
  `gender:confidence == 1.0` and scoring all 17,678 out-of-sample; the script prints both accuracies
  side by side. Cite the printed pair in the report rather than a number copied here.

### Sun — Regression (`06_linear_regression.py`, `07_logistic_regression.py`) ✅ done

**Lab 5**, which is *two* tasks: linear regression **and** logistic regression.

**Logistic regression on `is_human` is the regression that fits this assignment.** The outcome is
categorical, so the Week 6 slide is explicit: *"Logistic regression — a better choice if the outcome
variable is categorical."*

The current file predicts `gender:confidence` with Ridge and a random forest and gets **R² = 0.05**.
That target is 74% the single value 1.0, so it is not a continuous quantity and linear regression is
misspecified from the start. Keep that experiment — a documented negative result earns marks — but:

- `06_linear_regression.py` keeps the Ridge / random-forest experiment on `gender:confidence` as a
  documented negative result and prints `corr(|residual|, gender:confidence)` to show why residuals
  are not a mislabel list. It does not vote.
- `07_logistic_regression.py` is the regression that fits a categorical outcome: fit on train, choose
  full vs RFE on validation, test once, then out-of-fold verdicts for every labelled row with the
  chosen configuration. It casts the **regression** vote; classification's vote goes to the best
  Lab 4 model, so logistic regression never votes twice.

### Nelly — Text (`08_text.py`) ✅ done

**Lab 7.** nltk tokenising and stop words, conditional frequency distribution, TF-IDF, gensim LDA.

Uses **text features only** — no structured features — so it stays an independent voter for the
consensus. `desc_clean` and `text_clean` were already lower-cased and stripped in 02.

- Text alone reaches **0.8187**, higher than all 38 structured features together (0.8052).
  Description alone 0.7069, tweet alone 0.7376 — the two fields are complementary.
- Nominates 357 profiles at `score >= 0.85`, 64.1% of which the crowd was also unsure about.
- The clearest linguistic finding for the report: **first-person pronouns separate the classes.**
  `my`, `me`, `I` are the strongest human weights; `we`, `us`, `our` the strongest non-human.
  People speak as individuals, organisations speak as groups.
- Distinctive words — non-human: `continuous, price, updates, subscribe, official, latest`;
  human: `husband, father, dad, graduate, actor, writer, snapchat`.
- LDA over the descriptions gives 8 readable topics, and their human rate ranges from 0.492
  (`news, social, media, free`) to 0.899 (`fan, writer, lover, love, girl`).

---

## 6. What happens after everyone is done

### Step 1 — `09_consensus.py`

Concatenate the five `*_flagged.csv` files on `_unit_id` and pivot:

```
_unit_id   name        gender  crowd_conf  rules  clustering  classification  regression  text  votes
815719226  Sunglare    brand      0.672      1        1             1              1        0      4
...
```

Rank by `votes` descending, then `crowd_conf` ascending. **That table is the Task 4 answer.**

### Step 2 — validate the consensus

Check the flags against evidence no model used. The current numbers, for reference:

| | nominated by 2+ methods (513) | all candidates (1,295) | all labelled |
|---|---|---|---|
| below full crowd confidence | 72.5% | 56.0% | 25.8% |
| `label_conflict` | 1.0% | 1.2% | 0.4% |

Caution: `label_conflict` has a base rate of 0.4%, so on a subset of 250 profiles you would expect
**one**. It is too rare to validate anything on small groups — `gender:confidence` is the only
corroborating signal dense enough to be useful.

### Step 3 — the unknown profiles

The 1,037 rows the crowd could not label still get a prediction from every method, read from each
method's `*_predictions.csv`. Where the methods agree, that is a suggested label. Smaller deliverable, but it is the second half of Task 4
("amend non-human and human profiles").

### Step 4 — the report

| Task | marks | what covers it |
|---|---|---|
| 1 — lifecycle design | 3 | `01_eda` findings → `02_preprocess` decisions, mapped to the six phases |
| 2 — process data, apply models | 10 | one section per method: what it is, why chosen, how it performed, what it flagged |
| 3 — visualise, and use visualisation to evaluate | 5 | every method's figures, **plus** the consensus and validation charts |
| 4 — multiple views, suggest amendments | 2 | the view comparison table below + the ranked consensus list |

**Phrase the conclusions as recommendations, never verdicts.** "These 336 profiles are recorded as
brands but behave like people, and the annotators were unsure about three-quarters of them — we
recommend they be re-examined." Not "this profile *is* a person." An 0.86 model has no authority to
overrule a human label outright.

---

## 7. Numbers already established — don't redo this work

All measured on `twitter_full.csv`, 5-fold stratified CV, `SEED = 7`.

**Dataset:** 18,715 profiles · 17,678 labelled (12,266 human / 5,412 brand) · 1,037 unknown.
Baseline "always guess human" = **0.6939**.

**Accuracy by method** (38 structured features, plain `LogisticRegression` unless noted):

| method | accuracy |
|---|---|
| logistic regression | 0.8052 |
| decision tree | 0.7953 |
| association rules as a classifier | 0.7745 |
| **text only** (TF-IDF, no structured features) | **0.8187** |
| everything + TF-IDF of both text fields | **0.8377** (`class_weight="balanced"`) / 0.8560 (plain) |

**Accuracy by view — this is the Task 4 "multiple modes" answer:**

| view | alone |
|---|---|
| activity (tweets, favourites, account age) | 0.7544 |
| profile flags (picture, bio, location) | 0.7475 |
| text content (TF-IDF, both fields) | 0.8187 |
| **colour** | **0.5516 — essentially uninformative** |
| all combined | **0.8377** |

Colour being near-useless is a *finding*, not a failure — the assignment names it as a view to study,
and two methods reached that conclusion independently.

**The accuracy ceiling** — why ~0.85 is not underwhelming:

| crowd confidence | n | model accuracy |
|---|---|---|
| 1.00 (annotators unanimous) | 13,119 | **0.8991** |
| 0.60 – 0.74 | 4,030 | 0.7404 |
| below 0.60 | 529 | 0.6692 |

Where the annotators agreed with each other, the model agrees with them 90% of the time. Where they
argued, it drops to 67%. **The model is hitting the same wall humans hit** — you cannot learn a
pattern from labels that were close to a coin flip. Roughly 0.90 is the ceiling these labels permit.

This is our strongest single result: the errors are concentrated exactly where the labels are least
reliable, which is independent support for the whole premise.

**Feature engineering:** dropping features does not help. Removing the timezone dummies and colour
changes accuracy by ±0.004. The problem was never too many features — it was the missing text.

---

## 8. Open questions for the team

1. **Resolved — classification stays structured-only** (`05`), text-only is `08`; the combined
   structured + TF-IDF number is reported separately as the best single model.
2. **Resolved — file names** are one script per lab, numbered in pipeline order (§2).
3. **Still open — one model configuration.** `05`'s logistic row uses `class_weight="balanced"`;
   `07` uses the plain model, as Lab 5 does. Whoever writes the regression section cites `07`'s
   numbers; the classification comparison table cites `05`'s and says the settings differ.
4. **Resolved — `09_consensus.py` is Andrew's** (`CONSENSUS_SPEC.md`).
