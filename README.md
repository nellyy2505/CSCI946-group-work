# CSCI446/946 Assignment 2 — misrecorded human / non-human Twitter profiles

Nine Python scripts, one per technique, that together produce a ranked list of profiles whose
crowd label looks wrong. `APPROACH.md` explains the design and the output contract;
`CONSENSUS_SPEC.md` explains the vote.

## Setup

Python 3.8 or later with the packages in `requirements.txt` (tested with Python 3.8.18 and the
versions listed there; a newer Python installs gensim 4.x, which has the same API calls used here):

    pip install -r requirements.txt

`08_text.py` downloads the nltk `stopwords` and `punkt_tab` resources on first run (internet needed once).

## Run order

Run from the repository root, in order. Each script resolves its paths from its own location, so
the working directory does not matter. Scripts show every figure with `plt.show()` and also save it;
to run without a display: `MPLBACKEND=Agg python code/01_eda.py`.

| script | what it does | writes |
|---|---|---|
| `01_eda.py` | inspect the raw file, record its problems | `fig_eda_*` |
| `02_preprocess.py` | clean, engineer features, 60/20/20 split | `data/processed/twitter_{full,train,validation,test}.csv`, `fig_preprocess_*` |
| `03_association_rules.py` | Apriori rules that conclude the label (Lab 6) | `association_{rules,predictions,flagged}.csv`, `fig_association_*` |
| `04_clustering.py` | k-means, hierarchical, DBSCAN on behaviour features (Lab 3) | `clustering_{assignments,predictions,flagged}.csv`, `fig_clustering_*` |
| `05_classification.py` | decision tree, KNN, naive Bayes, MLP, logistic; best Lab 4 model votes (Lab 4) | `classification_*.csv`, `fig_classification_*` |
| `06_linear_regression.py` | Ridge / random forest on `gender:confidence` — a documented negative result (Lab 5) | `regression_linear_model_comparison.csv`, `fig_regression_linear_*` |
| `07_logistic_regression.py` | logistic regression on `is_human` with RFE; the regression vote (Lab 5) | `regression_{logistic_model_comparison,predictions,flagged}.csv`, `fig_regression_logistic_*` |
| `08_text.py` | nltk tokens, TF-IDF + logistic, gensim LDA on the text fields only (Lab 7) | `text_{predictions,flagged}.csv`, `fig_text_*` |
| `09_consensus.py` | one equal vote per method; ranked review list and unknown-profile suggestions | `consensus_*.csv`, `fig_consensus_agreement.png` |

All outputs go to `data/output/`. `SEED = 7` everywhere, so a rerun reproduces every file.

## Submission

`A2.zip` = the report PDF + `code/` + `data/` + this README, `APPROACH.md` and `CONSENSUS_SPEC.md`.
Leave `.git/` out.
