# CSCI446/946 Assignment 2 — misrecorded human / non-human Twitter profiles
## Setup

Python 3.8–3.12. `requirements.txt` gives minimum versions; the versions every reported number
comes from are in its comments:

    pip install -r requirements.txt
    python -m nltk.downloader stopwords punkt_tab

The tested versions need Python 3.8–3.11. Python 3.12 needs newer libraries; the scripts run unchanged,
but a few text and logistic votes move (1,100 to 1,103 candidates instead of 1,103).

The second line is needed once, with internet access. `08_text.py` looks for the two nltk resources
first and downloads them only if they are missing; if that download fails it stops and prints the
line above. To keep the resources somewhere else, set `NLTK_DATA` to that folder. nltk is kept below
3.10 because 3.10 and later refuse downloads through a proxy.

The scripts expect the raw file at `data/raw/twitter_user_data.csv`.

## Run order

Run from the repository root, in order 01 → 11. Each script resolves its paths from its own
location. Scripts show every figure with `plt.show()` and also save it; to run without a display:
`MPLBACKEND=Agg python code/01_eda.py`. The full run takes about 2 minutes.

| script | what it does | writes |
|---|---|---|
| `01_eda.py` | inspect the raw file, record its problems | `fig_eda_*` |
| `02_preprocess.py` | clean, engineer features, stratified 60/20/20 train/validation/test split | `data/processed/twitter_{full,train,validation,test}.csv`, `fig_preprocess_*` |
| `03_association_rules.py` | Apriori rules that conclude the label (Lab 6) | `association_{rules,predictions,flagged}.csv`, `fig_association_*` |
| `04_clustering.py` | k-means, hierarchical, DBSCAN on behaviour features (Lab 3) | `clustering_{assignments,predictions,flagged}.csv`, `fig_clustering_*` |
| `05_classification.py` | decision tree, KNN, naive Bayes, MLP, logistic; the best non-logistic model votes (Lab 4) | `classification_*.csv`, `fig_classification_*` |
| `06_linear_regression.py` | Lab 5 `LinearRegression` on `is_human` (0/1): R², MSE and accuracy at 0.5, showing predictions outside [0, 1] and two-band residuals, so logistic (07) is the right tool; does not vote | `regression_linear_{coefficients,model_comparison}.csv`, `fig_regression_linear_*` |
| `07_logistic_regression.py` | logistic regression on `is_human` with RFE; the regression vote (Lab 5) | `regression_{logistic_model_comparison,logistic_coefficients,predictions,flagged}.csv`, `fig_regression_logistic_*` |
| `08_text.py` | nltk tokens, TF-IDF + logistic regression, gensim LDA on the text fields only (Lab 7) | `text_{predictions,flagged}.csv`, `fig_text_*` |
| `09_consensus.py` | one equal vote per method; ranked review list and unknown-profile suggestions | `consensus_*.csv`, `fig_consensus_agreement.png` |
| `10_views.py` | logistic regression on each view (activity, profile flags, colour, text counts, time zone, text) | `views_accuracy.csv`, `fig_views_accuracy.png` |
| `11_amendments.py` | first-person and organisation word groups, the recorded-human patterns and the suggested amendment for every candidate (report 10.6) | `amendments_{non_human_to_human,human_to_non_human,signals,suggested}.csv` |

All outputs go to `data/output/`. `SEED = 7` wherever there is randomness, so a rerun with the
tested versions in `requirements.txt` reproduces the files; other library versions can move a few flags.