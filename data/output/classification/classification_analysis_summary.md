# Classification analysis summary

## Experimental design

The analysis used 38 engineered metadata features and treated human as male/female (1) and non-human as brand (0). Only records with crowd-label confidence at or above 1.00 and no recorded label conflict were used as reliable supervised examples.

The fixed splits contained 7826 reliable training records, 2600 reliable validation records and 2657 reliable test records. Hyperparameters and decision thresholds were selected using validation macro F1; the test split was used once for final comparison.

## Model comparison

A Dummy Classifier established the majority-class baseline. Decision Tree, KNN, Naive Bayes and MLPClassifier were compared: they represent interpretable rule-based, local similarity-based, probabilistic and nonlinear learning approaches covered in the subject materials.

The reporting model was chosen by test macro F1, with any model scoring within 0.010 of the top score re-ranked by balanced accuracy and non-human recall rather than macro F1 alone, since the non-human class is the one this project is specifically trying to catch. The selected model was **Decision Tree**, with test macro F1 of 0.788, balanced accuracy of 0.792, ROC-AUC of 0.874, non-human recall of 0.709, and human recall of 0.875.

Macro F1 was the primary selection metric because the classes were imbalanced and ordinary accuracy could favour the larger human class. Balanced accuracy and class-specific recall show whether performance is distributed across both classes. ROC-AUC and non-human average precision assess ranking quality, while Brier score and calibration curves assess whether predicted probabilities are reliable enough for candidate prioritisation.

## Potentially mislabelled profiles

A disagreement was treated as evidence for review rather than proof of an incorrect label. Reliable labelled profiles received out-of-fold predictions, while low-confidence, conflicting and unknown records were predicted by models that had not trained on those records. A strong candidate required a majority of the compared models to contradict the recorded label with predicted-class probability above the configured threshold.

The classification analysis flagged 2808 profiles for review, including 1019 recorded human profiles and 1789 recorded non-human profiles.

These candidates should be combined with association-rule, clustering and text-analysis evidence before recommending a label amendment. Model disagreement can also reflect unusual but correctly labelled accounts, incomplete profile information, or noise in a single sampled tweet.

## Limitations

The recorded labels are crowd annotations rather than verified ground truth. Each account contributes only one sampled tweet, and profile characteristics may have changed since collection. Colour and timezone variables may encode platform defaults rather than identity. Naive Bayes assumes the features are conditionally independent given the class; several engineered features are correlated (tweet rate with tweet count, favourites rate with favourite count, and the sidebar RGB channels with each other), so its probability estimates are less reliable than the other models' even in periods where its accuracy is competitive. Text content (tweet and description text) was not used at this classification stage; a text-based model would need those columns carried through the preprocessing split files. The results therefore identify profiles requiring review, not confirmed annotation errors.