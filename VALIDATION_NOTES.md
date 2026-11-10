# Validation notes — real UCI execution

This project was evaluated with the original **Iranian Churn** CSV from UCI (3,150 rows, 495 churn events). The stored notebook is the executed final version using Python 3.13.2. All 15 code cells completed and have saved outputs. The package maintains nine standalone offline tests.

## Why the protocol changed

An initial row-wise split produced ROC-AUC 0.9848 and average precision 0.9364, but a quality audit revealed 300 full-row duplicates and 365 records whose nine-feature vectors repeated an earlier record (including full-row duplicates). Eighteen feature-profile groups contained conflicting churn labels. These observations motivated a revised evaluation that groups exact input profiles across development/holdout partitions and development CV folds. A repeated profile does **not** establish that two rows represent the same person.

The revised partition had **0 cross-split overlapping feature profiles**, 2,516 development records, and 634 holdout records (99 observed churn events). The updated procedure was adopted **after reviewing the first run** and must not be described as independently preregistered or externally validated.

## Final protocol results

- Model chosen by development five-fold mean average precision: **Histogram Gradient Boosting** (CV AP **0.8880**, CV ROC-AUC **0.9734**).
- Holdout: **average precision 0.9033**, **ROC-AUC 0.9752**, **Brier score 0.0415**.
- Frozen F2 threshold from development out-of-fold predictions: **0.1749**. At that threshold the test set has **precision 0.6118**, **recall 0.9394**, and **152/634 accounts flagged**.
- Illustrative 15% review budget: **96/634 accounts**, including **79/99 observed churners** (79.8% recall at capacity, 5.27x lift versus prevalence). This is not causal evidence of retention benefit.
- Conditional percentile bootstrap ranges using repeated predictor-profile clusters: AP **0.8433–0.9487**, ROC-AUC **0.9600–0.9887**, Brier **0.0288–0.0556**. Bootstrap does not include model-selection or cross-dataset uncertainty.

## Software checks and limits

- Notebook: code and execution source cross-checked against the revised packaged source; 15 cells executed, no error outputs.
- All five PNG plots in `figures/` were recovered **directly from the saved execution outputs**. No plots or scores were synthesized for publication.
- The original UCI CSV is **not redistributed**; the downloader obtains it again for local reproduction.
- Model and threshold were selected using development data only. Permutation importance and subgroup descriptions are post-hoc test-set diagnostics, not model-selection inputs.
- No independent temporal test, customer ID, documented action outcomes, or prospective production evaluation is available.

See `README.md` and `DATA_PROVENANCE.md` for full study documentation and data licensing.
