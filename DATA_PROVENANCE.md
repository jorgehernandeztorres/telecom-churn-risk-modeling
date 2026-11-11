# Dataset provenance, attribution and permitted use

| Item | Verified origin |
| --- | --- |
| Dataset | **Iranian Churn**, UCI Machine Learning Repository, ID **563** |
| Provider description | Collected from an Iranian telecommunications company's database across a 12-month period |
| Samples | 3,150 customer-level records |
| Features | 13 input attributes plus `Churn` target |
| Observation window | Inputs aggregated during first nine months; churn outcome by end of month 12 |
| UCI publication / donation | 2020 |
| Original dataset URL | https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset |
| DOI | https://doi.org/10.24432/C5JW3Z |
| Dataset license | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| License URL | https://creativecommons.org/licenses/by/4.0/ |

**Attribution:** *Iranian Churn* [Dataset] (2020), UCI Machine Learning Repository, DOI: 10.24432/C5JW3Z. The dataset is distributed under CC BY 4.0.

**Processing:** The source data are loaded without changing the raw CSV on disk. The analysis selects nine documented pre-outcome inputs, omits a near-label `Status` field from its primary predictive model, withholds demographic attributes, fits preprocessing only on development data or within cross-validation training folds, and generates derived model predictions and figures. No fabricated customers are added to the actual-data analysis.

**Distribution:** This repository does *not* bundle the original CSV. It includes a Python downloader which fetches the publicly hosted source archive directly from UCI. The downloaded `data/iranian_churn.csv` is ignored by Git. CC BY 4.0 permits redistribution with attribution if the author later chooses to include the dataset.

**Important caveats:** The source's exact timestamps per customer, sampling design, meaning of non-active `Status` versus `Churn`, formula behind `Customer Value`, and wider population representativeness are not documented sufficiently to support causal or production-deployment claims. We do not assume that contact with a highly ranked customer would prevent churn.

**Project scope:** This is a self-directed portfolio study based on the cited public UCI dataset. The analysis, code and visualizations were prepared specifically for this repository, with AI assistance for implementation review and documentation. The dataset is attributed to UCI; no claim of field deployment or external validation is made.

## Exact-profile duplicates and evaluation policy

During the initial execution on original UCI data, 300 rows were exact full-row duplicates, and 365 rows repeated a nine-feature predictor vector (overlapping categories, **not** 665 distinct duplicate rows). This is an observational data-quality finding; these records cannot be assumed to belong to the same person. Nevertheless, the revised public notebook groups records that match on the *exact predictor columns* across all development/holdout and cross-validation splits to limit duplicate-profile overlap. This change was made after an initial, row-wise analysis, and the resulting estimates are exploratory rather than prospective.

The full source CSV is not bundled. The executed notebook displays a brief preview of three public UCI records; attribution and dataset licensing remain applicable. Dataset source licensing is CC BY 4.0, independent of any code-license decision.
