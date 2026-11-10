"""Fixed, interpretable model candidates and fold-safe preprocessing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import NUMERIC_FEATURES, CATEGORICAL_FEATURES

SEED = 42


def make_preprocessor(*, include_status=False):
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)
    if include_status:
        categorical.append('Status')
    return ColumnTransformer([
        ('numerical', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), numeric),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical),
    ], remainder='drop')


def make_candidates(*, include_status=False, seed=SEED):
    preprocessing = make_preprocessor(include_status=include_status)
    return {
        'Prevalence baseline': DummyClassifier(strategy='prior'),
        'Logistic regression': Pipeline([
            ('preprocess', preprocessing),
            ('classifier', LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        'Random forest': Pipeline([
            ('preprocess', make_preprocessor(include_status=include_status)),
            ('classifier', RandomForestClassifier(
                n_estimators=180, min_samples_leaf=5, max_features='sqrt',
                random_state=seed, n_jobs=1)),
        ]),
        'Gradient boosting': Pipeline([
            ('preprocess', make_preprocessor(include_status=include_status)),
            ('classifier', HistGradientBoostingClassifier(
                max_iter=140, learning_rate=0.06, max_leaf_nodes=13,
                min_samples_leaf=16, l2_regularization=0.6, random_state=seed)),
        ]),
    }


def profile_disjoint_holdout(data, *, features, target, fraction=0.2, seed=SEED, attempts=100):
    """Deterministic approximate-stratified holdout with no shared input profiles.

    This is a *sensitivity protocol*, motivated by repeated predictor profiles
    found during dataset quality review. Same values do not prove same customer.
    Test size/prevalence matching never uses fitted models or their metrics.
    """
    if not 0 < fraction < 1:
        raise ValueError('fraction must be strictly between 0 and 1')
    from pandas import MultiIndex
    groups = pd.factorize(MultiIndex.from_frame(data.loc[:, list(features)]))[0]
    y = data[target].to_numpy(dtype=int)
    splitter = GroupShuffleSplit(n_splits=attempts, test_size=fraction, random_state=seed)
    chosen = None
    best = float('inf')
    for train_idx, test_idx in splitter.split(data, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        deviation = abs(len(test_idx) / len(y) - fraction) + 2 * abs(y[test_idx].mean() - y.mean())
        if deviation < best:
            best = deviation
            chosen = (train_idx, test_idx)
    if chosen is None:
        raise ValueError('Cannot construct a two-class, group-disjoint holdout')
    train_idx, test_idx = chosen
    if not set(groups[train_idx]).isdisjoint(groups[test_idx]):
        raise AssertionError('Repeated predictor profiles cross the holdout boundary')
    return train_idx, test_idx, groups


def compare_candidates(X, y, *, groups=None, n_splits=5, seed=SEED):
    """Fit preprocessing per fold; optionally isolate duplicate predictor profiles."""
    cv = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
          if groups is not None else
          StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    candidates = make_candidates(seed=seed)
    rows = []
    for name, model in candidates.items():
        scores = cross_validate(
            model, X, y, cv=cv, groups=groups,
            scoring={'ap': 'average_precision', 'auc': 'roc_auc'},
            n_jobs=1, error_score='raise',
        )
        rows.append({
            'model': name,
            'mean_cv_ap': float(np.mean(scores['test_ap'])),
            'std_cv_ap': float(np.std(scores['test_ap'], ddof=1)),
            'mean_cv_auc': float(np.mean(scores['test_auc'])),
        })
    return pd.DataFrame(rows).sort_values('mean_cv_ap', ascending=False, kind='stable').reset_index(drop=True)
