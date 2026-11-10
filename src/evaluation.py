"""Leakage-aware thresholding, probability evaluation and capacity ranking."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, fbeta_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def validate_binary_scores(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or y.size != scores.size or y.size < 2:
        raise ValueError("Expected matching one-dimensional nonempty label and score arrays")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes (0 and 1) are required")
    if not np.isfinite(scores).all() or np.min(scores) < 0 or np.max(scores) > 1:
        raise ValueError("Probabilities must be finite and between zero and one")
    return y, scores


def select_f2_threshold(y, scores):
    """Optimize F2 on development-only OOF probabilities, NOT on the holdout.

    Tie breaks favor higher thresholds (smaller alerted groups). Thresholds
    are drawn from observed scores and therefore avoid an arbitrary grid.
    """
    y, scores = validate_binary_scores(y, scores)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    num = 5.0 * precision[:-1] * recall[:-1]
    den = 4.0 * precision[:-1] + recall[:-1]
    f2 = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    best = np.flatnonzero(np.isclose(f2, np.max(f2), rtol=1e-12, atol=1e-12))[-1]
    return float(thresholds[best])


def operating_point(y, scores, threshold):
    y, scores = validate_binary_scores(y, scores)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")
    predicted = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "flagged_fraction": float(predicted.mean()),
        "flagged_count": int(predicted.sum()),
    }


def capacity_metrics(y, scores, *, fraction=0.15):
    """Descriptive concentration: not incremental retention or causal uplift."""
    y, scores = validate_binary_scores(y, scores)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = int(np.ceil(len(y) * fraction))
    # Deterministic ties by original position for reproducibility.
    order = np.lexsort((np.arange(len(scores)), -scores))
    hit_count = int(y[order[:k]].sum())
    positives = int(y.sum())
    precision_at_k = hit_count / k
    prevalence = positives / len(y)
    return {
        "fraction": float(fraction),
        "reviewed_count": k,
        "churners_in_reviewed": hit_count,
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(hit_count / positives),
        "lift_at_k": float(precision_at_k / prevalence),
        "prevalence": float(prevalence),
    }


def probability_metrics(y, scores):
    y, scores = validate_binary_scores(y, scores)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "prevalence": float(y.mean()),
    }


def stratified_bootstrap_intervals(y, scores, *, n_resamples=300, seed=42):
    """Conditional percentile intervals for metrics, not external validation.

    Resamples each observed class separately, preserving holdout class counts;
    these intervals ignore model-development uncertainty and dataset shift.
    """
    y, scores = validate_binary_scores(y, scores)
    if n_resamples < 20:
        raise ValueError("n_resamples must be at least 20")
    rng = np.random.default_rng(seed)
    class_zero = np.flatnonzero(y == 0)
    class_one = np.flatnonzero(y == 1)
    estimates = {"average_precision": [], "roc_auc": [], "brier_score": []}
    for _ in range(n_resamples):
        idx = np.concatenate([
            rng.choice(class_zero, size=len(class_zero), replace=True),
            rng.choice(class_one, size=len(class_one), replace=True),
        ])
        metrics = probability_metrics(y[idx], scores[idx])
        for name in estimates:
            estimates[name].append(metrics[name])
    return {name: tuple(map(float, np.quantile(values, [0.025, 0.975])))
            for name, values in estimates.items()}


def profile_bootstrap_intervals(y, scores, groups, *, n_resamples=300, seed=42):
    """Resample entire repeated-profile clusters for conditional metric intervals.

    This acknowledges record dependence within identical observed profiles.
    It does not include model-fit, temporal-shift or source-selection uncertainty.
    """
    y, scores = validate_binary_scores(y, scores)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(y):
        raise ValueError('groups must match labels')
    if n_resamples < 20:
        raise ValueError('At least 20 resamples are required')
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    rng = np.random.default_rng(seed)
    estimates = {'average_precision': [], 'roc_auc': [], 'brier_score': []}
    trials = 0
    while len(estimates['roc_auc']) < n_resamples and trials < 10 * n_resamples:
        trials += 1
        chosen = rng.integers(0, len(members), len(members))
        sample_idx = np.concatenate([members[i] for i in chosen])
        if len(np.unique(y[sample_idx])) < 2:
            continue
        metrics = probability_metrics(y[sample_idx], scores[sample_idx])
        for name in estimates:
            estimates[name].append(metrics[name])
    if len(estimates['roc_auc']) < n_resamples:
        raise RuntimeError('Insufficient two-class cluster bootstrap resamples')
    return {key: tuple(map(float, np.quantile(val, [0.025, 0.975])))
            for key, val in estimates.items()}
