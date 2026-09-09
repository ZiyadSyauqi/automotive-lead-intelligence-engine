"""Threshold metrics and capacity-constrained ranking metrics."""

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    """Top-K uses stable input order to break ties; undefined denominators are None."""
    y = np.asarray(y_true)
    scores = np.asarray(scores)
    if y.ndim != 1 or scores.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected equally sized one-dimensional binary labels and scores")
    if not np.isfinite(scores).all() or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(y):
        raise ValueError("Scores must be finite and K must be an integer in [1, n]")
    hits = int(y[np.argsort(-scores, kind="stable")[:k]].sum())
    positives = int(y.sum())
    precision = hits / k
    return {"k": int(k), "conversions_in_top_k": hits, "precision_at_k": precision,
            "recall_at_k": hits / positives if positives else None,
            "lift_at_k": precision / y.mean() if positives else None}


def evaluate(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    """PR-AUC is reported using average precision (not trapezoidal integration)."""
    y = np.asarray(y_true)
    predicted = scores >= .5
    return {"roc_auc": roc_auc_score(y, scores),
            "pr_auc_average_precision": average_precision_score(y, scores),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0),
            "f1": f1_score(y, predicted, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            "threshold": .5, **ranking_metrics(y, scores, k)}
