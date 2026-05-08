"""Standard classification metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss,
)


def compute_classification_metrics(y_true, y_pred, y_prob_positive):
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob_positive)
    except ValueError:
        metrics["auroc"] = np.nan

    metrics["brier_score"] = brier_score_loss(y_true, y_prob_positive)

    return metrics


def compute_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def compute_majority_baseline(y_true):
    """Compute metrics for a majority-class baseline."""
    majority_class = int(np.round(y_true.mean()))
    y_pred = np.full_like(y_true, majority_class)
    y_prob = np.full(len(y_true), y_true.mean() if majority_class == 1 else 1 - y_true.mean())

    return compute_classification_metrics(y_true, y_pred, y_prob)


def compute_all_model_metrics(df):
    """Compute metrics for all three models + majority baseline."""
    results = {}
    y_true = df["label"].values

    results["majority_baseline"] = compute_majority_baseline(y_true)

    for name, pred_col, prob_col in [
        ("text_only", "text_pred", "text_prob_positive"),
        ("metadata_only", "meta_pred", "meta_prob_positive"),
        ("multimodal", "mm_pred", "mm_prob_positive"),
    ]:
        results[name] = compute_classification_metrics(
            y_true, df[pred_col].values, df[prob_col].values
        )

    return results
