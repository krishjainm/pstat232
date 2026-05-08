"""Tests for evaluation metrics and statistical tests."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.calibration import expected_calibration_error, confidence_stats


def test_classification_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.8, 0.7])

    m = compute_classification_metrics(y_true, y_pred, y_prob)
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["f1"] <= 1
    assert 0 <= m["auroc"] <= 1
    assert m["accuracy"] == 4 / 6


def test_ece():
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
    ece, _, _, _ = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0 <= ece <= 1


def test_confidence_stats():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    conf = np.array([0.95, 0.80, 0.90, 0.92])

    stats = confidence_stats(y_true, y_pred, conf, high_conf_threshold=0.90)
    assert stats["high_confidence_error_count"] == 1
    assert stats["n_high_confidence"] == 3


def test_bootstrap_metric():
    from src.evaluation.statistical_tests import bootstrap_metric

    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    y_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.8, 0.7, 0.85, 0.9, 0.15, 0.6])

    def acc_fn(yt, yp, _):
        return (yt == yp).mean()

    point, lo, hi = bootstrap_metric(y_true, y_pred, y_prob, acc_fn, n_resamples=500, seed=42)
    assert lo <= point <= hi
    assert 0 <= lo
    assert hi <= 1


def test_mcnemar():
    from src.evaluation.statistical_tests import mcnemar_test

    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    y_pred_a = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    y_pred_b = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0])

    chi2, p = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert chi2 >= 0
    assert 0 <= p <= 1


def test_mcnemar_identical():
    from src.evaluation.statistical_tests import mcnemar_test

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])

    chi2, p = mcnemar_test(y_true, y_pred, y_pred)
    assert chi2 == 0
    assert p == 1.0


if __name__ == "__main__":
    test_classification_metrics()
    test_ece()
    test_confidence_stats()
    test_bootstrap_metric()
    test_mcnemar()
    test_mcnemar_identical()
    print("All metrics tests passed.")
