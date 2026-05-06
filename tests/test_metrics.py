"""Tests for evaluation metrics."""

import sys
sys.path.insert(0, ".")

import numpy as np
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
    assert stats["high_confidence_error_count"] == 1  # last one: conf=0.92, wrong
    assert stats["n_high_confidence"] == 3  # 0.95, 0.90, 0.92


if __name__ == "__main__":
    test_classification_metrics()
    test_ece()
    test_confidence_stats()
    print("All metrics tests passed.")
