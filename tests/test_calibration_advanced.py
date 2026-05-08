"""Tests for advanced calibration: temperature scaling, selective prediction."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.evaluation.temperature_scaling import (
    find_optimal_temperature,
    apply_temperature_scaling,
    compute_group_conditional_calibration,
    selective_prediction_curve,
    compute_selective_prediction,
)


def test_find_optimal_temperature():
    np.random.seed(42)
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.95, 0.05, 0.80, 0.20, 0.70, 0.90, 0.10, 0.15, 0.85, 0.25])

    T = find_optimal_temperature(y_true, y_prob, n_bins=5)
    assert 0.1 <= T <= 10.0


def test_apply_temperature_scaling():
    y_prob = np.array([0.9, 0.1, 0.7, 0.3])
    calibrated = apply_temperature_scaling(y_prob, temperature=2.0)

    assert len(calibrated) == 4
    assert all(0 <= p <= 1 for p in calibrated)
    # Higher temperature should push probabilities toward 0.5
    assert abs(calibrated[0] - 0.5) < abs(y_prob[0] - 0.5)


def test_group_conditional_calibration():
    df = pd.DataFrame({
        "label": [1, 0, 1, 0, 1, 0],
        "text_prob_positive": [0.9, 0.1, 0.8, 0.3, 0.7, 0.2],
        "meta_prob_positive": [0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
        "mm_prob_positive": [0.85, 0.15, 0.75, 0.25, 0.65, 0.35],
        "agreement_status": ["agreement", "agreement", "disagreement",
                             "disagreement", "agreement", "disagreement"],
    })

    result = compute_group_conditional_calibration(df, n_bins=3)
    assert len(result) > 0
    assert "model" in result.columns
    assert "group" in result.columns
    assert "ece" in result.columns


def test_selective_prediction_curve():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 1])
    confidence = np.array([0.95, 0.90, 0.80, 0.60, 0.55])

    curve = selective_prediction_curve(y_true, y_pred, confidence, n_thresholds=5)
    assert len(curve) == 5
    assert "threshold" in curve.columns
    assert "accuracy" in curve.columns
    assert "coverage" in curve.columns
    # Higher threshold should give higher accuracy (or NaN)
    high_thresh = curve[curve["threshold"] > 0.85]
    if len(high_thresh) > 0 and not high_thresh["accuracy"].isna().all():
        assert high_thresh["accuracy"].dropna().iloc[0] >= 0.5


def test_compute_selective_prediction():
    df = pd.DataFrame({
        "label": [1, 0, 1, 0, 1, 0, 1, 0],
        "text_pred": [1, 0, 1, 1, 1, 0, 1, 0],
        "meta_pred": [1, 0, 0, 0, 1, 1, 1, 0],
        "mm_pred": [1, 0, 1, 0, 1, 0, 0, 1],
        "text_confidence": [0.95, 0.90, 0.80, 0.60, 0.85, 0.75, 0.92, 0.88],
        "meta_confidence": [0.80, 0.85, 0.70, 0.90, 0.75, 0.60, 0.88, 0.82],
        "mm_confidence": [0.90, 0.88, 0.78, 0.65, 0.82, 0.70, 0.91, 0.85],
        "agreement_status": ["agreement", "agreement", "disagreement", "disagreement",
                             "agreement", "disagreement", "agreement", "disagreement"],
    })

    result = compute_selective_prediction(df)
    assert len(result) > 0
    assert set(result.columns) >= {"threshold", "accuracy", "coverage", "model", "group"}


if __name__ == "__main__":
    test_find_optimal_temperature()
    test_apply_temperature_scaling()
    test_group_conditional_calibration()
    test_selective_prediction_curve()
    test_compute_selective_prediction()
    print("All advanced calibration tests passed.")
