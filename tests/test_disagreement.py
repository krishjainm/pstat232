"""Tests for disagreement definition and modality dominance."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.features.define_disagreement import assign_disagreement


def test_assign_disagreement():
    cfg = {"disagreement": {"weak_threshold": 0.70, "medium_threshold": 0.90, "strong_threshold": 0.90}}

    df = pd.DataFrame({
        "label": [1, 1, 0, 0, 1],
        "text_sentiment_label": [1, 0, 0, 1, 0],
        "text_sentiment_confidence": [0.95, 0.60, 0.85, 0.95, 0.80],
    })

    result = assign_disagreement(df, cfg)

    assert result["agreement_status"].iloc[0] == "agreement"
    assert result["agreement_status"].iloc[1] == "disagreement"
    assert result["agreement_status"].iloc[2] == "agreement"
    assert result["agreement_status"].iloc[3] == "disagreement"

    assert result["disagreement_group"].iloc[1] == "weak_disagreement"
    assert result["disagreement_group"].iloc[3] == "strong_disagreement"
    assert result["disagreement_group"].iloc[4] == "medium_disagreement"


def test_modality_dominance():
    from src.evaluation.modality_dominance import classify_dominance

    row_text = pd.Series({"text_pred": 1, "meta_pred": 0, "mm_pred": 1})
    assert classify_dominance(row_text) == "text_dominant"

    row_meta = pd.Series({"text_pred": 1, "meta_pred": 0, "mm_pred": 0})
    assert classify_dominance(row_meta) == "metadata_dominant"

    row_both = pd.Series({"text_pred": 1, "meta_pred": 1, "mm_pred": 1})
    assert classify_dominance(row_both) == "both_agree"


def test_probability_dominance():
    from src.evaluation.modality_dominance import compute_probability_dominance

    df = pd.DataFrame({
        "mm_prob_positive": [0.9, 0.2, 0.6, 0.8],
        "text_prob_positive": [0.85, 0.3, 0.7, 0.1],
        "meta_prob_positive": [0.5, 0.1, 0.5, 0.9],
        "agreement_status": ["agreement", "disagreement", "disagreement", "disagreement"],
    })

    result_df, summary = compute_probability_dominance(df)
    assert "dominance_ratio" in result_df.columns
    assert len(result_df) == 4
    assert 0 <= result_df["dominance_ratio"].min()
    assert result_df["dominance_ratio"].max() <= 1
    assert "disagree_mean_ratio" in summary


def test_conflict_dominance():
    from src.evaluation.modality_dominance import compute_conflict_dominance

    df = pd.DataFrame({
        "text_pred": [1, 0, 1, 0, 1],
        "meta_pred": [0, 1, 1, 0, 0],
        "mm_pred":   [1, 0, 1, 0, 0],
        "label":     [1, 0, 1, 0, 1],
    })

    results, stats = compute_conflict_dominance(df)
    assert len(results) == 3
    assert stats["n_conflict_cases"] == 3  # rows 0, 1, 4 have text != meta
    assert results["count"].sum() == 3


def test_conflict_dominance_no_conflicts():
    from src.evaluation.modality_dominance import compute_conflict_dominance

    df = pd.DataFrame({
        "text_pred": [1, 0, 1],
        "meta_pred": [1, 0, 1],
        "mm_pred":   [1, 0, 1],
        "label":     [1, 0, 1],
    })

    results, stats = compute_conflict_dominance(df)
    assert len(results) == 0


if __name__ == "__main__":
    test_assign_disagreement()
    test_modality_dominance()
    test_probability_dominance()
    test_conflict_dominance()
    test_conflict_dominance_no_conflicts()
    print("All disagreement tests passed.")
