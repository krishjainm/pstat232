"""Tests for disagreement definition."""

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

    # row 1: disagree, conf=0.60 < 0.70 → weak
    assert result["disagreement_group"].iloc[1] == "weak_disagreement"
    # row 3: disagree, conf=0.95 >= 0.90 → strong
    assert result["disagreement_group"].iloc[3] == "strong_disagreement"
    # row 4: disagree, conf=0.80, 0.70 <= 0.80 < 0.90 → medium
    assert result["disagreement_group"].iloc[4] == "medium_disagreement"


def test_modality_dominance():
    from src.evaluation.modality_dominance import classify_dominance

    row_text = pd.Series({"text_pred": 1, "meta_pred": 0, "mm_pred": 1})
    assert classify_dominance(row_text) == "text_dominant"

    row_meta = pd.Series({"text_pred": 1, "meta_pred": 0, "mm_pred": 0})
    assert classify_dominance(row_meta) == "metadata_dominant"

    row_both = pd.Series({"text_pred": 1, "meta_pred": 1, "mm_pred": 1})
    assert classify_dominance(row_both) == "both_agree"


if __name__ == "__main__":
    test_assign_disagreement()
    test_modality_dominance()
    print("All disagreement tests passed.")
