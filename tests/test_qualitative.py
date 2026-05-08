"""Tests for qualitative error analysis module."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.evaluation.qualitative import (
    classify_error_type,
    build_error_taxonomy,
    generate_case_studies,
)


def test_classify_error_type_sarcasm():
    text = "Love this product! It broke on the first day."
    assert classify_error_type(text, 0, 1) == "sarcasm_irony"


def test_classify_error_type_mixed():
    text = "The product is great quality but unfortunately it arrived damaged."
    assert classify_error_type(text, 0, 1) == "mixed_sentiment"


def test_classify_error_type_rating_misuse():
    text = "Shipping was late and the package arrived damaged by USPS."
    assert classify_error_type(text, 0, 1) == "rating_misuse"


def test_classify_error_type_ambiguous():
    text = "ok"
    assert classify_error_type(text, 0, 1) == "short_review"


def test_classify_error_type_empty():
    assert classify_error_type("", 0, 1) == "empty_text"
    assert classify_error_type(None, 0, 1) == "empty_text"


def test_classify_error_type_normal():
    text = "This is a really amazing device that works perfectly for my needs every single day."
    result = classify_error_type(text, 1, 0)
    assert result == "other"


def test_build_error_taxonomy():
    df = pd.DataFrame({
        "review_text": [
            "Love this, broke in a day",
            "Great but unfortunately damaged",
            "Shipping was terrible and late",
            "ok",
            "Really good product works well for me every day",
            "Perfect quality, highly recommend it to everyone I know",
            "Bad product, do not buy this ever again",
        ],
        "label":      [0, 0, 0, 0, 1, 1, 0],
        "mm_pred":    [1, 1, 1, 1, 0, 0, 1],
        "mm_confidence": [0.95, 0.80, 0.90, 0.60, 0.85, 0.70, 0.92],
        "text_pred":  [1, 1, 1, 0, 0, 1, 1],
        "meta_pred":  [0, 0, 1, 1, 1, 0, 0],
        "agreement_status": ["disagreement"] * 4 + ["agreement"] * 3,
        "disagreement_group": ["strong_disagreement"] * 4 + ["agreement"] * 3,
    })

    taxonomy, errors = build_error_taxonomy(df, max_errors=100)
    assert len(taxonomy) > 0
    assert "error_type" in taxonomy.columns
    assert "count" in taxonomy.columns
    assert "percentage" in taxonomy.columns


def test_generate_case_studies():
    df = pd.DataFrame({
        "review_text": [
            "Love this, broke in a day just terrible",
            "Great but unfortunately was damaged on arrival",
            "Shipping was terrible and late and bad overall",
            "This is a simple okay product for the price",
            "Really good product works very well for me",
        ],
        "label":      [0, 0, 0, 0, 1],
        "mm_pred":    [1, 1, 1, 1, 0],
        "mm_confidence": [0.95, 0.92, 0.90, 0.88, 0.86],
        "text_pred":  [1, 1, 1, 0, 0],
        "meta_pred":  [0, 0, 1, 1, 1],
        "agreement_status": ["disagreement"] * 5,
        "disagreement_group": ["strong_disagreement"] * 5,
    })

    cases = generate_case_studies(df, n_cases=3)
    assert len(cases) <= 3
    assert "text_preview" in cases.columns
    assert "error_type" in cases.columns


if __name__ == "__main__":
    test_classify_error_type_sarcasm()
    test_classify_error_type_mixed()
    test_classify_error_type_rating_misuse()
    test_classify_error_type_ambiguous()
    test_classify_error_type_empty()
    test_classify_error_type_normal()
    test_build_error_taxonomy()
    test_generate_case_studies()
    print("All qualitative tests passed.")
