"""Tests for preprocessing functions."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.data.preprocess import clean_text, create_binary_label, build_metadata_features


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("normal text") == "normal text"


def test_create_binary_label():
    cfg = {"data": {"positive_threshold": 4, "negative_threshold": 2}}
    df = pd.DataFrame({"rating": [1, 2, 3, 4, 5]})
    result = create_binary_label(df, cfg)
    assert len(result) == 4  # rating 3 dropped
    assert result["label"].tolist() == [0, 0, 1, 1]


def test_build_metadata_features():
    df = pd.DataFrame({
        "text": ["great product", "bad item", "ok thing"],
        "label": [1, 0, 1],
        "rating": [5, 1, 4],
    })
    result = build_metadata_features(df)
    assert "review_text" in result.columns
    assert "review_length_words" in result.columns
    assert "review_length_chars" in result.columns
    assert result["review_length_words"].iloc[0] == 2


if __name__ == "__main__":
    test_clean_text()
    test_create_binary_label()
    test_build_metadata_features()
    print("All preprocess tests passed.")
