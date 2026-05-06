"""Utilities for metadata feature standardization."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


METADATA_FEATURES = [
    "review_length_words",
    "review_length_chars",
    "helpful_vote",
    "verified_purchase",
    "product_average_rating",
    "product_rating_number",
    "product_price",
    "review_year",
]


def get_available_metadata(df):
    """Return metadata feature columns that exist in the dataframe."""
    return [c for c in METADATA_FEATURES if c in df.columns]


def fit_scaler(train_df, save_path="models/metadata_scaler.joblib"):
    """Fit a StandardScaler on training metadata and save it."""
    feats = get_available_metadata(train_df)
    scaler = StandardScaler()
    scaler.fit(train_df[feats].values)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Fitted scaler on {len(feats)} features, saved to {save_path}")
    return scaler


def transform_metadata(df, scaler, feature_cols=None):
    """Apply fitted scaler to metadata features."""
    if feature_cols is None:
        feature_cols = get_available_metadata(df)
    scaled = scaler.transform(df[feature_cols].values)
    return scaled, feature_cols


def load_scaler(path="models/metadata_scaler.joblib"):
    return joblib.load(path)
