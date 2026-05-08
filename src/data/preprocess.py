"""Preprocess raw review data: clean text, create labels, build metadata features."""

import os
import re
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_dir="data/raw"):
    reviews = pd.read_parquet(os.path.join(raw_dir, "reviews.parquet"))

    meta_path = os.path.join(raw_dir, "metadata.parquet")
    meta = pd.read_parquet(meta_path) if os.path.exists(meta_path) else None

    return reviews, meta


def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def create_binary_label(df, cfg):
    """Create binary sentiment label from rating, drop neutrals."""
    pos_thresh = cfg["data"]["positive_threshold"]
    neg_thresh = cfg["data"]["negative_threshold"]

    df = df.copy()
    df = df[df["rating"].notna()]
    df["rating"] = df["rating"].astype(float)

    df = df[(df["rating"] >= pos_thresh) | (df["rating"] <= neg_thresh)]
    df["label"] = (df["rating"] >= pos_thresh).astype(int)

    print(f"After dropping neutrals: {len(df)} reviews")
    print(f"  Positive: {df['label'].sum()} | Negative: {(df['label'] == 0).sum()}")
    return df


def merge_metadata(reviews_df, meta_df):
    """Merge product metadata into reviews."""
    if meta_df is None:
        print("No metadata available, skipping merge.")
        return reviews_df

    meta_cols = ["parent_asin"]
    useful_meta = []

    for col in ["average_rating", "rating_number", "price", "main_category"]:
        if col in meta_df.columns:
            useful_meta.append(col)

    if not useful_meta:
        print("No useful metadata columns found.")
        return reviews_df

    meta_subset = meta_df[["parent_asin"] + useful_meta].drop_duplicates(subset=["parent_asin"])
    merge_key = "parent_asin" if "parent_asin" in reviews_df.columns else "asin"
    merged = reviews_df.merge(meta_subset, left_on=merge_key, right_on="parent_asin", how="left")

    print(f"Merged metadata. Columns added: {useful_meta}")
    return merged


def build_metadata_features(df):
    """Create metadata features from available columns."""
    df = df.copy()

    df["review_text"] = df["text"].apply(clean_text)
    df["review_length_words"] = df["review_text"].apply(lambda x: len(x.split()))
    df["review_length_chars"] = df["review_text"].apply(len)

    if "helpful_vote" not in df.columns:
        df["helpful_vote"] = 0
    df["helpful_vote"] = df["helpful_vote"].fillna(0).astype(int)

    if "verified_purchase" in df.columns:
        df["verified_purchase"] = df["verified_purchase"].astype(int)
    else:
        df["verified_purchase"] = 1

    if "average_rating" in df.columns:
        df["product_average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    else:
        df["product_average_rating"] = np.nan

    if "rating_number" in df.columns:
        df["product_rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce")
    else:
        df["product_rating_number"] = np.nan

    if "price" in df.columns:
        df["product_price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce"
        )
    else:
        df["product_price"] = np.nan

    if "timestamp" in df.columns:
        df["review_year"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce").dt.year
    else:
        df["review_year"] = np.nan

    # Drop reviews with empty text
    df = df[df["review_text"].str.len() > 0]

    return df


def select_final_columns(df):
    """Keep only the columns needed for modeling."""
    keep = [
        "review_text", "label",
        "review_length_words", "review_length_chars",
        "helpful_vote", "verified_purchase",
        "product_average_rating", "product_rating_number",
        "product_price", "review_year",
        "rating",  # kept for disagreement computation, NOT as a feature
    ]
    available = [c for c in keep if c in df.columns]
    return df[available].reset_index(drop=True)


def balance_classes(df, seed=42):
    """Downsample majority class to match minority class size."""
    counts = df["label"].value_counts()
    minority_n = counts.min()
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    majority_df = df[df["label"] == majority_label].sample(n=minority_n, random_state=seed)
    minority_df = df[df["label"] == minority_label]
    balanced = pd.concat([majority_df, minority_df]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Balanced classes: {len(balanced)} samples ({minority_n} per class)")
    return balanced


def preprocess_pipeline(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    reviews_df, meta_df = load_raw_data()

    reviews_df = create_binary_label(reviews_df, cfg)
    reviews_df = merge_metadata(reviews_df, meta_df)
    reviews_df = build_metadata_features(reviews_df)
    reviews_df = select_final_columns(reviews_df)

    if cfg["data"].get("balance_classes", False):
        reviews_df = balance_classes(reviews_df, seed=cfg["project"]["seed"])

    meta_feats = [
        "product_average_rating", "product_rating_number",
        "product_price", "review_year"
    ]
    for col in meta_feats:
        if col in reviews_df.columns:
            median_val = reviews_df[col].median()
            reviews_df[col] = reviews_df[col].fillna(median_val)

    os.makedirs("data/interim", exist_ok=True)
    out_path = "data/interim/preprocessed.parquet"
    reviews_df.to_parquet(out_path, index=False)
    print(f"Saved preprocessed data to {out_path}: {reviews_df.shape}")
    return reviews_df


if __name__ == "__main__":
    preprocess_pipeline()
