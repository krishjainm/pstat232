"""Create stratified train/val/test splits."""

import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_splits(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    seed = cfg["project"]["seed"]
    train_size = cfg["splits"]["train_size"]
    val_size = cfg["splits"]["val_size"]
    test_size = cfg["splits"]["test_size"]

    df = pd.read_parquet("data/interim/preprocessed.parquet")
    print(f"Loaded {len(df)} preprocessed samples")

    # First split: train vs (val + test)
    val_test_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df, test_size=val_test_size, random_state=seed,
        stratify=df["label"]
    )

    # Second split: val vs test (proportional within the held-out set)
    relative_test = test_size / val_test_size
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, random_state=seed,
        stratify=temp_df["label"]
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train pos rate: {train_df['label'].mean():.3f}")
    print(f"Val   pos rate: {val_df['label'].mean():.3f}")
    print(f"Test  pos rate: {test_df['label'].mean():.3f}")

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_parquet("data/processed/train.parquet", index=False)
    val_df.to_parquet("data/processed/val.parquet", index=False)
    test_df.to_parquet("data/processed/test.parquet", index=False)
    print("Saved train/val/test splits to data/processed/")

    return train_df, val_df, test_df


if __name__ == "__main__":
    create_splits()
