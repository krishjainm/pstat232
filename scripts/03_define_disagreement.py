"""Script 03: Define agreement/disagreement labels using text sentiment scoring."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import yaml
from src.features.define_disagreement import add_disagreement_labels


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    # Try transformer first, fall back to TF-IDF if it fails
    try:
        print("Using pretrained transformer for text sentiment scoring...")
        method = "transformer"
        train_df = add_disagreement_labels(train_df, cfg, method=method)
        val_df = add_disagreement_labels(val_df, cfg, method=method)
        test_df = add_disagreement_labels(test_df, cfg, method=method)
    except Exception as e:
        print(f"Transformer failed ({e}), falling back to TF-IDF...")
        method = "tfidf"
        train_df = add_disagreement_labels(train_df, cfg, method=method, train_df=train_df)
        val_df = add_disagreement_labels(val_df, cfg, method=method, train_df=train_df)
        test_df = add_disagreement_labels(test_df, cfg, method=method, train_df=train_df)

    train_df.to_parquet("data/processed/train.parquet", index=False)
    val_df.to_parquet("data/processed/val.parquet", index=False)
    test_df.to_parquet("data/processed/test.parquet", index=False)

    print("\nOverall disagreement stats (train):")
    print(train_df["disagreement_group"].value_counts())
    print(f"\nDisagreement rate: {(train_df['agreement_status'] == 'disagreement').mean():.3f}")

    print("\nDone. Updated splits saved to data/processed/")


if __name__ == "__main__":
    main()
