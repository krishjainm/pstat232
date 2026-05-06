"""Script 05: Train metadata-only sentiment model."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import yaml
from src.models.train_metadata_model import train_metadata_model


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")

    train_metadata_model(train_df, val_df, cfg)
    print("Metadata model training complete.")


if __name__ == "__main__":
    main()
