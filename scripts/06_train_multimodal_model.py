"""Script 06: Train multimodal fusion model."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import yaml
from src.models.train_multimodal_model import train_multimodal


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")

    train_multimodal(train_df, val_df, cfg)
    print("Multimodal model training complete.")


if __name__ == "__main__":
    main()
