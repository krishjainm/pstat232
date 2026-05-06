"""Script 04: Train text-only sentiment model."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import yaml
from src.models.train_text_model import (
    train_tfidf_logreg, train_distilbert,
    predict_tfidf_logreg, predict_distilbert,
)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")

    model_type = cfg["models"]["text_model"]["type"]

    if model_type == "distilbert":
        try:
            print("Training DistilBERT text model...")
            train_distilbert(train_df, val_df, cfg)
        except Exception as e:
            print(f"DistilBERT training failed ({e}), falling back to TF-IDF + LogReg...")
            train_tfidf_logreg(train_df, val_df)
    else:
        print("Training TF-IDF + LogReg text model...")
        train_tfidf_logreg(train_df, val_df)

    print("Text model training complete.")


if __name__ == "__main__":
    main()
