"""Script 09: Train and compare all fusion architectures (early, late, gated)."""

import sys
sys.path.insert(0, ".")

import os
import copy
import yaml
import numpy as np
import pandas as pd
import shutil

from src.models.train_multimodal_model import train_multimodal, predict_multimodal
from src.evaluation.metrics import compute_classification_metrics
from src.models.predict import load_config


def main():
    cfg = load_config()
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    fusion_types = ["early", "late", "gated"]
    results = []

    for ft in fusion_types:
        print(f"\n{'='*60}")
        print(f"Training fusion_type={ft}")
        print(f"{'='*60}")

        save_dir = f"models/multimodal_{ft}"
        ft_cfg = copy.deepcopy(cfg)
        ft_cfg["models"]["multimodal_model"]["fusion_type"] = ft

        train_multimodal(train_df, val_df, ft_cfg, save_dir=save_dir)

        preds, prob_pos, conf = predict_multimodal(test_df, "test", ft_cfg, save_dir=save_dir)
        y_true = test_df["label"].values

        metrics = compute_classification_metrics(y_true, preds, prob_pos)
        metrics["fusion_type"] = ft

        disagree_mask = test_df["agreement_status"] == "disagreement"
        if disagree_mask.sum() > 0:
            d_metrics = compute_classification_metrics(
                y_true[disagree_mask], preds[disagree_mask], prob_pos[disagree_mask]
            )
            metrics["disagree_accuracy"] = d_metrics["accuracy"]
            metrics["disagree_f1"] = d_metrics["f1"]

        results.append(metrics)
        print(f"{ft}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

    results_df = pd.DataFrame(results)
    os.makedirs("reports/tables", exist_ok=True)
    results_df.to_csv("reports/tables/fusion_comparison_results.csv", index=False)
    print("\n--- Fusion Comparison ---")
    print(results_df.to_string(index=False))
    print("\nSaved to reports/tables/fusion_comparison_results.csv")


if __name__ == "__main__":
    main()
