"""Script 07: Evaluate all models — metrics, calibration, group analysis, modality dominance."""

import sys
sys.path.insert(0, ".")

import os
import pandas as pd
import numpy as np
import yaml

from src.models.predict import get_all_predictions, load_config
from src.evaluation.metrics import compute_all_model_metrics
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.group_analysis import compute_group_metrics
from src.evaluation.modality_dominance import compute_modality_dominance, compute_dominance_by_severity


TABLE_DIR = "reports/tables"
os.makedirs(TABLE_DIR, exist_ok=True)


def save_table(df, name):
    path = os.path.join(TABLE_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def main():
    cfg = load_config()

    # Load data
    test_df = pd.read_parquet("data/processed/test.parquet")
    train_df = pd.read_parquet("data/processed/train.parquet")

    # ------ Dataset summary ------
    summary = {
        "split": ["train", "test"],
        "n_samples": [len(train_df), len(test_df)],
        "n_positive": [int(train_df["label"].sum()), int(test_df["label"].sum())],
        "n_negative": [int((train_df["label"] == 0).sum()), int((test_df["label"] == 0).sum())],
        "positive_rate": [train_df["label"].mean(), test_df["label"].mean()],
    }
    save_table(pd.DataFrame(summary), "dataset_summary.csv")

    # ------ Get predictions ------
    print("\nGenerating predictions on test set...")
    test_df = get_all_predictions(test_df, "test", cfg)

    # ------ Main results ------
    main_metrics = compute_all_model_metrics(test_df)
    main_df = pd.DataFrame(main_metrics).T
    main_df.index.name = "model"
    main_df = main_df.reset_index()
    save_table(main_df, "main_results.csv")
    print("\n--- Main Results ---")
    print(main_df.to_string(index=False))

    # ------ Group results ------
    group_df = compute_group_metrics(test_df)
    save_table(group_df, "group_results.csv")
    print("\n--- Group Results ---")
    print(group_df.to_string(index=False))

    # ------ Calibration results ------
    cal_rows = []
    model_specs = [
        ("text_only", "text_pred", "text_prob_positive", "text_confidence"),
        ("metadata_only", "meta_pred", "meta_prob_positive", "meta_confidence"),
        ("multimodal", "mm_pred", "mm_prob_positive", "mm_confidence"),
    ]
    for model_name, pred_col, prob_col, conf_col in model_specs:
        stats, _, _, _ = compute_calibration_metrics(
            test_df, model_name, pred_col, prob_col, conf_col,
            n_bins=cfg["evaluation"]["n_bins_calibration"],
            threshold=cfg["evaluation"]["high_confidence_threshold"],
        )
        stats["model"] = model_name
        cal_rows.append(stats)

    cal_df = pd.DataFrame(cal_rows)
    save_table(cal_df, "calibration_results.csv")
    print("\n--- Calibration Results ---")
    print(cal_df.to_string(index=False))

    # ------ Modality dominance ------
    dom_df = compute_modality_dominance(test_df)
    save_table(dom_df, "modality_dominance_results.csv")

    dom_sev_df = compute_dominance_by_severity(test_df)
    if len(dom_sev_df) > 0:
        save_table(dom_sev_df, "modality_dominance_by_severity.csv")

    # ------ Error examples ------
    incorrect_mask = test_df["mm_pred"] != test_df["label"]
    high_conf_mask = test_df["mm_confidence"] >= cfg["evaluation"]["high_confidence_threshold"]
    disagree_mask = test_df["agreement_status"] == "disagreement"
    error_df = test_df[incorrect_mask & high_conf_mask & disagree_mask].head(20)

    if len(error_df) > 0:
        error_cols = [
            "review_text", "rating", "label",
            "text_sentiment_label", "text_pred", "meta_pred", "mm_pred",
            "mm_confidence", "disagreement_group",
        ]
        available_cols = [c for c in error_cols if c in error_df.columns]
        save_table(error_df[available_cols], "error_examples.csv")

    # Save predictions for figure generation
    test_df.to_parquet("data/processed/test_with_predictions.parquet", index=False)
    print("\nEvaluation complete. All tables saved to reports/tables/")


if __name__ == "__main__":
    main()
