"""Script 08: Generate all figures for the report."""

import sys
sys.path.insert(0, ".")

import os
import pandas as pd
import yaml
from src.visualization.plots import (
    plot_class_distribution,
    plot_agreement_distribution,
    plot_accuracy_by_group,
    plot_f1_by_group,
    plot_calibration_curves,
    plot_confidence_correct_vs_incorrect,
    plot_modality_dominance,
    plot_high_confidence_errors,
    plot_confusion_matrices,
    plot_disagreement_severity_trend,
    plot_probability_dominance_histogram,
    plot_metadata_feature_importance,
    plot_calibration_by_group,
    plot_selective_prediction,
    plot_cross_category_results,
)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def main():
    cfg = load_config()
    test_df = pd.read_parquet("data/processed/test_with_predictions.parquet")
    group_df = pd.read_csv("reports/tables/group_results.csv")
    dom_df = pd.read_csv("reports/tables/modality_dominance_results.csv")

    print("Generating figures...")

    # Original figures (1-10)
    plot_class_distribution(test_df)
    plot_agreement_distribution(test_df)
    plot_accuracy_by_group(group_df)
    plot_f1_by_group(group_df)
    plot_calibration_curves(test_df, n_bins=cfg["evaluation"]["n_bins_calibration"])
    plot_confidence_correct_vs_incorrect(test_df)
    plot_modality_dominance(dom_df)
    plot_high_confidence_errors(test_df, threshold=cfg["evaluation"]["high_confidence_threshold"])
    plot_confusion_matrices(test_df)
    plot_disagreement_severity_trend(group_df)

    # New figures (11-15)
    if "dominance_ratio" in test_df.columns:
        plot_probability_dominance_histogram(test_df)

    feat_imp_df = safe_read_csv("reports/tables/metadata_feature_importance.csv")
    if feat_imp_df is not None:
        plot_metadata_feature_importance(feat_imp_df)

    plot_calibration_by_group(test_df, n_bins=cfg["evaluation"]["n_bins_calibration"])

    sel_pred_df = safe_read_csv("reports/tables/selective_prediction_curves.csv")
    if sel_pred_df is not None:
        plot_selective_prediction(sel_pred_df)

    cross_cat_df = safe_read_csv("reports/tables/cross_category_results.csv")
    if cross_cat_df is not None:
        plot_cross_category_results(cross_cat_df)

    print("\nAll figures saved to reports/figures/")


if __name__ == "__main__":
    main()
