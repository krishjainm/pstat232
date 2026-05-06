"""Script 08: Generate all figures for the report."""

import sys
sys.path.insert(0, ".")

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
)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    test_df = pd.read_parquet("data/processed/test_with_predictions.parquet")
    group_df = pd.read_csv("reports/tables/group_results.csv")
    dom_df = pd.read_csv("reports/tables/modality_dominance_results.csv")

    print("Generating figures...")

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

    print("\nAll figures saved to reports/figures/")


if __name__ == "__main__":
    main()
