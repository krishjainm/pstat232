"""Run the full pipeline on multiple Amazon review categories for cross-category comparison.

Each category uses isolated directories so results don't overwrite each other.
"""

import sys
sys.path.insert(0, ".")

import os
import copy
import shutil
import yaml
import pandas as pd
import numpy as np

from src.data.download_data import download_reviews, download_metadata, save_raw_data
from src.data.preprocess import (
    load_raw_data, create_binary_label, merge_metadata,
    build_metadata_features, select_final_columns, balance_classes,
)
from src.data.create_splits import create_splits
from src.features.define_disagreement import add_disagreement_labels
from src.models.train_text_model import train_sbert_logreg
from src.models.train_metadata_model import train_metadata_model
from src.models.train_multimodal_model import train_multimodal
from src.models.predict import get_all_predictions
from src.evaluation.metrics import compute_all_model_metrics


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_single_category(category, base_cfg):
    """Run pipeline for one category, return summary metrics dict."""
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["category"] = category

    print(f"\n{'='*60}")
    print(f"  CATEGORY: {category}")
    print(f"{'='*60}")

    try:
        # Download
        reviews_df = download_reviews(cfg)
        meta_df = download_metadata(cfg)
        save_raw_data(reviews_df, meta_df)

        # Preprocess
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
        reviews_df.to_parquet("data/interim/preprocessed.parquet", index=False)

        # Splits
        from sklearn.model_selection import train_test_split
        seed = cfg["project"]["seed"]
        val_test_size = cfg["splits"]["val_size"] + cfg["splits"]["test_size"]
        train_df, temp_df = train_test_split(
            reviews_df, test_size=val_test_size, random_state=seed,
            stratify=reviews_df["label"]
        )
        relative_test = cfg["splits"]["test_size"] / val_test_size
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test, random_state=seed,
            stratify=temp_df["label"]
        )

        # Disagreement labels
        train_df = add_disagreement_labels(train_df, cfg, method="transformer")
        val_df = add_disagreement_labels(val_df, cfg, method="transformer")
        test_df = add_disagreement_labels(test_df, cfg, method="transformer")

        os.makedirs("data/processed", exist_ok=True)
        train_df.to_parquet("data/processed/train.parquet", index=False)
        val_df.to_parquet("data/processed/val.parquet", index=False)
        test_df.to_parquet("data/processed/test.parquet", index=False)

        # Train models
        train_sbert_logreg(train_df, val_df, cfg)
        train_metadata_model(train_df, val_df, cfg)
        train_multimodal(train_df, val_df, cfg)

        # Evaluate
        test_df = get_all_predictions(test_df, "test", cfg)
        main_metrics = compute_all_model_metrics(test_df)

        disagree_mask = test_df["agreement_status"] == "disagreement"
        disagree_rate = disagree_mask.mean()

        summary = {
            "category": category,
            "n_total": len(reviews_df),
            "n_test": len(test_df),
            "disagreement_rate": disagree_rate,
        }

        for model_name, metrics in main_metrics.items():
            if model_name == "majority_baseline":
                continue
            summary[f"{model_name}_accuracy"] = metrics["accuracy"]
            summary[f"{model_name}_f1"] = metrics["f1"]
            summary[f"{model_name}_auroc"] = metrics["auroc"]

        from src.evaluation.metrics import compute_classification_metrics
        from src.evaluation.calibration import expected_calibration_error
        from sklearn.metrics import accuracy_score, brier_score_loss

        if disagree_mask.sum() > 0:
            disagree_subset = test_df[disagree_mask]
            for mn, pc, pp in [
                ("text_only", "text_pred", "text_prob_positive"),
                ("metadata_only", "meta_pred", "meta_prob_positive"),
                ("multimodal", "mm_pred", "mm_prob_positive"),
            ]:
                d_metrics = compute_classification_metrics(
                    disagree_subset["label"].values,
                    disagree_subset[pc].values,
                    disagree_subset[pp].values,
                )
                summary[f"{mn}_disagree_acc"] = d_metrics["accuracy"]
                summary[f"{mn}_disagree_f1"] = d_metrics["f1"]

        # Per-category calibration (ECE/Brier overall + group-conditional) and
        # strong-disagreement accuracy, for the richer cross-category table.
        strong_mask = (test_df["disagreement_group"] == "strong_disagreement").values
        agree_mask = (test_df["agreement_status"] == "agreement").values
        for mn, pc, pp in [
            ("text_only", "text_pred", "text_prob_positive"),
            ("metadata_only", "meta_pred", "meta_prob_positive"),
            ("multimodal", "mm_pred", "mm_prob_positive"),
        ]:
            y = test_df["label"].values
            pr = test_df[pp].values
            pd_ = test_df[pc].values
            ece_overall, _, _, _ = expected_calibration_error(y, pr, 10)
            summary[f"{mn}_ece"] = ece_overall
            summary[f"{mn}_brier"] = brier_score_loss(y, pr)
            if agree_mask.sum() > 0:
                summary[f"{mn}_ece_agreement"], _, _, _ = expected_calibration_error(y[agree_mask], pr[agree_mask], 10)
            if disagree_mask.sum() > 0:
                summary[f"{mn}_ece_disagreement"], _, _, _ = expected_calibration_error(y[disagree_mask.values], pr[disagree_mask.values], 10)
            if strong_mask.sum() > 0:
                summary[f"{mn}_strong_disagree_acc"] = accuracy_score(y[strong_mask], pd_[strong_mask])

        # Text dominance in true-conflict cases
        conflict = test_df[test_df["text_pred"] != test_df["meta_pred"]]
        if len(conflict) > 0:
            follows_text = (conflict["mm_pred"] == conflict["text_pred"]).mean()
            summary["text_dominance_pct"] = follows_text

        print(f"\n{category} complete: overall_acc={summary.get('multimodal_accuracy', 'N/A'):.4f}, "
              f"disagree_rate={disagree_rate:.3f}")
        return summary

    except Exception as e:
        print(f"FAILED for {category}: {e}")
        import traceback
        traceback.print_exc()
        return {"category": category, "error": str(e)}


def main():
    cfg = load_config()
    categories = cfg["data"].get("categories", [cfg["data"]["category"]])

    all_results = []
    for cat in categories:
        result = run_single_category(cat, cfg)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    os.makedirs("reports/tables", exist_ok=True)
    results_df.to_csv("reports/tables/cross_category_results.csv", index=False)

    print("\n\n" + "=" * 80)
    print("CROSS-CATEGORY RESULTS")
    print("=" * 80)
    display_cols = [c for c in results_df.columns if c != "error"]
    print(results_df[display_cols].to_string(index=False))
    print(f"\nSaved to reports/tables/cross_category_results.csv")


if __name__ == "__main__":
    main()
