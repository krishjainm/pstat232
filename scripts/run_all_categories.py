"""Run the full pipeline on multiple Amazon review categories for cross-category comparison."""

import sys
sys.path.insert(0, ".")

import os
import copy
import yaml
import pandas as pd

from src.data.download_data import main as download_main
from src.data.preprocess import preprocess_pipeline
from src.data.create_splits import create_splits
from src.features.define_disagreement import add_disagreement_labels
from src.models.train_metadata_model import train_metadata_model
from src.models.train_multimodal_model import train_multimodal
from src.models.predict import get_all_predictions
from src.evaluation.metrics import compute_all_model_metrics
from src.evaluation.group_analysis import compute_group_metrics
from src.evaluation.modality_dominance import (
    compute_modality_dominance, compute_conflict_dominance,
)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_single_category(category, base_cfg):
    """Run the pipeline for a single category, returning summary metrics."""
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["category"] = category

    cat_dir = f"data/categories/{category}"
    os.makedirs(cat_dir, exist_ok=True)

    tmp_config_path = os.path.join(cat_dir, "config.yaml")
    with open(tmp_config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"\n{'='*60}")
    print(f"Processing category: {category}")
    print(f"{'='*60}")

    try:
        download_main(config_path=tmp_config_path)
        preprocess_pipeline(config_path=tmp_config_path)
        create_splits(config_path=tmp_config_path)

        train_df = pd.read_parquet("data/processed/train.parquet")
        val_df = pd.read_parquet("data/processed/val.parquet")
        test_df = pd.read_parquet("data/processed/test.parquet")

        try:
            train_df = add_disagreement_labels(train_df, cfg, method="transformer")
            val_df = add_disagreement_labels(val_df, cfg, method="transformer")
            test_df = add_disagreement_labels(test_df, cfg, method="transformer")
        except Exception:
            train_df = add_disagreement_labels(train_df, cfg, method="tfidf", train_df=train_df)
            val_df = add_disagreement_labels(val_df, cfg, method="tfidf", train_df=train_df)
            test_df = add_disagreement_labels(test_df, cfg, method="tfidf", train_df=train_df)

        train_df.to_parquet("data/processed/train.parquet", index=False)
        val_df.to_parquet("data/processed/val.parquet", index=False)
        test_df.to_parquet("data/processed/test.parquet", index=False)

        from scripts import _run_training as _rt
        _rt.run_all_training(cfg, train_df, val_df)

        test_df = get_all_predictions(test_df, "test", cfg)
        main_metrics = compute_all_model_metrics(test_df)
        group_df = compute_group_metrics(test_df)

        disagree_rate = (test_df["agreement_status"] == "disagreement").mean()

        summary = {
            "category": category,
            "n_samples": len(test_df),
            "disagreement_rate": disagree_rate,
        }
        for model_name, metrics in main_metrics.items():
            summary[f"{model_name}_accuracy"] = metrics["accuracy"]
            summary[f"{model_name}_f1"] = metrics["f1"]

        disagree_subset = test_df[test_df["agreement_status"] == "disagreement"]
        if len(disagree_subset) > 0:
            for mn, pc in [("text_only", "text_pred"), ("metadata_only", "meta_pred"), ("multimodal", "mm_pred")]:
                summary[f"{mn}_disagree_accuracy"] = (
                    disagree_subset["label"] == disagree_subset[pc]
                ).mean()

        return summary

    except Exception as e:
        print(f"Failed for category {category}: {e}")
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
    print("\n\nCross-category results:")
    print(results_df.to_string(index=False))
    print("\nSaved to reports/tables/cross_category_results.csv")


if __name__ == "__main__":
    main()
