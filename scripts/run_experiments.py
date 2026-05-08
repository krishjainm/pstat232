"""Run the full pipeline with multiple seeds and report mean +/- std."""

import sys
sys.path.insert(0, ".")

import os
import copy
import yaml
import numpy as np
import pandas as pd
import logging

from src.data.download_data import main as download_main
from src.data.preprocess import preprocess_pipeline
from src.data.create_splits import create_splits
from src.features.define_disagreement import add_disagreement_labels
from src.models.train_metadata_model import train_metadata_model
from src.models.train_multimodal_model import train_multimodal
from src.models.predict import get_all_predictions
from src.evaluation.metrics import compute_all_model_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment_log.txt", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_single_seed(seed, base_cfg):
    """Run the full pipeline with a specific seed."""
    cfg = copy.deepcopy(base_cfg)
    cfg["project"]["seed"] = seed
    logger.info(f"Running pipeline with seed={seed}")

    try:
        from src.data.create_splits import create_splits as _cs
        train_df = pd.read_parquet("data/processed/train.parquet")
        val_df = pd.read_parquet("data/processed/val.parquet")
        test_df = pd.read_parquet("data/processed/test.parquet")

        np.random.seed(seed)

        from src.models.train_text_model import train_sbert_logreg, train_tfidf_logreg
        model_type = cfg["models"]["text_model"]["type"]
        if model_type == "sentence_transformer":
            train_sbert_logreg(train_df, val_df, cfg)
        else:
            train_tfidf_logreg(train_df, val_df)

        train_metadata_model(train_df, val_df, cfg)
        train_multimodal(train_df, val_df, cfg)

        test_df = get_all_predictions(test_df, "test", cfg)
        metrics = compute_all_model_metrics(test_df)

        result = {"seed": seed}
        for model_name, m in metrics.items():
            for metric_name, value in m.items():
                result[f"{model_name}_{metric_name}"] = value

        disagree = test_df[test_df["agreement_status"] == "disagreement"]
        if len(disagree) > 0:
            for mn, pc in [("text_only", "text_pred"), ("metadata_only", "meta_pred"), ("multimodal", "mm_pred")]:
                result[f"{mn}_disagree_acc"] = (disagree["label"] == disagree[pc]).mean()

        logger.info(f"Seed {seed} complete. Multimodal accuracy: {result.get('multimodal_accuracy', 'N/A')}")
        return result

    except Exception as e:
        logger.error(f"Seed {seed} failed: {e}")
        return {"seed": seed, "error": str(e)}


def main():
    cfg = load_config()
    seeds = cfg["project"].get("seeds", [42, 123, 456])

    logger.info(f"Running experiments with {len(seeds)} seeds: {seeds}")

    all_results = []
    for seed in seeds:
        result = run_single_seed(seed, cfg)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    os.makedirs("reports/tables", exist_ok=True)
    results_df.to_csv("reports/tables/multi_seed_results.csv", index=False)

    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "seed"]

    summary_rows = []
    for col in numeric_cols:
        vals = results_df[col].dropna()
        summary_rows.append({
            "metric": col,
            "mean": vals.mean(),
            "std": vals.std(),
            "min": vals.min(),
            "max": vals.max(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("reports/tables/multi_seed_summary.csv", index=False)

    logger.info("\nMulti-seed summary:")
    logger.info(summary_df.to_string(index=False))
    print("\nResults saved to reports/tables/multi_seed_results.csv")
    print("Summary saved to reports/tables/multi_seed_summary.csv")


if __name__ == "__main__":
    main()
