"""Script 10: Train disagreement-aware model and selective abstention baseline.

Compares:
  1. Standard early fusion (already trained)
  2. Disagreement-aware early fusion (upweighted loss on disagreement samples)
  3. Selective abstention (fall back to metadata on detected disagreement)
"""

import sys
sys.path.insert(0, ".")

import os
import copy
import numpy as np
import pandas as pd

from src.models.train_multimodal_model import train_multimodal, predict_multimodal
from src.models.predict import get_all_predictions, load_config
from src.evaluation.metrics import compute_classification_metrics


def selective_abstention(test_df, conf_gap_threshold=0.15):
    """When text and metadata strongly disagree, fall back to metadata prediction.

    Detects likely disagreement using the gap between text and metadata confidence,
    then substitutes metadata predictions for those samples.
    """
    text_prob = test_df["text_prob_positive"].values
    meta_prob = test_df["meta_prob_positive"].values

    text_pred = test_df["text_pred"].values
    meta_pred = test_df["meta_pred"].values
    mm_pred = test_df["mm_pred"].values.copy()
    mm_prob = test_df["mm_prob_positive"].values.copy()

    pred_disagree = text_pred != meta_pred
    conf_gap = np.abs(text_prob - meta_prob)
    flagged = pred_disagree & (conf_gap > conf_gap_threshold)

    mm_pred[flagged] = meta_pred[flagged]
    mm_prob[flagged] = meta_prob[flagged]

    n_flagged = flagged.sum()
    return mm_pred, mm_prob, n_flagged, flagged


def evaluate_model(y_true, y_pred, y_prob, disagree_mask, label):
    """Compute overall and disagreement metrics."""
    overall = compute_classification_metrics(y_true, y_pred, y_prob)
    disagree = compute_classification_metrics(
        y_true[disagree_mask], y_pred[disagree_mask], y_prob[disagree_mask]
    )
    agree = compute_classification_metrics(
        y_true[~disagree_mask], y_pred[~disagree_mask], y_prob[~disagree_mask]
    )
    return {
        "method": label,
        "overall_accuracy": overall["accuracy"],
        "overall_f1": overall["f1"],
        "overall_auroc": overall["auroc"],
        "agree_accuracy": agree["accuracy"],
        "agree_f1": agree["f1"],
        "disagree_accuracy": disagree["accuracy"],
        "disagree_f1": disagree["f1"],
        "disagree_auroc": disagree["auroc"],
    }


def main():
    cfg = load_config()
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    test_df = get_all_predictions(test_df, "test", cfg)

    y_true = test_df["label"].values
    disagree_mask = test_df["agreement_status"].values == "disagreement"

    results = []

    # 1. Standard fusion (already trained, use existing predictions)
    results.append(evaluate_model(
        y_true, test_df["mm_pred"].values, test_df["mm_prob_positive"].values,
        disagree_mask, "Standard Fusion"
    ))

    # 2. Disagreement-aware fusion (retrain with upweighted loss)
    for weight in [3.0, 5.0]:
        save_dir = f"models/multimodal_disagree_w{int(weight)}"
        train_multimodal(
            train_df, val_df, cfg, save_dir=save_dir,
            disagree_aware=True, disagree_weight=weight,
        )
        preds, prob_pos, conf = predict_multimodal(test_df, "test", cfg, save_dir=save_dir)
        results.append(evaluate_model(
            y_true, preds, prob_pos, disagree_mask,
            f"Disagree-Aware (w={int(weight)})"
        ))

    # 3. Selective abstention baselines
    for threshold in [0.10, 0.15, 0.20, 0.30]:
        sa_pred, sa_prob, n_flagged, flagged = selective_abstention(test_df, threshold)
        label = f"Selective Abstention (gap>{threshold:.2f})"
        results.append(evaluate_model(
            y_true, sa_pred, sa_prob, disagree_mask, label
        ))
        n_dis_flagged = (flagged & disagree_mask).sum()
        print(f"{label}: {n_flagged} flagged ({n_dis_flagged} actual disagreement)")

    results_df = pd.DataFrame(results)
    os.makedirs("reports/tables", exist_ok=True)
    results_df.to_csv("reports/tables/disagreement_aware_results.csv", index=False)

    print("\n" + "=" * 80)
    print("DISAGREEMENT-AWARE METHOD COMPARISON")
    print("=" * 80)
    for _, row in results_df.iterrows():
        print(f"\n{row['method']}:")
        print(f"  Overall:      acc={row['overall_accuracy']:.4f}  f1={row['overall_f1']:.4f}")
        print(f"  Agreement:    acc={row['agree_accuracy']:.4f}  f1={row['agree_f1']:.4f}")
        print(f"  Disagreement: acc={row['disagree_accuracy']:.4f}  f1={row['disagree_f1']:.4f}")

    print(f"\nSaved to reports/tables/disagreement_aware_results.csv")


if __name__ == "__main__":
    main()
