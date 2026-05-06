"""Group-level evaluation: metrics by agreement/disagreement group."""

import pandas as pd
import numpy as np
from src.evaluation.metrics import compute_classification_metrics


GROUPS = ["agreement", "weak_disagreement", "medium_disagreement", "strong_disagreement"]
MODEL_SPECS = [
    ("text_only", "text_pred", "text_prob_positive"),
    ("metadata_only", "meta_pred", "meta_prob_positive"),
    ("multimodal", "mm_pred", "mm_prob_positive"),
]


def compute_group_metrics(df):
    """Compute classification metrics per model per disagreement group."""
    rows = []

    for group in ["overall"] + GROUPS + ["all_disagreement"]:
        if group == "overall":
            subset = df
        elif group == "all_disagreement":
            subset = df[df["agreement_status"] == "disagreement"]
        else:
            subset = df[df["disagreement_group"] == group]

        if len(subset) == 0:
            continue

        for model_name, pred_col, prob_col in MODEL_SPECS:
            metrics = compute_classification_metrics(
                subset["label"].values,
                subset[pred_col].values,
                subset[prob_col].values,
            )
            metrics["model"] = model_name
            metrics["group"] = group
            metrics["n"] = len(subset)
            rows.append(metrics)

    return pd.DataFrame(rows)
