"""Modality dominance analysis: which modality does the multimodal model follow?"""

import numpy as np
import pandas as pd


def classify_dominance(row):
    """Classify multimodal behavior for a single prediction."""
    text_pred = row["text_pred"]
    meta_pred = row["meta_pred"]
    mm_pred = row["mm_pred"]

    if text_pred == meta_pred:
        return "both_agree"
    elif mm_pred == text_pred and mm_pred != meta_pred:
        return "text_dominant"
    elif mm_pred == meta_pred and mm_pred != text_pred:
        return "metadata_dominant"
    else:
        return "neither"


def compute_modality_dominance(df):
    """Compute modality dominance for disagreement cases."""
    disagree_df = df[df["agreement_status"] == "disagreement"].copy()

    if len(disagree_df) == 0:
        print("No disagreement cases found.")
        return pd.DataFrame()

    disagree_df["dominance"] = disagree_df.apply(classify_dominance, axis=1)

    counts = disagree_df["dominance"].value_counts()
    total = len(disagree_df)

    results = []
    for cat in ["text_dominant", "metadata_dominant", "both_agree", "neither"]:
        n = counts.get(cat, 0)
        results.append({
            "category": cat,
            "count": n,
            "percentage": 100 * n / total if total > 0 else 0,
        })

    results_df = pd.DataFrame(results)
    print("Modality dominance (disagreement cases):")
    print(results_df.to_string(index=False))
    return results_df


def compute_dominance_by_severity(df):
    """Break down modality dominance by disagreement severity."""
    disagree_df = df[df["agreement_status"] == "disagreement"].copy()
    if len(disagree_df) == 0:
        return pd.DataFrame()

    disagree_df["dominance"] = disagree_df.apply(classify_dominance, axis=1)

    rows = []
    for group in ["weak_disagreement", "medium_disagreement", "strong_disagreement"]:
        subset = disagree_df[disagree_df["disagreement_group"] == group]
        total = len(subset)
        if total == 0:
            continue
        for cat in ["text_dominant", "metadata_dominant", "both_agree", "neither"]:
            n = (subset["dominance"] == cat).sum()
            rows.append({
                "severity": group,
                "dominance": cat,
                "count": n,
                "percentage": 100 * n / total,
            })

    return pd.DataFrame(rows)
