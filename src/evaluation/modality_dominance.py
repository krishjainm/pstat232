"""Modality dominance analysis: which modality does the multimodal model follow?

Includes label-based dominance, probability-based dominance, and
true-conflict-only dominance (where unimodal models actually disagree).
"""

import numpy as np
import pandas as pd


def classify_dominance(row):
    """Classify multimodal behavior for a single prediction (label-based)."""
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
    """Compute label-based modality dominance for disagreement cases."""
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


# ---------------------------------------------------------------------------
# Upgrade 3a: Probability-based dominance
# ---------------------------------------------------------------------------

def compute_probability_dominance(df):
    """Compute probability-based dominance using predicted probabilities.

    For each sample, compute how close the multimodal probability is to each
    unimodal model's probability. A dominance_ratio near 0 means multimodal
    tracks text; near 1 means it tracks metadata.
    """
    required = ["mm_prob_positive", "text_prob_positive", "meta_prob_positive"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    result = df.copy()
    text_dist = np.abs(result["mm_prob_positive"].values - result["text_prob_positive"].values)
    meta_dist = np.abs(result["mm_prob_positive"].values - result["meta_prob_positive"].values)

    denom = text_dist + meta_dist + 1e-8
    result["dominance_ratio"] = text_dist / denom
    result["text_distance"] = text_dist
    result["meta_distance"] = meta_dist

    disagree = result[result["agreement_status"] == "disagreement"]
    if len(disagree) > 0:
        mean_ratio = disagree["dominance_ratio"].mean()
        print(f"Probability dominance (disagreement cases): mean ratio = {mean_ratio:.3f}")
        print(f"  (0 = follows text, 1 = follows metadata)")
        text_leaning = (disagree["dominance_ratio"] < 0.5).sum()
        meta_leaning = (disagree["dominance_ratio"] >= 0.5).sum()
        print(f"  Text-leaning: {text_leaning} ({100*text_leaning/len(disagree):.1f}%)")
        print(f"  Meta-leaning: {meta_leaning} ({100*meta_leaning/len(disagree):.1f}%)")

    summary = {
        "overall_mean_ratio": result["dominance_ratio"].mean(),
        "disagree_mean_ratio": disagree["dominance_ratio"].mean() if len(disagree) > 0 else np.nan,
        "disagree_median_ratio": disagree["dominance_ratio"].median() if len(disagree) > 0 else np.nan,
        "disagree_text_leaning_pct": 100 * (disagree["dominance_ratio"] < 0.5).mean() if len(disagree) > 0 else np.nan,
    }

    return result, summary


# ---------------------------------------------------------------------------
# Upgrade 3b: True-conflict-only dominance
# ---------------------------------------------------------------------------

def compute_conflict_dominance(df):
    """Dominance analysis restricted to cases where text_pred != meta_pred.

    These are the cases where unimodal models actually disagree, providing
    the clearest signal for which modality the fusion model trusts.
    """
    conflict = df[df["text_pred"] != df["meta_pred"]].copy()

    if len(conflict) == 0:
        print("No true-conflict cases (text_pred == meta_pred everywhere).")
        return pd.DataFrame(), {}

    total = len(conflict)
    follows_text = (conflict["mm_pred"] == conflict["text_pred"]).sum()
    follows_meta = (conflict["mm_pred"] == conflict["meta_pred"]).sum()
    follows_neither = total - follows_text - follows_meta

    results = pd.DataFrame([
        {"category": "follows_text", "count": int(follows_text),
         "percentage": 100 * follows_text / total},
        {"category": "follows_metadata", "count": int(follows_meta),
         "percentage": 100 * follows_meta / total},
        {"category": "follows_neither", "count": int(follows_neither),
         "percentage": 100 * follows_neither / total},
    ])

    correct_when_text = (
        (conflict["mm_pred"] == conflict["text_pred"]) &
        (conflict["mm_pred"] == conflict["label"])
    ).sum()
    correct_when_meta = (
        (conflict["mm_pred"] == conflict["meta_pred"]) &
        (conflict["mm_pred"] == conflict["label"])
    ).sum()

    stats = {
        "n_conflict_cases": total,
        "follows_text_pct": 100 * follows_text / total,
        "follows_meta_pct": 100 * follows_meta / total,
        "accuracy_when_follows_text": correct_when_text / max(follows_text, 1),
        "accuracy_when_follows_meta": correct_when_meta / max(follows_meta, 1),
    }

    print(f"True-conflict dominance ({total} cases where text_pred != meta_pred):")
    print(results.to_string(index=False))

    return results, stats
