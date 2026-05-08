"""Qualitative error analysis: error taxonomy and case study generation."""

import numpy as np
import pandas as pd
import re


ERROR_CATEGORIES = {
    "sarcasm_irony": [
        r"(?i)\blove\b.*(?:broke|terrible|worst|never|don'?t|didn'?t|waste)",
        r"(?i)(?:great|amazing|wonderful).*(?:not|broke|fail|return|refund|junk)",
        r"(?i)yeah.*right",
        r"(?i)(?:thanks|thank).*(?:nothing|waste)",
    ],
    "mixed_sentiment": [
        r"(?i)(?:good|great|nice|love).*(?:but|however|although|except|unfortunately)",
        r"(?i)(?:but|however|although|except).*(?:good|great|nice|love)",
        r"(?i)\b(?:pro|pros)\b.*\b(?:con|cons)\b",
    ],
    "rating_misuse": [
        r"(?i)(?:shipping|delivery|package|arrived|shipped|ups|fedex|usps).*(?:late|slow|damaged|wrong|broken)",
        r"(?i)(?:seller|vendor|customer service|return|refund).*(?:bad|terrible|awful|rude|unhelpful)",
    ],
    "ambiguous": [
        r"(?i)^(?:ok|okay|fine|decent|average|alright|meh|so-so|not bad)[\.\,\!\s]*$",
        r"(?i)(?:it'?s? (?:ok|okay|fine|alright|decent))",
    ],
    "short_review": [],
}


def classify_error_type(text, label, pred_label):
    """Heuristically classify an error into a category.

    This is an approximate taxonomy — manual review is still recommended.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "empty_text"

    if len(text.split()) <= 5:
        return "short_review"

    for category, patterns in ERROR_CATEGORIES.items():
        if category == "short_review":
            continue
        for pattern in patterns:
            if re.search(pattern, text):
                return category

    return "other"


def build_error_taxonomy(df, max_errors=200):
    """Classify errors for the multimodal model and return a taxonomy table.

    Parameters
    ----------
    df : DataFrame with columns: review_text, label, mm_pred, mm_confidence,
         text_pred, meta_pred, agreement_status, disagreement_group
    """
    errors = df[df["mm_pred"] != df["label"]].copy()
    if len(errors) > max_errors:
        errors = errors.sample(n=max_errors, random_state=42)

    errors["error_type"] = errors.apply(
        lambda row: classify_error_type(row["review_text"], row["label"], row["mm_pred"]),
        axis=1,
    )

    taxonomy = errors.groupby("error_type").agg(
        count=("error_type", "size"),
        mean_confidence=("mm_confidence", "mean"),
        pct_high_conf=("mm_confidence", lambda x: (x >= 0.9).mean() * 100),
        pct_disagreement=("agreement_status", lambda x: (x == "disagreement").mean() * 100),
    ).reset_index()
    taxonomy["percentage"] = 100 * taxonomy["count"] / taxonomy["count"].sum()
    taxonomy = taxonomy.sort_values("count", ascending=False)

    return taxonomy, errors


def generate_case_studies(df, n_cases=5):
    """Select diverse, illustrative error cases for the paper.

    Selects high-confidence errors from different categories and disagreement groups.
    """
    errors = df[
        (df["mm_pred"] != df["label"]) &
        (df["mm_confidence"] >= 0.85)
    ].copy()

    errors["error_type"] = errors.apply(
        lambda row: classify_error_type(row["review_text"], row["label"], row["mm_pred"]),
        axis=1,
    )

    cases = []
    seen_types = set()

    for _, row in errors.sort_values("mm_confidence", ascending=False).iterrows():
        if len(cases) >= n_cases:
            break
        etype = row["error_type"]
        if etype not in seen_types or len(cases) < 3:
            seen_types.add(etype)
            text_preview = row["review_text"][:300]
            if len(row["review_text"]) > 300:
                text_preview += "..."
            cases.append({
                "text_preview": text_preview,
                "true_label": "positive" if row["label"] == 1 else "negative",
                "mm_prediction": "positive" if row["mm_pred"] == 1 else "negative",
                "mm_confidence": row["mm_confidence"],
                "text_pred": "positive" if row["text_pred"] == 1 else "negative",
                "meta_pred": "positive" if row["meta_pred"] == 1 else "negative",
                "agreement_status": row["agreement_status"],
                "disagreement_group": row.get("disagreement_group", ""),
                "error_type": etype,
            })

    return pd.DataFrame(cases)


def per_model_error_taxonomy(df, max_errors=200):
    """Build error taxonomy for each model."""
    model_specs = [
        ("text_only", "text_pred", "text_confidence"),
        ("metadata_only", "meta_pred", "meta_confidence"),
        ("multimodal", "mm_pred", "mm_confidence"),
    ]

    all_taxonomies = []
    for model_name, pred_col, conf_col in model_specs:
        errors = df[df[pred_col] != df["label"]].copy()
        if len(errors) > max_errors:
            errors = errors.sample(n=max_errors, random_state=42)

        errors["error_type"] = errors.apply(
            lambda row: classify_error_type(row["review_text"], row["label"], row[pred_col]),
            axis=1,
        )

        taxonomy = errors["error_type"].value_counts().reset_index()
        taxonomy.columns = ["error_type", "count"]
        taxonomy["percentage"] = 100 * taxonomy["count"] / taxonomy["count"].sum()
        taxonomy["model"] = model_name
        all_taxonomies.append(taxonomy)

    return pd.concat(all_taxonomies, ignore_index=True)
