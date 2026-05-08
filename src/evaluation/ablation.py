"""Ablation studies: feature importance, modality ablation, fusion comparison."""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score


def extract_metadata_feature_importance(model_dir="models/metadata_only"):
    """Extract feature importances from the metadata XGBoost model."""
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    feat_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.zeros(len(feat_cols))

    df = pd.DataFrame({
        "feature": feat_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return df


def compute_shap_values(model_dir="models/metadata_only", X_sample=None, feature_cols=None):
    """Compute SHAP values for the metadata model.

    Returns a DataFrame of mean absolute SHAP values per feature.
    Falls back to feature_importances_ if SHAP is unavailable.
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    if feature_cols is None:
        feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))

    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        if X_sample is not None:
            shap_values = explainer.shap_values(X_sample)
        else:
            return extract_metadata_feature_importance(model_dir)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        return df

    except ImportError:
        print("SHAP not available, using feature_importances_ instead.")
        return extract_metadata_feature_importance(model_dir)


def modality_ablation(df, predict_fn, cfg, text_dim):
    """Zero out text or metadata features and measure accuracy drop.

    Parameters
    ----------
    df : DataFrame with 'label' column
    predict_fn : callable(features_array) -> predictions
    cfg : config dict
    text_dim : int, number of text feature dimensions
    """
    y_true = df["label"].values

    from src.models.train_multimodal_model import (
        _get_text_features, combine_features,
    )
    from src.features.build_metadata_features import (
        get_available_metadata, load_scaler, transform_metadata,
    )

    X_text = _get_text_features(df, cfg, "models/multimodal")
    scaler = load_scaler("models/metadata_only/scaler.joblib")
    feat_cols = get_available_metadata(df)
    X_meta, _ = transform_metadata(df, scaler, feat_cols)

    X_full = combine_features(X_text, X_meta)
    X_no_text = combine_features(np.zeros_like(X_text), X_meta)
    X_no_meta = combine_features(X_text, np.zeros_like(X_meta))

    results = {}
    for name, features in [("full", X_full), ("no_text", X_no_text), ("no_meta", X_no_meta)]:
        preds = predict_fn(features)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        results[name] = {"accuracy": acc, "f1": f1}

    results["text_ablation_drop"] = results["full"]["accuracy"] - results["no_text"]["accuracy"]
    results["meta_ablation_drop"] = results["full"]["accuracy"] - results["no_meta"]["accuracy"]

    return results


def ablation_by_group(df, predict_fn, cfg, text_dim):
    """Run modality ablation separately for agreement and disagreement subsets."""
    rows = []
    for group_name in ["overall", "agreement", "disagreement"]:
        if group_name == "overall":
            subset = df
        elif group_name == "agreement":
            subset = df[df["agreement_status"] == "agreement"]
        else:
            subset = df[df["agreement_status"] == "disagreement"]

        if len(subset) < 10:
            continue

        try:
            abl = modality_ablation(subset.reset_index(drop=True), predict_fn, cfg, text_dim)
            rows.append({
                "group": group_name,
                "n": len(subset),
                "accuracy_full": abl["full"]["accuracy"],
                "accuracy_no_text": abl["no_text"]["accuracy"],
                "accuracy_no_meta": abl["no_meta"]["accuracy"],
                "text_drop": abl["text_ablation_drop"],
                "meta_drop": abl["meta_ablation_drop"],
            })
        except Exception as e:
            print(f"Ablation failed for {group_name}: {e}")

    return pd.DataFrame(rows)
