"""Temperature scaling and selective prediction analysis."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from src.evaluation.calibration import expected_calibration_error


def _apply_temperature(logits, temperature):
    """Scale logits by temperature and return calibrated probabilities."""
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def _probs_to_logits(probs):
    """Convert predicted probabilities back to logits."""
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return np.log(probs)


def find_optimal_temperature(y_true, y_prob_positive, n_bins=10):
    """Find the temperature that minimizes ECE on a validation set.

    y_prob_positive: P(class=1) from the model.
    """
    probs_2d = np.column_stack([1 - y_prob_positive, y_prob_positive])
    logits = _probs_to_logits(probs_2d)

    def objective(T):
        cal_probs = _apply_temperature(logits, T)
        ece, _, _, _ = expected_calibration_error(y_true, cal_probs[:, 1], n_bins)
        return ece

    result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    return result.x


def apply_temperature_scaling(y_prob_positive, temperature):
    """Apply a learned temperature to probabilities."""
    probs_2d = np.column_stack([1 - y_prob_positive, y_prob_positive])
    logits = _probs_to_logits(probs_2d)
    calibrated = _apply_temperature(logits, temperature)
    return calibrated[:, 1]


def compute_group_conditional_calibration(df, n_bins=10):
    """Compute ECE separately for agreement and disagreement cases per model."""
    model_specs = [
        ("text_only", "text_prob_positive"),
        ("metadata_only", "meta_prob_positive"),
        ("multimodal", "mm_prob_positive"),
    ]

    rows = []
    for group_name in ["agreement", "disagreement"]:
        if group_name == "agreement":
            mask = df["agreement_status"] == "agreement"
        else:
            mask = df["agreement_status"] == "disagreement"
        subset = df[mask]
        if len(subset) == 0:
            continue

        y_true = subset["label"].values
        for model_name, prob_col in model_specs:
            y_prob = subset[prob_col].values
            ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
                y_true, y_prob, n_bins
            )
            rows.append({
                "model": model_name,
                "group": group_name,
                "n": len(subset),
                "ece": ece,
                "mean_confidence": np.max(
                    np.column_stack([y_prob, 1 - y_prob]), axis=1
                ).mean(),
            })

    return pd.DataFrame(rows)


def temperature_scaling_experiment(df, val_df, n_bins=10):
    """Run temperature scaling: learn T on val, evaluate on test.

    Returns a DataFrame comparing ECE before/after temperature scaling
    for agreement vs disagreement subsets.
    """
    model_specs = [
        ("text_only", "text_prob_positive"),
        ("metadata_only", "meta_prob_positive"),
        ("multimodal", "mm_prob_positive"),
    ]

    rows = []
    for model_name, prob_col in model_specs:
        if prob_col not in val_df.columns or prob_col not in df.columns:
            continue

        # Learn temperature on validation set
        T = find_optimal_temperature(
            val_df["label"].values, val_df[prob_col].values, n_bins
        )

        # Evaluate on test set subsets
        for group_name in ["overall", "agreement", "disagreement"]:
            if group_name == "overall":
                subset = df
            elif group_name == "agreement":
                subset = df[df["agreement_status"] == "agreement"]
            else:
                subset = df[df["agreement_status"] == "disagreement"]

            if len(subset) == 0:
                continue

            y_true = subset["label"].values
            y_prob_orig = subset[prob_col].values
            y_prob_cal = apply_temperature_scaling(y_prob_orig, T)

            ece_orig, _, _, _ = expected_calibration_error(y_true, y_prob_orig, n_bins)
            ece_cal, _, _, _ = expected_calibration_error(y_true, y_prob_cal, n_bins)

            rows.append({
                "model": model_name,
                "group": group_name,
                "temperature": T,
                "ece_before": ece_orig,
                "ece_after": ece_cal,
                "ece_improvement": ece_orig - ece_cal,
                "n": len(subset),
            })

    return pd.DataFrame(rows)


def selective_prediction_curve(y_true, y_pred, confidence, n_thresholds=20):
    """Compute accuracy vs coverage at various confidence thresholds.

    Returns arrays of (threshold, accuracy, coverage) tuples.
    """
    thresholds = np.linspace(0.5, 0.99, n_thresholds)
    results = []

    for t in thresholds:
        mask = confidence >= t
        coverage = mask.mean()
        if coverage == 0:
            acc = np.nan
        else:
            acc = (y_true[mask] == y_pred[mask]).mean()
        results.append({"threshold": t, "accuracy": acc, "coverage": coverage})

    return pd.DataFrame(results)


def compute_selective_prediction(df):
    """Compute selective prediction curves for all models, split by group."""
    model_specs = [
        ("text_only", "text_pred", "text_confidence"),
        ("metadata_only", "meta_pred", "meta_confidence"),
        ("multimodal", "mm_pred", "mm_confidence"),
    ]

    all_curves = []
    for model_name, pred_col, conf_col in model_specs:
        for group_name in ["overall", "agreement", "disagreement"]:
            if group_name == "overall":
                subset = df
            elif group_name == "agreement":
                subset = df[df["agreement_status"] == "agreement"]
            else:
                subset = df[df["agreement_status"] == "disagreement"]

            if len(subset) == 0:
                continue

            curve = selective_prediction_curve(
                subset["label"].values,
                subset[pred_col].values,
                subset[conf_col].values,
            )
            curve["model"] = model_name
            curve["group"] = group_name
            all_curves.append(curve)

    return pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
