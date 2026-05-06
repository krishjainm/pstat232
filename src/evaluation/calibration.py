"""Calibration metrics: ECE, Brier score, reliability data."""

import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_in_bin = mask.sum()

        if n_in_bin == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
            continue

        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (n_in_bin / total) * abs(avg_acc - avg_conf)

        bin_accs.append(avg_acc)
        bin_confs.append(avg_conf)
        bin_counts.append(n_in_bin)

    return ece, np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def confidence_stats(y_true, y_pred, confidence, high_conf_threshold=0.90):
    """Compute confidence statistics."""
    correct = y_true == y_pred
    incorrect = ~correct

    stats = {
        "mean_confidence_overall": confidence.mean(),
        "mean_confidence_correct": confidence[correct].mean() if correct.sum() > 0 else np.nan,
        "mean_confidence_incorrect": confidence[incorrect].mean() if incorrect.sum() > 0 else np.nan,
    }

    high_conf = confidence >= high_conf_threshold
    high_conf_errors = high_conf & incorrect
    stats["high_confidence_error_count"] = high_conf_errors.sum()
    stats["high_confidence_error_rate"] = (
        high_conf_errors.sum() / high_conf.sum() if high_conf.sum() > 0 else 0.0
    )
    stats["n_high_confidence"] = high_conf.sum()

    return stats


def compute_calibration_metrics(df, model_name, pred_col, prob_col, conf_col, n_bins=10, threshold=0.90):
    """Full calibration analysis for one model."""
    y_true = df["label"].values
    y_pred = df[pred_col].values
    y_prob = df[prob_col].values
    conf = df[conf_col].values

    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(y_true, y_prob, n_bins)
    stats = confidence_stats(y_true, y_pred, conf, threshold)
    stats["ece"] = ece

    return stats, bin_accs, bin_confs, bin_counts
