"""Statistical significance testing: bootstrap CIs and McNemar's test."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.evaluation.calibration import expected_calibration_error


def bootstrap_metric(y_true, y_pred, y_prob, metric_fn, n_resamples=1000,
                     confidence_level=0.95, seed=42):
    """Compute a metric with bootstrap confidence interval.

    Parameters
    ----------
    metric_fn : callable(y_true, y_pred, y_prob) -> float
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = np.empty(n_resamples)

    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        try:
            scores[i] = metric_fn(y_true[idx], y_pred[idx], y_prob[idx])
        except (ValueError, ZeroDivisionError):
            scores[i] = np.nan

    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return np.nan, np.nan, np.nan

    alpha = 1 - confidence_level
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    point = metric_fn(y_true, y_pred, y_prob)

    return point, lo, hi


def _acc_fn(yt, yp, _):
    return accuracy_score(yt, yp)


def _f1_fn(yt, yp, _):
    return f1_score(yt, yp, zero_division=0)


def _auroc_fn(yt, _, yprob):
    if len(np.unique(yt)) < 2:
        return np.nan
    return roc_auc_score(yt, yprob)


def _ece_fn(yt, _, yprob):
    ece, _, _, _ = expected_calibration_error(yt, yprob, n_bins=10)
    return ece


METRIC_FNS = {
    "accuracy": _acc_fn,
    "f1": _f1_fn,
    "auroc": _auroc_fn,
    "ece": _ece_fn,
}


def bootstrap_all_metrics(y_true, y_pred, y_prob, n_resamples=1000,
                          confidence_level=0.95, seed=42):
    """Compute all key metrics with bootstrap CIs."""
    results = {}
    for name, fn in METRIC_FNS.items():
        point, lo, hi = bootstrap_metric(
            y_true, y_pred, y_prob, fn,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        results[name] = point
        results[f"{name}_ci_lo"] = lo
        results[f"{name}_ci_hi"] = hi

    return results


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test comparing two classifiers on the same test set.

    Returns chi-squared statistic and two-sided p-value.
    """
    from scipy.stats import chi2

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int((correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a & correct_b).sum())  # A wrong, B right

    if b + c == 0:
        return 0.0, 1.0

    # continuity-corrected McNemar
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return float(chi2_stat), float(p_value)


def run_all_statistical_tests(df, cfg):
    """Run bootstrap CIs and McNemar's tests for all model pairs."""
    n_boot = cfg["evaluation"].get("bootstrap_n_resamples", 1000)
    conf_level = cfg["evaluation"].get("bootstrap_confidence_level", 0.95)
    seed = cfg["project"]["seed"]
    y_true = df["label"].values

    model_specs = [
        ("text_only", "text_pred", "text_prob_positive"),
        ("metadata_only", "meta_pred", "meta_prob_positive"),
        ("multimodal", "mm_pred", "mm_prob_positive"),
    ]

    # Bootstrap CIs per model (overall + disagreement subset)
    ci_rows = []
    for model_name, pred_col, prob_col in model_specs:
        for subset_name, mask in [("overall", np.ones(len(df), dtype=bool)),
                                  ("disagreement", df["agreement_status"].values == "disagreement")]:
            yt = y_true[mask]
            yp = df[pred_col].values[mask]
            yprob = df[prob_col].values[mask]
            if len(yt) == 0:
                continue
            metrics = bootstrap_all_metrics(yt, yp, yprob, n_boot, conf_level, seed)
            metrics["model"] = model_name
            metrics["subset"] = subset_name
            metrics["n"] = int(mask.sum())
            ci_rows.append(metrics)

    ci_df = pd.DataFrame(ci_rows)

    # McNemar's tests between model pairs on disagreement cases
    mcnemar_rows = []
    disagree_mask = df["agreement_status"].values == "disagreement"
    yt_d = y_true[disagree_mask]

    pairs = [
        ("text_only_vs_multimodal", "text_pred", "mm_pred"),
        ("metadata_only_vs_multimodal", "meta_pred", "mm_pred"),
        ("text_only_vs_metadata_only", "text_pred", "meta_pred"),
    ]
    for pair_name, col_a, col_b in pairs:
        for subset_name, mask in [("overall", np.ones(len(df), dtype=bool)),
                                  ("disagreement", disagree_mask)]:
            yt = y_true[mask]
            ya = df[col_a].values[mask]
            yb = df[col_b].values[mask]
            if len(yt) == 0:
                continue
            chi2_stat, p_val = mcnemar_test(yt, ya, yb)
            mcnemar_rows.append({
                "comparison": pair_name,
                "subset": subset_name,
                "n": int(mask.sum()),
                "chi2": chi2_stat,
                "p_value": p_val,
                "significant": p_val < cfg["evaluation"].get("mcnemar_alpha", 0.05),
            })

    mcnemar_df = pd.DataFrame(mcnemar_rows)

    return ci_df, mcnemar_df


def format_metric_with_ci(row, metric_name):
    """Format a metric as 'value (CI: lo-hi)' for paper tables."""
    val = row[metric_name]
    lo = row.get(f"{metric_name}_ci_lo", np.nan)
    hi = row.get(f"{metric_name}_ci_hi", np.nan)
    if np.isnan(lo) or np.isnan(hi):
        return f"{val:.3f}"
    return f"{val:.3f} ({lo:.3f}-{hi:.3f})"
