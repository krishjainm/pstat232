"""Extended statistical testing for the research upgrade.

Operates ONLY on real, already-generated per-sample predictions
(`data/processed/test_with_predictions.parquet`, the validated All_Beauty
test set) plus a leakage-free half-split for temperature-scaling comparison.

Produces: reports/tables/statistical_tests_extended.csv

No model retraining is performed here; every number is computed from
real predictions. Comparisons that require artifacts we do not have on disk
(e.g. per-sample disagreement-aware predictions, a stored validation split)
are emitted with status="requires_retraining" rather than fabricated.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import accuracy_score, f1_score
from src.evaluation.statistical_tests import mcnemar_test
from src.evaluation.temperature_scaling import (
    find_optimal_temperature, apply_temperature_scaling,
)
from src.evaluation.calibration import expected_calibration_error

PRED_PATH = "data/processed/test_with_predictions.parquet"
OUT_PATH = "reports/tables/statistical_tests_extended.csv"
SEED = 42
N_BOOT = 2000


def _metric(y_true, y_pred, kind):
    if kind == "accuracy":
        return accuracy_score(y_true, y_pred)
    if kind == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    raise ValueError(kind)


def paired_bootstrap_diff(y_true, pred_a, pred_b, kind="accuracy",
                          n_boot=N_BOOT, seed=SEED):
    """Two-sided paired bootstrap on metric(A) - metric(B).

    Returns point difference, 95% CI, and a bootstrap p-value for H0: diff=0.
    """
    rng = np.random.RandomState(seed)
    N = len(y_true)
    if N == 0:
        return np.nan, np.nan, np.nan, np.nan
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, N, size=N)
        diffs[i] = _metric(y_true[idx], pred_a[idx], kind) - _metric(y_true[idx], pred_b[idx], kind)
    point = _metric(y_true, pred_a, kind) - _metric(y_true, pred_b, kind)
    lo = np.percentile(diffs, 2.5)
    hi = np.percentile(diffs, 97.5)
    # two-sided bootstrap p-value
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = min(p, 1.0)
    return point, lo, hi, p


def compare(df, mask, name_a, col_a, name_b, col_b, subset_label):
    y = df["label"].values[mask]
    a = df[col_a].values[mask]
    b = df[col_b].values[mask]
    rows = []
    for kind in ["accuracy", "f1"]:
        diff, lo, hi, p_boot = paired_bootstrap_diff(y, a, b, kind)
        chi2, p_mc = mcnemar_test(y, a, b)
        rows.append({
            "comparison": f"{name_a}_vs_{name_b}",
            "subset": subset_label,
            "n": int(mask.sum()),
            "metric": kind,
            f"value_{name_a}": _metric(y, a, kind),
            f"value_{name_b}": _metric(y, b, kind),
            "diff_a_minus_b": diff,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "paired_bootstrap_p": p_boot,
            "mcnemar_chi2": chi2,
            "mcnemar_p": p_mc,
            "status": "computed",
        })
    return rows


def temp_scaling_half_split(df, seed=SEED, n_bins=10):
    """Leakage-free global vs group-wise temperature scaling.

    We do NOT have a stored validation split for this category, and learning a
    temperature on the test set then evaluating on the same set is leakage.
    Instead we split the test set 50/50 (stratified by agreement_status),
    learn temperatures on split-1, and report ECE on the held-out split-2.
    """
    rng = np.random.RandomState(seed)
    rows = []
    prob_col = "mm_prob_positive"
    df = df.reset_index(drop=True)

    # stratified 50/50 split by agreement_status
    fit_idx, eval_idx = [], []
    for status, g in df.groupby("agreement_status"):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        half = len(idx) // 2
        fit_idx.extend(idx[:half])
        eval_idx.extend(idx[half:])
    fit = df.loc[fit_idx]
    ev = df.loc[eval_idx]

    # ----- global temperature (learned on fit, all groups) -----
    T_global = find_optimal_temperature(fit["label"].values, fit[prob_col].values, n_bins)

    # ----- group-wise temperatures (learned per group on fit) -----
    T_group = {}
    for status in ["agreement", "disagreement"]:
        sub = fit[fit["agreement_status"] == status]
        if len(sub) >= 20:
            T_group[status] = find_optimal_temperature(sub["label"].values, sub[prob_col].values, n_bins)
        else:
            T_group[status] = T_global

    for status in ["overall", "agreement", "disagreement"]:
        if status == "overall":
            sub = ev
        else:
            sub = ev[ev["agreement_status"] == status]
        if len(sub) == 0:
            continue
        yt = sub["label"].values
        p_orig = sub[prob_col].values
        ece_before, _, _, _ = expected_calibration_error(yt, p_orig, n_bins)
        # global
        p_glob = apply_temperature_scaling(p_orig, T_global)
        ece_glob, _, _, _ = expected_calibration_error(yt, p_glob, n_bins)
        # group-wise: choose T by the row's own agreement status
        if status == "overall":
            p_grp = np.empty_like(p_orig)
            for st in ["agreement", "disagreement"]:
                m = (sub["agreement_status"] == st).values
                if m.any():
                    p_grp[m] = apply_temperature_scaling(p_orig[m], T_group[st])
        else:
            p_grp = apply_temperature_scaling(p_orig, T_group[status])
        ece_grp, _, _, _ = expected_calibration_error(yt, p_grp, n_bins)

        rows.append({
            "comparison": "global_vs_groupwise_temperature",
            "subset": status,
            "n": int(len(sub)),
            "metric": "ECE_held_out_split",
            "ece_no_scaling": ece_before,
            "ece_global_T": ece_glob,
            "ece_groupwise_T": ece_grp,
            "T_global": T_global,
            "T_agreement": T_group["agreement"],
            "T_disagreement": T_group["disagreement"],
            "status": "computed_leakage_free_halfsplit",
        })
    return rows


def main():
    df = pd.read_parquet(PRED_PATH)
    print(f"Loaded {len(df)} rows from {PRED_PATH}")
    assert len(df) == 3102, "Expected the All_Beauty test predictions (3102 rows)."

    strong = (df["disagreement_group"] == "strong_disagreement").values
    disagree = (df["agreement_status"] == "disagreement").values
    overall = np.ones(len(df), dtype=bool)

    all_rows = []
    # 1. Text-only vs Multimodal (overall + strong disagreement)
    all_rows += compare(df, overall, "text_only", "text_pred", "multimodal", "mm_pred", "overall")
    all_rows += compare(df, disagree, "text_only", "text_pred", "multimodal", "mm_pred", "disagreement")
    all_rows += compare(df, strong, "text_only", "text_pred", "multimodal", "mm_pred", "strong_disagreement")
    # 2. Multimodal vs Metadata-only on strong disagreement
    all_rows += compare(df, strong, "multimodal", "mm_pred", "metadata_only", "meta_pred", "strong_disagreement")
    all_rows += compare(df, disagree, "multimodal", "mm_pred", "metadata_only", "meta_pred", "disagreement")
    # 3. Global vs group-wise temperature scaling (leakage-free half-split)
    all_rows += temp_scaling_half_split(df)

    # 4. Multimodal vs best disagreement-aware method:
    #    per-sample DA predictions are not stored on disk, so this is left to a
    #    provisioned run rather than fabricated.
    all_rows.append({
        "comparison": "multimodal_vs_best_disagreement_aware",
        "subset": "disagreement",
        "metric": "accuracy/f1",
        "status": "requires_retraining_per_sample_DA_predictions_not_stored",
    })

    out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(out)} rows)")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(out.to_string())


if __name__ == "__main__":
    main()
