"""PSTAT 232: resampling-based inference for classifier metrics.

Computational-statistics core of the project. Instead of reporting point
estimates only, we quantify sampling uncertainty with the nonparametric
bootstrap and test pairwise differences with the *paired* bootstrap and
McNemar's test.

1. Bootstrap percentile 95% CIs for accuracy, F1, AUROC, ECE, and Brier score,
   for each model, on the overall test set and on the disagreement subset.
2. Paired comparisons (same resampled indices for both models) of accuracy and
   F1 differences, with two-sided bootstrap p-values, reported overall and on
   the disagreement subset. McNemar's exact-style chi-square test is reported
   alongside for the accuracy comparison.

Inputs (produced by scripts/pstat232_leakage_controlled_main.py)
    data_icml/processed/{category}_pstat232_test_predictions.parquet

Outputs
    reports/tables/pstat232_bootstrap_ci.csv
    reports/tables/pstat232_paired_comparisons.csv

Usage
    python scripts/pstat232_resampling_inference.py --category All_Beauty --n-boot 2000
"""

import argparse
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from scripts_icml import icml_common as C

TABLE_DIR = "reports/tables"
PRED_DIR = os.path.join("data_icml", "processed")

MODELS = [
    ("text_only", "text_pred", "text_prob"),
    ("metadata_only_lc", "meta_pred", "meta_prob"),
    ("multimodal_lc", "mm_pred", "mm_prob"),
    ("disagreement_aware_lc", "da_pred", "da_prob"),
]


def _test_path(category):
    return os.path.join(PRED_DIR, f"{category}_pstat232_test_predictions.parquet")


def bootstrap_ci_table(df, n_boot, seed):
    rows = []
    subsets = {
        "overall": np.ones(len(df), dtype=bool),
        "disagreement": df["agreement_status"].values == "disagreement",
    }
    for model, pred_col, prob_col in MODELS:
        if pred_col not in df.columns:
            continue
        y = df["label"].values
        pred = df[pred_col].values
        prob = df[prob_col].values
        for subset_name, mask in subsets.items():
            for metric in ["accuracy", "f1", "auroc", "ece"]:
                point, lo, hi = C.bootstrap_ci(
                    y, pred, prob, metric=metric, n=n_boot, seed=seed, mask=mask)
                rows.append({
                    "model": model, "subset": subset_name, "metric": metric,
                    "n": int(mask.sum()), "estimate": point,
                    "ci_lo": lo, "ci_hi": hi,
                })
            # Brier handled directly (not in bootstrap_ci helper's metric set).
            point, lo, hi = _bootstrap_brier(y, prob, mask, n_boot, seed)
            rows.append({
                "model": model, "subset": "overall" if subset_name == "overall" else subset_name,
                "metric": "brier", "n": int(mask.sum()),
                "estimate": point, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def _bootstrap_brier(y, prob, mask, n_boot, seed):
    y = np.asarray(y)[mask]
    prob = np.asarray(prob)[mask]
    if len(y) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    N = len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, N, N)
        vals[i] = np.mean((prob[idx] - y[idx]) ** 2)
    point = float(np.mean((prob - y) ** 2))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_comparisons_table(df, n_boot, seed):
    y = df["label"].values
    comparisons = [
        ("multimodal_lc", "text_only"),
        ("multimodal_lc", "metadata_only_lc"),
        ("disagreement_aware_lc", "multimodal_lc"),
    ]
    subsets = {
        "overall": np.ones(len(df), dtype=bool),
        "disagreement": df["agreement_status"].values == "disagreement",
        "strong_disagreement": df["disagreement_group"].values == "strong_disagreement",
    }
    col = {m: (p, pr) for m, p, pr in MODELS}
    rows = []
    for a, b in comparisons:
        if a not in col or b not in col:
            continue
        pa = df[col[a][0]].values
        pb = df[col[b][0]].values
        for subset_name, mask in subsets.items():
            if mask.sum() < 10:
                continue
            for metric in ["accuracy", "f1"]:
                delta, p_boot = C.paired_bootstrap_test(
                    y, pa, pb, mask=mask, metric=metric, n=n_boot, seed=seed)
                row = {
                    "model_a": a, "model_b": b, "subset": subset_name,
                    "metric": metric, "n": int(mask.sum()),
                    "delta_a_minus_b": delta, "boot_p_two_sided": p_boot,
                }
                if metric == "accuracy":
                    stat, p_mc, disc_b, disc_c = C.mcnemar(y, pa, pb, mask=mask)
                    row["mcnemar_chi2"] = stat
                    row["mcnemar_p"] = p_mc
                    row["mcnemar_b_a_correct_b_wrong"] = disc_b
                    row["mcnemar_c_b_correct_a_wrong"] = disc_c
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default="All_Beauty")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = _test_path(args.category)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}\nRun: python scripts/pstat232_leakage_controlled_main.py "
            f"--category {args.category}")

    df = pd.read_parquet(path)
    os.makedirs(TABLE_DIR, exist_ok=True)

    ci = bootstrap_ci_table(df, args.n_boot, args.seed)
    ci_path = os.path.join(TABLE_DIR, "pstat232_bootstrap_ci.csv")
    ci.to_csv(ci_path, index=False)
    print(f"[table] {ci_path}")
    print(ci.to_string(index=False))

    paired = paired_comparisons_table(df, args.n_boot, args.seed)
    paired_path = os.path.join(TABLE_DIR, "pstat232_paired_comparisons.csv")
    paired.to_csv(paired_path, index=False)
    print(f"\n[table] {paired_path}")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
