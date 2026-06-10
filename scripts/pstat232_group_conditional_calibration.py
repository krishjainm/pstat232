"""PSTAT 232 extension: group-conditional calibration via an inference-time
conflict proxy.

Motivation (computational statistics)
-------------------------------------
A single global temperature-scaling parameter assumes the *miscalibration* of a
classifier is homogeneous across the input space. Multimodal fusion models
violate this: they are well-calibrated when the modalities agree but badly
over-confident when they conflict. We test whether *conditioning* the
post-hoc calibration map on a cheap, label-free conflict proxy reduces
calibration error where it matters.

Conflict proxy (inference-time, no label used)
----------------------------------------------
    conflict = (text_pred != meta_pred)  OR  (|text_prob - meta_prob| > tau)

Method
------
1. Fit ONE global temperature T_global on the validation set (all samples).
2. Fit SEPARATE temperatures T_conflict and T_nonconflict on the validation
   conflict / non-conflict subsets.
3. Evaluate uncalibrated, global, and group-conditional calibration on the test
   set; report accuracy (unchanged by temperature, sanity check), NLL, Brier,
   ECE, and group-specific ECE.

The model being calibrated is the leakage-controlled early-fusion model
(``mm_prob``). Temperatures are learned on validation and frozen for test, so
there is no test-set leakage in the calibration step.

Inputs (produced by scripts/pstat232_leakage_controlled_main.py)
    data_icml/processed/{category}_pstat232_val_predictions.parquet
    data_icml/processed/{category}_pstat232_test_predictions.parquet

Outputs
    reports/tables/pstat232_group_calibration.csv
    reports/figures/pstat232_group_calibration.png

Usage
    python scripts/pstat232_group_conditional_calibration.py --category All_Beauty --tau 0.5
"""

import argparse
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.evaluation.calibration import expected_calibration_error

TABLE_DIR = "reports/tables"
FIG_DIR = "reports/figures"
PRED_DIR = os.path.join("data_icml", "processed")


def _paths(category):
    return (
        os.path.join(PRED_DIR, f"{category}_pstat232_val_predictions.parquet"),
        os.path.join(PRED_DIR, f"{category}_pstat232_test_predictions.parquet"),
    )


def _conflict_mask(df, tau):
    hard = df["text_pred"].values != df["meta_pred"].values
    soft = np.abs(df["text_prob"].values - df["meta_prob"].values) > tau
    return hard | soft


def _prob_to_logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _apply_temperature(prob_pos, T):
    """Temperature-scale a binary positive-class probability."""
    z = _prob_to_logit(prob_pos)
    return 1.0 / (1.0 + np.exp(-z / T))


def _nll(y_true, prob_pos):
    p = np.clip(prob_pos, 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _brier(y_true, prob_pos):
    return float(np.mean((prob_pos - y_true) ** 2))


def _ece(y_true, prob_pos, n_bins=10):
    ece, *_ = expected_calibration_error(np.asarray(y_true), np.asarray(prob_pos), n_bins)
    return float(ece)


def fit_temperature(y_true, prob_pos, objective="nll", n_bins=10):
    """Fit a scalar temperature on (y_true, prob_pos) by minimizing NLL (default)
    or ECE. NLL is the proper-scoring-rule choice and is smooth in T."""
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return 1.0

    def obj(T):
        cal = _apply_temperature(prob_pos, T)
        if objective == "ece":
            return _ece(y_true, cal, n_bins)
        return _nll(y_true, cal)

    res = minimize_scalar(obj, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def _metric_block(method, group, y_true, prob_pos, T):
    pred = (prob_pos >= 0.5).astype(int)
    return {
        "method": method,
        "group": group,
        "n": int(len(y_true)),
        "temperature": float(T),
        "accuracy": float((pred == y_true).mean()) if len(y_true) else np.nan,
        "nll": _nll(y_true, prob_pos) if len(y_true) else np.nan,
        "brier": _brier(y_true, prob_pos) if len(y_true) else np.nan,
        "ece": _ece(y_true, prob_pos) if len(y_true) >= 10 else np.nan,
    }


def run(category, tau, objective):
    val_path, test_path = _paths(category)
    if not (os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Missing prediction artifacts for {category}.\n"
            f"Run: python scripts/pstat232_leakage_controlled_main.py "
            f"--category {category}")

    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    val_conf = _conflict_mask(val, tau)
    test_conf = _conflict_mask(test, tau)

    yv, pv = val["label"].values, val["mm_prob"].values
    yt, pt = test["label"].values, test["mm_prob"].values

    # ----- Fit temperatures on validation only -----
    T_global = fit_temperature(yv, pv, objective)
    T_conflict = fit_temperature(yv[val_conf], pv[val_conf], objective)
    T_nonconflict = fit_temperature(yv[~val_conf], pv[~val_conf], objective)
    print(f"[temps] global={T_global:.3f} conflict={T_conflict:.3f} "
          f"nonconflict={T_nonconflict:.3f} "
          f"(val conflict n={int(val_conf.sum())}/{len(val)})")

    # ----- Build calibrated test probabilities -----
    p_uncal = pt.copy()
    p_global = _apply_temperature(pt, T_global)
    p_group = np.where(test_conf,
                       _apply_temperature(pt, T_conflict),
                       _apply_temperature(pt, T_nonconflict))

    methods = [
        ("uncalibrated", p_uncal, {"all": 1.0, "conflict": 1.0, "nonconflict": 1.0}),
        ("global_temperature", p_global,
         {"all": T_global, "conflict": T_global, "nonconflict": T_global}),
        ("group_conditional_temperature", p_group,
         {"all": np.nan, "conflict": T_conflict, "nonconflict": T_nonconflict}),
    ]

    rows = []
    for name, probs, temps in methods:
        rows.append(_metric_block(name, "all", yt, probs, temps["all"]))
        rows.append(_metric_block(name, "conflict", yt[test_conf],
                                  probs[test_conf], temps["conflict"]))
        rows.append(_metric_block(name, "nonconflict", yt[~test_conf],
                                  probs[~test_conf], temps["nonconflict"]))
    results = pd.DataFrame(rows)
    results.insert(0, "category", category)
    results.insert(1, "tau", tau)
    results.insert(2, "calibration_objective", objective)
    return results, (p_uncal, p_global, p_group, yt, test_conf)


def make_figure(results, arrays, category, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_uncal, p_global, p_group, yt, test_conf = arrays
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: ECE by group x method (the headline result).
    groups = ["all", "nonconflict", "conflict"]
    methods = ["uncalibrated", "global_temperature", "group_conditional_temperature"]
    labels = ["Uncalibrated", "Global T", "Group-conditional T"]
    colors = ["#BBBBBB", "#4C72B0", "#55A868"]
    x = np.arange(len(groups))
    width = 0.26
    ax = axes[0]
    for i, (m, lab, col) in enumerate(zip(methods, labels, colors)):
        vals = [results[(results["method"] == m) & (results["group"] == g)]["ece"].values[0]
                for g in groups]
        ax.bar(x + (i - 1) * width, vals, width, label=lab, color=col, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["All", "Non-conflict", "Conflict"])
    ax.set_ylabel("Expected Calibration Error (lower is better)")
    ax.set_title(f"{category}: ECE by conflict group and calibration method")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: reliability diagram on the CONFLICT subset.
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="perfect")
    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for probs, lab, col in [(p_uncal, "Uncalibrated", "#BBBBBB"),
                            (p_global, "Global T", "#4C72B0"),
                            (p_group, "Group-conditional T", "#55A868")]:
        pc = probs[test_conf]
        yc = yt[test_conf]
        accs = []
        for i in range(n_bins):
            m = (pc >= edges[i]) & (pc < edges[i + 1] if i < n_bins - 1 else pc <= edges[i + 1])
            accs.append(yc[m].mean() if m.sum() else np.nan)
        ax.plot(centers, accs, marker="o", label=lab, color=col, alpha=0.9)
    ax.set_xlabel("Predicted P(positive)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{category}: reliability on the conflict subset")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default="All_Beauty")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="Probability-gap threshold for the soft conflict proxy")
    ap.add_argument("--objective", choices=["nll", "ece"], default="nll",
                    help="Temperature-fitting objective on validation set")
    args = ap.parse_args()

    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    results, arrays = run(args.category, args.tau, args.objective)
    out_csv = os.path.join(TABLE_DIR, "pstat232_group_calibration.csv")
    results.to_csv(out_csv, index=False)
    print(f"[table] {out_csv}")
    print(results.to_string(index=False))

    make_figure(results, arrays, args.category,
                os.path.join(FIG_DIR, "pstat232_group_calibration.png"))


if __name__ == "__main__":
    main()
