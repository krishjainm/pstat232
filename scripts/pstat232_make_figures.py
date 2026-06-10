"""PSTAT 232: generate the clean figure set for the report.

Each figure is generated only if its backing data exists; otherwise the script
prints the exact command needed to regenerate that data and continues. This
keeps the figure step idempotent and avoids inventing results.

Figures
    reports/figures/pstat232_accuracy_by_disagreement.png
    reports/figures/pstat232_calibration_by_group.png
    reports/figures/pstat232_group_calibration.png       (delegated)
    reports/figures/pstat232_leakage_ablation.png
    reports/figures/pstat232_selective_prediction.png

Inputs
    reports/tables/pstat232_accuracy_by_disagreement.csv
    reports/tables/pstat232_leakage_ablation.csv
    reports/tables/pstat232_selective_prediction.csv
    reports/tables/pstat232_group_calibration.csv
    data_icml/processed/{category}_pstat232_test_predictions.parquet

Usage
    python scripts/pstat232_make_figures.py --category All_Beauty
"""

import argparse
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.calibration import expected_calibration_error

TABLE_DIR = "reports/tables"
FIG_DIR = "reports/figures"
PRED_DIR = os.path.join("data_icml", "processed")

MODEL_LABELS = {
    "text_only": "Text-only",
    "metadata_only_lc": "Metadata-only (LC)",
    "multimodal_lc": "Multimodal (LC)",
    "disagreement_aware_lc": "Disagree-aware (LC)",
}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _missing(path, regen_cmd):
    print(f"[skip] missing {path}\n        regenerate with: {regen_cmd}")


def fig_accuracy_by_disagreement():
    path = os.path.join(TABLE_DIR, "pstat232_accuracy_by_disagreement.csv")
    if not os.path.exists(path):
        _missing(path, "python scripts/pstat232_leakage_controlled_main.py")
        return
    df = pd.read_csv(path)
    groups = ["agreement", "weak_disagreement", "medium_disagreement",
              "strong_disagreement"]
    groups = [g for g in groups if g in df["disagreement_group"].unique()]
    models = [m for m in MODEL_LABELS if m in df["model"].unique()]
    x = np.arange(len(groups))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("disagreement_group")
        vals = [sub.loc[g, "accuracy"] if g in sub.index else np.nan for g in groups]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
               label=MODEL_LABELS[model], color=PALETTE[i % len(PALETTE)], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", "\n") for g in groups])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, ls="--", c="k", alpha=0.5, label="chance")
    ax.set_title("Accuracy by disagreement severity (leakage-controlled)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(FIG_DIR, "pstat232_accuracy_by_disagreement.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_calibration_by_group(category):
    path = os.path.join(PRED_DIR, f"{category}_pstat232_test_predictions.parquet")
    if not os.path.exists(path):
        _missing(path, f"python scripts/pstat232_leakage_controlled_main.py --category {category}")
        return
    df = pd.read_parquet(path)
    specs = [("text_only", "text_prob"), ("metadata_only_lc", "meta_prob"),
             ("multimodal_lc", "mm_prob")]
    if "da_prob" in df.columns:
        specs.append(("disagreement_aware_lc", "da_prob"))
    groups = ["agreement", "disagreement"]
    x = np.arange(len(specs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for gi, g in enumerate(groups):
        mask = df["agreement_status"].values == g
        vals = []
        for _, prob_col in specs:
            y = df["label"].values[mask]
            p = df[prob_col].values[mask]
            ece = expected_calibration_error(y, p)[0] if mask.sum() >= 10 else np.nan
            vals.append(ece)
        ax.bar(x + (gi - 0.5) * width, vals, width,
               label=g.capitalize(), color=PALETTE[gi], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m, _ in specs], rotation=15)
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title(f"{category}: ECE by agreement vs disagreement (leakage-controlled)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(FIG_DIR, "pstat232_calibration_by_group.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_leakage_ablation():
    path = os.path.join(TABLE_DIR, "pstat232_leakage_ablation.csv")
    if not os.path.exists(path):
        _missing(path, "python scripts/pstat232_leakage_controlled_main.py")
        return
    df = pd.read_csv(path)
    models = df["model"].unique().tolist()
    metrics = ["accuracy", "auroc", "disagreement_acc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    width = 0.35
    for ax, metric in zip(axes, metrics):
        x = np.arange(len(models))
        for fi, (fs, col) in enumerate([("leakage_controlled", "#55A868"),
                                        ("full", "#C44E52")]):
            vals = []
            for m in models:
                r = df[(df["model"] == m) & (df["feature_set"] == fs)]
                vals.append(r[metric].values[0] if len(r) else np.nan)
            ax.bar(x + (fi - 0.5) * width, vals, width,
                   label="Leakage-controlled" if fs == "leakage_controlled" else "Full (leakage-prone)",
                   color=col, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_ylabel("Score")
    fig.suptitle("Leakage ablation: effect of removing product_average_rating")
    out = os.path.join(FIG_DIR, "pstat232_leakage_ablation.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_selective_prediction():
    path = os.path.join(TABLE_DIR, "pstat232_selective_prediction.csv")
    if not os.path.exists(path):
        _missing(path, "python scripts/pstat232_leakage_controlled_main.py")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"overall": "#4C72B0", "agreement": "#55A868", "disagreement": "#C44E52"}
    for g in ["overall", "agreement", "disagreement"]:
        sub = df[df["group"] == g].sort_values("coverage")
        if len(sub) == 0:
            continue
        ax.plot(sub["coverage"], sub["accuracy"], marker="o",
                label=g.capitalize(), color=colors.get(g, "#333333"))
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("Selective accuracy")
    ax.set_title("Selective prediction: multimodal (LC) accuracy vs coverage")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(FIG_DIR, "pstat232_selective_prediction.png")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_group_calibration(category, tau, objective):
    """Delegate to the calibration script if its table is missing; otherwise it
    will already have written the figure. We (re)build from the CSV for
    idempotence when the table exists."""
    table = os.path.join(TABLE_DIR, "pstat232_group_calibration.csv")
    fig_path = os.path.join(FIG_DIR, "pstat232_group_calibration.png")
    if os.path.exists(fig_path):
        print(f"[fig] {fig_path} (already present)")
        return
    if not os.path.exists(table):
        _missing(table,
                 f"python scripts/pstat232_group_conditional_calibration.py --category {category}")
        return
    # Rebuild a compact ECE-by-group figure from the table alone.
    df = pd.read_csv(table)
    groups = ["all", "nonconflict", "conflict"]
    methods = ["uncalibrated", "global_temperature", "group_conditional_temperature"]
    labels = ["Uncalibrated", "Global T", "Group-conditional T"]
    colors = ["#BBBBBB", "#4C72B0", "#55A868"]
    x = np.arange(len(groups)); width = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (m, lab, col) in enumerate(zip(methods, labels, colors)):
        vals = []
        for g in groups:
            r = df[(df["method"] == m) & (df["group"] == g)]
            vals.append(r["ece"].values[0] if len(r) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=lab, color=col, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["All", "Non-conflict", "Conflict"])
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title(f"{category}: group-conditional calibration")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {fig_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default="All_Beauty")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--objective", default="nll")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_accuracy_by_disagreement()
    fig_calibration_by_group(args.category)
    fig_leakage_ablation()
    fig_selective_prediction()
    fig_group_calibration(args.category, args.tau, args.objective)
    print("\n[pstat232] figure generation complete.")


if __name__ == "__main__":
    main()
