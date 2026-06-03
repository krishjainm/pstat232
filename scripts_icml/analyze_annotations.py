"""Phase 6: analyze completed human annotations.

Reads every CSV in reports_icml/annotations/ that contains filled human-label
columns and computes:
  * percentage genuine conflict vs label noise (and other categories)
  * inter-annotator agreement (Cohen's kappa) when >=2 annotators overlap
  * model accuracy on genuine-conflict rows vs label-noise rows

If no annotation files (with any filled labels) exist, prints that manual
annotation is pending and exits cleanly (exit 0). No labels are fabricated.

Usage:
  python -m scripts_icml.analyze_annotations
"""
from __future__ import annotations

import os
import glob
import sys
import numpy as np
import pandas as pd

ANN_DIR = "reports_icml/annotations"
HUMAN_COLS = ["genuine_conflict", "label_noise", "sarcasm",
              "mixed_sentiment", "unclear"]
OUT_TABLE = "reports_icml/tables/human_annotation_results.csv"
OUT_FIG = "reports_icml/figures/human_annotation_breakdown.png"


def _has_labels(df):
    present = [c for c in HUMAN_COLS if c in df.columns]
    if not present:
        return False
    filled = df[present].apply(pd.to_numeric, errors="coerce")
    return bool(filled.notna().any().any() and (filled.fillna(0).sum().sum() > 0))


def _cohens_kappa(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n == 0:
        return np.nan
    po = (a == b).mean()
    cats = np.unique(np.concatenate([a, b]))
    pe = sum(((a == c).mean()) * ((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else np.nan


def main():
    files = []
    if os.path.isdir(ANN_DIR):
        files = [f for f in glob.glob(os.path.join(ANN_DIR, "*.csv"))]
    labeled = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if _has_labels(df):
            df["__annotator__"] = os.path.splitext(os.path.basename(f))[0]
            labeled.append(df)

    if not labeled:
        print("Manual annotation is PENDING: no completed annotation files found "
              f"in {ANN_DIR}/. Fill reports_icml/manual_annotation_sample.csv "
              "(see reports_icml/annotation_guide.md) and save completed files as "
              f"{ANN_DIR}/annotations_<id>.csv, then re-run this script.")
        sys.exit(0)

    os.makedirs("reports_icml/tables", exist_ok=True)
    os.makedirs("reports_icml/figures", exist_ok=True)

    allann = pd.concat(labeled, ignore_index=True)
    for c in HUMAN_COLS:
        if c in allann.columns:
            allann[c] = pd.to_numeric(allann[c], errors="coerce").fillna(0).astype(int)

    n_total = len(allann)
    breakdown = {c: int(allann[c].sum()) for c in HUMAN_COLS if c in allann.columns}
    pct = {c: round(v / n_total, 4) for c, v in breakdown.items()}

    # model accuracy by human category (multimodal prediction vs binary label)
    acc_by_cat = {}
    if {"multimodal_prediction", "binary_label"}.issubset(allann.columns):
        correct = (allann["multimodal_prediction"] == allann["binary_label"])
        for c in HUMAN_COLS:
            if c in allann.columns and allann[c].sum() > 0:
                acc_by_cat[c] = round(correct[allann[c] == 1].mean(), 4)

    # inter-annotator agreement on genuine_conflict (overlapping uids)
    kappas = {}
    annotators = allann["__annotator__"].unique()
    if len(annotators) >= 2 and "genuine_conflict" in allann.columns:
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                a = allann[allann["__annotator__"] == annotators[i]][["uid", "genuine_conflict"]]
                b = allann[allann["__annotator__"] == annotators[j]][["uid", "genuine_conflict"]]
                m = a.merge(b, on="uid", suffixes=("_a", "_b"))
                if len(m):
                    kappas[f"{annotators[i]}|{annotators[j]}"] = round(
                        _cohens_kappa(m["genuine_conflict_a"], m["genuine_conflict_b"]), 4)

    rows = [{"metric": f"pct_{c}", "value": pct[c]} for c in pct]
    rows += [{"metric": f"count_{c}", "value": breakdown[c]} for c in breakdown]
    rows += [{"metric": f"model_acc_on_{c}", "value": v} for c, v in acc_by_cat.items()]
    rows += [{"metric": f"kappa_{k}", "value": v} for k, v in kappas.items()]
    rows.append({"metric": "n_annotations", "value": n_total})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_TABLE, index=False)
    print(res.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(breakdown.keys())
    ax.bar(cats, [pct[c] for c in cats], color="#4C72B0")
    ax.set_ylabel("fraction of annotated candidates")
    ax.set_title("Human annotation breakdown of disagreement candidates")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"[save] {OUT_TABLE}\n[save] {OUT_FIG}")


if __name__ == "__main__":
    main()
