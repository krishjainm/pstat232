"""Cross-category figures from the REAL existing cross_category_results.csv.

The committed `reports/tables/cross_category_results.csv` is the output of a
prior full multi-category run (All_Beauty, Digital_Music, Appliances succeeded;
Video_Games failed on a metadata parsing error). We visualize that real data.

Note on the calibration-gap panel: per-category, per-group ECE was not stored by
the original cross-category run, so a faithful cross-category calibration figure
cannot be drawn from existing artifacts. We therefore plot the agreement-vs-
disagreement ECE gap for All_Beauty (the one category with stored group-conditional
ECE, `group_conditional_calibration.csv`) and clearly label it as such. The
extended `run_all_categories.py` now records per-category ECE so a full version
can be produced on a provisioned rerun.
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CC = "reports/tables/cross_category_results.csv"
GC = "reports/tables/group_conditional_calibration.csv"
FIG_DIR = "reports/figures"


def main():
    df = pd.read_csv(CC)
    df = df[df["error"].isna()].copy() if "error" in df.columns else df
    df = df.sort_values("category")
    cats = df["category"].tolist()
    x = np.arange(len(cats))
    w = 0.25

    # ---- 1. overall accuracy by model & category ----
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - w, df["text_only_accuracy"], w, label="Text-only", color="#4C72B0")
    plt.bar(x, df["metadata_only_accuracy"], w, label="Metadata-only", color="#DD8452")
    plt.bar(x + w, df["multimodal_accuracy"], w, label="Multimodal", color="#55A868")
    plt.xticks(x, cats)
    plt.ylabel("Overall test accuracy")
    plt.ylim(0.5, 1.0)
    plt.title("Cross-category overall accuracy")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cross_category_accuracy.png", dpi=150)
    plt.close()
    print("Wrote cross_category_accuracy.png")

    # ---- 2. disagreement gap (multimodal overall vs disagreement acc) ----
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - w / 2, df["multimodal_accuracy"], w, label="MM overall", color="#55A868")
    plt.bar(x + w / 2, df["multimodal_disagree_acc"], w, label="MM disagreement", color="#C44E52")
    for i, (ov, dz) in enumerate(zip(df["multimodal_accuracy"], df["multimodal_disagree_acc"])):
        plt.annotate(f"-{(ov - dz) * 100:.0f}pp", (x[i], dz - 0.04), ha="center", fontsize=9, color="#C44E52")
    plt.xticks(x, cats)
    plt.ylabel("Multimodal accuracy")
    plt.ylim(0.5, 1.0)
    plt.title("Cross-category disagreement gap (overall - disagreement)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cross_category_disagreement_gap.png", dpi=150)
    plt.close()
    print("Wrote cross_category_disagreement_gap.png")

    # ---- 3. calibration gap (All_Beauty, real group-conditional ECE) ----
    gc = pd.read_csv(GC)
    mm = gc[gc["model"] == "multimodal"]
    agree_ece = float(mm[mm["group"] == "agreement"]["ece"].iloc[0])
    dis_ece = float(mm[mm["group"] == "disagreement"]["ece"].iloc[0])
    plt.figure(figsize=(6, 4.6))
    bars = plt.bar(["Agreement", "Disagreement"], [agree_ece, dis_ece],
                   color=["#55A868", "#C44E52"])
    for b, v in zip(bars, [agree_ece, dis_ece]):
        plt.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v + 0.005), ha="center")
    plt.ylabel("Expected Calibration Error (ECE)")
    plt.title("Calibration gap, multimodal model (All_Beauty)\n"
              f"{dis_ece / agree_ece:.0f}x worse under disagreement")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cross_category_calibration_gap.png", dpi=150)
    plt.close()
    print("Wrote cross_category_calibration_gap.png (All_Beauty; see module docstring)")

    print("\nReal cross-category numbers used:")
    print(df[["category", "n_test", "disagreement_rate", "multimodal_accuracy",
              "multimodal_disagree_acc", "multimodal_disagree_f1", "text_dominance_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
