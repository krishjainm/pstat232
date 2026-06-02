"""Phase 1: clean canonical multi-seed reproduction on All_Beauty.

Runs 8 models across 5 redrawn splits (seeds) on the fixed balanced pool,
reusing cached SBERT embeddings. Produces:
  reports_icml/tables/canonical_allbeauty_multiseed.csv   (per seed x model)
  reports_icml/tables/canonical_allbeauty_summary.csv     (mean +/- std)
  reports_icml/tables/canonical_allbeauty_stats.csv        (paired tests per seed)
  reports_icml/figures/canonical_allbeauty_errorbars.png
  reports_icml/canonical_reproduction_report.md

Usage:
    python -m scripts_icml.phase1_canonical --category All_Beauty
"""

import argparse
import os

import numpy as np
import pandas as pd

from scripts_icml import icml_common as C

SEEDS = [42, 123, 456, 789, 2026]

# Metric columns to summarize with mean +/- std across seeds.
SUMMARY_METRICS = [
    "accuracy", "f1", "auroc", "ece", "brier",
    "agreement_acc", "disagreement_acc", "strong_disagreement_acc",
    "agreement_ece", "disagreement_ece", "calibration_gap",
    "high_conf_error_rate", "conf_correct", "conf_incorrect",
    "modality_dominance_text_pct",
]


def run_seed(pool, emb, seed, category):
    """Train all 8 models for one seed; return list of metric rows + raw preds."""
    tr, va, te = C.make_split(pool, seed)
    sid = C.split_id(category, seed)

    train_df = pool.iloc[tr]
    val_df = pool.iloc[va]
    test_df = pool.iloc[te].reset_index(drop=True)

    y_tr, y_va = train_df["label"].values, val_df["label"].values
    emb_tr, emb_va, emb_te = emb[tr], emb[va], emb[te]

    # Metadata feature matrices (fit scaler on train only).
    full_cols = [c for c in C.METADATA_FULL if c in pool.columns]
    lc_cols = [c for c in C.METADATA_LEAKAGE_CONTROLLED if c in pool.columns]
    mf_tr, mf_va, mf_te = C.scale_meta(train_df, [train_df, val_df, test_df], full_cols)
    ml_tr, ml_va, ml_te = C.scale_meta(train_df, [train_df, val_df, test_df], lc_cols)

    preds = {}

    # 1. Text-only
    preds["text_only"] = C.train_text_only(emb_tr, y_tr, emb_te, seed)
    # 2. Metadata-only full
    preds["metadata_full"] = C.train_metadata_only(mf_tr, y_tr, mf_te, seed)
    # 3. Metadata-only leakage-controlled
    preds["metadata_lc"] = C.train_metadata_only(ml_tr, y_tr, ml_te, seed)
    # 4. Early fusion full
    preds["early_fusion_full"] = C.train_fusion(
        "early", emb_tr, mf_tr, y_tr, emb_va, mf_va, y_va, emb_te, mf_te, seed)
    # 5. Early fusion leakage-controlled
    preds["early_fusion_lc"] = C.train_fusion(
        "early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed)
    # 6. Late fusion (leakage-controlled)
    preds["late_fusion_lc"] = C.train_fusion(
        "late", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed)
    # 7. Gated fusion (leakage-controlled)
    preds["gated_fusion_lc"] = C.train_fusion(
        "gated", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed)
    # 8. Disagreement-aware reweighting (early, lc, w=5 on disagreement train rows)
    w = np.ones(len(train_df), dtype=np.float32)
    w[train_df["agreement_status"].values == "disagreement"] = 5.0
    preds["disagree_aware_lc"] = C.train_fusion(
        "early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed,
        sample_weights=w)

    # Reference unimodal preds for modality dominance (main setting).
    text_pred = preds["text_only"][0]
    meta_pred = preds["metadata_lc"][0]

    meta_info = {
        "text_only": ("sbert_text", False, False),
        "metadata_full": ("metadata_full(8)", True, False),
        "metadata_lc": ("metadata_lc(6)", False, False),
        "early_fusion_full": ("sbert+metadata_full", True, False),
        "early_fusion_lc": ("sbert+metadata_lc", False, False),
        "late_fusion_lc": ("sbert+metadata_lc", False, False),
        "gated_fusion_lc": ("sbert+metadata_lc", False, False),
        "disagree_aware_lc": ("sbert+metadata_lc", False, True),
    }
    is_fusion = {"early_fusion_full", "early_fusion_lc", "late_fusion_lc",
                 "gated_fusion_lc", "disagree_aware_lc"}

    rows = []
    for name, (pred, prob) in preds.items():
        feats, has_par, dis_train = meta_info[name]
        tp = text_pred if name in is_fusion else None
        mp = meta_pred if name in is_fusion else None
        m = C.compute_metric_suite(test_df, pred, prob, text_pred=tp, meta_pred=mp)
        # bootstrap CIs for headline metrics
        y = test_df["label"].values
        for met in ["accuracy", "f1", "auroc"]:
            _, lo, hi = C.bootstrap_ci(y, pred, prob, metric=met, seed=seed)
            m[f"{met}_ci_lo"] = lo
            m[f"{met}_ci_hi"] = hi
        m.update({"category": category, "seed": seed, "split_id": sid,
                  "model": name, "features": feats,
                  "includes_product_avg_rating": has_par,
                  "disagreement_labels_in_training": dis_train})
        rows.append(m)

        C.log_experiment({
            "phase": "1", "dataset": "Amazon-Reviews-2023", "category": category,
            "seed": seed, "split_id": sid, "model": name, "features": feats,
            "includes_product_avg_rating": has_par,
            "disagreement_labels_in_training": dis_train,
            "command": "python -m scripts_icml.phase1_canonical",
            "output_files": "reports_icml/tables/canonical_allbeauty_multiseed.csv",
            "key_metrics": (f"acc={m['accuracy']:.3f},disag_acc={m['disagreement_acc']:.3f},"
                            f"ece={m['ece']:.3f},cal_gap={m['calibration_gap']:.3f}"),
        })

    # paired tests for key comparisons on this seed
    y = test_df["label"].values
    disag_mask = (test_df["agreement_status"].values == "disagreement")
    stat_rows = []
    comparisons = [
        ("early_fusion_lc", "text_only", "overall", None),
        ("early_fusion_lc", "text_only", "disagreement", disag_mask),
        ("disagree_aware_lc", "early_fusion_lc", "disagreement", disag_mask),
        ("early_fusion_full", "early_fusion_lc", "overall", None),
    ]
    for a, b, subset, mask in comparisons:
        pa = preds[a][0]
        pb = preds[b][0]
        d_acc, p_boot = C.paired_bootstrap_test(y, pa, pb, mask=mask, metric="accuracy")
        chi2, p_mc, nb, nc = C.mcnemar(y, pa, pb, mask=mask)
        stat_rows.append({
            "category": category, "seed": seed, "subset": subset,
            "model_a": a, "model_b": b,
            "delta_accuracy": d_acc, "paired_bootstrap_p": p_boot,
            "mcnemar_chi2": chi2, "mcnemar_p": p_mc,
            "n": int(mask.sum()) if mask is not None else len(y),
        })

    return rows, stat_rows


def make_figure(summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["text_only", "metadata_lc", "metadata_full", "early_fusion_lc",
             "early_fusion_full", "late_fusion_lc", "gated_fusion_lc",
             "disagree_aware_lc"]
    summary = summary.set_index("model")
    order = [m for m in order if m in summary.index]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        ("accuracy", "Overall Accuracy"),
        ("disagreement_acc", "Disagreement Accuracy"),
        ("calibration_gap", "Calibration Gap (disagree ECE - agree ECE)"),
    ]
    x = np.arange(len(order))
    for ax, (metric, title) in zip(axes, panels):
        means = summary.loc[order, f"{metric}_mean"].values
        stds = summary.loc[order, f"{metric}_std"].values
        ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Canonical All_Beauty reproduction (mean +/- std over 5 seeds)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="All_Beauty")
    ap.add_argument("--pool-size", type=int, default=20000)
    args = ap.parse_args()

    pool = C.build_canonical_pool(args.category, pool_size=args.pool_size)
    emb = C.load_sbert(args.category)
    assert len(pool) == len(emb), "pool / sbert length mismatch"

    all_rows, all_stats = [], []
    for seed in SEEDS:
        print(f"\n===== SEED {seed} =====")
        rows, stats = run_seed(pool, emb, seed, args.category)
        all_rows.extend(rows)
        all_stats.extend(stats)

    df = pd.DataFrame(all_rows)
    os.makedirs("reports_icml/tables", exist_ok=True)
    df.to_csv("reports_icml/tables/canonical_allbeauty_multiseed.csv", index=False)

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv("reports_icml/tables/canonical_allbeauty_stats.csv", index=False)

    # aggregate mean +/- std across seeds
    agg = {}
    for m in SUMMARY_METRICS:
        if m in df.columns:
            agg[f"{m}_mean"] = (m, "mean")
            agg[f"{m}_std"] = (m, "std")
    summary = df.groupby("model").agg(**agg).reset_index()
    summary.to_csv("reports_icml/tables/canonical_allbeauty_summary.csv", index=False)

    make_figure(summary, "reports_icml/figures/canonical_allbeauty_errorbars.png")
    write_report(df, summary, stats_df, args.category, pool)
    print("\n[Phase 1] complete.")


def write_report(df, summary, stats_df, category, pool):
    lines = []
    lines.append(f"# Canonical Reproduction Report - {category}\n")
    lines.append("## Protocol\n")
    lines.append(f"- Dataset: Amazon Reviews 2023, category `{category}`.\n")
    lines.append(f"- Balanced pool: {len(pool)} reviews "
                 f"({int((pool['label']==1).sum())} pos / {int((pool['label']==0).sum())} neg), "
                 f"fixed with master seed {C.MASTER_SEED}.\n")
    lines.append(f"- Seeds (redrawn 70/15/15 stratified splits): {SEEDS}.\n")
    lines.append("- Leakage-controlled metadata drops "
                 f"`{', '.join(C.LEAKAGE_DROP)}`.\n")
    n_dis = int((pool['agreement_status']=='disagreement').sum())
    lines.append(f"- Pool disagreement rate (Definition A, pretrained sentiment vs "
                 f"rating label): {100*n_dis/len(pool):.1f}% ({n_dis}/{len(pool)}).\n")
    lines.append("\n## Headline results (mean +/- std over 5 seeds)\n")
    cols = ["accuracy", "disagreement_acc", "strong_disagreement_acc", "auroc",
            "ece", "calibration_gap", "high_conf_error_rate"]
    s = summary.set_index("model")
    header = "| model | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)
    order = ["text_only", "metadata_lc", "metadata_full", "early_fusion_lc",
             "early_fusion_full", "late_fusion_lc", "gated_fusion_lc",
             "disagree_aware_lc"]
    for m in [o for o in order if o in s.index]:
        cells = []
        for c in cols:
            mean = s.loc[m, f"{c}_mean"]
            std = s.loc[m, f"{c}_std"]
            cells.append(f"{mean:.3f}+/-{std:.3f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines.append("\n## Key paired tests (per seed, then summarized)\n")
    if len(stats_df):
        g = stats_df.groupby(["model_a", "model_b", "subset"]).agg(
            mean_delta_acc=("delta_accuracy", "mean"),
            std_delta_acc=("delta_accuracy", "std"),
            median_bootstrap_p=("paired_bootstrap_p", "median"),
            median_mcnemar_p=("mcnemar_p", "median"),
            n_seeds=("seed", "count"),
        ).reset_index()
        lines.append("| comparison (A vs B) | subset | mean delta acc | median boot p | median McNemar p |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, r in g.iterrows():
            lines.append(f"| {r['model_a']} vs {r['model_b']} | {r['subset']} | "
                         f"{r['mean_delta_acc']:+.3f}+/-{r['std_delta_acc']:.3f} | "
                         f"{r['median_bootstrap_p']:.3f} | {r['median_mcnemar_p']:.3f} |")

    lines.append("\n## Honest reading\n")
    # quick automated observations
    acc_text = s.loc["text_only", "accuracy_mean"]
    acc_ef = s.loc["early_fusion_lc", "accuracy_mean"]
    dis_text = s.loc["text_only", "disagreement_acc_mean"]
    dis_ef = s.loc["early_fusion_lc", "disagreement_acc_mean"]
    lines.append(f"- Early fusion (leakage-controlled) overall accuracy "
                 f"{acc_ef:.3f} vs text-only {acc_text:.3f} "
                 f"(delta {acc_ef-acc_text:+.3f}).\n")
    lines.append(f"- On disagreement cases: early fusion {dis_ef:.3f} vs text-only "
                 f"{dis_text:.3f} (delta {dis_ef-dis_text:+.3f}).\n")
    lines.append("- See `canonical_allbeauty_summary.csv` for the full metric suite "
                 "and `canonical_allbeauty_stats.csv` for per-seed significance.\n")

    with open("reports_icml/canonical_reproduction_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[report] wrote reports_icml/canonical_reproduction_report.md")


if __name__ == "__main__":
    main()
