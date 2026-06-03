"""Phase 4 benchmark: UGCA-Fusion vs baselines + ablations on All_Beauty.

For each of 5 seeds (identical redrawn splits, leakage-controlled metadata) we
train, on the SAME split:
  baselines : text_only, metadata_lc, early_fusion_lc, gated_fusion_lc,
              disagree_aware_lc
  UGCA      : ugca_full
  ablations : ugca_no_conflict, ugca_no_entropy, ugca_no_js, ugca_no_cal

We also measure the conflict detector's ability to predict the proxy
disagreement label on the test set WITHOUT using the true class label
(roc_auc(disagreement_label, c_hat)) -- the core inference-time claim.

Outputs:
  reports_icml/tables/ugca_results.csv     (main comparison, per seed + summary)
  reports_icml/tables/ugca_ablation.csv    (ablations, per seed)
  reports_icml/figures/ugca_vs_baselines.png
  reports_icml/figures/ugca_ablation.png
  reports_icml/figures/conflict_detector_auc.png

Usage:
  python -m scripts_icml.evaluate_ugca --category All_Beauty
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts_icml import icml_common as C
from scripts_icml.train_ugca import train_ugca, predict_ugca

SEEDS = [42, 123, 456, 789, 2026]

ABLATIONS = {
    "ugca_full": {},
    "ugca_no_conflict": {"use_conflict_detector": False},
    "ugca_no_entropy": {"use_entropy": False},
    "ugca_no_js": {"use_js": False},
    "ugca_no_cal": {"use_calibration_loss": False},
}


def run_seed(pool, emb, seed, category):
    tr, va, te = C.make_split(pool, seed)
    train_df, val_df, test_df = pool.iloc[tr], pool.iloc[va], pool.iloc[te].reset_index(drop=True)
    y_tr, y_va = train_df["label"].values, val_df["label"].values
    emb_tr, emb_va, emb_te = emb[tr], emb[va], emb[te]
    lc = [c for c in C.METADATA_LEAKAGE_CONTROLLED if c in pool.columns]
    ml_tr, ml_va, ml_te = C.scale_meta(train_df, [train_df, val_df, test_df], lc)
    disagree_tr = (train_df["agreement_status"].values == "disagreement").astype(np.float32)
    disagree_te = (test_df["agreement_status"].values == "disagreement").astype(int)

    # reference unimodal predictions (for modality dominance, same split)
    text_pred, _ = C.train_text_only(emb_tr, y_tr, emb_te, seed)
    meta_pred, _ = C.train_metadata_only(ml_tr, y_tr, ml_te, seed)

    rows_main, rows_abl, conflict_auc = [], [], {}

    # --- baselines (identical split) ---
    baselines = {}
    baselines["early_fusion_lc"] = C.train_fusion(
        "early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed)
    baselines["gated_fusion_lc"] = C.train_fusion(
        "gated", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed)
    w = np.ones(len(train_df), dtype=np.float32)
    w[train_df["agreement_status"].values == "disagreement"] = 5.0
    baselines["disagree_aware_lc"] = C.train_fusion(
        "early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed,
        sample_weights=w)

    for name, (pred, prob) in baselines.items():
        m = C.compute_metric_suite(test_df, pred, prob, text_pred=text_pred, meta_pred=meta_pred)
        m.update({"category": category, "seed": seed, "model": name,
                  "conflict_auc": np.nan})
        rows_main.append(m)
        rows_abl.append({**m})

    # --- UGCA + ablations ---
    for name, flags in ABLATIONS.items():
        model = train_ugca(emb_tr, ml_tr, y_tr, disagree_tr,
                           emb_va, ml_va, y_va, seed, flags=flags)
        pred, prob, c_hat, gate = predict_ugca(model, emb_te, ml_te)
        m = C.compute_metric_suite(test_df, pred, prob, text_pred=text_pred, meta_pred=meta_pred)
        # conflict detector AUROC vs proxy disagreement label (no true class label used)
        cauc = np.nan
        if flags.get("use_conflict_detector", True) and len(np.unique(disagree_te)) > 1:
            try:
                cauc = roc_auc_score(disagree_te, c_hat)
            except Exception:
                cauc = np.nan
        m.update({"category": category, "seed": seed, "model": name,
                  "conflict_auc": cauc, "mean_gate": float(np.mean(gate))})
        rows_abl.append(m)
        if name == "ugca_full":
            rows_main.append(m)
            conflict_auc[seed] = cauc

        C.log_experiment({
            "phase": "4", "dataset": "Amazon-Reviews-2023", "category": category,
            "seed": seed, "split_id": C.split_id(category, seed), "model": name,
            "features": "sbert+metadata_lc", "includes_product_avg_rating": False,
            "disagreement_labels_in_training": flags.get("use_conflict_detector", True),
            "command": "python -m scripts_icml.evaluate_ugca",
            "output_files": "reports_icml/tables/ugca_results.csv,ugca_ablation.csv",
            "key_metrics": (f"acc={m['accuracy']:.3f},disag_acc={m['disagreement_acc']:.3f},"
                            f"ece={m['ece']:.3f},cal_gap={m['calibration_gap']:.3f},"
                            f"conflict_auc={cauc if np.isnan(cauc) else round(cauc,3)}"),
        })

    return rows_main, rows_abl, conflict_auc


def summarize(df, models):
    metrics = ["accuracy", "disagreement_acc", "strong_disagreement_acc",
               "ece", "calibration_gap", "high_conf_error_rate", "auroc",
               "conflict_auc"]
    agg = {}
    for m in metrics:
        if m in df.columns:
            agg[f"{m}_mean"] = (m, "mean")
            agg[f"{m}_std"] = (m, "std")
    s = df.groupby("model").agg(**agg).reset_index()
    s["__order"] = s["model"].apply(lambda x: models.index(x) if x in models else 99)
    return s.sort_values("__order").drop(columns="__order")


def fig_vs_baselines(summary, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["early_fusion_lc", "gated_fusion_lc", "disagree_aware_lc", "ugca_full"]
    s = summary.set_index("model")
    order = [m for m in order if m in s.index]
    panels = [("accuracy", "Overall acc"), ("disagreement_acc", "Disagreement acc"),
              ("ece", "ECE (lower better)"), ("calibration_gap", "Calib gap (lower better)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    x = np.arange(len(order))
    for ax, (met, title) in zip(axes, panels):
        ax.bar(x, s.loc[order, f"{met}_mean"], yerr=s.loc[order, f"{met}_std"],
               capsize=4, color="#4C72B0", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("UGCA-Fusion vs baselines, All_Beauty (mean +/- std, 5 seeds)")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_ablation(summary, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["ugca_full", "ugca_no_conflict", "ugca_no_entropy", "ugca_no_js",
             "ugca_no_cal", "early_fusion_lc", "gated_fusion_lc"]
    s = summary.set_index("model")
    order = [m for m in order if m in s.index]
    panels = [("disagreement_acc", "Disagreement acc"),
              ("calibration_gap", "Calib gap (lower better)"),
              ("conflict_auc", "Conflict-detector AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(order))
    for ax, (met, title) in zip(axes, panels):
        vals = [s.loc[m, f"{met}_mean"] if (m in s.index and f"{met}_mean" in s.columns) else np.nan for m in order]
        errs = [s.loc[m, f"{met}_std"] if (m in s.index and f"{met}_std" in s.columns) else 0 for m in order]
        ax.bar(x, vals, yerr=errs, capsize=4, color="#55A868", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("UGCA-Fusion ablations, All_Beauty (mean +/- std, 5 seeds)")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_conflict_auc(abl_df, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = abl_df[abl_df["model"] == "ugca_full"].sort_values("seed")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    ax.bar(x, sub["conflict_auc"].values, color="#C44E52", alpha=0.85)
    ax.axhline(0.5, ls="--", c="k", alpha=0.6, label="chance (0.5)")
    mean_auc = sub["conflict_auc"].mean()
    ax.axhline(mean_auc, ls="-", c="navy", label=f"mean={mean_auc:.3f}")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in sub["seed"]])
    ax.set_ylabel("AUROC(disagreement label, c_hat)")
    ax.set_ylim(0, 1)
    ax.set_title("UGCA conflict detector: predicting proxy disagreement\n"
                 "at inference WITHOUT the true class label")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="All_Beauty")
    args = ap.parse_args()

    pool = C.build_canonical_pool(args.category)
    emb = C.load_sbert(args.category)

    main_rows, abl_rows = [], []
    for seed in SEEDS:
        print(f"\n===== UGCA SEED {seed} =====", flush=True)
        rm, ra, _ = run_seed(pool, emb, seed, args.category)
        main_rows.extend(rm); abl_rows.extend(ra)
        # incremental save
        os.makedirs("reports_icml/tables", exist_ok=True)
        pd.DataFrame(main_rows).to_csv("reports_icml/tables/ugca_results.csv", index=False)
        pd.DataFrame(abl_rows).to_csv("reports_icml/tables/ugca_ablation.csv", index=False)
        print(f"[save] seed {seed} done")

    main_df = pd.DataFrame(main_rows)
    abl_df = pd.DataFrame(abl_rows)

    main_models = ["early_fusion_lc", "gated_fusion_lc", "disagree_aware_lc", "ugca_full"]
    abl_models = ["ugca_full", "ugca_no_conflict", "ugca_no_entropy", "ugca_no_js",
                  "ugca_no_cal", "early_fusion_lc", "gated_fusion_lc"]
    summarize(main_df, main_models).to_csv(
        "reports_icml/tables/ugca_results_summary.csv", index=False)
    abl_summary = summarize(abl_df, abl_models)
    abl_summary.to_csv("reports_icml/tables/ugca_ablation_summary.csv", index=False)

    fig_vs_baselines(summarize(main_df, main_models),
                     "reports_icml/figures/ugca_vs_baselines.png")
    fig_ablation(abl_summary, "reports_icml/figures/ugca_ablation.png")
    fig_conflict_auc(abl_df, "reports_icml/figures/conflict_detector_auc.png")

    print("\n[Phase 4] complete.")


if __name__ == "__main__":
    main()
