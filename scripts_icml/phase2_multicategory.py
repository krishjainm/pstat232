"""Phase 2: multi-category generalization of the conflict-failure pattern.

For each Amazon product category we build a fixed balanced pool (cached) and run
seeds 42/123/456 with redrawn splits, using leakage-controlled features as the
main setting (full-feature early fusion kept only as an appendix row).

Models (lean set):
  text_only, metadata_lc, early_fusion_lc, disagree_aware_lc   (main)
  early_fusion_full                                            (appendix)

UGCA-Fusion (Phase 4) is added to this grid after it is implemented.

Outputs:
  reports_icml/tables/multicategory_multiseed_results.csv
  reports_icml/tables/multicategory_summary.csv
  reports_icml/figures/multicategory_disagreement_gap.png
  reports_icml/figures/multicategory_calibration_gap.png
  reports_icml/figures/multicategory_mitigation_effect.png

Usage:
  python -m scripts_icml.phase2_multicategory
"""

import argparse
import os
import traceback

import numpy as np
import pandas as pd

from scripts_icml import icml_common as C

SEEDS = [42, 123, 456]

# (category, pool_size). All_Beauty reuses the Phase-1 pool (20k); others 12k to
# keep the one-time CPU sentiment pass tractable. Documented in the report.
CATEGORY_POOLS = [
    ("All_Beauty", 20000),
    ("Digital_Music", 12000),
    ("Gift_Cards", 12000),
    ("Appliances", 12000),
    ("Video_Games", 12000),
]

MAIN_MODELS = ["text_only", "metadata_lc", "early_fusion_lc", "disagree_aware_lc"]
APPENDIX_MODELS = ["early_fusion_full"]


def run_seed_lean(pool, emb, seed, category):
    tr, va, te = C.make_split(pool, seed)
    sid = C.split_id(category, seed)
    train_df, val_df, test_df = pool.iloc[tr], pool.iloc[va], pool.iloc[te].reset_index(drop=True)
    y_tr, y_va = train_df["label"].values, val_df["label"].values
    emb_tr, emb_va, emb_te = emb[tr], emb[va], emb[te]

    full_cols = [c for c in C.METADATA_FULL if c in pool.columns]
    lc_cols = [c for c in C.METADATA_LEAKAGE_CONTROLLED if c in pool.columns]
    mf_tr, mf_va, mf_te = C.scale_meta(train_df, [train_df, val_df, test_df], full_cols)
    ml_tr, ml_va, ml_te = C.scale_meta(train_df, [train_df, val_df, test_df], lc_cols)

    preds = {}
    preds["text_only"] = C.train_text_only(emb_tr, y_tr, emb_te, seed)
    preds["metadata_lc"] = C.train_metadata_only(ml_tr, y_tr, ml_te, seed)
    preds["early_fusion_lc"] = C.train_fusion(
        "early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed)
    preds["early_fusion_full"] = C.train_fusion(
        "early", emb_tr, mf_tr, y_tr, emb_va, mf_va, y_va, emb_te, mf_te, seed)
    w = np.ones(len(train_df), dtype=np.float32)
    w[train_df["agreement_status"].values == "disagreement"] = 5.0
    preds["disagree_aware_lc"] = C.train_fusion(
        "early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed,
        sample_weights=w)

    text_pred, meta_pred = preds["text_only"][0], preds["metadata_lc"][0]
    is_fusion = {"early_fusion_lc", "early_fusion_full", "disagree_aware_lc"}
    info = {
        "text_only": ("sbert_text", False, False, "main"),
        "metadata_lc": ("metadata_lc", False, False, "main"),
        "early_fusion_lc": ("sbert+metadata_lc", False, False, "main"),
        "disagree_aware_lc": ("sbert+metadata_lc", False, True, "main"),
        "early_fusion_full": ("sbert+metadata_full", True, False, "appendix"),
    }

    rows = []
    for name, (pred, prob) in preds.items():
        feats, has_par, dis_train, tier = info[name]
        tp = text_pred if name in is_fusion else None
        mp = meta_pred if name in is_fusion else None
        m = C.compute_metric_suite(test_df, pred, prob, text_pred=tp, meta_pred=mp)
        m.update({"category": category, "seed": seed, "split_id": sid,
                  "model": name, "features": feats, "tier": tier,
                  "includes_product_avg_rating": has_par,
                  "disagreement_labels_in_training": dis_train})
        rows.append(m)
        C.log_experiment({
            "phase": "2", "dataset": "Amazon-Reviews-2023", "category": category,
            "seed": seed, "split_id": sid, "model": name, "features": feats,
            "includes_product_avg_rating": has_par,
            "disagreement_labels_in_training": dis_train,
            "command": "python -m scripts_icml.phase2_multicategory",
            "output_files": "reports_icml/tables/multicategory_multiseed_results.csv",
            "key_metrics": (f"acc={m['accuracy']:.3f},agree_acc={m['agreement_acc']:.3f},"
                            f"disag_acc={m['disagreement_acc']:.3f},cal_gap={m['calibration_gap']:.3f}"),
        })
    return rows


def build_summary(df):
    metrics = ["accuracy", "agreement_acc", "disagreement_acc",
               "strong_disagreement_acc", "auroc", "ece",
               "agreement_ece", "disagreement_ece", "calibration_gap",
               "high_conf_error_rate", "modality_dominance_text_pct"]
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    return df.groupby(["category", "model"]).agg(**agg).reset_index()


def fig_disagreement_gap(summary, categories, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[summary["model"] == "early_fusion_lc"].set_index("category")
    cats = [c for c in categories if c in s.index]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, s.loc[cats, "agreement_acc_mean"], w,
           yerr=s.loc[cats, "agreement_acc_std"], capsize=3,
           label="agreement acc", color="#55A868")
    ax.bar(x + w/2, s.loc[cats, "disagreement_acc_mean"], w,
           yerr=s.loc[cats, "disagreement_acc_std"], capsize=3,
           label="disagreement acc", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Early fusion (leakage-controlled): agreement vs disagreement accuracy\n"
                 "across categories (mean +/- std, seeds 42/123/456)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_calibration_gap(summary, categories, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["text_only", "early_fusion_lc"]
    cats = [c for c in categories if c in summary["category"].values]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"text_only": "#4C72B0", "early_fusion_lc": "#C44E52"}
    for i, model in enumerate(models):
        s = summary[summary["model"] == model].set_index("category")
        vals = [s.loc[c, "calibration_gap_mean"] if c in s.index else np.nan for c in cats]
        errs = [s.loc[c, "calibration_gap_std"] if c in s.index else 0 for c in cats]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3,
               label=model, color=colors[model])
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("calibration gap (disagree ECE - agree ECE)")
    ax.set_title("Calibration gap under conflict across categories")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def fig_mitigation_effect(summary, categories, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["early_fusion_lc", "disagree_aware_lc"]
    cats = [c for c in categories if c in summary["category"].values]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"early_fusion_lc": "#8172B3", "disagree_aware_lc": "#CCB974"}
    for i, model in enumerate(models):
        s = summary[summary["model"] == model].set_index("category")
        vals = [s.loc[c, "disagreement_acc_mean"] if c in s.index else np.nan for c in cats]
        errs = [s.loc[c, "disagreement_acc_std"] if c in s.index else 0 for c in cats]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3,
               label=model, color=colors[model])
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("disagreement accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Disagreement-aware reweighting effect on disagreement accuracy")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=None,
                    help="subset of categories to run (default: all configured)")
    args = ap.parse_args()

    pools = CATEGORY_POOLS
    if args.categories:
        pools = [(c, dict(CATEGORY_POOLS).get(c, 12000)) for c in args.categories]

    os.makedirs("reports_icml/tables", exist_ok=True)
    out_csv = "reports_icml/tables/multicategory_multiseed_results.csv"

    all_rows = []
    done_categories = []
    for category, psize in pools:
        print(f"\n########## CATEGORY: {category} (pool {psize}) ##########", flush=True)
        try:
            pool = C.build_canonical_pool(category, pool_size=psize)
            emb = C.load_sbert(category)
            if len(pool) != len(emb):
                raise RuntimeError("pool/sbert length mismatch")
        except Exception as e:
            print(f"[WARN] skipping {category}: {e}")
            traceback.print_exc()
            continue
        for seed in SEEDS:
            print(f"  -- seed {seed}", flush=True)
            all_rows.extend(run_seed_lean(pool, emb, seed, category))
        done_categories.append(category)
        # incremental save so partial progress is never lost
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"[save] {out_csv} now has {len(all_rows)} rows "
              f"({len(done_categories)} categories)")

    if not all_rows:
        print("[Phase 2] no categories completed.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    summary = build_summary(df)
    summary.to_csv("reports_icml/tables/multicategory_summary.csv", index=False)

    order = [c for c, _ in CATEGORY_POOLS if c in done_categories]
    fig_disagreement_gap(summary, order, "reports_icml/figures/multicategory_disagreement_gap.png")
    fig_calibration_gap(summary, order, "reports_icml/figures/multicategory_calibration_gap.png")
    fig_mitigation_effect(summary, order, "reports_icml/figures/multicategory_mitigation_effect.png")

    print(f"\n[Phase 2] complete. Categories: {done_categories}")


if __name__ == "__main__":
    main()
