"""PSTAT 232 main driver: leakage-controlled disagreement-aware fusion.

This is the *primary* PSTAT 232 pipeline. It is a computational-statistics
re-framing of the original PSTAT 262DS multimodal-disagreement study:

  * Empirical risk minimization (ERM) for text-only, metadata-only, and an
    early-fusion model, with a disagreement-aware reweighted ERM variant.
  * Leakage control: the primary metadata feature set DROPS the product-level
    aggregate ``product_average_rating`` (and ``product_rating_number``), which
    partially encodes the rating-derived label. The full (leakage-prone) set is
    retained only as a sensitivity/ablation experiment.
  * Per-sample test/validation predictions are persisted so that the
    calibration, resampling-inference, and figure scripts consume a single,
    reproducible prediction artifact (no re-training required downstream).

It reuses the validated, materialized canonical pools and cached SBERT
embeddings under ``data_icml/`` via ``scripts_icml.icml_common``; therefore it
runs offline with no data download.

Outputs
-------
data_icml/processed/{category}_pstat232_test_predictions.parquet
data_icml/processed/{category}_pstat232_val_predictions.parquet
reports/tables/pstat232_main_results.csv
reports/tables/pstat232_leakage_ablation.csv
reports/tables/pstat232_accuracy_by_disagreement.csv
reports/tables/pstat232_selective_prediction.csv

Usage
-----
    python scripts/pstat232_leakage_controlled_main.py --category All_Beauty --seed 42
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


def _ensure_dirs():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)


def test_pred_path(category):
    return os.path.join(PRED_DIR, f"{category}_pstat232_test_predictions.parquet")


def val_pred_path(category):
    return os.path.join(PRED_DIR, f"{category}_pstat232_val_predictions.parquet")


def _save_table(df, name):
    path = os.path.join(TABLE_DIR, name)
    df.to_csv(path, index=False)
    print(f"[table] {path}")


def _selective_curve(y_true, y_pred, confidence, n_thresholds=20):
    """Accuracy vs coverage at increasing confidence thresholds."""
    thresholds = np.linspace(0.5, 0.99, n_thresholds)
    rows = []
    for t in thresholds:
        mask = confidence >= t
        cov = float(mask.mean())
        acc = float((y_true[mask] == y_pred[mask]).mean()) if mask.sum() else np.nan
        rows.append({"threshold": float(t), "coverage": cov, "accuracy": acc})
    return pd.DataFrame(rows)


def build_predictions(category, seed):
    """Train ERM models on a single leakage-controlled split and return
    per-sample prediction frames for the validation and test partitions, plus
    the auxiliary metric/ablation tables."""
    pool = C.build_canonical_pool(category)
    emb = C.load_sbert(category)

    tr, va, te = C.make_split(pool, seed)
    train_df = pool.iloc[tr].reset_index(drop=True)
    val_df = pool.iloc[va].reset_index(drop=True)
    test_df = pool.iloc[te].reset_index(drop=True)
    y_tr, y_va, y_te = train_df["label"].values, val_df["label"].values, test_df["label"].values
    emb_tr, emb_va, emb_te = emb[tr], emb[va], emb[te]

    # Feature sets: primary (leakage-controlled) vs sensitivity (full).
    lc_cols = [c for c in C.METADATA_LEAKAGE_CONTROLLED if c in pool.columns]
    full_cols = [c for c in C.METADATA_FULL if c in pool.columns]
    print(f"[features] leakage-controlled metadata: {lc_cols}")
    print(f"[features] full (sensitivity) metadata:  {full_cols}")

    ml_tr, ml_va, ml_te = C.scale_meta(train_df, [train_df, val_df, test_df], lc_cols)
    mf_tr, mf_va, mf_te = C.scale_meta(train_df, [train_df, val_df, test_df], full_cols)

    # ----- Unimodal ERM -----
    text_pred_te, text_prob_te = C.train_text_only(emb_tr, y_tr, emb_te, seed)
    text_pred_va, text_prob_va = C.train_text_only(emb_tr, y_tr, emb_va, seed)
    meta_pred_te, meta_prob_te = C.train_metadata_only(ml_tr, y_tr, ml_te, seed)
    meta_pred_va, meta_prob_va = C.train_metadata_only(ml_tr, y_tr, ml_va, seed)

    # ----- Multimodal early-fusion ERM (leakage-controlled) -----
    mm_pred_te, mm_prob_te = C.train_fusion(
        "early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed)
    mm_pred_va, mm_prob_va = C.train_fusion(
        "early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_va, ml_va, seed)

    # ----- Disagreement-aware reweighted ERM (upweight conflict samples) -----
    w = np.ones(len(train_df), dtype=np.float32)
    w[train_df["agreement_status"].values == "disagreement"] = 5.0
    da_pred_te, da_prob_te = C.train_fusion(
        "early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va, emb_te, ml_te, seed,
        sample_weights=w)

    def _frame(df, y, text_pred, text_prob, meta_pred, meta_prob, mm_pred, mm_prob,
               da_pred=None, da_prob=None):
        out = pd.DataFrame({
            "uid": df["uid"].values,
            "label": y,
            "agreement_status": df["agreement_status"].values,
            "disagreement_group": df["disagreement_group"].values,
            "text_pred": text_pred,
            "text_prob": text_prob,
            "meta_pred": meta_pred,
            "meta_prob": meta_prob,
            "mm_pred": mm_pred,
            "mm_prob": mm_prob,
        })
        out["text_confidence"] = np.maximum(text_prob, 1 - text_prob)
        out["meta_confidence"] = np.maximum(meta_prob, 1 - meta_prob)
        out["mm_confidence"] = np.maximum(mm_prob, 1 - mm_prob)
        # Inference-time conflict proxy (uses only model outputs, not the label).
        out["conflict_hard"] = (out["text_pred"] != out["meta_pred"]).astype(int)
        out["prob_gap"] = np.abs(out["text_prob"] - out["meta_prob"])
        if da_pred is not None:
            out["da_pred"] = da_pred
            out["da_prob"] = da_prob
            out["da_confidence"] = np.maximum(da_prob, 1 - da_prob)
        return out

    test_frame = _frame(test_df, y_te, text_pred_te, text_prob_te, meta_pred_te,
                        meta_prob_te, mm_pred_te, mm_prob_te, da_pred_te, da_prob_te)
    val_frame = _frame(val_df, y_va, text_pred_va, text_prob_va, meta_pred_va,
                       meta_prob_va, mm_pred_va, mm_prob_va)

    # ----- Main metric suite (test set) -----
    main_rows = []
    specs = [
        ("text_only", text_pred_te, text_prob_te),
        ("metadata_only_lc", meta_pred_te, meta_prob_te),
        ("multimodal_lc", mm_pred_te, mm_prob_te),
        ("disagreement_aware_lc", da_pred_te, da_prob_te),
    ]
    for name, pred, prob in specs:
        m = C.compute_metric_suite(test_df, pred, prob,
                                   text_pred=text_pred_te, meta_pred=meta_pred_te)
        m = {"model": name, **m}
        main_rows.append(m)
    main_results = pd.DataFrame(main_rows)

    # ----- Leakage ablation: metadata-only and fusion, LC vs FULL -----
    meta_pred_full_te, meta_prob_full_te = C.train_metadata_only(mf_tr, y_tr, mf_te, seed)
    mm_pred_full_te, mm_prob_full_te = C.train_fusion(
        "early", emb_tr, mf_tr, y_tr, emb_va, mf_va, y_va, emb_te, mf_te, seed)
    abl_rows = []
    for name, feature_set, includes_avg, pred, prob in [
        ("metadata_only", "leakage_controlled", False, meta_pred_te, meta_prob_te),
        ("metadata_only", "full", True, meta_pred_full_te, meta_prob_full_te),
        ("multimodal", "leakage_controlled", False, mm_pred_te, mm_prob_te),
        ("multimodal", "full", True, mm_pred_full_te, mm_prob_full_te),
    ]:
        m = C.compute_metric_suite(test_df, pred, prob)
        abl_rows.append({
            "model": name, "feature_set": feature_set,
            "includes_product_average_rating": includes_avg,
            "accuracy": m["accuracy"], "f1": m["f1"], "auroc": m["auroc"],
            "ece": m["ece"], "brier": m["brier"],
            "disagreement_acc": m["disagreement_acc"],
        })
    leakage_ablation = pd.DataFrame(abl_rows)

    # ----- Accuracy by disagreement group (per model) -----
    acc_rows = []
    group_order = ["agreement", "weak_disagreement", "medium_disagreement",
                   "strong_disagreement"]
    for name, pred, _ in specs:
        for g in group_order:
            mask = test_df["disagreement_group"].values == g
            n = int(mask.sum())
            if n == 0:
                continue
            acc = float((np.asarray(pred)[mask] == y_te[mask]).mean())
            acc_rows.append({"model": name, "disagreement_group": g,
                             "n": n, "accuracy": acc})
    accuracy_by_disagreement = pd.DataFrame(acc_rows)

    # ----- Selective prediction (fusion model), overall + disagreement -----
    sel_frames = []
    for g, mask in [("overall", np.ones(len(test_df), dtype=bool)),
                    ("agreement", test_df["agreement_status"].values == "agreement"),
                    ("disagreement", test_df["agreement_status"].values == "disagreement")]:
        if mask.sum() == 0:
            continue
        cur = _selective_curve(y_te[mask], mm_pred_te[mask],
                               np.maximum(mm_prob_te, 1 - mm_prob_te)[mask])
        cur["group"] = g
        cur["model"] = "multimodal_lc"
        sel_frames.append(cur)
    selective = pd.concat(sel_frames, ignore_index=True)

    return {
        "test_frame": test_frame, "val_frame": val_frame,
        "main_results": main_results, "leakage_ablation": leakage_ablation,
        "accuracy_by_disagreement": accuracy_by_disagreement,
        "selective": selective,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default="All_Beauty",
                    help="Materialized category under data_icml/ (default All_Beauty)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _ensure_dirs()
    res = build_predictions(args.category, args.seed)

    res["test_frame"].to_parquet(test_pred_path(args.category), index=False)
    res["val_frame"].to_parquet(val_pred_path(args.category), index=False)
    print(f"[pred] {test_pred_path(args.category)} ({res['test_frame'].shape})")
    print(f"[pred] {val_pred_path(args.category)} ({res['val_frame'].shape})")

    _save_table(res["main_results"], "pstat232_main_results.csv")
    _save_table(res["leakage_ablation"], "pstat232_leakage_ablation.csv")
    _save_table(res["accuracy_by_disagreement"], "pstat232_accuracy_by_disagreement.csv")
    _save_table(res["selective"], "pstat232_selective_prediction.csv")

    print("\n--- PSTAT 232 main results (leakage-controlled) ---")
    cols = ["model", "accuracy", "disagreement_acc", "auroc", "ece", "brier"]
    print(res["main_results"][cols].to_string(index=False))
    print("\n--- Leakage ablation (LC vs full metadata) ---")
    print(res["leakage_ablation"].to_string(index=False))

    try:
        C.log_experiment({
            "phase": "pstat232", "dataset": "Amazon-Reviews-2023",
            "category": args.category, "seed": args.seed,
            "split_id": C.split_id(args.category, args.seed),
            "model": "text/meta_lc/early_lc/disagree_aware_lc",
            "features": "sbert+metadata_lc", "includes_product_avg_rating": False,
            "disagreement_labels_in_training": True,
            "command": "python scripts/pstat232_leakage_controlled_main.py",
            "output_files": "pstat232_main_results.csv,pstat232_leakage_ablation.csv",
            "key_metrics": "see reports/tables/pstat232_main_results.csv",
        })
    except Exception as e:
        print(f"[registry] skipped: {e}")

    print("\n[pstat232] main pipeline complete.")


if __name__ == "__main__":
    main()
