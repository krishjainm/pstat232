"""Phase 5: stronger / multiple disagreement definitions.

We reduce dependence on a single pretrained sentiment model by comparing four
disagreement definitions on the canonical All_Beauty pool:

  A. Pretrained-sentiment vs rating label   (the existing proxy)
  B. Cross-model: text-only pred != metadata-only (leakage-controlled) pred
  C. High-confidence text contradiction: text-only conf >= 0.90 AND text-only
     prediction disagrees with the rating label
  D. Human-review subset: 200 sampled candidates exported for manual annotation
     (no human labels fabricated -- columns left blank)

To make B and C leakage-free for *every* row, unimodal predictions are produced
by 5-fold cross-fitting over the pool (each row predicted by a model that never
saw it in training). The multimodal (early-fusion) prediction in the annotation
CSV is produced the same way.

Outputs:
  reports_icml/tables/disagreement_definition_comparison.csv
  reports_icml/figures/disagreement_definition_overlap.png
  reports_icml/manual_annotation_sample.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from scripts_icml import icml_common as ic

CATEGORY = "All_Beauty"
HIGH_CONF = 0.90
N_FOLDS = 5
CV_SEED = 42
ANNOTATION_N = 200


def cross_fit_predictions(df, emb):
    """Return per-row, out-of-fold predictions/confidences for text-only,
    metadata-only (leakage-controlled), and early-fusion models."""
    y = df["label"].values
    n = len(df)
    text_pred = np.full(n, -1)
    text_conf = np.zeros(n)
    meta_pred = np.full(n, -1)
    meta_conf = np.zeros(n)
    mm_pred = np.full(n, -1)
    mm_conf = np.zeros(n)

    feat_lc = ic.METADATA_LEAKAGE_CONTROLLED
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
        print(f"[cv] fold {fold+1}/{N_FOLDS}", flush=True)
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        emb_tr, emb_te = emb[tr], emb[te]
        y_tr = y[tr]

        # text-only (SBERT + LogReg)
        tp, tprob = ic.train_text_only(emb_tr, y_tr, emb_te, CV_SEED)
        text_pred[te] = tp
        text_conf[te] = np.maximum(tprob, 1 - tprob)

        # metadata-only leakage-controlled (XGBoost)
        Xtr, Xte = ic.scale_meta(tr_df, [tr_df, te_df], feat_lc)
        mp, mprob = ic.train_metadata_only(Xtr, y_tr, Xte, CV_SEED)
        meta_pred[te] = mp
        meta_conf[te] = np.maximum(mprob, 1 - mprob)

        # early fusion (carve a small val set out of the train fold)
        rng = np.random.RandomState(CV_SEED)
        perm = rng.permutation(len(tr))
        n_val = max(1, int(0.15 * len(tr)))
        va_loc, tr_loc = perm[:n_val], perm[n_val:]
        mtr_full, mte = ic.scale_meta(tr_df, [tr_df, te_df], feat_lc)
        fp, fprob = ic.train_fusion(
            "early",
            emb_tr[tr_loc], mtr_full[tr_loc], y_tr[tr_loc],
            emb_tr[va_loc], mtr_full[va_loc], y_tr[va_loc],
            emb_te, mte, CV_SEED)
        mm_pred[te] = fp
        mm_conf[te] = np.maximum(fprob, 1 - fprob)

    return dict(text_pred=text_pred, text_conf=text_conf,
                meta_pred=meta_pred, meta_conf=meta_conf,
                mm_pred=mm_pred, mm_conf=mm_conf)


def main():
    os.makedirs("reports_icml/tables", exist_ok=True)
    os.makedirs("reports_icml/figures", exist_ok=True)

    df = ic.build_canonical_pool(CATEGORY).reset_index(drop=True)
    emb = ic.load_sbert(CATEGORY)
    assert len(df) == len(emb), "pool/SBERT length mismatch"

    cv = cross_fit_predictions(df, emb)
    y = df["label"].values

    # --- Definitions (boolean flags over the full pool) ---
    defA = (df["text_sentiment_label"].values != y)
    defB = (cv["text_pred"] != cv["meta_pred"])
    defC = (cv["text_conf"] >= HIGH_CONF) & (cv["text_pred"] != y)

    defs = {"A_sentiment_vs_label": defA,
            "B_cross_model": defB,
            "C_highconf_text_contradiction": defC}

    # --- Comparison table: prevalence + model accuracy on flagged subset ---
    rows = []
    for name, flag in defs.items():
        n_flag = int(flag.sum())
        rate = n_flag / len(df)
        mm_acc_flag = float((cv["mm_pred"][flag] == y[flag]).mean()) if n_flag else np.nan
        mm_acc_unflag = float((cv["mm_pred"][~flag] == y[~flag]).mean())
        rows.append(dict(definition=name, n_flagged=n_flag,
                         prevalence=round(rate, 4),
                         multimodal_acc_on_flagged=round(mm_acc_flag, 4),
                         multimodal_acc_on_unflagged=round(mm_acc_unflag, 4)))
    comp = pd.DataFrame(rows)

    # --- Pairwise overlap (Jaccard + |intersection|) ---
    names = list(defs.keys())
    jac = pd.DataFrame(index=names, columns=names, dtype=float)
    inter = pd.DataFrame(index=names, columns=names, dtype=int)
    for a in names:
        for b in names:
            A_, B_ = defs[a], defs[b]
            i = int((A_ & B_).sum())
            u = int((A_ | B_).sum())
            jac.loc[a, b] = round(i / u, 4) if u else 0.0
            inter.loc[a, b] = i

    comp_path = "reports_icml/tables/disagreement_definition_comparison.csv"
    comp.to_csv(comp_path, index=False)
    with open(comp_path, "a", encoding="utf-8") as f:
        f.write("\n# Pairwise Jaccard overlap\n")
    jac.to_csv(comp_path, mode="a")
    with open(comp_path, "a", encoding="utf-8") as f:
        f.write("\n# Pairwise intersection counts\n")
    inter.to_csv(comp_path, mode="a")
    print(comp.to_string(index=False))
    print("\nJaccard:\n", jac)

    # --- Figure: prevalence bar + Jaccard heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar([n.split("_")[0] for n in names],
                [defs[n].mean() for n in names],
                color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("Disagreement prevalence by definition")
    axes[0].set_ylabel("fraction of pool flagged")
    for i, n in enumerate(names):
        axes[0].text(i, defs[n].mean(), f"{defs[n].sum()}", ha="center", va="bottom")

    im = axes[1].imshow(jac.values.astype(float), cmap="viridis", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_yticks(range(len(names)))
    short = [n.split("_")[0] for n in names]
    axes[1].set_xticklabels(short)
    axes[1].set_yticklabels(short)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[1].text(j, i, f"{jac.values[i, j]:.2f}",
                         ha="center", va="center", color="w")
    axes[1].set_title("Pairwise Jaccard overlap")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig_path = "reports_icml/figures/disagreement_definition_overlap.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # --- Definition D: 200-row manual annotation sample ---
    candidate = defA | defB | defC
    cand_idx = np.where(candidate)[0]
    rng = np.random.RandomState(CV_SEED)
    pick = rng.choice(cand_idx, size=min(ANNOTATION_N, len(cand_idx)),
                      replace=False)
    ann = pd.DataFrame({
        "uid": df["uid"].values[pick],
        "review_text": df["review_text"].values[pick],
        "star_rating": df["rating"].values[pick],
        "binary_label": y[pick],
        "text_only_prediction": cv["text_pred"][pick],
        "text_only_confidence": np.round(cv["text_conf"][pick], 4),
        "metadata_only_prediction": cv["meta_pred"][pick],
        "metadata_only_confidence": np.round(cv["meta_conf"][pick], 4),
        "multimodal_prediction": cv["mm_pred"][pick],
        "multimodal_confidence": np.round(cv["mm_conf"][pick], 4),
        "flag_A_sentiment_vs_label": defA[pick].astype(int),
        "flag_B_cross_model": defB[pick].astype(int),
        "flag_C_highconf_contradiction": defC[pick].astype(int),
        # blank columns for human labeling (DO NOT fabricate)
        "genuine_conflict": "",
        "label_noise": "",
        "sarcasm": "",
        "mixed_sentiment": "",
        "unclear": "",
        "notes": "",
    })
    ann_path = "reports_icml/manual_annotation_sample.csv"
    ann.to_csv(ann_path, index=False)
    print(f"\n[save] {comp_path}\n[save] {fig_path}\n[save] {ann_path} "
          f"({len(ann)} rows)")

    ic.log_experiment(dict(
        phase=5, dataset="Amazon-Reviews-2023", category=CATEGORY,
        seed=CV_SEED, split_id=f"{CATEGORY}_5foldCV",
        model="text_only+metadata_lc+early_fusion (cross-fit)",
        features="sbert+metadata_lc",
        includes_product_avg_rating=False,
        disagreement_labels_in_training=False,
        command="python -m scripts_icml.phase5_disagreement_definitions",
        output_files=f"{comp_path},{fig_path},{ann_path}",
        key_metrics=(f"prevA={defA.mean():.3f},prevB={defB.mean():.3f},"
                     f"prevC={defC.mean():.3f},"
                     f"jac_AB={jac.loc[names[0],names[1]]},"
                     f"jac_AC={jac.loc[names[0],names[2]]},"
                     f"jac_BC={jac.loc[names[1],names[2]]}"),
    ))


if __name__ == "__main__":
    main()
