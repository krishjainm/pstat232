"""Ablation study aggregation for the research upgrade.

Combines three real ablations into one table + figure:

  A/B. Metadata feature-set ablation (full / no product_average_rating /
       no product-level) -- computed in upgrade_leakage_audit.py via 5-fold
       stratified CV on the All_Beauty test partition.
  C.   Text-embedding ablation (SBERT+LogReg vs TF-IDF+LogReg), trained/evaluated
       on the materialized Appliances split (reuses cached SBERT embeddings).
  D.   Fusion-architecture ablation (early / late / gated) -- from the committed,
       real `fusion_comparison_results.csv` (All_Beauty test set).

distilbert-base-uncased embeddings were not run (extra heavy CPU download/encode);
this is noted rather than fabricated.

Outputs:
  reports/tables/ablation_results.csv
  reports/figures/ablation_summary.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

EMB_DIR = "data/interim"
META_JSON = "reports/tables/_metadata_ablation_rows.json"
FUSION_CSV = "reports/tables/fusion_comparison_results.csv"
OUT_CSV = "reports/tables/ablation_results.csv"
FIG = "reports/figures/ablation_summary.png"


def text_embedding_ablation():
    """SBERT+LogReg vs TF-IDF+LogReg on the materialized Appliances split."""
    tr = pd.read_parquet("data/processed/train.parquet")
    te = pd.read_parquet("data/processed/test.parquet")
    ytr, yte = tr["label"].values, te["label"].values
    dis = (te["agreement_status"] == "disagreement").values
    rows = []

    # SBERT (reuse cached embeddings from the mitigation run if present)
    emb_tr = os.path.join(EMB_DIR, "sbert_train.npy")
    emb_te = os.path.join(EMB_DIR, "sbert_test.npy")
    if os.path.exists(emb_tr) and os.path.exists(emb_te):
        Xtr, Xte = np.load(emb_tr), np.load(emb_te)
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        Xtr = m.encode(tr["review_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
        Xte = m.encode(te["review_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    rows.append({"ablation": "text_embedding", "variant": "sbert_minilm_logreg",
                 "data": "Appliances test", "metric_type": "test_accuracy",
                 "accuracy": round(accuracy_score(yte, pred), 4),
                 "disagree_acc": round(accuracy_score(yte[dis], pred[dis]), 4),
                 "note": "all-MiniLM-L6-v2 embeddings + logistic regression"})

    # TF-IDF
    vec = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
    Xtr_t = vec.fit_transform(tr["review_text"].values)
    Xte_t = vec.transform(te["review_text"].values)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(Xtr_t, ytr)
    pred = clf.predict(Xte_t)
    rows.append({"ablation": "text_embedding", "variant": "tfidf_logreg",
                 "data": "Appliances test", "metric_type": "test_accuracy",
                 "accuracy": round(accuracy_score(yte, pred), 4),
                 "disagree_acc": round(accuracy_score(yte[dis], pred[dis]), 4),
                 "note": "TF-IDF (1,2)-grams + logistic regression"})
    return rows


def main():
    rows = []

    # A/B metadata feature-set ablation
    if os.path.exists(META_JSON):
        for r in json.load(open(META_JSON)):
            rows.append({"ablation": "metadata_feature_set", "variant": r["variant"],
                         "data": "All_Beauty test (5-fold CV)", "metric_type": "cv_accuracy",
                         "accuracy": r["cv_accuracy_mean"], "disagree_acc": np.nan,
                         "note": f"{r['n_features']} feats; std={r['cv_accuracy_std']}"})

    # C text-embedding ablation
    print("Running text-embedding ablation...")
    rows += text_embedding_ablation()

    # D fusion-architecture ablation (real, committed)
    if os.path.exists(FUSION_CSV):
        fc = pd.read_csv(FUSION_CSV)
        for _, r in fc.iterrows():
            rows.append({"ablation": "fusion_architecture", "variant": r["fusion_type"],
                         "data": "All_Beauty test", "metric_type": "test_accuracy",
                         "accuracy": round(float(r["accuracy"]), 4),
                         "disagree_acc": round(float(r["disagree_accuracy"]), 4),
                         "note": "from fusion_comparison_results.csv"})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print(out.to_string(index=False))

    # figure: grouped by ablation
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, abl, title in zip(
        axes,
        ["metadata_feature_set", "text_embedding", "fusion_architecture"],
        ["Metadata feature set\n(All_Beauty CV acc)", "Text embedding\n(Appliances test acc)",
         "Fusion architecture\n(All_Beauty test acc)"],
    ):
        sub = out[out["ablation"] == abl]
        ax.barh(sub["variant"][::-1], sub["accuracy"][::-1], color="#4C72B0")
        ax.set_xlim(0.4, 1.0)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(sub["accuracy"][::-1]):
            ax.annotate(f"{v:.3f}", (v + 0.005, i), va="center", fontsize=8)
    fig.suptitle("Ablation summary (each panel uses the data noted; not directly comparable across panels)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG, dpi=150)
    plt.close()
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
