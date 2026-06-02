"""Mitigation method comparison for the research upgrade.

Compares simple, honest mitigation strategies for the disagreement failure mode:
  A. Fixed disagreement reweighting (w = 2, 3, 5, 8)
  B. Severity-aware reweighting (agree=1, weak=2, medium=4, strong=6)
  C. Focal loss (gamma = 1, 2; and focal-gamma2 + DA-w3)
  D. Post-hoc calibration on the standard model (global vs group-wise temperature)

Data provenance: this uses the materialized processed split currently on disk.
That split is the Appliances category (a real Amazon category with full metadata
and disagreement labels). We run the comparison here because it has a complete
train/val/test split on disk; the paper's primary disagreement-aware numbers
remain those reported for All_Beauty. Running the SAME comparison on a second
category is itself evidence about whether the mitigation transfers.

Every metric is computed from a real trained model. We report the full tradeoff
(overall vs disagreement vs calibration); no method is selected by a single metric.

Outputs:
  reports/tables/mitigation_comparison.csv
  reports/figures/mitigation_tradeoff.png
  reports/figures/mitigation_disagreement_accuracy.png
  reports/figures/mitigation_calibration.png
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

from src.models.train_multimodal_model import MultimodalMLP
from src.features.build_metadata_features import get_available_metadata
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.temperature_scaling import (
    find_optimal_temperature, apply_temperature_scaling,
)

SEED = 42
EPOCHS = 15
BATCH = 64
LR = 1e-3
EMB_DIR = "data/interim"
SBERT_MODEL = "all-MiniLM-L6-v2"

SEVERITY_WEIGHTS = {"agreement": 1.0, "weak_disagreement": 2.0,
                    "medium_disagreement": 4.0, "strong_disagreement": 6.0}


def sbert_encode_cached(texts, split):
    path = os.path.join(EMB_DIR, f"sbert_{split}.npy")
    meta = os.path.join(EMB_DIR, f"sbert_{split}.sha")
    h = hashlib.sha1(("||".join(map(str, texts[:50])) + f"|{len(texts)}").encode()).hexdigest()
    if os.path.exists(path) and os.path.exists(meta):
        with open(meta) as f:
            if f.read().strip() == h:
                return np.load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL)
    emb = model.encode(list(texts), batch_size=128, show_progress_bar=True,
                       normalize_embeddings=True)
    os.makedirs(EMB_DIR, exist_ok=True)
    np.save(path, emb)
    with open(meta, "w") as f:
        f.write(h)
    return emb


def build_features():
    tr = pd.read_parquet("data/processed/train.parquet")
    va = pd.read_parquet("data/processed/val.parquet")
    te = pd.read_parquet("data/processed/test.parquet")

    feat_cols = get_available_metadata(tr)
    scaler = StandardScaler().fit(tr[feat_cols].values)

    def feats(df, split):
        emb = sbert_encode_cached(df["review_text"].tolist(), split)
        meta = scaler.transform(df[feat_cols].values)
        return np.hstack([emb, meta]).astype(np.float32)

    Xtr, Xva, Xte = feats(tr, "train"), feats(va, "val"), feats(te, "test")
    return (tr, va, te, Xtr, Xva, Xte)


def train_model(Xtr, ytr, wtr, focal_gamma=0.0, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultimodalMLP(input_dim=Xtr.shape[1], hidden_dims=(256, 64), dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    Xt = torch.tensor(Xtr)
    yt = torch.tensor(ytr, dtype=torch.long)
    wt = torch.tensor(wtr, dtype=torch.float32)
    n = len(yt)

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb, wb = Xt[idx], yt[idx], wt[idx]
            opt.zero_grad()
            logits = model(xb)
            logp = torch.log_softmax(logits, dim=1)
            ce = -logp.gather(1, yb.unsqueeze(1)).squeeze(1)
            if focal_gamma > 0:
                pt = torch.exp(-ce)
                loss_i = (1 - pt) ** focal_gamma * ce
            else:
                loss_i = ce
            loss = (loss_i * wb).mean()
            loss.backward()
            opt.step()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs = torch.softmax(logits, dim=1).numpy()
    return probs[:, 1]


def evaluate(te, prob_pos, method, extra=None):
    y = te["label"].values
    pred = (prob_pos >= 0.5).astype(int)
    dis = (te["agreement_status"] == "disagreement").values
    strong = (te["disagreement_group"] == "strong_disagreement").values

    def ece(mask):
        if mask.sum() == 0:
            return np.nan
        e, _, _, _ = expected_calibration_error(y[mask], prob_pos[mask], 10)
        return e

    row = {
        "method": method,
        "overall_acc": accuracy_score(y, pred),
        "disagree_acc": accuracy_score(y[dis], pred[dis]),
        "strong_disagree_acc": accuracy_score(y[strong], pred[strong]),
        "overall_f1": f1_score(y, pred, zero_division=0),
        "disagree_f1": f1_score(y[dis], pred[dis], zero_division=0),
        "overall_auroc": roc_auc_score(y, prob_pos),
        "overall_ece": ece(np.ones(len(y), dtype=bool)),
        "disagree_ece": ece(dis),
        "overall_brier": brier_score_loss(y, prob_pos),
    }
    if extra:
        row.update(extra)
    return row


def main():
    print("Building features (SBERT + metadata) for the materialized split...")
    tr, va, te, Xtr, Xva, Xte = build_features()
    ytr = tr["label"].values
    yva = va["label"].values
    print(f"Train {Xtr.shape}, Val {Xva.shape}, Test {Xte.shape}")

    rows = []
    standard_prob = None

    # ---- A. fixed disagreement reweighting + baseline ----
    configs = [("Standard (w=1)", 1.0)] + [(f"Fixed DA (w={w})", w) for w in (2, 3, 5, 8)]
    for name, w in configs:
        wtr = np.where(tr["agreement_status"].values == "disagreement", w, 1.0).astype(np.float32)
        model = train_model(Xtr, ytr, wtr)
        prob = predict(model, Xte)
        if name.startswith("Standard"):
            standard_prob = prob
            standard_val_prob = predict(model, Xva)
        rows.append(evaluate(te, prob, name))
        print(f"  {name}: overall={rows[-1]['overall_acc']:.4f} disagree={rows[-1]['disagree_acc']:.4f}")

    # ---- B. severity-aware reweighting ----
    wtr = tr["disagreement_group"].map(SEVERITY_WEIGHTS).fillna(1.0).values.astype(np.float32)
    model = train_model(Xtr, ytr, wtr)
    rows.append(evaluate(te, predict(model, Xte), "Severity-aware (1/2/4/6)"))
    print(f"  Severity-aware: overall={rows[-1]['overall_acc']:.4f} disagree={rows[-1]['disagree_acc']:.4f}")

    # ---- C. focal loss ----
    for g in (1.0, 2.0):
        ones = np.ones(len(ytr), dtype=np.float32)
        model = train_model(Xtr, ytr, ones, focal_gamma=g)
        rows.append(evaluate(te, predict(model, Xte), f"Focal (gamma={int(g)})"))
        print(f"  Focal g={g}: overall={rows[-1]['overall_acc']:.4f} disagree={rows[-1]['disagree_acc']:.4f}")
    # focal + DA
    wtr = np.where(tr["agreement_status"].values == "disagreement", 3.0, 1.0).astype(np.float32)
    model = train_model(Xtr, ytr, wtr, focal_gamma=2.0)
    rows.append(evaluate(te, predict(model, Xte), "Focal(g=2)+DA(w=3)"))
    print(f"  Focal+DA: overall={rows[-1]['overall_acc']:.4f} disagree={rows[-1]['disagree_acc']:.4f}")

    # ---- D. post-hoc temperature scaling on the standard model ----
    # global temperature learned on val
    T_global = find_optimal_temperature(yva, standard_val_prob, 10)
    prob_glob = apply_temperature_scaling(standard_prob, T_global)
    rows.append(evaluate(te, prob_glob, "Standard + global temp",
                         extra={"note": f"T={T_global:.3f}; preds unchanged vs Standard"}))
    # group-wise temperature learned on val groups
    Tg = {}
    for st in ["agreement", "disagreement"]:
        m = (va["agreement_status"] == st).values
        Tg[st] = find_optimal_temperature(yva[m], standard_val_prob[m], 10) if m.sum() >= 20 else T_global
    prob_grp = standard_prob.copy()
    for st in ["agreement", "disagreement"]:
        m = (te["agreement_status"] == st).values
        prob_grp[m] = apply_temperature_scaling(standard_prob[m], Tg[st])
    rows.append(evaluate(te, prob_grp, "Standard + groupwise temp",
                         extra={"note": f"T_agree={Tg['agreement']:.3f}, T_dis={Tg['disagreement']:.3f}"}))

    out = pd.DataFrame(rows)
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4)
    out.insert(1, "data_split", "Appliances (materialized split)")
    os.makedirs("reports/tables", exist_ok=True)
    out.to_csv("reports/tables/mitigation_comparison.csv", index=False)
    print("\nWrote reports/tables/mitigation_comparison.csv")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(out.drop(columns=["data_split"]).to_string(index=False))

    _figures(out)


def _figures(out):
    fig_dir = "reports/figures"
    train_methods = out[~out["method"].str.contains("temp")].copy()

    # tradeoff: overall vs disagreement accuracy
    plt.figure(figsize=(7.5, 5.5))
    for _, r in train_methods.iterrows():
        plt.scatter(r["overall_acc"], r["disagree_acc"], s=70)
        plt.annotate(r["method"], (r["overall_acc"], r["disagree_acc"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Overall accuracy")
    plt.ylabel("Disagreement accuracy")
    plt.title("Mitigation tradeoff: overall vs disagreement accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/mitigation_tradeoff.png", dpi=150)
    plt.close()

    # disagreement accuracy bars (disagree + strong)
    x = np.arange(len(train_methods))
    w = 0.4
    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, train_methods["disagree_acc"], w, label="Disagreement acc", color="#C44E52")
    plt.bar(x + w / 2, train_methods["strong_disagree_acc"], w, label="Strong-disagreement acc", color="#8172B3")
    plt.xticks(x, train_methods["method"], rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Disagreement & strong-disagreement accuracy by mitigation method")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/mitigation_disagreement_accuracy.png", dpi=150)
    plt.close()

    # calibration: overall vs disagreement ECE per method
    x = np.arange(len(out))
    plt.figure(figsize=(11, 5))
    plt.bar(x - w / 2, out["overall_ece"], w, label="Overall ECE", color="#4C72B0")
    plt.bar(x + w / 2, out["disagree_ece"], w, label="Disagreement ECE", color="#DD8452")
    plt.xticks(x, out["method"], rotation=30, ha="right")
    plt.ylabel("ECE")
    plt.title("Calibration by mitigation method (lower is better)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/mitigation_calibration.png", dpi=150)
    plt.close()
    print("Wrote mitigation figures (tradeoff, disagreement_accuracy, calibration)")


if __name__ == "__main__":
    main()
