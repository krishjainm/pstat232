"""Multi-seed robustness for the multimodal fusion model.

Trains the early-fusion multimodal model and the best disagreement-aware variant
(fixed w=5) under seeds {42, 123, 456} and reports mean +/- std for each metric.

Scope / honesty: the data split and the SBERT/metadata features are held fixed
(we reuse the materialized Appliances split and cached embeddings). Only the
multimodal model's stochastic training (weight init + minibatch order) is varied
across seeds. This therefore measures *training* variance, a lower bound on total
variance; full seed-level variance would also re-draw the split (requires a
provisioned rerun of the whole pipeline). This is stated in the outputs.

Outputs:
  reports/tables/multiseed_results.csv
  reports/figures/multiseed_error_bars.png
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.models.train_multimodal_model import MultimodalMLP
from src.features.build_metadata_features import get_available_metadata
from src.evaluation.calibration import expected_calibration_error

SEEDS = [42, 123, 456]
EPOCHS, BATCH, LR = 15, 64, 1e-3
EMB_DIR = "data/interim"


def load_features():
    tr = pd.read_parquet("data/processed/train.parquet")
    te = pd.read_parquet("data/processed/test.parquet")
    emb_tr = np.load(os.path.join(EMB_DIR, "sbert_train.npy"))
    emb_te = np.load(os.path.join(EMB_DIR, "sbert_test.npy"))
    feat_cols = get_available_metadata(tr)
    scaler = StandardScaler().fit(tr[feat_cols].values)
    Xtr = np.hstack([emb_tr, scaler.transform(tr[feat_cols].values)]).astype(np.float32)
    Xte = np.hstack([emb_te, scaler.transform(te[feat_cols].values)]).astype(np.float32)
    return tr, te, Xtr, Xte


def train(Xtr, ytr, wtr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultimodalMLP(Xtr.shape[1], (256, 64), 0.2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)
    wt = torch.tensor(wtr, dtype=torch.float32)
    n = len(yt)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            logits = model(Xt[idx])
            ce = -torch.log_softmax(logits, 1).gather(1, yt[idx].unsqueeze(1)).squeeze(1)
            (ce * wt[idx]).mean().backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(torch.tensor(Xte_g)), 1).numpy()[:, 1]
    return prob


def metrics(te, prob):
    y = te["label"].values
    pred = (prob >= 0.5).astype(int)
    dis = (te["agreement_status"] == "disagreement").values
    strong = (te["disagreement_group"] == "strong_disagreement").values
    e, _, _, _ = expected_calibration_error(y, prob, 10)
    return {
        "overall_acc": accuracy_score(y, pred),
        "disagree_acc": accuracy_score(y[dis], pred[dis]),
        "strong_disagree_acc": accuracy_score(y[strong], pred[strong]),
        "overall_f1": f1_score(y, pred, zero_division=0),
        "overall_auroc": roc_auc_score(y, prob),
        "overall_ece": e,
    }


def main():
    global Xte_g
    tr, te, Xtr, Xte = load_features()
    Xte_g = Xte
    ytr = tr["label"].values
    methods = {
        "Standard": np.ones(len(ytr), dtype=np.float32),
        "DA (w=5)": np.where(tr["agreement_status"].values == "disagreement", 5.0, 1.0).astype(np.float32),
    }

    rows = []
    for mname, wtr in methods.items():
        for seed in SEEDS:
            prob = train(Xtr, ytr, wtr, seed)
            m = metrics(te, prob)
            m.update({"method": mname, "seed": seed})
            rows.append(m)
            print(f"  {mname} seed={seed}: acc={m['overall_acc']:.4f} dis={m['disagree_acc']:.4f}")

    df = pd.DataFrame(rows)
    metric_cols = ["overall_acc", "disagree_acc", "strong_disagree_acc",
                   "overall_f1", "overall_auroc", "overall_ece"]
    agg = df.groupby("method")[metric_cols].agg(["mean", "std"]).round(4)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = len(SEEDS)
    agg["data_split"] = "Appliances (materialized); training-only variance"

    os.makedirs("reports/tables", exist_ok=True)
    # save both per-seed and aggregated
    df.round(4).to_csv("reports/tables/multiseed_results.csv", index=False)
    agg.to_csv("reports/tables/multiseed_results_aggregated.csv", index=False)
    print("\nPer-seed -> reports/tables/multiseed_results.csv")
    print(agg.to_string(index=False))

    # error-bar figure
    plt.figure(figsize=(9, 5))
    labels = ["overall_acc", "disagree_acc", "strong_disagree_acc", "overall_auroc"]
    x = np.arange(len(labels))
    w = 0.35
    for k, mname in enumerate(methods):
        means = [df[df["method"] == mname][c].mean() for c in labels]
        stds = [df[df["method"] == mname][c].std() for c in labels]
        plt.bar(x + (k - 0.5) * w, means, w, yerr=stds, capsize=4, label=mname)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Score")
    plt.ylim(0.5, 1.0)
    plt.title(f"Multi-seed robustness (mean +/- std over seeds {SEEDS}; Appliances split, training variance)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/multiseed_error_bars.png", dpi=150)
    plt.close()
    print("Wrote reports/figures/multiseed_error_bars.png")


if __name__ == "__main__":
    main()
