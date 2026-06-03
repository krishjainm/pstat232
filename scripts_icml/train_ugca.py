"""Train UGCA-Fusion (Uncertainty-Guided Conflict-Aware Fusion).

Provides reusable `train_ugca` / `predict_ugca` helpers (used by the Phase-4
benchmark in evaluate_ugca.py) and a CLI to train a single model on a category
pool seed and save it under models_icml/.

Usage:
    python -m scripts_icml.train_ugca --category All_Beauty --seed 42
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.ugca_fusion import UGCAFusion, ugca_loss
from scripts_icml import icml_common as C


DEFAULT_LAMBDAS = dict(lambda_text=0.3, lambda_meta=0.3,
                       lambda_conflict=0.5, lambda_cal=0.2)


def train_ugca(text_tr, meta_tr, y_tr, disagree_tr,
               text_va, meta_va, y_va, seed,
               flags=None, lambdas=None, epochs=25, batch=64, lr=1e-3,
               hidden=128, dropout=0.2):
    """Train a UGCA-Fusion model; early-stop on val accuracy. Returns the model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    flags = flags or {}
    lambdas = {**DEFAULT_LAMBDAS, **(lambdas or {})}

    model = UGCAFusion(
        text_dim=text_tr.shape[1], meta_dim=meta_tr.shape[1],
        hidden=hidden, dropout=dropout,
        use_conflict_detector=flags.get("use_conflict_detector", True),
        use_entropy=flags.get("use_entropy", True),
        use_js=flags.get("use_js", True),
        use_learned_conflict_features=flags.get("use_learned_conflict_features", True),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tt = torch.tensor(text_tr, dtype=torch.float32)
    mt = torch.tensor(meta_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    dt = torch.tensor(disagree_tr, dtype=torch.float32)
    ds = TensorDataset(tt, mt, yt, dt)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    tva = torch.tensor(text_va, dtype=torch.float32)
    mva = torch.tensor(meta_va, dtype=torch.float32)

    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, mb, yb, db in dl:
            opt.zero_grad()
            out = model(xb, mb)
            loss = ugca_loss(
                out, yb, db, **lambdas,
                use_calibration_loss=flags.get("use_calibration_loss", True),
                use_conflict_detector=flags.get("use_conflict_detector", True))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_pred = model(tva, mva)["logit_fused"].argmax(1).numpy()
        acc = (va_pred == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_ugca(model, text, meta):
    """Return pred, prob_positive, c_hat, gate (numpy) on a test set."""
    with torch.no_grad():
        out = model(torch.tensor(text, dtype=torch.float32),
                    torch.tensor(meta, dtype=torch.float32))
        prob = torch.softmax(out["logit_fused"], dim=1).numpy()[:, 1]
        c_hat = out["c_hat"].numpy()
        gate = out["gate"].numpy()
    pred = (prob >= 0.5).astype(int)
    return pred, prob, c_hat, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="All_Beauty")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    pool = C.build_canonical_pool(args.category)
    emb = C.load_sbert(args.category)
    tr, va, te = C.make_split(pool, args.seed)
    train_df, val_df, test_df = pool.iloc[tr], pool.iloc[va], pool.iloc[te].reset_index(drop=True)
    lc_cols = [c for c in C.METADATA_LEAKAGE_CONTROLLED if c in pool.columns]
    ml_tr, ml_va, ml_te = C.scale_meta(train_df, [train_df, val_df, test_df], lc_cols)
    disagree_tr = (train_df["agreement_status"].values == "disagreement").astype(np.float32)

    model = train_ugca(emb[tr], ml_tr, train_df["label"].values, disagree_tr,
                       emb[va], ml_va, val_df["label"].values, args.seed,
                       epochs=args.epochs)
    pred, prob, c_hat, gate = predict_ugca(model, emb[te], ml_te)
    m = C.compute_metric_suite(test_df, pred, prob)
    print(f"[UGCA {args.category} seed {args.seed}] acc={m['accuracy']:.3f} "
          f"disag_acc={m['disagreement_acc']:.3f} ece={m['ece']:.3f} "
          f"cal_gap={m['calibration_gap']:.3f}")

    os.makedirs(f"models_icml/ugca_{args.category}_s{args.seed}", exist_ok=True)
    torch.save(model.state_dict(),
               f"models_icml/ugca_{args.category}_s{args.seed}/model.pt")
    print("saved model.")


if __name__ == "__main__":
    main()
