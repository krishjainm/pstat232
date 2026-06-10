"""Shared computational utilities for the disagreement-aware fusion study.

This module centralizes:
  * canonical data preparation (download -> preprocess -> disagreement -> SBERT cache)
  * leakage-controlled vs full metadata feature sets
  * CPU-light per-seed model training on cached SBERT embeddings
  * the full conflict-aware metric suite
  * an experiment-registry logger

Design choice (documented honestly): the *balanced pool* is built once with a
fixed master seed; only the train/val/test *split* is redrawn per seed. This is
the standard "fixed data, multiple random splits" protocol. It keeps the
expensive SBERT encoding and transformer-sentiment labeling to a single pass
(feasible on CPU) while still exposing split-level variance across seeds.
"""

from __future__ import annotations

import os
import json
import hashlib
import datetime as _dt

import numpy as np
import pandas as pd

# Validated preprocessing helpers (read-only use).
from src.data.preprocess import (
    create_binary_label,
    merge_metadata,
    build_metadata_features,
    select_final_columns,
    balance_classes,
)

DATASET = "McAuley-Lab/Amazon-Reviews-2023"
MASTER_SEED = 2024  # fixes the balanced pool; splits are redrawn per run-seed

# Metadata feature sets ------------------------------------------------------
# Full set (includes the leakage-prone product-level rating aggregate).
METADATA_FULL = [
    "review_length_words",
    "review_length_chars",
    "helpful_vote",
    "verified_purchase",
    "product_average_rating",
    "product_rating_number",
    "product_price",
    "review_year",
]
# Leakage-controlled set removes product-level aggregates that can encode the
# review-rating label (product_average_rating) and the prior-driven count
# (product_rating_number). Price and year are kept (not rating aggregates).
LEAKAGE_DROP = ["product_average_rating", "product_rating_number"]
METADATA_LEAKAGE_CONTROLLED = [c for c in METADATA_FULL if c not in LEAKAGE_DROP]

# Disagreement thresholds (match config/config.yaml).
WEAK_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.90  # >= medium == strong_disagreement


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def raw_dir(category):
    return _ensure(os.path.join("data_pool", "raw", category))


def interim_dir(category):
    return _ensure(os.path.join("data_pool", "interim", category))


def pool_path(category):
    return os.path.join(interim_dir(category), "canonical_pool.parquet")


def sbert_path(category):
    return os.path.join(interim_dir(category), "sbert.npy")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_category(category, max_raw=160000, seed=MASTER_SEED,
                      max_read_rows=500000):
    """Download reviews + metadata for a category, subsample reviews to max_raw.

    For very large categories (multi-GB review files), reading the entire file
    is infeasible on modest hardware; we stop after `max_read_rows` rows and
    subsample from those. This introduces file-order bias for huge categories,
    which is documented as a limitation.
    """
    from huggingface_hub import hf_hub_download

    rdir = raw_dir(category)
    rev_out = os.path.join(rdir, "reviews.parquet")
    meta_out = os.path.join(rdir, "metadata.parquet")

    if os.path.exists(rev_out) and os.path.exists(meta_out):
        print(f"[download] cached parquet for {category}")
        return rev_out, meta_out

    print(f"[download] reviews for {category} ...")
    rev_file = hf_hub_download(
        repo_id=DATASET,
        filename=f"raw/review_categories/{category}.jsonl",
        repo_type="dataset",
    )
    # Chunked read so very large files do not blow up memory; cap rows read.
    chunks = []
    total = 0
    capped = False
    for chunk in pd.read_json(rev_file, lines=True, chunksize=100000):
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_read_rows:
            capped = True
            break
    reviews = pd.concat(chunks, ignore_index=True)
    print(f"[download] {category}: read {len(reviews)} rows"
          + (f" (capped at {max_read_rows})" if capped else ""))
    if len(reviews) > max_raw:
        reviews = reviews.sample(n=max_raw, random_state=seed).reset_index(drop=True)
        print(f"[download] subsampled to {len(reviews)}")
    reviews.to_parquet(rev_out, index=False)

    print(f"[download] metadata for {category} ...")
    try:
        meta_file = hf_hub_download(
            repo_id=DATASET,
            filename=f"raw/meta_categories/meta_{category}.jsonl",
            repo_type="dataset",
        )
        # Cap metadata rows read so multi-GB meta files do not OOM. Products
        # beyond the cap get NaN metadata (median-filled later); documented.
        meta_chunks, mtot = [], 0
        for ch in pd.read_json(meta_file, lines=True, chunksize=100000):
            keep = [c for c in ["parent_asin", "average_rating", "rating_number",
                                "price", "main_category"] if c in ch.columns]
            meta_chunks.append(ch[keep])
            mtot += len(ch)
            if mtot >= max_read_rows:
                break
        meta = pd.concat(meta_chunks, ignore_index=True)
        meta.to_parquet(meta_out, index=False)
        print(f"[download] metadata rows: {len(meta)}")
    except Exception as e:  # pragma: no cover
        print(f"[download] WARNING no metadata: {e}")
        pd.DataFrame({"parent_asin": []}).to_parquet(meta_out, index=False)

    return rev_out, meta_out


# ---------------------------------------------------------------------------
# Canonical pool: preprocess + balance + disagreement + SBERT cache
# ---------------------------------------------------------------------------
def _cfg_for_preprocess():
    """Minimal cfg dict matching what the preprocess functions need."""
    return {
        "data": {"positive_threshold": 4, "negative_threshold": 2},
    }


def build_canonical_pool(category, pool_size=20000, force=False):
    """Create (or load) the fixed balanced pool with disagreement labels + SBERT.

    Returns the pool DataFrame; SBERT embeddings are cached at sbert_path().
    """
    ppath = pool_path(category)
    spath = sbert_path(category)
    if os.path.exists(ppath) and os.path.exists(spath) and not force:
        print(f"[pool] cached for {category}")
        return pd.read_parquet(ppath)

    rev_out, meta_out = download_category(category)
    reviews = pd.read_parquet(rev_out)
    # Only load the lightweight metadata columns we actually use; the raw
    # metadata parquet has huge nested columns (images/videos/details) that
    # cause memory pressure and swap-thrashing during inference.
    meta_cols = ["parent_asin", "average_rating", "rating_number",
                 "price", "main_category"]
    try:
        meta = pd.read_parquet(meta_out, columns=meta_cols)
    except Exception:
        meta = pd.read_parquet(meta_out)
    if len(meta) == 0:
        meta = None

    cfg = _cfg_for_preprocess()
    df = create_binary_label(reviews, cfg)
    df = merge_metadata(df, meta)
    df = build_metadata_features(df)
    df = select_final_columns(df)
    # Free large intermediates before the heavy (CPU/memory) sentiment + SBERT.
    del reviews, meta
    import gc as _gc
    _gc.collect()

    # median-fill numeric metadata (mirror preprocess_pipeline)
    for col in ["product_average_rating", "product_rating_number",
                "product_price", "review_year"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Balance, then cap to pool_size (fixed master seed). Use explicit
    # per-class sampling (robust across pandas versions).
    df = balance_classes(df, seed=MASTER_SEED)
    if len(df) > pool_size:
        per = pool_size // 2
        parts = []
        for lab in sorted(df["label"].unique()):
            g = df[df["label"] == lab]
            parts.append(g.sample(n=min(per, len(g)), random_state=MASTER_SEED))
        df = pd.concat(parts).sample(frac=1, random_state=MASTER_SEED)
    df = df.reset_index(drop=True)
    df["uid"] = np.arange(len(df))

    # --- Checkpoint A: sentiment (the most expensive step). Saved immediately
    #     after scoring so a later crash never wastes the compute. ---
    sent_ckpt = os.path.join(interim_dir(category), "sentiment_ckpt.parquet")
    if os.path.exists(sent_ckpt):
        prev = pd.read_parquet(sent_ckpt)
        if len(prev) == len(df):
            print(f"[pool] reusing sentiment checkpoint {sent_ckpt}")
            df["text_sentiment_label"] = prev["text_sentiment_label"].values
            df["text_sentiment_confidence"] = prev["text_sentiment_confidence"].values
        else:
            prev = None
    else:
        prev = None
    if "text_sentiment_label" not in df.columns:
        print(f"[pool] scoring text sentiment on {len(df)} reviews (one-time)...")
        labels, confs = compute_text_sentiment_fast(df["review_text"].tolist())
        df["text_sentiment_label"] = labels
        df["text_sentiment_confidence"] = confs
        df[["uid", "text_sentiment_label", "text_sentiment_confidence"]].to_parquet(
            sent_ckpt, index=False)
        print(f"[pool] saved sentiment checkpoint -> {sent_ckpt}")

    df = _assign_disagreement(df)

    # --- Checkpoint B: SBERT embeddings ---
    if os.path.exists(spath) and np.load(spath, mmap_mode="r").shape[0] == len(df):
        print(f"[pool] reusing SBERT cache {spath}")
    else:
        print(f"[pool] encoding SBERT embeddings for {len(df)} reviews (one-time)...")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        emb = enc.encode(df["review_text"].tolist(), batch_size=128,
                         show_progress_bar=True, normalize_embeddings=True)
        np.save(spath, emb.astype(np.float32))

    df.to_parquet(ppath, index=False)
    print(f"[pool] saved {ppath} ({df.shape})")
    print(df["disagreement_group"].value_counts())
    return df


def compute_text_sentiment_fast(texts, char_cap=300, batch_size=64):
    """Fast pretrained-sentiment proxy (Definition A).

    Long reviews dominate CPU cost; we truncate each review to the first
    `char_cap` characters before scoring with DistilBERT-SST-2. Review
    sentiment is overwhelmingly expressed early, so this is a documented,
    compute-bounded approximation of the proxy disagreement signal (the
    disagreement label is itself a proxy, see limitations).
    """
    # Force offline (model is cached) so pipeline construction never stalls on
    # an HF Hub network round-trip, and cap CPU threads to avoid oversubscription.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    except Exception:
        pass

    from transformers import pipeline

    print("[sentiment] constructing pipeline (offline)...", flush=True)
    pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
        truncation=True,
        max_length=96,
    )
    print("[sentiment] pipeline ready; scoring...", flush=True)
    capped = [(t[:char_cap] if isinstance(t, str) and len(t) > char_cap else (t or "."))
              for t in texts]
    capped = [t if t.strip() else "." for t in capped]
    labels, confs = [], []
    n_batches = (len(capped) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(capped), batch_size)):
        for r in pipe(capped[i:i + batch_size]):
            labels.append(1 if r["label"] == "POSITIVE" else 0)
            confs.append(r["score"])
        if bi % 20 == 0 or bi == n_batches - 1:
            print(f"[sentiment] batch {bi+1}/{n_batches}", flush=True)
    return np.array(labels), np.array(confs)


def _assign_disagreement(df):
    df = df.copy()
    agrees = df["text_sentiment_label"] == df["label"]
    df["agreement_status"] = np.where(agrees, "agreement", "disagreement")
    conf = df["text_sentiment_confidence"]
    conditions = [
        agrees,
        (~agrees) & (conf < WEAK_THRESHOLD),
        (~agrees) & (conf >= WEAK_THRESHOLD) & (conf < MEDIUM_THRESHOLD),
        (~agrees) & (conf >= MEDIUM_THRESHOLD),
    ]
    choices = ["agreement", "weak_disagreement",
               "medium_disagreement", "strong_disagreement"]
    df["disagreement_group"] = np.select(conditions, choices, default="agreement")
    return df


def load_sbert(category):
    return np.load(sbert_path(category))


# ---------------------------------------------------------------------------
# Per-seed split (redrawn per seed)
# ---------------------------------------------------------------------------
def make_split(df, seed, train=0.70, val=0.15):
    """Stratified train/val/test split; returns boolean index arrays via 'uid'."""
    from sklearn.model_selection import train_test_split
    idx = df.index.values
    y = df["label"].values
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=(1 - train), random_state=seed, stratify=y)
    rel_test = (1 - train - val) / (1 - train)
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=rel_test, random_state=seed, stratify=y[tmp_idx])
    return tr_idx, va_idx, te_idx


def split_id(category, seed):
    return f"{category}_s{seed}_70-15-15"


# ---------------------------------------------------------------------------
# Metadata scaling helpers (per split, fit on train only)
# ---------------------------------------------------------------------------
def scale_meta(train_df, other_dfs, feature_cols):
    """Fit StandardScaler on train metadata; robust to residual NaNs.

    Some categories have a fully-missing metadata column (e.g. price or
    review_year), whose median is itself NaN, so median-fill leaves NaNs.
    We impute remaining NaNs with the train-column mean (or 0 if a column is
    entirely NaN) before scaling.
    """
    from sklearn.preprocessing import StandardScaler

    Xtr = train_df[feature_cols].to_numpy(dtype=float)
    col_means = np.nanmean(Xtr, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    def _impute(df):
        X = df[feature_cols].to_numpy(dtype=float)
        idx = np.where(~np.isfinite(X))
        if idx[0].size:
            X[idx] = np.take(col_means, idx[1])
        return X

    sc = StandardScaler().fit(_impute(train_df))
    return [sc.transform(_impute(d)) for d in other_dfs]


# ---------------------------------------------------------------------------
# Models (CPU-light; reuse cached SBERT embeddings)
# ---------------------------------------------------------------------------
def train_text_only(emb_tr, y_tr, emb_te, seed):
    """SBERT + LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(emb_tr, y_tr)
    prob = clf.predict_proba(emb_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def train_metadata_only(Xtr, y_tr, Xte, seed):
    """XGBoost on scaled metadata."""
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=seed, n_jobs=4, verbosity=0,
    )
    clf.fit(Xtr, y_tr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def _torch_fusion(fusion_type, text_tr, meta_tr, y_tr, text_va, meta_va, y_va,
                  text_te, meta_te, seed, sample_weights=None,
                  epochs=15, hidden=(256, 64), dropout=0.2, lr=1e-3, batch=64):
    """Train an early/late/gated fusion MLP; early-stop on val acc.

    Returns (pred, prob_positive) on the test set.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    text_dim = text_tr.shape[1]
    meta_dim = meta_tr.shape[1]

    Xtr = np.hstack([text_tr, meta_tr]).astype(np.float32)
    Xva = np.hstack([text_va, meta_va]).astype(np.float32)
    Xte = np.hstack([text_te, meta_te]).astype(np.float32)

    class EarlyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], text_dim + meta_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 2))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class LateMLP(nn.Module):
        def __init__(self):
            super().__init__()
            h = hidden[0]
            self.t = nn.Sequential(nn.Linear(text_dim, h), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(h, 2))
            self.m = nn.Sequential(nn.Linear(meta_dim, h), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(h, 2))

        def forward(self, x):
            return (self.t(x[:, :text_dim]) + self.m(x[:, text_dim:])) / 2

    class GatedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            h = hidden[0]
            self.t = nn.Sequential(nn.Linear(text_dim, h), nn.ReLU(), nn.Dropout(dropout))
            self.m = nn.Sequential(nn.Linear(meta_dim, h), nn.ReLU(), nn.Dropout(dropout))
            self.gate = nn.Sequential(nn.Linear(text_dim + meta_dim, 1), nn.Sigmoid())
            self.cls = nn.Linear(h, 2)

        def forward(self, x):
            g = self.gate(x)
            fused = g * self.t(x[:, :text_dim]) + (1 - g) * self.m(x[:, text_dim:])
            return self.cls(fused)

    model = {"early": EarlyMLP, "late": LateMLP, "gated": GatedMLP}[fusion_type]()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(reduction="none")

    w = np.ones(len(y_tr), dtype=np.float32) if sample_weights is None else sample_weights.astype(np.float32)
    tr_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(y_tr),
                          torch.tensor(w))
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)

    Xva_t = torch.tensor(Xva)
    Xte_t = torch.tensor(Xte)

    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb, wb in tr_dl:
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_pred = model(Xva_t).argmax(1).numpy()
        acc = (va_pred == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        prob = torch.softmax(logits, 1).numpy()[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def train_fusion(fusion_type, text_tr, meta_tr, y_tr, text_va, meta_va, y_va,
                 text_te, meta_te, seed, sample_weights=None):
    return _torch_fusion(fusion_type, text_tr, meta_tr, y_tr, text_va, meta_va,
                         y_va, text_te, meta_te, seed, sample_weights)


# ---------------------------------------------------------------------------
# Conflict-aware metric suite
# ---------------------------------------------------------------------------
def _ece(y_true, y_prob, n_bins=10):
    from src.evaluation.calibration import expected_calibration_error
    ece, *_ = expected_calibration_error(np.asarray(y_true), np.asarray(y_prob), n_bins)
    return ece


def compute_metric_suite(test_df, pred, prob, text_pred=None, meta_pred=None,
                         high_conf=0.90):
    """Compute the full conflict-aware metric dictionary for one model.

    test_df must include: label, agreement_status, disagreement_group.
    prob is P(class=1); pred is the hard prediction.
    text_pred / meta_pred (optional) enable modality-dominance %.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

    y = test_df["label"].values
    pred = np.asarray(pred)
    prob = np.asarray(prob)
    conf = np.maximum(prob, 1 - prob)
    correct = (pred == y)

    out = {}
    out["accuracy"] = accuracy_score(y, pred)
    out["f1"] = f1_score(y, pred, zero_division=0)
    out["auroc"] = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    out["ece"] = _ece(y, prob)
    out["brier"] = brier_score_loss(y, prob)

    agree = test_df["agreement_status"].values == "agreement"
    disag = ~agree
    strong = test_df["disagreement_group"].values == "strong_disagreement"

    out["agreement_acc"] = accuracy_score(y[agree], pred[agree]) if agree.sum() else np.nan
    out["disagreement_acc"] = accuracy_score(y[disag], pred[disag]) if disag.sum() else np.nan
    out["strong_disagreement_acc"] = accuracy_score(y[strong], pred[strong]) if strong.sum() else np.nan

    out["agreement_ece"] = _ece(y[agree], prob[agree]) if agree.sum() > 10 else np.nan
    out["disagreement_ece"] = _ece(y[disag], prob[disag]) if disag.sum() > 10 else np.nan
    out["calibration_gap"] = (out["disagreement_ece"] - out["agreement_ece"]
                              if not (np.isnan(out["disagreement_ece"]) or np.isnan(out["agreement_ece"]))
                              else np.nan)

    hc = conf >= high_conf
    hc_err = hc & (~correct)
    out["high_conf_error_rate"] = hc_err.sum() / hc.sum() if hc.sum() else 0.0
    out["n_high_conf"] = int(hc.sum())

    out["conf_correct"] = conf[correct].mean() if correct.sum() else np.nan
    out["conf_incorrect"] = conf[~correct].mean() if (~correct).sum() else np.nan

    out["n_agreement"] = int(agree.sum())
    out["n_disagreement"] = int(disag.sum())
    out["n_strong_disagreement"] = int(strong.sum())
    out["n_test"] = int(len(y))

    # Modality dominance %: of true-conflict cases (text_pred != meta_pred),
    # fraction where the fusion model follows text.
    if text_pred is not None and meta_pred is not None:
        tp = np.asarray(text_pred)
        mp = np.asarray(meta_pred)
        conflict = tp != mp
        nconf = conflict.sum()
        if nconf:
            follows_text = ((pred == tp) & conflict).sum()
            out["modality_dominance_text_pct"] = 100.0 * follows_text / nconf
            out["n_true_conflict"] = int(nconf)
        else:
            out["modality_dominance_text_pct"] = np.nan
            out["n_true_conflict"] = 0
    return out


# ---------------------------------------------------------------------------
# Statistics: bootstrap CI + paired bootstrap + McNemar
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, pred, prob, metric="accuracy", n=1000, seed=42, mask=None):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = np.asarray(y_true); pred = np.asarray(pred); prob = np.asarray(prob)
    if mask is not None:
        y_true, pred, prob = y_true[mask], pred[mask], prob[mask]

    def m(yt, yp, pr):
        if metric == "accuracy":
            return accuracy_score(yt, yp)
        if metric == "f1":
            return f1_score(yt, yp, zero_division=0)
        if metric == "auroc":
            return roc_auc_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan
        if metric == "ece":
            return _ece(yt, pr)
        raise ValueError(metric)

    rng = np.random.RandomState(seed)
    N = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        try:
            vals.append(m(y_true[idx], pred[idx], prob[idx]))
        except Exception:
            pass
    vals = np.array([v for v in vals if not np.isnan(v)])
    point = m(y_true, pred, prob)
    if len(vals) == 0:
        return point, np.nan, np.nan
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def paired_bootstrap_test(y_true, pred_a, pred_b, mask=None, metric="accuracy",
                          n=2000, seed=42):
    """Paired bootstrap test of metric(A) - metric(B). Returns (delta, p_two_sided)."""
    from sklearn.metrics import accuracy_score, f1_score
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    if mask is not None:
        y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]

    def m(yt, yp):
        if metric == "accuracy":
            return accuracy_score(yt, yp)
        if metric == "f1":
            return f1_score(yt, yp, zero_division=0)
        raise ValueError(metric)

    rng = np.random.RandomState(seed)
    N = len(y_true)
    if N == 0:
        return np.nan, np.nan
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.randint(0, N, N)
        deltas[i] = m(y_true[idx], pred_a[idx]) - m(y_true[idx], pred_b[idx])
    point = m(y_true, pred_a) - m(y_true, pred_b)
    # two-sided p: fraction of resamples on the opposite side of 0
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(point), float(min(p, 1.0))


def mcnemar(y_true, pred_a, pred_b, mask=None):
    from scipy.stats import chi2
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    if mask is not None:
        y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]
    ca = pred_a == y_true
    cb = pred_b == y_true
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(1 - chi2.cdf(stat, df=1))
    return float(stat), p, b, c


def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


# ---------------------------------------------------------------------------
# Experiment registry logger
# ---------------------------------------------------------------------------
REGISTRY_PATH = os.path.join("reports", "experiment_registry.md")


def log_experiment(row: dict):
    """Append a single experiment record to the registry markdown table."""
    _ensure("reports")
    cols = ["timestamp", "phase", "dataset", "category", "seed", "split_id",
            "model", "features", "includes_product_avg_rating",
            "disagreement_labels_in_training", "command", "output_files",
            "key_metrics"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    row = dict(row)
    row.setdefault("timestamp", _dt.datetime.now().isoformat(timespec="seconds"))
    line = "| " + " | ".join(str(row.get(c, "")).replace("|", "/").replace("\n", " ")
                             for c in cols) + " |"
    if not os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            f.write("# Experiment Registry\n\n")
            f.write("Experiment records are logged here automatically by "
                    "`pipeline/common.py`.\n\n")
            f.write(header + "\n" + sep + "\n")
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
