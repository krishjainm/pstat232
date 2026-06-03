"""Phase 6: create a manual-annotation batch of disagreement candidates.

Generalizes the Phase 5 export: builds cross-fitted unimodal/multimodal
predictions on a category pool, flags disagreement candidates with definitions
A/B/C, samples N of them, and writes a CSV with empty human-label columns.

Usage:
  python -m scripts_icml.create_annotation_batch --category All_Beauty --n 200 \
      --out reports_icml/annotations/batch_allbeauty.csv

The output schema matches reports_icml/manual_annotation_sample.csv so the same
analysis script can consume it. Human labels are NEVER fabricated.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from scripts_icml import icml_common as ic
from scripts_icml.phase5_disagreement_definitions import (
    cross_fit_predictions, HIGH_CONF)


def build_batch(category, n, seed):
    df = ic.build_canonical_pool(category).reset_index(drop=True)
    emb = ic.load_sbert(category)
    cv = cross_fit_predictions(df, emb)
    y = df["label"].values

    defA = (df["text_sentiment_label"].values != y)
    defB = (cv["text_pred"] != cv["meta_pred"])
    defC = (cv["text_conf"] >= HIGH_CONF) & (cv["text_pred"] != y)
    candidate = defA | defB | defC
    cand_idx = np.where(candidate)[0]

    rng = np.random.RandomState(seed)
    pick = rng.choice(cand_idx, size=min(n, len(cand_idx)), replace=False)
    return pd.DataFrame({
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
        "genuine_conflict": "",
        "label_noise": "",
        "sarcasm": "",
        "mixed_sentiment": "",
        "unclear": "",
        "notes": "",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="All_Beauty")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports_icml/annotations/annotation_batch.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    batch = build_batch(args.category, args.n, args.seed)
    batch.to_csv(args.out, index=False)
    print(f"[save] {args.out} ({len(batch)} rows) -- human-label columns blank")


if __name__ == "__main__":
    main()
