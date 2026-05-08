"""Unified prediction interface for all models."""

import os
import numpy as np
import pandas as pd
import yaml

from src.models.train_text_model import (
    predict_tfidf_logreg, predict_distilbert, predict_sbert_logreg,
)
from src.models.train_metadata_model import predict_metadata_model
from src.models.train_multimodal_model import predict_multimodal


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def detect_text_model_type(save_dir="models/text_only"):
    """Detect whether saved text model is SBERT, TF-IDF, or DistilBERT."""
    if os.path.exists(os.path.join(save_dir, "sbert_logreg_model.joblib")):
        return "sentence_transformer"
    elif os.path.exists(os.path.join(save_dir, "logreg_model.joblib")):
        return "tfidf"
    elif os.path.exists(os.path.join(save_dir, "config.json")):
        return "distilbert"
    else:
        raise FileNotFoundError(f"No text model found in {save_dir}")


def get_all_predictions(df, split_name, cfg, config_path="config/config.yaml"):
    """Get predictions from all three models and return augmented dataframe."""
    df = df.copy()

    text_type = detect_text_model_type()
    if text_type == "sentence_transformer":
        t_preds, t_prob_pos, t_conf = predict_sbert_logreg(df)
    elif text_type == "tfidf":
        t_preds, t_prob_pos, t_conf = predict_tfidf_logreg(df)
    else:
        t_preds, t_prob_pos, t_conf = predict_distilbert(df)

    df["text_pred"] = t_preds
    df["text_prob_positive"] = t_prob_pos
    df["text_confidence"] = t_conf

    m_preds, m_prob_pos, m_conf = predict_metadata_model(df)
    df["meta_pred"] = m_preds
    df["meta_prob_positive"] = m_prob_pos
    df["meta_confidence"] = m_conf

    mm_preds, mm_prob_pos, mm_conf = predict_multimodal(df, split_name, cfg)
    df["mm_pred"] = mm_preds
    df["mm_prob_positive"] = mm_prob_pos
    df["mm_confidence"] = mm_conf

    return df
