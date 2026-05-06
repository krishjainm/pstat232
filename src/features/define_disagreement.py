"""Define agreement/disagreement between text sentiment and rating label."""

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_text_sentiment_transformer(texts, batch_size=64):
    """Use a pretrained HF sentiment pipeline to score text sentiment."""
    from transformers import pipeline

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # CPU; set to 0 for GPU
        truncation=True,
        max_length=512,
    )

    labels = []
    confidences = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring text sentiment"):
        batch = texts[i : i + batch_size]
        results = sentiment_pipe(batch)
        for r in results:
            if r["label"] == "POSITIVE":
                labels.append(1)
                confidences.append(r["score"])
            else:
                labels.append(0)
                confidences.append(r["score"])

    return np.array(labels), np.array(confidences)


def compute_text_sentiment_tfidf(train_df, target_df):
    """Fallback: Train a TF-IDF + LogReg model and predict sentiment."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    print("Training TF-IDF + LogReg for text sentiment scoring...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X_train = vectorizer.fit_transform(train_df["review_text"].values)
    y_train = train_df["label"].values

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    X_target = vectorizer.transform(target_df["review_text"].values)
    labels = clf.predict(X_target)
    probs = clf.predict_proba(X_target)
    confidences = np.max(probs, axis=1)

    return labels, confidences


def assign_disagreement(df, cfg):
    """Add disagreement columns given text_sentiment_label and text_sentiment_confidence."""
    weak_thresh = cfg["disagreement"]["weak_threshold"]
    medium_thresh = cfg["disagreement"]["medium_threshold"]

    df = df.copy()

    agrees = df["text_sentiment_label"] == df["label"]
    df["agreement_status"] = np.where(agrees, "agreement", "disagreement")

    conditions = [
        agrees,
        (~agrees) & (df["text_sentiment_confidence"] < weak_thresh),
        (~agrees) & (df["text_sentiment_confidence"] >= weak_thresh) & (df["text_sentiment_confidence"] < medium_thresh),
        (~agrees) & (df["text_sentiment_confidence"] >= medium_thresh),
    ]
    choices = ["agreement", "weak_disagreement", "medium_disagreement", "strong_disagreement"]
    df["disagreement_group"] = np.select(conditions, choices, default="agreement")

    return df


def add_disagreement_labels(df, cfg, method="transformer", train_df=None):
    """Full pipeline: score text sentiment, then assign disagreement."""
    texts = df["review_text"].tolist()

    if method == "transformer":
        labels, confs = compute_text_sentiment_transformer(texts)
    elif method == "tfidf":
        if train_df is None:
            raise ValueError("train_df required for tfidf method")
        labels, confs = compute_text_sentiment_tfidf(train_df, df)
    else:
        raise ValueError(f"Unknown method: {method}")

    df = df.copy()
    df["text_sentiment_label"] = labels
    df["text_sentiment_confidence"] = confs
    df = assign_disagreement(df, cfg)

    print(f"Disagreement distribution:\n{df['disagreement_group'].value_counts()}")
    return df
