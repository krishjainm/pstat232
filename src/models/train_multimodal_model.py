"""Train multimodal fusion model: text features + metadata → MLP."""

import os
import yaml
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse
from tqdm import tqdm

from src.features.build_metadata_features import (
    get_available_metadata, load_scaler, transform_metadata, fit_scaler
)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# MLP Fusion Model
# ---------------------------------------------------------------------------

class MultimodalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FusionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# TF-IDF + Metadata Fusion
# ---------------------------------------------------------------------------

def build_tfidf_features(train_texts, target_texts, max_features=20000, save_dir="models/multimodal"):
    """Fit TF-IDF on train, transform both train and target."""
    vec_path = os.path.join(save_dir, "tfidf_vectorizer.joblib")

    if os.path.exists(vec_path) and target_texts is not None:
        vectorizer = joblib.load(vec_path)
        X = vectorizer.transform(target_texts)
    else:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
        vectorizer.fit(train_texts)
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(vectorizer, vec_path)
        X = vectorizer.transform(target_texts if target_texts is not None else train_texts)

    if issparse(X):
        X = X.toarray()
    return X, vectorizer


def combine_features(text_feats, meta_feats):
    """Concatenate text and metadata features."""
    return np.hstack([text_feats, meta_feats])


def train_multimodal(train_df, val_df, cfg, save_dir="models/multimodal"):
    os.makedirs(save_dir, exist_ok=True)
    mm_cfg = cfg["models"]["multimodal_model"]
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training multimodal model on {device}")

    # Text features via TF-IDF
    max_feats = mm_cfg.get("text_max_features", 20000)
    train_texts = train_df["review_text"].tolist()
    val_texts = val_df["review_text"].tolist()

    print("Building TF-IDF features for multimodal model...")
    X_train_text, vectorizer = build_tfidf_features(train_texts, train_texts, max_feats, save_dir)
    X_val_text, _ = build_tfidf_features(train_texts, val_texts, max_feats, save_dir)

    # Metadata features
    scaler_path = "models/metadata_only/scaler.joblib"
    if os.path.exists(scaler_path):
        scaler = load_scaler(scaler_path)
    else:
        scaler = fit_scaler(train_df, save_path=scaler_path)

    feat_cols = get_available_metadata(train_df)
    train_meta, _ = transform_metadata(train_df, scaler, feat_cols)
    val_meta, _ = transform_metadata(val_df, scaler, feat_cols)

    # Combine
    X_train = combine_features(X_train_text, train_meta)
    X_val = combine_features(X_val_text, val_meta)
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    print(f"Combined feature dim: {X_train.shape[1]} (text={X_train_text.shape[1]}, meta={train_meta.shape[1]})")

    # Datasets
    train_ds = FusionDataset(X_train, y_train)
    val_ds = FusionDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=mm_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=mm_cfg["batch_size"])

    input_dim = X_train.shape[1]
    model = MultimodalMLP(
        input_dim,
        hidden_dims=tuple(mm_cfg["hidden_dims"]),
        dropout=mm_cfg["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=mm_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(mm_cfg["epochs"]):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)

        val_acc = accuracy_score(y_val, val_preds)
        print(f"  Epoch {epoch+1}/{mm_cfg['epochs']}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save architecture info
    joblib.dump({
        "input_dim": input_dim,
        "hidden_dims": tuple(mm_cfg["hidden_dims"]),
        "dropout": mm_cfg["dropout"],
    }, os.path.join(save_dir, "model_config.joblib"))

    print(f"Best val accuracy: {best_val_acc:.4f}")
    return model


def predict_multimodal(df, split_name, cfg, save_dir="models/multimodal"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_info = joblib.load(os.path.join(save_dir, "model_config.joblib"))
    model = MultimodalMLP(
        model_info["input_dim"],
        model_info["hidden_dims"],
        model_info["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt"), map_location=device))
    model.eval()

    # Text features
    vectorizer = joblib.load(os.path.join(save_dir, "tfidf_vectorizer.joblib"))
    X_text = vectorizer.transform(df["review_text"].tolist())
    if issparse(X_text):
        X_text = X_text.toarray()

    # Metadata
    scaler = load_scaler("models/metadata_only/scaler.joblib")
    feat_cols = get_available_metadata(df)
    meta_feats, _ = transform_metadata(df, scaler, feat_cols)

    X = combine_features(X_text, meta_feats)
    ds = FusionDataset(X, df["label"].values)
    loader = DataLoader(ds, batch_size=128)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for X_batch, _ in loader:
            logits = model(X_batch.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    prob_positive = all_probs[:, 1]
    confidence = np.max(all_probs, axis=1)

    return all_preds, prob_positive, confidence
