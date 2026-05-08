"""Train multimodal fusion model: text features + metadata → MLP.

Supports three text feature backends: sentence-transformers (default), TF-IDF, or DistilBERT.
Also supports late fusion and gated fusion variants.
"""

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
# Fusion Model Architectures
# ---------------------------------------------------------------------------

class MultimodalMLP(nn.Module):
    """Early fusion: concatenate features then MLP."""
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


class LateFusionModel(nn.Module):
    """Late fusion: separate MLPs per modality, average logits."""
    def __init__(self, text_dim, meta_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, text_dim=None):
        text_x = x[:, :text_dim]
        meta_x = x[:, text_dim:]
        return (self.text_net(text_x) + self.meta_net(meta_x)) / 2


class GatedFusionModel(nn.Module):
    """Gated fusion: learned gate decides modality weighting."""
    def __init__(self, text_dim, meta_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(text_dim + meta_dim, 1), nn.Sigmoid(),
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, text_dim=None):
        text_x = x[:, :text_dim]
        meta_x = x[:, text_dim:]
        g = self.gate(x)
        fused = g * self.text_net(text_x) + (1 - g) * self.meta_net(meta_x)
        return self.classifier(fused)


class FusionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Text Feature Builders
# ---------------------------------------------------------------------------

def build_sbert_features(texts, save_dir="models/multimodal", sbert_model="all-MiniLM-L6-v2"):
    """Encode texts with sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(sbert_model)
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


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


def _get_text_features(df, cfg, save_dir, is_train=False, train_df=None):
    """Route to the configured text feature backend."""
    mm_cfg = cfg["models"]["multimodal_model"]
    text_type = mm_cfg.get("text_feature_type", "tfidf")
    texts = df["review_text"].tolist()

    if text_type == "sentence_transformer":
        sbert_name = mm_cfg.get("sbert_model", "all-MiniLM-L6-v2")
        return build_sbert_features(texts, save_dir, sbert_name)
    else:
        max_feats = mm_cfg.get("text_max_features", 20000)
        train_texts = train_df["review_text"].tolist() if train_df is not None else texts
        X, _ = build_tfidf_features(train_texts, texts, max_feats, save_dir)
        return X


def _build_model(fusion_type, input_dim, text_dim, meta_dim, mm_cfg):
    """Instantiate the right fusion model variant."""
    hidden_dims = tuple(mm_cfg.get("hidden_dims", [256, 64]))
    dropout = mm_cfg.get("dropout", 0.2)
    hidden_dim = hidden_dims[0] if hidden_dims else 64

    if fusion_type == "late":
        return LateFusionModel(text_dim, meta_dim, hidden_dim, dropout)
    elif fusion_type == "gated":
        return GatedFusionModel(text_dim, meta_dim, hidden_dim, dropout)
    else:
        return MultimodalMLP(input_dim, hidden_dims, dropout)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_multimodal(train_df, val_df, cfg, save_dir="models/multimodal"):
    os.makedirs(save_dir, exist_ok=True)
    mm_cfg = cfg["models"]["multimodal_model"]
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_type = mm_cfg.get("fusion_type", "early")
    print(f"Training multimodal model on {device} (fusion={fusion_type})")

    # Text features
    X_train_text = _get_text_features(train_df, cfg, save_dir, is_train=True)
    X_val_text = _get_text_features(val_df, cfg, save_dir, train_df=train_df)

    # Metadata features
    scaler_path = "models/metadata_only/scaler.joblib"
    if os.path.exists(scaler_path):
        scaler = load_scaler(scaler_path)
    else:
        scaler = fit_scaler(train_df, save_path=scaler_path)

    feat_cols = get_available_metadata(train_df)
    train_meta, _ = transform_metadata(train_df, scaler, feat_cols)
    val_meta, _ = transform_metadata(val_df, scaler, feat_cols)

    text_dim = X_train_text.shape[1]
    meta_dim = train_meta.shape[1]

    X_train = combine_features(X_train_text, train_meta)
    X_val = combine_features(X_val_text, val_meta)
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    print(f"Combined feature dim: {X_train.shape[1]} (text={text_dim}, meta={meta_dim})")

    train_ds = FusionDataset(X_train, y_train)
    val_ds = FusionDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=mm_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=mm_cfg["batch_size"])

    input_dim = X_train.shape[1]
    model = _build_model(fusion_type, input_dim, text_dim, meta_dim, mm_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mm_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(mm_cfg["epochs"]):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if fusion_type in ("late", "gated"):
                logits = model(X_batch, text_dim=text_dim)
            else:
                logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if fusion_type in ("late", "gated"):
                    logits = model(X_batch.to(device), text_dim=text_dim)
                else:
                    logits = model(X_batch.to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)

        val_acc = accuracy_score(y_val, val_preds)
        print(f"  Epoch {epoch+1}/{mm_cfg['epochs']}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    joblib.dump({
        "input_dim": input_dim,
        "text_dim": text_dim,
        "meta_dim": meta_dim,
        "hidden_dims": tuple(mm_cfg.get("hidden_dims", [256, 64])),
        "dropout": mm_cfg.get("dropout", 0.2),
        "fusion_type": fusion_type,
    }, os.path.join(save_dir, "model_config.joblib"))

    print(f"Best val accuracy: {best_val_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_multimodal(df, split_name, cfg, save_dir="models/multimodal"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_info = joblib.load(os.path.join(save_dir, "model_config.joblib"))
    fusion_type = model_info.get("fusion_type", "early")
    text_dim = model_info.get("text_dim", model_info["input_dim"])
    meta_dim = model_info.get("meta_dim", 0)
    mm_cfg = cfg["models"]["multimodal_model"]

    model = _build_model(
        fusion_type, model_info["input_dim"], text_dim, meta_dim,
        {"hidden_dims": list(model_info["hidden_dims"]), "dropout": model_info["dropout"]},
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt"), map_location=device))
    model.eval()

    # Text features — use same backend that was used during training
    text_type = mm_cfg.get("text_feature_type", "tfidf")
    texts = df["review_text"].tolist()

    if text_type == "sentence_transformer":
        sbert_name = mm_cfg.get("sbert_model", "all-MiniLM-L6-v2")
        X_text = build_sbert_features(texts, save_dir, sbert_name)
    else:
        vectorizer = joblib.load(os.path.join(save_dir, "tfidf_vectorizer.joblib"))
        X_text = vectorizer.transform(texts)
        if issparse(X_text):
            X_text = X_text.toarray()

    scaler = load_scaler("models/metadata_only/scaler.joblib")
    feat_cols = get_available_metadata(df)
    meta_feats, _ = transform_metadata(df, scaler, feat_cols)

    X = combine_features(X_text, meta_feats)
    ds = FusionDataset(X, df["label"].values)
    loader = DataLoader(ds, batch_size=128)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for X_batch, _ in loader:
            if fusion_type in ("late", "gated"):
                logits = model(X_batch.to(device), text_dim=text_dim)
            else:
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
