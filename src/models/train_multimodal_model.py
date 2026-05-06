"""Train multimodal fusion model: text embeddings + metadata → MLP."""

import os
import yaml
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.features.build_metadata_features import (
    get_available_metadata, load_scaler, transform_metadata
)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Text embeddings (frozen)
# ---------------------------------------------------------------------------

def extract_text_embeddings(texts, model_name="distilbert-base-uncased", batch_size=32):
    """Extract [CLS] embeddings from a frozen DistilBERT."""
    from transformers import DistilBertTokenizer, DistilBertModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch = texts[i : i + batch_size]
        encoding = tokenizer(
            batch, max_length=256, padding=True, truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings)


def get_or_cache_embeddings(df, split_name, cache_dir="data/interim", model_name="distilbert-base-uncased"):
    """Load cached embeddings or compute and cache them."""
    cache_path = os.path.join(cache_dir, f"{split_name}_text_embeddings.npy")
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    texts = df["review_text"].tolist()
    embeddings = extract_text_embeddings(texts, model_name)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Cached embeddings to {cache_path}")
    return embeddings


# ---------------------------------------------------------------------------
# MLP Fusion Model
# ---------------------------------------------------------------------------

class MultimodalMLP(nn.Module):
    def __init__(self, text_dim, meta_dim, hidden_dims=(256, 64), dropout=0.2):
        super().__init__()
        input_dim = text_dim + meta_dim
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
    def __init__(self, text_emb, meta_feats, labels):
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32)
        self.meta = torch.tensor(meta_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        combined = torch.cat([self.text_emb[idx], self.meta[idx]])
        return combined, self.labels[idx]


def train_multimodal(train_df, val_df, cfg, save_dir="models/multimodal"):
    os.makedirs(save_dir, exist_ok=True)
    mm_cfg = cfg["models"]["multimodal_model"]
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training multimodal model on {device}")

    # Text embeddings
    train_emb = get_or_cache_embeddings(train_df, "train", model_name=mm_cfg["text_encoder"])
    val_emb = get_or_cache_embeddings(val_df, "val", model_name=mm_cfg["text_encoder"])

    # Metadata
    scaler_path = "models/metadata_only/scaler.joblib"
    if os.path.exists(scaler_path):
        scaler = load_scaler(scaler_path)
    else:
        from src.features.build_metadata_features import fit_scaler
        scaler = fit_scaler(train_df, save_path=scaler_path)

    feat_cols = get_available_metadata(train_df)
    train_meta, _ = transform_metadata(train_df, scaler, feat_cols)
    val_meta, _ = transform_metadata(val_df, scaler, feat_cols)

    # Datasets
    train_ds = FusionDataset(train_emb, train_meta, train_df["label"].values)
    val_ds = FusionDataset(val_emb, val_meta, val_df["label"].values)
    train_loader = DataLoader(train_ds, batch_size=mm_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=mm_cfg["batch_size"])

    text_dim = train_emb.shape[1]
    meta_dim = train_meta.shape[1]

    model = MultimodalMLP(
        text_dim, meta_dim,
        hidden_dims=tuple(mm_cfg["hidden_dims"]),
        dropout=mm_cfg["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=mm_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(mm_cfg["epochs"]):
        model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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

        val_acc = accuracy_score(val_df["label"].values, val_preds)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save architecture info for loading later
    joblib.dump({
        "text_dim": text_dim, "meta_dim": meta_dim,
        "hidden_dims": tuple(mm_cfg["hidden_dims"]),
        "dropout": mm_cfg["dropout"],
    }, os.path.join(save_dir, "model_config.joblib"))

    print(f"Best val accuracy: {best_val_acc:.4f}")
    return model


def predict_multimodal(df, split_name, cfg, save_dir="models/multimodal"):
    mm_cfg = cfg["models"]["multimodal_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model config and weights
    model_info = joblib.load(os.path.join(save_dir, "model_config.joblib"))
    model = MultimodalMLP(
        model_info["text_dim"], model_info["meta_dim"],
        model_info["hidden_dims"], model_info["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt"), map_location=device))
    model.eval()

    # Embeddings and metadata
    text_emb = get_or_cache_embeddings(df, split_name, model_name=mm_cfg["text_encoder"])
    scaler = load_scaler("models/metadata_only/scaler.joblib")
    feat_cols = get_available_metadata(df)
    meta_feats, _ = transform_metadata(df, scaler, feat_cols)

    ds = FusionDataset(text_emb, meta_feats, df["label"].values)
    loader = DataLoader(ds, batch_size=64)

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
