"""Train text-only sentiment model: DistilBERT or TF-IDF + LogReg fallback."""

import os
import yaml
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression Fallback
# ---------------------------------------------------------------------------

def train_tfidf_logreg(train_df, val_df, save_dir="models/text_only"):
    os.makedirs(save_dir, exist_ok=True)

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["review_text"].values)
    y_train = train_df["label"].values

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    X_val = vectorizer.transform(val_df["review_text"].values)
    y_val = val_df["label"].values
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Text-only (TF-IDF+LogReg) val accuracy: {val_acc:.4f}")

    joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(save_dir, "logreg_model.joblib"))
    print(f"Saved text-only model to {save_dir}")

    return vectorizer, clf


def predict_tfidf_logreg(df, save_dir="models/text_only"):
    vectorizer = joblib.load(os.path.join(save_dir, "tfidf_vectorizer.joblib"))
    clf = joblib.load(os.path.join(save_dir, "logreg_model.joblib"))

    X = vectorizer.transform(df["review_text"].values)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    confidence = np.max(probs, axis=1)
    prob_positive = probs[:, 1]

    return preds, prob_positive, confidence


# ---------------------------------------------------------------------------
# DistilBERT Fine-tuning
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_distilbert(train_df, val_df, cfg, save_dir="models/text_only"):
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    os.makedirs(save_dir, exist_ok=True)
    model_cfg = cfg["models"]["text_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DistilBERT on {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(model_cfg["pretrained_name"])
    model = DistilBertForSequenceClassification.from_pretrained(
        model_cfg["pretrained_name"], num_labels=2
    ).to(device)

    train_ds = ReviewDataset(
        train_df["review_text"].tolist(), train_df["label"].tolist(),
        tokenizer, model_cfg["max_length"]
    )
    val_ds = ReviewDataset(
        val_df["review_text"].tolist(), val_df["label"].tolist(),
        tokenizer, model_cfg["max_length"]
    )
    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=model_cfg["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg["learning_rate"])

    for epoch in range(model_cfg["epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch["label"].numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved DistilBERT model to {save_dir}")
    return model, tokenizer


def predict_distilbert(df, save_dir="models/text_only", batch_size=32):
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(save_dir)
    model = DistilBertForSequenceClassification.from_pretrained(save_dir).to(device)
    model.eval()

    texts = df["review_text"].tolist()
    all_preds, all_probs = [], []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoding = tokenizer(
            batch_texts, max_length=256, padding=True, truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_probs.extend(probs)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    prob_positive = all_probs[:, 1]
    confidence = np.max(all_probs, axis=1)

    return all_preds, prob_positive, confidence
