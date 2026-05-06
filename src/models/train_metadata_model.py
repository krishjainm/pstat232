"""Train metadata-only sentiment model: XGBoost or RandomForest fallback."""

import os
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

from src.features.build_metadata_features import (
    get_available_metadata, fit_scaler, transform_metadata, load_scaler
)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_metadata_model(train_df, val_df, cfg, save_dir="models/metadata_only"):
    os.makedirs(save_dir, exist_ok=True)
    seed = cfg["project"]["seed"]
    model_type = cfg["models"]["metadata_model"]["type"]

    scaler = fit_scaler(train_df, save_path=os.path.join(save_dir, "scaler.joblib"))

    feat_cols = get_available_metadata(train_df)
    X_train, _ = transform_metadata(train_df, scaler, feat_cols)
    y_train = train_df["label"].values

    X_val, _ = transform_metadata(val_df, scaler, feat_cols)
    y_val = val_df["label"].values

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=seed, eval_metric="logloss",
                use_label_encoder=False,
            )
        except ImportError:
            print("XGBoost not available, falling back to RandomForest.")
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)

    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Metadata-only val accuracy: {val_acc:.4f}")

    joblib.dump(clf, os.path.join(save_dir, "model.joblib"))
    joblib.dump(feat_cols, os.path.join(save_dir, "feature_cols.joblib"))
    print(f"Saved metadata model to {save_dir}")
    return clf, scaler


def predict_metadata_model(df, save_dir="models/metadata_only"):
    clf = joblib.load(os.path.join(save_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(save_dir, "scaler.joblib"))
    feat_cols = joblib.load(os.path.join(save_dir, "feature_cols.joblib"))

    X, _ = transform_metadata(df, scaler, feat_cols)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    prob_positive = probs[:, 1]
    confidence = np.max(probs, axis=1)

    return preds, prob_positive, confidence
