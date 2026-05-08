"""Tests for model architectures (no data/GPU required)."""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch
from src.models.train_multimodal_model import (
    MultimodalMLP, LateFusionModel, GatedFusionModel, FusionDataset,
)


def test_multimodal_mlp():
    model = MultimodalMLP(input_dim=100, hidden_dims=(64, 32), dropout=0.1)
    x = torch.randn(8, 100)
    out = model(x)
    assert out.shape == (8, 2)


def test_late_fusion_model():
    model = LateFusionModel(text_dim=80, meta_dim=20, hidden_dim=32, dropout=0.1)
    x = torch.randn(8, 100)
    out = model(x, text_dim=80)
    assert out.shape == (8, 2)


def test_gated_fusion_model():
    model = GatedFusionModel(text_dim=80, meta_dim=20, hidden_dim=32, dropout=0.1)
    x = torch.randn(8, 100)
    out = model(x, text_dim=80)
    assert out.shape == (8, 2)


def test_fusion_dataset():
    features = np.random.randn(10, 50).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    ds = FusionDataset(features, labels)
    assert len(ds) == 10
    feat, label, weight = ds[0]
    assert feat.shape == (50,)
    assert label.item() in (0, 1)
    assert weight.item() == 1.0


def test_gated_fusion_gate_values():
    model = GatedFusionModel(text_dim=5, meta_dim=3, hidden_dim=4, dropout=0.0)
    model.eval()
    x = torch.randn(4, 8)
    with torch.no_grad():
        gate_vals = model.gate(x)
    assert gate_vals.shape == (4, 1)
    assert (gate_vals >= 0).all() and (gate_vals <= 1).all()


if __name__ == "__main__":
    test_multimodal_mlp()
    test_late_fusion_model()
    test_gated_fusion_model()
    test_fusion_dataset()
    test_gated_fusion_gate_values()
    print("All model tests passed.")
