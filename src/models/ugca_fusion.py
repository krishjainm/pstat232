"""UGCA-Fusion: Uncertainty-Guided Conflict-Aware Fusion.

A learned, inference-time conflict-aware multimodal fusion model. At inference
the model estimates *conflict* between the two modalities purely from the
relationship between their unimodal predictive distributions (it never sees the
true label), and conditions the fusion gate on that estimated conflict.

Components
----------
1. Text encoder      : text embedding -> h_text
2. Metadata encoder  : metadata vector -> h_meta
3. Unimodal heads    : h_text -> p_text, h_meta -> p_meta
4. Conflict detector : predicts a conflict score c_hat in [0,1] from
                       |p_text - p_meta|, entropy_text, entropy_meta,
                       JS(p_text || p_meta), and (optionally) learned features
                       of (h_text, h_meta).
5. Conflict-conditioned gate g : sigmoid gate fed h_text, h_meta and the
                       conflict signals; under high estimated conflict the gate
                       can route toward the lower-entropy (more certain) modality.
6. Fused prediction  : h_fused = g * h_text + (1-g) * h_meta ; classifier(h_fused).

Loss
----
L = CE(y_hat, y)
  + lambda_text * CE(text_head, y)
  + lambda_meta * CE(meta_head, y)
  + lambda_conflict * BCE(c_hat, proxy_disagreement_label)   [training only]
  + lambda_cal * Brier(p_fused, y)                            [calibration surrogate]

The conflict detector uses the proxy disagreement label ONLY during training
(via the BCE term). At inference it relies solely on the unimodal-prediction
relationship, so no label leakage occurs.

Ablation flags let us disable the conflict detector, the entropy features, the
JS-divergence feature, or the calibration loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy(p, eps=1e-8):
    return -(p * torch.log(p + eps)).sum(dim=1)


def _js_divergence(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


class UGCAFusion(nn.Module):
    def __init__(self, text_dim, meta_dim, hidden=128, dropout=0.2,
                 use_conflict_detector=True, use_entropy=True, use_js=True,
                 use_learned_conflict_features=True, learned_conflict_dim=16):
        super().__init__()
        self.use_conflict_detector = use_conflict_detector
        self.use_entropy = use_entropy
        self.use_js = use_js
        self.use_learned_conflict_features = use_learned_conflict_features
        H = hidden

        self.text_enc = nn.Sequential(
            nn.Linear(text_dim, H), nn.ReLU(), nn.Dropout(dropout))
        self.meta_enc = nn.Sequential(
            nn.Linear(meta_dim, H), nn.ReLU(), nn.Dropout(dropout))
        self.text_head = nn.Linear(H, 2)
        self.meta_head = nn.Linear(H, 2)

        # --- conflict feature dimensionality ---
        # hand-crafted: abs_gap always; +2 if entropy; +1 if js
        cf_dim = 1 + (2 if use_entropy else 0) + (1 if use_js else 0)
        if use_learned_conflict_features:
            self.conflict_proj = nn.Sequential(
                nn.Linear(2 * H, learned_conflict_dim), nn.ReLU())
            cf_dim += learned_conflict_dim
        self._cf_dim = cf_dim

        if use_conflict_detector:
            self.conflict_detector = nn.Sequential(
                nn.Linear(cf_dim, 32), nn.ReLU(), nn.Linear(32, 1))

        # --- gate input: h_text, h_meta, abs_gap, [c_hat], [entropies] ---
        gate_in = 2 * H + 1  # + abs_gap
        if use_conflict_detector:
            gate_in += 1       # + c_hat
        if use_entropy:
            gate_in += 2       # + entropy_text, entropy_meta
        self.gate = nn.Sequential(
            nn.Linear(gate_in, H), nn.ReLU(), nn.Linear(H, 1))

        self.classifier = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H // 2, 2))

    def forward(self, text, meta):
        h_text = self.text_enc(text)
        h_meta = self.meta_enc(meta)
        logit_text = self.text_head(h_text)
        logit_meta = self.meta_head(h_meta)
        p_text = F.softmax(logit_text, dim=1)
        p_meta = F.softmax(logit_meta, dim=1)

        abs_gap = (p_text[:, 1] - p_meta[:, 1]).abs()
        ent_text = _entropy(p_text)
        ent_meta = _entropy(p_meta)
        js = _js_divergence(p_text, p_meta)

        cf = [abs_gap.unsqueeze(1)]
        if self.use_entropy:
            cf += [ent_text.unsqueeze(1), ent_meta.unsqueeze(1)]
        if self.use_js:
            cf += [js.unsqueeze(1)]
        if self.use_learned_conflict_features:
            cf += [self.conflict_proj(torch.cat([h_text, h_meta], dim=1))]
        cf = torch.cat(cf, dim=1)

        if self.use_conflict_detector:
            c_logit = self.conflict_detector(cf).squeeze(1)
            c_hat = torch.sigmoid(c_logit)
        else:
            c_logit = torch.zeros(text.shape[0], device=text.device)
            c_hat = torch.zeros(text.shape[0], device=text.device)

        gate_feats = [h_text, h_meta, abs_gap.unsqueeze(1)]
        if self.use_conflict_detector:
            gate_feats.append(c_hat.detach().unsqueeze(1))
        if self.use_entropy:
            gate_feats += [ent_text.unsqueeze(1), ent_meta.unsqueeze(1)]
        g = torch.sigmoid(self.gate(torch.cat(gate_feats, dim=1)))  # (B,1)

        h_fused = g * h_text + (1 - g) * h_meta
        logit_fused = self.classifier(h_fused)

        return {
            "logit_fused": logit_fused,
            "logit_text": logit_text,
            "logit_meta": logit_meta,
            "p_text": p_text,
            "p_meta": p_meta,
            "c_logit": c_logit,
            "c_hat": c_hat,
            "gate": g.squeeze(1),
            "entropy_text": ent_text,
            "entropy_meta": ent_meta,
            "js": js,
            "abs_gap": abs_gap,
        }


def ugca_loss(out, y, proxy_disagree, lambda_text=0.3, lambda_meta=0.3,
              lambda_conflict=0.5, lambda_cal=0.2, use_calibration_loss=True,
              use_conflict_detector=True):
    """Composite UGCA loss. proxy_disagree is a float tensor in {0,1}."""
    ce = nn.functional.cross_entropy
    loss = ce(out["logit_fused"], y)
    loss = loss + lambda_text * ce(out["logit_text"], y)
    loss = loss + lambda_meta * ce(out["logit_meta"], y)

    if use_conflict_detector and lambda_conflict > 0:
        loss = loss + lambda_conflict * F.binary_cross_entropy_with_logits(
            out["c_logit"], proxy_disagree)

    if use_calibration_loss and lambda_cal > 0:
        # Brier score on the positive-class probability: a proper scoring rule
        # that promotes calibration, fully differentiable.
        p_pos = F.softmax(out["logit_fused"], dim=1)[:, 1]
        brier = ((p_pos - y.float()) ** 2).mean()
        loss = loss + lambda_cal * brier

    return loss
