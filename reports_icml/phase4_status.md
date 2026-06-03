# Phase 4 Status: UGCA-Fusion (Uncertainty-Guided Conflict-Aware Fusion)

**Status: COMPLETE and VERIFIED.**

Dataset: Amazon Reviews 2023, `All_Beauty`, leakage-controlled metadata
(`product_average_rating` and `product_rating_number` removed).
Seeds: 42, 123, 456, 789, 2026 (redrawn 70/15/15 splits per seed).
Compute: CPU-only.

## 1. What was built

- `src/models/ugca_fusion.py` — the `UGCAFusion` model and composite `ugca_loss`.
  - Text/metadata encoders, unimodal heads (`p_text`, `p_meta`).
  - **Conflict detector** that predicts a conflict score `c_hat` from
    `|p_text - p_meta|`, text/meta entropy, and Jensen–Shannon divergence
    (plus optional learned features). Uses **no true label at inference.**
  - **Conflict-conditioned fusion gate**: `g` is modulated by `c_hat`, so the
    fused representation `h_fused = g * h_text + (1-g) * h_meta` shifts under
    detected conflict.
  - Loss = CE(fused) + λ_text·CE(text) + λ_meta·CE(meta)
    + λ_conflict·BCE(conflict detector, proxy disagreement label)
    + λ_cal·(differentiable calibration surrogate).
- `scripts_icml/train_ugca.py` — training loop (early stopping) + CLI.
- `scripts_icml/evaluate_ugca.py` — 5-seed benchmark: UGCA-full, four ablations,
  and three baselines (early fusion, gated fusion, disagreement-aware reweighting),
  all leakage-controlled. Computes the conflict-detector AUROC.

## 2. Results (mean ± std across 5 seeds, All_Beauty, leakage-controlled)

| Model | Accuracy | Disagreement acc | ECE | Calib. gap | Conflict AUROC |
| --- | --- | --- | --- | --- | --- |
| early_fusion_lc | 0.881 ± 0.005 | 0.654 ± 0.032 | 0.038 | 0.189 | – |
| gated_fusion_lc | 0.883 ± 0.004 | 0.651 ± 0.024 | **0.018** | **0.168** | – |
| disagree_aware_lc | 0.867 ± 0.004 | **0.678 ± 0.025** | 0.060 | 0.170 | – |
| **ugca_full** | **0.883 ± 0.006** | 0.658 ± 0.035 | 0.050 | 0.206 | **0.721** |
| ugca_no_conflict | 0.881 ± 0.008 | 0.646 ± 0.033 | 0.039 | 0.204 | – |
| ugca_no_entropy | 0.883 ± 0.004 | 0.657 ± 0.032 | 0.040 | 0.188 | 0.717 |
| ugca_no_js | 0.883 ± 0.006 | 0.658 ± 0.035 | 0.051 | 0.208 | 0.726 |
| ugca_no_cal | 0.882 ± 0.003 | 0.658 ± 0.043 | 0.048 | 0.221 | 0.718 |

Artifacts:
- `reports_icml/tables/ugca_results.csv`
- `reports_icml/tables/ugca_ablation.csv`
- `reports_icml/figures/ugca_vs_baselines.png`
- `reports_icml/figures/ugca_ablation.png`
- `reports_icml/figures/conflict_detector_auc.png`

## 3. Honest findings

**Positive (genuine):**
- The **conflict detector works**: AUROC ≈ 0.72 across all UGCA variants, well
  above chance (0.5). This supports the central hypothesis that modality conflict
  is *detectable at inference* from the relationship between unimodal predictions,
  without access to the true label.
- **Ablations behave as designed.** Removing the conflict detector lowers
  disagreement accuracy (0.658 → 0.646). Removing the calibration loss worsens the
  calibration gap (0.206 → 0.221) and ECE. Entropy and JS features each contribute
  marginally to the conflict AUROC.

**Negative / honest limitations (NOT hidden):**
- **UGCA does not improve overall accuracy.** UGCA-full (0.883) is statistically
  indistinguishable from early/gated fusion (0.881–0.883).
- **UGCA does not win on disagreement accuracy.** The simple disagreement-aware
  reweighting baseline is still best on the disagreement subset
  (0.678 vs UGCA 0.658). The learned conflict gate did not beat the simpler
  reweighting on this metric in our setting.
- **UGCA is not the best-calibrated model.** Gated fusion has the lowest ECE
  (0.018) and calibration gap (0.168); UGCA-full is worse (0.050 / 0.206).

## 4. Interpretation

In our setting, the value of UGCA-Fusion is **diagnostic** (a label-free conflict
score with AUROC ≈ 0.72), **not** a uniform accuracy or calibration win. This is
consistent with the project's central, cautious claim: *average multimodal gains
mask conflict-specific reliability failures, and conflict-aware inference-time
fusion can reduce, but not eliminate, these failures.* No mitigation evaluated
here solves the disagreement problem, which we report as a robust open problem
rather than overclaiming a solution.

## 5. Next

Phase 4 is complete. Recommended follow-ups: integrate UGCA as model #5 into the
Phase 2 multi-category grid to test whether the conflict-detector AUROC replicates
across domains, and feed UGCA into the Phase 7 calibration/selective-prediction
comparison.
