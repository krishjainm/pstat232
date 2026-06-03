# Phase 1 Status: Clean Canonical Reproduction (All_Beauty)

**Status: COMPLETE and VERIFIED.**

## Protocol
- Dataset: Amazon Reviews 2023, category `All_Beauty`.
- Fixed balanced pool: 20,000 reviews (10k pos / 10k neg), master seed 2024.
- Pool disagreement rate (Definition A: pretrained sentiment vs rating label):
  11.6%.
- Seeds (redrawn 70/15/15 stratified splits): 42, 123, 456, 789, 2026.
- Leakage-controlled (LC) metadata drops `product_average_rating` and
  `product_rating_number`.
- 8 models trained per seed; full conflict-aware metric suite; bootstrap 95% CIs
  and per-seed paired tests (McNemar + paired bootstrap).

## Headline results (mean ± std over 5 seeds)

| Model | Accuracy | Disagree. acc | AUROC | ECE | Calib. gap |
| --- | --- | --- | --- | --- | --- |
| text_only | 0.876±0.005 | 0.652±0.025 | 0.944 | 0.031 | 0.120 |
| metadata_lc | 0.578±0.007 | 0.508±0.022 | 0.610 | 0.040 | 0.113 |
| metadata_full | 0.686±0.004 | 0.628±0.014 | 0.758 | 0.027 | 0.099 |
| early_fusion_lc | 0.881±0.005 | 0.654±0.032 | 0.947 | 0.038 | 0.189 |
| early_fusion_full | 0.889±0.007 | 0.678±0.040 | 0.955 | 0.041 | 0.188 |
| late_fusion_lc | 0.884±0.005 | 0.650±0.038 | 0.952 | 0.021 | 0.181 |
| gated_fusion_lc | 0.883±0.004 | 0.651±0.024 | 0.951 | 0.018 | 0.168 |
| disagree_aware_lc | 0.867±0.004 | 0.678±0.025 | 0.937 | 0.060 | 0.170 |

## Key findings (honest)
- Early fusion (LC) improves overall accuracy over text-only by only ~0.5 pts.
- On the disagreement subset, fusion provides essentially no gain.
- Every fusion variant **increases the calibration gap** vs text-only
  (0.120 → 0.168–0.189): average gains hide conflict-specific reliability loss.
- Full-feature fusion outperforms LC fusion, confirming the dropped product-level
  aggregates were leakage-prone; the LC setting is the honest main baseline.

## Artifacts
- `reports_icml/tables/canonical_allbeauty_multiseed.csv` (per seed × model)
- `reports_icml/tables/canonical_allbeauty_summary.csv` (mean ± std)
- `reports_icml/tables/canonical_allbeauty_stats.csv` (per-seed paired tests)
- `reports_icml/figures/canonical_allbeauty_errorbars.png`
- `reports_icml/canonical_reproduction_report.md`

Reproduce with: `python -m scripts_icml.phase1_canonical --category All_Beauty`.
