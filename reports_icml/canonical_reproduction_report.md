# Canonical Reproduction Report - All_Beauty

## Protocol

- Dataset: Amazon Reviews 2023, category `All_Beauty`.

- Balanced pool: 20000 reviews (10000 pos / 10000 neg), fixed with master seed 2024.

- Seeds (redrawn 70/15/15 stratified splits): [42, 123, 456, 789, 2026].

- Leakage-controlled metadata drops `product_average_rating, product_rating_number`.

- Pool disagreement rate (Definition A, pretrained sentiment vs rating label): 11.6% (2317/20000).


## Headline results (mean +/- std over 5 seeds)

| model | accuracy | disagreement_acc | strong_disagreement_acc | auroc | ece | calibration_gap | high_conf_error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text_only | 0.876+/-0.005 | 0.652+/-0.025 | 0.619+/-0.031 | 0.944+/-0.002 | 0.031+/-0.004 | 0.120+/-0.029 | 0.027+/-0.002 |
| metadata_lc | 0.578+/-0.007 | 0.508+/-0.022 | 0.485+/-0.017 | 0.610+/-0.007 | 0.040+/-0.006 | 0.113+/-0.028 | 0.201+/-0.045 |
| metadata_full | 0.686+/-0.004 | 0.628+/-0.014 | 0.619+/-0.018 | 0.758+/-0.007 | 0.027+/-0.008 | 0.099+/-0.010 | 0.062+/-0.011 |
| early_fusion_lc | 0.881+/-0.005 | 0.654+/-0.032 | 0.626+/-0.038 | 0.947+/-0.003 | 0.038+/-0.020 | 0.189+/-0.036 | 0.052+/-0.014 |
| early_fusion_full | 0.889+/-0.007 | 0.678+/-0.040 | 0.658+/-0.043 | 0.955+/-0.002 | 0.041+/-0.020 | 0.188+/-0.038 | 0.049+/-0.013 |
| late_fusion_lc | 0.884+/-0.005 | 0.650+/-0.038 | 0.622+/-0.054 | 0.952+/-0.002 | 0.021+/-0.006 | 0.181+/-0.041 | 0.037+/-0.005 |
| gated_fusion_lc | 0.883+/-0.004 | 0.651+/-0.024 | 0.625+/-0.032 | 0.951+/-0.002 | 0.018+/-0.005 | 0.168+/-0.043 | 0.034+/-0.003 |
| disagree_aware_lc | 0.867+/-0.004 | 0.678+/-0.025 | 0.660+/-0.032 | 0.937+/-0.003 | 0.060+/-0.018 | 0.170+/-0.060 | 0.066+/-0.022 |

## Key paired tests (per seed, then summarized)

| comparison (A vs B) | subset | mean delta acc | median boot p | median McNemar p |
| --- | --- | --- | --- | --- |
| disagree_aware_lc vs early_fusion_lc | disagreement | +0.023+/-0.014 | 0.520 | 0.551 |
| early_fusion_full vs early_fusion_lc | overall | +0.008+/-0.005 | 0.025 | 0.026 |
| early_fusion_lc vs text_only | disagreement | +0.002+/-0.016 | 0.612 | 0.643 |
| early_fusion_lc vs text_only | overall | +0.005+/-0.002 | 0.399 | 0.397 |

## Honest reading

- Early fusion (leakage-controlled) overall accuracy 0.881 vs text-only 0.876 (delta +0.005).

- On disagreement cases: early fusion 0.654 vs text-only 0.652 (delta +0.002).

- See `canonical_allbeauty_summary.csv` for the full metric suite and `canonical_allbeauty_stats.csv` for per-seed significance.

