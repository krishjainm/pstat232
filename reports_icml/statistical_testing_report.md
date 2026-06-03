# Statistical Testing Report (Phase 8)

All comparisons use the leakage-controlled setting on All_Beauty unless noted. Two units of analysis are reported: **seed-paired** (n=5 redrawn splits; t-test, Wilcoxon, bootstrap-over-seeds CI, Cohen's dz) and **example-level** (McNemar + paired bootstrap per seed, median p over seeds). With only 5 seeds, p-values are interpreted cautiously and we emphasize effect sizes and CIs over thresholds.


## Seed-level paired tests

| comparison | subset | metric | mean delta | t-CI | boot-CI | t-test p | Wilcoxon p | Cohen's dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_fusion_lc vs text_only | overall | accuracy | +0.0050 | [0.0031, 0.0069] | [0.0041, 0.0064] | 0.0019 | 0.0625 | 3.235 |
| early_fusion_lc vs text_only | overall | disagreement_acc | +0.0020 | [-0.0178, 0.0218] | [-0.0112, 0.013] | 0.7911 | 0.8125 | 0.127 |
| early_fusion_lc vs text_only | overall | calibration_gap | +0.0685 | [0.0422, 0.0947] | [0.052, 0.0849] | 0.0019 | 0.0625 | 3.237 |
| ugca_full vs early_fusion_lc | overall | disagreement_acc | +0.0031 | [-0.0187, 0.0248] | [-0.0115, 0.0153] | 0.716 | 0.8125 | 0.175 |
| ugca_full vs disagree_aware_lc | overall | disagreement_acc | -0.0204 | [-0.0421, 0.0014] | [-0.0332, -0.007] | 0.0599 | 0.125 | -1.164 |
| ugca_full vs disagree_aware_lc | overall | accuracy | +0.0156 | [0.0106, 0.0206] | [0.0126, 0.0186] | 0.001 | 0.0625 | 3.884 |
| ugca_full vs early_fusion_lc | overall | calibration_gap | +0.0175 | [-0.0187, 0.0538] | [0.0006, 0.044] | 0.2503 | 0.1875 | 0.601 |
| ugca_full vs early_fusion_lc | overall | ece | +0.0111 | [-0.0216, 0.0438] | [-0.0112, 0.0301] | 0.4005 | 0.625 | 0.42 |
| early_fusion_lc vs early_fusion_full | overall | accuracy | -0.0080 | [-0.0143, -0.0017] | [-0.0118, -0.0039] | 0.024 | 0.0625 | -1.584 |
| early_fusion_lc vs early_fusion_full | overall | disagreement_acc | -0.0239 | [-0.0387, -0.0092] | [-0.033, -0.0149] | 0.0108 | 0.0625 | -2.014 |
| early_fusion_lc vs text_only | pooled-multicat | accuracy | +0.0069 | [0.0035, 0.0102] | [0.0039, 0.0099] | 0.0006 | 0.001 | 1.137 |
| early_fusion_lc vs text_only | pooled-multicat | disagreement_acc | +0.0240 | [0.0008, 0.0471] | [0.0041, 0.0447] | 0.0434 | 0.048 | 0.573 |

## Example-level tests (median over seeds)

| comparison | subset | mean delta acc | median McNemar p | median bootstrap p | n seeds |
| --- | --- | --- | --- | --- | --- |
| disagree_aware_lc vs early_fusion_lc | disagreement | +0.0234 | 0.551 | 0.52 | 5 |
| early_fusion_full vs early_fusion_lc | overall | +0.0080 | 0.0264 | 0.025 | 5 |
| early_fusion_lc vs text_only | disagreement | +0.0020 | 0.6434 | 0.612 | 5 |
| early_fusion_lc vs text_only | overall | +0.0050 | 0.3973 | 0.399 | 5 |
| ugca_full vs early_fusion_lc | overall | +0.0020 | 0.7094 | 0.662 | 5 |
| ugca_full vs early_fusion_lc | disagreement | +0.0031 | 0.5596 | 0.534 | 5 |
| ugca_full vs disagree_aware_lc | disagreement | -0.0204 | 0.461 | 0.422 | 5 |

## Honest reading

- **Early fusion vs text-only (overall):** see mean delta and CI above; in our setting the overall-accuracy difference is small.

- **Disagreement subset:** the large raw gaps reported in Phases 1/5 are between agreement and disagreement *subsets*, not necessarily a significant *model-vs-model* difference on the same subset.

- **UGCA vs baselines:** UGCA does not show a statistically clear advantage over early fusion or over disagreement-aware reweighting on disagreement accuracy (CIs include 0 / large median p). Its demonstrated value is the conflict detector (Phase 4, AUROC ~0.72), not an accuracy win.

- **Leakage control:** full-feature early fusion vs leakage-controlled early fusion quantifies how much product-level priors inflate accuracy; see the corresponding row.

- Effect sizes are in `effect_sizes.csv`; full numbers in `statistical_tests_main.csv`.

