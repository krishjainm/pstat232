# Phase 1 Status — Clean Canonical Reproduction (All_Beauty)

**Status: COMPLETE and VERIFIED.** Phase 2 has not been started (per the
instruction to verify Phase 1 first).

## 1. What ran

Canonical multi-seed reproduction on **Amazon Reviews 2023 / All_Beauty**, run
entirely on the `icml-neurips-upgrade` branch with all outputs under
`reports_icml/`, `data_icml/`, `models_icml/`.

- **Data**: freshly downloaded All_Beauty (701,528 raw reviews → 160k subsample →
  binary-labeled, metadata-merged, class-balanced) → a **fixed balanced pool of
  20,000 reviews** (10k pos / 10k neg), built once with master seed 2024.
- **Disagreement (Definition A)**: pretrained DistilBERT-SST-2 sentiment vs. the
  rating-derived label. Pool disagreement rate **11.6%** (2,317/20,000;
  1,809 strong / 298 medium / 210 weak). Computed once and cached.
- **Embeddings**: SBERT `all-MiniLM-L6-v2`, encoded once for the whole pool and
  cached (`data_icml/interim/All_Beauty/sbert.npy`).
- **Protocol**: 5 redrawn stratified 70/15/15 splits, seeds **42, 123, 456, 789,
  2026** (fixed pool, multiple random splits). Each split fits a fresh scaler on
  train only.
- **8 models per seed** (40 fits total):
  1. text-only (SBERT + LogReg)
  2. metadata-only **full** (8 feats, incl. `product_average_rating`)
  3. metadata-only **leakage-controlled** (6 feats; drops
     `product_average_rating`, `product_rating_number`)
  4. early fusion **full**
  5. early fusion **leakage-controlled** (main setting)
  6. late fusion (leakage-controlled)
  7. gated fusion (leakage-controlled)
  8. disagreement-aware loss reweighting (early, leakage-controlled, w=5 on
     disagreement training rows — the only model that uses disagreement labels
     during training)

Full metric suite computed per (seed, model): accuracy, F1, AUROC, ECE, Brier,
agreement/disagreement/strong-disagreement accuracy, agreement/disagreement ECE,
calibration gap, high-confidence error rate, confidence on correct vs incorrect,
and modality-dominance %. Bootstrap 95% CIs (accuracy/F1/AUROC) per model;
paired bootstrap + McNemar tests per seed for the key comparisons.

## 2. Outputs produced (verified to exist)

| File | Rows / size |
| --- | --- |
| `reports_icml/tables/canonical_allbeauty_multiseed.csv` | 40 rows (8 models × 5 seeds) |
| `reports_icml/tables/canonical_allbeauty_summary.csv` | per-model mean±std |
| `reports_icml/tables/canonical_allbeauty_stats.csv` | per-seed paired tests |
| `reports_icml/figures/canonical_allbeauty_errorbars.png` | 3-panel error-bar figure |
| `reports_icml/canonical_reproduction_report.md` | narrative report |
| `reports_icml/experiment_registry.md` | 40 experiment rows logged |

## 3. Headline results (mean ± std over 5 seeds, leakage-controlled = main)

| model | acc | disagree acc | strong-disagree acc | AUROC | ECE | calib gap | modality text-dominance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text_only | 0.876±0.005 | 0.652±0.025 | 0.619±0.031 | 0.944 | 0.031 | 0.120 | — |
| metadata_lc | 0.578±0.007 | 0.508±0.022 | 0.485±0.017 | 0.610 | 0.040 | 0.113 | — |
| early_fusion_lc | 0.881±0.005 | 0.654±0.032 | 0.626±0.038 | 0.947 | 0.038 | 0.189 | 91.3% |
| late_fusion_lc | 0.884±0.005 | 0.650±0.038 | 0.622±0.054 | 0.952 | 0.021 | 0.181 | 93.3% |
| gated_fusion_lc | 0.883±0.004 | 0.651±0.024 | 0.625±0.032 | 0.951 | 0.018 | 0.168 | 93.4% |
| disagree_aware_lc | 0.867±0.004 | 0.678±0.025 | 0.660±0.032 | 0.937 | 0.060 | 0.170 | 89.6% |
| metadata_full* | 0.686±0.004 | 0.628±0.014 | 0.619±0.018 | 0.758 | 0.027 | 0.099 | — |
| early_fusion_full* | 0.889±0.007 | 0.678±0.040 | 0.658±0.043 | 0.955 | 0.041 | 0.188 | 90.1% |

\* full-feature rows are the secondary/appendix setting (include leakage feature).

## 4. Honest reading (what the numbers actually say)

1. **The conflict-failure pattern replicates cleanly.** Every fusion model
   scores ~0.91 on agreement but only ~0.65 on disagreement (~26 pp gap), and
   calibration degrades ~8× (agreement ECE ≈0.025 vs disagreement ECE ≈0.21;
   calibration gap ≈0.17–0.19). This is robust across all 5 seeds.

2. **Under leakage control, fusion's advantage over text-only is small and not
   significant.** early_fusion_lc vs text_only: overall +0.005 acc
   (median paired-bootstrap p≈0.40), on disagreement +0.002 acc (p≈0.61). So the
   "multimodal helps overall" headline is *weak* once product-level rating
   aggregates are removed. This is a genuine, non-inflated negative-leaning
   result and is the kind of finding the upgrade is meant to surface.

3. **Leakage is real and measurable.** metadata_full (0.686) vs metadata_lc
   (0.578) is a 10.8 pp gap, and early_fusion_full beats early_fusion_lc by
   +0.008 acc with a *significant* paired test (median p≈0.025). So
   `product_average_rating`/`product_rating_number` carry label-encoding signal,
   justifying the leakage-controlled main setting.

4. **Disagreement-aware reweighting helps disagreement modestly but not
   significantly, and costs calibration.** disag acc 0.678 vs early_fusion_lc
   0.654 (+0.023, median p≈0.52, NOT significant) and strong-disagree 0.660 vs
   0.626; but overall acc drops to 0.867 and ECE worsens to 0.060. No free lunch.

5. **Strong text dominance.** In true-conflict cases (text_pred≠meta_pred) the
   fusion models follow text 89–93% of the time.

## 5. Verification checklist

- [x] Branch `icml-neurips-upgrade` created; outputs isolated under `*_icml/`.
- [x] Original `reports/`, `paper/`, `models/`, `data/` untouched.
- [x] All 8 models ran on all 5 seeds (40/40 rows present).
- [x] Redrawn splits per seed (distinct `split_id` per seed).
- [x] Leakage-controlled features are the main setting; full features secondary.
- [x] Full metric suite + bootstrap CIs + paired tests produced.
- [x] Every experiment logged to `experiment_registry.md` with the required
      fields (dataset, category, seed, split, model, features,
      product_average_rating inclusion, disagreement-labels-in-training, metrics,
      command, output paths).
- [x] Figure renders; report written.

## 6. Reproduce

```bash
# one-time pool build (cached; sentiment + SBERT checkpointed)
python -m scripts_icml.canonical_data --category All_Beauty --pool-size 20000
# multi-seed run
python -m scripts_icml.phase1_canonical --category All_Beauty
```

## 7. Known caveats / limitations carried into later phases

- Disagreement is still a single-model proxy (Definition A) — Phase 5 will add
  alternative definitions and a human-annotation sample.
- Protocol fixes the balanced pool and redraws only the split (compute-bounded on
  CPU); resampling the pool itself is left as a robustness extension.
- The sentiment proxy truncates review text to the first 300 characters for
  tractable CPU scoring; documented and applied uniformly.

**Phase 1 is verified. Awaiting go-ahead before starting Phase 2
(multi-category evaluation).**
