# PSTAT 232 Project Plan

**Project title:** *A Computational Study of Disagreement-Aware Fusion,
Calibration, and Resampling Inference in Text and Metadata Classification*

This document describes the project's goals, computational design, file layout,
and reproduction commands. Work is on branch **`pstat232-final`**.

---

## 1. Goals

A computational-statistics study of a binary text-and-metadata classification
problem in which the two modalities frequently disagree. The project is built
around four computational themes:

1. **Empirical risk minimization (ERM)** and a disagreement-aware reweighted
   variant, with the conditional risk decomposed across agreement and
   graded-disagreement strata.
2. **Leakage-controlled evaluation** as the primary protocol: a label-encoding
   product-level rating aggregate is removed from the main pipeline; the leaky
   configuration is retained only as a sensitivity ablation.
3. **Post-hoc calibration / UQ**: temperature scaling and a new
   **group-conditional** temperature-scaling estimator driven by a label-free,
   inference-time conflict proxy.
4. **Resampling inference**: nonparametric bootstrap confidence intervals and
   paired bootstrap / McNemar tests for every key comparison.

---

## 2. Statistical design

- **Data.** Amazon Reviews 2023 (McAuley Lab), primary category `All_Beauty`,
  with four additional materialized categories for sensitivity. A class-balanced
  pool of 20,000 reviews with cached sentence-transformer embeddings and a
  pretrained-sentiment proxy; a stratified 70/15/15 split (seed 42).
- **Models (ERM estimators).** Text-only (logistic regression on SBERT
  embeddings), metadata-only (gradient-boosted trees), early-fusion MLP, and a
  disagreement-aware reweighted-ERM fusion model.
- **Leakage control.** The primary metadata feature set drops
  `product_average_rating` and `product_rating_number`; the full set is trained
  only for the ablation table.
- **Calibration.** A single global temperature vs. separate temperatures fit on
  validation conflict / non-conflict subsets, evaluated by ECE, NLL, and Brier
  score (and group-specific ECE).
- **Inference.** Bootstrap (`B=2000`) percentile CIs and paired bootstrap +
  McNemar tests.

---

## 3. Repository layout

| Path | Purpose |
|---|---|
| `scripts/pstat232_leakage_controlled_main.py` | Primary leakage-controlled ERM pipeline; persists per-sample val/test predictions; writes main, leakage-ablation, accuracy-by-disagreement, and selective-prediction tables. |
| `scripts/pstat232_group_conditional_calibration.py` | Global vs group-conditional temperature scaling via the inference-time conflict proxy. |
| `scripts/pstat232_resampling_inference.py` | Bootstrap CIs + paired bootstrap / McNemar tests. |
| `scripts/pstat232_make_figures.py` | The five report figures (data-guarded). |
| `pipeline/common.py` | Canonical pool loading, cached SBERT, CPU-light ERM training, the conflict-aware metric suite, and bootstrap/McNemar helpers. |
| `src/` | Reusable modules: data, features, models, evaluation, visualization. |
| `data_pool/` | Materialized pools, cached SBERT embeddings, and prediction artifacts. |
| `reports/tables/`, `reports/figures/` | Generated CSVs and PNGs. |
| `paper/pstat232_report.tex` | The report (compiles to PDF). |
| `tests/` | 32 unit tests. |
| `archive/legacy_materials/` | Supplementary outline notes. |

The `scripts/01..10_*.py` and `scripts/upgrade_*.py` provide a full
data-download-to-figures pipeline for the broader analysis (download, preprocess,
disagreement labeling, training, evaluation, figures, ablations, statistics).

---

## 4. Data flow

`pstat232_leakage_controlled_main.py` is the single source of per-sample
predictions: it persists
`data_pool/processed/All_Beauty_pstat232_{val,test}_predictions.parquet`, which
the calibration, resampling-inference, and figure scripts consume directly. No
retraining is required for steps 2–4.

---

## 5. Outputs

**Tables** (`reports/tables/`): `pstat232_main_results.csv`,
`pstat232_leakage_ablation.csv`, `pstat232_accuracy_by_disagreement.csv`,
`pstat232_group_calibration.csv`, `pstat232_bootstrap_ci.csv`,
`pstat232_paired_comparisons.csv`, `pstat232_selective_prediction.csv`.

**Figures** (`reports/figures/`): `pstat232_accuracy_by_disagreement.png`,
`pstat232_calibration_by_group.png`, `pstat232_group_calibration.png`,
`pstat232_leakage_ablation.png`, `pstat232_selective_prediction.png`.

**Predictions** (`data_pool/processed/`):
`All_Beauty_pstat232_{val,test}_predictions.parquet`.

**Report**: `paper/pstat232_report.tex`.

---

## 6. Reproduction commands

Runs **offline** from `data_pool/` (no download/encoding required):

```bash
pip install -r requirements.txt

make pstat232            # main + calibration + inference + figures
make pstat232-report     # compile paper/pstat232_report.tex -> PDF
make test                # 32 unit tests
```

Or individually:

```bash
python scripts/pstat232_leakage_controlled_main.py --category All_Beauty --seed 42
python scripts/pstat232_group_conditional_calibration.py --category All_Beauty --tau 0.5
python scripts/pstat232_resampling_inference.py --category All_Beauty --n-boot 2000
python scripts/pstat232_make_figures.py --category All_Beauty
```

Other materialized categories can be substituted via `--category`
(`Appliances`, `Digital_Music`, `Gift_Cards`, `Video_Games`) or
`make pstat232 CATEGORY=Appliances`.

---

## 7. Verification status

- All four scripts run end-to-end on `All_Beauty` (verified).
- All 7 tables and 5 figures regenerated from materialized data (verified).
- `paper/pstat232_report.tex` compiles to PDF with `pdflatex` (TinyTeX), no
  undefined references (verified).
- `pytest tests/` — **32 passed** (verified).
- No invented numbers: every value in the report and README is produced by the
  scripts above and matches the generated CSVs.

---

## 8. Known issues / TODO

- **Headline claims use one primary category/seed** (`All_Beauty`, seed 42).
  Cross-category replication is supported by the scripts (`--category`) but not
  run for every category; this is noted as a limitation in the report.
- **`sentence-transformers` is only needed to rebuild embeddings.** The cached
  `data_pool/.../sbert.npy` is used by default; it is not imported at runtime
  unless the pool/cache is missing.
- **ECE bootstrap CIs** are for a binned, non-smooth functional and are reported
  as indicative (noted in the report's Limitations).
- **TODO (optional):** run `make pstat232 CATEGORY=...` across all five
  categories and aggregate a cross-category sensitivity table/figure.
