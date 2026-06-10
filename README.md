# A Computational Study of Disagreement-Aware Fusion, Calibration, and Resampling Inference in Text and Metadata Classification

**PSTAT 232 — Computational Techniques in Statistics — Final Project**

## Project Scope

This is a PSTAT 232 computational-statistics final project studying
**disagreement-aware fusion**, **leakage-controlled evaluation**,
**calibration and uncertainty quantification**, and **resampling inference** for a
binary text-and-metadata classification problem in which the two modalities
frequently conflict. The project is organized around four computational themes:

- **Empirical risk minimization (ERM)** and a disagreement-aware reweighted
  variant.
- **Leakage-controlled evaluation** as the primary protocol (a label-encoding
  product-level rating aggregate is removed; the leaky configuration is kept only
  as a sensitivity ablation).
- **Post-hoc calibration / UQ**, including a group-conditional temperature-scaling
  estimator driven by a label-free, inference-time conflict proxy.
- **Resampling-based inference**: nonparametric bootstrap confidence intervals
  and paired bootstrap / McNemar tests for every key comparison.

See [`PSTAT232_PROJECT_PLAN.md`](PSTAT232_PROJECT_PLAN.md) for the full project
plan, file map, and reproduction commands.

## Main Research Question

When two modalities (review **text** and structured **metadata**) disagree, how
should we *fuse*, *calibrate*, and *do inference* about a classifier — and how
much of the apparent multimodal benefit survives once we (a) control for label
leakage and (b) attach honest sampling uncertainty?

Sub-questions:

1. How does empirical risk decompose across agreement and graded-disagreement
   strata, and does reweighting the ERM objective toward conflict cases change
   that decomposition?
2. Is the fusion model's accuracy advantage statistically real under resampling,
   or within sampling noise?
3. Is miscalibration homogeneous, or does it concentrate in the conflict region
   — and can a **group-conditional** post-hoc calibrator using a label-free
   conflict proxy exploit that?
4. How much of the metadata model's measured skill is an artifact of a
   label-encoding product-level rating aggregate?

## Statistical Methods Used

- **Empirical risk minimization (ERM)** for text-only (logistic regression on
  sentence-transformer embeddings), metadata-only (gradient-boosted trees), and
  early-fusion (MLP) models, plus a **disagreement-aware reweighted ERM** variant.
- **Post-hoc calibration / UQ**: temperature scaling fit by NLL on validation;
  **group-conditional temperature scaling** driven by an inference-time conflict
  proxy; ECE, NLL, Brier score, reliability diagrams, selective-prediction curves.
- **Resampling inference**: nonparametric bootstrap 95% percentile CIs for
  accuracy / F1 / AUROC / ECE / Brier; **paired bootstrap** tests and
  **McNemar's** continuity-corrected χ² test for model comparisons.
- **Experimental-design control**: a **leakage-controlled** primary pipeline that
  removes `product_average_rating` (a product-level aggregate of the rating that
  the label is derived from); the leaky configuration is kept only as a
  sensitivity ablation.

## Leakage-Controlled Setup (primary version)

The label `Y` is derived from the per-review star rating. The metadata feature
`product_average_rating` is a **product-level aggregate of star ratings**, so it
partially encodes the label — a textbook target-leakage source. The **primary
pipeline removes** `product_average_rating` and the prior-driven
`product_rating_number`. The full (leaky) feature set is run **only** as the
sensitivity ablation in `reports/tables/pstat232_leakage_ablation.csv`.

Effect of removing the leaky feature (`All_Beauty` test set):

| Model | Feature set | Accuracy | AUROC | ECE |
|-------|-------------|----------|-------|-----|
| Metadata-only | leakage-controlled | 0.566 | 0.600 | 0.050 |
| Metadata-only | full (leaky) | 0.687 | 0.755 | 0.028 |
| Multimodal | leakage-controlled | 0.876 | 0.947 | 0.026 |
| Multimodal | full (leaky) | 0.878 | 0.953 | 0.062 |

The leaky aggregate inflates metadata accuracy by ~12 pp and AUROC by +0.155, but
adds almost nothing to fusion accuracy while *tripling* fusion ECE.

## Key Results (`All_Beauty`, seed 42, leakage-controlled)

| Model | Accuracy | Disagree. acc. | AUROC | ECE | Brier |
|-------|----------|----------------|-------|-----|-------|
| Text-only | 0.873 | 0.641 | 0.943 | 0.028 | 0.093 |
| Metadata-only (LC) | 0.566 | 0.500 | 0.600 | 0.050 | 0.245 |
| Multimodal (LC) | **0.876** | 0.649 | **0.947** | **0.026** | **0.090** |
| Disagreement-aware (LC) | 0.862 | **0.663** | 0.936 | 0.072 | 0.108 |

Under resampling inference, **fusion is not statistically better than text-only**
(paired bootstrap *p* = 0.42; McNemar *p* = 0.42), but is overwhelmingly better
than the leakage-controlled metadata model (*p* < 0.001). The disagreement-aware
model improves disagreement accuracy but is significantly worse overall
(*p* = 0.002). Group-conditional calibration reduces non-conflict ECE
(0.0157 → 0.0137) and overall NLL; after leakage control the conflict-driven
miscalibration is mild.

## Reproducibility Commands

All steps run **offline** from the materialized canonical pool and cached
embeddings under `data_pool/` (no download or re-encoding needed).

```bash
pip install -r requirements.txt

# Full pipeline (or run the steps individually below)
make pstat232

# 1. Leakage-controlled ERM models + per-sample predictions + main/ablation tables
python scripts/pstat232_leakage_controlled_main.py --category All_Beauty --seed 42

# 2. Group-conditional calibration (table + figure)
python scripts/pstat232_group_conditional_calibration.py --category All_Beauty --tau 0.5

# 3. Bootstrap CIs + paired comparisons
python scripts/pstat232_resampling_inference.py --category All_Beauty --n-boot 2000

# 4. All figures
python scripts/pstat232_make_figures.py --category All_Beauty

# Targeted Make wrappers
make pstat232-tables     # steps 1 + 2 + 3 (CSV outputs)
make pstat232-figures    # step 4
make pstat232-report     # compile paper/pstat232_report.tex

# Tests
make test                # 32 unit tests (pytest)
```

> Step 1 persists `data_pool/processed/All_Beauty_pstat232_{val,test}_predictions.parquet`,
> which steps 2–4 consume, so calibration / inference / figures need no retraining.

## Main Outputs

**Report**

- `paper/pstat232_report.tex` — the project report (compiles to PDF with
  `make pstat232-report`).

**Tables** (`reports/tables/`)

- `pstat232_main_results.csv` — primary leakage-controlled model metrics.
- `pstat232_leakage_ablation.csv` — LC vs full (leaky) feature sets.
- `pstat232_accuracy_by_disagreement.csv` — accuracy per severity stratum.
- `pstat232_group_calibration.csv` — uncalibrated / global / group-conditional T.
- `pstat232_bootstrap_ci.csv` — bootstrap 95% CIs.
- `pstat232_paired_comparisons.csv` — paired bootstrap + McNemar tests.
- `pstat232_selective_prediction.csv` — selective accuracy vs coverage.

**Figures** (`reports/figures/`)

- `pstat232_accuracy_by_disagreement.png`
- `pstat232_calibration_by_group.png`
- `pstat232_group_calibration.png`
- `pstat232_leakage_ablation.png`
- `pstat232_selective_prediction.png`

**Prediction artifacts** (`data_pool/processed/`)

- `All_Beauty_pstat232_{val,test}_predictions.parquet`

## Dataset

**Amazon Reviews 2023** (McAuley Lab), primary category `All_Beauty`
(`Appliances`, `Digital_Music`, `Gift_Cards`, `Video_Games` materialized for
sensitivity). Binary sentiment: positive (rating ≥ 4) vs negative (rating ≤ 2),
neutrals dropped, class-balanced pool of 20,000 with a 70/15/15 stratified split.

## Project Structure

```
├── paper/
│   └── pstat232_report.tex          # project report (primary deliverable)
├── scripts/
│   ├── pstat232_leakage_controlled_main.py
│   ├── pstat232_group_conditional_calibration.py
│   ├── pstat232_resampling_inference.py
│   ├── pstat232_make_figures.py
│   └── 01..10_*.py, upgrade_*.py    # full data-to-figures pipeline
├── pipeline/                        # canonical data, CPU-light training, stats utils
├── src/                             # reusable modules (data/features/models/eval/viz)
├── data_pool/                       # materialized pools, cached SBERT, predictions
├── reports/figures, reports/tables  # generated outputs
├── archive/legacy_materials/        # supplementary outline notes
├── tests/                           # 32 unit tests
├── Makefile                         # includes pstat232* targets
├── requirements.txt
└── PSTAT232_PROJECT_PLAN.md
```

## Tests

```bash
python -m pytest tests/ -v   # 32 unit tests
```

## Requirements

Python 3.10+ with `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`,
`torch`, `xgboost`, `sentence-transformers` (only needed to *rebuild* embeddings;
the cached `data_pool/.../sbert.npy` is used by default). See `requirements.txt`.
