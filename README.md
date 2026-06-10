# A Computational Study of Disagreement-Aware Fusion, Calibration, and Resampling Inference in Text and Metadata Classification

**PSTAT 232 — Computational Techniques in Statistics — Final Project**

> This repository was previously a PSTAT 262DS machine-learning project on
> multimodal disagreement failures. It has been refactored into a
> **computational-statistics** study: empirical risk minimization and a
> disagreement-aware variant, post-hoc calibration with a new group-conditional
> estimator, resampling-based inference (bootstrap CIs + paired tests), and a
> **leakage-controlled** evaluation protocol. See
> [`PSTAT232_REFACTOR_PLAN.md`](PSTAT232_REFACTOR_PLAN.md) for a file-by-file
> account of what was reused, changed, and archived, and the relationship to the
> previous project below.

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
PSTAT 232 pipeline removes** `product_average_rating` and the prior-driven
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
miscalibration is mild — much of the previously reported calibration pathology
was a leakage artifact.

## Reproducibility Commands

All PSTAT 232 steps run **offline** from the materialized canonical pool and
cached embeddings under `data_icml/` (no download or re-encoding needed).

```bash
pip install -r requirements.txt

# Full PSTAT 232 pipeline (or run the steps individually below)
make pstat232

# 1. Leakage-controlled ERM models + per-sample predictions + main/ablation tables
python scripts/pstat232_leakage_controlled_main.py --category All_Beauty --seed 42

# 2. Group-conditional calibration (table + figure)
python scripts/pstat232_group_conditional_calibration.py --category All_Beauty --tau 0.5

# 3. Bootstrap CIs + paired comparisons
python scripts/pstat232_resampling_inference.py --category All_Beauty --n-boot 2000

# 4. All PSTAT 232 figures
python scripts/pstat232_make_figures.py --category All_Beauty

# Targeted Make wrappers
make pstat232-tables     # steps 1 + 2 + 3 (CSV outputs)
make pstat232-figures    # step 4
make pstat232-report     # compile paper/pstat232_report.tex

# Tests
make test                # 32 unit tests (pytest)
```

> Step 1 persists `data_icml/processed/All_Beauty_pstat232_{val,test}_predictions.parquet`,
> which steps 2–4 consume, so calibration / inference / figures need no retraining.

## Main Outputs

**Report**

- `paper/pstat232_report.tex` — the PSTAT 232 report (compiles to PDF with
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

**Prediction artifacts** (`data_icml/processed/`)

- `All_Beauty_pstat232_{val,test}_predictions.parquet`

## Relationship to the Previous PSTAT 262DS Project

The earlier PSTAT 262DS project ("When Modalities Disagree: Failure Modes and
Mitigations in Multimodal Sentiment Classification") framed the task as a
multimodal **ML benchmark**: which fusion architecture wins, by how much, and
which modality dominates under conflict. Its paper, the ICML/NeurIPS upgrade
(`paper_icml/`, `scripts_icml/`, `reports_icml/`), and the original analyses are
**retained** as prior work and reused infrastructure.

### What changed from the old version

| Aspect | PSTAT 262DS (old) | PSTAT 232 (this version) |
|--------|-------------------|--------------------------|
| Framing | Multimodal ML / leaderboard | Computational statistics |
| Leakage | Audited in an appendix; leaky feature used in main results | **Leakage-controlled is the primary pipeline**; leaky set is a sensitivity ablation only |
| Inference | Bootstrap CIs + McNemar reported | **Resampling inference is central**: bootstrap CIs *and* paired bootstrap tests for every key comparison |
| Calibration | Global temperature scaling | **Group-conditional temperature scaling** via a label-free inference-time conflict proxy (new extension) |
| Main deliverable | `paper/main.tex` | `paper/pstat232_report.tex` |
| New scripts | — | `scripts/pstat232_*.py` (4 scripts) |
| Old class/submission files | in `reports/` | moved to `archive/pstat262ds/` |

The reused code lives in `src/` (data, features, models, evaluation,
visualization) and `scripts_icml/icml_common.py` (canonical pool loading, cached
SBERT, CPU-light ERM training, the metric suite, and bootstrap/McNemar helpers),
on which the new `scripts/pstat232_*.py` are built.

## Dataset

**Amazon Reviews 2023** (McAuley Lab), primary category `All_Beauty`
(`Appliances`, `Digital_Music`, `Gift_Cards`, `Video_Games` materialized for
sensitivity). Binary sentiment: positive (rating ≥ 4) vs negative (rating ≤ 2),
neutrals dropped, class-balanced pool of 20,000 with a 70/15/15 stratified split.

## Project Structure

```
├── paper/
│   ├── pstat232_report.tex          # PSTAT 232 report (primary deliverable)
│   └── main.tex                     # prior PSTAT 262DS paper (retained)
├── scripts/
│   ├── pstat232_leakage_controlled_main.py
│   ├── pstat232_group_conditional_calibration.py
│   ├── pstat232_resampling_inference.py
│   ├── pstat232_make_figures.py
│   └── 01..10_*.py                  # original pipeline (retained)
├── scripts_icml/                    # reused infra (canonical data, training, stats)
├── src/                             # reusable modules (data/features/models/eval/viz)
├── data_icml/                       # materialized pools, cached SBERT, predictions
├── reports/figures, reports/tables  # PSTAT 232 + original outputs
├── archive/pstat262ds/              # archived class/submission-specific files
├── tests/                           # 32 unit tests
├── Makefile                         # includes pstat232* targets
├── requirements.txt
└── PSTAT232_REFACTOR_PLAN.md
```

## Requirements

Python 3.10+ with `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`,
`torch`, `xgboost`, `sentence-transformers` (only needed to *rebuild* embeddings;
the cached `data_icml/.../sbert.npy` is used by default). See `requirements.txt`.
