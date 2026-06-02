# When Modalities Disagree: Failure Modes and Mitigations in Multimodal Sentiment Classification

## Research Question

How do multimodal models behave when input modalities conflict, which modality dominates under disagreement, and can simple training-time interventions help?

## Overview

Multimodal models combine multiple sources of information (e.g., text and metadata) under the assumption that more data improves predictions. This project investigates what happens when those sources *disagree*—for example, when a review's text sounds positive but its star rating is low. We show that fusion models degrade dramatically under modality conflict, then demonstrate that **disagreement-aware loss reweighting** can partially mitigate these failures.

## Key Findings

| Metric | Text-Only | Metadata-Only | Multimodal | DA (w=5) |
|--------|-----------|---------------|------------|----------|
| Overall Accuracy | 0.868 | 0.679 | **0.888** | 0.873 |
| Disagreement Accuracy | 0.592 | 0.615 | 0.641 | **0.673** |
| Disagreement F1 | 0.648 | 0.670 | 0.685 | **0.728** |
| Strong Disagree Acc. | 0.545 | **0.613** | 0.586 | — |
| AUROC | 0.943 | 0.749 | **0.956** | 0.944 |

- Multimodal fusion improves average accuracy (+2.0 pp over text-only) but degrades disproportionately under modality conflict (64.1% on disagreement vs. 91.9% on agreement).
- **Disagreement-aware reweighting (w=5)** improves disagreement accuracy by +3.2 pp and F1 by +4.3 pp, at only 1.5 pp overall cost.
- The fusion model follows text predictions in **82.2%** of true-conflict cases.
- All three fusion architectures (early, late, gated) perform comparably (88.3–88.9%), suggesting the problem is paradigmatic, not architectural.
- Disagreement induces severe miscalibration: ECE of 0.219 vs. 0.017 for agreement cases (13× increase). Temperature scaling fails to close this gap.

## Reliability Caveats (read before citing the numbers)

These follow from the research-upgrade audit (`reports/research_upgrade_baseline_check.md`, `reports/leakage_audit.md`, `reports/neurlips_icml_upgrade_summary.md`) and are reported honestly:

- **Fusion's advantage concentrates in the agreement regime.** On the *strong*-disagreement subset, multimodal is **not** statistically better than text-only (McNemar p = 0.136) or metadata-only (p = 0.53). The headline accuracy gain is real overall but erodes exactly where modalities conflict.
- **Metadata leakage.** The metadata model's skill is largely driven by `product_average_rating`, a product-level aggregate that partially encodes the label: removing it drops metadata CV accuracy from 0.66 → 0.55. The `no_product_average_rating` configuration is the more conservative setting.
- **"Disagreement" is a proxy.** It is defined via a pretrained sentiment model vs. the rating-derived label, so the disagreement subset is enriched for hard/mislabeled reviews (see qualitative case 4).
- **Mitigations help modestly.** Severity-aware reweighting gives the best disagreement-accuracy gains; focal loss / temperature scaling help calibration but not disagreement accuracy. No method fixes both, and multi-seed runs show gains are comparable to training variance. The extended mitigation and multi-seed studies were run on the Appliances split (the materialized data on hand).

## Dataset

**Amazon Reviews 2023** (McAuley Lab) — `All_Beauty` category (configurable to other categories).

- Binary sentiment classification: positive (rating ≥ 4) vs. negative (rating ≤ 2); neutrals dropped.
- Class-balanced via majority-class downsampling: 20,674 samples (10,337 per class).
- 70/15/15 stratified train/val/test split → 14,471 / 3,101 / 3,102 samples.
- 11.1% disagreement rate on test set (343 of 3,102 samples).

## Models

| Model | Input | Architecture |
|-------|-------|-------------|
| Text-only | Review text | Sentence-Transformer (`all-MiniLM-L6-v2`) + Logistic Regression |
| Metadata-only | Structured features (8-d) | XGBoost (200 estimators, max depth 6) |
| Multimodal (Early) | Text + metadata (392-d) | 2-layer MLP (256→64→2) with ReLU + dropout |
| Multimodal (Late) | Text + metadata | Separate MLPs, averaged logits |
| Multimodal (Gated) | Text + metadata | Learned sigmoid gate over modality representations |
| Disagree-Aware | Text + metadata | Early fusion + disagreement-weighted cross-entropy loss |
| Majority Baseline | — | Always predicts majority class (50% on balanced data) |

## Analyses

1. **Disagreement severity**: Classify reviews by how strongly text sentiment contradicts the rating label (weak / medium / strong disagreement).
2. **Group-level evaluation**: Accuracy, F1, AUROC with 95% bootstrap CIs for agreement vs. disagreement subsets.
3. **Modality dominance**: Label-based, probability-based, and true-conflict dominance analysis.
4. **Fusion comparison**: Side-by-side evaluation of early, late, and gated fusion architectures.
5. **Disagreement-aware training**: Loss reweighting that upweights disagreement samples during training (w ∈ {3, 5}).
6. **Selective abstention**: Baseline that falls back to metadata predictions when text/metadata disagree.
7. **Statistical significance**: McNemar's test (continuity-corrected) for all pairwise model comparisons.
8. **Calibration**: Group-conditional ECE, temperature scaling experiments, selective prediction curves.
9. **Feature importance**: XGBoost feature importances for metadata model.
10. **Error taxonomy**: Heuristic classification of failure modes (sarcasm, mixed sentiment, short reviews, conditional sentiment, topic drift, etc.).

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/01_download_data.py
python scripts/02_preprocess_data.py
python scripts/03_define_disagreement.py
python scripts/04_train_text_model.py
python scripts/05_train_metadata_model.py
python scripts/06_train_multimodal_model.py
python scripts/07_evaluate_models.py
python scripts/08_generate_figures.py

# Additional scripts
python scripts/09_fusion_comparison.py          # Compare early/late/gated fusion
python scripts/10_disagreement_aware.py         # Disagreement-aware training + selective abstention
python scripts/run_all_categories.py            # Run across multiple product categories
python scripts/run_experiments.py               # Multi-seed experiments

# Research-upgrade analyses (rigor / generalization / honesty checks)
python scripts/upgrade_extended_stats.py        # Paired-bootstrap + McNemar on key comparisons
python scripts/upgrade_qualitative.py           # 6 representative qualitative case studies
python scripts/upgrade_leakage_audit.py         # Metadata leakage audit + correlations
python scripts/upgrade_ablation.py              # Metadata / text-encoder / fusion ablations
python scripts/upgrade_cross_category_figs.py   # Cross-category figures from results CSV
python scripts/upgrade_mitigation.py            # Mitigation comparison (needs SBERT; trains variants)
python scripts/upgrade_multiseed.py             # Multi-seed robustness (needs cached SBERT features)

# Or use Make
make all                # Core pipeline (scripts 01-08)
make extras             # Fusion comparison + disagreement-aware (scripts 09-10)
make multi-category     # Cross-category comparison
make multi-seed         # Multi-seed robustness
make test               # Run unit tests
make paper              # Compile LaTeX paper
```

## Tests

```bash
python -m pytest tests/ -v
```

32 unit tests covering metrics, calibration, disagreement, models, preprocessing, qualitative analysis, and statistical tests.

## Outputs

- **Tables** (`reports/tables/`): main/group results, bootstrap CIs, McNemar's tests, fusion comparison, calibration, modality dominance, temperature scaling, selective prediction, error taxonomy, case studies, disagreement-aware comparison, plus research-upgrade tables: `statistical_tests_extended.csv`, `mitigation_comparison.csv`, `ablation_results.csv`, `metadata_correlations.csv`, `multiseed_results.csv`, `qualitative_case_studies.csv`.
- **Figures** (`reports/figures/`): the original 14 PNGs plus cross-category (`cross_category_accuracy/disagreement_gap/calibration_gap.png`), mitigation (`mitigation_tradeoff/disagreement_accuracy/calibration.png`), `ablation_summary.png`, `multiseed_error_bars.png`, and `metadata_feature_importance_no_leakage.png`.
- **Models** (`models/`): Saved model checkpoints and artifacts for text, metadata, multimodal, and disagreement-aware models.
- **Paper** (`paper/main.tex`): Full LaTeX paper (12 pages) with cross-category generalization, mitigation comparison, ablation + leakage audit, and a strengthened limitations section.
- **Reports** (`reports/`): `verification_report.md`, `research_upgrade_baseline_check.md`, `leakage_audit.md`, `qualitative_case_studies.md`, `neurlips_icml_upgrade_summary.md`.

## Project Structure

```
├── config/config.yaml              # All hyperparameters and settings
├── data/                           # Raw, interim, and processed data
├── notebooks/                      # Exploration and analysis notebooks
├── src/                            # Reusable source modules
│   ├── data/                       # Download, preprocess, split
│   ├── features/                   # Metadata features, disagreement labels
│   ├── models/                     # Training and prediction
│   ├── evaluation/                 # Metrics, calibration, group analysis,
│   │                               # statistical tests, temperature scaling,
│   │                               # ablation, qualitative analysis
│   └── visualization/              # Plotting functions
├── scripts/                        # Runnable pipeline scripts (01–10)
│   ├── run_all_categories.py       # Multi-category pipeline
│   └── run_experiments.py          # Multi-seed experiments
├── paper/                          # LaTeX paper and references
├── models/                         # Saved model artifacts
├── reports/                        # Figures, tables, outlines
├── tests/                          # 32 unit tests
├── Makefile                        # Build automation
└── requirements.txt                # Pinned dependencies
```

## Requirements

Python 3.10+ with key dependencies: `torch`, `transformers`, `sentence-transformers`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `datasets`, `shap`. See `requirements.txt` for the full list with version pins.
