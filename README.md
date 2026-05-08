# When Modalities Disagree: Failure Modes in Multimodal Sentiment Classification

**PSTAT 262DS Final Project**

## Research Question

How do multimodal models behave when input modalities conflict, and which modality dominates under disagreement?

## Overview

Multimodal models combine multiple sources of information (e.g., text and metadata) under the assumption that more data improves predictions. This project investigates what happens when those sources *disagree*—for example, when a review's text sounds positive but its star rating is low. We study whether multimodal fusion improves robustness or introduces new failure modes under modality conflict.

## Key Findings

| Metric | Text-Only | Metadata-Only | Multimodal (Early) |
|--------|-----------|---------------|-------------------|
| Overall Accuracy | 0.868 | 0.679 | **0.888** |
| Disagreement Accuracy | 0.592 | 0.615 | 0.641 |
| Strong Disagree Accuracy | 0.545 | **0.613** | 0.586 |
| AUROC | 0.943 | 0.749 | **0.956** |

- Multimodal fusion improves average accuracy (+2.0 pp over text-only) but degrades disproportionately under modality conflict (64.1% on disagreement vs. 91.9% on agreement).
- The fusion model follows text predictions in **82.2%** of true-conflict cases.
- All three fusion architectures (early, late, gated) perform comparably (88.3–88.9%), suggesting the problem is paradigmatic, not architectural.
- Disagreement induces severe miscalibration: ECE of 0.219 vs. 0.017 for agreement cases (13× increase). Temperature scaling fails to close this gap.

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
| Metadata-only | Structured features (7-d) | XGBoost (200 estimators) |
| Multimodal (Early) | Text + metadata (392-d) | 2-layer MLP (256→64→2) with ReLU + dropout |
| Multimodal (Late) | Text + metadata | Separate MLPs, averaged logits |
| Multimodal (Gated) | Text + metadata | Learned sigmoid gate over modality representations |
| Majority Baseline | — | Always predicts majority class (50% on balanced data) |

## Analyses

1. **Disagreement severity**: Classify reviews by how strongly text sentiment contradicts the rating label (weak / medium / strong disagreement).
2. **Group-level evaluation**: Accuracy, F1, AUROC with 95% bootstrap CIs for agreement vs. disagreement subsets.
3. **Modality dominance**: Label-based, probability-based, and true-conflict dominance analysis.
4. **Fusion comparison**: Side-by-side evaluation of early, late, and gated fusion architectures.
5. **Statistical significance**: McNemar's test (continuity-corrected) for all pairwise model comparisons.
6. **Calibration**: Group-conditional ECE, temperature scaling experiments, selective prediction curves.
7. **Feature importance**: XGBoost feature importances for metadata model.
8. **Error taxonomy**: Heuristic classification of failure modes (sarcasm, mixed sentiment, short reviews, etc.).

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
python scripts/09_fusion_comparison.py      # Compare early/late/gated fusion
python scripts/run_all_categories.py        # Run across multiple product categories
python scripts/run_experiments.py           # Multi-seed experiments

# Or use Make
make all                # Full pipeline
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

- **Tables** (`reports/tables/`): 19 CSV files including main results, group results, bootstrap CIs, McNemar's tests, fusion comparison, calibration, modality dominance, temperature scaling, selective prediction, error taxonomy, and case studies.
- **Figures** (`reports/figures/`): 14 PNG figures including class distribution, agreement distribution, accuracy/F1 by group, calibration curves, modality dominance, probability dominance histogram, confidence analysis, confusion matrices, feature importance, calibration by group, and selective prediction curves.
- **Models** (`models/`): Saved model checkpoints and artifacts for text, metadata, and multimodal models.
- **Paper** (`paper/main.tex`): Full LaTeX paper (9 pages) with real results, bootstrap CIs, and embedded figures.

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
├── scripts/                        # Runnable pipeline scripts (01–09)
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
