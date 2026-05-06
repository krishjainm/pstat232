# When Modalities Disagree: Failure Modes in Multimodal Models

**PSTAT 262DS Final Project**

## Research Question

How do multimodal models behave when input modalities conflict, and which modality dominates under disagreement?

## Overview

Multimodal models combine multiple sources of information (e.g., text and metadata) under the assumption that more data improves predictions. This project investigates what happens when those sources *disagree*—for example, when a review's text sounds positive but its star rating is low. We study whether multimodal fusion improves robustness or introduces new failure modes under modality conflict.

## Dataset

**Amazon Reviews 2023** (McAuley Lab) — `All_Beauty` category.

- Binary sentiment classification: positive (rating ≥ 4) vs. negative (rating ≤ 2).
- Neutral reviews (rating = 3) are dropped.
- Text modality: review text.
- Metadata modality: review length, helpful votes, verified purchase, product average rating, price, etc.

## Models

| Model | Input | Architecture |
|-------|-------|-------------|
| Text-only | Review text | DistilBERT (or TF-IDF + LogReg fallback) |
| Metadata-only | Structured features | XGBoost |
| Multimodal | Text + metadata | DistilBERT embeddings + metadata → MLP |

## Key Analyses

1. **Disagreement severity**: Classify reviews by how strongly text sentiment contradicts the rating label.
2. **Group-level evaluation**: Report accuracy, F1, calibration for agreement vs. disagreement subsets.
3. **Modality dominance**: When modalities conflict, does the multimodal model follow text or metadata?
4. **Confidence & calibration**: Do models become overconfident under disagreement?

## How to Run

```bash
pip install -r requirements.txt

python scripts/01_download_data.py
python scripts/02_preprocess_data.py
python scripts/03_define_disagreement.py
python scripts/04_train_text_model.py
python scripts/05_train_metadata_model.py
python scripts/06_train_multimodal_model.py
python scripts/07_evaluate_models.py
python scripts/08_generate_figures.py
```

## Outputs

- **Tables**: `reports/tables/` — dataset summary, main results, group results, calibration, modality dominance, error examples.
- **Figures**: `reports/figures/` — class distribution, agreement distribution, accuracy by group, calibration curves, modality dominance chart, confidence analysis, confusion matrices.
- **Models**: `models/` — saved model checkpoints and artifacts.

## Project Structure

```
├── config/config.yaml          # All hyperparameters and settings
├── data/                       # Raw, interim, and processed data
├── notebooks/                  # Exploration and analysis notebooks
├── src/                        # Reusable source modules
│   ├── data/                   # Download, preprocess, split
│   ├── features/               # Metadata features, disagreement labels
│   ├── models/                 # Training and prediction
│   ├── evaluation/             # Metrics, calibration, group analysis
│   └── visualization/          # Plotting functions
├── scripts/                    # Runnable pipeline scripts (01–08)
├── models/                     # Saved model artifacts
├── reports/                    # Figures, tables, report/presentation outlines
└── tests/                      # Unit tests
```
