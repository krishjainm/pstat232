# Research-Upgrade Baseline Check

**Date:** 2026-06-01
**Purpose:** Establish and freeze the current baseline before any research-upgrade
changes, and confirm the committed results match the README and paper.

---

## 1. Commands run (this environment: Windows, Python 3.13, CPU-only)

| Command | Result |
|---------|--------|
| `pip install` core deps (`numpy pandas scikit-learn scipy pyyaml joblib tqdm torch pytest pyarrow matplotlib seaborn xgboost sentence-transformers`) | PASS |
| `python -m pytest tests/ -v` | **PASS — 32/32** |
| `pdflatex + bibtex + pdflatex×2` (paper) | **PASS — compiles** |
| Inspect saved model configs / feature columns | PASS (`meta_dim=8`, `input_dim=392`) |

### Full training pipeline (`make all`) — not re-executed
Re-running download + training would overwrite the validated artifacts and risk
seed/library drift. The committed artifacts are therefore treated as the frozen
baseline, and were verified for internal consistency (Section 2). HuggingFace is
reachable, so a provisioned rerun is possible; see the upgrade summary for which
new experiments require it.

---

## 2. Frozen baseline reference (All_Beauty test set, n=3102)

Taken directly from the committed CSVs in `reports/tables/` (all verified to match
the README tables and `paper/main.tex`):

| Model | Overall Acc | Overall F1 | AUROC | Disagree Acc | Disagree F1 | Strong-Disagree Acc | Overall ECE |
|-------|------------:|-----------:|------:|-------------:|------------:|--------------------:|------------:|
| Majority | .500 | .000 | .500 | — | — | — | — |
| Text-only | .868 | .866 | .943 | .592 | .648 | .545 | .028 |
| Metadata-only | .679 | .680 | .749 | .615 | .670 | .613 | .035 |
| Multimodal | .888 | .886 | .956 | .641 | .685 | .586 | .023 |
| Disagree-Aware (w=5) | .873 | .875 | .944 | .673 | .728 | — | — |

Other frozen reference numbers (real, from committed CSVs):
- True-conflict text dominance: **82.2%** (903/1098).
- Group-conditional calibration (multimodal): ECE **0.017** (agreement) vs **0.219** (disagreement) — **~13×**.
- McNemar text-vs-multimodal overall: χ²=17.6, p<0.001.

**Match status:** every README/paper headline number was traced to a generating
CSV in the prior verification pass and matches within rounding. See
`reports/verification_report.md`.

---

## 3. Data-state note (important for reproducibility)

The on-disk processed splits were inspected:

| Artifact | Rows | Identity |
|----------|-----:|----------|
| `data/raw/reviews.parquet` | 50,000 | subsampled raw (max_samples) |
| `data/interim/preprocessed.parquet` | 15,624 | **Appliances** (balanced) |
| `data/processed/train.parquet` | 10,936 | **Appliances** train |
| `data/processed/val.parquet` | — | **Appliances** val |
| `data/processed/test.parquet` | 2,344 | **Appliances** test |
| `data/processed/test_with_predictions.parquet` | 3,102 | **All_Beauty** (paper baseline, intact) |

A prior cross-category run (which produced `cross_category_results.csv`) left the
materialized split as **Appliances**, while the validated **All_Beauty** test set
with all per-model predictions survives as `test_with_predictions.parquet`.

**Implications for the upgrade (followed throughout):**
- Analyses that need no retraining (extended statistics, qualitative case studies,
  leakage correlations) are run on the intact **All_Beauty** predictions — fully
  consistent with the paper.
- The mitigation-method comparison, which requires a complete train/val/test split,
  is run on the materialized **Appliances** split (a real category) and is labeled
  as such; this doubles as a cross-domain test of the mitigation.
- Reproducing the All_Beauty *training* pipeline (and per-category ECE/Brier) requires
  a provisioned rerun (`make all`, `make extras`, `make multi-category`).

No baseline artifacts were modified by the upgrade; new results were written to new files.
