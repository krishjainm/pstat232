# PSTAT 232 Refactor Plan

**Project title:** *A Computational Study of Disagreement-Aware Fusion,
Calibration, and Resampling Inference in Text and Metadata Classification*

This document records how the original **PSTAT 262DS** project ("When Modalities
Disagree: Failure Modes and Mitigations in Multimodal Sentiment Classification")
was refactored into a **PSTAT 232** (computational techniques in statistics)
final project. Work is on branch **`pstat232-final`**.

---

## 1. Reframing summary

| | PSTAT 262DS (original) | PSTAT 232 (this refactor) |
|---|---|---|
| Framing | Multimodal ML / leaderboard | Computational statistics |
| Central object | Best fusion architecture | Empirical risk, calibration, resampling inference |
| Leakage | Audited in an appendix; `product_average_rating` used in main results | **Leakage control is the primary pipeline**; leaky set is a sensitivity ablation only |
| Inference | Bootstrap CIs + McNemar reported | **Resampling inference is central**: bootstrap CIs + paired bootstrap tests for every key comparison |
| Calibration | Global temperature scaling | **Group-conditional temperature scaling** via a label-free inference-time conflict proxy (new) |
| Deliverable paper | `paper/main.tex` | `paper/pstat232_report.tex` |

---

## 2. What is reused from PSTAT 262DS

Reused **unchanged** (the new scripts depend on these):

- `src/data/preprocess.py` — labeling, metadata merge, feature build, balancing.
- `src/evaluation/calibration.py` — `expected_calibration_error` (ECE binning).
- `src/evaluation/temperature_scaling.py`, `metrics.py`, `group_analysis.py`,
  `statistical_tests.py`, `modality_dominance.py`, `qualitative.py`, `ablation.py`.
- `src/models/*`, `src/features/*`, `src/visualization/plots.py`.
- `scripts_icml/icml_common.py` — **key reuse**: canonical pool loading,
  cached SBERT, CPU-light ERM training (`train_text_only`, `train_metadata_only`,
  `train_fusion`), `compute_metric_suite`, and `bootstrap_ci` / `paired_bootstrap_test`
  / `mcnemar` helpers. The new PSTAT 232 scripts are built directly on this.
- `data_icml/interim/<category>/{canonical_pool.parquet, sbert.npy, sentiment_ckpt.parquet}`
  — materialized, balanced 20k pools with cached embeddings for 5 categories.
  This is why the PSTAT 232 pipeline runs **offline with no download/encoding**.

Reused as **prior-work record** (kept, not archived):

- `paper/main.tex`, `paper/references.bib`.
- `paper_icml/`, `scripts_icml/` (rest), `reports_icml/`.
- Original `reports/figures/*.png` and `reports/tables/*.csv` (the leaky-config
  baseline is referenced for comparison).

---

## 3. What is changed for PSTAT 232

- **Leakage control is primary.** `scripts/pstat232_leakage_controlled_main.py`
  drops `product_average_rating` and `product_rating_number` from the primary
  metadata feature set (using `icml_common.METADATA_LEAKAGE_CONTROLLED`). The
  full (leaky) set is trained only for the ablation table.
- **Resampling inference is foregrounded** in its own script and report sections.
- **New computational-statistics extension:** group-conditional temperature
  scaling driven by a label-free conflict proxy.
- **New report** with a statistics-style structure (problem formulation,
  leakage audit, ERM, calibration/UQ, resampling results, interpretation,
  limitations).
- **README** rewritten as the PSTAT 232 landing page.

---

## 4. Files kept (in place)

- All of `src/`, `scripts/01..10_*.py`, `scripts/upgrade_*.py`,
  `scripts/run_*.py`.
- All of `scripts_icml/`, `data_icml/`, `reports_icml/`, `paper_icml/`.
- `paper/main.tex`, `paper/references.bib`.
- `tests/` (32 tests, all passing).
- `config/config.yaml`, `requirements.txt`, `.gitignore`.

## 5. Files archived → `archive/pstat262ds/`

| Moved from | Moved to |
|---|---|
| `reports/presentation_outline.md` | `archive/pstat262ds/presentation_outline.md` |
| `reports/final_report_outline.md` | `archive/pstat262ds/final_report_outline.md` |

Plus a new `archive/pstat262ds/README.md` explaining the archive.

> Conservative archiving: only clearly class/submission-specific *outline* docs
> were moved. Source code, the ICML infrastructure, and result artifacts were
> retained because the PSTAT 232 pipeline reuses or references them, and moving
> them would break imports.

## 6. New files added

**Scripts** (`scripts/`)

- `pstat232_leakage_controlled_main.py` — primary leakage-controlled ERM pipeline;
  trains text/metadata/fusion/disagreement-aware models on one split; persists
  per-sample val/test predictions; writes main results, leakage ablation,
  accuracy-by-disagreement, and selective-prediction tables.
- `pstat232_group_conditional_calibration.py` — global vs group-conditional
  temperature scaling via the inference-time conflict proxy.
- `pstat232_resampling_inference.py` — bootstrap CIs + paired bootstrap / McNemar.
- `pstat232_make_figures.py` — the five PSTAT 232 figures (data-guarded).

**Report**

- `paper/pstat232_report.tex` (compiles to PDF).

**Plan / docs**

- `PSTAT232_REFACTOR_PLAN.md` (this file), `archive/pstat262ds/README.md`,
  rewritten `README.md`.

**Generated artifacts** (produced by the scripts above)

- Tables: `reports/tables/pstat232_main_results.csv`,
  `pstat232_leakage_ablation.csv`, `pstat232_accuracy_by_disagreement.csv`,
  `pstat232_group_calibration.csv`, `pstat232_bootstrap_ci.csv`,
  `pstat232_paired_comparisons.csv`, `pstat232_selective_prediction.csv`.
- Figures: `reports/figures/pstat232_accuracy_by_disagreement.png`,
  `pstat232_calibration_by_group.png`, `pstat232_group_calibration.png`,
  `pstat232_leakage_ablation.png`, `pstat232_selective_prediction.png`.
- Predictions: `data_icml/processed/All_Beauty_pstat232_{val,test}_predictions.parquet`.

**Makefile**: added `pstat232`, `pstat232-tables`, `pstat232-figures`,
`pstat232-report` (and `pstat232-main/-calibration/-inference`).

---

## 7. Commands to reproduce the PSTAT 232 version

Runs **offline** from `data_icml/` (no download/encoding required):

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
(`Appliances`, `Digital_Music`, `Gift_Cards`, `Video_Games`) or `make pstat232 CATEGORY=Appliances`.

---

## 8. Verification status

- All four PSTAT 232 scripts run end-to-end on `All_Beauty` (verified).
- All 7 tables and 5 figures regenerated from materialized data (verified).
- `paper/pstat232_report.tex` compiles to PDF with `pdflatex` (TinyTeX), no
  undefined references (verified).
- `pytest tests/` — **32 passed** (verified).
- No invented numbers: every value in the report and README is produced by the
  scripts above and matches the generated CSVs.

---

## 9. Known issues / missing data / TODO

- **`data/processed/` (original pipeline) is gitignored and not materialized.**
  The original `scripts/01..10_*.py` pipeline reads `data/processed/*.parquet`,
  which requires a network download (`scripts/01_download_data.py`) and SBERT
  encoding. The PSTAT 232 pipeline deliberately avoids this by using the
  materialized `data_icml/` pools instead. To run the *original* pipeline:
  `make download preprocess disagreement train evaluate figures`.
- **Single primary category/seed for headline claims.** `All_Beauty`, seed 42.
  Cross-category replication is supported by the scripts (`--category`) but not
  run for every category here; this is noted as a limitation in the report.
- **`sentence-transformers` import is slow / only needed to rebuild embeddings.**
  The cached `data_icml/.../sbert.npy` is used by default; ST is not imported at
  runtime unless the pool/cache is missing.
- **ECE bootstrap CIs** are for a binned, non-smooth functional and are reported
  as indicative (noted in the report's Limitations).
- **TODO (optional future work):** run `make pstat232 CATEGORY=...` across all
  five categories and aggregate a cross-category sensitivity table/figure.
