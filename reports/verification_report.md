# Verification Report

**Project:** When Modalities Disagree: Failure Modes and Mitigations in Multimodal Sentiment Classification
**Date:** 2026-06-01
**Scope:** Reproducibility audit, consistency check, and submission-readiness polish for final submission. No new experiments or model complexity were added; metrics were not altered.

---

## 1. Environment

| Item | Value |
|------|-------|
| OS | Windows 10 (win32 10.0.26200), PowerShell |
| Python | 3.13.13 |
| Installed for verification | `numpy`, `pandas`, `scikit-learn`, `scipy`, `pyyaml`, `joblib`, `tqdm`, `torch 2.12.0+cpu`, `pytest 9.0.3` |
| LaTeX | TeX Live 2026 (`pdfTeX 3.141592653`, `BibTeX 0.99e`) |

The repository already contained generated artifacts (`data/`, `models/`, `reports/tables/`, `reports/figures/`, `paper/main.pdf`).

---

## 2. Commands Run

| Command | Purpose | Result |
|---------|---------|--------|
| `pip install numpy pandas scikit-learn scipy pyyaml joblib tqdm` | Core deps for src/tests | **PASS** |
| `pip install torch` | Required by model/test modules | **PASS** (2.12.0+cpu) |
| `pip install pytest` | Test runner | **PASS** |
| `python -m pytest tests/ -v` | Run unit tests (`make test`) | **PASS — 32 passed in ~42s** |
| `pdflatex + bibtex + pdflatex x2` (`make paper`) | Compile paper | **PASS — main.pdf, 10 pages, no undefined citations/refs** |
| Inspect `models/metadata_only/feature_cols.joblib` and `models/multimodal/model_config.joblib` | Confirm true feature dimensions | **PASS — meta_dim=8, text_dim=384, input_dim=392** |

### Full retraining pipeline (`make all`) — NOT executed (intentional)

`make all` (scripts `01_download_data` → `08_generate_figures`), `make extras` (09–10),
`make multi-category`, and `make multi-seed` were **not run**. Rationale:

- They require multi-GB downloads (`torch`, `transformers`, `sentence-transformers`, `datasets`)
  and a live HuggingFace download of *Amazon Reviews 2023*, plus CPU training of
  SBERT/XGBoost/MLP models.
- Re-running training would **overwrite the existing CSV/figure artifacts** that the README and
  paper were validated against. Per the task constraint ("do not change metrics unless generated
  by the code; do not invent results"), the safer verification is to confirm the **committed
  artifacts** are internally consistent with the README and paper, which they are (Section 3).
- The code paths that train these models are exercised structurally by the unit tests
  (model forward passes, metrics, calibration, disagreement, preprocessing, qualitative).

**Recommended manual step before final hand-in (optional):** on a machine with the full
dependency stack and network access, run `make all && make extras && make multi-category`
once to confirm end-to-end regeneration. Numbers are seeded (`seed: 42`) but minor
floating-point / library-version drift is possible.

---

## 3. Reproducibility: README / Paper numbers vs generated CSVs

All headline numbers were cross-checked against the CSVs in `reports/tables/`. **All match** (rounding aside).

### 3.1 Main results (`main_results.csv`, `bootstrap_ci_results.csv`)

| Metric | Source | README/Paper | CSV | Match |
|--------|--------|--------------|-----|-------|
| Text overall acc | main_results | 0.868 | 0.8675 | ✓ |
| Metadata overall acc | main_results | 0.679 | 0.6786 | ✓ |
| Multimodal overall acc | main_results | 0.888 | 0.8881 | ✓ |
| Multimodal overall AUROC | main_results | 0.956 | 0.9556 | ✓ |
| Text disagree acc | group_results | 0.592 | 0.5918 | ✓ |
| Metadata disagree acc | group_results | 0.615 | 0.6152 | ✓ |
| Multimodal disagree acc | group_results | 0.641 | 0.6414 | ✓ |
| Multimodal disagree F1 | group_results | 0.685 | 0.6854 | ✓ |
| Strong-disagree acc (text/meta/mm) | group_results | .545/.613/.586 | .5451/.6128/.5865 | ✓ |
| All bootstrap CIs (Table 1) | bootstrap_ci_results | as printed | identical | ✓ |

### 3.2 Group accuracy table (`group_results.csv`, paper Table 2)
Every cell (overall / agreement / weak / medium / strong / all-disagree for the three models) matches, including subgroup sizes **n = 2759 / 27 / 50 / 266 / 343**. ✓

### 3.3 Fusion comparison (`fusion_comparison_results.csv`, paper Table 3)
Early/Late/Gated values (.888/.889/.883 acc; .641/.650/.630 disagree acc) match. ✓

### 3.4 Modality dominance
- True-conflict dominance (`conflict_dominance_results.csv`): follows_text 903 / (903+195) = **82.2%** ✓ (README + paper).
- Probability dominance (`probability_dominance_summary.csv`): mean ratio 0.380, 69.4% text-leaning ✓.

### 3.5 Calibration (`group_conditional_calibration.csv`, `temperature_scaling_results.csv`, paper Table 5)
Agreement vs disagreement ECE (e.g. multimodal 0.017 vs 0.219 → **13×**), and before/after temperature values and temperatures (0.662 / 1.311 / 1.153) all match. ✓

### 3.6 McNemar (`mcnemar_test_results.csv`, paper Table 6)
χ² and p-values match (17.6/<0.001; 4.65/0.031; 451.0/<0.001; 0.56/0.456; 311.7/<0.001; 0.33/0.568). ✓

### 3.7 Disagreement-aware training (`disagreement_aware_results.csv`, paper Table 7)
Standard / DA(w=3) / DA(w=5) overall, agreement, and disagreement Acc/F1 match; w=5 disagree acc 0.673, F1 0.728. ✓

### 3.8 Error taxonomy (`error_taxonomy.csv`, paper Table 8)
All counts, percentages, mean confidence, and %-disagreement match (Mixed 45/22.5%, Sarcasm 23/11.5%, Short 17/8.5%, … Other 92/46.0%). ✓

### 3.9 Cross-category (`cross_category_results.csv`, paper Table 9)
All Beauty / Digital Music / Appliances rows (n_test, disagreement %, MM acc, disagree acc/F1, text dominance) match. Video Games row failed (recorded `error` in CSV; acknowledged in paper Limitations). ✓

### 3.10 Dataset counts
- 20,674 balanced samples (10,337/class): `cross_category_results.csv` All_Beauty n_total = 20674 ✓
- Splits 14,471 / 3,101 / 3,102 (sum = 20,674) ✓ (`dataset_summary.csv` shows train 14471, test 3102).
- 11.1% test disagreement = 343/3102 ✓ (= 27 weak + 50 medium + 266 strong).

---

## 4. Figures and Tables Referenced in `paper/main.tex`

### 4.1 Figures (all present in `reports/figures/`)
| LaTeX include | File exists |
|---------------|-------------|
| `disagreement_severity_trend.png` | ✓ |
| `probability_dominance_histogram.png` | ✓ |
| `selective_prediction.png` | ✓ |

The paper uses **relative paths** (`../reports/figures/...`) and compiles cleanly from inside `paper/`.
The other 11 figures in `reports/figures/` exist but are not referenced by the paper (used in notebooks/slides) — not an error.

### 4.2 Tables
All 9 paper tables are **directly embedded** as LaTeX `tabular` (no `\input` of external CSVs), so there are no missing-file risks. Each embedded value was traced back to a generating CSV (Section 3).

### 4.3 Counts claimed in README
- "14 PNG figures" → exactly 14 present ✓
- "20+ CSV files" → 21 present ✓
- "32 unit tests" → `pytest` collected and passed exactly 32 ✓
- "(9/10 pages)" → updated to **10** after polish (see Section 6).

---

## 5. Inconsistencies Found

| # | Type | Description | Status |
|---|------|-------------|--------|
| 1 | Code vs docs (model spec) | Metadata modality is **8 features** in code (`METADATA_FEATURES` includes `product_price`; saved `feature_cols.joblib` has 8; `meta_dim=8`, fused `input_dim=392`). README said "7-d", `config.yaml` listed 7, paper said "seven features". | **FIXED** |
| 2 | Build automation | `Makefile` had no targets for scripts `09_fusion_comparison` and `10_disagreement_aware`, even though their outputs appear in the paper; README implied `make all` is the "full pipeline". | **FIXED** (added `fusion`, `disagree-aware`, `extras`; clarified README) |
| 3 | Citation accuracy | Paper described `huang2021makes` ("What Makes Multi-modal Learning Better than Single (Provably)") as "attention-based fusion", which it is not. | **FIXED** (reworded to its actual theoretical claim) |
| 4 | Overstated wording | "modality conflict is a paradigm limitation, not architectural" and the heading "Fusion Architecture Does Not Matter" overgeneralize from 3 tested architectures. | **FIXED** (softened to "not specific to a single fusion architecture" / "Has Little Effect") |
| 5 | Doc drift | Paper page count in README (9) vs actual after polish. | **FIXED** (now 10) |

No metric mismatches were found. No broken figure paths were found.

---

## 6. Fixes Applied

**Consistency / correctness (safe):**
- `README.md`: metadata model now "Structured features (8-d)" and "XGBoost (200 estimators, max depth 6)".
- `config/config.yaml`: added `product_price` to `features.metadata_features` (now 8, matching code).
- `paper/main.tex` Data section: "seven features … and review year" → "eight features … product price, and review year".
- `paper/main.tex` Limitations: "seven structured features" → "eight structured tabular features".
- `Makefile`: added `fusion`, `disagree-aware`, and `extras` targets for scripts 09–10; updated `.PHONY`.
- `README.md`: documented `make extras`; clarified `make all` = scripts 01–08.

**Paper quality (no result changes):**
- **Abstract** rewritten for clarity and to frame the novelty as a *systematic, statistically grounded characterization* of disagreement / dominance / calibration / mitigation — explicitly *not* a claim to have discovered that modalities disagree.
- **Contributions** consolidated into 5 precise, bolded items tied to concrete evidence and uncertainty (CIs, McNemar).
- **Introduction** adds a sentence positioning prior work (modality over-reliance, modality gap) as the *starting point*, not the finding.
- **Related Work**: corrected the `huang2021makes` characterization.
- **Discussion**: softened over-generalized architecture claims.
- **Limitations** substantially strengthened: scope (Amazon-only), proxy/circularity of the disagreement definition, small weak/medium subgroups, single-split headline numbers, limited metadata, heuristic taxonomy with large "other" bucket, train-time-label requirement, incomplete category sweep.

**Verification:** paper recompiles to a 10-page PDF with no undefined references or citations; all 32 tests pass.

---

## 7. What Still Needs Manual Review

1. **End-to-end regeneration (optional but recommended):** run `make all && make extras && make multi-category` on a fully provisioned machine to confirm artifacts regenerate and numbers reproduce within tolerance. Not done here to avoid overwriting validated artifacts and because it needs large downloads + network.
2. **Multi-seed for the mitigation:** the disagreement-aware gain (+3.2 pp acc / +4.3 pp F1) is from a single seed/split. `make multi-seed` exists (`run_experiments.py`); running it would let the paper report variance on the mitigation effect.
3. **Author/affiliation block** in `paper/main.tex` (currently "Krishna … krish@ucsb.edu") — confirm this is the desired final attribution for submission.
4. **Error-taxonomy interpretation:** 46% of errors are "other"; treat category-level rates as indicative, per the strengthened limitation.

---

## 8. PSTAT 262DS Final-Project Assessment

The project comfortably meets the expectations of a graduate data-science final project:

- **Clear research question** with a non-trivial, well-motivated hypothesis (fusion under modality conflict).
- **End-to-end, reproducible pipeline** (config-driven scripts 01–10, Makefile, pinned `requirements.txt`, 32 passing unit tests).
- **Statistical rigor**: bootstrap 95% CIs, McNemar's tests, calibration (ECE, temperature scaling), selective prediction.
- **Methodological contribution**: a graded-disagreement evaluation protocol, three dominance lenses, group-conditional calibration, and a simple, effective mitigation (disagreement-aware reweighting).
- **Robustness**: three fusion architectures and three product categories.
- **Communication**: a complete 10-page paper with embedded results and figures, plus report/presentation outlines.

After the fixes above, the README, config, code, and paper are mutually consistent and all reported numbers trace to generated CSVs. Remaining items (Section 7) are confirmatory/optional rather than blocking. **The project is submission-ready.**
