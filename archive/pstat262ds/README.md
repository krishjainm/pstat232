# Archived PSTAT 262DS materials

This folder holds class/submission-specific artifacts from the original
**PSTAT 262DS** project ("When Modalities Disagree"). They are kept for
provenance but are **not** part of the PSTAT 232 deliverable.

## Contents

- `presentation_outline.md` — slide/presentation outline for the 262DS submission.
- `final_report_outline.md` — outline of the original 262DS report.

## What was *not* archived (intentionally retained)

The following were **kept in place** because the PSTAT 232 work reuses them or
references them as prior work:

- `src/` — data, features, models, evaluation, and visualization modules
  (reused directly by the new PSTAT 232 scripts via `scripts_icml/icml_common.py`).
- `scripts/01..10_*.py` and `scripts/upgrade_*.py` — original pipeline scripts.
- `scripts_icml/`, `data_icml/`, `reports_icml/`, `paper_icml/` — the
  ICML/NeurIPS upgrade infrastructure (canonical pools, cached SBERT, the metric
  suite, and bootstrap/McNemar helpers the PSTAT 232 scripts are built on).
- `paper/main.tex`, `paper/references.bib` — the original 262DS paper, retained
  as the prior-work record.
- `reports/figures/*.png`, `reports/tables/*.csv` (non-`pstat232_*`) — original
  results, some of which are discussed as the leaky-configuration baseline.

See `../../PSTAT232_REFACTOR_PLAN.md` for the full mapping of reused / changed /
archived files.
