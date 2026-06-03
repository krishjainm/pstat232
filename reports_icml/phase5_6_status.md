# Phase 5 & 6 Status: Disagreement Definitions + Human Annotation Pipeline

**Status: COMPLETE and VERIFIED.** Dataset: All_Beauty canonical pool (20k,
leakage-controlled), 5-fold cross-fitted predictions (no leakage).

## Phase 5 — Multiple disagreement definitions

Implemented in `scripts_icml/phase5_disagreement_definitions.py`.

| Definition | Prevalence | Multimodal acc (flagged) | Multimodal acc (unflagged) |
| --- | --- | --- | --- |
| A: sentiment vs rating label | 11.6% | 0.638 | 0.912 |
| B: cross-model (text vs meta) | 42.3% | 0.859 | 0.896 |
| C: high-conf text contradiction (≥0.90) | 1.3% | **0.008** | 0.892 |

Pairwise Jaccard overlap: A∩B = 0.11, A∩C = 0.068, B∩C = 0.014.

**Honest findings:**
- The three definitions flag **largely disjoint** instance sets (low Jaccard) —
  "disagreement" is strongly definition-dependent, so results must always state
  which definition is used.
- Definition A reproduces the canonical large failure gap (0.638 vs 0.912).
- Definition B is much looser (42% of the pool) with only a modest gap; it is a
  weak conflict signal on its own.
- Definition C is rare but the multimodal model is almost always wrong (0.008)
  on it — these are prime label-noise / sarcasm candidates, which is exactly what
  the human-annotation phase is designed to disentangle.

Artifacts: `reports_icml/tables/disagreement_definition_comparison.csv`,
`reports_icml/figures/disagreement_definition_overlap.png`,
`reports_icml/manual_annotation_sample.csv` (200 candidates, human-label columns
left blank).

## Phase 6 — Human annotation pipeline

- `reports_icml/annotation_guide.md` — six-category guide (genuine conflict,
  label noise, mixed sentiment, sarcasm, unclear, metadata-prior) with a
  decision order.
- `scripts_icml/create_annotation_batch.py` — generates additional annotation
  batches for any category (same schema), human labels blank.
- `scripts_icml/analyze_annotations.py` — aggregates completed annotations:
  category breakdown, Cohen's kappa inter-annotator agreement, and model accuracy
  per human category. Verified to **exit gracefully** ("annotation pending") when
  no completed files exist; no labels are fabricated.

Pending (requires human effort, not compute): actual manual labeling of the 200
candidates, after which `analyze_annotations.py` produces
`reports_icml/tables/human_annotation_results.csv` and
`reports_icml/figures/human_annotation_breakdown.png`.
