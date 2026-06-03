# Phase 2 Status — Multi-Category Evaluation

**Status: COMPLETE.** All 5 target categories ran successfully.

## What ran

- Categories: **All_Beauty, Digital_Music, Gift_Cards, Appliances, Video_Games**.
- Seeds 42/123/456 with redrawn splits; leakage-controlled features as the main
  setting; full-feature early fusion as an appendix row.
- Models: text_only, metadata_lc, early_fusion_lc, disagree_aware_lc, early_fusion_full.
- 75 result rows = 5 categories × 5 models × 3 seeds, all logged to the registry.

## Outputs (verified)

| file | content |
| --- | --- |
| `reports_icml/tables/multicategory_multiseed_results.csv` | 75 rows |
| `reports_icml/tables/multicategory_summary.csv` | per category × model mean±std |
| `reports_icml/figures/multicategory_disagreement_gap.png` | agreement vs disagreement acc per category |
| `reports_icml/figures/multicategory_calibration_gap.png` | calibration gap per category (text vs fusion) |
| `reports_icml/figures/multicategory_mitigation_effect.png` | disagreement acc: fusion vs reweighting |
| `reports_icml/multicategory_report.md` | full narrative + honest claim assessment |

## Key findings (honest)

1. **The disagreement-failure pattern replicates in all 5 categories**: early
   fusion drops 25–33 pp from agreement (~0.91–0.95) to disagreement
   (~0.59–0.66), with calibration gaps of 0.19–0.26.
2. **Fusion's overall gain over text-only is small but consistent** (+0.002 to
   +0.010 across categories).
3. **Fusion vs text-only on the disagreement subset is category-dependent**
   (helps in 3, neutral in 1, hurts in 1) — so we do *not* claim fusion is
   uniquely harmful under conflict; we claim the conflict regime is universally
   hard.
4. **Disagreement-aware reweighting helps disagreement in 4/5 categories** (up to
   +0.070) and reduces calibration gaps, but is not universal (hurts
   Digital_Music) and costs ~1 pp overall — motivating UGCA-Fusion.
5. **Text dominance (84–91%)** is category-independent.

## Caveats / limitations

- Pool sizes differ (All_Beauty 20k; others 12k) and Video_Games/Appliances use
  capped/file-order subsamples — acceptable for the within-category relative
  pattern, documented in the report.
- UGCA-Fusion (Phase 4) is not yet in this grid; it will be added after Phase 4.
- Cross-category paired significance is produced in Phase 8.

**Phase 2 verified. Ready to proceed (Phase 3 second dataset, or Phase 4
UGCA-Fusion) on request.**
