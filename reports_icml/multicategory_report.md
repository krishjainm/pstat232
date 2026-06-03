# Phase 2 — Multi-Category Generalization Report

## Protocol

- **Categories (5):** All_Beauty, Digital_Music, Gift_Cards, Appliances, Video_Games (Amazon Reviews 2023).
- **Seeds:** 42, 123, 456, with **redrawn** 70/15/15 stratified splits per seed.
- **Main setting:** leakage-controlled metadata (drops `product_average_rating`, `product_rating_number`). Full-feature early fusion (`early_fusion_full`) is reported only as a secondary/appendix row.
- **Pool sizes:** All_Beauty reuses the Phase-1 pool (20k); the other four use 12k balanced pools (capped by available negatives). Because Phase 2 tests whether a *within-category relative pattern* replicates, the differing pool sizes are not a confound for the gap/mitigation claims; they are documented for transparency.
- **Large categories:** Video_Games review and metadata JSONL reads are capped at 500k rows (file-order subsample) to remain feasible on CPU; Appliances reuses the already-materialized 50k subsample. Documented as a limitation.
- **Models:** text_only, metadata_lc, early_fusion_lc, disagree_aware_lc (main); early_fusion_full (appendix). UGCA-Fusion (Phase 4) will be added to this grid once implemented.

Outputs: `multicategory_multiseed_results.csv` (75 rows = 5×5×3), `multicategory_summary.csv`, and figures `multicategory_disagreement_gap.png`, `multicategory_calibration_gap.png`, `multicategory_mitigation_effect.png`.

## Cross-category summary (mean over 3 seeds, leakage-controlled)

| category | model | acc | agree acc | disagree acc | agree–disagree gap | calib gap | text-dominance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All_Beauty | text_only | 0.873 | 0.905 | 0.636 | 0.269 | 0.132 | — |
| All_Beauty | early_fusion_lc | 0.878 | 0.910 | 0.636 | 0.274 | 0.207 | 90.6% |
| All_Beauty | disagree_aware_lc | 0.865 | 0.892 | 0.660 | 0.232 | 0.203 | 89.0% |
| Digital_Music | text_only | 0.876 | 0.915 | 0.596 | 0.319 | 0.108 | — |
| Digital_Music | early_fusion_lc | 0.886 | 0.921 | 0.636 | 0.285 | 0.190 | 91.0% |
| Digital_Music | disagree_aware_lc | 0.860 | 0.894 | 0.618 | 0.276 | 0.147 | 88.7% |
| Gift_Cards | text_only | 0.914 | 0.953 | 0.568 | 0.385 | 0.249 | — |
| Gift_Cards | early_fusion_lc | 0.924 | 0.954 | 0.646 | 0.308 | 0.263 | 88.1% |
| Gift_Cards | disagree_aware_lc | 0.911 | 0.936 | 0.684 | 0.252 | 0.209 | 83.7% |
| Appliances | text_only | 0.861 | 0.903 | 0.644 | 0.259 | 0.177 | — |
| Appliances | early_fusion_lc | 0.871 | 0.911 | 0.662 | 0.249 | 0.231 | 86.3% |
| Appliances | disagree_aware_lc | 0.856 | 0.880 | 0.732 | 0.148 | 0.125 | 85.8% |
| Video_Games | text_only | 0.870 | 0.912 | 0.603 | 0.309 | 0.158 | — |
| Video_Games | early_fusion_lc | 0.872 | 0.916 | 0.586 | 0.330 | 0.253 | 89.9% |
| Video_Games | disagree_aware_lc | 0.853 | 0.886 | 0.639 | 0.247 | 0.134 | 88.4% |

(Full-feature appendix rows are in `multicategory_summary.csv`.)

## Main claim under test

> *Does multimodal fusion consistently improve overall performance while failing disproportionately under modality disagreement?*

**Honest answer, split into its two parts:**

1. **"Improves overall" — yes, but small and consistent.** Leakage-controlled early fusion beats text-only on overall accuracy in all 5 categories, by **+0.002 to +0.010** (All_Beauty +0.005, Digital_Music +0.010, Gift_Cards +0.010, Appliances +0.010, Video_Games +0.002). The gains are uniformly small once leakage is removed (significance deferred to Phase 8).

2. **"Fails disproportionately under disagreement" — the *gap* is universal; the *fusion-vs-text disadvantage* is not.** In **every** category, the disagreement subset is dramatically harder than agreement for every model: the early-fusion agreement→disagreement drop is **0.25–0.33** (e.g. Video_Games 0.916→0.586). Calibration also degrades under conflict (fusion calib gap 0.19–0.26 across categories). **However**, whether fusion is specifically *worse than text-only on the disagreement subset* is category-dependent: fusion helps disagreement in Gift_Cards (+0.078), Digital_Music (+0.040) and Appliances (+0.018), is neutral in All_Beauty (0.000), and *hurts* in Video_Games (−0.017).

**Replicated conclusion (cautious):** *Across the five evaluated categories, multimodal fusion yields small, consistent average gains while the disagreement subset remains a severe, universally replicated failure regime (25–33 pp accuracy drop and elevated miscalibration). Fusion does not consistently rescue or consistently worsen the disagreement subset relative to text-only; the conflict regime is hard for all models.* This is a robust pattern, not a claim that fusion is uniquely harmful.

## Mitigation (disagreement-aware reweighting)

`disagree_aware_lc` improves disagreement accuracy over `early_fusion_lc` in **4 of 5** categories — Appliances +0.070, Video_Games +0.053, Gift_Cards +0.038, All_Beauty +0.024 — but *reduces* it in Digital_Music (−0.018). It also shrinks the calibration gap in several categories (Appliances 0.231→0.125, Video_Games 0.253→0.134) and consistently costs ~1 pp of overall accuracy. So the simple mitigation is a modest, mostly-positive but **not universal** intervention — motivating the learned UGCA-Fusion method (Phase 4).

## Modality dominance

In true-conflict cases the fusion model follows text **84–91%** of the time across all categories — a robust, category-independent text-dominance.

## Limitations carried forward

- Differing pool sizes across categories (12k vs 20k) and file-order subsampling for Video_Games/Appliances.
- Disagreement is still the single-model proxy (Definition A); Phase 5 adds alternatives.
- Per-seed paired significance tests across categories are produced in Phase 8.
