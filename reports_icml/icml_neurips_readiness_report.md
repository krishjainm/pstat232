# ICML/NeurIPS Readiness Report

Branch: `icml-neurips-upgrade`. All new artifacts are isolated under
`reports_icml/`, `models_icml/`, `data_icml/`, `paper_icml/`, `scripts_icml/`;
the original `reports/` and `paper/` are untouched. Every run is logged in
`reports_icml/experiment_registry.md`. Compute: **CPU-only** throughout.

## 1. What was completed

| Phase | Description | Status |
| --- | --- | --- |
| 0 | Branch + isolated dirs + experiment registry | Done |
| 1 | Canonical reproduction, All\_Beauty, 5 seeds, 8 models, full metric suite, bootstrap CIs + paired tests | Done & verified |
| 2 | Multi-category (5 categories x 3 seeds), leakage-controlled main setting | Done & verified |
| 3 | Second-dataset **feasibility report** (Yelp chosen; MOSI/MOSEI/Hateful Memes = GPU future work) | Report done; implementation deferred |
| 4 | **UGCA-Fusion** model + trainer + 5-seed benchmark + 4 ablations + conflict-detector AUROC | Done & verified |
| 5 | Four disagreement definitions (A/B/C/D) via cross-fitted predictions | Done & verified |
| 6 | Human-annotation pipeline (guide + batch generator + analyzer, graceful "pending") | Done & verified |
| 7 | Calibration / selective prediction | **Skipped by request** (metric suite already reports ECE/calibration gap by group) |
| 8 | Statistical rigor: seed-paired + example-level (McNemar, paired bootstrap, effect sizes) | Done & verified |
| 9 | ICML/NeurIPS paper (`paper_icml/main.tex` + `references.bib`, compiles to 6-page PDF) | Done & verified |
| 10 | This report | Done |

### Figure coverage note
Cross-category figures (`multicategory_disagreement_gap.png`,
`multicategory_calibration_gap.png`, `multicategory_mitigation_effect.png`)
include **text-only, metadata-LC, early-fusion-LC, and disagreement-aware
reweighting only**. UGCA-Fusion is benchmarked separately on the canonical
All_Beauty 5-seed run and is **not** mixed into the cross-category figures
(to avoid combining complete and incomplete category coverage); each figure now
carries this caption. The UGCA figures (`ugca_vs_baselines.png`,
`ugca_ablation.png`, `conflict_detector_auc.png`) are labeled **"All_Beauty
only"**. All figures are regenerated from, and consistent with, their current
source CSVs.

## 2. Experiments that ran successfully

- Phase 1: `canonical_allbeauty_multiseed.csv` (40 rows = 8 models x 5 seeds),
  summary, per-seed stats, error-bar figure, narrative report.
- Phase 2: `multicategory_multiseed_results.csv` (75 rows = 5 models x 5 cats x
  3 seeds), summary, three cross-category figures, report.
- Phase 4: `ugca_results.csv`, `ugca_ablation.csv` (40 rows), three figures.
- Phase 5: `disagreement_definition_comparison.csv`, overlap figure,
  `manual_annotation_sample.csv` (200 candidates).
- Phase 8: `statistical_tests_main.csv`, `effect_sizes.csv`, report.

## 3. Strongest results

1. **Fusion widens the conflict calibration gap (replicated).** Early fusion
   raises the agreement/disagreement calibration gap from 0.120 to 0.189 on
   All\_Beauty (paired $p{=}0.002$, $d_z{=}3.2$), and the increase holds in
   **all 5 categories**. This is the paper's most robust, multi-seed,
   cross-domain finding.
2. **Leakage control matters and is honestly costly.** Removing
   `product_average_rating`/`product_rating_number` significantly lowers accuracy
   ($-0.8$ pts, $p{=}0.024$) and disagreement accuracy ($-2.4$ pts, $p{=}0.011$);
   example-level McNemar agrees ($p{=}0.026$). The phenomenon is therefore not a
   product-prior artifact.
3. **Conflict is detectable at inference without labels.** UGCA's conflict
   detector reaches AUROC ${\approx}0.72$ across 5 seeds.
4. **Disagreement is definition-dependent.** The three automatic definitions
   flag near-disjoint sets (Jaccard $\le0.11$).

## 4. Weak or negative results (reported honestly)

- **Fusion does not help on the disagreement subset.** Early vs.\ text-only
  disagreement accuracy: $+0.002$, $p{=}0.79$ (All\_Beauty); sign is inconsistent
  across categories (e.g.\ Video\_Games worse).
- **Fusion's overall accuracy gain is tiny** ($+0.5$ pts) despite being
  statistically significant.
- **UGCA-Fusion does not beat baselines on accuracy/disagreement.** Indistinct
  from early fusion (paired $p{>}0.7$; McNemar median $p{>}0.5$); gated fusion is
  better calibrated (ECE 0.018 vs 0.050).

## 5. Does UGCA-Fusion beat baselines?

**Partially / diagnostically, not decisively.**
- YES on overall accuracy vs.\ disagreement-aware reweighting ($+1.6$ pts,
  $p{=}0.001$): UGCA avoids the overall-accuracy sacrifice that reweighting makes.
- YES as a **label-free conflict detector** (AUROC ${\approx}0.72$).
- NO on disagreement accuracy vs.\ early fusion or vs.\ reweighting (not
  significant; reweighting is marginally best on the disagreement subset).
- NO on calibration vs.\ gated fusion.
Conclusion: UGCA-Fusion is a useful diagnostic, not a solution to conflict.

## 6. Are improvements statistically significant?

- Significant: fusion's overall accuracy gain; fusion's calibration-gap increase;
  leakage-control effect; UGCA vs.\ reweighting on overall accuracy. (All paired
  $t$-test $p<0.05$ with large $d_z$, corroborated by bootstrap CIs and, where
  applicable, McNemar.)
- Not significant: fusion / UGCA gains on the disagreement subset.
- Caveat: only 5 seeds; we emphasize effect sizes and CIs over $p$ thresholds.
  See `reports_icml/statistical_testing_report.md`.

## 7. Remaining gaps before real submission

1. **No completed human annotations** (pipeline ready; 200 candidates exported).
   Needed to quantify genuine conflict vs.\ label noise.
2. **Second dataset not yet run** (Yelp scoped in feasibility report). A true
   cross-modality dataset (CMU-MOSEI / Hateful Memes) requires GPU.
3. **Frozen encoders / capped pools** due to CPU-only compute; end-to-end
   fine-tuning untested.
4. **Phase 7** (temperature scaling, risk-coverage curves) not run.
5. UGCA needs a setting where it clearly wins, or an explicit framing as a
   diagnostic tool.

## 8. Recommended next experiments

1. Collect the 200 human annotations; recompute model accuracy on genuine
   conflict vs.\ label noise (closes the largest credibility gap cheaply).
2. Implement the scoped **Yelp** second dataset (full reuse of `icml_common`).
3. On GPU: fine-tune encoders end-to-end; add CMU-MOSEI or Hateful Memes for a
   genuine image/audio${+}$text conflict test.
4. Phase 7: group-wise temperature scaling and risk-coverage curves per model.
5. Increase to 10+ seeds and add Holm correction for the multiple comparisons.
6. Stronger UGCA: contrastive conflict objective, conflict-conditioned abstention.

## 9. Honest rating

- As **a class project**: clearly exceeds it.
- As **a workshop paper**: yes, comfortably submittable now (rigorous,
  leakage-controlled, multi-seed, multi-category, honest negatives, a novel
  diagnostic method).
- As **a main-conference paper (ICML/NeurIPS)**: **not yet.** The core
  observational finding (fusion widens the conflict calibration gap, robustly)
  is strong, but the method (UGCA-Fusion) does not yet deliver a decisive
  improvement, there is no completed human-annotation evidence, and there is no
  second dataset *family* or cross-modality result. With (a) human annotations,
  (b) the Yelp run, and (c) one GPU cross-modality dataset plus a sharper UGCA
  result, this would be a credible main-conference submission.

**Overall: a strong workshop paper today; a clear, well-scoped path to main
conference.** Negative results are reported rather than hidden, consistent with
the project's stated success criteria.
