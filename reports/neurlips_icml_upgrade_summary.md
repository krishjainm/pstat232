# Research-Upgrade Summary (toward an ICML/NeurIPS-style prototype)

**Date:** 2026-06-01
**Guiding rule followed:** no fabricated, cherry-picked, or invented results. Every
number below comes from code run on real data. Where an experiment could not be run
in this environment without overwriting validated artifacts or large downloads, it is
labeled as such rather than faked.

---

## 1. What was added

| Area | Artifact(s) | Data used | Real? |
|------|-------------|-----------|-------|
| Baseline freeze + data-state audit | `research_upgrade_baseline_check.md` | committed All_Beauty artifacts | ✅ |
| Extended statistical tests (paired bootstrap + McNemar; leakage-free half-split temp scaling) | `tables/statistical_tests_extended.csv` | All_Beauty test predictions (n=3102) | ✅ |
| Qualitative case studies (6 archetypes) | `tables/qualitative_case_studies.csv`, `qualitative_case_studies.md` | All_Beauty test predictions | ✅ |
| Leakage audit (correlations + CV ablation + figure) | `leakage_audit.md`, `tables/metadata_correlations.csv`, `figures/metadata_feature_importance_no_leakage.png` | All_Beauty test set | ✅ |
| Ablations (metadata / text-encoder / fusion) | `tables/ablation_results.csv`, `figures/ablation_summary.png` | All_Beauty CV + Appliances + committed fusion CSV | ✅ |
| Mitigation comparison (fixed/severity/focal/temperature) | `tables/mitigation_comparison.csv`, 3 `figures/mitigation_*.png` | Appliances split (materialized) | ✅ |
| Multi-seed robustness (42/123/456) | `tables/multiseed_results*.csv`, `figures/multiseed_error_bars.png` | Appliances split | ✅ (training variance) |
| Cross-category figures | 3 `figures/cross_category_*.png` | committed `cross_category_results.csv` | ✅ |
| Paper sections + conservative reframing | `paper/main.tex` (now 12 pp) | — | — |
| Richer cross-category metrics (ECE/Brier/strong/calibration gap) | extended `scripts/run_all_categories.py` | **requires provisioned rerun** | code-ready |

A note on data state: the on-disk processed split is **Appliances** (a prior
cross-category run overwrote it); the validated **All_Beauty** test set survives as
`test_with_predictions.parquet`. No-retraining analyses were therefore run on
All_Beauty; retraining-dependent analyses (mitigation, multi-seed, text-encoder) were
run on the materialized Appliances split and labeled accordingly. Running mitigation
on a *second* category doubles as a cross-domain transfer check.

---

## 2. Which results improved / got stronger evidence

- **Cross-category replication is real and consistent.** The disagreement gap (overall −
  disagreement accuracy) replicates across all three successful categories:
  All_Beauty −25 pp, Digital_Music −24 pp, Appliances −15 pp; text dominance 68–82%.
- **A genuine, honest mitigation comparison** now exists. On Appliances, **severity-aware
  reweighting** is the best accuracy-oriented mitigation: disagreement acc 0.791→**0.815**
  (+2.4 pp), strong-disagreement 0.767→**0.800** (+3.3 pp), for −1.4 pp overall.
  **Focal loss / temperature scaling** improve calibration (overall ECE 0.064→0.034 / **0.020**)
  but not disagreement accuracy. This "division of labor" is a cleaner story than the
  single-method claim.
- **Leakage is now quantified.** `product_average_rating` (point-biserial r=0.38 with the
  label) drives most of the metadata model's skill: CV accuracy 0.66 (8 feats) → **0.55**
  without it. We designate the no-`product_average_rating` setting as the conservative one.
- **Statistical rigor strengthened with an honest negative.** Paired bootstrap + McNemar
  confirm fusion > text overall (p<0.001), but show fusion is **not** significantly better
  than text-only on *strong* disagreement (p=0.136) nor better than metadata-only there
  (p=0.53). The fusion advantage concentrates in the agreement regime.
- **Encoder robustness.** TF-IDF+LogReg (0.880) is competitive with SBERT (0.870) on
  Appliances → the disagreement phenomenon is not an artifact of the text encoder.

## 3. Which results stayed weak / negative (reported, not hidden)

- **Mitigation gains are modest vs. seed variance.** Across seeds 42/123/456 (Appliances):
  Standard disagreement acc 0.776±0.016 vs. DA(w=5) 0.790±0.015; strong 0.752±0.021 vs.
  0.773±0.013. The improvement is real but within ~1 std — not a decisive fix.
- **No method closes the calibration gap under disagreement.** Disagreement-subset ECE stays
  ≈0.15–0.18 across all mitigations, including group-wise temperature scaling.
- **Fixed reweighting did not transfer cleanly.** On Appliances, small fixed weights
  (w=2,3) did *not* improve disagreement accuracy (unlike All_Beauty's w=5 result); only
  severity-aware / larger weights helped. Mitigation efficacy is category-dependent.
- **"Disagreement" remains a proxy.** Qualitative case 4 (text clearly negative, rating 5★ →
  label "positive") shows some disagreement cases are label noise, not genuine semantic
  conflict.

## 4. What would still be needed for a true ICML/NeurIPS submission

1. **A clean, single-category, fully-reproduced run** of the entire upgrade on one canonical
   category (preferably All_Beauty) so mitigation, multi-seed, and ablations all share one
   split. (Here they are split across All_Beauty and Appliances for environmental reasons.)
2. **Full multi-seed × multi-category** with re-drawn splits (not just training stochasticity),
   reporting CIs on the *mitigation effect*, plus the richer per-category table
   (ECE/Brier/strong/calibration gap) — the extended `run_all_categories.py` is ready for this.
3. **A real baseline beyond Amazon text+metadata** — at least one vision–language or
   audio–text dataset — to support any claim of generality. Current scope is Amazon reviews only.
4. **A leakage-controlled main result**: re-run the headline comparison with
   `product_average_rating` removed and confirm the disagreement story is unchanged.
5. **Human-annotated disagreement / error taxonomy** on a sample, to separate genuine
   semantic conflict from rating/label noise, and to validate the heuristic taxonomy.
6. **A learned or theoretically motivated mitigation** (e.g. uncertainty-weighted fusion,
   modality dropout, modality-conditioned calibration) benchmarked against the simple ones
   here, ideally with detection of disagreement at inference time (no ground-truth labels).
7. **Compute**: GPU runs to make the multi-seed/multi-category/mitigation grid affordable and
   to add variance estimates everywhere.

## 5. Honest assessment of remaining limitations

- Scope is a single dataset family (Amazon Reviews 2023); we claim the *methodology*
  generalizes, not the specific numbers.
- The disagreement definition is proxy-based and the disagreement subset is enriched for
  hard/mislabeled examples.
- The metadata modality has a leakage-prone feature; the conservative setting is provided
  but the paper's main table still uses the full feature set.
- Mitigation and multi-seed evidence live on the Appliances split and vary only training
  stochasticity; this is a lower bound on variance.
- Small weak/medium disagreement subgroups (n=27/50 on All_Beauty) give wide CIs.

## 6. Bottom line

The project moved from "a class project that observes modality disagreement" to "a
**systematic, statistically grounded study** of disagreement, dominance, calibration, and
a comparison of simple mitigations, with an explicit leakage audit and honest negative
results." It is a credible **research prototype / strong workshop-paper** in its current
form. It is **not yet** a main-conference ICML/NeurIPS submission, primarily because of
(a) single-dataset scope, (b) a split-inconsistent mitigation/robustness study, and
(c) the absence of a novel, learned mitigation benchmarked under re-drawn-split,
multi-seed, multi-category conditions. None of these gaps were papered over with invented
numbers; they are the concrete, fundable next steps listed in §4.
