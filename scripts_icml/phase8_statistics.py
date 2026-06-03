"""Phase 8: statistical rigor for the key claims.

Two complementary levels of analysis:

(1) Seed-level paired analysis (the unit is a redrawn split / seed). For each key
    comparison and metric we take the per-seed paired differences and report:
    mean +/- std, a t-based 95% CI, a bootstrap-over-seeds 95% CI, a paired
    t-test and Wilcoxon p-value, and the paired effect size Cohen's dz.
    Source: the multi-seed metric CSVs already produced in Phases 1/2/4.

(2) Example-level paired tests (the unit is a test instance) for accuracy-style
    claims: McNemar's test + paired bootstrap on the difference in accuracy,
    overall and on the disagreement subset. We reuse the cached per-seed
    example-level stats from Phase 1 (canonical_allbeauty_stats.csv) and
    recompute the UGCA comparisons across all 5 seeds (not previously cached).

Outputs:
  reports_icml/tables/statistical_tests_main.csv
  reports_icml/tables/effect_sizes.csv
  reports_icml/statistical_testing_report.md
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

from scripts_icml import icml_common as C
from scripts_icml.train_ugca import train_ugca, predict_ugca

CAT = "All_Beauty"
SEEDS = [42, 123, 456, 789, 2026]

CANON = "reports_icml/tables/canonical_allbeauty_multiseed.csv"
ABL = "reports_icml/tables/ugca_ablation.csv"
MULTI = "reports_icml/tables/multicategory_multiseed_results.csv"
CANON_STATS = "reports_icml/tables/canonical_allbeauty_stats.csv"


# ---------------------------------------------------------------------------
# (1) Seed-level paired analysis
# ---------------------------------------------------------------------------
def _paired_vectors(df, model_a, model_b, metric, key=("seed",)):
    a = df[df["model"] == model_a].set_index(list(key))[metric]
    b = df[df["model"] == model_b].set_index(list(key))[metric]
    common = a.index.intersection(b.index)
    return a.loc[common].values.astype(float), b.loc[common].values.astype(float)


def _boot_ci_over_units(deltas, n=5000, seed=0):
    rng = np.random.RandomState(seed)
    N = len(deltas)
    if N < 2:
        return np.nan, np.nan
    means = [deltas[rng.randint(0, N, N)].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seed_level_row(df, model_a, model_b, metric, subset_label, key=("seed",),
                   source=""):
    a, b = _paired_vectors(df, model_a, model_b, metric, key)
    d = a - b
    nseed = len(d)
    mean_d, std_d = float(d.mean()), float(d.std(ddof=1)) if nseed > 1 else np.nan
    se = std_d / np.sqrt(nseed) if nseed > 1 else np.nan
    tcrit = stats.t.ppf(0.975, nseed - 1) if nseed > 1 else np.nan
    t_lo = mean_d - tcrit * se if nseed > 1 else np.nan
    t_hi = mean_d + tcrit * se if nseed > 1 else np.nan
    b_lo, b_hi = _boot_ci_over_units(d)
    # paired t-test
    if nseed > 1 and np.any(d != 0):
        t_p = float(stats.ttest_rel(a, b).pvalue)
    else:
        t_p = np.nan
    # Wilcoxon (needs nonzero diffs)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue) if nseed >= 5 and np.any(d != 0) else np.nan
    except Exception:
        w_p = np.nan
    dz = mean_d / std_d if (nseed > 1 and std_d and std_d > 0) else np.nan
    return {
        "comparison": f"{model_a} vs {model_b}", "subset": subset_label,
        "metric": metric, "level": "seed-paired", "source": source,
        "n_units": nseed, "mean_delta": round(mean_d, 4),
        "std_delta": round(std_d, 4) if nseed > 1 else np.nan,
        "t_ci_lo": round(t_lo, 4) if nseed > 1 else np.nan,
        "t_ci_hi": round(t_hi, 4) if nseed > 1 else np.nan,
        "boot_ci_lo": round(b_lo, 4) if not np.isnan(b_lo) else np.nan,
        "boot_ci_hi": round(b_hi, 4) if not np.isnan(b_hi) else np.nan,
        "ttest_p": round(t_p, 4) if not np.isnan(t_p) else np.nan,
        "wilcoxon_p": round(w_p, 4) if not np.isnan(w_p) else np.nan,
        "cohens_dz": round(dz, 3) if not np.isnan(dz) else np.nan,
    }


def part1_seed_level():
    canon = pd.read_csv(CANON)
    abl = pd.read_csv(ABL)
    multi = pd.read_csv(MULTI)
    rows, eff = [], []

    comps = [
        # (df, A, B, subset_label, [metrics], source)
        (canon, "early_fusion_lc", "text_only", "overall", ["accuracy"], "canon"),
        (canon, "early_fusion_lc", "text_only", "overall", ["disagreement_acc", "calibration_gap"], "canon"),
        (abl, "ugca_full", "early_fusion_lc", "overall", ["disagreement_acc"], "ugca"),
        (abl, "ugca_full", "disagree_aware_lc", "overall", ["disagreement_acc", "accuracy"], "ugca"),
        (abl, "ugca_full", "early_fusion_lc", "overall", ["calibration_gap", "ece"], "ugca"),
        (canon, "early_fusion_lc", "early_fusion_full", "overall", ["accuracy", "disagreement_acc"], "canon"),
    ]
    for df, a, b, sub, metrics, src in comps:
        for met in metrics:
            r = seed_level_row(df, a, b, met, sub, source=src)
            rows.append(r)
            eff.append({"comparison": r["comparison"], "subset": sub, "metric": met,
                        "n_seeds": r["n_units"], "mean_delta": r["mean_delta"],
                        "std_delta": r["std_delta"], "cohens_dz": r["cohens_dz"]})

    # Cross-domain (pooled over category x seed; flagged as non-independent)
    for met in ["accuracy", "disagreement_acc"]:
        r = seed_level_row(multi, "early_fusion_lc", "text_only", met,
                           "pooled-multicat", key=("category", "seed"), source="multicat")
        rows.append(r)
        eff.append({"comparison": r["comparison"], "subset": "pooled-multicat",
                    "metric": met, "n_seeds": r["n_units"],
                    "mean_delta": r["mean_delta"], "std_delta": r["std_delta"],
                    "cohens_dz": r["cohens_dz"]})
    return rows, eff


# ---------------------------------------------------------------------------
# (2) Example-level paired tests
# ---------------------------------------------------------------------------
def part2_example_level(pool, emb):
    """Reuse cached Phase-1 example-level stats; recompute UGCA comparisons."""
    rows = []

    # --- 2a. cached canonical example-level stats (already 5 seeds) ---
    if os.path.exists(CANON_STATS):
        cs = pd.read_csv(CANON_STATS)
        g = cs.groupby(["model_a", "model_b", "subset"]).agg(
            mean_delta=("delta_accuracy", "mean"),
            median_boot_p=("paired_bootstrap_p", "median"),
            median_mcnemar_p=("mcnemar_p", "median"),
            n_seeds=("seed", "count"),
        ).reset_index()
        for _, r in g.iterrows():
            rows.append({
                "comparison": f"{r['model_a']} vs {r['model_b']}",
                "subset": r["subset"], "metric": "accuracy",
                "level": "example (per-seed McNemar+bootstrap, median over seeds)",
                "source": "canon_stats", "n_units": int(r["n_seeds"]),
                "mean_delta": round(r["mean_delta"], 4),
                "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
                "ttest_p": np.nan, "wilcoxon_p": np.nan, "cohens_dz": np.nan,
                "mcnemar_p_median": round(r["median_mcnemar_p"], 4),
                "bootstrap_p_median": round(r["median_boot_p"], 4),
            })

    # --- 2b. UGCA example-level comparisons across 5 seeds (recompute) ---
    lc = [c for c in C.METADATA_LEAKAGE_CONTROLLED if c in pool.columns]
    ug_records = {("ugca_full", "early_fusion_lc", "overall"): [],
                  ("ugca_full", "early_fusion_lc", "disagreement"): [],
                  ("ugca_full", "disagree_aware_lc", "disagreement"): []}
    for seed in SEEDS:
        print(f"[p8] example-level UGCA recompute seed {seed}", flush=True)
        tr, va, te = C.make_split(pool, seed)
        train_df, val_df, test_df = pool.iloc[tr], pool.iloc[va], pool.iloc[te].reset_index(drop=True)
        y_tr, y_va = train_df["label"].values, val_df["label"].values
        y_te = test_df["label"].values
        emb_tr, emb_va, emb_te = emb[tr], emb[va], emb[te]
        ml_tr, ml_va, ml_te = C.scale_meta(train_df, [train_df, val_df, test_df], lc)
        disag_mask = (test_df["agreement_status"].values == "disagreement")
        disagree_tr = (train_df["agreement_status"].values == "disagreement").astype(np.float32)

        early = C.train_fusion("early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va,
                               emb_te, ml_te, seed)[0]
        w = np.ones(len(train_df), dtype=np.float32)
        w[train_df["agreement_status"].values == "disagreement"] = 5.0
        daware = C.train_fusion("early", emb_tr, ml_tr, y_tr, emb_va, ml_va, y_va,
                                emb_te, ml_te, seed, sample_weights=w)[0]
        model = train_ugca(emb_tr, ml_tr, y_tr, disagree_tr, emb_va, ml_va, y_va, seed)
        ugca = predict_ugca(model, emb_te, ml_te)[0]

        for (a, b, sub), preds in [
            (("ugca_full", "early_fusion_lc", "overall"), (ugca, early)),
            (("ugca_full", "early_fusion_lc", "disagreement"), (ugca, early)),
            (("ugca_full", "disagree_aware_lc", "disagreement"), (ugca, daware)),
        ]:
            mask = None if sub == "overall" else disag_mask
            d_acc, p_boot = C.paired_bootstrap_test(y_te, preds[0], preds[1], mask=mask)
            chi2, p_mc, nb, nc = C.mcnemar(y_te, preds[0], preds[1], mask=mask)
            ug_records[(a, b, sub)].append((d_acc, p_boot, p_mc))

    for (a, b, sub), recs in ug_records.items():
        arr = np.array(recs)
        rows.append({
            "comparison": f"{a} vs {b}", "subset": sub, "metric": "accuracy",
            "level": "example (per-seed McNemar+bootstrap, median over seeds)",
            "source": "ugca_recompute", "n_units": len(recs),
            "mean_delta": round(float(arr[:, 0].mean()), 4),
            "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
            "ttest_p": np.nan, "wilcoxon_p": np.nan, "cohens_dz": np.nan,
            "mcnemar_p_median": round(float(np.median(arr[:, 2])), 4),
            "bootstrap_p_median": round(float(np.median(arr[:, 1])), 4),
        })
    return rows


def write_report(main_df, eff_df):
    L = ["# Statistical Testing Report (Phase 8)\n",
         "All comparisons use the leakage-controlled setting on All_Beauty unless "
         "noted. Two units of analysis are reported: **seed-paired** (n=5 redrawn "
         "splits; t-test, Wilcoxon, bootstrap-over-seeds CI, Cohen's dz) and "
         "**example-level** (McNemar + paired bootstrap per seed, median p over "
         "seeds). With only 5 seeds, p-values are interpreted cautiously and we "
         "emphasize effect sizes and CIs over thresholds.\n",
         "\n## Seed-level paired tests\n",
         "| comparison | subset | metric | mean delta | t-CI | boot-CI | t-test p | Wilcoxon p | Cohen's dz |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in main_df[main_df["level"] == "seed-paired"].iterrows():
        L.append(f"| {r['comparison']} | {r['subset']} | {r['metric']} | "
                 f"{r['mean_delta']:+.4f} | [{r['t_ci_lo']}, {r['t_ci_hi']}] | "
                 f"[{r['boot_ci_lo']}, {r['boot_ci_hi']}] | {r['ttest_p']} | "
                 f"{r['wilcoxon_p']} | {r['cohens_dz']} |")

    L += ["\n## Example-level tests (median over seeds)\n",
          "| comparison | subset | mean delta acc | median McNemar p | median bootstrap p | n seeds |",
          "| --- | --- | --- | --- | --- | --- |"]
    ex = main_df[main_df["level"].str.startswith("example")]
    for _, r in ex.iterrows():
        L.append(f"| {r['comparison']} | {r['subset']} | {r['mean_delta']:+.4f} | "
                 f"{r.get('mcnemar_p_median','')} | {r.get('bootstrap_p_median','')} | "
                 f"{r['n_units']} |")

    L += ["\n## Honest reading\n",
          "- **Early fusion vs text-only (overall):** see mean delta and CI above; "
          "in our setting the overall-accuracy difference is small.\n",
          "- **Disagreement subset:** the large raw gaps reported in Phases 1/5 are "
          "between agreement and disagreement *subsets*, not necessarily a "
          "significant *model-vs-model* difference on the same subset.\n",
          "- **UGCA vs baselines:** UGCA does not show a statistically clear "
          "advantage over early fusion or over disagreement-aware reweighting on "
          "disagreement accuracy (CIs include 0 / large median p). Its demonstrated "
          "value is the conflict detector (Phase 4, AUROC ~0.72), not an accuracy win.\n",
          "- **Leakage control:** full-feature early fusion vs leakage-controlled "
          "early fusion quantifies how much product-level priors inflate accuracy; "
          "see the corresponding row.\n",
          "- Effect sizes are in `effect_sizes.csv`; full numbers in "
          "`statistical_tests_main.csv`.\n"]
    with open("reports_icml/statistical_testing_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[report] reports_icml/statistical_testing_report.md")


def main():
    os.makedirs("reports_icml/tables", exist_ok=True)
    rows1, eff = part1_seed_level()
    pool = C.build_canonical_pool(CAT)
    emb = C.load_sbert(CAT)
    rows2 = part2_example_level(pool, emb)

    main_df = pd.DataFrame(rows1 + rows2)
    main_df.to_csv("reports_icml/tables/statistical_tests_main.csv", index=False)
    eff_df = pd.DataFrame(eff)
    eff_df.to_csv("reports_icml/tables/effect_sizes.csv", index=False)
    write_report(main_df, eff_df)

    C.log_experiment(dict(
        phase=8, dataset="Amazon-Reviews-2023", category=CAT, seed="42-2026",
        split_id="multi", model="all (post-hoc stats)",
        features="n/a", includes_product_avg_rating="mixed",
        disagreement_labels_in_training="mixed",
        command="python -m scripts_icml.phase8_statistics",
        output_files="reports_icml/tables/statistical_tests_main.csv,effect_sizes.csv,reports_icml/statistical_testing_report.md",
        key_metrics="seed-paired + example-level McNemar/bootstrap for 7 key comparisons"))
    print("\n[Phase 8] complete.")
    print(main_df.to_string(index=False))


if __name__ == "__main__":
    main()
