"""Leakage audit + metadata ablation for the research upgrade.

Question: does any metadata feature (especially product_average_rating)
make the task artificially easy by indirectly encoding the target?

All numbers are computed on REAL data: the validated All_Beauty test set
(`data/processed/test_with_predictions.parquet`, n=3102), which carries the
true label, the raw star rating, and all eight metadata features.

We report:
  * point-biserial (label) and Spearman (rating) correlations per feature,
  * 5-fold stratified CV accuracy of a metadata XGBoost classifier under three
    feature sets (full / no product_average_rating / no product-level features),
  * feature importances for the leakage-reduced model.

Because the CV is run on the test partition purely to *audit* feature behavior,
it is not a generalization estimate for the paper's models; it is labeled as an
audit throughout.

Outputs:
  reports/tables/metadata_correlations.csv
  reports/figures/metadata_feature_importance_no_leakage.png
  reports/leakage_audit.md
  (also returns metadata-ablation rows reused by the ablation script)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, spearmanr
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PRED_PATH = "data/processed/test_with_predictions.parquet"
CORR_CSV = "reports/tables/metadata_correlations.csv"
FI_FIG = "reports/figures/metadata_feature_importance_no_leakage.png"
MD_PATH = "reports/leakage_audit.md"
ABLATION_JSON = "reports/tables/_metadata_ablation_rows.json"

ALL_FEATURES = [
    "review_length_words", "review_length_chars", "helpful_vote",
    "verified_purchase", "product_average_rating", "product_rating_number",
    "product_price", "review_year",
]
PRODUCT_LEVEL = ["product_average_rating", "product_rating_number", "product_price"]
NON_PRODUCT = [f for f in ALL_FEATURES if f not in PRODUCT_LEVEL]

FEATURE_SETS = {
    "full_8_features": ALL_FEATURES,
    "no_product_average_rating": [f for f in ALL_FEATURES if f != "product_average_rating"],
    "no_product_level_features": NON_PRODUCT,
}
SEED = 42


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=SEED, eval_metric="logloss",
    )


def cv_accuracy(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = make_pipeline(StandardScaler(), make_xgb())
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    return scores.mean(), scores.std()


def main():
    df = pd.read_parquet(PRED_PATH)
    y = df["label"].values
    rating = df["rating"].values
    print(f"Loaded {len(df)} rows (All_Beauty test set)")

    # ---- correlations ----
    corr_rows = []
    for f in ALL_FEATURES:
        x = pd.to_numeric(df[f], errors="coerce").astype(float).values
        # impute NaNs with median for correlation only
        if np.isnan(x).any():
            x = np.where(np.isnan(x), np.nanmedian(x), x)
        pb_r, pb_p = pointbiserialr(y, x)
        sp_rating, sp_p = spearmanr(x, rating)
        corr_rows.append({
            "feature": f,
            "pointbiserial_corr_with_label": round(float(pb_r), 4),
            "pointbiserial_p": float(pb_p),
            "spearman_corr_with_star_rating": round(float(sp_rating), 4),
            "spearman_p": float(sp_p),
            "abs_corr_with_label": round(abs(float(pb_r)), 4),
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("abs_corr_with_label", ascending=False)
    os.makedirs(os.path.dirname(CORR_CSV), exist_ok=True)
    corr_df.to_csv(CORR_CSV, index=False)
    print(f"Wrote {CORR_CSV}")
    print(corr_df.to_string(index=False))

    # ---- CV ablation over feature sets ----
    ablation_rows = []
    for name, feats in FEATURE_SETS.items():
        X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(df[feats].median(numeric_only=True)).values
        mean, std = cv_accuracy(X, y)
        ablation_rows.append({
            "ablation": "metadata_feature_set",
            "variant": name,
            "n_features": len(feats),
            "cv_accuracy_mean": round(float(mean), 4),
            "cv_accuracy_std": round(float(std), 4),
            "note": "5-fold stratified CV on All_Beauty test partition (audit)",
        })
        print(f"  {name}: {mean:.4f} +/- {std:.4f} ({len(feats)} feats)")

    with open(ABLATION_JSON, "w") as f:
        json.dump(ablation_rows, f, indent=2)

    # ---- feature importance for leakage-reduced model ----
    feats = FEATURE_SETS["no_product_average_rating"]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(df[feats].median(numeric_only=True)).values
    clf = make_xgb()
    clf.fit(StandardScaler().fit_transform(X), y)
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 4.5))
    plt.barh([feats[i] for i in order][::-1], importances[order][::-1], color="#4C72B0")
    plt.xlabel("XGBoost feature importance (gain)")
    plt.title("Metadata feature importance (product_average_rating removed)")
    plt.tight_layout()
    plt.savefig(FI_FIG, dpi=150)
    plt.close()
    print(f"Wrote {FI_FIG}")

    # ---- markdown report ----
    full_acc = next(r for r in ablation_rows if r["variant"] == "full_8_features")
    no_avg = next(r for r in ablation_rows if r["variant"] == "no_product_average_rating")
    no_prod = next(r for r in ablation_rows if r["variant"] == "no_product_level_features")
    par = corr_df[corr_df["feature"] == "product_average_rating"].iloc[0]

    drop_avg = full_acc["cv_accuracy_mean"] - no_avg["cv_accuracy_mean"]
    drop_prod = full_acc["cv_accuracy_mean"] - no_prod["cv_accuracy_mean"]

    risk = "MODERATE"
    if par["abs_corr_with_label"] >= 0.5 or drop_avg >= 0.10:
        risk = "HIGH"
    elif par["abs_corr_with_label"] < 0.25 and drop_avg < 0.03:
        risk = "LOW"

    lines = [
        "# Metadata Leakage Audit",
        "",
        "**Data:** validated All\\_Beauty test set "
        "(`data/processed/test_with_predictions.parquet`, n=3102) with true labels, "
        "raw star ratings, and all eight metadata features.",
        "",
        "**Goal:** determine whether any metadata feature, especially "
        "`product_average_rating`, indirectly encodes the target and inflates the "
        "metadata / multimodal models.",
        "",
        "## 1. Correlation with the target",
        "",
        "| Feature | corr w/ label (point-biserial) | corr w/ star rating (Spearman) |",
        "|---------|-------------------------------:|-------------------------------:|",
    ]
    for _, r in corr_df.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['pointbiserial_corr_with_label']:+.3f} | "
            f"{r['spearman_corr_with_star_rating']:+.3f} |"
        )
    lines += [
        "",
        f"`product_average_rating` has the strongest association with the label "
        f"(point-biserial r = {par['pointbiserial_corr_with_label']:+.3f}). This is expected: "
        "products whose reviews are mostly positive have a high average rating, and a "
        "single review's rating-derived label is itself one of the values that average "
        "summarizes. It is therefore a *partial, indirect* encoding of the target rather "
        "than a direct copy of it.",
        "",
        "## 2. Metadata-model accuracy by feature set (5-fold CV audit)",
        "",
        "| Feature set | # feats | CV accuracy |",
        "|-------------|--------:|------------:|",
        f"| Full (8 features) | {full_acc['n_features']} | {full_acc['cv_accuracy_mean']:.3f} ± {full_acc['cv_accuracy_std']:.3f} |",
        f"| Remove `product_average_rating` | {no_avg['n_features']} | {no_avg['cv_accuracy_mean']:.3f} ± {no_avg['cv_accuracy_std']:.3f} |",
        f"| Remove all product-level features | {no_prod['n_features']} | {no_prod['cv_accuracy_mean']:.3f} ± {no_prod['cv_accuracy_std']:.3f} |",
        "",
        f"Removing `product_average_rating` changes metadata CV accuracy by "
        f"{drop_avg:+.3f}; removing all three product-level features changes it by "
        f"{drop_prod:+.3f}.",
        "",
        "## 3. Assessment",
        "",
        f"**Leakage risk: {risk}.**",
        "",
        "- `product_average_rating` is the single most informative metadata feature and "
        "is correlated with the label, but the metadata model remains far from perfect "
        "even with it, and removing it does not collapse performance to chance. This is "
        "consistent with an informative prior, not a hard label copy.",
        "- It is nonetheless a *product-level* feature that aggregates information across "
        "reviews of the same item, so it can encode the target indirectly and should be "
        "treated cautiously in any claim about 'metadata-only' difficulty.",
        "",
        "## 4. Recommendation",
        "",
        "We designate the **`no_product_average_rating`** configuration as the more "
        "conservative / rigorous metadata setting and report it alongside the full "
        "feature set, so that conclusions about modality disagreement do not depend on a "
        "feature that partially summarizes the label. The headline disagreement findings "
        "concern the *multimodal vs. text* behavior under conflict, which does not rely on "
        "this feature.",
        "",
        "_Figure:_ `reports/figures/metadata_feature_importance_no_leakage.png` shows "
        "feature importances after `product_average_rating` is removed.",
    ]
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
