# Metadata Leakage Audit

**Data:** validated All\_Beauty test set (`data/processed/test_with_predictions.parquet`, n=3102) with true labels, raw star ratings, and all eight metadata features.

**Goal:** determine whether any metadata feature, especially `product_average_rating`, indirectly encodes the target and inflates the metadata / multimodal models.

## 1. Correlation with the target

| Feature | corr w/ label (point-biserial) | corr w/ star rating (Spearman) |
|---------|-------------------------------:|-------------------------------:|
| `product_average_rating` | +0.380 | +0.388 |
| `review_year` | -0.120 | -0.127 |
| `product_rating_number` | +0.064 | +0.095 |
| `product_price` | +0.060 | +0.046 |
| `verified_purchase` | -0.056 | -0.034 |
| `review_length_chars` | +0.016 | -0.069 |
| `review_length_words` | +0.014 | -0.075 |
| `helpful_vote` | -0.004 | -0.055 |

`product_average_rating` has the strongest association with the label (point-biserial r = +0.380). This is expected: products whose reviews are mostly positive have a high average rating, and a single review's rating-derived label is itself one of the values that average summarizes. It is therefore a *partial, indirect* encoding of the target rather than a direct copy of it.

## 2. Metadata-model accuracy by feature set (5-fold CV audit)

| Feature set | # feats | CV accuracy |
|-------------|--------:|------------:|
| Full (8 features) | 8 | 0.660 ± 0.021 |
| Remove `product_average_rating` | 7 | 0.551 ± 0.017 |
| Remove all product-level features | 5 | 0.546 ± 0.025 |

Removing `product_average_rating` changes metadata CV accuracy by +0.110; removing all three product-level features changes it by +0.114.

## 3. Assessment

**Leakage risk: HIGH.**

- `product_average_rating` is the single most informative metadata feature and is correlated with the label, but the metadata model remains far from perfect even with it, and removing it does not collapse performance to chance. This is consistent with an informative prior, not a hard label copy.
- It is nonetheless a *product-level* feature that aggregates information across reviews of the same item, so it can encode the target indirectly and should be treated cautiously in any claim about 'metadata-only' difficulty.

## 4. Recommendation

We designate the **`no_product_average_rating`** configuration as the more conservative / rigorous metadata setting and report it alongside the full feature set, so that conclusions about modality disagreement do not depend on a feature that partially summarizes the label. The headline disagreement findings concern the *multimodal vs. text* behavior under conflict, which does not rely on this feature.

_Figure:_ `reports/figures/metadata_feature_importance_no_leakage.png` shows feature importances after `product_average_rating` is removed.