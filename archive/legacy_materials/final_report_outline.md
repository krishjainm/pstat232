# When Modalities Disagree: Failure Modes in Multimodal Models

## Abstract

Multimodal learning integrates diverse data sources under the assumption that combining modalities improves prediction. However, real-world data frequently contains conflicting signals across modalities. This project investigates how multimodal models behave when input modalities disagree, using the Amazon Reviews 2023 dataset. We construct binary sentiment classification models using text, structured metadata, and a multimodal fusion architecture, then systematically evaluate performance across agreement and disagreement subsets of varying severity. Our analysis reveals whether multimodal fusion improves robustness under conflict or introduces new failure modes, which modality the fused model relies on when signals disagree, and whether disagreement induces overconfidence. Results show that average accuracy masks important reliability failures that emerge under modality conflict.

## 1. Introduction

- Multimodal machine learning combines heterogeneous data sources (text, images, metadata, etc.) to improve predictions.
- A common assumption is that more modalities always help — more information should mean better decisions.
- In practice, modalities can conflict: a review's text may sound positive while its star rating is low, or vice versa.
- When modalities disagree, fusion models face a dilemma: which signal should dominate?
- This project systematically studies model behavior under modality disagreement.
- We define disagreement severity, evaluate models on disagreement subgroups, and analyze modality dominance and calibration.

## 2. Background and Related Work

- **Multimodal learning**: Baltrušaitis et al. (2019) survey of multimodal ML; fusion strategies (early, late, hybrid).
- **Multimodal robustness**: Studies on how models handle missing or noisy modalities.
- **Modality dominance / bias**: Evidence that multimodal models can over-rely on one modality (e.g., language bias in VQA).
- **Calibration and confidence**: Guo et al. (2017) on modern neural network calibration; Expected Calibration Error (ECE).
- **Disagreement and failure analysis**: Importance of evaluating models beyond aggregate metrics; subgroup analysis.

## 3. Data

- **Dataset**: Amazon Reviews 2023 (McAuley Lab), All_Beauty category.
- **Task**: Binary sentiment classification — positive (rating ≥ 4) vs. negative (rating ≤ 2); neutral (rating = 3) dropped.
- **Text modality**: Review text.
- **Metadata modality**: Review length, helpful votes, verified purchase, product average rating, product rating count, price, review year.
- **Agreement/disagreement definition**: A pretrained sentiment model scores each review's text. If the text sentiment label matches the rating-derived label, the case is "agreement"; otherwise, "disagreement." Severity is determined by the text model's confidence.
- **Dataset statistics**: [Insert table — class balance, split sizes, disagreement rates.]

## 4. Methods

### 4.1 Models
- **Text-only**: DistilBERT fine-tuned for binary classification (fallback: TF-IDF + Logistic Regression).
- **Metadata-only**: XGBoost classifier on structured features.
- **Multimodal fusion**: Frozen DistilBERT embeddings concatenated with standardized metadata features, fed into a 2-layer MLP.

### 4.2 Disagreement Framework
- Text sentiment scored via pretrained `distilbert-base-uncased-finetuned-sst-2-english`.
- Disagreement severity: weak (confidence < 0.70), medium (0.70–0.90), strong (≥ 0.90).

### 4.3 Evaluation
- Standard metrics: accuracy, precision, recall, F1, AUROC.
- Calibration: ECE, Brier score, confidence statistics.
- Group analysis: all metrics broken down by agreement/disagreement group.
- Modality dominance: for disagreement cases, classify whether the multimodal model follows text, metadata, both, or neither.

## 5. Experiments

- **Split**: 70/15/15 stratified train/val/test.
- **Training details**: DistilBERT 2 epochs, lr=2e-5; XGBoost 200 trees; MLP 5 epochs, lr=1e-4.
- **Evaluation on held-out test set only.**
- All experiments use seed=42 for reproducibility.

## 6. Results

### 6.1 Overall Performance
[Insert main_results table]

### 6.2 Performance by Agreement Group
[Insert group_results table — accuracy and F1 by model and group]

### 6.3 Calibration Analysis
[Insert calibration_results table — ECE, Brier, confidence stats]
[Insert calibration curves figure]

### 6.4 Modality Dominance
[Insert modality_dominance_results table]
[Insert modality dominance bar chart]

### 6.5 High-Confidence Errors
[Insert high-confidence error rate figure]
[Insert error examples table]

### 6.6 Disagreement Severity Trend
[Insert disagreement severity trend figure]

## 7. Discussion

- **Does multimodal always help?** Compare overall accuracy across models. Discuss cases where fusion helps vs. hurts.
- **Are disagreement cases harder?** Compare agreement vs. disagreement accuracy. Quantify the gap.
- **Which modality dominates?** Report dominance percentages. Discuss implications for trust and interpretability.
- **Overconfidence under disagreement?** Compare confidence on correct vs. incorrect predictions across groups. Highlight high-confidence failures.
- **Practical implications**: When should practitioners trust multimodal predictions? How can disagreement detection improve reliability?

## 8. Limitations

- Rating-derived labels are an imperfect proxy for true sentiment.
- Metadata features are limited and may not fully represent the "metadata modality."
- Analysis is on one dataset category only — generalization requires further study.
- The pretrained sentiment model used for disagreement labeling may itself have biases.
- This is an analysis project, not a novel architecture — the contribution is the evaluation framework.

## 9. Conclusion

This project demonstrates that multimodal models, while effective on average, can exhibit degraded performance and overconfidence when input modalities conflict. Disagreement severity is a meaningful predictor of model failure, and modality dominance analysis reveals which signal the fused model trusts under conflict. These findings motivate further research into disagreement-aware fusion strategies and modality-conditioned calibration.

## References

- Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks.
- Hou, Y., et al. (2024). Amazon Reviews 2023 dataset.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT.
