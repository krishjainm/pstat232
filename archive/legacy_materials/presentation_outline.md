# Presentation Outline: When Modalities Disagree
**10-minute presentation plan**

---

## Slide 1: Title
- **When Modalities Disagree: Failure Modes in Multimodal Models**
- PSTAT 232 Final Project
- Author name, date

## Slide 2: Motivation
- Multimodal models combine text, images, metadata, etc.
- Common assumption: more modalities = better predictions.
- But what happens when modalities **conflict**?
- Example: "This product is amazing!" → 1 star.
- **Research question**: How do multimodal models behave when input modalities disagree?

## Slide 3: Dataset and Task
- Amazon Reviews 2023, All_Beauty category.
- Binary sentiment: positive (rating ≥ 4) vs. negative (rating ≤ 2).
- Two modalities: review text + structured metadata.
- [Show class distribution figure]

## Slide 4: Defining Agreement vs. Disagreement
- Use pretrained sentiment model to score review text.
- **Agreement**: text sentiment matches rating label.
- **Disagreement**: text sentiment contradicts rating label.
- Severity: weak / medium / strong (based on text model confidence).
- [Show agreement distribution figure]

## Slide 5: Models Compared
- **Text-only**: DistilBERT (or TF-IDF + LogReg).
- **Metadata-only**: XGBoost on structured features.
- **Multimodal**: DistilBERT embeddings + metadata → MLP fusion.
- All evaluated on the same held-out test set.

## Slide 6: Main Performance Results
- Overall accuracy, F1, AUROC for each model.
- [Show main results table]
- Key takeaway: multimodal performs well *on average*.

## Slide 7: Disagreement Severity Results
- Accuracy drops as disagreement severity increases.
- Gap between agreement and strong disagreement is substantial.
- [Show accuracy by group bar chart + severity trend figure]

## Slide 8: Confidence and Calibration
- Models become overconfident on disagreement cases.
- High-confidence error rate is elevated for strong disagreement.
- [Show calibration curves + confidence histogram]

## Slide 9: Modality Dominance Analysis
- When text and metadata disagree, which does the multimodal model follow?
- [Show modality dominance bar chart]
- Key finding: the model tends to follow [text/metadata] under conflict.
- This reveals an implicit modality bias.

## Slide 10: Conclusion and Future Work
- **Key takeaways**:
  1. Average accuracy masks failures under modality conflict.
  2. Disagreement cases are harder, and severity predicts error rate.
  3. Multimodal fusion exhibits modality dominance under conflict.
  4. Overconfidence under disagreement is a reliability concern.
- **Future work**: disagreement-aware fusion, modality-conditioned calibration, extension to image+text modalities.
- **Main message**: Multimodal models must be evaluated under conflict — disagreement reveals hidden failure modes.
