# Qualitative Case Studies

Real examples drawn from the validated All\_Beauty test set (`data/processed/test_with_predictions.parquet`, n=3102). Predictions and confidences are the actual saved model outputs; text is truncated for readability.

## 1. Agreement, correct

> Very disappointing

- **Star rating:** 1  |  **True label:** negative  |  **Disagreement group:** agreement
- **Text-only:** negative (conf 0.998)
- **Metadata-only:** negative (conf 0.989)
- **Multimodal:** negative (conf 1.0)
- **Notes:** True=negative; text=negative(1.00), meta=negative(0.99), mm=negative(1.00). Modalities agree and the fusion model is correct; the easy regime.

## 2. Disagreement, correct

> Did not stick.

- **Star rating:** 1  |  **True label:** negative  |  **Disagreement group:** medium_disagreement
- **Text-only:** negative (conf 0.999)
- **Metadata-only:** negative (conf 0.993)
- **Multimodal:** negative (conf 1.0)
- **Notes:** True=negative; text=negative(1.00), meta=negative(0.99), mm=negative(1.00). Text sentiment contradicts the rating-derived label, yet fusion still recovers the correct answer.

## 3. Disagreement, incorrect

> Nice and comfy

- **Star rating:** 1  |  **True label:** negative  |  **Disagreement group:** strong_disagreement
- **Text-only:** positive (conf 0.999)
- **Metadata-only:** positive (conf 0.955)
- **Multimodal:** positive (conf 1.0)
- **Notes:** True=negative; text=positive(1.00), meta=positive(0.95), mm=positive(1.00). Under conflict the fusion model errs, illustrating the disagreement failure mode.

## 4. High-confidence disagreement failure

> The zipper is broken can not use them waste of money and not use and can not return that is not right

- **Star rating:** 5  |  **True label:** positive  |  **Disagreement group:** strong_disagreement
- **Text-only:** negative (conf 0.996)
- **Metadata-only:** negative (conf 0.754)
- **Multimodal:** negative (conf 1.0)
- **Notes:** True=positive; text=negative(1.00), meta=negative(0.75), mm=negative(1.00). A confident (>=0.9) fusion error inside a disagreement case: miscalibration under conflict.

## 5. Multimodal follows text

> It wasn’t reusable

- **Star rating:** 4  |  **True label:** positive  |  **Disagreement group:** strong_disagreement
- **Text-only:** negative (conf 0.998)
- **Metadata-only:** positive (conf 0.515)
- **Multimodal:** negative (conf 0.998)
- **Notes:** True=positive; text=negative(1.00), meta=positive(0.52), mm=negative(1.00). Text and metadata predict opposite labels; fusion adopts the TEXT decision.

## 6. Multimodal follows metadata

> Worked as Advertised

- **Star rating:** 5  |  **True label:** positive  |  **Disagreement group:** strong_disagreement
- **Text-only:** negative (conf 0.605)
- **Metadata-only:** positive (conf 0.999)
- **Multimodal:** positive (conf 0.989)
- **Notes:** True=positive; text=negative(0.61), meta=positive(1.00), mm=positive(0.99). Text and metadata predict opposite labels; fusion adopts the METADATA decision.
