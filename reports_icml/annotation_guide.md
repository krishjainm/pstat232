# Human Annotation Guide: Modality Conflict vs Label Noise

This guide accompanies the disagreement candidates exported in
`reports_icml/manual_annotation_sample.csv` (and any batch produced by
`scripts_icml/create_annotation_batch.py`). The goal is to separate **genuine
semantic conflict** between modalities from **label noise** and other
explanations, so the paper can report what fraction of "disagreement" failures
are real model failures vs. dataset artifacts.

> Do **not** guess the model's prediction or use it to decide your label. Read
> the review text and the star rating, then choose the category that best
> explains the mismatch.

## How to annotate

For each row, fill in the human-label columns with `1` (applies) or `0` (does
not apply). A row may have more than one category set to `1`, but try to pick
the single **primary** explanation when possible. Use `notes` for anything
unusual.

Columns to fill:

| Column | Set to 1 when... |
| --- | --- |
| `genuine_conflict` | The text clearly expresses one sentiment while the star rating clearly indicates the opposite (e.g. glowing text + 1 star, or harsh text + 5 stars), and this is **not** explained by sarcasm or a typo. |
| `label_noise` | The star rating looks simply wrong or inconsistent with an otherwise unambiguous review (likely a mis-click or wrong-product rating). |
| `mixed_sentiment` | The text contains substantial positive **and** negative content, so neither label is clearly correct. |
| `sarcasm` | The text is sarcastic/ironic; literal sentiment is opposite to intended sentiment. |
| `unclear` | The text is too short, off-topic, non-English, or otherwise impossible to judge. |
| (notes) | A 6th category, *metadata-prior conflict*, is recorded in `notes` as `metadata_prior` when the conflict appears driven by product-level priors rather than the review's own semantics. |

## Category definitions (detailed)

1. **Genuine conflict** — The two modalities genuinely disagree. Example:
   "This serum completely cleared my skin, I love it" rated 1 star. This is the
   case our method most cares about.
2. **Label noise** — The rating appears erroneous. Example: a clearly negative
   review ("broke after one use, terrible") rated 5 stars with no irony.
3. **Mixed sentiment** — "Great color but the pump broke immediately." Both
   sentiments present; a single binary label is reductive.
4. **Sarcasm / irony** — "Oh sure, *love* paying \$40 for empty packaging."
   Literal text positive-sounding, intent negative (or vice versa).
5. **Ambiguous / too short** — "ok", "as described", emojis only, or non-English.
6. **Metadata-prior conflict** — The disagreement is an artifact of metadata
   priors (e.g. product average rating) rather than the review semantics; record
   in `notes`.

## Decision order (when multiple apply)

1. If text is uninterpretable -> `unclear`.
2. Else if clearly sarcastic -> `sarcasm`.
3. Else if both polarities strongly present -> `mixed_sentiment`.
4. Else if rating looks like a mistake -> `label_noise`.
5. Else if text and rating cleanly oppose -> `genuine_conflict`.

## Saving annotations

Save each annotator's completed file as
`reports_icml/annotations/annotations_<annotator_id>.csv`, keeping the `uid`
column intact. `scripts_icml/analyze_annotations.py` aggregates all files in that
folder, computes the genuine-conflict vs label-noise breakdown, inter-annotator
agreement (Cohen's kappa when >=2 annotators), and model accuracy on each
human category. If no annotation files exist, the script reports that manual
annotation is pending and exits cleanly.
