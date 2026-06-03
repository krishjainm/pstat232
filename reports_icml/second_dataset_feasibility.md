# Phase 3: Second Dataset Family — Feasibility Report

**Goal.** Move beyond Amazon Reviews so the conflict phenomenon is not tied to a
single dataset family. We require a dataset that (i) has ≥2 modalities, (ii)
supports classification / sentiment-style prediction, (iii) admits a
disagreement / conflict proxy analogous to "pretrained-sentiment vs label" or
"cross-modal prediction disagreement", and (iv) is feasible on our **CPU-only**
compute budget.

> Status: this is a **feasibility analysis with a concrete recommendation**.
> Per the project's compute constraints and the agreed plan, the second dataset
> is scoped here and earmarked for implementation; it is **not yet benchmarked**.
> The repository remains honest about this (see `phase3` rows omitted from the
> results tables, and the readiness report's "remaining gaps").

## Compute context

All experiments to date run on CPU only (no CUDA). The Amazon pipeline is kept
feasible by (a) building a fixed balanced pool of ~20k reviews per category,
(b) caching SBERT embeddings and DistilBERT-SST-2 sentiment once, and
(c) redrawing only the train/val/test split per seed. Any second dataset must be
small enough that one-time encoding is affordable on CPU (tens of thousands of
short text items, or a few thousand images at most).

## Candidate analysis

| Candidate | Task | Modalities | Size | Labels | Disagreement proxy | Compute (CPU) | Download feasibility | Chosen? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Yelp Open Dataset** | Review sentiment | Text + structured metadata (stars, useful/funny/cool votes, business attributes, avg business rating) | ~7M reviews (subsample to ~20k) | 1–5 stars → binary | **Direct analog**: DistilBERT-SST-2(text) vs star label; cross-model text-vs-metadata; product/business avg-rating leakage control | **Low** (same SBERT + sentiment pipeline as Amazon; reuse `icml_common`) | Public (Yelp dataset download / HF mirrors); large JSON but we cap rows | **YES (recommended)** |
| MM-IMDb | Genre / sentiment | Text (plot) + image (poster) | ~25k movies | multi-label genre | Weak: genre is not sentiment; conflict proxy is awkward | Medium (image CNN features once) | Public (HF `mm_imdb`) | No — task mismatch (multi-label genre, not a clean conflict signal) |
| Hateful Memes | Hateful vs not | Image + text (overlaid) | 10k memes | binary hateful | Strong conceptually (text benign, image hateful → conflict) | **High** on CPU (needs vision-language features; designed for GPU) | Gated download (Facebook/DrivenData license) | No — license friction + CPU-infeasible image-text encoding |
| CMU-MOSI | Sentiment | Text + audio + video | ~2.2k clips | continuous sentiment | Strong (modality-specific sentiment can conflict) | **High** (audio/video feature extraction; aligned features needed) | CMU SDK download, nontrivial | No — small N, heavy feature pipeline, CPU-infeasible |
| CMU-MOSEI | Sentiment/emotion | Text + audio + video | ~23k clips | sentiment + emotion | Strong | **Very high** on CPU | Large (GBs), CMU SDK | No — too heavy for CPU |
| HF `sst2` / `imdb` | Sentiment | Text only | 67k / 50k | binary | None usable | Low | Easy | No — single modality |

## Why Yelp is the recommended second family

1. **Same conflict construct, different domain.** Yelp gives review *text* plus
   rich *structured metadata* (star rating, vote counts, and crucially a
   business-level `stars`/`review_count` aggregate). This mirrors the Amazon
   text+metadata setup almost exactly, so:
   - Definition A (pretrained-sentiment vs star label) transfers directly.
   - Definition B (text-only vs metadata-only prediction) transfers directly.
   - The **leakage control** story transfers: the business-level average rating
     is the Yelp analog of `product_average_rating`, letting us repeat the
     leakage-controlled vs full-feature comparison in a non-Amazon domain.
2. **Compute-compatible.** Text is short-to-medium; we reuse the exact
   `icml_common` pipeline (SBERT + DistilBERT-SST-2 + XGBoost/LogReg + fusion
   MLPs + UGCA). One-time encoding of a 20k subsample is the same cost as one
   Amazon category (feasible on CPU).
3. **Strengthens the paper's external validity** without introducing a new,
   CPU-infeasible modality (audio/video/image).

### Honest limitation of choosing Yelp
Yelp is still a *review + metadata* family, so it does not test a fundamentally
different modality pair (e.g. image+text). It does, however, break the
"single-dataset / single-source" weakness and re-tests the leakage-controlled
conflict phenomenon in an independent domain. A true cross-modality test
(MOSI/MOSEI or Hateful Memes) is listed as a **GPU-required future experiment**
in the readiness report.

## Proposed implementation plan (when compute allows)

1. Add `download_yelp()` to a new `scripts_icml/yelp_common.py` (or extend
   `icml_common`) that:
   - reads a capped subsample of the Yelp reviews JSON (reuse the `max_read_rows`
     pattern), joins business metadata, builds binary labels (stars ≥4 = pos,
     ≤2 = neg, drop 3), and balances to a ~20k pool.
   - defines `METADATA_FULL_YELP` (incl. business avg stars, review_count) and
     `METADATA_LC_YELP` (drop business-level rating aggregates).
2. Reuse the existing pool→sentiment→SBERT→split machinery unchanged.
3. Run the Phase-1 model set + UGCA for seeds {42,123,456}.
4. Save:
   - `data_icml/second_dataset/` (pool + caches)
   - `reports_icml/tables/second_dataset_results.csv`
   - `reports_icml/figures/second_dataset_disagreement_results.png`

## Decision

**Chosen dataset family: Yelp (text + business/review metadata).**
Rationale: maximal reuse of the validated pipeline, direct transfer of all
disagreement definitions and the leakage-control design, and CPU feasibility.
Implementation is scoped above and deferred to a compute window; the
cross-modality datasets (MOSI/MOSEI, Hateful Memes) are documented as
GPU-required follow-ups rather than assumed available.
