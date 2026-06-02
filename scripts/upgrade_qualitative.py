"""Curated qualitative case studies for the research upgrade.

Selects six representative, real test examples from the validated All_Beauty
test predictions and writes both a CSV and a readable Markdown report.

Cases: agreement-correct, disagreement-correct, disagreement-incorrect,
high-confidence disagreement failure, multimodal-follows-text,
multimodal-follows-metadata.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRED_PATH = "data/processed/test_with_predictions.parquet"
CSV_PATH = "reports/tables/qualitative_case_studies.csv"
MD_PATH = "reports/qualitative_case_studies.md"

LBL = {0: "negative", 1: "positive"}


def _preview(text, n=280):
    text = str(text).replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def pick(df, mask, sort_col=None, ascending=False):
    sub = df[mask]
    if len(sub) == 0:
        return None
    if sort_col is not None:
        sub = sub.sort_values(sort_col, ascending=ascending)
    return sub.iloc[0]


def explain(row, case):
    t = f"text={LBL[row.text_pred]}({row.text_confidence:.2f})"
    m = f"meta={LBL[row.meta_pred]}({row.meta_confidence:.2f})"
    mm = f"mm={LBL[row.mm_pred]}({row.mm_confidence:.2f})"
    truth = LBL[row.label]
    base = f"True={truth}; {t}, {m}, {mm}."
    notes = {
        "agreement_correct": "Modalities agree and the fusion model is correct; the easy regime.",
        "disagreement_correct": "Text sentiment contradicts the rating-derived label, yet fusion still recovers the correct answer.",
        "disagreement_incorrect": "Under conflict the fusion model errs, illustrating the disagreement failure mode.",
        "high_conf_disagreement_failure": "A confident (>=0.9) fusion error inside a disagreement case: miscalibration under conflict.",
        "mm_follows_text": "Text and metadata predict opposite labels; fusion adopts the TEXT decision.",
        "mm_follows_metadata": "Text and metadata predict opposite labels; fusion adopts the METADATA decision.",
    }
    return base + " " + notes[case]


def main():
    df = pd.read_parquet(PRED_PATH)
    for c in ["text_pred", "meta_pred", "mm_pred", "label"]:
        df[c] = df[c].astype(int)
    print(f"Loaded {len(df)} rows")

    dis = df["agreement_status"] == "disagreement"
    agree = df["agreement_status"] == "agreement"
    mm_correct = df["mm_pred"] == df["label"]
    conflict = df["text_pred"] != df["meta_pred"]
    follows_text = conflict & (df["mm_pred"] == df["text_pred"])
    follows_meta = conflict & (df["mm_pred"] == df["meta_pred"])

    selections = [
        ("agreement_correct", agree & mm_correct, "mm_confidence", False),
        ("disagreement_correct", dis & mm_correct, "mm_confidence", False),
        ("disagreement_incorrect", dis & ~mm_correct, "mm_confidence", False),
        ("high_conf_disagreement_failure", dis & ~mm_correct & (df["mm_confidence"] >= 0.9), "mm_confidence", False),
        ("mm_follows_text", follows_text & dis, "mm_confidence", False),
        ("mm_follows_metadata", follows_meta, "mm_confidence", False),
    ]

    rows = []
    used_idx = set()
    for case, mask, sort_col, asc in selections:
        # avoid reusing the exact same review across cases when possible
        m = mask & ~df.index.isin(used_idx)
        row = pick(df, m, sort_col, asc)
        if row is None:
            row = pick(df, mask, sort_col, asc)
        if row is None:
            print(f"[warn] no example found for {case}")
            continue
        used_idx.add(row.name)
        rows.append({
            "case": case,
            "review_text": _preview(row.review_text),
            "rating": int(row.rating),
            "true_label": LBL[int(row.label)],
            "text_pred": LBL[int(row.text_pred)],
            "text_confidence": round(float(row.text_confidence), 3),
            "meta_pred": LBL[int(row.meta_pred)],
            "meta_confidence": round(float(row.meta_confidence), 3),
            "mm_pred": LBL[int(row.mm_pred)],
            "mm_confidence": round(float(row.mm_confidence), 3),
            "disagreement_group": row.disagreement_group,
            "explanation": explain(row, case),
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    out.to_csv(CSV_PATH, index=False)
    print(f"Wrote {CSV_PATH} ({len(out)} cases)")

    # Markdown report
    titles = {
        "agreement_correct": "1. Agreement, correct",
        "disagreement_correct": "2. Disagreement, correct",
        "disagreement_incorrect": "3. Disagreement, incorrect",
        "high_conf_disagreement_failure": "4. High-confidence disagreement failure",
        "mm_follows_text": "5. Multimodal follows text",
        "mm_follows_metadata": "6. Multimodal follows metadata",
    }
    lines = [
        "# Qualitative Case Studies",
        "",
        "Real examples drawn from the validated All\\_Beauty test set "
        "(`data/processed/test_with_predictions.parquet`, n=3102). "
        "Predictions and confidences are the actual saved model outputs; "
        "text is truncated for readability.",
        "",
    ]
    for r in rows:
        lines.append(f"## {titles[r['case']]}")
        lines.append("")
        lines.append(f"> {r['review_text']}")
        lines.append("")
        lines.append(f"- **Star rating:** {r['rating']}  |  **True label:** {r['true_label']}  |  **Disagreement group:** {r['disagreement_group']}")
        lines.append(f"- **Text-only:** {r['text_pred']} (conf {r['text_confidence']})")
        lines.append(f"- **Metadata-only:** {r['meta_pred']} (conf {r['meta_confidence']})")
        lines.append(f"- **Multimodal:** {r['mm_pred']} (conf {r['mm_confidence']})")
        lines.append(f"- **Notes:** {r['explanation']}")
        lines.append("")

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
